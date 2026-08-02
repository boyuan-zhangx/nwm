"""Deterministic four-slot context selection for Phase A experiments.

The functions in this module do not depend on PyTorch or model code.  A policy
selects source frame indices only; the dataset remains responsible for loading
the corresponding real observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONTEXT_POLICIES = (
    "recent",
    "random_history",
    "oracle_manifest",
    "pose_aligned",
)


@dataclass(frozen=True)
class ContextSelection:
    """One fixed-size context selection and its retrieval diagnostics."""

    policy: str
    context_indices: tuple[int, ...]
    selected_source_index: int
    replaced_native_index: int
    temporal_gap: int
    pose_distance: float | None
    heading_difference: float | None
    retrieval_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wrapped_heading_difference(first: float, second: float) -> float:
    """Return the absolute wrapped angular distance in radians."""

    return float(abs((first - second + np.pi) % (2 * np.pi) - np.pi))


def _stable_index(size: int, *, seed: int, trajectory: str, query_index: int) -> int:
    if size <= 0:
        raise ValueError("cannot sample from an empty history")
    payload = f"{seed}\0{trajectory}\0{query_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % size


def _pose_diagnostics(
    source_index: int,
    query_index: int,
    positions: np.ndarray | None,
    yaws: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if positions is None or yaws is None:
        return None, None
    position_array = np.asarray(positions, dtype=np.float64)
    yaw_array = np.asarray(yaws, dtype=np.float64).squeeze()
    if position_array.ndim != 2 or position_array.shape[1] < 2:
        raise ValueError("positions must have shape [T, >=2]")
    if yaw_array.ndim != 1 or len(yaw_array) != len(position_array):
        raise ValueError("yaws must have shape [T] and match positions")
    if not 0 <= source_index < len(position_array) or not 0 <= query_index < len(position_array):
        raise ValueError("source or query index lies outside trajectory metadata")
    distance = float(
        np.linalg.norm(position_array[source_index, :2] - position_array[query_index, :2])
    )
    heading = wrapped_heading_difference(yaw_array[source_index], yaw_array[query_index])
    return distance, heading


def select_context_indices(
    *,
    policy: str,
    trajectory: str,
    query_index: int,
    context_size: int = 4,
    seed: int = 0,
    positive_indices: Sequence[int] = (),
    positions: np.ndarray | None = None,
    yaws: np.ndarray | None = None,
    min_temporal_gap: int | None = None,
    position_scale: float = 0.75,
    heading_scale_radians: float = np.deg2rad(20.0),
) -> ContextSelection:
    """Select exactly ``context_size`` frame indices for a frozen NWM.

    Replacement policies keep the latest ``context_size - 1`` frames and
    replace only the oldest native slot.  Historical candidates are always
    outside the native context and satisfy ``min_temporal_gap``.
    """

    if policy not in CONTEXT_POLICIES:
        raise ValueError(f"unknown context policy {policy!r}; choose from {CONTEXT_POLICIES}")
    if context_size < 2:
        raise ValueError("context_size must be at least 2")
    if query_index < context_size - 1:
        raise ValueError("query does not have enough native context frames")
    if position_scale <= 0 or heading_scale_radians <= 0:
        raise ValueError("pose score scales must be positive")

    recent = tuple(range(query_index - context_size + 1, query_index + 1))
    if policy == "recent":
        source = recent[0]
        pose_distance, heading_difference = _pose_diagnostics(
            source, query_index, positions, yaws
        )
        return ContextSelection(
            policy=policy,
            context_indices=recent,
            selected_source_index=source,
            replaced_native_index=recent[0],
            temporal_gap=query_index - source,
            pose_distance=pose_distance,
            heading_difference=heading_difference,
            retrieval_score=None,
        )

    effective_gap = max(context_size, min_temporal_gap or context_size)
    last_candidate = query_index - effective_gap
    if last_candidate < 0:
        raise ValueError("query does not have any history outside the native context")
    candidates = tuple(range(last_candidate + 1))

    score: float | None = None
    if policy == "random_history":
        positive_set = {int(index) for index in positive_indices}
        random_candidates = tuple(
            candidate for candidate in candidates if candidate not in positive_set
        )
        if not random_candidates:
            raise ValueError("manifest has no eligible non-positive random history")
        source = random_candidates[
            _stable_index(
                len(random_candidates),
                seed=seed,
                trajectory=trajectory,
                query_index=query_index,
            )
        ]
    elif policy == "oracle_manifest":
        candidate_set = set(candidates)
        valid_positives = [int(index) for index in positive_indices if int(index) in candidate_set]
        if not valid_positives:
            raise ValueError("manifest has no oracle positive in the eligible history")
        source = valid_positives[0]
    else:
        if positions is None or yaws is None:
            raise ValueError("pose_aligned requires positions and yaws")
        scored: list[tuple[float, float, float, int]] = []
        for candidate in candidates:
            distance, heading = _pose_diagnostics(candidate, query_index, positions, yaws)
            assert distance is not None and heading is not None
            candidate_score = distance / position_scale + heading / heading_scale_radians
            scored.append((candidate_score, distance, heading, candidate))
        score, _, _, source = min(scored)

    pose_distance, heading_difference = _pose_diagnostics(
        source, query_index, positions, yaws
    )
    replacement = (source, *recent[1:])
    if len(replacement) != context_size or len(set(replacement)) != context_size:
        raise AssertionError("context replacement did not produce unique fixed-size indices")
    return ContextSelection(
        policy=policy,
        context_indices=replacement,
        selected_source_index=source,
        replaced_native_index=recent[0],
        temporal_gap=query_index - source,
        pose_distance=pose_distance,
        heading_difference=heading_difference,
        retrieval_score=score,
    )


def load_revisit_manifest(path: Path | str) -> list[dict[str, Any]]:
    """Load and minimally validate a JSONL revisit manifest."""

    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{manifest_path}:{line_number}: record must be an object")
            if "trajectory" not in value or "query_index" not in value:
                raise ValueError(
                    f"{manifest_path}:{line_number}: trajectory and query_index are required"
                )
            record = dict(value)
            record["trajectory"] = str(record["trajectory"])
            record["query_index"] = int(record["query_index"])
            record.setdefault("positive_indices", [])
            key = (record["trajectory"], record["query_index"])
            if key in seen:
                raise ValueError(f"{manifest_path}:{line_number}: duplicate query {key}")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError(f"{manifest_path}: manifest contains no records")
    return records
