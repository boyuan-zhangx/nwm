#!/usr/bin/env python3
"""Build geometry-based revisit labels from NWM trajectory metadata.

The generated JSONL is a benchmark seed, not a final semantic annotation. It
defines long-gap pose/heading revisits independently of model predictions. Main
paper cases should additionally receive landmark/scenario labels from a frozen
annotation protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np


def wrapped_angle_difference(first: np.ndarray, second: float) -> np.ndarray:
    """Absolute wrapped angular distance in radians."""

    return np.abs((first - second + np.pi) % (2 * np.pi) - np.pi)


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # NWM metadata is pickle-based. Only open trusted project datasets.
    with path.open("rb") as stream:
        metadata: Any = pickle.load(stream)
    if not isinstance(metadata, dict) or not {"position", "yaw"} <= metadata.keys():
        raise ValueError(f"{path}: expected position and yaw arrays")
    position = np.asarray(metadata["position"], dtype=np.float64)
    yaw = np.asarray(metadata["yaw"], dtype=np.float64).squeeze()
    if position.ndim != 2 or position.shape[1] < 2:
        raise ValueError(f"{path}: position must have shape [T,>=2], got {position.shape}")
    if yaw.ndim != 1 or len(yaw) != len(position):
        raise ValueError(f"{path}: yaw must have shape [T] and match position length")
    if not np.isfinite(position).all() or not np.isfinite(yaw).all():
        raise ValueError(f"{path}: non-finite pose values")
    return position, yaw


def _ordered(indices: np.ndarray, distance: np.ndarray, heading: np.ndarray) -> list[int]:
    if not len(indices):
        return []
    ranking = np.lexsort((indices, heading[indices], distance[indices]))
    return indices[ranking].astype(int).tolist()


def trajectory_revisits(
    trajectory: str,
    position: np.ndarray,
    yaw: np.ndarray,
    *,
    min_temporal_gap: int,
    position_threshold: float,
    heading_threshold: float,
    heading_wrong_threshold: float,
    query_stride: int,
    max_queries: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Create one record per query with at least one geometric positive."""

    records: list[dict[str, Any]] = []
    for query in range(min_temporal_gap, len(position) - 1, query_stride):
        candidate_indices = np.arange(0, query - min_temporal_gap + 1)
        distance = np.linalg.norm(position[:, :2] - position[query, :2], axis=1)
        heading = wrapped_angle_difference(yaw, yaw[query])

        positive_mask = (
            (distance[candidate_indices] <= position_threshold)
            & (heading[candidate_indices] <= heading_threshold)
        )
        positives = candidate_indices[positive_mask]
        if not len(positives):
            continue

        near_position = candidate_indices[
            (distance[candidate_indices] <= position_threshold)
            & (heading[candidate_indices] >= heading_wrong_threshold)
        ]
        far_position = candidate_indices[
            (distance[candidate_indices] > position_threshold)
            & (heading[candidate_indices] <= heading_threshold)
        ]
        wrong_pool = sorted(set(candidate_indices.tolist()) - set(positives.tolist()))
        random_wrong = rng.choice(wrong_pool) if wrong_pool else None
        temporally_wrong = max(wrong_pool) if wrong_pool else None

        action_xy = position[query + 1, :2] - position[query, :2]
        action_yaw = float(
            (yaw[query + 1] - yaw[query] + np.pi) % (2 * np.pi) - np.pi
        )
        records.append(
            {
                "trajectory": trajectory,
                "query_index": query,
                "query_position_xy": position[query, :2].tolist(),
                "query_yaw": float(yaw[query]),
                "query_action": [float(action_xy[0]), float(action_xy[1]), action_yaw],
                "positive_indices": _ordered(positives, distance, heading),
                "heading_wrong_indices": _ordered(near_position, distance, heading),
                "spatial_wrong_indices": _ordered(far_position, distance, heading),
                "random_wrong_index": random_wrong,
                "temporally_wrong_index": temporally_wrong,
                "positive_pose_distance": [
                    float(distance[index]) for index in _ordered(positives, distance, heading)
                ],
                "positive_heading_difference": [
                    float(heading[index]) for index in _ordered(positives, distance, heading)
                ],
                "label_source": "geometry_v1",
            }
        )
        if max_queries > 0 and len(records) >= max_queries:
            break
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-temporal-gap", type=int, default=8)
    parser.add_argument("--position-threshold", type=float, default=0.75)
    parser.add_argument("--heading-threshold-deg", type=float, default=20.0)
    parser.add_argument("--heading-wrong-threshold-deg", type=float, default=90.0)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--max-queries-per-trajectory", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_temporal_gap < 1 or args.query_stride < 1:
        raise SystemExit("min-temporal-gap and query-stride must be positive")
    if args.position_threshold <= 0:
        raise SystemExit("position-threshold must be positive")
    if args.heading_wrong_threshold_deg <= args.heading_threshold_deg:
        raise SystemExit("heading-wrong threshold must exceed positive heading threshold")

    data_root = args.data_root.expanduser().resolve()
    split = args.split.expanduser().resolve()
    names_path = split / "traj_names.txt"
    if not data_root.is_dir() or not names_path.is_file():
        raise SystemExit("data-root or split/traj_names.txt does not exist")

    names = [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for name in names:
        position, yaw = load_trajectory(data_root / name / "traj_data.pkl")
        records.extend(
            trajectory_revisits(
                name,
                position,
                yaw,
                min_temporal_gap=args.min_temporal_gap,
                position_threshold=args.position_threshold,
                heading_threshold=np.deg2rad(args.heading_threshold_deg),
                heading_wrong_threshold=np.deg2rad(args.heading_wrong_threshold_deg),
                query_stride=args.query_stride,
                max_queries=args.max_queries_per_trajectory,
                rng=rng,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "trajectories": len(names),
        "revisit_queries": len(records),
        "output": str(args.output),
        "seed": args.seed,
        "thresholds": {
            "min_temporal_gap": args.min_temporal_gap,
            "position": args.position_threshold,
            "heading_deg": args.heading_threshold_deg,
            "heading_wrong_deg": args.heading_wrong_threshold_deg,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
