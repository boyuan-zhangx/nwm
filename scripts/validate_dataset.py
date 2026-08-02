#!/usr/bin/env python3
"""Validate the NWM trajectory contract without starting a training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


def validate_trajectory(root: Path, name: str) -> list[str]:
    problems: list[str] = []
    trajectory = root / name
    metadata_path = trajectory / "traj_data.pkl"
    if not metadata_path.is_file():
        return [f"missing {metadata_path}"]

    # NWM datasets are trusted project artifacts. Never open untrusted pickle files.
    with metadata_path.open("rb") as stream:
        metadata: Any = pickle.load(stream)
    if not isinstance(metadata, dict):
        return [f"{metadata_path}: expected dict, got {type(metadata).__name__}"]

    for key in ("position", "yaw"):
        if key not in metadata:
            problems.append(f"{metadata_path}: missing key {key!r}")
    if problems:
        return problems

    position = np.asarray(metadata["position"])
    yaw = np.asarray(metadata["yaw"]).squeeze()
    if position.ndim != 2 or position.shape[1] < 2:
        problems.append(f"{metadata_path}: position must have shape [T,>=2], got {position.shape}")
    if yaw.ndim != 1:
        problems.append(f"{metadata_path}: yaw must have shape [T] or [T,1], got {yaw.shape}")
    if position.shape[0] != yaw.shape[0]:
        problems.append(
            f"{metadata_path}: position/yaw length mismatch {position.shape[0]} != {yaw.shape[0]}"
        )
    if not np.isfinite(position).all() or not np.isfinite(yaw).all():
        problems.append(f"{metadata_path}: pose contains NaN or infinity")

    frame_count = min(position.shape[0], yaw.shape[0])
    if frame_count:
        for index in {0, frame_count - 1}:
            candidates = (trajectory / f"{index}.jpg", trajectory / f"{index}.png")
            if not any(candidate.is_file() for candidate in candidates):
                problems.append(
                    f"{trajectory}: frame {index} missing (expected .jpg or .png)"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="directory containing traj_names.txt")
    parser.add_argument("--max-trajectories", type=int, default=0, help="0 validates every listed trajectory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split = args.split.expanduser().resolve()
    names_path = split / "traj_names.txt"
    if not data_root.is_dir():
        parser.error(f"data root does not exist: {data_root}")
    if not names_path.is_file():
        parser.error(f"split file does not exist: {names_path}")

    names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_trajectories > 0:
        names = names[: args.max_trajectories]
    failures = {name: issues for name in names if (issues := validate_trajectory(data_root, name))}
    result = {
        "data_root": str(data_root),
        "split": str(split),
        "checked": len(names),
        "passed": len(names) - len(failures),
        "failed": len(failures),
        "failures": failures,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Checked {result['checked']} trajectories: "
            f"{result['passed']} passed, {result['failed']} failed"
        )
        for name, issues in failures.items():
            for issue in issues:
                print(f"[FAIL] {name}: {issue}")
    return 1 if failures or not names else 0


if __name__ == "__main__":
    raise SystemExit(main())
