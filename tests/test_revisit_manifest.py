import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from scripts.build_revisit_manifest import trajectory_revisits


def test_trajectory_revisits_labels_return_to_start():
    position = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.1]]
    )
    yaw = np.zeros(len(position))
    records = trajectory_revisits(
        "loop",
        position,
        yaw,
        min_temporal_gap=3,
        position_threshold=0.2,
        heading_threshold=np.deg2rad(10),
        heading_wrong_threshold=np.deg2rad(90),
        query_stride=1,
        max_queries=0,
        rng=__import__("random").Random(0),
    )

    assert records
    return_record = next(record for record in records if record["query_index"] == 4)
    assert return_record["positive_indices"] == [0]
    assert return_record["query_action"] == [0.0, 0.1, 0.0]


def test_cli_writes_deterministic_jsonl(tmp_path):
    data_root = tmp_path / "data"
    trajectory = data_root / "loop"
    split = tmp_path / "split"
    trajectory.mkdir(parents=True)
    split.mkdir()
    (split / "traj_names.txt").write_text("loop\n", encoding="utf-8")
    with (trajectory / "traj_data.pkl").open("wb") as stream:
        pickle.dump(
            {
                "position": np.array(
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.1]]
                ),
                "yaw": np.zeros(6),
            },
            stream,
        )
    output = tmp_path / "manifest.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_revisit_manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--split",
            str(split),
            "--output",
            str(output),
            "--min-temporal-gap",
            "3",
            "--position-threshold",
            "0.2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row["query_index"] == 4 and row["positive_indices"] == [0] for row in rows)
