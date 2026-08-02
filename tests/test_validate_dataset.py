from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np


def test_dataset_validator_accepts_minimal_valid_trajectory(tmp_path):
    data_root = tmp_path / "data"
    split = tmp_path / "split"
    trajectory = data_root / "traj_001"
    trajectory.mkdir(parents=True)
    split.mkdir()
    (split / "traj_names.txt").write_text("traj_001\n", encoding="utf-8")
    with (trajectory / "traj_data.pkl").open("wb") as stream:
        pickle.dump(
            {"position": np.zeros((2, 2)), "yaw": np.zeros(2)},
            stream,
        )
    (trajectory / "0.jpg").write_bytes(b"placeholder")
    (trajectory / "1.jpg").write_bytes(b"placeholder")

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_dataset.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(data_root),
            "--split",
            str(split),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 0 failed" in result.stdout
