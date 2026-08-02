from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Bash wrapper tests run on Linux/WSL")


def _fake_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['NAVWARE_TEST_LOG'], 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _environment(fake_python: Path, log_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["NAVWARE_PYTHON"] = str(fake_python)
    environment["NAVWARE_TEST_LOG"] = str(log_path)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    return environment


def _calls(log_path: Path) -> list[list[str]]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_train_wrapper_resolves_caller_paths_and_forwards_arguments(tmp_path: Path) -> None:
    caller = tmp_path / "caller"
    caller.mkdir()
    experiment = caller / "experiment with spaces.yaml"
    paths = caller / "paths with spaces.yaml"
    experiment.write_text("model: test\n", encoding="utf-8")
    paths.write_text("paths: {}\n", encoding="utf-8")
    log_path = tmp_path / "calls.jsonl"
    fake_python = _fake_python(tmp_path / "bin" / "python")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "train.sh"),
            experiment.name,
            paths.name,
            "--epochs",
            "1",
        ],
        cwd=caller,
        env=_environment(fake_python, log_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = _calls(log_path)
    assert calls[0] == [
        "scripts/navware.py",
        "doctor",
        "--profile",
        "nwm",
        "--config",
        str(experiment),
        "--paths-config",
        str(paths),
    ]
    assert calls[1] == [
        "train.py",
        "--config",
        str(experiment),
        "--paths-config",
        str(paths),
        "--epochs",
        "1",
    ]


def test_infer_wrapper_resolves_repository_paths(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.jsonl"
    fake_python = _fake_python(tmp_path / "bin" / "python")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "infer.sh"),
            "config/nwm_cdit_xl.yaml",
            "config/paths.example.yaml",
            "--batch_size",
            "1",
        ],
        cwd=tmp_path,
        env=_environment(fake_python, log_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = _calls(log_path)
    experiment = str(REPO_ROOT / "config" / "nwm_cdit_xl.yaml")
    paths = str(REPO_ROOT / "config" / "paths.example.yaml")
    assert calls[1] == [
        "isolated_nwm_infer.py",
        "--exp",
        experiment,
        "--paths-config",
        paths,
        "--batch_size",
        "1",
    ]


def test_active_virtualenv_takes_precedence_over_repository_environment(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.jsonl"
    active_python = _fake_python(tmp_path / "active-venv" / "bin" / "python")
    environment = os.environ.copy()
    environment.pop("NAVWARE_PYTHON", None)
    environment.pop("CONDA_PREFIX", None)
    environment["VIRTUAL_ENV"] = str(tmp_path / "active-venv")
    environment["NAVWARE_TEST_LOG"] = str(log_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "train.sh"),
            "config/nwm_cdit_xl.yaml",
            "config/paths.example.yaml",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"[navware] Python: {active_python}" in result.stdout
    assert len(_calls(log_path)) == 2


@pytest.mark.parametrize("wrapper", ["train.sh", "infer.sh"])
def test_wrapper_rejects_missing_config_before_launch(tmp_path: Path, wrapper: str) -> None:
    log_path = tmp_path / "calls.jsonl"
    fake_python = _fake_python(tmp_path / "bin" / "python")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / wrapper),
            "missing-experiment.yaml",
            "missing-paths.yaml",
        ],
        cwd=tmp_path,
        env=_environment(fake_python, log_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Experiment config not found or not readable" in result.stderr
    assert not log_path.exists()
