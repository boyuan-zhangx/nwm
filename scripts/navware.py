#!/usr/bin/env python3
"""Small, dependency-light entry point for NavWare environment checks.

This script is safe to run before downloading model weights or datasets. It
does not install packages, submit jobs, start training, or contact W&B.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "core": ("torch", "torchvision", "timm", "yaml", "einops"),
    "nwm": (
        "torch",
        "torchvision",
        "timm",
        "yaml",
        "einops",
        "diffusers",
        "transformers",
        "decord",
        "lpips",
        "dreamsim",
        "torcheval",
    ),
    "worldmem": (
        "torch",
        "torchvision",
        "lightning",
        "wandb",
        "hydra",
        "omegaconf",
        "torchmetrics",
        "cv2",
        "pandas",
    ),
}

NWM_VERSION_PINS = {
    "accelerate": "0.34.2",
    "diffusers": "0.30.3",
    "transformers": "4.44.2",
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def check(name: str, passed: bool, detail: str, *, warning: bool = False) -> Check:
    if passed:
        status = "PASS"
    elif warning:
        status = "WARN"
    else:
        status = "FAIL"
    return Check(name=name, status=status, detail=detail)


def python_checks() -> list[Check]:
    version = sys.version_info
    exact = version.major == 3 and version.minor == 10
    implementation = platform.python_implementation()
    return [
        check(
            "python",
            exact,
            f"{implementation} {platform.python_version()} at {sys.executable}; project target is 3.10",
        ),
        check(
            "virtualenv",
            sys.prefix != getattr(sys, "base_prefix", sys.prefix),
            f"prefix={sys.prefix}",
        ),
    ]


def platform_checks() -> list[Check]:
    is_linux = sys.platform.startswith("linux")
    is_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
    details = f"system={platform.system()} release={platform.release()}"
    if is_linux:
        details += f" wsl={is_wsl}"
    return [
        check(
            "linux-runtime",
            is_linux,
            details,
            warning=not is_linux,
        )
    ]


def package_checks(profile: str) -> list[Check]:
    checks = []
    for module in PROFILES[profile]:
        found = importlib.util.find_spec(module) is not None
        checks.append(check(f"module:{module}", found, "available" if found else "missing"))
    if profile == "nwm":
        for package, expected in NWM_VERSION_PINS.items():
            try:
                installed = metadata.version(package)
            except metadata.PackageNotFoundError:
                installed = None
            checks.append(
                check(
                    f"version:{package}",
                    installed == expected,
                    f"installed={installed!r}; required={expected}",
                )
            )
    return checks


def runtime_checks() -> list[Check]:
    checks = [
        check(
            "ffmpeg",
            shutil.which("ffmpeg") is not None,
            shutil.which("ffmpeg") or "not found on PATH",
        )
    ]
    if importlib.util.find_spec("torch") is None:
        checks.append(check("torch-runtime", False, "torch is not installed"))
        return checks

    import torch

    cuda = torch.cuda.is_available()
    detail = f"torch={torch.__version__}; cuda_available={cuda}"
    if cuda:
        detail += f"; device={torch.cuda.get_device_name(0)}; torch_cuda={torch.version.cuda}"
    checks.append(check("torch-runtime", cuda, detail, warning=not cuda))
    return checks


def _load_yaml(path: Path) -> tuple[Optional[dict], Optional[str]]:
    if importlib.util.find_spec("yaml") is None:
        return None, "PyYAML is not installed"
    import yaml

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:  # configuration diagnostics should report, not crash
        return None, str(error)
    if not isinstance(content, dict):
        return None, "top-level YAML value must be a mapping"
    return content, None


def config_checks(config_path: Path, paths_path: Optional[Path] = None) -> list[Check]:
    checks = [check("config:file", config_path.is_file(), str(config_path))]
    if not config_path.is_file():
        return checks

    if paths_path is None:
        config, error = _load_yaml(config_path)
    else:
        if not paths_path.is_file():
            checks.append(check("config:paths-file", False, str(paths_path)))
            return checks
        try:
            from config_utils import load_config

            config = load_config(
                REPO_ROOT / "config" / "eval_config.yaml", config_path, paths_path
            )
            error = None
        except Exception as config_error:
            config = None
            error = str(config_error)
    checks.append(check("config:yaml", error is None, error or "valid YAML mapping"))
    if config is None:
        return checks

    required = (
        "run_name",
        "results_dir",
        "model",
        "image_size",
        "context_size",
        "datasets",
    )
    missing = [key for key in required if key not in config]
    checks.append(
        check(
            "config:required-keys",
            not missing,
            "all present" if not missing else f"missing {missing}",
        )
    )

    image_size = config.get("image_size")
    checks.append(
        check(
            "config:image-size",
            isinstance(image_size, int) and image_size > 0 and image_size % 8 == 0,
            f"image_size={image_size}; must be a positive multiple of 8",
        )
    )

    model = config.get("model", "")
    hybrid_flag = bool(config.get("use_hybrid_model"))
    is_hybrid_name = isinstance(model, str) and model.startswith("HybridCDiT-")
    checks.append(
        check(
            "config:model-selection",
            hybrid_flag == is_hybrid_name or (not hybrid_flag and not is_hybrid_name),
            f"model={model!r}, use_hybrid_model={hybrid_flag}; use a HybridCDiT-* name for LT-NWM",
        )
    )
    if hybrid_flag or is_hybrid_name:
        checks.append(
            check(
                "config:hybrid-training-data",
                False,
                "retrieval/model unit tests work, but train.py does not yet provide memory_latents batches",
                warning=True,
            )
        )

    if bool(config.get("phase_a_frozen")):
        phase_a_model = isinstance(model, str) and model in {
            "CDiT-S/2",
            "CDiT-B/2",
            "CDiT-L/2",
            "CDiT-XL/2",
        }
        context_size = config.get("context_size")
        train_flag = config.get("train")
        checks.extend(
            [
                check(
                    "config:phase-a-model",
                    phase_a_model and not hybrid_flag,
                    f"model={model!r}; Phase A requires a standard CDiT checkpoint",
                ),
                check(
                    "config:phase-a-context",
                    context_size == 4,
                    f"context_size={context_size!r}; context replacement requires 4",
                ),
                check(
                    "config:phase-a-frozen",
                    train_flag is False,
                    f"train={train_flag!r}; Phase A configs must set train: false",
                ),
            ]
        )

    datasets = config.get("datasets")
    if isinstance(datasets, dict):
        for dataset_name, dataset in datasets.items():
            if not isinstance(dataset, dict):
                checks.append(check(f"dataset:{dataset_name}", False, "entry is not a mapping"))
                continue
            raw_path = dataset.get("data_folder")
            if not raw_path:
                checks.append(check(f"dataset:{dataset_name}:data", False, "data_folder is missing"))
                continue
            path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
            if not path.is_absolute():
                path = REPO_ROOT / path
            placeholder = "/path/to/" in str(raw_path).replace("\\", "/")
            checks.append(
                check(
                    f"dataset:{dataset_name}:data",
                    path.is_dir() and not placeholder,
                    str(path),
                )
            )
    else:
        checks.append(check("config:datasets", False, "datasets must be a mapping"))
    return checks


def git_checks() -> list[Check]:
    try:
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=D"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        return [check("git-worktree", False, f"could not inspect git: {error}", warning=True)]

    overlap = sorted(set(staged) & set(untracked))
    return [
        check(
            "git-index",
            not overlap,
            "clean index mapping"
            if not overlap
            else f"{len(overlap)} paths are staged deleted and simultaneously untracked",
            warning=bool(overlap),
        )
    ]


def render(checks: Iterable[Check], *, json_output: bool) -> int:
    checks = list(checks)
    if json_output:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        width = max(len(item.name) for item in checks)
        for item in checks:
            print(f"[{item.status:4}] {item.name:<{width}}  {item.detail}")
        counts = {status: sum(item.status == status for item in checks) for status in ("PASS", "WARN", "FAIL")}
        print(f"\nSummary: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    return 1 if any(item.status == "FAIL" for item in checks) else 0


def doctor(args: argparse.Namespace) -> int:
    checks = python_checks() + platform_checks() + package_checks(args.profile)
    checks += runtime_checks() + git_checks()
    if args.config is not None:
        config_path = args.config
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        paths_path = args.paths_config
        if paths_path is not None and not paths_path.is_absolute():
            paths_path = REPO_ROOT / paths_path
        checks += config_checks(
            config_path.resolve(), None if paths_path is None else paths_path.resolve()
        )
    return render(checks, json_output=args.json)


def smoke(_: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_hybrid_models.py",
        "tests/test_retrieval_context.py",
        "tests/test_revisit_manifest.py",
        "tests/test_phase_a_eval.py",
    ]
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)

    doctor_parser = subcommands.add_parser("doctor", help="check an environment and optional run config")
    doctor_parser.add_argument("--profile", choices=tuple(PROFILES), default="core")
    doctor_parser.add_argument("--config", type=Path)
    doctor_parser.add_argument("--paths-config", type=Path)
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.set_defaults(func=doctor)

    smoke_parser = subcommands.add_parser("smoke", help="run fast CPU memory/model tests")
    smoke_parser.set_defaults(func=smoke)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
