#!/usr/bin/env python3
"""Install or diagnose the vendored WorldMem reference implementation.

LT-NWM does not import WorldMem at runtime; install this optional profile only
when reproducing the upstream WorldMem baseline or its Minecraft experiments.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent


def repo_root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not (path / "WorldMem" / "requirements.txt").is_file():
        raise argparse.ArgumentTypeError(
            f"{path} does not contain WorldMem/requirements.txt"
        )
    return path


def require_supported_environment(*, allow_system: bool) -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"WorldMem requires the project Python 3.10 environment; found {sys.version.split()[0]}"
        )
    in_virtualenv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_virtualenv and not allow_system:
        raise SystemExit(
            "Refusing to install into the system interpreter. Activate .venv or pass --allow-system."
        )


def install(args: argparse.Namespace) -> int:
    require_supported_environment(allow_system=args.allow_system)
    requirements = args.repo_root / "WorldMem" / "requirements.txt"
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    print("Resolved repository:", args.repo_root)
    print("Interpreter:", sys.executable)
    print("Command:", " ".join(command))
    if args.dry_run:
        return 0
    result = subprocess.run(command, cwd=args.repo_root, check=False)
    if result.returncode != 0:
        return result.returncode
    return doctor(args)


def doctor(args: argparse.Namespace) -> int:
    navware = args.repo_root / "scripts" / "navware.py"
    command = [sys.executable, str(navware), "doctor", "--profile", "worldmem"]
    result = subprocess.run(command, cwd=args.repo_root, check=False)

    print("\nWorldMem source checks")
    worldmem_root = args.repo_root / "WorldMem"
    sys.path.insert(0, str(worldmem_root))
    source_failures = 0
    for module_name in (
        "algorithms.worldmem",
        "algorithms.worldmem.pose_prediction",
        "datasets.video.minecraft_video_dataset",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as error:
            source_failures += 1
            print(f"[FAIL] {module_name}: {type(error).__name__}: {error}")
        else:
            print(f"[PASS] {module_name}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"[{'PASS' if ffmpeg else 'FAIL'}] ffmpeg: {ffmpeg or 'not found on PATH'}")
    return 1 if result.returncode or source_failures or not ffmpeg else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=repo_root,
        default=DEFAULT_REPO_ROOT,
        help="NWM repository root (auto-detected by default)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    install_parser = subcommands.add_parser("install", help="install optional WorldMem packages")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--allow-system", action="store_true")
    install_parser.set_defaults(func=install)

    doctor_parser = subcommands.add_parser("doctor", help="run read-only WorldMem diagnostics")
    doctor_parser.set_defaults(func=doctor)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
