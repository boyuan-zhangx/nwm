#!/usr/bin/env python3
"""Evaluate Phase A policy outputs against ground-truth future frames.

Example:
    python scripts/evaluate_phase_a.py \
      --gt-dir /results/gt/phase_a/recon/time \
      --run recent=/results/nwm_cdit_s/phase_a/recent/.../recon/time \
      --run oracle=/results/nwm_cdit_s/phase_a/oracle_manifest/.../recon/time \
      --metrics ssim,lpips,dreamsim \
      --output /results/nwm_cdit_s/phase_a/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


SUPPORTED_METRICS = ("ssim", "lpips", "dreamsim")


def load_image_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def structural_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    """Compute RGB SSIM with a Gaussian window and image range [0, 1]."""

    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("SSIM inputs must have identical [B,C,H,W] shapes")
    _, channels, height, width = first.shape
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires images at least 3x3 pixels")

    coordinates = torch.arange(window_size, dtype=first.dtype, device=first.device)
    coordinates -= (window_size - 1) / 2
    sigma = 1.5 if window_size >= 11 else max(window_size / 6, 0.5)
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] @ gaussian[None, :]).expand(
        channels, 1, window_size, window_size
    )

    mean_first = functional.conv2d(first, window, groups=channels)
    mean_second = functional.conv2d(second, window, groups=channels)
    mean_first_sq = mean_first.square()
    mean_second_sq = mean_second.square()
    mean_product = mean_first * mean_second
    variance_first = functional.conv2d(first.square(), window, groups=channels) - mean_first_sq
    variance_second = (
        functional.conv2d(second.square(), window, groups=channels) - mean_second_sq
    )
    covariance = (
        functional.conv2d(first * second, window, groups=channels) - mean_product
    )

    constant_one = 0.01**2
    constant_two = 0.03**2
    score = (
        (2 * mean_product + constant_one)
        * (2 * covariance + constant_two)
        / (
            (mean_first_sq + mean_second_sq + constant_one)
            * (variance_first + variance_second + constant_two)
        )
    )
    return float(score.mean().clamp(-1, 1).item())


class MetricSuite:
    def __init__(self, metric_names: list[str], device: str):
        unknown = sorted(set(metric_names) - set(SUPPORTED_METRICS))
        if unknown:
            raise ValueError(f"unsupported metrics: {unknown}")
        self.metric_names = metric_names
        self.device = torch.device(device)
        self._lpips = None
        self._dreamsim = None
        self._dreamsim_preprocess: Callable | None = None

    def _load_lpips(self):
        if self._lpips is None:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError("LPIPS requested but the lpips package is unavailable") from error
            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()

    def _load_dreamsim(self):
        if self._dreamsim is None:
            try:
                from dreamsim import dreamsim
            except ImportError as error:
                raise RuntimeError(
                    "DreamSim requested but the dreamsim package is unavailable"
                ) from error
            self._dreamsim, self._dreamsim_preprocess = dreamsim(
                pretrained=True, device=str(self.device)
            )
            self._dreamsim.eval()

    @torch.no_grad()
    def evaluate_pair(self, ground_truth: Path, prediction: Path) -> dict[str, float]:
        result: dict[str, float] = {}
        first = load_image_tensor(ground_truth)
        second = load_image_tensor(prediction)
        if first.shape != second.shape:
            raise ValueError(
                f"image shape mismatch: {ground_truth} {tuple(first.shape)} vs "
                f"{prediction} {tuple(second.shape)}"
            )
        if "ssim" in self.metric_names:
            result["ssim"] = structural_similarity(first, second)
        if "lpips" in self.metric_names:
            self._load_lpips()
            assert self._lpips is not None
            distance = self._lpips(
                first.to(self.device) * 2 - 1, second.to(self.device) * 2 - 1
            )
            result["lpips"] = float(distance.mean().item())
        if "dreamsim" in self.metric_names:
            self._load_dreamsim()
            assert self._dreamsim is not None and self._dreamsim_preprocess is not None
            first_preprocessed = self._dreamsim_preprocess(
                Image.open(ground_truth).convert("RGB")
            ).to(self.device)
            second_preprocessed = self._dreamsim_preprocess(
                Image.open(prediction).convert("RGB")
            ).to(self.device)
            distance = self._dreamsim(first_preprocessed, second_preprocessed)
            result["dreamsim"] = float(distance.mean().item())
        return result


def image_inventory(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    inventory = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.png"))
        if path.is_file()
    }
    if not inventory:
        raise ValueError(f"no PNG predictions found under {root}")
    return inventory


def parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run must use LABEL=PATH syntax: {value}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in runs:
            raise ValueError(f"empty or duplicate run label: {label!r}")
        runs[label] = Path(raw_path).expanduser().resolve()
    return runs


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_runs(
    *, gt_dir: Path, runs: dict[str, Path], metric_names: list[str], device: str
) -> tuple[dict, list[dict]]:
    ground_truth = image_inventory(gt_dir)
    suite = MetricSuite(metric_names, device)
    summary = {
        "schema_version": 1,
        "ground_truth_dir": str(gt_dir),
        "metrics": metric_names,
        "runs": {},
    }
    details: list[dict] = []

    for label, run_dir in runs.items():
        predictions = image_inventory(run_dir)
        missing = sorted(set(ground_truth) - set(predictions))
        extra = sorted(set(predictions) - set(ground_truth))
        if missing or extra:
            raise ValueError(
                f"{label}: prediction inventory differs from ground truth; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        collected = {metric: [] for metric in metric_names}
        for relative_path, gt_path in ground_truth.items():
            values = suite.evaluate_pair(gt_path, predictions[relative_path])
            details.append(
                {"run": label, "relative_path": relative_path, **values}
            )
            for metric, value in values.items():
                collected[metric].append(value)
        summary["runs"][label] = {
            "path": str(run_dir),
            "count": len(ground_truth),
            "metrics": {
                metric: aggregate(metric_values)
                for metric, metric_values in collected.items()
            },
        }
    return summary, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument(
        "--run", action="append", required=True, help="repeatable LABEL=PATH run"
    )
    parser.add_argument("--metrics", default="ssim")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    if not metric_names:
        raise SystemExit("at least one metric is required")
    runs = parse_runs(args.run)
    summary, details = evaluate_runs(
        gt_dir=args.gt_dir.expanduser().resolve(),
        runs=runs,
        metric_names=metric_names,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    details_path = args.details_output or args.output.with_suffix(".jsonl")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as stream:
        for record in details:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
