import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from scripts.evaluate_phase_a import structural_similarity, load_image_tensor


def test_structural_similarity_is_one_for_identical_images(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), color=(64, 128, 192)).save(image_path)
    tensor = load_image_tensor(image_path)

    assert structural_similarity(tensor, tensor) == pytest.approx(1.0, abs=1e-6)


def test_phase_a_eval_cli_matches_relative_prediction_inventory(tmp_path):
    gt_dir = tmp_path / "gt"
    recent_dir = tmp_path / "recent"
    oracle_dir = tmp_path / "oracle"
    for root in (gt_dir, recent_dir, oracle_dir):
        (root / "id_0").mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(gt_dir / "id_0" / "1.png")
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(recent_dir / "id_0" / "1.png")
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(
        oracle_dir / "id_0" / "1.png"
    )
    output = tmp_path / "metrics.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_phase_a.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gt-dir",
            str(gt_dir),
            "--run",
            f"recent={recent_dir}",
            "--run",
            f"oracle={oracle_dir}",
            "--metrics",
            "ssim",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["runs"]["oracle"]["metrics"]["ssim"]["mean"] == pytest.approx(1.0)
    assert (
        metrics["runs"]["recent"]["metrics"]["ssim"]["mean"]
        < metrics["runs"]["oracle"]["metrics"]["ssim"]["mean"]
    )
    assert output.with_suffix(".jsonl").is_file()
