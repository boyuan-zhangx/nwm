import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest

from retrieval_context import load_revisit_manifest, select_context_indices


def test_recent_preserves_native_four_frame_context():
    selection = select_context_indices(
        policy="recent", trajectory="loop", query_index=9
    )

    assert selection.context_indices == (6, 7, 8, 9)
    assert selection.selected_source_index == 6


def test_random_history_is_deterministic_and_outside_native_context():
    first = select_context_indices(
        policy="random_history",
        trajectory="loop",
        query_index=12,
        min_temporal_gap=6,
        positive_indices=[0, 1],
        seed=7,
    )
    second = select_context_indices(
        policy="random_history",
        trajectory="loop",
        query_index=12,
        min_temporal_gap=6,
        positive_indices=[0, 1],
        seed=7,
    )

    assert first == second
    assert first.selected_source_index <= 6
    assert first.selected_source_index not in {0, 1}
    assert first.context_indices[1:] == (10, 11, 12)


def test_oracle_uses_first_eligible_manifest_positive():
    selection = select_context_indices(
        policy="oracle_manifest",
        trajectory="loop",
        query_index=9,
        min_temporal_gap=5,
        positive_indices=[8, 1, 0],
    )

    assert selection.context_indices == (1, 7, 8, 9)


def test_pose_aligned_uses_position_and_wrapped_heading():
    positions = np.array(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 0.0], [5.0, 0.0], [0.0, 0.0]]
    )
    yaws = np.array([np.pi - 0.02, 0.0, 0.0, 0.0, -np.pi + 0.02])

    selection = select_context_indices(
        policy="pose_aligned",
        trajectory="loop",
        query_index=4,
        positions=positions,
        yaws=yaws,
        min_temporal_gap=4,
    )

    assert selection.selected_source_index == 0
    assert selection.heading_difference == pytest.approx(0.04)


def test_manifest_rejects_duplicate_queries(tmp_path):
    manifest = tmp_path / "duplicates.jsonl"
    row = {"trajectory": "loop", "query_index": 4, "positive_indices": [0]}
    manifest.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate query"):
        load_revisit_manifest(manifest)


def test_revisit_dataset_loads_oracle_frame_into_oldest_slot(tmp_path):
    pytest.importorskip("matplotlib")
    from torchvision import transforms

    from datasets import RevisitEvalDataset

    data_root = tmp_path / "data"
    trajectory_root = data_root / "loop"
    split_root = tmp_path / "split"
    trajectory_root.mkdir(parents=True)
    split_root.mkdir()
    (split_root / "traj_names.txt").write_text("loop\n", encoding="utf-8")

    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
        ]
    )
    with (trajectory_root / "traj_data.pkl").open("wb") as stream:
        pickle.dump({"position": positions, "yaw": np.zeros(len(positions))}, stream)
    for index in range(len(positions)):
        Image.new("RGB", (8, 8), color=(index * 20, 0, 0)).save(
            trajectory_root / f"{index}.jpg"
        )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "trajectory": "loop",
                "query_index": 6,
                "positive_indices": [0],
                "min_temporal_gap": 4,
                "label_source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = RevisitEvalDataset(
        data_folder=str(data_root),
        data_split_folder=str(split_root),
        dataset_name="recon",
        image_size=8,
        min_dist_cat=-8,
        max_dist_cat=8,
        len_traj_pred=1,
        traj_stride=1,
        context_size=4,
        transform=transforms.ToTensor(),
        manifest_path=str(manifest),
        context_policy="oracle_manifest",
        selection_seed=0,
        normalize=True,
    )

    _, observations, future, delta = dataset[0]
    diagnostic = dataset.get_diagnostic(0)

    assert diagnostic["context_indices"] == (0, 4, 5, 6)
    assert observations.shape == (4, 3, 8, 8)
    assert float(observations[0, 0].mean()) == pytest.approx(0.0, abs=0.01)
    assert float(observations[-1, 0].mean()) > 0.4
    assert future.shape[0] == 1
    assert delta.shape == (1, 3)
