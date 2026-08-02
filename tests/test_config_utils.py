from pathlib import Path

import pytest

from config_utils import deep_merge, load_config


def test_deep_merge_preserves_nested_defaults():
    merged = deep_merge(
        {"datasets": {"recon": {"goals_per_obs": 4, "data_folder": "old"}}},
        {"datasets": {"recon": {"data_folder": "new"}}},
    )
    assert merged == {
        "datasets": {"recon": {"goals_per_obs": 4, "data_folder": "new"}}
    }


def test_load_config_applies_path_overlay_and_expands_environment(tmp_path, monkeypatch):
    defaults = tmp_path / "defaults.yaml"
    experiment = tmp_path / "experiment.yaml"
    paths = tmp_path / "paths.yaml"
    defaults.write_text("datasets:\n  recon:\n    goals_per_obs: 4\n", encoding="utf-8")
    experiment.write_text("run_name: smoke\n", encoding="utf-8")
    paths.write_text("datasets:\n  recon:\n    data_folder: ${NAVWARE_DATA}/recon\n", encoding="utf-8")
    monkeypatch.setenv("NAVWARE_DATA", str(tmp_path / "data"))

    config = load_config(defaults, experiment, paths)

    assert config["datasets"]["recon"]["goals_per_obs"] == 4
    assert Path(config["datasets"]["recon"]["data_folder"]) == tmp_path / "data" / "recon"


def test_load_config_rejects_unresolved_environment_variable(tmp_path):
    defaults = tmp_path / "defaults.yaml"
    experiment = tmp_path / "experiment.yaml"
    defaults.write_text("results_dir: ${NOT_DEFINED}/results\n", encoding="utf-8")
    experiment.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="results_dir"):
        load_config(defaults, experiment)


@pytest.mark.parametrize(
    ("filename", "model", "run_name"),
    [
        ("nwm_cdit_s.yaml", "CDiT-S/2", "nwm_cdit_s"),
        ("nwm_cdit_b.yaml", "CDiT-B/2", "nwm_cdit_b"),
    ],
)
def test_phase_a_small_configs_are_frozen_and_four_frame(filename, model, run_name):
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(
        repo_root / "config" / "eval_config.yaml",
        repo_root / "config" / filename,
    )

    assert config["model"] == model
    assert config["run_name"] == run_name
    assert config["context_size"] == 4
    assert config["image_size"] == 224
    assert config["train"] is False
    assert config["phase_a_frozen"] is True
