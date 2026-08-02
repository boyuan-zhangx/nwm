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
