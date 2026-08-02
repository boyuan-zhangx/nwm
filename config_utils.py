"""Shared YAML configuration loading with deep overlays and env expansion."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import yaml


ENV_PATTERN = re.compile(r"\$\{[^}]+\}")


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; non-mapping overlay values replace base values."""

    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return value


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _unresolved(value: Any, prefix: str = "") -> list[str]:
    missing = []
    if isinstance(value, str) and ENV_PATTERN.search(value):
        missing.append(prefix or "<root>")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            missing.extend(_unresolved(item, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            missing.extend(_unresolved(item, child))
    return missing


def load_config(
    defaults: Path | str,
    experiment: Path | str,
    paths: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Load defaults, experiment, then optional machine-specific path overlay."""

    config = deep_merge(_read_yaml(Path(defaults)), _read_yaml(Path(experiment)))
    if paths is not None:
        config = deep_merge(config, _read_yaml(Path(paths)))
    config = _expand(config)
    unresolved = _unresolved(config)
    if unresolved:
        raise ValueError(
            "unresolved environment variables in config keys: " + ", ".join(unresolved)
        )
    return config
