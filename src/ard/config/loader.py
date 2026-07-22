"""YAML loading, environment expansion, dot overrides, and resolved snapshots."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig

ENV_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value
    missing = {first or second for first, second in ENV_PATTERN.findall(value) if (first or second) not in os.environ}
    if missing:
        raise ValueError("missing environment variables: " + ", ".join(sorted(missing)))
    return os.path.expandvars(value)


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must have key=value form: {override!r}")
    dotted, raw_value = override.split("=", maxsplit=1)
    keys = dotted.split(".")
    if any(not key for key in keys):
        raise ValueError(f"invalid override path: {dotted!r}")
    target: dict[str, Any] = data
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: Path, overrides: list[str] | tuple[str, ...] = ()) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a YAML mapping")
    expanded = _expand_environment(raw)
    for override in overrides:
        _apply_override(expanded, override)
    return ExperimentConfig.model_validate(expanded)


def resolved_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(config.model_dump_json())


def save_resolved_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(resolved_config_dict(config), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)
