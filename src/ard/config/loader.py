"""YAML loading, environment expansion, dot overrides, and resolved snapshots."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import EvaluationConfig, ExperimentConfig

ENV_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass(frozen=True)
class EvaluationResolvedConfig:
    """Validated evaluation view plus immutable raw training lineage."""

    config: ExperimentConfig
    raw_config_hash: str
    migration: dict[str, object]


def _mapping_digest(value: Mapping[str, Any]) -> str:
    """Match the checkpoint digest while excluding operational W&B policy."""
    import hashlib

    canonical = deepcopy(dict(value))
    tracking = canonical.get("tracking")
    if isinstance(tracking, dict):
        tracking.pop("artifact_retention", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def load_evaluation_config(
    path: Path,
    *,
    training_config: ExperimentConfig,
    overrides: list[str] | tuple[str, ...] = (),
) -> ExperimentConfig:
    """Load a full evaluation config or a strict ``evaluation:`` overlay.

    The sibling resolved training config remains the only source of training
    identity.  A partial overlay may replace evaluation fields only; omitted
    fields retain the saved training evaluation contract.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    expanded = _expand_environment(raw)
    for override in overrides:
        _apply_override(expanded, override)
    if "schema_version" in expanded:
        return ExperimentConfig.model_validate(expanded)
    if set(expanded) != {"evaluation"} or not isinstance(expanded["evaluation"], dict):
        raise ValueError("partial evaluation config may contain only an evaluation mapping")
    evaluation = training_config.evaluation.model_dump(mode="json")
    evaluation.update(expanded["evaluation"])
    validated = EvaluationConfig.model_validate(evaluation)
    return training_config.model_copy(update={"evaluation": validated})


def load_resolved_config_for_evaluation(path: Path) -> EvaluationResolvedConfig:
    """Load historical resolved training lineage without reopening train support.

    Checkpoint identity is calculated from the untouched saved YAML mapping.
    Only the in-memory evaluation view is migrated: the former one-off
    logging-only method becomes ordinary RSLAD with explicit observations,
    obsolete gate metadata is discarded, and newer optional observation state
    defaults to ``off``.  Normal config loading remains strict.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("resolved training config must be a YAML mapping")
    raw_mapping = deepcopy(raw)
    raw_hash = _mapping_digest(raw_mapping)
    migrated = deepcopy(raw_mapping)
    method = migrated.get("method")
    if not isinstance(method, dict) or not isinstance(method.get("id"), str):
        raise ValueError("resolved training config has no method ID")
    source_method_id = method["id"]
    applied: list[str] = []
    if "research_design" in migrated:
        migrated.pop("research_design")
        applied.append("research_design_removed")
    if source_method_id == "rslad_logging_only":
        method["id"] = "rslad"
        observation = migrated.get("observation")
        if observation is not None and observation != {"profile": "teacher_response"}:
            raise ValueError("legacy rslad_logging_only resolved config has incompatible observation metadata")
        migrated["observation"] = {"profile": "teacher_response"}
        applied.append("rslad_logging_only_to_rslad_teacher_response")
    elif "observation" not in migrated:
        migrated["observation"] = {"profile": "off"}
        applied.append("observation_defaulted_off")
    config = ExperimentConfig.model_validate(migrated)
    return EvaluationResolvedConfig(
        config=config,
        raw_config_hash=raw_hash,
        migration={
            "source_method_id": source_method_id,
            "runtime_method_id": config.method.id,
            "applied": applied,
            "raw_mapping_hash": raw_hash,
        },
    )


def resolved_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(config.model_dump_json())


def save_resolved_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(resolved_config_dict(config), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)
