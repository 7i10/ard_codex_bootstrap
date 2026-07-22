"""Strict, standalone configuration for bounded RobustBench teacher audits.

This is intentionally not an :class:`ExperimentConfig`: it never constructs a
student, optimizer, tracker, or training state.  The only permitted result is
a local clean/PGD screening measurement for one hash-registered teacher.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from .schema import AttackConfig, StrictModel

_ENV_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_APPROVED_TEACHERS = frozenset({"chen2021_ltd_wrn34_10", "bartoldson2024_adversarial_wrn94_16"})


class TeacherAuditTeacherConfig(StrictModel):
    """Only a project-owned registry ID may identify an audited teacher."""

    registry_id: Literal["chen2021_ltd_wrn34_10", "bartoldson2024_adversarial_wrn94_16"]


class TeacherAuditDatasetConfig(StrictModel):
    """The official, already-present CIFAR-10 test split only."""

    name: Literal["cifar10"]
    root: Path
    split: Literal["test"]
    download: Literal[False]


class TeacherAuditRunConfig(StrictModel):
    """Bounded execution identity, separate from a training tier."""

    max_samples: int = Field(ge=10)
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    output_dir: Path


class TeacherAuditConfig(StrictModel):
    """Exact local-only teacher screening contract."""

    schema_version: Literal[1]
    teacher: TeacherAuditTeacherConfig
    dataset: TeacherAuditDatasetConfig
    run: TeacherAuditRunConfig
    attack: AttackConfig

    @model_validator(mode="after")
    def validate_contract(self) -> TeacherAuditConfig:
        if self.teacher.registry_id not in _APPROVED_TEACHERS:
            raise ValueError("teacher audit registry ID is not approved")
        expected = {
            "norm": "linf",
            "input_domain": "pixel_0_1",
            "epsilon": "8/255",
            "step_size": "2/255",
            "steps": 20,
            "random_start": True,
            "loss": "ce",
            "kl_target": None,
            "temperature": 1.0,
            "temperature_squared": True,
            "student_mode": "eval",
            "teacher_mode": "eval",
            "trace_step_losses": False,
        }
        mismatches = [key for key, value in expected.items() if getattr(self.attack, key) != value]
        if mismatches:
            raise ValueError(
                "teacher audit requires exact PGD-20 hard-label CE identity; mismatched: " + ", ".join(mismatches)
            )
        return self


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value
    missing = {first or second for first, second in _ENV_PATTERN.findall(value) if (first or second) not in os.environ}
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
    target = data
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[keys[-1]] = yaml.safe_load(raw_value)


def load_teacher_audit_config(path: Path, overrides: list[str] | tuple[str, ...] = ()) -> TeacherAuditConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("teacher audit config must be a YAML mapping")
    expanded = _expand_environment(raw)
    for override in overrides:
        _apply_override(expanded, override)
    return TeacherAuditConfig.model_validate(expanded)


def resolved_teacher_audit_config(config: TeacherAuditConfig) -> dict[str, Any]:
    """Return JSON-safe resolved values, including rational attack quantities."""
    return json.loads(config.model_dump_json())


def atomic_write_text(path: Path, content: str) -> None:
    """Durably replace a small local audit artifact without partial results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_resolved_teacher_audit_config(config: TeacherAuditConfig, path: Path) -> None:
    atomic_write_text(path, yaml.safe_dump(resolved_teacher_audit_config(config), sort_keys=False))
