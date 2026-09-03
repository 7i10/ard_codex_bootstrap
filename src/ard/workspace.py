"""Tracked workspace contract for operational ARD paths.

This module owns *future runtime-write* locations only.  Historical artifact
roots remain readable because reports and frozen manifests may legitimately
refer to them by absolute path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUNTIME_FIELDS = (
    "runtime_root",
    "run_root",
    "analysis_root",
    "staging_root",
    "worktree_root",
    "orchestration_root",
    "task_context_root",
    "lock_root",
    "temp_root",
)
REQUIRED_FIELDS = (
    "repo_root",
    "dataset_root",
    "ard_dataset_root",
    "imagenet_root",
    *RUNTIME_FIELDS,
    "python",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_registry_path() -> Path:
    return repository_root() / "configs" / "workspace" / "ard_workspace_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class WorkspaceContract:
    """Validated, resolved view of one tracked workspace registry."""

    registry_path: Path
    registry_sha256: str
    values: dict[str, Any]

    def path(self, field: str) -> Path:
        value = self.values[field]
        if not isinstance(value, str):  # guarded by ``load_workspace_contract``
            raise ValueError(f"workspace field is not a path: {field}")
        return Path(value).resolve(strict=False)

    @property
    def runtime_root(self) -> Path:
        return self.path("runtime_root")

    @property
    def runtime_paths(self) -> dict[str, Path]:
        return {field: self.path(field) for field in RUNTIME_FIELDS}

    def ensure_runtime_layout(self) -> tuple[Path, ...]:
        """Create exactly the registered runtime layout, and nothing else."""
        created: list[Path] = []
        for path in self.runtime_paths.values():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        return tuple(created)

    def require_runtime_write(self, value: str | Path) -> Path:
        """Reject a future ARD write outside the canonical runtime root."""
        path = Path(value).resolve(strict=False)
        if not _is_relative_to(path, self.runtime_root):
            raise ValueError(
                f"future ARD runtime write must be beneath {self.runtime_root}, got {path}"
            )
        return path


def load_workspace_contract(path: Path | None = None) -> WorkspaceContract:
    registry_path = (path or default_registry_path()).resolve()
    try:
        values = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read workspace registry {registry_path}: {exc}") from exc
    if not isinstance(values, dict) or values.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{registry_path}: schema_version must be {SCHEMA_VERSION}")
    missing = [field for field in REQUIRED_FIELDS if not isinstance(values.get(field), str) or not values[field]]
    if missing:
        raise ValueError(f"{registry_path}: missing required workspace paths: {', '.join(missing)}")
    for field in REQUIRED_FIELDS:
        if not Path(str(values[field])).is_absolute():
            raise ValueError(f"{registry_path}: {field} must be an absolute path")
    runtime_root = Path(str(values["runtime_root"])).resolve(strict=False)
    for field in RUNTIME_FIELDS:
        candidate = Path(str(values[field])).resolve(strict=False)
        if not _is_relative_to(candidate, runtime_root):
            raise ValueError(f"{registry_path}: {field} must be beneath runtime_root")
    hosts = values.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError(f"{registry_path}: hosts must be a non-empty mapping")
    for name, profile in hosts.items():
        if not isinstance(name, str) or not isinstance(profile, dict):
            raise ValueError(f"{registry_path}: invalid host profile")
        if not isinstance(profile.get("hostname"), str) or not profile["hostname"]:
            raise ValueError(f"{registry_path}: {name}.hostname is required")
        if profile.get("execution_class") not in {"local", "external"}:
            raise ValueError(f"{registry_path}: {name}.execution_class is invalid")
    return WorkspaceContract(registry_path=registry_path, registry_sha256=sha256(registry_path), values=values)
