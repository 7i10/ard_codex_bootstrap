"""Append-only pass cache keyed by every reproducibility-relevant test input."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkout_identity(checkout: Path) -> dict[str, str | None]:
    if not checkout.exists():
        return {"head": None, "origin": None, "dirty": None}
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    head = completed.stdout.strip() if completed.returncode == 0 else None
    if head is None:
        return {"head": None, "origin": None, "dirty": None}
    origin = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return {
        "head": head,
        "origin": origin.stdout.strip() if origin.returncode == 0 else None,
        "dirty": "true" if dirty.returncode == 0 and bool(dirty.stdout) else "false" if dirty.returncode == 0 else None,
    }


def external_identity(root: Path) -> dict[str, dict[str, str | None]]:
    """Describe every repository declared by the lock, in deterministic name order."""
    lock_path = root / "external.lock.yaml"
    try:
        raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError):
        return {}
    repositories = raw.get("repositories") if isinstance(raw, dict) else None
    if not isinstance(repositories, dict):
        return {}
    identity: dict[str, dict[str, str | None]] = {}
    for name in sorted(repositories):
        entry = repositories[name]
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        identity[name] = {
            "locked_commit": entry.get("commit") if isinstance(entry.get("commit"), str) else None,
            "locked_url": entry.get("url") if isinstance(entry.get("url"), str) else None,
            **_checkout_identity(root / ".external" / name),
        }
    return identity


def external_sha(root: Path) -> str | None:
    """Backward-compatible access to the historical SAAD checkout SHA."""
    return external_identity(root).get("saad", {}).get("head")


def environment_identity() -> dict[str, object]:
    """Describe the runtime that can affect a numerical or CUDA test result."""
    identity: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "gpus": [],
    }
    try:
        import torch

        identity["torch"] = str(torch.__version__)
        cuda_version = torch.version.cuda
        identity["cuda"] = None if cuda_version is None else str(cuda_version)
        if torch.cuda.is_available():
            gpus: list[dict[str, object]] = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                uuid = getattr(properties, "uuid", None)
                gpus.append(
                    {
                        "index": int(index),
                        "name": str(properties.name),
                        "capability": [int(properties.major), int(properties.minor)],
                        "uuid": None if uuid is None else str(uuid),
                    }
                )
            identity["gpus"] = gpus
    except ImportError:
        pass
    return identity


def _hash_paths(root: Path, paths: Iterable[str], *, prefixes: tuple[str, ...]) -> dict[str, str]:
    """Hash declared test inputs without accidentally traversing a checkout."""
    result: dict[str, str] = {}
    for relative in sorted(set(paths)):
        if prefixes and not relative.startswith(prefixes):
            continue
        path = root / relative
        result[relative] = sha256_file(path) if path.is_file() else "missing"
    return result


def fingerprint(
    *,
    root: Path,
    command: tuple[str, ...],
    relevant_paths: Iterable[str],
    environment: Mapping[str, str] | None = None,
    fixture_version: str = "1",
    seed: str = "0",
) -> str:
    declared_paths = tuple(sorted(set(relevant_paths)))
    files = _hash_paths(root, declared_paths, prefixes=())
    payload = {
        # The argv vector is intentional: shell-normalized strings can merge
        # materially different commands such as marker expressions.
        "exact_command": command,
        "environment": dict(sorted((environment or {}).items())),
        "test_hashes": _hash_paths(root, declared_paths, prefixes=("tests/",)),
        "source_config_hashes": _hash_paths(root, declared_paths, prefixes=("src/", "configs/", "pyproject.toml")),
        "other_input_hashes": files,
        "runtime": environment_identity(),
        "external_identity": external_identity(root),
        "fixture_version": fixture_version,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CacheRecord:
    fingerprint: str
    command: tuple[str, ...]
    status: str


class PassCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[CacheRecord]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                result.append(CacheRecord(item["fingerprint"], tuple(item["command"]), item["status"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        return result

    def has_pass(self, key: str) -> bool:
        for record in reversed(self.records()):
            if record.fingerprint == key:
                return record.status == "passed"
        return False

    def failed_commands(self) -> tuple[tuple[str, ...], ...]:
        latest: dict[tuple[str, ...], CacheRecord] = {}
        for record in self.records():
            latest[record.command] = record
        return tuple(command for command, record in latest.items() if record.status == "failed")

    def append(self, record: CacheRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"fingerprint": record.fingerprint, "command": list(record.command), "status": record.status},
                    sort_keys=True,
                )
                + "\n"
            )
