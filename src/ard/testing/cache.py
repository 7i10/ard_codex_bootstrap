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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_sha(root: Path) -> str | None:
    checkout = root / ".external" / "saad"
    if not checkout.exists():
        return None
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


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
        "external_sha": external_sha(root),
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
