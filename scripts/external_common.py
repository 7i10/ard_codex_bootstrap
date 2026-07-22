"""Shared, deliberately small helpers for pinned external repositories."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOCKED_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md", "COPYING.txt")


class ExternalError(RuntimeError):
    """A local external repository does not satisfy its recorded contract."""


@dataclass(frozen=True)
class LockedRepository:
    name: str
    url: str
    commit: str
    fetched_at: str | None
    license_file: str | None
    license_status: str
    license_evidence: dict[str, str] | None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        raise ExternalError(completed.stderr.strip() or "git command failed: " + " ".join(args))
    return completed.stdout.strip()


def load_lock(path: Path) -> tuple[dict[str, Any], LockedRepository]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExternalError(f"external lock is missing: {path}") from exc
    except yaml.YAMLError as exc:
        raise ExternalError(f"external lock is invalid YAML: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ExternalError("external lock must have version: 1")
    entry = raw.get("repositories", {}).get("saad")
    if not isinstance(entry, dict):
        raise ExternalError("external lock must define repositories.saad")
    url, commit = entry.get("url"), entry.get("commit")
    if not isinstance(url, str) or not url:
        raise ExternalError("repositories.saad.url must be a non-empty string")
    if not isinstance(commit, str) or not LOCKED_SHA_RE.fullmatch(commit):
        raise ExternalError("repositories.saad.commit must be an exact lowercase 40-character SHA")
    status = entry.get("license_status", "unclear")
    if status not in {"verified", "absent", "unclear"}:
        raise ExternalError("license_status must be verified, absent, or unclear")
    license_file = entry.get("license_file")
    evidence = entry.get("license_evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise ExternalError("license_evidence must be a mapping or null")
    has_file = isinstance(license_file, str) and bool(license_file)
    digest = evidence.get("sha256") if isinstance(evidence, dict) else None
    has_evidence = isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest))
    if has_file != has_evidence:
        raise ExternalError("license_file and valid sha256 license_evidence must both be present or both be null")
    if license_file is not None and not has_file:
        raise ExternalError("license_file must be a non-empty string or null")
    if evidence is not None and not has_evidence:
        raise ExternalError("license_evidence.sha256 must be an exact lowercase 64-character digest")
    if status == "verified" and not has_file:
        raise ExternalError("verified license status requires a license file and sha256 evidence")
    if status == "absent" and (has_file or has_evidence):
        raise ExternalError("absent license status requires null file and evidence")
    return raw, LockedRepository(
        name="saad",
        url=url,
        commit=commit,
        fetched_at=entry.get("fetched_at"),
        license_file=license_file,
        license_status=status,
        license_evidence=evidence,
    )


def write_lock(path: Path, raw: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def validate_checkout(path: Path, locked: LockedRepository) -> None:
    if not path.is_dir():
        raise ExternalError(f"external checkout is missing: {path}")
    if git(["rev-parse", "--is-inside-work-tree"], cwd=path) != "true":
        raise ExternalError(f"external path is not a Git work tree: {path}")
    remote = git(["remote", "get-url", "origin"], cwd=path)
    if remote != locked.url:
        raise ExternalError(f"origin mismatch for {path}: expected {locked.url!r}, got {remote!r}")
    head = git(["rev-parse", "HEAD"], cwd=path)
    if head != locked.commit:
        raise ExternalError(f"HEAD mismatch for {path}: expected {locked.commit}, got {head}")
    if git(["status", "--porcelain", "--untracked-files=all"], cwd=path):
        raise ExternalError(f"external checkout is dirty and will not be overwritten: {path}")


def license_evidence(path: Path) -> tuple[str | None, dict[str, str] | None]:
    for relative in LICENSE_NAMES:
        candidate = path / relative
        if candidate.is_file():
            return relative, {"sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    return None, None
