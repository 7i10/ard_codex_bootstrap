#!/usr/bin/env python3
"""Stage and validate canonical, identity-bound collected artifacts.

This utility deliberately has no SSH or experiment logic.  A remote executor
first places bytes in a local staging path; this command copies them to the
declared canonical path, verifies their SHA-256, and rejects incomplete or
foreign inventory cells before an aggregation command consumes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
IDENTITY_FIELDS = (
    "campaign_id",
    "source_sha",
    "job_id",
    "seed",
    "arm",
    "epoch",
    "split",
    "attack_identity",
)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(char in "0123456789abcdefABCDEF" for char in value)


def resolve(value: Any, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read inventory manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("inventory manifest must be a mapping")
    return payload


def issue(errors: list[dict[str, Any]], index: int | None, field: str, expected: Any, observed: Any) -> None:
    errors.append({"artifact_index": index, "field": field, "expected": expected, "observed": observed})


def identity_key(identity: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(identity.get(field) for field in IDENTITY_FIELDS)


def inspect_manifest(payload: dict[str, Any], base: Path, *, require_collected: bool) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    campaign_id = payload.get("campaign_id")
    source_sha = payload.get("source_sha")
    if payload.get("schema_version") != SCHEMA_VERSION:
        issue(errors, None, "schema_version", SCHEMA_VERSION, payload.get("schema_version"))
    if not isinstance(campaign_id, str) or not campaign_id:
        issue(errors, None, "campaign_id", "non-empty string", campaign_id)
    if not is_sha(source_sha, 40):
        issue(errors, None, "source_sha", "40-hex SHA", source_sha)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        issue(errors, None, "artifacts", "non-empty list", artifacts)
        artifacts = []
    seen: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            issue(errors, index, "artifact", "mapping", raw)
            continue
        origin = raw.get("origin")
        if (
            not isinstance(origin, dict)
            or not isinstance(origin.get("host"), str)
            or not isinstance(origin.get("path"), str)
        ):
            issue(errors, index, "origin", "{host,path}", origin)
        identity = raw.get("identity")
        if not isinstance(identity, dict):
            issue(errors, index, "identity", f"fields {IDENTITY_FIELDS}", identity)
            continue
        missing = [field for field in IDENTITY_FIELDS if field not in identity]
        if missing:
            issue(errors, index, "identity", f"fields {IDENTITY_FIELDS}", {"missing": missing})
        if identity.get("campaign_id") != campaign_id:
            issue(errors, index, "identity.campaign_id", campaign_id, identity.get("campaign_id"))
        if identity.get("source_sha") != source_sha:
            issue(errors, index, "identity.source_sha", source_sha, identity.get("source_sha"))
        key = identity_key(identity)
        if key in seen:
            issue(errors, index, "identity", "unique inventory cell", {"duplicates": [seen[key], index]})
        else:
            seen[key] = index
        expected_sha = raw.get("sha256")
        if not is_sha(expected_sha, 64):
            issue(errors, index, "sha256", "64-hex SHA", expected_sha)
        collected = resolve(raw.get("collected_path"), base)
        if collected is None:
            issue(errors, index, "collected_path", "path", raw.get("collected_path"))
        elif require_collected:
            if not collected.is_file():
                issue(errors, index, "collected_path", "existing local file", str(collected))
            elif is_sha(expected_sha, 64) and digest(collected) != str(expected_sha).lower():
                issue(errors, index, "collected_path.sha256", str(expected_sha).lower(), digest(collected))
        rows.append(
            {
                "identity": identity,
                "origin_host": origin.get("host") if isinstance(origin, dict) else None,
                "origin_path": origin.get("path") if isinstance(origin, dict) else None,
                "collected_path": str(collected) if collected else None,
                "expected_sha256": expected_sha,
                "observed_sha256": digest(collected)
                if require_collected and collected and collected.is_file()
                else None,
            }
        )
    required = payload.get("required_cells", [])
    if not isinstance(required, list):
        issue(errors, None, "required_cells", "list", required)
        required = []
    actual_keys = set(seen)
    required_keys: set[tuple[Any, ...]] = set()
    for index, identity in enumerate(required):
        if not isinstance(identity, dict) or any(field not in identity for field in IDENTITY_FIELDS):
            issue(
                errors,
                None,
                "required_cells",
                f"identity fields {IDENTITY_FIELDS}",
                {"index": index, "value": identity},
            )
            continue
        if identity.get("campaign_id") != campaign_id or identity.get("source_sha") != source_sha:
            issue(
                errors,
                None,
                "required_cells.identity",
                {"campaign_id": campaign_id, "source_sha": source_sha},
                identity,
            )
        key = identity_key(identity)
        if key in required_keys:
            issue(errors, None, "required_cells", "unique cells", {"duplicate_index": index})
        required_keys.add(key)
    if required_keys:
        missing = sorted(required_keys - actual_keys, key=repr)
        unexpected = sorted(actual_keys - required_keys, key=repr)
        if missing:
            issue(errors, None, "required_cells", "all required inventory cells", {"missing": missing})
        if unexpected:
            issue(errors, None, "required_cells", "no unexpected inventory cells", {"unexpected": unexpected})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "campaign_id": campaign_id,
        "source_sha": source_sha,
        "require_collected": require_collected,
        "artifacts": rows,
        "errors": errors,
    }


def stage(payload: dict[str, Any], base: Path) -> dict[str, Any]:
    report = inspect_manifest(payload, base, require_collected=False)
    if report["status"] != "pass":
        return report
    errors: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["artifacts"]):
        staging = resolve(raw.get("staging_path"), base)
        collected = resolve(raw.get("collected_path"), base)
        if staging is None or not staging.is_file():
            issue(errors, index, "staging_path", "existing locally collected file", str(staging))
            continue
        if collected is None:
            issue(errors, index, "collected_path", "path", raw.get("collected_path"))
            continue
        expected = str(raw.get("sha256", "")).lower()
        observed_staging = digest(staging)
        if not is_sha(expected, 64) or observed_staging != expected:
            issue(errors, index, "staging_path.sha256", expected, observed_staging)
            continue
        collected.parent.mkdir(parents=True, exist_ok=True)
        if collected.exists():
            observed_collected = digest(collected) if collected.is_file() else None
            if observed_collected != expected:
                issue(errors, index, "collected_path", "absent or matching immutable artifact", observed_collected)
            continue
        # The final path is intentionally untouched until the complete staged
        # copy has been fsync'd and hash-verified.  The temporary lives in the
        # target directory, so os.replace is a same-filesystem atomic commit.
        temporary = collected.with_name(f".{collected.name}.{os.getpid()}.tmp")
        try:
            with staging.open("rb") as source, temporary.open("xb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            observed_temporary = digest(temporary)
            if observed_temporary != expected:
                issue(errors, index, "temporary_path.sha256", expected, observed_temporary)
                temporary.unlink(missing_ok=True)
                continue
            os.replace(temporary, collected)
        finally:
            temporary.unlink(missing_ok=True)
    if errors:
        report["status"] = "fail"
        report["errors"].extend(errors)
        return report
    return inspect_manifest(payload, base, require_collected=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "stage", "validate"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        payload = load(manifest_path)
        if args.command == "stage":
            report = stage(payload, manifest_path.parent)
        else:
            report = inspect_manifest(payload, manifest_path.parent, require_collected=args.command == "validate")
    except ValueError as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "errors": [{"field": "manifest", "observed": str(exc)}],
        }
    if args.report is not None:
        atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
