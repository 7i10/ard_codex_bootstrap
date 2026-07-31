"""Read-only, hash-bound exports of logging-only SampleStateStore anchors.

This module deliberately prepares no predictor and computes no scores.  It
only turns the immutable epoch-99 and epoch-199 state payloads into a stable
sample-ID matrix for the separately frozen prediction implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class LoggingOnlyStateError(ValueError):
    """Raised when a logging-only state pair cannot prove its identity."""


@dataclass(frozen=True)
class LoggingOnlyStateAnalysis:
    """Immutable source identity and stable-ID rows for an external matrix."""

    identity: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LoggingOnlyStateError("sample state cannot be canonically hash-bound") from exc
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_clean_analysis_provenance() -> dict[str, Any]:
    """Hash-bind the tracked-clean implementation that defines this export."""
    root = Path(__file__).resolve().parents[3]
    paths = {
        "analysis": Path(__file__).resolve(),
        "cli": root / "src/ard/cli/logging_only_state.py",
        "sample_store": root / "src/ard/state/sample_store.py",
    }
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise LoggingOnlyStateError("logging-only analysis requires tracked source files") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha) or dirty:
        raise LoggingOnlyStateError("logging-only analysis requires a tracked-clean Git revision")
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    return {"git_sha": sha, "dirty": False, "source_files": hashes, "source_sha256": _canonical_sha256(hashes)}


def _validate_analysis_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    git_sha = value.get("git_sha")
    source_files = value.get("source_files")
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(character not in "0123456789abcdef" for character in git_sha)
        or value.get("dirty") is not False
        or not isinstance(source_files, Mapping)
        or not source_files
    ):
        raise LoggingOnlyStateError("logging-only analysis provenance is incomplete")
    hashes = dict(source_files)
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes.values()
    ):
        raise LoggingOnlyStateError("logging-only analysis source hashes are invalid")
    expected = _canonical_sha256(hashes)
    if value.get("source_sha256") != expected:
        raise LoggingOnlyStateError("logging-only analysis source aggregate hash is invalid")
    return {"git_sha": git_sha, "dirty": False, "source_files": hashes, "source_sha256": expected}


def _load_checkpoint(path: Path, *, expected_epoch: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise LoggingOnlyStateError(f"checkpoint is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise LoggingOnlyStateError(f"checkpoint is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise LoggingOnlyStateError("checkpoint payload must be a mapping")
    if payload.get("epoch") != expected_epoch:
        raise LoggingOnlyStateError(f"checkpoint epoch must be exactly {expected_epoch}")
    if not isinstance(payload.get("config_hash"), str) or not payload["config_hash"]:
        raise LoggingOnlyStateError("checkpoint lacks a config hash")
    if not isinstance(payload.get("tracker_run_id"), str) or not payload["tracker_run_id"]:
        raise LoggingOnlyStateError("checkpoint lacks a tracking run ID")
    if not isinstance(payload.get("world_size"), int) or isinstance(payload["world_size"], bool):
        raise LoggingOnlyStateError("checkpoint lacks a valid world size")
    state = payload.get("sample_state")
    if not isinstance(state, Mapping) or state.get("format_version") != 3:
        raise LoggingOnlyStateError("checkpoint must contain format-v3 logging-only sample state")
    if state.get("pending") != []:
        raise LoggingOnlyStateError("epoch-boundary logging-only sample state must not retain pending observations")
    records = state.get("records")
    if not isinstance(records, Mapping) or not records:
        raise LoggingOnlyStateError("logging-only sample state must contain records")
    materialized = dict(state)
    if any(not isinstance(record, Mapping) for record in records.values()):
        raise LoggingOnlyStateError("logging-only sample records must be mappings")
    return dict(payload), materialized


def _manifest_identity(
    path: Path,
    *,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Verify both checkpoints are canonical versioned model artifacts."""
    if not path.is_file():
        raise LoggingOnlyStateError(f"run-bundle manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoggingOnlyStateError("run-bundle manifest is unreadable JSON") from exc
    if not isinstance(manifest, Mapping):
        raise LoggingOnlyStateError("run-bundle manifest must be a mapping")
    if manifest.get("status") not in {"completed", "sync_pending"}:
        raise LoggingOnlyStateError("run-bundle manifest is not terminal-success")
    completion_path = path.parent / "completion.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoggingOnlyStateError("run-bundle completion marker is missing or unreadable") from exc
    if not isinstance(completion, Mapping) or completion.get("status") not in {"completed", "sync_pending"}:
        raise LoggingOnlyStateError("run-bundle completion marker is not terminal-success")
    for field in ("run_id", "config_hash", "world_size"):
        expected = checkpoint_payload["tracker_run_id"] if field == "run_id" else checkpoint_payload[field]
        if manifest.get(field) != expected:
            raise LoggingOnlyStateError(f"run-bundle manifest {field} does not match checkpoint identity")
    git = manifest.get("git")
    if (
        not isinstance(git, Mapping)
        or not isinstance(git.get("sha"), str)
        or len(git["sha"]) != 40
        or any(character not in "0123456789abcdef" for character in git["sha"])
        or git.get("dirty") is not False
    ):
        raise LoggingOnlyStateError("run-bundle manifest lacks a clean scientific Git SHA")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise LoggingOnlyStateError("run-bundle manifest lacks its artifact inventory")
    matched: dict[str, dict[str, Any]] = {}
    for role, digest in checkpoint_sha256.items():
        candidates = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            and artifact.get("type") == "model"
            and artifact.get("sha256") == digest
            and isinstance(artifact.get("aliases"), list)
            and "last" in artifact["aliases"]
        ]
        if len(candidates) != 1:
            raise LoggingOnlyStateError(f"{role} checkpoint is not one exact last-model artifact in the run bundle")
        matched[role] = {
            "artifact_name": candidates[0].get("name"),
            "artifact_local_path": candidates[0].get("local_path"),
            "sha256": digest,
        }
    return {
        "run_bundle_manifest_sha256": _file_sha256(path),
        "run_bundle_completion_sha256": _file_sha256(completion_path),
        "scientific_git_sha": git["sha"],
        "checkpoint_artifacts": matched,
    }


def _logging_only_record(record: Mapping[str, Any]) -> None:
    required = {
        "true_label",
        "last_margin",
        "margin_ema",
        "robust_correct_count",
        "seen",
        "forgetting_count",
        "previous_robust_correct",
        "first_robustly_learned_epoch",
        "current_correct_streak",
        "longest_correct_streak",
        "margin_mean",
        "margin_m2",
        "margin_time_sum",
        "margin_time_squared_sum",
        "margin_time_margin_sum",
        "history_statistics_complete",
        "teacher_clean_entropy",
        "teacher_adversarial_entropy",
        "teacher_clean_to_adversarial_margin_response",
        "teacher_clean_to_adversarial_js_response",
    }
    if not required.issubset(record):
        raise LoggingOnlyStateError("sample record does not satisfy the logging-only v3 primitive contract")
    if record["true_label"] is None or isinstance(record["true_label"], bool):
        raise LoggingOnlyStateError("logging-only sample record lacks its stable true label")
    if record["history_statistics_complete"] is not True:
        raise LoggingOnlyStateError("logging-only sample history was migrated from an incomplete legacy state")
    teacher_fields = (
        "teacher_clean_entropy",
        "teacher_adversarial_entropy",
        "teacher_clean_to_adversarial_margin_response",
        "teacher_clean_to_adversarial_js_response",
    )
    if any(record[field] is None for field in teacher_fields):
        raise LoggingOnlyStateError("sample record lacks logging-only teacher response primitives")
    finite_fields = (
        "last_margin",
        "margin_ema",
        "margin_mean",
        "margin_m2",
        "margin_time_sum",
        "margin_time_squared_sum",
        "margin_time_margin_sum",
        *teacher_fields,
    )
    for field in finite_fields:
        if (
            isinstance(record[field], bool)
            or not isinstance(record[field], (int, float))
            or not math.isfinite(float(record[field]))
        ):
            raise LoggingOnlyStateError("sample record has non-finite logging-only primitives")


def _exact_counter(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoggingOnlyStateError(f"sample state {field} must be an exact integer")
    return value


def logging_only_state_analysis(
    *,
    anchor_checkpoint: Path,
    final_checkpoint: Path,
    expected_count: int = 45000,
    run_bundle_manifest: Path,
    analysis_provenance: Mapping[str, Any] | None = None,
    anchor_epoch: int = 99,
    final_epoch: int = 199,
) -> LoggingOnlyStateAnalysis:
    """Hash-bind exact anchor/final states and export an ID-keyed raw matrix."""
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise LoggingOnlyStateError("expected sample count must be a positive integer")
    if anchor_epoch != 99 or final_epoch != 199:
        raise LoggingOnlyStateError("the frozen logging-only analysis requires anchor epoch 99 and final epoch 199")
    anchor_payload, anchor_state = _load_checkpoint(anchor_checkpoint, expected_epoch=anchor_epoch)
    final_payload, final_state = _load_checkpoint(final_checkpoint, expected_epoch=final_epoch)
    anchor_checkpoint_sha256 = _file_sha256(anchor_checkpoint)
    final_checkpoint_sha256 = _file_sha256(final_checkpoint)
    identity_keys = ("config_hash", "tracker_run_id", "world_size")
    if any(anchor_payload[key] != final_payload[key] for key in identity_keys):
        raise LoggingOnlyStateError("anchor and final checkpoints do not share one execution identity")
    anchor_records = anchor_state["records"]
    final_records = final_state["records"]
    assert isinstance(anchor_records, Mapping) and isinstance(final_records, Mapping)
    if set(anchor_records) != set(final_records):
        raise LoggingOnlyStateError("anchor and final sample states do not cover the same stable IDs")
    if len(anchor_records) != expected_count:
        raise LoggingOnlyStateError(
            f"logging-only sample-state record count is {len(anchor_records)}, expected exactly {expected_count}"
        )
    rows: list[dict[str, Any]] = []
    for raw_id in sorted(anchor_records, key=lambda value: int(value)):
        anchor = anchor_records[raw_id]
        final = final_records[raw_id]
        assert isinstance(anchor, Mapping) and isinstance(final, Mapping)
        _logging_only_record(anchor)
        _logging_only_record(final)
        if anchor["true_label"] != final["true_label"]:
            raise LoggingOnlyStateError("stable sample ID changed true label between anchor and final states")
        anchor_seen, final_seen = _exact_counter(anchor, "seen"), _exact_counter(final, "seen")
        anchor_correct = _exact_counter(anchor, "robust_correct_count")
        final_correct = _exact_counter(final, "robust_correct_count")
        anchor_forgetting = _exact_counter(anchor, "forgetting_count")
        final_forgetting = _exact_counter(final, "forgetting_count")
        if (
            anchor_seen != anchor_epoch + 1
            or final_seen != final_epoch + 1
            or not 0 <= anchor_correct <= anchor_seen
            or not 0 <= final_correct <= final_seen
            or not 0 <= anchor_forgetting < anchor_seen
            or not 0 <= final_forgetting < final_seen
        ):
            raise LoggingOnlyStateError("sample state counters do not cover every protocol epoch exactly")
        if final_forgetting < anchor_forgetting:
            raise LoggingOnlyStateError("sample forgetting count regressed between anchor and final state")
        if not isinstance(anchor["previous_robust_correct"], bool) or not isinstance(
            final["previous_robust_correct"], bool
        ):
            raise LoggingOnlyStateError("sample state current robust correctness must be bool")
        row: dict[str, Any] = {
            "namespace": "train",
            "sample_id": int(raw_id),
            "true_label": int(anchor["true_label"]),
            "anchor_epoch": anchor_epoch,
            "final_epoch": final_epoch,
            "anchor_robust_correct_frequency": anchor_correct / anchor_seen,
            "final_robust_correct_frequency": final_correct / final_seen,
            "subsequent_forgetting_increment": final_forgetting - anchor_forgetting,
            "future_online_forgetting": int(final_forgetting > anchor_forgetting),
            "final_robust_error": int(not bool(final["previous_robust_correct"])),
        }
        row.update({f"anchor_{key}": value for key, value in anchor.items()})
        row.update({f"final_{key}": value for key, value in final.items()})
        rows.append(row)
    manifest_identity = _manifest_identity(
        run_bundle_manifest,
        checkpoint_payload=anchor_payload,
        checkpoint_sha256={"anchor": anchor_checkpoint_sha256, "final": final_checkpoint_sha256},
    )
    provenance = _validate_analysis_provenance(
        _tracked_clean_analysis_provenance() if analysis_provenance is None else analysis_provenance
    )
    identity = {
        "schema_version": 1,
        "contract": "logging_only_exact_state_anchor99_final199_v1",
        "run_id": anchor_payload["tracker_run_id"],
        "config_hash": anchor_payload["config_hash"],
        "world_size": anchor_payload["world_size"],
        "expected_count": expected_count,
        **manifest_identity,
        "analysis_provenance": provenance,
        "anchor": {
            "epoch": anchor_epoch,
            "checkpoint_sha256": anchor_checkpoint_sha256,
            "sample_state_sha256": _canonical_sha256(anchor_state),
        },
        "final": {
            "epoch": final_epoch,
            "checkpoint_sha256": final_checkpoint_sha256,
            "sample_state_sha256": _canonical_sha256(final_state),
        },
        "row_count": len(rows),
    }
    return LoggingOnlyStateAnalysis(identity=identity, rows=tuple(rows))
