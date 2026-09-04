#!/usr/bin/env python3
"""Resolve and validate a scientific campaign before handing it to the orchestrator.

The gate is deliberately small and method agnostic.  It protects the inputs and
lineage of a campaign, then delegates scheduling and detached execution to
``multi-gpu-experiment-orchestrator/scripts/orchestrate.py``.  JSON is the
canonical interchange format; YAML is accepted when PyYAML is installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUNTIME_FINGERPRINT_VERSION = 1
FAST_EXISTING_RUNTIME = "FAST_EXISTING_RUNTIME"
FULL_NEW_INTEGRATION = "FULL_NEW_INTEGRATION"
OPERATIONAL_PROFILES = frozenset({FAST_EXISTING_RUNTIME, FULL_NEW_INTEGRATION})
JOB_TYPES = frozenset({"training", "evaluation", "collection", "inventory", "aggregation", "report", "finalization", "publish"})
# A change in any of these integration layers is intentionally a Full-path
# concern.  They are operational declarations, not scientific identity fields.
FULL_INTEGRATION_INDICATORS = frozenset(
    {
        "new_objective_runtime",
        "new_trainer_execution_path",
        "new_ddp_process_model",
        "new_dataset_loader",
        "new_remote_execution_mechanism",
        "new_checkpoint_serialization",
        "new_artifact_schema",
        "new_dependency_topology",
        "new_dependency_output_binding",
    }
)
SMOKE_EQUIVALENCE_FIELDS = frozenset(
    {
        "public_cli",
        "output_semantics",
        "config_schema",
        "checkpoint_load_path",
        "treatment_branch",
    }
)
FORBIDDEN_RETRY_FIELDS = {
    "parent",
    "seed",
    "arm",
    "scientific_config",
    "attack",
    "augmentation",
    "mask",
    "calibration",
    "dataset",
    "teacher",
}
REQUIRED_ATTACK_FIELDS = {"loss", "epsilon", "step_size", "steps", "random_start", "target"}
REMOTE_PREFLIGHT_SCHEMA = 1
MAX_PARALLEL_GATE_CHECKS = 4

# Resolve one immutable input once per observed file state.  The cache key
# deliberately includes path identity and stat evidence; a changed file (or a
# distinct host-local path) is rehashed.  Revalidation still invokes this
# helper and therefore observes a changed stat instead of trusting an earlier
# preparation pass.
_FILE_DIGEST_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}


def runtime_signature_required(spec: dict[str, Any]) -> bool:
    """Production/workspace campaigns must prove a validated runtime class."""
    return bool(
        spec.get("runtime_signature_required") is True
        or spec.get("require_validated_runtime_signature") is True
        or spec.get("tier") == "production"
        or (
            isinstance(spec.get("workspace_contract"), dict)
            and spec["workspace_contract"].get("enforce_future_writes") is True
        )
    )


_SCIENTIFIC_ARG_FLAGS = frozenset(
    {
        "--seed", "--parent", "--checkpoint", "--teacher", "--teacher-checkpoint",
        "--dataset", "--data", "--data-root", "--output", "--output-dir", "--run-id",
        "--wandb-run-id", "--mask", "--calibration", "--config", "--epochs", "--start-epoch",
        "--end-epoch", "--epsilon", "--step-size", "--steps", "--attack-seed", "--eval-seed",
    }
)


def _runtime_command_shape(command: Any) -> list[str]:
    """Retain executable/runtime shape while excluding scientific values.

    Explicit runtime metadata remains preferred.  The fallback keeps enough
    argv structure to invalidate a signature when the public CLI changes,
    without making seed, parent, checkpoint, or output bytes part of the
    operational class.
    """
    if not isinstance(command, list):
        return []
    shape: list[str] = []
    skip = False
    for token in command:
        if skip:
            skip = False
            continue
        value = str(token)
        if value in _SCIENTIFIC_ARG_FLAGS:
            skip = True
            continue
        if any(value.startswith(flag + "=") for flag in _SCIENTIFIC_ARG_FLAGS):
            continue
        shape.append(value)
    return shape


def _runtime_value(job: dict[str, Any], manifest: dict[str, Any], key: str, default: Any = None) -> Any:
    value = job.get(key)
    if value is not None:
        return value
    runtime = job.get("runtime_identity")
    if isinstance(runtime, dict) and runtime.get(key) is not None:
        return runtime[key]
    runtime = manifest.get("runtime_identity")
    if isinstance(runtime, dict) and runtime.get(key) is not None:
        return runtime[key]
    return default


def derive_runtime_fingerprint(resolved_manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive the operational runtime class from a resolved manifest.

    Scientific values (seed, parent/Teacher bytes, metrics, W&B IDs and
    timestamps) are intentionally excluded.  The returned object contains a
    canonical ``fingerprint`` and its SHA-256 ``digest`` so registry entries
    cannot be granted by an authored ID alone.
    """
    jobs = []
    for job in sorted(resolved_manifest.get("jobs", []), key=lambda item: str(item.get("job_id", ""))):
        executor = job.get("executor") if isinstance(job.get("executor"), dict) else {}
        deps = sorted(str(value) for value in job.get("dependencies", []) if isinstance(value, str))
        bindings = []
        for item in job.get("inputs", []) if isinstance(job.get("inputs", []), list) else []:
            if not isinstance(item, dict) or item.get("kind") != "dependency_output":
                continue
            bindings.append(
                {
                    "producer_job_id": item.get("producer_job_id"),
                    "path_binding": item.get("path_binding") or item.get("path") or None,
                }
            )
        jobs.append(
            {
                "job_type": str(job.get("job_type", "training")).lower(),
                "execution_class": "external" if executor.get("type") == "external_probe" else "local",
                "public_cli": _runtime_value(job, resolved_manifest, "public_cli") or _runtime_command_shape(job.get("command")),
                "runtime_contract_version": _runtime_value(job, resolved_manifest, "runtime_contract_version", "unversioned"),
                "config_schema": _runtime_value(job, resolved_manifest, "config_schema", "unversioned"),
                "checkpoint_load_contract": _runtime_value(job, resolved_manifest, "checkpoint_load_contract", "default"),
                "output_semantics": _runtime_value(
                    job,
                    resolved_manifest,
                    "output_semantics",
                    "attempt-scoped" if job.get("attempt_scoped_output") else "non-overwrite",
                ),
                "artifact_schema": _runtime_value(job, resolved_manifest, "artifact_schema", "default"),
                "overwrite_semantics": _runtime_value(job, resolved_manifest, "overwrite_semantics", "reject"),
                "dependencies": deps,
                "dependency_output_bindings": sorted(bindings, key=lambda value: str(value.get("producer_job_id", ""))),
                "completion_marker_contract": _runtime_value(job, resolved_manifest, "completion_marker_contract", "schema-v1"),
            }
        )
    topology = [
        {"job_type": row["job_type"], "dependencies": row["dependencies"]}
        for row in jobs
    ]
    fingerprint = {
        "version": RUNTIME_FINGERPRINT_VERSION,
        "public_cli_identity": _runtime_value(resolved_manifest, resolved_manifest, "public_cli_identity", "resolved-argv"),
        "runtime_contract_version": resolved_manifest.get("runtime_contract_version", "unversioned"),
        "config_schema": resolved_manifest.get("config_schema", "unversioned"),
        "checkpoint_load_contract": resolved_manifest.get("checkpoint_load_contract", "default"),
        "artifact_schema": resolved_manifest.get("artifact_schema", "default"),
        "execution_class_compatibility": sorted({row["execution_class"] for row in jobs}),
        "job_types": sorted({row["job_type"] for row in jobs}),
        "dependency_topology_class": topology,
        "jobs": jobs,
        "required_completion_marker_contract": resolved_manifest.get("completion_marker_contract", "schema-v1"),
    }
    return {"fingerprint": fingerprint, "digest": digest(fingerprint)}


def runtime_signature_report_for_manifest(manifest: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    """Match a resolved-manifest-derived fingerprint against one registry entry."""
    observed = derive_runtime_fingerprint(manifest)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "registry_invalid", "registry": str(registry_path), "observed": observed}
    entries = registry.get("signatures") if isinstance(registry, dict) else None
    requested = manifest.get("launch_gate", {}).get("runtime_signature")
    requested_id = requested.get("id") if isinstance(requested, dict) else requested
    if not isinstance(entries, list):
        return {"status": "registry_invalid", "registry": str(registry_path), "observed": observed}
    match = next((entry for entry in entries if isinstance(entry, dict) and entry.get("id") == requested_id), None)
    if match is None:
        return {"status": "unknown", "registry": str(registry_path), "observed": observed}
    if match.get("fingerprint_digest") != observed["digest"] or match.get("fingerprint") != observed["fingerprint"]:
        return {"status": "mismatch", "signature": match, "registry": str(registry_path), "observed": observed}
    return {
        "status": "validated",
        "signature": match,
        "registry": str(registry_path),
        "registry_sha256": file_digest(registry_path),
        "observed": observed,
    }


def runtime_signature_report(spec: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve one immutable signature against the tracked validation registry."""
    requested = spec.get("runtime_signature")
    required = runtime_signature_required(spec)
    if requested is None:
        return {"required": required, "status": "missing", "signature": None, "registry": None}
    registry_path = resolve_path(
        spec.get("runtime_signature_registry") or "configs/operational/validated_runtime_signatures_v1.json", base
    )
    if registry_path is None or not registry_path.is_file():
        return {"required": required, "status": "registry_missing", "signature": None, "registry": str(registry_path)}
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"required": required, "status": "registry_invalid", "signature": None, "registry": str(registry_path)}
    entries = registry.get("signatures") if isinstance(registry, dict) else None
    if not isinstance(entries, list):
        return {"required": required, "status": "registry_invalid", "signature": None, "registry": str(registry_path)}
    requested_id = requested.get("id") if isinstance(requested, dict) else requested
    if not isinstance(requested_id, str) or not requested_id:
        return {"required": required, "status": "invalid_request", "signature": None, "registry": str(registry_path)}
    match = next((entry for entry in entries if isinstance(entry, dict) and entry.get("id") == requested_id), None)
    if match is None:
        return {"required": required, "status": "unknown", "signature": None, "registry": str(registry_path)}
    if isinstance(requested, dict):
        comparable = {key: value for key, value in requested.items() if key != "id"}
        observed = {key: match.get(key) for key in comparable}
        if observed != comparable:
            return {
                "required": required,
                "status": "mismatch",
                "signature": match,
                "registry": str(registry_path),
            }
    return {
        "required": required,
        "status": "validated",
        "signature": match,
        "registry": str(registry_path),
        "registry_sha256": file_digest(registry_path),
    }


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def argv_digest(value: list[str] | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    key = (
        str(resolved),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    cached = _FILE_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    observed = hashlib.sha256(resolved.read_bytes()).hexdigest()
    _FILE_DIGEST_CACHE[key] = observed
    return observed


def classify_operational_profile(spec: dict[str, Any], errors: list[dict[str, Any]]) -> str:
    """Classify only the two permitted operational execution profiles.

    Existing public runtime integrations default to Fast.  An author can
    explicitly select Full when integration uncertainty remains; any declared
    new integration mechanism forces Full and rejects an unsafe Fast label.
    """
    declared = spec.get("operational_profile")
    if declared is not None and declared not in OPERATIONAL_PROFILES:
        error(
            errors,
            None,
            "operational_profile",
            sorted(OPERATIONAL_PROFILES),
            declared,
            "declare FAST_EXISTING_RUNTIME or FULL_NEW_INTEGRATION",
        )
    changes = spec.get("integration_changes", {})
    if changes is None:
        changes = {}
    if not isinstance(changes, dict):
        error(
            errors,
            None,
            "integration_changes",
            "mapping of known integration changes",
            type(changes).__name__,
            "declare new runtime mechanisms explicitly",
        )
        changes = {}
    unknown_changes = sorted(name for name in changes if name not in FULL_INTEGRATION_INDICATORS)
    if unknown_changes:
        error(
            errors,
            None,
            "integration_changes",
            sorted(FULL_INTEGRATION_INDICATORS),
            unknown_changes,
            "classify an unknown integration change conservatively as Full outside the Fast contract",
        )
    active_changes = sorted(name for name, changed in changes.items() if bool(changed))
    inferred = FULL_NEW_INTEGRATION if active_changes else FAST_EXISTING_RUNTIME
    if declared == FAST_EXISTING_RUNTIME and active_changes:
        error(
            errors,
            None,
            "operational_profile",
            FULL_NEW_INTEGRATION,
            FAST_EXISTING_RUNTIME,
            "new runtime integration cannot use the Fast profile",
        )
    # Explicit Full is a conservative decision and remains valid even when a
    # currently known trigger is not machine-readable in the spec.
    if declared == FULL_NEW_INTEGRATION:
        return FULL_NEW_INTEGRATION
    return inferred


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


def experiment_state_path(spec: dict[str, Any], manifest: dict[str, Any], base: Path) -> Path | None:
    """Resolve the canonical control-plane state path without guessing.

    New production specs should opt into the workspace registry.  Tests and
    legacy callers may provide an explicit path; otherwise no bridge state is
    emitted rather than writing beside an arbitrary scientific output.
    """
    explicit = spec.get("experiment_state_path")
    if explicit is not None:
        return resolve_path(explicit, base)
    contract = spec.get("workspace_contract")
    if not isinstance(contract, dict):
        return None
    registry = resolve_path(contract.get("registry"), base)
    if registry is None or not registry.is_file():
        return None
    try:
        values = json.loads(registry.read_text(encoding="utf-8"))
        runtime_root = values.get("runtime_root")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(runtime_root, str) or not runtime_root:
        return None
    return Path(runtime_root).expanduser().resolve() / "runs" / str(manifest["campaign_id"]) / "experiment-state.json"


def _declared_terminal_jobs(spec: dict[str, Any], jobs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    by_id = {str(job["job_id"]): job for job in jobs}
    declared_training = spec.get("required_training_jobs")
    declared_results = spec.get("terminal_result_jobs")
    training = (
        [str(value) for value in declared_training]
        if isinstance(declared_training, list)
        else [job["job_id"] for job in jobs if str(job.get("job_type", "")).lower() == "training"]
    )
    results = (
        [str(value) for value in declared_results]
        if isinstance(declared_results, list)
        else [
            job["job_id"]
            for job in jobs
            if str(job.get("job_type", "")).lower() in {"evaluation", "collection", "inventory", "aggregation", "report", "finalization", "publish"}
            and job.get("required_for_terminal", True) is not False
        ]
    )
    missing = sorted((set(training) | set(results)) - set(by_id))
    if missing:
        raise ValueError(f"terminal job declarations reference unknown IDs: {missing}")
    incompatible_training = [job_id for job_id in training if str(by_id[job_id].get("job_type", "")).lower() != "training"]
    if incompatible_training:
        raise ValueError(f"required_training_jobs must be training jobs: {incompatible_training}")
    overlap = sorted(set(training) & set(results))
    if overlap:
        raise ValueError(f"training and terminal result jobs overlap: {overlap}")
    return training, results


def build_experiment_state(
    spec: dict[str, Any], manifest: dict[str, Any], frozen_manifest: Path, *, state_path: Path
) -> dict[str, Any]:
    """Build the v2 reconciler bridge from one frozen orchestrator manifest."""
    training_jobs, result_jobs = _declared_terminal_jobs(spec, manifest.get("jobs", []))
    postprocess = spec.get("postprocess", {})
    if not isinstance(postprocess, dict):
        postprocess = {}
    owner_kind = postprocess.get("owner_kind")
    if owner_kind is None:
        owner_kind = "orchestrator_dag" if result_jobs else "external_registered_command"
    if owner_kind not in {"orchestrator_dag", "external_registered_command"}:
        raise ValueError(
            "postprocess.owner_kind must be orchestrator_dag or external_registered_command: "
            f"{owner_kind}"
        )
    identity_hash = digest(
        {
            "source_sha": manifest["source"]["git_sha"],
            "campaign_id": manifest["campaign_id"],
            "jobs": manifest.get("launch_gate", {}).get("scientific_identity_hashes", {}),
        }
    )
    configured_postprocess = dict(postprocess)
    configured_postprocess["owner_kind"] = owner_kind
    recovery = spec.get("recovery", {})
    if not isinstance(recovery, dict):
        recovery = {}
    publish = spec.get("result_publish", {})
    if not isinstance(publish, dict):
        publish = {}
    recovery_command = recovery.get("command") if isinstance(recovery.get("command"), list) else None
    postprocess_command = (
        configured_postprocess.get("command") if isinstance(configured_postprocess.get("command"), list) else None
    )
    return {
        "schema_version": 2,
        "experiment_id": manifest["campaign_id"],
        "mode": "orchestrator_campaign",
        "source_sha": manifest["source"]["git_sha"],
        "scientific_identity_hash": identity_hash,
        "campaign_id": manifest["campaign_id"],
        "manifest_path": str(frozen_manifest),
        "manifest_sha256": file_digest(frozen_manifest),
        "orchestrator_state_path": manifest["state_path"],
        "required_training_jobs": training_jobs,
        "terminal_result_jobs": result_jobs,
        "training_failure_markers": {
            job["job_id"]: job.get("_technical_failure_marker")
            for job in manifest.get("jobs", [])
            if job["job_id"] in training_jobs and job.get("_technical_failure_marker")
        },
        "state": "LAUNCHING",
        "launch": {"status": "pending", "attempts": []},
        "postprocess": configured_postprocess,
        "postprocess_command_sha256": argv_digest(postprocess_command),
        "recovery": dict(recovery),
        "recovery_command_sha256": argv_digest(recovery_command),
        "result_publish": dict(publish),
        "created_at": now(),
        "last_reconciled_at": None,
        "control_state_path": str(state_path),
    }


def update_experiment_state_launch(state_path: Path, *, status: str, attempt: dict[str, Any]) -> None:
    """Atomically record whether the detached controller launch was proven."""
    if status not in {"confirmed", "failed", "recovery"}:
        raise ValueError("launch status must be confirmed, failed, or recovery")
    state = load_document(state_path)
    if state.get("schema_version") != 2:
        raise ValueError("experiment-state must be schema-v2")
    launch = state.setdefault("launch", {"status": "pending", "attempts": []})
    attempts = launch.setdefault("attempts", [])
    attempts.append(dict(attempt, status=status))
    launch["status"] = status
    state["state"] = "TRAINING" if status == "confirmed" else ("LAUNCH_FAILED" if status == "failed" else "NEEDS_TECHNICAL_RECOVERY")
    state["last_reconciled_at"] = now()
    atomic_json(state_path, state)


def write_experiment_state(spec: dict[str, Any], manifest: dict[str, Any], frozen_manifest: Path) -> Path | None:
    """Materialize the campaign bridge only for an actual launch."""
    # The gate's resolved in-memory manifest is intentionally free of the
    # orchestrator's private loader fields.  The frozen manifest path is the
    # authoritative base for resolving the explicit state/registry paths.
    base = frozen_manifest.resolve().parent
    path = experiment_state_path(spec, manifest, base)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_state = build_experiment_state(spec, manifest, frozen_manifest, state_path=path)
    if path.exists():
        existing = load_document(path)
        if (
            existing.get("schema_version") != 2
            or existing.get("campaign_id") != manifest["campaign_id"]
            or existing.get("manifest_sha256") != file_digest(frozen_manifest)
            or existing.get("source_sha") != expected_state["source_sha"]
            or existing.get("scientific_identity_hash") != expected_state["scientific_identity_hash"]
            or existing.get("orchestrator_state_path") != expected_state["orchestrator_state_path"]
        ):
            raise ValueError(f"experiment-state already exists for a different campaign/manifest: {path}")
        return path
    atomic_json(path, expected_state)
    return path


def load_document(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read campaign spec {path}: {exc}") from exc
    try:
        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ValueError("YAML specs require PyYAML; use JSON or install the optional dependency") from exc
            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid campaign spec {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("campaign spec must be a mapping")
    return value


def resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def error(errors: list[dict[str, Any]], job: str | None, field: str, expected: Any, observed: Any, action: str) -> None:
    errors.append(
        {
            "job": job,
            "field": field,
            "expected": expected,
            "observed": observed,
            "remediation": action,
        }
    )


def _is_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and bool(re.fullmatch(r"[0-9a-fA-F]+", value))


def run_git(repo: Path, *args: str) -> tuple[int, str, str]:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def source_audit(spec: dict[str, Any], base: Path, errors: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    source = spec.get("source", {})
    if not isinstance(source, dict):
        error(errors, None, "source", "mapping", type(source).__name__, "provide source.git_sha and repo_path")
        return {}
    sha = source.get("git_sha") or spec.get("scientific_source_git_sha")
    if not isinstance(sha, str) or not _is_sha(sha, 40):
        error(errors, None, "source.git_sha", "full 40-hex commit", sha, "freeze a full Git commit SHA")
        return {}
    sha = sha.lower()
    repo = resolve_path(source.get("repo_path") or spec.get("repo_path"), base)
    if repo is None:
        error(errors, None, "source.repo_path", "existing repository", None, "set the repository used by the command")
        return {"git_sha": sha}
    code, head, stderr = run_git(repo, "rev-parse", "HEAD")
    if code != 0:
        error(errors, None, "source.repo_path", "Git repository", str(repo), stderr or "git rev-parse failed")
    elif head.lower() != sha:
        error(errors, None, "source.git_sha", sha, head, "check out the requested source before freezing")
    code, _, stderr = run_git(repo, "cat-file", "-e", f"{sha}^{{commit}}")
    if code != 0:
        error(
            errors, None, "source.git_sha", "commit exists", sha, stderr or "fetch or materialize the requested commit"
        )
    dirty_code, dirty, _ = run_git(repo, "status", "--porcelain")
    source_policy = spec.get("source_policy", {})
    if not isinstance(source_policy, dict):
        source_policy = {}
    allow_dirty = bool(source_policy.get("allow_dirty", False))
    generated = source_policy.get("generated_paths", [])
    generated_paths = [str(resolve_path(item, repo)) for item in generated] if isinstance(generated, list) else []
    dirty_lines = dirty.splitlines() if dirty else []
    unmanaged_dirty = [
        line
        for line in dirty_lines
        if not any(str((repo / line[3:].strip()).resolve()).startswith(path) for path in generated_paths)
    ]
    if dirty_code == 0 and unmanaged_dirty and not allow_dirty:
        error(
            errors,
            None,
            "source.dirty",
            "clean",
            unmanaged_dirty,
            "commit or explicitly record a non-production dirty policy",
        )
    elif dirty and allow_dirty:
        warnings.append("source repository is dirty; source_policy.allow_dirty=true")
    registered = source_policy.get("registered_files", {})
    if not isinstance(registered, dict):
        error(
            errors,
            None,
            "source_policy.registered_files",
            "mapping",
            type(registered).__name__,
            "use path -> SHA-256 entries",
        )
    else:
        for raw, expected in registered.items():
            path = resolve_path(raw, repo)
            if path is None or not path.is_file():
                error(
                    errors,
                    None,
                    f"source file {raw}",
                    "regular file",
                    str(path),
                    "materialize the registered source file",
                )
            elif not _is_sha(expected):
                error(errors, None, f"source file {raw}", "64-hex SHA-256", expected, "record a file SHA-256")
            elif file_digest(path) != expected.lower():
                error(
                    errors,
                    None,
                    f"source file {raw}",
                    expected.lower(),
                    file_digest(path),
                    "refresh the source binding",
                )
    return {"git_sha": sha, "repo_path": str(repo)}


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _descriptor(value: Any, default_kind: str = "existing_input") -> dict[str, Any]:
    if isinstance(value, str):
        return {"path": value, "kind": default_kind}
    if isinstance(value, dict):
        return dict(value)
    return {"path": None, "kind": default_kind, "invalid": value}


def validate_artifact(
    descriptor: Any,
    *,
    base: Path,
    errors: list[dict[str, Any]],
    job_id: str | None,
    role: str,
    allow_deferred: bool = False,
) -> dict[str, Any]:
    item = _descriptor(descriptor)
    raw_path = item.get("path")
    path = resolve_path(raw_path, base) if isinstance(raw_path, (str, Path)) else None
    kind = item.get("kind", "existing_input")
    if path is None:
        error(errors, job_id, role, "path", raw_path, "provide an artifact path")
        return item
    item["path"] = str(path)
    if kind == "dependency_output" and allow_deferred:
        return item
    if not path.is_file():
        error(errors, job_id, role, "existing regular file", str(path), "materialize the input before launch")
        return item
    expected = item.get("sha256")
    observed = file_digest(path)
    if expected is None and (role.startswith("mask") or role.startswith("calibration") or role.startswith("teacher")):
        error(
            errors, job_id, role + ".sha256", "registered SHA-256", None, "bind this scientific artifact to exact bytes"
        )
    elif expected is not None and (not _is_sha(expected) or observed != str(expected).lower()):
        error(errors, job_id, role + ".sha256", expected, observed, "bind the artifact to its exact bytes")
    for field in (
        "schema_version",
        "stable_id_hash",
        "parent_checkpoint_sha256",
        "epoch",
        "coefficient",
        "frozen_before_outcome",
    ):
        if field in item:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if not isinstance(payload, dict) or payload.get(field) != item[field]:
                error(
                    errors,
                    job_id,
                    f"{role}.{field}",
                    item[field],
                    payload.get(field) if isinstance(payload, dict) else None,
                    "use the registered artifact with matching metadata",
                )
    return item


def check_parent(
    parent: Any,
    *,
    base: Path,
    errors: list[dict[str, Any]],
    job_id: str,
) -> dict[str, Any] | None:
    if parent is None:
        return None
    item = _descriptor(parent)
    path = resolve_path(item.get("path"), base)
    if path is None:
        error(errors, job_id, "parent.path", "path", item.get("path"), "provide the parent checkpoint path")
        return item
    item["path"] = str(path)
    if not path.is_file():
        if item.get("kind") == "dependency_output":
            return item
        error(
            errors,
            job_id,
            "parent.path",
            "existing readable file",
            str(path),
            "materialize the exact parent checkpoint",
        )
        return item
    expected = item.get("sha256") or item.get("parent_checkpoint_sha256")
    if expected is None and item.get("kind") != "dependency_output":
        error(errors, job_id, "parent.sha256", "registered SHA-256", None, "bind the exact parent checkpoint bytes")
    elif expected is not None:
        parent_hash = file_digest(path)
        if not _is_sha(expected) or parent_hash != str(expected).lower():
            error(errors, job_id, "parent.sha256", expected, parent_hash, "resolve the exact parent alias/checkpoint")
    alias = resolve_path(item.get("alias_path"), base)
    if alias is not None:
        if not alias.is_file():
            error(errors, job_id, "parent.alias_path", "existing file", str(alias), "repair the parent alias")
        elif file_digest(alias) != file_digest(path):
            error(
                errors,
                job_id,
                "parent.alias_path",
                file_digest(path),
                file_digest(alias),
                "point the alias at the registered parent",
            )
    metadata_path = resolve_path(item.get("metadata_path"), base)
    if metadata_path is None and path.suffix == ".json":
        metadata_path = path
    if metadata_path is not None and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict):
            for field in ("epoch", "seed", "source_sha", "teacher_sha256", "config_sha256"):
                if field in item:
                    metadata_value: Any = metadata.get(field)
                    expected_value = item[field]
                    if field == "source_sha" and isinstance(metadata_value, str):
                        metadata_value = metadata_value.lower()
                    if metadata_value != expected_value:
                        error(
                            errors,
                            job_id,
                            f"parent.{field}",
                            expected_value,
                            metadata_value,
                            "use a checkpoint with matching embedded lineage",
                        )
    elif any(field in item for field in ("epoch", "seed", "source_sha", "teacher_sha256", "config_sha256")):
        error(
            errors,
            job_id,
            "parent.metadata",
            "metadata sidecar or JSON payload",
            None,
            "provide embedded/sidecar checkpoint lineage",
        )
    return item


def identity_artifact(value: Any) -> Any:
    """Keep scientific artifact identity, excluding host-local paths."""
    if not isinstance(value, dict):
        return value
    return {
        key: value[key]
        for key in ("sha256", "parent_checkpoint_sha256", "schema_version", "stable_id_hash", "epoch", "coefficient")
        if key in value
    }


def attack_identity(
    attack: Any, errors: list[dict[str, Any]], job_id: str | None, role: str
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(attack, dict):
        error(errors, job_id, role, "mapping", type(attack).__name__, "provide the complete attack contract")
        return {}, None
    missing = sorted(REQUIRED_ATTACK_FIELDS - set(attack))
    if missing:
        error(
            errors,
            job_id,
            role,
            "fields " + ", ".join(sorted(REQUIRED_ATTACK_FIELDS)),
            missing,
            "record loss, budget, steps, start, and target semantics",
        )
    fields = {key: value for key, value in attack.items() if key not in {"sha256", "label"}}
    computed = digest(fields)
    expected = attack.get("sha256")
    if expected is not None and (not _is_sha(expected) or str(expected).lower() != computed):
        error(errors, job_id, role + ".sha256", computed, expected, "refresh the canonical attack identity")
    label = attack.get("label")
    expected_label = attack.get("identity_label")
    if expected_label is not None and label != expected_label:
        # A free-form label is descriptive; an explicitly registered label is
        # authoritative and must not silently disagree with the contract.
        error(
            errors,
            job_id,
            role + ".label",
            expected_label,
            label,
            "correct the label or remove it; fields remain authoritative",
        )
    return fields, computed


def _replace_epoch_bound(command: list[str], final_epoch: int, errors: list[dict[str, Any]], job_id: str) -> list[str]:
    expected = final_epoch + 1
    result = list(command)
    if "--epochs" in result:
        index = result.index("--epochs")
        if index + 1 >= len(result):
            error(errors, job_id, "command.--epochs", expected, None, "provide an exclusive epoch upper bound")
        else:
            try:
                authored = int(result[index + 1])
            except ValueError:
                authored = None
            if authored is None:
                error(errors, job_id, "command.--epochs", expected, result[index + 1], "use an integer exclusive bound")
            else:
                result[index + 1] = str(expected)
    elif "runtime_epochs" in command:
        error(
            errors, job_id, "command", "argv with --epochs", command, "bind the scientific final epoch to the runtime"
        )
    else:
        error(errors, job_id, "command.--epochs", expected, None, "include the runtime exclusive epoch bound")
    return result


def _job_epoch_binding(
    job: dict[str, Any],
    *,
    campaign_start_epoch: int | None,
    campaign_final_epoch: int | None,
    errors: list[dict[str, Any]],
    job_id: str,
) -> tuple[int | None, int | None]:
    """Resolve an optional, narrower epoch contract for one training DAG node.

    A campaign can legitimately contain a short materialization prefix followed
    by longer child continuations.  The top-level contract remains the campaign
    envelope, while a training job may narrow it through ``epoch_binding``.
    The local range is identity-bound below and still has to be expressed as an
    exclusive ``--epochs`` runtime argument.
    """

    raw = job.get("epoch_binding")
    if raw is None:
        return campaign_start_epoch, campaign_final_epoch
    if not isinstance(raw, dict):
        error(
            errors,
            job_id,
            "epoch_binding",
            "mapping with scientific_start_epoch/scientific_final_epoch",
            raw,
            "declare a job-local inclusive epoch range",
        )
        return campaign_start_epoch, campaign_final_epoch
    start_epoch = raw.get("scientific_start_epoch", campaign_start_epoch)
    final_epoch = raw.get("scientific_final_epoch", campaign_final_epoch)
    if not isinstance(start_epoch, int) or not isinstance(final_epoch, int) or final_epoch < start_epoch:
        error(
            errors,
            job_id,
            "epoch_binding",
            "integer inclusive start/final with final >= start",
            raw,
            "declare a valid job-local epoch range",
        )
        return campaign_start_epoch, campaign_final_epoch
    if isinstance(campaign_start_epoch, int) and start_epoch < campaign_start_epoch:
        error(
            errors,
            job_id,
            "epoch_binding.scientific_start_epoch",
            f">= campaign start {campaign_start_epoch}",
            start_epoch,
            "keep the job-local range inside the campaign envelope",
        )
    if isinstance(campaign_final_epoch, int) and final_epoch > campaign_final_epoch:
        error(
            errors,
            job_id,
            "epoch_binding.scientific_final_epoch",
            f"<= campaign final {campaign_final_epoch}",
            final_epoch,
            "keep the job-local range inside the campaign envelope",
        )
    return start_epoch, final_epoch


def _argv(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        return None
    return list(value)


def validate_remote_wrapper(value: Any, *, host: str, role: str, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Require executable proof for direct remote shell wrappers.

    Remote files cannot be inspected locally.  A remote preflight report must
    therefore attest to the executable bit for a direct ``*.sh`` argv.  An
    explicit shell interpreter is self-describing and does not need that bit.
    """
    if not isinstance(value, dict):
        error(errors, host, role, "{argv, executable}", value, "declare remote launcher metadata")
        return None
    argv = _argv(value.get("argv"))
    if argv is None:
        error(errors, host, role + ".argv", "non-empty argv", value.get("argv"), "use structured argv metadata")
        return None
    direct_shell = argv[0].endswith(".sh")
    if direct_shell and value.get("executable") is not True:
        error(
            errors,
            host,
            role + ".executable",
            True,
            value.get("executable"),
            "invoke the wrapper as ['bash', 'wrapper.sh', ...] or report an executable remote wrapper",
        )
    return {"argv": argv, "executable": bool(value.get("executable", False))}


def remote_artifact_bindings(
    *, jobs: list[dict[str, Any]], dataset_identity: Any, teacher_identity: Any, teacher: Any
) -> list[dict[str, Any]]:
    """Build path-free identity expectations for one remote-host preflight."""
    bindings: list[dict[str, Any]] = []
    teacher_sha = None
    if isinstance(teacher, dict):
        checkpoint = teacher.get("checkpoint")
        if isinstance(checkpoint, dict):
            teacher_sha = checkpoint.get("sha256")
        teacher_sha = teacher_sha or teacher.get("sha256")
    for job in jobs:
        bindings.append({"role": "dataset", "job_id": job["job_id"], "identity": dataset_identity, "available": True})
        bindings.append(
            {
                "role": "teacher",
                "job_id": job["job_id"],
                "identity": teacher_identity,
                "sha256": teacher_sha,
                "available": True,
            }
        )
        for role, item in (("parent", job.get("parent")), ("scientific_config", job.get("scientific_config"))):
            if isinstance(item, dict):
                bindings.append(
                    {
                        "role": role,
                        "job_id": job["job_id"],
                        "sha256": item.get("sha256") or item.get("parent_checkpoint_sha256"),
                        "available": True,
                    }
                )
        for role, items in (("mask", job.get("masks", [])), ("calibration", job.get("calibration", []))):
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict):
                    bindings.append(
                        {"role": role, "job_id": job["job_id"], "sha256": item.get("sha256"), "available": True}
                    )
    return bindings


def _binding_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item.get("role"), item.get("job_id"), item.get("identity"), item.get("sha256"))


def run_remote_preflight(
    *, host: str, profile: dict[str, Any], source_sha: str, bindings: list[dict[str, Any]], errors: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Execute and validate the host executor's bounded JSON preflight.

    The command is normally a thin SSH wrapper supplied by ``run-on-ferret``.
    The gate validates its result but deliberately does not implement a second
    SSH or worktree manager.
    """
    contract = profile.get("remote_preflight")
    if not isinstance(contract, dict):
        error(errors, host, "remote_preflight", "mapping", contract, "declare a bounded remote preflight contract")
        return None
    command = _argv(contract.get("command"))
    if command is None:
        error(
            errors,
            host,
            "remote_preflight.command",
            "non-empty argv",
            contract.get("command"),
            "provide an SSH/status argv",
        )
        return None
    launcher = validate_remote_wrapper(
        contract.get("launcher"), host=host, role="remote_preflight.launcher", errors=errors
    )
    completion_probe = validate_remote_wrapper(
        contract.get("completion_probe"), host=host, role="remote_preflight.completion_probe", errors=errors
    )
    timeout = contract.get("timeout_seconds", 30)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = 0
    if timeout <= 0 or timeout > 120:
        error(
            errors,
            host,
            "remote_preflight.timeout_seconds",
            "0 < seconds <= 120",
            contract.get("timeout_seconds"),
            "use a bounded preflight",
        )
        return None
    minimum_disk = contract.get("minimum_disk_free_bytes", 0)
    if not isinstance(minimum_disk, int) or minimum_disk < 0:
        error(
            errors,
            host,
            "remote_preflight.minimum_disk_free_bytes",
            "non-negative integer",
            minimum_disk,
            "declare disk minimum",
        )
        return None
    output_root = contract.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        error(
            errors,
            host,
            "remote_preflight.output_root",
            "non-empty remote path",
            output_root,
            "declare the remote output root",
        )
        return None
    expected = {
        "schema_version": REMOTE_PREFLIGHT_SCHEMA,
        "host": host,
        "source_sha": source_sha.lower(),
        "python": profile.get("python"),
        "gpus": [{"index": gpu["index"], "uuid": gpu.get("uuid")} for gpu in profile.get("gpus", [])],
        "artifacts": bindings,
        "output_root": output_root,
        "launcher": launcher,
        "completion_probe": completion_probe,
    }
    env = os.environ.copy()
    env["ARD_LAUNCH_GATE_REMOTE_EXPECTED"] = json.dumps(expected, sort_keys=True)
    try:
        completed = subprocess.run(command, text=True, capture_output=True, env=env, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        error(
            errors,
            host,
            "remote_preflight.command",
            "successful bounded command",
            str(exc),
            "repair the remote executor",
        )
        return None
    try:
        report = json.loads(completed.stdout) if completed.returncode == 0 else None
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        error(
            errors,
            host,
            "remote_preflight.report",
            "JSON report with exit 0",
            completed.stderr or completed.stdout or completed.returncode,
            "make the remote preflight emit the contract report",
        )
        return None
    for field, expected_value in (
        ("schema_version", REMOTE_PREFLIGHT_SCHEMA),
        ("host", host),
        ("source_sha", source_sha.lower()),
    ):
        observed = report.get(field)
        if str(observed).lower() != str(expected_value).lower():
            error(
                errors, host, "remote_preflight." + field, expected_value, observed, "repair the remote source binding"
            )
    python_report = report.get("python")
    if (
        not isinstance(python_report, dict)
        or python_report.get("path") != profile.get("python")
        or python_report.get("available") is not True
    ):
        error(
            errors,
            host,
            "remote_preflight.python",
            {"path": profile.get("python"), "available": True},
            python_report,
            "repair the remote Python environment",
        )
    output_report = report.get("output")
    if (
        not isinstance(output_report, dict)
        or output_report.get("path") != output_root
        or output_report.get("writable") is not True
    ):
        error(
            errors,
            host,
            "remote_preflight.output",
            {"path": output_root, "writable": True},
            output_report,
            "repair remote output storage",
        )
    disk = report.get("disk_free_bytes")
    if not isinstance(disk, int) or disk < minimum_disk:
        error(
            errors,
            host,
            "remote_preflight.disk_free_bytes",
            f">= {minimum_disk}",
            disk,
            "free remote disk before launch",
        )
    reported_gpus = report.get("gpus")
    if not isinstance(reported_gpus, list):
        reported_gpus = []
    by_index = {item.get("index"): item for item in reported_gpus if isinstance(item, dict)}
    for gpu in expected["gpus"]:
        seen = by_index.get(gpu["index"])
        if not isinstance(seen, dict) or (gpu.get("uuid") and seen.get("uuid") != gpu["uuid"]):
            error(errors, host, "remote_preflight.gpus", gpu, seen, "repair GPU UUID/index binding")
    reported_bindings = report.get("artifacts")
    if not isinstance(reported_bindings, list):
        reported_bindings = []
    reported = {
        _binding_key(item) for item in reported_bindings if isinstance(item, dict) and item.get("available") is True
    }
    missing = [_binding_key(item) for item in bindings if _binding_key(item) not in reported]
    if missing:
        error(
            errors,
            host,
            "remote_preflight.artifacts",
            "all declared remote inputs",
            missing,
            "materialize and hash-bind remote inputs",
        )
    for role, metadata in (("launcher", launcher), ("completion_probe", completion_probe)):
        observed = report.get(role)
        if (
            not isinstance(observed, dict)
            or observed.get("argv") != metadata["argv"]
            or observed.get("valid") is not True
        ):
            error(
                errors,
                host,
                "remote_preflight." + role,
                {"argv": metadata["argv"], "valid": True},
                observed,
                "repair remote wrapper/probe",
            )
    return report


def run_remote_preflights(
    *,
    hosts: dict[str, dict[str, Any]],
    source_sha: str,
    bindings_by_host: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run distinct external-host preflights concurrently and render in host order.

    A remote preflight is host-scoped, read-only, and already bounded by its
    contract timeout.  Running different hosts concurrently removes needless
    control-plane serialization without relaxing any host/input validation.
    Each worker owns a local error list; the caller merges those lists in
    deterministic host order after every bounded command has returned.
    """
    tasks = [
        (host, profile, bindings_by_host[host])
        for host, profile in sorted(hosts.items())
        if profile.get("backend", "local") == "external" and host in bindings_by_host
    ]
    if not tasks:
        return {}

    def execute(
        host: str, profile: dict[str, Any], bindings: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        local_errors: list[dict[str, Any]] = []
        try:
            report = run_remote_preflight(
                host=host,
                profile=profile,
                source_sha=source_sha,
                bindings=bindings,
                errors=local_errors,
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper boundary
            error(
                local_errors,
                host,
                "remote_preflight.internal",
                "bounded preflight result",
                str(exc),
                "repair the remote preflight implementation",
            )
            report = None
        return report, local_errors

    results: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_GATE_CHECKS, len(tasks))) as executor:
        futures = {host: executor.submit(execute, host, profile, bindings) for host, profile, bindings in tasks}
        for host, _, _ in tasks:
            results[host] = futures[host].result()

    reports: dict[str, Any] = {}
    for host, _, _ in tasks:
        report, local_errors = results[host]
        errors.extend(local_errors)
        if report is not None:
            reports[host] = report
    return reports


def artifact_inventory_script() -> Path:
    return Path(__file__).with_name("artifact_inventory.py")


def validate_artifact_collection_contract(
    value: Any,
    *,
    base: Path,
    jobs: list[dict[str, Any]],
    external_hosts_used: bool,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Require a collection -> inventory barrier before remote aggregation."""
    if not external_hosts_used and value is None:
        return None
    if not isinstance(value, dict):
        error(
            errors,
            None,
            "artifact_collection",
            "mapping with manifest_path, collection_job_id, inventory_job_id",
            value,
            "declare canonical collection for every remote campaign",
        )
        return None
    raw_path = value.get("manifest_path")
    path = resolve_path(raw_path, base)
    if path is None or not path.is_file():
        error(
            errors,
            None,
            "artifact_collection.manifest_path",
            "existing inventory manifest",
            str(path),
            "materialize the inventory schema",
        )
        return None
    inspected = subprocess.run(
        [sys.executable, str(artifact_inventory_script()), "inspect", "--manifest", str(path)],
        text=True,
        capture_output=True,
    )
    if inspected.returncode != 0:
        error(
            errors,
            None,
            "artifact_collection.manifest",
            "valid identity-bound inventory schema",
            inspected.stdout or inspected.stderr,
            "repair the required artifact matrix before launch",
        )
    collection_id = value.get("collection_job_id")
    inventory_id = value.get("inventory_job_id")
    by_id = {job["job_id"]: job for job in jobs}
    collection = by_id.get(collection_id)
    inventory = by_id.get(inventory_id)
    if collection is None or collection.get("job_type") != "collection":
        error(
            errors,
            None,
            "artifact_collection.collection_job_id",
            "declared collection job",
            collection_id,
            "add a collection DAG node",
        )
    if inventory is None or inventory.get("job_type") != "inventory":
        error(
            errors,
            None,
            "artifact_collection.inventory_job_id",
            "declared inventory job",
            inventory_id,
            "add an inventory DAG node",
        )
    elif collection_id not in inventory.get("dependencies", []):
        error(
            errors,
            inventory_id,
            "dependencies",
            f"includes {collection_id}",
            inventory.get("dependencies"),
            "make inventory wait for canonical collection",
        )
    for job in jobs:
        if job.get("job_type") == "aggregation" and inventory_id not in job.get("dependencies", []):
            error(
                errors,
                job["job_id"],
                "dependencies",
                f"includes {inventory_id}",
                job.get("dependencies"),
                "make aggregation wait for the validated inventory",
            )
    return {
        "manifest_path": str(path),
        "collection_job_id": collection_id,
        "inventory_job_id": inventory_id,
    }


def _logical_host_path(
    spec: dict[str, Any],
    host_profile: dict[str, Any],
    identity: str,
    key: str,
    errors: list[dict[str, Any]],
    job_id: str,
) -> Path | None:
    campaign = spec.get(key, {})
    campaign_paths = campaign.get("host_paths", {}) if isinstance(campaign, dict) else {}
    profile_paths = host_profile.get(key + "_paths", {})
    if not isinstance(profile_paths, dict):
        profile_paths = {}
    raw = profile_paths.get(identity, campaign_paths.get(host_profile.get("name")))
    if raw is None:
        error(errors, job_id, key + ".identity", identity, None, "add a host-local logical path mapping")
        return None
    return resolve_path(raw, Path(host_profile.get("_base", ".")))


def resolve_campaign(spec: dict[str, Any], spec_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base = spec_path.parent.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    operational_profile = classify_operational_profile(spec, errors)
    signature = runtime_signature_report(spec, base)
    if operational_profile == FAST_EXISTING_RUNTIME and signature["required"] and signature["status"] != "validated":
        error(
            errors,
            None,
            "runtime_signature",
            "validated registry entry matching the requested runtime",
            signature["status"],
            "declare a registered runtime signature or use FULL_NEW_INTEGRATION",
        )
    if spec.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        error(
            errors,
            None,
            "schema_version",
            SCHEMA_VERSION,
            spec.get("schema_version"),
            "use a supported campaign schema",
        )
    campaign_id = spec.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        error(errors, None, "campaign_id", "non-empty string", campaign_id, "name the campaign")
        campaign_id = "invalid-campaign"
    source = source_audit(spec, base, errors, warnings)
    hosts_spec = spec.get("hosts")
    if not isinstance(hosts_spec, dict) or not hosts_spec:
        error(errors, None, "hosts", "non-empty mapping", hosts_spec, "define host profiles")
        hosts_spec = {}
    hosts: dict[str, Any] = {}
    for host, raw_profile in hosts_spec.items():
        if not isinstance(raw_profile, dict):
            error(errors, host, "host", "mapping", raw_profile, "define a host profile")
            continue
        profile = dict(raw_profile)
        profile["name"] = host
        profile["_base"] = str(base)
        repo = resolve_path(profile.get("repo_path") or source.get("repo_path"), base)
        if repo is not None:
            profile["repo_path"] = str(repo)
        python_path = profile.get("python")
        if python_path:
            resolved_python = resolve_path(python_path, base)
            profile["python"] = str(resolved_python) if resolved_python else python_path
            if (
                profile.get("backend", "local") == "local"
                and resolved_python is not None
                and not resolved_python.is_file()
            ):
                error(
                    errors,
                    host,
                    "python",
                    "existing executable",
                    str(resolved_python),
                    "select a valid host-local interpreter",
                )
        if profile.get("backend", "local") == "local":
            repo_path = Path(profile["repo_path"]) if profile.get("repo_path") else None
            if repo_path is not None and not repo_path.is_dir():
                error(errors, host, "repo_path", "existing directory", str(repo_path), "select the host-local checkout")
            required_paths = profile.get("required_paths", [])
            if isinstance(required_paths, list):
                for raw_path in required_paths:
                    candidate = resolve_path(raw_path, base)
                    if candidate is None or not candidate.exists():
                        error(
                            errors,
                            host,
                            "required_path",
                            "existing path",
                            str(candidate),
                            "materialize the host prerequisite",
                        )
            if repo_path is not None and repo_path.is_dir() and source.get("git_sha"):
                code, host_head, stderr = run_git(repo_path, "rev-parse", "HEAD")
                if code != 0 or host_head.lower() != str(source["git_sha"]).lower():
                    error(
                        errors,
                        host,
                        "repo_path.git_sha",
                        source.get("git_sha"),
                        host_head or stderr,
                        "check out the frozen source on this host",
                    )
        profile.pop("name", None)
        profile.pop("_base", None)
        hosts[host] = profile
    jobs_spec = spec.get("jobs")
    if not isinstance(jobs_spec, list) or not jobs_spec:
        error(errors, None, "jobs", "non-empty list", jobs_spec, "define campaign jobs")
        jobs_spec = []
    ids: set[str] = set()
    resolved_jobs: list[dict[str, Any]] = []
    output_paths: dict[str, str] = {}
    dataset = spec.get("dataset", {}) if isinstance(spec.get("dataset", {}), dict) else {}
    dataset_identity = dataset.get("identity")
    split_identity = dataset.get("split_identity")
    attacks = spec.get("attacks", {}) if isinstance(spec.get("attacks", {}), dict) else {}
    training = spec.get("training", {}) if isinstance(spec.get("training", {}), dict) else {}
    final_epoch = training.get("scientific_final_epoch")
    start_epoch = training.get("scientific_start_epoch")
    if not isinstance(start_epoch, int) or not isinstance(final_epoch, int) or final_epoch < start_epoch:
        error(
            errors,
            None,
            "training.epoch_contract",
            "integer start/final with final >= start",
            {"scientific_start_epoch": start_epoch, "scientific_final_epoch": final_epoch},
            "declare inclusive scientific epoch bounds",
        )
    if not isinstance(dataset_identity, str) or not dataset_identity:
        error(
            errors,
            None,
            "dataset.identity",
            "non-empty logical identity",
            dataset_identity,
            "declare the dataset identity",
        )
    teacher_spec = spec.get("teacher", {})
    if not isinstance(teacher_spec, dict) or not isinstance(teacher_spec.get("identity"), str):
        error(
            errors, None, "teacher.identity", "non-empty identity", teacher_spec, "declare the frozen teacher identity"
        )
    explicit_roles_required = runtime_signature_required(spec) or spec.get("schema_version") == 2
    for raw_job in jobs_spec:
        if not isinstance(raw_job, dict):
            error(errors, None, "job", "mapping", raw_job, "define each job as a mapping")
            continue
        job = dict(raw_job)
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            error(errors, None, "job_id", "non-empty string", job_id, "name each job")
            continue
        if job_id in ids:
            error(errors, job_id, "job_id", "unique", job_id, "rename the duplicate job")
            continue
        ids.add(job_id)
        if "job_type" not in job:
            if explicit_roles_required:
                error(
                    errors,
                    job_id,
                    "job_type",
                    sorted(JOB_TYPES),
                    None,
                    "declare an explicit schema-v2 production job role",
                )
            job_type = "training"
        else:
            job_type = str(job.get("job_type", "")).lower()
            if job_type not in JOB_TYPES:
                error(errors, job_id, "job_type", sorted(JOB_TYPES), job.get("job_type"), "use a registered job role")
                job_type = "invalid"
        job["job_type"] = job_type
        host = job.get("host") or (next(iter(hosts)) if hosts else None)
        if host not in hosts:
            error(errors, job_id, "host", sorted(hosts), host, "choose a declared host profile")
            host = next(iter(hosts), "")
        profile = hosts.get(host, {})
        command = job.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            error(errors, job_id, "command", "argv list", command, "use structured argv, never a shell string")
            command = ["false"]
        else:
            command = list(command)
        if command and command[0] in {"python", "${PYTHON}"} and profile.get("python"):
            command[0] = str(profile["python"])
        job_start_epoch, job_final_epoch = _job_epoch_binding(
            job,
            campaign_start_epoch=start_epoch,
            campaign_final_epoch=final_epoch,
            errors=errors,
            job_id=job_id,
        )
        if isinstance(job_final_epoch, int) and job_type == "training":
            command = _replace_epoch_bound(command, job_final_epoch, errors, job_id)
            authored_runtime_epochs = job.get("runtime_epochs")
            if authored_runtime_epochs is not None and authored_runtime_epochs != job_final_epoch + 1:
                error(
                    errors,
                    job_id,
                    "runtime_epochs",
                    job_final_epoch + 1,
                    authored_runtime_epochs,
                    "use the exclusive runtime bound derived from the job-local scientific final epoch",
                )
            authored_expected_final_epoch = job.get("expected_final_epoch")
            if (
                authored_expected_final_epoch is not None
                and authored_expected_final_epoch != job_final_epoch
            ):
                error(
                    errors,
                    job_id,
                    "expected_final_epoch",
                    job_final_epoch,
                    authored_expected_final_epoch,
                    "bind the expected final checkpoint to the job-local scientific final epoch",
                )
        cwd = resolve_path(job.get("cwd") or profile.get("repo_path") or base, base)
        output = resolve_path(job.get("output_dir"), base)
        if output is None:
            error(errors, job_id, "output_dir", "path", None, "provide a unique output directory")
            output = base / "invalid-output" / job_id
        output_key = str(output)
        overlap_owner = next(
            (
                output_paths[existing]
                for existing in output_paths
                if output_paths[existing] != job_id and paths_overlap(output, Path(existing))
            ),
            None,
        )
        if overlap_owner is not None:
            error(errors, job_id, "output_dir", "unique path", output_key, f"it collides with {overlap_owner}")
        output_paths[output_key] = job_id
        marker = resolve_path(job.get("completion_marker") or "completion.json", output)
        if marker is None:
            marker = output / "completion.json"
        if str(marker) in output_paths and output_paths[str(marker)] != job_id:
            error(
                errors,
                job_id,
                "completion_marker",
                "unique path",
                str(marker),
                f"it collides with {output_paths[str(marker)]}",
            )
        output_paths[str(marker)] = job_id
        if dataset_identity is not None and job.get("dataset_identity", dataset_identity) != dataset_identity:
            error(
                errors,
                job_id,
                "dataset_identity",
                dataset_identity,
                job.get("dataset_identity"),
                "use the campaign logical dataset identity",
            )
        dataset_path = None
        if dataset_identity is not None:
            profile_for_lookup = dict(profile)
            profile_for_lookup["name"] = host
            profile_for_lookup["_base"] = str(base)
            dataset_path = _logical_host_path(
                spec, profile_for_lookup, str(dataset_identity), "dataset", errors, job_id
            )
            if dataset_path is not None:
                specified = resolve_path(job.get("dataset_path"), base)
                if specified is not None and specified != dataset_path:
                    error(
                        errors,
                        job_id,
                        "dataset_path",
                        str(dataset_path),
                        str(specified),
                        "use the selected host's logical dataset path",
                    )
                if profile.get("backend", "local") == "local":
                    required_files = dataset.get("required_files", [])
                    for relative in required_files if isinstance(required_files, list) else []:
                        candidate = dataset_path / str(relative)
                        if not candidate.exists():
                            error(
                                errors,
                                job_id,
                                "dataset.required_file",
                                "existing path",
                                str(candidate),
                                "materialize the host-local dataset",
                            )
                    identity_file = dataset.get("identity_file")
                    if identity_file is not None:
                        identity_descriptor = {
                            "path": str(dataset_path / str(identity_file))
                            if not Path(str(identity_file)).is_absolute()
                            else str(identity_file),
                            "sha256": dataset.get("sha256"),
                        }
                        validate_artifact(
                            identity_descriptor,
                            base=base,
                            errors=errors,
                            job_id=job_id,
                            role="dataset.identity_file",
                        )
        if split_identity is not None and job.get("split_identity", split_identity) != split_identity:
            error(
                errors, job_id, "split_identity", split_identity, job.get("split_identity"), "bind the registered split"
            )
        parent = check_parent(job.get("parent"), base=base, errors=errors, job_id=job_id)
        teacher = spec.get("teacher", {})
        teacher_identity = teacher.get("identity") if isinstance(teacher, dict) else None
        teacher_desc = teacher.get("checkpoint") if isinstance(teacher, dict) else None
        teacher_item: dict[str, Any] | None = None
        if isinstance(teacher, dict) and teacher_desc is None and teacher.get("path") is not None:
            teacher_desc = {"path": teacher.get("path"), "sha256": teacher.get("sha256")}
        if isinstance(teacher, dict) and teacher_desc is not None:
            teacher_item = _descriptor(teacher_desc)
            profile_teacher_paths = profile.get("teacher_paths", {})
            if isinstance(profile_teacher_paths, dict) and teacher_identity in profile_teacher_paths:
                teacher_item["path"] = profile_teacher_paths[teacher_identity]
            validate_artifact(teacher_item, base=base, errors=errors, job_id=job_id, role="teacher.checkpoint")
        config_desc = job.get("scientific_config_path") or job.get("config_path")
        config_item = None
        if config_desc is not None:
            config_item = validate_artifact(
                {"path": config_desc, "sha256": job.get("config_sha256")},
                base=base,
                errors=errors,
                job_id=job_id,
                role="scientific_config",
            )
        masks = job.get("masks", [])
        if not isinstance(masks, list):
            masks = []
        masks = masks + (spec.get("masks", []) if isinstance(spec.get("masks", []), list) else [])
        masks = [
            validate_artifact(artifact, base=base, errors=errors, job_id=job_id, role="mask") for artifact in masks
        ]
        calibrations = job.get("calibration", [])
        if not isinstance(calibrations, list):
            calibrations = []
        calibrations = calibrations + (
            spec.get("calibration_artifacts", []) if isinstance(spec.get("calibration_artifacts", []), list) else []
        )
        calibrations = [
            validate_artifact(artifact, base=base, errors=errors, job_id=job_id, role="calibration")
            for artifact in calibrations
        ]
        attack_ref = job.get("attack")
        attack = attacks.get(attack_ref, attack_ref) if isinstance(attack_ref, str) else attack_ref
        attack_fields, attack_sha = (
            attack_identity(attack, errors, job_id, "attack") if attack is not None else ({}, None)
        )
        dependencies = job.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            error(errors, job_id, "dependencies", "job ID list", dependencies, "use declared dependency IDs")
            dependencies = []
        inputs = []
        for raw_input in job.get("inputs", []) if isinstance(job.get("inputs", []), list) else []:
            item = _descriptor(raw_input)
            if item.get("kind") == "dependency_output":
                producer = item.get("producer_job_id")
                if producer not in ids and producer not in {j.get("job_id") for j in jobs_spec if isinstance(j, dict)}:
                    error(
                        errors,
                        job_id,
                        "input.producer_job_id",
                        "declared producer job",
                        producer,
                        "declare the producer in this manifest",
                    )
                if producer not in dependencies:
                    error(
                        errors,
                        job_id,
                        "input.producer_job_id",
                        dependencies,
                        producer,
                        "add the producer as a dependency",
                    )
                item = validate_artifact(
                    item, base=base, errors=errors, job_id=job_id, role="input", allow_deferred=True
                )
            else:
                item = validate_artifact(item, base=base, errors=errors, job_id=job_id, role="input")
            inputs.append(item)
        expected_outputs = []
        for raw_output in job.get("expected_outputs", []) if isinstance(job.get("expected_outputs", []), list) else []:
            expected_outputs.append(_descriptor(raw_output, "output"))
        identity = {
            "arm": job.get("arm"),
            "seed": job.get("seed"),
            "scientific_config": identity_artifact(config_item),
            "dataset_identity": dataset_identity,
            "split_identity": split_identity,
            "parent": identity_artifact(parent),
            "teacher": {"identity": teacher_identity, "checkpoint": identity_artifact(teacher_item)},
            "attack": {"fields": attack_fields, "sha256": attack_sha},
            "augmentation": spec.get("augmentation_identity", job.get("augmentation")),
            "rng": spec.get("rng_contract", job.get("rng_contract")),
            "masks": [identity_artifact(item) for item in masks],
            "calibration": [identity_artifact(item) for item in calibrations],
            "epoch_binding": {
                "scientific_start_epoch": job_start_epoch,
                "scientific_final_epoch": job_final_epoch,
            },
        }
        identity_hash = digest({"source_sha": source.get("git_sha"), **identity})
        base_run = str(job.get("wandb_run_id") or f"{campaign_id}-{job_id}-{identity_hash[:12]}")
        resolved = {
            "job_id": job_id,
            "run_id": job.get("run_id", base_run),
            "arm": job.get("arm"),
            "seed": job.get("seed"),
            "host": host,
            "host_constraints": job.get("host_constraints", []),
            "gpu": job.get("gpu"),
            "gpu_count": job.get("gpu_count", 1),
            "command": command,
            "cwd": str(cwd) if cwd else str(base),
            "env": dict(job.get("env", {})) if isinstance(job.get("env", {}), dict) else {},
            "required_paths": [
                str(item["path"]) for item in inputs if item.get("kind") != "dependency_output" and item.get("path")
            ],
            "required_env": job.get("required_env", []),
            "output_dir": str(output),
            "completion_marker": str(marker),
            "dependencies": dependencies,
            "estimated_work": job.get("estimated_work", 1),
            "transfer_seconds": job.get("transfer_seconds", 0),
            "scientific_identity": {"source_sha": source.get("git_sha"), **identity},
            "identity_hash": identity_hash,
            "wandb_run_id_template": f"{base_run}-attempt-{{attempt}}",
            "retry_policy": job.get("retry_policy", {"max_attempts": 1}),
            "executor": job.get("executor", {"type": "local"}),
            "job_type": job_type,
            # Preserve external completion-probe fields when freezing the
            # resolved manifest.  The orchestrator validates these fields for
            # external jobs; dropping them here makes an otherwise valid
            # campaign fail only after the gate has frozen its manifest.
            "completion_probe": list(job["completion_probe"])
            if isinstance(job.get("completion_probe"), list)
            else None,
            "probe_interval_seconds": job.get("probe_interval_seconds", 30),
            "probe_timeout_seconds": job.get("probe_timeout_seconds"),
            "host_confirm_probe": list(job["host_confirm_probe"])
            if isinstance(job.get("host_confirm_probe"), list)
            else None,
            "host_confirm_timeout_seconds": job.get("host_confirm_timeout_seconds", 30),
            "host_confirm_interval_seconds": job.get("host_confirm_interval_seconds", 1),
            "remote_command": list(job["remote_command"]) if isinstance(job.get("remote_command"), list) else None,
            "host_confirmation_marker": job.get("host_confirmation_marker"),
            "expected_origin_host": job.get("expected_origin_host"),
            "attempt_scoped_output": job.get("attempt_scoped_output"),
            "fanout_group": job.get("fanout_group"),
            "smoke_group": job.get("smoke_group"),
            "smoke_equivalence": job.get("smoke_equivalence"),
            "expected_outputs": expected_outputs,
            "expected_final_epoch": job.get("expected_final_epoch", job_final_epoch),
            "epoch_binding": {
                "scientific_start_epoch": job_start_epoch,
                "scientific_final_epoch": job_final_epoch,
                "runtime_exclusive_epochs": job_final_epoch + 1 if isinstance(job_final_epoch, int) else None,
            },
            "dataset": {
                "identity": dataset_identity,
                "split_identity": split_identity,
                "path": str(dataset_path) if dataset_path else None,
            },
            "parent": parent,
            "scientific_config": config_item,
            "masks": masks,
            "calibration": calibrations,
            "inputs": inputs,
        }
        retry_mutations = job.get("retry_mutations", {})
        if isinstance(retry_mutations, dict):
            forbidden = sorted(set(retry_mutations) & FORBIDDEN_RETRY_FIELDS)
            if forbidden:
                error(
                    errors,
                    job_id,
                    "retry_mutations",
                    "technical fields only",
                    forbidden,
                    "keep scientific identity unchanged on retry",
                )
        resolved_jobs.append(resolved)
    # Dependency output producers and paths are checked after all jobs exist.
    by_id = {job["job_id"]: job for job in resolved_jobs}
    for job in resolved_jobs:
        for item in job.get("inputs", []):
            if item.get("kind") != "dependency_output":
                continue
            producer = by_id.get(item.get("producer_job_id"))
            if producer is None:
                continue
            producer_paths = {
                str(resolve_path(out.get("path"), Path(producer["output_dir"])) or "")
                for out in producer.get("expected_outputs", [])
            }
            expected_path = str(resolve_path(item.get("path"), base) or "")
            if expected_path not in producer_paths and expected_path != producer["output_dir"]:
                error(
                    errors,
                    job["job_id"],
                    "input.path",
                    sorted(producer_paths),
                    expected_path,
                    "bind dependency input to a producer output",
                )
    for job in resolved_jobs:
        job["wandb_run_id"] = job["wandb_run_id_template"].format(attempt=1)
    wandb_ids = [job["wandb_run_id"] for job in resolved_jobs]
    if len(wandb_ids) != len(set(wandb_ids)):
        error(
            errors,
            None,
            "wandb_run_id",
            "unique per job attempt",
            wandb_ids,
            "use campaign/job/identity-specific run IDs",
        )
    remote_bindings: dict[str, list[dict[str, Any]]] = {}
    for host, profile in hosts.items():
        if profile.get("backend", "local") != "external":
            continue
        host_jobs = [job for job in resolved_jobs if job.get("host") == host]
        if not host_jobs:
            continue
        remote_bindings[host] = remote_artifact_bindings(
            jobs=host_jobs,
            dataset_identity=dataset_identity,
            teacher_identity=teacher_spec.get("identity") if isinstance(teacher_spec, dict) else None,
            teacher=teacher_spec,
        )
    remote_preflights = run_remote_preflights(
        hosts=hosts,
        source_sha=str(source.get("git_sha", "")),
        bindings_by_host=remote_bindings,
        errors=errors,
    )
    artifact_collection = validate_artifact_collection_contract(
        spec.get("artifact_collection"),
        base=base,
        jobs=resolved_jobs,
        external_hosts_used=bool(remote_preflights)
        or any(
            profile.get("backend", "local") == "external" and any(job.get("host") == host for job in resolved_jobs)
            for host, profile in hosts.items()
        ),
        errors=errors,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "production_schema_version": 2 if (runtime_signature_required(spec) or spec.get("schema_version") == 2) else 1,
        "campaign_id": campaign_id,
        "source": {"git_sha": source.get("git_sha")},
        "state_path": str(resolve_path(spec.get("state_path") or f".orchestration/{campaign_id}.state.json", base)),
        "reservation_root": spec.get("reservation_root"),
        "orchestration_root": spec.get("orchestration_root"),
        "workspace_contract": spec.get("workspace_contract"),
        "hosts": hosts,
        "jobs": resolved_jobs,
        "required_training_jobs": spec.get("required_training_jobs"),
        "terminal_result_jobs": spec.get("terminal_result_jobs"),
        # Operational bridge declarations are carried through the immutable
        # manifest; they never enter scientific identity hashes.
        "postprocess": spec.get("postprocess", {}) if isinstance(spec.get("postprocess", {}), dict) else {},
        "recovery": spec.get("recovery", {}) if isinstance(spec.get("recovery", {}), dict) else {},
        "result_publish": spec.get("result_publish", {})
        if isinstance(spec.get("result_publish", {}), dict)
        else {},
        "experiment_state_path": str(experiment_state_path(spec, {"campaign_id": campaign_id}, base) or ""),
        "launch_gate": {
            "schema_version": SCHEMA_VERSION,
            "operational_profile": operational_profile,
            "runtime_signature": signature,
            "scientific_identity_hashes": {job["job_id"]: job["identity_hash"] for job in resolved_jobs},
            "source_repo": source.get("repo_path"),
            "dataset_identity": dataset_identity,
            "split_identity": split_identity,
            "canary": spec.get("canary", {}),
            "remote_preflight": remote_preflights,
            "artifact_collection": artifact_collection,
            "require_exact_smoke": bool(
                isinstance(spec.get("canary"), dict) and spec["canary"].get("require_exact_smoke") is True
            ),
        },
    }
    # Fast eligibility is granted only after the complete resolved runtime is
    # available.  Authored IDs/optional comparison fields are insufficient.
    registry_path = resolve_path(
        spec.get("runtime_signature_registry") or "configs/operational/validated_runtime_signatures_v1.json", base
    )
    runtime_fingerprint = derive_runtime_fingerprint(manifest)
    manifest["launch_gate"]["runtime_fingerprint"] = runtime_fingerprint
    if operational_profile == FAST_EXISTING_RUNTIME and signature["required"]:
        if registry_path is None or not registry_path.is_file():
            errors.append({"job": None, "field": "runtime_signature", "expected": "tracked registry", "observed": "missing", "remediation": "promote a smoke-proven runtime"})
        else:
            derived_report = runtime_signature_report_for_manifest(manifest, registry_path)
            if derived_report.get("status") != "validated":
                errors.append({"job": None, "field": "runtime_signature.fingerprint", "expected": "exact resolved-manifest fingerprint match", "observed": derived_report.get("status"), "remediation": "use FULL_NEW_INTEGRATION or promote a matching exact smoke"})
            else:
                manifest["launch_gate"]["runtime_signature_validation"] = derived_report
    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "resolved_jobs": resolved_jobs,
        "remote_preflight": remote_preflights,
        "operational_profile": operational_profile,
    }
    return manifest, report


def orchestrator_module() -> Any:
    path = Path(__file__).resolve().parents[2] / "multi-gpu-experiment-orchestrator" / "scripts" / "orchestrate.py"
    spec = importlib.util.spec_from_file_location("ard_orchestrate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load orchestrator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(manifest: dict[str, Any], out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved-manifest.json"
    payload = canonical(manifest)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    if path.exists() and file_digest(path) != manifest_sha:
        raise ValueError(f"frozen manifest already exists with a different SHA: {path}")
    if not path.exists():
        path.write_bytes(payload)
    freeze_record = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha,
        "source_sha": manifest["source"]["git_sha"],
        "scientific_identity_hashes": manifest["launch_gate"]["scientific_identity_hashes"],
    }
    atomic_json(out_dir / "freeze.json", freeze_record)
    return path, manifest_sha


def promote_runtime_signature(
    resolved_manifest_path: Path,
    exact_smoke_report_path: Path,
    signature_id: str,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically promote one fingerprint proven by the current exact smoke."""
    manifest = load_document(resolved_manifest_path.resolve())
    smoke = load_document(exact_smoke_report_path.resolve())
    if smoke.get("status") != "pass":
        raise ValueError("exact smoke report must have status=pass")
    if not isinstance(signature_id, str) or not SAFE_COMPONENT.fullmatch(signature_id):
        raise ValueError("signature-id must be one safe path component")
    source_sha = str(manifest.get("source", {}).get("git_sha", "")).lower()
    source_repo = manifest.get("launch_gate", {}).get("source_repo")
    if source_repo and source_sha:
        code, head, stderr = run_git(Path(source_repo), "rev-parse", "HEAD")
        if code != 0 or head.lower() != source_sha:
            raise ValueError(f"source is stale for signature promotion: expected {source_sha}, got {head or stderr}")
    exact = [
        item for item in smoke.get("results", [])
        if isinstance(item, dict) and item.get("kind") == "exact_public_cli" and item.get("status") == "pass"
    ]
    if not exact:
        raise ValueError("promotion requires at least one passed exact_public_cli smoke")
    expected_classes = {
        execution_class(manifest, job)
        for job in manifest.get("jobs", [])
        if requires_exact_smoke(job)
    }
    observed_classes = {item.get("binding", {}).get("execution_class") for item in exact}
    if not expected_classes <= observed_classes:
        raise ValueError("exact smoke does not cover every execution class in the resolved campaign")
    fingerprint = derive_runtime_fingerprint(manifest)
    binding = exact[0].get("binding")
    if not isinstance(binding, dict) or binding.get("source_sha", "").lower() != source_sha:
        raise ValueError("exact smoke binding is not bound to the resolved source")
    registry = registry_path or (Path(__file__).resolve().parents[4] / "configs/operational/validated_runtime_signatures_v1.json")
    if not registry.is_absolute():
        registry = (resolved_manifest_path.resolve().parent / registry).resolve()
    if registry.exists():
        data = load_document(registry)
    else:
        data = {"schema_version": 1, "signatures": [], "policy": {"unknown_is_full_integration": True}}
    entries = data.setdefault("signatures", [])
    existing = next((row for row in entries if isinstance(row, dict) and row.get("id") == signature_id), None)
    if existing is not None:
        if existing.get("fingerprint_digest") != fingerprint["digest"] or existing.get("fingerprint") != fingerprint["fingerprint"]:
            raise ValueError("signature ID already exists with a different fingerprint")
        return {"status": "NO_OP", "signature_id": signature_id, "fingerprint_digest": fingerprint["digest"]}
    entries.append(
        {
            "id": signature_id,
            "fingerprint": fingerprint["fingerprint"],
            "fingerprint_digest": fingerprint["digest"],
            "validated_execution_classes": sorted(observed_classes),
            "validating_source_sha": source_sha,
            "resolved_manifest_sha256": file_digest(resolved_manifest_path.resolve()),
            "exact_smoke_report_sha256": file_digest(exact_smoke_report_path.resolve()),
            "validated_at": now(),
            "description": "Validated public runtime topology promoted from an exact bounded smoke",
        }
    )
    data["schema_version"] = 1
    atomic_json(registry, data)
    return {"status": "PROMOTED", "signature_id": signature_id, "fingerprint_digest": fingerprint["digest"], "registry": str(registry)}


def plan_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        module = orchestrator_module()
        by_id = {job["job_id"]: job for job in manifest["jobs"]}
        remaining = set(by_id)
        completed: set[str] = set()
        unavailable: set[str] = set()
        rows = []
        while remaining:
            wave = sorted(
                (
                    by_id[job_id]
                    for job_id in remaining
                    if set(by_id[job_id].get("dependencies", [])) <= completed
                    or set(by_id[job_id].get("dependencies", [])) & unavailable
                ),
                key=lambda item: (-float(item.get("estimated_work", 1)), item["job_id"]),
            )
            if not wave:
                for job_id in sorted(remaining):
                    rows.append({"job_id": job_id, "status": "dependency_unavailable"})
                    unavailable.add(job_id)
                    remaining.remove(job_id)
                break
            occupied: set[tuple[str, int]] = set()
            assigned = 0
            for job in wave:
                if set(job.get("dependencies", [])) & unavailable:
                    rows.append({"job_id": job["job_id"], "status": "dependency_unavailable"})
                    unavailable.add(job["job_id"])
                    remaining.remove(job["job_id"])
                    continue
                slot = module.assign_slot(manifest, job, occupied)
                if slot is None:
                    if module.has_candidate_slot(manifest, job):
                        continue
                    rows.append({"job_id": job["job_id"], "status": "resource_conflict"})
                    unavailable.add(job["job_id"])
                    remaining.remove(job["job_id"])
                    continue
                occupied.add((slot[0], slot[1]))
                rows.append(
                    {
                        "job_id": job["job_id"],
                        "arm": job.get("arm"),
                        "seed": job.get("seed"),
                        "host": slot[0],
                        "gpu": slot[1],
                        "gpu_uuid": slot[2],
                        "dependencies": job.get("dependencies", []),
                        "scientific_identity_hash": job["identity_hash"],
                        "wandb_run_id": job["wandb_run_id"],
                    }
                )
                completed.add(job["job_id"])
                remaining.remove(job["job_id"])
                assigned += 1
            if assigned == 0:
                for job in wave:
                    if job["job_id"] not in remaining:
                        continue
                    rows.append({"job_id": job["job_id"], "status": "resource_conflict"})
                    unavailable.add(job["job_id"])
                    remaining.remove(job["job_id"])
        return rows
    except Exception as exc:  # pragma: no cover - defensive CLI formatting
        return [{"status": "plan_error", "error": str(exc)}]


def stage_remote_lifecycle_payload(
    manifest: dict[str, Any], *, host: str, payload: dict[str, Any], output: Path, timeout: float
) -> dict[str, Any]:
    """Validate and stage one remote lifecycle/collection proof.

    Both the legacy generic probe and a stronger exact public-CLI smoke use
    this same proof path.  The latter may therefore subsume the former without
    dropping host, source, completion, or SHA-verified collection evidence.
    """
    required = (
        payload.get("schema_version") == REMOTE_PREFLIGHT_SCHEMA
        and payload.get("host") == host
        and str(payload.get("source_sha", "")).lower() == manifest["source"]["git_sha"].lower()
        and payload.get("process_confirmed") is True
        and isinstance(payload.get("remote_manifest"), str)
        and bool(payload["remote_manifest"])
        and payload.get("completion_marker") is True
    )
    artifact = payload.get("artifact")
    if not required or not isinstance(artifact, dict):
        return {"host": host, "status": "fail", "error": "remote lifecycle evidence incomplete"}
    inventory = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "source_sha": manifest["source"]["git_sha"],
        "artifacts": [artifact],
        "required_cells": [artifact.get("identity")],
    }
    inventory_path = output / "collection-manifest.json"
    atomic_json(inventory_path, inventory)
    staged = subprocess.run(
        [sys.executable, str(artifact_inventory_script()), "stage", "--manifest", str(inventory_path)],
        text=True,
        capture_output=True,
    )
    (output / "collection.stdout.txt").write_text(staged.stdout, encoding="utf-8")
    (output / "collection.stderr.txt").write_text(staged.stderr, encoding="utf-8")
    return {
        "host": host,
        "status": "pass" if staged.returncode == 0 else "fail",
        "returncode": staged.returncode,
        "timeout_seconds": timeout,
        "artifact_collection": str(inventory_path),
    }


def run_remote_lifecycle_canaries(
    manifest: dict[str, Any], gate_dir: Path, *, subsumed_hosts: set[str] | None = None
) -> list[dict[str, Any]]:
    """Run one bounded launch/status/collection roundtrip per uncovered host."""
    subsumed_hosts = subsumed_hosts or set()
    canary = manifest.get("launch_gate", {}).get("canary", {})
    entries = canary.get("remote_lifecycle", []) if isinstance(canary, dict) else []
    entries = entries if isinstance(entries, list) else []
    external_hosts = {
        host
        for host, profile in manifest.get("hosts", {}).items()
        if isinstance(profile, dict)
        and profile.get("backend", "local") == "external"
        and any(job.get("host") == host for job in manifest.get("jobs", []))
    }
    by_host = {
        entry.get("host"): entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("host"), str)
    }
    results: list[dict[str, Any]] = []
    for host in sorted(external_hosts):
        if host in subsumed_hosts:
            continue
        entry = by_host.get(host)
        if entry is None:
            results.append({"host": host, "status": "fail", "error": "remote lifecycle canary missing"})
            continue
        command = _argv(entry.get("command"))
        timeout = entry.get("timeout_seconds", 30)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 0
        if command is None or timeout <= 0 or timeout > 120:
            results.append({"host": host, "status": "fail", "error": "remote lifecycle canary requires bounded argv"})
            continue
        output = gate_dir / "canary" / f"remote-{host}"
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "ARD_LAUNCH_GATE_CANARY": "1",
                "ARD_LAUNCH_GATE_EXPECTED_SOURCE_SHA": manifest["source"]["git_sha"],
                "ARD_LAUNCH_GATE_EXPECTED_HOST": host,
            }
        )
        try:
            completed = subprocess.run(command, text=True, capture_output=True, env=env, timeout=timeout)
            (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            payload = json.loads(completed.stdout) if completed.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            results.append({"host": host, "status": "fail", "error": str(exc)})
            continue
        if not isinstance(payload, dict):
            results.append({"host": host, "status": "fail", "error": "canary did not emit JSON"})
            continue
        results.append(
            stage_remote_lifecycle_payload(manifest, host=host, payload=payload, output=output, timeout=timeout)
        )
    return results


def execution_class(manifest: dict[str, Any], job: dict[str, Any]) -> str:
    profile = manifest["hosts"].get(job["host"], {})
    return "external" if profile.get("backend", "local") == "external" else "local"


def smoke_group_contract(manifest: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Return explicit smoke groups and reject unsafe equivalence claims.

    A representative smoke proves an execution branch, not every seed's
    scientific state.  Every member remains independently identity-validated
    during campaign resolution; grouping is permitted only for a complete,
    author-declared operational equivalence descriptor and one execution
    class.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for job in manifest.get("jobs", []):
        group = job.get("smoke_group")
        if group is None:
            continue
        if not isinstance(group, str) or not group:
            errors.append(f"job {job.get('job_id')} has an invalid smoke_group")
            continue
        groups.setdefault(group, []).append(job)
    for group, jobs in groups.items():
        descriptors: list[dict[str, Any]] = []
        classes: set[str] = set()
        for job in jobs:
            descriptor = job.get("smoke_equivalence")
            if not isinstance(descriptor, dict):
                errors.append(f"smoke_group {group} requires smoke_equivalence for {job.get('job_id')}")
                continue
            missing = sorted(SMOKE_EQUIVALENCE_FIELDS - set(descriptor))
            if missing:
                errors.append(f"smoke_group {group} is missing equivalence fields: {', '.join(missing)}")
                continue
            descriptors.append({field: descriptor[field] for field in sorted(SMOKE_EQUIVALENCE_FIELDS)})
            classes.add(execution_class(manifest, job))
        if len(classes) > 1:
            errors.append(f"smoke_group {group} mixes local and external execution classes")
        if descriptors and any(value != descriptors[0] for value in descriptors[1:]):
            errors.append(f"smoke_group {group} contains non-equivalent public runtime branches")
    return groups, errors


def smoke_group_coverage(manifest: dict[str, Any], job: dict[str, Any]) -> list[str]:
    group = job.get("smoke_group")
    if not isinstance(group, str) or not group:
        return [str(job["job_id"])]
    return [str(candidate["job_id"]) for candidate in manifest.get("jobs", []) if candidate.get("smoke_group") == group]


def requires_exact_smoke(job: dict[str, Any]) -> bool:
    """Keep public-runtime smoke coverage focused on executable method paths.

    Collection, inventory, aggregation, and report nodes are protected by
    their dependency/collection contracts.  Treating them as a second local
    public-runtime class would force a duplicate unrelated smoke for every
    remote scientific campaign.
    """
    return job.get("job_type", "training") not in {"collection", "inventory", "aggregation", "report", "finalization"}


def orchestrator_source_sha256() -> str:
    path = Path(__file__).resolve().parents[2] / "multi-gpu-experiment-orchestrator" / "scripts" / "orchestrate.py"
    return file_digest(path)


def dependency_topology_sha256(manifest: dict[str, Any]) -> str:
    topology = [
        {"job_id": job.get("job_id"), "dependencies": sorted(job.get("dependencies", []))}
        for job in manifest.get("jobs", [])
    ]
    return digest(topology)


def exact_smoke_binding(manifest: dict[str, Any], job: dict[str, Any], command: list[str]) -> dict[str, Any]:
    parent = job.get("parent") if isinstance(job.get("parent"), dict) else {}
    config = job.get("scientific_config") if isinstance(job.get("scientific_config"), dict) else {}
    return {
        "source_sha": manifest["source"]["git_sha"].lower(),
        "production_argv_sha256": digest(job["command"]),
        "smoke_argv_sha256": digest(command),
        "config_sha256": config.get("sha256"),
        "parent_checkpoint_sha256": parent.get("sha256") or parent.get("parent_checkpoint_sha256"),
        "orchestrator_source_sha256": orchestrator_source_sha256(),
        "dependency_topology_sha256": dependency_topology_sha256(manifest),
        "runtime_signature": manifest.get("launch_gate", {}).get("runtime_signature"),
        "manifest_schema_version": manifest["schema_version"],
        "execution_class": execution_class(manifest, job),
        "fanout_group": job.get("fanout_group") or "__all__",
        "smoke_group": job.get("smoke_group"),
    }


def _run_static_cli_entry(manifest: dict[str, Any], gate_dir: Path, entry: Any) -> dict[str, Any]:
    """Run one static entry; commands inside an entry intentionally stay ordered."""
    started_at = now()
    job_id = entry.get("job_id") if isinstance(entry, dict) else None
    commands = entry.get("commands") if isinstance(entry, dict) else None
    job = next((item for item in manifest["jobs"] if item["job_id"] == job_id), None)
    valid_commands = isinstance(commands, list) and commands and all(_argv(item) for item in commands)
    if job is None or not valid_commands:
        return {
            "job_id": job_id,
            "status": "fail",
            "error": "static_cli requires known job_id and commands",
            "started_at": started_at,
            "finished_at": now(),
        }
    try:
        timeout = float(entry.get("timeout_seconds", 30))
    except (AttributeError, TypeError, ValueError):
        timeout = 0
    if timeout <= 0 or timeout > 120:
        return {
            "job_id": job_id,
            "status": "fail",
            "error": "static_cli timeout must be bounded",
            "started_at": started_at,
            "finished_at": now(),
        }
    command_results: list[dict[str, Any]] = []
    for ordinal, command in enumerate(commands):
        output = gate_dir / "static-cli" / str(job_id) / str(ordinal)
        try:
            output.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, cwd=job["cwd"], text=True, capture_output=True, timeout=timeout)
            (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            command_results.append(
                {"argv": command, "argv_sha256": digest(command), "returncode": completed.returncode}
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            command_results.append(
                {
                    "argv": command,
                    "argv_sha256": digest(command),
                    "error": str(exc),
                    "returncode": 127,
                }
            )
    return {
        "job_id": job_id,
        "status": "pass" if all(row["returncode"] == 0 for row in command_results) else "fail",
        "commands": command_results,
        "parallel_safe": bool(isinstance(entry, dict) and entry.get("parallel_safe") is True),
        "started_at": started_at,
        "finished_at": now(),
    }


def _parallel_static_indexes(entries: list[Any]) -> set[int]:
    """Return explicitly isolated static checks with unique output namespaces."""
    safe_job_ids = [
        entry.get("job_id")
        for entry in entries
        if isinstance(entry, dict) and entry.get("parallel_safe") is True and isinstance(entry.get("job_id"), str)
    ]
    return {
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, dict)
        and entry.get("parallel_safe") is True
        and isinstance(entry.get("job_id"), str)
        and safe_job_ids.count(entry["job_id"]) == 1
    }


def run_static_cli_smokes(manifest: dict[str, Any], gate_dir: Path) -> dict[str, Any]:
    """Run static checks with opt-in concurrency and deterministic result order."""
    canary = manifest.get("launch_gate", {}).get("canary", {})
    entries = canary.get("static_cli", []) if isinstance(canary, dict) else []
    entries = entries if isinstance(entries, list) else []
    results: list[dict[str, Any] | None] = [None] * len(entries)
    parallel_indexes = _parallel_static_indexes(entries)
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_GATE_CHECKS, max(1, len(parallel_indexes)))) as executor:
        futures = {
            index: executor.submit(_run_static_cli_entry, manifest, gate_dir, entries[index])
            for index in sorted(parallel_indexes)
        }
        for index, entry in enumerate(entries):
            if index not in parallel_indexes:
                results[index] = _run_static_cli_entry(manifest, gate_dir, entry)
        for index in sorted(parallel_indexes):
            results[index] = futures[index].result()
    rendered = [item for item in results if item is not None]
    required = bool(manifest.get("launch_gate", {}).get("require_exact_smoke"))
    if required and not rendered:
        rendered.append({"status": "fail", "error": "static_cli checks are required before exact smoke/fan-out"})
    return {
        "status": "pass" if all(row.get("status") == "pass" for row in rendered) else "fail",
        "parallel_entries": len(parallel_indexes),
        "results": rendered,
    }


def validate_exact_smoke_report(manifest: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Reject fan-out when a source/argv/parent/config-bound smoke is stale."""
    if not manifest.get("launch_gate", {}).get("require_exact_smoke"):
        return []
    explicit_groups, errors = smoke_group_contract(manifest)
    exact = [
        item
        for item in report.get("results", [])
        if item.get("kind") == "exact_public_cli" and item.get("status") == "pass"
    ]
    smoke_jobs = [job for job in manifest["jobs"] if requires_exact_smoke(job)]
    required_classes = {execution_class(manifest, job) for job in smoke_jobs}
    observed_classes = {item.get("binding", {}).get("execution_class") for item in exact}
    if not required_classes <= observed_classes:
        errors.append("exact smoke does not cover every local/external execution class")
    for entry in exact:
        job = next((item for item in manifest["jobs"] if item["job_id"] == entry.get("job_id")), None)
        command = entry.get("command")
        valid_binding = (
            job is not None
            and isinstance(command, list)
            and entry.get("binding") == exact_smoke_binding(manifest, job, command)
        )
        if not valid_binding:
            errors.append(f"exact smoke binding is stale for {entry.get('job_id')}")
        coverage = entry.get("covered_job_ids", [entry.get("job_id")])
        if not isinstance(coverage, list) or not all(isinstance(value, str) for value in coverage):
            errors.append(f"exact smoke coverage is malformed for {entry.get('job_id')}")
            continue
        group = job.get("smoke_group") if job is not None else None
        if isinstance(group, str) and group:
            expected_coverage = set(smoke_group_coverage(manifest, job))
            if set(coverage) != expected_coverage:
                errors.append(f"smoke_group {group} has incomplete representative coverage")
            if entry.get("binding", {}).get("smoke_group") != group:
                errors.append(f"smoke_group {group} binding is stale for {entry.get('job_id')}")
        elif coverage != [entry.get("job_id")]:
            errors.append(f"non-group exact smoke cannot cover other jobs: {entry.get('job_id')}")
    for group, jobs in explicit_groups.items():
        expected = {str(job["job_id"]) for job in jobs}
        entries = [
            item
            for item in exact
            if item.get("binding", {}).get("smoke_group") == group and set(item.get("covered_job_ids", [])) == expected
        ]
        if not entries:
            errors.append(f"smoke_group {group} has no exact public-CLI representative")
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in smoke_jobs:
        if job.get("smoke_group") is not None:
            continue
        groups.setdefault(job.get("fanout_group") or "__all__", []).append(job)
    for group, jobs in groups.items():
        if len(jobs) <= 1:
            continue
        entries = [item for item in exact if item.get("binding", {}).get("fanout_group") == group]
        if not entries:
            errors.append(f"fan-out group {group} has no exact public-CLI smoke")
            continue
    return errors


def _run_canary_entry(
    manifest: dict[str, Any],
    gate_dir: Path,
    entry: dict[str, Any],
    job: dict[str, Any],
    kind: str,
    binding: dict[str, Any] | None,
    smoke_group: str | None,
) -> dict[str, Any]:
    """Execute one prevalidated canary entry in its isolated gate namespace."""
    started_at = now()
    output = gate_dir / "canary" / entry["job_id"]
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "job_id": job["job_id"],
            "status": "fail",
            "error": str(exc),
            "started_at": started_at,
            "finished_at": now(),
        }
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
    env["ARD_LAUNCH_GATE_CANARY"] = "1"
    env["ARD_ORCH_JOB_ID"] = job["job_id"]
    env["ARD_ORCH_SOURCE_SHA"] = manifest["source"]["git_sha"]
    env["ARD_LAUNCH_GATE_EXPECTED_HOST"] = job["host"]
    env["ARD_LAUNCH_GATE_EXPECTED_EXECUTION_CLASS"] = execution_class(manifest, job)
    if job.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    try:
        timeout = float(entry.get("timeout_seconds", 30))
    except (TypeError, ValueError):
        timeout = 0
    if timeout <= 0 or timeout > 120:
        return {
            "job_id": job["job_id"],
            "status": "fail",
            "error": "canary timeout must be bounded",
            "started_at": started_at,
            "finished_at": now(),
        }
    try:
        completed = subprocess.run(
            entry["command"], cwd=job["cwd"], env=env, text=True, capture_output=True, timeout=timeout
        )
        (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "job_id": job["job_id"],
            "status": "fail",
            "error": "canary timeout" if isinstance(exc, subprocess.TimeoutExpired) else str(exc),
            "timeout_seconds": timeout,
            "started_at": started_at,
            "finished_at": now(),
        }
    result = {
        "job_id": job["job_id"],
        "kind": kind,
        "command": entry["command"],
        "binding": binding,
        "smoke_group": smoke_group,
        "covered_job_ids": smoke_group_coverage(manifest, job) if kind == "exact_public_cli" else [job["job_id"]],
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "timeout_seconds": timeout,
        "parallel_safe": entry.get("parallel_safe") is True,
        "parallel_resource_key": entry.get("parallel_resource_key"),
        "started_at": started_at,
        "finished_at": now(),
    }
    if (
        result["status"] == "pass"
        and kind == "exact_public_cli"
        and execution_class(manifest, job) == "external"
        and entry.get("subsumes_remote_lifecycle") is True
    ):
        try:
            lifecycle_payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            lifecycle_payload = None
        if not isinstance(lifecycle_payload, dict):
            result.update(status="fail", error="exact remote smoke did not emit lifecycle JSON")
        else:
            lifecycle = stage_remote_lifecycle_payload(
                manifest,
                host=str(job["host"]),
                payload=lifecycle_payload,
                output=output,
                timeout=timeout,
            )
            result["remote_lifecycle"] = lifecycle
            if lifecycle["status"] != "pass":
                result.update(status="fail", error="exact remote smoke lifecycle proof failed")
            else:
                result["subsumes_remote_lifecycle"] = True
    return result


def _parallel_canary_indexes(
    entries: list[tuple[int, dict[str, Any], dict[str, Any], str, dict[str, Any] | None, str | None]],
) -> set[int]:
    """Allow only explicitly isolated exact-smoke entries to overlap.

    Exact public-CLI smoke may touch a GPU or external runner.  A truthful
    `parallel_safe` declaration therefore also needs a unique resource key and
    a fixed, unique host/GPU assignment.  Bounded generic canaries stay
    serial because the gate cannot infer their mutable-resource footprint.
    """
    candidates = [
        item
        for item in entries
        if item[3] == "exact_public_cli"
        and item[1].get("parallel_safe") is True
        and isinstance(item[1].get("parallel_resource_key"), str)
        and item[1]["parallel_resource_key"]
        and isinstance(item[2].get("gpu"), int)
    ]
    resource_keys = [item[1]["parallel_resource_key"] for item in candidates]
    slots = [(item[2]["host"], item[2]["gpu"]) for item in candidates]
    return {
        item[0]
        for item in candidates
        if resource_keys.count(item[1]["parallel_resource_key"]) == 1
        and slots.count((item[2]["host"], item[2]["gpu"])) == 1
    }


def run_canary(manifest: dict[str, Any], gate_dir: Path) -> dict[str, Any]:
    started_at = now()
    canary = manifest.get("launch_gate", {}).get("canary", {})
    entries = canary.get("jobs", []) if isinstance(canary, dict) else []
    entries = entries if isinstance(entries, list) else []
    prefix_results: list[dict[str, Any]] = []
    entry_results: dict[int, dict[str, Any]] = {}
    _, group_errors = smoke_group_contract(manifest)
    for message in group_errors:
        prefix_results.append({"status": "fail", "kind": "smoke_group", "error": message})
    seen_groups: set[str] = set()
    prepared: list[tuple[int, dict[str, Any], dict[str, Any], str, dict[str, Any] | None, str | None]] = []
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("job_id"), str)
            or not isinstance(entry.get("command"), list)
        ):
            entry_results[index] = {"status": "fail", "error": "canary jobs require job_id and argv command"}
            continue
        job = next((item for item in manifest["jobs"] if item["job_id"] == entry["job_id"]), None)
        if job is None:
            entry_results[index] = {"job_id": entry["job_id"], "status": "fail", "error": "unknown production job"}
            continue
        kind = entry.get("kind", "bounded_canary")
        if kind not in {"bounded_canary", "exact_public_cli"}:
            entry_results[index] = {"job_id": job["job_id"], "status": "fail", "error": "unknown canary kind"}
            continue
        smoke_group = job.get("smoke_group") if kind == "exact_public_cli" else None
        if entry.get("smoke_group") is not None and entry.get("smoke_group") != smoke_group:
            entry_results[index] = {
                "job_id": job["job_id"],
                "status": "fail",
                "error": "canary smoke_group differs from the resolved production job",
            }
            continue
        if isinstance(smoke_group, str) and smoke_group:
            if smoke_group in seen_groups:
                entry_results[index] = {
                    "job_id": job["job_id"],
                    "kind": kind,
                    "smoke_group": smoke_group,
                    "status": "skipped_equivalent",
                    "covered_by": smoke_group,
                }
                continue
            seen_groups.add(smoke_group)
        binding = exact_smoke_binding(manifest, job, entry["command"]) if kind == "exact_public_cli" else None
        if kind == "exact_public_cli" and (
            binding["config_sha256"] is None or binding["parent_checkpoint_sha256"] is None
        ):
            entry_results[index] = {
                "job_id": job["job_id"],
                "status": "fail",
                "error": "exact public-CLI smoke requires bound config and parent SHA-256",
            }
            continue
        if kind == "exact_public_cli" and entry.get("execution_class") != binding["execution_class"]:
            entry_results[index] = {
                "job_id": job["job_id"],
                "status": "fail",
                "error": "exact public-CLI smoke execution_class differs from job host class",
            }
            continue
        prepared.append((index, entry, job, kind, binding, smoke_group))

    parallel_indexes = _parallel_canary_indexes(prepared)
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_GATE_CHECKS, max(1, len(parallel_indexes)))) as executor:
        futures = {
            index: executor.submit(_run_canary_entry, manifest, gate_dir, entry, job, kind, binding, smoke_group)
            for index, entry, job, kind, binding, smoke_group in prepared
            if index in parallel_indexes
        }
        for index, entry, job, kind, binding, smoke_group in prepared:
            if index not in parallel_indexes:
                entry_results[index] = _run_canary_entry(manifest, gate_dir, entry, job, kind, binding, smoke_group)
        for index in sorted(parallel_indexes):
            entry_results[index] = futures[index].result()

    results = prefix_results + [entry_results[index] for index in range(len(entries))]
    if not entries:
        results.append(
            {
                "status": "fail",
                "error": "no canary jobs declared; use --canary-only with a bounded non-scientific canary",
            }
        )
    # Preserve the historical lifecycle rule: only a passed exact external
    # smoke may suppress the generic host lifecycle proof.
    jobs_by_id = {job["job_id"]: job for job in manifest["jobs"]}
    subsumed_hosts = {
        str(jobs_by_id[item["job_id"]]["host"])
        for item in results
        if item.get("status") == "pass"
        and item.get("subsumes_remote_lifecycle") is True
        and item.get("job_id") in jobs_by_id
    }
    results.extend(run_remote_lifecycle_canaries(manifest, gate_dir, subsumed_hosts=subsumed_hosts))
    return {
        "status": "pass" if all(item.get("status") in {"pass", "skipped_equivalent"} for item in results) else "fail",
        "started_at": started_at,
        "finished_at": now(),
        "parallel_entries": len(parallel_indexes),
        "results": results,
    }


def revalidate_frozen(resolved_path: Path, freeze_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(resolved_path.read_text(encoding="utf-8"))
    freeze_record = json.loads(freeze_path.read_text(encoding="utf-8"))
    errors: list[dict[str, Any]] = []
    observed = file_digest(resolved_path)
    if observed != freeze_record.get("manifest_sha256"):
        error(
            errors,
            None,
            "manifest_sha256",
            freeze_record.get("manifest_sha256"),
            observed,
            "do not edit a frozen manifest",
        )
    source_repo = manifest.get("launch_gate", {}).get("source_repo")
    expected_sha = manifest.get("source", {}).get("git_sha")
    if source_repo and expected_sha:
        code, head, stderr = run_git(Path(source_repo), "rev-parse", "HEAD")
        if code != 0 or head.lower() != expected_sha.lower():
            error(
                errors, None, "source.git_sha", expected_sha, head or stderr, "restore the frozen source before launch"
            )
    for job in manifest.get("jobs", []):
        parent = job.get("parent")
        if isinstance(parent, dict) and parent.get("kind") != "dependency_output" and parent.get("path"):
            parent_path = Path(str(parent["path"]))
            expected_parent = parent.get("sha256") or parent.get("parent_checkpoint_sha256")
            if not parent_path.is_file():
                error(
                    errors,
                    job.get("job_id"),
                    "parent.path",
                    "existing file",
                    str(parent_path),
                    "restore the frozen parent",
                )
            elif expected_parent and file_digest(parent_path) != str(expected_parent).lower():
                error(
                    errors,
                    job.get("job_id"),
                    "parent.sha256",
                    expected_parent,
                    file_digest(parent_path),
                    "restore the frozen parent bytes",
                )
        config = job.get("scientific_config")
        if isinstance(config, dict) and config.get("path"):
            config_path = Path(str(config["path"]))
            expected_config = config.get("sha256")
            if not config_path.is_file():
                error(
                    errors,
                    job.get("job_id"),
                    "scientific_config.path",
                    "existing file",
                    str(config_path),
                    "restore the frozen config",
                )
            elif expected_config and file_digest(config_path) != str(expected_config).lower():
                error(
                    errors,
                    job.get("job_id"),
                    "scientific_config.sha256",
                    expected_config,
                    file_digest(config_path),
                    "restore the frozen config bytes",
                )
    remote_hosts = {
        host: profile
        for host, profile in manifest.get("hosts", {}).items()
        if isinstance(profile, dict) and profile.get("backend", "local") == "external"
    }
    remote_bindings: dict[str, list[dict[str, Any]]] = {}
    for host in remote_hosts:
        host_jobs = [job for job in manifest.get("jobs", []) if job.get("host") == host]
        if not host_jobs:
            continue
        first_identity = host_jobs[0].get("scientific_identity", {})
        teacher = first_identity.get("teacher", {}) if isinstance(first_identity, dict) else {}
        remote_bindings[host] = remote_artifact_bindings(
            jobs=host_jobs,
            dataset_identity=manifest.get("launch_gate", {}).get("dataset_identity"),
            teacher_identity=teacher.get("identity") if isinstance(teacher, dict) else None,
            teacher=teacher if isinstance(teacher, dict) else {},
        )
    run_remote_preflights(
        hosts=remote_hosts,
        source_sha=str(expected_sha or ""),
        bindings_by_host=remote_bindings,
        errors=errors,
    )
    return manifest, errors


def validate_run(resolved_path: Path) -> tuple[bool, dict[str, Any]]:
    manifest = json.loads(resolved_path.read_text(encoding="utf-8"))
    state_path = Path(manifest["state_path"])
    errors: list[dict[str, Any]] = []
    if not state_path.is_file():
        error(errors, None, "state_path", "existing state", str(state_path), "resume the detached controller")
        return False, {"status": "fail", "errors": errors}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    module = orchestrator_module()
    for job in manifest.get("jobs", []):
        record = state.get("jobs", {}).get(job["job_id"], {})
        if record.get("status") != "completed":
            error(
                errors,
                job["job_id"],
                "state.status",
                "completed",
                record.get("status"),
                "inspect the job failure before declaring the campaign complete",
            )
            continue
        marker = Path(job["completion_marker"])
        if not module.valid_marker(manifest, job, marker):
            error(
                errors,
                job["job_id"],
                "completion_marker",
                "valid marker",
                str(marker),
                "repair the producer output/lineage",
            )
        for raw in job.get("expected_outputs", []):
            item = _descriptor(raw, "output")
            if "epoch" not in item and job.get("expected_final_epoch") is not None:
                item["epoch"] = job["expected_final_epoch"]
            validate_artifact(
                item,
                base=Path(job["output_dir"]),
                errors=errors,
                job_id=job["job_id"],
                role="expected_output",
            )
            path = resolve_path(item.get("path"), Path(job["output_dir"]))
            if path is None or not path.is_file():
                error(
                    errors,
                    job["job_id"],
                    "expected_output",
                    "existing file",
                    str(path),
                    "produce every required output",
                )
            elif item.get("sha256") and file_digest(path) != str(item["sha256"]).lower():
                error(
                    errors,
                    job["job_id"],
                    "expected_output.sha256",
                    item["sha256"],
                    file_digest(path),
                    "restore the expected output bytes",
                )
    collection = manifest.get("launch_gate", {}).get("artifact_collection")
    if isinstance(collection, dict) and isinstance(collection.get("manifest_path"), str):
        inventory = Path(collection["manifest_path"])
        checked = subprocess.run(
            [sys.executable, str(artifact_inventory_script()), "validate", "--manifest", str(inventory)],
            text=True,
            capture_output=True,
        )
        if checked.returncode != 0:
            error(
                errors,
                None,
                "artifact_collection",
                "complete canonical SHA-verified inventory",
                checked.stdout or checked.stderr,
                "run collection/inventory before aggregation completion",
            )
    return not errors, {"status": "pass" if not errors else "fail", "errors": errors, "state": state}


def invoke_orchestrator_capture(command: str, manifest_path: Path, *, detached: bool = False) -> tuple[int, str, str]:
    script = Path(__file__).resolve().parents[2] / "multi-gpu-experiment-orchestrator" / "scripts" / "orchestrate.py"
    args = [sys.executable, str(script), command, "--manifest", str(manifest_path)]
    result = subprocess.run(args, text=True, capture_output=True)
    return result.returncode, result.stdout, result.stderr


def invoke_orchestrator(command: str, manifest_path: Path, *, detached: bool = False) -> int:
    return invoke_orchestrator_capture(command, manifest_path, detached=detached)[0]


def write_remote_preflight_record(gate_dir: Path, report: dict[str, Any]) -> None:
    """Persist one host-indexed bounded-preflight record beside the freeze."""
    records = report.get("remote_preflight", {})
    if isinstance(records, dict):
        atomic_json(gate_dir / "remote-preflight.json", {"schema_version": REMOTE_PREFLIGHT_SCHEMA, "hosts": records})


def _timing_value(ledger: list[dict[str, Any]], event_type: str) -> dt.datetime | None:
    for event in ledger:
        if event.get("event_type") == event_type and isinstance(event.get("timestamp"), str):
            try:
                value = dt.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            except ValueError:
                continue
            return value.astimezone(dt.UTC) if value.tzinfo is not None else None
    return None


def _timing_seconds(ledger: list[dict[str, Any]], start: str, finish: str) -> float | None:
    first = _timing_value(ledger, start)
    last = _timing_value(ledger, finish)
    return (last - first).total_seconds() if first is not None and last is not None else None


def preparation_summary(manifest: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Render the compact, non-scientific Fast preparation budget."""
    requested = _timing_value(ledger, "request_received") or _timing_value(ledger, "gate_started")
    ready = _timing_value(ledger, "ready_to_launch") or _timing_value(ledger, "manifest_frozen")
    controller = _timing_value(ledger, "controller_launched")
    return {
        "schema_version": 1,
        "operational_profile": manifest.get("launch_gate", {}).get("operational_profile"),
        "request_to_ready_seconds": (ready - requested).total_seconds()
        if requested is not None and ready is not None
        else None,
        "request_to_controller_seconds": (controller - requested).total_seconds()
        if requested is not None and controller is not None
        else None,
        "static_check_duration_seconds": _timing_seconds(ledger, "static_checks_started", "static_checks_passed"),
        "smoke_duration_seconds": _timing_seconds(ledger, "cli_smoke_started", "integration_smoke_passed"),
        "preflight_duration_seconds": _timing_seconds(ledger, "preflight_started", "preflight_passed"),
        "manifest_duration_seconds": _timing_seconds(ledger, "manifest_freeze_started", "manifest_frozen"),
        "manifest_freeze_cycles": sum(event.get("event_type") == "manifest_frozen" for event in ledger),
        "controller_launch_attempts": sum(event.get("event_type") == "controller_launch_attempt" for event in ledger),
        "events": ledger,
    }


def fast_launch_errors(manifest: dict[str, Any]) -> list[str]:
    """Require the existing safe proof set before accepting one-command Fast launch."""
    errors: list[str] = []
    profile = manifest.get("launch_gate", {}).get("operational_profile")
    if profile != FAST_EXISTING_RUNTIME:
        errors.append("fast launch requires FAST_EXISTING_RUNTIME; use the Full integration workflow")
    signature = manifest.get("launch_gate", {}).get("runtime_signature", {})
    if isinstance(signature, dict) and signature.get("required") and signature.get("status") != "validated":
        errors.append("fast launch requires a matching validated runtime signature")
    canary = manifest.get("launch_gate", {}).get("canary", {})
    if not isinstance(canary, dict) or not manifest.get("launch_gate", {}).get("require_exact_smoke"):
        errors.append("fast launch requires canary.require_exact_smoke=true")
    elif not isinstance(canary.get("static_cli"), list) or not canary["static_cli"]:
        errors.append("fast launch requires at least one bounded static CLI check")
    for job in manifest.get("jobs", []):
        if job.get("gpu_count", 1) != 1:
            errors.append(f"fast launch only supports the existing one-GPU-per-job contract: {job.get('job_id')}")
    _, group_errors = smoke_group_contract(manifest)
    errors.extend(group_errors)
    return errors


def write_preparation_summary(
    manifest: dict[str, Any], ledger: list[dict[str, Any]], gate_dir: Path, report: dict[str, Any]
) -> None:
    summary = preparation_summary(manifest, ledger)
    report["timing_ledger"] = ledger
    report["preparation_summary"] = summary
    if manifest.get("launch_gate", {}).get("operational_profile") == FAST_EXISTING_RUNTIME:
        atomic_json(gate_dir / "fast-path-summary.json", summary)


def gate_main(args: argparse.Namespace) -> int:
    gate_started_at = now()
    spec_path = args.campaign_spec.resolve()
    gate_dir = (args.output_dir or spec_path.parent / ".launch-gate" / spec_path.stem).resolve()
    freeze_path = gate_dir / "freeze.json"
    if args.validate_run:
        ok, report = validate_run(args.resolved_manifest.resolve())
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ok else 2
    ledger: list[dict[str, Any]] = [
        {"event_type": "gate_started", "timestamp": gate_started_at},
        {"event_type": "preflight_started", "timestamp": gate_started_at},
    ]
    try:
        spec = load_document(spec_path)
        manifest, report = resolve_campaign(spec, spec_path)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "errors": [{"field": "spec", "expected": "valid campaign spec", "observed": str(exc)}],
                },
                indent=2,
            )
        )
        return 2
    requested = spec.get("timing", {}) if isinstance(spec.get("timing", {}), dict) else {}
    if isinstance(requested.get("request_received"), str):
        ledger.insert(
            0,
            {
                "event_type": "request_received",
                "timestamp": requested["request_received"],
                "precision": requested.get("request_precision", "unknown"),
            },
        )
    ledger.append({"event_type": "input_inventory_complete", "timestamp": now()})
    ledger.append({"event_type": "source_ready", "timestamp": now()})
    if report["status"] != "pass":
        write_preparation_summary(manifest, ledger, gate_dir, report)
        write_remote_preflight_record(gate_dir, report)
        atomic_json(gate_dir / "preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    dry_rows = plan_rows(manifest)
    plan_errors = [row for row in dry_rows if row.get("status") in {"plan_error", "resource_conflict"}]
    if plan_errors:
        report["status"] = "fail"
        report["errors"].extend(
            {
                "job": row.get("job_id"),
                "field": "resource_plan",
                "expected": "every job has a valid host/GPU slot",
                "observed": row,
                "remediation": "declare an available compatible slot or reduce concurrent jobs",
            }
            for row in plan_errors
        )
        report["dry_run"] = dry_rows
        write_preparation_summary(manifest, ledger, gate_dir, report)
        write_remote_preflight_record(gate_dir, report)
        atomic_json(gate_dir / "preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    manifest["launch_gate"]["planned_slots"] = {
        row["job_id"]: {key: row[key] for key in ("host", "gpu", "gpu_uuid") if key in row}
        for row in dry_rows
        if row.get("status") != "resource_conflict" and "job_id" in row
    }
    report["host_job_matrix"] = [
        {
            "job_id": row["job_id"],
            "host": row.get("host"),
            "gpu": row.get("gpu"),
            "gpu_uuid": row.get("gpu_uuid"),
            "scientific_identity_hash": row.get("scientific_identity_hash"),
            "smoke_group": next(
                (job.get("smoke_group") for job in manifest["jobs"] if job["job_id"] == row["job_id"]), None
            ),
        }
        for row in dry_rows
        if "job_id" in row
    ]
    ledger.append({"event_type": "host_config_matrix_complete", "timestamp": now()})
    ledger.append({"event_type": "remote_preflight_passed", "timestamp": now()})
    ledger.append({"event_type": "preflight_passed", "timestamp": now()})
    if args.fast_launch:
        errors = fast_launch_errors(manifest)
        if errors:
            report["status"] = "fail"
            report["errors"].extend(
                {
                    "job": None,
                    "field": "fast_launch",
                    "expected": "eligible existing runtime with exact smoke coverage",
                    "observed": message,
                    "remediation": "use the Full integration workflow or repair the complete Fast contract",
                }
                for message in errors
            )
            write_preparation_summary(manifest, ledger, gate_dir, report)
            write_remote_preflight_record(gate_dir, report)
            atomic_json(gate_dir / "preflight.json", report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2
    if args.canary_only or args.launch:
        ledger.append({"event_type": "static_checks_started", "timestamp": now()})
        static_report = run_static_cli_smokes(manifest, gate_dir)
        atomic_json(gate_dir / "static-cli.json", static_report)
        if static_report["status"] != "pass":
            print(json.dumps(static_report, indent=2, sort_keys=True))
            return 2
        ledger.append({"event_type": "static_checks_passed", "timestamp": now()})
        ledger.append({"event_type": "cli_smoke_started", "timestamp": now()})
        canary_report = run_canary(manifest, gate_dir)
        atomic_json(gate_dir / "canary.json", canary_report)
        if canary_report["status"] != "pass":
            print(json.dumps(canary_report, indent=2, sort_keys=True))
            return 2
        exact_errors = validate_exact_smoke_report(manifest, canary_report)
        if exact_errors:
            print(json.dumps({"status": "fail", "errors": exact_errors}, indent=2, sort_keys=True))
            return 2
        ledger.append({"event_type": "integration_smoke_passed", "timestamp": now()})
    ledger.append({"event_type": "manifest_freeze_started", "timestamp": now()})
    # Preparation timestamps are operational evidence, not immutable campaign
    # identity.  Keeping them out of the frozen manifest lets the ordinary
    # preflight -> canary -> launch workflow reuse the same scientific freeze;
    # they are persisted in preflight.json / fast-path-summary.json instead.
    report["dry_run"] = dry_rows
    try:
        frozen_path, manifest_sha = freeze(manifest, gate_dir)
    except ValueError as exc:
        report = {
            "status": "fail",
            "errors": [{"field": "freeze", "expected": "unchanged manifest", "observed": str(exc)}],
            "warnings": [],
        }
        atomic_json(gate_dir / "preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    report["manifest_sha256"] = manifest_sha
    ledger.append({"event_type": "manifest_frozen", "timestamp": now(), "manifest_sha256": manifest_sha})
    ledger.append({"event_type": "ready_to_launch", "timestamp": now()})
    write_preparation_summary(manifest, ledger, gate_dir, report)
    write_remote_preflight_record(gate_dir, report)
    atomic_json(gate_dir / "preflight.json", report)
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "manifest": str(frozen_path),
                    "manifest_sha256": manifest_sha,
                    "jobs": report["dry_run"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.launch:
        _, revalidation_errors = revalidate_frozen(frozen_path, freeze_path)
        revalidation_errors.extend(validate_exact_smoke_report(manifest, canary_report))
        if revalidation_errors:
            write_preparation_summary(manifest, ledger, gate_dir, report)
            atomic_json(gate_dir / "preflight.json", report)
            print(json.dumps({"status": "fail", "errors": revalidation_errors}, indent=2, sort_keys=True))
            return 2
        if invoke_orchestrator("preflight", frozen_path) != 0:
            return 2
        try:
            bridge_path = write_experiment_state(spec, manifest, frozen_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["status"] = "fail"
            report["errors"].append(
                {
                    "job": None,
                    "field": "experiment_state",
                    "expected": "new or matching schema-v2 campaign bridge",
                    "observed": str(exc),
                    "remediation": "repair the registered runtime state before launching",
                }
            )
            atomic_json(gate_dir / "preflight.json", report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2
        if bridge_path is not None:
            report["experiment_state_path"] = str(bridge_path)
        ledger.append({"event_type": "controller_launch_attempt", "timestamp": now()})
        launched, stdout, stderr = invoke_orchestrator_capture("run", frozen_path, detached=True)
        launch_attempt = {
            "manifest_sha256": manifest_sha,
            "source_sha": manifest["source"]["git_sha"],
            "returncode": launched,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        controller_pid = None
        if launched == 0:
            try:
                parsed = json.loads(stdout or "{}")
                controller_pid = parsed.get("controller_pid")
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(controller_pid, int) and controller_pid > 0:
                launch_attempt["controller_pid"] = controller_pid
                launch_attempt["controller_status"] = parsed.get("status")
                ledger.append({"event_type": "controller_launched", "timestamp": now(), "controller_pid": controller_pid})
                if bridge_path is not None:
                    update_experiment_state_launch(bridge_path, status="confirmed", attempt=launch_attempt)
            else:
                launched = 2
                launch_attempt["error"] = "detached controller did not return a valid controller_pid"
        if launched != 0 and bridge_path is not None:
            update_experiment_state_launch(bridge_path, status="failed", attempt=launch_attempt)
        write_preparation_summary(manifest, ledger, gate_dir, report)
        atomic_json(gate_dir / "preflight.json", report)
        return launched
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": str(frozen_path),
                "manifest_sha256": manifest_sha,
                "jobs": report["dry_run"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--campaign-spec", type=Path, required=False, help="human-authored JSON/YAML campaign spec")
    parser_obj.add_argument("--output-dir", type=Path)
    parser_obj.add_argument("--preflight-only", action="store_true")
    parser_obj.add_argument("--dry-run", action="store_true")
    parser_obj.add_argument("--canary-only", action="store_true")
    parser_obj.add_argument("--launch", action="store_true")
    parser_obj.add_argument(
        "--fast-launch",
        action="store_true",
        help="one-command Fast existing-runtime prepare -> exact smoke -> freeze -> detached launch",
    )
    parser_obj.add_argument("--validate-run", action="store_true")
    parser_obj.add_argument("--resolved-manifest", type=Path)
    parser_obj.add_argument("--promote-runtime-signature", action="store_true")
    parser_obj.add_argument("--exact-smoke-report", type=Path)
    parser_obj.add_argument("--signature-id")
    parser_obj.add_argument("--runtime-signature-registry", type=Path)
    return parser_obj


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.promote_runtime_signature:
        if args.resolved_manifest is None or args.exact_smoke_report is None or not args.signature_id:
            parser().error("--promote-runtime-signature requires --resolved-manifest, --exact-smoke-report, and --signature-id")
        try:
            result = promote_runtime_signature(
                args.resolved_manifest,
                args.exact_smoke_report,
                args.signature_id,
                args.runtime_signature_registry,
            )
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.fast_launch:
        incompatible = [
            flag
            for flag, selected in (
                ("--preflight-only", args.preflight_only),
                ("--dry-run", args.dry_run),
                ("--canary-only", args.canary_only),
                ("--launch", args.launch),
                ("--validate-run", args.validate_run),
            )
            if selected
        ]
        if incompatible:
            parser().error("--fast-launch cannot be combined with " + ", ".join(incompatible))
        args.launch = True
    if args.validate_run:
        if args.resolved_manifest is None:
            parser().error("--validate-run requires --resolved-manifest")
        args.campaign_spec = args.resolved_manifest
    elif args.campaign_spec is None:
        parser().error("--campaign-spec is required")
    return gate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
