#!/usr/bin/env python3
"""Resolve and validate a scientific campaign before handing it to the orchestrator.

The gate is deliberately small and method agnostic.  It protects the inputs and
lineage of a campaign, then delegates scheduling and detached execution to
``multi-gpu-experiment-orchestrator/scripts/orchestrate.py``.  JSON is the
canonical interchange format; YAML is accepted when PyYAML is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
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


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


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
        if isinstance(final_epoch, int) and job.get("job_type", "training") == "training":
            command = _replace_epoch_bound(command, final_epoch, errors, job_id)
            authored_runtime_epochs = job.get("runtime_epochs")
            if authored_runtime_epochs is not None and authored_runtime_epochs != final_epoch + 1:
                error(
                    errors,
                    job_id,
                    "runtime_epochs",
                    final_epoch + 1,
                    authored_runtime_epochs,
                    "use the exclusive runtime bound derived from scientific_final_epoch",
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
            "expected_outputs": expected_outputs,
            "expected_final_epoch": job.get("expected_final_epoch", final_epoch),
            "epoch_binding": {
                "scientific_final_epoch": final_epoch,
                "runtime_exclusive_epochs": final_epoch + 1 if isinstance(final_epoch, int) else None,
            },
            "dataset": {
                "identity": dataset_identity,
                "split_identity": split_identity,
                "path": str(dataset_path) if dataset_path else None,
            },
            "parent": parent,
            "scientific_config": config_item,
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
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "source": {"git_sha": source.get("git_sha")},
        "state_path": str(resolve_path(spec.get("state_path") or f".orchestration/{campaign_id}.state.json", base)),
        "reservation_root": spec.get("reservation_root"),
        "hosts": hosts,
        "jobs": resolved_jobs,
        "launch_gate": {
            "schema_version": SCHEMA_VERSION,
            "scientific_identity_hashes": {job["job_id"]: job["identity_hash"] for job in resolved_jobs},
            "source_repo": source.get("repo_path"),
            "dataset_identity": dataset_identity,
            "split_identity": split_identity,
            "canary": spec.get("canary", {}),
        },
    }
    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "resolved_jobs": resolved_jobs,
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


def run_canary(manifest: dict[str, Any], gate_dir: Path) -> dict[str, Any]:
    canary = manifest.get("launch_gate", {}).get("canary", {})
    entries = canary.get("jobs", []) if isinstance(canary, dict) else []
    results: list[dict[str, Any]] = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("job_id"), str)
            or not isinstance(entry.get("command"), list)
        ):
            results.append({"status": "fail", "error": "canary jobs require job_id and argv command"})
            continue
        job = next((item for item in manifest["jobs"] if item["job_id"] == entry["job_id"]), None)
        if job is None:
            results.append({"job_id": entry["job_id"], "status": "fail", "error": "unknown production job"})
            continue
        output = gate_dir / "canary" / entry["job_id"]
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
        env["ARD_LAUNCH_GATE_CANARY"] = "1"
        env["ARD_ORCH_JOB_ID"] = job["job_id"]
        env["ARD_ORCH_SOURCE_SHA"] = manifest["source"]["git_sha"]
        if job.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        timeout = float(entry.get("timeout_seconds", 30))
        try:
            completed = subprocess.run(
                entry["command"], cwd=job["cwd"], env=env, text=True, capture_output=True, timeout=timeout
            )
            (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            results.append(
                {
                    "job_id": job["job_id"],
                    "status": "pass" if completed.returncode == 0 else "fail",
                    "returncode": completed.returncode,
                    "timeout_seconds": timeout,
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {"job_id": job["job_id"], "status": "fail", "error": "canary timeout", "timeout_seconds": timeout}
            )
    if not entries:
        results.append(
            {
                "status": "fail",
                "error": "no canary jobs declared; use --canary-only with a bounded non-scientific canary",
            }
        )
    return {"status": "pass" if all(item.get("status") == "pass" for item in results) else "fail", "results": results}


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
    return not errors, {"status": "pass" if not errors else "fail", "errors": errors, "state": state}


def invoke_orchestrator(command: str, manifest_path: Path, *, detached: bool = False) -> int:
    script = Path(__file__).resolve().parents[2] / "multi-gpu-experiment-orchestrator" / "scripts" / "orchestrate.py"
    args = [sys.executable, str(script), command, "--manifest", str(manifest_path)]
    if command == "run" and detached:
        result = subprocess.run(args, text=True)
    else:
        result = subprocess.run(args, text=True)
    return result.returncode


def gate_main(args: argparse.Namespace) -> int:
    spec_path = args.campaign_spec.resolve()
    gate_dir = (args.output_dir or spec_path.parent / ".launch-gate" / spec_path.stem).resolve()
    freeze_path = gate_dir / "freeze.json"
    if args.validate_run:
        ok, report = validate_run(args.resolved_manifest.resolve())
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if ok else 2
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
    if report["status"] != "pass":
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
        atomic_json(gate_dir / "preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    manifest["launch_gate"]["planned_slots"] = {
        row["job_id"]: {key: row[key] for key in ("host", "gpu", "gpu_uuid") if key in row}
        for row in dry_rows
        if row.get("status") != "resource_conflict" and "job_id" in row
    }
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
    if args.canary_only or args.launch:
        canary_report = run_canary(manifest, gate_dir)
        atomic_json(gate_dir / "canary.json", canary_report)
        if canary_report["status"] != "pass":
            print(json.dumps(canary_report, indent=2, sort_keys=True))
            return 2
    if args.launch:
        _, revalidation_errors = revalidate_frozen(frozen_path, freeze_path)
        if revalidation_errors:
            print(json.dumps({"status": "fail", "errors": revalidation_errors}, indent=2, sort_keys=True))
            return 2
        if invoke_orchestrator("preflight", frozen_path) != 0:
            return 2
        return invoke_orchestrator("run", frozen_path, detached=True)
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
    parser_obj.add_argument("--validate-run", action="store_true")
    parser_obj.add_argument("--resolved-manifest", type=Path)
    return parser_obj


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.validate_run:
        if args.resolved_manifest is None:
            parser().error("--validate-run requires --resolved-manifest")
        args.campaign_spec = args.resolved_manifest
    elif args.campaign_spec is None:
        parser().error("--campaign-spec is required")
    return gate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
