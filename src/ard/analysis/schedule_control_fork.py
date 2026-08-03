"""Fail-closed epoch-79 scheduler-only counterfactual forks.

The fork is intentionally a control-plane operation: it copies every training
state byte-for-byte except the future MultiStepLR milestones, child identity,
and the explicitly reset post-fork model-selection state.  It never loads a
dataset or runs an optimizer step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.state import SampleStateStore
from ard.tracking import stable_run_id
from ard.tracking.adapter import collect_git_state

SCHEDULE_CONTROL_KIND = "delayed_multistep_schedule_control_v1"
PARENT_MILESTONES = (100, 150)
CHILD_MILESTONES = (120, 170)
PARENT_EPOCH = 79
PARENT_GLOBAL_STEP = 28_160
_LOGGING_ONLY_METHOD_ID = "rslad_logging_only"
_LOGGING_ONLY_DESIGN_ID = "logging_only_history_confirmatory_v1"
_LOGGING_ONLY_DESIGN_MANIFEST = "configs/analysis/logging_only_history_confirmatory_v1.yaml"
_LOGGING_ONLY_DESIGN_SHA256 = "d653d9ef08cfa94976a0e3279166b47543d16f3eaadb69810769470b77838c12"


class ScheduleControlForkError(RuntimeError):
    """The requested schedule-only fork lacks immutable parent evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_yaml_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScheduleControlForkError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ScheduleControlForkError(f"{name} must be a YAML mapping")
    return value


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScheduleControlForkError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ScheduleControlForkError(f"{name} must be a JSON mapping")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleControlForkError(f"{name} must be a mapping")
    return value


def _validate_historical_logging_only_metadata(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept exactly the retired, parity-attested L1 envelope.

    This is deliberately narrower than evaluation compatibility.  A schedule
    fork is a new training run, so only the known historical config shape may
    cross that boundary; arbitrary legacy metadata must not become a general
    resume escape hatch.
    """
    method = _require_mapping(raw.get("method"), name="historical parent method")
    if method.get("id") != _LOGGING_ONLY_METHOD_ID:
        raise ScheduleControlForkError("historical parent is not retired logging-only RSLAD")
    if "observation" in raw:
        raise ScheduleControlForkError("historical logging-only parent must have no observation metadata")
    design = _require_mapping(raw.get("research_design"), name="historical logging-only research design")
    if (
        set(design) != {"id", "manifest", "sha256"}
        or design.get("id") != _LOGGING_ONLY_DESIGN_ID
        or design.get("manifest") != _LOGGING_ONLY_DESIGN_MANIFEST
    ):
        raise ScheduleControlForkError("historical logging-only parent has incompatible research-design metadata")
    digest = design.get("sha256")
    if digest != _LOGGING_ONLY_DESIGN_SHA256:
        raise ScheduleControlForkError("historical logging-only parent has incompatible research-design SHA-256")
    return design


def parent_runtime_view(raw: Mapping[str, Any]) -> tuple[ExperimentConfig, dict[str, object] | None]:
    """Build the strict runtime view while retaining the original config hash.

    ``rslad_logging_only`` was loss-identical to RSLAD and was retired after
    L1.  Its exact, hash-bound source mapping remains the checkpoint identity;
    this function creates only the child runtime comparison view.
    """
    method = _require_mapping(raw.get("method"), name="parent method")
    if method.get("id") == _LOGGING_ONLY_METHOD_ID:
        design = _validate_historical_logging_only_metadata(raw)
        migrated = copy.deepcopy(dict(raw))
        migrated_method = _require_mapping(migrated.get("method"), name="historical parent method")
        migrated_method["id"] = "rslad"
        migrated.pop("research_design")
        migrated["observation"] = {"profile": "teacher_response"}
        try:
            config = ExperimentConfig.model_validate(migrated)
        except ValueError as exc:
            raise ScheduleControlForkError("historical logging-only parent cannot migrate to strict RSLAD") from exc
        return config, {
            "kind": "historical_logging_only_runtime_migration_v1",
            "source_method_id": _LOGGING_ONLY_METHOD_ID,
            "runtime_method_id": "rslad",
            "source_observation": "absent",
            "runtime_observation_profile": "teacher_response",
            "research_design": dict(design),
            "applied": [
                "rslad_logging_only_to_rslad",
                "research_design_removed",
                "teacher_response_observation_added",
            ],
        }
    if "research_design" in raw:
        raise ScheduleControlForkError("ordinary RSLAD parent cannot contain research-design metadata")
    try:
        return ExperimentConfig.model_validate(raw), None
    except ValueError as exc:
        raise ScheduleControlForkError("parent resolved config is not strict runnable configuration") from exc


# Historical callers used the private spelling while this is a narrow, shared
# compatibility boundary for the two epoch-boundary continuation tools.
_parent_runtime_view = parent_runtime_view


def _spec_parent(spec_path: Path) -> tuple[dict[str, Any], str]:
    """Load the analysis-owned immutable parent evidence, not a train config field."""
    spec = _load_yaml_mapping(spec_path, name="schedule-control spec")
    parent = spec.get("parent")
    expected = {"schema_version", "kind", "parent"}
    if set(spec) != expected or spec.get("schema_version") != 1 or spec.get("kind") != SCHEDULE_CONTROL_KIND:
        raise ScheduleControlForkError("schedule-control spec has unexpected or missing identity fields")
    if not isinstance(parent, Mapping):
        raise ScheduleControlForkError("schedule-control spec parent must be a mapping")
    required = {
        "checkpoint_sha256",
        "raw_config_sha256",
        "git_sha",
        "epoch",
        "world_size",
        "teacher_checkpoint_sha256",
        "sample_state_records",
        "sample_state_sha256",
        "train_partition_manifest",
        "train_partition_manifest_sha256",
        "train_partition_ids_labels_sha256",
        "artifact_attestation",
        "artifact_attestation_sha256",
        "artifact_inventory",
        "artifact_inventory_sha256",
    }
    if set(parent) != required:
        raise ScheduleControlForkError("schedule-control spec parent has unexpected or missing fields")
    values = dict(parent)
    if values["epoch"] != PARENT_EPOCH or values["world_size"] != 1 or values["sample_state_records"] != 45_000:
        raise ScheduleControlForkError("schedule-control spec does not bind the epoch-79 single-rank 45k parent")
    for name, length in (
        ("checkpoint_sha256", 64),
        ("raw_config_sha256", 64),
        ("git_sha", 40),
        ("teacher_checkpoint_sha256", 64),
        ("sample_state_sha256", 64),
        ("train_partition_manifest_sha256", 64),
        ("train_partition_ids_labels_sha256", 64),
        ("artifact_attestation_sha256", 64),
        ("artifact_inventory_sha256", 64),
    ):
        value = values[name]
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ScheduleControlForkError(
                f"schedule-control spec {name} must be a lowercase {length}-character digest"
            )
    return values, sha256_file(spec_path)


def validate_train_partition(parent: Mapping[str, Any], *, labels: Mapping[int, int], num_classes: int) -> None:
    path = Path(str(parent["train_partition_manifest"]))
    if not path.is_file() or sha256_file(path) != parent["train_partition_manifest_sha256"]:
        raise ScheduleControlForkError("train-partition manifest bytes do not match bound SHA-256")
    manifest = _load_json_mapping(path, name="train-partition manifest")
    if set(manifest) != {"schema_version", "namespace", "ids_labels", "ids_labels_sha256"}:
        raise ScheduleControlForkError("train-partition manifest has unexpected or missing fields")
    rows = manifest.get("ids_labels")
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or len(row) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in row)
        for row in rows
    ):
        raise ScheduleControlForkError("train-partition manifest IDs and labels must be integer pairs")
    normalized = [(row[0], row[1]) for row in rows]
    if manifest.get("schema_version") != 1 or manifest.get("namespace") != "train" or normalized != sorted(normalized):
        raise ScheduleControlForkError("train-partition manifest must be sorted exact train-only identity")
    if len(normalized) != len(set(sample_id for sample_id, _ in normalized)) or any(
        label < 0 or label >= num_classes for _, label in normalized
    ):
        raise ScheduleControlForkError("train-partition manifest has duplicate IDs or invalid labels")
    digest = _canonical_sha256(normalized)
    if manifest.get("ids_labels_sha256") != digest or digest != parent["train_partition_ids_labels_sha256"]:
        raise ScheduleControlForkError("train-partition manifest ID/label digest does not match")
    if dict(normalized) != dict(labels):
        raise ScheduleControlForkError("parent sample state does not exactly match train partition identity")


_validate_train_partition = validate_train_partition


def _validate_scheduler_parent(payload: Mapping[str, Any]) -> None:
    state = _require_mapping(payload.get("scheduler"), name="parent scheduler")
    expected_milestones = Counter({100: 1, 150: 1})
    if (
        state.get("milestones") != expected_milestones
        or state.get("gamma") != 0.1
        or state.get("last_epoch") != 80
        or state.get("_step_count") != 81
        or state.get("base_lrs") != [0.1]
        or state.get("_last_lr") != [0.1]
    ):
        raise ScheduleControlForkError("parent scheduler is not the epoch-79 pre-decay MultiStepLR [100,150]")
    optimizer = _require_mapping(payload.get("optimizer"), name="parent optimizer")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get("lr") != 0.1:
        raise ScheduleControlForkError("parent optimizer LR is not the required pre-decay 0.1")


def _replace_scheduler_milestones(parent_state: Mapping[str, Any]) -> dict[str, Any]:
    _validate_scheduler_parent({"scheduler": parent_state, "optimizer": {"param_groups": [{"lr": 0.1}]}})
    child = copy.deepcopy(dict(parent_state))
    child["milestones"] = Counter({120: 1, 170: 1})
    return child


def _post_fork_selection_metadata(parent: Mapping[str, Any]) -> dict[str, Any]:
    child = copy.deepcopy(dict(parent))
    child["selected_epoch"] = None
    for key in (
        "selected_clean_accuracy",
        "selected_pgd_accuracy",
        "last_epoch",
        "last_clean_accuracy",
        "last_pgd_accuracy",
    ):
        child.pop(key, None)
    child["scope"] = "post_fork_best"
    return child


def _validate_parent(
    *,
    checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    artifact_inventory: Path,
    artifact_attestation: Path,
    child: ExperimentConfig,
    parent: Mapping[str, Any],
) -> tuple[dict[str, Any], ExperimentConfig, dict[str, Any], dict[str, object] | None]:
    if artifact_inventory.resolve() != Path(str(parent["artifact_inventory"])).resolve():
        raise ScheduleControlForkError("artifact inventory path does not match schedule-control spec")
    if artifact_attestation.resolve() != Path(str(parent["artifact_attestation"])).resolve():
        raise ScheduleControlForkError("artifact attestation path does not match schedule-control spec")
    if not checkpoint.is_file() or sha256_file(checkpoint) != parent["checkpoint_sha256"]:
        raise ScheduleControlForkError("parent checkpoint bytes do not match bound SHA-256")
    raw = _load_yaml_mapping(parent_resolved_config, name="parent resolved config")
    raw_hash = config_digest(raw)
    if raw_hash != parent["raw_config_sha256"]:
        raise ScheduleControlForkError("parent raw resolved config SHA-256 does not match schedule-control lineage")
    source, compatibility_migration = _parent_runtime_view(raw)
    if source.intervention is not None:
        raise ScheduleControlForkError("schedule-control parent must be ordinary observed RSLAD")
    if (
        source.protocol.id != "controlled_cifar10_r18_v1"
        or source.method.id != "rslad"
        or source.observation.profile != "teacher_response"
        or source.dataset.name != "cifar10"
        or source.training.epochs != 200
        or source.training.per_rank_batch_size != 128
        or source.training.global_batch_size != 128
        or source.scheduler.id != "multistep"
        or source.scheduler.milestones != PARENT_MILESTONES
        or source.scheduler.gamma != 0.1
        or source.scheduler.step_at != "epoch_end"
    ):
        raise ScheduleControlForkError("parent is not the controlled epoch-79 Bartoldson RSLAD schedule baseline")
    if (
        source.teacher is None
        or source.teacher.registry_id != "bartoldson2024_adversarial_wrn94_16"
        or source.teacher.checkpoint_sha256 != parent["teacher_checkpoint_sha256"]
    ):
        raise ScheduleControlForkError("parent teacher SHA does not match schedule-control lineage")
    manifest = _load_json_mapping(parent_manifest, name="parent manifest")
    manifest_git = _require_mapping(manifest.get("git"), name="parent manifest git")
    manifest_teacher = _require_mapping(manifest.get("teacher"), name="parent manifest teacher")
    if (
        manifest.get("config_hash") != raw_hash
        or manifest_git.get("sha") != parent["git_sha"]
        or manifest_teacher.get("checkpoint_sha256") != parent["teacher_checkpoint_sha256"]
    ):
        raise ScheduleControlForkError("parent manifest config, Git, or teacher lineage does not match")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ScheduleControlForkError("parent checkpoint must be a mapping")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ScheduleControlForkError("parent checkpoint is incomplete; missing: " + ", ".join(sorted(missing)))
    if (
        payload.get("epoch") != PARENT_EPOCH
        or payload.get("epoch_boundary") != "end"
        or payload.get("world_size") != 1
        or payload.get("config_hash") != raw_hash
        or payload.get("global_step") != PARENT_GLOBAL_STEP
        or manifest.get("run_id") != payload.get("tracker_run_id")
    ):
        raise ScheduleControlForkError("parent checkpoint is not the exact epoch-79 single-rank boundary")
    _validate_scheduler_parent(payload)
    rng = payload.get("rng")
    sampler_states = payload.get("sampler_state")
    if not isinstance(rng, list) or len(rng) != 1 or not isinstance(sampler_states, list) or len(sampler_states) != 1:
        raise ScheduleControlForkError("parent checkpoint lacks complete single-rank RNG or sampler state")
    rank_rng = _require_mapping(rng[0], name="parent rank-zero RNG")
    if set(rank_rng) != {"python", "torch_cpu", "torch_cuda", "numpy"} or any(
        rank_rng[key] is None for key in ("python", "torch_cpu", "torch_cuda", "numpy")
    ):
        raise ScheduleControlForkError("parent checkpoint RNG state is incomplete")
    expected_sampler = {
        "epoch": PARENT_EPOCH,
        "seed": source.seeds.data_order,
        "rank": 0,
        "world_size": 1,
        "shuffle": True,
    }
    if payload.get("sampler_epoch") != [PARENT_EPOCH] or sampler_states[0] != expected_sampler:
        raise ScheduleControlForkError("parent sampler state is not the exact epoch-79 single-rank identity")
    sample_state = _require_mapping(payload.get("sample_state"), name="parent sample state")
    if _canonical_sha256(sample_state) != parent["sample_state_sha256"]:
        raise ScheduleControlForkError("parent sample-state SHA-256 does not match lineage")
    if sample_state.get("format_version") != SampleStateStore.FORMAT_VERSION or sample_state.get("pending") != []:
        raise ScheduleControlForkError("parent sample state is not complete at an epoch boundary")
    store = SampleStateStore(ema_decay=source.method.student_ema_decay)
    try:
        store.load_state_dict(sample_state)
    except (TypeError, ValueError) as exc:
        raise ScheduleControlForkError("parent sample state is invalid") from exc
    if len(store.records) != parent["sample_state_records"]:
        raise ScheduleControlForkError("parent sample state does not contain the full 45k train partition")
    labels: dict[int, int] = {}
    for sample_id, record in store.records.items():
        if record.true_label is None or record.seen != 80:
            raise ScheduleControlForkError("parent sample state lacks an epoch-79 train label/history record")
        labels[sample_id] = record.true_label
    _validate_train_partition(parent, labels=labels, num_classes=child.dataset.num_classes)
    if not artifact_inventory.is_file() or sha256_file(artifact_inventory) != parent["artifact_inventory_sha256"]:
        raise ScheduleControlForkError("artifact inventory bytes do not match bound SHA-256")
    inventory = _load_json_mapping(artifact_inventory, name="artifact inventory")
    artifact = _require_mapping(inventory.get("artifact"), name="artifact inventory artifact")
    if (
        set(artifact) != {"name", "version", "digest", "checkpoint_sha256"}
        or artifact.get("checkpoint_sha256") != parent["checkpoint_sha256"]
    ):
        raise ScheduleControlForkError("artifact inventory does not attest exact parent checkpoint")
    if not artifact_attestation.is_file() or sha256_file(artifact_attestation) != parent["artifact_attestation_sha256"]:
        raise ScheduleControlForkError("artifact attestation bytes do not match bound SHA-256")
    attestation = _load_json_mapping(artifact_attestation, name="artifact attestation")
    if (
        attestation.get("parent_manifest_path") != str(parent_manifest.resolve())
        or attestation.get("parent_manifest_sha256") != sha256_file(parent_manifest)
        or attestation.get("artifact_inventory_path") != str(artifact_inventory.resolve())
        or attestation.get("artifact_inventory_sha256") != sha256_file(artifact_inventory)
        or attestation.get("run_id") != payload.get("tracker_run_id")
        or attestation.get("config_hash") != raw_hash
        or attestation.get("git_sha") != parent["git_sha"]
        or attestation.get("teacher_checkpoint_sha256") != parent["teacher_checkpoint_sha256"]
        or attestation.get("checkpoint_sha256") != parent["checkpoint_sha256"]
        or attestation.get("artifact") != artifact
    ):
        raise ScheduleControlForkError("artifact attestation does not bind immutable parent lineage")
    return payload, source, dict(artifact), compatibility_migration


def _validate_allowed_delta(*, parent: ExperimentConfig, child: ExperimentConfig) -> None:
    parent_runtime = resolved_config_dict(parent)
    child_runtime = resolved_config_dict(child)
    parent_protocol = parent_runtime.pop("protocol")
    child_protocol = child_runtime.pop("protocol")
    if parent_protocol != {"id": "controlled_cifar10_r18_v1"} or child_protocol != {
        "id": "controlled_cifar10_r18_delayed_multistep_v1"
    }:
        raise ScheduleControlForkError(
            "schedule-control requires the registered baseline and delayed protocol identities"
        )
    for key in ("output_dir", "tracking"):
        parent_runtime.pop(key, None)
        child_runtime.pop(key, None)
    parent_scheduler = parent_runtime.pop("scheduler")
    child_scheduler = child_runtime.pop("scheduler")
    if parent_runtime != child_runtime:
        raise ScheduleControlForkError("child changes fields outside output/tracking/protocol/scheduler whitelist")
    if not isinstance(parent_scheduler, Mapping) or not isinstance(child_scheduler, Mapping):
        raise ScheduleControlForkError("parent and child scheduler mappings are required")
    expected_parent = {"id": "multistep", "milestones": list(PARENT_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"}
    expected_child = {"id": "multistep", "milestones": list(CHILD_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"}
    if dict(parent_scheduler) != expected_parent or dict(child_scheduler) != expected_child:
        raise ScheduleControlForkError("schedule-control may change only [100,150] to [120,170] at epoch end")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_schedule_control_fork(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    artifact_inventory: Path,
    artifact_attestation: Path,
    spec_path: Path,
    child_config_path: Path,
    root: Path,
    git_state_collector: Callable[[Path], Mapping[str, Any]] = collect_git_state,
) -> Path:
    """Create one strict epoch-79 delayed-schedule child checkpoint atomically."""
    child = load_config(child_config_path)
    if child.intervention is not None:
        raise ScheduleControlForkError("schedule-control child cannot be an intervention arm")
    parent, spec_sha256 = _spec_parent(spec_path)
    payload, source, artifact, compatibility_migration = _validate_parent(
        checkpoint=parent_checkpoint,
        parent_resolved_config=parent_resolved_config,
        parent_manifest=parent_manifest,
        artifact_inventory=artifact_inventory,
        artifact_attestation=artifact_attestation,
        child=child,
        parent=parent,
    )
    _validate_allowed_delta(parent=source, child=child)
    output = child.output_dir.resolve()
    if output == parent_checkpoint.resolve().parent or output.exists():
        raise ScheduleControlForkError("schedule-control child output must be a new directory distinct from parent")
    git = git_state_collector(root)
    git_sha = git.get("sha")
    if not isinstance(git_sha, str) or len(git_sha) != 40 or git.get("dirty") is not False:
        raise ScheduleControlForkError("schedule-control fork requires a clean addressable current Git SHA")
    child_hash = config_digest(resolved_config_dict(child))
    child_run_id = stable_run_id(child, config_hash=child_hash, git_sha=git_sha)
    if child_run_id == payload.get("tracker_run_id"):
        raise ScheduleControlForkError("schedule-control child must not reuse the parent tracking run ID")
    transformed = copy.deepcopy(payload)
    transformed["config_hash"] = child_hash
    transformed["tracker_run_id"] = child_run_id
    transformed["scheduler"] = _replace_scheduler_milestones(
        _require_mapping(payload["scheduler"], name="parent scheduler")
    )
    transformed["best_metric"] = float("-inf")
    transformed["selection_metadata"] = _post_fork_selection_metadata(
        _require_mapping(payload["selection_metadata"], name="parent selection metadata")
    )
    fork_lineage: dict[str, object] = {
        "kind": SCHEDULE_CONTROL_KIND,
        "child_tracker_run_id": child_run_id,
        "parent_tracker_run_id": payload["tracker_run_id"],
        "child_config_sha256": child_hash,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "schedule_control_spec_sha256": spec_sha256,
        "parent_raw_config_sha256": parent["raw_config_sha256"],
        "parent_git_sha": parent["git_sha"],
        "parent_epoch": PARENT_EPOCH,
        "parent_world_size": 1,
        "parent_teacher_checkpoint_sha256": parent["teacher_checkpoint_sha256"],
        "parent_sample_state_records": 45000,
        "parent_sample_state_sha256": parent["sample_state_sha256"],
        "parent_artifact_attestation_sha256": parent["artifact_attestation_sha256"],
        "parent_wandb_checkpoint_artifact": artifact,
        "parent_scheduler": {"milestones": list(PARENT_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"},
        "child_scheduler": {"milestones": list(CHILD_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"},
        "fork_git_sha": git_sha,
        "post_fork_best_scope": True,
    }
    if compatibility_migration is not None:
        fork_lineage["parent_config_compatibility_migration"] = compatibility_migration
    transformed["fork_lineage"] = fork_lineage
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".schedule-control-", dir=output.parent))
    try:
        _atomic_torch_save(transformed, staging / "last.pt")
        save_resolved_config(child, staging / "resolved_config.yaml")
        (staging / "fork-lineage.json").write_text(
            json.dumps(transformed["fork_lineage"], sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "schedule-control-complete.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": SCHEDULE_CONTROL_KIND,
                    "status": "complete",
                    "config_hash": child_hash,
                    "run_id": child_run_id,
                    "fork_checkpoint_sha256": sha256_file(staging / "last.pt"),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "last.pt"
