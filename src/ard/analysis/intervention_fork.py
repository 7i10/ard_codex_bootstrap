"""Fail-closed common-state fork for the registered H3 intervention screen."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.intervention_selector import SelectorBundleError, verify_selector_bundle
from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.config.schema import InterventionConfig
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.policies import FixedMaskError, load_fixed_intervention_mask
from ard.state import SampleStateStore
from ard.tracking import stable_run_id
from ard.tracking.adapter import collect_git_state


class InterventionForkError(RuntimeError):
    """A requested child cannot prove common-state scientific lineage."""


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


def build_parent_artifact_attestation(
    *, parent_manifest: Path, artifact_inventory: Path, checkpoint: Path
) -> dict[str, object]:
    manifest = _load_json_mapping(parent_manifest, name="parent manifest")
    inventory = _load_json_mapping(artifact_inventory, name="artifact inventory")
    return {
        "schema_version": 1,
        "parent_manifest_path": str(parent_manifest.resolve()),
        "parent_manifest_sha256": sha256_file(parent_manifest),
        "artifact_inventory_path": str(artifact_inventory.resolve()),
        "artifact_inventory_sha256": sha256_file(artifact_inventory),
        "run_id": manifest.get("run_id"),
        "config_hash": manifest.get("config_hash"),
        "git_sha": _require_mapping(manifest.get("git"), name="parent manifest git").get("sha"),
        "teacher_checkpoint_sha256": _require_mapping(manifest.get("teacher"), name="parent manifest teacher").get(
            "checkpoint_sha256"
        ),
        "artifact": inventory.get("artifact"),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InterventionForkError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise InterventionForkError(f"{name} must be a YAML mapping")
    return value


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterventionForkError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise InterventionForkError(f"{name} must be a JSON mapping")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterventionForkError(f"{name} must be a mapping")
    return value


def _load_train_partition(
    parent: object,
    *,
    expected_labels: Mapping[int, int],
    num_classes: int,
) -> None:
    path = getattr(parent, "train_partition_manifest")
    if (
        not isinstance(path, Path)
        or not path.is_file()
        or sha256_file(path) != getattr(parent, "train_partition_manifest_sha256")
    ):
        raise InterventionForkError("train-partition manifest bytes do not match the bound parent SHA-256")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterventionForkError("train-partition manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "namespace",
        "ids_labels",
        "ids_labels_sha256",
    }:
        raise InterventionForkError("train-partition manifest has unexpected or missing fields")
    rows = manifest.get("ids_labels")
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or len(row) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in row)
        for row in rows
    ):
        raise InterventionForkError("train-partition manifest IDs and labels must be integer pairs")
    normalized = [(row[0], row[1]) for row in rows]
    if manifest.get("schema_version") != 1 or manifest.get("namespace") != "train" or normalized != sorted(normalized):
        raise InterventionForkError("train-partition manifest must be sorted exact train-only identity")
    if len({sample_id for sample_id, _ in normalized}) != len(normalized) or any(
        label < 0 or label >= num_classes for _, label in normalized
    ):
        raise InterventionForkError("train-partition manifest has duplicate IDs or invalid labels")
    digest = _canonical_sha256(normalized)
    if manifest.get("ids_labels_sha256") != digest or digest != getattr(parent, "train_partition_ids_labels_sha256"):
        raise InterventionForkError("train-partition manifest ID/label digest does not match")
    if dict(normalized) != dict(expected_labels):
        raise InterventionForkError("parent sample-state IDs/labels do not exactly match the train partition manifest")


def _validate_parent(
    *,
    checkpoint: Path,
    parent_raw_config: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    parent_manifest_path: Path,
    arm: ExperimentConfig,
) -> tuple[dict[str, Any], dict[int, int], dict[str, Any]]:
    intervention = arm.intervention
    assert intervention is not None
    parent = intervention.parent
    if not checkpoint.is_file() or sha256_file(checkpoint) != parent.checkpoint_sha256:
        raise InterventionForkError("parent checkpoint bytes do not match the arm's bound SHA-256")
    raw_hash = config_digest(parent_raw_config)
    if raw_hash != parent.raw_config_sha256:
        raise InterventionForkError("parent raw resolved config SHA-256 does not match the arm")
    try:
        source = ExperimentConfig.model_validate(parent_raw_config)
    except ValueError as exc:
        raise InterventionForkError("parent raw resolved config is not a strict runnable configuration") from exc
    if source.method.id != "rslad" or source.observation.profile != "teacher_response":
        raise InterventionForkError("parent must be observed baseline RSLAD with teacher_response")
    if (
        source.protocol.id != "controlled_cifar10_r18_v1"
        or source.dataset.name != "cifar10"
        or source.training.epochs != 200
        or source.training.per_rank_batch_size != 128
        or source.training.global_batch_size != 128
        or parent.epoch not in {79, 99}
    ):
        raise InterventionForkError("parent must retain the controlled CIFAR-10 45k/128 protocol identity")
    if source.teacher is None or source.teacher.checkpoint_sha256 != parent.teacher_checkpoint_sha256:
        raise InterventionForkError("parent teacher SHA does not match the bound arm lineage")
    manifest_git = _require_mapping(parent_manifest.get("git"), name="parent manifest git")
    manifest_teacher = _require_mapping(parent_manifest.get("teacher"), name="parent manifest teacher")
    if parent_manifest.get("config_hash") != raw_hash or manifest_git.get("sha") != parent.git_sha:
        raise InterventionForkError("parent manifest config or Git lineage does not match the arm")
    if manifest_teacher.get("checkpoint_sha256") != parent.teacher_checkpoint_sha256:
        raise InterventionForkError("parent manifest teacher lineage does not match the arm")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise InterventionForkError("parent checkpoint must be a mapping")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise InterventionForkError("parent checkpoint is incomplete; missing: " + ", ".join(sorted(missing)))
    if (
        payload.get("epoch") != parent.epoch
        or payload.get("epoch_boundary") != "end"
        or payload.get("world_size") != parent.world_size
        or payload.get("config_hash") != raw_hash
    ):
        raise InterventionForkError("parent checkpoint is not the exact bound single-world-size state")
    if parent_manifest.get("run_id") != payload.get("tracker_run_id"):
        raise InterventionForkError("parent manifest run ID does not match the checkpoint tracker identity")
    attestation_path = parent.artifact_attestation
    if not attestation_path.is_file() or sha256_file(attestation_path) != parent.artifact_attestation_sha256:
        raise InterventionForkError("parent artifact attestation bytes do not match the bound SHA-256")
    attestation = _load_json_mapping(attestation_path, name="parent artifact attestation")
    inventory_path = parent.artifact_inventory
    if not inventory_path.is_file() or sha256_file(inventory_path) != parent.artifact_inventory_sha256:
        raise InterventionForkError("parent artifact inventory bytes do not match the bound SHA-256")
    inventory = _load_json_mapping(inventory_path, name="parent artifact inventory")
    inventory_artifact = _require_mapping(inventory.get("artifact"), name="parent artifact inventory artifact")
    if set(inventory_artifact) != {"name", "version", "digest", "checkpoint_sha256"} or any(
        not isinstance(inventory_artifact[key], str) or not inventory_artifact[key]
        for key in ("name", "version", "digest", "checkpoint_sha256")
    ):
        raise InterventionForkError(
            "parent artifact inventory must select exactly artifact name/version/digest/checkpoint SHA"
        )
    if inventory_artifact["checkpoint_sha256"] != parent.checkpoint_sha256:
        raise InterventionForkError("parent artifact inventory checkpoint SHA does not match the bound checkpoint")
    if (
        attestation.get("parent_manifest_path") != str(parent_manifest_path.resolve())
        or attestation.get("parent_manifest_sha256") != sha256_file(parent_manifest_path)
        or attestation.get("artifact_inventory_path") != str(inventory_path.resolve())
        or attestation.get("artifact_inventory_sha256") != sha256_file(inventory_path)
        or attestation.get("run_id") != payload.get("tracker_run_id")
        or attestation.get("config_hash") != raw_hash
        or attestation.get("git_sha") != parent.git_sha
        or attestation.get("teacher_checkpoint_sha256") != parent.teacher_checkpoint_sha256
        or attestation.get("checkpoint_sha256") != parent.checkpoint_sha256
        or attestation.get("artifact") != inventory_artifact
    ):
        raise InterventionForkError(
            "parent artifact attestation does not bind immutable manifest/inventory/checkpoint lineage"
        )
    if not isinstance(payload.get("rng"), list) or len(payload["rng"]) != 1:
        raise InterventionForkError("parent checkpoint lacks complete single-rank RNG state")
    if not isinstance(payload.get("sampler_state"), list) or len(payload["sampler_state"]) != 1:
        raise InterventionForkError("parent checkpoint lacks complete single-rank sampler state")
    rng = _require_mapping(payload["rng"][0], name="parent checkpoint rank-zero RNG")
    if set(rng) != {"python", "torch_cpu", "torch_cuda", "numpy"} or any(
        rng[key] is None for key in ("python", "torch_cpu", "torch_cuda", "numpy")
    ):
        raise InterventionForkError("parent checkpoint RNG state is incomplete")
    if payload.get("sampler_epoch") != [parent.epoch] or payload["sampler_state"][0] != {
        "epoch": parent.epoch,
        "seed": source.seeds.data_order,
        "rank": 0,
        "world_size": 1,
        "shuffle": True,
    }:
        raise InterventionForkError("parent sampler state does not prove the bound single-rank training identity")
    expected_steps = (parent.epoch + 1) * 352
    if payload.get("global_step") != expected_steps:
        raise InterventionForkError("parent global step is inconsistent with the bound epoch")
    for key in ("model", "optimizer", "selection_metadata"):
        _require_mapping(payload.get(key), name=f"parent checkpoint {key}")
    sample_state = _require_mapping(payload.get("sample_state"), name="parent checkpoint sample_state")
    if _canonical_sha256(sample_state) != parent.sample_state_sha256:
        raise InterventionForkError("parent sample-state SHA-256 does not match the bound arm")
    if sample_state.get("format_version") != SampleStateStore.FORMAT_VERSION:
        raise InterventionForkError("parent checkpoint requires format-v3 sample state")
    store = SampleStateStore(ema_decay=source.method.student_ema_decay)
    try:
        store.load_state_dict(sample_state)
    except (TypeError, ValueError) as exc:
        raise InterventionForkError("parent format-v3 sample state is invalid") from exc
    required_observation_fields = (
        "true_label",
        "teacher_clean_entropy",
        "teacher_clean_true_probability",
        "teacher_clean_max_wrong_probability",
        "teacher_clean_prediction",
        "teacher_clean_correct",
        "teacher_adversarial_entropy",
        "teacher_adversarial_true_probability",
        "teacher_adversarial_max_wrong_probability",
        "teacher_adversarial_prediction",
        "teacher_adversarial_correct",
        "teacher_clean_to_adversarial_margin_response",
        "teacher_clean_to_adversarial_js_response",
    )
    if (
        store.pending
        or len(store.records) != parent.sample_state_records
        or not all(
            record.history_statistics_complete
            and record.seen == parent.epoch + 1
            and all(getattr(record, field) is not None for field in required_observation_fields)
            for record in store.records.values()
        )
    ):
        raise InterventionForkError("parent format-v3 sample state is incomplete for the full train partition")
    train_labels: dict[int, int] = {}
    for sample_id, record in store.records.items():
        if record.true_label is None:  # guarded above; retain an explicit type/runtime fence.
            raise InterventionForkError("parent format-v3 sample state lacks a train label")
        train_labels[sample_id] = record.true_label
    if any(label < 0 or label >= arm.dataset.num_classes for label in train_labels.values()):
        raise InterventionForkError("parent sample-state labels are incompatible with the arm dataset")
    _load_train_partition(parent, expected_labels=train_labels, num_classes=arm.dataset.num_classes)
    return payload, train_labels, dict(inventory_artifact)


def _validate_allowed_delta(*, parent: ExperimentConfig, arm: ExperimentConfig) -> None:
    parent_runtime = resolved_config_dict(parent)
    arm_runtime = resolved_config_dict(arm)
    for key in ("output_dir", "tracking", "intervention"):
        parent_runtime.pop(key, None)
        arm_runtime.pop(key, None)
    if parent_runtime != arm_runtime:
        raise InterventionForkError("arm changes fields outside the registered intervention/tracking/output whitelist")
    if arm.training.epochs != parent.training.epochs:
        causal = arm.intervention is not None and arm.intervention.arm in {"C79", "RA", "RAR", "RB", "RBR"}
        if not causal or arm.training.epochs not in {84, 89, 94}:
            raise InterventionForkError(
                "common-state fork cannot alter continuation epochs outside the registered FFNR horizons"
            )


def _validate_mask(arm: ExperimentConfig, *, train_labels: Mapping[int, int] | None = None) -> None:
    intervention = arm.intervention
    assert intervention is not None
    if intervention.mask is None:
        return
    # The fork is intentionally data-free.  Runtime validates the complete
    # label namespace before training; here verify immutable mask bytes and
    # self-consistent declared class budget without loading CIFAR.
    path = intervention.mask.path
    if not path.is_file() or sha256_file(path) != intervention.mask.sha256:
        raise InterventionForkError("arm mask bytes do not match the configured SHA-256")
    if train_labels is not None:
        try:
            load_fixed_intervention_mask(
                path,
                expected_sha256=intervention.mask.sha256,
                expected_selected_ids_sha256=intervention.mask.selected_ids_sha256,
                expected_selected_count=intervention.mask.selected_count,
                expected_class_counts=intervention.mask.selected_class_counts,
                expected_provenance=intervention.mask.provenance.exact_payload(),
                train_labels=train_labels,
                num_classes=arm.dataset.num_classes,
            )
        except FixedMaskError as exc:
            raise InterventionForkError("arm mask fails strict train-ID validation") from exc


def _validate_selector_bundle(arms: Mapping[str, ExperimentConfig], *, train_labels: Mapping[int, int]) -> None:
    """Recompute both fixed masks before accepting the factorial fork.

    Hashes in an arm config only establish byte identity.  The selector bundle
    additionally proves that those bytes are the deterministic seed-0 fit and
    L3 checkpoint-panel score, which rejects an attacker replacing IDs while
    recomputing every superficial mask/config hash.
    """
    history = _intervention(arms["HS"]).mask
    random = _intervention(arms["RS"]).mask
    assert history is not None and random is not None
    provenance = history.provenance
    assert provenance.selector_spec_path is not None and provenance.approved_selector_spec_sha256 is not None
    if (
        not provenance.selector_spec_path.is_file()
        or sha256_file(provenance.selector_spec_path) != provenance.approved_selector_spec_sha256
    ):
        raise InterventionForkError("history selector bundle bytes do not match the bound SHA-256")
    parent = _intervention(arms["C"]).parent
    try:
        verified = verify_selector_bundle(
            bundle_path=provenance.selector_spec_path,
            history_mask_path=history.path,
            random_mask_path=random.path,
            expected_parent={
                "parent_checkpoint_sha256": parent.checkpoint_sha256,
                "parent_sample_state_sha256": parent.sample_state_sha256,
                "parent_raw_config_sha256": parent.raw_config_sha256,
            },
            expected_train_labels=train_labels,
        )
    except SelectorBundleError as exc:
        raise InterventionForkError("fixed selector bundle does not reproduce") from exc
    if verified["history_mask_sha256"] != history.sha256 or verified["random_mask_sha256"] != random.sha256:
        raise InterventionForkError("selector bundle mask hashes do not match registered arms")
    if history.provenance.approved_selector_spec_sha256 != verified["bundle_sha256"]:
        raise InterventionForkError("history provenance does not bind the verified selector bundle")
    random_provenance = random.provenance
    if (
        random_provenance.reference_history_mask_sha256 != verified["history_mask_sha256"]
        or random_provenance.reference_history_selector_spec_sha256 != verified["bundle_sha256"]
    ):
        raise InterventionForkError("random provenance does not bind the verified history selector")


def _intervention(arm: ExperimentConfig) -> InterventionConfig:
    intervention = arm.intervention
    if intervention is None:
        raise InterventionForkError("each common-state fork arm requires registered intervention metadata")
    return intervention


def _validate_screen(arms: Sequence[ExperimentConfig]) -> dict[str, ExperimentConfig]:
    by_name: dict[str, ExperimentConfig] = {
        str(_intervention(arm).arm): arm for arm in arms if arm.intervention is not None
    }
    causal_names = {"C79", "RA", "RAR", "RB", "RBR"}
    legacy_names = {"C", "HS", "RS", "HD", "RD"}
    if (set(by_name) != legacy_names and set(by_name) != causal_names) or len(by_name) != len(arms):
        raise InterventionForkError("fork requires exactly one registered five-arm screen")
    if set(by_name) == causal_names:
        for selected_name, random_name in (("RA", "RAR"), ("RB", "RBR")):
            selected = _intervention(by_name[selected_name])
            random = _intervention(by_name[random_name])
            assert selected.mask is not None and random.mask is not None
            if (
                selected.mask.selected_count != random.mask.selected_count
                or selected.mask.selected_class_counts != random.mask.selected_class_counts
            ):
                raise InterventionForkError(f"{selected_name} and {random_name} must share the exact selected budget")
        common_parent = _intervention(by_name["C79"]).parent
        if any(_intervention(arm).parent != common_parent for arm in by_name.values()):
            raise InterventionForkError("all FFNR causal arms must bind the exact same parent lineage")
        return by_name
    for first, second in (("HS", "HD"), ("RS", "RD")):
        left, right = _intervention(by_name[first]), _intervention(by_name[second])
        assert left.mask is not None and right.mask is not None
        left_identity = left.mask.model_dump(mode="json", exclude={"path"})
        right_identity = right.mask.model_dump(mode="json", exclude={"path"})
        if left_identity != right_identity:
            raise InterventionForkError(f"{first} and {second} must bind the exact same fixed mask")
    history = _intervention(by_name["HS"]).mask
    random = _intervention(by_name["RS"]).mask
    assert history is not None and random is not None
    if (
        history.selected_count != random.selected_count
        or history.selected_class_counts != random.selected_class_counts
        or random.provenance.reference_history_mask_sha256 != history.sha256
        or random.provenance.reference_history_selector_spec_sha256 != history.provenance.approved_selector_spec_sha256
        or random.provenance.reference_selected_count != history.selected_count
        or random.provenance.reference_selected_class_counts != history.selected_class_counts
    ):
        raise InterventionForkError("history and random arms must share the exact selected count and class budget")
    common_parent = _intervention(by_name["C"]).parent
    for arm in by_name.values():
        if _intervention(arm).parent != common_parent:
            raise InterventionForkError("all intervention arms must bind the exact same parent lineage")
    return by_name


def _fork_selection_metadata(parent: Mapping[str, Any]) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(parent))
    metadata["selected_epoch"] = None
    metadata.pop("selected_clean_accuracy", None)
    metadata.pop("selected_pgd_accuracy", None)
    metadata.pop("last_epoch", None)
    metadata.pop("last_clean_accuracy", None)
    metadata.pop("last_pgd_accuracy", None)
    metadata["scope"] = "post_fork_best"
    return metadata


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_intervention_forks(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    arm_config_paths: Sequence[Path],
    root: Path,
    git_state_collector: Callable[[Path], Mapping[str, Any]] = collect_git_state,
) -> dict[str, Path]:
    """Create all five child epoch-99 checkpoints without invoking training."""
    probe_arms = [load_config(path) for path in arm_config_paths]
    if probe_arms and all(
        arm.intervention is not None and arm.intervention.arm in {"PF_TA", "PF_R", "NR_TA", "NR_R"}
        for arm in probe_arms
    ):
        # The H5-derived epoch-39 route is a separate scientific contract; do
        # not make its looser arm shape an exception in this legacy H3 path.
        from ard.analysis.history_routing_v2 import create_history_routing_v2_forks

        return create_history_routing_v2_forks(
            parent_checkpoint=parent_checkpoint,
            parent_resolved_config=parent_resolved_config,
            parent_manifest=parent_manifest,
            arm_config_paths=arm_config_paths,
            root=root,
            git_state_collector=git_state_collector,
        )
    parent_raw = _load_mapping(parent_resolved_config, name="parent resolved config")
    source = ExperimentConfig.model_validate(parent_raw)
    manifest = _load_json_mapping(parent_manifest, name="parent manifest")
    arms = probe_arms
    by_name = _validate_screen(arms)
    payload, train_labels, parent_artifact = _validate_parent(
        checkpoint=parent_checkpoint,
        parent_raw_config=parent_raw,
        parent_manifest=manifest,
        parent_manifest_path=parent_manifest,
        arm=by_name["C"],
    )
    if set(by_name) == {"C", "HS", "RS", "HD", "RD"}:
        _validate_selector_bundle(by_name, train_labels=train_labels)
    output_dirs = [arm.output_dir.resolve() for arm in by_name.values()]
    if len(set(output_dirs)) != len(output_dirs):
        raise InterventionForkError("intervention arms cannot share an output directory")
    screen_root = output_dirs[0].parent
    if any(
        output.parent != screen_root or output.name != name
        for name, output in ((name, arm.output_dir.resolve()) for name, arm in by_name.items())
    ):
        raise InterventionForkError("arm outputs must be C/HS/RS/HD/RD children of one new screen root")
    if screen_root.exists():
        raise InterventionForkError("refusing to create an intervention screen in an existing root")
    for arm in by_name.values():
        _validate_allowed_delta(parent=source, arm=arm)
        _validate_mask(arm, train_labels=train_labels)
        if arm.output_dir.resolve() == parent_checkpoint.parent.resolve():
            raise InterventionForkError("an arm output directory must be distinct from its parent checkpoint directory")
    fork_git_state = git_state_collector(root)
    fork_git = fork_git_state.get("sha")
    if not isinstance(fork_git, str) or len(fork_git) != 40 or fork_git_state.get("dirty") is not False:
        raise InterventionForkError("fork requires a clean addressable current Git SHA")
    planned: dict[str, tuple[ExperimentConfig, str, str]] = {}
    parent_run_id = payload.get("tracker_run_id")
    child_run_ids: set[str] = set()
    for name, arm in by_name.items():
        child_hash = config_digest(resolved_config_dict(arm))
        child_run_id = stable_run_id(arm, config_hash=child_hash, git_sha=fork_git)
        if child_run_id == parent_run_id or child_run_id in child_run_ids:
            raise InterventionForkError("child arm run IDs must be unique and cannot reuse the parent run ID")
        child_run_ids.add(child_run_id)
        planned[name] = (arm, child_hash, child_run_id)
    causal = set(by_name) == {"C79", "RA", "RAR", "RB", "RBR"}
    screen_kind = "ffnr_causal_intervention_v1" if causal else "common_state_intervention_v1"
    parent_epoch = _intervention(next(iter(by_name.values()))).parent.epoch
    screen_identity = {
        "schema_version": 1,
        "kind": screen_kind,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "parent_run_id": parent_run_id,
        "fork_git_sha": fork_git,
        "arms": [
            {
                "arm": name,
                "config_hash": child_hash,
                "run_id": child_run_id,
                "output": name,
            }
            for name, (_, child_hash, child_run_id) in sorted(planned.items())
        ],
    }
    screen_id = _canonical_sha256(screen_identity)
    staging = Path(tempfile.mkdtemp(prefix=".intervention-screen-", dir=screen_root.parent))
    created: dict[str, Path] = {}
    parent_sha = sha256_file(parent_checkpoint)
    try:
        entries: list[dict[str, object]] = []
        for name, (arm, child_hash, child_run_id) in sorted(planned.items()):
            child = copy.deepcopy(payload)
            child["config_hash"] = child_hash
            child["tracker_run_id"] = child_run_id
            child["best_metric"] = float("-inf")
            child["selection_metadata"] = _fork_selection_metadata(
                _require_mapping(payload["selection_metadata"], name="selection_metadata")
            )
            child["fork_lineage"] = {
                "kind": screen_kind,
                "screen_id": screen_id,
                "arm": name,
                "child_tracker_run_id": child_run_id,
                "parent_checkpoint_sha256": parent_sha,
                "parent_raw_config_sha256": _intervention(arm).parent.raw_config_sha256,
                "parent_git_sha": _intervention(arm).parent.git_sha,
                "parent_epoch": parent_epoch,
                "parent_world_size": 1,
                "parent_teacher_checkpoint_sha256": _intervention(arm).parent.teacher_checkpoint_sha256,
                "parent_sample_state_records": 45000,
                "parent_sample_state_sha256": _intervention(arm).parent.sample_state_sha256,
                "parent_best_metric": payload["best_metric"],
                "parent_selection_metadata": copy.deepcopy(payload["selection_metadata"]),
                "parent_artifact_attestation_sha256": _intervention(arm).parent.artifact_attestation_sha256,
                "parent_wandb_checkpoint_artifact": parent_artifact,
                "fork_git_sha": fork_git,
                "post_fork_best_scope": True,
            }
            output = staging / name
            _atomic_torch_save(child, output / "last.pt")
            save_resolved_config(arm, output / "resolved_config.yaml")
            lineage = dict(child["fork_lineage"])
            lineage["child_config_sha256"] = child_hash
            lineage["child_tracker_run_id"] = child_run_id
            (output / "fork-lineage.json").write_text(
                json.dumps(lineage, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            entries.append(
                {
                    "arm": name,
                    "output": name,
                    "config_hash": child_hash,
                    "run_id": child_run_id,
                    "fork_checkpoint_sha256": sha256_file(output / "last.pt"),
                }
            )
            created[name] = screen_root / name / "last.pt"
        screen_manifest = {**screen_identity, "screen_id": screen_id, "status": "complete", "arms": entries}
        (staging / "screen-complete.json").write_text(
            json.dumps(screen_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        for name in planned:
            shutil.copy2(staging / "screen-complete.json", staging / name / "screen-complete.json")
        os.replace(staging, screen_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return created
