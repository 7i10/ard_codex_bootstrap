"""Fail-closed epoch-39 online-history routing bundles and continuations.

This module is deliberately separate from the retired H3 epoch-99 frozen-
predictor screen.  It consumes only the exact checkpointed SampleStateStore at
the anchor boundary and produces immutable masks before any child training
starts.  The route selector is therefore deployable: it has no replay panel,
future outcome, or fitted predictor input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.schedule_control_fork import (
    ScheduleControlForkError,
    parent_runtime_view,
    validate_train_partition,
)
from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.policies import FixedMaskError, load_fixed_intervention_mask, selected_ids_sha256
from ard.state import SampleStateStore
from ard.tracking import stable_run_id
from ard.tracking.adapter import collect_git_state

KIND = "history_routing_v2_intervention_v1"
ANCHOR_EPOCH = 39
Q = 0.10
ROUTES: dict[str, bool] = {"peak_failure": True, "non_recovery": False}
ARMS = {"PF_TA", "PF_R", "NR_TA", "NR_R"}
PARENT_MILESTONES = (100, 150)
CHILD_MILESTONES = (120, 170)


class HistoryRoutingV2Error(RuntimeError):
    """The v2 selector or continuation cannot prove its fixed lineage."""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HistoryRoutingV2Error(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise HistoryRoutingV2Error(f"{name} must be a mapping")
    return value


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryRoutingV2Error(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise HistoryRoutingV2Error(f"{name} must be a mapping")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoryRoutingV2Error(f"{name} must be a mapping")
    return value


def _parent_identity(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    arm: ExperimentConfig,
    validate_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, int], ExperimentConfig, dict[str, object] | None]:
    intervention = arm.intervention
    if intervention is None or intervention.parent.epoch != ANCHOR_EPOCH:
        raise HistoryRoutingV2Error("v2 arm must bind an epoch-39 intervention parent")
    parent = intervention.parent
    if not parent_checkpoint.is_file() or sha256_file(parent_checkpoint) != parent.checkpoint_sha256:
        raise HistoryRoutingV2Error("parent checkpoint bytes do not match bound lineage")
    raw = _load_yaml(parent_resolved_config, name="parent resolved config")
    raw_hash = config_digest(raw)
    if raw_hash != parent.raw_config_sha256:
        raise HistoryRoutingV2Error("parent raw config SHA-256 does not match bound lineage")
    try:
        source, runtime_migration = parent_runtime_view(raw)
    except ScheduleControlForkError as exc:
        raise HistoryRoutingV2Error("parent resolved config is not a strict runnable configuration") from exc
    if (
        source.intervention is not None
        or source.method.id != "rslad"
        or source.observation.profile != "teacher_response"
        or source.protocol.id != "controlled_cifar10_r18_v1"
        or source.dataset.name != "cifar10"
        or source.training.epochs != 200
        or source.training.per_rank_batch_size != 128
        or source.training.global_batch_size != 128
        or source.scheduler.id != "multistep"
        or tuple(source.scheduler.milestones) != PARENT_MILESTONES
        or source.scheduler.gamma != 0.1
        or source.scheduler.step_at != "epoch_end"
        or source.teacher is None
        or source.teacher.checkpoint_sha256 != parent.teacher_checkpoint_sha256
    ):
        raise HistoryRoutingV2Error("parent is not the controlled ordinary RSLAD epoch-39 protocol")
    manifest = _load_json(parent_manifest, name="parent manifest")
    git, teacher = (
        _mapping(manifest.get("git"), name="parent manifest git"),
        _mapping(manifest.get("teacher"), name="parent manifest teacher"),
    )
    if (
        manifest.get("config_hash") != raw_hash
        or git.get("sha") != parent.git_sha
        or teacher.get("checkpoint_sha256") != parent.teacher_checkpoint_sha256
    ):
        raise HistoryRoutingV2Error("parent manifest config, Git, or teacher lineage does not match")
    if validate_artifacts:
        if (
            not parent.artifact_inventory.is_file()
            or sha256_file(parent.artifact_inventory) != parent.artifact_inventory_sha256
            or not parent.artifact_attestation.is_file()
            or sha256_file(parent.artifact_attestation) != parent.artifact_attestation_sha256
        ):
            raise HistoryRoutingV2Error("parent artifact inventory or attestation bytes do not match bound lineage")
        inventory = _load_json(parent.artifact_inventory, name="parent artifact inventory")
        artifact = _mapping(inventory.get("artifact"), name="parent artifact inventory artifact")
        if artifact.get("checkpoint_sha256") != parent.checkpoint_sha256:
            raise HistoryRoutingV2Error("parent artifact inventory does not bind the selected checkpoint")
        attestation = _load_json(parent.artifact_attestation, name="parent artifact attestation")
        if (
            attestation.get("parent_manifest_path") != str(parent_manifest.resolve())
            or attestation.get("parent_manifest_sha256") != sha256_file(parent_manifest)
            or attestation.get("artifact_inventory_path") != str(parent.artifact_inventory.resolve())
            or attestation.get("artifact_inventory_sha256") != parent.artifact_inventory_sha256
            or attestation.get("checkpoint_sha256") != parent.checkpoint_sha256
            or attestation.get("config_hash") != raw_hash
            or attestation.get("git_sha") != parent.git_sha
            or attestation.get("teacher_checkpoint_sha256") != parent.teacher_checkpoint_sha256
        ):
            raise HistoryRoutingV2Error("parent artifact attestation does not bind immutable source lineage")
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or REQUIRED_KEYS.difference(payload):
        raise HistoryRoutingV2Error("parent checkpoint is incomplete")
    if (
        payload.get("epoch") != ANCHOR_EPOCH
        or payload.get("epoch_boundary") != "end"
        or payload.get("world_size") != parent.world_size
        or payload.get("config_hash") != raw_hash
        or payload.get("tracker_run_id") != manifest.get("run_id")
    ):
        raise HistoryRoutingV2Error("parent checkpoint is not the exact epoch-39 state")
    # 45k / 128 has 352 batches under the source sampler; this is an identity
    # fence, rather than an estimate made from the checkpoint alone.
    if payload.get("global_step") != 14_080:
        raise HistoryRoutingV2Error("parent global step is not the controlled epoch-39 boundary")
    if not isinstance(payload.get("rng"), list) or len(payload["rng"]) != 1:
        raise HistoryRoutingV2Error("parent checkpoint lacks complete single-rank RNG state")
    if not isinstance(payload.get("sampler_state"), list) or len(payload["sampler_state"]) != 1:
        raise HistoryRoutingV2Error("parent checkpoint lacks complete single-rank sampler state")
    expected_sampler = {
        "epoch": ANCHOR_EPOCH,
        "seed": source.seeds.data_order,
        "rank": 0,
        "world_size": 1,
        "shuffle": True,
    }
    if payload.get("sampler_epoch") != [ANCHOR_EPOCH] or payload["sampler_state"][0] != expected_sampler:
        raise HistoryRoutingV2Error("parent sampler state is not the exact epoch-39 single-rank identity")
    state = _mapping(payload.get("sample_state"), name="parent sample state")
    if _sha(state) != parent.sample_state_sha256:
        raise HistoryRoutingV2Error("parent sample state SHA-256 does not match bound lineage")
    store = SampleStateStore(ema_decay=source.method.student_ema_decay)
    try:
        store.load_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise HistoryRoutingV2Error("parent sample state is invalid") from exc
    if store.pending or len(store.records) != parent.sample_state_records:
        raise HistoryRoutingV2Error("parent sample state does not cover the exact train partition")
    labels: dict[int, int] = {}
    for sample_id, record in store.records.items():
        if (
            record.true_label is None
            or not 0 <= record.true_label < arm.dataset.num_classes
            or record.seen != ANCHOR_EPOCH + 1
            or record.robust_correct_count < 0
            or record.robust_correct_count > record.seen
            or not isinstance(record.previous_robust_correct, bool)
            or not math.isfinite(record.margin_ema)
        ):
            raise HistoryRoutingV2Error("parent sample state lacks complete anchor-inclusive online history")
        labels[sample_id] = record.true_label
    if len(labels) != parent.sample_state_records:
        raise HistoryRoutingV2Error("parent state stable-ID namespace is incomplete")
    try:
        validate_train_partition(parent.model_dump(mode="json"), labels=labels, num_classes=arm.dataset.num_classes)
    except ScheduleControlForkError as exc:
        raise HistoryRoutingV2Error(
            "parent train partition does not bind the exact sample-state IDs and labels"
        ) from exc
    return payload, labels, source, runtime_migration


def _midrank(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        raise HistoryRoutingV2Error("cannot rank an empty route population")
    ordered = sorted((float(value), int(sample_id)) for sample_id, value in values.items())
    output: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        value = (start + (end - start) / 2.0) / len(ordered)
        for _, sample_id in ordered[start:end]:
            output[sample_id] = value
        start = end
    return output


def _route_selection(state: Mapping[str, Any], *, route: str) -> tuple[list[int], dict[int, int], dict[str, Any]]:
    expected_correct = ROUTES[route]
    records = _mapping(state.get("records"), name="sample-state records")
    all_records: dict[int, Mapping[str, Any]] = {}
    for raw_id, raw_record in records.items():
        if not isinstance(raw_id, str) or not raw_id.isdigit():
            raise HistoryRoutingV2Error("sample-state stable IDs must be canonical integers")
        record = _mapping(raw_record, name="sample-state record")
        seen = record.get("seen")
        hits = record.get("robust_correct_count")
        margin = record.get("margin_ema")
        if (
            isinstance(seen, bool)
            or not isinstance(seen, int)
            or seen < 1
            or isinstance(hits, bool)
            or not isinstance(hits, int)
            or not 0 <= hits <= seen
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise HistoryRoutingV2Error("sample-state history score inputs violate the epoch-39 contract")
        all_records[int(raw_id)] = record
    # H5-Early ranks each input feature over the complete 45k training state
    # before it stratifies on anchor correctness.  Route-local midranks would
    # silently change the selector when a route's composition changes.
    frequency = _midrank(
        {
            sample_id: 1.0 - int(record["robust_correct_count"]) / int(record["seen"])
            for sample_id, record in all_records.items()
        }
    )
    margin = _midrank({sample_id: -float(record["margin_ema"]) for sample_id, record in all_records.items()})
    scores = {sample_id: 0.5 * (frequency[sample_id] + margin[sample_id]) for sample_id in all_records}
    eligible: dict[int, Mapping[str, Any]] = {}
    for sample_id, record in all_records.items():
        if record.get("previous_robust_correct") is expected_correct:
            eligible[sample_id] = record
    count = math.floor(Q * len(eligible))
    if count < 1:
        raise HistoryRoutingV2Error("route population is too small for the fixed 10% budget")
    selected = [
        sample_id
        for sample_id, _ in sorted(
            ((sample_id, scores[sample_id]) for sample_id in eligible), key=lambda item: (-item[1], item[0])
        )[:count]
    ]
    labels = {sample_id: int(eligible[sample_id]["true_label"]) for sample_id in eligible}
    metadata = {
        "route": route,
        "anchor_epoch": ANCHOR_EPOCH,
        "anchor_robust_correct": expected_correct,
        "eligible_count": len(eligible),
        "selected_count": count,
        "q": Q,
        "score": "0.5*midrank_all_train(1-robust_correct_frequency_inclusive)+0.5*midrank_all_train(-margin_ema)",
        "rank_population": "all_train_sample_state_records_before_anchor_correctness_route",
        "tie_break": "stable_sample_id_ascending",
    }
    return sorted(selected), labels, metadata


def _class_counts(ids: Sequence[int], labels: Mapping[int, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample_id in ids:
        label = str(labels[sample_id])
        counts[label] = counts.get(label, 0) + 1
    return {key: counts[key] for key in sorted(counts, key=int)}


def _random_ids(
    *, eligible_labels: Mapping[int, int], selected_counts: Mapping[str, int], parent_sha: str, route: str
) -> tuple[list[int], int]:
    seed = int(hashlib.sha256(f"{parent_sha}:{route}:v2".encode()).hexdigest()[:16], 16)
    chosen: list[int] = []
    for class_id, required in sorted(selected_counts.items(), key=lambda item: int(item[0])):
        candidates = [sample_id for sample_id, label in eligible_labels.items() if label == int(class_id)]
        ranked = sorted(
            candidates,
            key=lambda sample_id: hashlib.sha256(f"{seed}:{route}:{class_id}:{sample_id}".encode()).hexdigest(),
        )
        if len(ranked) < required:
            raise HistoryRoutingV2Error("class/state random control cannot match the selected history budget")
        chosen.extend(ranked[:required])
    return sorted(chosen), seed


def _mask_payload(
    *, selected_ids: Sequence[int], labels: Mapping[int, int], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    ids = tuple(sorted(int(sample_id) for sample_id in selected_ids))
    return {
        "schema_version": 1,
        "namespace": "train",
        "num_classes": 10,
        "selected_ids": list(ids),
        "selected_ids_sha256": selected_ids_sha256(ids),
        "selected_count": len(ids),
        "selected_class_counts": _class_counts(ids, labels),
        "provenance": dict(provenance),
    }


def build_history_routing_v2_bundle(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    train_partition_manifest: Path,
    train_partition_manifest_sha256: str,
    train_partition_ids_labels_sha256: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Materialize two route masks and their deterministic matched-random controls."""
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a history-routing v2 selector bundle")
    # A light local arm validates the exact parent schema without deciding any
    # route-specific child config.
    raw = _load_yaml(parent_resolved_config, name="parent resolved config")
    probe = copy.deepcopy(raw)
    probe["intervention"] = None
    try:
        source, _ = parent_runtime_view(probe)
    except ScheduleControlForkError as exc:
        raise HistoryRoutingV2Error("parent resolved config is not a strict runnable configuration") from exc
    if source.teacher is None:
        raise HistoryRoutingV2Error("parent requires a frozen teacher")
    # Build a temporary config-shaped arm solely to reuse the strict parent
    # validator.  It never reaches a Trainer.
    arm_raw = resolved_config_dict(source)
    parent_sha = sha256_file(parent_checkpoint)
    state = torch.load(parent_checkpoint, map_location="cpu", weights_only=False).get("sample_state")
    if not isinstance(state, Mapping):
        raise HistoryRoutingV2Error("parent checkpoint lacks sample state")
    parent = {
        "checkpoint_sha256": parent_sha,
        "raw_config_sha256": config_digest(raw),
        "git_sha": _mapping(_load_json(parent_manifest, name="parent manifest").get("git"), name="parent git").get(
            "sha"
        ),
        "epoch": ANCHOR_EPOCH,
        "world_size": 1,
        "teacher_checkpoint_sha256": source.teacher.checkpoint_sha256,
        "sample_state_records": 45000,
        "sample_state_sha256": _sha(state),
        "train_partition_manifest": str(train_partition_manifest),
        "train_partition_manifest_sha256": train_partition_manifest_sha256,
        "train_partition_ids_labels_sha256": train_partition_ids_labels_sha256,
        "artifact_attestation": "unused-by-selector.json",
        "artifact_attestation_sha256": "0" * 64,
        "artifact_inventory": "unused-by-selector.json",
        "artifact_inventory_sha256": "0" * 64,
    }
    # The selector validates checkpoint/config/manifest/state directly below;
    # artifact paths are bound later by the fork config and are not needed to
    # rank an already-local checkpoint.
    arm_raw["intervention"] = {
        "arm": "PF_TA",
        "selector": "online_history",
        "kind": "teacher_target_true_label_mix",
        "parent": parent,
        "selector_bundle_path": "placeholder.json",
        "selector_bundle_sha256": "0" * 64,
        "mask": {
            "path": "placeholder.json",
            "sha256": "0" * 64,
            "selected_ids_sha256": "0" * 64,
            "selected_count": 1,
            "selected_class_counts": {"0": 1},
            "provenance": {
                "source": "online_history_epoch39_v2",
                "approved_selector_spec_sha256": "0" * 64,
                "selector_spec_path": "placeholder.json",
                "parent_checkpoint_sha256": parent_sha,
                "parent_sample_state_sha256": _sha(state),
                "route": "peak_failure",
                "anchor_robust_correct": True,
            },
        },
    }
    arm = ExperimentConfig.model_validate(arm_raw)
    payload, labels, _, _ = _parent_identity(
        parent_checkpoint=parent_checkpoint,
        parent_resolved_config=parent_resolved_config,
        parent_manifest=parent_manifest,
        arm=arm,
    )
    output_dir.mkdir(parents=True)
    paths = {route: output_dir / f"{route}-history-mask.json" for route in ROUTES}
    random_paths = {route: output_dir / f"{route}-random-mask.json" for route in ROUTES}
    bundle_path = output_dir / "history-routing-v2-bundle.json"
    selection: dict[str, dict[str, Any]] = {}
    route_rows: dict[str, tuple[list[int], dict[int, int], dict[str, Any], list[int], int]] = {}
    for route in ROUTES:
        selected, eligible_labels, metadata = _route_selection(payload["sample_state"], route=route)
        random_ids, seed = _random_ids(
            eligible_labels=eligible_labels,
            selected_counts=_class_counts(selected, eligible_labels),
            parent_sha=parent_sha,
            route=route,
        )
        selection[route] = {**metadata, "selected_class_counts": _class_counts(selected, eligible_labels)}
        route_rows[route] = (selected, eligible_labels, metadata, random_ids, seed)
    bundle = {
        "schema_version": 1,
        "kind": "history_routing_v2_online_selector_v1",
        "parent": {
            "checkpoint_path": str(parent_checkpoint.resolve()),
            "checkpoint_sha256": parent_sha,
            "raw_config_path": str(parent_resolved_config.resolve()),
            "raw_config_sha256": config_digest(raw),
            "manifest_path": str(parent_manifest.resolve()),
            "manifest_sha256": sha256_file(parent_manifest),
            "sample_state_sha256": _sha(payload["sample_state"]),
            "train_partition_manifest_path": str(train_partition_manifest.resolve()),
            "train_partition_manifest_sha256": train_partition_manifest_sha256,
            "train_partition_ids_labels_sha256": train_partition_ids_labels_sha256,
            "epoch": ANCHOR_EPOCH,
            "run_id": payload["tracker_run_id"],
        },
        "selection": selection,
        "mask_paths": {
            route: {"history": str(paths[route].resolve()), "random": str(random_paths[route].resolve())}
            for route in ROUTES
        },
    }
    bundle_path.write_bytes(canonical_json(bundle))
    bundle_sha = sha256_file(bundle_path)
    for route, (selected, eligible_labels, metadata, random_ids, seed) in route_rows.items():
        history = _mask_payload(
            selected_ids=selected,
            labels=eligible_labels,
            provenance={
                "source": "online_history_epoch39_v2",
                "approved_selector_spec_sha256": bundle_sha,
                "selector_spec_path": str(bundle_path.resolve()),
                "parent_checkpoint_sha256": parent_sha,
                "parent_sample_state_sha256": _sha(payload["sample_state"]),
                "route": route,
                "anchor_robust_correct": ROUTES[route],
            },
        )
        paths[route].write_bytes(canonical_json(history))
        random = _mask_payload(
            selected_ids=random_ids,
            labels=eligible_labels,
            provenance={
                "source": "class_state_count_matched_random_epoch39_v2",
                "parent_checkpoint_sha256": parent_sha,
                "parent_sample_state_sha256": _sha(payload["sample_state"]),
                "random_seed": seed,
                "generator": "sha256_rank",
                "generator_version": "parent_route_class_sample_id_v1",
                "reference_history_mask_sha256": sha256_file(paths[route]),
                "reference_selected_count": len(selected),
                "reference_selected_class_counts": _class_counts(selected, eligible_labels),
                "reference_history_selector_spec_sha256": bundle_sha,
                "route": route,
                "anchor_robust_correct": ROUTES[route],
            },
        )
        random_paths[route].write_bytes(canonical_json(random))
    return {
        "bundle": bundle_path,
        **{f"{route}_history": paths[route] for route in ROUTES},
        **{f"{route}_random": random_paths[route] for route in ROUTES},
    }


def verify_history_routing_v2_bundle(*, bundle_path: Path) -> dict[str, Any]:
    bundle = _load_json(bundle_path, name="history-routing v2 bundle")
    if bundle.get("schema_version") != 1 or bundle.get("kind") != "history_routing_v2_online_selector_v1":
        raise HistoryRoutingV2Error("history-routing v2 selector bundle identity drifted")
    parent = _mapping(bundle.get("parent"), name="selector parent")
    required = {
        "checkpoint_path",
        "checkpoint_sha256",
        "raw_config_path",
        "raw_config_sha256",
        "manifest_path",
        "manifest_sha256",
        "sample_state_sha256",
        "train_partition_manifest_path",
        "train_partition_manifest_sha256",
        "train_partition_ids_labels_sha256",
        "epoch",
        "run_id",
    }
    if set(parent) != required or parent.get("epoch") != ANCHOR_EPOCH:
        raise HistoryRoutingV2Error("history-routing v2 parent bundle is incomplete")
    checkpoint, raw_config, manifest = (
        Path(str(parent["checkpoint_path"])),
        Path(str(parent["raw_config_path"])),
        Path(str(parent["manifest_path"])),
    )
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != parent["checkpoint_sha256"]
        or not raw_config.is_file()
        or config_digest(_load_yaml(raw_config, name="parent resolved config")) != parent["raw_config_sha256"]
        or not manifest.is_file()
        or sha256_file(manifest) != parent["manifest_sha256"]
    ):
        raise HistoryRoutingV2Error("history-routing v2 selector source bytes drifted")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _mapping(payload.get("sample_state"), name="parent sample state")
    if _sha(state) != parent["sample_state_sha256"] or payload.get("tracker_run_id") != parent["run_id"]:
        raise HistoryRoutingV2Error("history-routing v2 selector state identity drifted")
    store = SampleStateStore(ema_decay=0.9)
    try:
        store.load_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise HistoryRoutingV2Error("history-routing v2 selector sample state is invalid") from exc
    labels = {
        sample_id: record.true_label for sample_id, record in store.records.items() if record.true_label is not None
    }
    if len(labels) != len(store.records):
        raise HistoryRoutingV2Error("history-routing v2 selector lacks exact train labels")
    try:
        validate_train_partition(
            {
                "train_partition_manifest": parent["train_partition_manifest_path"],
                "train_partition_manifest_sha256": parent["train_partition_manifest_sha256"],
                "train_partition_ids_labels_sha256": parent["train_partition_ids_labels_sha256"],
            },
            labels=labels,
            num_classes=10,
        )
    except ScheduleControlForkError as exc:
        raise HistoryRoutingV2Error("history-routing v2 selector train partition identity drifted") from exc
    paths = _mapping(bundle.get("mask_paths"), name="selector mask paths")
    bundle_sha = sha256_file(bundle_path)
    verified: dict[str, Any] = {"bundle_sha256": bundle_sha, "routes": {}}
    for route in ROUTES:
        pair = _mapping(paths.get(route), name=f"{route} mask paths")
        history_path, random_path = Path(str(pair.get("history"))), Path(str(pair.get("random")))
        selected, labels, metadata = _route_selection(state, route=route)
        expected_history = _mask_payload(
            selected_ids=selected,
            labels=labels,
            provenance={
                "source": "online_history_epoch39_v2",
                "approved_selector_spec_sha256": bundle_sha,
                "selector_spec_path": str(bundle_path.resolve()),
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "parent_sample_state_sha256": parent["sample_state_sha256"],
                "route": route,
                "anchor_robust_correct": ROUTES[route],
            },
        )
        actual_history = _load_json(history_path, name=f"{route} history mask")
        if actual_history != expected_history:
            raise HistoryRoutingV2Error("history-routing v2 history mask does not reproduce exact online selection")
        random_ids, seed = _random_ids(
            eligible_labels=labels,
            selected_counts=_class_counts(selected, labels),
            parent_sha=str(parent["checkpoint_sha256"]),
            route=route,
        )
        expected_random = _mask_payload(
            selected_ids=random_ids,
            labels=labels,
            provenance={
                "source": "class_state_count_matched_random_epoch39_v2",
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "parent_sample_state_sha256": parent["sample_state_sha256"],
                "random_seed": seed,
                "generator": "sha256_rank",
                "generator_version": "parent_route_class_sample_id_v1",
                "reference_history_mask_sha256": sha256_file(history_path),
                "reference_selected_count": len(selected),
                "reference_selected_class_counts": _class_counts(selected, labels),
                "reference_history_selector_spec_sha256": bundle_sha,
                "route": route,
                "anchor_robust_correct": ROUTES[route],
            },
        )
        if _load_json(random_path, name=f"{route} random mask") != expected_random:
            raise HistoryRoutingV2Error(
                "history-routing v2 random mask does not reproduce exact class/state/count control"
            )
        if bundle.get("selection", {}).get(route) != {
            **metadata,
            "selected_class_counts": _class_counts(selected, labels),
        }:
            raise HistoryRoutingV2Error("history-routing v2 selector metadata does not reproduce")
        verified["routes"][route] = {
            "history_mask_sha256": sha256_file(history_path),
            "random_mask_sha256": sha256_file(random_path),
            "selected_count": len(selected),
            "selected_class_counts": _class_counts(selected, labels),
        }
    return verified


def _intervention(arm: ExperimentConfig):
    if arm.intervention is None:
        raise HistoryRoutingV2Error("each v2 child requires intervention metadata")
    return arm.intervention


def _validate_arms(arms: Sequence[ExperimentConfig]) -> dict[str, ExperimentConfig]:
    by_arm = {str(_intervention(arm).arm): arm for arm in arms}
    if set(by_arm) != ARMS or len(by_arm) != len(arms):
        raise HistoryRoutingV2Error("v2 fork requires exactly PF_TA/PF_R/NR_TA/NR_R once")
    parent = _intervention(by_arm["PF_TA"]).parent
    for arm in by_arm.values():
        if _intervention(arm).parent != parent:
            raise HistoryRoutingV2Error("all v2 arms must bind the exact same epoch-39 parent lineage")
        if _intervention(arm).kind != "teacher_target_true_label_mix":
            raise HistoryRoutingV2Error("v2 arms must use only the registered true-label target mix")
        if arm.protocol.id != "controlled_cifar10_r18_delayed_multistep_v1":
            raise HistoryRoutingV2Error("v2 arms require the registered delayed-schedule protocol identity")
        if (
            tuple(arm.scheduler.milestones) != CHILD_MILESTONES
            or arm.scheduler.gamma != 0.1
            or arm.scheduler.step_at != "epoch_end"
        ):
            raise HistoryRoutingV2Error("v2 arms require the fixed delayed schedule [120,170]")
        if arm.intervention is None or arm.intervention.selector_bundle_path is None:
            raise HistoryRoutingV2Error("v2 arms require a selector bundle path")
    for first, second in (("PF_TA", "PF_R"), ("NR_TA", "NR_R")):
        left, right = _intervention(by_arm[first]).mask, _intervention(by_arm[second]).mask
        assert left is not None and right is not None
        if left.selected_count != right.selected_count or left.selected_class_counts != right.selected_class_counts:
            raise HistoryRoutingV2Error("v2 history/random arms must share exact class/state/count budgets")
        if right.provenance.reference_history_mask_sha256 != left.sha256:
            raise HistoryRoutingV2Error("v2 random arm does not bind its route history mask")
    return by_arm


def _validate_child_delta(*, source: ExperimentConfig, arm: ExperimentConfig) -> None:
    parent_runtime, arm_runtime = resolved_config_dict(source), resolved_config_dict(arm)
    for key in ("output_dir", "tracking", "intervention", "scheduler", "protocol"):
        parent_runtime.pop(key, None)
        arm_runtime.pop(key, None)
    if parent_runtime != arm_runtime:
        raise HistoryRoutingV2Error("v2 child changes fields outside scheduler/intervention/tracking/output")


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_history_routing_v2_forks(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    arm_config_paths: Sequence[Path],
    root: Path,
    git_state_collector: Callable[[Path], Mapping[str, Any]] = collect_git_state,
) -> dict[str, Path]:
    arms = [load_config(path) for path in arm_config_paths]
    by_arm = _validate_arms(arms)
    payload, labels, source, runtime_migration = _parent_identity(
        parent_checkpoint=parent_checkpoint,
        parent_resolved_config=parent_resolved_config,
        parent_manifest=parent_manifest,
        arm=by_arm["PF_TA"],
        validate_artifacts=True,
    )
    for arm in by_arm.values():
        _validate_child_delta(source=source, arm=arm)
        mask = _intervention(arm).mask
        assert mask is not None
        try:
            load_fixed_intervention_mask(
                mask.path,
                expected_sha256=mask.sha256,
                expected_selected_ids_sha256=mask.selected_ids_sha256,
                expected_selected_count=mask.selected_count,
                expected_class_counts=mask.selected_class_counts,
                expected_provenance=mask.provenance.exact_payload(),
                train_labels=labels,
                num_classes=arm.dataset.num_classes,
            )
        except FixedMaskError as exc:
            raise HistoryRoutingV2Error("v2 child mask fails strict train-ID validation") from exc
    selector_path = _intervention(by_arm["PF_TA"]).selector_bundle_path
    assert selector_path is not None
    verified = verify_history_routing_v2_bundle(bundle_path=selector_path)
    parent = _intervention(by_arm["PF_TA"]).parent
    for arm in by_arm.values():
        intervention = _intervention(arm)
        if (
            intervention.selector_bundle_path is None
            or intervention.selector_bundle_path.resolve() != selector_path.resolve()
            or intervention.selector_bundle_sha256 != verified["bundle_sha256"]
        ):
            raise HistoryRoutingV2Error("v2 arms do not bind one immutable selector bundle")
    if verified["bundle_sha256"] != _intervention(by_arm["PF_TA"]).mask.provenance.approved_selector_spec_sha256:
        raise HistoryRoutingV2Error("v2 selector bundle SHA does not bind history mask provenance")
    for arm_name, route, kind in (
        ("PF_TA", "peak_failure", "history"),
        ("PF_R", "peak_failure", "random"),
        ("NR_TA", "non_recovery", "history"),
        ("NR_R", "non_recovery", "random"),
    ):
        mask = _intervention(by_arm[arm_name]).mask
        assert mask is not None
        if mask.sha256 != verified["routes"][route][f"{kind}_mask_sha256"]:
            raise HistoryRoutingV2Error("v2 arm mask does not match its verified selector bundle output")
    state = _mapping(payload.get("scheduler"), name="parent scheduler")
    if (
        state.get("milestones") != Counter({100: 1, 150: 1})
        or state.get("last_epoch") != ANCHOR_EPOCH + 1
        or state.get("_last_lr") != [0.1]
    ):
        raise HistoryRoutingV2Error("parent scheduler is not the pre-decay epoch-39 [100,150] state")
    git = git_state_collector(root)
    if git.get("dirty") is not False or not isinstance(git.get("sha"), str) or len(str(git["sha"])) != 40:
        raise HistoryRoutingV2Error("v2 fork requires a clean addressable Git SHA")
    root_dirs = [arm.output_dir.resolve() for arm in by_arm.values()]
    screen_root = root_dirs[0].parent
    if (
        screen_root.exists()
        or len(set(root_dirs)) != len(root_dirs)
        or any(path.parent != screen_root for path in root_dirs)
    ):
        raise HistoryRoutingV2Error("v2 children must have four distinct paths in one new screen root")
    plans: dict[str, tuple[ExperimentConfig, str, str]] = {}
    for name, arm in by_arm.items():
        digest = config_digest(resolved_config_dict(arm))
        run_id = stable_run_id(arm, config_hash=digest, git_sha=str(git["sha"]))
        plans[name] = (arm, digest, run_id)
    if len({run_id for _, _, run_id in plans.values()}) != len(plans):
        raise HistoryRoutingV2Error("v2 child tracking identities are not unique")
    identity = {
        "schema_version": 1,
        "kind": KIND,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "parent_run_id": payload.get("tracker_run_id"),
        "fork_git_sha": git["sha"],
        "arms": [
            {"arm": name, "config_hash": digest, "run_id": run_id, "output": name}
            for name, (_, digest, run_id) in sorted(plans.items())
        ],
    }
    screen_id = _sha(identity)
    staging = Path(tempfile.mkdtemp(prefix=".history-routing-v2-", dir=screen_root.parent))
    created: dict[str, Path] = {}
    try:
        entries = []
        for name, (arm, digest, run_id) in sorted(plans.items()):
            child = copy.deepcopy(payload)
            child["config_hash"], child["tracker_run_id"], child["best_metric"] = digest, run_id, float("-inf")
            child["scheduler"] = copy.deepcopy(dict(state))
            child["scheduler"]["milestones"] = Counter({120: 1, 170: 1})
            metadata = copy.deepcopy(dict(_mapping(payload["selection_metadata"], name="selection metadata")))
            metadata["selected_epoch"] = None
            for key in (
                "selected_clean_accuracy",
                "selected_pgd_accuracy",
                "last_epoch",
                "last_clean_accuracy",
                "last_pgd_accuracy",
            ):
                metadata.pop(key, None)
            metadata["scope"] = "post_fork_best"
            child["selection_metadata"] = metadata
            child["fork_lineage"] = {
                "kind": KIND,
                "screen_id": screen_id,
                "arm": name,
                "child_tracker_run_id": run_id,
                "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
                "parent_raw_config_sha256": parent.raw_config_sha256,
                "parent_runtime_migration": runtime_migration,
                "parent_git_sha": parent.git_sha,
                "parent_epoch": ANCHOR_EPOCH,
                "parent_world_size": 1,
                "parent_teacher_checkpoint_sha256": parent.teacher_checkpoint_sha256,
                "parent_sample_state_records": parent.sample_state_records,
                "parent_sample_state_sha256": parent.sample_state_sha256,
                "parent_best_metric": payload["best_metric"],
                "parent_selection_metadata": copy.deepcopy(payload["selection_metadata"]),
                "fork_git_sha": git["sha"],
                "selector_bundle_sha256": verified["bundle_sha256"],
                "post_fork_best_scope": True,
            }
            out = staging / name
            _atomic_save(child, out / "last.pt")
            save_resolved_config(arm, out / "resolved_config.yaml")
            lineage = {**child["fork_lineage"], "child_config_sha256": digest, "child_tracker_run_id": run_id}
            (out / "fork-lineage.json").write_text(
                json.dumps(lineage, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            entries.append(
                {
                    "arm": name,
                    "output": name,
                    "config_hash": digest,
                    "run_id": run_id,
                    "fork_checkpoint_sha256": sha256_file(out / "last.pt"),
                }
            )
            created[name] = screen_root / name / "last.pt"
        manifest = {**identity, "screen_id": screen_id, "status": "complete", "arms": entries}
        (staging / "screen-complete.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        for name in plans:
            shutil.copy2(staging / "screen-complete.json", staging / name / "screen-complete.json")
        os.replace(staging, screen_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return created
