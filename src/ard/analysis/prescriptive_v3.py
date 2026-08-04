"""Fail-closed epoch-34-online / epoch-79-state masks for prescriptive v3."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.rslad_signal_replay import canonical_json
from ard.analysis.schedule_control_fork import (
    CHILD_MILESTONES,
    PARENT_MILESTONES,
    _atomic_torch_save,
    _post_fork_selection_metadata,
    _replace_scheduler_milestones,
    _spec_parent,
    _validate_parent,
)
from ard.analysis.schedule_control_fork import (
    sha256_file as schedule_sha256_file,
)
from ard.analysis.signal_audit import sha256_file
from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.policies import load_fixed_intervention_mask, selected_ids_sha256
from ard.tracking import stable_run_id
from ard.tracking.adapter import collect_git_state

ANCHOR34, PARENT79, Q = 34, 79, 0.10


class PrescriptiveV3Error(ValueError):
    pass


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except Exception as exc:  # pragma: no cover
        raise PrescriptiveV3Error("v3 replay input parquet is unreadable") from exc


def _midrank(values: Mapping[int, float]) -> dict[int, float]:
    ordered = sorted((float(value), int(sample_id)) for sample_id, value in values.items())
    if not ordered:
        raise PrescriptiveV3Error("cannot rank an empty online population")
    result: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank = (start + (end - start) / 2) / len(ordered)
        for _, sample_id in ordered[start:end]:
            result[sample_id] = rank
        start = end
    return result


def _state(parent_checkpoint: Path) -> tuple[dict[int, tuple[int, bool]], str, str, Mapping[str, Any]]:
    if not parent_checkpoint.is_file():
        raise PrescriptiveV3Error("epoch-79 parent checkpoint is unavailable")
    checkpoint_sha = sha256_file(parent_checkpoint)
    try:
        payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover
        raise PrescriptiveV3Error("epoch-79 parent checkpoint is unreadable") from exc
    state = payload.get("sample_state") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or REQUIRED_KEYS.difference(payload)
        or payload.get("epoch") != PARENT79
        or payload.get("epoch_boundary") != "end"
        or not isinstance(state, Mapping)
        or state.get("format_version") != 3
        or state.get("pending") != []
        or not isinstance(state.get("records"), Mapping)
    ):
        raise PrescriptiveV3Error("parent is not an epoch-79 settled format-v3 checkpoint")
    records: dict[int, tuple[int, bool]] = {}
    for raw_id, record in state["records"].items():
        if not isinstance(raw_id, str) or not raw_id.isdigit() or not isinstance(record, Mapping):
            raise PrescriptiveV3Error("parent state stable-ID record drifted")
        sample_id, label, previous = int(raw_id), record.get("true_label"), record.get("previous_robust_correct")
        if sample_id in records or not isinstance(label, int) or not 0 <= label < 10 or not isinstance(previous, bool):
            raise PrescriptiveV3Error("parent state label/correctness drifted")
        records[sample_id] = (label, previous)
    return records, checkpoint_sha, _hash(state), payload


def build_prescriptive_v3_masks(
    *,
    online_observations: Path,
    online_lineage: Path,
    feature_observations: Path,
    feature_lineage: Path,
    parent_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Freeze H/R masks; no outcome observation is accepted by this API."""
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite prescriptive-v3 route bundle")
    try:
        online_meta = json.loads(online_lineage.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrescriptiveV3Error("epoch-34 online lineage is unreadable") from exc
    if (
        not isinstance(online_meta, Mapping)
        or online_meta.get("contract") != "pre39_online_state_candidate_v1"
        or online_meta.get("anchor_epoch") != ANCHOR34
        or online_meta.get("observations_sha256") != sha256_file(online_observations)
    ):
        raise PrescriptiveV3Error("epoch-34 online lineage/hash contract drifted")
    parent, parent_sha, state_sha, parent_payload = _state(parent_checkpoint)
    if online_meta.get("run_id") is None or online_meta.get("config_hash") is None:
        raise PrescriptiveV3Error("epoch-34 online lineage lacks run/config identity")
    online_rows = _rows(online_observations)
    online = {int(row["sample_id"]): row for row in online_rows}
    if len(online) != len(online_rows) or set(online) != set(parent) or any(
        row.get("anchor_epoch") != ANCHOR34
        or row.get("class_id") != parent[item][0]
        or not isinstance(row.get("current_robust_correct"), bool)
        for item, row in online.items()
    ):
        raise PrescriptiveV3Error("epoch-34 online/epoch-79 parent sparse-ID/class join drifted")
    try:
        feature_meta = json.loads(feature_lineage.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrescriptiveV3Error("epoch-34 feature lineage is unreadable") from exc
    if (
        not isinstance(feature_meta, Mapping)
        or feature_meta.get("observation_schema_version") != 2
        or feature_meta.get("feature_observations_sha256") != sha256_file(feature_observations)
        or online_meta.get("feature_observations_sha256") != sha256_file(feature_observations)
        or any(
            feature_meta.get(key) != online_meta.get(key)
            for key in ("run_id", "config_hash", "teacher", "dataset_identity", "attack_identity")
        )
        or parent_payload.get("tracker_run_id") != online_meta.get("run_id")
        or parent_payload.get("config_hash") != online_meta.get("config_hash")
    ):
        raise PrescriptiveV3Error("epoch-34 feature/online/epoch-79 parent lineage identity drifted")
    feature_rows = [
        row for row in _rows(feature_observations) if row.get("epoch") == ANCHOR34 and row.get("namespace") == "train"
    ]
    feature = {
        int(row["sample_id"]): row
        for row in feature_rows
    }
    if (
        len(feature) != len(feature_rows)
        or set(feature) != set(parent)
        or any(feature[item].get("class_id") != parent[item][0] for item in parent)
    ):
        raise PrescriptiveV3Error("epoch-34 feature sparse-ID/class join drifted")
    if any(
        not isinstance(row.get("robust_correct_frequency_inclusive"), (float, int))
        or isinstance(row.get("robust_correct_frequency_inclusive"), bool)
        or not math.isfinite(float(row["robust_correct_frequency_inclusive"]))
        or not isinstance(row.get("margin_ema"), (float, int))
        or isinstance(row.get("margin_ema"), bool)
        or not math.isfinite(float(row["margin_ema"]))
        or not isinstance(feature[item].get("teacher_adversarial_correct"), bool)
        for item, row in online.items()
    ):
        raise PrescriptiveV3Error("online score or feature teacher-correctness values drifted")
    frequency = _midrank({item: 1 - float(row["robust_correct_frequency_inclusive"]) for item, row in online.items()})
    margin = _midrank({item: -float(row["margin_ema"]) for item, row in online.items()})
    score = {item: (frequency[item] + margin[item]) / 2 for item in online}
    routes = {"PF_RET": True, "NR_PFX": False}
    histories: dict[str, list[int]] = {}
    for route, current in routes.items():
        eligible34 = [item for item, row in online.items() if row["current_robust_correct"] is current]
        ranked = sorted(
            eligible34,
            key=lambda item: (-score[item], item),
        )
        top = set(ranked[: max(1, int(len(ranked) * Q))])
        histories[route] = sorted(item for item in top if parent[item][1] is current)
        if not histories[route]:
            raise PrescriptiveV3Error("route intersection is empty")
    output_dir.mkdir(parents=True)
    bundle = {
        "schema_version": 1,
        "kind": "prescriptive_v3_epoch34_online_epoch79_route_v1",
        "parent": {"checkpoint_sha256": parent_sha, "sample_state_sha256": state_sha, "epoch": PARENT79},
        "online": {
            "observations_sha256": sha256_file(online_observations),
            "lineage_sha256": sha256_file(online_lineage),
            "anchor_epoch": ANCHOR34,
        },
        "feature_observations_sha256": sha256_file(feature_observations),
        "feature_lineage_sha256": sha256_file(feature_lineage),
        "score_contract": (
            "equal_midrank_all_of_one_minus_inclusive_robust_correct_frequency_and_negative_margin_ema_v1"
        ),
        "routes": {},
    }
    bundle_path = output_dir / "prescriptive-v3-bundle.json"
    for route, current in routes.items():
        history = histories[route]
        strata: dict[tuple[int, bool, bool], list[int]] = defaultdict(list)
        for item, (label, state) in parent.items():
            strata[(label, state, bool(feature[item]["teacher_adversarial_correct"]))].append(item)
        selected: set[int] = set(history)
        for item in history:
            key = (parent[item][0], parent[item][1], feature[item]["teacher_adversarial_correct"])
            candidates = [candidate for candidate in strata[key] if candidate not in selected]
            if not candidates:
                raise PrescriptiveV3Error("matched random stratum lacks an unselected candidate")
            chosen = min(
                candidates,
                key=lambda candidate: _hash(["prescriptive-v3", parent_sha, route, "random-control", candidate]),
            )
            selected.add(chosen)
        random = sorted(selected - set(history))
        route_payload = {"history": history, "random": random, "anchor_robust_correct": current}
        bundle["routes"][route] = {
            **route_payload,
            "history_ids_sha256": selected_ids_sha256(tuple(history)),
            "random_ids_sha256": selected_ids_sha256(tuple(random)),
            "history_mask": f"{route.lower()}_h.json",
            "random_mask": f"{route.lower()}_r.json",
        }
    bundle_path.write_bytes(canonical_json(bundle) + b"\n")
    bundle_sha = sha256_file(bundle_path)

    def write_mask(path: Path, ids: list[int], provenance: Mapping[str, Any]) -> str:
        counts = {
            str(label): sum(parent[item][0] == label for item in ids)
            for label in sorted({parent[item][0] for item in ids})
        }
        payload = {
            "schema_version": 1,
            "namespace": "train",
            "num_classes": 10,
            "selected_ids": ids,
            "selected_ids_sha256": selected_ids_sha256(tuple(ids)),
            "selected_count": len(ids),
            "selected_class_counts": counts,
            "provenance": dict(provenance),
        }
        path.write_bytes(canonical_json(payload) + b"\n")
        return sha256_file(path)

    created: dict[str, Path] = {"bundle": bundle_path}
    for route, current in routes.items():
        route_value = bundle["routes"][route]
        history, random = route_value["history"], route_value["random"]
        route_name = "peak_failure" if current else "non_recovery"
        common = {
            "parent_checkpoint_sha256": parent_sha,
            "parent_sample_state_sha256": state_sha,
            "route": route_name,
            "anchor_robust_correct": current,
        }
        history_path = output_dir / route_value["history_mask"]
        history_sha = write_mask(
            history_path,
            history,
            {
                **common,
                "source": "prescriptive_v3_online_history",
                "approved_selector_spec_sha256": bundle_sha,
                "selector_spec_path": str(bundle_path.resolve()),
            },
        )
        random_path = output_dir / route_value["random_mask"]
        write_mask(
            random_path,
            random,
            {
                **common,
                "source": "prescriptive_v3_matched_random",
                "random_seed": 0,
                "generator": "sha256",
                "generator_version": "prescriptive-v3-v1",
                "reference_history_mask_sha256": history_sha,
                "reference_selected_count": len(history),
                "reference_selected_class_counts": json.loads(history_path.read_text(encoding="utf-8"))[
                    "selected_class_counts"
                ],
                "reference_history_selector_spec_sha256": bundle_sha,
            },
        )
        created[f"{route}_H"] = history_path
        created[f"{route}_R"] = random_path
    return created


PRESCRIPTIVE_V3_KIND = "prescriptive_v3_intervention_v1"
ARMS = ("PF_RET_H", "PF_RET_R", "NR_PFX_H", "NR_PFX_R")


def _read_mapping(path: Path, *, name: str, yaml_input: bool = False) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_input else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise PrescriptiveV3Error(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PrescriptiveV3Error(f"{name} must be a mapping")
    return value


def _labels_from_parent(payload: Mapping[str, Any]) -> dict[int, int]:
    state = payload.get("sample_state")
    records = state.get("records") if isinstance(state, Mapping) else None
    if not isinstance(records, Mapping):
        raise PrescriptiveV3Error("parent sample state records are unavailable")
    labels: dict[int, int] = {}
    for raw_id, record in records.items():
        if not isinstance(raw_id, str) or not raw_id.isdigit() or not isinstance(record, Mapping):
            raise PrescriptiveV3Error("parent stable-ID state drifted")
        label = record.get("true_label")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 10:
            raise PrescriptiveV3Error("parent stable-ID label drifted")
        labels[int(raw_id)] = label
    return labels


def _validate_v3_delta(*, source: ExperimentConfig, arm: ExperimentConfig) -> None:
    if arm.prescriptive_v3 is None or arm.intervention is not None:
        raise PrescriptiveV3Error("v3 arm must use only the standalone prescriptive-v3 contract")
    parent = resolved_config_dict(source)
    child = resolved_config_dict(arm)
    if parent.pop("protocol") != {"id": "controlled_cifar10_r18_v1"} or child.pop("protocol") != {
        "id": "controlled_cifar10_r18_prescriptive_v3_v1"
    }:
        raise PrescriptiveV3Error("v3 fork requires baseline and dedicated prescriptive protocol identities")
    for key in ("output_dir", "tracking", "prescriptive_v3"):
        parent.pop(key, None)
        child.pop(key, None)
    parent_scheduler = parent.pop("scheduler", None)
    child_scheduler = child.pop("scheduler", None)
    if parent != child:
        raise PrescriptiveV3Error("v3 child changes fields outside protocol/scheduler/tracking/output/treatment")
    expected_parent = {"id": "multistep", "milestones": list(PARENT_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"}
    expected_child = {"id": "multistep", "milestones": list(CHILD_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"}
    if parent_scheduler != expected_parent or child_scheduler != expected_child:
        raise PrescriptiveV3Error("v3 requires the fixed [100,150] to [120,170] schedule transformation")


def _validate_v3_generation_delta(*, delayed: ExperimentConfig, arm: ExperimentConfig) -> None:
    """Check config derivation before the parent-state fork verifies `[100,150]`."""
    parent = resolved_config_dict(delayed)
    child = resolved_config_dict(arm)
    if parent.pop("protocol") != {"id": "controlled_cifar10_r18_delayed_multistep_v1"} or child.pop(
        "protocol"
    ) != {"id": "controlled_cifar10_r18_prescriptive_v3_v1"}:
        raise PrescriptiveV3Error("v3 config must derive from the registered delayed-control protocol")
    for key in ("output_dir", "tracking", "prescriptive_v3"):
        parent.pop(key, None)
        child.pop(key, None)
    if parent != child:
        raise PrescriptiveV3Error("generated v3 config changes fields outside protocol/tracking/output/treatment")


def _validate_bundle_and_masks(
    *, arms: list[ExperimentConfig], labels: Mapping[int, int], parent_sha: str, state_sha: str
) -> tuple[Path, str]:
    first = arms[0].prescriptive_v3
    assert first is not None
    bundle_path = first.selector_bundle_path
    bundle_sha = first.selector_bundle_sha256
    if not bundle_path.is_file() or sha256_file(bundle_path) != bundle_sha:
        raise PrescriptiveV3Error("v3 selector bundle bytes do not match every arm registration")
    bundle = _read_mapping(bundle_path, name="v3 selector bundle")
    if (
        bundle.get("schema_version") != 1
        or bundle.get("kind") != "prescriptive_v3_epoch34_online_epoch79_route_v1"
        or not isinstance(bundle.get("parent"), Mapping)
        or bundle["parent"].get("checkpoint_sha256") != parent_sha
        or bundle["parent"].get("sample_state_sha256") != state_sha
    ):
        raise PrescriptiveV3Error("v3 selector bundle does not bind the exact epoch-79 parent")
    routes = bundle.get("routes")
    if not isinstance(routes, Mapping):
        raise PrescriptiveV3Error("v3 selector bundle routes are unavailable")
    for arm in arms:
        spec = arm.prescriptive_v3
        assert spec is not None
        if spec.selector_bundle_path.resolve() != bundle_path.resolve() or spec.selector_bundle_sha256 != bundle_sha:
            raise PrescriptiveV3Error("v3 arms must use one exact selector bundle")
        route = "PF_RET" if spec.arm.startswith("PF_") else "NR_PFX"
        route_value = routes.get(route)
        if not isinstance(route_value, Mapping):
            raise PrescriptiveV3Error("v3 selector bundle lacks an arm route")
        expected_ids = (
            route_value["history_ids_sha256"] if spec.arm.endswith("_H") else route_value["random_ids_sha256"]
        )
        mask = spec.mask
        if mask.selected_ids_sha256 != expected_ids:
            raise PrescriptiveV3Error("v3 mask selected IDs do not match the frozen route bundle")
        try:
            load_fixed_intervention_mask(
                mask.path,
                expected_sha256=mask.sha256,
                expected_selected_ids_sha256=mask.selected_ids_sha256,
                expected_selected_count=mask.selected_count,
                expected_class_counts=mask.selected_class_counts,
                expected_provenance=mask.provenance.exact_payload(),
                train_labels=labels,
                num_classes=10,
            )
        except ValueError as exc:
            raise PrescriptiveV3Error("v3 mask is invalid for the exact source partition") from exc
    return bundle_path, bundle_sha


def create_prescriptive_v3_forks(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    artifact_inventory: Path,
    artifact_attestation: Path,
    schedule_spec: Path,
    arm_config_paths: list[Path],
    root: Path,
    git_state_collector: Any = collect_git_state,
) -> dict[str, Path]:
    """Atomically fork four epoch-79 children with delayed scheduler state.

    This composes the exact schedule-control parent verification rather than
    copying an epoch-79 checkpoint with stale `[100,150]` scheduler bytes.
    """
    arms = [load_config(path) for path in arm_config_paths]
    names = {arm.prescriptive_v3.arm if arm.prescriptive_v3 is not None else None for arm in arms}
    expected = set(ARMS)
    if len(arms) != 4 or names != expected:
        raise PrescriptiveV3Error("v3 fork requires one config for each PF_RET/NR_PFX H/R arm")
    spec_parent, spec_sha = _spec_parent(schedule_spec)
    try:
        payload, source, artifact, migration = _validate_parent(
            checkpoint=parent_checkpoint,
            parent_resolved_config=parent_resolved_config,
            parent_manifest=parent_manifest,
            artifact_inventory=artifact_inventory,
            artifact_attestation=artifact_attestation,
            child=arms[0],
            parent=spec_parent,
        )
    except Exception as exc:
        raise PrescriptiveV3Error("v3 parent schedule-control evidence is invalid") from exc
    parent_sha = schedule_sha256_file(parent_checkpoint)
    state_sha = _hash(payload["sample_state"])
    labels = _labels_from_parent(payload)
    if any(
        arm.prescriptive_v3 is None
        or arm.prescriptive_v3.parent.model_dump(mode="json") != spec_parent
        or arm.prescriptive_v3.anchor_checkpoint_sha256 != parent_sha
        or arm.prescriptive_v3.parent.checkpoint_sha256 != parent_sha
        or arm.prescriptive_v3.parent.sample_state_sha256 != state_sha
        for arm in arms
    ):
        raise PrescriptiveV3Error("v3 arm config parent/anchor lineage drifted")
    for arm in arms:
        _validate_v3_delta(source=source, arm=arm)
    bundle_path, bundle_sha = _validate_bundle_and_masks(
        arms=arms, labels=labels, parent_sha=parent_sha, state_sha=state_sha
    )
    outputs = [arm.output_dir.resolve() for arm in arms]
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        raise PrescriptiveV3Error("v3 fork outputs must be distinct and absent")
    screen_root = outputs[0].parent
    if any(
        output.parent != screen_root or output.name != arm.prescriptive_v3.arm
        for output, arm in zip(outputs, arms)
    ):
        raise PrescriptiveV3Error("v3 outputs must be named arm children of one absent screen root")
    if screen_root.exists():
        raise PrescriptiveV3Error("refusing to create a v3 screen in an existing root")
    git = git_state_collector(root)
    fork_git = git.get("sha")
    if not isinstance(fork_git, str) or len(fork_git) != 40 or git.get("dirty") is not False:
        raise PrescriptiveV3Error("v3 fork requires a clean addressable current Git SHA")
    planned: dict[str, tuple[ExperimentConfig, str, str]] = {}
    parent_run_id = payload.get("tracker_run_id")
    run_ids: set[str] = set()
    for arm in arms:
        assert arm.prescriptive_v3 is not None
        name = arm.prescriptive_v3.arm
        child_hash = config_digest(resolved_config_dict(arm))
        run_id = stable_run_id(arm, config_hash=child_hash, git_sha=fork_git)
        if run_id == parent_run_id or run_id in run_ids:
            raise PrescriptiveV3Error("v3 child tracker run IDs must be unique and distinct from parent")
        run_ids.add(run_id)
        planned[name] = (arm, child_hash, run_id)
    identity = {
        "schema_version": 1,
        "kind": PRESCRIPTIVE_V3_KIND,
        "parent_checkpoint_sha256": parent_sha,
        "parent_run_id": parent_run_id,
        "fork_git_sha": fork_git,
        "arms": [
            {"arm": name, "config_hash": digest, "run_id": run_id, "output": name}
            for name, (_, digest, run_id) in sorted(planned.items())
        ],
    }
    screen_id = _hash(identity)
    screen_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prescriptive-v3-", dir=screen_root.parent))
    created: dict[str, Path] = {}
    try:
        entries: list[dict[str, Any]] = []
        for name, (arm, child_hash, run_id) in sorted(planned.items()):
            assert arm.prescriptive_v3 is not None
            child = copy.deepcopy(payload)
            child["config_hash"] = child_hash
            child["tracker_run_id"] = run_id
            child["scheduler"] = _replace_scheduler_milestones(payload["scheduler"])
            child["best_metric"] = float("-inf")
            metadata = _post_fork_selection_metadata(payload["selection_metadata"])
            metadata["prescriptive_v3"] = arm.prescriptive_v3.model_dump(mode="json")
            child["selection_metadata"] = metadata
            lineage: dict[str, Any] = {
                "kind": PRESCRIPTIVE_V3_KIND,
                "screen_id": screen_id,
                "arm": name,
                "child_tracker_run_id": run_id,
                "child_config_sha256": child_hash,
                "parent_tracker_run_id": parent_run_id,
                "parent_checkpoint_sha256": parent_sha,
                "parent_raw_config_sha256": spec_parent["raw_config_sha256"],
                "parent_git_sha": spec_parent["git_sha"],
                "parent_epoch": PARENT79,
                "parent_world_size": 1,
                "parent_teacher_checkpoint_sha256": spec_parent["teacher_checkpoint_sha256"],
                "parent_sample_state_records": 45_000,
                "parent_sample_state_sha256": state_sha,
                "parent_artifact_attestation_sha256": spec_parent["artifact_attestation_sha256"],
                "parent_wandb_checkpoint_artifact": artifact,
                "schedule_control_spec_sha256": spec_sha,
                "selector_bundle_path": str(bundle_path.resolve()),
                "selector_bundle_sha256": bundle_sha,
                "parent_scheduler": {"milestones": list(PARENT_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"},
                "child_scheduler": {"milestones": list(CHILD_MILESTONES), "gamma": 0.1, "step_at": "epoch_end"},
                "fork_git_sha": fork_git,
                "post_fork_best_scope": True,
            }
            if migration is not None:
                lineage["parent_config_compatibility_migration"] = migration
            child["fork_lineage"] = lineage
            output = staging / name
            _atomic_torch_save(child, output / "last.pt")
            save_resolved_config(arm, output / "resolved_config.yaml")
            (output / "fork-lineage.json").write_bytes(canonical_json(lineage) + b"\n")
            entries.append(
                {
                    "arm": name,
                    "output": name,
                    "config_hash": child_hash,
                    "run_id": run_id,
                    "fork_checkpoint_sha256": schedule_sha256_file(output / "last.pt"),
                }
            )
            created[name] = screen_root / name / "last.pt"
        screen = {**identity, "screen_id": screen_id, "status": "complete", "arms": entries}
        (staging / "screen-complete.json").write_bytes(canonical_json(screen) + b"\n")
        for name in planned:
            (staging / name / "screen-complete.json").write_bytes(canonical_json(screen) + b"\n")
        os.replace(staging, screen_root)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return created


def write_prescriptive_v3_arm_configs(
    *,
    delayed_config: Path,
    schedule_spec: Path,
    parent_checkpoint: Path,
    selector_bundle: Path,
    masks_dir: Path,
    config_dir: Path,
    output_root: Path,
    run_prefix: str,
) -> dict[str, Path]:
    """Derive four immutable arm configs from a verified delayed-control config.

    The generated files are inputs to the fork operation, never children of
    ``output_root``; this keeps a later atomic fork able to reject preexisting
    outputs.
    """
    if config_dir.exists() or output_root.exists():
        raise FileExistsError("v3 config and output roots must both be absent")
    raw = _read_mapping(delayed_config, name="verified delayed child config", yaml_input=True)
    try:
        base = ExperimentConfig.model_validate(raw)
    except ValueError as exc:
        raise PrescriptiveV3Error("verified delayed child config is not strict") from exc
    if base.protocol.id != "controlled_cifar10_r18_delayed_multistep_v1" or base.intervention is not None:
        raise PrescriptiveV3Error("v3 configs must derive from an ordinary registered delayed-control config")
    # The config input is accepted through the strict schema, which may fill
    # defaults omitted by a hand-written delayed-control YAML.  Generate from
    # that resolved representation so every child has an explicit tracking
    # mapping and a stable config digest rather than depending on omission.
    raw = resolved_config_dict(base)
    parent, _ = _spec_parent(schedule_spec)
    parent_sha = schedule_sha256_file(parent_checkpoint)
    if parent_sha != parent["checkpoint_sha256"]:
        raise PrescriptiveV3Error("parent checkpoint bytes do not match schedule-control spec")
    bundle_sha = sha256_file(selector_bundle)
    bundle = _read_mapping(selector_bundle, name="v3 selector bundle")
    if not isinstance(bundle.get("parent"), Mapping) or bundle["parent"].get("checkpoint_sha256") != parent_sha:
        raise PrescriptiveV3Error("selector bundle does not bind supplied epoch-79 parent")
    config_dir.mkdir(parents=True)
    generated: dict[str, Path] = {}
    for arm in ARMS:
        route = "PF_RET" if arm.startswith("PF_") else "NR_PFX"
        suffix = "h" if arm.endswith("_H") else "r"
        mask_path = masks_dir / f"{route.lower()}_{suffix}.json"
        mask = _read_mapping(mask_path, name=f"{arm} mask")
        raw_arm = copy.deepcopy(raw)
        raw_arm["protocol"] = {"id": "controlled_cifar10_r18_prescriptive_v3_v1"}
        raw_arm["output_dir"] = str((output_root / arm).resolve())
        tracking = raw_arm.get("tracking")
        if not isinstance(tracking, dict):
            raise PrescriptiveV3Error("verified delayed config lacks a tracking mapping")
        tracking["run_id"] = f"{run_prefix}-{arm.lower()}"
        raw_arm["intervention"] = None
        raw_arm["prescriptive_v3"] = {
            "arm": arm,
            "parent": parent,
            "mask": {
                "path": str(mask_path.resolve()),
                "sha256": sha256_file(mask_path),
                "selected_ids_sha256": mask["selected_ids_sha256"],
                "selected_count": mask["selected_count"],
                "selected_class_counts": mask["selected_class_counts"],
                "provenance": mask["provenance"],
            },
            "selector_bundle_path": str(selector_bundle.resolve()),
            "selector_bundle_sha256": bundle_sha,
            "anchor_checkpoint": str(parent_checkpoint.resolve()),
            "anchor_checkpoint_sha256": parent_sha,
        }
        try:
            parsed = ExperimentConfig.model_validate(raw_arm)
        except ValueError as exc:
            raise PrescriptiveV3Error(f"generated {arm} config violates its strict contract") from exc
        _validate_v3_generation_delta(delayed=base, arm=parsed)
        path = config_dir / f"{arm.lower()}.yaml"
        save_resolved_config(parsed, path)
        generated[arm] = path
    return generated
