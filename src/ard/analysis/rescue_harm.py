"""Hash-bound completed-v2 rescue/harm checkpoint replay and paired reports.

This module is deliberately read-only.  It evaluates fixed saved checkpoints
on the raw, unaugmented CIFAR-10 train partition and treats paired outcomes as
exploratory model-level moderation, never as unit-level causal effects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ard.analysis.h4a_taxonomy import _domain_panel, _lineage
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, canonical_json, runtime_identity
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import CheckpointInventory, inventory_run_bundle, logical_dataset_identity, sha256_file
from ard.analysis.teacher_risk_replay import build_replay_loader, load_historical_student
from ard.attacks import AttackRequest, LinfPGD
from ard.config.loader import load_resolved_config_for_evaluation, resolved_config_dict
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.policies import selected_ids_sha256
from ard.state import SampleStateStore


class RescueHarmError(ValueError):
    """Raised when a replay or paired report cannot prove its frozen inputs."""


EPOCHS = (99, 104, 109, 199)
ARMS = ("control", "PF_H", "PF_R", "NR_H", "NR_R")
SOURCE_ARM = {"control": "control", "PF_H": "PF_TA", "PF_R": "PF_R", "NR_H": "NR_TA", "NR_R": "NR_R"}
OBSERVATION_COLUMNS = (
    "namespace",
    "run_id",
    "arm",
    "seed",
    "epoch",
    "sample_id",
    "class_id",
    "clean_prediction",
    "clean_correct",
    "clean_probability_margin",
    "robust_prediction",
    "robust_correct",
    "robust_probability_margin",
)
CATEGORIES = ("rescued", "harmed", "stable_correct", "unchanged_failure")

# The v3 panel is deliberately separate from completed-v2.  In particular,
# epoch 79 is represented by one exact shared parent checkpoint, never by a
# copied child artifact.
V3_EPOCHS = (79, 99, 119, 129, 149, 199)
V3_ARMS = ("C", "PF-H", "PF-R", "NR-H", "NR-R")
V3_CHILD_TO_CONFIG_ARM = {"PF-H": "PF_RET_H", "PF-R": "PF_RET_R", "NR-H": "NR_PFX_H", "NR-R": "NR_PFX_R"}
V3_OBSERVATION_COLUMNS = OBSERVATION_COLUMNS + (
    "teacher_clean_prediction",
    "teacher_clean_correct",
    "teacher_clean_true_probability",
    "teacher_clean_probability_margin",
    "teacher_clean_entropy_normalized",
    "teacher_adversarial_prediction",
    "teacher_adversarial_correct",
    "teacher_adversarial_true_probability",
    "teacher_adversarial_probability_margin",
    "teacher_adversarial_entropy_normalized",
    "teacher_clean_to_adversarial_kl",
    "teacher_clean_to_adversarial_prediction_flip",
    "teacher_clean_to_adversarial_true_probability_delta",
    "teacher_clean_to_adversarial_margin_delta",
    "route",
    "mask_selected",
    "intervention_active",
    "intervention_identity",
    "pf_anchor_clean_probability_margin",
    "pf_anchor_adversarial_probability_margin",
)


@dataclass(frozen=True)
class Inventory:
    run_id: str
    arm: str
    seed: int
    teacher: Mapping[str, Any]
    config_hash: str
    checkpoints: tuple[CheckpointInventory, ...]


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RescueHarmError("expected lowercase SHA-256")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RescueHarmError(f"{name} must be an integer >= {minimum}")
    return value


def _json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueHarmError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise RescueHarmError(f"{name} must be a JSON object")
    return value


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _tracked_clean_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__).resolve(), root / "src/ard/cli/rescue_harm.py")
    try:
        relative = [str(path.relative_to(root)) for path in paths]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RescueHarmError("rescue/harm replay requires tracked source and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise RescueHarmError("rescue/harm replay requires a tracked-clean revision")
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }


def _student_identity(config: Any) -> dict[str, Any]:
    return config.student.model_dump(mode="json")


def _teacher_identity(value: object) -> tuple[str | None, str]:
    if not isinstance(value, Mapping):
        raise RescueHarmError("teacher identity must be a mapping")
    registry, checkpoint = value.get("registry_id"), value.get("checkpoint_sha256")
    if registry is not None and not isinstance(registry, str):
        raise RescueHarmError("teacher registry identity is invalid")
    return registry, _sha(checkpoint)


def load_checkpoint_inventory(path: Path) -> Inventory:
    """Validate immutable bytes and exact payload epochs before GPU replay."""
    value = _json(path, name="checkpoint inventory")
    required = {"schema_version", "run_id", "arm", "seed", "teacher", "config_hash", "checkpoints"}
    if set(value) != required or value.get("schema_version") != 1:
        raise RescueHarmError("checkpoint inventory schema drifted")
    run_id, arm, seed = value["run_id"], value["arm"], value["seed"]
    if (
        not isinstance(run_id, str)
        or not run_id
        or arm not in set(SOURCE_ARM.values())
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise RescueHarmError("checkpoint inventory run/arm/seed identity is invalid")
    if not isinstance(value["teacher"], Mapping):
        raise RescueHarmError("checkpoint inventory teacher identity is invalid")
    config_hash = _sha(value["config_hash"])
    raw = value["checkpoints"]
    if not isinstance(raw, list) or len(raw) != len(EPOCHS):
        raise RescueHarmError("checkpoint inventory requires exactly four fixed epochs")
    result: list[CheckpointInventory] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"epoch", "path", "sha256", "scientific_git_sha"}:
            raise RescueHarmError("checkpoint inventory entry schema drifted")
        epoch = _integer(item["epoch"], name="checkpoint epoch")
        checkpoint_path = Path(item["path"])
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != _sha(item["sha256"]):
            raise RescueHarmError("checkpoint inventory byte hash drifted")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - torch exception varies
            raise RescueHarmError("checkpoint payload is unreadable") from exc
        if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload):
            raise RescueHarmError("checkpoint payload lacks complete lineage")
        if (
            payload.get("epoch") != epoch
            or payload.get("config_hash") != config_hash
            or payload.get("tracker_run_id") != run_id
        ):
            raise RescueHarmError("checkpoint payload epoch/config/run lineage drifted")
        git_sha = item["scientific_git_sha"]
        if not isinstance(git_sha, str) or len(git_sha) not in {40, 64}:
            raise RescueHarmError("checkpoint inventory Git identity is invalid")
        result.append(
            CheckpointInventory(
                run_id=run_id,
                artifact_name=f"{arm}-epoch{epoch}",
                aliases=("last",),
                publication_order=epoch,
                path=str(checkpoint_path.resolve()),
                sha256=item["sha256"],
                epoch=epoch,
                sample_state_present=True,
                sample_state_count=0,
                config_hash=config_hash,
                scientific_git_sha=git_sha,
            )
        )
    if tuple(sorted(checkpoint.epoch for checkpoint in result)) != EPOCHS:
        raise RescueHarmError("checkpoint inventory epochs must be exactly 99/104/109/199")
    return Inventory(
        run_id=run_id,
        arm=arm,
        seed=seed,
        teacher=dict(value["teacher"]),
        config_hash=config_hash,
        checkpoints=tuple(sorted(result, key=lambda item: item.epoch)),
    )


def build_checkpoint_inventory(
    *, manifest: Path, resolved_config: Path, arm: str, seed: int, output: Path
) -> dict[str, Any]:
    """Freeze one completed local run bundle at the four paired snapshots."""
    if output.exists():
        raise FileExistsError("refusing to overwrite checkpoint inventory")
    if arm not in {"control", "PF_TA", "PF_R", "NR_TA", "NR_R"}:
        raise RescueHarmError("inventory arm must be control/PF_TA/PF_R/NR_TA/NR_R")
    entries = [item for item in inventory_run_bundle(manifest) if item.epoch in EPOCHS]
    if tuple(sorted(item.epoch for item in entries)) != EPOCHS:
        raise RescueHarmError("local run bundle lacks exactly the fixed 99/104/109/199 checkpoints")
    if len({item.run_id for item in entries}) != 1 or len({item.config_hash for item in entries}) != 1:
        raise RescueHarmError("fixed checkpoints do not share one run/config identity")
    evaluation = load_resolved_config_for_evaluation(resolved_config)
    if evaluation.raw_config_hash != entries[0].config_hash or evaluation.config.teacher is None:
        raise RescueHarmError("resolved config does not bind completed checkpoint inventory")
    value = {
        "schema_version": 1,
        "run_id": entries[0].run_id,
        "arm": arm,
        "seed": seed,
        "teacher": evaluation.config.teacher.model_dump(mode="json"),
        "config_hash": entries[0].config_hash,
        "checkpoints": [
            {
                "epoch": item.epoch,
                "path": str(Path(item.path).resolve()),
                "sha256": item.sha256,
                "scientific_git_sha": item.scientific_git_sha,
            }
            for item in sorted(entries, key=lambda item: item.epoch)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(value) + b"\n")
    return value


def _checkpoint_state_sha(path: Path) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch error details vary
        raise RescueHarmError("checkpoint payload is unreadable") from exc
    if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload):
        raise RescueHarmError("checkpoint payload lacks complete lineage")
    state = payload.get("sample_state")
    if not isinstance(state, Mapping):
        raise RescueHarmError("checkpoint payload lacks sample-state lineage")
    return _hash(state)


def _v3_mask_and_parent(config: Any, *, arm: str) -> dict[str, Any]:
    """Return only bytes explicitly frozen by the v3 child config."""
    if arm == "C":
        return {"kind": "shared_parent_control"}
    spec = config.prescriptive_v3
    if spec is None or spec.arm != V3_CHILD_TO_CONFIG_ARM[arm]:
        raise RescueHarmError("v3 resolved config arm contract drifted")
    mask_path, bundle_path, parent_path = spec.mask.path, spec.selector_bundle_path, spec.anchor_checkpoint
    for path, digest, name in (
        (mask_path, spec.mask.sha256, "mask"),
        (bundle_path, spec.selector_bundle_sha256, "selector bundle"),
        (parent_path, spec.anchor_checkpoint_sha256, "epoch-79 parent"),
    ):
        if not path.is_file() or sha256_file(path) != _sha(digest):
            raise RescueHarmError(f"v3 {name} bytes do not match the resolved config")
    if spec.parent.checkpoint_sha256 != spec.anchor_checkpoint_sha256:
        raise RescueHarmError("v3 parent/anchor checkpoint identity drifted")
    for path, digest, name in (
        (spec.parent.train_partition_manifest, spec.parent.train_partition_manifest_sha256, "parent partition"),
        (spec.parent.artifact_attestation, spec.parent.artifact_attestation_sha256, "parent attestation"),
        (spec.parent.artifact_inventory, spec.parent.artifact_inventory_sha256, "parent artifact inventory"),
    ):
        if not path.is_file() or sha256_file(path) != _sha(digest):
            raise RescueHarmError(f"v3 {name} bytes do not match the resolved config")
    parent_state = _checkpoint_state_sha(parent_path)
    if parent_state != _sha(spec.parent.sample_state_sha256):
        raise RescueHarmError("v3 parent sample-state identity drifted")
    return {
        "kind": "prescriptive_v3_epoch79_parent_v1",
        "checkpoint_path": str(parent_path.resolve()),
        "checkpoint_sha256": spec.anchor_checkpoint_sha256,
        "sample_state_sha256": parent_state,
        "parent_raw_config_sha256": spec.parent.raw_config_sha256,
        "parent_git_sha": spec.parent.git_sha,
        "parent_train_partition_manifest_sha256": spec.parent.train_partition_manifest_sha256,
        "parent_train_partition_ids_labels_sha256": spec.parent.train_partition_ids_labels_sha256,
        "parent_artifact_attestation_sha256": spec.parent.artifact_attestation_sha256,
        "parent_artifact_inventory_sha256": spec.parent.artifact_inventory_sha256,
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": spec.mask.sha256,
        "selected_ids_sha256": spec.mask.selected_ids_sha256,
        "selector_bundle_path": str(bundle_path.resolve()),
        "selector_bundle_sha256": spec.selector_bundle_sha256,
        "route": "PF" if arm.startswith("PF-") else "NR",
        "arm": spec.arm,
    }


def _explicit_v3_control_parent(*, checkpoint: Path, children: Sequence[CheckpointInventory]) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise RescueHarmError("v3 control shared epoch-79 parent checkpoint is unavailable")
    parent_sha, state_sha = sha256_file(checkpoint), _checkpoint_state_sha(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload) or payload.get("epoch") != 79:
        raise RescueHarmError("v3 control shared parent is not an exact epoch-79 checkpoint")
    for child in children:
        child_payload = torch.load(Path(child.path), map_location="cpu", weights_only=False)
        lineage = child_payload.get("fork_lineage") if isinstance(child_payload, Mapping) else None
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("parent_epoch") != 79
            or lineage.get("parent_checkpoint_sha256") != parent_sha
            or lineage.get("parent_sample_state_sha256") != state_sha
        ):
            raise RescueHarmError("v3 delayed-control checkpoint fork lineage does not bind the shared epoch-79 parent")
    return {
        "kind": "explicit_shared_epoch79_parent_v1",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": parent_sha,
        "sample_state_sha256": state_sha,
    }


def build_v3_checkpoint_inventory(
    *,
    manifest: Path,
    resolved_config: Path,
    arm: str,
    seed: int,
    output: Path,
    epochs: Sequence[int] = V3_EPOCHS,
    shared_parent_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Freeze a v3 panel from checkpoint bytes, not artifact names/versions."""
    if output.exists():
        raise FileExistsError("refusing to overwrite checkpoint inventory")
    requested = tuple(epochs)
    if arm not in V3_ARMS or requested != tuple(sorted(set(requested))) or not set(requested).issubset(V3_EPOCHS):
        raise RescueHarmError("v3 inventory requires a sorted unique requested v3 common-epoch set")
    if 79 not in requested:
        raise RescueHarmError("v3 common epoch contract must retain the shared epoch-79 parent")
    evaluation = load_resolved_config_for_evaluation(resolved_config)
    config = evaluation.config
    if config.teacher is None:
        raise RescueHarmError("v3 inventory requires a frozen teacher")
    if arm != "C" and shared_parent_checkpoint is not None:
        raise RescueHarmError("v3 child inventory represents epoch-79 only through its config parent")
    entries = list(inventory_run_bundle(manifest))
    if not entries or len({item.run_id for item in entries}) != 1:
        raise RescueHarmError("v3 manifest lacks one immutable run identity")
    wanted = set(requested) - {79}
    selected = [item for item in entries if item.epoch in wanted and item.periodic_last]
    if tuple(sorted(item.epoch for item in selected)) != tuple(sorted(wanted)) or len(
        {item.epoch for item in selected}
    ) != len(selected):
        raise RescueHarmError("v3 manifest lacks exactly the requested periodic checkpoint bytes")
    if any(item.config_hash != evaluation.raw_config_hash for item in selected):
        raise RescueHarmError("v3 checkpoint config hash does not match resolved config")
    if arm == "C":
        if shared_parent_checkpoint is None:
            raise RescueHarmError("v3 control inventory requires --shared-parent-checkpoint for epoch-79")
        parent = _explicit_v3_control_parent(checkpoint=shared_parent_checkpoint, children=selected)
    else:
        parent = _v3_mask_and_parent(config, arm=arm)
    checkpoint_rows = []
    for item in sorted(selected, key=lambda value: value.epoch):
        path = Path(item.path)
        checkpoint_rows.append(
            {
                "epoch": item.epoch,
                "path": str(path.resolve()),
                "sha256": item.sha256,
                "sample_state_sha256": _checkpoint_state_sha(path),
                "scientific_git_sha": item.scientific_git_sha,
            }
        )
    value = {
        "schema_version": 3,
        "contract": "prescriptive_v3_rescue_harm_inventory_v1",
        "run_id": entries[0].run_id,
        "arm": arm,
        "seed": seed,
        "requested_epochs": list(requested),
        "config_sha256": evaluation.raw_config_hash,
        "scientific_git_sha": entries[0].scientific_git_sha,
        "teacher": config.teacher.model_dump(mode="json"),
        "dataset_identity": logical_dataset_identity(resolved_config_dict(config), train_expected_count=45_000),
        "attack_identity": config.method.selection_attack.model_dump(mode="json"),
        "source_manifest_sha256": sha256_file(manifest),
        "source_resolved_config_sha256": sha256_file(resolved_config),
        "parent": parent,
        "checkpoints": checkpoint_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(value) + b"\n")
    return value


def _ce_pgd20(config: Any) -> None:
    attack = config.method.selection_attack
    if (
        attack is None
        or attack.loss != "ce"
        or attack.steps != 20
        or attack.norm != "linf"
        or attack.input_domain != "pixel_0_1"
        or attack.student_mode != "eval"
    ):
        raise RescueHarmError("rescue/harm replay requires the saved CE PGD-20 eval pixel-space selection attack")


def _primitives(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits.float(), dim=1)
    prediction = probabilities.argmax(dim=1)
    true = probabilities.gather(1, labels[:, None]).squeeze(1)
    masked = probabilities.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    wrong = masked.max(dim=1).values
    return prediction, prediction.eq(labels), true - wrong


def _teacher_primitives(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    probabilities = F.softmax(logits.float(), dim=1)
    prediction, correct, margin = _primitives(logits, labels)
    true = probabilities.gather(1, labels[:, None]).squeeze(1)
    entropy = -(probabilities * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
    normalized_entropy = entropy / torch.log(torch.tensor(logits.shape[1], device=logits.device, dtype=entropy.dtype))
    if not torch.isfinite(normalized_entropy).all():
        raise RescueHarmError("teacher response primitives are non-finite")
    return {
        "probabilities": probabilities,
        "prediction": prediction,
        "correct": correct,
        "true": true,
        "margin": margin,
        "entropy": normalized_entropy,
    }


def _teacher_response_kl(clean_logits: torch.Tensor, adversarial_logits: torch.Tensor) -> torch.Tensor:
    """Compute KL(p_T(clean) || p_T(x_adv^S)) without probability-log NaNs."""
    clean_log = F.log_softmax(clean_logits.float(), dim=1)
    adversarial_log = F.log_softmax(adversarial_logits.float(), dim=1)
    value = (clean_log.exp() * (clean_log - adversarial_log)).sum(dim=1)
    if not torch.isfinite(value).all():
        raise RescueHarmError("teacher clean-to-adversarial KL is non-finite")
    return value


def _load_v3_inventory(path: Path) -> dict[str, Any]:
    value = _json(path, name="v3 checkpoint inventory")
    required = {
        "schema_version",
        "contract",
        "run_id",
        "arm",
        "seed",
        "requested_epochs",
        "config_sha256",
        "scientific_git_sha",
        "teacher",
        "dataset_identity",
        "attack_identity",
        "source_manifest_sha256",
        "source_resolved_config_sha256",
        "parent",
        "checkpoints",
    }
    if (
        set(value) != required
        or value.get("schema_version") != 3
        or value.get("contract") != "prescriptive_v3_rescue_harm_inventory_v1"
    ):
        raise RescueHarmError("v3 checkpoint inventory schema/contract drifted")
    requested = tuple(value.get("requested_epochs", ()))
    if (
        value.get("arm") not in V3_ARMS
        or not requested
        or requested != tuple(sorted(set(requested)))
        or not set(requested).issubset(V3_EPOCHS)
        or 79 not in requested
    ):
        raise RescueHarmError("v3 inventory common-epoch contract drifted")
    if not isinstance(value.get("parent"), Mapping) or not isinstance(value.get("checkpoints"), list):
        raise RescueHarmError("v3 inventory parent/checkpoint lineage is invalid")
    for key in ("config_sha256", "source_manifest_sha256", "source_resolved_config_sha256"):
        _sha(value.get(key))
    parent = value["parent"]
    if value["arm"] == "C":
        for key in ("checkpoint_path", "checkpoint_sha256", "sample_state_sha256"):
            if key not in parent:
                raise RescueHarmError("v3 control inventory shared-parent lineage is incomplete")
        parent_path = Path(str(parent["checkpoint_path"]))
        if (
            parent.get("kind") != "explicit_shared_epoch79_parent_v1"
            or not parent_path.is_file()
            or sha256_file(parent_path) != _sha(parent["checkpoint_sha256"])
            or _checkpoint_state_sha(parent_path) != _sha(parent["sample_state_sha256"])
        ):
            raise RescueHarmError("v3 control shared epoch-79 parent bytes drifted")
        parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        if not isinstance(parent_payload, Mapping) or parent_payload.get("epoch") != 79:
            raise RescueHarmError("v3 control shared parent epoch drifted")
    else:
        for key in (
            "checkpoint_path",
            "checkpoint_sha256",
            "sample_state_sha256",
            "mask_path",
            "mask_sha256",
            "selector_bundle_path",
            "selector_bundle_sha256",
        ):
            if key not in parent:
                raise RescueHarmError("v3 child inventory parent/mask lineage is incomplete")
        parent_path = Path(str(parent["checkpoint_path"]))
        if not parent_path.is_file() or sha256_file(parent_path) != _sha(parent["checkpoint_sha256"]):
            raise RescueHarmError("v3 shared parent checkpoint bytes drifted")
        if _checkpoint_state_sha(parent_path) != _sha(parent["sample_state_sha256"]):
            raise RescueHarmError("v3 shared parent sample-state bytes drifted")
        for path_key, hash_key in (("mask_path", "mask_sha256"), ("selector_bundle_path", "selector_bundle_sha256")):
            source = Path(str(parent[path_key]))
            if not source.is_file() or sha256_file(source) != _sha(parent[hash_key]):
                raise RescueHarmError("v3 mask/bundle bytes drifted")
    epochs = set(value["requested_epochs"])
    seen: set[int] = set()
    for row in value["checkpoints"]:
        if not isinstance(row, Mapping) or set(row) != {
            "epoch",
            "path",
            "sha256",
            "sample_state_sha256",
            "scientific_git_sha",
        }:
            raise RescueHarmError("v3 checkpoint inventory row schema drifted")
        epoch = _integer(row["epoch"], name="v3 checkpoint epoch")
        if epoch not in epochs or epoch in seen or (value["arm"] != "C" and epoch == 79):
            raise RescueHarmError("v3 checkpoint epochs must use the shared-parent representation")
        source = Path(str(row["path"]))
        if (
            not source.is_file()
            or sha256_file(source) != _sha(row["sha256"])
            or _checkpoint_state_sha(source) != _sha(row["sample_state_sha256"])
        ):
            raise RescueHarmError("v3 checkpoint/hash/sample-state drifted")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if (
            payload.get("epoch") != epoch
            or payload.get("tracker_run_id") != value["run_id"]
            or payload.get("config_hash") != value["config_sha256"]
        ):
            raise RescueHarmError("v3 checkpoint payload identity drifted")
        seen.add(epoch)
    if seen != epochs - {79}:
        raise RescueHarmError("v3 inventory lacks requested common checkpoint bytes")
    return value


def _v3_selected_ids(parent: Mapping[str, Any]) -> set[int]:
    if parent.get("kind") in {"shared_parent_control", "explicit_shared_epoch79_parent_v1"}:
        return set()
    mask = _json(Path(str(parent["mask_path"])), name="v3 mask")
    ids = mask.get("selected_ids")
    if (
        not isinstance(ids, list)
        or tuple(ids) != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        or selected_ids_sha256(tuple(ids)) != parent.get("selected_ids_sha256")
    ):
        raise RescueHarmError("v3 mask selected stable IDs/hash drifted")
    return set(ids)


def replay_inventory(
    *,
    resolved_config: Path,
    inventory_path: Path,
    output_parquet: Path,
    output_lineage: Path,
    device: torch.device,
    batch_size: int,
    analysis_seed: int,
    epochs: Sequence[int] = EPOCHS,
) -> dict[str, Any]:
    """Replay all four immutable snapshots for one arm without teacher forwards."""
    if output_parquet.exists() or output_lineage.exists():
        raise FileExistsError("refusing to overwrite rescue/harm replay output")
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RescueHarmError("requested replay device is unavailable")
    if batch_size < 1:
        raise RescueHarmError("replay batch_size must be positive")
    provenance = _tracked_clean_provenance()
    inventory = load_checkpoint_inventory(inventory_path)
    evaluation = load_resolved_config_for_evaluation(resolved_config)
    config = evaluation.config
    if config.dataset.name != "cifar10" or config.dataset.split != "train":
        raise RescueHarmError("rescue/harm replay is restricted to CIFAR-10 raw train data")
    if evaluation.raw_config_hash != inventory.config_hash or config.teacher is None:
        raise RescueHarmError("resolved config does not match immutable checkpoint inventory")
    if inventory.teacher != config.teacher.model_dump(mode="json"):
        raise RescueHarmError("checkpoint inventory teacher does not match resolved config teacher")
    _ce_pgd20(config)
    loader = build_replay_loader(config, batch_size=batch_size)
    if len(loader.dataset) != 45_000:
        raise RescueHarmError("raw CIFAR-10 replay loader must expose exactly 45,000 stable train IDs")
    rows: list[dict[str, Any]] = []
    attack = LinfPGD(config.method.selection_attack)
    selected_epochs = tuple(sorted(set(epochs)))
    if not selected_epochs or any(epoch not in EPOCHS for epoch in selected_epochs):
        raise RescueHarmError("replay epoch selection must be a non-empty subset of fixed snapshots")
    for checkpoint in (item for item in inventory.checkpoints if item.epoch in selected_epochs):
        student, _ = load_historical_student(
            checkpoint, config=config, device=device, expected_config_hash=inventory.config_hash
        )
        student.eval()
        for batch_index, raw_batch in enumerate(loader):
            batch = raw_batch.to(device)
            generator = torch.Generator(device=device).manual_seed(analysis_seed + 1_000_003 * batch_index)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                clean_logits = student(batch.images.float()).float()
            result = attack.generate(
                AttackRequest(inputs=batch.images, labels=batch.labels, student=student, generator=generator)
            )
            if result.max_abs_delta > float(config.method.selection_attack.epsilon_value) + 1e-7:
                raise RescueHarmError("CE PGD replay violated its pixel-space Linf bound")
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                robust_logits = student(result.adversarial.float()).float()
                clean_prediction, clean_correct, clean_margin = _primitives(clean_logits, batch.labels)
                robust_prediction, robust_correct, robust_margin = _primitives(robust_logits, batch.labels)
            rows.extend(
                {
                    "namespace": "train",
                    "run_id": inventory.run_id,
                    "arm": inventory.arm,
                    "seed": inventory.seed,
                    "epoch": checkpoint.epoch,
                    "sample_id": int(sample_id),
                    "class_id": int(class_id),
                    "clean_prediction": int(cp),
                    "clean_correct": bool(cc),
                    "clean_probability_margin": float(cm),
                    "robust_prediction": int(rp),
                    "robust_correct": bool(rc),
                    "robust_probability_margin": float(rm),
                }
                for sample_id, class_id, cp, cc, cm, rp, rc, rm in zip(
                    batch.sample_ids.tolist(),
                    batch.labels.tolist(),
                    clean_prediction.tolist(),
                    clean_correct.tolist(),
                    clean_margin.tolist(),
                    robust_prediction.tolist(),
                    robust_correct.tolist(),
                    robust_margin.tolist(),
                    strict=True,
                )
            )
        student.zero_grad(set_to_none=True)
    expected = len(loader.dataset)
    if len(rows) != expected * len(selected_epochs) or any(
        sum(row["epoch"] == epoch for row in rows) != expected for epoch in selected_epochs
    ):
        raise RescueHarmError("replay lacks exact stable-ID coverage at one or more epochs")
    write_sample_parquet(sorted(rows, key=lambda row: (int(row["epoch"]), int(row["sample_id"]))), output_parquet)
    lineage = {
        "schema_version": 1,
        "contract": "completed_v2_rescue_harm_replay_v1",
        "observations_sha256": sha256_file(output_parquet),
        "run_id": inventory.run_id,
        "arm": inventory.arm,
        "seed": inventory.seed,
        "teacher": inventory.teacher,
        "config_sha256": inventory.config_hash,
        "source_resolved_config_sha256": sha256_file(resolved_config),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "checkpoints": [
            {"epoch": item.epoch, "sha256": item.sha256}
            for item in inventory.checkpoints
            if item.epoch in selected_epochs
        ],
        "dataset_identity": logical_dataset_identity(resolved_config_dict(config), train_expected_count=expected),
        "attack_identity": config.method.selection_attack.model_dump(mode="json"),
        "analysis_seed": analysis_seed,
        "student_identity": _student_identity(config),
        "runtime": runtime_identity(device),
        "row_count": len(rows),
        "analysis_provenance": provenance,
    }
    output_lineage.parent.mkdir(parents=True, exist_ok=True)
    output_lineage.write_bytes(canonical_json(lineage) + b"\n")
    return lineage


def replay_v3_inventory(
    *,
    resolved_config: Path,
    inventory_path: Path,
    output_parquet: Path,
    output_lineage: Path,
    device: torch.device,
    batch_size: int,
    analysis_seed: int,
    expected_count: int = 45_000,
    expected_epochs: Sequence[int] | None = None,
    emit_epochs: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Replay the frozen v3 union, including teacher response on student CE-PGD20 inputs.

    This is intentionally a separate entry point: accepting a v2 inventory here
    would silently omit the v3 parent/mask and teacher-response contracts.
    """
    if output_parquet.exists() or output_lineage.exists():
        raise FileExistsError("refusing to overwrite rescue/harm replay output")
    if (
        device.type not in {"cpu", "cuda"}
        or (device.type == "cuda" and not torch.cuda.is_available())
        or batch_size < 1
    ):
        raise RescueHarmError("v3 replay device or batch size is invalid")
    provenance = _tracked_clean_provenance()
    inventory = _load_v3_inventory(inventory_path)
    if expected_epochs is not None and tuple(expected_epochs) != tuple(inventory["requested_epochs"]):
        raise RescueHarmError("v3 replay --epoch selection does not match the immutable inventory epochs")
    emitted = tuple(
        (
            inventory["requested_epochs"]
            if inventory["arm"] == "C"
            else [epoch for epoch in inventory["requested_epochs"] if epoch != 79]
        )
        if emit_epochs is None
        else emit_epochs
    )
    if (
        not emitted
        or emitted != tuple(sorted(set(emitted)))
        or not set(emitted).issubset(inventory["requested_epochs"])
    ):
        raise RescueHarmError("v3 replay emitted epochs must be a non-empty immutable inventory subset")
    evaluation = load_resolved_config_for_evaluation(resolved_config)
    config = evaluation.config
    if config.teacher is None or evaluation.raw_config_hash != inventory["config_sha256"]:
        raise RescueHarmError("v3 resolved config does not match immutable inventory")
    if (
        config.teacher.model_dump(mode="json") != inventory["teacher"]
        or config.dataset.name != "cifar10"
        or config.dataset.split != "train"
    ):
        raise RescueHarmError("v3 teacher/dataset identity drifted")
    _ce_pgd20(config)
    loader = build_replay_loader(config, batch_size=batch_size)
    if expected_count < 1 or len(loader.dataset) != expected_count:
        raise RescueHarmError("v3 replay requires the exact frozen stable train-ID population")
    selected_ids = _v3_selected_ids(inventory["parent"])
    if selected_ids and not selected_ids.issubset(
        {int(item) for batch in loader for item in batch.sample_ids.tolist()}
    ):
        raise RescueHarmError("v3 mask is outside the frozen train stable-ID population")
    teacher = build_teacher(config.teacher, tier=config.tier).to(device).eval()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RescueHarmError("v3 replay teacher is not frozen")
    rows: list[dict[str, Any]] = []
    source_rows = {int(row["epoch"]): row for row in inventory["checkpoints"]}
    parent = inventory["parent"]
    source_rows[79] = {"epoch": 79, "path": parent["checkpoint_path"], "sha256": parent["checkpoint_sha256"]}
    attack = LinfPGD(config.method.selection_attack)
    anchor: torch.nn.Module | None = None
    for epoch in inventory["requested_epochs"]:
        source = source_rows[epoch]
        student = build_student(config.student, tier=config.tier).to(device)
        payload = load_saved_student_checkpoint(Path(str(source["path"])), student)
        if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload) or payload.get("epoch") != epoch:
            raise RescueHarmError("v3 replay checkpoint payload drifted")
        student.eval()
        if inventory["arm"].startswith("PF-") and epoch == 79:
            anchor = student
        if inventory["arm"].startswith("PF-") and anchor is None:
            raise RescueHarmError("v3 PF replay cannot derive anchor alignment from shared epoch-79 parent")
        if epoch not in emitted:
            continue
        for batch_index, raw_batch in enumerate(loader):
            batch = raw_batch.to(device)
            generator = torch.Generator(device=device).manual_seed(analysis_seed + 1_000_003 * batch_index)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                clean_logits = student(batch.images.float()).float()
            result = attack.generate(
                AttackRequest(inputs=batch.images, labels=batch.labels, student=student, generator=generator)
            )
            if result.max_abs_delta > float(config.method.selection_attack.epsilon_value) + 1e-7:
                raise RescueHarmError("v3 CE-PGD20 replay violated pixel-space Linf bound")
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                robust_logits = student(result.adversarial.float()).float()
                teacher_clean_logits = teacher(batch.images.float()).float()
                teacher_adversarial_logits = teacher(result.adversarial.float()).float()
                tc = _teacher_primitives(teacher_clean_logits, batch.labels)
                ta = _teacher_primitives(teacher_adversarial_logits, batch.labels)
                kl = _teacher_response_kl(teacher_clean_logits, teacher_adversarial_logits)
                clean_prediction, clean_correct, clean_margin = _primitives(clean_logits, batch.labels)
                robust_prediction, robust_correct, robust_margin = _primitives(robust_logits, batch.labels)
                if anchor is not None:
                    ac = _primitives(anchor(batch.images.float()).float(), batch.labels)[2]
                    aa = _primitives(anchor(result.adversarial.float()).float(), batch.labels)[2]
                else:
                    ac = aa = None
            route = "control" if inventory["arm"] == "C" else ("PF" if inventory["arm"].startswith("PF-") else "NR")
            active = bool(epoch >= 80 and ((route == "PF" and epoch <= 129) or (route == "NR" and epoch <= 99)))
            identity = (
                "control"
                if route == "control"
                else "pf_teacher_0.75_anchor_0.25_epochs80_129"
                if route == "PF"
                else "nr_prefix_pgd5_selected_epochs80_99_else_pgd10"
            )
            for offset, sample_id in enumerate(batch.sample_ids.tolist()):
                is_selected = int(sample_id) in selected_ids
                rows.append(
                    {
                        "namespace": "train",
                        "run_id": inventory["run_id"],
                        "arm": inventory["arm"],
                        "seed": inventory["seed"],
                        "epoch": epoch,
                        "sample_id": int(sample_id),
                        "class_id": int(batch.labels[offset]),
                        "clean_prediction": int(clean_prediction[offset]),
                        "clean_correct": bool(clean_correct[offset]),
                        "clean_probability_margin": float(clean_margin[offset]),
                        "robust_prediction": int(robust_prediction[offset]),
                        "robust_correct": bool(robust_correct[offset]),
                        "robust_probability_margin": float(robust_margin[offset]),
                        "teacher_clean_prediction": int(tc["prediction"][offset]),
                        "teacher_clean_correct": bool(tc["correct"][offset]),
                        "teacher_clean_true_probability": float(tc["true"][offset]),
                        "teacher_clean_probability_margin": float(tc["margin"][offset]),
                        "teacher_clean_entropy_normalized": float(tc["entropy"][offset]),
                        "teacher_adversarial_prediction": int(ta["prediction"][offset]),
                        "teacher_adversarial_correct": bool(ta["correct"][offset]),
                        "teacher_adversarial_true_probability": float(ta["true"][offset]),
                        "teacher_adversarial_probability_margin": float(ta["margin"][offset]),
                        "teacher_adversarial_entropy_normalized": float(ta["entropy"][offset]),
                        "teacher_clean_to_adversarial_kl": float(kl[offset]),
                        "teacher_clean_to_adversarial_prediction_flip": bool(
                            tc["prediction"][offset] != ta["prediction"][offset]
                        ),
                        "teacher_clean_to_adversarial_true_probability_delta": float(
                            ta["true"][offset] - tc["true"][offset]
                        ),
                        "teacher_clean_to_adversarial_margin_delta": float(ta["margin"][offset] - tc["margin"][offset]),
                        "route": route,
                        "mask_selected": is_selected,
                        "intervention_active": bool(active and is_selected),
                        "intervention_identity": identity,
                        "pf_anchor_clean_probability_margin": None if ac is None else float(ac[offset]),
                        "pf_anchor_adversarial_probability_margin": None if aa is None else float(aa[offset]),
                    }
                )
        if student is not anchor:
            student.zero_grad(set_to_none=True)
    expected = expected_count * len(emitted)
    if len(rows) != expected:
        raise RescueHarmError("v3 replay lacks exact stable-ID coverage")
    write_sample_parquet(sorted(rows, key=lambda row: (int(row["epoch"]), int(row["sample_id"]))), output_parquet)
    lineage = {
        "schema_version": 3,
        "contract": "prescriptive_v3_rescue_harm_replay_v1",
        "observations_sha256": sha256_file(output_parquet),
        "inventory_sha256": sha256_file(inventory_path),
        "run_id": inventory["run_id"],
        "arm": inventory["arm"],
        "seed": inventory["seed"],
        "requested_epochs": list(emitted),
        "parent_epochs": inventory["requested_epochs"],
        "config_sha256": inventory["config_sha256"],
        "scientific_git_sha": inventory["scientific_git_sha"],
        "teacher": inventory["teacher"],
        "dataset_identity": inventory["dataset_identity"],
        "attack_identity": inventory["attack_identity"],
        "source_manifest_sha256": inventory["source_manifest_sha256"],
        "source_resolved_config_sha256": inventory["source_resolved_config_sha256"],
        "parent": inventory["parent"],
        "checkpoints": inventory["checkpoints"],
        "analysis_seed": analysis_seed,
        "row_count": len(rows),
        "observation_columns": list(V3_OBSERVATION_COLUMNS),
        "analysis_provenance": provenance,
    }
    output_lineage.parent.mkdir(parents=True, exist_ok=True)
    output_lineage.write_bytes(canonical_json(lineage) + b"\n")
    return lineage


def _read_observations(
    path: Path, lineage_path: Path, *, arm: str
) -> tuple[dict[str, Any], dict[int, dict[int, dict[str, Any]]]]:
    lineage = _json(lineage_path, name=f"{arm} lineage")
    if (
        lineage.get("contract") != "completed_v2_rescue_harm_replay_v1"
        or lineage.get("arm") != arm
        or lineage.get("observations_sha256") != sha256_file(path)
    ):
        raise RescueHarmError(f"{arm} observation lineage drifted")
    for key in (
        "seed",
        "config_sha256",
        "dataset_identity",
        "attack_identity",
        "analysis_seed",
        "teacher",
        "student_identity",
    ):
        if key not in lineage:
            raise RescueHarmError(f"{arm} observation lineage is incomplete")
    if not isinstance(lineage["teacher"], Mapping) or not isinstance(lineage["student_identity"], Mapping):
        raise RescueHarmError(f"{arm} observation lineage model identity is invalid")
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:  # pragma: no cover
        raise RescueHarmError(f"{arm} observations are unreadable") from exc
    if tuple(table.column_names) != OBSERVATION_COLUMNS:
        raise RescueHarmError(f"{arm} observation schema drifted")
    panels = {epoch: {} for epoch in EPOCHS}
    for row in table.to_pylist():
        epoch, sample_id, class_id = row.get("epoch"), row.get("sample_id"), row.get("class_id")
        if (
            epoch not in panels
            or not isinstance(sample_id, int)
            or not isinstance(class_id, int)
            or row.get("namespace") != "train"
            or sample_id in panels[epoch]
        ):
            raise RescueHarmError(f"{arm} stable-ID/epoch contract drifted")
        if (
            row.get("run_id") != lineage.get("run_id")
            or row.get("arm") != arm
            or row.get("seed") != lineage.get("seed")
        ):
            raise RescueHarmError(f"{arm} row identity drifted")
        if not isinstance(row.get("robust_correct"), bool) or not isinstance(row.get("clean_correct"), bool):
            raise RescueHarmError(f"{arm} correctness schema drifted")
        panels[epoch][sample_id] = dict(row)
    expected = lineage.get("row_count")
    if (
        not isinstance(expected, int)
        or expected != sum(len(panel) for panel in panels.values())
        or any(not panel for panel in panels.values())
    ):
        raise RescueHarmError(f"{arm} row count drifted")
    reference = panels[EPOCHS[0]]
    if any(
        set(panel) != set(reference)
        or any(panel[sample_id]["class_id"] != reference[sample_id]["class_id"] for sample_id in reference)
        for panel in panels.values()
    ):
        raise RescueHarmError(f"{arm} stable ID/class join drifted")
    return lineage, panels


def merge_epoch_replays(
    *, inputs: Mapping[int, tuple[Path, Path]], output_parquet: Path, output_lineage: Path
) -> dict[str, Any]:
    """Combine four independently replayed snapshots into one formal arm panel."""
    if output_parquet.exists() or output_lineage.exists():
        raise FileExistsError("refusing to overwrite merged rescue/harm replay output")
    if set(inputs) != set(EPOCHS):
        raise RescueHarmError("merge requires exactly one single-epoch input for 99/104/109/199")
    rows_by_epoch: dict[int, list[dict[str, Any]]] = {}
    lineages: dict[int, dict[str, Any]] = {}
    identity_keys = (
        "run_id",
        "arm",
        "seed",
        "teacher",
        "config_sha256",
        "dataset_identity",
        "attack_identity",
        "analysis_seed",
        "student_identity",
        "analysis_provenance",
    )
    for epoch in EPOCHS:
        path, lineage_path = inputs[epoch]
        lineage = _json(lineage_path, name=f"epoch-{epoch} replay lineage")
        if (
            lineage.get("contract") != "completed_v2_rescue_harm_replay_v1"
            or lineage.get("observations_sha256") != sha256_file(path)
            or lineage.get("row_count") is None
        ):
            raise RescueHarmError("single-epoch replay lineage drifted")
        checkpoints = lineage.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1 or checkpoints[0].get("epoch") != epoch:
            raise RescueHarmError("single-epoch replay lineage checkpoint identity drifted")
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
        except Exception as exc:  # pragma: no cover
            raise RescueHarmError("single-epoch observations are unreadable") from exc
        if tuple(table.column_names) != OBSERVATION_COLUMNS:
            raise RescueHarmError("single-epoch observation schema drifted")
        rows = [dict(row) for row in table.to_pylist()]
        if not rows or len(rows) != lineage["row_count"] or any(row.get("epoch") != epoch for row in rows):
            raise RescueHarmError("single-epoch observation count/epoch drifted")
        ids: set[int] = set()
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, int) or sample_id in ids or row.get("namespace") != "train":
                raise RescueHarmError("single-epoch stable-ID contract drifted")
            ids.add(sample_id)
            if (
                row.get("run_id") != lineage.get("run_id")
                or row.get("arm") != lineage.get("arm")
                or row.get("seed") != lineage.get("seed")
            ):
                raise RescueHarmError("single-epoch row identity drifted")
        rows_by_epoch[epoch], lineages[epoch] = rows, lineage
    reference = lineages[EPOCHS[0]]
    if any(any(lineages[epoch].get(key) != reference.get(key) for key in identity_keys) for epoch in EPOCHS[1:]):
        raise RescueHarmError("single-epoch replay identity drifted")
    reference_rows = {int(row["sample_id"]): int(row["class_id"]) for row in rows_by_epoch[EPOCHS[0]]}
    if any(
        {int(row["sample_id"]): int(row["class_id"]) for row in rows_by_epoch[epoch]} != reference_rows
        for epoch in EPOCHS[1:]
    ):
        raise RescueHarmError("single-epoch stable ID/class join drifted")
    rows = [row for epoch in EPOCHS for row in sorted(rows_by_epoch[epoch], key=lambda item: int(item["sample_id"]))]
    write_sample_parquet(rows, output_parquet)
    lineage = {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"observations_sha256", "row_count", "checkpoints", "runtime"}
        },
        "observations_sha256": sha256_file(output_parquet),
        "checkpoints": [lineages[epoch]["checkpoints"][0] for epoch in EPOCHS],
        "runtime": {"merged_from_single_epoch_replays": True},
        "row_count": len(rows),
    }
    output_lineage.parent.mkdir(parents=True, exist_ok=True)
    output_lineage.write_bytes(canonical_json(lineage) + b"\n")
    return lineage


def merge_v3_epoch_replays(
    *, inputs: Mapping[int, tuple[Path, Path]], output_parquet: Path, output_lineage: Path
) -> dict[str, Any]:
    """Merge v3 smoke replays, retaining one shared epoch-79 parent row set."""
    if output_parquet.exists() or output_lineage.exists():
        raise FileExistsError("refusing to overwrite merged rescue/harm replay output")
    if set(inputs) != set(V3_EPOCHS):
        raise RescueHarmError("v3 merge requires exactly one input for each frozen common epoch")
    lineages: dict[int, dict[str, Any]] = {}
    rows_by_epoch: dict[int, list[dict[str, Any]]] = {}
    identity_keys = (
        "run_id",
        "arm",
        "seed",
        "config_sha256",
        "scientific_git_sha",
        "teacher",
        "dataset_identity",
        "attack_identity",
        "source_manifest_sha256",
        "source_resolved_config_sha256",
        "parent",
        "analysis_seed",
        "analysis_provenance",
        "observation_columns",
    )
    for epoch in V3_EPOCHS:
        path, lineage_path = inputs[epoch]
        lineage = _json(lineage_path, name=f"v3 epoch-{epoch} lineage")
        requested = tuple(lineage.get("requested_epochs", ()))
        if (
            lineage.get("schema_version") != 3
            or lineage.get("contract") != "prescriptive_v3_rescue_harm_replay_v1"
            or lineage.get("observations_sha256") != sha256_file(path)
            or epoch not in requested
            or requested != tuple(sorted(set(requested)))
            or not set(requested).issubset(V3_EPOCHS)
            or tuple(lineage.get("observation_columns", ())) != V3_OBSERVATION_COLUMNS
        ):
            raise RescueHarmError("v3 single-epoch replay lineage drifted or mixes a v2 contract")
        try:
            import pyarrow.parquet as pq

            rows = [dict(row) for row in pq.read_table(path).to_pylist() if row.get("epoch") == epoch]
        except Exception as exc:  # pragma: no cover
            raise RescueHarmError("v3 single-epoch observations are unreadable") from exc
        if not rows or any(row.get("epoch") != epoch for row in rows):
            raise RescueHarmError("v3 single-epoch observation epoch drifted")
        ids = [row.get("sample_id") for row in rows]
        if any(not isinstance(item, int) for item in ids) or len(ids) != len(set(ids)):
            raise RescueHarmError("v3 single-epoch stable-ID contract drifted")
        lineages[epoch], rows_by_epoch[epoch] = lineage, rows
    reference = lineages[79]
    if any(any(lineages[epoch].get(key) != reference.get(key) for key in identity_keys) for epoch in V3_EPOCHS[1:]):
        raise RescueHarmError("v3 single-epoch replay identity drifted")
    classes = {int(row["sample_id"]): int(row["class_id"]) for row in rows_by_epoch[79]}
    if any(
        {int(row["sample_id"]): int(row["class_id"]) for row in rows_by_epoch[epoch]} != classes
        for epoch in V3_EPOCHS[1:]
    ):
        raise RescueHarmError("v3 single-epoch stable ID/class join drifted")
    rows = [
        row for epoch in V3_EPOCHS for row in sorted(rows_by_epoch[epoch], key=lambda value: int(value["sample_id"]))
    ]
    write_sample_parquet(rows, output_parquet)
    checkpoints = [
        row for epoch in V3_EPOCHS for row in lineages[epoch].get("checkpoints", []) if row.get("epoch") == epoch
    ]
    output = {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"observations_sha256", "requested_epochs", "row_count", "checkpoints"}
        },
        "observations_sha256": sha256_file(output_parquet),
        "requested_epochs": list(V3_EPOCHS),
        "checkpoints": checkpoints,
        "row_count": len(rows),
    }
    output_lineage.parent.mkdir(parents=True, exist_ok=True)
    output_lineage.write_bytes(canonical_json(output) + b"\n")
    return output


def _resolve_mask_path(bundle: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RescueHarmError("selector mask path is invalid")
    declared = Path(value)
    if declared.is_file():
        return declared
    fallback = bundle.parent / declared.name
    if fallback.is_file():
        return fallback
    raise RescueHarmError("selector mask path is unavailable locally and has no bundle-relative fallback")


def _parent_online_state(
    checkpoint: Path, *, parent: Mapping[str, Any], feature: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[int, bool], dict[str, str]]:
    """Bind eligibility to the exact settled online state that made the masks."""
    if not checkpoint.is_file() or sha256_file(checkpoint) != _sha(parent.get("checkpoint_sha256")):
        raise RescueHarmError("parent checkpoint SHA does not match selector bundle")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch error details vary
        raise RescueHarmError("parent checkpoint is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or REQUIRED_KEYS.difference(payload)
        or payload.get("epoch") != 39
        or payload.get("epoch_boundary") != "end"
    ):
        raise RescueHarmError("parent checkpoint is not the exact epoch-39 end boundary")
    state = payload.get("sample_state")
    expected_keys = {"format_version", "ema_decay", "records", "pending", "next_order"}
    if (
        not isinstance(state, Mapping)
        or set(state) != expected_keys
        or state.get("format_version") != 3
        or state.get("pending") != []
        or isinstance(state.get("next_order"), bool)
        or not isinstance(state.get("next_order"), int)
        or state["next_order"] < 0
        or isinstance(state.get("ema_decay"), bool)
        or not isinstance(state.get("ema_decay"), (int, float))
        or not 0 <= float(state["ema_decay"]) < 1
        or not isinstance(state.get("records"), Mapping)
    ):
        raise RescueHarmError("parent checkpoint lacks a settled format-v3 SampleStateStore")
    state_sha = _hash(state)
    if state_sha != _sha(parent.get("sample_state_sha256")):
        raise RescueHarmError("parent sample-state SHA does not match selector bundle")
    store = SampleStateStore(ema_decay=float(state["ema_decay"]))
    try:
        store.load_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise RescueHarmError("parent checkpoint SampleStateStore records are invalid") from exc
    if store.pending or len(store.records) != len(feature):
        raise RescueHarmError("parent checkpoint SampleStateStore is not settled/exact")
    online: dict[int, bool] = {}
    for raw_id, record in state["records"].items():
        if not isinstance(raw_id, str) or not raw_id.isdigit() or not isinstance(record, Mapping):
            raise RescueHarmError("parent sample-state stable-ID record is invalid")
        sample_id = int(raw_id)
        label, previous, seen, hits = (
            record.get("true_label"),
            record.get("previous_robust_correct"),
            record.get("seen"),
            record.get("robust_correct_count"),
        )
        if (
            sample_id in online
            or isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < 10
            or not isinstance(previous, bool)
            or isinstance(seen, bool)
            or not isinstance(seen, int)
            or seen != 40
            or isinstance(hits, bool)
            or not isinstance(hits, int)
            or not 0 <= hits <= seen
            or record.get("history_statistics_complete") is not True
            or sample_id not in feature
            or feature[sample_id]["class_id"] != label
        ):
            raise RescueHarmError("parent sample-state stable-ID/class/temporal join drifted")
        online[sample_id] = previous
    if set(online) != set(feature):
        raise RescueHarmError("parent sample-state and feature replay stable-ID set drifted")
    return online, {"checkpoint_sha256": sha256_file(checkpoint), "sample_state_sha256": state_sha}


def _mask_bundle(
    path: Path, *, feature: Mapping[int, Mapping[str, Any]], parent_checkpoint: Path
) -> tuple[dict[str, set[int]], dict[str, str]]:
    """Load frozen masks and their exact online, rather than replay, eligibility."""
    value = _json(path, name="epoch39 selector bundle")
    required = {"schema_version", "kind", "parent", "selection", "mask_paths"}
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("kind") != "history_routing_v2_online_selector_v1"
    ):
        raise RescueHarmError("selector bundle schema/version drifted")
    parent, selection, paths = value["parent"], value["selection"], value["mask_paths"]
    if (
        not isinstance(parent, Mapping)
        or parent.get("epoch") != 39
        or not isinstance(selection, Mapping)
        or not isinstance(paths, Mapping)
    ):
        raise RescueHarmError("selector bundle parent/selection lineage drifted")
    online, parent_identity = _parent_online_state(parent_checkpoint, parent=parent, feature=feature)
    route_specs = {
        "PF_H": ("peak_failure", "history", True),
        "PF_R": ("peak_failure", "random", True),
        "NR_H": ("non_recovery", "history", False),
        "NR_R": ("non_recovery", "random", False),
    }
    result: dict[str, set[int]] = {}
    labels = {sample_id: int(row["class_id"]) for sample_id, row in feature.items()}
    bundle_sha = sha256_file(path)
    for arm, (route, kind, anchor_correct) in route_specs.items():
        pair = paths.get(route)
        metadata = selection.get(route)
        if not isinstance(pair, Mapping) or not isinstance(metadata, Mapping):
            raise RescueHarmError("selector bundle route metadata drifted")
        mask_path = _resolve_mask_path(path, pair.get(kind))
        mask = _json(mask_path, name=f"{arm} selector mask")
        expected_keys = {
            "schema_version",
            "namespace",
            "num_classes",
            "selected_ids",
            "selected_ids_sha256",
            "selected_count",
            "selected_class_counts",
            "provenance",
        }
        ids = mask.get("selected_ids")
        if (
            set(mask) != expected_keys
            or mask.get("schema_version") != 1
            or mask.get("namespace") != "train"
            or mask.get("num_classes") != 10
            or not isinstance(ids, list)
        ):
            raise RescueHarmError("selector mask schema drifted")
        if (
            tuple(ids) != tuple(sorted(ids))
            or len(ids) != len(set(ids))
            or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        ):
            raise RescueHarmError("selector mask selected IDs are not sorted unique integers")
        selected = set(ids)
        if (
            not selected.issubset(labels)
            or mask.get("selected_count") != len(ids)
            or mask.get("selected_ids_sha256") != selected_ids_sha256(tuple(ids))
        ):
            raise RescueHarmError("selector mask selected IDs/count/hash drifted")
        counts = {
            str(class_id): sum(labels[sample_id] == class_id for sample_id in ids)
            for class_id in sorted(set(labels.values()))
        }
        counts = {key: value for key, value in counts.items() if value}
        if (
            mask.get("selected_class_counts") != counts
            or metadata.get("selected_count") != len(ids)
            or metadata.get("selected_class_counts") != counts
        ):
            raise RescueHarmError("selector mask class/count metadata drifted")
        provenance = mask.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("route") != route
            or provenance.get("anchor_robust_correct") is not anchor_correct
            or provenance.get("parent_checkpoint_sha256") != parent.get("checkpoint_sha256")
            or provenance.get("parent_sample_state_sha256") != parent.get("sample_state_sha256")
        ):
            raise RescueHarmError("selector mask parent/route provenance drifted")
        if kind == "history":
            if provenance.get("approved_selector_spec_sha256") != bundle_sha:
                raise RescueHarmError("history selector mask does not bind this bundle")
        elif provenance.get("reference_history_selector_spec_sha256") != bundle_sha:
            raise RescueHarmError("random selector mask does not bind this bundle")
        eligible = {sample_id for sample_id, previous in online.items() if previous is anchor_correct}
        if not selected.issubset(eligible) or metadata.get("eligible_count") != len(eligible):
            raise RescueHarmError("selector mask does not match epoch39 online-state route eligibility")
        result[arm] = selected
    # Eligibility is the parent checkpoint's online pre-update observation;
    # replay robust correctness remains exclusive to Control-to-arm outcomes.
    result["PF"] = {sample_id for sample_id, previous in online.items() if previous}
    result["NR"] = set(feature) - result["PF"]
    return result, parent_identity


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = {name: sum(row["category"] == name for row in rows) for name in CATEGORIES}
    count = len(rows)
    return {
        "count": count,
        "categories": categories,
        "net_rescue": categories["rescued"] - categories["harmed"],
        "net_rescue_rate": (categories["rescued"] - categories["harmed"]) / count if count else None,
    }


def _category(control: bool, arm: bool) -> str:
    return (
        "rescued"
        if not control and arm
        else "harmed"
        if control and not arm
        else "stable_correct"
        if arm
        else "unchanged_failure"
    )


def _feature_panel(
    path: Path, lineage_path: Path, *, expected_count: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    try:
        meta = _lineage(
            lineage_path,
            path,
            key="feature_observations_sha256",
            expected_count=expected_count,
            protocol="feature_protocol",
        )
        import pyarrow.parquet as pq

        raw = [dict(row) for row in pq.read_table(path).to_pylist()]
        panel = _domain_panel(raw, epochs=FEATURE_EPOCHS, expected_count=expected_count, name="feature")
    except ValueError as exc:
        raise RescueHarmError(str(exc)) from exc
    # Preserve the exact stored true-label probability only after validating
    # the schema-v2 record with H4a's frozen field/algebra checks.
    for row in raw:
        if row["epoch"] == 39:
            panel[39][row["sample_id"]]["teacher_clean_true_probability"] = float(row["teacher_clean_true_probability"])
    if not isinstance(meta.get("teacher"), Mapping):
        raise RescueHarmError("feature replay lacks teacher identity")
    return panel[39], meta


def report_rescue_harm(
    *,
    observations: Mapping[str, tuple[Path, Path]],
    mask_bundle: Path,
    feature_observations: Path,
    feature_lineage: Path,
    parent_checkpoint: Path,
    output: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Report exhaustive paired categories; masks are frozen before moderators join."""
    if output.exists():
        raise FileExistsError("refusing to overwrite rescue/harm report")
    if set(observations) != set(ARMS):
        raise RescueHarmError("report requires control and all four PF/NR history/random arms")
    parsed = {arm: _read_observations(*observations[arm], arm=SOURCE_ARM[arm]) for arm in ARMS}
    control_meta, control = parsed["control"]
    identity_keys = ("seed", "dataset_identity", "attack_identity", "analysis_seed", "teacher", "student_identity")
    if any(any(parsed[arm][0].get(key) != control_meta.get(key) for key in identity_keys) for arm in ARMS[1:]):
        raise RescueHarmError("arm replay attack/student/teacher/seed/dataset identity drifted")
    ids = set(control[EPOCHS[0]])
    if len(ids) != expected_count or any(
        set(parsed[arm][1][EPOCHS[0]]) != ids
        or any(
            parsed[arm][1][EPOCHS[0]][sample_id]["class_id"] != control[EPOCHS[0]][sample_id]["class_id"]
            for sample_id in ids
        )
        for arm in ARMS[1:]
    ):
        raise RescueHarmError("arms do not share one exact stable-ID population")
    feature, feature_meta = _feature_panel(feature_observations, feature_lineage, expected_count=expected_count)
    if _teacher_identity(feature_meta.get("teacher")) != _teacher_identity(control_meta.get("teacher")):
        raise RescueHarmError("epoch39 feature replay teacher lineage identity drifted")
    if set(feature) != ids or any(
        feature[sample_id]["class_id"] != control[EPOCHS[0]][sample_id]["class_id"] for sample_id in ids
    ):
        raise RescueHarmError("epoch39 feature replay stable-ID/class join drifted")
    masks, parent_identity = _mask_bundle(mask_bundle, feature=feature, parent_checkpoint=parent_checkpoint)
    per_epoch: dict[str, Any] = {}
    for epoch in EPOCHS:
        epoch_report: dict[str, Any] = {}
        for arm in ARMS[1:]:
            rows = []
            for sample_id in sorted(ids):
                c, a, moderator = control[epoch][sample_id], parsed[arm][1][epoch][sample_id], feature[sample_id]
                selected = sample_id in masks[arm]
                route = "PF" if arm.startswith("PF_") else "NR"
                eligible = sample_id in masks[route]
                rows.append(
                    {
                        "category": _category(bool(c["robust_correct"]), bool(a["robust_correct"])),
                        "selected": selected,
                        "eligible": eligible,
                        "route": route,
                        "selection": "history" if arm.endswith("_H") else "random",
                        "teacher_clean_correct": bool(moderator["teacher_clean_correct"]),
                        "teacher_adversarial_correct": bool(moderator["teacher_adversarial_correct"]),
                        "teacher_clean_to_adversarial_flip": bool(moderator["teacher_prediction_flip"]),
                        "true_label_mix_l1_distance": 1 - float(moderator["teacher_clean_true_probability"]),
                    }
                )
            if sum(_summary(rows)["categories"].values()) != len(rows):
                raise RescueHarmError("rescue/harm categories are not exhaustive")
            groups = {
                "all": rows,
                "selected": [row for row in rows if row["selected"]],
                "non_selected": [row for row in rows if not row["selected"]],
                "eligible": [row for row in rows if row["eligible"]],
                "non_eligible": [row for row in rows if not row["eligible"]],
                "teacher_clean_correct": [row for row in rows if row["teacher_clean_correct"]],
                "teacher_clean_wrong": [row for row in rows if not row["teacher_clean_correct"]],
                "teacher_adversarial_correct": [row for row in rows if row["teacher_adversarial_correct"]],
                "teacher_adversarial_wrong": [row for row in rows if not row["teacher_adversarial_correct"]],
                "teacher_clean_to_adversarial_flip": [row for row in rows if row["teacher_clean_to_adversarial_flip"]],
                "teacher_clean_to_adversarial_stable_prediction": [
                    row for row in rows if not row["teacher_clean_to_adversarial_flip"]
                ],
            }
            epoch_report[arm] = {
                "route": route,
                "selection": "history" if arm.endswith("_H") else "random",
                "categories": {name: _summary(group) for name, group in groups.items()},
                "true_label_mix_l1_distance": {
                    name: _float_summary([float(row["true_label_mix_l1_distance"]) for row in group])
                    for name, group in groups.items()
                },
            }
        per_epoch[str(epoch)] = epoch_report
    result = {
        "schema_version": 1,
        "contract": "completed_v2_rescue_harm_report_v1",
        "exploratory_model_level_moderation_not_identifiable_unit_causal_effect": True,
        "epochs": list(EPOCHS),
        "input_identity": {
            "arm_lineage_sha256": {arm: sha256_file(observations[arm][1]) for arm in ARMS},
            "mask_bundle_sha256": sha256_file(mask_bundle),
            "feature_lineage_sha256": sha256_file(feature_lineage),
            "attack_identity": control_meta["attack_identity"],
            "parent_checkpoint_sha256": parent_identity["checkpoint_sha256"],
            "parent_sample_state_sha256": parent_identity["sample_state_sha256"],
            "eligibility_domain": "epoch39_parent_sample_state.previous_robust_correct",
            "outcome_domain": "fixed_checkpoint_common_ce_pgd20_replay.robust_correct",
        },
        "epochs_report": per_epoch,
        "diagnostics": {
            "kl_js": "not_available_without_full_distribution",
            "gradient": "not_available_without_full_distribution",
            "official_test": "not_used",
        },
        "true_label_mix_l1_formula": (
            "||p_teacher - (0.5*p_teacher + 0.5*one_hot(y))||_1 = 1 - p_teacher_clean(y); exact for all samples"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result) + b"\n")
    return result


def _float_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _read_v3_observations(
    path: Path, lineage_path: Path, *, arm: str, require_parent_epoch: bool = True
) -> tuple[dict[str, Any], dict[int, dict[int, dict[str, Any]]]]:
    lineage = _json(lineage_path, name=f"{arm} v3 lineage")
    required = {
        "schema_version",
        "contract",
        "observations_sha256",
        "inventory_sha256",
        "run_id",
        "arm",
        "seed",
        "requested_epochs",
        "parent_epochs",
        "config_sha256",
        "scientific_git_sha",
        "teacher",
        "dataset_identity",
        "attack_identity",
        "source_manifest_sha256",
        "source_resolved_config_sha256",
        "parent",
        "checkpoints",
        "analysis_seed",
        "row_count",
        "observation_columns",
        "analysis_provenance",
    }
    if (
        set(lineage) != required
        or lineage.get("schema_version") != 3
        or lineage.get("contract") != "prescriptive_v3_rescue_harm_replay_v1"
        or lineage.get("arm") != arm
        or lineage.get("observations_sha256") != sha256_file(path)
    ):
        raise RescueHarmError("v3 observation lineage drifted or mixes a v2 contract")
    epochs = tuple(lineage.get("requested_epochs", ()))
    parent_epochs = tuple(lineage.get("parent_epochs", ()))
    if (
        not epochs
        or epochs != tuple(sorted(set(epochs)))
        or not set(epochs).issubset(V3_EPOCHS)
        or (require_parent_epoch and 79 not in epochs)
        or not parent_epochs
        or parent_epochs != tuple(sorted(set(parent_epochs)))
        or not set(epochs).issubset(parent_epochs)
        or 79 not in parent_epochs
        or tuple(lineage.get("observation_columns", ())) != V3_OBSERVATION_COLUMNS
    ):
        raise RescueHarmError("v3 observations do not contain the frozen common epoch/schema union")
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:  # pragma: no cover
        raise RescueHarmError("v3 observations are unreadable") from exc
    if tuple(table.column_names) != V3_OBSERVATION_COLUMNS:
        raise RescueHarmError("v3 observation parquet schema drifted")
    panels = {epoch: {} for epoch in epochs}
    for row in table.to_pylist():
        epoch, sample_id, class_id = row.get("epoch"), row.get("sample_id"), row.get("class_id")
        if (
            epoch not in panels
            or not isinstance(sample_id, int)
            or not isinstance(class_id, int)
            or sample_id in panels[epoch]
            or row.get("namespace") != "train"
            or row.get("run_id") != lineage["run_id"]
            or row.get("arm") != arm
            or row.get("seed") != lineage["seed"]
        ):
            raise RescueHarmError("v3 common epoch/stable-ID row identity drifted")
        if (
            not isinstance(row.get("clean_correct"), bool)
            or not isinstance(row.get("robust_correct"), bool)
            or not isinstance(row.get("mask_selected"), bool)
            or not isinstance(row.get("intervention_active"), bool)
        ):
            raise RescueHarmError("v3 correctness/mask contract drifted")
        if arm.startswith("PF-") and (
            row.get("pf_anchor_clean_probability_margin") is None
            or row.get("pf_anchor_adversarial_probability_margin") is None
        ):
            raise RescueHarmError("v3 PF anchor alignment is unavailable; refusing to fabricate it")
        if not arm.startswith("PF-") and (
            row.get("pf_anchor_clean_probability_margin") is not None
            or row.get("pf_anchor_adversarial_probability_margin") is not None
        ):
            raise RescueHarmError("v3 non-PF row contains fabricated anchor alignment")
        panels[epoch][sample_id] = dict(row)
    if lineage["row_count"] != sum(len(panel) for panel in panels.values()) or any(
        not panel for panel in panels.values()
    ):
        raise RescueHarmError("v3 observation row count drifted")
    reference = panels[79] if 79 in panels else panels[epochs[0]]
    if any(
        set(panel) != set(reference)
        or any(panel[item]["class_id"] != reference[item]["class_id"] for item in reference)
        for panel in panels.values()
    ):
        raise RescueHarmError("v3 stable sparse ID/class join drifted")
    return lineage, panels


def smoke_v3_report(
    *, observations: Path, lineage: Path, arm: str, epoch: int, expected_count: int, output: Path
) -> dict[str, Any]:
    """Validate one v3 target replay without constructing paired outcomes."""
    if output.exists():
        raise FileExistsError("refusing to overwrite v3 smoke report")
    if arm not in V3_ARMS or epoch not in V3_EPOCHS:
        raise RescueHarmError("v3 smoke report arm/epoch is invalid")
    meta, panels = _read_v3_observations(observations, lineage, arm=arm, require_parent_epoch=False)
    if tuple(meta["requested_epochs"]) != (epoch,) or epoch not in panels or len(panels[epoch]) != expected_count:
        raise RescueHarmError(
            "v3 smoke report requires exactly one emitted target epoch and expected sparse population"
        )
    rows = panels[epoch]
    if any(row["class_id"] < 0 or row["class_id"] >= 10 for row in rows.values()):
        raise RescueHarmError("v3 smoke report class IDs are invalid")
    result = {
        "schema_version": 3,
        "contract": "prescriptive_v3_rescue_harm_smoke_report_v1",
        "arm": arm,
        "epoch": epoch,
        "input_identity": {
            "observations_sha256": sha256_file(observations),
            "lineage_sha256": sha256_file(lineage),
            "parent": meta["parent"],
            "attack_identity": meta["attack_identity"],
            "analysis_provenance": meta["analysis_provenance"],
        },
        "stable_sparse_ids_sha256": _hash(sorted((sample_id, row["class_id"]) for sample_id, row in rows.items())),
        "count": len(rows),
        "intervention": {
            "active_count": sum(bool(row["intervention_active"]) for row in rows.values()),
            "identity": sorted({str(row["intervention_identity"]) for row in rows.values()}),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result) + b"\n")
    return result


def report_v3_rescue_harm(
    *,
    observations: Mapping[str, tuple[Path, Path]],
    output: Path,
    expected_count: int,
    report_epochs: Sequence[int] = V3_EPOCHS,
) -> dict[str, Any]:
    """Point report for v3 moderation and spillover; never an individual causal estimate."""
    if output.exists():
        raise FileExistsError("refusing to overwrite rescue/harm report")
    if set(observations) != set(V3_ARMS):
        raise RescueHarmError("v3 report requires C/PF-H/PF-R/NR-H/NR-R observations")
    parsed = {arm: _read_v3_observations(*observations[arm], arm=arm, require_parent_epoch=False) for arm in V3_ARMS}
    control_meta, control = parsed["C"]
    source_epochs = tuple(control_meta["requested_epochs"])
    selected_epochs = tuple(report_epochs)
    if (
        not selected_epochs
        or selected_epochs != tuple(sorted(set(selected_epochs)))
        or not set(selected_epochs).issubset(source_epochs)
        or source_epochs != V3_EPOCHS
        or any(tuple(parsed[arm][0]["requested_epochs"]) != V3_EPOCHS[1:] for arm in V3_ARMS[1:])
        or any(tuple(parsed[arm][0]["parent_epochs"]) != V3_EPOCHS for arm in V3_ARMS)
    ):
        raise RescueHarmError("v3 report epochs do not match one immutable common replay panel")
    if selected_epochs != V3_EPOCHS and len(selected_epochs) != 1:
        raise RescueHarmError("v3 smoke report permits exactly one explicit common epoch")
    ids = set(control[79])
    identity = ("seed", "dataset_identity", "attack_identity", "teacher", "analysis_seed", "analysis_provenance")
    if len(ids) != expected_count or any(
        any(parsed[arm][0].get(key) != control_meta.get(key) for key in identity) for arm in V3_ARMS[1:]
    ):
        raise RescueHarmError("v3 arm attack/teacher/seed/dataset identity drifted")
    c79 = next((row for row in control_meta["checkpoints"] if row.get("epoch") == 79), control_meta["parent"])
    if not isinstance(c79, Mapping):
        raise RescueHarmError("v3 control lineage lacks its shared epoch-79 parent checkpoint")
    c79_sha = c79.get("sha256", c79.get("checkpoint_sha256"))
    c79_state = c79.get("sample_state_sha256")
    if not isinstance(c79_sha, str) or not isinstance(c79_state, str):
        raise RescueHarmError("v3 control shared parent SHA/state is incomplete")
    child_git = {parsed[arm][0].get("scientific_git_sha") for arm in V3_ARMS[1:]}
    if len(child_git) != 1:
        raise RescueHarmError("v3 PF/NR children must share one scientific Git identity")
    for arm in V3_ARMS[1:]:
        parent = parsed[arm][0]["parent"]
        if parent.get("checkpoint_sha256") != c79_sha or parent.get("sample_state_sha256") != c79_state:
            raise RescueHarmError("v3 child does not bind the control shared epoch-79 parent/state")
        child_reference = parsed[arm][1][V3_EPOCHS[1]]
        if set(child_reference) != ids or any(
            child_reference[item]["class_id"] != control[V3_EPOCHS[1]][item]["class_id"] for item in ids
        ):
            raise RescueHarmError("v3 arms do not share one exact sparse-ID/class population")
    reports: dict[str, Any] = {}
    for epoch in selected_epochs:
        reports[str(epoch)] = {}
        for arm in V3_ARMS[1:]:
            if epoch == 79:
                reports[str(epoch)][arm] = {
                    "shared_parent_baseline": True,
                    "count": len(ids),
                    "robust_correct": sum(bool(row["robust_correct"]) for row in control[79].values()),
                }
                continue
            rows = []
            for sample_id in sorted(ids):
                c, a = control[epoch][sample_id], parsed[arm][1][epoch][sample_id]
                rows.append(
                    {
                        "category": _category(bool(c["robust_correct"]), bool(a["robust_correct"])),
                        "selected": bool(a["mask_selected"]),
                        "clean_transition": f"{int(bool(c['clean_correct']))}->{int(bool(a['clean_correct']))}",
                        "robust_transition": f"{int(bool(c['robust_correct']))}->{int(bool(a['robust_correct']))}",
                        "clean_margin_delta": float(a["clean_probability_margin"])
                        - float(c["clean_probability_margin"]),
                        "robust_margin_delta": float(a["robust_probability_margin"])
                        - float(c["robust_probability_margin"]),
                        "teacher_clean_to_adversarial_kl": float(a["teacher_clean_to_adversarial_kl"]),
                        "teacher_clean_to_adversarial_true_probability_delta": float(
                            a["teacher_clean_to_adversarial_true_probability_delta"]
                        ),
                        "teacher_clean_to_adversarial_margin_delta": float(
                            a["teacher_clean_to_adversarial_margin_delta"]
                        ),
                        "intervention_active": bool(a["intervention_active"]),
                        "intervention_identity": str(a["intervention_identity"]),
                        "pf_anchor_clean_probability_margin": a["pf_anchor_clean_probability_margin"],
                        "pf_anchor_adversarial_probability_margin": a["pf_anchor_adversarial_probability_margin"],
                        "student_minus_anchor_clean_margin": None
                        if a["pf_anchor_clean_probability_margin"] is None
                        else float(a["clean_probability_margin"]) - float(a["pf_anchor_clean_probability_margin"]),
                        "student_minus_anchor_robust_margin": None
                        if a["pf_anchor_adversarial_probability_margin"] is None
                        else float(a["robust_probability_margin"])
                        - float(a["pf_anchor_adversarial_probability_margin"]),
                    }
                )
            if sum(_summary(rows)["categories"].values()) != len(rows):
                raise RescueHarmError("v3 rescue/harm categories are not exhaustive")
            groups = {
                "all": rows,
                "selected": [row for row in rows if row["selected"]],
                "non_selected": [row for row in rows if not row["selected"]],
            }
            reports[str(epoch)][arm] = {
                "categories": {name: _summary(group) for name, group in groups.items()},
                "spillover_net_rescue": _summary(groups["non_selected"])["net_rescue"],
                "transitions": {
                    name: {
                        "clean": {
                            value: sum(row["clean_transition"] == value for row in group)
                            for value in ("0->0", "0->1", "1->0", "1->1")
                        },
                        "robust": {
                            value: sum(row["robust_transition"] == value for row in group)
                            for value in ("0->0", "0->1", "1->0", "1->1")
                        },
                    }
                    for name, group in groups.items()
                },
                "margin_change": {
                    name: {
                        "clean": _float_summary([row["clean_margin_delta"] for row in group]),
                        "robust": _float_summary([row["robust_margin_delta"] for row in group]),
                    }
                    for name, group in groups.items()
                },
                "teacher_response": {
                    name: {
                        "clean_to_student_adversarial_kl": _float_summary(
                            [row["teacher_clean_to_adversarial_kl"] for row in group]
                        ),
                        "true_probability_delta": _float_summary(
                            [row["teacher_clean_to_adversarial_true_probability_delta"] for row in group]
                        ),
                        "margin_delta": _float_summary(
                            [row["teacher_clean_to_adversarial_margin_delta"] for row in group]
                        ),
                    }
                    for name, group in groups.items()
                },
                "intervention": {
                    "active_count": sum(row["intervention_active"] for row in rows),
                    "active_selected_count": sum(row["intervention_active"] and row["selected"] for row in rows),
                    "identity": sorted({row["intervention_identity"] for row in rows}),
                    "nr_phase": "pre_window"
                    if arm.startswith("NR-") and epoch < 80
                    else "active_window"
                    if arm.startswith("NR-") and epoch <= 99
                    else "post_window"
                    if arm.startswith("NR-")
                    else "not_applicable",
                },
            }
            if arm.startswith("PF-"):
                reports[str(epoch)][arm]["pf_anchor_alignment"] = {
                    name: {
                        "anchor_clean_margin": _float_summary(
                            [float(row["pf_anchor_clean_probability_margin"]) for row in group]
                        ),
                        "anchor_adversarial_margin": _float_summary(
                            [float(row["pf_anchor_adversarial_probability_margin"]) for row in group]
                        ),
                        "student_minus_anchor_clean_margin": _float_summary(
                            [float(row["student_minus_anchor_clean_margin"]) for row in group]
                        ),
                        "student_minus_anchor_adversarial_margin": _float_summary(
                            [float(row["student_minus_anchor_robust_margin"]) for row in group]
                        ),
                    }
                    for name, group in groups.items()
                }
    result = {
        "schema_version": 3,
        "contract": "prescriptive_v3_rescue_harm_report_v1",
        "exploratory_model_level_moderation_not_identifiable_unit_causal_effect": True,
        "epochs": list(selected_epochs),
        "input_identity": {
            "arm_lineage_sha256": {arm: sha256_file(observations[arm][1]) for arm in V3_ARMS},
            "shared_parent_checkpoint_sha256": c79_sha,
            "shared_parent_sample_state_sha256": c79_state,
            "attack_identity": control_meta["attack_identity"],
            "arm_scientific_git_sha": {arm: parsed[arm][0]["scientific_git_sha"] for arm in V3_ARMS},
        },
        "epochs_report": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result) + b"\n")
    return result
