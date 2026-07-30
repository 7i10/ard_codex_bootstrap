"""Fail-closed frozen training-set oracle masks.

The oracle is deliberately an external, immutable input to a new scientific
method. It is constructed only from source RSLAD checkpoints and never reads
official-test outputs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.attacks import AttackRequest, LinfPGD
from ard.config import ExperimentConfig
from ard.config.loader import resolved_config_dict
from ard.config.schema import AttackConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedBatch,
    build_dataset,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.checkpoint import config_digest
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint, validate_checkpoint_lineage
from ard.models import build_student, build_teacher

from .teacher_risk_replay import git_identity, repository_root_from_source


class FrozenOracleError(RuntimeError):
    """A frozen-oracle input cannot prove its train-only provenance."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_base64_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hex(value: object, *, name: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise FrozenOracleError(f"{name} must be a lowercase {length}-character SHA-256/commit digest")
    return value


@dataclass(frozen=True)
class FrozenRiskLookup:
    """Immutable binary train-ID lookup used by the shared Trainer."""

    risks: Mapping[int, int]
    manifest_sha256: str
    source: Mapping[str, Any]

    def values(self, sample_ids: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if sample_ids.ndim != 1:
            raise FrozenOracleError("frozen oracle sample IDs must be a one-dimensional vector")
        values: list[int] = []
        for sample_id in sample_ids.detach().cpu().tolist():
            if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id not in self.risks:
                raise FrozenOracleError("frozen oracle has no risk for a training stable sample ID")
            values.append(self.risks[sample_id])
        return torch.tensor(values, device=device, dtype=dtype)


def train_labels(config: ExperimentConfig) -> dict[int, int]:
    """Return the exact source train partition labels, never official test labels."""
    if config.dataset.name != "cifar10" or config.dataset.split != "train":
        raise FrozenOracleError("frozen oracle is restricted to the CIFAR-10 official training split")
    from ard.data import build_train_validation_views

    train, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    raw_targets = getattr(train.dataset.dataset, "targets", None)
    if not isinstance(raw_targets, (list, tuple)):
        raise FrozenOracleError("source train dataset does not expose immutable integer labels")
    labels = {int(sample_id): int(raw_targets[sample_id]) for sample_id in train.indices}
    valid_labels = all(label >= 0 and label < config.dataset.num_classes for label in labels.values())
    if len(labels) != len(train.indices) or not valid_labels:
        raise FrozenOracleError("source training label namespace is invalid")
    return labels


def _source_manifest(path: Path, *, source_config_hash: str, source_teacher_sha256: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenOracleError("source run manifest is unreadable") from exc
    if not isinstance(parsed, Mapping):
        raise FrozenOracleError("source run manifest must be a JSON mapping")
    if parsed.get("config_hash") != source_config_hash:
        raise FrozenOracleError("source run manifest config hash does not match the source config")
    git = parsed.get("git")
    teacher = parsed.get("teacher")
    if not isinstance(git, Mapping):
        raise FrozenOracleError("source run manifest has no scientific Git lineage")
    git_sha = _hex(git.get("sha"), name="source scientific Git SHA", length=40)
    if not isinstance(teacher, Mapping) or teacher.get("checkpoint_sha256") != source_teacher_sha256:
        raise FrozenOracleError("source run manifest teacher lineage does not match source config")
    run_id = parsed.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise FrozenOracleError("source run manifest has no run ID")
    return {
        "run_id": run_id,
        "scientific_git_sha": git_sha,
        "source_manifest_sha256": sha256_file(path),
    }


def _build_rows(
    *,
    labels: Mapping[int, int],
    historical_correct: Mapping[int, bool],
    final_correct: Mapping[int, bool],
    selected_ids: set[int],
) -> list[dict[str, Any]]:
    expected = set(labels)
    if set(historical_correct) != expected or set(final_correct) != expected:
        raise FrozenOracleError("source checkpoints do not exactly cover the frozen training ID namespace")
    rows: list[dict[str, Any]] = []
    for sample_id, class_id in sorted(labels.items()):
        prior_correct = historical_correct[sample_id]
        final_value = final_correct[sample_id]
        if prior_correct and final_value:
            transition = "stable_correct"
        elif prior_correct and not final_value:
            transition = "future_forgetting"
        elif not prior_correct and final_value:
            transition = "recovered"
        else:
            transition = "persistent_failure"
        rows.append(
            {
                "namespace": "train",
                "sample_id": sample_id,
                "class_id": class_id,
                "risk": int(sample_id in selected_ids),
                "transition": transition,
                "source_historical_robust_correct": prior_correct,
                "source_final_robust_correct": final_value,
            }
        )
    return rows


def replay_source_hashes() -> dict[str, str]:
    """Hash the exact implementation used to construct the frozen mask."""
    analysis_path = Path(__file__).resolve()
    cli_path = analysis_path.parents[1] / "cli" / "build_frozen_oracle.py"
    return {"analysis_module": sha256_file(analysis_path), "cli_module": sha256_file(cli_path)}


def builder_git_identity() -> dict[str, Any]:
    """Require an addressable clean commit for the replay implementation."""
    try:
        identity = git_identity(root=repository_root_from_source())
    except Exception as exc:  # pragma: no cover - delegated helper detail
        raise FrozenOracleError("frozen-oracle builder requires a readable clean Git identity") from exc
    if identity.get("dirty"):
        raise FrozenOracleError("frozen-oracle builder requires a clean Git worktree")
    sha = _hex(identity.get("sha"), name="frozen-oracle builder Git SHA", length=40)
    return {"sha": sha, "dirty": False}


def replay_robust_correctness(
    *,
    source_config: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    attack_seed_base: int,
) -> dict[str, Any]:
    """Read-only raw-train PGD replay for one stateless RSLAD checkpoint."""
    if batch_size < 1 or attack_seed_base < 0:
        raise FrozenOracleError("frozen-oracle replay batch size and attack seed base must be non-negative/positive")
    if device.type != "cuda":
        raise FrozenOracleError("frozen-oracle replay is restricted to explicit CUDA execution")
    if (
        source_config.protocol.id != "controlled_cifar10_r18_v1"
        or source_config.method.id != "rslad"
        or source_config.teacher is None
    ):
        raise FrozenOracleError("frozen-oracle replay requires a baseline RSLAD source and frozen teacher")
    attack_config = source_config.method.attack
    expected_attack = AttackConfig(loss="kl", kl_target="teacher_clean")
    if attack_config.identity() != expected_attack.identity():
        raise FrozenOracleError("frozen-oracle replay requires the complete controlled KL PGD-10 attack identity")
    expected_hash = config_digest(resolved_config_dict(source_config))
    try:
        payload = validate_checkpoint_lineage(checkpoint, expected_config_hash=expected_hash)
    except (OSError, ValueError) as exc:
        raise FrozenOracleError("source replay checkpoint does not satisfy strict lineage") from exc
    if not isinstance(payload, Mapping):  # defensive after strict helper
        raise FrozenOracleError("source replay checkpoint is unreadable")
    checkpoint_sha = sha256_file(checkpoint)
    raw = build_dataset(source_config.dataset)
    train, _ = stratified_train_validation_split(
        raw, validation_fraction=source_config.training.validation_fraction, seed=source_config.seeds.split
    )
    loader = DataLoader(
        train,
        batch_size=batch_size,
        sampler=EpochShuffleSampler(len(train), seed=source_config.seeds.data_order, shuffle=False),
        num_workers=source_config.training.num_workers,
        collate_fn=collate_indexed,
    )
    student = build_student(source_config.student, tier=source_config.tier).to(device).eval()
    try:
        load_saved_student_checkpoint(checkpoint, student)
    except (OSError, ValueError) as exc:
        raise FrozenOracleError("source replay checkpoint cannot strictly load the student") from exc
    teacher = build_teacher(source_config.teacher, tier=source_config.tier).to(device).eval()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise FrozenOracleError("frozen-oracle replay teacher parameters must be frozen")
    attack = LinfPGD(attack_config)
    correctness: dict[int, bool] = {}
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        if not isinstance(raw_batch, IndexedBatch):
            raise FrozenOracleError("frozen-oracle replay loader must emit IndexedBatch")
        batch = raw_batch.to(device)
        generator = torch.Generator(device=device).manual_seed(attack_seed_base + 1_000_003 * batch_index)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            teacher_logits = teacher(batch.images.float()).detach().float()
        result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_logits,
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, result.max_abs_delta)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            predictions = student(result.adversarial.float()).argmax(dim=1)
        for sample_id, correct in zip(batch.sample_ids.tolist(), predictions.eq(batch.labels).tolist(), strict=True):
            if sample_id in correctness:
                raise FrozenOracleError("frozen-oracle replay produced duplicate source IDs")
            correctness[int(sample_id)] = bool(correct)
        student.zero_grad(set_to_none=True)
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise FrozenOracleError("frozen-oracle replay populated a teacher parameter gradient")
    epsilon = attack_config.epsilon_value
    assert epsilon is not None
    if max_abs_delta > epsilon + 1e-7:
        raise FrozenOracleError("frozen-oracle replay violated the pixel-space Linf bound")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "epoch": payload.get("epoch"),
        "tracker_run_id": payload.get("tracker_run_id"),
        "config_sha256": payload.get("config_hash"),
        "correctness": correctness,
        "correctness_sha256": hashlib.sha256(
            canonical_json([[sample_id, correctness[sample_id]] for sample_id in sorted(correctness)]).encode()
        ).hexdigest(),
        "replay_protocol": {
            "input_view": "raw_unaugmented_train_partition",
            "attack_seed_base": attack_seed_base,
            "seed_formula": "attack_seed_base + 1000003 * batch_index",
            "batch_size": batch_size,
            "device_type": device.type,
        },
        "max_abs_delta": max_abs_delta,
    }


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(int(row["class_id"]) for row in rows if row["risk"] == 1)
    return {str(class_id): counter[class_id] for class_id in sorted(counter)}


def train_namespace_fingerprint(labels: Mapping[int, int]) -> str:
    """Portable identity of the exact eligible source-ID/label namespace."""
    payload = {"namespace": "train", "labels": [[sample_id, labels[sample_id]] for sample_id in sorted(labels)]}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def validate_wandb_checkpoint_inventory(
    inventory: Mapping[str, Any],
    *,
    historical_checkpoint: Path,
    final_checkpoint: Path,
    source_config_hash: str,
    source_run_id: str,
    source_scientific_git_sha: str,
) -> dict[str, Any]:
    """Bind the two local checkpoint byte streams to audited W&B versions."""
    if inventory.get("schema_version") != 1:
        raise FrozenOracleError("W&B checkpoint inventory requires schema_version=1")
    expected_artifact_name = f"model-{source_run_id}-last"
    if (
        inventory.get("run_id") != source_run_id
        or inventory.get("artifact_name") != expected_artifact_name
        or inventory.get("config_sha256") != source_config_hash
        or inventory.get("scientific_git_sha") != source_scientific_git_sha
    ):
        raise FrozenOracleError("W&B checkpoint inventory source identity does not match the source run")
    checkpoints = inventory.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise FrozenOracleError("W&B checkpoint inventory has no checkpoint mapping")
    canonical: dict[str, Any] = {
        "run_id": source_run_id,
        "artifact_name": expected_artifact_name,
        "config_sha256": source_config_hash,
        "scientific_git_sha": source_scientific_git_sha,
        "checkpoints": {},
    }
    for name, path, epoch, version in (
        ("historical", historical_checkpoint, 99, "v19"),
        ("final", final_checkpoint, 199, "v39"),
    ):
        row = checkpoints.get(name)
        if not isinstance(row, Mapping):
            raise FrozenOracleError(f"W&B checkpoint inventory lacks {name}")
        if (
            row.get("epoch") != epoch
            or row.get("version") != version
            or row.get("file_name") != "last.pt"
            or row.get("size") != path.stat().st_size
            or row.get("file_md5") != md5_base64_file(path)
            or row.get("checkpoint_sha256") != sha256_file(path)
        ):
            raise FrozenOracleError(f"W&B checkpoint inventory {name} bytes/version do not match local checkpoint")
        _hex(row.get("artifact_digest"), name=f"W&B {name} artifact digest", length=32)
        canonical["checkpoints"][name] = {
            key: row[key]
            for key in (
                "epoch",
                "version",
                "artifact_digest",
                "file_name",
                "file_md5",
                "size",
                "checkpoint_sha256",
            )
        }
    return canonical


def build_frozen_oracle_manifests(
    *,
    source_config: ExperimentConfig,
    source_manifest: Path,
    historical_replay: Mapping[str, Any],
    final_replay: Mapping[str, Any],
    labels: Mapping[int, int],
    builder_git: Mapping[str, Any],
    wandb_checkpoint_inventory: Mapping[str, Any],
    control_seeds: Sequence[int] = (101, 202, 303),
) -> dict[str, dict[str, Any]]:
    """Construct oracle plus deterministic class-matched controls in memory."""
    if source_config.protocol.id != "controlled_cifar10_r18_v1" or source_config.method.id != "rslad":
        raise FrozenOracleError("frozen oracle source must be controlled_cifar10_r18_v1 baseline RSLAD")
    if source_config.teacher is None or source_config.teacher.registry_id != "bartoldson2024_adversarial_wrn94_16":
        raise FrozenOracleError("frozen oracle source must use the registered Bartoldson teacher")
    if source_config.teacher.checkpoint_sha256 is None:
        raise FrozenOracleError("frozen oracle source teacher must have an exact checkpoint SHA")
    expected_attack = AttackConfig(loss="kl", kl_target="teacher_clean")
    if source_config.method.attack.identity() != expected_attack.identity():
        raise FrozenOracleError("frozen oracle source attack must match the complete controlled KL PGD-10 identity")
    valid_control_seeds = len(control_seeds) == 3 and len(set(control_seeds)) == 3
    if not valid_control_seeds or any(not isinstance(seed, int) for seed in control_seeds):
        raise FrozenOracleError("frozen oracle requires exactly three distinct integer class-matched control seeds")
    if builder_git.get("dirty") is not False:
        raise FrozenOracleError("frozen-oracle builder Git identity must be clean")
    builder_sha = _hex(builder_git.get("sha"), name="frozen-oracle builder Git SHA", length=40)
    source_resolved = resolved_config_dict(source_config)
    source_hash = config_digest(source_resolved)

    def validate_replay(replay: Mapping[str, Any], *, epoch: int, name: str) -> tuple[dict[int, bool], dict[str, Any]]:
        if replay.get("epoch") != epoch or replay.get("config_sha256") != source_hash:
            raise FrozenOracleError(f"{name} replay does not bind the expected source checkpoint/config")
        if replay.get("tracker_run_id") is None or not isinstance(replay.get("tracker_run_id"), str):
            raise FrozenOracleError(f"{name} replay lacks source run lineage")
        _hex(replay.get("checkpoint_sha256"), name=f"{name} replay checkpoint SHA")
        _hex(replay.get("correctness_sha256"), name=f"{name} replay correctness SHA")
        correctness = replay.get("correctness")
        protocol = replay.get("replay_protocol")
        if not isinstance(correctness, Mapping) or not isinstance(protocol, Mapping):
            raise FrozenOracleError(f"{name} replay lacks correctness/protocol")
        output: dict[int, bool] = {}
        for sample_id, value in correctness.items():
            if isinstance(sample_id, bool) or not isinstance(sample_id, int) or not isinstance(value, bool):
                raise FrozenOracleError(f"{name} replay correctness mapping is invalid")
            output[sample_id] = value
        correctness_bytes = canonical_json([[item, output[item]] for item in sorted(output)]).encode()
        recomputed = hashlib.sha256(correctness_bytes).hexdigest()
        if recomputed != replay["correctness_sha256"]:
            raise FrozenOracleError(f"{name} replay correctness SHA does not match its rows")
        maximum = replay.get("max_abs_delta")
        if not isinstance(maximum, (int, float)) or maximum < 0:
            raise FrozenOracleError(f"{name} replay max Linf delta is invalid")
        epsilon = source_config.method.attack.epsilon_value
        assert epsilon is not None
        if maximum > epsilon + 1e-7:
            raise FrozenOracleError(f"{name} replay exceeds the configured pixel-space Linf bound")
        return output, dict(protocol)

    historical_correct, historical_protocol = validate_replay(historical_replay, epoch=99, name="historical")
    final_correct, final_protocol = validate_replay(final_replay, epoch=199, name="final")
    if historical_replay["tracker_run_id"] != final_replay["tracker_run_id"]:
        raise FrozenOracleError("source replays do not have a common tracking run ID")
    if historical_protocol != final_protocol:
        raise FrozenOracleError("historical/final replay protocols must be exactly identical")
    lineage = _source_manifest(
        source_manifest, source_config_hash=source_hash, source_teacher_sha256=source_config.teacher.checkpoint_sha256
    )
    if lineage["run_id"] != historical_replay["tracker_run_id"]:
        raise FrozenOracleError("source manifest run ID does not match source replays")
    if wandb_checkpoint_inventory.get("run_id") != lineage["run_id"]:
        raise FrozenOracleError("validated W&B checkpoint inventory run ID does not match source lineage")
    inventory_checkpoints = wandb_checkpoint_inventory.get("checkpoints")
    if not isinstance(inventory_checkpoints, Mapping):
        raise FrozenOracleError("validated W&B checkpoint inventory is incomplete")
    for name, replay in (("historical", historical_replay), ("final", final_replay)):
        row = inventory_checkpoints.get(name)
        if not isinstance(row, Mapping) or row.get("checkpoint_sha256") != replay.get("checkpoint_sha256"):
            raise FrozenOracleError("W&B checkpoint inventory SHA does not match replayed bytes")
    # The preregistered oracle is final robust error: both future forgetting
    # and persistent failure.  Transition remains a reporting-only field.
    selected = {sample_id for sample_id, correct in final_correct.items() if not correct}
    oracle_rows = _build_rows(
        labels=labels, historical_correct=historical_correct, final_correct=final_correct, selected_ids=selected
    )
    selected_counts = _class_counts(oracle_rows)
    common = {
        "schema_version": 1,
        "kind": "frozen_oracle_mask",
        "namespace": "train",
        "method_id": "rslad_frozen_oracle_softening",
        "risk_value": 1.0,
        "dataset": {
            "name": source_config.dataset.name,
            "split": source_config.dataset.split,
            "num_classes": source_config.dataset.num_classes,
            "train_sample_count": len(labels),
            "validation_fraction": source_config.training.validation_fraction,
            "split_seed": source_config.seeds.split,
            "train_namespace_sha256": train_namespace_fingerprint(labels),
        },
        "source": {
            **lineage,
            "config_sha256": source_hash,
            "teacher_checkpoint_sha256": source_config.teacher.checkpoint_sha256,
            "attack_identity": source_config.method.attack.identity(),
            "replay_source_files": replay_source_hashes(),
            "builder_git": {"sha": builder_sha, "dirty": False},
            "wandb_checkpoint_inventory": dict(wandb_checkpoint_inventory),
            "replays": {
                "historical": {
                    key: historical_replay[key]
                    for key in ("epoch", "checkpoint_sha256", "correctness_sha256", "replay_protocol", "max_abs_delta")
                },
                "final": {
                    key: final_replay[key]
                    for key in ("epoch", "checkpoint_sha256", "correctness_sha256", "replay_protocol", "max_abs_delta")
                },
            },
        },
        "selected_class_counts": selected_counts,
        "selected_count": len(selected),
    }
    outputs: dict[str, dict[str, Any]] = {
        "oracle": {**common, "assignment": {"kind": "oracle_final_robust_error"}, "rows": oracle_rows}
    }
    available_by_class: dict[int, list[int]] = {}
    for sample_id, class_id in labels.items():
        available_by_class.setdefault(class_id, []).append(sample_id)
    for index, seed in enumerate(control_seeds, start=1):
        control_selected: set[int] = set()
        generator = random.Random(seed)
        for class_id, count in sorted((int(key), value) for key, value in selected_counts.items()):
            candidates = sorted(available_by_class.get(class_id, ()))
            if count > len(candidates):
                raise FrozenOracleError("class-matched control count exceeds source train class population")
            control_selected.update(generator.sample(candidates, count))
        rows = _build_rows(
            labels=labels,
            historical_correct=historical_correct,
            final_correct=final_correct,
            selected_ids=control_selected,
        )
        outputs[f"control-{index}"] = {
            **common,
            "assignment": {"kind": "class_matched_random", "index": index, "seed": seed},
            "rows": rows,
        }
    return outputs


def write_frozen_oracle_manifests(output_dir: Path, manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Atomically write four new manifests and return their exact file SHA-256s."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FrozenOracleError("refusing to overwrite a frozen-oracle output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    try:
        for name, payload in sorted(manifests.items()):
            path = output_dir / f"{name}.json"
            if path.exists():
                raise FrozenOracleError("refusing to overwrite a frozen-oracle manifest")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            temporary.replace(path)
            outputs[name] = sha256_file(path)
    except Exception:
        # Do not report a partial collection as a valid generated set.
        for path in output_dir.glob("*.json"):
            path.unlink()
        raise
    return outputs


def load_frozen_risk_lookup(
    path: Path,
    *,
    expected_sha256: str,
    expected_dataset_name: str,
    expected_num_classes: int,
    expected_train_labels: Mapping[int, int],
    expected_attack_identity: Mapping[str, Any],
    expected_teacher_checkpoint_sha256: str,
) -> FrozenRiskLookup:
    """Validate a full train-namespace manifest before any training starts."""
    _hex(expected_sha256, name="configured frozen-oracle manifest SHA")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise FrozenOracleError("frozen-oracle manifest bytes do not match the configured SHA-256")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenOracleError("frozen-oracle manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise FrozenOracleError("frozen-oracle manifest must be a mapping")
    if payload.get("schema_version") != 1 or payload.get("kind") != "frozen_oracle_mask":
        raise FrozenOracleError("frozen-oracle manifest schema/kind is invalid")
    if payload.get("method_id") != "rslad_frozen_oracle_softening" or payload.get("namespace") != "train":
        raise FrozenOracleError("frozen-oracle manifest is not a train-only method input")
    if payload.get("risk_value") != 1.0:
        raise FrozenOracleError("frozen-oracle manifest must fix selected binary risk to exactly 1.0")
    dataset = payload.get("dataset")
    source = payload.get("source")
    rows = payload.get("rows")
    if not isinstance(dataset, Mapping) or not isinstance(source, Mapping) or not isinstance(rows, list):
        raise FrozenOracleError("frozen-oracle manifest lacks required lineage/dataset/rows")
    if dataset.get("name") != expected_dataset_name or dataset.get("split") != "train":
        raise FrozenOracleError("frozen-oracle manifest dataset is not the active official training split")
    if dataset.get("num_classes") != expected_num_classes or dataset.get("train_sample_count") != len(
        expected_train_labels
    ):
        raise FrozenOracleError("frozen-oracle manifest class/count identity differs from active training data")
    if dataset.get("train_namespace_sha256") != train_namespace_fingerprint(expected_train_labels):
        raise FrozenOracleError("frozen-oracle manifest train namespace fingerprint differs from active data")
    _hex(source.get("config_sha256"), name="frozen-oracle source config SHA")
    _hex(source.get("teacher_checkpoint_sha256"), name="frozen-oracle source teacher SHA")
    _hex(source.get("source_manifest_sha256"), name="frozen-oracle source manifest SHA")
    _hex(source.get("scientific_git_sha"), name="frozen-oracle scientific Git SHA", length=40)
    attack = source.get("attack_identity")
    replays = source.get("replays")
    replay_files = source.get("replay_source_files")
    builder_git = source.get("builder_git")
    wandb_inventory = source.get("wandb_checkpoint_inventory")
    if (
        not isinstance(attack, Mapping)
        or not isinstance(replays, Mapping)
        or not isinstance(replay_files, Mapping)
        or not isinstance(builder_git, Mapping)
        or not isinstance(wandb_inventory, Mapping)
    ):
        raise FrozenOracleError("frozen-oracle source attack/replay lineage is incomplete")
    if builder_git.get("dirty") is not False:
        raise FrozenOracleError("frozen-oracle builder lineage is not clean")
    _hex(builder_git.get("sha"), name="frozen-oracle builder Git SHA", length=40)
    if canonical_json(attack) != canonical_json(expected_attack_identity):
        raise FrozenOracleError("frozen-oracle source attack identity differs from active training attack")
    if source.get("teacher_checkpoint_sha256") != expected_teacher_checkpoint_sha256:
        raise FrozenOracleError("frozen-oracle source teacher checkpoint differs from active training teacher")
    for value in replay_files.values():
        _hex(value, name="frozen-oracle replay source file SHA")
    if wandb_inventory.get("run_id") != source.get("run_id"):
        raise FrozenOracleError("frozen-oracle W&B inventory run ID differs from source lineage")
    wandb_checkpoints = wandb_inventory.get("checkpoints")
    if not isinstance(wandb_checkpoints, Mapping):
        raise FrozenOracleError("frozen-oracle W&B checkpoint inventory is incomplete")
    for name, epoch in (("historical", 99), ("final", 199)):
        checkpoint = replays.get(name)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("epoch") != epoch:
            raise FrozenOracleError("frozen-oracle source checkpoint epoch lineage is invalid")
        _hex(checkpoint.get("checkpoint_sha256"), name=f"frozen-oracle {name} checkpoint SHA")
        _hex(checkpoint.get("correctness_sha256"), name=f"frozen-oracle {name} correctness SHA")
        if not isinstance(checkpoint.get("replay_protocol"), Mapping):
            raise FrozenOracleError("frozen-oracle replay protocol lineage is invalid")
        wandb_checkpoint = wandb_checkpoints.get(name)
        if (
            not isinstance(wandb_checkpoint, Mapping)
            or wandb_checkpoint.get("epoch") != epoch
            or wandb_checkpoint.get("version") != ("v19" if name == "historical" else "v39")
            or wandb_checkpoint.get("checkpoint_sha256") != checkpoint.get("checkpoint_sha256")
        ):
            raise FrozenOracleError("frozen-oracle W&B checkpoint inventory differs from replay lineage")
    risks: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("namespace") != "train":
            raise FrozenOracleError("frozen-oracle rows may only name the training namespace")
        sample_id, class_id, risk = row.get("sample_id"), row.get("class_id"), row.get("risk")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id in risks:
            raise FrozenOracleError("frozen-oracle rows have duplicate/invalid stable sample IDs")
        if expected_train_labels.get(sample_id) != class_id:
            raise FrozenOracleError("frozen-oracle row ID/class does not match active training partition")
        if risk not in {0, 1}:
            raise FrozenOracleError("frozen-oracle risk must be binary 0 or 1")
        risks[sample_id] = int(risk)
    if set(risks) != set(expected_train_labels):
        raise FrozenOracleError("frozen-oracle manifest has missing or foreign training sample IDs")
    selected_count = sum(risks.values())
    expected_counts = Counter(expected_train_labels[item] for item, risk in risks.items() if risk)
    expected_serialized_counts = {str(key): value for key, value in sorted(expected_counts.items())}
    if (
        payload.get("selected_count") != selected_count
        or payload.get("selected_class_counts") != expected_serialized_counts
    ):
        raise FrozenOracleError("frozen-oracle selected class counts are inconsistent with binary rows")
    return FrozenRiskLookup(risks=risks, manifest_sha256=expected_sha256, source=source)
