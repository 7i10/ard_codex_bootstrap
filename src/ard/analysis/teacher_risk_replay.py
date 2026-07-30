"""Deterministic, read-only historical teacher-risk replay."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ard.attacks import AttackRequest, LinfPGD
from ard.config import ExperimentConfig
from ard.config.loader import resolved_config_dict
from ard.config.schema import training_execution_identity
from ard.data import (
    EpochShuffleSampler,
    IndexedBatch,
    build_dataset,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.engine.distributed import unwrap_model
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint, validate_checkpoint_lineage
from ard.models import build_student, build_teacher

from .signal_audit import (
    CheckpointInventory,
    SignalAuditError,
    _sha256_mapping,
    canonical_json,
    logical_dataset_fingerprint,
    logical_dataset_identity,
    replay_protocol,
    sha256_file,
)


class TeacherRiskReplayError(SignalAuditError):
    """Raised when a historical replay cannot prove its scientific identity."""


@dataclass(frozen=True)
class ReplayResult:
    rows: tuple[dict[str, Any], ...]
    max_abs_delta: float
    attack_seed_base: int


def replay_source_hashes() -> dict[str, str]:
    """Hash the analysis implementation and its executable entry point."""
    analysis_path = Path(__file__).resolve()
    cli_path = analysis_path.parents[1] / "cli" / "replay_teacher_risk.py"
    return {
        "analysis_module": sha256_file(analysis_path),
        "cli_module": sha256_file(cli_path),
    }


def repository_root_from_source() -> Path:
    """Return the repository containing this installed source tree, never the caller cwd."""
    return Path(__file__).resolve().parents[3]


def git_identity(*, root: Path) -> dict[str, Any]:
    """Read repository identity without changing the worktree."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TeacherRiskReplayError("replay requires a readable Git identity") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise TeacherRiskReplayError("replay Git SHA must be an exact lowercase commit SHA")
    return {"sha": sha, "dirty": dirty}


def build_replay_loader(config: ExperimentConfig, *, batch_size: int) -> DataLoader[IndexedBatch]:
    """Rebuild the raw, unaugmented training partition with source IDs intact."""
    if config.dataset.name not in {"cifar10", "cifar100"} or config.dataset.split != "train":
        raise TeacherRiskReplayError("teacher-risk replay is restricted to the raw CIFAR official training split")
    if batch_size < 1:
        raise TeacherRiskReplayError("replay batch_size must be positive")
    raw_view = build_dataset(config.dataset)
    train_view, _ = stratified_train_validation_split(
        raw_view,
        validation_fraction=config.training.validation_fraction,
        seed=config.seeds.split,
    )
    sampler = EpochShuffleSampler(len(train_view), seed=config.seeds.data_order, shuffle=False)
    return DataLoader(
        train_view,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
    )


def load_historical_student(
    checkpoint: CheckpointInventory, *, config: ExperimentConfig, device: torch.device
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Strictly load the exact periodic checkpoint selected by the audit."""
    path = Path(checkpoint.path)
    if sha256_file(path) != checkpoint.sha256:
        raise TeacherRiskReplayError("selected replay checkpoint SHA no longer matches its immutable inventory")
    resolved = resolved_config_dict(config)
    expected_hash = config_digest(resolved)
    try:
        payload = validate_checkpoint_lineage(path, expected_config_hash=expected_hash)
    except (OSError, ValueError) as exc:
        raise TeacherRiskReplayError("selected replay checkpoint is not a complete training checkpoint") from exc
    if REQUIRED_KEYS.difference(payload) or checkpoint.config_hash != expected_hash:
        raise TeacherRiskReplayError(
            "selected replay checkpoint config hash does not match strict resolved training config"
        )
    if payload.get("tracker_run_id") != checkpoint.run_id or payload.get("epoch") != checkpoint.epoch:
        raise TeacherRiskReplayError("selected replay checkpoint run ID or epoch does not match inventory")
    student = build_student(config.student, tier=config.tier)
    load_saved_student_checkpoint(path, unwrap_model(student))
    return student.to(device).eval(), payload


def _teacher_risk(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.detach().float()
    probabilities = F.softmax(logits, dim=1)
    entropy = -(probabilities * F.log_softmax(logits, dim=1)).sum(dim=1)
    risk = 1.0 - entropy / math.log(logits.shape[1])
    if not torch.isfinite(entropy).all() or not torch.isfinite(risk).all():
        raise TeacherRiskReplayError("teacher entropy/risk must be finite FP32 values")
    if bool((risk < -1e-6).any()) or bool((risk > 1.0 + 1e-6).any()):
        raise TeacherRiskReplayError("teacher risk is outside the Shannon-entropy [0, 1] range")
    return entropy, risk


def replay_rows(
    *,
    student: nn.Module,
    teacher: nn.Module,
    loader: Iterable[IndexedBatch],
    attack: LinfPGD,
    device: torch.device,
    attack_seed_base: int,
) -> ReplayResult:
    """Replay one fixed student checkpoint without optimizer, state, or tracking mutation."""
    if attack.config.loss != "kl" or attack.config.kl_target != "teacher_clean" or attack.config.steps != 10:
        raise TeacherRiskReplayError("teacher-risk replay requires the exact KL PGD-10 teacher_clean attack")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise TeacherRiskReplayError("teacher-risk replay requires frozen teacher parameters")
    teacher.eval()
    student.eval()
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        if not isinstance(raw_batch, IndexedBatch):
            raise TeacherRiskReplayError("teacher-risk replay loader must emit IndexedBatch values")
        batch = raw_batch.to(device)
        generator = torch.Generator(device=device).manual_seed(attack_seed_base + 1_000_003 * batch_index)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            teacher_clean_logits = teacher(batch.images.float()).detach().float()
        result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits,
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, result.max_abs_delta)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            logits = teacher(result.adversarial.float()).detach().float()
            entropy, risk = _teacher_risk(logits)
            prediction = logits.argmax(dim=1)
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise TeacherRiskReplayError("teacher-risk replay populated a teacher parameter gradient")
        for sample_id, class_id, entropy_value, risk_value, predicted in zip(
            batch.sample_ids.tolist(),
            batch.labels.tolist(),
            entropy.tolist(),
            risk.tolist(),
            prediction.tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": int(sample_id),
                    "class_id": int(class_id),
                    "teacher_entropy": float(entropy_value),
                    "teacher_risk": float(risk_value),
                    "teacher_prediction": int(predicted),
                    "teacher_correct": bool(predicted == class_id),
                }
            )
        student.zero_grad(set_to_none=True)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise TeacherRiskReplayError("teacher-risk replay produced duplicate stable source sample IDs")
    epsilon = attack.config.epsilon_value
    assert epsilon is not None
    if max_abs_delta > epsilon + 1e-7:
        raise TeacherRiskReplayError("teacher-risk replay violated the configured pixel-space Linf bound")
    return ReplayResult(rows=tuple(rows), max_abs_delta=max_abs_delta, attack_seed_base=attack_seed_base)


def replay_envelope(
    *,
    audit_config: Mapping[str, Any],
    training_config: ExperimentConfig,
    historical: CheckpointInventory,
    device: torch.device,
    batch_size: int,
    repository_root: Path,
) -> dict[str, Any]:
    """Create a complete provenance envelope from immutable local inputs only."""
    if training_config.teacher is None:
        raise TeacherRiskReplayError("teacher-risk replay requires a registered teacher")
    if historical.epoch != 99:
        raise TeacherRiskReplayError("formal teacher-risk replay is fixed to the selected epoch-99 periodic checkpoint")
    if audit_config.get("method_id") != training_config.method.id:
        raise TeacherRiskReplayError("analysis method_id does not match strict resolved training method")
    if audit_config.get("training_seed") != training_config.seeds.model_init:
        raise TeacherRiskReplayError("analysis training_seed does not match strict resolved training config")
    declared_teacher = audit_config.get("teacher")
    if (
        not isinstance(declared_teacher, Mapping)
        or declared_teacher.get("registry_id") != training_config.teacher.registry_id
    ):
        raise TeacherRiskReplayError("analysis teacher registry_id does not match strict resolved training teacher")
    resolved = resolved_config_dict(training_config)
    expected_count = audit_config.get("train_expected_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise TeacherRiskReplayError("analysis config requires a positive train_expected_count")
    configured_batch_size = audit_config.get("replay_batch_size")
    if configured_batch_size != batch_size:
        raise TeacherRiskReplayError("--batch-size must exactly match analysis replay_batch_size")
    configured_device_type = audit_config.get("replay_device_type")
    if configured_device_type != device.type:
        raise TeacherRiskReplayError("requested replay device.type must exactly match analysis replay_device_type")
    expected_fingerprint = logical_dataset_fingerprint(resolved, train_expected_count=expected_count)
    if audit_config.get("dataset_fingerprint") != expected_fingerprint:
        raise TeacherRiskReplayError("analysis dataset fingerprint does not match strict resolved training config")
    declared_attack = audit_config.get("threat_identity", audit_config.get("attack_identity"))
    if canonical_json(declared_attack) != canonical_json(training_config.method.attack.model_dump(mode="json")):
        raise TeacherRiskReplayError("analysis attack identity does not match the strict resolved training method")
    if training_config.method.attack.steps != 10:
        raise TeacherRiskReplayError("teacher-risk replay requires a training method with exact PGD-10")
    repository_identity = git_identity(root=repository_root)
    if repository_identity["dirty"]:
        raise TeacherRiskReplayError("teacher-risk replay requires a clean Git worktree")
    student, checkpoint_payload = load_historical_student(historical, config=training_config, device=device)
    teacher = build_teacher(training_config.teacher, tier=training_config.tier).to(device)
    loader = build_replay_loader(training_config, batch_size=batch_size)
    replay = replay_rows(
        student=student,
        teacher=teacher,
        loader=loader,
        attack=LinfPGD(training_config.method.attack),
        device=device,
        attack_seed_base=training_config.seeds.train_attack + 1_000_003 * int(checkpoint_payload["global_step"]),
    )
    if len(replay.rows) != expected_count:
        raise TeacherRiskReplayError("replay split cardinality does not match configured train population")
    source_hashes = replay_source_hashes()
    dataset_identity = logical_dataset_identity(resolved, train_expected_count=expected_count)["dataset"]
    protocol = replay_protocol(
        batch_size=batch_size,
        attack_seed_base=replay.attack_seed_base,
        device_type=device.type,
    )
    checkpoint_world_size = checkpoint_payload.get("world_size")
    if isinstance(checkpoint_world_size, bool) or not isinstance(checkpoint_world_size, int):
        raise TeacherRiskReplayError("selected replay checkpoint world_size is invalid")
    checkpoint_training = {
        "world_size": checkpoint_world_size,
        "execution_identity": training_execution_identity(
            training=training_config.training,
            world_size=checkpoint_world_size,
        ),
    }
    return {
        "run_id": historical.run_id,
        "historical_epoch": historical.epoch,
        "historical_checkpoint_sha256": historical.sha256,
        "teacher_checkpoint_sha256": teacher.metadata.checkpoint_sha256,
        "dataset_fingerprint": expected_fingerprint,
        "dataset_identity": dataset_identity,
        "attack_identity": training_config.method.attack.model_dump(mode="json"),
        "replay_protocol": protocol,
        "rows": list(replay.rows),
        "replay_output_sha256": _sha256_mapping(replay.rows),
        "replay_source_files": source_hashes,
        "replay_source_sha256": _sha256_mapping(source_hashes),
        "git": repository_identity,
        "max_abs_delta": replay.max_abs_delta,
        "checkpoint_training": checkpoint_training,
        "execution": {
            "device": str(device),
        },
    }
