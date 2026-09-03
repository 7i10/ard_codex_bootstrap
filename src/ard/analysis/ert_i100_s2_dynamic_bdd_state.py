"""Read-only CE-PGD20 state replay for the I100 dynamic-BDD screen.

The production endpoint rows retain Student outcomes but not the adversarial
images or Teacher outputs.  This module reconstructs those *diagnostics* from
saved checkpoints without changing training, selection, or checkpoint state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.ert_stage_a_endpoint import _probability_margin, split_identity
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.tracking.adapter import collect_git_state


class DynamicBDDStateReplayError(RuntimeError):
    """The immutable state-replay contract was not satisfied."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches_sha256(path: str | Path | None, expected_sha256: str) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    return candidate.is_file() and _sha256(candidate) == expected_sha256


def _q10_fragile_ids(
    rows: Mapping[int, Mapping[str, Any]], *, correct_key: str, margin_key: str
) -> tuple[set[int], float]:
    """Return the stable-ID tie-broken lower decile of correct samples."""
    positive = sorted(
        (sample_id for sample_id, row in rows.items() if bool(row[correct_key])),
        key=lambda sample_id: (float(rows[sample_id][margin_key]), sample_id),
    )
    if not positive:
        raise DynamicBDDStateReplayError(f"no adversarial-correct rows for {correct_key}")
    count = math.ceil(0.10 * len(positive))
    selected = set(positive[:count])
    boundary_id = max(selected, key=lambda sample_id: (float(rows[sample_id][margin_key]), sample_id))
    return selected, float(rows[boundary_id][margin_key])


def canonical_state_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify all rows into current S1/S2/S3 and T1/T2/T3 states.

    The q10 fragile subsets are recomputed on the current checkpoint's full
    split, exactly as in the registered canonical-state construction.
    """
    materialized = [dict(row) for row in rows]
    by_id = {int(row["sample_id"]): row for row in materialized}
    if not by_id:
        raise DynamicBDDStateReplayError("state replay produced no rows")
    if len(by_id) != len(materialized):
        raise DynamicBDDStateReplayError("duplicate stable IDs in state replay")
    student_s2, student_floor = _q10_fragile_ids(
        by_id, correct_key="student_ce20_adv_correct", margin_key="student_ce20_adv_margin"
    )
    teacher_t2, teacher_floor = _q10_fragile_ids(
        by_id, correct_key="teacher_ce20_adv_correct", margin_key="teacher_ce20_adv_margin"
    )
    state_by_id: dict[int, dict[str, str]] = {}
    counts = {f"S{student}xT{teacher}": 0 for student in range(1, 4) for teacher in range(1, 4)}
    for sample_id, row in by_id.items():
        student = "S3"
        if bool(row["student_ce20_adv_correct"]):
            student = "S2" if sample_id in student_s2 else "S1"
        teacher = "T3"
        if bool(row["teacher_ce20_adv_correct"]):
            teacher = "T2" if sample_id in teacher_t2 else "T1"
        state_by_id[sample_id] = {"student": student, "teacher": teacher, "joint": f"{student}x{teacher}"}
        counts[f"{student}x{teacher}"] += 1
    count = len(by_id)
    return {
        "row_count": count,
        "student_q10_floor": student_floor,
        "teacher_q10_floor": teacher_floor,
        "joint_counts": counts,
        "joint_fractions": {key: value / count for key, value in counts.items()},
        "state_by_id": state_by_id,
    }


def replay_train_states(
    *,
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    expected_epoch: int,
    expected_checkpoint_sha256: str | None = None,
    expected_teacher_sha256: str | None = None,
) -> dict[str, Any]:
    """Generate CE-PGD20 Student/Teacher state rows on the fixed train split."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DynamicBDDStateReplayError(f"refusing to overwrite state replay output: {output_dir}")
    if expected_checkpoint_sha256 is not None and _sha256(checkpoint) != expected_checkpoint_sha256:
        raise DynamicBDDStateReplayError("checkpoint SHA-256 differs from frozen input")
    config = load_config(config_path)
    attack_config = config.method.selection_attack
    if attack_config is None or attack_config.loss != "ce" or attack_config.steps != 20:
        raise DynamicBDDStateReplayError("state replay requires the configured CE-PGD20 selection attack")
    if attack_config.epsilon != "8/255" or attack_config.step_size != "2/255" or not attack_config.random_start:
        raise DynamicBDDStateReplayError("state replay attack differs from the frozen CE-PGD20 contract")
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False:
        raise DynamicBDDStateReplayError("state replay requires a clean analysis source tree")

    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, student)
    if payload.get("epoch") != expected_epoch or payload.get("epoch_boundary") != "end":
        raise DynamicBDDStateReplayError(f"checkpoint is not epoch-{expected_epoch} end state")
    teacher = build_teacher(config.teacher, tier=config.tier)
    teacher_source = getattr(config.teacher, "checkpoint", None)
    if expected_teacher_sha256 is not None:
        if not _matches_sha256(teacher_source, expected_teacher_sha256):
            raise DynamicBDDStateReplayError("Teacher checkpoint SHA-256 differs from frozen input")

    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    identity = split_identity(train_dataset, split="train")
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_rank_batch_size,
        sampler=EpochShuffleSampler(len(train_dataset), seed=0, rank=0, world_size=1, shuffle=False),
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
    )
    student.to(device).eval()
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    attack = LinfPGD(attack_config)
    generator = torch.Generator(device=device).manual_seed(config.seeds.evaluation_attack)
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            student_clean = student(batch.images)
            teacher_clean = teacher(batch.images)
        adversarial = attack.generate(
            AttackRequest(inputs=batch.images, labels=batch.labels, student=student, teacher=None, generator=generator)
        ).adversarial
        with torch.no_grad():
            student_adv = student(adversarial)
            teacher_adv = teacher(adversarial)
        student_clean_margin = _probability_margin(student_clean, batch.labels)
        student_adv_margin = _probability_margin(student_adv, batch.labels)
        teacher_clean_margin = _probability_margin(teacher_clean, batch.labels)
        teacher_adv_margin = _probability_margin(teacher_adv, batch.labels)
        student_clean_probability = (
            torch.softmax(student_clean.float(), dim=1).gather(1, batch.labels[:, None]).squeeze(1)
        )
        student_adv_probability = torch.softmax(student_adv.float(), dim=1).gather(1, batch.labels[:, None]).squeeze(1)
        teacher_clean_probability = (
            torch.softmax(teacher_clean.float(), dim=1).gather(1, batch.labels[:, None]).squeeze(1)
        )
        teacher_adv_probability = torch.softmax(teacher_adv.float(), dim=1).gather(1, batch.labels[:, None]).squeeze(1)
        for index in range(batch.labels.shape[0]):
            label = batch.labels[index]
            rows.append(
                {
                    "sample_id": int(batch.sample_ids[index]),
                    "class_id": int(label),
                    "epoch": expected_epoch,
                    "student_clean_correct": bool(student_clean[index].argmax() == label),
                    "student_clean_probability": float(student_clean_probability[index]),
                    "student_clean_margin": float(student_clean_margin[index]),
                    "student_ce20_adv_correct": bool(student_adv[index].argmax() == label),
                    "student_ce20_adv_probability": float(student_adv_probability[index]),
                    "student_ce20_adv_margin": float(student_adv_margin[index]),
                    "teacher_clean_correct": bool(teacher_clean[index].argmax() == label),
                    "teacher_clean_probability": float(teacher_clean_probability[index]),
                    "teacher_clean_margin": float(teacher_clean_margin[index]),
                    "teacher_ce20_adv_correct": bool(teacher_adv[index].argmax() == label),
                    "teacher_ce20_adv_probability": float(teacher_adv_probability[index]),
                    "teacher_ce20_adv_margin": float(teacher_adv_margin[index]),
                }
            )
    if len(rows) != identity["count"]:
        raise DynamicBDDStateReplayError("state replay row count differs from the fixed train split")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "state-rows.parquet"
    write_sample_parquet(rows, rows_path)
    summary = canonical_state_summary(rows)
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_dynamic_bdd_state_replay_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": expected_epoch,
        "teacher_checkpoint_sha256": expected_teacher_sha256,
        "attack": attack_config.identity(),
        "attack_identity_sha256": attack_config.identity_sha256(),
        "split_identity": identity,
        "source_git_sha": source["sha"],
        "row_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": _sha256(rows_path),
        "state_summary": {key: value for key, value in summary.items() if key != "state_by_id"},
    }
    (output_dir / "state-replay.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
