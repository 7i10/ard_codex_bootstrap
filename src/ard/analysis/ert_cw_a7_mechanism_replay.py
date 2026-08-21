"""No-update replay primitives for the frozen ERT A5--A8 mechanism audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.ert_clean_wrong_broad_screen import fixed_clean_wrong_mask
from ard.analysis.ert_clean_wrong_subtypes import _probability_stats
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.tracking.adapter import collect_git_state


class A7MechanismReplayError(RuntimeError):
    """Raised when the frozen no-update replay contract is violated."""


PARENTS = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}

TARGET_MODES = ("fixed", "teacher_zero", "teacher_floor", "teacher_abstain")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _treatment(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    treatment = raw.get("treatment") if isinstance(raw, Mapping) else None
    if not isinstance(treatment, Mapping):
        raise A7MechanismReplayError(f"missing treatment contract: {config_path}")
    mode = treatment.get("margin_target_mode")
    if mode not in TARGET_MODES:
        raise A7MechanismReplayError(f"A5--A8 replay requires a frozen margin target mode, got {mode!r}")
    calibration = raw.get("calibration") if isinstance(raw, Mapping) else None
    pooled = calibration.get("pooled_positive_margin_quantiles") if isinstance(calibration, Mapping) else None
    if isinstance(pooled, Mapping):
        if treatment.get("margin_floor") is None:
            treatment = {**treatment, "margin_floor": pooled.get("q25")}
        if treatment.get("margin_cap") is None:
            treatment = {**treatment, "margin_cap": pooled.get("q75")}
    if not isinstance(treatment.get("margin_coefficient"), (int, float)):
        raise A7MechanismReplayError(f"incomplete margin calibration in {config_path}")
    if mode != "fixed" and not isinstance(treatment.get("margin_cap"), (int, float)):
        raise A7MechanismReplayError(f"teacher target lacks cap in {config_path}")
    if mode == "fixed" and not isinstance(treatment.get("margin_gamma"), (int, float)):
        raise A7MechanismReplayError("fixed target lacks gamma")
    if mode == "teacher_floor" and not isinstance(treatment.get("margin_floor"), (int, float)):
        raise A7MechanismReplayError("floor target lacks floor")
    return {str(key): value for key, value in treatment.items()}


def _runtime_config(config_path: Path):
    """Load the immutable parent config behind a resolved arm overlay."""
    try:
        return load_config(config_path)
    except Exception as exc:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        parent = raw.get("parent_config") if isinstance(raw, Mapping) else None
        if not isinstance(parent, str) or not Path(parent).is_file():
            raise A7MechanismReplayError(f"arm config is not a runtime config: {config_path}") from exc
        try:
            return load_config(Path(parent))
        except Exception as parent_exc:
            raise A7MechanismReplayError(f"parent runtime config is invalid: {parent}") from parent_exc


def _targets(
    teacher_margin: torch.Tensor, treatment: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(treatment["margin_target_mode"])
    cap = float(treatment["margin_cap"])
    if mode == "fixed":
        target = torch.full_like(teacher_margin, float(treatment["margin_gamma"]))
        active = torch.ones_like(teacher_margin)
    elif mode == "teacher_zero":
        target = teacher_margin.clamp(min=0.0, max=cap)
        active = torch.ones_like(teacher_margin)
    elif mode == "teacher_floor":
        target = teacher_margin.clamp(min=float(treatment["margin_floor"]), max=cap)
        active = torch.ones_like(teacher_margin)
    else:
        target = teacher_margin.clamp(min=0.0, max=cap)
        active = (teacher_margin > 0).to(dtype=teacher_margin.dtype)
    return target.detach(), active.detach()


def _regime(margin: float, *, floor: float, cap: float) -> str:
    if margin <= 0.0:
        return "R0"
    if margin < floor:
        return "R1"
    if margin <= cap:
        return "R2"
    return "R3"


def replay_checkpoint(
    *,
    config_path: Path,
    checkpoint: Path,
    mask_path: Path,
    output_dir: Path,
    run: str,
    arm: str,
    device: torch.device,
    expected_epoch: int,
) -> dict[str, Any]:
    """Replay one checkpoint with the exact KL-PGD10 attack, without updates."""
    config = _runtime_config(config_path)
    attack_config = config.method.attack
    if (
        attack_config.loss != "kl"
        or attack_config.kl_target != "teacher_clean"
        or attack_config.steps != 10
        or attack_config.epsilon != "8/255"
        or attack_config.step_size != "2/255"
        or not attack_config.random_start
    ):
        raise A7MechanismReplayError("replay requires the canonical Teacher-clean KL-PGD10 attack")
    if run not in PARENTS:
        raise A7MechanismReplayError(f"unknown run {run}")
    treatment = _treatment(config_path)
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise A7MechanismReplayError("replay requires a clean source tree")
    mask = fixed_clean_wrong_mask(mask_path, run=run)
    selected = {int(item) for item in mask["selected_ids"]}
    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, student)
    if payload.get("epoch") != expected_epoch or payload.get("epoch_boundary") != "end":
        raise A7MechanismReplayError(f"checkpoint is not epoch-{expected_epoch} end state: {checkpoint}")
    if config.teacher is None:
        raise A7MechanismReplayError("replay requires the registered Teacher")
    teacher = build_teacher(config.teacher, tier=config.tier)
    device = torch.device(device)
    student.to(device).eval()
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_rank_batch_size,
        sampler=EpochShuffleSampler(len(train_dataset), seed=0, rank=0, world_size=1, shuffle=False),
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
    )
    attack = LinfPGD(attack_config)
    global_step = payload.get("global_step")
    if not isinstance(global_step, int):
        raise A7MechanismReplayError("checkpoint lacks integer global_step")
    cap = float(treatment["margin_cap"])
    floor = float(treatment.get("margin_floor") or 0.0)
    coefficient = float(treatment["margin_coefficient"])
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(device)
        with torch.no_grad():
            student_clean_logits = student(batch.images.float())
            teacher_clean_logits = teacher(batch.images.float())
        generator = torch.Generator(device=device).manual_seed(
            config.seeds.model_init + 1_000_003 * (global_step + batch_index)
        )
        attack_result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits.detach().float(),
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, attack_result.max_abs_delta)
        with torch.no_grad():
            student_adv_logits = student(attack_result.adversarial.float())
            teacher_adv_logits = teacher(attack_result.adversarial.float())
        student_clean = _probability_stats(student_clean_logits, batch.labels)
        student_adv = _probability_stats(student_adv_logits, batch.labels)
        teacher_clean = _probability_stats(teacher_clean_logits, batch.labels)
        teacher_adv = _probability_stats(teacher_adv_logits, batch.labels)
        student_margin = student_adv["margin"]
        teacher_margin = teacher_adv["margin"]
        target, active = _targets(teacher_margin, treatment)
        hinge = torch.relu(target - student_margin)
        weighted_loss = coefficient * active * hinge
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise A7MechanismReplayError("replay populated Teacher parameter gradients")
        for idx, sample_id in enumerate(batch.sample_ids.tolist()):
            if int(sample_id) not in selected:
                continue
            tm = float(teacher_margin[idx])
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "true_label": int(batch.labels[idx]),
                    "epoch": expected_epoch,
                    "run": run,
                    "arm": arm,
                    "student_clean_correct": bool(student_clean["correct"][idx]),
                    "student_adv_correct": bool(student_adv["correct"][idx]),
                    "student_clean_margin": float(student_clean["margin"][idx]),
                    "student_adv_margin": float(student_margin[idx]),
                    "student_clean_true_probability": float(student_clean["true_probability"][idx]),
                    "student_adv_true_probability": float(student_adv["true_probability"][idx]),
                    "teacher_clean_correct": bool(teacher_clean["correct"][idx]),
                    "teacher_adv_correct": bool(teacher_adv["correct"][idx]),
                    "teacher_clean_margin": float(teacher_clean["margin"][idx]),
                    "teacher_adv_margin": tm,
                    "teacher_clean_true_probability": float(teacher_clean["true_probability"][idx]),
                    "teacher_adv_true_probability": float(teacher_adv["true_probability"][idx]),
                    "regime": _regime(tm, floor=floor, cap=cap),
                    "target": float(target[idx]),
                    "target_active": bool(active[idx]),
                    "raw_deficit": float(target[idx] - student_margin[idx]),
                    "positive_deficit": float(hinge[idx]),
                    "hinge_active": bool(hinge[idx] > 0),
                    "margin_loss": float(weighted_loss[idx]),
                }
            )
        student.zero_grad(set_to_none=True)
    if len(rows) != len(selected) or {row["sample_id"] for row in rows} != selected:
        raise A7MechanismReplayError("replay did not recover the exact sparse Clean-Wrong ID set")
    epsilon_value = attack_config.epsilon_value
    if epsilon_value is None:
        raise A7MechanismReplayError("attack epsilon is unresolved")
    if max_abs_delta > float(epsilon_value) + 1e-7:
        raise A7MechanismReplayError("replay exceeded its pixel-space Linf bound")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "a7-mechanism-sample-stats.parquet"
    write_sample_parquet(rows, rows_path)
    result = {
        "schema_version": 1,
        "contract": "ert_cw_a7_mechanism_replay_v1",
        "no_training": True,
        "run": run,
        "arm": arm,
        "feature_epoch": expected_epoch,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": expected_epoch,
        "checkpoint_global_step": global_step,
        "parent_checkpoint_sha256": PARENTS[run],
        "mask_path": mask["mask_path"],
        "mask_sha256": mask["mask_sha256"],
        "selected_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": _sha256(rows_path),
        "source_git_sha": source["sha"],
        "attack": attack_config.identity(),
        "attack_identity_sha256": attack_config.identity_sha256(),
        "attack_seed_protocol": "model_init_seed + 1000003*(checkpoint_global_step + batch_index)",
        "max_abs_delta": max_abs_delta,
        "full_train_order_replayed": True,
        "treatment": treatment,
        "regime_contract": {"R0": "mT<=0", "R1": "0<mT<floor", "R2": "floor<=mT<=cap", "R3": "mT>cap"},
    }
    (output_dir / "a7-mechanism-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
