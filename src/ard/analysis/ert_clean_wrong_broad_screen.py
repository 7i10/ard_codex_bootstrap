"""Contracts and no-update calibration for the ERT Clean-Wrong screen.

The actual continuation remains the shared Stage-A runtime.  This module only
owns the frozen arm table, the fixed epoch-79 cohort contract, and the C12
coefficient calibration so that a shell launcher cannot silently invent an arm.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_train_validation_views, collate_indexed
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective


class CleanWrongScreenError(RuntimeError):
    """Raised when a Clean-Wrong screen contract is not satisfied."""


@dataclass(frozen=True)
class BroadArm:
    name: str
    selected_epsilon: str = "8/255"
    selected_step: str = "2/255"
    advkd_multiplier: float = 1.0
    clean_ce: float = 0.0
    bce: bool = False
    adaptive_pressure: bool = False
    teacher_gate: bool = False
    iad_inspired: bool = False


ARMS: tuple[BroadArm, ...] = (
    BroadArm("C0"),
    BroadArm("C1", "4/255", "1/255"),
    BroadArm("C2", advkd_multiplier=0.5),
    BroadArm("C3", "4/255", "1/255", advkd_multiplier=0.5),
    BroadArm("C4", clean_ce=0.075),
    BroadArm("C5", "4/255", "1/255", clean_ce=0.075),
    BroadArm("C6", advkd_multiplier=0.5, clean_ce=0.075),
    BroadArm("C7", "4/255", "1/255", advkd_multiplier=0.5, clean_ce=0.075),
    BroadArm("C8", "2/255", "0.5/255"),
    BroadArm("C9", advkd_multiplier=0.25),
    BroadArm("C10", clean_ce=0.15),
    BroadArm("C11", advkd_multiplier=1.5),
    BroadArm("C12", bce=True),
    BroadArm("C13", adaptive_pressure=True),
    BroadArm("C14", teacher_gate=True),
    BroadArm("C15", iad_inspired=True),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def arm_by_name(name: str) -> BroadArm:
    for arm in ARMS:
        if arm.name == name:
            return arm
    raise CleanWrongScreenError(f"unknown Clean-Wrong arm: {name}")


def validate_arm(arm: BroadArm) -> None:
    if _fraction(arm.selected_step) > _fraction(arm.selected_epsilon):
        raise CleanWrongScreenError(f"{arm.name} selected step exceeds epsilon")
    if not 0.0 <= arm.advkd_multiplier:
        raise CleanWrongScreenError(f"{arm.name} AdvKD multiplier is negative")
    if arm.clean_ce < 0:
        raise CleanWrongScreenError(f"{arm.name} CleanCE is negative")


def fixed_clean_wrong_mask(mask_path: Path, *, run: str) -> dict[str, Any]:
    """Return the registered epoch-79 Clean-Wrong IDs and class counts."""
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    if payload.get("anchor_epoch") != 79 or payload.get("contract") != "ert_state_overlay_v1":
        raise CleanWrongScreenError("Clean-Wrong mask is not the registered epoch-79 overlay")
    masks = payload.get("masks")
    if not isinstance(masks, Mapping) or not isinstance(masks.get("student_clean_wrong"), Mapping):
        raise CleanWrongScreenError("registered student_clean_wrong mask is missing")
    record = masks["student_clean_wrong"]
    ids = record.get("selected_ids")
    if not isinstance(ids, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise CleanWrongScreenError("Clean-Wrong mask IDs are not integer stable IDs")
    if ids != sorted(set(ids)):
        raise CleanWrongScreenError("Clean-Wrong IDs must be sorted and unique")
    return {
        "run": run,
        "anchor_epoch": 79,
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": _sha256(mask_path),
        "selected_ids": ids,
        "selected_count": len(ids),
        "selected_class_counts": record.get("selected_class_counts", {}),
    }


def _bce_adv(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=1)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-8)
    wrong = probabilities.scatter(1, labels[:, None], 0.0).amax(dim=1)
    return -torch.log(true_probability) - torch.log((1.0 - wrong).clamp_min(1e-8))


def calibrate_bce_beta(
    *,
    config_path: Path,
    checkpoint_path: Path,
    mask_path: Path,
    output_path: Path,
    device: str = "cuda",
    max_samples: int = 128,
    seed: int = 20260816,
) -> dict[str, Any]:
    """Freeze C12 beta from no-update AdvKD/BCE gradient ratios."""
    config = load_config(config_path)
    if config.method.id != "rslad" or config.method.attack.loss != "kl" or config.method.attack.steps != 10:
        raise CleanWrongScreenError("C12 calibration requires the canonical RSLAD KL-PGD10 parent")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("epoch") != 79 or payload.get("epoch_boundary") != "end":
        raise CleanWrongScreenError("C12 calibration requires an epoch-79 end checkpoint")
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    student = build_student(config.student, tier=config.tier).to(device_obj)
    student.load_state_dict(payload["model"], strict=True)
    student.eval()
    teacher = build_teacher(config.teacher, tier=config.tier).to(device_obj) if config.teacher is not None else None
    if teacher is None:
        raise CleanWrongScreenError("C12 calibration requires a frozen teacher")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    mask = fixed_clean_wrong_mask(mask_path, run=config.seeds.model_init.__str__())
    selected = set(mask["selected_ids"])
    positions = [index for index, sample_id in enumerate(train_dataset.indices) if int(sample_id) in selected]
    rng = random.Random(seed)
    rng.shuffle(positions)
    positions = sorted(positions[:max_samples])
    if not positions:
        raise CleanWrongScreenError("C12 calibration cohort is empty")
    loader = DataLoader(
        Subset(train_dataset, positions),
        batch_size=min(config.training.per_rank_batch_size, 64),
        shuffle=False,
        collate_fn=collate_indexed,
    )
    attack = LinfPGD(config.method.attack)
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    ratios: list[float] = []
    for batch in loader:
        batch = batch.to(device_obj)
        with torch.no_grad():
            teacher_clean = teacher(batch.images.float()).detach().float()
        adversarial = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean,
                generator=torch.Generator(device=device_obj).manual_seed(seed),
            )
        ).adversarial
        student_adv = student(adversarial)
        with torch.no_grad():
            student_clean = student(batch.images)
        terms = objective(
            student_logits=student_adv,
            labels=batch.labels,
            teacher_logits=teacher_clean,
            clean_student_logits=student_clean,
        )
        if terms.adversarial_kd is None:
            raise CleanWrongScreenError("RSLAD calibration did not expose adversarial KD")
        student.zero_grad(set_to_none=True)
        terms.adversarial_kd.mean().backward(retain_graph=True)
        base = torch.cat(
            [p.grad.detach().float().reshape(-1) for p in student.parameters() if p.grad is not None]
        ).norm()
        student.zero_grad(set_to_none=True)
        _bce_adv(student_adv, batch.labels).mean().backward()
        bce = torch.cat(
            [p.grad.detach().float().reshape(-1) for p in student.parameters() if p.grad is not None]
        ).norm()
        if not torch.isfinite(base) or not torch.isfinite(bce) or float(bce) <= 0:
            raise CleanWrongScreenError("C12 calibration produced a non-finite or zero gradient")
        ratios.append(float((base / bce).detach().cpu()))
    median_ratio = float(torch.tensor(ratios, dtype=torch.float64).median())
    beta = 0.25 * median_ratio
    result = {
        "schema_version": 1,
        "contract": "ert_clean_wrong_bce_calibration_v1",
        "tau": 2.0,
        "target_gradient_ratio": 0.25,
        "beta_bce": beta,
        "base_over_bce_ratio_median": median_ratio,
        "batch_ratios": ratios,
        "sample_count": len(positions),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "mask": mask,
        "seed": seed,
        "source_git_sha": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
