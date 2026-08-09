#!/usr/bin/env python3
# ruff: noqa: E501
"""One-batch loss/gradient scale dry-run for FFNR route coefficients.

The command loads an exact epoch-79 parent checkpoint and computes ordinary
RSLAD plus hypothetical selected-sample CE/KD changes.  It performs no
optimizer update, no validation, and no training artifact mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_train_validation_views
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective


def _mask_ids(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != "ffnr_causal_pilot_mask_v1":
        raise ValueError(f"unexpected mask contract: {path}")
    ids = payload.get("selected_ids")
    if not isinstance(ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ValueError(f"mask IDs are invalid: {path}")
    return set(ids)


def _grad_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def _candidate(
    *,
    terms: Any,
    logits: torch.Tensor,
    labels: torch.Tensor,
    selected: torch.Tensor,
    kd_multiplier: float,
    ce_coefficient: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(logits, labels, reduction="none")
    delta = selected * ((kd_multiplier - 1.0) * (5.0 / 6.0) * terms.adversarial_kd + ce_coefficient * ce)
    total = terms.total + delta
    chosen = selected.bool()
    stats = {
        "selected_count": float(chosen.sum().item()),
        "baseline_selected_median": float(terms.total.detach()[chosen].median().item()) if bool(chosen.any()) else 0.0,
        "candidate_selected_median": float(total.detach()[chosen].median().item()) if bool(chosen.any()) else 0.0,
        "baseline_batch_mean": float(terms.total.detach().mean().item()),
        "candidate_batch_mean": float(total.detach().mean().item()),
        "candidate_to_baseline_batch_ratio": float(
            total.detach().mean().item() / max(abs(terms.total.detach().mean().item()), 1e-12)
        ),
    }
    return total, stats


def run(*, config_path: Path, parent: Path, masks: list[Path], device: str, batch_size: int) -> dict[str, Any]:
    config = load_config(config_path)
    if config.method.id != "rslad" or config.dataset.split != "train":
        raise ValueError("dry-run requires the resolved Chen RSLAD train config")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 79 or payload.get("epoch_boundary") != "end":
        raise ValueError("parent must be the exact epoch-79 end-boundary checkpoint")
    target_device = torch.device(device)
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    train_dataset.set_epoch(79)
    if batch_size < 8 or batch_size > len(train_dataset):
        raise ValueError("batch size is outside the dry-run range")
    items = [train_dataset[index] for index in range(batch_size)]
    images = torch.stack([item[0] for item in items]).to(target_device)
    labels = torch.tensor([int(item[1]) for item in items], device=target_device)
    sample_ids = [int(item[2]) for item in items]
    student = build_student(config.student, tier=config.tier).to(target_device)
    student.load_state_dict(payload["model"], strict=True)
    student.eval()
    teacher = build_teacher(config.teacher, tier=config.tier).to(target_device)
    teacher.eval()
    for parameter in teacher.parameters():
        if parameter.grad is not None:
            raise ValueError("teacher unexpectedly has a parameter gradient before dry-run")
        parameter.requires_grad_(False)
    with torch.no_grad():
        teacher_clean = teacher(images).detach().float()
    attack = LinfPGD(config.method.attack)
    generator = torch.Generator(device=target_device).manual_seed(config.seeds.train_attack + 79_000_003)
    adversarial = attack.generate(
        AttackRequest(
            inputs=images,
            labels=labels,
            student=student,
            teacher=teacher,
            target_logits=teacher_clean,
            generator=generator,
        )
    ).adversarial
    logits = student(adversarial)
    clean_logits = student(images)
    objective = RSLADObjective(
        temperature=config.method.temperature, temperature_squared=config.method.temperature_squared
    )
    terms = objective(
        student_logits=logits, labels=labels, teacher_logits=teacher_clean, clean_student_logits=clean_logits
    )
    base_loss = float(terms.total.detach().mean().item())
    student.zero_grad(set_to_none=True)
    terms.total.mean().backward(retain_graph=True)
    baseline_gradient_norm = _grad_norm(student)
    student.zero_grad(set_to_none=True)
    results: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ffnr_loss_scale_dry_run_v1",
        "parent": str(parent),
        "parent_epoch": 79,
        "sample_ids": sample_ids,
        "base_loss": base_loss,
        "baseline_gradient_norm": baseline_gradient_norm,
        "masks": {},
    }
    for mask_path in masks:
        selected_ids = _mask_ids(mask_path)
        selected = torch.tensor(
            [sample_id in selected_ids for sample_id in sample_ids], device=target_device, dtype=logits.dtype
        )
        mask_results: dict[str, Any] = {
            "path": str(mask_path),
            "selected_in_batch": int(selected.sum().item()),
            "candidates": {},
        }
        candidates = (
            [("kd05_ce025", 0.5, 0.25), ("kd05_ce050", 0.5, 0.5)]
            if "route_a" in mask_path.name
            else [("kd10_ce025", 1.0, 0.25), ("kd10_ce050", 1.0, 0.5)]
        )
        for name, kd_multiplier, ce_coefficient in candidates:
            total, stats = _candidate(
                terms=terms,
                logits=logits,
                labels=labels,
                selected=selected,
                kd_multiplier=kd_multiplier,
                ce_coefficient=ce_coefficient,
            )
            student.zero_grad(set_to_none=True)
            total.mean().backward(retain_graph=True)
            stats.update(
                {
                    "kd_multiplier": kd_multiplier,
                    "ce_coefficient": ce_coefficient,
                    "gradient_norm": _grad_norm(student),
                    "gradient_to_baseline": _grad_norm(student) / max(baseline_gradient_norm, 1e-12),
                }
            )
            mask_results["candidates"][name] = stats
        results["masks"][mask_path.name] = mask_results
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--mask", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        config_path=args.config, parent=args.parent, masks=args.mask, device=args.device, batch_size=args.batch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
