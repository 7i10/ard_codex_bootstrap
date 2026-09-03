#!/usr/bin/env python3
"""Bounded, non-training canary for the I100 boundary-distance contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ard.analysis.ert_stage_a_runtime import StageATreatment
from ard.data import IndexedBatch
from ard.engine import Trainer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def _trainer(mode: str) -> Trainer:
    student = _Linear()
    teacher = _Linear()
    with torch.no_grad():
        student.linear.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        teacher.linear.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainer = Trainer.__new__(Trainer)
    trainer.model = student
    trainer.teacher = teacher
    trainer.device = torch.device("cpu")
    trainer.boundary_intervention = mode
    trainer.boundary_coefficient = 1.0
    trainer.boundary_epsilon = 1e-12
    trainer._boundary_epoch_stats = {
        "boundary_active_count": 0.0,
        "boundary_gate_positive_count": 0.0,
        "boundary_zero_rho_count": 0.0,
        "boundary_input_gradient_calls": 0.0,
    }
    trainer._teacher_adversarial_forward_calls = 0.0
    return trainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", action="append", required=True)
    parser.add_argument("--parent-sha", action="append", required=True)
    parser.add_argument("--mask", action="append", required=True)
    parser.add_argument("--mask-sha", action="append", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-sha", required=True)
    args = parser.parse_args()
    if not (len(args.parent) == len(args.parent_sha) == len(args.mask) == len(args.mask_sha) == 2):
        raise SystemExit("canary expects two parent and mask pairs")
    records = []
    pairs = zip(args.parent, args.parent_sha, args.mask, args.mask_sha, strict=True)
    for parent, expected, mask, mask_expected in pairs:
        parent_path, mask_path = Path(parent), Path(mask)
        if _sha256(parent_path) != expected:
            raise SystemExit(f"parent SHA mismatch: {parent_path}")
        if _sha256(mask_path) != mask_expected:
            raise SystemExit(f"mask SHA mismatch: {mask_path}")
        records.append({"parent": str(parent_path), "parent_sha256": expected, "mask": str(mask_path)})
    teacher = Path(args.teacher)
    if _sha256(teacher) != args.teacher_sha:
        raise SystemExit("Teacher SHA mismatch")

    images = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    labels = torch.tensor([0, 1])
    batch = IndexedBatch(images, labels, torch.tensor([10, 11]), torch.tensor([True, True]))
    student_adv = torch.tensor([[1.2, 0.6, -0.2], [0.3, 1.1, -0.4]], requires_grad=True)
    student_clean = torch.tensor([[1.0, 0.8, -0.2], [0.2, 1.0, -0.4]], requires_grad=True)
    teacher_adv = torch.tensor([[1.5, 0.7, -0.3], [0.4, 1.3, -0.4]])
    teacher_clean = torch.tensor([[1.2, 0.9, -0.3], [0.3, 1.1, -0.4]])
    adversarial = images + 0.1
    mode_stats = {}
    for mode in ("pair_margin", "detached_boundary_distance", "secant_boundary_distance"):
        trainer = _trainer(mode)
        values = trainer._boundary_terms(
            batch=batch,
            adversarial=adversarial,
            logits=student_adv,
            clean_student_logits=student_clean,
            teacher_clean_logits=teacher_clean,
            teacher_adversarial_logits=teacher_adv,
            treatment_risk=torch.ones(2),
        )
        if not bool(torch.isfinite(values).all()):
            raise SystemExit(f"non-finite canary values for {mode}")
        values.sum().backward()
        if any(parameter.grad is not None for parameter in trainer.teacher.parameters()):
            raise SystemExit(f"Teacher received gradient for {mode}")
        mode_stats[mode] = trainer._boundary_epoch_stats
    # The public treatment contract must reject a boundary intervention without
    # a fixed mask key; this guards against accidental dynamic routing.
    try:
        StageATreatment(
            arm="invalid",
            mask_key=None,
            kind="broad",
            boundary_intervention="pair_margin",
            boundary_coefficient=1.0,
        )
    except Exception:
        pass
    else:
        raise SystemExit("boundary treatment accepted without a fixed mask")
    print(
        json.dumps(
            {"status": "pass", "lineage": records, "teacher_sha256": args.teacher_sha, "modes": mode_stats},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
