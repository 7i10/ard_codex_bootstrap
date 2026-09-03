#!/usr/bin/env python3
"""Pooled no-update calibration for the I100 dynamic boundary screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
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

try:
    from scripts.prepare_ert_i100_s2_rbp import _class_stratified, _norm, _rows, sha256
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    from prepare_ert_i100_s2_rbp import _class_stratified, _norm, _rows, sha256

PARENT_SHA = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
TRAIN_ATTACK_ID = "97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d"
BOUNDARY_EPSILON = 1e-12


def _ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


@contextmanager
def _preserve_buffers(model: torch.nn.Module):
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    mode = model.training
    try:
        yield
    finally:
        model.train(mode)
        with torch.no_grad():
            for name, value in model.named_buffers():
                if name in buffers:
                    value.copy_(buffers[name])


def _pair(logits: torch.Tensor, labels: torch.Tensor, rival: torch.Tensor | None = None):
    if rival is None:
        masked = logits.detach().clone()
        masked.scatter_(1, labels[:, None], float("-inf"))
        rival = masked.argmax(dim=1)
    else:
        rival = rival.detach()
    margin = logits.float().gather(1, labels[:, None]).squeeze(1) - logits.float().gather(1, rival[:, None]).squeeze(1)
    return margin, rival


def _geometry_losses(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    images: torch.Tensor,
    adversarial: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    modes: tuple[str, ...] = ("pair_margin", "detached_boundary_distance", "secant_boundary_distance"),
) -> dict[str, torch.Tensor]:
    # Geometry views use eval mode to preserve BN buffers and avoid
    # batch-composition-dependent input gradients.
    student_adv = student(adversarial)
    student_clean = student(images)
    with torch.no_grad():
        teacher_adv = teacher(adversarial).detach()
        teacher_clean = teacher(images).detach()
    student_margin, rival = _pair(student_adv, labels)
    student_clean_margin, _ = _pair(student_clean, labels, rival)
    teacher_margin, _ = _pair(teacher_adv, labels, rival)
    teacher_clean_margin, _ = _pair(teacher_clean, labels, rival)
    active = (teacher_margin > 0).to(dtype=student_adv.dtype)
    rho = (adversarial.detach() - images.detach()).abs().flatten(1).amax(dim=1)
    losses: dict[str, torch.Tensor] = {}
    if "pair_margin" in modes:
        losses["pair_margin"] = 0.5 * F.relu(teacher_margin - student_margin).square() * active
    if "secant_boundary_distance" in modes:
        # The Student denominator is intentionally graph-bearing.  This is
        # the first-order S-BDD contract used by the runtime, not the prior
        # detached-denominator approximation.
        q_student = (student_margin - student_clean_margin).abs() / (rho + epsilon)
        q_teacher = (teacher_margin - teacher_clean_margin).abs() / (rho + epsilon)
        losses["secant_boundary_distance"] = (
            0.5
            * F.relu(teacher_margin / (q_teacher + epsilon) - student_margin / (q_student + epsilon)).square()
            * active
            * (rho > epsilon).to(dtype=active.dtype)
        )
    if "detached_boundary_distance" in modes:
        # D-BDD uses a fresh eval-mode Student/Teacher forward with input gradients.
        x = adversarial.detach().requires_grad_(True)
        with torch.enable_grad():
            student_geom, _ = _pair(student(x), labels, rival)
        grad_s = torch.autograd.grad(student_geom.sum(), x, create_graph=False, retain_graph=False)[0].detach()
        with torch.enable_grad():
            teacher_geom, _ = _pair(teacher(x), labels, rival)
        grad_t = torch.autograd.grad(teacher_geom.sum(), x, create_graph=False, retain_graph=False)[0].detach()
        losses["detached_boundary_distance"] = (
            0.5
            * F.relu(
                teacher_margin / (grad_t.abs().flatten(1).sum(dim=1) + epsilon)
                - student_margin / (grad_s.abs().flatten(1).sum(dim=1).detach() + epsilon)
            ).square()
            * active
        )
    return losses


def _run(
    run: str,
    config_path: Path,
    checkpoint: Path,
    mask_path: Path,
    replay_path: Path,
    device: torch.device,
    modes: tuple[str, ...] = ("pair_margin", "detached_boundary_distance", "secant_boundary_distance"),
    max_batches: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256(checkpoint) != PARENT_SHA[run]:
        raise ValueError(f"{run}: parent SHA mismatch")
    config = load_config(config_path)
    keyed = config.method.attack.model_copy(update={"random_start_keying": "sample_keyed_v1"})
    if keyed.identity_sha256() != TRAIN_ATTACK_ID:
        raise ValueError(f"{run}: training attack identity mismatch")
    config = config.model_copy(update={"method": config.method.model_copy(update={"attack": keyed})})
    rows = _rows(replay_path)
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
    ids = _class_stratified(set(mask_payload["masks"]["s2_t1"]["selected_ids"]), rows)
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    train_dataset.set_epoch(100)
    positions = {int(sid): pos for pos, sid in enumerate(train_dataset.indices)}
    subset = Subset(train_dataset, [positions[sid] for sid in ids])
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_indexed)
    student = build_student(config.student, tier=config.tier).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(payload["model"], strict=True)
    student.eval()
    teacher = build_teacher(config.teacher, tier=config.tier).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    attack = LinfPGD(keyed)
    objective = RSLADObjective(
        temperature=config.method.temperature, temperature_squared=config.method.temperature_squared
    )
    measurements: list[dict[str, Any]] = []
    with _preserve_buffers(student), _preserve_buffers(teacher):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(device)
            with torch.no_grad():
                teacher_clean = teacher(batch.images.float()).detach().float()
            attack_result = attack.generate(
                AttackRequest(
                    inputs=batch.images,
                    labels=batch.labels,
                    student=student,
                    teacher=teacher,
                    target_logits=teacher_clean,
                    generator=torch.Generator(device=device).manual_seed(config.seeds.train_attack + batch_index),
                    source_ids=batch.sample_ids,
                    epoch=100,
                    attack_seed=config.seeds.train_attack,
                    stream_tag="train_pgd",
                    restart_index=0,
                )
            )
            adversarial = attack_result.adversarial.detach()
            adv_logits = student(adversarial.float())
            clean_logits = student(batch.images.float())
            terms = objective(
                student_logits=adv_logits,
                clean_student_logits=clean_logits,
                teacher_logits=teacher_clean,
                labels=batch.labels,
            )
            if terms.adversarial_kd is None:
                raise ValueError("RSLAD did not expose adversarial KD")
            base = objective.ADVERSARIAL_COEFFICIENT * terms.adversarial_kd
            # Recompute intervention tensors with parameter graphs only where
            # a norm is measured; no optimizer/scheduler/state update occurs.
            losses = _geometry_losses(
                student=student,
                teacher=teacher,
                images=batch.images.float(),
                adversarial=adversarial.float(),
                labels=batch.labels,
                epsilon=BOUNDARY_EPSILON,
                modes=modes,
            )
            row: dict[str, Any] = {"run": run, "batch": batch_index, "n": int(batch.labels.numel())}
            base_norm = _norm(student, base)
            row["base_advkd_norm"] = base_norm
            for mode, values in losses.items():
                norm = _norm(student, values)
                row[f"{mode}_norm"] = norm
                row[f"{mode}_ratio_at_1"] = norm / base_norm if base_norm else math.nan
                row[f"{mode}_active_count"] = float((values.detach() > 0).sum().item())
            measurements.append(row)
            student.zero_grad(set_to_none=True)
    if not measurements:
        raise ValueError(f"{run}: no calibration measurements")
    return measurements, {
        "run": run,
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "mask_sha256": sha256(mask_path),
        "replay_sha256": sha256(replay_path),
        "sample_count": len(ids),
        "sample_ids_sha256": _ids_sha(ids),
        "training_attack_identity_sha256": keyed.identity_sha256(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    for run in ("dev1", "dev2"):
        p.add_argument(f"--{run}-config", type=Path, required=True)
        p.add_argument(f"--{run}-checkpoint", type=Path, required=True)
        p.add_argument(f"--{run}-mask", type=Path, required=True)
        p.add_argument(f"--{run}-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    specs = {
        "dev-1": (args.dev1_config, args.dev1_checkpoint, args.dev1_mask, args.dev1_replay),
        "dev-2": (args.dev2_config, args.dev2_checkpoint, args.dev2_mask, args.dev2_replay),
    }
    all_measurements: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    for run, values in specs.items():
        measurements, metadata = _run(run, *values, device=torch.device(args.device))
        all_measurements.extend(measurements)
        inputs[run] = metadata
    base = torch.tensor([m["base_advkd_norm"] for m in all_measurements], dtype=torch.float64)
    coefficients: dict[str, float] = {}
    for mode in ("pair_margin", "detached_boundary_distance", "secant_boundary_distance"):
        norms = torch.tensor([m[f"{mode}_norm"] for m in all_measurements], dtype=torch.float64)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
            raise ValueError(f"{mode}: non-finite/zero gradient norm")
        coefficients[mode] = float(0.25 * torch.median(base / norms).item())
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_dynamic_bdd_calibration_v1",
        "status": "complete_no_update",
        "parent_epoch": 99,
        "calibration_view_epoch": 100,
        "target_gradient_ratio": 0.25,
        "boundary_epsilon": BOUNDARY_EPSILON,
        "inputs": inputs,
        "coefficients": coefficients,
        "measurements": all_measurements,
        "achieved_ratios": {
            mode: [float(coefficients[mode] * m[f"{mode}_norm"] / m["base_advkd_norm"]) for m in all_measurements]
            for mode in coefficients
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256(args.output)
    args.output.with_name(args.output.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    result["artifact_sha256"] = digest
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
