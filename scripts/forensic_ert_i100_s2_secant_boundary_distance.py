#!/usr/bin/env python3
"""No-update real-checkpoint forensic for corrected I100 Secant BDD.

This command deliberately does not instantiate an optimiser or scheduler.  It
uses rank-local, natural training batches and the frozen KL-PGD10 runtime view
to distinguish the registered selected-only/eval calibration from a closer
checkpoint no-update runtime proxy.  It is not a historical training replay.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.analysis.ert_i100_action_transfer import sha256
from ard.analysis.ert_i100_s2_secant_forensic import (
    central_difference,
    dynamic_pair_margin,
    quantile_summary,
    rank_correlation,
    scalar_secant_loss,
    secant_components,
    state_tensor_hash,
)
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective

EPSILON = 1e-12


@contextmanager
def preserve_state(model: torch.nn.Module):
    """Restore parameters and buffers bitwise after a read-only forensic."""
    snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before = state_tensor_hash(snapshot)
    mode = model.training
    try:
        yield before
    finally:
        model.load_state_dict(snapshot, strict=True)
        model.train(mode)
        after = state_tensor_hash({name: value.detach() for name, value in model.state_dict().items()})
        if before != after:
            raise RuntimeError("forensic did not restore Student parameters/buffers bitwise")


def _loader(config: Any, *, epoch: int) -> DataLoader:
    train, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    train.set_epoch(epoch)
    sampler = EpochShuffleSampler(len(train), seed=config.seeds.data_order, rank=0, world_size=1, shuffle=True)
    sampler.set_epoch(epoch)
    return DataLoader(
        train,
        batch_size=config.training.per_rank_batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_indexed,
    )


def _norm(parameters: list[torch.Tensor | None]) -> float:
    total = 0.0
    for value in parameters:
        if value is not None:
            total += float(value.detach().float().square().sum().item())
    return math.sqrt(total)


def _head_parameter(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    candidates = [(name, parameter) for name, parameter in model.named_parameters() if parameter.ndim >= 2]
    if not candidates:
        raise RuntimeError("Student has no matrix parameter for directional finite difference")
    return candidates[-1]


def _scalar_fd(row: dict[str, float]) -> dict[str, Any]:
    adv = torch.tensor([row["student_adv_margin"]], dtype=torch.float64, requires_grad=True)
    clean = torch.tensor([row["student_clean_margin"]], dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([row["rho"]], dtype=torch.float64)
    d_teacher = torch.tensor([row["d_teacher"]], dtype=torch.float64)
    loss = scalar_secant_loss(adv, clean, rho=rho, d_teacher=d_teacher, epsilon=EPSILON).sum()
    grad_adv, grad_clean = torch.autograd.grad(loss, (adv, clean))
    output: dict[str, Any] = {"sample_id": int(row["sample_id"]), "kink_safe": False, "steps": []}

    # The two partial derivatives have distinct perturbation paths.  Check
    # their abs and ReLU regions separately: a safe perturbation in m_adv does
    # not imply that m_clean is safe (or vice versa).
    def region(adv_value: float, clean_value: float) -> tuple[float, bool]:
        delta = adv_value - clean_value
        d_student = adv_value / (abs(delta) / (float(rho) + EPSILON) + EPSILON)
        return math.copysign(1.0, delta), bool(float(d_teacher) - d_student > 0.0)

    base_sign, base_hinge = region(float(adv.detach()), float(clean.detach()))

    def partial_safe(*, perturb_adv: bool, step: float) -> bool:
        for signed in (-step, step):
            adv_value = float(adv.detach()) + signed if perturb_adv else float(adv.detach())
            clean_value = float(clean.detach()) if perturb_adv else float(clean.detach()) + signed
            sign, hinge = region(adv_value, clean_value)
            if sign != base_sign or hinge != base_hinge:
                return False
        return True

    for step in (1e-4, 1e-5, 1e-6):

        def adv_fn(value: torch.Tensor) -> torch.Tensor:
            return scalar_secant_loss(value, clean.detach(), rho=rho, d_teacher=d_teacher, epsilon=EPSILON).sum()

        def clean_fn(value: torch.Tensor) -> torch.Tensor:
            return scalar_secant_loss(adv.detach(), value, rho=rho, d_teacher=d_teacher, epsilon=EPSILON).sum()

        safe_adv = partial_safe(perturb_adv=True, step=step)
        safe_clean = partial_safe(perturb_adv=False, step=step)
        safe = safe_adv and safe_clean
        record: dict[str, Any] = {
            "step": step,
            "kink_safe_adv": safe_adv,
            "kink_safe_clean": safe_clean,
            "kink_safe": safe,
        }
        if safe_adv:
            fd_adv = central_difference(adv_fn, adv.detach(), step=step)
            record.update(
                {
                    "autograd_adv": float(grad_adv),
                    "finite_difference_adv": float(fd_adv),
                    "absolute_error_adv": float((grad_adv - fd_adv).abs()),
                }
            )
        if safe_clean:
            fd_clean = central_difference(clean_fn, clean.detach(), step=step)
            record.update(
                {
                    "autograd_clean": float(grad_clean),
                    "finite_difference_clean": float(fd_clean),
                    "absolute_error_clean": float((grad_clean - fd_clean).abs()),
                }
            )
            output["kink_safe"] = True
        output["steps"].append(record)
    return output


def _parameter_fd(
    *,
    student: torch.nn.Module,
    images: torch.Tensor,
    adversarial: torch.Tensor,
    labels: torch.Tensor,
    rival: torch.Tensor,
    teacher_adv_margin: torch.Tensor,
    teacher_clean_margin: torch.Tensor,
    rho: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, Any]:
    """Train-mode, state-restored head-direction finite-difference check.

    The frozen adversarial tensor, rival, Teacher values, rho and selected
    mask isolate the outer S-BDD formula.  Every forward begins from the same
    complete Student state_dict, so BatchNorm buffers follow the production
    train-mode path without contaminating the neighbouring +/- evaluations.
    """
    name, parameter = _head_parameter(student)
    generator = torch.Generator(device=parameter.device).manual_seed(917_431)
    direction = torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype)
    direction = direction / direction.norm()
    parameter_snapshot = parameter.detach().clone()
    full_snapshot = {key: value.detach().clone() for key, value in student.state_dict().items()}
    before = state_tensor_hash(full_snapshot)
    original_mode = student.training
    result: dict[str, Any] = {}
    try:

        def restore(offset: float) -> None:
            with torch.no_grad():
                student.load_state_dict(full_snapshot, strict=True)
                parameter.copy_(parameter_snapshot + offset * direction)
            student.train()

        def loss_at(offset: float) -> torch.Tensor:
            restore(offset)
            adv_logits = student(adversarial.float())
            clean_logits = student(images.float())
            adv_margin, observed_rival = dynamic_pair_margin(adv_logits, labels, rival)
            clean_margin, _ = dynamic_pair_margin(clean_logits, labels, rival)
            if not torch.equal(observed_rival, rival):
                raise RuntimeError("frozen rival unexpectedly changed")
            values = secant_components(
                student_adv_margin=adv_margin,
                student_clean_margin=clean_margin,
                teacher_adv_margin=teacher_adv_margin,
                teacher_clean_margin=teacher_clean_margin,
                rho=rho,
                selected=selected,
                epsilon=EPSILON,
            )
            return values["raw_loss"].mean(), values

        baseline, baseline_values = loss_at(0.0)
        gradient = torch.autograd.grad(baseline, parameter, retain_graph=False)[0]
        autograd_direction = float((gradient * direction).sum())
        checks: list[dict[str, Any]] = []
        for step in (1e-3, 3e-4, 1e-4):
            plus, plus_values = loss_at(step)
            minus, minus_values = loss_at(-step)
            same_hinge = bool(torch.equal(plus_values["hinge_positive"], baseline_values["hinge_positive"])) and bool(
                torch.equal(minus_values["hinge_positive"], baseline_values["hinge_positive"])
            )
            same_abs_sign = bool(
                torch.equal(plus_values["student_margin_delta_sign"], baseline_values["student_margin_delta_sign"])
            ) and bool(
                torch.equal(minus_values["student_margin_delta_sign"], baseline_values["student_margin_delta_sign"])
            )
            same_teacher_gate = bool(
                torch.equal(plus_values["teacher_pair_gate"], baseline_values["teacher_pair_gate"])
            ) and bool(torch.equal(minus_values["teacher_pair_gate"], baseline_values["teacher_pair_gate"]))
            same_rho_gate = bool(torch.equal(plus_values["nonzero_rho"], baseline_values["nonzero_rho"])) and bool(
                torch.equal(minus_values["nonzero_rho"], baseline_values["nonzero_rho"])
            )
            safe = same_abs_sign and same_hinge and same_teacher_gate and same_rho_gate
            record: dict[str, Any] = {
                "step": step,
                "kink_safe": safe,
                "abs_sign_preserved": same_abs_sign,
                "hinge_preserved": same_hinge,
                "teacher_pair_gate_preserved": same_teacher_gate,
                "rho_gate_preserved": same_rho_gate,
            }
            if safe:
                central = float((plus - minus) / (2.0 * step))
                record.update(
                    {
                        "autograd_directional": autograd_direction,
                        "central_difference": central,
                        "absolute_error": float(abs(autograd_direction - central)),
                    }
                )
            checks.append(record)
        result = {
            "parameter": name,
            "student_mode": "train_with_full_state_restore",
            "state_hash_before": before,
            "checks": checks,
        }
    finally:
        with torch.no_grad():
            student.load_state_dict(full_snapshot, strict=True)
        student.train(original_mode)
        after = state_tensor_hash({key: value.detach() for key, value in student.state_dict().items()})
        if before != after:
            raise RuntimeError("parameter directional finite difference failed to restore Student state")
        result["state_hash_after"] = after
        result["state_hash_before_after"] = "identical"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_batches < 2:
        raise ValueError("forensic needs multiple real batches")
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("e99 parent checkpoint SHA mismatch")
    config = load_config(args.config)
    attack_config = config.method.attack.model_copy(update={"random_start_keying": "sample_keyed_v1"})
    if attack_config.loss != "kl" or attack_config.steps != 10 or attack_config.kl_target != "teacher_clean":
        raise ValueError("forensic requires registered sample-keyed KL-PGD10")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 99 or payload.get("epoch_boundary") != "end":
        raise ValueError("forensic requires exact e99 end-boundary parent")
    device = torch.device(args.device)
    student = build_student(config.student, tier=config.tier).to(device)
    student.load_state_dict(payload["model"], strict=True)
    teacher = build_teacher(config.teacher, tier=config.tier).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    selected_ids = set(json.loads(args.mask.read_text(encoding="utf-8"))["masks"]["s2_t1"]["selected_ids"])
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    attack = LinfPGD(attack_config)
    batch_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    scalar_candidates: list[dict[str, float]] = []
    parameter_fd: dict[str, Any] | None = None
    with preserve_state(student):
        student.train()
        for batch_index, batch in enumerate(_loader(config, epoch=args.epoch)):
            if batch_index >= args.max_batches:
                break
            batch = batch.to(device)
            selected = torch.as_tensor(
                [int(sample_id) in selected_ids for sample_id in batch.sample_ids.tolist()],
                device=device,
                dtype=torch.float32,
            )
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                teacher_clean = teacher(batch.images.float()).detach().float()
            adversarial = attack.generate(
                AttackRequest(
                    inputs=batch.images,
                    labels=batch.labels,
                    student=student,
                    teacher=teacher,
                    target_logits=teacher_clean,
                    source_ids=batch.sample_ids,
                    epoch=args.epoch,
                    attack_seed=config.seeds.train_attack,
                    stream_tag="train_pgd",
                    restart_index=0,
                    generator=torch.Generator(device=device),
                )
            ).adversarial.detach()
            student_adv_logits = student(adversarial.float())
            student_clean_logits = student(batch.images.float())
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                teacher_adv_logits = teacher(adversarial.float()).detach().float()
            student_adv_margin, rival = dynamic_pair_margin(student_adv_logits, batch.labels)
            student_clean_margin, _ = dynamic_pair_margin(student_clean_logits, batch.labels, rival)
            teacher_adv_margin, _ = dynamic_pair_margin(teacher_adv_logits, batch.labels, rival)
            teacher_clean_margin, _ = dynamic_pair_margin(teacher_clean, batch.labels, rival)
            rho = (adversarial.detach() - batch.images.detach()).abs().flatten(1).amax(dim=1)
            values = secant_components(
                student_adv_margin=student_adv_margin,
                student_clean_margin=student_clean_margin,
                teacher_adv_margin=teacher_adv_margin,
                teacher_clean_margin=teacher_clean_margin,
                rho=rho,
                selected=selected,
                epsilon=EPSILON,
            )
            terms = objective(
                student_logits=student_adv_logits,
                clean_student_logits=student_clean_logits,
                teacher_logits=teacher_clean,
                labels=batch.labels,
            )
            assert terms.adversarial_kd is not None
            base_per_sample = objective.ADVERSARIAL_COEFFICIENT * terms.adversarial_kd
            base_grad = torch.autograd.grad(
                base_per_sample.mean(),
                tuple(student.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            secant_grad = torch.autograd.grad(
                values["raw_loss"].mean(),
                tuple(student.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            batch_records.append(
                {
                    "batch": batch_index,
                    "batch_size": int(batch.labels.numel()),
                    "selected_count": int(selected.sum()),
                    "teacher_pair_gate_count": int(values["teacher_pair_gate"].sum()),
                    "hinge_positive_count": int((values["hinge_positive"] * values["active"]).sum()),
                    "base_advkd_gradient_norm": _norm(list(base_grad)),
                    "secant_at_coefficient_1_gradient_norm": _norm(list(secant_grad)),
                    "secant_to_base_ratio_at_1": _norm(list(secant_grad)) / _norm(list(base_grad)),
                }
            )
            active_indices = torch.nonzero(values["active"] > 0, as_tuple=False).flatten().tolist()
            for index in active_indices:
                base_one = base_per_sample[index] / batch.labels.numel()
                secant_one = values["raw_loss"][index] / batch.labels.numel()
                base_one_grad = torch.autograd.grad(
                    base_one, tuple(student.parameters()), retain_graph=True, allow_unused=True
                )
                secant_one_grad = torch.autograd.grad(
                    secant_one, tuple(student.parameters()), retain_graph=True, allow_unused=True
                )
                record = {
                    "sample_id": int(batch.sample_ids[index]),
                    "batch": batch_index,
                    "student_adv_margin": float(student_adv_margin[index].detach()),
                    "student_clean_margin": float(student_clean_margin[index].detach()),
                    "teacher_adv_margin": float(teacher_adv_margin[index].detach()),
                    "teacher_clean_margin": float(teacher_clean_margin[index].detach()),
                    "abs_student_margin_delta": float(
                        (student_adv_margin[index] - student_clean_margin[index]).abs().detach()
                    ),
                    "rho": float(rho[index].detach()),
                    "q_student": float(values["q_student"][index].detach()),
                    "q_teacher": float(values["q_teacher"][index].detach()),
                    "d_student": float(values["d_student"][index].detach()),
                    "d_teacher": float(values["d_teacher"][index].detach()),
                    "hinge_gap": float(values["hinge_gap"][index].detach()),
                    "hinge_positive": bool(values["hinge_positive"][index]),
                    "teacher_student_pair_gate": bool(values["teacher_pair_gate"][index]),
                    "raw_loss": float(values["raw_loss"][index].detach()),
                    "base_advkd_gradient_norm": _norm(list(base_one_grad)),
                    "secant_gradient_norm_at_1": _norm(list(secant_one_grad)),
                }
                record["gradient_ratio_at_1"] = (
                    record["secant_gradient_norm_at_1"] / record["base_advkd_gradient_norm"]
                    if record["base_advkd_gradient_norm"]
                    else math.nan
                )
                sample_records.append(record)
                if len(scalar_candidates) < 8 and record["hinge_positive"]:
                    scalar_candidates.append(record)
            if parameter_fd is None and active_indices:
                parameter_fd = _parameter_fd(
                    student=student,
                    images=batch.images,
                    adversarial=adversarial,
                    labels=batch.labels,
                    rival=rival,
                    teacher_adv_margin=teacher_adv_margin,
                    teacher_clean_margin=teacher_clean_margin,
                    rho=rho,
                    selected=selected,
                )
            student.zero_grad(set_to_none=True)
    if not sample_records:
        raise RuntimeError("no selected teacher-pair-gated real samples in forensic batches")
    scalar_fd = [_scalar_fd(record) for record in scalar_candidates]
    tensors = {
        "q_student": torch.tensor([record["q_student"] for record in sample_records]),
        "inverse_q_student": torch.tensor([1.0 / record["q_student"] for record in sample_records]),
        "abs_student_margin_delta": torch.tensor([record["abs_student_margin_delta"] for record in sample_records]),
        "d_student": torch.tensor([record["d_student"] for record in sample_records]),
        "d_teacher": torch.tensor([record["d_teacher"] for record in sample_records]),
        "hinge_gap": torch.tensor([record["hinge_gap"] for record in sample_records]),
        "raw_loss": torch.tensor([record["raw_loss"] for record in sample_records]),
        "gradient_ratio_at_1": torch.tensor([record["gradient_ratio_at_1"] for record in sample_records]),
    }
    epsilon_sensitivity: dict[str, Any] = {}
    for epsilon in (1e-12, 1e-9, 1e-6, 1e-4):
        # Diagnostic scalars only: no coefficient re-selection and no update.
        losses = []
        for record in sample_records:
            q_teacher = abs(record["teacher_adv_margin"] - record["teacher_clean_margin"]) / (record["rho"] + epsilon)
            d_teacher = record["teacher_adv_margin"] / (q_teacher + epsilon)
            value = scalar_secant_loss(
                torch.tensor([record["student_adv_margin"]], dtype=torch.float64),
                torch.tensor([record["student_clean_margin"]], dtype=torch.float64),
                rho=torch.tensor([record["rho"]], dtype=torch.float64),
                d_teacher=torch.tensor([d_teacher], dtype=torch.float64),
                epsilon=epsilon,
            )
            losses.append(value)
        epsilon_sensitivity[str(epsilon)] = {"raw_loss": quantile_summary(torch.cat(losses))}
    active = torch.tensor([record["hinge_positive"] for record in sample_records], dtype=torch.bool)
    correlations = {
        key: rank_correlation(value, tensors["gradient_ratio_at_1"])
        for key, value in tensors.items()
        if key != "gradient_ratio_at_1"
    }
    correlations_active = {
        key: rank_correlation(value[active], tensors["gradient_ratio_at_1"][active])
        for key, value in tensors.items()
        if key != "gradient_ratio_at_1"
    }
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_secant_forensic_runtime_proxy_v1",
        "scope": (
            "no optimizer/scheduler/model update; rank-local natural training batches; not historical per-visit replay"
        ),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_epoch": 99,
        "config_sha256": sha256(args.config),
        "mask_sha256": sha256(args.mask),
        "formula_version": "student_parameter_graph_v2",
        "formula": "0.5*relu(dT-dS)^2; Student qS graph preserved; Teacher quantities detached; Student-selected rival",
        "epsilon": EPSILON,
        "coefficient_diagnostics_only": [1.0, 1.5219638832872224],
        "runtime_proxy": {
            "train_mode_student": True,
            "teacher_eval_frozen": True,
            "full_rank_batch_mean": True,
            "selected_count_normalization": False,
            "attack_identity_sha256": attack_config.identity_sha256(),
            "epoch": args.epoch,
            "max_natural_batches": args.max_batches,
            "global_ddp_batch_reconstructed": False,
        },
        "state_hash_before_after": "identical",
        "batch_measurements": batch_records,
        "sample_summary": {key: quantile_summary(value) for key, value in tensors.items()},
        "sample_count": len(sample_records),
        "top_gradient_ratio_samples": sorted(
            sample_records, key=lambda row: float(row["gradient_ratio_at_1"]), reverse=True
        )[:20],
        "rank_correlations": {
            "all_teacher_pair_gated_selected": correlations,
            "hinge_positive_subset": correlations_active,
        },
        "scalar_finite_difference": scalar_fd,
        "parameter_directional_finite_difference": parameter_fd,
        "epsilon_sensitivity_diagnostic_only": epsilon_sensitivity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(sha256(args.output) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
