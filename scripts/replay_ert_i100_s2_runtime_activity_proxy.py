#!/usr/bin/env python3
"""Checkpoint no-update training-proxy activity for fixed-cohort DPM/D-BDD.

This is deliberately *not* a reconstruction of historical per-visit activity:
the saved checkpoint is post-epoch and distributed rank peers are unavailable.
It does preserve the configured augmented training view, sample-keyed KL-PGD10,
Student train mode, rank-local natural batch composition, fixed mask, pair gate,
and full-rank-batch loss reduction.  The report joins it to the separate
canonical CE20 action branch only after replay.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ard.analysis.ert_i100_s2_forensic import dynamic_pair_margin, state_tensor_hash
from torch.utils.data import DataLoader

from ard.analysis.ert_i100_action_transfer import sha256
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.models import build_student, build_teacher

EPSILON = 1e-12


@contextmanager
def preserve_state(model: torch.nn.Module):
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
            raise RuntimeError("runtime-proxy replay did not restore Student parameters/buffers bitwise")


def loader(config: Any, *, epoch: int) -> DataLoader:
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


def dbdd_loss(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    adversarial: torch.Tensor,
    labels: torch.Tensor,
    rival: torch.Tensor,
    student_adv_margin: torch.Tensor,
    teacher_adv_margin: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runtime-identical detached D-BDD selected loss and hinge indicator."""
    result = torch.zeros_like(student_adv_margin)
    selected_indices = torch.nonzero(active > 0, as_tuple=False).flatten()
    if not selected_indices.numel():
        return result, torch.zeros_like(active, dtype=torch.bool)
    x = adversarial.detach().index_select(0, selected_indices).requires_grad_(True)
    pair = rival.index_select(0, selected_indices)
    selected_labels = labels.index_select(0, selected_indices)
    original_student_mode, original_teacher_mode = student.training, teacher.training
    try:
        student.eval()
        teacher.eval()
        student_margin, _ = dynamic_pair_margin(student(x.float()), selected_labels, pair)
        grad_student = torch.autograd.grad(student_margin.sum(), x, create_graph=False, retain_graph=False)[0].detach()
        x_teacher = x.detach().requires_grad_(True)
        teacher_margin, _ = dynamic_pair_margin(teacher(x_teacher.float()), selected_labels, pair)
        grad_teacher = torch.autograd.grad(teacher_margin.sum(), x_teacher, create_graph=False, retain_graph=False)[
            0
        ].detach()
    finally:
        student.train(original_student_mode)
        teacher.train(original_teacher_mode)
    d_student = student_adv_margin.index_select(0, selected_indices) / (
        grad_student.abs().flatten(1).sum(dim=1) + EPSILON
    )
    d_teacher = teacher_adv_margin.index_select(0, selected_indices) / (
        grad_teacher.abs().flatten(1).sum(dim=1) + EPSILON
    )
    hinge = d_teacher.detach() - d_student
    values = 0.5 * F.relu(hinge).square()
    result.index_copy_(0, selected_indices, values)
    positive = torch.zeros_like(active, dtype=torch.bool)
    positive.index_copy_(0, selected_indices, hinge > 0)
    return result, positive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=("dev-1", "dev-2"), required=True)
    parser.add_argument("--arm", choices=("dpm", "dbdd"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA mismatch")
    config = load_config(args.config)
    attack_config = config.method.attack.model_copy(update={"random_start_keying": "sample_keyed_v1"})
    if attack_config.loss != "kl" or attack_config.steps != 10 or attack_config.kl_target != "teacher_clean":
        raise ValueError("runtime proxy requires registered KL-PGD10")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != args.epoch or payload.get("epoch_boundary") != "end":
        raise ValueError("checkpoint payload does not match requested endpoint")
    device = torch.device(args.device)
    student = build_student(config.student, tier=config.tier).to(device)
    student.load_state_dict(payload["model"], strict=True)
    teacher = build_teacher(config.teacher, tier=config.tier).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    fixed_ids = set(json.loads(args.mask.read_text(encoding="utf-8"))["masks"]["s2_t1"]["selected_ids"])
    attack = LinfPGD(attack_config)
    per_sample: list[dict[str, Any]] = []
    counts = {"seen": 0, "selected": 0, "teacher_student_pair_gate": 0, "hinge_positive": 0, "extra_loss_positive": 0}
    with preserve_state(student):
        student.train()
        for batch in loader(config, epoch=args.epoch):
            batch = batch.to(device)
            selected = torch.as_tensor(
                [int(sample_id) in fixed_ids for sample_id in batch.sample_ids.tolist()],
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
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                teacher_adv_logits = teacher(adversarial.float()).detach().float()
            student_margin, rival = dynamic_pair_margin(student_adv_logits, batch.labels)
            teacher_margin, _ = dynamic_pair_margin(teacher_adv_logits, batch.labels, rival)
            pair_gate = teacher_margin > 0
            active = selected * pair_gate.to(dtype=selected.dtype)
            if args.arm == "dpm":
                hinge = teacher_margin - student_margin
                loss = 0.5 * F.relu(hinge).square() * active
                hinge_positive = hinge > 0
            else:
                loss, hinge_positive = dbdd_loss(
                    student=student,
                    teacher=teacher,
                    adversarial=adversarial,
                    labels=batch.labels,
                    rival=rival,
                    student_adv_margin=student_margin,
                    teacher_adv_margin=teacher_margin,
                    active=active,
                )
            selected_indices = torch.nonzero(selected > 0, as_tuple=False).flatten().tolist()
            counts["seen"] += int(batch.labels.numel())
            counts["selected"] += len(selected_indices)
            for index in selected_indices:
                row = {
                    "sample_id": int(batch.sample_ids[index]),
                    "teacher_student_pair_gate": bool(pair_gate[index]),
                    "hinge_positive": bool(hinge_positive[index]),
                    "extra_loss_positive": bool(loss[index].detach() > 0),
                    "raw_extra_loss": float(loss[index].detach()),
                }
                counts["teacher_student_pair_gate"] += int(row["teacher_student_pair_gate"])
                counts["hinge_positive"] += int(row["hinge_positive"] and row["teacher_student_pair_gate"])
                counts["extra_loss_positive"] += int(row["extra_loss_positive"])
                per_sample.append(row)
            student.zero_grad(set_to_none=True)
    complete_mask = len({row["sample_id"] for row in per_sample}) == len(fixed_ids)
    if counts["seen"] != 45_000 or counts["selected"] != len(fixed_ids) or not complete_mask:
        raise RuntimeError("runtime proxy did not cover the fixed train mask exactly once")
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_checkpoint_no_update_runtime_activity_proxy_v1",
        "seed": args.seed,
        "arm": args.arm,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_epoch": args.epoch,
        "config_sha256": sha256(args.config),
        "mask_sha256": sha256(args.mask),
        "scope": "checkpoint no-update rank-local training proxy; not historical per-visit activity",
        "runtime_contract": {
            "student_train_mode": True,
            "teacher_eval_frozen": True,
            "train_view_augmentation_epoch": args.epoch,
            "rank_local_natural_batch_size": config.training.per_rank_batch_size,
            "full_rank_batch_mean": True,
            "selected_count_normalization": False,
            "attack_identity_sha256": attack_config.identity_sha256(),
            "ddp_peer_batch_unavailable": True,
        },
        "student_state_hash_before_after": "identical",
        "counts": counts,
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(sha256(args.output) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
