#!/usr/bin/env python3
"""No-update gradient geometry at the exact I100 epoch-99 parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from ard.attacks import AttackRequest, LinfPGD
from ard.cli.train import _seed_everything
from ard.config import load_config
from ard.data import build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective
from ard.tracking.adapter import collect_git_state

PARENT_HASHES = {
    1: "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    2: "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(grads: tuple[torch.Tensor | None, ...], parameters: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().reshape(-1)
            for gradient, parameter in zip(grads, parameters, strict=True)
        ]
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(left.norm().item() * right.norm().item())
    return None if denominator == 0.0 else float(torch.dot(left, right).item() / denominator)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def group_ids(parent: dict[str, Any]) -> dict[str, list[int]]:
    records = parent.get("sample_state", {}).get("records")
    if not isinstance(records, dict) or len(records) < 1024:
        raise ValueError("e99 parent has no complete sample-state records")
    scored = sorted(
        ((float(record["margin_ema"]), int(sample_id)) for sample_id, record in records.items()),
        key=lambda item: (item[0], item[1]),
    )
    n = len(scored)
    high_n, mid_n = n // 5, (3 * n) // 5
    high = [sample_id for _, sample_id in scored[:high_n]]
    mid = [sample_id for _, sample_id in scored[high_n : high_n + mid_n]]
    low = [sample_id for _, sample_id in scored[high_n + mid_n :]]
    generator = torch.Generator(device="cpu").manual_seed(81_000 + int(parent["epoch"]))
    random_order = torch.randperm(n, generator=generator).tolist()
    all_ids = [sample_id for _, sample_id in scored]
    return {
        "HIGH_ONLY": high[: 8 * 128],
        "MID_ONLY": mid[: 8 * 128],
        "LOW_ONLY": low[: 8 * 128],
        "RANDOM": [all_ids[index] for index in random_order[: 8 * 128]],
        "MIXED_20_60_20": [
            sample_id
            for batch in range(8)
            for sample_id in (
                high[batch * 26 : (batch + 1) * 26]
                + mid[batch * 76 : (batch + 1) * 76]
                + low[batch * 26 : (batch + 1) * 26]
            )
        ],
    }


def run_seed(*, seed: int, parent_path: Path, config_path: Path, output: Path, device: str) -> dict[str, Any]:
    if sha256(parent_path) != PARENT_HASHES[seed]:
        raise ValueError(f"parent SHA mismatch for seed {seed}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, dict) or parent.get("epoch") != 99 or parent.get("epoch_boundary") != "end":
        raise ValueError("gradient geometry requires the exact e99 end-boundary parent")
    config = load_config(config_path, ["method.attack.random_start_keying=sample_keyed_v1"])
    if config.seeds.model_init != seed:
        raise ValueError("config model seed does not match parent seed")
    if config.teacher is None:
        raise ValueError("gradient geometry requires the registered teacher")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    torch_device = torch.device(device)
    _seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    train_dataset.set_epoch(99)
    position = {int(sample_id): index for index, sample_id in enumerate(train_dataset.indices)}
    groups = group_ids(parent)
    if any(sample_id not in position for ids in groups.values() for sample_id in ids):
        raise ValueError("gradient geometry group contains an ID outside the train view")
    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(parent_path, student)
    student = student.to(torch_device).eval()
    teacher = build_teacher(config.teacher, tier=config.tier).to(torch_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    attack = LinfPGD(config.method.attack)
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    parameters = tuple(student.parameters())
    summaries: dict[str, Any] = {}
    vectors: dict[str, torch.Tensor] = {}
    batch_size = 128
    for group_name, ids in groups.items():
        batch_vectors: list[torch.Tensor] = []
        kd_vectors: list[torch.Tensor] = []
        total_norms: list[float] = []
        kd_norms: list[float] = []
        for batch_index in range(8):
            batch_ids = ids[batch_index * batch_size : (batch_index + 1) * batch_size]
            items = [train_dataset[position[sample_id]] for sample_id in batch_ids]
            batch = collate_indexed(items).to(torch_device)
            with torch.no_grad():
                teacher_clean = teacher(batch.images.float()).detach().float()
                student_clean = student(batch.images.float())
            attack_result = attack.generate(
                AttackRequest(
                    inputs=batch.images,
                    labels=batch.labels,
                    student=student,
                    teacher=teacher,
                    target_logits=teacher_clean,
                    source_ids=batch.sample_ids,
                    epoch=99,
                    attack_seed=config.seeds.train_attack,
                    stream_tag="train_pgd",
                    restart_index=0,
                )
            )
            student_adv = student(attack_result.adversarial.float())
            terms = objective(
                student_logits=student_adv,
                labels=batch.labels,
                teacher_logits=teacher_clean,
                clean_student_logits=student_clean,
            )
            total_loss = terms.total.mean()
            adv_kd_loss = terms.adversarial_kd.mean() if terms.adversarial_kd is not None else terms.kd.mean()
            total_grad = flatten(
                torch.autograd.grad(total_loss, parameters, retain_graph=True, allow_unused=True), parameters
            )
            kd_grad = flatten(
                torch.autograd.grad(adv_kd_loss, parameters, retain_graph=False, allow_unused=True), parameters
            )
            batch_vectors.append(total_grad)
            kd_vectors.append(kd_grad)
            total_norms.append(float(total_grad.norm().item()))
            kd_norms.append(float(kd_grad.norm().item()))
            student.zero_grad(set_to_none=True)
        vector = torch.stack(batch_vectors).mean(dim=0)
        kd_vector = torch.stack(kd_vectors).mean(dim=0)
        vectors[group_name] = vector
        summaries[group_name] = {
            "sample_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "batch_count": 8,
            "batch_size": batch_size,
            "total_gradient_norm": float(vector.norm().item()),
            "adv_kd_gradient_norm": float(kd_vector.norm().item()),
            "batch_total_gradient_norm_median": statistics.median(total_norms),
            "batch_adv_kd_gradient_norm_median": statistics.median(kd_norms),
            "batch_total_gradient_norm_iqr": [percentile(total_norms, 0.25), percentile(total_norms, 0.75)],
            "batch_adv_kd_gradient_norm_iqr": [percentile(kd_norms, 0.25), percentile(kd_norms, 0.75)],
        }
    summary = {
        "schema_version": 1,
        "contract": "ert_rslad_ordering_gradient_geometry_v1",
        "no_update": True,
        "source_git_sha": collect_git_state(Path.cwd())["sha"],
        "seed": seed,
        "epoch": 99,
        "parent_checkpoint": str(parent_path.resolve()),
        "parent_checkpoint_sha256": sha256(parent_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": payload.get("config_hash"),
        "risk_definition": "-margin_ema; HIGH=lowest-margin top20%, LOW=highest-margin bottom20%",
        "group_summaries": summaries,
        "pairwise_cosine_total": {
            "HIGH_vs_MID": cosine(vectors["HIGH_ONLY"], vectors["MID_ONLY"]),
            "HIGH_vs_LOW": cosine(vectors["HIGH_ONLY"], vectors["LOW_ONLY"]),
            "MID_vs_LOW": cosine(vectors["MID_ONLY"], vectors["LOW_ONLY"]),
            "BALANCED_vs_RANDOM": cosine(vectors["MIXED_20_60_20"], vectors["RANDOM"]),
        },
        "gradient_cancellation_high_low": (
            1.0
            - float((vectors["HIGH_ONLY"] + vectors["LOW_ONLY"]).norm().item())
            / max(1e-12, float(vectors["HIGH_ONLY"].norm().item() + vectors["LOW_ONLY"].norm().item()))
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False:
        raise RuntimeError("gradient geometry requires a clean source tree")
    result = run_seed(
        seed=args.seed,
        parent_path=args.parent.resolve(),
        config_path=args.config.resolve(),
        output=args.output.resolve(),
        device=args.device,
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "parent": result["parent_checkpoint_sha256"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
