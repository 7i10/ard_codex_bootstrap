"""No-update gradient probe for the frozen A5--A8 margin mechanisms."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ard.analysis.ert_clean_wrong_subtypes import _probability_stats
from ard.analysis.ert_cw_a7_mechanism_replay import _regime, _runtime_config, _targets, _treatment
from ard.attacks import AttackRequest, LinfPGD
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective
from ard.tracking.adapter import collect_git_state


class A7GradientProbeError(RuntimeError):
    """Raised when a no-update gradient probe violates its contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector_grad(loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([value.detach().float().reshape(-1) for value in values if value is not None], dim=0)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denom = float(left.norm().item() * right.norm().item())
    if denom == 0.0:
        return None
    return float(torch.dot(left, right).item() / denom)


def probe_checkpoint(
    *,
    config_path: Path,
    checkpoint: Path,
    probe_rows: Path,
    run: str,
    arm: str,
    epoch: int,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    config = _runtime_config(config_path)
    treatment = _treatment(config_path)
    probe = json.loads(probe_rows.read_text(encoding="utf-8")) if probe_rows.suffix == ".json" else None
    if probe is not None:
        raise A7GradientProbeError("probe rows must be a Parquet path")
    import pyarrow.parquet as pq

    probe_values = pq.read_table(probe_rows).to_pylist()
    if len(probe_values) < 16:
        raise A7GradientProbeError("gradient probe requires at least 16 fixed IDs")
    # The probe panel is fixed by sorted IDs; no endpoint outcome enters selection.
    ids = sorted(int(row["sample_id"]) for row in probe_values)[:128]
    probe_ids = set(ids)
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False:
        raise A7GradientProbeError("gradient probe requires clean source")
    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, student)
    if payload.get("epoch") != epoch or payload.get("epoch_boundary") != "end":
        raise A7GradientProbeError("checkpoint epoch mismatch")
    if config.teacher is None:
        raise A7GradientProbeError("gradient probe requires Teacher")
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
    attack_config = config.method.attack
    attack = LinfPGD(attack_config)
    global_step = payload.get("global_step")
    if not isinstance(global_step, int):
        raise A7GradientProbeError("checkpoint lacks global_step")
    parameters = tuple(student.parameters())
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    coefficient = float(treatment["margin_coefficient"])
    sums: dict[str, list[float]] = defaultdict(list)
    regime_sums: dict[str, list[float]] = defaultdict(list)
    seen: set[int] = set()
    for batch_index, raw_batch in enumerate(loader):
        selected_indices = [index for index, item in enumerate(raw_batch.sample_ids.tolist()) if int(item) in probe_ids]
        if not selected_indices:
            continue
        batch = raw_batch.to(device)
        index = torch.tensor(selected_indices, device=device, dtype=torch.long)
        images = batch.images.index_select(0, index).float()
        labels = batch.labels.index_select(0, index)
        sample_ids = [int(item) for item in batch.sample_ids.index_select(0, index).tolist()]
        # Keep the Student clean graph: the probe measures its clean-CE
        # gradient.  Only the frozen Teacher target is detached.
        student_clean_logits = student(images)
        with torch.no_grad():
            teacher_clean_logits = teacher(images)
        generator = torch.Generator(device=device).manual_seed(
            config.seeds.model_init + 1_000_003 * (global_step + batch_index)
        )
        attack_result = attack.generate(
            AttackRequest(
                inputs=images,
                labels=labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits.detach().float(),
                generator=generator,
            )
        )
        student_adv_logits = student(attack_result.adversarial.float())
        with torch.no_grad():
            teacher_adv_logits = teacher(attack_result.adversarial.float())
        terms = objective(
            student_logits=student_adv_logits,
            labels=labels,
            teacher_logits=teacher_clean_logits,
            clean_student_logits=student_clean_logits,
        )
        teacher_margin = _probability_stats(teacher_adv_logits, labels)["margin"]
        target, active = _targets(teacher_margin, treatment)
        student_margin = _probability_stats(student_adv_logits, labels)["margin"]
        margin_loss = (coefficient * active * F.relu(target - student_margin)).mean()
        base_loss = terms.total.mean()
        clean_loss = F.cross_entropy(student_clean_logits, labels)
        gradients = {
            "base": _vector_grad(base_loss, parameters),
            "clean": _vector_grad(clean_loss, parameters),
            "margin": _vector_grad(margin_loss, parameters),
        }
        base_norm = float(gradients["base"].norm().item())
        margin_norm = float(gradients["margin"].norm().item())
        clean_norm = float(gradients["clean"].norm().item())
        sums["base_norm"].append(base_norm)
        sums["clean_norm"].append(clean_norm)
        sums["margin_norm"].append(margin_norm)
        sums["weighted_margin_base_ratio"].append(margin_norm / base_norm if base_norm else math.nan)
        for left, right in (("margin", "base"), ("margin", "clean"), ("clean", "base")):
            value = _cosine(gradients[left], gradients[right])
            if value is not None:
                sums[f"cosine_{left}_{right}"].append(value)
        for sample_id, margin in zip(sample_ids, teacher_margin.tolist(), strict=True):
            seen.add(sample_id)
            regime_sums[
                _regime(
                    float(margin),
                    floor=float(treatment.get("margin_floor") or 0.03221710026264191),
                    cap=float(treatment.get("margin_cap") or 0.13952550292015076),
                )
            ].append(margin_norm)
        student.zero_grad(set_to_none=True)
    if seen != probe_ids:
        raise A7GradientProbeError("probe did not recover all fixed IDs")
    summary = {key: {"mean": sum(values) / len(values), "n": len(values)} for key, values in sums.items() if values}
    result = {
        "schema_version": 1,
        "contract": "ert_cw_a7_gradient_probe_v1",
        "no_update": True,
        "run": run,
        "arm": arm,
        "epoch": epoch,
        "checkpoint_sha256": _sha256(checkpoint),
        "probe_rows_sha256": _sha256(probe_rows),
        "probe_count": len(probe_ids),
        "source_git_sha": source["sha"],
        "summary": summary,
        "regime_margin_norm_count": {key: len(value) for key, value in regime_sums.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
