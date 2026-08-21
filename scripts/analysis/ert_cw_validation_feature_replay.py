#!/usr/bin/env python3
"""Replay epoch-79 CE-PGD20 or KL-PGD10 features on the fixed validation view.

This is an evaluation-only producer for the Clean-Wrong generalization
diagnostic.  It never updates model, optimizer, scheduler, or sample state.
The validation feature rows are intentionally separate from the train-only
Clean-Wrong replay artifacts used to define the train-derived quantile bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.ert_clean_wrong_subtypes import _probability_stats
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.tracking.adapter import collect_git_state

ROOT = Path(__file__).resolve().parents[2]
CE_CONTRACT = "ert_clean_wrong_validation_ce_pgd20_features_v1"
KL_CONTRACT = "ert_clean_wrong_validation_kl_pgd10_features_v1"


class ReplayError(RuntimeError):
    """Raised when the validation replay contract is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_identity(rows: list[dict[str, Any]]) -> str:
    pairs = [(int(row["sample_id"]), int(row["true_label"])) for row in rows]
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def replay(
    *,
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    run: str,
    attack_kind: str,
    device: torch.device,
) -> dict[str, Any]:
    config = load_config(config_path)
    attack_config = config.method.selection_attack if attack_kind == "CE20" else config.method.attack
    if attack_kind == "CE20":
        expected = {"loss": "ce", "steps": 20, "epsilon": "8/255", "step_size": "2/255", "kl_target": None}
    else:
        expected = {"loss": "kl", "steps": 10, "epsilon": "8/255", "step_size": "2/255", "kl_target": "teacher_clean"}
    if attack_config is None or any(getattr(attack_config, key) != value for key, value in expected.items()):
        raise ReplayError(f"{run}/{attack_kind}: configured attack does not match the frozen contract")
    if not attack_config.random_start or attack_config.input_domain != "pixel_0_1" or attack_config.norm != "linf":
        raise ReplayError(f"{run}/{attack_kind}: attack domain/random-start contract mismatch")
    source = collect_git_state(ROOT)
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise ReplayError("validation replay requires a clean source tree")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 79 or payload.get("epoch_boundary") != "end":
        raise ReplayError(f"{run}: checkpoint is not epoch-79 end")
    student = build_student(config.student, tier=config.tier)
    load_saved_student_checkpoint(checkpoint, student)
    if config.teacher is None:
        raise ReplayError("validation replay requires a teacher")
    teacher = build_teacher(config.teacher, tier=config.tier)
    device = torch.device(device)
    student.to(device).eval()
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    _, validation_dataset = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    loader = DataLoader(
        validation_dataset,
        batch_size=config.training.per_rank_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
    )
    attack = LinfPGD(attack_config)
    global_step = payload.get("global_step")
    if not isinstance(global_step, int):
        raise ReplayError("epoch-79 checkpoint lacks integer global_step")
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(device)
        with torch.no_grad():
            student_clean_logits = student(batch.images.float())
            teacher_clean_logits = teacher(batch.images.float())
        if attack_kind == "CE20":
            generator = torch.Generator(device=device).manual_seed(config.seeds.evaluation_attack + batch_index)
            request = AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=None,
                generator=generator,
            )
        else:
            generator = torch.Generator(device=device).manual_seed(
                config.seeds.model_init + 1_000_003 * (global_step + batch_index)
            )
            request = AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits.detach().float(),
                generator=generator,
            )
        attack_result = attack.generate(request)
        max_abs_delta = max(max_abs_delta, float(attack_result.max_abs_delta))
        with torch.no_grad():
            student_adv_logits = student(attack_result.adversarial.float())
            teacher_adv_logits = teacher(attack_result.adversarial.float())
        student_clean = _probability_stats(student_clean_logits, batch.labels)
        student_adv = _probability_stats(student_adv_logits, batch.labels)
        teacher_clean = _probability_stats(teacher_clean_logits, batch.labels)
        teacher_adv = _probability_stats(teacher_adv_logits, batch.labels)
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise ReplayError("teacher parameter gradient was populated")
        for index, sample_id in enumerate(batch.sample_ids.tolist()):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "true_label": int(batch.labels[index]),
                    "student_clean_correct": bool(student_clean["correct"][index]),
                    "student_adv_correct": bool(student_adv["correct"][index]),
                    "student_clean_margin": float(student_clean["margin"][index]),
                    "student_adv_margin": float(student_adv["margin"][index]),
                    "student_clean_true_probability": float(student_clean["true_probability"][index]),
                    "student_adv_true_probability": float(student_adv["true_probability"][index]),
                    "teacher_clean_correct": bool(teacher_clean["correct"][index]),
                    "teacher_adv_correct": bool(teacher_adv["correct"][index]),
                    "teacher_clean_margin": float(teacher_clean["margin"][index]),
                    "teacher_adv_margin": float(teacher_adv["margin"][index]),
                    "teacher_clean_true_probability": float(teacher_clean["true_probability"][index]),
                    "teacher_adv_true_probability": float(teacher_adv["true_probability"][index]),
                }
            )
    if len(rows) != len(validation_dataset) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ReplayError(f"{run}/{attack_kind}: validation row count or stable IDs are incomplete")
    if max_abs_delta > float(attack_config.epsilon_value) + 1e-7:
        raise ReplayError(f"{run}/{attack_kind}: attack exceeded Linf epsilon")
    rows.sort(key=lambda row: int(row["sample_id"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "validation-feature-stats.parquet"
    write_sample_parquet(rows, rows_path)
    result = {
        "schema_version": 1,
        "contract": CE_CONTRACT if attack_kind == "CE20" else KL_CONTRACT,
        "dataset_scope": "validation",
        "feature_epoch": 79,
        "run": run,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": 79,
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "row_count": len(rows),
        "split_identity_sha256": _split_identity(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256(rows_path),
        "source_git_sha": source["sha"],
        "attack": attack_config.identity(),
        "attack_identity_sha256": attack_config.identity_sha256(),
        "attack_seed_protocol": (
            "CE: evaluation_attack + batch_index; KL: model_init + 1000003*(checkpoint_global_step + batch_index)"
        ),
        "checkpoint_global_step": global_step,
        "max_abs_delta": max_abs_delta,
        "full_validation_order_replayed": True,
    }
    (output_dir / "validation-feature-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", choices=("L2", "L4"), required=True)
    parser.add_argument("--attack", choices=("CE20", "KL10"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = replay(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        run=args.run,
        attack_kind=args.attack,
        device=torch.device(args.device),
    )
    print(json.dumps({"contract": result["contract"], "rows_sha256": result["rows_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
