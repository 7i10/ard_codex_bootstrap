#!/usr/bin/env python3
"""Prepare fixed canonical S2xT1 masks and pooled no-update calibration.

The script is deliberately separate from the continuation runtime.  It only
reads the hash-bound e99 replay rows, builds fixed train/validation masks, and
measures intervention gradients without an optimizer or scheduler update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_train_validation_views, collate_indexed
from ard.models import build_student, build_teacher
from ard.objectives import ObjectiveTerms, RSLADObjective

PARENT_SHA = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
TEACHER_SHA = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
ENDPOINT_ATTACK_ID = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
SAMPLE_KEYED_KL10_ATTACK_IDENTITY = "97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d"
REPLAY_DIR = Path(".cache/analysis/ert-i100-cw-gap-completion-replay")
VALIDATION_DIR = Path(".cache/analysis/ert-i100-cw-gap-e99")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(map(int, ids)), separators=(",", ":")).encode()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate stable IDs in {path}")
    return result


def _q10_boundary(rows: Mapping[int, Mapping[str, Any]], correct_key: str, margin_key: str) -> tuple[float, set[int]]:
    positive = sorted(
        (sid for sid, row in rows.items() if bool(row[correct_key])),
        key=lambda sid: (float(rows[sid][margin_key]), sid),
    )
    if not positive:
        raise ValueError("positive-margin state cohort is empty")
    count = math.ceil(0.10 * len(positive))
    fragile = set(positive[:count])
    # The largest member of the tie-broken q10 cohort is the frozen floor.
    return float(rows[max(fragile, key=lambda sid: (float(rows[sid][margin_key]), sid))][margin_key]), fragile


def canonical_s2_t1(rows: Mapping[int, Mapping[str, Any]]) -> tuple[set[int], dict[str, Any]]:
    # Train replay rows carry explicit CE20 names; the registered validation
    # feature artifact uses the shorter ``*_adv_*`` names.  Both are the same
    # CE-PGD20 contract, and selecting by schema keeps this conversion
    # explicit instead of silently treating a different attack as CE20.
    if "student_ce20_adv_correct" in next(iter(rows.values())):
        student_correct_key, student_margin_key = "student_ce20_adv_correct", "student_ce20_adv_margin"
        teacher_correct_key, teacher_margin_key = "teacher_ce20_adv_correct", "teacher_ce20_adv_margin"
    elif "student_adv_correct" in next(iter(rows.values())):
        student_correct_key, student_margin_key = "student_adv_correct", "student_adv_margin"
        teacher_correct_key, teacher_margin_key = "teacher_adv_correct", "teacher_adv_margin"
    else:
        raise ValueError("rows do not expose the registered CE-PGD20 correctness/margin fields")
    student_floor, student_s2 = _q10_boundary(rows, student_correct_key, student_margin_key)
    _teacher_floor, teacher_t2 = _q10_boundary(rows, teacher_correct_key, teacher_margin_key)
    student_correct = {sid for sid, row in rows.items() if bool(row[student_correct_key])}
    teacher_correct = {sid for sid, row in rows.items() if bool(row[teacher_correct_key])}
    selected = student_s2 & (teacher_correct - teacher_t2)
    if not selected:
        raise ValueError("canonical S2xT1 cohort is empty")
    return selected, {
        "student_s2_count": len(student_s2),
        "student_s1_count": len(student_correct - student_s2),
        "student_s3_count": len(rows) - len(student_correct),
        "teacher_t1_count": len(teacher_correct - teacher_t2),
        "teacher_t2_count": len(teacher_t2),
        "teacher_t3_count": len(rows) - len(teacher_correct),
        "student_q10_floor": student_floor,
        "teacher_q10_boundary": _teacher_floor,
    }


def _class_counts(rows: Mapping[int, Mapping[str, Any]], ids: Iterable[int]) -> dict[str, int]:
    labels = (
        int(rows[sid]["class_id"] if "class_id" in rows[sid] else rows[sid]["true_label"])
        for sid in ids
    )
    return {str(key): value for key, value in sorted(Counter(labels).items())}


def write_mask(
    *,
    run: str,
    train_rows: Mapping[int, Mapping[str, Any]],
    val_rows: Mapping[int, Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    train_ids, train_states = canonical_s2_t1(train_rows)
    val_ids, val_states = canonical_s2_t1(val_rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_rbp_masks_v1",
        "anchor_epoch": 99,
        "run": run,
        "parent_checkpoint_sha256": PARENT_SHA[run],
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "feature_attack": "CE-PGD20, epsilon=8/255, step=2/255, random_start=true, sample_keyed_v1",
        "canonical_state_contract": (
            "S1/S2/S3 and T1/T2/T3: adversarial-correct positive-margin "
            "q10 with stable-ID tie break"
        ),
        "masks": {
            "s2_t1": {
                "namespace": "train",
                "selected_ids": sorted(train_ids),
                "selected_ids_sha256": ids_sha256(train_ids),
                "selected_count": len(train_ids),
                "selected_class_counts": _class_counts(train_rows, train_ids),
            },
            "validation_s2_t1": {
                "namespace": "validation",
                "selected_ids": sorted(val_ids),
                "selected_ids_sha256": ids_sha256(val_ids),
                "selected_count": len(val_ids),
                "selected_class_counts": _class_counts(val_rows, val_ids),
            },
        },
        "state_counts": {"train": train_states, "validation": val_states},
        "provenance": {"source_git_sha": _git_sha()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"path": str(output.resolve()), "sha256": sha256(output), "train": train_states, "validation": val_states}


def _class_stratified(ids: set[int], rows: Mapping[int, Mapping[str, Any]], limit: int = 512) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for sid in sorted(ids):
        by_class[int(rows[sid].get("class_id", rows[sid].get("true_label")))].append(sid)
    selected: list[int] = []
    while len(selected) < min(limit, len(ids)):
        progressed = False
        for cls in sorted(by_class):
            if by_class[cls]:
                selected.append(by_class[cls].pop(0))
                progressed = True
                if len(selected) == min(limit, len(ids)):
                    break
        if not progressed:
            break
    return selected


def _prob_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits.float(), dim=1)
    true = prob.gather(1, labels[:, None]).squeeze(1)
    wrong = prob.scatter(1, labels[:, None], 0.0).amax(dim=1)
    return true - wrong


def _grad_vector(student: torch.nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().float().reshape(-1) for p in student.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError("no student gradient")
    return torch.cat(parts)


def _norm(student: torch.nn.Module, values: torch.Tensor) -> float:
    student.zero_grad(set_to_none=True)
    values.mean().backward(retain_graph=True)
    return float(_grad_vector(student).norm().item())


@contextmanager
def _preserve_buffers(model: torch.nn.Module):
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    training = model.training
    try:
        yield
    finally:
        model.train(training)
        with torch.no_grad():
            for name, value in model.named_buffers():
                if name in buffers:
                    value.copy_(buffers[name])


def _calibrate_run(
    *,
    run: str,
    config_path: Path,
    checkpoint: Path,
    mask_path: Path,
    replay_path: Path,
    device: torch.device,
    tpfm_floor: float,
    tpfm_cap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256(checkpoint) != PARENT_SHA[run]:
        raise ValueError(f"{run} parent SHA mismatch")
    config = load_config(config_path)
    if config.method.id != "rslad" or config.method.attack.loss != "kl" or config.method.attack.steps != 10:
        raise ValueError("calibration requires I100 KL-PGD10")
    keyed_attack = config.method.attack.model_copy(update={"random_start_keying": "sample_keyed_v1"})
    if keyed_attack.identity_sha256() != SAMPLE_KEYED_KL10_ATTACK_IDENTITY:
        raise ValueError("calibration attack does not match the registered sample-keyed KL10 identity")
    config = config.model_copy(update={"method": config.method.model_copy(update={"attack": keyed_attack})})
    rows = _rows(replay_path)
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    ids = set(mask["masks"]["s2_t1"]["selected_ids"])
    sample_ids = _class_stratified(ids, rows)
    device = torch.device(device)
    student = build_student(config.student, tier=config.tier).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(payload["model"], strict=True)
    student.train()
    teacher = build_teacher(config.teacher, tier=config.tier).to(device)
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
    train_dataset.set_epoch(99)
    positions = {int(sid): pos for pos, sid in enumerate(train_dataset.indices)}
    if set(sample_ids) - set(positions):
        raise ValueError(f"{run} calibration IDs are outside the train split")
    subset = Subset(train_dataset, [positions[sid] for sid in sample_ids])
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_indexed)
    attack = LinfPGD(config.method.attack)
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    measurements: list[dict[str, Any]] = []
    with _preserve_buffers(student):
        for batch_index, batch in enumerate(loader):
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
                    source_ids=batch.sample_ids,
                    epoch=99,
                    attack_seed=config.seeds.train_attack,
                    stream_tag="train_pgd",
                    restart_index=0,
                )
            )
            adv_logits = student(attack_result.adversarial.float())
            clean_logits = student(batch.images.float())
            with torch.no_grad():
                teacher_adv = teacher(attack_result.adversarial.float()).detach().float()
            terms = objective(
                student_logits=adv_logits,
                clean_student_logits=clean_logits,
                teacher_logits=teacher_clean,
                labels=batch.labels,
            )
            if terms.adversarial_kd is None:
                raise ValueError("RSLAD did not expose adversarial KD")
            base_adv = objective.ADVERSARIAL_COEFFICIENT * terms.adversarial_kd
            student_margin = _prob_margin(adv_logits, batch.labels)
            teacher_margin = _prob_margin(teacher_adv, batch.labels).detach()
            sbf_floor = float(
                mask.get("state_counts", {}).get("train", {}).get("student_q10_floor", 0.0)
            )
            zero = torch.zeros_like(base_adv)
            sbf_terms = ObjectiveTerms(
                hard=zero,
                kd=zero,
                regularization=zero,
                adversarial_kd=zero,
                clean_kd=zero,
            ).add_adversarial_margin(
                adv_logits,
                batch.labels,
                torch.full_like(student_margin, sbf_floor),
                torch.ones_like(student_margin),
                coefficient=1.0,
            )
            tpfm_terms = ObjectiveTerms(
                hard=zero,
                kd=zero,
                regularization=zero,
                adversarial_kd=zero,
                clean_kd=zero,
            ).add_adversarial_margin(
                adv_logits,
                batch.labels,
                teacher_margin.clamp(min=tpfm_floor, max=tpfm_cap),
                torch.ones_like(student_margin),
                coefficient=1.0,
            )
            base_norm = _norm(student, base_adv)
            sbf_norm = _norm(student, sbf_terms.hard)
            tpfm_norm = _norm(student, tpfm_terms.hard)
            measurements.append(
                {
                    "run": run,
                    "batch": batch_index,
                    "n": int(batch.labels.shape[0]),
                    "base_advkd_norm": base_norm,
                    "sbf_margin_norm": sbf_norm,
                    "tpfm_margin_norm": tpfm_norm,
                    "sbf_floor": sbf_floor,
                    "tpfm_floor": tpfm_floor,
                    "tpfm_cap": tpfm_cap,
                    "sbf_ratio_at_1": sbf_norm / base_norm if base_norm else math.nan,
                    "tpfm_ratio_at_1": tpfm_norm / base_norm if base_norm else math.nan,
                }
            )
            student.zero_grad(set_to_none=True)
    required_norms = ("base_advkd_norm", "sbf_margin_norm", "tpfm_margin_norm")
    if not measurements or any(
        not math.isfinite(float(measurement[key])) or float(measurement[key]) <= 0
        for measurement in measurements
        for key in required_norms
    ):
        raise ValueError(f"{run} calibration has a non-finite/zero gradient")
    return measurements, {
        "run": run,
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config_path),
        "mask_sha256": sha256(mask_path),
        "replay_sha256": sha256(replay_path),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": ids_sha256(sample_ids),
        "student_q10_floor": float(mask["state_counts"]["train"]["student_q10_floor"]),
        "training_attack_identity_sha256": keyed_attack.identity_sha256(),
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    specs = {
        "dev-1": (args.dev1_config, args.dev1_checkpoint, args.dev1_mask, args.dev1_replay),
        "dev-2": (args.dev2_config, args.dev2_checkpoint, args.dev2_mask, args.dev2_replay),
    }
    all_measurements: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    # Freeze TPFM's teacher-floor target from the complete selected positive
    # pre-treatment cohort before measuring any gradients.  Per-batch
    # quantiles would make the calibration depend on DataLoader chunking.
    pooled_teacher: list[float] = []
    for run, (_, _, mask_path, replay_path) in specs.items():
        rows = _rows(Path(replay_path))
        mask = json.loads(Path(mask_path).read_text(encoding="utf-8"))
        ids = set(mask["masks"]["s2_t1"]["selected_ids"])
        teacher_key = (
            "teacher_ce20_adv_margin"
            if "teacher_ce20_adv_margin" in next(iter(rows.values()))
            else "teacher_adv_margin"
        )
        pooled_teacher.extend(
            float(rows[sid][teacher_key])
            for sid in ids
            if float(rows[sid][teacher_key]) > 0
        )
    if not pooled_teacher:
        raise ValueError("TPFM calibration cohort has no positive Teacher margins")
    teacher_tensor = torch.tensor(pooled_teacher, dtype=torch.float64)
    quantiles = torch.quantile(
        teacher_tensor,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    floor, cap = [float(value) for value in quantiles]
    if not (math.isfinite(floor) and math.isfinite(cap) and 0 < floor <= cap):
        raise ValueError(f"invalid pooled TPFM floor/cap: {floor}, {cap}")
    for run, values in specs.items():
        measurements, metadata = _calibrate_run(
            run=run,
            config_path=Path(values[0]),
            checkpoint=Path(values[1]),
            mask_path=Path(values[2]),
            replay_path=Path(values[3]),
            device=torch.device(args.device),
            tpfm_floor=floor,
            tpfm_cap=cap,
        )
        all_measurements.extend(measurements)
        inputs[run] = metadata
    base = torch.tensor([m["base_advkd_norm"] for m in all_measurements], dtype=torch.float64)
    sbf = torch.tensor([m["sbf_margin_norm"] for m in all_measurements], dtype=torch.float64)
    tpfm = torch.tensor([m["tpfm_margin_norm"] for m in all_measurements], dtype=torch.float64)
    sbf_coefficient = float(0.25 * torch.median(base / sbf).item())
    tpfm_coefficient = float(0.25 * torch.median(base / tpfm).item())
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_rbp_calibration_v1",
        "status": "complete_no_update",
        "tau": 2.0,
        "target_ratio": 0.25,
        "parent_epoch": 99,
        "canonical_cohort": "Train S2xT1",
        "sbf": {
            "margin_target_mode": "fixed",
            "coefficient": sbf_coefficient,
            "floor_by_run": {run: value["student_q10_floor"] for run, value in inputs.items()},
        },
        "tpfm": {"margin_target_mode": "teacher_floor", "coefficient": tpfm_coefficient, "floor": floor, "cap": cap},
        "inputs": inputs,
        "measurements": all_measurements,
        "achieved_ratios": {
            "sbf": [float(sbf_coefficient * m["sbf_margin_norm"] / m["base_advkd_norm"]) for m in all_measurements],
            "tpfm": [float(tpfm_coefficient * m["tpfm_margin_norm"] / m["base_advkd_norm"]) for m in all_measurements],
        },
        "provenance": {"source_git_sha": _git_sha(), "attack_identity": "KL-PGD10 teacher-clean sample_keyed_v1"},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["artifact_sha256"] = sha256(output)
    output.with_name(output.name + ".sha256").write_text(result["artifact_sha256"] + "\n", encoding="utf-8")
    return result


def prepare(args: argparse.Namespace) -> None:
    for run, replay, validation, output in (
        ("dev-1", args.dev1_replay, args.dev1_validation, args.dev1_output),
        ("dev-2", args.dev2_replay, args.dev2_validation, args.dev2_output),
    ):
        train_rows, val_rows = _rows(Path(replay)), _rows(Path(validation))
        metadata = write_mask(run=run, train_rows=train_rows, val_rows=val_rows, output=Path(output))
        print(json.dumps(metadata, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--dev1-replay", default=str(REPLAY_DIR / "dev-1/e99-observations.parquet"))
    q.add_argument("--dev2-replay", default=str(REPLAY_DIR / "dev-2/e99-observations.parquet"))
    q.add_argument("--dev1-validation", default=str(VALIDATION_DIR / "L2/CE20/validation-feature-stats.parquet"))
    q.add_argument("--dev2-validation", default=str(VALIDATION_DIR / "L4/CE20/validation-feature-stats.parquet"))
    q.add_argument("--dev1-output", required=True)
    q.add_argument("--dev2-output", required=True)
    c = sub.add_parser("calibrate")
    for run in ("dev1", "dev2"):
        c.add_argument(f"--{run}-config", required=True)
        c.add_argument(f"--{run}-checkpoint", required=True)
        c.add_argument(f"--{run}-mask", required=True)
        c.add_argument(f"--{run}-replay", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        print(json.dumps(calibrate(args), sort_keys=True))
