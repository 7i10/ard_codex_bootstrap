"""No-update gradient calibration for the ERT Stage A treatment screen."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_train_validation_views, collate_indexed
from ard.engine import config_digest
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective
from ard.targets import TeacherOnlyTemperatureTargetPolicy
from ard.tracking.adapter import collect_git_state


class StageACalibrationError(RuntimeError):
    """Raised when calibration inputs or invariants are invalid."""


COHORTS = ("s3_t1", "s3_t2", "s3_t3", "clean_wrong")
S3_COHORTS = ("s3_t1", "s3_t2", "s3_t3")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageACalibrationError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise StageACalibrationError(f"{name} must be a JSON object")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StageACalibrationError("calibration config is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StageACalibrationError("calibration config schema_version must be 1")
    if value.get("contract") != "ert_stage_a_calibration_v1":
        raise StageACalibrationError("unexpected calibration contract")
    runs = value.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {"L2", "L4"}:
        raise StageACalibrationError("calibration must contain exactly L2 and L4 runs")
    return value


def _resolve(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise StageACalibrationError(f"{name} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _state_masks(path: Path) -> dict[str, set[int]]:
    payload = _load_json(path, name="state mask")
    if payload.get("anchor_epoch") != 79 or payload.get("contract") != "ert_state_overlay_v1":
        raise StageACalibrationError("Stage A masks must be the registered epoch-79 overlay masks")
    masks = payload.get("masks")
    if not isinstance(masks, Mapping):
        raise StageACalibrationError("registered state masks are missing")
    required = {"s3_t1_q10", "s3_t2_q10", "s3_t3_q10", "student_clean_wrong"}
    if not required.issubset(masks):
        raise StageACalibrationError("registered Stage A masks are incomplete")
    result: dict[str, set[int]] = {}
    for name in required:
        record = masks[name]
        if not isinstance(record, Mapping) or not isinstance(record.get("selected_ids"), list):
            raise StageACalibrationError(f"mask {name} has no stable ID list")
        ids = [item for item in record["selected_ids"] if isinstance(item, int) and not isinstance(item, bool)]
        if len(ids) != len(record["selected_ids"]) or len(set(ids)) != len(ids):
            raise StageACalibrationError(f"mask {name} has invalid or duplicate IDs")
        result[name.removesuffix("_q10")] = set(ids)
    return result


def _class_stratified_ids(ids: set[int], labels: Mapping[int, int], *, limit: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for sample_id in sorted(ids):
        if sample_id not in labels:
            raise StageACalibrationError(f"calibration ID {sample_id} is outside the train label namespace")
        by_class[labels[sample_id]].append(sample_id)
    generator = random.Random(seed)
    for members in by_class.values():
        generator.shuffle(members)
    selected: list[int] = []
    classes = sorted(by_class)
    while len(selected) < min(limit, len(ids)):
        progressed = False
        for class_id in classes:
            members = by_class[class_id]
            if members:
                selected.append(members.pop())
                progressed = True
                if len(selected) == min(limit, len(ids)):
                    break
        if not progressed:
            break
    return sorted(selected)


@contextmanager
def _preserve_model_state(model: nn.Module) -> Iterator[None]:
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


def _gradient_vector(model: nn.Module, *, head_only: bool = False) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or (
            head_only and not any(token in name.lower() for token in ("fc", "classifier", "head", "linear"))
        ):
            continue
        values.append(parameter.grad.detach().float().reshape(-1))
    if not values:
        raise StageACalibrationError("calibration produced an empty gradient")
    return torch.cat(values)


def _measure_gradient(model: nn.Module, values: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    values.mean().backward()
    vector = _gradient_vector(model)
    head = _gradient_vector(model, head_only=True)
    return float(vector.norm().item()), float(head.norm().item()), vector


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator.item()) == 0:
        raise StageACalibrationError("gradient cosine has a zero denominator")
    return float(torch.dot(left, right).div(denominator).item())


@dataclass(frozen=True)
class BatchMeasurement:
    run: str
    cohort: str
    batch: int
    base_adv_norm: float
    base_adv_head_norm: float
    soft_adv_norm: float
    soft_adv_head_norm: float
    adv_ce_norm: float
    adv_ce_head_norm: float
    base_total_norm: float
    base_total_head_norm: float
    clean_ce_norm: float
    clean_ce_head_norm: float
    adv_ce_cosine: float


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values or not all(torch.isfinite(torch.tensor(values))):
        raise StageACalibrationError("calibration summary has no finite values")
    tensor = torch.tensor(values, dtype=torch.float64)
    quartiles = torch.quantile(tensor, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return {
        "count": len(values),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "median": float(quartiles[1]),
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
        "iqr": float(quartiles[2] - quartiles[0]),
    }


def _load_parent(
    *, config_path: Path, checkpoint_path: Path, device: torch.device
) -> tuple[Any, nn.Module, nn.Module, dict[str, Any]]:
    config = load_config(config_path)
    if config.method.id != "rslad" or config.method.attack.loss != "kl" or config.method.attack.steps != 10:
        raise StageACalibrationError("calibration parent is not the exact observed RSLAD KL-PGD10 run")
    if config.method.attack.kl_target != "teacher_clean":
        raise StageACalibrationError("calibration parent must use teacher-clean KL target")
    if config.method.attack.temperature != 1.0 or not config.method.attack.temperature_squared:
        raise StageACalibrationError("calibration parent temperature contract drifted")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("epoch") != 79 or payload.get("epoch_boundary") != "end":
        raise StageACalibrationError("calibration checkpoint is not epoch-79 end-boundary state")
    student = build_student(config.student, tier=config.tier).to(device)
    student.load_state_dict(payload["model"], strict=True)
    teacher = build_teacher(config.teacher, tier=config.tier).to(device) if config.teacher is not None else None
    if teacher is None:
        raise StageACalibrationError("calibration requires a frozen Teacher")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return config, student, teacher, dict(payload)


def calibrate(*, config_path: Path, output: Path, device: str = "cuda") -> dict[str, Any]:
    """Run deterministic no-update gradient calibration and freeze coefficients."""
    root = Path.cwd().resolve()
    raw = _load_config(config_path.resolve())
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if raw.get("tau") != 2.0:
        raise StageACalibrationError("Stage A calibration freezes tau=2.0")
    all_measurements: list[BatchMeasurement] = []
    input_manifest: dict[str, Any] = {}
    for run_name in ("L2", "L4"):
        run = raw["runs"][run_name]
        if not isinstance(run, Mapping):
            raise StageACalibrationError(f"{run_name} run entry is invalid")
        parent_config_path = _resolve(root, run.get("parent_config"), name=f"{run_name} parent config")
        checkpoint_path = _resolve(root, run.get("checkpoint"), name=f"{run_name} checkpoint")
        mask_path = _resolve(root, run.get("mask"), name=f"{run_name} mask")
        if not checkpoint_path.is_file() or not mask_path.is_file() or not parent_config_path.is_file():
            raise StageACalibrationError(f"{run_name} calibration input is missing")
        config, student, teacher, payload = _load_parent(
            config_path=parent_config_path, checkpoint_path=checkpoint_path, device=device_obj
        )
        train_dataset, _ = build_train_validation_views(
            config.dataset,
            validation_fraction=config.training.validation_fraction,
            split_seed=config.seeds.split,
            augmentation_seed=config.seeds.augmentation,
        )
        train_dataset.set_epoch(79)
        raw_targets = getattr(train_dataset.dataset.dataset, "targets", None)
        if not isinstance(raw_targets, (tuple, list)):
            raise StageACalibrationError("parent dataset does not expose immutable train labels")
        labels = {int(sample_id): int(raw_targets[sample_id]) for sample_id in train_dataset.indices}
        masks = _state_masks(mask_path)
        attack = LinfPGD(config.method.attack)
        objective = RSLADObjective(
            temperature=config.method.temperature,
            temperature_squared=config.method.temperature_squared,
        )
        target_policy = TeacherOnlyTemperatureTargetPolicy(target_temperature=2.0, baseline_temperature=1.0)
        input_manifest[run_name] = {
            "config_sha256": _sha256(parent_config_path),
            "config_hash": config_digest(config.model_dump(mode="json")),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": payload.get("epoch"),
            "mask_sha256": _sha256(mask_path),
            "teacher_checkpoint_sha256": config.teacher.checkpoint_sha256 if config.teacher else None,
            "cohort_counts": {name: len(ids) for name, ids in masks.items()},
        }
        for cohort_index, cohort in enumerate(COHORTS):
            ids = _class_stratified_ids(
                masks[cohort], labels, limit=int(raw.get("max_per_cohort", 512)), seed=79_000 + cohort_index
            )
            id_to_position = {sample_id: position for position, sample_id in enumerate(train_dataset.indices)}
            positions = [id_to_position[sample_id] for sample_id in ids if sample_id in id_to_position]
            subset = Subset(train_dataset, positions)
            loader = DataLoader(
                subset,
                batch_size=int(raw.get("batch_size", 64)),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_indexed,
            )
            student.train()
            with _preserve_model_state(student):
                for batch_index, batch in enumerate(loader):
                    images = batch.images.to(device_obj)
                    batch_labels = batch.labels.to(device_obj)
                    generator = torch.Generator(device=device_obj).manual_seed(
                        20260811 + 1009 * cohort_index + batch_index
                    )
                    with torch.no_grad(), torch.autocast(device_type=device_obj.type, enabled=False):
                        teacher_clean = teacher(images.float()).detach().float()
                    attack_result = attack.generate(
                        request=AttackRequest(
                            inputs=images,
                            labels=batch_labels,
                            student=student,
                            teacher=teacher,
                            target_logits=teacher_clean,
                            generator=generator,
                        )
                    )
                    adversarial_logits = student(attack_result.adversarial)
                    clean_logits = student(images)
                    terms = objective(
                        student_logits=adversarial_logits,
                        clean_student_logits=clean_logits,
                        teacher_logits=teacher_clean,
                        labels=batch_labels,
                    )
                    assert terms.adversarial_kd is not None and terms.clean_kd is not None
                    base_adv = objective.ADVERSARIAL_COEFFICIENT * terms.adversarial_kd
                    base_total = terms.kd
                    soft_target = target_policy(
                        teacher_logits=teacher_clean,
                        risk=torch.ones(images.shape[0], device=device_obj),
                        temperature=2.0,
                    )
                    soft_adv = objective(
                        student_logits=adversarial_logits,
                        clean_student_logits=clean_logits,
                        teacher_logits=teacher_clean,
                        labels=batch_labels,
                        adversarial_target_probabilities=soft_target.probabilities,
                    ).adversarial_kd
                    assert soft_adv is not None
                    soft_adv = objective.ADVERSARIAL_COEFFICIENT * soft_adv
                    adv_ce = F.cross_entropy(adversarial_logits, batch_labels, reduction="none")
                    clean_ce = F.cross_entropy(clean_logits, batch_labels, reduction="none")
                    base_adv_norm, base_adv_head, base_adv_vec = _measure_gradient(student, base_adv)
                    soft_norm, soft_head, _ = _measure_gradient(student, soft_adv)
                    ce_norm, ce_head, ce_vec = _measure_gradient(student, adv_ce)
                    total_norm, total_head, _ = _measure_gradient(student, base_total)
                    clean_norm, clean_head, _ = _measure_gradient(student, clean_ce)
                    all_measurements.append(
                        BatchMeasurement(
                            run=run_name,
                            cohort=cohort,
                            batch=batch_index,
                            base_adv_norm=base_adv_norm,
                            base_adv_head_norm=base_adv_head,
                            soft_adv_norm=soft_norm,
                            soft_adv_head_norm=soft_head,
                            adv_ce_norm=ce_norm,
                            adv_ce_head_norm=ce_head,
                            base_total_norm=total_norm,
                            base_total_head_norm=total_head,
                            clean_ce_norm=clean_norm,
                            clean_ce_head_norm=clean_head,
                            adv_ce_cosine=_cosine(base_adv_vec, ce_vec),
                        )
                    )
                    student.zero_grad(set_to_none=True)
    ratios_soft = [m.base_adv_norm / m.soft_adv_norm for m in all_measurements if m.soft_adv_norm > 0]
    advce_reference = [m.base_adv_norm / m.adv_ce_norm for m in all_measurements if m.adv_ce_norm > 0]
    cleance_reference = [m.base_total_norm / m.clean_ce_norm for m in all_measurements if m.clean_ce_norm > 0]
    if not ratios_soft or not advce_reference or not cleance_reference:
        raise StageACalibrationError("calibration has a zero gradient denominator")
    alpha_soft = float(torch.median(torch.tensor(ratios_soft)).item())
    advce_scale = float(torch.median(torch.tensor(advce_reference)).item())
    clean_scale = float(torch.median(torch.tensor(cleance_reference)).item())
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_stage_a_calibration_v1",
        "status": "complete_no_update",
        "tau": 2.0,
        "alpha_soft": alpha_soft,
        "beta_advce_weak": 0.25 * advce_scale,
        "beta_advce_moderate": 0.50 * advce_scale,
        "beta_cleance_weak": 0.25 * clean_scale,
        "device": str(device_obj),
        "inputs": input_manifest,
        "measurements": [m.__dict__ for m in all_measurements],
        "summaries": {
            "base_adv_norm": _summarize([m.base_adv_norm for m in all_measurements]),
            "soft_adv_norm": _summarize([m.soft_adv_norm for m in all_measurements]),
            "adv_ce_norm": _summarize([m.adv_ce_norm for m in all_measurements]),
            "base_total_norm": _summarize([m.base_total_norm for m in all_measurements]),
            "clean_ce_norm": _summarize([m.clean_ce_norm for m in all_measurements]),
            "adv_ce_cosine": _summarize([m.adv_ce_cosine for m in all_measurements]),
            "achieved_soft_ratio": _summarize(
                [alpha_soft * m.soft_adv_norm / m.base_adv_norm for m in all_measurements]
            ),
            "achieved_weak_advce_ratio": _summarize(
                [0.25 * advce_scale * m.adv_ce_norm / m.base_adv_norm for m in all_measurements]
            ),
            "achieved_moderate_advce_ratio": _summarize(
                [0.50 * advce_scale * m.adv_ce_norm / m.base_adv_norm for m in all_measurements]
            ),
            "achieved_cleance_ratio": _summarize(
                [0.25 * clean_scale * m.clean_ce_norm / m.base_total_norm for m in all_measurements]
            ),
        },
        "provenance": {"git": collect_git_state(root)},
        "artifact_sha256": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["artifact_sha256"] = _sha256(output)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
