"""No-update calibration for the Clean-Wrong margin action screen."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from ard.analysis.ert_stage_a_calibration import (
    _class_stratified_ids,
    _gradient_vector,
    _load_parent,
    _resolve,
    _sha256,
    _state_masks,
)
from ard.attacks import AttackRequest, LinfPGD
from ard.data import build_train_validation_views, collate_indexed
from ard.objectives import ObjectiveTerms, RSLADObjective
from ard.tracking.adapter import collect_git_state


class CleanWrongMarginCalibrationError(RuntimeError):
    """Raised when the frozen calibration contract cannot be satisfied."""


def _config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CleanWrongMarginCalibrationError("calibration config is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise CleanWrongMarginCalibrationError("calibration config schema_version must be 1")
    if value.get("contract") != "ert_cw_margin_calibration_v1":
        raise CleanWrongMarginCalibrationError("unexpected margin calibration contract")
    if set(value.get("runs", {})) != {"L2", "L4"}:
        raise CleanWrongMarginCalibrationError("calibration requires exactly L2 and L4")
    return value


def _rows(path: Path, *, expected_checkpoint: str, expected_mask: str) -> tuple[list[int], list[float]]:
    table = pq.read_table(path, columns=["sample_id", "teacher_adv_margin"])
    ids = [int(x) for x in table.column("sample_id").to_pylist()]
    margins = [float(x) for x in table.column("teacher_adv_margin").to_pylist()]
    if len(ids) != len(set(ids)) or not all(torch.isfinite(torch.tensor(margins))):
        raise CleanWrongMarginCalibrationError("feature rows contain invalid IDs or margins")
    if expected_checkpoint is None or expected_mask is None:
        raise CleanWrongMarginCalibrationError("feature lineage is incomplete")
    positive = [(sample_id, margin) for sample_id, margin in zip(ids, margins) if margin > 0.0]
    if not positive:
        raise CleanWrongMarginCalibrationError("no positive Teacher margins for calibration")
    return [x[0] for x in positive], [x[1] for x in positive]


def _quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(tensor, torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64))
    return {"q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2])}


def _norm(model: torch.nn.Module, value: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    value.mean().backward(retain_graph=True)
    return float(_gradient_vector(model).norm().item())


def calibrate(*, config_path: Path, output: Path, device: str = "cuda") -> dict[str, Any]:
    root = Path.cwd().resolve()
    raw = _config(config_path.resolve())
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    measurements: list[dict[str, float | int | str]] = []
    inputs: dict[str, Any] = {}
    for run_name in ("L2", "L4"):
        spec = raw["runs"][run_name]
        if not isinstance(spec, Mapping):
            raise CleanWrongMarginCalibrationError(f"{run_name} entry is invalid")
        parent_config = _resolve(root, spec.get("parent_config"), name=f"{run_name} parent config")
        checkpoint = _resolve(root, spec.get("checkpoint"), name=f"{run_name} checkpoint")
        mask_path = _resolve(root, spec.get("mask"), name=f"{run_name} mask")
        feature_meta = _resolve(root, spec.get("feature_meta"), name=f"{run_name} feature metadata")
        feature_rows = _resolve(root, spec.get("feature_rows"), name=f"{run_name} feature rows")
        metadata = json.loads(feature_meta.read_text(encoding="utf-8"))
        if (
            metadata.get("checkpoint_sha256") != _sha256(checkpoint)
            or metadata.get("mask_sha256") != _sha256(mask_path)
        ):
            raise CleanWrongMarginCalibrationError(f"{run_name} feature lineage does not match parent/mask")
        if metadata.get("rows_sha256") != _sha256(feature_rows):
            raise CleanWrongMarginCalibrationError(f"{run_name} feature row hash does not match metadata")
        if metadata.get("contract") != "ert_clean_wrong_c0_kl_pgd10_features_v1":
            raise CleanWrongMarginCalibrationError(f"{run_name} feature attack contract is not KL-PGD10")
        positive_ids, positive_margins = _rows(
            feature_rows,
            expected_checkpoint=metadata.get("checkpoint_sha256"),
            expected_mask=metadata.get("mask_sha256"),
        )
        config, student, teacher, payload = _load_parent(
            config_path=parent_config, checkpoint_path=checkpoint, device=device_obj
        )
        if _sha256(checkpoint) not in {
            "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
            "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
        }:
            raise CleanWrongMarginCalibrationError("calibration parent is not a frozen exact epoch-79 parent")
        train_dataset, _ = build_train_validation_views(
            config.dataset,
            validation_fraction=config.training.validation_fraction,
            split_seed=config.seeds.split,
            augmentation_seed=config.seeds.augmentation,
        )
        train_dataset.set_epoch(79)
        targets = getattr(train_dataset.dataset.dataset, "targets", None)
        if not isinstance(targets, (tuple, list)):
            raise CleanWrongMarginCalibrationError("parent dataset does not expose immutable labels")
        labels = {int(sample_id): int(targets[int(sample_id)]) for sample_id in train_dataset.indices}
        masks = _state_masks(mask_path)
        cw_ids = masks["clean_wrong"]
        selected_ids = sorted(set(positive_ids) & cw_ids)
        if not selected_ids:
            raise CleanWrongMarginCalibrationError(f"{run_name} has no positive Clean-Wrong IDs")
        ids = _class_stratified_ids(
            set(selected_ids), labels, limit=int(raw.get("max_samples_per_seed", 256)), seed=79_005 + len(run_name)
        )
        positions = [train_dataset.indices.index(sample_id) for sample_id in ids]
        loader = DataLoader(
            Subset(train_dataset, positions),
            batch_size=int(raw.get("batch_size", 64)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_indexed,
        )
        attack = LinfPGD(config.method.attack)
        objective = RSLADObjective(
            temperature=config.method.temperature,
            temperature_squared=config.method.temperature_squared,
        )
        gamma = _quantiles(positive_margins)
        for batch_index, batch in enumerate(loader):
            images = batch.images.to(device_obj)
            batch_labels = batch.labels.to(device_obj)
            with torch.no_grad():
                teacher_clean = teacher(images.float()).detach().float()
            attack_result = attack.generate(
                request=AttackRequest(
                    inputs=images,
                    labels=batch_labels,
                    student=student,
                    teacher=teacher,
                    target_logits=teacher_clean,
                    generator=torch.Generator(device=device_obj).manual_seed(20260821 + batch_index),
                )
            )
            adv_logits = student(attack_result.adversarial)
            clean_logits = student(images)
            terms = objective(
                student_logits=adv_logits,
                clean_student_logits=clean_logits,
                teacher_logits=teacher_clean,
                labels=batch_labels,
            )
            if terms.adversarial_kd is None:
                raise CleanWrongMarginCalibrationError("RSLAD did not expose adversarial KD")
            base = objective.ADVERSARIAL_COEFFICIENT * terms.adversarial_kd
            ce = F.cross_entropy(adv_logits, batch_labels, reduction="none")
            zero = torch.zeros_like(base)
            target = torch.full_like(base, gamma["q50"])
            margin_terms = ObjectiveTerms(
                hard=zero,
                kd=zero,
                regularization=zero,
                adversarial_kd=zero,
                clean_kd=zero,
            ).add_adversarial_margin(adv_logits, batch_labels, target, torch.ones_like(base), coefficient=1.0)
            base_norm = _norm(student, base)
            ce_norm = _norm(student, ce)
            margin_norm = _norm(student, margin_terms.hard)
            measurements.append(
                {
                    "run": run_name,
                    "batch": batch_index,
                    "base_advkd_norm": base_norm,
                    "advce_norm": ce_norm,
                    "margin_norm": margin_norm,
                }
            )
            student.zero_grad(set_to_none=True)
        inputs[run_name] = {
            "parent_config": str(parent_config),
            "checkpoint_sha256": _sha256(checkpoint),
            "mask_sha256": _sha256(mask_path),
            "feature_meta_sha256": _sha256(feature_meta),
            "feature_rows_sha256": _sha256(feature_rows),
            "positive_teacher_margin_count": len(positive_margins),
            "calibration_sample_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "quantiles": gamma,
        }
    base = torch.tensor([float(x["base_advkd_norm"]) for x in measurements])
    ce = torch.tensor([float(x["advce_norm"]) for x in measurements])
    margin = torch.tensor([float(x["margin_norm"]) for x in measurements])
    if (
        not bool(torch.isfinite(torch.stack([base, ce, margin])).all())
        or bool((ce <= 0).any())
        or bool((margin <= 0).any())
    ):
        raise CleanWrongMarginCalibrationError("calibration has a non-finite or zero gradient denominator")
    advce_scale = float(torch.median(base / ce).item())
    margin_scale = float(torch.median(base / margin).item())
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_margin_calibration_v1",
        "status": "complete_no_update",
        "tau": 2.0,
        "target_ratio": 0.25,
        "beta_advce": 0.25 * advce_scale,
        "margin_coefficient": 0.25 * margin_scale,
        "device": str(device_obj),
        "inputs": inputs,
        "measurements": measurements,
        "provenance": {
            "config_sha256": _sha256(config_path),
            "git": collect_git_state(root),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifact_sha = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(artifact_sha + "\n", encoding="utf-8")
    result["artifact_sha256"] = artifact_sha
    return result
