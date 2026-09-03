#!/usr/bin/env python3
"""Read-only local boundary geometry replay and cross-seed audit.

The replay uses the exact epoch-99 I100 parents and the already registered
validation CE-PGD20 random-start stream.  It stores only scalar geometry
summaries; input-gradient tensors are discarded before the next batch.  The
analysis subcommand joins those scalars to existing I100_CONTROL endpoint
rows and performs fixed, cross-seed descriptive prediction tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher

ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
TEACHER_SHA = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
ENDPOINT_ATTACK_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
REPLAY_PROTOCOL = "registered_validation_ce20_batch_keyed_v1: evaluation_attack + batch_index"
SPLIT_IDENTITY = "16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4"
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def ids_sha(ids: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(int(x) for x in ids), separators=(",", ":")).encode()).hexdigest()


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def _logit_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, labels[:, None]).squeeze(1)


def _pair_margin(logits: torch.Tensor, labels: torch.Tensor, rivals: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, labels[:, None]).squeeze(1) - logits.gather(1, rivals[:, None]).squeeze(1)


def _probability_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=1)
    true = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong = probabilities.scatter(1, labels[:, None], 0.0).amax(dim=1)
    return true - wrong


def _rivals(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    masked = logits.detach().clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return masked.argmax(dim=1)


def _input_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, rivals: torch.Tensor
) -> torch.Tensor:
    pixels = inputs.detach().float().requires_grad_(True)
    with torch.autocast(device_type=pixels.device.type, enabled=False):
        logits = model(pixels).float()
        margins = _pair_margin(logits, labels, rivals)
    gradient = torch.autograd.grad(margins.sum(), pixels, only_inputs=True, create_graph=False)[0]
    return gradient.detach()


def _gradient_scalars(gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = gradient.float().flatten(1)
    return flat.abs().sum(dim=1), flat.norm(p=2, dim=1)


def _check_parent(checkpoint: Path, seed: str) -> dict[str, Any]:
    expected = PARENT_SHA[seed]
    actual = sha256(checkpoint)
    if actual != expected:
        raise ValueError(f"{seed}: parent SHA mismatch: {actual}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 99 or payload.get("epoch_boundary") != "end":
        raise ValueError(f"{seed}: parent is not the exact e99 end boundary")
    return payload


def _load_mask(mask_path: Path, seed: str, scope: str) -> tuple[list[int], dict[str, Any]]:
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    if payload.get("anchor_epoch") != 99 or payload.get("parent_checkpoint_sha256") != PARENT_SHA[seed]:
        raise ValueError(f"{seed}: mask lineage mismatch")
    if payload.get("teacher_checkpoint_sha256") != TEACHER_SHA:
        raise ValueError(f"{seed}: teacher lineage mismatch")
    key = "validation_s2_t1" if scope == "validation" else "s2_t1"
    record = payload.get("masks", {}).get(key)
    if not isinstance(record, dict) or not isinstance(record.get("selected_ids"), list):
        raise ValueError(f"{seed}: missing {scope} S2xT1 IDs")
    ids = [int(value) for value in record["selected_ids"]]
    if len(ids) != len(set(ids)) or ids_sha(ids) != record.get("selected_ids_sha256"):
        raise ValueError(f"{seed}: mask stable-ID digest mismatch")
    return ids, payload


def _config_with_local_overrides(config_path: Path, dataset_root: Path, teacher_path: Path) -> Any:
    return load_config(
        config_path,
        [
            f"dataset.root={dataset_root}",
            f"evaluation.dataset.root={dataset_root}",
            f"teacher.checkpoint={teacher_path}",
        ],
    )


def _loader(config: Any, scope: str) -> tuple[Any, DataLoader]:
    train, validation = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    dataset = train if scope == "train" else validation
    loader = DataLoader(
        dataset,
        batch_size=config.training.per_rank_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_indexed,
    )
    return dataset, loader


def _canary_gradient(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    adv: torch.Tensor,
    labels: torch.Tensor,
    rivals: torch.Tensor,
    student_batch_grad: torch.Tensor,
    teacher_batch_grad: torch.Tensor,
) -> dict[str, Any]:
    index = 0
    single_s = _input_gradient(student, adv[index : index + 1], labels[index : index + 1], rivals[index : index + 1])[0]
    single_t = _input_gradient(teacher, adv[index : index + 1], labels[index : index + 1], rivals[index : index + 1])[0]
    s_error = float((single_s - student_batch_grad[index]).abs().amax().item())
    t_error = float((single_t - teacher_batch_grad[index]).abs().amax().item())
    if s_error > 1e-5 or t_error > 1e-5:
        raise ValueError(f"batched/single gradient mismatch: student={s_error}, teacher={t_error}")
    return {"checked": True, "sample_index": index, "student_max_abs_error": s_error, "teacher_max_abs_error": t_error}


def replay(
    *,
    seed: str,
    scope: str,
    config_path: Path,
    checkpoint: Path,
    mask_path: Path,
    dataset_root: Path,
    teacher_path: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    if git_sha() is None:
        raise ValueError("unable to resolve source SHA")
    _check_parent(checkpoint, seed)
    selected_ids, mask_payload = _load_mask(mask_path, seed, scope)
    config = _config_with_local_overrides(config_path, dataset_root, teacher_path)
    if config.teacher is None or config.teacher.checkpoint_sha256 != TEACHER_SHA:
        raise ValueError(f"{seed}: Teacher config SHA mismatch")
    attack_config = config.method.selection_attack
    if (
        attack_config is None
        or attack_config.loss != "ce"
        or attack_config.steps != 20
        or attack_config.random_start_keying != "batch"
    ):
        raise ValueError("geometry anchor requires the registered batch-keyed CE-PGD20 attack")
    if attack_config.identity_sha256() != ENDPOINT_ATTACK_SHA:
        raise ValueError("geometry anchor attack identity mismatch")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch_device = torch.device(device)
    student = build_student(config.student, tier=config.tier)
    load_saved_student_checkpoint(checkpoint, student)
    teacher = build_teacher(config.teacher, tier=config.tier)
    student.to(torch_device).eval()
    teacher.to(torch_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    dataset, loader = _loader(config, scope)
    selected = set(selected_ids)
    positions = {int(sid): index for index, sid in enumerate(dataset.indices)}
    if not selected.issubset(positions):
        raise ValueError(f"{seed}: selected {scope} IDs outside dataset view")
    attack = LinfPGD(attack_config)
    rows: list[dict[str, Any]] = []
    canary: dict[str, Any] | None = None
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(torch_device)
        with torch.no_grad(), torch.autocast(device_type=torch_device.type, enabled=False):
            s_clean = student(batch.images.float()).detach().float()
            t_clean = teacher(batch.images.float()).detach().float()
        generator = torch.Generator(device=torch_device).manual_seed(int(config.seeds.evaluation_attack) + batch_index)
        attack_result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=None,
                source_ids=batch.sample_ids,
                epoch=99,
                attack_seed=int(config.seeds.evaluation_attack),
                stream_tag="selection_pgd",
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, float(attack_result.max_abs_delta))
        adv = attack_result.adversarial.float()
        with torch.no_grad(), torch.autocast(device_type=torch_device.type, enabled=False):
            s_adv = student(adv).detach().float()
            t_adv = teacher(adv).detach().float()
        labels = batch.labels.long()
        rivals = _rivals(s_adv, labels)
        s_adv_grad = _input_gradient(student, adv, labels, rivals)
        t_adv_grad = _input_gradient(teacher, adv, labels, rivals)
        s_clean_grad = _input_gradient(student, batch.images.float(), labels, rivals)
        t_clean_grad = _input_gradient(teacher, batch.images.float(), labels, rivals)
        if canary is None:
            canary = _canary_gradient(student, teacher, adv, labels, rivals, s_adv_grad, t_adv_grad)
        s_l1, s_l2 = _gradient_scalars(s_adv_grad)
        t_l1, t_l2 = _gradient_scalars(t_adv_grad)
        sc_l1, sc_l2 = _gradient_scalars(s_clean_grad)
        tc_l1, tc_l2 = _gradient_scalars(t_clean_grad)
        s_pair = _pair_margin(s_adv, labels, rivals)
        t_pair = _pair_margin(t_adv, labels, rivals)
        s_clean_pair = _pair_margin(s_clean, labels, rivals)
        t_clean_pair = _pair_margin(t_clean, labels, rivals)
        cosine_adv = F.cosine_similarity(s_adv_grad.flatten(1), t_adv_grad.flatten(1), dim=1, eps=EPS)
        cosine_clean_s = F.cosine_similarity(s_clean_grad.flatten(1), s_adv_grad.flatten(1), dim=1, eps=EPS)
        cosine_clean_t = F.cosine_similarity(t_clean_grad.flatten(1), t_adv_grad.flatten(1), dim=1, eps=EPS)
        selected_positions = [i for i, sid in enumerate(batch.sample_ids.tolist()) if int(sid) in selected]
        for i in selected_positions:
            values = {
                "scope": scope,
                "seed": seed,
                "sample_id": int(batch.sample_ids[i]),
                "true_label": int(labels[i]),
                "student_adv_correct": bool(s_adv.argmax(dim=1)[i] == labels[i]),
                "teacher_adv_correct": bool(t_adv.argmax(dim=1)[i] == labels[i]),
                "student_clean_logit_margin": float(s_clean_pair[i]),
                "student_adv_logit_margin": float(s_pair[i]),
                "teacher_clean_logit_margin": float(t_clean_pair[i]),
                "teacher_adv_logit_margin": float(t_pair[i]),
                "student_adv_probability_margin": float(_probability_margin(s_adv, labels)[i]),
                "teacher_adv_probability_margin": float(_probability_margin(t_adv, labels)[i]),
                "student_rival_class": int(rivals[i]),
                "student_grad_l1": float(s_l1[i]),
                "student_grad_l2": float(s_l2[i]),
                "teacher_grad_l1": float(t_l1[i]),
                "teacher_grad_l2": float(t_l2[i]),
                "gradient_l2_ratio_teacher_over_student": float(t_l2[i] / max(float(s_l2[i]), EPS)),
                "normal_cosine": float(cosine_adv[i]),
                "normal_mismatch": float(1.0 - cosine_adv[i]),
                "student_distance_inf": float(s_pair[i] / max(float(s_l1[i]), EPS)),
                "teacher_distance_inf": float(t_pair[i] / max(float(t_l1[i]), EPS)),
                "student_clean_distance_inf": float(s_clean_pair[i] / max(float(sc_l1[i]), EPS)),
                "teacher_clean_distance_inf": float(t_clean_pair[i] / max(float(tc_l1[i]), EPS)),
                "student_clean_adv_normal_cosine": float(cosine_clean_s[i]),
                "teacher_clean_adv_normal_cosine": float(cosine_clean_t[i]),
            }
            values["delta_distance_inf"] = values["teacher_distance_inf"] - values["student_distance_inf"]
            values["distance_ratio"] = values["student_distance_inf"] / (values["teacher_distance_inf"] + EPS)
            for key, value in values.items():
                if isinstance(value, float):
                    _finite(value, key)
            rows.append(values)
        del s_adv_grad, t_adv_grad, s_clean_grad, t_clean_grad
    if len(rows) != len(selected_ids) or {int(row["sample_id"]) for row in rows} != selected:
        raise ValueError(f"{seed}: incomplete selected scalar coverage {len(rows)} != {len(selected_ids)}")
    if max_abs_delta > float(attack_config.epsilon_value) + 1e-7:
        raise ValueError(f"{seed}: attack exceeded epsilon")
    rows.sort(key=lambda row: int(row["sample_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output, compression="zstd")
    meta = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_boundary_geometry_scalars_v1",
        "source_git_sha": git_sha(),
        "seed": seed,
        "scope": scope,
        "anchor_epoch": 99,
        "parent_checkpoint_sha256": sha256(checkpoint),
        "parent_checkpoint": str(checkpoint.resolve()),
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "mask_sha256": sha256(mask_path),
        "selected_count": len(selected_ids),
        "selected_ids_sha256": ids_sha(selected_ids),
        "dataset_scope_count": len(dataset),
        "endpoint_attack_identity_sha256": attack_config.identity_sha256(),
        "replay_protocol": REPLAY_PROTOCOL,
        "input_coordinate": "pixel_0_1; model normalization belongs to adapter",
        "gradient_dtype": "float32",
        "rival_definition": "Student strongest non-true logit on x_adv^S; shared with Teacher",
        "no_update": True,
        "raw_gradients_persisted": False,
        "rows_path": str(output.resolve()),
        "rows_sha256": sha256(output),
        "row_count": len(rows),
        "max_abs_delta": max_abs_delta,
        "batched_single_canary": canary,
        "mask_canonical_state_contract": mask_payload.get("canonical_state_contract"),
    }
    output.with_name(output.stem + ".json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _read_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate stable IDs in {path}")
    return result


def _load_endpoints(root: Path, seed: str, ids: set[int]) -> dict[int, dict[int, bool]]:
    run_dir = root / ("dev1-control" if seed == "dev-1" else "dev2-control-r6") / "endpoints"
    result: dict[int, dict[int, bool]] = {}
    for epoch in (104, 109, 114):
        endpoint_json = run_dir / f"e{epoch}-validation" / "endpoint.json"
        rows_path = run_dir / f"e{epoch}-validation" / "endpoint-sample-stats.parquet"
        if not endpoint_json.is_file() or not rows_path.is_file():
            raise FileNotFoundError(f"missing control endpoint e{epoch} for {seed}")
        meta = json.loads(endpoint_json.read_text(encoding="utf-8"))
        if meta.get("attack_identity_sha256") != ENDPOINT_ATTACK_SHA or meta.get("row_count") != 5000:
            raise ValueError(f"{seed} e{epoch}: endpoint identity mismatch")
        if meta.get("split_identity", {}).get("sample_id_label_sha256") != SPLIT_IDENTITY:
            raise ValueError(f"{seed} e{epoch}: validation split mismatch")
        if meta.get("rows_sha256") != sha256(rows_path):
            raise ValueError(f"{seed} e{epoch}: endpoint row SHA mismatch")
        rows = _read_rows(rows_path)
        result[epoch] = {sid: bool(rows[sid]["robust_correct"]) for sid in ids}
    return result


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    # Mann–Whitney with ties assigned half credit.
    comparisons = (positives[:, None] > negatives[None, :]).sum() + 0.5 * (
        positives[:, None] == negatives[None, :]
    ).sum()
    return float(comparisons / (len(positives) * len(negatives)))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    total = int(labels.sum())
    if total == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.flatnonzero(sorted_labels) + 1
    return float((cumulative[ranks - 1] / ranks).sum() / total)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def _ridge_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    train_x, test_x = _standardize(train_x, test_x)
    n_features = train_x.shape[1]
    design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    y = train_y.astype(np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ theta
        loss = np.logaddexp(0.0, logits).sum() - np.dot(y, logits) + 0.5 * alpha * np.dot(theta[1:], theta[1:])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        grad = design.T @ (probabilities - y)
        grad[1:] += alpha * theta[1:]
        return float(loss / len(y)), grad / len(y)

    result = minimize(
        lambda theta: objective(theta),
        np.zeros(n_features + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise ValueError(f"fixed ridge logistic failed: {result.message}")
    return 1.0 / (1.0 + np.exp(-np.clip(test_design @ result.x, -60.0, 60.0)))


FEATURES = {
    "student_margin": ("student_adv_logit_margin",),
    "teacher_margin": ("teacher_adv_logit_margin",),
    "student_teacher_margins": ("student_adv_logit_margin", "teacher_adv_logit_margin"),
    "student_distance": ("student_distance_inf",),
    "normal_mismatch": ("normal_mismatch",),
    "geometry_minimal": ("student_distance_inf", "normal_mismatch", "delta_distance_inf"),
    "geometry_distance_relation": (
        "student_distance_inf",
        "teacher_distance_inf",
        "delta_distance_inf",
        "distance_ratio",
    ),
    "margin_plus_geometry": (
        "student_adv_logit_margin",
        "teacher_adv_logit_margin",
        "student_distance_inf",
        "normal_mismatch",
        "delta_distance_inf",
    ),
}


def _univariate(rows: dict[int, dict[str, Any]], outcomes: Mapping[int, bool], feature: str) -> dict[str, Any]:
    common = [(float(row[feature]), int(outcomes[sid])) for sid, row in rows.items() if sid in outcomes]
    values = np.asarray([x for x, _ in common], dtype=np.float64)
    labels = np.asarray([y for _, y in common], dtype=np.int64)
    risk_sign = (
        -1.0
        if feature in {"student_adv_logit_margin", "teacher_adv_logit_margin", "student_distance_inf", "distance_ratio"}
        else 1.0
    )
    risk = risk_sign * values
    failure_values = values[labels == 1]
    correct_values = values[labels == 0]
    pooled = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    smd = (
        float((failure_values.mean() - correct_values.mean()) / pooled)
        if len(failure_values) and len(correct_values) and pooled > 0
        else None
    )
    rho = spearmanr(risk, labels).statistic if len(set(labels.tolist())) > 1 else None
    return {
        "feature": feature,
        "n": len(values),
        "failure_n": int(labels.sum()),
        "failure_median": float(np.median(failure_values)) if len(failure_values) else None,
        "correct_median": float(np.median(correct_values)) if len(correct_values) else None,
        "standardized_difference": smd,
        "risk_direction": "higher_risk" if risk_sign > 0 else "lower_feature_is_risk",
        "auroc": _auc(risk, labels),
        "average_precision": _average_precision(risk, labels),
        "spearman": float(rho) if rho is not None and math.isfinite(float(rho)) else None,
    }


def _predictor_rows(
    seed_fit: dict[int, dict[str, Any]],
    seed_eval: dict[int, dict[str, Any]],
    outcome_fit: Mapping[int, bool],
    outcome_eval: Mapping[int, bool],
    feature_keys: tuple[str, ...],
) -> dict[str, Any]:
    ids_fit = sorted(set(seed_fit) & set(outcome_fit))
    ids_eval = sorted(set(seed_eval) & set(outcome_eval))
    x_fit = np.asarray([[float(seed_fit[sid][key]) for key in feature_keys] for sid in ids_fit], dtype=np.float64)
    x_eval = np.asarray([[float(seed_eval[sid][key]) for key in feature_keys] for sid in ids_eval], dtype=np.float64)
    y_fit = np.asarray([int(outcome_fit[sid]) for sid in ids_fit], dtype=np.int64)
    y_eval = np.asarray([int(outcome_eval[sid]) for sid in ids_eval], dtype=np.int64)
    probabilities = _ridge_logistic(x_fit, y_fit, x_eval)
    return {
        "n_fit": len(ids_fit),
        "n_eval": len(ids_eval),
        "failure_fit": int(y_fit.sum()),
        "failure_eval": int(y_eval.sum()),
        "auroc": _auc(probabilities, y_eval),
        "average_precision": _average_precision(probabilities, y_eval),
        "spearman": float(spearmanr(probabilities, y_eval).statistic) if len(set(y_eval.tolist())) > 1 else None,
        "feature_columns": list(feature_keys),
        "ridge_alpha": 1.0,
        "standardization": "fit-seed mean/std only",
    }


def _cells(rows: dict[int, dict[str, Any]], outcomes: Mapping[int, bool]) -> dict[str, Any]:
    ids = sorted(set(rows) & set(outcomes))
    med_align = float(np.median([float(rows[sid]["normal_cosine"]) for sid in ids]))
    med_dist = float(np.median([float(rows[sid]["student_distance_inf"]) for sid in ids]))
    result: dict[str, Any] = {"median_normal_cosine": med_align, "median_student_distance_inf": med_dist, "cells": {}}
    for align_name, align_pred in (
        ("high_alignment", lambda r: float(r["normal_cosine"]) >= med_align),
        ("low_alignment", lambda r: float(r["normal_cosine"]) < med_align),
    ):
        for dist_name, dist_pred in (
            ("adequate_distance", lambda r: float(r["student_distance_inf"]) >= med_dist),
            ("low_student_distance", lambda r: float(r["student_distance_inf"]) < med_dist),
        ):
            members = [sid for sid in ids if align_pred(rows[sid]) and dist_pred(rows[sid])]
            result["cells"][f"{align_name}+{dist_name}"] = {
                "n": len(members),
                "failure_n": int(sum(bool(outcomes[sid]) for sid in members)),
                "failure_rate": float(sum(bool(outcomes[sid]) for sid in members) / len(members)) if members else None,
            }
    return result


def _contract(
    *,
    geometry_paths: Mapping[str, Path],
    geometry_meta: Mapping[str, Mapping[str, Any]],
    masks: Mapping[str, Mapping[str, Any]],
    endpoint_root: Path,
) -> dict[str, Any]:
    """Return the immutable replay/analysis contract used by every output."""
    return {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_boundary_geometry_contract_v1",
        "source_git_sha": git_sha(),
        "anchor_epoch": 99,
        "parents": dict(PARENT_SHA),
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "cohort": "canonical validation S2xT1 from fixed positive-margin q10 mask",
        "mask_artifacts": dict(masks),
        "geometry_scalar_artifacts": {
            seed: {
                "path": str(path.resolve()),
                "rows_sha256": str(geometry_meta[seed]["rows_sha256"]),
                "row_count": int(geometry_meta[seed]["row_count"]),
                "replay_protocol": geometry_meta[seed]["replay_protocol"],
            }
            for seed, path in geometry_paths.items()
        },
        "anchor_attack": {
            "identity_sha256": ENDPOINT_ATTACK_SHA,
            "protocol": REPLAY_PROTOCOL,
            "random_start_keying": "batch",
            "input_domain": "pixel_0_1",
            "epsilon": "8/255",
            "step_size": "2/255",
            "steps": 20,
            "student_mode": "eval",
            "teacher_mode": "eval",
        },
        "endpoint_root": str(endpoint_root.resolve()),
        "future_failure": "I100_CONTROL robust_wrong at existing CE-PGD20 e104/e109/e114",
        "rival_definition": "Student strongest non-true logit on x_adv^S; shared with Teacher",
        "distance_proxy": "pair_logit_margin / input_gradient_l1",
        "no_update": True,
        "raw_gradients_persisted": False,
        "fixed_ridge": {"alpha": 1.0, "fit_seed_standardization": True, "pooled_fit": False},
        "analysis_is_descriptive": True,
        "mask_label_note": (
            "Mask artifact descriptive label says sample_keyed_v1; registered replay metadata "
            "and regenerated anchor use batch-index CE seed stream."
        ),
    }


def analyze(
    *,
    geometry: Mapping[str, Path],
    mask_paths: Mapping[str, Path],
    endpoint_root: Path,
    output_json: Path,
    report: Path,
) -> dict[str, Any]:
    rows_by_seed = {seed: _read_rows(path) for seed, path in geometry.items()}
    geometry_meta: dict[str, Mapping[str, Any]] = {}
    for seed, path in geometry.items():
        metadata_path = path.with_name(path.stem + ".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{seed}: missing geometry metadata {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("rows_sha256") != sha256(path) or metadata.get("row_count") != len(rows_by_seed[seed]):
            raise ValueError(f"{seed}: geometry metadata/rows mismatch")
        if metadata.get("endpoint_attack_identity_sha256") != ENDPOINT_ATTACK_SHA:
            raise ValueError(f"{seed}: geometry attack identity mismatch")
        geometry_meta[seed] = metadata
    masks = {}
    outcomes: dict[str, dict[int, dict[int, bool]]] = {}
    for seed, mask_path in mask_paths.items():
        ids, _ = _load_mask(mask_path, seed, "validation")
        if set(ids) != set(rows_by_seed[seed]):
            raise ValueError(f"{seed}: geometry/mask coverage mismatch")
        masks[seed] = {"count": len(ids), "ids_sha256": ids_sha(ids), "mask_sha256": sha256(mask_path)}
        outcomes[seed] = _load_endpoints(endpoint_root, seed, set(ids))
    univariate: dict[str, Any] = {}
    predictors: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    for seed in ("dev-1", "dev-2"):
        univariate[seed] = {}
        cells[seed] = {}
        for epoch, labels in outcomes[seed].items():
            univariate[seed][str(epoch)] = [
                _univariate(rows_by_seed[seed], labels, feature)
                for feature in (
                    "student_adv_logit_margin",
                    "teacher_adv_logit_margin",
                    "student_distance_inf",
                    "normal_mismatch",
                    "delta_distance_inf",
                    "distance_ratio",
                )
            ]
            cells[seed][str(epoch)] = _cells(rows_by_seed[seed], labels)
    for epoch in (104, 109, 114):
        predictors[str(epoch)] = {}
        for name, feature_keys in FEATURES.items():
            predictors[str(epoch)][name] = {
                "dev-1_to_dev-2": _predictor_rows(
                    rows_by_seed["dev-1"],
                    rows_by_seed["dev-2"],
                    outcomes["dev-1"][epoch],
                    outcomes["dev-2"][epoch],
                    feature_keys,
                ),
                "dev-2_to_dev-1": _predictor_rows(
                    rows_by_seed["dev-2"],
                    rows_by_seed["dev-1"],
                    outcomes["dev-2"][epoch],
                    outcomes["dev-1"][epoch],
                    feature_keys,
                ),
            }
    geometry_gain: dict[str, Any] = {}
    for epoch in (104, 109, 114):
        geometry_gain[str(epoch)] = {}
        for direction in ("dev-1_to_dev-2", "dev-2_to_dev-1"):
            geometry_gain[str(epoch)][direction] = {
                "normal_mismatch_minus_student_margin": predictors[str(epoch)]["normal_mismatch"][direction]["auroc"]
                - predictors[str(epoch)]["student_margin"][direction]["auroc"],
                "student_distance_minus_student_margin": predictors[str(epoch)]["student_distance"][direction]["auroc"]
                - predictors[str(epoch)]["student_margin"][direction]["auroc"],
                "delta_distance_minus_student_margin": predictors[str(epoch)]["geometry_minimal"][direction]["auroc"]
                - predictors[str(epoch)]["student_margin"][direction]["auroc"],
                "minimal_geometry_minus_student_margin": predictors[str(epoch)]["geometry_minimal"][direction]["auroc"]
                - predictors[str(epoch)]["student_margin"][direction]["auroc"],
            }

    def stable_positive(key: str) -> bool:
        values = [
            geometry_gain[str(epoch)][direction][key]
            for epoch in (104, 109, 114)
            for direction in ("dev-1_to_dev-2", "dev-2_to_dev-1")
        ]
        return all(value > 0.0 for value in values)

    normal_supported = stable_positive("normal_mismatch_minus_student_margin")
    distance_supported = stable_positive("delta_distance_minus_student_margin")
    if normal_supported and distance_supported:
        decision = "BG3_NORMAL_AND_DISTANCE_SUPPORTED"
    elif normal_supported:
        decision = "BG1_NORMAL_MISMATCH_SUPPORTED"
    elif distance_supported:
        decision = "BG2_DISTANCE_GAP_SUPPORTED"
    elif any(
        geometry_gain[str(epoch)][direction][key] > 0.0
        for epoch in (104, 109, 114)
        for direction in ("dev-1_to_dev-2", "dev-2_to_dev-1")
        for key in ("normal_mismatch_minus_student_margin", "delta_distance_minus_student_margin")
    ):
        decision = "BG4_GEOMETRY_WEAK"
    else:
        decision = "BG5_NOT_SUPPORTED"
    contract = _contract(geometry_paths=geometry, geometry_meta=geometry_meta, masks=masks, endpoint_root=endpoint_root)
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_boundary_geometry_audit_v1",
        "status": "complete_read_only",
        "source_git_sha": contract["source_git_sha"],
        "anchor_epoch": 99,
        "parents": PARENT_SHA,
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "anchor_attack": {"identity_sha256": ENDPOINT_ATTACK_SHA, "protocol": REPLAY_PROTOCOL},
        "cohort": "canonical validation S2xT1 from fixed positive-margin q10 mask",
        "masks": masks,
        "contract_details": contract,
        "future_failure": "I100_CONTROL robust_wrong at existing CE-PGD20 e104/e109/e114",
        "univariate": univariate,
        "cross_seed_predictors": predictors,
        "geometry_added_value": geometry_gain,
        "geometry_cells": cells,
        "history_comparator": "not used; no new History reconstruction",
        "decision": decision,
        "decision_rationale": (
            "Support requires strictly positive cross-seed AUROC gain over Student margin in both "
            "fit→eval directions at all registered horizons; otherwise isolated gains are classified "
            "as weak and no causal intervention follows."
        ),
        "no_training": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# ERT I100 canonical S2×T1 boundary geometry audit",
        "",
        "Status: complete read-only analysis. No training, intervention, threshold tuning, new seed, "
        "official test, or AutoAttack was run.",
        "",
        "## Executive answer",
        "",
        "- Exact e99 I100 parents and fixed validation S2×T1 masks were used. The anchor is the "
        "registered Student CE-PGD20 point; replay metadata resolves its random start as batch-index.",
        "- Geometry uses the Student strongest non-true logit as a shared class pair. Input gradients "
        "are through pixel-space adapters; Teacher parameters are frozen and have no parameter grads.",
        "- Primary scalars are normal cosine/mismatch and first-order L∞ distance proxies "
        "`m_pair / ||g||_1`; these are not exact distances.",
        f"- Decision: **{decision}**. Geometry is descriptive; no geometry-based intervention follows.",
        "",
        "## Cohort and lineage",
        "",
        f"Parents: dev-1 `{PARENT_SHA['dev-1']}`, dev-2 `{PARENT_SHA['dev-2']}`. "
        f"Teacher `{TEACHER_SHA}`. Endpoint attack `{ENDPOINT_ATTACK_SHA}`. "
        f"Validation split `{SPLIT_IDENTITY}`.",
        "",
        "| seed | fixed validation S2×T1 n | mask IDs SHA |",
        "|---|---:|---|",
    ]
    for seed in ("dev-1", "dev-2"):
        lines.append(f"| {seed} | {masks[seed]['count']} | `{masks[seed]['ids_sha256']}` |")
    lines.extend(
        [
            "",
            "## Cross-seed predictor AUROC",
            "",
            "AUROC is descriptive; fit uses one seed and evaluates the other with fixed ridge "
            "logistic alpha=1.0 and fit-seed standardization.",
            "",
        ]
    )
    for epoch in (104, 109, 114):
        lines.extend([f"### e{epoch}", "", "| predictor | dev1→dev2 | dev2→dev1 |", "|---|---:|---:|"])
        for name in FEATURES:
            a = predictors[str(epoch)][name]["dev-1_to_dev-2"]["auroc"]
            b = predictors[str(epoch)][name]["dev-2_to_dev-1"]["auroc"]
            lines.append(
                f"| {name} | {a:.4f} | {b:.4f} |" if a is not None and b is not None else f"| {name} | n/a | n/a |"
            )
        lines.append("")
    lines.extend(
        ["## Geometry cells", "", "Cells use within-seed medians descriptively only; they are not selectors.", ""]
    )
    for seed in ("dev-1", "dev-2"):
        lines.extend([f"### {seed}", "", "| epoch | cell | n | failure rate |", "|---:|---|---:|---:|"])
        for epoch in (104, 109, 114):
            for cell, value in cells[seed][str(epoch)]["cells"].items():
                lines.append(
                    f"| {epoch} | {cell} | {value['n']} | {value['failure_rate']:.4f} |"
                    if value["failure_rate"] is not None
                    else f"| {epoch} | {cell} | 0 | n/a |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation and stop boundary",
            "",
            "Geometry does not establish that mismatch or distance causes future failure, nor that "
            "alignment/distance distillation improves robustness. No new loss, route, threshold, or "
            "training is started. Full scalar details and hashes are in the machine artifact.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--seed", choices=("dev-1", "dev-2"), required=True)
    replay_parser.add_argument("--scope", choices=("validation", "train"), default="validation")
    replay_parser.add_argument("--config", type=Path, required=True)
    replay_parser.add_argument("--checkpoint", type=Path, required=True)
    replay_parser.add_argument("--mask", type=Path, required=True)
    replay_parser.add_argument("--dataset-root", type=Path, required=True)
    replay_parser.add_argument("--teacher", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--device", default="cuda")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--dev1-geometry", type=Path, required=True)
    analyze_parser.add_argument("--dev2-geometry", type=Path, required=True)
    analyze_parser.add_argument("--dev1-mask", type=Path, required=True)
    analyze_parser.add_argument("--dev2-mask", type=Path, required=True)
    analyze_parser.add_argument("--endpoint-root", type=Path, required=True)
    analyze_parser.add_argument("--output-json", type=Path, required=True)
    analyze_parser.add_argument("--report", type=Path, required=True)
    analyze_parser.add_argument("--contract-json", type=Path)
    analyze_parser.add_argument("--univariate-json", type=Path)
    analyze_parser.add_argument("--crossseed-json", type=Path)
    analyze_parser.add_argument("--cells-json", type=Path)
    analyze_parser.add_argument("--combined-scalars", type=Path)
    args = parser.parse_args()
    if args.command == "replay":
        result = replay(
            seed=args.seed,
            scope=args.scope,
            config_path=args.config,
            checkpoint=args.checkpoint,
            mask_path=args.mask,
            dataset_root=args.dataset_root,
            teacher_path=args.teacher,
            output=args.output,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "rows": result["row_count"],
                    "rows_sha256": result["rows_sha256"],
                    "canary": result["batched_single_canary"],
                },
                sort_keys=True,
            )
        )
    else:
        result = analyze(
            geometry={"dev-1": args.dev1_geometry, "dev-2": args.dev2_geometry},
            mask_paths={"dev-1": args.dev1_mask, "dev-2": args.dev2_mask},
            endpoint_root=args.endpoint_root,
            output_json=args.output_json,
            report=args.report,
        )
        contract = result["contract_details"]
        for path, payload in (
            (args.contract_json, contract),
            (args.univariate_json, {"contract": contract, "per_seed": result["univariate"]}),
            (
                args.crossseed_json,
                {
                    "contract": contract,
                    "predictors": result["cross_seed_predictors"],
                    "geometry_added_value": result["geometry_added_value"],
                },
            ),
            (args.cells_json, {"contract": contract, "per_seed": result["geometry_cells"]}),
        ):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.combined_scalars is not None:
            all_rows: list[dict[str, Any]] = []
            for seed, path in (("dev-1", args.dev1_geometry), ("dev-2", args.dev2_geometry)):
                for row in _read_rows(path).values():
                    all_rows.append({"seed": seed, **row})
            args.combined_scalars.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(all_rows), args.combined_scalars, compression="zstd")
        print(json.dumps({"decision": result["decision"], "source_git_sha": result["source_git_sha"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
