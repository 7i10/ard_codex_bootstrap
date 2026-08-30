#!/usr/bin/env python3
"""Read-only BASE reconvergence and I100 Student-history validity analysis.

The script intentionally consumes only existing JSONL metrics, endpoint rows,
and epoch-boundary checkpoints.  It never trains a model or generates an
attack.  Large per-sample arrays remain in the ignored output directory; the
tracked JSON reports contain lineage and compact summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
N_TRAIN = 45_000
SEEDS = ("dev-1", "dev-2", "confirm-a", "confirm-b", "confirm-c")
DEV = ("dev-1", "dev-2")
CONFIRM = ("confirm-a", "confirm-b", "confirm-c")
CUTOFFS = (49, 99, 149)
STATE_EPOCHS = (49, 99, 149, 199)
SOURCE_CACHE = ROOT / ".cache/analysis/ert-rslad-student-history-v1/sources"
OLD_CACHE = ROOT / ".cache/analysis/ert-rslad-five-seed-stochasticity-v1/sources"
ENDPOINT_INVENTORY = ROOT / "docs/experiments/ert_rslad_five_seed_artifact_inventory_v1.json"
ATTACK_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
SPLIT_SHA = "16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty metrics: {path}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(epochs[0], epochs[-1] + 1)) or len(set(epochs)) != len(epochs):
        raise ValueError(f"metrics are not contiguous/unique: {path}")
    for row in rows:
        for key in ("val_pgd_accuracy", "val_clean_accuracy"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"non-finite {key}: {path}")
    return rows


def _dev_checkpoint(seed: str, cutoff: int) -> Path:
    s = 1 if seed == "dev-1" else 2
    suffix = "r2" if s == 1 else "r1"
    crop = Path(
        f"/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        f"ert-rslad-static-trajstab-v1/cropshift-s{s}-{suffix}"
    )
    if cutoff < 100:
        return crop / f"epoch-{cutoff:03d}.pt"
    suffix = Path(
        f"/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/idbh-s100-s{s}"
    )
    return suffix / f"epoch-{cutoff:03d}.pt"


def _confirm_checkpoint(seed: str, cutoff: int) -> Path:
    if seed in {"confirm-a", "confirm-b"} and cutoff >= 100:
        root = (
            Path(f"/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/unseen-{seed}-i100-suffix")
            / "outputs/student"
        )
        return root / f"epoch-{cutoff:03d}.pt"
    root = SOURCE_CACHE / seed
    if cutoff < 100:
        return root / "prefix" / f"epoch-{cutoff:03d}.pt"
    return root / "i100-suffix" / f"epoch-{cutoff:03d}.pt"


def checkpoint_path(seed: str, cutoff: int) -> Path:
    path = _dev_checkpoint(seed, cutoff) if seed.startswith("dev-") else _confirm_checkpoint(seed, cutoff)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_state(path: Path, scientific_epoch: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload is not a mapping: {path}")
    if payload.get("epoch") != scientific_epoch - 1 or payload.get("epoch_boundary") != "end":
        raise ValueError(
            f"epoch contract mismatch for {path}: payload={payload.get('epoch')} expected={scientific_epoch - 1}"
        )
    state = payload.get("sample_state")
    if not isinstance(state, dict) or state.get("format_version") != 3 or state.get("pending") != []:
        raise ValueError(f"incomplete format-v3 state: {path}")
    records = state.get("records")
    if not isinstance(records, dict) or len(records) != N_TRAIN:
        raise ValueError(f"sample-state count mismatch: {path}")
    ids = sorted(int(k) for k in records)
    if len(ids) != N_TRAIN or len(set(ids)) != N_TRAIN:
        raise ValueError(f"stable-ID count/uniqueness contract failed: {path}")
    fields: dict[str, list[Any]] = {
        k: []
        for k in (
            "true_label",
            "seen",
            "robust_correct_count",
            "previous_robust_correct",
            "forgetting_count",
            "current_correct_streak",
            "margin_ema",
            "last_margin",
            "history_statistics_complete",
        )
    }
    labels: list[int] = []
    for sample_id in ids:
        raw_id = str(sample_id)
        record = records[raw_id]
        if not isinstance(record, dict) or record.get("history_statistics_complete") is not True:
            raise ValueError(f"history statistics incomplete: {path} id={raw_id}")
        if int(record["seen"]) != scientific_epoch:
            raise ValueError(
                f"seen denominator mismatch: {path} id={raw_id} seen={record.get('seen')} expected={scientific_epoch}"
            )
        if not 0 <= int(record["robust_correct_count"]) <= int(record["seen"]):
            raise ValueError(f"invalid correctness counter: {path} id={raw_id}")
        label = int(record["true_label"])
        if not 0 <= label < 10:
            raise ValueError(f"invalid true label: {path} id={raw_id}")
        labels.append(label)
        for field in fields:
            fields[field].append(record[field])
    arrays = {
        "sample_id": np.asarray(ids, dtype=np.int64),
        "true_label": np.asarray(fields["true_label"], dtype=np.int64),
        "seen": np.asarray(fields["seen"], dtype=np.int64),
        "hits": np.asarray(fields["robust_correct_count"], dtype=np.int64),
        "current_correct": np.asarray(fields["previous_robust_correct"], dtype=bool),
        "forgetting": np.asarray(fields["forgetting_count"], dtype=np.int64),
        "current_streak": np.asarray(fields["current_correct_streak"], dtype=np.int64),
        "margin_ema": np.asarray(fields["margin_ema"], dtype=np.float64),
        "last_margin": np.asarray(fields["last_margin"], dtype=np.float64),
    }
    for key in ("margin_ema", "last_margin"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite state field {key}: {path}")
    return arrays, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "payload_epoch": int(payload["epoch"]),
        "scientific_epoch": scientific_epoch,
        "seen": int(arrays["seen"][0]),
        "tracker_run_id": payload.get("tracker_run_id"),
        "config_hash": payload.get("config_hash"),
        "global_step": payload.get("global_step"),
        "sample_state_sha256": canonical_sha(state),
        "sample_state_records": len(records),
        "sample_state_format_version": state.get("format_version"),
        "sample_state_pending": len(state.get("pending", [])),
        "history_statistics_complete": True,
        "teacher_identity_fields_present": all(
            records["0"].get(k) is not None for k in ("teacher_clean_correct", "teacher_adversarial_correct")
        ),
    }


def metric_sources() -> dict[str, dict[str, Path]]:
    d: dict[str, dict[str, Path]] = {}
    for seed, base_dir, crop_dir in (
        ("dev-1", OLD_CACHE / "dev/base-s1", OLD_CACHE / "dev/cropshift-s1"),
        ("dev-2", OLD_CACHE / "dev/base-s2", OLD_CACHE / "dev/cropshift-s2"),
        (
            "confirm-a",
            Path(
                "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/unseen-confirm-a-base-r2/outputs/student"
            ),
            SOURCE_CACHE / "confirm-a/prefix",
        ),
        (
            "confirm-b",
            Path(
                "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/unseen-confirm-b-base-r3/outputs/student"
            ),
            SOURCE_CACHE / "confirm-b/prefix",
        ),
        ("confirm-c", OLD_CACHE / "confirm-c/base", SOURCE_CACHE / "confirm-c/prefix"),
    ):
        suffix = OLD_CACHE / (
            "dev/i100-s1" if seed == "dev-1" else "dev/i100-s2" if seed == "dev-2" else f"{seed}/i100-suffix"
        )
        if seed == "confirm-a":
            suffix = OLD_CACHE / "confirm-a/crop-suffix"
        if seed == "confirm-b":
            suffix = Path(
                "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/unseen-confirm-b-i100-suffix/outputs/student"
            )
        if seed == "confirm-c":
            suffix = OLD_CACHE / "confirm-c/i100-suffix"
        d[seed] = {
            "base": base_dir / "epoch-metrics.jsonl",
            "crop": crop_dir / "epoch-metrics.jsonl",
            "suffix": suffix / "epoch-metrics.jsonl",
        }
    # confirmation A's suffix was copied under the historical crop-suffix name;
    # its metrics are I100 only in the prior cache's suffix-independent run.
    d["confirm-a"]["suffix"] = Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/unseen-confirm-a-i100-suffix/outputs/student/epoch-metrics.jsonl"
    )
    d["confirm-a"]["crop_suffix"] = OLD_CACHE / "confirm-a/crop-suffix/epoch-metrics.jsonl"
    d["confirm-b"]["crop_suffix"] = Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/unseen-confirm-b-crop-suffix/outputs/student/epoch-metrics.jsonl"
    )
    d["confirm-c"]["crop_suffix"] = OLD_CACHE / "confirm-c/crop-suffix/epoch-metrics.jsonl"
    return d


def hybrid_metrics(seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = metric_sources()[seed]
    base = read_metrics(paths["base"])
    crop_observed = read_metrics(paths["crop"])
    suffix = read_metrics(paths["suffix"])
    if seed.startswith("dev-"):
        crop = crop_observed
        prefix = crop[:100]
    else:
        prefix = crop_observed
        crop_suffix = read_metrics(paths["crop_suffix"])
        if [int(r["epoch"]) for r in crop_suffix] != list(range(100, 200)):
            raise ValueError(f"CROPSHIFT suffix coverage is not 100..199 for {seed}")
        crop = prefix + crop_suffix
    if [int(r["epoch"]) for r in prefix] != list(range(100)):
        raise ValueError(f"prefix coverage is not 0..99 for {seed}")
    if [int(r["epoch"]) for r in suffix] != list(range(100, 200)):
        raise ValueError(f"I100 suffix coverage is not 100..199 for {seed}")
    i100 = prefix + suffix
    if [int(r["epoch"]) for r in i100] != list(range(200)):
        raise ValueError(f"I100 hybrid coverage mismatch for {seed}")
    return base, crop, i100


def endpoint_base_check() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    inv = read_json(ENDPOINT_INVENTORY)
    out = []
    for cell in inv["cells"]:
        if cell["arm"] != "BASE" or int(cell["epoch"]) not in (99, 149, 199):
            continue
        path = ROOT / cell["path"]
        table = pq.read_table(path).to_pylist()
        if len(table) != 5000 or len({int(r["sample_id"]) for r in table}) != 5000:
            raise ValueError(f"endpoint stable-ID contract failed: {path}")
        out.append(
            {
                "seed": cell["seed"],
                "epoch": int(cell["epoch"]),
                "path": cell["path"],
                "rows_sha256": cell["sha256"],
                "row_count": len(table),
                "clean_accuracy": float(np.mean([bool(r["clean_correct"]) for r in table])),
                "robust_accuracy": float(np.mean([bool(r["robust_correct"]) for r in table])),
                "adversarial_margin_mean": float(np.mean([float(r["adversarial_probability_margin"]) for r in table])),
            }
        )
    if len(out) != 15:
        raise ValueError(f"expected 15 BASE endpoint cells, got {len(out)}")
    return sorted(out, key=lambda r: (r["seed"], r["epoch"]))


def base_reconvergence(
    metrics_by_seed: dict[str, list[dict[str, Any]]], endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    matrix = np.asarray([[float(r["val_pgd_accuracy"]) for r in metrics_by_seed[s]] for s in SEEDS])
    sd = matrix.std(axis=0, ddof=1)
    peak_index = int(np.argmax(sd))
    t_star = peak_index
    per_seed = []
    for idx, seed in enumerate(SEEDS):
        values = matrix[idx]
        best_idx = int(np.argmax(values))
        per_seed.append(
            {
                "seed": seed,
                "robust_at_t_star": float(values[t_star]),
                "robust_at_149": float(values[149]),
                "robust_at_199": float(values[199]),
                "gain_tstar_to_199": float(values[199] - values[t_star]),
                "gain_149_to_199": float(values[199] - values[149]),
                "best_robust": float(values[best_idx]),
                "best_epoch": best_idx,
                "best_minus_final": float(values[best_idx] - values[-1]),
            }
        )
    rho = float(spearmanr(matrix[:, t_star], matrix[:, 199] - matrix[:, t_star]).statistic)
    if not math.isfinite(rho):
        classification = "inconclusive"
    elif rho <= -0.5:
        classification = "catch-up"
    elif rho >= 0.5:
        classification = "RO-driven"
    else:
        classification = "mixed"
    return {
        "contract": {
            "metric": "val_pgd_accuracy",
            "seed_count": 5,
            "sample_sd_ddof": 1,
            "t_star_rule": "argmax five-seed BASE robust SD",
            "endpoint_attack_identity_sha256": ATTACK_SHA,
        },
        "t_star": t_star,
        "peak_sd": float(sd[t_star]),
        "peak_range": float(matrix[:, t_star].max() - matrix[:, t_star].min()),
        "final_sd": float(sd[199]),
        "final_range": float(matrix[:, 199].max() - matrix[:, 199].min()),
        "macro_reconvergence_ratio": float(sd[199] / sd[t_star]) if sd[t_star] else None,
        "per_seed": per_seed,
        "spearman_baseline_tstar_vs_final_gain": rho,
        "classification": classification,
        "endpoint_check": endpoint_rows,
        "conclusion": "descriptive only; not a causal attribution of robust overfitting or reconvergence",
    }


def state_inventory() -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ert_rslad_student_history_inventory",
        "expected_train_count": N_TRAIN,
        "cutoffs": list(STATE_EPOCHS),
        "feature_cutoffs": list(CUTOFFS),
        "seeds": list(SEEDS),
        "checkpoints": [],
        "lineage": {},
    }
    states: dict[str, dict[int, np.ndarray]] = {seed: {} for seed in SEEDS}
    ids_ref: np.ndarray | None = None
    labels_ref: np.ndarray | None = None
    for seed in SEEDS:
        for cutoff in STATE_EPOCHS:
            path = checkpoint_path(seed, cutoff)
            arr, meta = extract_state(path, cutoff)
            if labels_ref is None:
                ids_ref = arr["sample_id"].copy()
                labels_ref = arr["true_label"].copy()
            elif not np.array_equal(ids_ref, arr["sample_id"]) or not np.array_equal(labels_ref, arr["true_label"]):
                raise ValueError(f"stable ID/label mapping drift: {seed} cutoff {cutoff}")
            states[seed][cutoff] = arr
            inventory["checkpoints"].append({"seed": seed, "cutoff": cutoff, **meta})
    inventory["stable_id_label_sha256"] = canonical_sha(
        {
            "ids": states[SEEDS[0]][STATE_EPOCHS[0]]["sample_id"].tolist(),
            "labels": labels_ref.tolist() if labels_ref is not None else [],
        }
    )
    inventory["sample_state_contract"] = {
        "format_version": 3,
        "epoch_boundary": "end",
        "observation_semantics": "detached FP32 one valid observation per sample per epoch",
        "correctness": "Student robust correctness under training attack",
        "teacher_fields": "snapshot fields are present but not used as predictive features",
        "ema_decay": 0.9,
        "actual_seen_denominator": True,
    }
    inventory["checkpoint_sha256_aggregate"] = canonical_sha(
        [(r["seed"], r["cutoff"], r["sha256"]) for r in inventory["checkpoints"]]
    )
    return inventory, states


def features(arr: dict[str, np.ndarray], family: str) -> np.ndarray:
    current = arr["current_correct"].astype(np.float64)
    margin = arr["last_margin"]
    hist = np.column_stack(
        (
            arr["hits"] / arr["seen"],
            arr["forgetting"] / arr["seen"],
            arr["margin_ema"],
            arr["current_streak"] / arr["seen"],
        )
    )
    if family == "P0":
        return current[:, None]
    if family == "P1":
        return margin[:, None]
    if family == "P2":
        return np.column_stack((current, margin))
    if family == "P3":
        return hist
    if family == "P4":
        return np.column_stack((current, margin, hist))
    raise ValueError(family)


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z_train = (x_train - mean) / std
    z_eval = (x_eval - mean) / std
    aug = np.column_stack((np.ones(len(z_train)), z_train))
    reg = np.eye(aug.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(aug.T @ aug + alpha * reg, aug.T @ y_train)
    return np.column_stack((np.ones(len(z_eval)), z_eval)) @ coef


def logistic_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, c: float = 1.0) -> np.ndarray:
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z_train = (x_train - mean) / std
    z_eval = (x_eval - mean) / std
    aug = np.column_stack((np.ones(len(z_train)), z_train))
    penalty = np.r_[0.0, np.ones(z_train.shape[1])] / c

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        logits = aug @ w
        loss = float(np.logaddexp(0.0, logits).sum() - (y_train * logits).sum() + 0.5 * (penalty * w * w).sum())
        grad = aug.T @ (expit(logits) - y_train) + penalty * w
        return loss, grad

    result = minimize(
        lambda w: objective(w),
        np.zeros(aug.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return expit(np.column_stack((np.ones(len(z_eval)), z_eval)) @ result.x)


def auc_binary(y: np.ndarray, score: np.ndarray) -> float | None:
    y = y.astype(int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    """Deterministic PR-AUC/average precision for a binary target."""
    y = y.astype(int)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    positions = np.arange(1, len(y) + 1)
    return float(np.sum((cumulative / positions) * sorted_y) / positives)


def predictive(states: dict[str, dict[int, dict[str, np.ndarray]]]) -> dict[str, Any]:
    families = ("P0", "P1", "P2", "P3", "P4")
    out: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ert_rslad_student_history_predictive_validity",
        "cutoffs": list(CUTOFFS),
        "primary_target": "1 - (robust_correct_count_199 - robust_correct_count_149) / (seen_199 - seen_149),",
        "feature_families": {
            "P0": "current robust correctness",
            "P1": "current last robust margin",
            "P2": "current correctness + current margin",
            "P3": "inclusive correctness frequency + forgetting rate + margin EMA + current correct streak rate",
            "P4": "P2 + P3",
        },
        "ridge_alpha": 1.0,
        "logistic_l2_C": 1.0,
        "fit_seeds": list(DEV),
        "evaluation_seeds": list(CONFIRM),
        "results": [],
    }
    for cutoff in CUTOFFS:
        y_by_seed: dict[str, np.ndarray] = {}
        wrong_by_seed: dict[str, np.ndarray] = {}
        for seed in SEEDS:
            before, after = states[seed][149], states[seed][199]
            denominator = after["seen"] - before["seen"]
            if not np.all(denominator == 50):
                raise ValueError(f"future denominator is not 50: {seed}")
            y_by_seed[seed] = 1.0 - (after["hits"] - before["hits"]) / denominator
            wrong_by_seed[seed] = (~after["current_correct"]).astype(int)
        for family in families:
            x_train = np.concatenate([features(states[s][cutoff], family) for s in DEV])
            y_train = np.concatenate([y_by_seed[s] for s in DEV])
            train_pred = ridge_fit_predict(x_train, y_train, x_train)
            for split, seeds in (("dev_fit", DEV), ("confirmation", CONFIRM)):
                for seed in seeds:
                    pred = (
                        train_pred
                        if split == "dev_fit" and seed in ("pooled",)
                        else ridge_fit_predict(x_train, y_train, features(states[seed][cutoff], family))
                    )
                    y = y_by_seed[seed]
                    rho = float(spearmanr(pred, y).statistic)
                    out["results"].append(
                        {
                            "cutoff": cutoff,
                            "family": family,
                            "task": "ridge_future_failure",
                            "split": split,
                            "seed": seed,
                            "n": len(y),
                            "spearman": rho,
                            "target_mean": float(y.mean()),
                            "prediction_mean": float(pred.mean()),
                        }
                    )
            if family in ("P2", "P4"):
                for seed in CONFIRM:
                    try:
                        x_eval = features(states[seed][cutoff], family)
                        prob = logistic_fit_predict(x_train, np.concatenate([wrong_by_seed[s] for s in DEV]), x_eval)
                        y = wrong_by_seed[seed]
                        out["results"].append(
                            {
                                "cutoff": cutoff,
                                "family": family,
                                "task": "logistic_final_robust_wrong",
                                "split": "confirmation",
                                "seed": seed,
                                "n": len(y),
                                "roc_auc": auc_binary(y, prob),
                                "pr_auc": average_precision(y, prob),
                                "brier": float(np.mean((prob - y) ** 2)),
                                "prevalence": float(y.mean()),
                            }
                        )
                    except RuntimeError as exc:
                        out["results"].append(
                            {
                                "cutoff": cutoff,
                                "family": family,
                                "task": "logistic_final_robust_wrong",
                                "split": "confirmation",
                                "seed": seed,
                                "status": "unavailable",
                                "reason": str(exc),
                            }
                        )
    # Fixed support gate is evaluated only at the primary cutoff and only on
    # confirmation rows: no family/threshold is selected from outcomes.
    primary = [
        r
        for r in out["results"]
        if r.get("task") == "ridge_future_failure" and r.get("split") == "confirmation" and r["cutoff"] == 99
    ]
    by_family = {f: [r["spearman"] for r in primary if r["family"] == f] for f in families}
    p2, p4 = np.asarray(by_family["P2"]), np.asarray(by_family["P4"])
    diff = p4 - p2
    out["primary_gate"] = {
        "cutoff": 99,
        "metric": "Spearman(predicted future failure, observed future failure)",
        "P4_minus_P2_by_confirmation_seed": dict(zip(CONFIRM, diff.tolist(), strict=True)),
        "mean_difference": float(diff.mean()),
        "all_positive": bool(np.all(diff > 0)),
        "strong_support": bool(np.all(diff > 0) and diff.mean() >= 0.02),
        "partial_support": bool(diff.mean() > 0 and not (np.all(diff > 0) and diff.mean() >= 0.02)),
        "decision": "strong_support"
        if np.all(diff > 0) and diff.mean() >= 0.02
        else "partial_support"
        if diff.mean() > 0
        else "no_added_value",
    }
    # Cross-seed final train-state sensitivity is a descriptive secondary.
    final = np.column_stack([states[s][199]["current_correct"] for s in SEEDS])
    k = final.sum(axis=1)
    out["cross_seed_final_train_sensitivity"] = {
        "k_histogram": {str(i): int(np.sum(k == i)) for i in range(6)},
        "all_seed_correct_fraction": float(np.mean(k == 5)),
        "all_seed_wrong_fraction": float(np.mean(k == 0)),
        "seed_sensitive_fraction": float(np.mean((k >= 1) & (k <= 4))),
        "highly_sensitive_fraction": float(np.mean((k == 2) | (k == 3))),
    }
    return out


def global_v2(metric_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_epoch: list[dict[str, Any]] = []
    for arm in ("BASE", "CROPSHIFT", "I100"):
        for epoch in range(200):
            vals = [float(metric_rows[s][arm][epoch]["val_pgd_accuracy"]) for s in SEEDS]
            cleans = [float(metric_rows[s][arm][epoch]["val_clean_accuracy"]) for s in SEEDS]
            by_epoch.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "n_seeds": 5,
                    "seeds": list(SEEDS),
                    "mean_robust": float(np.mean(vals)),
                    "sd_robust": float(np.std(vals, ddof=1)),
                    "range_robust": float(max(vals) - min(vals)),
                    "mean_clean": float(np.mean(cleans)),
                }
            )
    return {
        "schema_version": 2,
        "kind": "ert_rslad_five_seed_global_stochasticity",
        "source_git_sha": git_sha(),
        "seeds": list(SEEDS),
        "arms": ["BASE", "CROPSHIFT", "I100"],
        "coverage": {a: {s: 200 for s in SEEDS} for a in ("BASE", "CROPSHIFT", "I100")},
        "by_epoch": by_epoch,
        "recovered_confirmation_prefix_dense_metrics": True,
        "previous_artifact": "docs/experiments/ert_rslad_five_seed_global_stochasticity_v1.json",
        "no_imputation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/analysis/ert-rslad-student-history-v1/outputs"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for seed in SEEDS:
        base, crop, i100 = hybrid_metrics(seed)
        metrics[seed] = {"BASE": base, "CROPSHIFT": crop, "I100": i100}
    endpoint_rows = endpoint_base_check()
    base_diag = base_reconvergence({s: metrics[s]["BASE"] for s in SEEDS}, endpoint_rows)
    inventory, states = state_inventory()
    prediction = predictive(states)
    recovery = {
        "schema_version": 1,
        "kind": "ert_rslad_confirmation_prefix_dense_metric_recovery",
        "source_git_sha": git_sha(),
        "metric_contract": "epoch-metrics.jsonl val_pgd_accuracy/val_clean_accuracy, epochs 0..99 contiguous",
        "seeds": list(CONFIRM),
        "recovered": [],
        "previous_v1_limitation": (
            "confirmation CROPSHIFT prefix dense metrics were unavailable in the prior local inventory"
        ),
        "recovery_method": (
            "targeted Ferret paths recorded in historical endpoint manifests; rsync checksum verification"
        ),
        "no_training": True,
    }
    for seed in CONFIRM:
        path = metric_sources()[seed]["crop"]
        rows = read_metrics(path)
        recovery["recovered"].append(
            {
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "row_count": len(rows),
                "epoch_start": int(rows[0]["epoch"]),
                "epoch_end": int(rows[-1]["epoch"]),
                "source": "Ferret historical prefix run",
            }
        )
    inventory["source_git_sha"] = git_sha()
    inventory["metric_sources"] = {
        s: {k: {"path": str(v.resolve()), "sha256": sha256(v)} for k, v in metric_sources()[s].items()} for s in SEEDS
    }
    global_art = global_v2(metrics)
    for name, data in (
        ("ert_rslad_base_reconvergence_ro_diagnostic_v1.json", base_diag),
        ("ert_rslad_confirmation_prefix_dense_metric_recovery_v1.json", recovery),
        ("ert_rslad_student_history_inventory_v1.json", inventory),
        ("ert_rslad_student_history_predictive_validity_v1.json", prediction),
        ("ert_rslad_five_seed_global_stochasticity_v2.json", global_art),
    ):
        (args.output_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Compact trajectory table used by the human report and later review.
    lines = ["arm,seed,epoch,clean,robust"]
    for seed in SEEDS:
        for arm in ("BASE", "CROPSHIFT", "I100"):
            for row in metrics[seed][arm]:
                lines.append(
                    f"{arm},{seed},{int(row['epoch'])},{float(row['val_clean_accuracy']):.10f},{float(row['val_pgd_accuracy']):.10f}"
                )
    (args.output_dir / "global_trajectory_v2.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "t_star": base_diag["t_star"],
                "base_final_sd": base_diag["final_sd"],
                "history_checkpoints": len(inventory["checkpoints"]),
                "confirmation_prefix_recovered": len(recovery["recovered"]),
                "primary_gate": prediction["primary_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
