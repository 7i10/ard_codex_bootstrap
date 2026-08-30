#!/usr/bin/env python3
"""Freeze the smallest preregistered Student-history predictor.

This is a read-only analysis over the already materialized epoch-boundary
sample-state checkpoints.  Development seeds alone determine the selected
feature subset and the frozen Ridge coefficients; confirmation seeds are
reported only after that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from analyze_ert_rslad_student_history import checkpoint_path, extract_state
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEV = ("dev-1", "dev-2")
CONFIRM = ("confirm-a", "confirm-b", "confirm-c")
ALL_SEEDS = DEV + CONFIRM
N_TRAIN = 45_000
CUTOFF = 99
TARGET_BEFORE = 149
TARGET_AFTER = 199

FEATURE_NAMES = (
    "correctness_frequency",
    "margin_ema",
    "forgetting_rate",
    "correct_streak_rate",
)
CANDIDATES: dict[str, tuple[int, ...]] = {
    "H1": (0,),
    "H2": (1,),
    "H3": (2,),
    "H4": (3,),
    "H5": (0, 1),
    "H6": (0, 2),
    "H7": (0, 3),
    "H8": (0, 1, 2),
    "H9": (0, 2, 1, 3),
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_states() -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    states: dict[str, dict[int, dict[str, np.ndarray]]] = {seed: {} for seed in ALL_SEEDS}
    checkpoints: list[dict[str, Any]] = []
    ids_ref: np.ndarray | None = None
    labels_ref: np.ndarray | None = None
    for seed in ALL_SEEDS:
        for epoch in (CUTOFF, TARGET_BEFORE, TARGET_AFTER):
            path = checkpoint_path(seed, epoch)
            state, metadata = extract_state(path, epoch)
            if ids_ref is None:
                ids_ref = state["sample_id"].copy()
                labels_ref = state["true_label"].copy()
            elif not np.array_equal(ids_ref, state["sample_id"]) or not np.array_equal(labels_ref, state["true_label"]):
                raise ValueError(f"stable-ID/label mapping drift: {seed} epoch {epoch}")
            states[seed][epoch] = state
            checkpoints.append({"seed": seed, "epoch": epoch, **metadata})
    return states, checkpoints


def _history_features(state: dict[str, np.ndarray]) -> np.ndarray:
    seen = state["seen"].astype(np.float64)
    return np.column_stack(
        (
            state["hits"] / seen,
            state["margin_ema"],
            state["forgetting"] / seen,
            state["current_streak"] / seen,
        )
    )


def _future_failure(states: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    before = states[TARGET_BEFORE]
    after = states[TARGET_AFTER]
    denominator = after["seen"] - before["seen"]
    if not np.all(denominator == TARGET_AFTER - TARGET_BEFORE):
        raise ValueError("future failure denominator is not 50 for every sample")
    return 1.0 - (after["hits"] - before["hits"]) / denominator


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict[str, Any]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    z = (x - mean) / std
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
    return {"mean": mean, "std": std, "coef": coef}


def _predict(fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - fit["mean"]) / fit["std"]
    return np.column_stack((np.ones(len(z)), z)) @ fit["coef"]


def _result_row(
    *,
    candidate: str,
    fit: dict[str, Any],
    states: dict[str, dict[int, dict[str, np.ndarray]]],
    seed: str,
    split: str,
) -> dict[str, Any]:
    x = _history_features(states[seed][CUTOFF])[:, CANDIDATES[candidate]]
    y = _future_failure(states[seed])
    pred = _predict(fit, x)
    rho = float(spearmanr(pred, y).statistic)
    if not np.isfinite(rho):
        raise ValueError(f"non-finite Spearman for {candidate}/{seed}")
    return {
        "candidate": candidate,
        "seed": seed,
        "split": split,
        "cutoff": CUTOFF,
        "target": "1-(hits_199-hits_149)/(seen_199-seen_149)",
        "n": int(len(y)),
        "spearman": rho,
        "target_mean": float(y.mean()),
        "prediction_mean": float(pred.mean()),
    }


def select_candidate(dev_by_candidate: dict[str, list[float]]) -> tuple[str, list[str]]:
    """Apply the frozen, dev-only minimality rule without touching confirmation rows."""
    h9_values = dev_by_candidate["H9"]
    h9_seed = dict(zip(DEV, h9_values, strict=True))
    h9_mean = float(np.mean(h9_values))
    eligible = []
    for candidate, values in dev_by_candidate.items():
        if len(CANDIDATES[candidate]) >= len(CANDIDATES["H9"]):
            continue
        if float(np.mean(values)) < h9_mean - 0.01:
            continue
        if any(value < h9_seed[seed] - 0.015 for seed, value in zip(DEV, values, strict=True)):
            continue
        eligible.append(candidate)
    if not eligible:
        return "H9", eligible
    return (
        sorted(
            eligible,
            key=lambda candidate: (
                len(CANDIDATES[candidate]),
                -float(np.mean(dev_by_candidate[candidate])),
                int(candidate[1:]),
            ),
        )[0],
        eligible,
    )


def analyze(states: dict[str, dict[int, dict[str, np.ndarray]]], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    fits: dict[str, dict[str, Any]] = {}
    for candidate, columns in CANDIDATES.items():
        x_train = np.concatenate([_history_features(states[seed][CUTOFF])[:, columns] for seed in DEV])
        y_train = np.concatenate([_future_failure(states[seed]) for seed in DEV])
        fit = _fit_ridge(x_train, y_train)
        fits[candidate] = fit
        results.extend(
            _result_row(candidate=candidate, fit=fit, states=states, seed=seed, split="dev") for seed in DEV
        )
        results.extend(
            _result_row(candidate=candidate, fit=fit, states=states, seed=seed, split="confirmation")
            for seed in CONFIRM
        )

    dev_by_candidate: dict[str, list[float]] = {
        candidate: [r["spearman"] for r in results if r["candidate"] == candidate and r["split"] == "dev"]
        for candidate in CANDIDATES
    }
    selected, eligible = select_candidate(dev_by_candidate)
    selected_fit = fits[selected]
    frozen = {
        "candidate": selected,
        "feature_names": [FEATURE_NAMES[index] for index in CANDIDATES[selected]],
        "columns": list(CANDIDATES[selected]),
        "ridge_alpha": 1.0,
        "standardization": "pooled development seeds only",
        "mean": selected_fit["mean"].tolist(),
        "std": selected_fit["std"].tolist(),
        "coef_intercept_and_standardized": selected_fit["coef"].tolist(),
    }
    return {
        "schema_version": 1,
        "kind": "ert_rslad_history_minimality",
        "source_git_sha": _git_sha(),
        "no_training": True,
        "selection": {
            "cutoff": CUTOFF,
            "target": "1-(hits_199-hits_149)/(seen_199-seen_149)",
            "fit_seeds": list(DEV),
            "confirmation_seeds": list(CONFIRM),
            "metric": "Spearman(predicted future failure, observed future failure)",
            "reference": "H9",
            "mean_tolerance": 0.01,
            "per_seed_tolerance": 0.015,
            "tie_break": "fewest features, then higher dev mean Spearman, then lower H number",
            "eligible_candidates": eligible,
            "selected_candidate": selected,
            "h9_dev_mean": float(np.mean(dev_by_candidate["H9"])),
            "h9_dev_by_seed": dict(zip(DEV, dev_by_candidate["H9"], strict=True)),
        },
        "feature_contract": {
            "H1": "robust_correct_count / seen",
            "H2": "margin_ema (SampleStateStore EMA decay 0.9)",
            "H3": "forgetting_count / seen",
            "H4": "current_correct_streak / seen",
            "state_boundary": "epoch-end state; one valid training-attack observation per sample per epoch",
            "future_target_window": "epochs 150-199 inclusive observations, represented by state 149 to state 199",
        },
        "candidate_features": {name: [FEATURE_NAMES[i] for i in cols] for name, cols in CANDIDATES.items()},
        "dev_results": [r for r in results if r["split"] == "dev"],
        "confirmation_results": [r for r in results if r["split"] == "confirmation"],
        "frozen_predictor": frozen,
        "frozen_predictor_sha256": _sha(frozen),
        "stable_id_sha256": _sha(
            {seed: states[seed][CUTOFF]["sample_id"].tolist() for seed in ALL_SEEDS}
        ),
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/ert_rslad_history_minimality_v1.json")
    args = parser.parse_args()
    states, checkpoints = _load_states()
    artifact = analyze(states, checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": artifact["selection"]["selected_candidate"],
                "eligible": artifact["selection"]["eligible_candidates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
