#!/usr/bin/env python3
"""Point report for the fixed Chen CE/KL x PGD10/20 replay matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONDITIONS = ("ce_pgd10", "ce_pgd20", "kl_pgd10", "kl_pgd20")
EPOCHS = (189, 194, 199)


def _mask_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("failure masks must be non-empty, matching vectors")
    intersection = float(np.logical_and(first, second).sum())
    union = float(np.logical_or(first, second).sum())
    p = float(first.mean())
    q = float(second.mean())
    jaccard = intersection / union if union else 1.0
    null_jaccard = (p * q) / (p + q - p * q) if p + q - p * q else 1.0
    chance_adjusted = (jaccard - null_jaccard) / (1.0 - null_jaccard) if null_jaccard < 1.0 else 1.0
    agree = float((first == second).mean())
    expected_agree = p * q + (1 - p) * (1 - q)
    kappa = (agree - expected_agree) / (1 - expected_agree) if expected_agree < 1.0 else 1.0
    return {
        "n": float(first.size),
        "failure_prevalence_first": p,
        "failure_prevalence_second": q,
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
        "chance_null_jaccard": null_jaccard,
        "chance_adjusted_jaccard": chance_adjusted,
        "cohen_kappa": kappa,
    }


def _rank_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_rank = pd.Series(first).rank(method="average").to_numpy()
    second_rank = pd.Series(second).rank(method="average").to_numpy()
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _read(path: Path, *, expected_epochs: tuple[int, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"sample_id", "class_id", "epoch", "student_robust_correct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if tuple(sorted(frame["epoch"].unique())) != expected_epochs:
        raise ValueError(f"{path} epoch coverage is not exactly {expected_epochs}")
    if frame["sample_id"].duplicated().any():
        # A sample occurs once per epoch, but never twice in one epoch.
        if frame.duplicated(["sample_id", "epoch"]).any():
            raise ValueError(f"{path} has duplicate sample/epoch rows")
    return frame.sort_values(["epoch", "sample_id"], kind="mergesort").reset_index(drop=True)


def _condition_paths(root: Path, seed: str, condition: str) -> Path:
    if condition == "ce_pgd20":
        return root / f"{seed}-ce20.parquet"
    return root / seed / condition / "strong-observations.parquet"


def _condition_frame(root: Path, seed: str, condition: str, existing_ce20: Path | None) -> pd.DataFrame:
    path = (
        existing_ce20
        if condition == "ce_pgd20" and existing_ce20 is not None
        else _condition_paths(root, seed, condition)
    )
    return _read(path, expected_epochs=EPOCHS)


def build_report(*, root: Path, l2_ce20: Path, l4_ce20: Path) -> dict[str, Any]:
    by_seed: dict[str, dict[str, pd.DataFrame]] = {}
    for seed, ce20 in (("L2", l2_ce20), ("L4", l4_ce20)):
        by_seed[seed] = {condition: _condition_frame(root, seed, condition, ce20) for condition in CONDITIONS}
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for epoch in EPOCHS:
            l2 = by_seed["L2"][condition].query("epoch == @epoch").sort_values("sample_id")
            l4 = by_seed["L4"][condition].query("epoch == @epoch").sort_values("sample_id")
            if not np.array_equal(l2["sample_id"].to_numpy(), l4["sample_id"].to_numpy()):
                raise ValueError(f"stable sample universe mismatch for {condition} epoch {epoch}")
            metrics = _mask_metrics(~l2["student_robust_correct"].to_numpy(), ~l4["student_robust_correct"].to_numpy())
            rows.append({"condition": condition, "epoch": epoch, **metrics})
        left = by_seed["L2"][condition].pivot(index="sample_id", columns="epoch", values="student_robust_correct")
        right = by_seed["L4"][condition].pivot(index="sample_id", columns="epoch", values="student_robust_correct")
        left_frequency = (~left[list(EPOCHS)]).sum(axis=1).to_numpy()
        right_frequency = (~right[list(EPOCHS)]).sum(axis=1).to_numpy()
        rows.append(
            {
                "condition": condition,
                "epoch": "frequency",
                "failure_frequency_spearman": _rank_corr(left_frequency, right_frequency),
                "failure_frequency_pearson": float(np.corrcoef(left_frequency, right_frequency)[0, 1]),
            }
        )
    return {
        "contract": "ffnr_attack_factorial_report_v1",
        "epochs": list(EPOCHS),
        "conditions": list(CONDITIONS),
        "definitions": {
            "failure": "not student_robust_correct",
            "chance_adjusted_jaccard": "(observed Jaccard - independent-prevalence null) / (1 - null)",
            "frequency_correlation": "per-sample count of failures over epochs 189,194,199",
        },
        "rows": rows,
    }


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# FFNR CE/KL × PGD10/20 factorial",
        "",
        "Point estimates only; no bootstrap was preregistered for this diagnostic.",
        "",
        "| condition | epoch | L2 failure | L4 failure | Jaccard | chance-adjusted | κ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        if row["epoch"] == "frequency":
            continue
        lines.append(
            f"| {row['condition']} | {row['epoch']} | {row['failure_prevalence_first']:.4f} | "
            f"{row['failure_prevalence_second']:.4f} | {row['jaccard']:.4f} | "
            f"{row['chance_adjusted_jaccard']:.4f} | {row['cohen_kappa']:.4f} |"
        )
    lines += ["", "| condition | frequency Spearman | frequency Pearson |", "|---|---:|---:|"]
    for row in report["rows"]:
        if row["epoch"] == "frequency":
            lines.append(
                f"| {row['condition']} | {row['failure_frequency_spearman']:.4f} | "
                f"{row['failure_frequency_pearson']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--l2-ce20", type=Path, required=True)
    parser.add_argument("--l4-ce20", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(root=args.root, l2_ce20=args.l2_ce20, l4_ce20=args.l4_ce20)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown.write_text(_render(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json), "markdown": str(args.markdown)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
