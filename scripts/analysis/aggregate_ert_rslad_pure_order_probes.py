#!/usr/bin/env python3
"""Aggregate pure-order probe telemetry and apply the preregistered gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SCHEDULE_IDS = tuple(f"SHUFFLE_PLUS_{index}" for index in range(8))
DESCRIPTORS = (
    "D1_batch_mean_risk_sd",
    "D2_within_batch_risk_sd_mean",
    "D3_high_risk_fraction_sd",
    "D4_lag1_batch_mean_risk_acf",
    "D5_hard_batch_longest_run",
    "D6_position_vs_batch_mean_risk_spearman",
)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for index in order[position:end]:
            result[index] = rank
        position = end
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    )
    return None if denominator == 0.0 else numerator / denominator


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be mappings: {path}")
    return rows


def _auc(rows: list[dict[str, Any]], field: str) -> float:
    points = [(int(row["epoch"]), float(row[field])) for row in rows]
    points.sort()
    if len(points) != 15 or [epoch for epoch, _ in points] != list(range(100, 115)):
        raise ValueError("probe metrics must contain exactly epochs 100--114")
    return (
        sum(
            (right_epoch - left_epoch) * (left_value + right_value) / 2.0
            # Adjacent point lists intentionally differ in length by one.
            # ``strict=True`` would reject every valid trajectory here.
            for (left_epoch, left_value), (right_epoch, right_value) in zip(points, points[1:])
        )
        / 14.0
    )


def aggregate_seed(root: Path, seed: int) -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    for schedule_id in SCHEDULE_IDS:
        run = root / f"{schedule_id.lower()}-s{seed}"
        metrics = _read_jsonl(run / "epoch-metrics.jsonl")
        descriptors = _read_jsonl(run / "ordering-telemetry" / "ordering-descriptors.jsonl")
        metrics = [row for row in metrics if 100 <= int(row.get("epoch", -1)) <= 114]
        descriptors = [row for row in descriptors if 100 <= int(row.get("epoch", -1)) <= 114]
        if len(metrics) != 15 or len(descriptors) != 15:
            raise ValueError(f"incomplete pure-order probe: seed={seed} schedule={schedule_id}")
        if len({int(row["epoch"]) for row in metrics}) != 15 or len({int(row["epoch"]) for row in descriptors}) != 15:
            raise ValueError(f"duplicate probe epochs: seed={seed} schedule={schedule_id}")
        for row in descriptors:
            if row.get("risk_definition") != "-margin_ema" or int(row.get("valid_sample_count", 0)) != 45000:
                raise ValueError(f"telemetry contract mismatch: {run}")
            if any(row.get(key) is None for key in DESCRIPTORS if key.startswith("D4")):
                # D4 can be undefined only for a degenerate batch sequence;
                # retain the row and make the association missing explicitly.
                continue
        by_epoch = {int(row["epoch"]): row for row in descriptors}
        summaries = {
            key: statistics.fmean(
                float(by_epoch[epoch][key]) for epoch in range(100, 115) if by_epoch[epoch].get(key) is not None
            )
            for key in DESCRIPTORS
            if any(by_epoch[epoch].get(key) is not None for epoch in range(100, 115))
        }
        schedules[schedule_id] = {
            "run": str(run.resolve()),
            "probe_auc": _auc(metrics, "train_robust_accuracy"),
            "robust_at_114": float(next(row for row in metrics if int(row["epoch"]) == 114)["train_robust_accuracy"]),
            "clean_at_114": float(next(row for row in metrics if int(row["epoch"]) == 114)["train_clean_accuracy"]),
            "gain_100_to_114": float(metrics[-1]["train_robust_accuracy"])
            - float(metrics[0]["train_robust_accuracy"]),
            "best_robust": max(float(row["train_robust_accuracy"]) for row in metrics),
            "best_minus_last": max(float(row["train_robust_accuracy"]) for row in metrics)
            - float(metrics[-1]["train_robust_accuracy"]),
            "descriptor_mean": summaries,
            "descriptor_rows": descriptors,
            "metrics_rows": metrics,
        }
    associations: dict[str, Any] = {}
    for descriptor in DESCRIPTORS:
        if not all(descriptor in schedules[schedule]["descriptor_mean"] for schedule in SCHEDULE_IDS):
            associations[descriptor] = {"rho": None, "reason": "undefined_descriptor"}
            continue
        x = [float(schedules[schedule]["descriptor_mean"][descriptor]) for schedule in SCHEDULE_IDS]
        y = [float(schedules[schedule]["probe_auc"]) for schedule in SCHEDULE_IDS]
        associations[descriptor] = {"rho": _spearman(x, y), "n": len(x), "descriptor_values": x, "probe_auc": y}
    return {"seed": seed, "schedules": schedules, "associations": associations}


def select_mechanism(per_seed: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = []
    for descriptor in DESCRIPTORS:
        rhos = [per_seed[str(seed)]["associations"][descriptor].get("rho") for seed in (1, 2)]
        if any(rho is None for rho in rhos):
            continue
        same_sign = (float(rhos[0]) >= 0) == (float(rhos[1]) >= 0)
        abs_values = [abs(float(rho)) for rho in rhos]
        eligible = same_sign and min(abs_values) >= 0.40 and statistics.fmean(abs_values) >= 0.50
        candidate_rows.append(
            {
                "descriptor": descriptor,
                "rho_dev1": rhos[0],
                "rho_dev2": rhos[1],
                "same_sign": same_sign,
                "mean_abs_rho": statistics.fmean(abs_values),
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in candidate_rows if row["eligible"]]
    priority = {descriptor: index for index, descriptor in enumerate(DESCRIPTORS)}
    eligible_rows.sort(
        key=lambda row: (
            -min(abs(float(row["rho_dev1"])), abs(float(row["rho_dev2"]))),
            priority[row["descriptor"]],
        )
    )
    selected = eligible_rows[0] if eligible_rows else None
    policy = None
    if selected is not None:
        descriptor = selected["descriptor"]
        rho = statistics.fmean(float(selected[key]) for key in ("rho_dev1", "rho_dev2"))
        if descriptor in {"D1_batch_mean_risk_sd", "D3_high_risk_fraction_sd"} and rho > 0:
            policy = "RISK_CLUSTERED_BATCHES_V1"
        elif descriptor == "D2_within_batch_risk_sd_mean" and rho > 0:
            policy = "STOP_D2_CLOSE_TO_FAILED_BALANCED_POLICY"
        elif descriptor == "D4_lag1_batch_mean_risk_acf" and rho < 0:
            policy = "RISK_ALTERNATING_BATCHES_V1"
        elif descriptor == "D4_lag1_batch_mean_risk_acf" and rho > 0:
            policy = "RISK_BLOCKED_BATCHES_V1"
        elif descriptor == "D6_position_vs_batch_mean_risk_spearman":
            policy = "EASY_TO_HARD_BATCHES_V1" if rho > 0 else "HARD_TO_EASY_BATCHES_V1"
    return {
        "eligible_descriptors": eligible_rows,
        "all_descriptor_associations": candidate_rows,
        "status": "mechanism_identified" if selected is not None else "mechanism_not_identified",
        "selected_descriptor": None if selected is None else selected["descriptor"],
        "selected_policy": policy,
        "second_intervention_allowed": selected is not None
        and policy not in {None, "STOP_D2_CLOSE_TO_FAILED_BALANCED_POLICY"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "frozen_before_training" or len(registry.get("schedules", [])) != 8:
        raise ValueError("probe registry is not the frozen eight-schedule contract")
    per_seed = {str(seed): aggregate_seed(args.root, seed) for seed in (1, 2)}
    selection = select_mechanism(per_seed)
    artifact = {
        "schema_version": 1,
        "kind": "ert_rslad_pure_order_probe_results_v1",
        "status": "completed",
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "root": str(args.root.resolve()),
        "per_seed": per_seed,
        "mechanism_selection": selection,
        "existing_run_consistency": {
            "status": "not_assessed",
            "reason": "historical runs lack batch-level telemetry and are discovery-only evidence",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **selection}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
