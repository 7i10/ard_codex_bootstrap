#!/usr/bin/env python3
"""Aggregate attack-seed probe trajectories, endpoints, and sensitivity rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL rows: {path}")
    return rows


def auc(rows: list[dict[str, Any]], field: str = "train_robust_accuracy") -> float:
    points = sorted((int(row["epoch"]), float(row[field])) for row in rows if 100 <= int(row["epoch"]) <= 114)
    if [epoch for epoch, _ in points] != list(range(100, 115)):
        raise ValueError("attack-seed trajectory must contain epochs 100--114")
    return (
        sum(
            (b_epoch - a_epoch) * (a_value + b_value) / 2
            for (a_epoch, a_value), (b_epoch, b_value) in zip(points, points[1:])
        )
        / 14
    )


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + end - 1) / 2
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x, y = rank(left), rank(right)
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    denominator = math.sqrt(sum((value - xm) ** 2 for value in x) * sum((value - ym) ** 2 for value in y))
    return None if denominator == 0 else sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denominator


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(
    *, registry_path: Path, root: Path, fixed_root: Path, pure_order_path: Path, output: Path, report: Path
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    seed_rows = registry.get("seeds")
    if registry.get("status") != "frozen_before_training" or not isinstance(seed_rows, list) or len(seed_rows) != 8:
        raise ValueError("registry is not the frozen eight-seed contract")
    seeds = [int(row["value"]) for row in seed_rows]
    fixed_model: dict[str, Any] = {}
    for dev_seed in (1, 2):
        result_path = fixed_root / f"seed{dev_seed}" / "fixed-model-result.json"
        fixed_model[str(dev_seed)] = json.loads(result_path.read_text(encoding="utf-8"))
    per_seed: dict[str, Any] = {}
    sample_sensitivity: dict[str, Any] = {}
    for dev_seed in (1, 2):
        arms: dict[str, Any] = {}
        endpoint_rows: dict[int, list[dict[str, Any]]] = {}
        for index, attack_seed_value in enumerate(seeds):
            run = root / f"attack-seed-{index}-s{dev_seed}"
            metrics = read_jsonl(run / "epoch-metrics.jsonl")
            probe = [row for row in metrics if 100 <= int(row["epoch"]) <= 114]
            endpoint = run / "endpoint" / "validation" / "endpoint.json"
            endpoint_payload = json.loads(endpoint.read_text(encoding="utf-8"))
            if (
                endpoint_payload.get("row_count") != 5000
                or endpoint_payload.get("attack_identity_sha256")
                != "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
            ):
                raise ValueError(f"endpoint identity mismatch: {endpoint}")
            rows_path = Path(endpoint_payload["rows_path"])
            endpoint_rows[index] = _read_parquet(rows_path)
            arms[f"ATTACK_SEED_{index}"] = {
                "index": index,
                "attack_seed": attack_seed_value,
                "run": str(run.resolve()),
                "metrics_sha256": sha256(run / "epoch-metrics.jsonl"),
                "probe_auc_100_114": auc(probe),
                "robust_at_114": float(probe[-1]["train_robust_accuracy"]),
                "clean_at_114": float(probe[-1]["train_clean_accuracy"]),
                "gain_100_to_114": float(probe[-1]["train_robust_accuracy"]) - float(probe[0]["train_robust_accuracy"]),
                "best_robust": max(float(row["train_robust_accuracy"]) for row in probe),
                "best_minus_last": max(float(row["train_robust_accuracy"]) for row in probe)
                - float(probe[-1]["train_robust_accuracy"]),
                "endpoint": {
                    "path": str(endpoint.resolve()),
                    "sha256": sha256(endpoint),
                    "clean_accuracy": endpoint_payload["clean_accuracy"],
                    "robust_accuracy": endpoint_payload["robust_accuracy"],
                    "rows_sha256": endpoint_payload["rows_sha256"],
                },
            }
        per_seed[str(dev_seed)] = {"arms": arms}
        per_seed[str(dev_seed)]["probe_auc_summary"] = _summary([item["probe_auc_100_114"] for item in arms.values()])
        per_seed[str(dev_seed)]["robust_at_114_summary"] = _summary([item["robust_at_114"] for item in arms.values()])
        by_id: dict[int, list[dict[str, Any]]] = {}
        for index, rows in endpoint_rows.items():
            for row in rows:
                by_id.setdefault(int(row["sample_id"]), []).append({"attack_index": index, **row})
        sample_rows: list[dict[str, Any]] = []
        for sample_id, rows in sorted(by_id.items()):
            margins = [float(row["adversarial_probability_margin"]) for row in rows]
            correctness = [bool(row["robust_correct"]) for row in rows]
            sample_rows.append(
                {
                    "seed": dev_seed,
                    "sample_id": sample_id,
                    "true_label": int(rows[0]["true_label"]),
                    "robust_frequency": sum(correctness),
                    "non_unanimous": len(set(correctness)) > 1,
                    "margin_sd": statistics.stdev(margins) if len(margins) > 1 else 0.0,
                    "margin_range": max(margins) - min(margins),
                }
            )
        sample_sensitivity[str(dev_seed)] = {
            "sample_count": len(sample_rows),
            "frequency_histogram": {
                str(k): sum(int(row["robust_frequency"] == k) for row in sample_rows) for k in range(9)
            },
            "non_unanimous_fraction": statistics.fmean(bool(row["non_unanimous"]) for row in sample_rows)
            if sample_rows
            else None,
            "margin_sd_summary": _summary([float(row["margin_sd"]) for row in sample_rows]),
            "rows": sample_rows,
        }
    pure = json.loads(pure_order_path.read_text(encoding="utf-8"))
    order_summary = {}
    for dev_seed in (1, 2):
        schedules = pure["per_seed"][str(dev_seed)]["schedules"]
        aucs = [float(schedules[name]["probe_auc"]) for name in sorted(schedules)]
        order_summary[str(dev_seed)] = {"auc_summary": _summary(aucs), "source_sha256": sha256(pure_order_path)}
    classification = {}
    for dev_seed in (1, 2):
        attack = per_seed[str(dev_seed)]["probe_auc_summary"]
        order = order_summary[str(dev_seed)]["auc_summary"]
        ratio = None if order["sd"] == 0 else attack["sd"] / order["sd"]
        strong = (
            ratio is not None
            and ratio >= 2
            and (attack["max"] - attack["min"]) >= 0.0015
            and (order["max"] - order["min"]) >= 0
        )
        classification[str(dev_seed)] = {
            "attack_vs_order_sd_ratio": ratio,
            "attack_auc_range": attack["max"] - attack["min"],
            "order_auc_range": order["max"] - order["min"],
            "classification": "STRONG" if strong else "MODERATE" if ratio is not None and ratio > 1 else "WEAK",
        }
    artifact = {
        "schema_version": 1,
        "kind": "ert_rslad_attack_seed_probe_results_v1",
        "status": "completed",
        "registry_sha256": sha256(registry_path),
        "root": str(root.resolve()),
        "fixed_model_root": str(fixed_root.resolve()),
        "fixed_model": fixed_model,
        "per_seed": per_seed,
        "sample_sensitivity": sample_sensitivity,
        "pure_order_reference": order_summary,
        "classification": classification,
        "no_intervention": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    fixed_artifact = {
        "schema_version": 1,
        "kind": "ert_rslad_attack_randomness_fixed_model_v1",
        "status": "completed",
        "registry_sha256": artifact["registry_sha256"],
        "fixed_model": artifact["fixed_model"],
    }
    fixed_output = output.parent / "ert_rslad_attack_randomness_fixed_model_v1.json"
    fixed_output.write_text(json.dumps(fixed_artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    sample_output = output.parent / "ert_rslad_attack_seed_sample_sensitivity_v1.json"
    sample_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ert_rslad_attack_seed_sample_sensitivity_v1",
                "status": "completed",
                "registry_sha256": artifact["registry_sha256"],
                "sample_sensitivity": artifact["sample_sensitivity"],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(artifact, report)
    return artifact


def _write_report(artifact: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# ERT / RSLAD Attack Random-Start Randomness Characterization",
        "",
        "This is a descriptive fixed-model plus 15-epoch attack-seed characterization. "
        "No attack intervention or seed promotion was performed.",
        "",
        "## Fixed-model direct replay",
        "",
        "| dev seed | n | risk→margin-SD Spearman | risk→attack-loss-SD Spearman |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in (1, 2):
        fixed = (
            artifact.get("fixed_model", {}).get(str(seed), {})
            if isinstance(artifact.get("fixed_model"), Mapping)
            else {}
        )
        lines.append(
            f"| {seed} | {fixed.get('sample_count', 'not recorded')} | "
            f"{fixed.get('risk_margin_sd_spearman', 'not recorded')} | "
            f"{fixed.get('risk_attack_kl_sd_spearman', 'not recorded')} |"
        )
    lines += [
        "",
        "## 15-epoch trajectory probe",
        "",
        "| dev seed | attack AUC mean | SD | range | e114 robust mean |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in (1, 2):
        summary = artifact["per_seed"][str(seed)]["probe_auc_summary"]
        robust = artifact["per_seed"][str(seed)]["robust_at_114_summary"]
        lines.append(
            f"| {seed} | {summary['mean']:.6f} | {summary['sd']:.6f} | "
            f"{summary['min']:.6f}–{summary['max']:.6f} | {robust['mean']:.6f} |"
        )
    lines += [
        "",
        "## Sample-level e114 sensitivity",
        "",
        "| dev seed | validation rows | non-unanimous fraction | margin-SD mean |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in (1, 2):
        row = artifact["sample_sensitivity"][str(seed)]
        lines.append(
            f"| {seed} | {row['sample_count']} | {row['non_unanimous_fraction']:.6f} | "
            f"{row['margin_sd_summary']['mean']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Attack-seed dispersion is reported against the pre-existing pure-order reference. "
        "The classification is characterization only; it does not authorize a training intervention, "
        "seed selection, or extension.",
        "",
        "## Lineage",
        "",
        f"Registry SHA-256: `{artifact['registry_sha256']}`",
        "Source SHA: recorded in the frozen registry.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--pure-order-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    aggregate(
        registry_path=args.registry.resolve(),
        root=args.root.resolve(),
        fixed_root=args.fixed_root.resolve(),
        pure_order_path=args.pure_order_results.resolve(),
        output=args.output.resolve(),
        report=args.report.resolve(),
    )
    print(json.dumps({"output": str(args.output.resolve()), "report": str(args.report.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
