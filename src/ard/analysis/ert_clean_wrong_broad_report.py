"""Aggregate the fixed Clean-Wrong screen without selecting a winner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ard.analysis.ert_clean_wrong_broad_screen import ARMS, fixed_clean_wrong_mask


class CleanWrongReportError(RuntimeError):
    """Raised when screen endpoint lineage or joins are incomplete."""


def _paired(base: list[dict[str, Any]], treatment: list[dict[str, Any]], *, key: str) -> dict[str, float]:
    b = {int(row["sample_id"]): row for row in base}
    t = {int(row["sample_id"]): row for row in treatment}
    if set(b) != set(t):
        raise CleanWrongReportError("endpoint stable-ID universe differs between arm and C0")
    deltas = np.asarray([float(t[item][key]) - float(b[item][key]) for item in sorted(b)], dtype=np.float64)
    correct_key = "clean_correct" if key == "clean_correct" else "robust_correct"
    base_correct = np.asarray([bool(b[item][correct_key]) for item in sorted(b)])
    treatment_correct = np.asarray([bool(t[item][correct_key]) for item in sorted(b)])
    rescue = (~base_correct) & treatment_correct
    harm = base_correct & (~treatment_correct)
    return {
        "delta_accuracy": float(deltas.mean()),
        "delta_margin": float(deltas.mean()),
        "rescue_rate": float(rescue.mean()),
        "harm_rate": float(harm.mean()),
        "net_rescue": float(rescue.mean() - harm.mean()),
    }


def _bootstrap_delta(
    base: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    *,
    key: str,
    labels: dict[int, int],
    seed: int,
    replicates: int = 2000,
) -> dict[str, object]:
    ids = sorted(int(row["sample_id"]) for row in base)
    base_map = {int(row["sample_id"]): row for row in base}
    treatment_map = {int(row["sample_id"]): row for row in treatment}
    if set(ids) != set(treatment_map):
        raise CleanWrongReportError("bootstrap endpoint universe mismatch")
    classes = sorted({labels[item] for item in ids})
    deltas = {
        cls: np.asarray(
            [float(treatment_map[item][key]) - float(base_map[item][key]) for item in ids if labels[item] == cls],
            dtype=np.float64,
        )
        for cls in classes
    }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = [values_for[rng.integers(0, len(values_for), len(values_for))] for values_for in deltas.values()]
        values[index] = float(np.mean(np.concatenate(sampled)))
    return {
        "mean": float(values.mean()),
        "lower_2_5": float(np.quantile(values, 0.025)),
        "upper_97_5": float(np.quantile(values, 0.975)),
        "replicates": replicates,
        "seed": seed,
        "stratification": "true_class",
    }


def build_report(
    *,
    root: Path,
    masks: dict[str, Path],
    output_json: Path,
    output_markdown: Path,
    bootstrap: bool = True,
) -> dict[str, Any]:
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_clean_wrong_broad_screen_results_v1",
        "endpoint_epoch": 84,
        "seeds": {},
        "bootstrap": {"replicates": 2000, "seed": 20260816, "enabled": bootstrap},
    }
    for run, mask_path in masks.items():
        mask = fixed_clean_wrong_mask(mask_path, run=run)
        run_data: dict[str, Any] = {"mask": mask, "arms": {}}
        for arm in ARMS:
            endpoint_data: dict[str, Any] = {}
            for split in ("train", "validation"):
                endpoint = root / run / arm.name / "endpoint" / split / "endpoint.json"
                rows_path = endpoint.with_name("endpoint-sample-stats.parquet")
                if not endpoint.is_file() or not rows_path.is_file():
                    raise CleanWrongReportError(f"missing endpoint for {run}/{arm.name}/{split}: {endpoint}")
                payload = json.loads(endpoint.read_text(encoding="utf-8"))
                import pyarrow.parquet as pq

                rows = pq.read_table(rows_path).to_pylist()
                endpoint_data[split] = {"meta": payload, "rows": rows}
            run_data["arms"][arm.name] = endpoint_data
        machine["seeds"][run] = {"mask": mask, "arms": {}}
        base_data = run_data["arms"]["C0"]
        selected = set(mask["selected_ids"])
        labels = {int(row["sample_id"]): int(row["true_label"]) for row in base_data["train"]["rows"]}
        for arm in ARMS:
            result: dict[str, Any] = {}
            for split in ("train", "validation"):
                base_rows = base_data[split]["rows"]
                treatment_rows = run_data["arms"][arm.name][split]["rows"]
                cohorts = (
                    {
                        "direct": [row for row in base_rows if int(row["sample_id"]) in selected],
                        "spillover": [row for row in base_rows if int(row["sample_id"]) not in selected],
                    }
                    if split == "train"
                    else {"held_out": base_rows}
                )
                result[split] = {}
                treatment_by_id = {int(row["sample_id"]): row for row in treatment_rows}
                for cohort, cohort_base in cohorts.items():
                    cohort_ids = {int(row["sample_id"]) for row in cohort_base}
                    cohort_treatment = [treatment_by_id[item] for item in sorted(cohort_ids)]
                    robust = _paired(cohort_base, cohort_treatment, key="robust_correct")
                    clean = _paired(cohort_base, cohort_treatment, key="clean_correct")
                    robust["delta_margin"] = _paired(
                        cohort_base, cohort_treatment, key="adversarial_probability_margin"
                    )["delta_accuracy"]
                    clean["delta_margin"] = _paired(
                        cohort_base, cohort_treatment, key="clean_probability_margin"
                    )["delta_accuracy"]
                    result[split][cohort] = {"robust": robust, "clean": clean}
                    if bootstrap and arm.name != "C0":
                        result[split][cohort]["robust_ci"] = _bootstrap_delta(
                            cohort_base,
                            cohort_treatment,
                            key="robust_correct",
                            labels={
                                item: labels.get(item, int(treatment_by_id[item]["true_label"]))
                                for item in cohort_ids
                            },
                            seed=20260816,
                        )
            machine["seeds"][run]["arms"][arm.name] = result
    machine["source"] = {"root": str(root.resolve())}
    machine["source_sha256"] = hashlib.sha256(json.dumps(machine, sort_keys=True).encode()).hexdigest()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong Broad Treatment Screen",
        "",
        "Status: completed only after all 32 arms and independent epoch-84 endpoints exist; no automatic promotion.",
        "",
        "Direct is the fixed epoch-79 Clean-Wrong train cohort; spillover is the remaining train IDs;",
        "held-out is the fixed internal validation split. Bootstrap intervals are sample uncertainty,",
        "not training-seed uncertainty.",
        "",
        "| seed | arm | direct robust Δ | spillover robust Δ | held-out robust Δ | held-out clean Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in masks:
        for arm in ARMS:
            row = machine["seeds"][run]["arms"][arm.name]
            lines.append(
                f"| {run} | {arm.name} | {row['train']['direct']['robust']['delta_accuracy']:.6f} | "
                f"{row['train']['spillover']['robust']['delta_accuracy']:.6f} | "
                f"{row['validation']['held_out']['robust']['delta_accuracy']:.6f} | "
                f"{row['validation']['held_out']['clean']['delta_accuracy']:.6f} |"
            )
    lines += [
        "",
        "No winner, coefficient, threshold, +15 continuation, official test, or AutoAttack was selected automatically.",
    ]
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return machine
