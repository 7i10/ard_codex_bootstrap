#!/usr/bin/env python3
"""Aggregate the preregistered A7 teacher-floor lambda sensitivity screen.

This is a read-only point analysis.  It consumes hash-bound CE-PGD20 endpoint
rows and the already frozen epoch-79 CE20/KL10 feature rows; it never loads a
checkpoint or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("L2", "L4")
EPOCHS = (84, 89, 94)
ENDPOINT_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
PARENT = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}
MASK = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}
ARMS = {
    "L0_BASE": ("A0", 0.0),
    "L1_010": ("L1_010", 0.10),
    "L2_CAL": ("A7", 0.2388051152229309),
    "L3_025": ("L3_025", 0.25),
    "L4_050": ("L4_050", 0.50),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["sample_id"]): row for row in pq.read_table(path).to_pylist()}


def load_endpoint(path: Path, run: str, epoch: int, split: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    meta_path = path / "endpoint.json"
    row_path = path / "endpoint-sample-stats.parquet"
    if not meta_path.is_file() or not row_path.is_file():
        raise RuntimeError(f"missing endpoint: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("contract") != "ert_stage_a_common_ce_pgd20_endpoint_v1":
        raise RuntimeError(f"endpoint contract mismatch: {path}")
    if meta.get("checkpoint_epoch") != epoch or meta.get("attack_identity_sha256") != ENDPOINT_SHA:
        raise RuntimeError(f"endpoint attack/epoch mismatch: {path}")
    expected_rows = 45000 if split == "train" else 5000
    if meta.get("row_count") != expected_rows or meta.get("rows_sha256") != sha256(row_path):
        raise RuntimeError(f"endpoint row hash/count mismatch: {path}")
    data = rows(row_path)
    if len(data) != expected_rows:
        raise RuntimeError(f"endpoint IDs are incomplete: {path}")
    return meta, data


def binary(base: list[bool], treatment: list[bool]) -> dict[str, float]:
    n = len(base)
    rescue = sum((not b) and t for b, t in zip(base, treatment)) / n
    harm = sum(b and (not t) for b, t in zip(base, treatment)) / n
    delta = sum(float(t) - float(b) for b, t in zip(base, treatment)) / n
    if abs(delta - (rescue - harm)) > 1e-12:
        raise RuntimeError("accuracy delta is not rescue minus harm")
    return {"accuracy_delta": delta, "rescue_rate": rescue, "harm_rate": harm, "net_rescue": rescue - harm}


def effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    if not ids:
        # Train-derived quantile boundaries may yield an empty validation bin.
        # Preserve that fact explicitly instead of fabricating an effect.
        return {
            "n": 0,
            "clean": None,
            "robust": None,
        }
    clean = binary([bool(base[i]["clean_correct"]) for i in ids], [bool(treatment[i]["clean_correct"]) for i in ids])
    robust = binary([bool(base[i]["robust_correct"]) for i in ids], [bool(treatment[i]["robust_correct"]) for i in ids])

    def mean(key: str) -> float:
        return sum(float(treatment[i][key]) - float(base[i][key]) for i in ids) / len(ids)

    return {
        "n": len(ids),
        "clean": {**clean, "margin_delta": mean("clean_probability_margin")},
        "robust": {**robust, "margin_delta": mean("adversarial_probability_margin")},
    }


def quantile_ids(
    train_features: dict[int, dict[str, Any]], validation_features: dict[int, dict[str, Any]], key: str
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    ordered = sorted(train_features, key=lambda i: (float(train_features[i][key]), i))
    groups: dict[str, list[int]] = {}
    bounds: dict[str, Any] = {}
    base, rem = divmod(len(ordered), 5)
    for index in range(5):
        selected = ordered[
            sum(base + (j < rem) for j in range(index)) : sum(base + (j < rem) for j in range(index + 1))
        ]
        name = f"Q{index + 1}"
        groups[name] = selected
        bounds[name] = {
            "n": len(selected),
            "min": float(train_features[selected[0]][key]),
            "max": float(train_features[selected[-1]][key]),
            "ids_sha256": hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest(),
        }
    # Apply train-derived boundaries to validation, never re-bin on outcomes.
    edges = [
        float(train_features[ordered[min(sum(base + (j < rem) for j in range(index + 1)) - 1, len(ordered) - 1)]][key])
        for index in range(4)
    ]
    validation_groups = {f"Q{i + 1}": [] for i in range(5)}
    for sample_id in sorted(validation_features):
        value = float(validation_features[sample_id][key])
        bucket = 0
        while bucket < 4 and value > edges[bucket]:
            bucket += 1
        validation_groups[f"Q{bucket + 1}"].append(sample_id)
    return {"train": groups, "validation": validation_groups}, bounds


def endpoint_roots(fresh: Path, historical: Path, run: str, arm: str, epoch: int, split: str) -> Path:
    disk_arm = ARMS[arm][0]
    root = historical if disk_arm in {"A0", "A7"} else fresh
    return root / run / disk_arm / f"epoch-{epoch}" / split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh-root", type=Path, default=ROOT / ".cache/analysis/ert-cw-margin-lambda-sensitivity-v1-endpoints"
    )
    parser.add_argument(
        "--historical-root", type=Path, default=ROOT / ".cache/analysis/ert-cw-margin-screen-v1-r3-endpoints-v2"
    )
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "docs/experiments/ert_cw_margin_lambda_sensitivity_v1.json"
    )
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md")
    args = parser.parse_args()
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_margin_lambda_sensitivity_v1",
        "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "endpoint_attack_identity_sha256": ENDPOINT_SHA,
        "lambdas": {arm: value[1] for arm, value in ARMS.items()},
        "clean_ce": 0.0,
        "floor": 0.03221710026264191,
        "cap": 0.13952550292015076,
        "results": {},
        "lineage": {},
        "trajectory": {},
        "cross_seed": {},
    }
    mask_root = ROOT / ".cache/analysis/ert-state-overlay-v1-review"
    feature_root = ROOT / ".cache/analysis/ert-clean-wrong-subtypes-v4"
    kl_root = ROOT / ".cache/analysis/ert-clean-wrong-reliability-proxy-v1"
    validation_feature_root = ROOT / ".cache/analysis/ert-cw-generalization-v1"
    trajectory_root = ROOT / ".cache/analysis/ert-cw-margin-lambda-sensitivity-v1-runs"
    for run in RUNS:
        mask_path = mask_root / f"anchor79-fixed-masks-{run}.json"
        if sha256(mask_path) != MASK[run]:
            raise RuntimeError(f"mask hash mismatch: {run}")
        mask = json.loads(mask_path.read_text(encoding="utf-8"))["masks"]["student_clean_wrong"]["selected_ids"]
        train_cw = [int(x) for x in mask]
        ce_train = rows(feature_root / run / "clean-wrong-feature-stats.parquet")
        kl_train = rows(kl_root / run / "clean-wrong-kl10-feature-stats.parquet")
        ce_val = rows(validation_feature_root / run / "CE20/validation-feature-stats.parquet")
        kl_val = rows(validation_feature_root / run / "KL10/validation-feature-stats.parquet")
        ce_q, ce_bounds = quantile_ids({i: ce_train[i] for i in train_cw}, ce_val, "teacher_adv_margin")
        kl_q, kl_bounds = quantile_ids({i: kl_train[i] for i in train_cw}, kl_val, "teacher_adv_margin")
        machine["lineage"][run] = {
            "mask_path": str(mask_path),
            "mask_sha256": sha256(mask_path),
            "parent_checkpoint_sha256": PARENT[run],
            "ce_bounds": ce_bounds,
            "kl_bounds": kl_bounds,
        }
        machine["results"][run] = {}
        machine["trajectory"][run] = {}
        for arm in ARMS:
            metrics_root = (
                trajectory_root
                if arm not in {"L0_BASE", "L2_CAL"}
                else ROOT / ".cache/analysis/ert-cw-margin-screen-v1-r3"
            )
            metrics_arm = ARMS[arm][0] if arm not in {"L0_BASE", "L2_CAL"} else {
                "L0_BASE": "A0",
                "L2_CAL": "A7",
            }[arm]
            metrics_path = metrics_root / run / metrics_arm / "epoch-metrics.jsonl"
            if not metrics_path.is_file():
                raise RuntimeError(f"missing trajectory metrics: {metrics_path}")
            metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            by_epoch = {int(row["epoch"]): row for row in metrics}
            machine["trajectory"][run][arm] = {
                str(epoch): by_epoch[epoch]
                for epoch in EPOCHS
                if epoch in by_epoch
            }
            if set(machine["trajectory"][run][arm]) != {str(epoch) for epoch in EPOCHS}:
                raise RuntimeError(f"trajectory epochs are incomplete: {metrics_path}")
            machine["lineage"][run].setdefault("trajectory", {})[arm] = {
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
            }
        for epoch in EPOCHS:
            machine["results"][run][str(epoch)] = {}
            loaded: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
            for arm in ARMS:
                loaded[arm] = {}
                for split in ("train", "validation"):
                    meta, data = load_endpoint(
                        endpoint_roots(args.fresh_root, args.historical_root, run, arm, epoch, split), run, epoch, split
                    )
                    loaded[arm][split] = data
                    machine["lineage"].setdefault(run, {}).setdefault("endpoints", {}).setdefault(
                        str(epoch), {}
                    ).setdefault(arm, {})[split] = {
                        "rows_sha256": meta["rows_sha256"],
                        "checkpoint_sha256": meta["checkpoint_sha256"],
                        "source_git_sha": meta.get("source_git_sha"),
                    }
            base_train = loaded["L0_BASE"]["train"]
            base_val = loaded["L0_BASE"]["validation"]
            non_cw = sorted(set(base_train) - set(train_cw))
            val_cw = [i for i in ce_val if not bool(ce_val[i]["student_clean_correct"])]
            record: dict[str, Any] = {}
            for arm in ARMS:
                rec = {
                    "train_direct_cw": effect(base_train, loaded[arm]["train"], train_cw),
                    "train_spillover": effect(base_train, loaded[arm]["train"], non_cw),
                    "validation_overall": effect(base_val, loaded[arm]["validation"], sorted(base_val)),
                    "validation_cw": effect(base_val, loaded[arm]["validation"], val_cw),
                }
                rec["validation_q"] = {
                    "CE20": {
                        q: effect(base_val, loaded[arm]["validation"], [i for i in ids if i in base_val])
                        for q, ids in ce_q["validation"].items()
                    },
                    "KL10": {
                        q: effect(base_val, loaded[arm]["validation"], [i for i in ids if i in base_val])
                        for q, ids in kl_q["validation"].items()
                    },
                }
                record[arm] = rec
            machine["results"][run][str(epoch)] = record
    for epoch in EPOCHS:
        machine["cross_seed"][str(epoch)] = {}
        for arm in ARMS:
            values = [
                machine["results"][run][str(epoch)][arm]["validation_overall"]["robust"]["accuracy_delta"]
                for run in RUNS
            ]
            machine["cross_seed"][str(epoch)][arm] = {
                "L2_delta": values[0],
                "L4_delta": values[1],
                "mean_delta": sum(values) / len(values),
                "same_nonnegative_direction": values[0] >= 0 and values[1] >= 0,
                "same_nonpositive_direction": values[0] <= 0 and values[1] <= 0,
            }
    payload = json.dumps(machine, sort_keys=True, separators=(",", ":")).encode()
    machine["report_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong teacher-adaptive margin lambda sensitivity",
        "",
        "Status: completed fixed margin-only lambda sensitivity; CleanCE was zero in every arm.",
        "",
        "| seed | epoch | arm | lambda | held-out clean Δ | held-out robust Δ | spillover robust Δ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        for epoch in EPOCHS:
            for arm, (_, value) in ARMS.items():
                rec = machine["results"][run][str(epoch)][arm]
                clean_delta = 100 * rec["validation_overall"]["clean"]["accuracy_delta"]
                robust_delta = 100 * rec["validation_overall"]["robust"]["accuracy_delta"]
                spillover_delta = 100 * rec["train_spillover"]["robust"]["accuracy_delta"]
                lines.append(
                    f"| {run} | {epoch} | {arm} | {value:.6f} | "
                    f"{clean_delta:.3f} | {robust_delta:.3f} | {spillover_delta:.3f} |"
                )
    lines += [
        "",
        "## Epoch-94 Clean-Wrong direct cohort",
        "",
        "| seed | arm | n | clean Δ | robust Δ | clean net rescue | robust net rescue |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        for arm, (_, value) in ARMS.items():
            rec = machine["results"][run]["94"][arm]["train_direct_cw"]
            lines.append(
                f"| {run} | {arm} (λ={value:.6f}) | {rec['n']} | "
                f"{100 * rec['clean']['accuracy_delta']:.3f} | "
                f"{100 * rec['robust']['accuracy_delta']:.3f} | "
                f"{100 * rec['clean']['net_rescue']:.3f} | "
                f"{100 * rec['robust']['net_rescue']:.3f} |"
            )
    lines += [
        "",
        "## Epoch-94 held-out Clean-Wrong subgroup",
        "",
        "| seed | arm | n | clean Δ | robust Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for run in RUNS:
        for arm, (_, value) in ARMS.items():
            rec = machine["results"][run]["94"][arm]["validation_cw"]
            lines.append(
                f"| {run} | {arm} (λ={value:.6f}) | {rec['n']} | "
                f"{100 * rec['clean']['accuracy_delta']:.3f} | "
                f"{100 * rec['robust']['accuracy_delta']:.3f} |"
            )
    for domain in ("CE20", "KL10"):
        lines += [
            "",
            f"## Epoch-94 held-out {domain} Teacher-margin Q1--Q5",
            "",
            "Q bins are derived from the train Clean-Wrong feature distribution; "
            "empty validation bins are reported as `n=0` and are not assigned an effect.",
            "",
            "| seed | arm | Q1 robust Δ | Q2 | Q3 | Q4 | Q5 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for run in RUNS:
            for arm, (_, value) in ARMS.items():
                cells = []
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                    rec = machine["results"][run]["94"][arm]["validation_q"][domain][q]
                    cells.append("n=0" if rec["n"] == 0 else f"{100 * rec['robust']['accuracy_delta']:.3f}")
                lines.append(
                    f"| {run} | {arm} (λ={value:.6f}) | " + " | ".join(cells) + " |"
                )
    lines += [
        "",
        "## Training trajectory validation PGD accuracy",
        "",
        "| seed | arm | epoch84 | epoch89 | epoch94 |",
        "|---|---|---:|---:|---:|",
    ]
    for run in RUNS:
        for arm, (_, value) in ARMS.items():
            cells = [f"{machine['trajectory'][run][arm][str(epoch)]['val_pgd_accuracy']:.4f}" for epoch in EPOCHS]
            lines.append(f"| {run} | {arm} (λ={value:.6f}) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Cross-seed epoch-94 held-out robust delta",
        "",
        "| arm | L2 Δ | L4 Δ | mean Δ | same nonnegative direction |",
        "|---|---:|---:|---:|---|",
    ]
    for arm, (_, value) in ARMS.items():
        rec = machine["cross_seed"]["94"][arm]
        lines.append(
            f"| {arm} (λ={value:.6f}) | {100 * rec['L2_delta']:.3f} | "
            f"{100 * rec['L4_delta']:.3f} | {100 * rec['mean_delta']:.3f} | "
            f"{rec['same_nonnegative_direction']} |"
        )
    lines += [
        "",
        "The primary endpoint is epoch-94 held-out CE-PGD20 robust accuracy. "
        "Effects are descriptive paired sample effects; they are not "
        "training-seed confidence intervals. No lambda was selected automatically.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
