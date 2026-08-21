#!/usr/bin/env python3
"""Combine the hash-bound F0--F3 Clean-Wrong factorial endpoints.

F0/F1/F2 are historical A0/A1/A7 endpoint artifacts.  F3 is the only fresh
trajectory.  The script is read-only with respect to checkpoints and computes
paired stable-ID effects; it never chooses a treatment or a threshold.
"""

# ruff: noqa: E501

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
ARMS = ("F0", "F1", "F2", "F3")
HORIZONS = (84, 89, 94)
ENDPOINT_ATTACK_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
TRAIN_SOURCE = "bb59b512185af7bb70633c3266efd95bb24a563f"
F3_SOURCE = "514c585740ac17ce2e75427687bf5ec4afa8ba6c"
PARENT_SHA = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}
MASK_SHA = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}
HIST_ENDPOINT_ROOT = ROOT / ".cache/analysis/ert-cw-margin-screen-v1-r3-endpoints-v2"
F3_ENDPOINT_ROOT = ROOT / ".cache/analysis/ert-cw-a7-cleance-ablation-v1-endpoints"
TRAIN_FEATURE = {
    "L2": ROOT / ".cache/analysis/ert-clean-wrong-subtypes-v4/L2/clean-wrong-feature-stats.parquet",
    "L4": ROOT / ".cache/analysis/ert-clean-wrong-subtypes-v4/L4/clean-wrong-feature-stats.parquet",
}
KL_FEATURE = {
    "L2": ROOT / ".cache/analysis/ert-clean-wrong-reliability-proxy-v1/L2/clean-wrong-kl10-feature-stats.parquet",
    "L4": ROOT / ".cache/analysis/ert-clean-wrong-reliability-proxy-v1/L4/clean-wrong-kl10-feature-stats.parquet",
}
VAL_FEATURE = {
    "L2": ROOT / ".cache/analysis/ert-cw-generalization-v1/L2/CE20/validation-feature-stats.parquet",
    "L4": ROOT / ".cache/analysis/ert-cw-generalization-v1/L4/CE20/validation-feature-stats.parquet",
}
VAL_KL_FEATURE = {
    "L2": ROOT / ".cache/analysis/ert-cw-generalization-v1/L2/KL10/validation-feature-stats.parquet",
    "L4": ROOT / ".cache/analysis/ert-cw-generalization-v1/L4/KL10/validation-feature-stats.parquet",
}


class AblationError(RuntimeError):
    """Raised when an endpoint or feature join is not hash-bound."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AblationError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    rows = {int(row["sample_id"]): row for row in values}
    if len(rows) != len(values):
        raise AblationError(f"duplicate stable IDs: {path}")
    return rows


def endpoint_path(run: str, arm: str, epoch: int, split: str) -> Path:
    if arm == "F3":
        root = F3_ENDPOINT_ROOT / run / "F3"
    else:
        root = HIST_ENDPOINT_ROOT / run / {"F0": "A0", "F1": "A1", "F2": "A7"}[arm]
    return root / f"epoch-{epoch}" / split


def load_endpoint(run: str, arm: str, epoch: int, split: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    root = endpoint_path(run, arm, epoch, split)
    meta_path = root / "endpoint.json"
    rows_path = root / "endpoint-sample-stats.parquet"
    if not meta_path.is_file() or not rows_path.is_file():
        raise AblationError(f"missing endpoint: {root}")
    meta = read_json(meta_path)
    if meta.get("contract") != "ert_stage_a_common_ce_pgd20_endpoint_v1":
        raise AblationError(f"endpoint contract mismatch: {meta_path}")
    if meta.get("checkpoint_epoch") != epoch or meta.get("attack_identity_sha256") != ENDPOINT_ATTACK_SHA:
        raise AblationError(f"endpoint attack/epoch mismatch: {meta_path}")
    if meta.get("rows_sha256") != sha256(rows_path) or meta.get("row_count") not in {45000, 5000}:
        raise AblationError(f"endpoint row identity mismatch: {meta_path}")
    if meta.get("source_git_sha") not in {TRAIN_SOURCE, F3_SOURCE}:
        raise AblationError(f"unexpected endpoint source: {meta_path}")
    rows = read_rows(rows_path)
    if len(rows) != int(meta["row_count"]):
        raise AblationError(f"endpoint row count mismatch: {meta_path}")
    return meta, rows


def feature_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise AblationError(f"missing feature rows: {path}")
    return read_rows(path)


def check_universe(reference: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]], label: str) -> None:
    if set(reference) != set(candidate):
        raise AblationError(f"{label}: stable-ID universe mismatch")
    if any(int(reference[i]["true_label"]) != int(candidate[i]["true_label"]) for i in reference):
        raise AblationError(f"{label}: class mapping mismatch")


def scalar_effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    if not ids:
        return {"n": 0}

    def one(correct: str, margin: str) -> dict[str, Any]:
        b = [bool(base[i][correct]) for i in ids]
        t = [bool(treatment[i][correct]) for i in ids]
        rescue = sum(not before and after for before, after in zip(b, t))
        harm = sum(before and not after for before, after in zip(b, t))
        n = len(ids)
        rescue_rate, harm_rate = rescue / n, harm / n
        delta = rescue_rate - harm_rate
        if abs(delta - (rescue - harm) / n) > 1e-12:
            raise AblationError("accuracy delta != rescue - harm")
        return {
            "accuracy_delta": delta,
            "rescue_rate": rescue_rate,
            "harm_rate": harm_rate,
            "net_rescue_rate": delta,
            "rescue_count": rescue,
            "harm_count": harm,
            "margin_delta": sum(float(treatment[i][margin]) - float(base[i][margin]) for i in ids) / n,
        }

    return {"n": len(ids), "clean": one("clean_correct", "clean_probability_margin"), "robust": one("robust_correct", "adversarial_probability_margin")}


def absolute(meta: dict[str, Any]) -> dict[str, float]:
    return {key: float(meta[key]) for key in ("clean_accuracy", "robust_accuracy")}


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def quintiles(rows: dict[int, dict[str, Any]], ids: list[int], feature: str) -> tuple[dict[str, list[int]], dict[str, Any]]:
    ordered = sorted(ids, key=lambda item: (float(rows[item][feature]), item))
    groups = {f"Q{q}": ordered[(len(ordered) * (q - 1)) // 5 : (len(ordered) * q) // 5] for q in range(1, 6)}
    bounds = {
        name: {
            "count": len(members),
            "lower": float(rows[members[0]][feature]),
            "upper": float(rows[members[-1]][feature]),
            "ids_sha256": ids_hash(members),
        }
        for name, members in groups.items()
    }
    return groups, bounds


def validation_groups(train_rows: dict[int, dict[str, Any]], val_rows: dict[int, dict[str, Any]], feature: str) -> dict[str, list[int]]:
    train_ids = sorted(train_rows, key=lambda item: (float(train_rows[item][feature]), item))
    upper = [float(train_rows[train_ids[(len(train_ids) * q) // 5 - 1]][feature]) for q in range(1, 6)]
    groups = {f"Q{q}": [] for q in range(1, 6)}
    for item in val_rows:
        value = float(val_rows[item][feature])
        q = next((idx for idx, edge in enumerate(upper, 1) if value <= edge), 5)
        groups[f"Q{q}"].append(item)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs/experiments/ert_cw_a7_cleance_ablation_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/ERT_CW_A7_CLEANCE_ABLATION.md")
    args = parser.parse_args()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_a7_cleance_ablation_v1",
        "source_git_sha": source_sha,
        "training": {"fresh": ["L2/F3", "L4/F3"], "reused": {"F0": "A0", "F1": "A1", "F2": "A7"}},
        "parents": PARENT_SHA,
        "mask_sha256": MASK_SHA,
        "endpoint_attack_identity_sha256": ENDPOINT_ATTACK_SHA,
        "arms": {"F0": "baseline", "F1": "CleanCE 0.15", "F2": "teacher-floor margin only", "F3": "CleanCE 0.15 + teacher-floor margin"},
        "horizons": list(HORIZONS),
        "results": {},
        "quantiles": {},
        "factorial": {},
        "lineage": {},
    }
    for run in RUNS:
        train_features = feature_rows(TRAIN_FEATURE[run])
        kl_features = feature_rows(KL_FEATURE[run])
        val_features = feature_rows(VAL_FEATURE[run])
        val_kl_features = feature_rows(VAL_KL_FEATURE[run])
        check_universe(train_features, kl_features, f"{run} train CE/KL features")
        check_universe(val_features, val_kl_features, f"{run} validation CE/KL features")
        cw_ids = sorted(item for item, row in train_features.items() if not bool(row["student_clean_correct"]))
        val_cw_ids = sorted(item for item, row in val_features.items() if not bool(row["student_clean_correct"]))
        ce_groups, ce_bounds = quintiles(train_features, cw_ids, "teacher_adv_margin")
        kl_groups, kl_bounds = quintiles(kl_features, cw_ids, "teacher_adv_margin")
        ce_val_groups = validation_groups(train_features, val_features, "teacher_adv_margin")
        kl_val_groups = validation_groups(kl_features, val_kl_features, "teacher_adv_margin")
        ce_val_groups = {name: sorted(set(ids) & set(val_cw_ids)) for name, ids in ce_val_groups.items()}
        kl_val_groups = {name: sorted(set(ids) & set(val_cw_ids)) for name, ids in kl_val_groups.items()}
        machine["quantiles"][run] = {
            "train_ce20": ce_bounds,
            "train_kl10": kl_bounds,
            "validation_cw_count": len(val_cw_ids),
            "validation_ce20_counts": {key: len(value) for key, value in ce_val_groups.items()},
            "validation_kl10_counts": {key: len(value) for key, value in kl_val_groups.items()},
            "feature_sha256": {"train_ce20": sha256(TRAIN_FEATURE[run]), "train_kl10": sha256(KL_FEATURE[run]), "validation_ce20": sha256(VAL_FEATURE[run]), "validation_kl10": sha256(VAL_KL_FEATURE[run])},
        }
        machine["results"][run] = {}
        endpoint_rows: dict[int, dict[str, dict[str, dict[int, dict[str, Any]]]]] = {}
        for epoch in HORIZONS:
            endpoint_rows[epoch] = {}
            machine["results"][run][str(epoch)] = {}
            for arm in ARMS:
                endpoint_rows[epoch][arm] = {}
                machine["results"][run][str(epoch)][arm] = {}
                for split in ("train", "validation"):
                    meta, rows = load_endpoint(run, arm, epoch, split)
                    endpoint_rows[epoch][arm][split] = rows
                    machine["lineage"].setdefault(run, {}).setdefault(str(epoch), {}).setdefault(arm, {})[split] = {"meta_sha256": sha256(endpoint_path(run, arm, epoch, split) / "endpoint.json"), "rows_sha256": meta["rows_sha256"], "checkpoint_sha256": meta["checkpoint_sha256"], "source_git_sha": meta["source_git_sha"]}
                    machine["results"][run][str(epoch)][arm][split] = {"absolute": absolute(meta), "n": int(meta["row_count"])}
            base_train = endpoint_rows[epoch]["F0"]["train"]
            base_val = endpoint_rows[epoch]["F0"]["validation"]
            for arm in ARMS:
                for split, ids, base in (
                    ("train_overall", sorted(base_train), base_train),
                    ("train_direct_cw", cw_ids, base_train),
                    ("train_spillover_non_cw", sorted(set(base_train) - set(cw_ids)), base_train),
                    ("validation_overall", sorted(base_val), base_val),
                    ("validation_cw", val_cw_ids, base_val),
                ):
                    cohort = ids
                    target = endpoint_rows[epoch][arm]["train" if split.startswith("train") else "validation"]
                    check_universe(base, target, f"{run} e{epoch} {arm} {split}")
                    machine["results"][run][str(epoch)][arm][split] = scalar_effect(base, target, cohort)
                for domain, groups, rows_split, base in (
                    ("CE20", ce_groups, "train", base_train),
                    ("KL10", kl_groups, "train", base_train),
                    ("CE20", ce_val_groups, "validation", base_val),
                    ("KL10", kl_val_groups, "validation", base_val),
                ):
                    target = endpoint_rows[epoch][arm][rows_split]
                    q_effects = {}
                    for name, ids in groups.items():
                        q_effects[name] = scalar_effect(base, target, ids)
                    machine["results"][run][str(epoch)][arm].setdefault("quantile_effects", {}).setdefault(domain, {})[rows_split] = q_effects
            # Factorial contrasts are computed from paired effect means where available.
            machine["factorial"].setdefault(run, {})[str(epoch)] = {}
            for cohort in ("train_direct_cw", "train_spillover_non_cw", "validation_cw", "validation_overall"):
                values = {arm: machine["results"][run][str(epoch)][arm][cohort] for arm in ARMS}
                for metric in ("clean", "robust"):
                    def val(arm: str) -> float: return float(values[arm][metric]["accuracy_delta"])
                    machine["factorial"][run][str(epoch)][cohort + "_" + metric] = {
                        "ce_no_margin": val("F1"),
                        "margin_no_ce": val("F2"),
                        "ce_given_margin": val("F3") - val("F2"),
                        "margin_given_ce": val("F3") - val("F1"),
                        "interaction": val("F3") - val("F2") - val("F1"),
                    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong A7 CleanCE ablation",
        "",
        "Status: completed 2x2 factorial assembly. Historical A0/A1/A7 are F0/F1/F2; only F3 was trained fresh.",
        "",
        "## Frozen interpretation boundary",
        "",
        "This is a two-seed, paired, descriptive ablation. Bootstrap/training-seed population inference is not claimed.",
        "Historical A7 is margin-only; treating it as full A7 would be incorrect.",
        "",
        "## F3 endpoint absolute metrics",
        "",
        "| seed | epoch | split | clean | robust |",
        "|---|---:|---|---:|---:|",
    ]
    for run in RUNS:
        for epoch in HORIZONS:
            for split in ("train", "validation"):
                absolute_values = machine["results"][run][str(epoch)]["F3"][split]
                lines.append(f"| {run} | {epoch} | {split} | {absolute_values['absolute']['clean_accuracy']:.4f} | {absolute_values['absolute']['robust_accuracy']:.4f} |")
    lines += ["", "## Epoch-94 factorial contrasts (paired accuracy deltas)", "", "| seed | cohort/metric | CE no margin | margin no CE | CE given margin | margin given CE | interaction |", "|---|---|---:|---:|---:|---:|---:|"]
    for run in RUNS:
        for key, value in machine["factorial"][run]["94"].items():
            lines.append(f"| {run} | {key} | {value['ce_no_margin']:+.4f} | {value['margin_no_ce']:+.4f} | {value['ce_given_margin']:+.4f} | {value['margin_given_ce']:+.4f} | {value['interaction']:+.4f} |")
    lines += ["", "## Held-out Clean-Wrong Q1--Q5", "", "The machine JSON contains CE20 and KL10 train-derived quantile boundaries and paired clean/robust effects for every arm and horizon. Validation Q groups are restricted to pre-treatment validation Clean-Wrong IDs; no outcome is used for binning.", "", "## Next decision", "", "Do not launch lambda/floor/cap sensitivity automatically. Human review is required after comparing F2 (margin-only) and F3 (full combination)."]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "source_git_sha": source_sha}))


if __name__ == "__main__":
    main()
