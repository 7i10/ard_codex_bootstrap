#!/usr/bin/env python3
"""Aggregate the sample-keyed attack RNG/history-ordering dev campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ATTACK_ID = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
ROOT = Path(
    "/home/shunsukenaito/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-history-ordering-v2-final"
)
ARMS = {"NEW_CONTROL": "epoch_shuffle_control", "NEW_HISTORY": "history_balanced_v1"}
REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values.sort(key=lambda item: int(item["epoch"]))
    if [int(item["epoch"]) for item in values] != list(range(100, 200)):
        raise ValueError(f"expected contiguous post-fork epochs 100..199: {path}")
    return values


def auc(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("AUC requires at least two points")
    return (values[0] / 2.0 + sum(values[1:-1]) + values[-1] / 2.0) / (len(values) - 1)


def endpoint(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("attack_identity_sha256") != ATTACK_ID or data.get("row_count") != 5000:
        raise ValueError(f"endpoint contract drift: {path}")
    return {
        "path": str(path.resolve()),
        "endpoint_json_sha256": sha256(path),
        "rows_sha256": data["rows_sha256"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "clean": float(data["clean_accuracy"]),
        "robust": float(data["robust_accuracy"]),
    }


def arm(seed: int, name: str) -> dict[str, Any]:
    directory = ROOT / f"{name.lower().replace('_', '-')}-s{seed}"
    metric_path = directory / "epoch-metrics.jsonl"
    metric_rows = rows(metric_path)
    robust = [float(item["val_pgd_accuracy"]) for item in metric_rows]
    clean = [float(item["val_clean_accuracy"]) for item in metric_rows]
    endpoints = {
        str(epoch): endpoint(directory / "endpoints" / f"epoch-{epoch}" / "validation" / "endpoint.json")
        for epoch in (149, 199)
    }
    return {
        "seed": seed,
        "arm": name,
        "ordering_policy": ARMS[name],
        "directory": str(directory.resolve()),
        "metrics_sha256": sha256(metric_path),
        "rows": len(metric_rows),
        "trajectory": {
            "first_epoch": 100,
            "last_epoch": 199,
            "last_robust": robust[-1],
            "last_clean": clean[-1],
            "best_robust": max(robust),
            "best_epoch": metric_rows[max(range(len(robust)), key=robust.__getitem__)]["epoch"],
            "post100_auc": auc(robust),
            "post100_clean_mean": sum(clean) / len(clean),
        },
        "endpoints": endpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "docs/experiments/ert_rslad_history_balanced_ordering_dev_v2_results.json",
    )
    parser.add_argument("--output-md", type=Path, default=REPO / "docs/ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md")
    args = parser.parse_args()
    arms = [arm(seed, name) for seed in (1, 2) for name in ARMS]
    by_key = {(item["seed"], item["arm"]): item for item in arms}
    comparisons = []
    for seed in (1, 2):
        control = by_key[(seed, "NEW_CONTROL")]
        history = by_key[(seed, "NEW_HISTORY")]
        comparisons.append(
            {
                "seed": seed,
                "history_minus_control": {
                    "post100_auc": history["trajectory"]["post100_auc"] - control["trajectory"]["post100_auc"],
                    "final_robust": history["trajectory"]["last_robust"] - control["trajectory"]["last_robust"],
                    "final_clean": history["trajectory"]["last_clean"] - control["trajectory"]["last_clean"],
                    "endpoint_149_robust": history["endpoints"]["149"]["robust"]
                    - control["endpoints"]["149"]["robust"],
                    "endpoint_199_robust": history["endpoints"]["199"]["robust"]
                    - control["endpoints"]["199"]["robust"],
                    "endpoint_199_clean": history["endpoints"]["199"]["clean"] - control["endpoints"]["199"]["clean"],
                },
            }
        )
    result = {
        "schema_version": 1,
        "kind": "ert_rslad_history_balanced_ordering_dev_v2_results",
        "status": "complete",
        "source_git_sha": "aafc5b7b18a557a027d9dcd4b0064bfcaf843404",
        "training_attack": {
            "keying": "sample_keyed_v1",
            "loss": "kl",
            "steps": 10,
            "epsilon": "8/255",
            "step_size": "2/255",
        },
        "endpoint_attack_identity_sha256": ATTACK_ID,
        "dataset": {"name": "cifar10", "train_count": 45000, "validation_count": 5000},
        "arms": arms,
        "paired_comparisons": comparisons,
        "promotion_gate": {
            "both_seeds_final_robust_positive": all(
                item["history_minus_control"]["final_robust"] > 0 for item in comparisons
            ),
            "both_seeds_post100_auc_nonnegative": all(
                item["history_minus_control"]["post100_auc"] >= 0 for item in comparisons
            ),
            "mean_final_robust_at_least_0_30pp": sum(
                item["history_minus_control"]["final_robust"] for item in comparisons
            )
            / 2
            >= 0.003,
            "clean_delta_at_least_minus_1pp": all(
                item["history_minus_control"]["final_clean"] >= -0.01 for item in comparisons
            ),
            "automatic_promotion": False,
        },
        "limitations": [
            "Two development seeds only; no confirmation seeds were run.",
            "Sample-keyed random-start generation uses a correctness-first CPU generator and exceeded the 5% "
            "random-start-only overhead target in the bounded benchmark.",
            "Internal validation only; no official test or AutoAttack.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT / RSLAD History-Balanced Ordering Dev v2",
        "",
        "Status: complete; dev seeds 1 and 2 only. The primary comparison is NEW_HISTORY minus NEW_CONTROL "
        "under the new sample-keyed training attack RNG contract.",
        "",
        "## Paired results",
        "",
        "| seed | final robust Δ | final clean Δ | post-100 AUC Δ | endpoint 149 robust Δ | endpoint 199 robust Δ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        delta = item["history_minus_control"]
        lines.append(
            f"| {item['seed']} | {delta['final_robust'] * 100:+.3f} pp | {delta['final_clean'] * 100:+.3f} pp | "
            f"{delta['post100_auc'] * 100:+.3f} pp | {delta['endpoint_149_robust'] * 100:+.3f} pp | "
            f"{delta['endpoint_199_robust'] * 100:+.3f} pp |"
        )
    lines += [
        "",
        "## Contract",
        "",
        "- Prefix: frozen I100 (CropShift epochs 0–99, IDBH_WEAK epochs 100–199).",
        "- Training attack: KL-PGD10, 8/255, 2/255, random start, teacher-clean target; random starts are keyed by "
        "attack seed, epoch, source ID, stream tag, and restart index only.",
        "- NEW_CONTROL uses canonical epoch shuffle; NEW_HISTORY uses frozen H2 `margin_ema` risk, HIGH/MID/LOW "
        "20/60/20, HIGH/MID/MID/LOW/MID interleave, exact-once exposure.",
        "- Endpoint: common CE-PGD20 on the fixed internal validation split (5,000 rows).",
        "- W&B policy: metrics-only; model and run-bundle uploads disabled.",
        "",
        "## Decision",
        "",
        "No automatic promotion is performed. Any confirmation-seed campaign requires human review.",
        "",
        f"Machine artifact: `{args.output_json}` (SHA-256 `{sha256(args.output_json)}`).",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "arms": len(arms)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
