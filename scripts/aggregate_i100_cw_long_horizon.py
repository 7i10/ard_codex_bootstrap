#!/usr/bin/env python3
"""Aggregate the completed I100 Clean-Wrong long-horizon screen.

The endpoint rows are copied from the immutable Ferret/Hamster run outputs into
an analysis staging root.  This script deliberately computes paired effects
from stable IDs and never selects an arm from the outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np

SEEDS = ("dev-1", "dev-2")
ARMS = ("I100_CONTROL", "CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM")
EPOCHS = (129, 149, 169, 189, 199)


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(r["sample_id"]): r for r in pq.read_table(path).to_pylist()}


def paired(c: dict[int, dict[str, Any]], t: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    ids = ids & set(c) & set(t)
    n = len(ids)
    if not n:
        return {"n": 0}
    cr = ch = rr = rh = 0
    cm = rm = 0.0
    for sid in ids:
        a, b = c[sid], t[sid]
        cc, tc = bool(a["clean_correct"]), bool(b["clean_correct"])
        ar, tr = bool(a["robust_correct"]), bool(b["robust_correct"])
        cr += int(not cc and tc); ch += int(cc and not tc)
        rr += int(not ar and tr); rh += int(ar and not tr)
        cm += float(b["clean_probability_margin"]) - float(a["clean_probability_margin"])
        rm += float(b["adversarial_probability_margin"]) - float(a["adversarial_probability_margin"])
    return {"n": n, "clean_rescue": cr, "clean_harm": ch,
            "clean_net_rescue": cr - ch, "clean_delta": (cr - ch) / n,
            "robust_rescue": rr, "robust_harm": rh,
            "robust_net_rescue": rr - rh, "robust_delta": (rr - rh) / n,
            "clean_margin_delta": cm / n, "robust_margin_delta": rm / n}


def endpoint(root: Path, seed: str, arm: str, epoch: int, split: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    d = root / "runs" / seed / arm / "endpoints" / f"e{epoch}-{split}"
    meta = json.loads((d / "endpoint.json").read_text(encoding="utf-8"))
    return load_rows(d / "endpoint-sample-stats.parquet"), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    root, masks, out = args.root.resolve(), args.masks.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mask_ids: dict[str, set[int]] = {}
    mask_meta: dict[str, Any] = {}
    for seed in SEEDS:
        m = json.loads((masks / f"{seed}.json").read_text(encoding="utf-8"))
        mask_meta[seed] = {"selected_count": m["masks"]["clean_wrong"]["selected_count"],
                           "selected_ids_sha256": m["masks"]["clean_wrong"]["selected_ids_sha256"],
                           "replay_lineage_sha256": m["replay_lineage_sha256"]}
        mask_ids[seed] = set(map(int, m["masks"]["clean_wrong"]["selected_ids"]))

    result: dict[str, Any] = {"schema_version": 1, "contract": "ert_rslad_i100_cw_long_horizon_results_v1",
        "root": str(root), "seeds": list(SEEDS), "arms": list(ARMS), "epochs": list(EPOCHS),
        "attack_identity_sha256": None, "endpoints": [], "effects": [], "trajectory": [], "runtime": [], "post114_summary": [],
        "mask": mask_meta, "technical_recovery": {"dev-2/CLEAN_WRONG_TPFM": "Hamster direct-chain recovery after Ferret pre-training hash validation failures"}}

    for seed in SEEDS:
        for arm in ARMS:
            metrics = [json.loads(x) for x in (root / "runs" / seed / arm / "epoch-metrics.jsonl").read_text().splitlines() if x.strip()]
            if metrics:
                result["runtime"].append({"seed": seed, "arm": arm, "epochs": len(metrics),
                    "first_epoch": metrics[0]["epoch"], "last_epoch": metrics[-1]["epoch"],
                    "mean_train_seconds": sum(float(m["train_seconds"]) for m in metrics) / len(metrics),
                    "mean_images_per_second": sum(float(m["train_images_per_second"]) for m in metrics) / len(metrics),
                    "total_train_seconds": sum(float(m["train_seconds"]) for m in metrics)})
                for m in metrics:
                    result["trajectory"].append({"seed": seed, "arm": arm, **m})
                x = np.asarray([float(m["epoch"]) for m in metrics])
                y = np.asarray([float(m["val_pgd_accuracy"]) for m in metrics])
                result["post114_summary"].append({"seed": seed, "arm": arm,
                    "epoch_first": int(x[0]), "epoch_last": int(x[-1]),
                    "normalized_auc": float(np.trapezoid(y, x) / (x[-1] - x[0])),
                    "mean_val_pgd_accuracy": float(y.mean()), "best_val_pgd_accuracy": float(y.max()),
                    "last_val_pgd_accuracy": float(y[-1])})
        for epoch in EPOCHS:
            loaded = {}
            for arm in ARMS:
                rows, meta = endpoint(root, seed, arm, epoch, "validation")
                loaded[arm] = rows
                if result["attack_identity_sha256"] is None:
                    result["attack_identity_sha256"] = meta["attack_identity_sha256"]
                if meta["attack_identity_sha256"] != result["attack_identity_sha256"]:
                    raise ValueError("endpoint attack identity mismatch")
                result["endpoints"].append({"seed": seed, "epoch": epoch, "arm": arm,
                    "split": "validation", "clean_accuracy": meta["clean_accuracy"],
                    "robust_accuracy": meta["robust_accuracy"], "checkpoint_sha256": meta["checkpoint_sha256"],
                    "rows_sha256": meta["rows_sha256"], "row_count": meta["row_count"]})
            control = loaded["I100_CONTROL"]
            for arm in ARMS:
                result["effects"].append({"seed": seed, "epoch": epoch, "arm": arm, "scope": "heldout", **paired(control, loaded[arm], set(control))})
        # Train endpoint at e199: direct is the fixed CW cohort; spillover its complement.
        train = {}
        for arm in ARMS:
            train[arm], meta = endpoint(root, seed, arm, 199, "train")
            result["endpoints"].append({"seed": seed, "epoch": 199, "arm": arm, "split": "train",
                "clean_accuracy": meta["clean_accuracy"], "robust_accuracy": meta["robust_accuracy"],
                "checkpoint_sha256": meta["checkpoint_sha256"], "rows_sha256": meta["rows_sha256"], "row_count": meta["row_count"]})
        universe = set(train["I100_CONTROL"])
        for arm in ARMS:
            ids = mask_ids[seed] if arm != "I100_CONTROL" else set()
            result["effects"].append({"seed": seed, "epoch": 199, "arm": arm, "scope": "direct", **paired(train["I100_CONTROL"], train[arm], ids)})
            result["effects"].append({"seed": seed, "epoch": 199, "arm": arm, "scope": "spillover", **paired(train["I100_CONTROL"], train[arm], universe - ids)})
    (out / "ert_rslad_i100_cw_long_horizon_results_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = {"schema_version": 1, "contract": "ert_rslad_i100_cw_long_horizon_contract_v1",
        "source_git_sha": "c6032f9dc09f938fd0b9fb87379cf16c3f0f26bb",
        "manifest_sha256": "2ebe1aa63a49b39c868d6f28d9dc19e52c44c5e92a9bb2ea7cdd3960cb040cb4",
        "parent_epoch": 114, "endpoint_epochs": list(EPOCHS), "attack_identity_sha256": result["attack_identity_sha256"],
        "masks": mask_meta, "coefficients": {"beta_advce": 0.11834514302628477, "margin_coefficient": 0.316427398202933, "margin_floor": 0.17963354289531708, "margin_cap": 0.5595575273036957},
        "training_attack": "KL-PGD10; eps=8/255; step=2/255; random_start=true; Teacher-clean target",
        "endpoint_attack": "CE-PGD20; eps=8/255; step=2/255; random_start=true; eval mode",
        "recovery": "dev-2/CLEAN_WRONG_TPFM rerun on Hamster after Ferret technical SHA/path failures; scientific identity unchanged"}
    (out / "ert_rslad_i100_cw_long_horizon_contract_v1.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "ert_rslad_i100_cw_long_horizon_direct_spillover_v1.json").write_text(
        json.dumps({"schema_version": 1, "contract": "ert_rslad_i100_cw_long_horizon_direct_spillover_v1",
                    "effects": [e for e in result["effects"] if e["scope"] in ("direct", "spillover")]},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "ert_rslad_i100_cw_long_horizon_runtime_v1.json").write_text(
        json.dumps({"schema_version": 1, "contract": "ert_rslad_i100_cw_long_horizon_runtime_v1",
                    "runtime": result["runtime"], "post114_summary": result["post114_summary"]},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "endpoint_rows": len(result["endpoints"]), "effects": len(result["effects"]), "trajectory_rows": len(result["trajectory"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
