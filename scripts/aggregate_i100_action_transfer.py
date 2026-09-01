#!/usr/bin/env python3
"""Aggregate fixed I100 action-transfer endpoint rows without refitting choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ARMS = (
    "I100_CONTROL",
    "PILOT_S3_T1_WEAK_ADVCE",
    "CLEAN_WRONG_PLAIN_ADVCE",
    "CLEAN_WRONG_A7_MARGIN_ONLY",
)


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["sample_id"]): row for row in pq.read_table(path).to_pylist()}


def _paired(control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    if not ids:
        return {"n": 0, "clean_delta": None, "robust_delta": None, "clean_rescue": 0, "clean_harm": 0, "robust_rescue": 0, "robust_harm": 0}
    common = ids & control.keys() & treatment.keys()
    if common != ids:
        raise ValueError(f"endpoint stable-ID mismatch for {len(ids - common)} rows")
    clean_rescue = clean_harm = robust_rescue = robust_harm = 0
    clean_margin = robust_margin = 0.0
    for sid in sorted(ids):
        c, t = control[sid], treatment[sid]
        cc, tc = bool(c["clean_correct"]), bool(t["clean_correct"])
        cr, tr = bool(c["robust_correct"]), bool(t["robust_correct"])
        clean_rescue += int(not cc and tc)
        clean_harm += int(cc and not tc)
        robust_rescue += int(not cr and tr)
        robust_harm += int(cr and not tr)
        clean_margin += float(t["clean_probability_margin"]) - float(c["clean_probability_margin"])
        robust_margin += float(t["adversarial_probability_margin"]) - float(c["adversarial_probability_margin"])
    n = len(common)
    return {
        "n": n,
        "clean_delta": (clean_rescue - clean_harm) / n,
        "robust_delta": (robust_rescue - robust_harm) / n,
        "clean_rescue": clean_rescue,
        "clean_harm": clean_harm,
        "clean_net_rescue": clean_rescue - clean_harm,
        "robust_rescue": robust_rescue,
        "robust_harm": robust_harm,
        "robust_net_rescue": robust_rescue - robust_harm,
        "clean_margin_delta": clean_margin / n,
        "robust_margin_delta": robust_margin / n,
    }


def _endpoint(root: Path, seed: str, arm: str, epoch: int, split: str) -> dict[int, dict[str, Any]]:
    suffix = f"e{epoch}-{split}"
    return _rows(root / "runs" / seed / arm / "endpoints" / suffix / "endpoint-sample-stats.parquet")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, masks_root = args.root.resolve(), args.masks.resolve()
    mask_payload = {
        seed: json.loads((masks_root / f"{seed}.json").read_text(encoding="utf-8"))
        for seed in ("dev-1", "dev-2")
    }
    results: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_action_transfer_results_v1",
        "root": str(root),
        "arms": list(ARMS),
        "endpoints": {},
        "effects": [],
    }
    for seed in ("dev-1", "dev-2"):
        selected = {arm: set(int(x) for x in mask_payload[seed]["masks"][key]["selected_ids"]) for arm, key in {
            "PILOT_S3_T1_WEAK_ADVCE": "pilot_s3_t1",
            "CLEAN_WRONG_PLAIN_ADVCE": "clean_wrong",
            "CLEAN_WRONG_A7_MARGIN_ONLY": "clean_wrong",
        }.items()}
        selected["I100_CONTROL"] = set()
        cw = set(int(x) for x in mask_payload[seed]["masks"]["clean_wrong"]["selected_ids"])
        for epoch in (104, 109, 114):
            control_val = _endpoint(root, seed, "I100_CONTROL", epoch, "validation")
            for arm in ARMS:
                rows = _endpoint(root, seed, arm, epoch, "validation")
                # Validation is the common held-out endpoint for every arm.
                effect = _paired(control_val, rows, set(control_val))
                results["effects"].append({"seed": seed, "epoch": epoch, "arm": arm, "scope": "heldout", **effect})
                endpoint_json = root / "runs" / seed / arm / "endpoints" / f"e{epoch}-validation" / "endpoint.json"
                payload = json.loads(endpoint_json.read_text(encoding="utf-8"))
                results["endpoints"][f"{seed}/{arm}/e{epoch}/validation"] = {
                    "clean_accuracy": payload["clean_accuracy"], "robust_accuracy": payload["robust_accuracy"],
                    "checkpoint_sha256": payload["checkpoint_sha256"], "rows_sha256": payload["rows_sha256"],
                }
        control_train = _endpoint(root, seed, "I100_CONTROL", 114, "train")
        for arm in ARMS:
            rows = _endpoint(root, seed, arm, 114, "train")
            group_ids = selected[arm]
            # Direct is the fixed action cohort; spillover is its complement in
            # the train universe.  For the control arm the selected set is empty
            # and the effect is the expected zero identity.
            for scope, ids in (("direct", group_ids), ("spillover", set(control_train) - group_ids)):
                results["effects"].append({"seed": seed, "epoch": 114, "arm": arm, "scope": scope, **_paired(control_train, rows, ids)})
            endpoint_json = root / "runs" / seed / arm / "endpoints" / "e114-train" / "endpoint.json"
            payload = json.loads(endpoint_json.read_text(encoding="utf-8"))
            results["endpoints"][f"{seed}/{arm}/e114/train"] = {
                "clean_accuracy": payload["clean_accuracy"], "robust_accuracy": payload["robust_accuracy"],
                "checkpoint_sha256": payload["checkpoint_sha256"], "rows_sha256": payload["rows_sha256"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "effects": len(results["effects"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
