#!/usr/bin/env python3
"""Write the immutable eight-schedule pure-order probe registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEDULES = [
    {
        "schedule_id": f"SHUFFLE_PLUS_{index}",
        "order_policy": "epoch_shuffle_offset",
        "order_seed_offset": index,
        "description": "ordinary epoch shuffle with a pre-registered data-order seed offset",
    }
    for index in range(8)
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or any(character not in "0123456789abcdef" for character in args.source_sha):
        raise ValueError("source SHA must be a lowercase 40-character Git SHA")
    registry = {
        "schema_version": 1,
        "kind": "ert_rslad_pure_order_probe_registry_v1",
        "status": "frozen_before_training",
        "source_git_sha": args.source_sha,
        "parent_payload_epoch": 99,
        "probe_epoch_start": 100,
        "probe_epoch_end_exclusive": 115,
        "risk_definition": "-margin_ema_snapshot_at_epoch_start",
        "telemetry": {
            "batch_stable_ids": True,
            "descriptors": [
                "D1_batch_mean_risk_sd",
                "D2_within_batch_risk_sd_mean",
                "D3_high_risk_fraction_sd",
                "D4_lag1_batch_mean_risk_acf",
                "D5_hard_batch_longest_run",
                "D6_position_vs_batch_mean_risk_spearman",
            ],
        },
        "selection_rule": {
            "metric": "probe_robust_auc_100_114",
            "eligible_same_sign_per_seed": True,
            "per_seed_abs_rho_min": 0.40,
            "mean_abs_rho_min": 0.50,
            "tie_priority": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "no_validation_accuracy_for_selection": True,
        },
        "schedules": SCHEDULES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(registry, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"path": str(args.output), "sha256": digest, "schedules": len(SCHEDULES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
