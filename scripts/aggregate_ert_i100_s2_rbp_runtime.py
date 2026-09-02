#!/usr/bin/env python3
"""Aggregate completion/endpoint lineage for the S2 boundary screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_ENDPOINT_ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
EXPECTED_EPOCHS = (104, 109, 114)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for arm in args.arms:
        arm_root = args.campaign_root / arm
        summary_path = arm_root / "endpoints" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("attack_identity_sha256") != EXPECTED_ENDPOINT_ATTACK:
            raise ValueError(f"{arm}: endpoint attack identity mismatch")
        outputs = summary.get("outputs", [])
        if len(outputs) != 4:
            raise ValueError(f"{arm}: expected three validation and one train endpoint")
        validation = [item for item in outputs if item.get("dataset_scope") == "validation"]
        epochs = sorted(int(item["checkpoint_epoch"]) for item in validation)
        if tuple(epochs) != EXPECTED_EPOCHS:
            raise ValueError(f"{arm}: validation endpoint epochs are {epochs}")
        rows.append(
            {
                "arm": arm,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256(summary_path),
                "validation": [
                    {
                        "epoch": int(item["checkpoint_epoch"]),
                        "clean_accuracy": item["clean_accuracy"],
                        "robust_accuracy": item["robust_accuracy"],
                        "rows_sha256": item["rows_sha256"],
                    }
                    for item in sorted(validation, key=lambda value: int(value["checkpoint_epoch"]))
                ],
                "train_e114": next(
                    {
                        "clean_accuracy": item["clean_accuracy"],
                        "robust_accuracy": item["robust_accuracy"],
                        "rows_sha256": item["rows_sha256"],
                    }
                    for item in outputs
                    if item.get("dataset_scope") == "train"
                ),
            }
        )
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_rbp_runtime_aggregation_v1",
        "endpoint_attack_identity_sha256": EXPECTED_ENDPOINT_ATTACK,
        "arms": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
