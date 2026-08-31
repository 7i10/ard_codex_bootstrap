#!/usr/bin/env python3
"""Combine the two immutable per-seed gradient geometry summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", type=Path, required=True)
    parser.add_argument("--seed2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {
        "1": json.loads(args.seed1.read_text(encoding="utf-8")),
        "2": json.loads(args.seed2.read_text(encoding="utf-8")),
    }
    for seed, row in rows.items():
        if row.get("contract") != "ert_rslad_ordering_gradient_geometry_v1" or row.get("no_update") is not True:
            raise ValueError(f"gradient contract mismatch for seed {seed}")
    source_shas = {row.get("source_git_sha") for row in rows.values()}
    parents = {seed: row["parent_checkpoint_sha256"] for seed, row in rows.items()}
    artifact = {
        "schema_version": 1,
        "contract": "ert_rslad_ordering_gradient_geometry_v1",
        "no_update": True,
        "source_git_shas": sorted(source_shas),
        "epoch": 99,
        "parents": parents,
        "per_seed": rows,
        "group_definition": "HIGH=lowest margin EMA top20%; MID=middle60%; LOW=highest margin EMA bottom20%",
        "selection_note": "descriptive diagnostic only; no optimizer/scheduler/sample-state update",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
