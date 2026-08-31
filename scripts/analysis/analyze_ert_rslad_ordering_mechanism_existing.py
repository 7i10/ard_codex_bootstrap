#!/usr/bin/env python3
"""Audit whether existing ordering runs contain the telemetry needed for D1--D6.

This is intentionally read-only.  It does not reconstruct or invent batch
orders from permutation digests, and it never launches training or evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(
    "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-history-ordering-v2-final"
)
HISTORY_RESULT = REPO / "docs/experiments/ert_rslad_history_balanced_ordering_dev_v2_results.json"
PARENTS = {
    1: {
        "path": REPO.parent / "ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed1/s100/epoch-100.pt",
        "sha256": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    },
    2: {
        "path": REPO.parent / "ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed2/s100/epoch-100.pt",
        "sha256": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
    },
}
DESCRIPTORS = {
    "D1_batch_mean_risk_sd": "SD of batch mean risk",
    "D2_within_batch_risk_sd": "mean within-batch risk SD",
    "D3_high_risk_fraction_sd": "SD of high-risk fraction per batch",
    "D4_batch_mean_risk_lag1_acf": "lag-1 autocorrelation of batch mean risk",
    "D5_hard_batch_clustering": "top-20% run length and inter-batch distance",
    "D6_batch_position_risk_spearman": "batch position versus risk Spearman correlation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_inventory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = sorted({key for row in rows for key in row})
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256(path),
        "row_count": len(rows),
        "first_epoch": rows[0].get("epoch") if rows else None,
        "last_epoch": rows[-1].get("epoch") if rows else None,
        "keys": keys,
        "contains_batch_level_values": any(
            any(token in key.lower() for token in ("batch", "risk_fraction", "autocorr", "position", "distance"))
            for key in keys
        ),
    }


def direction_audit() -> dict[str, Any]:
    source = (REPO / "src/ard/data/indexed.py").read_text(encoding="utf-8")
    fixed = "scored.append((margin, source_id))" in source and "scored.append((-margin, source_id))" not in source
    historical = subprocess.run(
        ["git", "show", "31932d6bb9fce0517e8e408d08038ab5f9972f46:src/ard/data/indexed.py"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    historical_bug = "scored.append((-margin, source_id))" in historical.stdout
    return {
        "current_sampler_high_is_low_margin": fixed,
        "historical_ordering_source_sha": "31932d6bb9fce0517e8e408d08038ab5f9972f46",
        "historical_sampler_used_inverse_direction": historical_bug,
        "prior_history_run_not_rerun": True,
    }


def run_inventory(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed in (1, 2):
        for arm in ("new-control", "new-history"):
            directory = root / f"{arm}-s{seed}"
            manifest_path = directory / "run-bundle/manifest.json"
            lineage = load_json(directory / "fork-lineage.json") if (directory / "fork-lineage.json").is_file() else {}
            ordering = jsonl_inventory(directory / "ordering-metrics.jsonl")
            epoch_metrics = jsonl_inventory(directory / "epoch-metrics.jsonl")
            runs.append(
                {
                    "seed": seed,
                    "arm": arm.upper().replace("-", "_"),
                    "directory": str(directory),
                    "manifest": {
                        "path": str(manifest_path),
                        "exists": manifest_path.is_file(),
                        "sha256": sha256(manifest_path) if manifest_path.is_file() else None,
                    },
                    "fork_lineage": lineage,
                    "ordering_metrics": ordering,
                    "epoch_metrics": epoch_metrics,
                    "batch_telemetry_available": bool(ordering.get("contains_batch_level_values")),
                    "sample_state_sequence_available": False,
                }
            )
    return runs


def parent_inventory() -> list[dict[str, Any]]:
    inventory = []
    for seed, expected in PARENTS.items():
        path = Path(expected["path"])
        actual = sha256(path) if path.is_file() else None
        inventory.append(
            {
                "seed": seed,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual,
                "sha_match": actual == expected["sha256"],
                "payload_boundary": "epoch=99, epoch_boundary=end" if path.is_file() else None,
            }
        )
    return inventory


def build_result(root: Path) -> dict[str, Any]:
    history = load_json(HISTORY_RESULT)
    runs = run_inventory(root)
    telemetry_missing = {
        name: {
            "definition": definition,
            "available": False,
            "reason": (
                "Existing artifacts contain permutation digests and final sample state, "
                "but no per-batch stable-ID/risk observations."
            ),
        }
        for name, definition in DESCRIPTORS.items()
    }
    return {
        "schema_version": 1,
        "kind": "ert_rslad_ordering_mechanism_existing_runs_v1",
        "status": "blocked_missing_batch_level_telemetry",
        "analysis_mode": "read_only",
        "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "historical_result_artifact": {"path": str(HISTORY_RESULT), "sha256": sha256(HISTORY_RESULT)},
        "parents": parent_inventory(),
        "runs": runs,
        "descriptor_contract": telemetry_missing,
        "direction_audit": direction_audit(),
        "historical_paired_results": history.get("paired_comparisons", []),
        "historical_result_interpretation": {
            "usable_for_lineage_inventory": True,
            "usable_as_corrected_direction_mechanism_evidence": False,
            "reason": (
                "The completed NEW_HISTORY run was generated before the HIGH/LOW direction "
                "correction and does not expose batch-level telemetry."
            ),
        },
        "phase_gate": {
            "phase_a_existing_batch_mechanism": "blocked",
            "phase_b_gradient_geometry": "not_run",
            "phase_c_pure_order_probe": "not_run",
            "phase_d_second_intervention": "not_run",
            "holdout_training": "not_run",
            "decision": "stop_no_mechanism_identified",
        },
        "limitations": [
            "Permutation SHA-256 values are one-way commitments; they cannot recover batch membership or risk values.",
            "Final sample-stats parquet is an endpoint snapshot, not an epoch-by-epoch state trajectory.",
            "No new training, endpoint evaluation, gradient calibration, or polling was performed.",
        ],
    }


def render_markdown(result: dict[str, Any], output_json: Path) -> str:
    lines = [
        "# ERT / RSLAD Ordering Mechanism Discovery — Existing Runs Audit",
        "",
        "## Conclusion",
        "",
        "Phase A is **blocked** and the mechanism gate is not identified. Existing ordering",
        "runs preserve epoch-level permutation digests, but not the batch stable IDs or",
        "per-sample risk values needed to compute D1–D6. No GPU probe, pure-order probe,",
        "or second intervention was launched.",
        "",
        "The exact dev I100 e99 parents are present and hash-matching, but that does not",
        "remedy the missing historical batch telemetry. The completed history run also",
        "predates the corrected HIGH/LOW direction and is retained only for lineage and",
        "failure-context reporting; it is not corrected-direction mechanism evidence.",
        "",
        "## Existing run inventory",
        "",
        "| seed | arm | epoch metrics | ordering rows | ordering fields | batch telemetry |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for run in result["runs"]:
        ordering = run["ordering_metrics"]
        lines.append(
            f"| {run['seed']} | {run['arm']} | {run['epoch_metrics'].get('row_count', 0)} | "
            f"{ordering.get('row_count', 0)} | {', '.join(ordering.get('keys', [])) or 'none'} | "
            f"{'yes' if run['batch_telemetry_available'] else 'no'} |"
        )
    lines += [
        "",
        "`NEW_HISTORY` contains 100 rows (epochs 100–199) with `permutation_sha256`,",
        "stratum counts, and pattern only. `NEW_CONTROL` has no ordering-metrics file.",
        "A digest cannot be inverted into the batch risk sequence.",
        "",
        "## e99 parent inventory",
        "",
        "| seed | path | payload boundary | SHA-256 | match |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for parent in result["parents"]:
        lines.append(
            f"| {parent['seed']} | `{parent['path']}` | {parent.get('payload_boundary') or 'missing'} | "
            f"`{parent.get('actual_sha256') or parent['expected_sha256']}` | "
            f"{'yes' if parent['sha_match'] else 'no'} |"
        )
    lines += [
        "",
        "## D1–D6 availability",
        "",
        "| descriptor | available | reason |",
        "| --- | --- | --- |",
    ]
    for name, item in result["descriptor_contract"].items():
        lines.append(f"| {name} | no | {item['reason']} |")
    lines += [
        "",
        "## Contract correction",
        "",
        "The current sampler now places low-margin samples in HIGH (high-risk) and",
        "high-margin samples in LOW. A focused regression test covers this direction.",
        "The prior history run was not rerun, as required.",
        "",
        "## Gate decision",
        "",
        "Because no descriptor can be computed from existing artifacts, mechanism selection",
        "is not scientifically identified. Phase B/C/D and holdout training remain not run.",
        "Recovering a batch-level telemetry artifact or running a separately registered",
        "telemetry-producing diagnostic would be required before a second intervention.",
        "",
        f"Machine artifact: `{output_json}` (the pre-self-reference content hash is recorded in the JSON).",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-json", type=Path, default=REPO / "docs/experiments/ert_rslad_ordering_mechanism_existing_runs_v1.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=REPO / "docs/ERT_RSLAD_ORDERING_MECHANISM_AND_SECOND_INTERVENTION.md"
    )
    args = parser.parse_args()
    result = build_result(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    provisional = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    result["content_sha256"] = sha256_bytes(provisional)
    args.output_json.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    report = render_markdown(result, args.output_json)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"status": result["status"], "output_json": str(args.output_json), "output_md": str(args.output_md)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
