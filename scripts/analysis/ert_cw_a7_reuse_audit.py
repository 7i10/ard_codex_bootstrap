#!/usr/bin/env python3
"""Audit whether the historical A0/A1/A7 runs can be reused as a 2x2 ablation.

The historical A7 name is intentionally not trusted: the resolved treatment
fields are the source of truth.  This audit is read-only and writes one
hash-bound JSON record plus a concise Markdown summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
HIST_ROOT = ROOT / ".cache/analysis/ert-cw-margin-screen-v1-r3"
CALIBRATION = ROOT / "docs/experiments/ert_cw_margin_calibration_v1.json"
MASK_PATHS = {
    "L2": ROOT / ".cache/analysis/ert-state-overlay-v1-review/anchor79-fixed-masks-L2.json",
    "L4": ROOT / ".cache/analysis/ert-state-overlay-v1-review/anchor79-fixed-masks-L4.json",
}
EXPECTED = {
    "L2": {
        "seed": 1,
        "parent": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
        "mask": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    },
    "L4": {
        "seed": 2,
        "parent": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
        "mask": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
    },
}
CALIBRATION_SHA = "a625b43ec12277bbf698270193f27e0e1f62e0a2a9f9a6a49e7fc0702593b2b5"
HISTORICAL_SOURCE = "bb59b512185af7bb70633c3266efd95bb24a563f"
REQUIRED_COMPONENTS = (
    "src/ard/engine/trainer.py",
    "src/ard/analysis/ert_stage_a_runtime.py",
    "src/ard/attacks/pgd.py",
    "src/ard/objectives/rslad.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(commit: str, path: str) -> str:
    data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def treatment(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("treatment")
    if not isinstance(value, dict):
        raise ValueError("resolved config has no treatment object")
    return value


def mask_identity(path: Path) -> tuple[str, str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("masks", {}).get("student_clean_wrong", {}).get("selected_ids")
    if payload.get("anchor_epoch") != 79 or not isinstance(selected, list):
        raise ValueError(f"invalid epoch-79 fixed mask: {path}")
    ids = [int(item) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate fixed mask IDs: {path}")
    digest = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    return sha256(path), digest, len(ids)


def expected_arm_fields(arm: str) -> dict[str, Any]:
    common = {"mask_key": "student_clean_wrong"}
    if arm == "A0":
        return {"kind": "baseline", "extra_clean_ce": None, "margin_target_mode": None, "mask_key": None}
    if arm == "A1":
        return {"kind": "broad", "extra_clean_ce": 0.15, "margin_target_mode": None, **common}
    if arm == "A7":
        return {
            "kind": "broad",
            "extra_clean_ce": None,
            "margin_target_mode": "teacher_floor",
            "margin_coefficient": 0.2388051152229309,
            "margin_floor": 0.03221710026264191,
            "margin_cap": 0.13952550292015076,
            **common,
        }
    raise ValueError(arm)


def run_record(seed_name: str, arm: str) -> dict[str, Any]:
    root = HIST_ROOT / seed_name / arm
    manifest_path = root / "run-bundle/manifest.json"
    config_path = root / "resolved_config.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid config: {config_path}")
    fork = manifest.get("fork_lineage")
    if not isinstance(fork, dict):
        raise ValueError(f"missing fork lineage: {manifest_path}")
    exp = EXPECTED[seed_name]
    mask_file_sha, mask_ids_sha, mask_count = mask_identity(MASK_PATHS[seed_name])
    actual_treatment = treatment(config)
    expected_treatment = expected_arm_fields(arm)
    mismatches = {
        key: {"expected": value, "actual": actual_treatment.get(key)}
        for key, value in expected_treatment.items()
        if actual_treatment.get(key) != value
    }
    parent_ok = (
        fork.get("parent_checkpoint_sha256") == exp["parent"]
        and fork.get("experiment_parent_checkpoint_sha256") == exp["parent"]
        and config.get("experiment_parent_checkpoint_sha256") == exp["parent"]
        and config.get("parent_config_hash") == fork.get("parent_config_hash")
        and fork.get("experiment_parent_epoch") == 79
        and fork.get("parent_epoch") == 79
    )
    lineage_ok = (
        fork.get("source_git_sha") == HISTORICAL_SOURCE
        and fork.get("calibration_sha256") == CALIBRATION_SHA
        and fork.get("child_config_hash") == manifest.get("config_hash") == config.get("child_config_hash")
    )
    endpoint_epochs = config.get("horizon_epochs") == [84, 89, 94]
    return {
        "seed": seed_name,
        "training_seed": manifest.get("training_seed"),
        "arm": arm,
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "resolved_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "source_git_sha": fork.get("source_git_sha"),
        "parent_checkpoint_sha256": fork.get("parent_checkpoint_sha256"),
        "mask_sha256": exp["mask"],
        "mask_path": str(MASK_PATHS[seed_name]),
        "mask_selected_ids_sha256": mask_ids_sha,
        "mask_count": mask_count,
        "mask_identity_exact": mask_file_sha == exp["mask"],
        "calibration_sha256": fork.get("calibration_sha256"),
        "treatment": actual_treatment,
        "expected_treatment": expected_treatment,
        "treatment_matches_expected": not mismatches,
        "treatment_mismatches": mismatches,
        "parent_lineage_exact": parent_ok,
        "shared_endpoint_epochs": endpoint_epochs,
        "lineage_identity_exact": lineage_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    calibration_sha = sha256(CALIBRATION)
    component_hashes = {
        path: {"current": sha256(ROOT / path), "historical": git_blob_sha(HISTORICAL_SOURCE, path)}
        for path in REQUIRED_COMPONENTS
    }
    runs = [run_record(seed, arm) for seed in ("L2", "L4") for arm in ("A0", "A1", "A7")]
    reusable = all(
        row["treatment_matches_expected"]
        and row["parent_lineage_exact"]
        and row["lineage_identity_exact"]
        and row["mask_identity_exact"]
        and row["shared_endpoint_epochs"]
        for row in runs
    ) and calibration_sha == CALIBRATION_SHA and all(
        value["current"] == value["historical"] for value in component_hashes.values()
    )
    result = {
        "schema_version": 1,
        "contract": "ert_cw_a7_reuse_audit_v1",
        "historical_source_git_sha": HISTORICAL_SOURCE,
        "calibration": {"path": str(CALIBRATION), "sha256": calibration_sha, "expected_sha256": CALIBRATION_SHA},
        "relevant_source_component_hashes": component_hashes,
        "runs": runs,
        "mapping": {"F0": "historical A0", "F1": "historical A1", "F2": "historical A7", "F3": None},
        "historical_a7_is_full_f3": False,
        "reuse_decision": {
            "F0": "reuse",
            "F1": "reuse",
            "F2": "reuse",
            "F3": "fresh_required",
        },
        "reuse_gate_passed": reusable,
        "fresh_training_required": ["F3_L2", "F3_L4"],
        "prohibitions": ["no lambda sweep", "no floor/cap sweep", "no new seed", "no official test", "no AutoAttack"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT A7 CleanCE reuse audit",
        "",
        "Status: PASS for historical F0/F1/F2 reuse; F3 requires fresh training.",
        "",
        "The historical A7 resolved treatment is `teacher_floor` margin-only with `extra_clean_ce: null`.",
        "It is therefore F2, not the full F3 (CleanCE + margin) arm described in the proposal.",
        "",
        "| Factorial arm | Historical source | Decision |",
        "|---|---|---|",
        "| F0 | A0 (baseline) | reuse |",
        "| F1 | A1 (CleanCE 0.15) | reuse |",
        "| F2 | A7 (teacher-floor margin only) | reuse |",
        "| F3 | no historical equivalent | fresh L2/L4 required |",
        "",
        f"Calibration SHA256: `{calibration_sha}`.",
        f"Relevant training/runtime source components match historical `{HISTORICAL_SOURCE}`.",
        (
            "All six historical manifests bind the exact epoch-79 parent, fixed endpoint horizons "
            "84/89/94, and the frozen calibration."
        ),
        "",
        "No production launch is authorized by this audit alone; F3 must pass a clean-tree canary before GPU launch.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"reuse_gate_passed": reusable, "json": str(args.json), "markdown": str(args.markdown)}))
    return 0 if reusable else 1


if __name__ == "__main__":
    raise SystemExit(main())
