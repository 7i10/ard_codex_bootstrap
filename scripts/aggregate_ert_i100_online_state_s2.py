#!/usr/bin/env python3
"""Aggregate the preregistered I100 Online-State S2×T1 preservation screen.

The script is deliberately read-only with respect to checkpoints and replay
rows.  It accepts only the canonical local collection tree produced by the
campaign DAG; execution-host absolute paths in manifests are provenance, not
an aggregation input.  The only writes are a non-overwriting JSON result and
its human-facing Markdown report after every required lineage assertion has
passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.analysis.ert_i100_s2_longitudinal import CONTRACT as CANONICAL_CONTRACT
from ard.analysis.ert_i100_s2_longitudinal import canonical_action_states

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "docs/experiments/ert_rslad_i100_online_state_s2_preservation_v1.json"
REPORT_PATH = ROOT / "docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md"
SOURCE_PREFIX_CONTRACT = "ert_rslad_i100_online_state_s2_prefix_v1"
THRESHOLD_CONTRACT = "ert_rslad_i100_online_state_s2_preservation_v1"
ENDPOINT_CONTRACT = "ert_stage_a_common_ce_pgd20_endpoint_v1"
ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
TRAIN_ATTACK_SHA256 = "97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d"
CALIBRATION_SHA256 = "37bf0a0e1aa6ff12951f1c05f59f6df55700be0e28291c6925670d7b6cb56840"
TEACHER_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
VALIDATION_SPLIT_SHA256 = "16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4"
PARENT_SHA256 = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
HISTORICAL_E99_ROWS = {
    "dev-1": {
        "path": ROOT / ".cache/analysis/ert-i100-cw-gap-completion-replay/dev-1/e99-observations.parquet",
        "sha256": "7899a0f7473862b7d7edd35e6009794d8cb4feb8e485c2ad28708c8a6c933d17",
    },
    "dev-2": {
        "path": ROOT / ".cache/analysis/ert-i100-cw-gap-completion-replay/dev-2/e99-observations.parquet",
        "sha256": "66390e55cf32615e32f179d88b7b0c407cc8afc9efadbfa43944bb22a309ce60",
    },
}
ARMS = ("control", "pmp", "dbdp")
SEEDS = ("dev-1", "dev-2")
HORIZONS = (104, 109, 114)


class AggregationError(RuntimeError):
    """An immutable lineage, endpoint, or paired-metric contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"missing required JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AggregationError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"expected a JSON object: {path}")
    return value


def _read_rows(path: Path, *, expected_count: int) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise AggregationError(f"missing required row artifact: {path}")
    rows = pq.read_table(path).to_pylist()
    by_id = {int(row["sample_id"]): dict(row) for row in rows}
    if len(rows) != expected_count or len(by_id) != expected_count:
        raise AggregationError(f"stable-ID row coverage differs at {path}: expected {expected_count}, got {len(rows)}")
    return by_id


def _require_clean_source(expected: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if actual != expected or dirty:
        raise AggregationError("aggregation requires the frozen clean production source")


def _non_overwriting(path: Path) -> None:
    if path.exists():
        raise AggregationError(f"refusing to overwrite registered output: {path}")


def _local_or_declared(*, declared: object, local: Path, expected_sha256: str, kind: str) -> Path:
    """Resolve a collected row file only after byte-hash verification."""

    candidates: list[Path] = []
    if isinstance(declared, str):
        candidates.append(Path(declared))
    candidates.append(local)
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == expected_sha256:
            return candidate.resolve()
    raise AggregationError(f"{kind} is unavailable locally or differs from its declared SHA-256: {local}")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise AggregationError("cannot aggregate an empty metric population")
    return float(sum(materialized) / len(materialized))


def _fmt_pp(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f} pp"


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _fmt_scalar(value: float | None) -> str:
    return "—" if value is None else f"{value:+.6f}"


def _effect(
    control: Mapping[int, Mapping[str, Any]],
    treatment: Mapping[int, Mapping[str, Any]],
    *,
    clean_key: str,
    robust_key: str,
    clean_margin_key: str,
    robust_margin_key: str,
) -> dict[str, Any]:
    """Stable-ID paired endpoint effect with explicit rescue/harm identity."""

    ids = sorted(control)
    if not ids or set(control) != set(treatment):
        raise AggregationError("paired treatment rows do not have identical stable-ID coverage")
    output: dict[str, Any] = {"n": len(ids)}
    for name, correct_key, margin_key in (
        ("clean", clean_key, clean_margin_key),
        ("robust", robust_key, robust_margin_key),
    ):
        rescue = sum(
            not bool(control[sample_id][correct_key]) and bool(treatment[sample_id][correct_key]) for sample_id in ids
        )
        harm = sum(
            bool(control[sample_id][correct_key]) and not bool(treatment[sample_id][correct_key]) for sample_id in ids
        )
        accuracy_delta = _mean(
            float(bool(treatment[sample_id][correct_key])) - float(bool(control[sample_id][correct_key]))
            for sample_id in ids
        )
        margin_delta = _mean(
            float(treatment[sample_id][margin_key]) - float(control[sample_id][margin_key]) for sample_id in ids
        )
        result = {
            "accuracy_delta": accuracy_delta,
            "margin_delta": margin_delta,
            "rescue_count": rescue,
            "harm_count": harm,
            "net_rescue_count": rescue - harm,
            "rescue_rate": rescue / len(ids),
            "harm_rate": harm / len(ids),
            "net_rescue_rate": (rescue - harm) / len(ids),
        }
        if not math.isclose(result["accuracy_delta"], result["net_rescue_rate"], abs_tol=1e-12):
            raise AggregationError(f"{name} accuracy delta differs from rescue minus harm")
        output[name] = result
    return output


def _endpoint_payload(endpoint_root: Path, *, seed: str, arm: str) -> dict[int, dict[str, Any]]:
    summary = _read_json(endpoint_root / "summary.json")
    if (
        summary.get("contract") != "ert_rslad_i100_online_state_s2_endpoint_v1"
        or summary.get("seed") != seed
        or summary.get("arm") != arm
    ):
        raise AggregationError(f"endpoint summary identity differs: {endpoint_root}")
    by_epoch: dict[int, dict[str, Any]] = {}
    outputs = summary.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(HORIZONS):
        raise AggregationError(f"endpoint summary lacks exactly e104/e109/e114 outputs: {endpoint_root}")
    for descriptor in outputs:
        if not isinstance(descriptor, dict):
            raise AggregationError("endpoint descriptor is malformed")
        epoch = descriptor.get("checkpoint_epoch")
        expected_sha = descriptor.get("rows_sha256")
        if (
            not isinstance(epoch, int)
            or epoch not in HORIZONS
            or descriptor.get("contract") != ENDPOINT_CONTRACT
            or descriptor.get("dataset_scope") != "validation"
            or descriptor.get("attack_identity_sha256") != ENDPOINT_ATTACK_SHA256
            or descriptor.get("split_identity", {}).get("sample_id_label_sha256") != VALIDATION_SPLIT_SHA256
            or not isinstance(expected_sha, str)
        ):
            raise AggregationError(f"endpoint contract differs at {endpoint_root}, e{epoch}")
        local = endpoint_root / f"e{epoch}-validation" / "endpoint-sample-stats.parquet"
        rows_path = _local_or_declared(
            declared=descriptor.get("rows_path"), local=local, expected_sha256=expected_sha, kind="endpoint rows"
        )
        rows = _read_rows(rows_path, expected_count=5_000)
        if any(int(row["true_label"]) < 0 for row in rows.values()):
            raise AggregationError("endpoint rows lack valid class labels")
        descriptor = dict(descriptor)
        descriptor["local_rows_path"] = str(rows_path)
        descriptor["rows"] = rows
        by_epoch[epoch] = descriptor
    if set(by_epoch) != set(HORIZONS):
        raise AggregationError("endpoint horizon set differs")
    return by_epoch


def _prefix_and_threshold(campaign: Path, *, seed: str, source_sha: str) -> dict[str, Any]:
    prefix_root = campaign / "prefix" / seed
    prefix = _read_json(prefix_root / "prefix-summary.json")
    result = prefix.get("result")
    if (
        prefix.get("contract") != SOURCE_PREFIX_CONTRACT
        or prefix.get("seed") != seed
        or prefix.get("source_git_sha") != source_sha
        or not isinstance(result, dict)
        or result.get("parent_checkpoint_sha256") != PARENT_SHA256[seed]
    ):
        raise AggregationError(f"shared e100 prefix lineage differs for {seed}")
    prefix_checkpoint = prefix_root / "training/checkpoints/epoch-100.pt"
    if not prefix_checkpoint.is_file():
        raise AggregationError(f"shared e100 prefix checkpoint is missing: {prefix_checkpoint}")
    horizon = result.get("horizon_checkpoints")
    descriptor = horizon.get("100") if isinstance(horizon, dict) else None
    if not isinstance(descriptor, dict) or descriptor.get("sha256") != sha256(prefix_checkpoint):
        raise AggregationError(f"shared e100 prefix checkpoint hash differs for {seed}")
    threshold_path = campaign / "thresholds" / seed / "frozen-thresholds.json"
    threshold = _read_json(threshold_path)
    sidecar = threshold_path.with_name(threshold_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256(threshold_path):
        raise AggregationError(f"frozen threshold sidecar differs for {seed}")
    thresholds = threshold.get("thresholds")
    if (
        threshold.get("contract") != THRESHOLD_CONTRACT
        or threshold.get("kind") != "frozen_thresholds"
        or threshold.get("source_git_sha") != source_sha
        or threshold.get("original_parent_checkpoint_sha256") != PARENT_SHA256[seed]
        or threshold.get("training_attack_identity_sha256") != TRAIN_ATTACK_SHA256
        or not isinstance(thresholds, dict)
        or not all(
            isinstance(thresholds.get(name), (int, float)) and thresholds[name] > 0
            for name in ("student_global_logit_q10", "teacher_global_logit_q10")
        )
    ):
        raise AggregationError(f"frozen e100 threshold contract differs for {seed}")
    return {
        "prefix_checkpoint": str(prefix_checkpoint.resolve()),
        "prefix_checkpoint_sha256": sha256(prefix_checkpoint),
        "threshold_path": str(threshold_path.resolve()),
        "threshold_sha256": sha256(threshold_path),
        "thresholds": thresholds,
    }


def _training_payload(
    campaign: Path,
    *,
    seed: str,
    arm: str,
    source_sha: str,
    prefix: Mapping[str, Any],
) -> dict[str, Any]:
    arm_root = campaign / "arms" / seed / arm
    training_root = arm_root / "training"
    summary = _read_json(arm_root / "arm-summary.json")
    result = summary.get("result")
    if (
        summary.get("contract") != "ert_rslad_i100_online_state_s2_arm_v1"
        or summary.get("seed") != seed
        or summary.get("arm") != arm
        or summary.get("source_git_sha") != source_sha
        or not isinstance(result, dict)
        or result.get("parent_checkpoint_sha256") != prefix["prefix_checkpoint_sha256"]
    ):
        raise AggregationError(f"child arm lineage differs for {seed}/{arm}")
    horizon = _read_json(training_root / "horizon-checkpoints.json")
    if set(horizon) != {str(epoch) for epoch in HORIZONS}:
        raise AggregationError(f"child horizon checkpoint set differs for {seed}/{arm}")
    checkpoints: dict[int, dict[str, Any]] = {}
    for epoch in HORIZONS:
        checkpoint = training_root / "checkpoints" / f"epoch-{epoch}.pt"
        descriptor = horizon[str(epoch)]
        if (
            not checkpoint.is_file()
            or not isinstance(descriptor, dict)
            or descriptor.get("sha256") != sha256(checkpoint)
            or descriptor.get("epoch") != epoch
        ):
            raise AggregationError(f"child e{epoch} checkpoint differs for {seed}/{arm}")
        checkpoints[epoch] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}
    state_manifest = _read_json(training_root / "online-state-manifest.json")
    state = state_manifest.get("state")
    threshold_binding = state_manifest.get("thresholds")
    if (
        state_manifest.get("contract") != THRESHOLD_CONTRACT
        or state_manifest.get("arm") != arm
        or not isinstance(state, dict)
        or not isinstance(threshold_binding, dict)
        or threshold_binding.get("sha256") != prefix["threshold_sha256"]
        or threshold_binding.get("student_global_logit_q10") != prefix["thresholds"]["student_global_logit_q10"]
        or threshold_binding.get("teacher_global_logit_q10") != prefix["thresholds"]["teacher_global_logit_q10"]
    ):
        raise AggregationError(f"child online-state lineage/threshold binding differs for {seed}/{arm}")
    epoch_states = state.get("epochs")
    if not isinstance(epoch_states, dict) or set(epoch_states) != {str(epoch) for epoch in range(101, 115)}:
        raise AggregationError(f"online state artifact horizon differs for {seed}/{arm}")
    online_rows: dict[int, dict[int, dict[str, Any]]] = {}
    for epoch in range(101, 115):
        descriptor = epoch_states[str(epoch)]
        expected_sha = descriptor.get("sha256") if isinstance(descriptor, dict) else None
        if not isinstance(expected_sha, str):
            raise AggregationError("online state descriptor lacks SHA-256")
        local = training_root / "online-state" / f"epoch-{epoch}.parquet"
        path = _local_or_declared(
            declared=descriptor.get("path"), local=local, expected_sha256=expected_sha, kind="online state rows"
        )
        rows = _read_rows(path, expected_count=45_000)
        required_state_columns = {
            "s2_t1_state_active",
            "s2_t1_state_entry",
            "s2_t1_state_exit",
            "s2_t1_state_reentry",
            "action_active",
            "action_entry",
            "action_exit",
            "action_reentry",
        }
        if any(not required_state_columns <= set(row) for row in rows.values()):
            raise AggregationError(f"online state rows lack separate state/action persistence fields for {seed}/{arm}")
        if any(bool(row["action_active"]) and str(row["branch"]) != "S2_T1" for row in rows.values()):
            raise AggregationError(f"online action escaped S2×T1 for {seed}/{arm}, e{epoch}")
        if arm == "control" and any(bool(row["action_active"]) for row in rows.values()):
            raise AggregationError(f"control has an active online action for {seed}, e{epoch}")
        online_rows[epoch] = rows
    metrics_path = training_root / "epoch-metrics.jsonl"
    if not metrics_path.is_file():
        raise AggregationError(f"child runtime metrics are missing: {metrics_path}")
    metrics: dict[int, dict[str, Any]] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("epoch"), int) and 101 <= row["epoch"] <= 114:
            metrics[int(row["epoch"])] = row
    if set(metrics) != set(range(101, 115)):
        raise AggregationError(f"online child metrics do not cover e101–e114 for {seed}/{arm}")
    return {
        "arm_root": str(arm_root.resolve()),
        "training_root": str(training_root.resolve()),
        "summary": summary,
        "checkpoints": checkpoints,
        "online_state_manifest": state_manifest,
        "online_rows": online_rows,
        "metrics": metrics,
    }


def _canonical_payload(campaign: Path, *, seed: str, arm: str, training: Mapping[str, Any]) -> dict[str, Any]:
    root = campaign / "canonical" / seed / arm
    metadata = _read_json(root / "state-replay.json")
    expected_checkpoint = training["checkpoints"][114]
    if (
        metadata.get("contract") != CANONICAL_CONTRACT
        or metadata.get("checkpoint_sha256") != expected_checkpoint["sha256"]
        or metadata.get("checkpoint_epoch") != 114
        or metadata.get("teacher_checkpoint_sha256") != TEACHER_SHA256
        or metadata.get("observation", {}).get("attack_identity_sha256") != ENDPOINT_ATTACK_SHA256
        or metadata.get("row_count") != 45_000
        or not isinstance(metadata.get("rows_sha256"), str)
    ):
        raise AggregationError(f"canonical e114 replay contract differs for {seed}/{arm}")
    rows_path = _local_or_declared(
        declared=metadata.get("rows_path"),
        local=root / "state-rows.parquet",
        expected_sha256=str(metadata["rows_sha256"]),
        kind="canonical raw-train rows",
    )
    rows = _read_rows(rows_path, expected_count=45_000)
    return {"metadata": metadata, "rows": rows, "rows_path": str(rows_path)}


def _route_mechanics(
    online_rows: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    arm: str,
    metrics: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    per_epoch: dict[str, dict[str, Any]] = {}
    all_rows: list[Mapping[str, Any]] = []
    for epoch in sorted(online_rows):
        rows = online_rows[epoch]
        all_rows.extend(rows.values())
        branches = Counter(str(row["branch"]) for row in rows.values())
        transitions = Counter(str(row["transition"]) for row in rows.values() if isinstance(row.get("transition"), str))
        active = sum(bool(row["action_active"]) for row in rows.values())
        state_active = sum(bool(row["s2_t1_state_active"]) for row in rows.values())
        per_epoch[str(epoch)] = {
            "branch_counts": dict(sorted(branches.items())),
            "eligible_s2_t1_count": sum(bool(row["eligible_s2_t1"]) for row in rows.values()),
            "s2_t1_state_active_count": state_active,
            "s2_t1_state_fraction": state_active / len(rows),
            "s2_t1_state_entries": sum(bool(row["s2_t1_state_entry"]) for row in rows.values()),
            "s2_t1_state_exits": sum(bool(row["s2_t1_state_exit"]) for row in rows.values()),
            "s2_t1_state_reentries": sum(bool(row["s2_t1_state_reentry"]) for row in rows.values()),
            "action_count": active,
            "action_fraction": active / len(rows),
            "entries": sum(bool(row["action_entry"]) for row in rows.values()),
            "exits": sum(bool(row["action_exit"]) for row in rows.values()),
            "reentries": sum(bool(row["action_reentry"]) for row in rows.values()),
            "state_switches": sum(
                row.get("previous_branch") is not None and row.get("previous_branch") != row["branch"]
                for row in rows.values()
            ),
            "action_switches": sum(bool(row["action_active"]) != bool(row["previous_action"]) for row in rows.values()),
            "transitions": dict(sorted(transitions.items())),
            "student_current_q10_minus_frozen": metrics[epoch].get("online_state_student_current_q10_minus_frozen"),
            "teacher_current_q10_minus_frozen": metrics[epoch].get("online_state_teacher_current_q10_minus_frozen"),
            "pair_gated_boundary_count": metrics[epoch].get("boundary_active_count", 0.0),
            "boundary_input_gradient_calls": metrics[epoch].get("boundary_input_gradient_calls", 0.0),
        }
    transition_totals = Counter(str(row["transition"]) for row in all_rows if isinstance(row.get("transition"), str))
    action_active = sum(bool(row["action_active"]) for row in all_rows)
    state_active = sum(bool(row["s2_t1_state_active"]) for row in all_rows)
    boundary_keys = (
        "boundary_loss_count",
        "boundary_unscaled_loss_sum",
        "boundary_unscaled_loss_max",
        "boundary_weighted_loss_sum",
        "boundary_weighted_loss_max",
        "boundary_hinge_sum",
        "boundary_hinge_max",
        "boundary_student_distance_sum",
        "boundary_student_distance_max",
        "boundary_teacher_distance_sum",
        "boundary_teacher_distance_max",
        "boundary_student_input_grad_l1_sum",
        "boundary_student_input_grad_l1_max",
        "boundary_teacher_input_grad_l1_sum",
        "boundary_teacher_input_grad_l1_max",
    )
    boundary_diagnostics: dict[str, float] = {}
    for key in boundary_keys:
        values = [float(metrics[epoch].get(key, 0.0)) for epoch in online_rows]
        boundary_diagnostics[key] = max(values) if key.endswith("_max") else sum(values)
    summary = {
        "per_epoch": per_epoch,
        "mean_active_fraction": action_active / len(all_rows),
        "total_action_exposure": action_active,
        "total_s2_t1_state_active_epochs": state_active,
        "mean_s2_t1_state_fraction": state_active / len(all_rows),
        "total_s2_t1_state_entries": sum(bool(row["s2_t1_state_entry"]) for row in all_rows),
        "total_s2_t1_state_exits": sum(bool(row["s2_t1_state_exit"]) for row in all_rows),
        "total_s2_t1_state_reentries": sum(bool(row["s2_t1_state_reentry"]) for row in all_rows),
        "total_entries": sum(bool(row["action_entry"]) for row in all_rows),
        "total_exits": sum(bool(row["action_exit"]) for row in all_rows),
        "total_reentries": sum(bool(row["action_reentry"]) for row in all_rows),
        "total_action_switches": sum(bool(row["action_active"]) != bool(row["previous_action"]) for row in all_rows),
        "total_pair_gated_boundary_count": sum(
            float(per_epoch[str(epoch)]["pair_gated_boundary_count"]) for epoch in online_rows
        ),
        "total_boundary_input_gradient_calls": sum(
            float(per_epoch[str(epoch)]["boundary_input_gradient_calls"]) for epoch in online_rows
        ),
        "transition_totals": dict(sorted(transition_totals.items())),
        "last_current_q10_minus_frozen": {
            "student": per_epoch["114"]["student_current_q10_minus_frozen"],
            "teacher": per_epoch["114"]["teacher_current_q10_minus_frozen"],
        },
        "boundary_diagnostics": boundary_diagnostics,
    }
    if arm == "control" and summary["total_action_exposure"] != 0:
        raise AggregationError("control routing mechanics have nonzero action exposure")
    return summary


def _online_canonical_agreement(
    online: Mapping[int, Mapping[str, Any]], canonical_rows: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(online) != set(canonical_rows):
        raise AggregationError("online/canonical e114 audit stable-ID coverage differs")
    canonical = canonical_action_states(canonical_rows.values())["state_by_id"]
    online_positive = {sample_id for sample_id, row in online.items() if bool(row["eligible_s2_t1"])}
    canonical_positive = {sample_id for sample_id, state in canonical.items() if str(state["joint"]) == "S2xT1"}
    tp = len(online_positive & canonical_positive)
    fp = len(online_positive - canonical_positive)
    fn = len(canonical_positive - online_positive)
    tn = len(online) - tp - fp - fn
    return {
        "n": len(online),
        "online_s2_t1_count": len(online_positive),
        "canonical_s2_t1_count": len(canonical_positive),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "jaccard": None if tp + fp + fn == 0 else tp / (tp + fp + fn),
        "interpretation": "diagnostic only: online augmented KL10 pre-update state versus canonical raw CE20 state",
    }


def _historical_e99_s2_t1(seed: str) -> dict[str, Any]:
    """Load the hash-bound e99 canonical reference for the e114 audit.

    This is deliberately an audit-only historical input.  It supplies the
    fixed e99 canonical S2×T1 IDs needed to describe their e114 canonical
    destinations; it does not enter the online selector or any treatment
    comparison.
    """

    descriptor = HISTORICAL_E99_ROWS[seed]
    path = Path(descriptor["path"])
    if not path.is_file() or sha256(path) != descriptor["sha256"]:
        raise AggregationError(f"historical canonical e99 rows are unavailable or changed for {seed}")
    rows = _read_rows(path, expected_count=45_000)
    states = canonical_action_states(rows.values())["state_by_id"]
    selected = {sample_id for sample_id, state in states.items() if str(state["joint"]) == "S2xT1"}
    if not selected:
        raise AggregationError(f"historical canonical e99 S2×T1 reference is empty for {seed}")
    return {
        "path": str(path.resolve()),
        "sha256": descriptor["sha256"],
        "n": len(selected),
        "ids": selected,
        "state_semantics": "historical canonical CE20 q10 state; audit-only fixed e99 reference",
    }


def _canonical_e99_outcomes(e99: Mapping[str, Any], canonical_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    current = canonical_action_states(canonical_rows.values())["state_by_id"]
    ids = set(e99["ids"])
    if not ids <= set(current):
        raise AggregationError("historical e99 canonical S2×T1 IDs do not join e114 canonical rows")
    outcomes = Counter(str(current[sample_id]["branch"]) for sample_id in ids)
    joints = Counter(str(current[sample_id]["joint"]) for sample_id in ids)
    return {
        "reference_n": int(e99["n"]),
        "e114_branch_counts": dict(sorted(outcomes.items())),
        "e114_joint_counts": dict(sorted(joints.items())),
        "e114_branch_rates": {key: value / int(e99["n"]) for key, value in sorted(outcomes.items())},
        "e114_joint_rates": {key: value / int(e99["n"]) for key, value in sorted(joints.items())},
        "state_semantics": (
            "e99 historical canonical reference -> e114 arm-specific canonical CE20 occupancy; descriptive only"
        ),
    }


def _runtime(metrics: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    seconds = [float(row["train_seconds"]) for row in metrics.values()]
    throughput = [float(row["train_images_per_second"]) for row in metrics.values()]
    if any(value <= 0 for value in seconds) or any(value <= 0 for value in throughput):
        raise AggregationError("online child reports nonpositive runtime/throughput")
    return {
        "epochs": len(seconds),
        "total_train_seconds": sum(seconds),
        "mean_train_seconds": _mean(seconds),
        "mean_train_images_per_second": _mean(throughput),
    }


def _decision(deltas: Mapping[str, float]) -> str:
    values = [float(deltas[seed]) for seed in SEEDS]
    if all(value > 0.0 for value in values):
        return "SUPPORTED"
    if all(value <= 0.0 for value in values):
        return "NOT_SUPPORTED"
    return "MIXED"


def _policy_decisions(*, pmp: str, dbdp: str, dbdp_specific: bool) -> dict[str, str]:
    """Return the preregistered, machine-readable policy decision strings."""

    if pmp == "SUPPORTED":
        pmp_label = "OS_PMP_SUPPORTED_FOR_NEXT_STAGE"
    elif pmp == "NOT_SUPPORTED":
        pmp_label = "OS_PMP_NOT_SUPPORTED"
    else:
        pmp_label = "OS_PMP_MIXED"
    if dbdp == "SUPPORTED":
        dbdp_label = "OS_DBDP_SUPPORTED_FOR_NEXT_STAGE"
    elif dbdp == "NOT_SUPPORTED":
        dbdp_label = "OS_DBDP_NOT_SUPPORTED"
    else:
        dbdp_label = "OS_DBDP_MIXED"
    overall = (
        "ONLINE_S2_NOT_SUPPORTED"
        if pmp == "NOT_SUPPORTED" and dbdp == "NOT_SUPPORTED"
        else "ONLINE_S2_MIXED"
        if "MIXED" in {pmp, dbdp}
        else "ONLINE_S2_SUPPORTED_FOR_NEXT_STAGE"
    )
    return {
        "pmp": pmp_label,
        "dbdp": dbdp_label,
        "overall": overall,
        "dbdp_specific": (
            "DBDP_SPECIFIC_SUPERIORITY_SUPPORTED" if dbdp_specific else "DBDP_SPECIFIC_SUPERIORITY_NOT_SUPPORTED"
        ),
    }


def _markdown(result: Mapping[str, Any]) -> str:
    endpoint_rows: list[str] = []
    for seed in SEEDS:
        for epoch in HORIZONS:
            for arm in ARMS:
                record = result["seeds"][seed]["arms"][arm]["endpoints"][str(epoch)]
                endpoint_rows.append(
                    f"| {seed} | {epoch} | {arm.upper()} | {_fmt_pct(record['clean_accuracy'])} | "
                    f"{_fmt_pct(record['robust_accuracy'])} | {_fmt_pp(record['clean_delta_vs_control'])} | "
                    f"{_fmt_pp(record['robust_delta_vs_control'])} |"
                )
    comparison_rows: list[str] = []
    for seed in SEEDS:
        for epoch in HORIZONS:
            comparisons = result["seeds"][seed]["comparisons"][str(epoch)]
            for name in ("pmp_vs_control", "dbdp_vs_control", "dbdp_vs_pmp"):
                effect = comparisons[name]
                comparison_rows.append(
                    f"| {seed} | {epoch} | {name} | {_fmt_pp(effect['clean']['accuracy_delta'])} | "
                    f"{_fmt_pp(effect['robust']['accuracy_delta'])} | {effect['robust']['rescue_count']} | "
                    f"{effect['robust']['harm_count']} |"
                )
    mechanics_state_rows: list[str] = []
    mechanics_action_rows: list[str] = []
    state_persistence_rows: list[str] = []
    transition_rows: list[str] = []
    threshold_drift_rows: list[str] = []
    canonical_occupancy_rows: list[str] = []
    canonical_fixed_outcome_rows: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            mechanics = result["seeds"][seed]["arms"][arm]["route_mechanics"]
            agreement = result["seeds"][seed]["arms"][arm]["online_vs_canonical_e114"]
            mechanics_state_rows.append(
                f"| {seed} | {arm.upper()} | {mechanics['total_s2_t1_state_active_epochs']} | "
                f"{mechanics['mean_s2_t1_state_fraction'] * 100:.3f}% | "
                f"{mechanics['total_action_exposure']} | {mechanics['total_pair_gated_boundary_count']:.0f} |"
            )
            mechanics_action_rows.append(
                f"| {seed} | {arm.upper()} | "
                f"{mechanics['mean_active_fraction'] * 100:.3f}% | {mechanics['total_entries']} | "
                f"{mechanics['total_reentries']} | {mechanics['total_action_switches']} | "
                f"{_fmt_pct(agreement['precision'])} | {_fmt_pct(agreement['recall'])} | "
                f"{_fmt_pct(agreement['jaccard'])} |"
            )
            state_persistence_rows.append(
                f"| {seed} | {arm.upper()} | {mechanics['total_s2_t1_state_entries']} | "
                f"{mechanics['total_s2_t1_state_exits']} | {mechanics['total_s2_t1_state_reentries']} | "
                f"{mechanics['total_action_exposure']} | {mechanics['total_entries']} | "
                f"{mechanics['total_reentries']} |"
            )
            transitions = mechanics["transition_totals"]
            transition_rows.append(
                f"| {seed} | {arm.upper()} | {transitions.get('S2_T1->S1', 0)} | "
                f"{transitions.get('S2_T1->S3', 0)} | {transitions.get('S2_T1->CW', 0)} | "
                f"{transitions.get('S1->S2_T1', 0)} | {transitions.get('S3->S2_T1', 0)} | "
                f"{transitions.get('CW->S2_T1', 0)} | "
                f"{transitions.get('S2_T2->S2_T1', 0) + transitions.get('S2_T3->S2_T1', 0)} |"
            )
            drift = mechanics["last_current_q10_minus_frozen"]
            threshold_drift_rows.append(
                f"| {seed} | {arm.upper()} | {_fmt_scalar(drift['student'])} | {_fmt_scalar(drift['teacher'])} |"
            )
            canonical_occupancy_rows.append(
                f"| {seed} | {arm.upper()} | {agreement['online_s2_t1_count']} | "
                f"{agreement['canonical_s2_t1_count']} | {agreement['true_positive']} | "
                f"{agreement['false_positive']} | {agreement['false_negative']} | {agreement['true_negative']} |"
            )
            outcomes = result["seeds"][seed]["arms"][arm]["canonical_e99_s2_t1_e114_outcomes"]
            counts = outcomes["e114_branch_counts"]
            canonical_fixed_outcome_rows.append(
                f"| {seed} | {arm.upper()} | {outcomes['reference_n']} | {counts.get('S1', 0)} | "
                f"{counts.get('S3-non-CW', 0)} | {counts.get('Clean-Wrong', 0)} | {counts.get('S2', 0)} |"
            )
    runtime_rows: list[str] = []
    for seed in SEEDS:
        control = result["seeds"][seed]["arms"]["control"]["runtime"]
        for arm in ARMS:
            runtime = result["seeds"][seed]["arms"][arm]["runtime"]
            relative = runtime["mean_train_seconds"] / control["mean_train_seconds"] - 1.0
            runtime_rows.append(
                f"| {seed} | {arm.upper()} | {runtime['mean_train_seconds']:.1f} | "
                f"{runtime['mean_train_images_per_second']:.1f} | {_fmt_pct(relative)} |"
            )
    primary = result["decision"]
    frozen = result["frozen_contract"]
    lineage_rows = "\n".join(
        "| {seed} | {parent} | {student:.6f} | {teacher:.6f} | {threshold} |".format(
            seed=seed,
            parent=frozen[seed]["parent_checkpoint_sha256"],
            student=frozen[seed]["thresholds"]["student_global_logit_q10"],
            teacher=frozen[seed]["thresholds"]["teacher_global_logit_q10"],
            threshold=frozen[seed]["threshold_sha256"],
        )
        for seed in SEEDS
    )
    return f"""# I100 Online-State S2×T1 Preservation Screen

## Decision

The registered primary endpoint is e114 held-out CE-PGD20 robust accuracy.
PMP versus Control is **{primary["pmp_vs_control"]}**; D-BDP versus Control is
**{primary["dbdp_vs_control"]}**; and the D-BDP-specific comparison is
**{primary["dbdp_vs_pmp"]}**.  These are two development-seed directional
classifications, not population-level significance claims.

Machine-readable decision: `{primary["machine_readable"]["overall"]}`;
`{primary["machine_readable"]["pmp"]}`;
`{primary["machine_readable"]["dbdp"]}`; and
`{primary["machine_readable"]["dbdp_specific"]}`.

The threshold was frozen once from each seed's shared e100 no-action prefix.
It was not recomputed for any child or later epoch.  The online action was
strictly limited to current `S2_T1`; `CW`, `S3`, `S2_T2`, and `S2_T3` remained
baseline.  PMP/D-BDP use the registered pre-update pair margin only after
this detached state decision, whereas the router itself uses global logit
margin.

## Frozen lineage

| seed | e99 parent SHA-256 | e100 Student q10 | e100 Teacher q10 | threshold SHA-256 |
| --- | --- | ---: | ---: | --- |
{lineage_rows}

The common Teacher SHA-256 is `{TEACHER_SHA256}`.  The exact calibration
artifact SHA-256 is `{CALIBRATION_SHA256}`. Training used sample-keyed
KL-PGD10 `{TRAIN_ATTACK_SHA256}` and endpoint evaluation used CE-PGD20
`{ENDPOINT_ATTACK_SHA256}`.

## Held-out endpoints

| seed | epoch | arm | held-out clean | held-out robust | clean Δ vs Control | robust Δ vs Control |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{chr(10).join(endpoint_rows)}

## Paired comparisons

| seed | epoch | comparison | clean Δ | robust Δ | robust rescue | robust harm |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

For every paired result, `accuracy_delta = rescue_rate - harm_rate` was
asserted.  e114 raw-train CE-PGD20 state replays are recorded separately as a
canonical audit; they are not relabelled as a fixed causal cohort after the
online policy has changed the trajectory.

## Online-state mechanics and canonical diagnostic

| seed | arm | S2×T1 state epochs | state fraction | action exposure | pair-gated loss |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(mechanics_state_rows)}

| seed | arm | action fraction | action entries | action re-entry | action switches | precision | recall | Jaccard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(mechanics_action_rows)}

Precision/recall/Jaccard compare the e114 online augmented KL10 pre-update
state with the arm's raw CE20 canonical state.  This is a proxy-alignment
diagnostic, not a causal subgroup definition.  State persistence and action
exposure are separately preserved in the machine artifact for every e101–e114
epoch.  `router exposure` is the
global-margin Online-S2×T1 action decision.  `pair-gated loss` is the subset
whose reused Student-rival Teacher pair also had positive pair margin inside
the already-frozen PMP/DBDP formula; it is not a second router state.

| seed | arm | state entries | state exits | state re-entry | action exposure | action entries | action re-entry |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(state_persistence_rows)}

The first three columns are state persistence, reported for Control, PMP, and
D-BDP alike. The final three columns are action events and are necessarily
zero for Control.

| seed | arm | S2×T1→S1 | →S3 | →CW | S1→S2×T1 | S3→S2×T1 | CW→S2×T1 | S2×T2/T3→S2×T1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(transition_rows)}

| seed | arm | e114 Student current q10 − frozen | e114 Teacher current q10 − frozen |
| --- | --- | ---: | ---: |
{chr(10).join(threshold_drift_rows)}

The q10 values in this table are diagnostics only.  They were not used to
update the frozen e100 thresholds.

| seed | arm | online S2×T1 | canonical S2×T1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(canonical_occupancy_rows)}

The e114 canonical audit measures current occupancy and online-proxy
agreement only.  Because this campaign does not replay an earlier canonical
state from the new trajectory, the following table instead uses the already
hash-bound historical e99 canonical S2×T1 IDs as a descriptive fixed reference.
It is not a post-treatment causal subgroup comparison.

| seed | arm | historical e99 S2×T1 | e114 S1 | e114 S3-non-CW | e114 Clean-Wrong | e114 S2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(canonical_fixed_outcome_rows)}

## Runtime

| seed | arm | mean train seconds / epoch | mean images/s | mean epoch-time Δ vs Control |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(runtime_rows)}

Runtime is descriptive because host/GPU placement is operational rather than
scientific identity.  W&B remained metrics-only; checkpoints and 45k-ID state
tables stayed local and hash-bound.

## Interpretation and stop rule

- `D-BDP > PMP` is considered D-BDP-specific only if it is strictly positive
  in both development seeds; the recorded result is **{primary["dbdp_vs_pmp"]}**.
- No coefficient, q10 threshold, arm, seed, e199 extension, official test, or
  AutoAttack was added from these results.
- A later Stable Indirect BDD design, if considered, requires a separate
  scientific contract and calibration.  This screen stops here.
"""


def aggregate(*, campaign: Path, expected_source_sha: str) -> dict[str, Any]:
    _require_clean_source(expected_source_sha)
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise AggregationError(f"campaign root is missing: {campaign}")
    frozen = {seed: _prefix_and_threshold(campaign, seed=seed, source_sha=expected_source_sha) for seed in SEEDS}
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_online_state_s2_preservation_v1",
        "source_git_sha": expected_source_sha,
        "campaign_root": str(campaign),
        "calibration_artifact_sha256": CALIBRATION_SHA256,
        "frozen_coefficients": {
            "pmp": 0.05380932585058825,
            "dbdp": 31.649566509850324,
            "boundary_epsilon": 1e-12,
        },
        "frozen_contract": frozen,
        "seeds": {},
        "decision": {},
    }
    pmp_deltas: dict[str, float] = {}
    dbdp_deltas: dict[str, float] = {}
    dbdp_pmp_deltas: dict[str, float] = {}
    for seed in SEEDS:
        e99_reference = _historical_e99_s2_t1(seed)
        seed_data: dict[str, Any] = {
            "historical_e99_canonical_s2_t1_reference": {
                key: value for key, value in e99_reference.items() if key != "ids"
            },
            "arms": {},
            "comparisons": {},
        }
        endpoint_rows: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
        canonical_rows: dict[str, dict[int, dict[str, Any]]] = {}
        for arm in ARMS:
            training = _training_payload(
                campaign, seed=seed, arm=arm, source_sha=expected_source_sha, prefix=frozen[seed]
            )
            endpoints = _endpoint_payload(campaign / "endpoints" / seed / arm, seed=seed, arm=arm)
            canonical = _canonical_payload(campaign, seed=seed, arm=arm, training=training)
            endpoint_rows[arm] = {epoch: value["rows"] for epoch, value in endpoints.items()}
            canonical_rows[arm] = canonical["rows"]
            endpoint_summary: dict[str, Any] = {}
            for epoch, descriptor in endpoints.items():
                rows = descriptor["rows"]
                endpoint_summary[str(epoch)] = {
                    "checkpoint_sha256": descriptor["checkpoint_sha256"],
                    "rows_sha256": descriptor["rows_sha256"],
                    "clean_accuracy": _mean(float(bool(row["clean_correct"])) for row in rows.values()),
                    "robust_accuracy": _mean(float(bool(row["robust_correct"])) for row in rows.values()),
                }
            seed_data["arms"][arm] = {
                "training": {
                    "checkpoints": training["checkpoints"],
                    "online_state_manifest": training["online_state_manifest"],
                },
                "endpoints": endpoint_summary,
                "canonical_e114": {
                    "metadata": canonical["metadata"],
                    "rows_path": canonical["rows_path"],
                },
                "route_mechanics": _route_mechanics(training["online_rows"], arm=arm, metrics=training["metrics"]),
                "online_vs_canonical_e114": _online_canonical_agreement(
                    training["online_rows"][114], canonical["rows"]
                ),
                "canonical_e99_s2_t1_e114_outcomes": _canonical_e99_outcomes(e99_reference, canonical["rows"]),
                "runtime": _runtime(training["metrics"]),
            }
        for epoch in HORIZONS:
            control = endpoint_rows["control"][epoch]
            comparisons = {
                "pmp_vs_control": _effect(
                    control,
                    endpoint_rows["pmp"][epoch],
                    clean_key="clean_correct",
                    robust_key="robust_correct",
                    clean_margin_key="clean_probability_margin",
                    robust_margin_key="adversarial_probability_margin",
                ),
                "dbdp_vs_control": _effect(
                    control,
                    endpoint_rows["dbdp"][epoch],
                    clean_key="clean_correct",
                    robust_key="robust_correct",
                    clean_margin_key="clean_probability_margin",
                    robust_margin_key="adversarial_probability_margin",
                ),
                "dbdp_vs_pmp": _effect(
                    endpoint_rows["pmp"][epoch],
                    endpoint_rows["dbdp"][epoch],
                    clean_key="clean_correct",
                    robust_key="robust_correct",
                    clean_margin_key="clean_probability_margin",
                    robust_margin_key="adversarial_probability_margin",
                ),
            }
            for arm in ARMS:
                if arm == "control":
                    seed_data["arms"][arm]["endpoints"][str(epoch)].update(
                        {"clean_delta_vs_control": 0.0, "robust_delta_vs_control": 0.0}
                    )
                else:
                    comparison = comparisons[f"{arm}_vs_control"]
                    seed_data["arms"][arm]["endpoints"][str(epoch)].update(
                        {
                            "clean_delta_vs_control": comparison["clean"]["accuracy_delta"],
                            "robust_delta_vs_control": comparison["robust"]["accuracy_delta"],
                        }
                    )
            seed_data["comparisons"][str(epoch)] = comparisons
        # Canonical e114 effects are audit-only, but preserve the specified
        # clean/robust rescue-harm accounting on exact 45k raw rows.
        seed_data["canonical_e114_audit"] = {
            "pmp_vs_control": _effect(
                canonical_rows["control"],
                canonical_rows["pmp"],
                clean_key="student_clean_correct",
                robust_key="student_ce20_adv_correct",
                clean_margin_key="student_clean_margin",
                robust_margin_key="student_ce20_adv_margin",
            ),
            "dbdp_vs_control": _effect(
                canonical_rows["control"],
                canonical_rows["dbdp"],
                clean_key="student_clean_correct",
                robust_key="student_ce20_adv_correct",
                clean_margin_key="student_clean_margin",
                robust_margin_key="student_ce20_adv_margin",
            ),
            "dbdp_vs_pmp": _effect(
                canonical_rows["pmp"],
                canonical_rows["dbdp"],
                clean_key="student_clean_correct",
                robust_key="student_ce20_adv_correct",
                clean_margin_key="student_clean_margin",
                robust_margin_key="student_ce20_adv_margin",
            ),
        }
        e114 = seed_data["comparisons"]["114"]
        pmp_deltas[seed] = float(e114["pmp_vs_control"]["robust"]["accuracy_delta"])
        dbdp_deltas[seed] = float(e114["dbdp_vs_control"]["robust"]["accuracy_delta"])
        dbdp_pmp_deltas[seed] = float(e114["dbdp_vs_pmp"]["robust"]["accuracy_delta"])
        result["seeds"][seed] = seed_data
    pmp_decision = _decision(pmp_deltas)
    dbdp_decision = _decision(dbdp_deltas)
    dbdp_pmp_decision = _decision(dbdp_pmp_deltas)
    dbdp_specific = all(value > 0.0 for value in dbdp_pmp_deltas.values())
    result["decision"] = {
        "primary_endpoint": "epoch114 held-out CE-PGD20 robust accuracy",
        "pmp_vs_control_per_seed": pmp_deltas,
        "dbdp_vs_control_per_seed": dbdp_deltas,
        "dbdp_vs_pmp_per_seed": dbdp_pmp_deltas,
        "pmp_vs_control": pmp_decision,
        "dbdp_vs_control": dbdp_decision,
        "dbdp_vs_pmp": dbdp_pmp_decision,
        "dbdp_specific_supported": dbdp_specific,
        "machine_readable": _policy_decisions(pmp=pmp_decision, dbdp=dbdp_decision, dbdp_specific=dbdp_specific),
        "stop": (
            "no automatic e199 extension, arm/threshold/coefficient change, fresh seed, official test, or AutoAttack"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="canonical local campaign collection root")
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    result_path = args.result.resolve()
    report_path = args.report.resolve()
    _non_overwriting(result_path)
    _non_overwriting(report_path)
    result = aggregate(campaign=args.campaign, expected_source_sha=args.expected_source_sha)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {"result": str(result_path), "report": str(report_path), "decision": result["decision"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
