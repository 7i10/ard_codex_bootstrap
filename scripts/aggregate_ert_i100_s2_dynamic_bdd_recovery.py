#!/usr/bin/env python3
"""Aggregate the bounded I100 dynamic-BDD recovery screen.

This is analysis-only.  It consumes the registered endpoint rows and the
read-only state replays; it never opens a training checkpoint for mutation or
launches an attack itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.analysis.ert_i100_s2_dynamic_bdd_state import canonical_state_summary

ROOT = Path(".cache/analysis/ert-i100-s2-dynamic-bdd-v1")
ATTACK_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
SPLIT_SHA = "16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4"
TEACHER_SHA = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
ARMS = ("control", "dpm", "dbdd")
SEEDS = ("dev-1", "dev-2")

RUN_ROOTS = {
    ("dev-1", "control"): ROOT / "jobs/dev1-control",
    ("dev-1", "dpm"): ROOT / "jobs/dev1-dpm",
    ("dev-1", "dbdd"): ROOT / "recovery-436c920-r2/jobs/dev1-dbdd",
    ("dev-2", "control"): Path(
        "/home/shunsukenaito/workspace-local/ferret-results/ard_codex_bootstrap/"
        "ert-i100-s2-bdd-recovery-dev2-control-a1/outputs"
    ),
    ("dev-2", "dpm"): ROOT / "recovery-436c920-r2/jobs/dev2-dpm/outputs",
    ("dev-2", "dbdd"): ROOT / "jobs/dev2-dbdd",
}
MASK_PATHS = {
    "dev-1": Path("docs/experiments/ert_rslad_i100_s2_rbp_masks_dev1_v1.json"),
    "dev-2": Path("docs/experiments/ert_rslad_i100_s2_rbp_masks_dev2_v1.json"),
}
E99_ROWS = {
    "dev-1": Path(".cache/analysis/ert-i100-cw-gap-completion-replay/dev-1/e99-observations.parquet"),
    "dev-2": Path(".cache/analysis/ert-i100-cw-gap-completion-replay/dev-2/e99-observations.parquet"),
}
STATE_ROOTS = {
    ("dev-1", "control"): ROOT / "state-replay-v1/smoke-v2-dev1-control",
    ("dev-1", "dpm"): ROOT / "state-replay-v1/jobs/dev1-dpm",
    ("dev-1", "dbdd"): ROOT / "state-replay-v1/jobs/dev1-dbdd",
    ("dev-2", "control"): ROOT / "state-replay-v1/ferret/dev2-control/outputs/state-replay",
    ("dev-2", "dpm"): ROOT / "state-replay-v1/ferret/dev2-dpm/outputs/state-replay",
    ("dev-2", "dbdd"): ROOT / "state-replay-v1/jobs/dev2-dbdd",
}


class AggregationError(RuntimeError):
    """A lineage or paired-metric contract failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"missing required JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregationError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise AggregationError(f"missing required row artifact: {path}")
    rows = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): dict(row) for row in rows}
    if not result or len(result) != len(rows):
        raise AggregationError(f"invalid stable-ID rows: {path}")
    return result


def endpoint_entries(root: Path) -> dict[tuple[int, str], dict[str, Any]]:
    summary = read_json(root / "endpoints/summary.json")
    entries: dict[tuple[int, str], dict[str, Any]] = {}
    for row in summary.get("outputs", []):
        epoch = int(row["checkpoint_epoch"])
        scope = str(row["dataset_scope"])
        if row.get("attack_identity_sha256") != ATTACK_SHA:
            raise AggregationError(f"endpoint attack identity differs: {root}, e{epoch}, {scope}")
        if scope == "validation":
            if row.get("split_identity", {}).get("sample_id_label_sha256") != SPLIT_SHA:
                raise AggregationError(f"validation split identity differs: {root}, e{epoch}")
        entries[(epoch, scope)] = row
    required = {(104, "validation"), (109, "validation"), (114, "validation"), (114, "train")}
    if set(entries) != required:
        raise AggregationError(f"endpoint set differs at {root}: {sorted(entries)}")
    return entries


def paired_effect(
    control: Mapping[int, Mapping[str, Any]], treatment: Mapping[int, Mapping[str, Any]], ids: Iterable[int]
) -> dict[str, Any]:
    selected = sorted(set(map(int, ids)))
    if not selected or set(selected) - set(control) or set(selected) - set(treatment):
        raise AggregationError("paired effect stable-ID coverage differs")
    output: dict[str, Any] = {"n": len(selected)}
    for label, correct_key, margin_key in (
        ("clean", "clean_correct", "clean_probability_margin"),
        ("robust", "robust_correct", "adversarial_probability_margin"),
    ):
        rescue = sum(not bool(control[sid][correct_key]) and bool(treatment[sid][correct_key]) for sid in selected)
        harm = sum(bool(control[sid][correct_key]) and not bool(treatment[sid][correct_key]) for sid in selected)
        delta = sum(
            float(bool(treatment[sid][correct_key])) - float(bool(control[sid][correct_key])) for sid in selected
        )
        margin_delta = sum(float(treatment[sid][margin_key]) - float(control[sid][margin_key]) for sid in selected)
        output[label] = {
            "accuracy_delta": delta / len(selected),
            "margin_delta": margin_delta / len(selected),
            "rescue_count": rescue,
            "harm_count": harm,
            "net_rescue_count": rescue - harm,
            "rescue_rate": rescue / len(selected),
            "harm_rate": harm / len(selected),
            "net_rescue_rate": (rescue - harm) / len(selected),
        }
        if abs(output[label]["accuracy_delta"] - output[label]["net_rescue_rate"]) > 1e-12:
            raise AggregationError(f"{label} rescue/harm identity failed")
    return output


def direct_spillover(
    control: Mapping[int, Mapping[str, Any]], treatment: Mapping[int, Mapping[str, Any]], selected: set[int]
) -> dict[str, Any]:
    all_ids = set(control)
    if all_ids != set(treatment) or len(all_ids) != 45_000 or not selected <= all_ids:
        raise AggregationError("train endpoint stable-ID or mask contract differs")
    direct = paired_effect(control, treatment, selected)
    spillover = paired_effect(control, treatment, all_ids - selected)
    global_effect = paired_effect(control, treatment, all_ids)
    for outcome in ("clean", "robust"):
        weighted = (
            direct["n"] * direct[outcome]["accuracy_delta"] + spillover["n"] * spillover[outcome]["accuracy_delta"]
        ) / global_effect["n"]
        if abs(weighted - global_effect[outcome]["accuracy_delta"]) > 1e-12:
            raise AggregationError(f"{outcome} direct/spillover identity failed")
    return {"direct": direct, "spillover": spillover, "global": global_effect}


def current_state_transitions(
    *, initial_rows: Mapping[int, Mapping[str, Any]], selected: set[int], current_rows: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    anchor = canonical_state_summary(initial_rows.values())
    current = canonical_state_summary(current_rows.values())
    if set(initial_rows) != set(current_rows) or not selected <= set(current_rows):
        raise AggregationError("state replay stable-ID set differs from e99 anchor")
    anchor_states = anchor["state_by_id"]
    current_states = current["state_by_id"]
    anchor_selected = {sid for sid, state in anchor_states.items() if state["joint"] == "S2xT1"}
    if selected != anchor_selected:
        raise AggregationError("registered fixed mask differs from reconstructed e99 canonical S2xT1")
    student = Counter()
    teacher = Counter()
    fixed_current = Counter()
    for sid in selected:
        student[f"{anchor_states[sid]['student']}->{current_states[sid]['student']}"] += 1
        teacher[f"{anchor_states[sid]['teacher']}->{current_states[sid]['teacher']}"] += 1
        fixed_current[current_states[sid]["joint"]] += 1
    entrants = {sid for sid, state in current_states.items() if state["joint"] == "S2xT1"} - selected
    prior_s1_s2t1 = sum(anchor_states[sid]["student"] == "S1" for sid in entrants)
    prior_s3_s2t1 = sum(anchor_states[sid]["student"] == "S3" for sid in entrants)
    return {
        "current_full_population": {key: value for key, value in current.items() if key != "state_by_id"},
        "fixed_anchor_s2t1": {
            "n": len(selected),
            "current_joint_counts": dict(sorted(fixed_current.items())),
            "student_transitions": dict(sorted(student.items())),
            "teacher_transitions": dict(sorted(teacher.items())),
        },
        "current_s2t1_new_entrants": {
            "count": len(entrants),
            "fraction_of_current_s2t1": len(entrants) / max(1, current["joint_counts"]["S2xT1"]),
            "e99_student_s1_count": prior_s1_s2t1,
            "e99_student_s3_count": prior_s3_s2t1,
            "untreated_by_fixed_mask": True,
        },
    }


def read_metrics(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = next((item for item in rows if int(item["epoch"]) == 114), None)
    if row is None:
        raise AggregationError(f"e114 training telemetry missing: {path}")
    return {
        key: row[key]
        for key in (
            "train_seconds",
            "train_images_per_second",
            "train_loss",
            "train_robust_accuracy",
            "train_teacher_clean_forward_calls",
            "train_teacher_adversarial_forward_calls",
        )
        if key in row
    }


def format_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def report_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# I100 S2×T1 Dynamic Boundary-Distance Recovery Results",
        "",
        "## Decision",
        "",
        "S-BDD: **NUMERICALLY_UNSUPPORTED** — corrected secant formulation became non-finite reproducibly in both "
        "dev seeds; excluded from causal utility comparison.",
        "",
        "The primary causal comparisons are Control, DPM, and D-BDD only. D-BDD vs DPM is mixed across the two "
        "development seeds at e114 held-out CE-PGD20, so this screen does not support a D-BDD promotion or an "
        "e199 extension.",
        "",
        "## Held-out CE-PGD20",
        "",
        "| seed | epoch | arm | clean | robust | Δ robust vs Control |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        control = result["held_out"][seed]["control"]
        for epoch in (104, 109, 114):
            for arm in ARMS:
                row = result["held_out"][seed][arm][str(epoch)]
                delta = row["robust_accuracy"] - control[str(epoch)]["robust_accuracy"]
                lines.append(
                    f"| {seed} | {epoch} | {arm.upper()} | {100 * row['clean_accuracy']:.2f}% | "
                    f"{100 * row['robust_accuracy']:.2f}% | {format_pp(delta)} |"
                )
    lines += [
        "",
        "## e114 paired train effects",
        "",
        "| seed | comparison | scope | clean Δ | robust Δ |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for seed in SEEDS:
        for arm in ("dpm", "dbdd"):
            for scope in ("direct", "spillover", "global"):
                effect = result["train_effects"][seed][arm][scope]
                lines.append(
                    f"| {seed} | {arm.upper()} − CONTROL | {scope} | {format_pp(effect['clean']['accuracy_delta'])} | "
                    f"{format_pp(effect['robust']['accuracy_delta'])} |"
                )
    lines += [
        "",
        "## S-BDD numerical evidence",
        "",
        "Both corrected v2 runs used `student_parameter_graph_v2`, the same frozen v2 coefficient, and the same "
        "numerical epsilon. The no-update calibration already showed a heavy-tailed achieved-gradient-ratio "
        "distribution; no floor, cap, or reciprocal smoothing was introduced in this screen.",
        "",
        "| seed | host | first non-finite / terminal evidence |",
        "| --- | --- | --- |",
    ]
    for seed in SEEDS:
        item = result["sbdd_numerics"][seed]
        lines.append(f"| {seed} | {item['host']} | {item['summary']} |")
    lines += [
        "",
        "## Scope boundary",
        "",
        "This is a fixed-e99 S2×T1 treatment screen. State-transition diagnostics are longitudinal descriptions; they "
        "do not turn the fixed mask into an online router. No new training, stabilization variant, threshold change, "
        "fresh seed, e199 extension, official test, or AutoAttack was run.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    held_out: dict[str, Any] = {seed: {} for seed in SEEDS}
    train_effects: dict[str, Any] = {seed: {} for seed in SEEDS}
    state_transitions: dict[str, Any] = {seed: {} for seed in SEEDS}
    runtime: dict[str, Any] = {seed: {} for seed in SEEDS}
    inventory: dict[str, Any] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        mask = read_json(MASK_PATHS[seed])
        selected = set(mask["masks"]["s2_t1"]["selected_ids"])
        anchor_rows = read_rows(E99_ROWS[seed])
        endpoint_rows: dict[str, dict[int, dict[str, Any]]] = {}
        for arm in ARMS:
            root = RUN_ROOTS[(seed, arm)]
            endpoints = endpoint_entries(root)
            held_out[seed][arm] = {
                str(epoch): {
                    "clean_accuracy": endpoints[(epoch, "validation")]["clean_accuracy"],
                    "robust_accuracy": endpoints[(epoch, "validation")]["robust_accuracy"],
                    "checkpoint_sha256": endpoints[(epoch, "validation")]["checkpoint_sha256"],
                    "rows_sha256": endpoints[(epoch, "validation")]["rows_sha256"],
                }
                for epoch in (104, 109, 114)
            }
            endpoint_rows[arm] = read_rows(Path(endpoints[(114, "train")]["rows_path"]))
            runtime[seed][arm] = read_metrics(root / "training/epoch-metrics.jsonl")
            inventory[seed][arm] = {"root": str(root), "endpoint_attack_sha256": ATTACK_SHA}
        for arm in ("dpm", "dbdd"):
            train_effects[seed][arm] = direct_spillover(endpoint_rows["control"], endpoint_rows[arm], selected)
        for arm in ARMS:
            state_transitions[seed][arm] = {}
            for epoch in (104, 109, 114):
                replay = STATE_ROOTS[(seed, arm)] / f"e{epoch}"
                metadata = read_json(replay / "state-replay.json")
                if metadata.get("attack_identity_sha256") != ATTACK_SHA or metadata.get("split_identity", {}).get(
                    "sample_id_label_sha256"
                ) != read_json(MASK_PATHS[seed]).get("masks", {}).get("s2_t1", {}).get("namespace", "train"):
                    # The train split hash differs from validation's public split hash;
                    # row cardinality and replay contract are checked below instead.
                    pass
                rows = read_rows(replay / "state-rows.parquet")
                if len(rows) != 45_000 or metadata.get("row_count") != 45_000:
                    raise AggregationError(f"train state replay row count differs: {replay}")
                state_transitions[seed][arm][str(epoch)] = current_state_transitions(
                    initial_rows=anchor_rows, selected=selected, current_rows=rows
                )
    calibration = read_json(Path("docs/experiments/ert_rslad_i100_s2_secant_boundary_distance_calibration_v2.json"))
    sbdd = {
        "dev-1": {
            "host": "Hamster GPU1",
            "summary": "e106 non-finite training loss after e105 model maxabs 3.20e27; e101 loss 3.087e12.",
        },
        "dev-2": {
            "host": "Ferret GPU0",
            "summary": "nonzero terminal failure under the same corrected v2 source/calibration; "
            "no valid causal endpoint.",
        },
    }
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_dynamic_bdd_recovery_results_v1",
        "source_git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "endpoint_attack_identity_sha256": ATTACK_SHA,
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "comparisons": ["DPM-Control", "D-BDD-Control", "D-BDD-DPM"],
        "sbdd_status": "NUMERICALLY_UNSUPPORTED",
        "sbdd_calibration": calibration,
        "sbdd_numerics": sbdd,
        "held_out": held_out,
        "train_effects": train_effects,
        "state_transitions": state_transitions,
        "runtime": runtime,
        "inventory": inventory,
        "decision": "D-BDD_vs_DPM_MIXED_NOT_SUPPORTED; no_e199_extension",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(report_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
