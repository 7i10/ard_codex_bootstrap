#!/usr/bin/env python3
"""Compute the Clean-Wrong Direct -> Spillover -> Held-out diagnostic."""

# Report table rows are intentionally long; they are emitted verbatim as
# Markdown and are not production-library code.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("L2", "L4")
BROAD_ARMS = ("C0", "C4", "C5", "C10", "C11", "C12", "C13")
GATED_ARMS = ("G0_BASE", "G1_CW_ALL_CE015", "G2_CW_R_CE20_CE015", "G3_CW_R_KL10_CE015")
PARENT_SHA = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}
MASK_SHA = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}
ENDPOINT_ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
TRAIN_SOURCE = "cbe03a7b3be0b11fa1555b573c6f453a3d10f27b"


class DiagnosticError(RuntimeError):
    """Raised when a generalization join is not scientifically safe."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): row for row in values}
    if len(result) != len(values):
        raise DiagnosticError(f"duplicate stable IDs: {path}")
    return result


def effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    if not ids:
        return {"n": 0, "clean": None, "robust": None}
    if set(ids) - set(base) or set(ids) - set(treatment):
        raise DiagnosticError("effect cohort is not contained in both endpoints")

    def one(correct: str, margin: str) -> dict[str, Any]:
        b = [bool(base[i][correct]) for i in ids]
        t = [bool(treatment[i][correct]) for i in ids]
        rescue = sum(not x and y for x, y in zip(b, t))
        harm = sum(x and not y for x, y in zip(b, t))
        n = len(ids)
        delta = (rescue - harm) / n
        if abs(delta - (rescue / n - harm / n)) > 1e-12:
            raise DiagnosticError("accuracy delta != rescue rate - harm rate")
        margin_delta = sum(float(treatment[i][margin]) - float(base[i][margin]) for i in ids) / n
        return {
            "accuracy_delta": delta,
            "rescue_rate": rescue / n,
            "harm_rate": harm / n,
            "net_rescue_rate": delta,
            "rescue_count": rescue,
            "harm_count": harm,
            "margin_delta": margin_delta,
        }

    return {
        "n": len(ids),
        "clean": one("clean_correct", "clean_probability_margin"),
        "robust": one("robust_correct", "adversarial_probability_margin"),
    }


def load_feature(
    meta_path: Path, *, expected_contract: str, run: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    meta = read_json(meta_path)
    if meta.get("contract") != expected_contract or meta.get("feature_epoch") != 79:
        raise DiagnosticError(f"{run}: feature contract/epoch mismatch: {meta_path}")
    if meta.get("checkpoint_sha256") != PARENT_SHA[run]:
        raise DiagnosticError(f"{run}: feature parent SHA mismatch")
    if meta.get("attack_identity_sha256") not in {
        "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2",
        "98194e2a6ee02add8c675b0df1146007f371ed1811ef34b9ef37d052997348bd",
    }:
        raise DiagnosticError(f"{run}: unexpected feature attack")
    path = Path(meta["rows_path"])
    if not path.is_file() or sha256(path) != meta.get("rows_sha256"):
        raise DiagnosticError(f"{run}: feature rows hash mismatch")
    return {
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": sha256(meta_path),
        "rows_path": str(path.resolve()),
        "rows_sha256": meta["rows_sha256"],
        "attack_identity_sha256": meta["attack_identity_sha256"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "mask_sha256": meta.get("mask_sha256"),
        "source_git_sha": meta.get("source_git_sha"),
    }, rows(path)


def quantile_edges(train_rows: dict[int, dict[str, Any]], ids: list[int], feature: str) -> dict[str, Any]:
    ordered = sorted(ids, key=lambda i: (float(train_rows[i][feature]), i))
    n = len(ordered)
    edges: dict[str, Any] = {}
    for q in range(5):
        end = (n * (q + 1)) // 5 - 1
        start = (n * q) // 5
        members = ordered[start : end + 1]
        edges[f"Q{q + 1}"] = {
            "count": len(members),
            "lower": float(train_rows[members[0]][feature]),
            "upper": float(train_rows[members[-1]][feature]),
            "ids_sha256": hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest(),
        }
    return edges


def quantile_groups(train_rows: dict[int, dict[str, Any]], ids: list[int], feature: str) -> dict[str, list[int]]:
    ordered = sorted(ids, key=lambda i: (float(train_rows[i][feature]), i))
    n = len(ordered)
    return {f"Q{q + 1}": ordered[(n * q) // 5 : (n * (q + 1)) // 5] for q in range(5)}


def assign_train_edges(
    validation_rows: dict[int, dict[str, Any]], ids: list[int], feature: str, edges: dict[str, Any]
) -> dict[str, list[int]]:
    result = {f"Q{i}": [] for i in range(1, 6)}
    for item in ids:
        value = float(validation_rows[item][feature])
        assigned = False
        for q in range(1, 6):
            if value <= float(edges[f"Q{q}"]["upper"]):
                result[f"Q{q}"].append(item)
                assigned = True
                break
        if not assigned:
            result["Q5"].append(item)
    return result


def endpoint(
    root: Path, run: str, arm: str, split: str, *, gated: bool = False, expected_source: str = TRAIN_SOURCE
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if gated:
        meta_path = root / run / arm / "epoch-94" / split / "endpoint.json"
    else:
        meta_path = root / run / arm / "endpoint" / split / "endpoint.json"
    rows_path = meta_path.with_name("endpoint-sample-stats.parquet")
    meta = read_json(meta_path)
    if meta.get("attack_identity_sha256") != ENDPOINT_ATTACK or meta.get("checkpoint_epoch") not in (
        {84} if not gated else {94}
    ):
        raise DiagnosticError(f"endpoint attack/epoch mismatch: {meta_path}")
    if meta.get("source_git_sha") != expected_source:
        raise DiagnosticError(f"endpoint source SHA mismatch: {meta_path}")
    if meta.get("rows_sha256") != sha256(rows_path):
        raise DiagnosticError(f"endpoint rows hash mismatch: {rows_path}")
    return {
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": sha256(meta_path),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": meta["rows_sha256"],
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "source_git_sha": meta.get("source_git_sha"),
    }, rows(rows_path)


def validate_endpoint_universe(
    reference: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]], *, label: str
) -> None:
    """Require stable IDs and class labels to match before paired effects."""
    if set(reference) != set(candidate):
        raise DiagnosticError(f"{label}: endpoint stable-ID universe mismatch")
    if any(int(reference[i]["true_label"]) != int(candidate[i]["true_label"]) for i in reference):
        raise DiagnosticError(f"{label}: endpoint class mapping mismatch")


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def pp(value: float | None) -> str:
    """Render a proportion as percentage points for the human report."""
    return "n/a" if value is None else f"{value * 100:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "docs/experiments/ert_cw_generalization_diagnostic_v1.json"
    )
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/ERT_CW_GENERALIZATION_DIAGNOSTIC.md")
    args = parser.parse_args()
    action_map = read_json(ROOT / "docs/experiments/ert_cw_margin_action_map_v1.json")
    if action_map.get("contract") != "ert_cw_margin_action_map_v1":
        raise DiagnosticError("action-map report contract mismatch")
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_generalization_diagnostic_v1",
        "analysis_kind": "direct_spillover_heldout_read_only",
        "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "no_training": True,
        "held_out_subtype_boundaries": "train-derived upper edges; validation outcome never defines bins",
        "seeds": {},
        "gpu_replay": {"jobs": 4, "scope": "L2/L4 x CE20/KL10 validation epoch-79", "peak_memory": "not instrumented"},
    }
    for run in RUNS:
        mask_path = ROOT / ".cache/analysis/ert-state-overlay-v1-review" / f"anchor79-fixed-masks-{run}.json"
        mask = read_json(mask_path)
        cw_ids = [int(i) for i in mask["masks"]["student_clean_wrong"]["selected_ids"]]
        if sha256(mask_path) != MASK_SHA[run]:
            raise DiagnosticError(f"{run}: mask SHA mismatch")
        ce_meta, ce_train = load_feature(
            ROOT / ".cache/analysis/ert-clean-wrong-subtypes-v4" / run / "clean-wrong-feature-replay.json",
            expected_contract="ert_clean_wrong_c0_ce_pgd20_features_v1",
            run=run,
        )
        kl_meta, kl_train = load_feature(
            ROOT
            / ".cache/analysis/ert-clean-wrong-reliability-proxy-v1"
            / run
            / "clean-wrong-kl10-feature-replay.json",
            expected_contract="ert_clean_wrong_c0_kl_pgd10_features_v1",
            run=run,
        )
        if ce_meta.get("mask_sha256") != MASK_SHA[run] or kl_meta.get("mask_sha256") != MASK_SHA[run]:
            raise DiagnosticError(f"{run}: train feature mask SHA mismatch")
        if set(cw_ids) != set(ce_train) or set(cw_ids) != set(kl_train):
            raise DiagnosticError(f"{run}: train feature/CW mask ID mismatch")
        ce_val_meta, ce_val = load_feature(
            ROOT / ".cache/analysis/ert-cw-generalization-v1" / run / "CE20" / "validation-feature-replay.json",
            expected_contract="ert_clean_wrong_validation_ce_pgd20_features_v1",
            run=run,
        )
        kl_val_meta, kl_val = load_feature(
            ROOT / ".cache/analysis/ert-cw-generalization-v1" / run / "KL10" / "validation-feature-replay.json",
            expected_contract="ert_clean_wrong_validation_kl_pgd10_features_v1",
            run=run,
        )
        if len(ce_val) != 5000 or set(ce_val) != set(kl_val):
            raise DiagnosticError(f"{run}: validation feature universe mismatch")
        if any(int(ce_val[i]["true_label"]) != int(kl_val[i]["true_label"]) for i in ce_val):
            raise DiagnosticError(f"{run}: validation CE/KL class mismatch")
        ce_edges = quantile_edges(ce_train, cw_ids, "teacher_adv_margin")
        kl_edges = quantile_edges(kl_train, cw_ids, "teacher_adv_margin")
        ce_train_q = quantile_groups(ce_train, cw_ids, "teacher_adv_margin")
        broad_meta: dict[str, Any] = {}
        broad_rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
        for arm in BROAD_ARMS:
            broad_rows[arm] = {}
            for split in ("train", "validation"):
                meta, data = endpoint(ROOT / ".cache/analysis/ert-clean-wrong-broad-v1", run, arm, split)
                broad_meta[f"{arm}/{split}"] = meta
                broad_rows[arm][split] = data
        base_train = broad_rows["C0"]["train"]
        base_val = broad_rows["C0"]["validation"]
        for arm in BROAD_ARMS:
            validate_endpoint_universe(base_train, broad_rows[arm]["train"], label=f"{run} broad {arm} train")
            validate_endpoint_universe(base_val, broad_rows[arm]["validation"], label=f"{run} broad {arm} validation")
        validate_endpoint_universe(base_val, ce_val, label=f"{run} validation features")
        spill_ids = sorted(set(base_train) - set(cw_ids))
        val_cw_ids = sorted(i for i, row in ce_val.items() if not bool(row["student_clean_correct"]))
        val_q_ce = assign_train_edges(ce_val, val_cw_ids, "teacher_adv_margin", ce_edges)
        val_q_kl = assign_train_edges(kl_val, val_cw_ids, "teacher_adv_margin", kl_edges)
        broad_effects: dict[str, Any] = {}
        for arm in BROAD_ARMS:
            direct_q = {q: effect(base_train, broad_rows[arm]["train"], ids) for q, ids in ce_train_q.items()}
            broad_effects[arm] = {
                "direct": effect(base_train, broad_rows[arm]["train"], cw_ids),
                "direct_ce20_q": direct_q,
                "spillover_non_cw_train": effect(base_train, broad_rows[arm]["train"], spill_ids),
                "heldout_validation_overall": effect(base_val, broad_rows[arm]["validation"], sorted(base_val)),
                "heldout_clean_wrong": effect(base_val, broad_rows[arm]["validation"], val_cw_ids),
                "heldout_clean_wrong_ce20_q": {
                    q: effect(base_val, broad_rows[arm]["validation"], ids) for q, ids in val_q_ce.items()
                },
                "heldout_clean_wrong_kl10_q": {
                    q: effect(base_val, broad_rows[arm]["validation"], ids) for q, ids in val_q_kl.items()
                },
            }
        gated_meta: dict[str, Any] = {}
        gated_rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
        for arm in GATED_ARMS:
            gated_rows[arm] = {}
            for split in ("train", "validation"):
                meta, data = endpoint(
                    ROOT / ".cache/analysis/ert-cw-reliability-gated-ce015-v1/endpoints",
                    run,
                    arm,
                    split,
                    gated=True,
                    expected_source="8544fed4505d423cefe6e89ad789f45c52488aac",
                )
                gated_meta[f"{arm}/{split}"] = meta
                gated_rows[arm][split] = data
        g0_train, g0_val = gated_rows["G0_BASE"]["train"], gated_rows["G0_BASE"]["validation"]
        for arm in GATED_ARMS:
            validate_endpoint_universe(g0_train, gated_rows[arm]["train"], label=f"{run} gated {arm} train")
            validate_endpoint_universe(g0_val, gated_rows[arm]["validation"], label=f"{run} gated {arm} validation")
        g2_train_ids = sorted(i for i in cw_ids if float(ce_train[i]["teacher_adv_margin"]) > 0)
        g3_train_ids = sorted(i for i in cw_ids if float(kl_train[i]["teacher_adv_margin"]) > 0)
        g2_val_ids = sorted(i for i in val_cw_ids if float(ce_val[i]["teacher_adv_margin"]) > 0)
        g3_val_ids = sorted(i for i in val_cw_ids if float(kl_val[i]["teacher_adv_margin"]) > 0)
        gated_effects: dict[str, Any] = {}
        for arm in GATED_ARMS:
            selected = (
                cw_ids
                if arm == "G1_CW_ALL_CE015"
                else g2_train_ids
                if arm == "G2_CW_R_CE20_CE015"
                else g3_train_ids
                if arm == "G3_CW_R_KL10_CE015"
                else []
            )
            val_selected = (
                val_cw_ids
                if arm == "G1_CW_ALL_CE015"
                else g2_val_ids
                if arm == "G2_CW_R_CE20_CE015"
                else g3_val_ids
                if arm == "G3_CW_R_KL10_CE015"
                else []
            )
            within = sorted(set(cw_ids) - set(selected))
            val_within = sorted(set(val_cw_ids) - set(val_selected))
            gated_effects[arm] = {
                "direct_train": effect(g0_train, gated_rows[arm]["train"], selected),
                "within_cw_spillover_train": effect(g0_train, gated_rows[arm]["train"], within),
                "non_cw_spillover_train": effect(
                    g0_train, gated_rows[arm]["train"], sorted(set(g0_train) - set(cw_ids))
                ),
                "heldout_selected_cw": effect(g0_val, gated_rows[arm]["validation"], val_selected),
                "heldout_excluded_cw": effect(g0_val, gated_rows[arm]["validation"], val_within),
                "heldout_clean_wrong_overall": effect(g0_val, gated_rows[arm]["validation"], val_cw_ids),
                "heldout_validation_overall": effect(g0_val, gated_rows[arm]["validation"], sorted(g0_val)),
                "selected_count_train": len(selected),
                "selected_count_validation": len(val_selected),
            }
        machine["seeds"][run] = {
            "clean_wrong_mask": {
                "path": str(mask_path.resolve()),
                "sha256": sha256(mask_path),
                "count": len(cw_ids),
                "ids_sha256": ids_hash(cw_ids),
            },
            "train_derived_edges": {"CE20": ce_edges, "KL10": kl_edges},
            "validation_clean_wrong": {
                "count": len(val_cw_ids),
                "ids_sha256": ids_hash(val_cw_ids),
                "CE20_q_counts": {q: len(v) for q, v in val_q_ce.items()},
                "KL10_q_counts": {q: len(v) for q, v in val_q_kl.items()},
            },
            "features": {
                "train_ce20": ce_meta,
                "train_kl10": kl_meta,
                "validation_ce20": ce_val_meta,
                "validation_kl10": kl_val_meta,
            },
            "broad_endpoint_lineage": broad_meta,
            "gated_endpoint_lineage": gated_meta,
            "broad_effects": broad_effects,
            "gated_effects": gated_effects,
        }
    machine["report_sha256"] = hashlib.sha256(
        json.dumps(machine, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong Generalization Diagnostic",
        "",
        "## Conclusion",
        "",
        "This is a read-only Direct → train Spillover → Held-out diagnostic. The validation CE-PGD20/KL-PGD10 features were replayed at epoch 79 using the exact L2/L4 parents; no training, threshold tuning, official test, or AutoAttack was run.",
        "",
        "The report distinguishes direct train-cohort correction from non-selected train effects and held-out Clean-Wrong effects. Held-out Q1–Q5 use train-derived upper boundaries; validation outcomes never define the bins.",
        "",
        "## Held-out cohort and boundary transfer",
        "",
        "| seed | train CW | held-out CW | held-out CE20 Q1–Q5 counts | held-out KL10 Q1–Q5 counts |",
        "|---|---:|---:|---|---|",
    ]
    for run in RUNS:
        value = machine["seeds"][run]
        lines.append(
            f"| {run} | {value['clean_wrong_mask']['count']} | {value['validation_clean_wrong']['count']} | {value['validation_clean_wrong']['CE20_q_counts']} | {value['validation_clean_wrong']['KL10_q_counts']} |"
        )
    lines += [
        "",
        "## Broad Screen: Direct → non-CW train Spillover → Held-out Clean Wrong",
        "",
        "| seed | arm | Direct robust Δ | Spillover robust Δ | Held-out robust Δ | Direct clean Δ | Spillover clean Δ | Held-out clean Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        for arm in BROAD_ARMS:
            value = machine["seeds"][run]["broad_effects"][arm]
            lines.append(
                f"| {run} | {arm} | {pp(value['direct']['robust']['accuracy_delta'])} | {pp(value['spillover_non_cw_train']['robust']['accuracy_delta'])} | {pp(value['heldout_clean_wrong']['robust']['accuracy_delta'])} | {pp(value['direct']['clean']['accuracy_delta'])} | {pp(value['spillover_non_cw_train']['clean']['accuracy_delta'])} | {pp(value['heldout_clean_wrong']['clean']['accuracy_delta'])} |"
            )
    lines += [
        "",
        "## Gated experiment: selected / within-CW / non-CW / held-out",
        "",
        "| seed | arm | train direct robust Δ | within-CW robust Δ | non-CW robust Δ | held-out selected robust Δ | held-out CW robust Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        for arm in GATED_ARMS:
            value = machine["seeds"][run]["gated_effects"][arm]
            lines.append(
                f"| {run} | {arm} | {pp(value['direct_train']['robust']['accuracy_delta']) if value['direct_train']['n'] else 'n/a'} | {pp(value['within_cw_spillover_train']['robust']['accuracy_delta']) if value['within_cw_spillover_train']['n'] else 'n/a'} | {pp(value['non_cw_spillover_train']['robust']['accuracy_delta']) if value['non_cw_spillover_train']['n'] else 'n/a'} | {pp(value['heldout_selected_cw']['robust']['accuracy_delta']) if value['heldout_selected_cw']['n'] else 'n/a'} | {pp(value['heldout_clean_wrong_overall']['robust']['accuracy_delta']) if value['heldout_clean_wrong_overall']['n'] else 'n/a'} |"
            )
    lines += [
        "",
        "All accuracy effects below are percentage points (pp); the machine JSON stores proportions. The JSON also contains paired rescue/harm/net-rescue rates and clean/robust probability-margin deltas.",
        "",
        "## Overall held-out validation (dilution check)",
        "",
        "This table uses all 5,000 fixed validation IDs, not only the validation Clean-Wrong subset. A positive selected-CW effect can be diluted or reversed by non-CW spillover; that is distinct from failure to transfer within the held-out Clean-Wrong subtype.",
        "",
        "| seed | arm | held-out overall robust Δ | held-out overall clean Δ | held-out CW robust Δ | held-out CW clean Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        for arm in GATED_ARMS:
            value = machine["seeds"][run]["gated_effects"][arm]
            overall = value["heldout_validation_overall"]
            cw = value["heldout_clean_wrong_overall"]
            lines.append(
                f"| {run} | {arm} | {pp(overall['robust']['accuracy_delta'])} | {pp(overall['clean']['accuracy_delta'])} | {pp(cw['robust']['accuracy_delta'])} | {pp(cw['clean']['accuracy_delta'])} |"
            )
    lines += [
        "",
        "For the broad-screen arms, the held-out Clean-Wrong table above is the subtype-transfer endpoint; the full-validation effects are available under `broad_effects[*].heldout_validation_overall` in the machine JSON.",
        "",
        "## Interpretation rules",
        "",
        "- Direct positive with held-out near zero is direct-only evidence.",
        "- Direct and Spillover positive with held-out near zero is train-distribution spillover without held-out transfer.",
        "- Positive held-out effects are evidence of transfer, not proof of generalization from two seeds.",
        "- Direct positive with negative Spillover and/or Held-out is harmful interference.",
        "- No ratio of Held-out/Direct is used as a primary metric.",
        "",
        "Held-out subtype transfer is reported only after the fixed epoch-79 validation feature lineage passed. No new intervention is selected automatically.",
        "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
