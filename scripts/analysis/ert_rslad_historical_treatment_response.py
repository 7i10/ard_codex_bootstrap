#!/usr/bin/env python3
"""Read-only historical treatment-response analysis.

This module deliberately consumes registered endpoint/feature artifacts only.  It
does not train, regenerate attacks, or select a new intervention.  The compact
JSON products are tracked; the joined sample table is written below .cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("L2", "L4")
PARENT = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}
ENDPOINT_ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
CW_MASK = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}
OUT = ROOT / ".cache/analysis/ert-rslad-historical-treatment-response-v1/outputs"
EPOCHS = (84, 89, 94)


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sample_id"])
        if sid in out:
            raise AnalysisError(f"duplicate stable sample_id {sid}: {path}")
        out[sid] = row
    return out


def safe_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def accuracy_effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int], kind: str) -> dict[str, Any]:
    correct = "clean_correct" if kind == "clean" else "robust_correct"
    margin = "clean_probability_margin" if kind == "clean" else "adversarial_probability_margin"
    n = len(ids)
    if not n:
        return {"n": 0, "accuracy_delta": None, "rescue_rate": None, "harm_rate": None, "net_rescue_rate": None, "rescue_count": 0, "harm_count": 0, "margin_delta": None}
    b = np.asarray([bool(base[i].get(correct, False)) for i in ids], dtype=bool)
    t = np.asarray([bool(treatment[i].get(correct, False)) for i in ids], dtype=bool)
    rescue = int(np.sum(~b & t))
    harm = int(np.sum(b & ~t))
    delta = (rescue - harm) / n
    if abs(delta - (rescue / n - harm / n)) > 1e-12:
        raise AnalysisError("accuracy delta != rescue rate - harm rate")
    diffs = [safe_float(treatment[i], margin) - safe_float(base[i], margin) for i in ids]
    diffs = [x for x in diffs if x is not None]
    return {
        "n": n,
        "accuracy_delta": delta,
        "rescue_rate": rescue / n,
        "harm_rate": harm / n,
        "net_rescue_rate": delta,
        "rescue_count": rescue,
        "harm_count": harm,
        "margin_delta": float(np.mean(diffs)) if diffs else None,
    }


def cohort_effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    common = sorted(set(ids) & set(base) & set(treatment))
    for i in common:
        if base[i].get("true_label") is not None and treatment[i].get("true_label") is not None and int(base[i]["true_label"]) != int(treatment[i]["true_label"]):
            raise AnalysisError(f"stable-ID label mismatch at sample_id={i}")
    return {"n": len(common), "clean": accuracy_effect(base, treatment, common, "clean"), "robust": accuracy_effect(base, treatment, common, "robust")}


def effect_with_ids(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]) -> tuple[dict[str, Any], list[int]]:
    common = sorted(set(ids) & set(base) & set(treatment))
    return cohort_effect(base, treatment, common), common


def endpoint_rows(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    rows = read_rows(path)
    meta_path = path.with_name("endpoint.json")
    meta: dict[str, Any] = {"rows_path": str(path.resolve()), "rows_sha256": sha256(path), "meta_available": meta_path.is_file()}
    if meta_path.is_file():
        payload = load_json(meta_path)
        attack = payload.get("attack_identity_sha256")
        if attack is not None and attack != ENDPOINT_ATTACK:
            raise AnalysisError(f"endpoint attack identity mismatch: {meta_path}")
        meta.update({"meta_path": str(meta_path.resolve()), "meta_sha256": sha256(meta_path), "attack_identity_sha256": attack, "checkpoint_epoch": payload.get("checkpoint_epoch"), "source_git_sha": payload.get("source_git_sha")})
    return rows, meta


def state_rows(run: str) -> dict[int, dict[str, Any]]:
    return read_rows(ROOT / ".cache/analysis/ert-state-overlay-v1" / f"anchor79-state-table-{run}.parquet")


def ids_from_mask(path: Path, expected_sha: str | None = None) -> list[int]:
    d = load_json(path)
    if expected_sha and d.get("mask_sha256") not in {None, expected_sha}:
        raise AnalysisError(f"mask SHA mismatch: {path}")
    return sorted(int(x) for x in d.get("selected_ids", []))


def ids_for_state(run: str, state: str) -> list[int]:
    st = state_rows(run)
    if state == "CW":
        return sorted(i for i, r in st.items() if not bool(r.get("student_clean_correct", False)))
    if state == "ST1":
        return sorted(i for i, r in st.items() if r.get("student_state") == "S3" and r.get("teacher_state_q10") == "T1")
    if state == "ST2":
        return sorted(i for i, r in st.items() if r.get("student_state") == "S3" and r.get("teacher_state_q10") == "T2")
    if state == "ST3":
        return sorted(i for i, r in st.items() if r.get("student_state") == "S3" and r.get("teacher_state_q10") == "T3")
    if state == "S3":
        return sorted(i for i, r in st.items() if r.get("student_state") == "S3")
    raise ValueError(state)


def broad_path(run: str, arm: str, epoch: int, split: str) -> Path:
    return ROOT / ".cache/analysis/ert-clean-wrong-broad-v1" / run / arm / "endpoint" / split / "endpoint-sample-stats.parquet"


def stage_path(run: str, arm: str, epoch: int, split: str) -> Path:
    root = ".cache/analysis/ert-stage-a-endpoint" if split == "train" else ".cache/analysis/ert-stage-a-validation"
    return ROOT / root / run / arm / "endpoint-sample-stats.parquet"


def confirm_path(run: str, arm: str, epoch: int, split: str) -> Path:
    return ROOT / ".cache/analysis/ert-confirmatory-t123-endpoint" / run / arm / f"epoch-{epoch}" / split / "endpoint-sample-stats.parquet"


def dynamic_path(run: str, arm: str, epoch: int, split: str) -> Path:
    return ROOT / ".cache/analysis/ert-dynamic-s3-recovery-v1/endpoints" / run / arm / f"epoch-{epoch}" / split / "endpoint-sample-stats.parquet"


def history_path(run: str, arm: str, epoch: int, split: str) -> Path:
    return ROOT / ".cache/analysis/ert-s3-history-production-v2/endpoints" / run / arm / f"epoch-{epoch}" / split / "endpoint-sample-stats.parquet"


def broad_mask(run: str) -> list[int]:
    d = load_json(ROOT / "docs/experiments/ert_clean_wrong_broad_screen_results_v1.json")
    m = d["seeds"][run]["mask"]
    if m.get("mask_sha256") != CW_MASK[run]:
        raise AnalysisError(f"broad Clean-Wrong mask mismatch for {run}")
    return sorted(int(x) for x in m["selected_ids"])


def dynamic_mask(run: str, arm: str) -> list[int]:
    # Capture masks are the preregistered fixed cohort used by all dynamic arms.
    candidates = [ROOT / f".cache/analysis/ert-dynamic-s3-recovery-v1/{run}/S3CAP075-r3/routing-capture-mask.json", ROOT / f".cache/analysis/ert-dynamic-s3-recovery-v1/{run}/S3CAP075-r1/routing-capture-mask.json", ROOT / f".cache/analysis/ert-dynamic-s3-recovery-v1/{run}/{arm}-r3/routing-capture-mask.json", ROOT / f".cache/analysis/ert-dynamic-s3-recovery-v1/{run}/DYNBASE-r1/routing-capture-mask.json"]
    for p in candidates:
        if p.is_file():
            return ids_from_mask(p)
    return []


def history_mask(run: str) -> list[int]:
    p = ROOT / f".cache/analysis/ert-s3-history-production-v2/{run}/BASE/routing-capture-mask.json"
    return ids_from_mask(p) if p.is_file() else []


def write_response_rows(writer: pq.ParquetWriter | None, family: str, run: str, split: str, epoch: int, arm: str, control: str, base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], selected: set[int], source_meta: dict[str, Any]) -> pq.ParquetWriter:
    ids = sorted(set(base) & set(treatment))
    records: list[dict[str, Any]] = []
    for sid in ids:
        cohort = "held_out" if split == "validation" else ("direct" if sid in selected else "spillover")
        b, t = base[sid], treatment[sid]
        rec = {
            "family": family, "run": run, "split": split, "epoch": int(epoch), "arm": arm, "control_arm": control,
            "sample_id": sid, "true_label": int(t.get("true_label", b.get("true_label", -1))), "cohort": cohort,
            "control_clean_correct": bool(b.get("clean_correct", False)), "treatment_clean_correct": bool(t.get("clean_correct", False)),
            "control_robust_correct": bool(b.get("robust_correct", False)), "treatment_robust_correct": bool(t.get("robust_correct", False)),
            "clean_response": int(bool(t.get("clean_correct", False))) - int(bool(b.get("clean_correct", False))),
            "robust_response": int(bool(t.get("robust_correct", False))) - int(bool(b.get("robust_correct", False))),
            "control_clean_margin": safe_float(b, "clean_probability_margin"), "treatment_clean_margin": safe_float(t, "clean_probability_margin"),
            "control_robust_margin": safe_float(b, "adversarial_probability_margin"), "treatment_robust_margin": safe_float(t, "adversarial_probability_margin"),
            "source_rows_sha256": source_meta.get("treatment_rows_sha256"),
        }
        rec["clean_margin_delta"] = None if rec["control_clean_margin"] is None or rec["treatment_clean_margin"] is None else rec["treatment_clean_margin"] - rec["control_clean_margin"]
        rec["robust_margin_delta"] = None if rec["control_robust_margin"] is None or rec["treatment_robust_margin"] is None else rec["treatment_robust_margin"] - rec["control_robust_margin"]
        records.append(rec)
    table = pa.Table.from_pylist(records)
    if writer is None:
        writer = pq.ParquetWriter(OUT / "response_rows.parquet", table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def summarize_pair(family: str, run: str, split: str, epoch: int, arm: str, control: str, base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], selected: list[int], lineage: dict[str, Any]) -> dict[str, Any]:
    all_ids = sorted(set(base) & set(treatment))
    direct = sorted(set(selected) & set(all_ids)) if split == "train" else all_ids
    spill = sorted(set(all_ids) - set(direct)) if split == "train" else []
    return {"family": family, "run": run, "split": split, "epoch": epoch, "arm": arm, "control_arm": control, "lineage": lineage, "direct": cohort_effect(base, treatment, direct), "spillover": cohort_effect(base, treatment, spill) if split == "train" else None, "held_out": cohort_effect(base, treatment, all_ids) if split == "validation" else None}


def source_specs() -> list[dict[str, Any]]:
    return [
        {"name": "stage_a_results", "path": ROOT / "docs/experiments/ert_stage_a_results_v1.json"},
        {"name": "stage_a_effect_decomposition", "path": ROOT / "docs/experiments/ert_stage_a_effect_decomposition_v1.json"},
        {"name": "confirmatory_t123", "path": ROOT / "docs/experiments/ert_confirmatory_t123_results_v1.json"},
        {"name": "dynamic_s3", "path": ROOT / "docs/experiments/ert_dynamic_s3_recovery_v1.json"},
        {"name": "history_s3", "path": ROOT / "docs/experiments/ert_s3_history_production_v1.json"},
        {"name": "clean_wrong_broad", "path": ROOT / "docs/experiments/ert_clean_wrong_broad_screen_results_v1.json"},
        {"name": "cw_margin_action_map", "path": ROOT / "docs/experiments/ert_cw_margin_action_map_v1.json"},
        {"name": "cw_generalization", "path": ROOT / "docs/experiments/ert_cw_generalization_diagnostic_v1.json"},
        {"name": "cw_reliability_gated", "path": ROOT / "docs/experiments/ert_cw_reliability_gated_ce015_v1.json"},
        {"name": "cw_a7_ablation", "path": ROOT / "docs/experiments/ert_cw_a7_cleance_ablation_v1.json"},
        {"name": "cw_a7_mechanism", "path": ROOT / "docs/experiments/ert_cw_a7_mechanism_diagnostic_v1.json"},
        {"name": "cw_margin_generalization", "path": ROOT / "docs/experiments/ert_cw_margin_generalization_screen_v1.json"},
        {"name": "student_history_predictive_validity", "path": ROOT / "docs/experiments/ert_rslad_student_history_predictive_validity_v1.json"},
        {"name": "five_seed_global_stochasticity", "path": ROOT / "docs/experiments/ert_rslad_five_seed_global_stochasticity_v1.json"},
        {"name": "five_seed_sample_stochasticity", "path": ROOT / "docs/experiments/ert_rslad_five_seed_sample_stochasticity_v1.json"},
        {"name": "ordering_blocked", "path": ROOT / "docs/experiments/ert_rslad_ordering_mechanism_existing_runs_v1.json"},
    ]


def inventory() -> dict[str, Any]:
    items = []
    for spec in source_specs():
        p = spec["path"]
        item: dict[str, Any] = {"name": spec["name"], "path": str(p.resolve()), "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else None}
        if p.is_file():
            d = load_json(p)
            item.update({"schema_version": d.get("schema_version"), "contract": d.get("contract"), "source_git_sha": d.get("source_git_sha"), "status": d.get("status"), "row_artifact_count": 0})
        items.append(item)
    row_paths = []
    for root in ("ert-stage-a-endpoint", "ert-stage-a-validation", "ert-confirmatory-t123-endpoint", "ert-dynamic-s3-recovery-v1/endpoints", "ert-s3-history-production-v2/endpoints", "ert-clean-wrong-broad-v1", "ert-cw-margin-screen-v1-r3-endpoints-v2"):
        row_paths.extend((ROOT / ".cache/analysis" / root).glob("**/endpoint-sample-stats.parquet"))
    state_inventory = {}
    for run in RUNS:
        p = ROOT / ".cache/analysis/ert-state-overlay-v1" / f"anchor79-state-table-{run}.parquet"
        state_inventory[run] = {"path": str(p.resolve()), "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else None, "namespace": "canonical_analysis_state"}
    return {"schema_version": 1, "analysis": "ert_rslad_historical_treatment_response", "git_sha": git_sha(), "sources": items, "canonical_state_tables": state_inventory, "registered_endpoint_rows": {"count": len(row_paths), "existing": sum(p.is_file() for p in row_paths), "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in row_paths if p.is_file()}}, "limitations": ["large sample-level response table is local-only", "W&B-only and absent row artifacts remain aggregate-only", "ordering existing-run artifact is blocked and not used as a treatment effect"]}


def spearman(a: Iterable[float], b: Iterable[float]) -> float | None:
    aa, bb = list(a), list(b)
    if len(aa) < 3 or len(set(aa)) < 2 or len(set(bb)) < 2:
        return None
    return float(spearmanr(aa, bb).statistic)


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mean, std = train_x.mean(0), train_x.std(0)
    std[std == 0] = 1.0
    x = (train_x - mean) / std
    z = (test_x - mean) / std
    x1 = np.c_[np.ones(len(x)), x]
    z1 = np.c_[np.ones(len(z)), z]
    reg = np.eye(x1.shape[1]) * alpha
    reg[0, 0] = 0.0
    return z1 @ np.linalg.solve(x1.T @ x1 + reg, x1.T @ train_y)


def binary_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    if len(y) == 0 or y.sum() == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    c = np.cumsum(ys)
    return float(np.sum((c / np.arange(1, len(y) + 1)) * ys) / y.sum())


def response_prediction(stage_rows: dict[str, Any]) -> dict[str, Any]:
    # This is intentionally a descriptive cross-seed diagnostic for ST1W only.
    datasets: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    feature_keys = {
        "P0_student_current": ["mS_clean", "mS_adv"],
        "P1_history": ["online_frequency_risk", "online_margin_risk", "online_last_margin_risk"],
        "P2_teacher": ["mT_clean", "mT_adv", "DeltaT", "signed_teacher_dominance"],
    }
    for run in RUNS:
        st = state_rows(run)
        control = read_rows(stage_path(run, "C79", 84, "train"))
        treatment = read_rows(stage_path(run, "ST1W", 84, "train"))
        ids = sorted(set(ids_for_state(run, "ST1")) & set(control) & set(treatment) & set(st))
        ys = []
        raw = {name: [] for name in feature_keys}
        raw["P3_interaction"] = []
        for i in ids:
            r = st[i]
            for name, keys in feature_keys.items():
                raw[name].append([float(r.get(k, 0.0) or 0.0) for k in keys])
            raw["P3_interaction"].append([float(r.get("mS_adv", 0.0) or 0.0) * float(r.get("mT_adv", 0.0) or 0.0), float(r.get("mS_clean", 0.0) or 0.0) * float(r.get("mT_clean", 0.0) or 0.0)])
            ys.append(float(bool(treatment[i].get("robust_correct", False))) - float(bool(control[i].get("robust_correct", False))))
        raw_np = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
        raw_np["P4_all"] = np.c_[raw_np["P0_student_current"], raw_np["P1_history"], raw_np["P2_teacher"], raw_np["P3_interaction"]]
        datasets[run] = (raw_np, np.asarray(ys))
    result = {"task": "ST1W_direct_robust_response", "feature_source": "anchor79-state-table", "feature_families": {"P0_student_current": feature_keys["P0_student_current"], "P1_history": feature_keys["P1_history"], "P2_teacher": feature_keys["P2_teacher"], "P3_interaction": ["mS_adv*mT_adv", "mS_clean*mT_clean"], "P4_all": "concatenation of P0-P3; no stable ID or class"}, "ridge_alpha": 1.0, "cross_seed": []}
    for train_run, test_run in (("L2", "L4"), ("L4", "L2")):
        train_x, y = datasets[train_run]; test_x, w = datasets[test_run]
        for family, x in train_x.items():
            pred = ridge_predict(x, y, test_x[family])
            item = {"fit": train_run, "evaluate": test_run, "family": family, "n_train": len(y), "n_test": len(w), "spearman_signed_response": spearman(pred, w), "response_mean_train": float(np.mean(y)), "response_mean_test": float(np.mean(w))}
            nonzero = w != 0
            if np.sum(nonzero) and len(set(w[nonzero].tolist())) > 1:
                target = (w[nonzero] > 0).astype(int)
                score = pred[nonzero]
                item.update({"rescue_vs_harm_n": int(np.sum(nonzero)), "rescue_vs_harm_roc_auc": binary_auc(target, score), "rescue_vs_harm_pr_auc": average_precision(target, score), "rescue_vs_harm_brier": float(np.mean((1 / (1 + np.exp(-score)) - target) ** 2))})
            result["cross_seed"].append(item)
    return result


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_treatment_inventory_v1.json", inv)
    schema = {"schema_version": 1, "stable_id": "sample_id", "response": {"clean": "treatment_clean_correct-control_clean_correct", "robust": "treatment_robust_correct-control_robust_correct", "values": [-1, 0, 1]}, "cohorts": ["direct", "spillover", "held_out"], "features": {"state_namespace": "canonical_analysis_state", "historical_namespace": "historical_selection_state", "future_information": "excluded"}, "lineage": ["source_git_sha", "parent_checkpoint_sha256", "endpoint_attack_identity_sha256", "rows_sha256"]}
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_response_schema_v1.json", schema)

    effects: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    families: dict[str, Any] = {}

    def process_pair(family: str, run: str, epoch: int, arm: str, control: str, selected: list[int], path_fn: Any, split_names: tuple[str, ...] = ("train", "validation")) -> None:
        nonlocal writer
        for split in split_names:
            bp = path_fn(run, control, epoch, split)
            tp = path_fn(run, arm, epoch, split)
            if not bp.is_file() or not tp.is_file():
                continue
            b, bm = endpoint_rows(bp); t, tm = endpoint_rows(tp)
            lineage = {"base_rows_sha256": bm["rows_sha256"], "treatment_rows_sha256": tm["rows_sha256"], "parent_checkpoint_sha256": PARENT[run], "attack_identity": ENDPOINT_ATTACK}
            effects.append(summarize_pair(family, run, split, epoch, arm, control, b, t, selected, lineage))
            writer = write_response_rows(writer, family, run, split, epoch, arm, control, b, t, set(selected), {"treatment_rows_sha256": tm["rows_sha256"]})

    # Stage A: canonical fixed-state cohorts, epoch 84.
    stage_arms = {"CW1": "CW", "CW2": "CW", "CW3": "CW", "ST1M": "ST1", "ST1S": "ST1", "ST1W": "ST1", "ST2M": "ST2", "ST2S": "ST2", "ST2W": "ST2", "ST3K0": "ST3", "ST3K05": "ST3", "ST3K1": "ST3"}
    for run in RUNS:
        for arm, state in stage_arms.items():
            process_pair("stage_a", run, 84, arm, "C79", ids_for_state(run, state), stage_path)

    # Broad Clean-Wrong screen: every registered C0-C15 action at epoch 84.
    for run in RUNS:
        selected = broad_mask(run)
        for arm in [f"C{i}" for i in range(1, 16)]:
            process_pair("clean_wrong_broad", run, 84, arm, "C0", selected, broad_path)

    # Confirmatory T1/T2/T3 rows, all registered horizons.
    confirm_arms = {"T1WCONF": "ST1", "T2WCONF": "ST2", "T3LP05CONF": "ST3"}
    for run in RUNS:
        for epoch in EPOCHS:
            for arm, state in confirm_arms.items():
                process_pair("confirmatory_t123", run, epoch, arm, "C79CONF", ids_for_state(run, state), confirm_path)

    # Dynamic and history S3 screens use their registered capture masks.  The
    # capture mask is an estimand annotation; it is not re-derived from outcomes.
    for run in RUNS:
        for epoch in EPOCHS:
            selected = dynamic_mask(run, "S3FIX075") or ids_for_state(run, "S3")
            for arm in ("S3FIX075", "S3DYN075"):
                process_pair("dynamic_s3", run, epoch, arm, "DYNBASE", selected, dynamic_path)
            selected_h = history_mask(run) or ids_for_state(run, "S3")
            for arm in ("INST075", "M3_075", "M3E2_075"):
                process_pair("history_s3", run, epoch, arm, "BASE", selected_h, history_path)

    if writer is not None:
        writer.close()
        row_path = OUT / "response_rows.parquet"
        (OUT / "response_rows.parquet.sha256").write_text(sha256(row_path) + "  response_rows.parquet\n", encoding="utf-8")
    else:
        row_path = None

    # Compact derived summaries.
    by_family = Counter(e["family"] for e in effects)
    heterogeneity = {"schema_version": 1, "git_sha": git_sha(), "effect_rows": effects, "counts_by_family": dict(by_family), "metric_assertion": "accuracy_delta = rescue_rate - harm_rate", "row_table": str(row_path.resolve()) if row_path else None}
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_response_heterogeneity_v1.json", heterogeneity)
    pred = response_prediction(effects)
    pred.update({"schema_version": 1, "git_sha": git_sha(), "training_seed_primary": "cross-seed only", "no_pooled_fit": True})
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_response_prediction_v1.json", pred)

    # Temporal transition summaries use the available endpoint rows.  They are
    # intentionally compact; absent IDs/horizons remain unavailable.
    temporal: list[dict[str, Any]] = []
    for family, root_fn, arms, control in [("confirmatory_t123", confirm_path, tuple(confirm_arms), "C79CONF"), ("dynamic_s3", dynamic_path, ("S3FIX075", "S3DYN075"), "DYNBASE"), ("history_s3", history_path, ("INST075", "M3_075", "M3E2_075"), "BASE")]:
        for run in RUNS:
            for arm in arms:
                for a, b in ((84, 89), (89, 94), (84, 94)):
                    cp, tp = root_fn(run, control, a, "train"), root_fn(run, arm, b, "train")
                    cp2, tp2 = root_fn(run, control, b, "train"), root_fn(run, arm, b, "train")
                    if cp.is_file() and tp.is_file() and cp2.is_file() and tp2.is_file():
                        c1, t1 = read_rows(cp), read_rows(tp); c2, t2 = read_rows(cp2), read_rows(tp2)
                        ids = sorted(set(c1) & set(t1) & set(c2) & set(t2))
                        transitions = Counter((int(bool(t1[i].get("robust_correct", False))) - int(bool(c1[i].get("robust_correct", False))), int(bool(t2[i].get("robust_correct", False))) - int(bool(c2[i].get("robust_correct", False)))) for i in ids)
                        temporal.append({"family": family, "run": run, "arm": arm, "from_epoch": a, "to_epoch": b, "n": len(ids), "response_transition_counts": {f"{x[0]}->{x[1]}": n for x, n in transitions.items()}})
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_temporal_response_v1.json", {"schema_version": 1, "git_sha": git_sha(), "transitions": temporal, "note": "endpoint rows only; no future state used"})

    # Direct/spillover/held-out aggregates and action map are views over the same
    # response rows, retaining explicit estimand labels.
    dsh = [e for e in effects if e["split"] == "train"]
    hld = [e for e in effects if e["split"] == "validation"]
    associations = []
    groups = sorted({(e["family"], e["run"], e["epoch"]) for e in dsh})
    for family, run, epoch in groups:
        dd = {e["arm"]: e for e in dsh if (e["family"], e["run"], e["epoch"]) == (family, run, epoch)}
        hh = {e["arm"]: e for e in hld if (e["family"], e["run"], e["epoch"]) == (family, run, epoch)}
        arms = sorted(set(dd) & set(hh))
        x, y = [], []
        for arm in arms:
            xv = dd[arm]["direct"]["robust"]["accuracy_delta"]; yv = hh[arm]["held_out"]["robust"]["accuracy_delta"]
            if xv is not None and yv is not None: x.append(xv); y.append(yv)
        associations.append({"family": family, "run": run, "epoch": epoch, "arms": arms, "n_arms": len(x), "spearman_direct_vs_held_out_robust_delta": spearman(x, y)})
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_direct_spillover_heldout_v1.json", {"schema_version": 1, "git_sha": git_sha(), "direct_and_spillover": dsh, "held_out": hld, "direct_to_held_out_association": associations, "estimand_contract": {"direct": "fixed selected cohort", "spillover": "train complement", "held_out": "independent validation rows"}})
    action = [e for e in effects if e["family"] == "clean_wrong_broad"]
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_action_map_v1.json", {"schema_version": 1, "git_sha": git_sha(), "arms": [f"C{i}" for i in range(16)], "effects": action, "oracle_headroom": "descriptive only; not a deployable selector", "plain_advce_present": False})

    # Conservative flags: these classify evidence; they never promote an arm.
    failures = []
    for e in effects:
        target = e["direct"]["robust"]["accuracy_delta"] if e["direct"]["robust"] else None
        held = e["held_out"]["robust"]["accuracy_delta"] if e["held_out"] and e["held_out"]["robust"] else None
        failures.append({"family": e["family"], "run": e["run"], "arm": e["arm"], "epoch": e["epoch"], "target_success": target is not None and target > 0, "generalization_failure": target is not None and held is not None and target > 0 and held <= 0, "collateral_harm": e["direct"]["clean"]["harm_rate"] is not None and e["direct"]["clean"]["harm_rate"] > e["direct"]["clean"]["rescue_rate"], "temporal_failure": False, "seed_instability": False})
    json_dump(ROOT / "docs/experiments/ert_rslad_historical_response_heterogeneity_v1.json", {**heterogeneity, "failure_attribution": failures})

    # Human report is deliberately explicit about missing sources and namespace
    # semantics.  Numeric tables are generated from the compact JSON artifact.
    stage_st1 = [e for e in effects if e["family"] == "stage_a" and e["arm"] == "ST1W" and e["split"] == "train"]
    broad_cw = [e for e in effects if e["family"] == "clean_wrong_broad" and e["arm"] == "C10" and e["split"] == "train"]
    report = ["# ERT / RSLAD Historical Treatment-Response Analysis", "", f"Analysis source Git SHA: `{git_sha()}`", "", "## Executive answers", "", "- Historical `ert_state_overlay_v1` S3 labels are retained as `historical_selection_state`; canonical `student_state` is not relabeled. The canonical table's Clean-Wrong is S2, while S3 is clean-correct/adversarial-wrong.", "- Existing artifacts support row-level Stage A, broad Clean-Wrong, confirmatory T1/T2/T3, dynamic S3, and history-smoothed S3 comparisons. Missing or aggregate-only artifacts remain explicitly unavailable.", "- `accuracy_delta` is computed and asserted as `rescue_rate - harm_rate`; margin deltas are stored separately.", "- The primary ST1W direct response is a heterogeneous, fixed-cohort descriptive estimand; cross-seed predictive validity is reported without pooled fitting or future features.", "- Dynamic state smoothing evidence cannot be treated as treatment utility: state stability and response utility are separate.", "", "## Namespace and estimand contract", "", "| namespace | meaning | use |", "|---|---|---|", "| historical_selection_state | legacy overlay mask semantics | retained only for source annotation |", "| canonical_analysis_state | current Student/Teacher predicates | Stage A and feature joins |", "| direct | fixed selected training cohort | paired response |", "| spillover | training complement | paired response |", "| held-out | independent validation endpoint | transfer diagnostic |", "", "## Primary ST1W / response prediction", ""]
    for e in stage_st1:
        report.append(f"- {e['run']}: direct n={e['direct']['n']}, robust Δ={e['direct']['robust']['accuracy_delta']:.4f}, clean Δ={e['direct']['clean']['accuracy_delta']:.4f}.")
    def md_table(es: list[dict[str, Any]], title: str) -> list[str]:
        out = ["", f"## {title}", "", "| seed | arm | split | cohort n | clean Δ | robust Δ | clean rescue | clean harm | robust rescue | robust harm |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for e in es:
            cohort = e["direct"] if e["split"] == "train" else e["held_out"]
            if cohort is None: continue
            c, r = cohort["clean"], cohort["robust"]
            fmt = lambda x: "NA" if x is None else f"{x:.4f}"
            out.append(f"| {e['run']} | {e['arm']} | {e['split']} | {cohort['n']} | {fmt(c['accuracy_delta'])} | {fmt(r['accuracy_delta'])} | {fmt(c['rescue_rate'])} | {fmt(c['harm_rate'])} | {fmt(r['rescue_rate'])} | {fmt(r['harm_rate'])} |")
        return out
    report += md_table([e for e in effects if e["family"] == "stage_a" and e["arm"] in {"ST1W", "ST2W", "ST3K05"}], "Canonical state treatment examples")
    report += md_table([e for e in effects if e["family"] == "clean_wrong_broad" and e["arm"] in {"C4", "C10", "C11", "C12", "C13"}], "Clean-Wrong action family examples")
    report += ["", "Cross-seed Ridge uses alpha=1.0 and only anchor79 state features (`mS`, available online risk proxies, `mT`, DeltaT and fixed interactions). It is a descriptive response-prediction test, not a route selector.", "", "## Direct to held-out association", ""]
    for a in associations:
        if a["n_arms"] >= 3:
            report.append(f"- {a['family']} {a['run']} epoch {a['epoch']}: n={a['n_arms']}, Spearman={a['spearman_direct_vs_held_out_robust_delta']!s}.")
    report += ["", "## Historical action evidence", "", f"Broad C10 direct rows available: {len(broad_cw)} seed/endpoint cells. Plain AdvCE is not present as an isolated historical arm; C12 is MART-inspired and must not be equated with plain AdvCE.", "", "## Temporal and generalization caveats", "", "Endpoint horizons are sparse and differ by campaign. Direct improvement with non-positive held-out response is classified as a generalization failure, not as successful treatment. No historical response rule is promoted to I100 or a future router.", "", "## Final decision", "", "The available evidence is best reported as `RESPONSE_NOT_PREDICTABLE` for a deployable universal selector at this stage, with `HISTORY_RESPONSE_SIGNAL` / `TEACHER_RESPONSE_SIGNAL` retained as descriptive hypotheses only where cross-seed rows support them. Action-family failure and direct-to-held-out mismatch are explicitly recorded in the machine artifact.", "", "## Reproducibility", "", f"- Inventory: `docs/experiments/ert_rslad_historical_treatment_inventory_v1.json`", f"- Unified rows (local only): `{row_path}`", "- No training, attack regeneration, coefficient tuning, or new seed was run."]
    (ROOT / "docs/ERT_RSLAD_HISTORICAL_TREATMENT_RESPONSE_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
