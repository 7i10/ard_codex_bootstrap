#!/usr/bin/env python3
"""Read-only I100 Clean-Wrong train-to-held-out generalization audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SEEDS = ("dev-1", "dev-2")
ARMS = ("I100_CONTROL", "CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM")
EPOCHS = (129, 149, 169, 189, 199)
TRAIN_MASKS = {
    "dev-1": Path("docs/experiments/ert_rslad_i100_action_transfer_masks_dev1_v1.json"),
    "dev-2": Path("docs/experiments/ert_rslad_i100_action_transfer_masks_dev2_v1.json"),
}
PARENTS = {"dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835", "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7"}
ENDPOINT_ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result = {int(r["sample_id"]): r for r in values}
    if len(result) != len(values):
        raise ValueError(f"duplicate stable IDs: {path}")
    return result


def effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    if not ids:
        return {"n": 0}
    if ids - set(base) or ids - set(treatment):
        raise ValueError("stable-ID join failure")
    out: dict[str, Any] = {"n": len(ids)}
    for kind, correct, margin in (("clean", "clean_correct", "clean_probability_margin"), ("robust", "robust_correct", "adversarial_probability_margin")):
        rescue = harm = 0
        md = 0.0
        for sid in ids:
            b, t = bool(base[sid][correct]), bool(treatment[sid][correct])
            rescue += int(not b and t)
            harm += int(b and not t)
            md += float(treatment[sid][margin]) - float(base[sid][margin])
        n = len(ids)
        delta = (rescue - harm) / n
        if abs(delta - (rescue / n - harm / n)) > 1e-12:
            raise ValueError("accuracy delta != rescue - harm")
        out[kind] = {"delta": delta, "rescue_rate": rescue / n, "harm_rate": harm / n,
                     "net_rescue_rate": delta, "rescue_count": rescue, "harm_count": harm,
                     "margin_delta": md / n}
    return out


def load_endpoint(root: Path, seed: str, arm: str, epoch: int, split: str) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    d = root / "runs" / seed / arm / "endpoints" / f"e{epoch}-{split}"
    meta = json.loads((d / "endpoint.json").read_text(encoding="utf-8"))
    if meta.get("attack_identity_sha256") != ENDPOINT_ATTACK:
        raise ValueError(f"attack identity mismatch: {d}")
    return rows(d / "endpoint-sample-stats.parquet"), meta


def feature_summary(feature_rows: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    keys = ("student_clean_margin", "student_adv_margin", "teacher_clean_margin", "teacher_adv_margin")
    result: dict[str, Any] = {"n": len(ids)}
    for key in keys:
        vals = sorted(float(feature_rows[i][key]) for i in ids)
        if not vals:
            continue
        result[key] = {"mean": sum(vals) / len(vals), "median": vals[len(vals) // 2],
                       "q20": vals[len(vals) // 5], "q80": vals[(4 * len(vals)) // 5]}
    result["teacher_adv_correct_rate"] = sum(float(feature_rows[i]["teacher_adv_margin"]) > 0 for i in ids) / len(ids) if ids else None
    return result


def overlap(a: set[int], b: set[int]) -> dict[str, Any]:
    union = a | b
    return {"a": len(a), "b": len(b), "intersection": len(a & b), "union": len(union),
            "jaccard": len(a & b) / len(union) if union else None,
            "a_only": len(a - b), "b_only": len(b - a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/shunsukenaito/workspace-local/ard-runs/ert-i100-cw-long-horizon-historical"))
    ap.add_argument("--feature-root", type=Path, default=Path(".cache/analysis/ert-i100-cw-gap-e99"))
    ap.add_argument("--output-dir", type=Path, default=Path("docs/experiments"))
    args = ap.parse_args()
    root, feature_root, out = args.root.resolve(), args.feature_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract = {"schema_version": 1, "contract": "ert_rslad_i100_cw_heldout_gap_contract_v1",
                "analysis": "read-only; e99 pre-treatment validation state; e129-e199 outcomes",
                "source_git_sha": "89f17818a3ee7d899c9a45cee57606aa60e9c93f", "parents": PARENTS,
                "endpoint_attack_identity_sha256": ENDPOINT_ATTACK,
                "training": "KL-PGD10 eps=8/255 step=2/255 random_start Teacher-clean target",
                "state_epoch": 99, "outcome_epochs": list(EPOCHS),
                "e114_row_artifact": "unavailable in current local artifact inventory; no imputation",
                "no_training": True, "no_threshold_tuning": True}
    (out / "ert_rslad_i100_cw_heldout_gap_contract_v1.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    state: dict[str, Any] = {"schema_version": 1, "contract": "ert_rslad_i100_cw_heldout_state_decomposition_v1", "seeds": {}}
    temporal: dict[str, Any] = {"schema_version": 1, "contract": "ert_rslad_i100_cw_heldout_temporal_response_v1", "seeds": {}}
    overlaps: dict[str, Any] = {"schema_version": 1, "contract": "ert_rslad_i100_cw_heldout_action_overlap_v1", "seeds": {}}
    shifts: dict[str, Any] = {"schema_version": 1, "contract": "ert_rslad_i100_cw_heldout_feature_shift_v1", "status": "unavailable", "reason": "e99 train feature parquet was not present; only registered e99 validation replay rows were available", "seeds": {}}
    for seed in SEEDS:
        ce = rows(feature_root / ("L2" if seed == "dev-1" else "L4") / "CE20" / "validation-feature-stats.parquet")
        kl = rows(feature_root / ("L2" if seed == "dev-1" else "L4") / "KL10" / "validation-feature-stats.parquet")
        if set(ce) != set(kl):
            raise ValueError("CE/KL validation ID mismatch")
        if any(bool(ce[i]["student_clean_correct"]) != bool(kl[i]["student_clean_correct"]) for i in ce):
            raise ValueError("clean state differs between CE and KL replay")
        val_cw = {i for i, r in ce.items() if not bool(r["student_clean_correct"])}
        val_non = set(ce) - val_cw
        local = "L2" if seed == "dev-1" else "L4"
        mask = json.loads((Path(".") / TRAIN_MASKS[seed]).read_text(encoding="utf-8"))
        train_cw = set(map(int, mask["masks"]["clean_wrong"]["selected_ids"]))
        state["seeds"][seed] = {"validation_cw": {"n": len(val_cw), "ids_sha256": hashlib.sha256(json.dumps(sorted(val_cw), separators=(",", ":")).encode()).hexdigest()},
                                 "validation_non_cw_n": len(val_non), "train_cw_n": len(train_cw),
                                 "validation_cw_features": feature_summary(ce, val_cw), "validation_non_cw_features": feature_summary(ce, val_non),
                                 "feature_lineage": {"CE20": sha256(feature_root / local / "CE20" / "validation-feature-stats.parquet"), "KL10": sha256(feature_root / local / "KL10" / "validation-feature-stats.parquet"),
                                                      "CE20_meta": sha256(feature_root / local / "CE20" / "validation-feature-replay.json"), "KL10_meta": sha256(feature_root / local / "KL10" / "validation-feature-replay.json")},
                                 "endpoint_lineage": {}}
        temporal["seeds"][seed] = {}
        overlaps["seeds"][seed] = {}
        # Cached outcome rows are available from e129 onward. e114 is recorded as missing.
        for arm in ARMS:
            temporal["seeds"][seed][arm] = {"available_epochs": list(EPOCHS), "e114": "unavailable"}
        for epoch in EPOCHS:
            loaded_meta = {arm: load_endpoint(root, seed, arm, epoch, "validation")[1] for arm in ARMS}
            loaded = {arm: load_endpoint(root, seed, arm, epoch, "validation")[0] for arm in ARMS}
            state["seeds"][seed]["endpoint_lineage"][f"e{epoch}-validation"] = {arm: {"checkpoint_sha256": loaded_meta[arm]["checkpoint_sha256"], "rows_sha256": loaded_meta[arm]["rows_sha256"], "meta_sha256": sha256(root / "runs" / seed / arm / "endpoints" / f"e{epoch}-validation" / "endpoint.json")} for arm in ARMS}
            base = loaded["I100_CONTROL"]
            if set(base) != set(ce):
                raise ValueError(f"{seed}/e{epoch}: validation feature/outcome ID mismatch")
            overall = {}
            for arm in ARMS:
                all_eff = effect(base, loaded[arm], set(base))
                cw_eff = effect(base, loaded[arm], val_cw)
                non_eff = effect(base, loaded[arm], val_non)
                weighted = (len(val_cw) * cw_eff["robust"]["delta"] + len(val_non) * non_eff["robust"]["delta"]) / len(base)
                if abs(weighted - all_eff["robust"]["delta"]) > 1e-12:
                    raise ValueError("weighted held-out reconciliation failed")
                overall[arm] = {"overall": all_eff, "validation_cw": cw_eff, "validation_non_cw": non_eff,
                                "weighted_reconciliation": {"robust_delta": weighted, "matches_overall": True}}
                state["seeds"][seed].setdefault("outcomes", {})[str(epoch)] = overall
        # e199 train direct/spillover and action overlap.
        train_meta = {arm: load_endpoint(root, seed, arm, 199, "train")[1] for arm in ARMS}
        train = {arm: load_endpoint(root, seed, arm, 199, "train")[0] for arm in ARMS}
        state["seeds"][seed]["endpoint_lineage"]["e199-train"] = {arm: {"checkpoint_sha256": train_meta[arm]["checkpoint_sha256"], "rows_sha256": train_meta[arm]["rows_sha256"], "meta_sha256": sha256(root / "runs" / seed / arm / "endpoints" / "e199-train" / "endpoint.json")} for arm in ARMS}
        for arm in ARMS:
            direct = effect(train["I100_CONTROL"], train[arm], train_cw)
            spill = effect(train["I100_CONTROL"], train[arm], set(train["I100_CONTROL"]) - train_cw)
            state["seeds"][seed].setdefault("train_e199", {})[arm] = {"direct_cw": direct, "spillover_non_cw": spill}
            for scope, ids in (("validation_cw", val_cw), ("train_cw", train_cw)):
                if arm == "I100_CONTROL":
                    continue
                c = load_endpoint(root, seed, arm, 199, "validation" if scope == "validation_cw" else "train")[0]
                b = load_endpoint(root, seed, "I100_CONTROL", 199, "validation" if scope == "validation_cw" else "train")[0]
                rescue = {i for i in ids if not bool(b[i]["robust_correct"]) and bool(c[i]["robust_correct"])}
                harm = {i for i in ids if bool(b[i]["robust_correct"]) and not bool(c[i]["robust_correct"])}
                overlaps["seeds"][seed].setdefault(scope, {})[arm] = {"rescue_n": len(rescue), "harm_n": len(harm), "rescue_ids_sha256": hashlib.sha256(json.dumps(sorted(rescue), separators=(",", ":")).encode()).hexdigest(), "harm_ids_sha256": hashlib.sha256(json.dumps(sorted(harm), separators=(",", ":")).encode()).hexdigest()}
            # Store actual sets temporarily for overlap arithmetic.
            def response(scope: str, arm_name: str) -> tuple[set[int], set[int]]:
                ids = val_cw if scope == "validation_cw" else train_cw
                split = "validation" if scope == "validation_cw" else "train"
                b = load_endpoint(root, seed, "I100_CONTROL", 199, split)[0]
                t = load_endpoint(root, seed, arm_name, 199, split)[0]
                return ({i for i in ids if not bool(b[i]["robust_correct"]) and bool(t[i]["robust_correct"])}, {i for i in ids if bool(b[i]["robust_correct"]) and not bool(t[i]["robust_correct"])})
            if arm != "I100_CONTROL":
                for scope in ("validation_cw", "train_cw"):
                    pr, ph = response(scope, "CLEAN_WRONG_PLAIN_ADVCE")
                    tr, th = response(scope, "CLEAN_WRONG_TPFM")
                    overlaps["seeds"]["rescue_overlap" if False else seed].setdefault("plain_vs_tpfm", {})[scope] = {"rescue": overlap(pr, tr), "harm": overlap(ph, th), "plain_only_rescue": len(pr-tr), "tpfm_only_rescue": len(tr-pr), "opposite_response": len((pr & th) | (tr & ph))}
        # temporal response transitions at available epochs for validation CW/non-CW.
        for arm in ("CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM"):
            for scope, ids in (("validation_cw", val_cw), ("validation_non_cw", val_non)):
                signs: dict[int, dict[int, int]] = {}
                for epoch in EPOCHS:
                    b = load_endpoint(root, seed, "I100_CONTROL", epoch, "validation")[0]
                    t = load_endpoint(root, seed, arm, epoch, "validation")[0]
                    signs[epoch] = {i: int(bool(t[i]["robust_correct"])) - int(bool(b[i]["robust_correct"])) for i in ids}
                transitions = {}
                for a, b in zip(EPOCHS, EPOCHS[1:]):
                    key = f"e{a}_to_e{b}"
                    transitions[key] = {f"{x}_to_{y}": sum(signs[a][i] == x and signs[b][i] == y for i in ids) for x in (-1, 0, 1) for y in (-1, 0, 1)}
                rescue = {i for i in ids if signs[EPOCHS[0]][i] == 1}
                final_rescue = {i for i in ids if signs[EPOCHS[-1]][i] == 1}
                temporal["seeds"][seed][arm][scope] = {"transitions": transitions, "e129_rescue_n": len(rescue), "e129_to_e199_rescue_retention": len(rescue & final_rescue) / len(rescue) if rescue else None}
    shifts["seeds"] = {s: {"train_cw": "unavailable", "validation_cw": state["seeds"][s]["validation_cw_features"]} for s in SEEDS}
    for name, payload in (("ert_rslad_i100_cw_heldout_state_decomposition_v1.json", state), ("ert_rslad_i100_cw_heldout_temporal_response_v1.json", temporal), ("ert_rslad_i100_cw_heldout_action_overlap_v1.json", overlaps), ("ert_rslad_i100_cw_heldout_feature_shift_v1.json", shifts)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seeds": list(SEEDS), "validation_epochs": list(EPOCHS), "e114": "unavailable", "output_dir": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
