#!/usr/bin/env python3
"""Read-only Teacher-margin x Clean-Wrong action-map analysis.

This intentionally consumes the completed broad-screen endpoint Parquets and
the hash-bound epoch-79 CE-PGD20/KL-PGD10 feature replays.  It never loads a
checkpoint or starts training/evaluation.  The analysis is descriptive: the
quantile boundaries are computed from pre-treatment features only and no
winner or new threshold is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BROAD = ROOT / ".cache/analysis/ert-clean-wrong-broad-v1"
DEFAULT_CE = ROOT / ".cache/analysis/ert-clean-wrong-subtypes-v4"
DEFAULT_KL = ROOT / ".cache/analysis/ert-clean-wrong-reliability-proxy-v1"
RUNS = ("L2", "L4")
ARMS = tuple(f"C{i}" for i in range(16))
CE_ATTACK_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
KL_ATTACK_SHA = "98194e2a6ee02add8c675b0df1146007f371ed1811ef34b9ef37d052997348bd"
EXPECTED_CKPT = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}
EXPECTED_MASK = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def feature_rows(meta_path: Path, run: str, *, kind: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    meta = read_json(meta_path)
    expected_contract = (
        "ert_clean_wrong_c0_ce_pgd20_features_v1" if kind == "CE20" else "ert_clean_wrong_c0_kl_pgd10_features_v1"
    )
    expected_attack = CE_ATTACK_SHA if kind == "CE20" else KL_ATTACK_SHA
    if meta.get("contract") != expected_contract or meta.get("feature_epoch") != 79:
        raise AnalysisError(f"{run}/{kind}: feature contract/epoch mismatch")
    if meta.get("checkpoint_sha256") != EXPECTED_CKPT[run]:
        raise AnalysisError(f"{run}/{kind}: parent checkpoint mismatch")
    if meta.get("mask_sha256") != EXPECTED_MASK[run]:
        raise AnalysisError(f"{run}/{kind}: Clean-Wrong mask mismatch")
    if meta.get("attack_identity_sha256") != expected_attack:
        raise AnalysisError(f"{run}/{kind}: attack identity mismatch")
    rows_path = Path(meta["rows_path"])
    if not rows_path.is_file() or sha256(rows_path) != meta.get("rows_sha256"):
        raise AnalysisError(f"{run}/{kind}: feature rows hash/path mismatch")
    rows = read_rows(rows_path)
    ids = [int(r["sample_id"]) for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != int(meta["selected_count"]):
        raise AnalysisError(f"{run}/{kind}: feature IDs are not unique/complete")
    by_id = {int(r["sample_id"]): r for r in rows}
    return {
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": sha256(meta_path),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": meta["rows_sha256"],
        "source_git_sha": meta.get("source_git_sha"),
        "attack_identity_sha256": meta["attack_identity_sha256"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "mask_sha256": meta["mask_sha256"],
        "selected_count": len(rows),
    }, by_id


def quantiles(values: dict[int, float]) -> tuple[dict[str, list[int]], dict[str, Any]]:
    ordered = sorted(values, key=lambda item: (values[item], item))
    n = len(ordered)
    base, rem = divmod(n, 5)
    groups: dict[str, list[int]] = {}
    boundaries: dict[str, Any] = {}
    start = 0
    for index in range(5):
        count = base + (1 if index < rem else 0)
        ids = ordered[start : start + count]
        name = f"Q{index + 1}"
        groups[name] = ids
        boundaries[name] = {
            "count": len(ids),
            "min_margin": values[ids[0]],
            "max_margin": values[ids[-1]],
            "first_id": ids[0],
            "last_id": ids[-1],
            "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        }
        start += count
    return groups, {
        "n": n,
        "rule": "sort by (pre_treatment_margin, sample_id), equal five groups",
        "boundaries": boundaries,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def metric(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    if not ids:
        raise AnalysisError("empty quantile")
    clean_b = [bool(base[i]["clean_correct"]) for i in ids]
    clean_t = [bool(treatment[i]["clean_correct"]) for i in ids]
    robust_b = [bool(base[i]["robust_correct"]) for i in ids]
    robust_t = [bool(treatment[i]["robust_correct"]) for i in ids]

    def binary(b: list[bool], t: list[bool]) -> dict[str, float]:
        rescue = sum((not x) and y for x, y in zip(b, t)) / len(b)
        harm = sum(x and (not y) for x, y in zip(b, t)) / len(b)
        net = rescue - harm
        delta = sum(float(y) - float(x) for x, y in zip(b, t)) / len(b)
        if not math.isclose(net, delta, rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisError("accuracy delta != rescue-harm")
        return {"accuracy_delta": delta, "rescue_rate": rescue, "harm_rate": harm, "net_rescue": net}

    clean = binary(clean_b, clean_t)
    robust = binary(robust_b, robust_t)
    clean_margin_delta = _mean(
        [float(treatment[i]["clean_probability_margin"]) - float(base[i]["clean_probability_margin"]) for i in ids]
    )
    robust_margin_delta = _mean(
        [
            float(treatment[i]["adversarial_probability_margin"]) - float(base[i]["adversarial_probability_margin"])
            for i in ids
        ]
    )
    return {
        "n": len(ids),
        "base_clean_accuracy": _mean([float(x) for x in clean_b]),
        "treatment_clean_accuracy": _mean([float(x) for x in clean_t]),
        "base_robust_accuracy": _mean([float(x) for x in robust_b]),
        "treatment_robust_accuracy": _mean([float(x) for x in robust_t]),
        "clean": {**clean, "margin_delta": clean_margin_delta},
        "robust": {**robust, "margin_delta": robust_margin_delta},
    }


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = _mean(ra), _mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return None if den == 0 else num / den


def pareto(effects: dict[str, dict[str, float]]) -> list[str]:
    result = []
    for arm, point in effects.items():
        dominated = False
        for other, candidate in effects.items():
            if other == arm:
                continue
            if (
                candidate["clean"] >= point["clean"]
                and candidate["robust"] >= point["robust"]
                and (candidate["clean"] > point["clean"] or candidate["robust"] > point["robust"])
            ):
                dominated = True
                break
        if not dominated:
            result.append(arm)
    return sorted(result)


def factorial(effects: dict[str, dict[str, float]], metric_name: str) -> dict[str, float]:
    def avg(names: list[str]) -> float:
        return _mean([effects[n][metric_name] for n in names])

    eps8 = ["C0", "C2", "C4", "C6"]
    eps4 = ["C1", "C3", "C5", "C7"]
    kd1 = ["C0", "C1", "C4", "C5"]
    kd05 = ["C2", "C3", "C6", "C7"]
    ce0 = ["C0", "C1", "C2", "C3"]
    ce075 = ["C4", "C5", "C6", "C7"]
    # Interaction terms are difference-in-differences, averaged over the third factor.
    eps_kd = 0.5 * (
        (effects["C3"][metric_name] - effects["C2"][metric_name])
        - (effects["C1"][metric_name] - effects["C0"][metric_name])
        + (effects["C7"][metric_name] - effects["C6"][metric_name])
        - (effects["C5"][metric_name] - effects["C4"][metric_name])
    )
    eps_ce = 0.5 * (
        (effects["C5"][metric_name] - effects["C1"][metric_name])
        - (effects["C4"][metric_name] - effects["C0"][metric_name])
        + (effects["C7"][metric_name] - effects["C3"][metric_name])
        - (effects["C6"][metric_name] - effects["C2"][metric_name])
    )
    kd_ce = 0.5 * (
        (effects["C6"][metric_name] - effects["C2"][metric_name])
        - (effects["C4"][metric_name] - effects["C0"][metric_name])
        + (effects["C7"][metric_name] - effects["C3"][metric_name])
        - (effects["C5"][metric_name] - effects["C1"][metric_name])
    )
    return {
        "epsilon_8_to_4": avg(eps4) - avg(eps8),
        "advkd_1_to_0.5": avg(kd05) - avg(kd1),
        "clean_ce_0_to_0.075": avg(ce075) - avg(ce0),
        "epsilon_x_advkd": eps_kd,
        "epsilon_x_clean_ce": eps_ce,
        "advkd_x_clean_ce": kd_ce,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs/experiments/ert_cw_margin_action_map_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/ERT_CW_MARGIN_ACTION_MAP.md")
    args = parser.parse_args()
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_margin_action_map_v1",
        "analysis_kind": "read_only_pre_treatment_teacher_margin_x_action_map",
        "endpoint_epoch": 84,
        "source_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "arms": {
            "C0": "BASE",
            "C1": "EPS4",
            "C2": "AdvKD x0.5",
            "C3": "EPS4 + AdvKD x0.5",
            "C4": "CleanCE 0.075",
            "C5": "EPS4 + CleanCE 0.075",
            "C6": "AdvKD x0.5 + CleanCE 0.075",
            "C7": "EPS4 + AdvKD x0.5 + CleanCE 0.075",
            "C8": "EPS2",
            "C9": "AdvKD x0.25",
            "C10": "CleanCE 0.15",
            "C11": "AdvKD x1.5",
            "C12": "MART-inspired boosted adversarial CE",
            "C13": "sample-adaptive AdvKD high-pressure",
            "C14": "Teacher-clean reliability gate",
            "C15": "IAD-inspired self-introspection",
        },
        "seeds": {},
        "held_out_subtype_transfer": {
            "status": "unavailable",
            "reason": "pre-treatment CE20/KL10 feature artifacts are train-only; no replay was started",
        },
        "selection": "Q1-Q5 are independent deterministic pre-treatment quantiles; no outcome or threshold tuning",
    }
    for run in RUNS:
        mask_path = ROOT / ".cache/analysis/ert-state-overlay-v1-review" / f"anchor79-fixed-masks-{run}.json"
        mask = read_json(mask_path)
        mask_record = mask["masks"]["student_clean_wrong"]
        selected = [int(x) for x in mask_record["selected_ids"]]
        if sha256(mask_path) != EXPECTED_MASK[run] or len(selected) != mask_record.get("selected_count", len(selected)):
            raise AnalysisError(f"{run}: registered mask hash/count mismatch")
        ce_meta, ce = feature_rows(DEFAULT_CE / run / "clean-wrong-feature-replay.json", run, kind="CE20")
        kl_meta, kl = feature_rows(DEFAULT_KL / run / "clean-wrong-kl10-feature-replay.json", run, kind="KL10")
        if set(ce) != set(selected) or set(kl) != set(selected) or set(ce) != set(kl):
            raise AnalysisError(f"{run}: feature/mask stable-ID universe mismatch")
        ce_q, ce_bounds = quantiles({i: float(ce[i]["teacher_adv_margin"]) for i in selected})
        kl_q, kl_bounds = quantiles({i: float(kl[i]["teacher_adv_margin"]) for i in selected})
        action_root = DEFAULT_BROAD / run
        base_path = action_root / "C0/endpoint/train/endpoint-sample-stats.parquet"
        base = {int(r["sample_id"]): r for r in read_rows(base_path)}
        if not set(selected) <= set(base):
            raise AnalysisError(f"{run}: endpoint does not contain all fixed Clean-Wrong IDs")
        for item in selected:
            if int(ce[item]["true_label"]) != int(kl[item]["true_label"]):
                raise AnalysisError(f"{run}: CE20/KL10 class mismatch for stable ID {item}")
            if int(ce[item]["true_label"]) != int(base[item]["true_label"]):
                raise AnalysisError(f"{run}: feature/endpoint class mismatch for stable ID {item}")
        lineage: dict[str, Any] = {
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": sha256(mask_path),
            "mask_count": len(selected),
            "ce20": ce_meta,
            "kl10": kl_meta,
            "endpoint": {},
        }
        effects: dict[str, Any] = {}
        maps: dict[str, Any] = {"CE20": {}, "KL10": {}}
        for arm in ARMS:
            endpoint_meta = read_json(action_root / arm / "endpoint/train/endpoint.json")
            rows_path = action_root / arm / "endpoint/train/endpoint-sample-stats.parquet"
            if (
                endpoint_meta.get("contract") != "ert_stage_a_common_ce_pgd20_endpoint_v1"
                or endpoint_meta.get("checkpoint_epoch") != 84
                or endpoint_meta.get("attack_identity_sha256") != CE_ATTACK_SHA
            ):
                raise AnalysisError(f"{run}/{arm}: endpoint contract/attack/epoch mismatch")
            if endpoint_meta.get("source_git_sha") != "cbe03a7b3be0b11fa1555b573c6f453a3d10f27b":
                raise AnalysisError(f"{run}/{arm}: broad-screen source SHA mismatch")
            if endpoint_meta.get("rows_sha256") != sha256(rows_path) or endpoint_meta.get("row_count") != 45000:
                raise AnalysisError(f"{run}/{arm}: endpoint rows hash/count mismatch")
            treatment = {int(r["sample_id"]): r for r in read_rows(rows_path)}
            if set(treatment) != set(base):
                raise AnalysisError(f"{run}/{arm}: endpoint stable-ID universe mismatch")
            lineage["endpoint"][arm] = {
                "meta_path": str((action_root / arm / "endpoint/train/endpoint.json").resolve()),
                "meta_sha256": sha256(action_root / arm / "endpoint/train/endpoint.json"),
                "rows_path": str(rows_path.resolve()),
                "rows_sha256": endpoint_meta["rows_sha256"],
                "checkpoint_sha256": endpoint_meta.get("checkpoint_sha256"),
                "source_git_sha": endpoint_meta.get("source_git_sha"),
                "attack_identity_sha256": endpoint_meta.get("attack_identity_sha256"),
            }
            arm_effects: dict[str, Any] = {}
            for kind, groups in (("CE20", ce_q), ("KL10", kl_q)):
                qmetrics = {q: metric(base, treatment, ids) for q, ids in groups.items()}
                arm_effects[kind] = qmetrics
                maps[kind][arm] = {
                    q: {
                        "clean_delta": qmetrics[q]["clean"]["accuracy_delta"],
                        "robust_delta": qmetrics[q]["robust"]["accuracy_delta"],
                    }
                    for q in groups
                }
            arm_effects["all_cw"] = metric(base, treatment, selected)
            effects[arm] = arm_effects
        factorials: dict[str, Any] = {}
        pareto_sets: dict[str, Any] = {"CE20": {}, "KL10": {}}
        for kind in ("CE20", "KL10"):
            factorials[kind] = {}
            for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                robust_effects = {
                    a: {
                        "clean": effects[a][kind][q]["clean"]["accuracy_delta"],
                        "robust": effects[a][kind][q]["robust"]["accuracy_delta"],
                    }
                    for a in ARMS
                }
                factorials[kind][q] = {
                    "clean": factorial(
                        {
                            a: {
                                "clean": effects[a][kind][q]["clean"]["accuracy_delta"],
                                "robust": effects[a][kind][q]["robust"]["accuracy_delta"],
                            }
                            for a in ARMS
                        },
                        "clean",
                    ),
                    "robust": factorial(robust_effects, "robust"),
                }
                pareto_sets[kind][q] = pareto(robust_effects)
        # Rank agreement compares all C0-C15 robust effects over the five pre-treatment bins.
        ce_flat = [maps["CE20"][a][q]["robust_delta"] for q in ("Q1", "Q2", "Q3", "Q4", "Q5") for a in ARMS]
        kl_flat = [maps["KL10"][a][q]["robust_delta"] for q in ("Q1", "Q2", "Q3", "Q4", "Q5") for a in ARMS]
        rank_agreement = {
            "flattened_robust_spearman": spearman(ce_flat, kl_flat),
            "per_quantile": {
                q: spearman(
                    [maps["CE20"][a][q]["robust_delta"] for a in ARMS],
                    [maps["KL10"][a][q]["robust_delta"] for a in ARMS],
                )
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
            },
            "pareto_jaccard": {
                q: (
                    len(set(pareto_sets["CE20"][q]) & set(pareto_sets["KL10"][q]))
                    / len(set(pareto_sets["CE20"][q]) | set(pareto_sets["KL10"][q]))
                )
                if (set(pareto_sets["CE20"][q]) | set(pareto_sets["KL10"][q]))
                else 1.0
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
            },
        }
        machine["seeds"][run] = {
            "quantiles": {"CE20": ce_bounds, "KL10": kl_bounds},
            "effects": effects,
            "maps": maps,
            "factorial": factorials,
            "pareto": pareto_sets,
            "rank_agreement": rank_agreement,
            "lineage": lineage,
        }
    machine["report_sha256"] = hashlib.sha256(
        json.dumps(machine, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong Teacher-Margin × Action Map",
        "",
        "Status: completed read-only analysis of the fixed epoch-79 Clean-Wrong cohort and epoch-84 CE-PGD20 endpoint.",
        "No training, threshold tuning, or new replay was performed.",
        "",
        "CE20 and KL10 Q1–Q5 are independent pre-treatment quantiles (sort by margin then stable ID).",
        "Accuracy deltas are always paired rescue minus harm; probability-margin deltas are reported separately.",
        "",
        "## Primary robust action maps (delta vs C0)",
        "",
    ]
    for run in RUNS:
        lines += [
            f"### {run} / CE-PGD20 Teacher margin",
            "",
            "| arm | Q1 | Q2 | Q3 | Q4 | Q5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            vals = [
                machine["seeds"][run]["maps"]["CE20"][arm][q]["robust_delta"] * 100
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
            ]
            lines.append(f"| {arm} | {' | '.join(f'{v:+.2f} pp' for v in vals)} |")
        lines += [
            "",
            f"### {run} / CE-PGD20 clean action map",
            "",
            "| arm | Q1 | Q2 | Q3 | Q4 | Q5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            vals = [
                machine["seeds"][run]["maps"]["CE20"][arm][q]["clean_delta"] * 100
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
            ]
            lines.append(f"| {arm} | {' | '.join(f'{v:+.2f} pp' for v in vals)} |")
        lines += [
            "",
            "### Pareto arms and CE20/KL10 agreement",
            "",
            f"- CE20 Pareto by Q: {machine['seeds'][run]['pareto']['CE20']}",
            f"- KL10 Pareto by Q: {machine['seeds'][run]['pareto']['KL10']}",
            "- flattened robust ranking Spearman (CE20 vs KL10): "
            f"{machine['seeds'][run]['rank_agreement']['flattened_robust_spearman']}",
            f"- per-Q robust ranking Spearman: {machine['seeds'][run]['rank_agreement']['per_quantile']}",
            "",
        ]
    lines += [
        "## Requested comparisons",
        "",
        "- CleanCE dose response is available for C0/C4/C10 in the machine artifact; it is descriptive.",
        "  No new coefficient was selected.",
        "- AdvKD pressure dose response is available for C9/C2/C0/C11.",
        "- Attack-budget comparison is available for C8/C1/C0.",
        "- Robust-side comparison is available for C10/C11/C12/C13/C0. C12 remains MART-inspired BCE, not plain AdvCE.",
        "- C0–C7 factorial main effects and two-way difference-in-differences are recorded per seed,",
        "  margin domain, and Q.",
        "- Held-out subtype transfer is not reported: pre-treatment CE20/KL10 artifacts are train-only.",
        "  No GPU replay was started.",
        "",
        "## Interpretation guardrails",
        "",
        "This is a subtype/action heterogeneity map, not a validated router. Q5 is not an optimal threshold.",
        "Direct train effects cannot be promoted to held-out effects.",
        "The previous four-arm sign-gate failure is not reclassified as solved.",
        "No new training or automatic winner promotion was performed.",
        "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
