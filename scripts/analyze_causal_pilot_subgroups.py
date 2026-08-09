#!/usr/bin/env python3
"""Subgroup and matched-random analysis for the completed Chen causal pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260809


def _mask(path: Path) -> set[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = value.get("selected_ids")
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise ValueError(f"invalid mask: {path}")
    return set(ids)


def _frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame["epoch"].nunique() != 1 or frame["sample_id"].duplicated().any():
        raise ValueError(f"pilot endpoint table is not one row per sample: {path}")
    return frame.set_index("sample_id").sort_index()


def _bootstrap_difference(values: pd.DataFrame, *, strata: str = "class_id") -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_group = [group for _, group in values.groupby(strata, sort=True)]
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = []
        for group in by_group:
            indexes = rng.integers(0, len(group), size=len(group))
            sampled.append(group.iloc[indexes])
        draw = pd.concat(sampled, axis=0)
        estimates[replicate] = float(draw.loc[draw["selected"], "effect"].mean()) - float(
            draw.loc[~draw["selected"], "effect"].mean()
        )
    point = float(values.loc[values["selected"], "effect"].mean()) - float(
        values.loc[~values["selected"], "effect"].mean()
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def _subgroup(frame: pd.DataFrame) -> pd.Series:
    teacher_clean = frame["teacher_clean_correct"].astype(bool)
    teacher_adv = frame["teacher_adversarial_correct"].astype(bool)
    student_clean = frame["clean_correct"].astype(bool)
    margin = frame["student_robust_margin_ema"].astype(float)
    margin_bin = pd.qcut(margin.rank(method="first"), 4, labels=["q0", "q1", "q2", "q3"])
    response = frame["teacher_clean_to_adversarial_margin_response"].astype(float)
    response_bin = pd.qcut(response.rank(method="first"), 4, labels=["r0", "r1", "r2", "r3"])
    return (
        pd.Series(
            np.select(
                [teacher_clean & teacher_adv, teacher_clean & ~teacher_adv, ~teacher_clean & teacher_adv],
                ["teacher_clean_and_adv_correct", "teacher_clean_correct_adv_wrong", "teacher_clean_wrong_adv_correct"],
                default="teacher_clean_and_adv_wrong",
            ),
            index=frame.index,
            name="teacher_state",
        )
        .astype(str)
        .str.cat(
            pd.Series(np.where(student_clean, "student_clean_correct", "student_clean_wrong"), index=frame.index),
            sep=";",
        )
        .str.cat(pd.Series(margin_bin.astype(str), index=frame.index), sep=";")
        .str.cat(pd.Series(response_bin.astype(str), index=frame.index), sep=";")
    )


def analyze(root: Path, *, human_review: Path | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "contract": "ffnr_causal_pilot_subgroups_v1",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "strata": "class_id"},
        "rows": [],
    }
    for seed in ("L2", "L4"):
        base = _frame(root / seed / "C79" / "sample-stats-train.parquet")
        treatment_paths = {
            "route_a": ("RA", "RAR", "route_a_selected-registered.json", "route_a_random-registered.json"),
            "route_b": ("RB", "RBR", "route_b_selected_q05-registered.json", "route_b_random_q05-registered.json"),
        }
        human = None
        if human_review is not None and human_review.is_file():
            raw = json.loads(human_review.read_text(encoding="utf-8"))
            human = {
                int(item["sample_id"]): str(item["label"])
                for item in raw.get("items", [])
                if isinstance(item, dict) and isinstance(item.get("sample_id"), int)
            }
        for route, (selected_arm, random_arm, selected_mask_name, random_mask_name) in treatment_paths.items():
            selected_ids = _mask(root.parent / "ffnr-causal-pilot-masks-v1" / seed / selected_mask_name)
            random_ids = _mask(root.parent / "ffnr-causal-pilot-masks-v1" / seed / random_mask_name)
            if len(selected_ids) != len(random_ids):
                raise ValueError(f"{seed} {route} masks are not count matched")
            for arm, ids, selected in ((selected_arm, selected_ids, True), (random_arm, random_ids, False)):
                treatment = _frame(root / seed / arm / "sample-stats-train.parquet")
                common = base.join(treatment, lsuffix="_base", rsuffix="_treatment").loc[sorted(ids)]
                common["effect"] = common["robust_correct_treatment"].astype(int) - common[
                    "robust_correct_base"
                ].astype(int)
                common["selected"] = selected
                common["teacher_state"] = _subgroup(
                    common.rename(
                        columns={
                            "teacher_clean_correct_base": "teacher_clean_correct",
                            "teacher_adversarial_correct_base": "teacher_adversarial_correct",
                            "clean_correct_base": "clean_correct",
                            "robust_correct_base": "robust_correct",
                            "student_robust_margin_ema_base": "student_robust_margin_ema",
                            "teacher_clean_to_adversarial_margin_response_base": (
                                "teacher_clean_to_adversarial_margin_response"
                            ),
                        }
                    )
                )
                if human is not None:
                    common["human_review"] = [human.get(int(sample_id)) for sample_id in common.index]
                output["rows"].append(
                    {
                        "seed": seed,
                        "route": route,
                        "arm": arm,
                        "selected": selected,
                        "n": len(common),
                        "mask_overlap_with_counterpart": len(selected_ids & random_ids),
                        "net_effect": float(common["effect"].mean()),
                        "rescue_rate": float(
                            ((common["robust_correct_base"] == 0) & (common["robust_correct_treatment"] == 1)).mean()
                        ),
                        "harm_rate": float(
                            ((common["robust_correct_base"] == 1) & (common["robust_correct_treatment"] == 0)).mean()
                        ),
                        "subgroups": common.groupby("teacher_state", observed=True)["effect"]
                        .agg(["count", "mean"])
                        .reset_index()
                        .to_dict(orient="records"),
                        "human_review": common["human_review"].value_counts(dropna=True).to_dict()
                        if human is not None
                        else {},
                        "class_effects": common.groupby("true_label_base", observed=True)["effect"]
                        .agg(["count", "mean"])
                        .reset_index()
                        .to_dict(orient="records"),
                        "robust_margin_ema_delta": float(
                            (
                                common["student_robust_margin_ema_treatment"] - common["student_robust_margin_ema_base"]
                            ).mean()
                        ),
                    }
                )
            # Bootstrap from the per-sample paired effects, recomputed below.
            selected_effects = _frame(root / seed / selected_arm / "sample-stats-train.parquet").loc[
                sorted(selected_ids)
            ]
            random_effects = _frame(root / seed / random_arm / "sample-stats-train.parquet").loc[sorted(random_ids)]
            selected_effects = base.join(selected_effects, lsuffix="_base", rsuffix="_treatment").loc[
                sorted(selected_ids)
            ]
            random_effects = base.join(random_effects, lsuffix="_base", rsuffix="_treatment").loc[sorted(random_ids)]
            combined = pd.concat(
                [selected_effects, random_effects], keys=[True, False], names=["selected", "sample_id"]
            ).reset_index()
            combined["effect"] = combined["robust_correct_treatment"].astype(int) - combined[
                "robust_correct_base"
            ].astype(int)
            combined["class_id"] = combined["true_label_base"]
            point, low, high = _bootstrap_difference(combined)
            output["rows"].append(
                {
                    "seed": seed,
                    "route": route,
                    "arm": "selected_minus_random",
                    "selected": None,
                    "n": len(combined),
                    "net_effect_difference": point,
                    "bootstrap_ci95": [low, high],
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, human_review=args.human_review)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summary = [
        "# Causal pilot subgroup analysis",
        "",
        f"Bootstrap: stratified by class, n={BOOTSTRAP_REPLICATES}, seed={BOOTSTRAP_SEED}.",
        "",
        "| seed | route | arm | n | net effect | rescue | harm |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        if "net_effect" in row:
            summary.append(
                f"| {row['seed']} | {row['route']} | {row['arm']} | {row['n']} | "
                f"{row['net_effect']:.4f} | {row['rescue_rate']:.4f} | {row['harm_rate']:.4f} |"
            )
        else:
            summary.append(
                f"| {row['seed']} | {row['route']} | selected−random | {row['n']} | "
                f"{row['net_effect_difference']:.4f} "
                f"({row['bootstrap_ci95'][0]:.4f},{row['bootstrap_ci95'][1]:.4f}) | — | — |"
            )
    args.markdown.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(args.json), "markdown": str(args.markdown)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
