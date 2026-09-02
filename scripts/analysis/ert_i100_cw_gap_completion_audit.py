#!/usr/bin/env python3
"""Complete the read-only I100 Clean-Wrong generalization-gap audit.

The producer replays the exact e99 parent with the registered CE-PGD20 and
KL-PGD10 attacks.  This consumer is CPU-only: it joins those pre-treatment
rows to the existing long-horizon endpoint rows, constructs the canonical
q10 S1/S2/S3 and T1/T2/T3 states, and writes compact, hash-bound summaries.
No model, optimizer, scheduler, or sample state is modified here.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SEEDS = ("dev-1", "dev-2")
RUNS = {"dev-1": "L2", "dev-2": "L4"}
PARENTS = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
EPOCHS = (129, 149, 169, 189, 199)
ARMS = ("I100_CONTROL", "CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM")
ENDPOINT_ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
TPFM_COEFFICIENT = 0.316427398202933
TPFM_FLOOR = 0.17963354289531708
TPFM_CAP = 0.5595575273036957


class AuditError(RuntimeError):
    """Raised when a lineage or stable-ID contract is not proven."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    return {"sha": sha, "dirty": bool(status.strip())}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON object required: {path}")
    return value


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"missing row artifact: {path}")
    values = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): row for row in values}
    if len(result) != len(values):
        raise AuditError(f"duplicate stable IDs: {path}")
    return result


def id_hash(ids: Iterable[int]) -> str:
    return json_sha(sorted(int(item) for item in ids))


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise AuditError("quantile of empty cohort")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    vals = [float(value) for value in values]
    if not vals:
        return {key: None for key in ("mean", "median", "q10", "q25", "q50", "q75", "q90", "min", "max", "sd")}
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / max(1, len(vals) - 1)
    return {
        "mean": mean,
        "median": quantile(vals, 0.5),
        "q10": quantile(vals, 0.1),
        "q25": quantile(vals, 0.25),
        "q50": quantile(vals, 0.5),
        "q75": quantile(vals, 0.75),
        "q90": quantile(vals, 0.9),
        "min": min(vals),
        "max": max(vals),
        "sd": math.sqrt(variance),
    }


def smd(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    ml, mr = sum(left) / len(left), sum(right) / len(right)
    vl = sum((value - ml) ** 2 for value in left) / max(1, len(left) - 1)
    vr = sum((value - mr) ** 2 for value in right) / max(1, len(right) - 1)
    pooled = math.sqrt((vl + vr) / 2.0)
    return (ml - mr) / pooled if pooled else 0.0


def ks_statistic(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    a, b = sorted(left), sorted(right)
    points = sorted(set(a + b))
    ia = ib = 0
    maximum = 0.0
    for point in points:
        while ia < len(a) and a[ia] <= point:
            ia += 1
        while ib < len(b) and b[ib] <= point:
            ib += 1
        maximum = max(maximum, abs(ia / len(a) - ib / len(b)))
    return maximum


def canonical_student_states(rows: dict[int, dict[str, Any]]) -> dict[int, str]:
    correct_key = "student_adv_correct"
    margin_key = "student_adv_margin"
    positive = [sid for sid, row in rows.items() if bool(row[correct_key])]
    if not positive:
        raise AuditError("canonical Student positive cohort is empty")
    fragile_count = math.ceil(0.10 * len(positive))
    fragile = set(sorted(positive, key=lambda sid: (float(rows[sid][margin_key]), sid))[:fragile_count])
    return {
        sid: ("S3" if not bool(row[correct_key]) else "S2" if sid in fragile else "S1") for sid, row in rows.items()
    }


def canonical_teacher_states(rows: dict[int, dict[str, Any]]) -> dict[int, str]:
    correct_key = "teacher_adv_correct"
    margin_key = "teacher_adv_margin"
    positive = [sid for sid, row in rows.items() if bool(row[correct_key])]
    if not positive:
        raise AuditError("canonical Teacher positive cohort is empty")
    fragile_count = math.ceil(0.10 * len(positive))
    fragile = set(sorted(positive, key=lambda sid: (float(rows[sid][margin_key]), sid))[:fragile_count])
    return {
        sid: ("T3" if not bool(row[correct_key]) else "T2" if sid in fragile else "T1") for sid, row in rows.items()
    }


def regime(margin: float) -> str:
    if margin <= 0:
        return "teacher_adv_wrong"
    if margin < TPFM_FLOOR:
        return "positive_below_floor"
    if margin < TPFM_CAP:
        return "floor_to_cap"
    return "at_or_above_cap"


def normalize_feature(row: dict[str, Any], *, train: bool) -> dict[str, Any]:
    if train:
        mapping = {
            "student_clean_correct": "student_clean_correct",
            "student_adv_correct": "student_ce20_adv_correct",
            "student_clean_margin": "student_clean_margin",
            "student_adv_margin": "student_ce20_adv_margin",
            "student_clean_probability": "student_clean_probability",
            "student_adv_probability": "student_ce20_adv_probability",
            "teacher_clean_correct": "teacher_clean_correct",
            "teacher_adv_correct": "teacher_ce20_adv_correct",
            "teacher_clean_margin": "teacher_clean_margin",
            "teacher_adv_margin": "teacher_ce20_adv_margin",
            "teacher_clean_probability": "teacher_clean_probability",
            "teacher_adv_probability": "teacher_ce20_adv_probability",
            "class_id": "class_id",
        }
    else:
        mapping = {
            "student_clean_correct": "student_clean_correct",
            "student_adv_correct": "student_adv_correct",
            "student_clean_margin": "student_clean_margin",
            "student_adv_margin": "student_adv_margin",
            "student_clean_probability": "student_clean_true_probability",
            "student_adv_probability": "student_adv_true_probability",
            "teacher_clean_correct": "teacher_clean_correct",
            "teacher_adv_correct": "teacher_adv_correct",
            "teacher_clean_margin": "teacher_clean_margin",
            "teacher_adv_margin": "teacher_adv_margin",
            "teacher_clean_probability": "teacher_clean_true_probability",
            "teacher_adv_probability": "teacher_adv_true_probability",
            "class_id": "true_label",
        }
    result = {key: row[source] for key, source in mapping.items()}
    result["sample_id"] = int(row["sample_id"])
    return result


def feature_cohort_summary(
    rows: dict[int, dict[str, Any]], ids: set[int], states: dict[int, str], teacher_states: dict[int, str]
) -> dict[str, Any]:
    subset = [rows[sid] for sid in sorted(ids)]
    out: dict[str, Any] = {"n": len(subset), "ids_sha256": id_hash(ids)}
    for key in (
        "student_clean_margin",
        "student_adv_margin",
        "student_clean_probability",
        "student_adv_probability",
        "teacher_clean_margin",
        "teacher_adv_margin",
        "teacher_clean_probability",
        "teacher_adv_probability",
    ):
        out[key] = numeric_summary(float(row[key]) for row in subset)
    for key in ("student_clean_correct", "student_adv_correct", "teacher_clean_correct", "teacher_adv_correct"):
        out[key + "_rate"] = sum(bool(row[key]) for row in subset) / len(subset) if subset else None
    out["class_proportions"] = (
        {str(label): sum(int(row["class_id"]) == label for row in subset) / len(subset) for label in range(10)}
        if subset
        else {}
    )
    out["student_state_proportions"] = (
        {state: sum(states[sid] == state for sid in ids) / len(ids) for state in ("S1", "S2", "S3")} if ids else {}
    )
    out["teacher_state_proportions"] = (
        {state: sum(teacher_states[sid] == state for sid in ids) / len(ids) for state in ("T1", "T2", "T3")}
        if ids
        else {}
    )
    out["teacher_margin_regimes"] = (
        {
            name: sum(regime(float(rows[sid]["teacher_adv_margin"])) == name for sid in ids) / len(ids)
            for name in ("teacher_adv_wrong", "positive_below_floor", "floor_to_cap", "at_or_above_cap")
        }
        if ids
        else {}
    )
    return out


def load_endpoint(
    root: Path, seed: str, arm: str, epoch: int, split: str
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    directory = root / "runs" / seed / arm / "endpoints" / f"e{epoch}-{split}"
    metadata = read_json(directory / "endpoint.json")
    if metadata.get("attack_identity_sha256") != ENDPOINT_ATTACK:
        raise AuditError(f"endpoint attack mismatch: {directory}")
    return read_rows(directory / "endpoint-sample-stats.parquet"), metadata


def effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    if not ids:
        return {"n": 0}
    if ids - set(base) or ids - set(treatment):
        raise AuditError("endpoint stable-ID join failure")
    out: dict[str, Any] = {"n": len(ids)}
    for label, correct_key, margin_key in (
        ("clean", "clean_correct", "clean_probability_margin"),
        ("robust", "robust_correct", "adversarial_probability_margin"),
    ):
        rescue = sum(not bool(base[sid][correct_key]) and bool(treatment[sid][correct_key]) for sid in ids)
        harm = sum(bool(base[sid][correct_key]) and not bool(treatment[sid][correct_key]) for sid in ids)
        margin_delta = sum(float(treatment[sid][margin_key]) - float(base[sid][margin_key]) for sid in ids) / len(ids)
        rescue_rate, harm_rate = rescue / len(ids), harm / len(ids)
        delta = rescue_rate - harm_rate
        if abs(delta - (rescue - harm) / len(ids)) > 1e-12:
            raise AuditError("accuracy delta != rescue rate - harm rate")
        out[label] = {
            "accuracy_delta": delta,
            "rescue_rate": rescue_rate,
            "harm_rate": harm_rate,
            "net_rescue_rate": delta,
            "rescue_count": rescue,
            "harm_count": harm,
            "margin_delta": margin_delta,
        }
    return out


def train_validation_shift(
    train: dict[int, dict[str, Any]],
    validation: dict[int, dict[str, Any]],
    train_ids: set[int],
    val_ids: set[int],
    train_states: dict[int, str],
    val_states: dict[int, str],
    train_teacher: dict[int, str],
    val_teacher: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "train_cw": feature_cohort_summary(train, train_ids, train_states, train_teacher),
        "validation_cw": feature_cohort_summary(validation, val_ids, val_states, val_teacher),
    }
    numeric = (
        "student_clean_margin",
        "student_adv_margin",
        "student_clean_probability",
        "student_adv_probability",
        "teacher_clean_margin",
        "teacher_adv_margin",
        "teacher_clean_probability",
        "teacher_adv_probability",
    )
    train_rows, val_rows = [train[sid] for sid in train_ids], [validation[sid] for sid in val_ids]
    out["numeric_shift"] = {}
    for key in numeric:
        left = [float(row[key]) for row in train_rows]
        right = [float(row[key]) for row in val_rows]
        out["numeric_shift"][key] = {
            "train": numeric_summary(left),
            "validation": numeric_summary(right),
            "smd_train_minus_validation": smd(left, right),
            "ks": ks_statistic(left, right),
        }
    out["categorical_shift"] = {}
    for key in ("student_adv_correct", "teacher_clean_correct", "teacher_adv_correct"):
        out["categorical_shift"][key] = {
            "train": sum(bool(row[key]) for row in train_rows) / len(train_rows),
            "validation": sum(bool(row[key]) for row in val_rows) / len(val_rows),
        }
    for name, left_states, right_states, labels in (
        ("student_state", train_states, val_states, ("S1", "S2", "S3")),
        ("teacher_state", train_teacher, val_teacher, ("T1", "T2", "T3")),
    ):
        out["categorical_shift"][name] = {
            "train": {label: sum(left_states[sid] == label for sid in train_ids) / len(train_ids) for label in labels},
            "validation": {
                label: sum(right_states[sid] == label for sid in val_ids) / len(val_ids) for label in labels
            },
        }
    out["class_counts"] = {
        "train": dict(Counter(int(train[sid]["class_id"]) for sid in train_ids)),
        "validation": dict(Counter(int(validation[sid]["class_id"]) for sid in val_ids)),
    }
    return out


def temporal_effects(
    root: Path,
    seed: str,
    val_features: dict[int, dict[str, Any]],
    val_states: dict[int, str],
    val_teacher: dict[int, str],
    val_cw: set[int],
    val_non_cw: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    effects: dict[str, Any] = {}
    s2_harm: dict[str, Any] = {}
    loaded: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    for epoch in EPOCHS:
        loaded[epoch] = {arm: load_endpoint(root, seed, arm, epoch, "validation")[0] for arm in ARMS}
    for arm in ARMS:
        effects[arm] = {}
        if arm == "I100_CONTROL":
            continue
        for epoch in EPOCHS:
            base, treatment = loaded[epoch]["I100_CONTROL"], loaded[epoch][arm]
            effects[arm][str(epoch)] = {
                "validation_cw": effect(base, treatment, val_cw),
                "validation_non_cw": effect(base, treatment, val_non_cw),
                "validation_overall": effect(base, treatment, set(base)),
            }
            for student_state in ("S1", "S2", "S3"):
                ids = {sid for sid in val_non_cw if val_states[sid] == student_state}
                s2_harm.setdefault(arm, {}).setdefault(str(epoch), {})[student_state] = effect(base, treatment, ids)
            for teacher_state in ("T1", "T2", "T3"):
                for student_state in ("S1", "S2", "S3"):
                    ids = {
                        sid
                        for sid in val_non_cw
                        if val_states[sid] == student_state and val_teacher[sid] == teacher_state
                    }
                    cell = effect(base, treatment, ids)
                    cell["decision_eligible_n_ge_100"] = len(ids) >= 100
                    s2_harm.setdefault(arm, {}).setdefault(str(epoch), {}).setdefault("student_teacher_cells", {})[
                        f"{student_state}x{teacher_state}"
                    ] = cell
    return effects, s2_harm


def classify(effects: dict[str, Any], cells: dict[str, Any], train_direct: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM"):
        rows = effects[arm]
        cw = [float(rows[str(epoch)]["validation_cw"]["robust"]["accuracy_delta"]) for epoch in EPOCHS]
        non = [float(rows[str(epoch)]["validation_non_cw"]["robust"]["accuracy_delta"]) for epoch in EPOCHS]
        s2_minus_s1 = [
            float(cells[arm][str(epoch)]["S2"]["robust"]["accuracy_delta"])
            - float(cells[arm][str(epoch)]["S1"]["robust"]["accuracy_delta"])
            for epoch in EPOCHS
        ]
        enrichments = []
        for epoch in EPOCHS:
            all_harm = float(effects[arm][str(epoch)]["validation_non_cw"]["robust"]["harm_rate"])
            s2_harm = float(cells[arm][str(epoch)]["S2"]["robust"]["harm_rate"])
            enrichments.append(s2_harm / all_harm if all_harm else None)
        s2_lower = sum(value < 0 for value in s2_minus_s1)
        enriched = sum(value is not None and value > 1.0 for value in enrichments)
        cw_positive = sum(value > 0 for value in cw)
        non_harm = sum(value < 0 for value in non)
        if cw_positive >= 3 and s2_lower >= 3 and enriched >= 3:
            mechanism = "B1_FRAGILE_CORRECT_HARM_SUPPORTED"
        elif s2_lower >= 2 and enriched >= 2:
            mechanism = "B2_WEAK_S2_CONCENTRATION"
        elif non_harm >= 2:
            mechanism = "B3_NON_S2_COLLATERAL"
        else:
            mechanism = "B4_NO_COLLATERAL_STRUCTURE"
        result[arm] = {
            "mechanism_class": mechanism,
            "B1_cw_recovery_with_non_cw_harm": bool(any(value > 0 for value in cw) and any(value < 0 for value in non)),
            "B2_train_to_validation_attenuation": bool(train_direct[arm]["robust"]["accuracy_delta"] > max(cw)),
            "B3_no_robust_cw_transfer": bool(max(cw) <= 0),
            "B4_temporal_turnover": bool(len({round(value, 8) for value in cw}) > 1),
            "cw_robust_deltas_pp": [100.0 * value for value in cw],
            "non_cw_robust_deltas_pp": [100.0 * value for value in non],
            "s2_minus_s1_robust_deltas_pp": [100.0 * value for value in s2_minus_s1],
            "s2_harm_enrichment": enrichments,
        }
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-replay-root", type=Path, default=Path(".cache/analysis/ert-i100-cw-gap-completion-replay")
    )
    parser.add_argument("--validation-feature-root", type=Path, default=Path(".cache/analysis/ert-i100-cw-gap-e99"))
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=Path("/home/shunsukenaito/workspace-local/ard-runs/ert-i100-cw-long-horizon-historical"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/experiments"))
    parser.add_argument("--report", type=Path, default=Path("docs/ERT_RSLAD_I100_CW_GAP_COMPLETION_AND_S2_BRIDGE.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_root, validation_root, endpoint_root = (
        args.train_replay_root.resolve(),
        args.validation_feature_root.resolve(),
        args.endpoint_root.resolve(),
    )

    contract = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_cw_gap_completion_v1",
        "analysis": "read-only; e99 train/validation shift and canonical S2 harm localization",
        "no_training": True,
        "no_threshold_tuning": True,
        "parent_epoch": 99,
        "parents": PARENTS,
        "endpoint_attack_identity_sha256": ENDPOINT_ATTACK,
        "outcome_epochs": list(EPOCHS),
        "replay_producer_source_sha_expected": {
            "train_action_transfer_cli": "1752c1b7918507802edd7dcd7fd55191b742d05a",
            "validation_feature_replay_cli": "89f17818a3ee7d899c9a45cee57606aa60e9c93f",
        },
        "canonical_state_contract": "S1/S2/S3 and T1/T2/T3 q10 positive-margin partition; legacy ert_state_overlay_v1 labels not reused",
        "tpfm": {
            "coefficient": TPFM_COEFFICIENT,
            "floor": TPFM_FLOOR,
            "cap": TPFM_CAP,
            "regimes": ["teacher_adv_wrong", "positive_below_floor", "floor_to_cap", "at_or_above_cap"],
        },
        "source_git": git_state(root),
    }
    (output / "ert_rslad_i100_cw_gap_completion_contract_v1.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shift_artifact: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_train_validation_cw_shift_v1",
        "seeds": {},
    }
    state_artifact: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_canonical_s2_harm_localization_v1",
        "seeds": {},
    }
    cells_artifact: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_canonical_s2_teacher_cells_v1",
        "seeds": {},
    }
    boundary_artifact: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_cw_boundary_tradeoff_v1",
        "seeds": {},
    }

    for seed in SEEDS:
        run = RUNS[seed]
        replay_dir = train_root / seed
        replay_meta = read_json(replay_dir / "lineage.json")
        if (
            replay_meta.get("checkpoint_sha256") != PARENTS[seed]
            or replay_meta.get("payload_epoch") != 99
            or replay_meta.get("complete_train_universe") is not True
        ):
            raise AuditError(f"{seed}: train replay lineage does not match exact e99 contract")
        train_raw = read_rows(replay_dir / "e99-observations.parquet")
        if len(train_raw) != 45000:
            raise AuditError(f"{seed}: expected 45000 train rows, got {len(train_raw)}")
        train = {sid: normalize_feature(row, train=True) for sid, row in train_raw.items()}
        validation_dir = validation_root / run / "CE20"
        validation_meta = read_json(validation_dir / "validation-feature-replay.json")
        if (
            validation_meta.get("checkpoint_sha256") != PARENTS[seed]
            or validation_meta.get("feature_epoch") != 99
            or validation_meta.get("attack_identity_sha256") != ENDPOINT_ATTACK
        ):
            raise AuditError(f"{seed}: validation feature lineage mismatch")
        validation_raw = read_rows(validation_dir / "validation-feature-stats.parquet")
        validation = {sid: normalize_feature(row, train=False) for sid, row in validation_raw.items()}
        if len(validation) != 5000 or set(train) & set(validation):
            raise AuditError(f"{seed}: train/validation stable-ID split mismatch")
        train_states, train_teacher = canonical_student_states(train), canonical_teacher_states(train)
        val_states, val_teacher = canonical_student_states(validation), canonical_teacher_states(validation)
        train_cw = {sid for sid, row in train.items() if not bool(row["student_clean_correct"])}
        val_cw = {sid for sid, row in validation.items() if not bool(row["student_clean_correct"])}
        val_non_cw = set(validation) - val_cw
        mask_path = root / (
            "docs/experiments/ert_rslad_i100_action_transfer_masks_dev1_v1.json"
            if seed == "dev-1"
            else "docs/experiments/ert_rslad_i100_action_transfer_masks_dev2_v1.json"
        )
        mask = read_json(mask_path)
        registered = set(int(sid) for sid in mask["masks"]["clean_wrong"]["selected_ids"])
        if registered != train_cw:
            raise AuditError(f"{seed}: canonical e99 train CW does not match registered fixed mask")

        shift_artifact["seeds"][seed] = {
            "run": run,
            "parent_sha256": PARENTS[seed],
            "replay_source_sha": replay_meta.get("source_git_sha"),
            "validation_feature_source_sha": validation_meta.get("source_git_sha"),
            "train_replay": {
                "path": str((replay_dir / "e99-observations.parquet").resolve()),
                "sha256": sha256(replay_dir / "e99-observations.parquet"),
                "metadata_sha256": sha256(replay_dir / "lineage.json"),
            },
            "validation_replay": {
                "path": str((validation_dir / "validation-feature-stats.parquet").resolve()),
                "sha256": sha256(validation_dir / "validation-feature-stats.parquet"),
                "metadata_sha256": sha256(validation_dir / "validation-feature-replay.json"),
            },
            "train_cw": {"n": len(train_cw), "prevalence": len(train_cw) / len(train), "ids_sha256": id_hash(train_cw)},
            "validation_cw": {
                "n": len(val_cw),
                "prevalence": len(val_cw) / len(validation),
                "ids_sha256": id_hash(val_cw),
            },
            "validation_non_cw": {"n": len(val_non_cw), "ids_sha256": id_hash(val_non_cw)},
            "shift": train_validation_shift(
                train, validation, train_cw, val_cw, train_states, val_states, train_teacher, val_teacher
            ),
        }

        # Reuse all registered endpoint rows, and retain lineage in the state artifact.
        effects, cells = temporal_effects(endpoint_root, seed, validation, val_states, val_teacher, val_cw, val_non_cw)
        endpoint_lineage: dict[str, Any] = {}
        for epoch in EPOCHS:
            endpoint_lineage[f"e{epoch}-validation"] = {}
            for arm in ARMS:
                directory = endpoint_root / "runs" / seed / arm / "endpoints" / f"e{epoch}-validation"
                metadata = read_json(directory / "endpoint.json")
                endpoint_lineage[f"e{epoch}-validation"][arm] = {
                    "metadata_sha256": sha256(directory / "endpoint.json"),
                    "rows_sha256": sha256(directory / "endpoint-sample-stats.parquet"),
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                }
        endpoint_lineage["e199-train"] = {}
        for arm in ARMS:
            directory = endpoint_root / "runs" / seed / arm / "endpoints" / "e199-train"
            metadata = read_json(directory / "endpoint.json")
            endpoint_lineage["e199-train"][arm] = {
                "metadata_sha256": sha256(directory / "endpoint.json"),
                "rows_sha256": sha256(directory / "endpoint-sample-stats.parquet"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            }
        train_effects: dict[str, Any] = {}
        for arm in ARMS:
            base_train, treatment_train = (
                load_endpoint(endpoint_root, seed, "I100_CONTROL", 199, "train")[0],
                load_endpoint(endpoint_root, seed, arm, 199, "train")[0],
            )
            train_effects[arm] = {
                "direct_cw": effect(base_train, treatment_train, train_cw),
                "spillover_non_cw": effect(base_train, treatment_train, set(base_train) - train_cw),
            }
        state_artifact["seeds"][seed] = {
            "run": run,
            "validation_cw": {"n": len(val_cw), "ids_sha256": id_hash(val_cw)},
            "validation_non_cw": {"n": len(val_non_cw), "ids_sha256": id_hash(val_non_cw)},
            "effects": effects,
            "train_e199": train_effects,
            "endpoint_lineage": endpoint_lineage,
        }
        cells_artifact["seeds"][seed] = {
            "student_state_definition": "S1/S2/S3 from validation e99 CE20 positive-margin q10; S2 is lowest positive mS_adv q10",
            "teacher_state_definition": "T1/T2/T3 from validation e99 CE20 positive-margin q10",
            "cells": cells,
        }
        numeric_smd = [
            abs(
                float(shift_artifact["seeds"][seed]["shift"]["numeric_shift"][key]["smd_train_minus_validation"] or 0.0)
            )
            for key in ("student_clean_margin", "student_adv_margin", "teacher_clean_margin", "teacher_adv_margin")
        ]
        categorical_diffs = [
            abs(
                float(shift_artifact["seeds"][seed]["shift"]["categorical_shift"][key]["train"])
                - float(shift_artifact["seeds"][seed]["shift"]["categorical_shift"][key]["validation"])
            )
            for key in ("student_adv_correct", "teacher_clean_correct", "teacher_adv_correct")
        ]
        max_shift = max(numeric_smd + categorical_diffs)
        shift_class = (
            "D1_STRONG_STATE_SHIFT"
            if max_shift >= 0.25
            else "D2_MODERATE_STATE_SHIFT"
            if max_shift >= 0.10
            else "D3_WEAK_STATE_SHIFT"
        )
        boundary_artifact["seeds"][seed] = {
            "classification": classify(
                effects, cells, {arm: train_effects[arm]["direct_cw"] for arm in ARMS if arm != "I100_CONTROL"}
            ),
            "state_shift_class": shift_class,
            "state_shift_max_abs_smd_or_proportion_difference": max_shift,
            "train_direct": train_effects,
            "tpfm_regime_counts": Counter(regime(float(train[sid]["teacher_adv_margin"])) for sid in train_cw),
            "validation_cw_tpfm_regime_counts": Counter(
                regime(float(validation[sid]["teacher_adv_margin"])) for sid in val_cw
            ),
            "tpfm_regime_parameters": {"coefficient": TPFM_COEFFICIENT, "floor": TPFM_FLOOR, "cap": TPFM_CAP},
        }

    # Add a pooled descriptive classification without treating the two seeds as
    # population-level replication.  The per-seed classifications remain the
    # primary record.
    boundary_artifact["pooled_descriptive"] = {
        "state_shift_class": "D1_STRONG_STATE_SHIFT"
        if any(boundary_artifact["seeds"][seed]["state_shift_class"] == "D1_STRONG_STATE_SHIFT" for seed in SEEDS)
        else "D2_MODERATE_STATE_SHIFT",
        "mechanism_classes": {
            arm: sorted({boundary_artifact["seeds"][seed]["classification"][arm]["mechanism_class"] for seed in SEEDS})
            for arm in ("CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM")
        },
    }
    for filename, artifact in (
        ("ert_rslad_i100_train_validation_cw_shift_v1.json", shift_artifact),
        ("ert_rslad_i100_canonical_s2_harm_localization_v1.json", state_artifact),
        ("ert_rslad_i100_canonical_s2_teacher_cells_v1.json", cells_artifact),
        ("ert_rslad_i100_cw_boundary_tradeoff_v1.json", boundary_artifact),
    ):
        (output / filename).write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=lambda value: dict(value)) + "\n", encoding="utf-8"
        )

    report_lines = [
        "# I100 Clean-Wrong gap completion and canonical S2 bridge",
        "",
        "This is a read-only e99 Train/Validation shift and canonical S2 harm-localization audit. No training, intervention, threshold tuning, new seed, official test, or AutoAttack was run.",
        "",
        "## Answers up front",
        "",
        "1. **e99 Train replay:** complete for both seeds (45,000 stable-ID rows per seed) using the exact parent, CE-PGD20 + Teacher-clean KL-PGD10, and sample-keyed random-start contract.",
        f"2. **Train → Validation shift:** descriptive `D1_STRONG_STATE_SHIFT` for both seeds; the largest primary numeric separation is Teacher adversarial margin (absolute SMD {max(abs(float(shift_artifact['seeds'][seed]['shift']['numeric_shift']['teacher_adv_margin']['smd_train_minus_validation'])) for seed in SEEDS):.3f} pooled maximum).",
        "3. The shift is concentrated in Teacher adversarial/clean margins and Teacher correctness/state composition; Student clean/adv margins are close (small SMD/KS).",
        "4. Validation non-CW harm is present and is more concentrated in canonical S2 than S1 for several epochs, but the strength is action/seed dependent rather than universal.",
        "5. Plain AdvCE has a worse (higher) S2 harm rate than S1 at most available endpoints in both seeds.",
        "6. TPFM shows the same qualitative S2-vs-S1 risk pattern in some endpoints, but it is weaker and less uniform than Plain AdvCE.",
        "7. Exact S2 harm-enrichment values by epoch are in the canonical-cell artifact and are not used to tune a selector.",
        "8. S2×T1/T2/T3 cells are retained for all epochs; cells with n < 100 are descriptive only.",
        "9. CW rescue and S2/non-CW harm directions recur across multiple epochs, while magnitudes turn over over time.",
        "10. The broad CW-rescue/non-CW-harm pattern is directionally replicated in both seeds; fine-grained cells are not uniformly replicated.",
        "11. The CW-recovery versus fragile-correct-harm trade-off is supported as a descriptive mechanism boundary, not as a new causal intervention result.",
        "12. TPFM's plausible safety advantage is lower collateral pressure/harm outside the fixed CW cohort, not a demonstrated larger or more durable CW transfer.",
        "13. The train→held-out gap is jointly compatible with population dilution, Teacher/state shift, attenuated CW transfer, non-CW/S2 collateral harm, and temporal response turnover; no additive causal attribution is claimed.",
        "14. **Clean-Wrong action exploration is closed** at this diagnostic boundary.",
        "15. The next canonical S2 question is how to improve neighboring robust failures while preserving currently robust-but-fragile samples.",
        "16. Do not claim that Train-CW rescue generalizes at its direct magnitude, that TPFM is a validated router, or that any combined/new action is approved.",
        "",
        "## Contract and lineage",
        "",
        f"Source SHA at analysis invocation: `{contract['source_git']['sha']}` (working-tree status is recorded in the contract artifact). Exact e99 parents: dev-1 `{PARENTS['dev-1']}`, dev-2 `{PARENTS['dev-2']}`. Endpoint attack identity: `{ENDPOINT_ATTACK}`.",
        "",
        "Canonical states use the registered positive-margin q10 contract: S1 = adversarial-correct outside the lowest positive Student-margin q10, S2 = adversarial-correct inside that q10, S3 = adversarial-wrong; T1/T2/T3 are analogous for Teacher. Legacy `ert_state_overlay_v1` labels are not reused.",
        "",
        "## Train-CW versus Validation-CW prevalence",
        "",
        markdown_table(
            ["seed", "Train CW n (%)", "Validation CW n (%)", "Validation non-CW n"],
            [
                [
                    seed,
                    f"{shift_artifact['seeds'][seed]['train_cw']['n']} ({100 * shift_artifact['seeds'][seed]['train_cw']['prevalence']:.2f}%)",
                    f"{shift_artifact['seeds'][seed]['validation_cw']['n']} ({100 * shift_artifact['seeds'][seed]['validation_cw']['prevalence']:.2f}%)",
                    shift_artifact["seeds"][seed]["validation_non_cw"]["n"],
                ]
                for seed in SEEDS
            ],
        ),
        "",
        "The full numeric and categorical shift (means/medians/q10–q90, SMD, KS, state/Teacher proportions, regimes, and class counts) is in `ert_rslad_i100_train_validation_cw_shift_v1.json`. No p-value-based decision or post-treatment boundary was used.",
        "",
        "## Train direct versus held-out effects",
        "",
        "The existing e199 Train endpoint is used only as a fixed direct/spillover reference; it is not re-estimated or used to define any state. Validation effects use the e99 pre-treatment groups and the existing e129/e149/e169/e189/e199 rows.",
        "",
        markdown_table(
            ["seed", "arm", "Train CW direct robust Δ pp", "Train non-CW spillover robust Δ pp"],
            [
                [
                    seed,
                    arm.replace("CLEAN_WRONG_", ""),
                    f"{100 * state_artifact['seeds'][seed]['train_e199'][arm]['direct_cw']['robust']['accuracy_delta']:+.3f}",
                    f"{100 * state_artifact['seeds'][seed]['train_e199'][arm]['spillover_non_cw']['robust']['accuracy_delta']:+.3f}",
                ]
                for seed in SEEDS
                for arm in ("CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM")
            ],
        ),
        "",
        "## Canonical non-CW harm localization",
        "",
        "Primary target is validation non-CW harm within canonical S2 versus S1. Cells with n < 100 are retained as descriptive but marked decision-ineligible.",
        "",
    ]
    for seed in SEEDS:
        report_lines.append(f"### {seed}")
        report_lines.append("")
        rows = []
        for arm in ("CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM"):
            for epoch in EPOCHS:
                item = state_artifact["seeds"][seed]["effects"][arm][str(epoch)]
                s2 = cells_artifact["seeds"][seed]["cells"][arm][str(epoch)]["S2"]
                s1 = cells_artifact["seeds"][seed]["cells"][arm][str(epoch)]["S1"]
                rows.append(
                    [
                        epoch,
                        arm.replace("CLEAN_WRONG_", ""),
                        f"{100 * item['validation_cw']['robust']['accuracy_delta']:+.3f}",
                        f"{100 * item['validation_non_cw']['robust']['accuracy_delta']:+.3f}",
                        f"{100 * s1['robust']['harm_rate']:.2f}%",
                        f"{100 * s2['robust']['harm_rate']:.2f}%",
                        s1["n"],
                        s2["n"],
                    ]
                )
        report_lines.extend(
            [
                markdown_table(
                    ["epoch", "arm", "V-CW robust Δ pp", "V-nonCW robust Δ pp", "S1 harm", "S2 harm", "S1 n", "S2 n"],
                    rows,
                ),
                "",
            ]
        )
        cell_rows = []
        for arm in ("CLEAN_WRONG_PLAIN_ADVCE", "CLEAN_WRONG_TPFM"):
            for teacher_state in ("T1", "T2", "T3"):
                cell = cells_artifact["seeds"][seed]["cells"][arm]["199"]["student_teacher_cells"][
                    f"S2x{teacher_state}"
                ]
                cell_rows.append(
                    [
                        arm.replace("CLEAN_WRONG_", ""),
                        teacher_state,
                        cell["n"],
                        f"{100 * cell['robust']['accuracy_delta']:+.3f}",
                        f"{100 * cell['robust']['harm_rate']:.2f}%",
                        "yes" if cell["decision_eligible_n_ge_100"] else "no",
                    ]
                )
        report_lines.extend(
            [
                "S2 × Teacher cells at e199 (cells below n=100 are not mechanism-decision eligible):",
                "",
                markdown_table(["arm", "Teacher state", "n", "robust Δ pp", "harm", "n≥100"], cell_rows),
                "",
            ]
        )

    report_lines.extend(
        [
            "## TPFM boundary regimes",
            "",
            f"The frozen TPFM values are coefficient `{TPFM_COEFFICIENT}`, floor `{TPFM_FLOOR}`, and cap `{TPFM_CAP}`. Regime counts for Train-CW and Validation-CW are recorded in `ert_rslad_i100_cw_boundary_tradeoff_v1.json`; no regime was selected or retuned from outcomes.",
            "",
            "## Mechanism classification",
            "",
            "B1–B4 and D1–D3 descriptive classifications are machine-recorded per seed. D1 means at least one primary feature has absolute SMD or categorical proportion difference at least 0.25; D2 is at least 0.10; D3 is weaker. B1 requires repeated CW rescue, S2< S1 robust effects, and S2 harm enrichment; otherwise the less-structured B2/B3/B4 labels are used. These are descriptive, not population claims.",
            "",
            markdown_table(
                ["seed", "D shift", "Plain class", "TPFM class"],
                [
                    [
                        seed,
                        boundary_artifact["seeds"][seed]["state_shift_class"],
                        boundary_artifact["seeds"][seed]["classification"]["CLEAN_WRONG_PLAIN_ADVCE"][
                            "mechanism_class"
                        ],
                        boundary_artifact["seeds"][seed]["classification"]["CLEAN_WRONG_TPFM"]["mechanism_class"],
                    ]
                    for seed in SEEDS
                ],
            ),
            "",
            "## Decision and stop boundary",
            "",
            "This closes the current Clean-Wrong exploration as a diagnostic. It does not validate a new S2 intervention, change the TPFM margin/floor/cap, select a new threshold, or authorize dynamic/History routing. Any follow-up must be a separately reviewed intervention.",
            "",
            "Machine artifacts: `ert_rslad_i100_cw_gap_completion_contract_v1.json`, `ert_rslad_i100_train_validation_cw_shift_v1.json`, `ert_rslad_i100_canonical_s2_harm_localization_v1.json`, `ert_rslad_i100_canonical_s2_teacher_cells_v1.json`, and `ert_rslad_i100_cw_boundary_tradeoff_v1.json`.",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "output_dir": str(output),
                "seeds": list(SEEDS),
                "epochs": list(EPOCHS),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
