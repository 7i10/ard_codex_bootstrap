"""Fail-closed CPU overlays for the frozen ERT epoch-79 state pilot.

This module only reads registered anchor-state and CE-PGD20 endpoint artifacts.
It never launches an attack or training job, and its anchor state table has no
future-outcome columns.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_causal_ce20 import ARMS, ENDPOINT_COLUMNS, HORIZONS, ce_pgd20_attack_identity
from ard.analysis.ffnr_state_mechanism import (
    EXPECTED_ONLINE_ATTACK,
    _margin_rows,
    _read_compact_observations,
    _validate_online_attack,
)
from ard.analysis.ffnr_strong_point import _online_panel, _strong_lineage
from ard.analysis.ffnr_strong_replay import EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import canonical_json, sha256_file

CONTRACT = "ert_state_overlay_v1"
LABELS = ("L2", "L4")
ANCHOR = 79
QUANTILES = (0.10, 0.20)
STATE_COLUMNS = (
    "namespace",
    "label",
    "anchor_epoch",
    "sample_id",
    "class_id",
    "student_clean_correct",
    "student_adv_correct",
    "teacher_clean_correct",
    "teacher_adv_correct",
    "mS_clean",
    "mS_adv",
    "mT_clean",
    "mT_adv",
    "DeltaS",
    "DeltaT",
    "signed_teacher_dominance",
    "online_current_correct",
    "online_frequency_risk",
    "online_margin_risk",
    "online_last_margin_risk",
    "student_state",
    "teacher_state_q10",
    "teacher_state_q20",
)


class ERTStateOverlayError(ValueError):
    """Raised when state or endpoint provenance cannot be established."""


def _json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ERTStateOverlayError(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ERTStateOverlayError(f"{name} must be a JSON object")
    return value


def _path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ERTStateOverlayError(f"{name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _tracked_clean_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__).resolve(), root / "src/ard/cli/ert_state_overlay.py")
    try:
        relative = [str(path.relative_to(root)) for path in paths]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ERTStateOverlayError("overlay requires tracked source and readable Git identity") from exc
    if len(sha) != 40 or dirty:
        raise ERTStateOverlayError("overlay requires a tracked-clean analysis revision")
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": hashes,
        "source_sha256": hashlib.sha256(canonical_json(hashes)).hexdigest(),
    }


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ERTStateOverlayError("state-overlay config is unreadable") from exc
    required = {
        "schema_version",
        "contract",
        "expected_count",
        "stable_id_class_universe_sha256",
        "anchor",
        "quantiles",
        "runs",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("contract") != CONTRACT
    ):
        raise ERTStateOverlayError("state-overlay config schema/contract drifted")
    if (
        raw.get("expected_count") != 45_000
        or raw.get("stable_id_class_universe_sha256") != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
    ):
        raise ERTStateOverlayError("state-overlay stable-ID universe drifted")
    if raw.get("anchor") != ANCHOR or tuple(raw.get("quantiles", ())) != QUANTILES:
        raise ERTStateOverlayError("state-overlay anchor/quantile contract drifted")
    runs = raw.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(LABELS):
        raise ERTStateOverlayError("state-overlay requires exactly L2/L4")
    source_names = {
        "feature_observations",
        "feature_lineage",
        "online_states",
        "online_lineage",
        "parent_fork_lineage",
    }
    endpoint_names = {"observations", "lineage", "report"}
    parsed: dict[str, Any] = {"expected_count": 45_000, "runs": {}}
    for label in LABELS:
        run = runs[label]
        if not isinstance(run, Mapping) or set(run) != {"anchor_state", "endpoints"}:
            raise ERTStateOverlayError(f"state-overlay {label} schema drifted")
        source = run["anchor_state"]
        endpoints = run["endpoints"]
        if not isinstance(source, Mapping) or set(source) != source_names:
            raise ERTStateOverlayError(f"state-overlay {label} anchor-state schema drifted")
        if not isinstance(endpoints, Mapping) or set(endpoints) != {str(item) for item in HORIZONS}:
            raise ERTStateOverlayError(f"state-overlay {label} endpoints must be exactly 84/89/94")
        parsed["runs"][label] = {
            "anchor_state": {name: _path(path.parent, source[name], name=f"{label}.{name}") for name in source_names},
            "endpoints": {},
        }
        for horizon in HORIZONS:
            entry = endpoints[str(horizon)]
            if not isinstance(entry, Mapping) or set(entry) != endpoint_names:
                raise ERTStateOverlayError(f"state-overlay {label} horizon-{horizon} schema drifted")
            parsed["runs"][label]["endpoints"][horizon] = {
                name: _path(path.parent, entry[name], name=f"{label}.{horizon}.{name}") for name in endpoint_names
            }
    return parsed


def _lower_positive_ids(rows: Mapping[int, Mapping[str, Any]], q: float) -> tuple[set[int], float]:
    positive = [item for item, row in rows.items() if bool(row["teacher_adv_correct"])]
    if not positive:
        raise ERTStateOverlayError("teacher-positive anchor cohort is empty")
    nominal = math.ceil(q * len(positive))
    ordered = sorted(positive, key=lambda item: (float(rows[item]["mT_adv"]), item))
    selected = set(ordered[:nominal])
    return selected, float(rows[ordered[nominal - 1]]["mT_adv"])


def build_state_bundle(
    *,
    label: str,
    feature_observations: Path,
    feature_lineage: Path,
    online_states: Path,
    online_lineage: Path,
    parent_fork_lineage: Path,
    expected_count: int = 45_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the no-future-outcome anchor table and deterministic fixed masks."""
    if label not in LABELS:
        raise ERTStateOverlayError("state overlay is frozen to Chen L2/L4")
    feature_meta = _strong_lineage(
        path=feature_lineage,
        observations=feature_observations,
        role="feature",
        expected_count=expected_count,
        expected_universe_sha256=EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
    )
    if ANCHOR not in tuple(feature_meta.get("requested_epochs", ())):
        raise ERTStateOverlayError("anchor replay lacks epoch 79")
    online, online_meta = _online_panel(online_states, online_lineage, expected_count)
    _validate_online_attack(online_meta)
    for key in ("run_id", "teacher", "dataset_identity"):
        if feature_meta.get(key) != online_meta.get(key):
            raise ERTStateOverlayError("anchor replay/online lineage identity drifted")
    if feature_meta.get("saved_resolved_config_mapping_sha256") != online_meta.get("config_hash"):
        raise ERTStateOverlayError("anchor replay/online config lineage drifted")
    # The legacy CE20 endpoint lineage predates an explicit parent binding.
    # Bind its use here to the immutable C79 fork lineage and the epoch-79
    # checkpoint inventory already used by the anchor replay.
    parent_fork = _json(parent_fork_lineage, name="C79 parent fork lineage")
    checkpoints = feature_meta.get("checkpoints")
    epoch79 = (
        next((item for item in checkpoints if isinstance(item, Mapping) and item.get("epoch") == ANCHOR), None)
        if isinstance(checkpoints, list)
        else None
    )
    if not isinstance(epoch79, Mapping) or parent_fork.get("parent_epoch") != ANCHOR:
        raise ERTStateOverlayError("C79 parent lineage lacks the frozen epoch-79 checkpoint")
    if (
        parent_fork.get("parent_checkpoint_sha256") != epoch79.get("sha256")
        or parent_fork.get("parent_raw_config_sha256") != feature_meta.get("saved_resolved_config_mapping_sha256")
        or parent_fork.get("parent_sample_state_records") != expected_count
    ):
        raise ERTStateOverlayError("C79 parent checkpoint/config/sample-state lineage drifted")
    feature = _read_compact_observations(
        feature_observations, epochs=(ANCHOR,), expected_count=expected_count, feature=True
    )
    rows = _margin_rows(feature, anchor=ANCHOR)
    if set(rows) != set(online[ANCHOR]) or any(
        rows[item]["class_id"] != online[ANCHOR][item]["class_id"] for item in rows
    ):
        raise ERTStateOverlayError("anchor stable-ID/class join drifted")
    low_ids, thresholds = {}, {}
    for q in QUANTILES:
        low_ids[q], thresholds[q] = _lower_positive_ids(rows, q)
    state_rows: list[dict[str, Any]] = []
    masks: dict[str, set[int]] = {
        "student_clean_wrong": set(),
        "student_clean_wrong_teacher_clean_correct": set(),
        "student_clean_wrong_teacher_clean_wrong": set(),
    }
    for q in QUANTILES:
        suffix = f"q{int(q * 100):02d}"
        masks.update({f"s3_t1_{suffix}": set(), f"s3_t2_{suffix}": set(), f"s3_t3_{suffix}": set()})
    for sample_id in sorted(rows):
        row, history = rows[sample_id], online[ANCHOR][sample_id]
        s3 = bool(row["student_clean_correct"]) and not bool(row["student_robust_correct"])
        # The pilot's Part 5 fixes S3, while the attachment also names S1/S2.
        # Make the complete mutually-exclusive partition explicit: S1 is
        # adversarial-correct; the adversarial-wrong remainder is split by
        # clean correctness into S2/S3.  No future endpoint appears here.
        student_state = "S1" if row["student_robust_correct"] else "S3" if s3 else "S2"
        teacher_states: dict[float, str] = {}
        for q in QUANTILES:
            state = "T3" if not row["teacher_adv_correct"] else "T2" if sample_id in low_ids[q] else "T1"
            teacher_states[q] = state
            if s3:
                masks[f"s3_{state.lower()}_q{int(q * 100):02d}"].add(sample_id)
        if not row["student_clean_correct"]:
            masks["student_clean_wrong"].add(sample_id)
            masks[
                "student_clean_wrong_teacher_clean_correct"
                if row["teacher_clean_correct"]
                else "student_clean_wrong_teacher_clean_wrong"
            ].add(sample_id)
        state_rows.append(
            {
                "namespace": "train",
                "label": label,
                "anchor_epoch": ANCHOR,
                "sample_id": sample_id,
                "class_id": row["class_id"],
                "student_clean_correct": row["student_clean_correct"],
                "student_adv_correct": row["student_robust_correct"],
                "teacher_clean_correct": row["teacher_clean_correct"],
                "teacher_adv_correct": row["teacher_adv_correct"],
                "mS_clean": row["mS_clean"],
                "mS_adv": row["mS_adv"],
                "mT_clean": row["mT_clean"],
                "mT_adv": row["mT_adv"],
                "DeltaS": row["DeltaS"],
                "DeltaT": row["DeltaT"],
                # Registered D signal: maximum wrong-class probability minus
                # teacher true-class probability, i.e. negative teacher margin.
                "signed_teacher_dominance": -row["mT_adv"],
                "online_current_correct": history["current_correct"],
                "online_frequency_risk": history["frequency_risk"],
                "online_margin_risk": history["margin_risk"],
                "online_last_margin_risk": history["last_margin_risk"],
                "student_state": student_state,
                "teacher_state_q10": teacher_states[0.10],
                "teacher_state_q20": teacher_states[0.20],
            }
        )
    if any(tuple(row) != STATE_COLUMNS for row in state_rows):
        raise ERTStateOverlayError("anchor state-table schema drifted")
    mask_records = {
        name: {
            "selected_count": len(ids),
            "selected_ids": sorted(ids),
            "selected_ids_sha256": hashlib.sha256(canonical_json(sorted(ids))).hexdigest(),
            "selected_class_counts": {
                str(class_id): sum(rows[item]["class_id"] == class_id for item in ids) for class_id in range(10)
            },
        }
        for name, ids in sorted(masks.items())
    }
    return state_rows, {
        "schema_version": 1,
        "contract": CONTRACT,
        "label": label,
        "namespace": "train",
        "anchor_epoch": ANCHOR,
        "definitions": {
            "student_state_definition": {
                "S1": "student_adv_correct",
                "S2": "student_adv_wrong and student_clean_wrong",
                "S3": "student_clean_correct and student_adv_wrong (pilot treatment state)",
            },
            "T1": "teacher_adv_correct and not lower_q_positive_mT_adv",
            "T2": "lower_q_positive_mT_adv",
            "T3": "teacher_adv_wrong",
            "signed_teacher_dominance": "max_wrong_probability_minus_true_probability = -mT_adv",
        },
        "quantiles": {
            str(q): {
                "positive_teacher_count": sum(bool(row["teacher_adv_correct"]) for row in rows.values()),
                "mT_adv_threshold": thresholds[q],
                "tie_break": "(mT_adv, stable_sample_id) ascending",
            }
            for q in QUANTILES
        },
        "masks": mask_records,
        "input_identity": {
            "run_id": feature_meta["run_id"],
            "config_hash": feature_meta["saved_resolved_config_mapping_sha256"],
            "teacher": feature_meta["teacher"],
            "dataset_identity": feature_meta["dataset_identity"],
            "online_attack_identity": EXPECTED_ONLINE_ATTACK,
            "parent_binding": {
                "fork_lineage": str(parent_fork_lineage.resolve()),
                "fork_lineage_sha256": sha256_file(parent_fork_lineage),
                "parent_checkpoint_sha256": parent_fork["parent_checkpoint_sha256"],
                "parent_raw_config_sha256": parent_fork["parent_raw_config_sha256"],
                "parent_sample_state_sha256": parent_fork.get("parent_sample_state_sha256"),
            },
            "input_sha256": {
                "feature_observations": sha256_file(feature_observations),
                "feature_lineage": sha256_file(feature_lineage),
                "online_states": sha256_file(online_states),
                "online_lineage": sha256_file(online_lineage),
            },
        },
    }


def _endpoint_rows(
    *, label: str, horizon: int, observations: Path, lineage: Path, report: Path, expected_count: int
) -> dict[str, dict[int, dict[str, Any]]]:
    meta, summary = _json(lineage, name="CE20 lineage"), _json(report, name="CE20 report")
    attack = ce_pgd20_attack_identity()
    if (
        meta.get("contract") != "ffnr_causal_ce20_v1"
        or meta.get("label") != label
        or meta.get("horizon") != horizon
        or meta.get("attack_identity") != attack
        or meta.get("attack_identity_sha256") != hashlib.sha256(canonical_json(attack)).hexdigest()
        or meta.get("observations_sha256") != sha256_file(observations)
    ):
        raise ERTStateOverlayError("CE20 endpoint lineage/attack binding drifted")
    if summary.get("label") != label or summary.get("horizon") != horizon:
        raise ERTStateOverlayError("CE20 endpoint report identity drifted")
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(observations)
    except Exception as exc:
        raise ERTStateOverlayError("CE20 endpoint observations are unreadable") from exc
    if set(table.column_names) != set(ENDPOINT_COLUMNS):
        raise ERTStateOverlayError("CE20 endpoint Parquet schema drifted")
    result: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for row in table.to_pylist():
        if (
            row.get("label") != label
            or row.get("horizon") != horizon
            or row.get("namespace") != "train"
            or row.get("arm") not in result
        ):
            raise ERTStateOverlayError("CE20 endpoint row identity drifted")
        sample_id, class_id = row.get("sample_id"), row.get("class_id")
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or sample_id in result[row["arm"]]
        ):
            raise ERTStateOverlayError("CE20 endpoint stable ID/class is invalid")
        result[row["arm"]][sample_id] = dict(row)
    reference = result["C79"]
    if len(reference) != expected_count or any(set(value) != set(reference) for value in result.values()):
        raise ERTStateOverlayError("CE20 endpoint stable-ID coverage drifted")
    if any(
        any(value[item]["class_id"] != reference[item]["class_id"] for item in reference) for value in result.values()
    ):
        raise ERTStateOverlayError("CE20 endpoint class join drifted")
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def effect(
    control: Mapping[int, Mapping[str, Any]], treatment: Mapping[int, Mapping[str, Any]], ids: Sequence[int]
) -> dict[str, Any]:
    if not ids or any(item not in control or item not in treatment for item in ids):
        raise ERTStateOverlayError("overlay paired endpoint stable-ID join is incomplete")
    robust = [
        int(bool(treatment[item]["student_robust_correct"])) - int(bool(control[item]["student_robust_correct"]))
        for item in ids
    ]
    clean = [
        int(bool(treatment[item]["student_clean_correct"])) - int(bool(control[item]["student_clean_correct"]))
        for item in ids
    ]
    rescue = [
        not bool(control[item]["student_robust_correct"]) and bool(treatment[item]["student_robust_correct"])
        for item in ids
    ]
    harm = [
        bool(control[item]["student_robust_correct"]) and not bool(treatment[item]["student_robust_correct"])
        for item in ids
    ]
    clean_harm = [
        bool(control[item]["student_clean_correct"]) and not bool(treatment[item]["student_clean_correct"])
        for item in ids
    ]
    return {
        "n": len(ids),
        "rescue_count": sum(rescue),
        "rescue_rate": _mean([float(x) for x in rescue]),
        "harm_count": sum(harm),
        "harm_rate": _mean([float(x) for x in harm]),
        "net_rescue_count": sum(robust),
        "net_rescue_rate": _mean(robust),
        "robust_accuracy_delta": _mean(robust),
        "clean_accuracy_delta": _mean(clean),
        "clean_harm_count": sum(clean_harm),
        "clean_harm_rate": _mean([float(x) for x in clean_harm]),
        "clean_margin_delta": _mean(
            [
                float(treatment[item]["student_clean_probability_margin"])
                - float(control[item]["student_clean_probability_margin"])
                for item in ids
            ]
        ),
        "adversarial_margin_delta": _mean(
            [
                float(treatment[item]["student_adversarial_probability_margin"])
                - float(control[item]["student_adversarial_probability_margin"])
                for item in ids
            ]
        ),
    }


def summarize_overlay(
    *,
    state_rows: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Mapping[str, Any]],
    endpoints: Mapping[int, Mapping[str, Mapping[int, Mapping[str, Any]]]],
) -> dict[str, Any]:
    state_by_id = {int(row["sample_id"]): row for row in state_rows}
    if len(state_by_id) != len(state_rows):
        raise ERTStateOverlayError("state table has duplicate sample IDs")
    mask_ids = {name: list(value["selected_ids"]) for name, value in masks.items()}
    result: dict[str, Any] = {"scientific_status": "exploratory_read_only_overlay_not_an_arm_selector", "horizons": {}}
    for horizon, by_arm in endpoints.items():
        control = by_arm["C79"]
        if set(control) != set(state_by_id) or any(
            control[item]["class_id"] != state_by_id[item]["class_id"] for item in control
        ):
            raise ERTStateOverlayError("anchor/CE20 stable-ID/class join drifted")
        for sample_id in control:
            flags = ("route_b_selected", "route_b_random")
            if any(by_arm[arm][sample_id][flag] != control[sample_id][flag] for arm in ARMS for flag in flags):
                raise ERTStateOverlayError("old Route-B cohort flags drift across endpoint arms")
        cohorts = {name: ids for name, ids in mask_ids.items()}
        cohorts["old_route_b_selected"] = [item for item in sorted(control) if control[item]["route_b_selected"]]
        cohorts["old_route_b_random"] = [item for item in sorted(control) if control[item]["route_b_random"]]
        horizon_report: dict[str, Any] = {}
        for name, ids in cohorts.items():
            if not ids:
                raise ERTStateOverlayError("overlay cohort is empty")
            anchor = {
                "n": len(ids),
                "teacher_clean_correct_rate": _mean(
                    [float(bool(state_by_id[item]["teacher_clean_correct"])) for item in ids]
                ),
                "teacher_adv_correct_rate": _mean(
                    [float(bool(state_by_id[item]["teacher_adv_correct"])) for item in ids]
                ),
                "mean_mT_adv": _mean([float(state_by_id[item]["mT_adv"]) for item in ids]),
                "mean_DeltaT": _mean([float(state_by_id[item]["DeltaT"]) for item in ids]),
            }
            horizon_report[name] = {
                "anchor": anchor,
                "effects": {arm: effect(control, by_arm[arm], ids) for arm in ARMS if arm != "C79"},
            }
        result["horizons"][str(horizon)] = horizon_report
    return result


def run_overlay(*, config_path: Path, output_dir: Path) -> dict[str, Path]:
    """Create immutable anchor state/mask artifacts and CPU-only endpoint overlays."""
    if output_dir.exists():
        raise ERTStateOverlayError("refusing to overwrite an overlay output directory")
    config = load_config(config_path)
    provenance = _tracked_clean_provenance()
    bundles, state_paths, mask_paths, all_reports, inputs = {}, {}, {}, {}, {}
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for label in LABELS:
            source = config["runs"][label]["anchor_state"]
            states, mask = build_state_bundle(label=label, expected_count=config["expected_count"], **source)
            state_path = write_sample_parquet(states, output_dir / f"anchor79-state-table-{label}.parquet")
            mask["state_table_sha256"] = sha256_file(state_path)
            mask["analysis_provenance"] = provenance
            mask_path = output_dir / f"anchor79-fixed-masks-{label}.json"
            mask_path.write_text(json.dumps(mask, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            endpoints = {
                horizon: _endpoint_rows(
                    label=label,
                    horizon=horizon,
                    expected_count=config["expected_count"],
                    **config["runs"][label]["endpoints"][horizon],
                )
                for horizon in HORIZONS
            }
            all_reports[label] = summarize_overlay(state_rows=states, masks=mask["masks"], endpoints=endpoints)
            bundles[label], state_paths[label], mask_paths[label] = mask, state_path, mask_path
            inputs[label] = {
                "anchor_state": {name: sha256_file(path) for name, path in source.items()},
                "endpoints": {
                    str(h): {name: sha256_file(path) for name, path in config["runs"][label]["endpoints"][h].items()}
                    for h in HORIZONS
                },
            }
        report_path = output_dir / "ert-state-overlay-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract": CONTRACT,
                    "config_sha256": sha256_file(config_path),
                    "analysis_provenance": provenance,
                    "reports": all_reports,
                },
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        lineage_path = output_dir / "lineage.json"
        lineage_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract": CONTRACT,
                    "config_sha256": sha256_file(config_path),
                    "analysis_provenance": provenance,
                    "inputs": inputs,
                    "state_tables": {label: sha256_file(path) for label, path in state_paths.items()},
                    "fixed_masks": {label: sha256_file(path) for label, path in mask_paths.items()},
                    "report_sha256": sha256_file(report_path),
                },
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        # A partial analysis directory must never be mistaken for a complete report.
        raise
    return {
        **{f"state_table_{label}": path for label, path in state_paths.items()},
        **{f"fixed_masks_{label}": path for label, path in mask_paths.items()},
        "report": report_path,
        "lineage": lineage_path,
    }
