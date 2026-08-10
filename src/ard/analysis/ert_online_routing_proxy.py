"""Read-only CPU diagnostics for the frozen ERT online-routing proxy plan."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ffnr_attack_factorial import CONDITIONS
from ard.analysis.ffnr_forecasting import _online_panel, equal_rank_score
from ard.analysis.ffnr_state_mechanism import _margin_rows, _read_compact_observations, _strong_lineage
from ard.analysis.ffnr_strong_replay import EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
from ard.analysis.signal_audit import canonical_json, sha256_file

CONTRACT = "ert_online_routing_proxy_v1"
LABELS = ("L2", "L4")
ANCHORS = (39, 59, 79)
TERMINAL_EPOCHS = (189, 194, 199)
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


class ERTOnlineRoutingProxyError(ValueError):
    pass


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ERTOnlineRoutingProxyError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ERTOnlineRoutingProxyError(f"{name} must be an object")
    return value


def _path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ERTOnlineRoutingProxyError(f"{name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__).resolve(), root / "src/ard/cli/ert_online_routing_proxy.py")
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
        raise ERTOnlineRoutingProxyError("proxy diagnostic requires tracked source and readable Git identity") from exc
    if len(sha) != 40 or dirty:
        raise ERTOnlineRoutingProxyError("proxy diagnostic requires a tracked-clean analysis revision")
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
        raise ERTOnlineRoutingProxyError("proxy config is unreadable") from exc
    required = {
        "schema_version",
        "contract",
        "expected_count",
        "stable_id_class_universe_sha256",
        "anchors",
        "terminal_epochs",
        "top_fractions",
        "runs",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("contract") != CONTRACT
    ):
        raise ERTOnlineRoutingProxyError("proxy config schema/contract drifted")
    if (
        raw.get("expected_count") != 45_000
        or raw.get("stable_id_class_universe_sha256") != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
    ):
        raise ERTOnlineRoutingProxyError("proxy config stable-ID universe drifted")
    if (
        tuple(raw.get("anchors", ())) != ANCHORS
        or tuple(raw.get("terminal_epochs", ())) != TERMINAL_EPOCHS
        or tuple(raw.get("top_fractions", ())) != TOP_FRACTIONS
    ):
        raise ERTOnlineRoutingProxyError("proxy config frozen epochs/fractions drifted")
    runs = raw.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(LABELS):
        raise ERTOnlineRoutingProxyError("proxy config requires exactly L2/L4")
    source = {
        "feature_observations",
        "feature_lineage",
        "outcome_observations",
        "outcome_lineage",
        "online_states",
        "online_lineage",
    }
    factorial = {condition: {"observations", "lineage"} for condition in CONDITIONS}
    parsed: dict[str, Any] = {"expected_count": 45_000, "runs": {}}
    for label in LABELS:
        entry = runs[label]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"source", "factorial"}
            or not isinstance(entry["source"], Mapping)
            or set(entry["source"]) != source
        ):
            raise ERTOnlineRoutingProxyError(f"proxy {label} source schema drifted")
        if not isinstance(entry["factorial"], Mapping) or set(entry["factorial"]) != set(CONDITIONS):
            raise ERTOnlineRoutingProxyError(f"proxy {label} factorial schema drifted")
        parsed["runs"][label] = {
            "source": {name: _path(path.parent, entry["source"][name], f"{label}.{name}") for name in source},
            "factorial": {},
        }
        for condition, keys in factorial.items():
            value = entry["factorial"][condition]
            if value is None:
                parsed["runs"][label]["factorial"][condition] = None
                continue
            if not isinstance(value, Mapping) or set(value) != keys:
                raise ERTOnlineRoutingProxyError(f"proxy {label}.{condition} factorial schema drifted")
            parsed["runs"][label]["factorial"][condition] = {
                name: _path(path.parent, value[name], f"{label}.{condition}.{name}") for name in keys
            }
    return parsed


def _future_failure(outcome: Mapping[int, Mapping[int, Mapping[str, Any]]]) -> dict[int, int]:
    if set(outcome) != set(TERMINAL_EPOCHS):
        raise ERTOnlineRoutingProxyError("majority FF requires exactly terminal 189/194/199")
    ids = set(outcome[TERMINAL_EPOCHS[0]])
    if any(set(outcome[epoch]) != ids for epoch in TERMINAL_EPOCHS):
        raise ERTOnlineRoutingProxyError("terminal stable-ID universe drifted")
    target: dict[int, int] = {}
    for item in ids:
        flags = [outcome[epoch][item].get("student_robust_correct") for epoch in TERMINAL_EPOCHS]
        if any(not isinstance(value, bool) for value in flags):
            raise ERTOnlineRoutingProxyError("terminal robust correctness must be boolean")
        target[item] = int(sum(not value for value in flags) >= 2)
    return target


def _top_metrics(score: Mapping[int, float], target: Mapping[int, int]) -> list[dict[str, Any]]:
    if set(score) != set(target) or not score:
        raise ERTOnlineRoutingProxyError("Top-K score/target coverage drifted")
    ordered = sorted(score, key=lambda item: (-float(score[item]), item))
    positives = sum(target.values())
    result: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        count = max(1, math.ceil(fraction * len(ordered)))
        selected = ordered[:count]
        hits = sum(target[item] for item in selected)
        result.append(
            {
                "fraction": fraction,
                "k": count,
                "ff_count": hits,
                "precision": hits / count,
                "recall": hits / positives if positives else None,
                "lift": (hits / count) / (positives / len(ordered)) if positives else None,
            }
        )
    count = positives
    selected = ordered[:count]
    hits = sum(target[item] for item in selected)
    result.append(
        {
            "selection": "gt_count",
            "fraction": None,
            "k": count,
            "ff_count": hits,
            "precision": hits / count if count else None,
            "recall": hits / positives if positives else None,
            "lift": (hits / count) / (positives / len(ordered)) if count and positives else None,
        }
    )
    return result


def _pearson(left: Mapping[int, float], right: Mapping[int, float]) -> float | None:
    if set(left) != set(right) or len(left) < 2:
        raise ERTOnlineRoutingProxyError("correlation stable-ID coverage drifted")
    x, y = [float(left[item]) for item in sorted(left)], [float(right[item]) for item in sorted(right)]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    denominator = math.sqrt(sum((item - mx) ** 2 for item in x) * sum((item - my) ** 2 for item in y))
    return None if denominator == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / denominator


def _spearman(left: Mapping[int, float], right: Mapping[int, float]) -> float | None:
    if set(left) != set(right):
        raise ERTOnlineRoutingProxyError("correlation stable-ID coverage drifted")
    def ranks(values: Mapping[int, float]) -> dict[int, float]:
        ordered = sorted((float(value), item) for item, value in values.items())
        result: dict[int, float] = {}
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            rank = (start + 1 + end) / 2.0
            for _, item in ordered[start:end]:
                result[item] = rank
            start = end
        return result
    return _pearson(ranks(left), ranks(right))


def _agreement(left: Mapping[int, float], right: Mapping[int, float]) -> list[dict[str, Any]]:
    result = []
    for fraction in TOP_FRACTIONS:
        count = max(1, math.ceil(fraction * len(left)))
        a = set(sorted(left, key=lambda item: (-left[item], item))[:count])
        b = set(sorted(right, key=lambda item: (-right[item], item))[:count])
        result.append(
            {"fraction": fraction, "k": count, "intersection": len(a & b), "jaccard": len(a & b) / len(a | b)}
        )
    return result


def _states_q10(
    *, robust_correct: Mapping[int, bool], risk: Mapping[int, float], lower_is_fragile: bool
) -> dict[int, str]:
    """Frozen S1/S2/S3 partition with stable-ID ties at q=10%."""
    if set(robust_correct) != set(risk):
        raise ERTOnlineRoutingProxyError("state score/correctness coverage drifted")
    positive = [item for item in risk if robust_correct[item]]
    if not positive:
        raise ERTOnlineRoutingProxyError("state robust-correct cohort is empty")
    ordered = sorted(positive, key=lambda item: (risk[item] if lower_is_fragile else -risk[item], item))
    fragile = set(ordered[: math.ceil(0.10 * len(ordered))])
    return {item: "S3" if not robust_correct[item] else "S2" if item in fragile else "S1" for item in risk}


def _teacher_states_q10(rows: Mapping[int, Mapping[str, Any]]) -> dict[int, str]:
    positive = [item for item, row in rows.items() if bool(row["teacher_adv_correct"])]
    if not positive:
        raise ERTOnlineRoutingProxyError("teacher-correct cohort is empty")
    fragile = set(
        sorted(positive, key=lambda item: (float(rows[item]["mT_adv"]), item))[: math.ceil(0.10 * len(positive))]
    )
    return {
        item: "T3" if not row["teacher_adv_correct"] else "T2" if item in fragile else "T1"
        for item, row in rows.items()
    }


def _state_cells(
    student: Mapping[int, str], teacher: Mapping[int, str], target: Mapping[int, int]
) -> list[dict[str, Any]]:
    if set(student) != set(teacher) or set(student) != set(target):
        raise ERTOnlineRoutingProxyError("state-cell stable-ID coverage drifted")
    result = []
    for student_state in ("S1", "S2", "S3"):
        for teacher_state in ("T1", "T2", "T3"):
            ids = [item for item in target if student[item] == student_state and teacher[item] == teacher_state]
            if ids:
                result.append(
                    {
                        "student_state": student_state,
                        "teacher_state": teacher_state,
                        "count": len(ids),
                        "majority_ff_rate": sum(target[item] for item in ids) / len(ids),
                    }
                )
    return result


def diagnose(
    *,
    feature: Mapping[int, Mapping[int, Mapping[str, Any]]],
    outcome: Mapping[int, Mapping[int, Mapping[str, Any]]],
    online: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    target = _future_failure(outcome)
    if set(feature) != set(ANCHORS) or set(online) != set(ANCHORS):
        raise ERTOnlineRoutingProxyError("proxy requires anchors 39/59/79 only")
    if any(set(feature[epoch]) != set(target) or set(online[epoch]) != set(target) for epoch in ANCHORS):
        raise ERTOnlineRoutingProxyError("feature/outcome/online stable-ID coverage drifted")
    if any(feature[epoch][item]["class_id"] != online[epoch][item]["class_id"] for epoch in ANCHORS for item in target):
        raise ERTOnlineRoutingProxyError("feature/online class join drifted")
    anchors: dict[str, Any] = {}
    transition_rows: list[dict[str, Any]] = []
    states: dict[int, dict[str, dict[int, tuple[str, str]]]] = {}
    for epoch in ANCHORS:
        margins = _margin_rows(feature, anchor=epoch)
        scores = {
            "teacher_signed_dominance": {item: -float(row["mT_adv"]) for item, row in margins.items()},
            "strong_student_margin_risk": {item: -float(row["mS_adv"]) for item, row in margins.items()},
            "online_margin_ema_risk": {item: float(online[epoch][item]["margin_risk"]) for item in target},
            "online_frequency_risk": {item: float(online[epoch][item]["frequency_risk"]) for item in target},
            "teacher_clean_margin_risk": {item: -float(row["mT_clean"]) for item, row in margins.items()},
            "mT_clean_plus_DeltaT_rank": equal_rank_score(
                {
                    "mT_clean": {item: -float(row["mT_clean"]) for item, row in margins.items()},
                    "DeltaT": {item: float(row["DeltaT"]) for item, row in margins.items()},
                }
            ),
        }
        strong_student = _states_q10(
            robust_correct={item: bool(row["student_robust_correct"]) for item, row in margins.items()},
            risk={item: float(row["mS_adv"]) for item, row in margins.items()},
            lower_is_fragile=True,
        )
        online_student = _states_q10(
            robust_correct={item: bool(online[epoch][item]["current_correct"]) for item in target},
            risk={item: float(online[epoch][item]["margin_risk"]) for item in target},
            lower_is_fragile=False,
        )
        teacher_state = _teacher_states_q10(margins)
        oracle_correct = {item: bool(row["student_robust_correct"]) for item, row in margins.items()}
        eligible_target = {item: target[item] for item in target if oracle_correct[item]}
        states[epoch] = {
            "strong_oracle": {item: (strong_student[item], teacher_state[item]) for item in target},
            "online": {item: (online_student[item], teacher_state[item]) for item in target},
        }
        anchors[str(epoch)] = {
            "top_k": {
                name: _top_metrics({item: score[item] for item in eligible_target}, eligible_target)
                for name, score in scores.items()
            },
            "eligibility": {
                "cohort": "strong_ce_pgd20_anchor_current_robust_correct",
                "n": len(eligible_target),
                "ff_count": sum(eligible_target.values()),
            },
            "ce20_vs_online": {
                "pearson": {
                    "strong_student_vs_online_margin": _pearson(
                        scores["strong_student_margin_risk"], scores["online_margin_ema_risk"]
                    ),
                    "teacher_dominance_vs_online_frequency": _pearson(
                        scores["teacher_signed_dominance"], scores["online_frequency_risk"]
                    ),
                },
                "spearman": {
                    "strong_student_vs_online_margin": _spearman(
                        scores["strong_student_margin_risk"], scores["online_margin_ema_risk"]
                    )
                },
                "student_correctness_agreement": sum(
                    bool(margins[item]["student_robust_correct"])
                    == bool(online[epoch][item]["current_correct"])
                    for item in target
                )
                / len(target),
                "student_wrong_precision": (
                    sum(
                        not bool(margins[item]["student_robust_correct"])
                        and not bool(online[epoch][item]["current_correct"])
                        for item in target
                    )
                    / max(1, sum(not bool(online[epoch][item]["current_correct"]) for item in target))
                ),
                "student_wrong_recall": (
                    sum(
                        not bool(margins[item]["student_robust_correct"])
                        and not bool(online[epoch][item]["current_correct"])
                        for item in target
                    )
                    / max(1, sum(not bool(margins[item]["student_robust_correct"]) for item in target))
                ),
                "agreement": _agreement(scores["strong_student_margin_risk"], scores["online_margin_ema_risk"]),
            },
            "state_cells": {
                "strong_oracle": _state_cells(strong_student, teacher_state, target),
                "online": _state_cells(online_student, teacher_state, target),
            },
            "s1_teacher_diagnostic": {
                teacher_state_name: {
                    "n": sum(
                        strong_student[item] == "S1" and teacher_state[item] == teacher_state_name for item in target
                    ),
                    "future_failure_rate": (
                        sum(
                            target[item]
                            for item in target
                            if strong_student[item] == "S1" and teacher_state[item] == teacher_state_name
                        )
                        / max(
                            1,
                            sum(
                                strong_student[item] == "S1" and teacher_state[item] == teacher_state_name
                                for item in target
                            ),
                        )
                    ),
                    "teacher_clean_correct_n": sum(
                        strong_student[item] == "S1"
                        and teacher_state[item] == teacher_state_name
                        and bool(margins[item]["teacher_clean_correct"])
                        for item in target
                    ),
                    "teacher_clean_wrong_n": sum(
                        strong_student[item] == "S1"
                        and teacher_state[item] == teacher_state_name
                        and not bool(margins[item]["teacher_clean_correct"])
                        for item in target
                    ),
                }
                for teacher_state_name in ("T1", "T2", "T3")
            },
        }
    for start, end in zip(ANCHORS, ANCHORS[1:]):
        for source_name in ("strong_oracle", "online"):
            current, following = states[start][source_name], states[end][source_name]
            for source_state, target_state in sorted({(current[item], following[item]) for item in target}):
                ids = [item for item in target if current[item] == source_state and following[item] == target_state]
                transition_rows.append(
                    {
                        "source": source_name,
                        "from_epoch": start,
                        "to_epoch": end,
                        "from_student_state": source_state[0],
                        "from_teacher_state": source_state[1],
                        "to_student_state": target_state[0],
                        "to_teacher_state": target_state[1],
                        "count": len(ids),
                        "majority_ff_rate": sum(target[item] for item in ids) / len(ids),
                    }
                )
    return {
        "contract": CONTRACT,
        "target": "majority_future_failure_over_189_194_199",
        "state_definition": {
            "S1": "robust_correct and not fragile q10",
            "S2": "robust_correct fragile q10",
            "S3": "robust_wrong",
            "T1": "teacher_adv_correct and not fragile q10",
            "T2": "teacher_adv_correct fragile q10",
            "T3": "teacher_adv_wrong",
        },
        "anchors": anchors,
        "transitions": transition_rows,
        "one_epoch_delayed": {"available": False, "reason": "no registered one-epoch-delayed replay/state artifact"},
        "threshold_calibration": {
            "absolute_transfer": {
                "available": False,
                "reason": "CE20 and KL10 margin domains have no frozen absolute-threshold mapping",
            },
            "quantile_transfer": {
                "available": True,
                "rule": "online current-correct lower-risk q10 fragile state with stable-ID ties",
            },
            "cross_seed_calibrated": {
                "available": False,
                "reason": "fit objective and threshold loss were not preregistered for this read-only diagnostic",
            },
        },
        "cost_accounting": {
            "available": False,
            "reason": "all observations were reused; no comparable new replay wall-clock was measured",
        },
    }


def _factorial_summary(
    paths: Mapping[str, Mapping[str, Path] | None], expected_count: int, oracle_target: Mapping[int, int]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"available": False, "conditions": {}}
    condition_targets: dict[str, dict[int, int]] = {}
    for condition in CONDITIONS:
        configured = paths[condition]
        if configured is None:
            summary["conditions"][condition] = {"available": False, "reason": "artifact_not_registered"}
            continue
        observations, lineage = configured["observations"], configured["lineage"]
        if not observations.is_file() or not lineage.is_file():
            summary["conditions"][condition] = {"available": False, "reason": "artifact_missing"}
            continue
        meta = _json(lineage, f"factorial {condition} lineage")
        if meta.get("contract") != "ffnr_attack_factorial_v1" or meta.get("observations_sha256") != sha256_file(
            observations
        ):
            raise ERTOnlineRoutingProxyError("factorial lineage hash/contract drifted")
        if meta.get("condition") != condition:
            raise ERTOnlineRoutingProxyError("factorial condition lineage drifted")
        try:
            import pyarrow.parquet as pq

            rows = pq.read_table(
                observations, columns=["sample_id", "class_id", "epoch", "student_robust_correct"]
            ).to_pylist()
        except Exception as exc:
            raise ERTOnlineRoutingProxyError("factorial observations are unreadable") from exc
        terminal = [row for row in rows if row.get("epoch") in TERMINAL_EPOCHS]
        if len(terminal) != expected_count * len(TERMINAL_EPOCHS):
            raise ERTOnlineRoutingProxyError("factorial terminal coverage drifted")
        by_epoch: dict[int, dict[int, bool]] = {epoch: {} for epoch in TERMINAL_EPOCHS}
        classes: dict[int, int] = {}
        for row in terminal:
            item, epoch, class_id = row.get("sample_id"), row.get("epoch"), row.get("class_id")
            if not isinstance(item, int) or epoch not in by_epoch or item in by_epoch[epoch]:
                raise ERTOnlineRoutingProxyError("factorial terminal ID/epoch drifted")
            by_epoch[epoch][item] = bool(row.get("student_robust_correct"))
            if item in classes and classes[item] != class_id:
                raise ERTOnlineRoutingProxyError("factorial stable-ID/class drifted")
            classes[item] = class_id
        if set(classes) != set(oracle_target) or any(set(by_epoch[epoch]) != set(oracle_target) for epoch in TERMINAL_EPOCHS):
            raise ERTOnlineRoutingProxyError("factorial terminal stable-ID universe drifted")
        condition_targets[condition] = {
            item: int(sum(not by_epoch[epoch][item] for epoch in TERMINAL_EPOCHS) >= 2) for item in oracle_target
        }
        target = condition_targets[condition]
        intersection = sum(target[item] and oracle_target[item] for item in target)
        union = sum(target[item] or oracle_target[item] for item in target)
        summary["available"] = True
        summary["conditions"][condition] = {
            "available": True,
            "row_count": len(terminal),
            "future_failure_count": sum(target.values()),
            "future_failure_rate": sum(target.values()) / len(target),
            "agreement_with_ce20_oracle": {
                "intersection": intersection,
                "jaccard": intersection / union if union else 1.0,
                "oracle_jaccard_denominator": sum(oracle_target.values()),
            },
            "lineage_sha256": sha256_file(lineage),
        }
    available = sorted(condition_targets)
    summary["pairwise_jaccard"] = {
        f"{left}__{right}": (
            sum(condition_targets[left][item] and condition_targets[right][item] for item in oracle_target)
            / max(
                1,
                sum(condition_targets[left][item] or condition_targets[right][item] for item in oracle_target),
            )
        )
        for index, left in enumerate(available)
        for right in available[index + 1 :]
    }
    return summary


def run_proxy(*, config_path: Path, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists():
        raise ERTOnlineRoutingProxyError("refusing to overwrite proxy output directory")
    config, provenance = load_config(config_path), _provenance()
    output_dir.mkdir(parents=True, exist_ok=False)
    reports, inputs = {}, {}
    for label in LABELS:
        source = config["runs"][label]["source"]
        feature_meta = _strong_lineage(
            path=source["feature_lineage"],
            observations=source["feature_observations"],
            role="feature",
            expected_count=config["expected_count"],
            expected_universe_sha256=EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
        )
        outcome_meta = _strong_lineage(
            path=source["outcome_lineage"],
            observations=source["outcome_observations"],
            role="outcome",
            expected_count=config["expected_count"],
            expected_universe_sha256=EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
        )
        for key in ("run_id", "teacher", "dataset_identity", "saved_resolved_config_mapping_sha256"):
            if feature_meta.get(key) != outcome_meta.get(key):
                raise ERTOnlineRoutingProxyError("feature/outcome lineage identity drifted")
        feature = _read_compact_observations(
            source["feature_observations"], epochs=ANCHORS, expected_count=config["expected_count"], feature=True
        )
        outcome = _read_compact_observations(
            source["outcome_observations"],
            epochs=TERMINAL_EPOCHS,
            expected_count=config["expected_count"],
            feature=False,
        )
        online, online_meta = _online_panel(source["online_states"], source["online_lineage"], config["expected_count"])
        if any(
            feature_meta.get(key)
            != (
                online_meta.get("config_hash")
                if key == "saved_resolved_config_mapping_sha256"
                else online_meta.get(key)
            )
            for key in ("run_id", "teacher", "dataset_identity", "saved_resolved_config_mapping_sha256")
        ):
            raise ERTOnlineRoutingProxyError("replay/online lineage identity drifted")
        reports[label] = {
            **diagnose(feature=feature, outcome=outcome, online=online),
            "factorial": _factorial_summary(
                config["runs"][label]["factorial"], config["expected_count"], _future_failure(outcome)
            ),
        }
        inputs[label] = {
            "source": {name: sha256_file(path) for name, path in source.items()},
            "factorial": {
                condition: (
                    None
                    if paths is None
                    else {
                        "available": all(path.is_file() for path in paths.values()),
                        "paths": {
                            name: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
                            for name, path in paths.items()
                        },
                    }
                )
                for condition, paths in config["runs"][label]["factorial"].items()
            },
        }
    report = output_dir / "ert-online-routing-proxy-report.json"
    report.write_text(
        json.dumps(
            {"schema_version": 1, "contract": CONTRACT, "analysis_provenance": provenance, "reports": reports},
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lineage = output_dir / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": CONTRACT,
                "config_sha256": sha256_file(config_path),
                "analysis_provenance": provenance,
                "inputs": inputs,
                "report_sha256": sha256_file(report),
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"report": report, "lineage": lineage}
