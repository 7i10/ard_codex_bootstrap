"""CPU-only PRE39 replay-domain forecasting and routing audit.

The point screen deliberately builds every routing score from observations at
or before its anchor.  Later common-PGD correctness is used only to name the
PF/NR labels and to describe already-selected masks; it never enters a score
or a mask construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.h4a_taxonomy import (
    PRIMARY_GROUPS,
    _domain_panel,
    _lineage,
    _primary_groups,
)
from ard.analysis.rslad_signal_replay import (
    FEATURE_EPOCHS,
    OUTCOME_EPOCHS,
    PANEL_EMA_BETA,
    canonical_json,
    repository_root_from_source,
)
from ard.analysis.signal_audit import SignalAuditError, _bootstrap_indices, binary_metrics, sha256_file


class Pre39PrescriptiveError(ValueError):
    """The immutable PRE39 replay input or report contract was violated."""


ANCHORS = (4, 9, 14, 19, 24, 29, 34)
OUTCOME_WINDOW = (99, 104, 109)
Q = 0.10
BOOTSTRAP_SEED = 2026080501
BOOTSTRAP_REPLICATES = 2000
MODEL_NAMES = (
    "student_history",
    "teacher_entropy",
    "student_teacher_additive",
    "student_teacher_product",
    "instantaneous_margin",
)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except Exception as exc:  # pragma: no cover - Arrow has version-specific errors
        raise Pre39PrescriptiveError("replay observations are unreadable") from exc


def _provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "pre39_prescriptive": Path(__file__).resolve(),
        "pre39_prescriptive_cli": root / "src/ard/cli/pre39_prescriptive.py",
        "h4a_taxonomy": root / "src/ard/analysis/h4a_taxonomy.py",
        "signal_audit": root / "src/ard/analysis/signal_audit.py",
    }
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
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
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise Pre39PrescriptiveError("PRE39 analysis requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise Pre39PrescriptiveError("PRE39 analysis requires a tracked-clean revision")
    source_files = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": source_files}


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return (
        {"count": 0, "mean": None, "min": None, "max": None}
        if not values
        else {"count": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    )


def _midrank(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        raise Pre39PrescriptiveError("cannot rank an empty population")
    ordered = sorted((float(value), int(sample_id)) for sample_id, value in values.items())
    result: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank = (start + (end - start) / 2.0) / len(ordered)
        for _, sample_id in ordered[start:end]:
            result[sample_id] = rank
        start = end
    return result


def _history_values(
    panel: Mapping[int, Mapping[int, Mapping[str, Any]]], *, anchor: int, inclusive: bool
) -> dict[str, dict[int, float]] | None:
    epochs = (
        tuple(epoch for epoch in FEATURE_EPOCHS if epoch <= anchor)
        if inclusive
        else tuple(epoch for epoch in FEATURE_EPOCHS if epoch < anchor)
    )
    if not epochs:
        return None
    if inclusive and epochs[-1] != anchor:
        raise Pre39PrescriptiveError("anchor is outside the frozen feature schedule")
    frequency_risk: dict[int, float] = {}
    margin_ema_risk: dict[int, float] = {}
    for sample_id in panel[anchor]:
        hits = sum(bool(panel[epoch][sample_id]["robust_correct"]) for epoch in epochs)
        ema: float | None = None
        for epoch in epochs:
            value = float(panel[epoch][sample_id]["student_margin"])
            # This is the frozen replay-panel EMA across observed sparse
            # checkpoints, not the exact online per-training-epoch EMA.
            ema = value if ema is None else PANEL_EMA_BETA * ema + (1 - PANEL_EMA_BETA) * value
        frequency_risk[sample_id] = 1 - hits / len(epochs)
        margin_ema_risk[sample_id] = (1 - float(ema)) / 2
    return {"frequency_risk": frequency_risk, "margin_ema_risk": margin_ema_risk}


def _scores(
    panel: Mapping[int, Mapping[int, Mapping[str, Any]]], *, anchor: int, inclusive: bool
) -> dict[str, dict[int, float]] | None:
    history = _history_values(panel, anchor=anchor, inclusive=inclusive)
    if history is None:
        return None
    frequency = _midrank(history["frequency_risk"])
    historical_margin = _midrank(history["margin_ema_risk"])
    student = {sample_id: (frequency[sample_id] + historical_margin[sample_id]) / 2 for sample_id in panel[anchor]}
    # Entropy is already normalized to [0, 1]; unlike the student composite
    # it remains its recorded primitive in the teacher-only comparator.
    teacher = {sample_id: float(row["teacher_entropy_normalized"]) for sample_id, row in panel[anchor].items()}
    instantaneous = _midrank(
        {sample_id: (1 - float(row["student_margin"])) / 2 for sample_id, row in panel[anchor].items()}
    )
    additive = {sample_id: (student[sample_id] + teacher[sample_id]) / 2 for sample_id in student}
    product = {
        sample_id: (student[sample_id] + teacher[sample_id] + student[sample_id] * teacher[sample_id]) / 3
        for sample_id in student
    }
    return {
        "student_history": student,
        "teacher_entropy": teacher,
        "student_teacher_additive": additive,
        "student_teacher_product": product,
        "instantaneous_margin": instantaneous,
    }


def _top_q(scores: Mapping[int, float]) -> set[int]:
    count = len(scores)
    if not count:
        return set()
    k = max(1, math.floor(Q * count))
    return {sample_id for sample_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]}


def _metrics(scores: Mapping[int, float], labels: Mapping[int, int]) -> dict[str, Any]:
    ordered = sorted(labels)
    result: dict[str, Any] = {
        "count": len(ordered),
        "prevalence": sum(labels.values()) / len(ordered) if ordered else None,
    }
    chosen = _top_q({sample_id: scores[sample_id] for sample_id in ordered})
    positives = sum(labels.values())
    result.update(
        {
            "q": Q,
            "top_q_count": len(chosen),
            "top_q_sample_ids_sha256": _hash(sorted(chosen)),
            "precision_at_q": sum(labels[sample_id] for sample_id in chosen) / len(chosen) if chosen else None,
            "recall_at_q": sum(labels[sample_id] for sample_id in chosen) / positives if positives else None,
        }
    )
    if not ordered or positives in {0, len(ordered)}:
        result.update({"auroc": None, "auprc": None, "log_loss": None})
        return result
    values = binary_metrics([labels[sample_id] for sample_id in ordered], [scores[sample_id] for sample_id in ordered])
    result.update(values)
    return result


def _overlap(scores: Mapping[str, Mapping[int, float]], eligible: set[int]) -> dict[str, Any]:
    selected = {
        name: _top_q({sample_id: value for sample_id, value in score.items() if sample_id in eligible})
        for name, score in scores.items()
    }
    return {
        "q": Q,
        "masks": {name: {"count": len(ids), "sample_ids_sha256": _hash(sorted(ids))} for name, ids in selected.items()},
        "jaccard": {
            left: {
                right: (
                    len(selected[left] & selected[right]) / len(selected[left] | selected[right])
                    if selected[left] | selected[right]
                    else None
                )
                for right in MODEL_NAMES
            }
            for left in MODEL_NAMES
        },
    }


def _trend(panel: Mapping[int, Mapping[int, Mapping[str, Any]]], sample_id: int, *, anchor: int) -> float | None:
    epochs = [epoch for epoch in FEATURE_EPOCHS if epoch <= anchor]
    if len(epochs) < 2:
        return None
    mean_epoch = sum(epochs) / len(epochs)
    denominator = sum((epoch - mean_epoch) ** 2 for epoch in epochs)
    return (
        sum((epoch - mean_epoch) * float(panel[epoch][sample_id]["student_margin"]) for epoch in epochs) / denominator
    )


def _routing_fields(
    *,
    ids: set[int],
    feature: Mapping[int, Mapping[int, Mapping[str, Any]]],
    outcome: Mapping[int, Mapping[int, Mapping[str, Any]]],
    anchor: int,
    stratum: str,
    stratum_count: int,
    labels: Mapping[int, int],
    primary_groups: Mapping[str, set[int]],
) -> dict[str, Any]:
    rows = [feature[anchor][sample_id] for sample_id in sorted(ids)]
    history = _history_values(feature, anchor=anchor, inclusive=True)
    assert history is not None
    group_mass = {name: sum(sample_id in group for sample_id in ids) for name, group in primary_groups.items()}
    endpoint = max(outcome)
    endpoint_wrong = sum(not bool(outcome[endpoint][sample_id]["robust_correct"]) for sample_id in ids)
    return {
        "count": len(rows),
        "selected_mask_fraction_of_stratum": len(rows) / stratum_count if stratum_count else None,
        "sample_ids_sha256": _hash(sorted(ids)),
        "outcome_positive_count": sum(labels[sample_id] for sample_id in ids),
        "teacher": {
            "clean_correct": sum(bool(row["teacher_clean_correct"]) for row in rows),
            "adversarial_correct": sum(bool(row["teacher_adversarial_correct"]) for row in rows),
            "clean_probability_margin": _summary([float(row["teacher_clean_margin"]) for row in rows]),
            "adversarial_probability_margin": _summary([float(row["teacher_adversarial_margin"]) for row in rows]),
            "clean_wrong_confidence": _summary([float(row["teacher_clean_wrong_confidence"]) for row in rows]),
            "adversarial_wrong_confidence": _summary(
                [float(row["teacher_adversarial_wrong_confidence"]) for row in rows]
            ),
            "clean_to_adversarial_margin_delta": _summary(
                [float(row["teacher_adversarial_margin"]) - float(row["teacher_clean_margin"]) for row in rows]
            ),
            "clean_to_adversarial_true_probability_delta": _summary(
                [
                    float(row["teacher_adversarial_true_probability"]) - float(row["teacher_clean_true_probability"])
                    for row in rows
                ]
            ),
            "clean_to_adversarial_prediction_flip": {
                "true": sum(bool(row["teacher_prediction_flip"]) for row in rows),
                "false": sum(not bool(row["teacher_prediction_flip"]) for row in rows),
            },
        },
        "student": {
            "clean_correct_robust_wrong": sum(
                bool(row["student_clean_correct"]) and not bool(row["robust_correct"]) for row in rows
            ),
            "robust_margin": _summary([float(row["student_margin"]) for row in rows]),
            "inclusive_correctness_frequency": _summary(
                [1 - history["frequency_risk"][int(row["sample_id"])] for row in rows]
            ),
            "margin_trend": _summary(
                [value for sample_id in ids if (value := _trend(feature, sample_id, anchor=anchor)) is not None]
            ),
        },
        "future_taxonomy_audit_only": {
            "masses": {
                "PF": sum(labels[sample_id] for sample_id in ids) if stratum == "PF" else 0,
                "NR": sum(labels[sample_id] for sample_id in ids) if stratum == "NR" else 0,
                **group_mass,
            },
            "definitions": ["PF", "NR", *PRIMARY_GROUPS],
        },
        "same_panel_oracle_headroom_audit_only": {
            "endpoint_epoch": endpoint,
            "endpoint_robust_error_count": endpoint_wrong,
            "coverage_pct_of_selected_mask": 100 * endpoint_wrong / len(rows) if rows else None,
        },
    }


def _validate_inputs(
    *,
    feature_observations: Path,
    outcome_observations: Path,
    feature_lineage: Path,
    outcome_lineage: Path,
    expected_count: int,
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, dict[int, dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise Pre39PrescriptiveError("expected_count must be positive")
    try:
        feature_meta = _lineage(
            feature_lineage,
            feature_observations,
            key="feature_observations_sha256",
            expected_count=expected_count,
            protocol="feature_protocol",
        )
        outcome_meta = _lineage(
            outcome_lineage,
            outcome_observations,
            key="outcome_observations_sha256",
            expected_count=expected_count,
            protocol="outcome_protocol",
        )
        feature_raw = _read_parquet(feature_observations)
        outcome_raw = _read_parquet(outcome_observations)
        feature = _domain_panel(feature_raw, epochs=FEATURE_EPOCHS, expected_count=expected_count, name="feature")
        outcome = _domain_panel(outcome_raw, epochs=OUTCOME_EPOCHS, expected_count=expected_count, name="outcome")
    except ValueError as exc:
        raise Pre39PrescriptiveError(str(exc)) from exc
    keys = ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher")
    if any(feature_meta[key] != outcome_meta[key] for key in keys):
        raise Pre39PrescriptiveError("feature/outcome lineage identity drifted")
    if set(feature[99]) != set(outcome[99]) or any(
        feature[99][sample_id]["class_id"] != outcome[99][sample_id]["class_id"] for sample_id in feature[99]
    ):
        raise Pre39PrescriptiveError("feature/outcome stable-ID/class join drifted")
    # H4a's validated compact rows intentionally omit entropy because its
    # taxonomy does not consume it.  PRE39 does, so retain this frozen schema
    # primitive after the full H4a validation above.
    for raw_rows, panel, name in ((feature_raw, feature, "feature"), (outcome_raw, outcome, "outcome")):
        for row in raw_rows:
            entropy = row.get("teacher_entropy_normalized")
            if (
                isinstance(entropy, bool)
                or not isinstance(entropy, (int, float))
                or not math.isfinite(float(entropy))
                or not 0 <= float(entropy) <= 1
            ):
                raise Pre39PrescriptiveError(f"{name} teacher entropy is outside contract")
            compact = panel[int(row["epoch"])][int(row["sample_id"])]
            compact["teacher_entropy_normalized"] = float(entropy)
            compact["teacher_clean_true_probability"] = float(row["teacher_clean_true_probability"])
            compact["teacher_adversarial_true_probability"] = float(row["teacher_adversarial_true_probability"])
    return feature, outcome, feature_meta, outcome_meta


def analyze_pre39_prescriptive(
    *,
    feature_observations: Path,
    outcome_observations: Path,
    feature_lineage: Path,
    outcome_lineage: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce point estimates and routing audit from one hash-bound replay pair."""
    feature, outcome, feature_meta, outcome_meta = _validate_inputs(
        feature_observations=feature_observations,
        outcome_observations=outcome_observations,
        feature_lineage=feature_lineage,
        outcome_lineage=outcome_lineage,
        expected_count=expected_count,
    )
    future_groups = _primary_groups(outcome, anchor=99)
    anchors: dict[str, Any] = {}
    bootstrap_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    mask_ids: dict[str, dict[str, dict[str, set[int]]]] = {}
    for anchor in ANCHORS:
        inclusive = _scores(feature, anchor=anchor, inclusive=True)
        assert inclusive is not None
        lead = _scores(feature, anchor=anchor, inclusive=False)
        anchor_correct = {sample_id for sample_id, row in feature[anchor].items() if bool(row["robust_correct"])}
        anchor_wrong = set(feature[anchor]) - anchor_correct
        pf = {
            sample_id: int(sum(not bool(outcome[epoch][sample_id]["robust_correct"]) for epoch in OUTCOME_WINDOW) >= 2)
            for sample_id in anchor_correct
        }
        nr = {
            sample_id: int(all(not bool(outcome[epoch][sample_id]["robust_correct"]) for epoch in OUTCOME_WINDOW))
            for sample_id in anchor_wrong
        }
        if set(pf) & set(nr):  # defensive; the defining anchor strata must be disjoint
            raise Pre39PrescriptiveError("PF/NR anchor strata overlap")
        strata = {"PF": pf, "NR": nr}
        anchor_report: dict[str, Any] = {
            "inclusive_history_primary": {},
            "prior_anchor_only_lead_time_diagnostic": None,
        }
        bootstrap_rows[str(anchor)] = {}
        mask_ids[str(anchor)] = {}
        for name, labels in strata.items():
            eligible = set(labels)
            models = {model: _metrics(score, labels) for model, score in inclusive.items()}
            selected = {
                model: _top_q({sample_id: score[sample_id] for sample_id in eligible})
                for model, score in inclusive.items()
            }
            mask_ids[str(anchor)][name] = selected
            anchor_report["inclusive_history_primary"][name] = {
                "definition": (
                    "anchor_robust_correct_and_wrong_at_least_two_of_99_104_109"
                    if name == "PF"
                    else "anchor_robust_wrong_and_wrong_all_of_99_104_109"
                ),
                "models": models,
                "top_q_overlap": _overlap(inclusive, eligible),
                "routing_audit": {
                    model: _routing_fields(
                        ids=ids,
                        feature=feature,
                        outcome=outcome,
                        anchor=anchor,
                        stratum=name,
                        stratum_count=len(eligible),
                        labels=labels,
                        primary_groups=future_groups,
                    )
                    for model, ids in selected.items()
                },
            }
            bootstrap_rows[str(anchor)][name] = [
                {
                    "sample_id": sample_id,
                    "class_id": feature[anchor][sample_id]["class_id"],
                    "outcome": labels[sample_id],
                    **{model: inclusive[model][sample_id] for model in MODEL_NAMES},
                }
                for sample_id in sorted(labels)
            ]
        if lead is not None:
            anchor_report["prior_anchor_only_lead_time_diagnostic"] = {
                name: {"models": {model: _metrics(score, labels) for model, score in lead.items()}}
                for name, labels in strata.items()
            }
        anchors[str(anchor)] = anchor_report
    return {
        "schema_version": 1,
        "contract": "pre39_prescriptive_replay_point_v1",
        "cpu_only": True,
        "routing_score_contract": "outcome_free_midrank_aggregation_v1",
        "model_predictors": {
            "student_history": ["inclusive_robust_correct_frequency_risk", "inclusive_margin_ema_risk"],
            "teacher_entropy": ["teacher_entropy_normalized"],
            "student_teacher_additive": ["student_history", "teacher_entropy"],
            "student_teacher_product": ["student_history", "teacher_entropy", "student_history_times_teacher_entropy"],
            "instantaneous_margin": ["anchor_student_margin_risk"],
            "routing_moderators_not_predictors": [
                "teacher_clean_correct",
                "teacher_adversarial_correct",
                "teacher_clean_probability_margin",
                "teacher_adversarial_probability_margin",
            ],
        },
        "anchors": list(ANCHORS),
        "outcome_window": list(OUTCOME_WINDOW),
        "input_identity": {
            "run_id": feature_meta["run_id"],
            "config_hash": feature_meta["config_hash"],
            "scientific_git_sha": feature_meta["scientific_git_sha"],
            "teacher_registry_id": feature_meta["teacher"].get("registry_id"),
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "feature_attack_domain": feature_meta["feature_protocol"],
            "outcome_attack_domain": outcome_meta["outcome_protocol"],
        },
        "anchor_reports": anchors,
        "analysis_provenance": dict(_provenance() if analysis_provenance is None else analysis_provenance),
        "_bootstrap_rows": bootstrap_rows,
        "_mask_ids": mask_ids,
    }


def collect_pre39_reports(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Collect L1--L4 point reports and compute teacher-specific mask overlap."""
    if not reports:
        raise Pre39PrescriptiveError("collection requires at least one run report")
    clean = {
        label: {key: value for key, value in report.items() if not key.startswith("_")}
        for label, report in sorted(reports.items())
    }
    by_teacher: dict[str, list[str]] = defaultdict(list)
    for label, report in reports.items():
        teacher = report.get("input_identity", {}).get("teacher_registry_id")
        if not isinstance(teacher, str) or not teacher:
            raise Pre39PrescriptiveError("report lacks teacher routing identity")
        by_teacher[teacher].append(label)
    cross_seed: dict[str, Any] = {}
    for teacher, labels in sorted(by_teacher.items()):
        pairs = []
        for index, left in enumerate(sorted(labels)):
            for right in sorted(labels)[index + 1 :]:
                rows = {}
                for anchor in ANCHORS:
                    for stratum in ("PF", "NR"):
                        for model in MODEL_NAMES:
                            left_ids = (
                                reports[left].get("_mask_ids", {}).get(str(anchor), {}).get(stratum, {}).get(model)
                            )
                            right_ids = (
                                reports[right].get("_mask_ids", {}).get(str(anchor), {}).get(stratum, {}).get(model)
                            )
                            if not isinstance(left_ids, set) or not isinstance(right_ids, set):
                                raise Pre39PrescriptiveError("cross-seed Jaccard requires in-memory hash-bound masks")
                            union = left_ids | right_ids
                            rows[f"epoch{anchor}:{stratum}:{model}"] = {
                                "left_mask_sha256": _hash(sorted(left_ids)),
                                "right_mask_sha256": _hash(sorted(right_ids)),
                                "jaccard": len(left_ids & right_ids) / len(union) if union else None,
                            }
                pairs.append({"left": left, "right": right, "masks": rows})
        cross_seed[teacher] = {
            "runs": sorted(labels),
            "pairs": pairs,
            "status": "available" if pairs else "unavailable_fewer_than_two_seeds",
        }
    return {
        "schema_version": 1,
        "contract": "pre39_prescriptive_collection_v1",
        "reports": clean,
        "cross_seed_route_prevalence": cross_seed,
    }


def _bootstrap_fingerprint(
    rows: Sequence[Mapping[str, Any]], *, anchor: int, stratum: str, baseline: str, candidate: str
) -> str:
    return _hash(
        {
            "contract": "pre39_paired_bootstrap_v1",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "anchor": anchor,
            "stratum": stratum,
            "baseline": baseline,
            "candidate": candidate,
            "rows": list(rows),
        }
    )


def run_pre39_bootstrap(
    *,
    report: Mapping[str, Any],
    anchor: int,
    stratum: str,
    baseline: str,
    candidate: str,
    output: Path,
    progress: Path,
    max_replicates: int | None = None,
) -> dict[str, Any]:
    """Run one explicit, class-stratified paired 2,000-replicate task."""
    if output.exists():
        raise FileExistsError("refusing to overwrite PRE39 bootstrap output")
    if (
        anchor not in ANCHORS
        or stratum not in {"PF", "NR"}
        or baseline not in MODEL_NAMES
        or candidate not in MODEL_NAMES
        or baseline == candidate
    ):
        raise Pre39PrescriptiveError("bootstrap task identity is invalid")
    rows = report.get("_bootstrap_rows", {}).get(str(anchor), {}).get(stratum)
    if not isinstance(rows, list) or not rows:
        raise Pre39PrescriptiveError("bootstrap rows are unavailable for selected anchor/stratum")
    if any(
        not isinstance(row, Mapping) or set(row) != {"sample_id", "class_id", "outcome", *MODEL_NAMES} for row in rows
    ):
        raise Pre39PrescriptiveError("bootstrap row schema drifted")
    if not any(int(row["outcome"]) for row in rows) or all(int(row["outcome"]) for row in rows):
        raise Pre39PrescriptiveError("bootstrap outcome requires both classes")
    limit = BOOTSTRAP_REPLICATES if max_replicates is None else min(max_replicates, BOOTSTRAP_REPLICATES)
    if limit < 1:
        raise Pre39PrescriptiveError("bootstrap replicate limit must be positive")
    fingerprint = _bootstrap_fingerprint(rows, anchor=anchor, stratum=stratum, baseline=baseline, candidate=candidate)
    state = {"fingerprint": fingerprint, "completed": {}}
    if progress.exists():
        try:
            state = json.loads(progress.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Pre39PrescriptiveError("bootstrap progress is unreadable") from exc
        if state.get("fingerprint") != fingerprint or not isinstance(state.get("completed"), dict):
            raise Pre39PrescriptiveError("bootstrap progress fingerprint mismatch")
    completed = {int(key): value for key, value in state["completed"].items()}
    for replicate in range(limit):
        if replicate in completed:
            continue
        selected = _bootstrap_indices(rows, seed=BOOTSTRAP_SEED, replicate=replicate, cluster=False)
        targets = [int(rows[index]["outcome"]) for index in selected]
        try:
            completed[replicate] = (
                binary_metrics(targets, [float(rows[index][candidate]) for index in selected])["auroc"]
                - binary_metrics(targets, [float(rows[index][baseline]) for index in selected])["auroc"]
            )
        except SignalAuditError:
            completed[replicate] = None
        if len(completed) % 50 == 0:
            _write_atomic(
                progress,
                {
                    "fingerprint": fingerprint,
                    "completed": {str(key): value for key, value in sorted(completed.items())},
                },
            )
    _write_atomic(
        progress,
        {"fingerprint": fingerprint, "completed": {str(key): value for key, value in sorted(completed.items())}},
    )
    deltas = sorted(float(value) for index, value in completed.items() if index < limit and value is not None)
    result = {
        "schema_version": 1,
        "contract": "pre39_paired_bootstrap_v1",
        "anchor": anchor,
        "stratum": stratum,
        "baseline": baseline,
        "candidate": candidate,
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "completed_replicates": len(deltas),
        "partial": limit != BOOTSTRAP_REPLICATES,
        "fingerprint": fingerprint,
        "lower": deltas[max(0, math.floor(0.025 * (len(deltas) - 1)))] if deltas else None,
        "upper": deltas[min(len(deltas) - 1, math.ceil(0.975 * (len(deltas) - 1)))] if deltas else None,
    }
    _write_atomic(output, result)
    return result


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def write_pre39_report(*, output: Path, report: Mapping[str, Any], overwrite: bool = False) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite PRE39 point report")
    public = {key: value for key, value in report.items() if not key.startswith("_")}
    _write_atomic(output, public)
    return output
