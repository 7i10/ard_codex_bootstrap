"""Read-only H4a sample taxonomy on schema-v2 common-PGD replay panels."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.rslad_signal_replay import (
    FEATURE_EPOCHS,
    OBSERVATION_SCHEMA_VERSION,
    OUTCOME_EPOCHS,
    canonical_json,
    repository_root_from_source,
)
from ard.analysis.signal_audit import sha256_file


class H4aTaxonomyError(ValueError):
    """An H4a report input violates the fixed replay contract."""


EARLY_ANCHORS = (39, 59, 79)
LATE_ANCHOR = 99
BLINDED_EXEMPLARS_PER_SIDE = 10
_BASE_COLUMNS = {
    "namespace",
    "sample_id",
    "class_id",
    "epoch",
    "observation_schema_version",
    "teacher_entropy_normalized",
    "student_probability_margin",
    "student_margin_risk",
    "robust_correct",
    "teacher_clean_prediction",
    "teacher_clean_correct",
    "teacher_clean_true_probability",
    "teacher_clean_max_wrong_probability",
    "teacher_clean_wrong_confidence",
    "teacher_clean_probability_margin",
    "teacher_clean_entropy_normalized",
    "teacher_adversarial_prediction",
    "teacher_adversarial_correct",
    "teacher_adversarial_true_probability",
    "teacher_adversarial_max_wrong_probability",
    "teacher_adversarial_wrong_confidence",
    "teacher_adversarial_probability_margin",
    "teacher_adversarial_entropy_normalized",
    "teacher_clean_to_adversarial_prediction_flip",
    "teacher_clean_to_adversarial_true_probability_delta",
    "teacher_clean_to_adversarial_margin_delta",
    "student_clean_prediction",
    "student_clean_correct",
    "student_clean_probability_margin",
}
PRIMARY_GROUPS = (
    "stable_correct",
    "future_forgetting",
    "persistent_wrong",
    "recovered_stable",
    "recovered_relapsed",
)


def _tracked_clean_provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "h4a_taxonomy": Path(__file__).resolve(),
        "h4a_taxonomy_cli": root / "src/ard/cli/h4a_taxonomy.py",
        "rslad_signal_replay": root / "src/ard/analysis/rslad_signal_replay.py",
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
        raise H4aTaxonomyError("H4a requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise H4aTaxonomyError("H4a requires a tracked-clean revision")
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": {name: sha256_file(path) for name, path in paths.items()},
    }


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H4aTaxonomyError("lineage is unreadable") from exc
    if not isinstance(value, dict):
        raise H4aTaxonomyError("lineage must be a mapping")
    return value


def _parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:
        raise H4aTaxonomyError("observation panel is unreadable") from exc
    if not _BASE_COLUMNS.issubset(table.column_names):
        raise H4aTaxonomyError("schema-v2 observation column contract drifted")
    return [dict(row) for row in table.to_pylist()]


def _integer(value: object, *, name: str, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (upper is not None and value >= upper):
        raise H4aTaxonomyError(f"{name} is outside its integer range")
    return value


def _float(value: object, *, name: str, lower: float, upper: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lower <= float(value) <= upper
    ):
        raise H4aTaxonomyError(f"{name} is outside its numeric range")
    return float(value)


def _bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise H4aTaxonomyError(f"{name} must be boolean")
    return value


def _close(left: float, right: float, *, name: str) -> None:
    if not math.isclose(left, right, abs_tol=1e-7, rel_tol=0.0):
        raise H4aTaxonomyError(f"{name} algebra drifted")


def _validated_row(row: Mapping[str, Any], *, expected_count: int) -> dict[str, Any]:
    if row.get("namespace") != "train" or row.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise H4aTaxonomyError("row namespace/schema contract drifted")
    # CIFAR validation uses a stable subset of original train IDs, not a
    # reindexed 0..N-1 range.  Completeness is established by exact count,
    # uniqueness, and cross-epoch ID/class joins in ``_domain_panel``.
    sample_id = _integer(row.get("sample_id"), name="sample ID")
    class_id = _integer(row.get("class_id"), name="class ID", upper=10)
    epoch = _integer(row.get("epoch"), name="epoch")
    robust_correct = _bool(row.get("robust_correct"), name="robust correctness")
    margin = _float(row.get("student_probability_margin"), name="student robust margin", lower=-1, upper=1)
    risk = _float(row.get("student_margin_risk"), name="student robust risk", lower=0, upper=1)
    _close(risk, (1 - margin) / 2, name="student robust risk")
    clean_prediction = _integer(row.get("student_clean_prediction"), name="student clean prediction", upper=10)
    clean_correct = _bool(row.get("student_clean_correct"), name="student clean correctness")
    if clean_correct != (clean_prediction == class_id):
        raise H4aTaxonomyError("student clean correctness/prediction drifted")
    _float(row.get("student_clean_probability_margin"), name="student clean margin", lower=-1, upper=1)
    result = {
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "robust_correct": robust_correct,
        "student_margin": margin,
        "student_clean_correct": clean_correct,
    }
    for domain in ("teacher_clean", "teacher_adversarial"):
        prediction = _integer(row.get(f"{domain}_prediction"), name=f"{domain} prediction", upper=10)
        correct = _bool(row.get(f"{domain}_correct"), name=f"{domain} correctness")
        true = _float(row.get(f"{domain}_true_probability"), name=f"{domain} true probability", lower=0, upper=1)
        wrong = _float(
            row.get(f"{domain}_max_wrong_probability"), name=f"{domain} max wrong probability", lower=0, upper=1
        )
        wrong_confidence = _float(
            row.get(f"{domain}_wrong_confidence"), name=f"{domain} wrong confidence", lower=-1, upper=1
        )
        probability_margin = _float(
            row.get(f"{domain}_probability_margin"), name=f"{domain} probability margin", lower=-1, upper=1
        )
        _float(row.get(f"{domain}_entropy_normalized"), name=f"{domain} entropy", lower=0, upper=1)
        if correct != (prediction == class_id):
            raise H4aTaxonomyError(f"{domain} correctness/prediction drifted")
        _close(wrong_confidence, wrong - true, name=f"{domain} wrong confidence")
        _close(probability_margin, true - wrong, name=f"{domain} probability margin")
        result[f"{domain}_correct"] = correct
        result[f"{domain}_wrong_confidence"] = wrong_confidence
        result[f"{domain}_margin"] = probability_margin
    flip = _bool(row.get("teacher_clean_to_adversarial_prediction_flip"), name="teacher prediction flip")
    if flip != (row["teacher_clean_prediction"] != row["teacher_adversarial_prediction"]):
        raise H4aTaxonomyError("teacher prediction flip drifted")
    clean_true = _float(
        row.get("teacher_clean_true_probability"), name="teacher clean true probability", lower=0, upper=1
    )
    adv_true = _float(
        row.get("teacher_adversarial_true_probability"), name="teacher adversarial true probability", lower=0, upper=1
    )
    _close(
        _float(
            row.get("teacher_clean_to_adversarial_true_probability_delta"),
            name="teacher true probability delta",
            lower=-1,
            upper=1,
        ),
        adv_true - clean_true,
        name="teacher true probability delta",
    )
    _close(
        _float(
            row.get("teacher_clean_to_adversarial_margin_delta"),
            name="teacher margin delta",
            lower=-2,
            upper=2,
        ),
        result["teacher_adversarial_margin"] - result["teacher_clean_margin"],
        name="teacher margin delta",
    )
    entropy = _float(row.get("teacher_entropy_normalized"), name="teacher entropy", lower=0, upper=1)
    _close(
        entropy,
        _float(row.get("teacher_adversarial_entropy_normalized"), name="teacher adv entropy", lower=0, upper=1),
        name="teacher entropy",
    )
    result["teacher_prediction_flip"] = flip
    return result


def _domain_panel(
    rows: Sequence[Mapping[str, Any]], *, epochs: Sequence[int], expected_count: int, name: str
) -> dict[int, dict[int, dict[str, Any]]]:
    result = {epoch: {} for epoch in epochs}
    for raw in rows:
        row = _validated_row(raw, expected_count=expected_count)
        if row["epoch"] not in result:
            raise H4aTaxonomyError(f"{name} epoch schedule drifted")
        if row["sample_id"] in result[row["epoch"]]:
            raise H4aTaxonomyError(f"{name} duplicate stable sample ID")
        result[row["epoch"]][row["sample_id"]] = row
    if any(len(panel) != expected_count for panel in result.values()):
        raise H4aTaxonomyError(f"{name} lacks exact stable-ID coverage")
    reference = result[epochs[0]]
    for panel in result.values():
        if set(panel) != set(reference) or any(
            panel[sid]["class_id"] != reference[sid]["class_id"] for sid in reference
        ):
            raise H4aTaxonomyError(f"{name} stable ID/class contract drifted")
    return result


def _lineage(path: Path, observations: Path, *, key: str, expected_count: int, protocol: str) -> dict[str, Any]:
    lineage = _json(path)
    if (
        lineage.get("schema_version") != 1
        or lineage.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION
        or lineage.get("train_expected_count") != expected_count
        or lineage.get(key) != sha256_file(observations)
        or not isinstance(lineage.get(protocol), Mapping)
    ):
        raise H4aTaxonomyError("lineage byte/schema/protocol binding drifted")
    for identity in ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher"):
        if identity not in lineage:
            raise H4aTaxonomyError("lineage identity is incomplete")
    return lineage


def _ids_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(canonical_json(sorted(ids))).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _margin_trends(panel: Mapping[int, Mapping[int, Mapping[str, Any]]], *, anchor: int) -> dict[int, float] | None:
    before = [epoch for epoch in sorted(panel) if epoch < anchor]
    if len(before) < 2:
        return None
    mean_epoch = sum(before) / len(before)
    denominator = sum((epoch - mean_epoch) ** 2 for epoch in before)
    return {
        sample_id: sum((epoch - mean_epoch) * float(panel[epoch][sample_id]["student_margin"]) for epoch in before)
        / denominator
        for sample_id in panel[anchor]
    }


def _primary_groups(panel: Mapping[int, Mapping[int, Mapping[str, Any]]], *, anchor: int) -> dict[str, set[int]]:
    epochs = [epoch for epoch in sorted(panel) if epoch >= anchor]
    if not epochs or epochs[0] != anchor:
        raise H4aTaxonomyError("domain lacks its requested anchor")
    groups = {name: set() for name in PRIMARY_GROUPS}
    for sample_id, row in panel[anchor].items():
        correctness = [bool(panel[epoch][sample_id]["robust_correct"]) for epoch in epochs]
        if correctness[0]:
            groups["future_forgetting" if not all(correctness) else "stable_correct"].add(sample_id)
            continue
        try:
            first_recovery = correctness.index(True)
        except ValueError:
            groups["persistent_wrong"].add(sample_id)
            continue
        groups["recovered_relapsed" if not all(correctness[first_recovery:]) else "recovered_stable"].add(sample_id)
    if set().union(*groups.values()) != set(panel[anchor]) or sum(len(ids) for ids in groups.values()) != len(
        panel[anchor]
    ):
        raise H4aTaxonomyError("primary taxonomy is not exhaustive/disjoint")
    return groups


def _group_report(
    *,
    ids: set[int],
    anchor_rows: Mapping[int, Mapping[str, Any]],
    endpoint_rows: Mapping[int, Mapping[str, Any]],
    expected_count: int,
    trends: Mapping[int, float] | None,
) -> dict[str, Any]:
    chosen = [anchor_rows[sample_id] for sample_id in sorted(ids)]
    target_error = sum(not bool(endpoint_rows[sample_id]["robust_correct"]) for sample_id in ids)
    classes: dict[str, int] = defaultdict(int)
    for row in chosen:
        classes[str(row["class_id"])] += 1
    cross_tabs = {
        "teacher_adversarial": {
            "correct": sum(bool(row["teacher_adversarial_correct"]) for row in chosen),
            "wrong": sum(not bool(row["teacher_adversarial_correct"]) for row in chosen),
        },
        "teacher_clean": {
            "correct": sum(bool(row["teacher_clean_correct"]) for row in chosen),
            "wrong": sum(not bool(row["teacher_clean_correct"]) for row in chosen),
        },
        "student_clean_correct_robust_wrong": sum(
            bool(row["student_clean_correct"]) and not bool(row["robust_correct"]) for row in chosen
        ),
        "teacher_clean_to_adversarial_prediction_flip": {
            "true": sum(bool(row["teacher_prediction_flip"]) for row in chosen),
            "false": sum(not bool(row["teacher_prediction_flip"]) for row in chosen),
        },
    }
    return {
        "count": len(chosen),
        "stable_sample_ids_sha256": _ids_hash(list(ids)),
        "class_counts": dict(sorted(classes.items(), key=lambda item: int(item[0]))),
        "cross_tabs": cross_tabs,
        "teacher_continuous": {
            "adversarial_wrong_confidence": _summary(
                [float(row["teacher_adversarial_wrong_confidence"]) for row in chosen]
            ),
            "adversarial_probability_margin": _summary([float(row["teacher_adversarial_margin"]) for row in chosen]),
            "clean_wrong_confidence": _summary([float(row["teacher_clean_wrong_confidence"]) for row in chosen]),
            "clean_probability_margin": _summary([float(row["teacher_clean_margin"]) for row in chosen]),
        },
        "pre_anchor_student_margin_trend": (
            {"available": False, "summary": None}
            if trends is None
            else {"available": True, "summary": _summary([float(trends[sample_id]) for sample_id in ids])}
        ),
        "same_panel_oracle_headroom": {
            "target": "endpoint_robust_error",
            "target_error_count_in_group": target_error,
            "denominator_train_panel": expected_count,
            "coverage_pct_of_train_panel": 100.0 * target_error / expected_count,
        },
    }


def _taxonomy(
    *,
    panel: Mapping[int, Mapping[int, Mapping[str, Any]]],
    anchors: Sequence[int],
    expected_count: int,
    name: str,
) -> tuple[dict[str, Any], dict[int, dict[str, set[int]]]]:
    output: dict[str, Any] = {"domain": name, "anchors": {}}
    ids_by_anchor: dict[int, dict[str, set[int]]] = {}
    endpoint = panel[max(panel)]
    for anchor in anchors:
        groups = _primary_groups(panel, anchor=anchor)
        ids_by_anchor[anchor] = groups
        trends = _margin_trends(panel, anchor=anchor)
        output["anchors"][str(anchor)] = {
            "endpoint_epoch": max(panel),
            "primary_groups": {
                group: _group_report(
                    ids=groups[group],
                    anchor_rows=panel[anchor],
                    endpoint_rows=endpoint,
                    expected_count=expected_count,
                    trends=trends,
                )
                for group in PRIMARY_GROUPS
            },
        }
    return output, ids_by_anchor


def _jaccard(left: set[int], right: set[int]) -> float | None:
    union = left | right
    return None if not union else len(left & right) / len(union)


def _cross_seed_jaccard(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_teacher: dict[str, list[str]] = defaultdict(list)
    for label, report in reports.items():
        by_teacher[str(report["input_identity"]["teacher_registry_id"])].append(label)
    result: dict[str, Any] = {}
    for teacher, labels in sorted(by_teacher.items()):
        labels = sorted(labels)
        if len(labels) < 2:
            result[teacher] = {"status": "unavailable_fewer_than_two_seeds", "pairs": []}
            continue
        pairs = []
        for index, left in enumerate(labels):
            for right in labels[index + 1 :]:
                values: dict[str, Any] = {}
                for domain in ("early", "late"):
                    for anchor, groups in reports[left]["_group_ids"][domain].items():
                        for group, ids in groups.items():
                            values[f"{domain}:epoch{anchor}:{group}"] = _jaccard(
                                ids, reports[right]["_group_ids"][domain][anchor][group]
                            )
                pairs.append({"left": left, "right": right, "groups": values})
        result[teacher] = {"status": "available", "pairs": pairs}
    return result


def _blinded_panel(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, report in sorted(reports.items()):
        for domain, panel, anchors in (
            ("early", report["_feature_panel"], EARLY_ANCHORS),
            ("late", report["_outcome_panel"], (LATE_ANCHOR,)),
        ):
            for anchor in anchors:
                for group, ids in report["_group_ids"][domain][anchor].items():
                    ordered = sorted(
                        ids,
                        key=lambda sample_id: (
                            float(panel[anchor][sample_id]["teacher_adversarial_wrong_confidence"]),
                            sample_id,
                        ),
                    )
                    selections = (
                        ("bottom", ordered[:BLINDED_EXEMPLARS_PER_SIDE]),
                        ("top", ordered[-BLINDED_EXEMPLARS_PER_SIDE:]),
                    )
                    for side, sample_ids in selections:
                        for rank, sample_id in enumerate(sample_ids, start=1):
                            rows.append(
                                {
                                    "diagnostic_only": True,
                                    "run_label": label,
                                    "domain": domain,
                                    "anchor_epoch": anchor,
                                    "primary_group": group,
                                    "side": side,
                                    "rank": rank,
                                    "sample_id": sample_id,
                                    "teacher_adversarial_wrong_confidence": float(
                                        panel[anchor][sample_id]["teacher_adversarial_wrong_confidence"]
                                    ),
                                }
                            )
    return rows


def analyze_h4a_taxonomy(
    *,
    feature_observations: Path,
    outcome_observations: Path,
    feature_lineage: Path,
    outcome_lineage: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    for key in ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher"):
        if feature_meta[key] != outcome_meta[key]:
            raise H4aTaxonomyError("feature/outcome lineage identity drifted")
    feature = _domain_panel(
        _parquet(feature_observations), epochs=FEATURE_EPOCHS, expected_count=expected_count, name="feature"
    )
    outcome = _domain_panel(
        _parquet(outcome_observations), epochs=OUTCOME_EPOCHS, expected_count=expected_count, name="outcome"
    )
    if set(feature[LATE_ANCHOR]) != set(outcome[LATE_ANCHOR]) or any(
        feature[LATE_ANCHOR][sample_id]["class_id"] != outcome[LATE_ANCHOR][sample_id]["class_id"]
        for sample_id in feature[LATE_ANCHOR]
    ):
        raise H4aTaxonomyError("feature/outcome epoch99 stable ID/class join drifted")
    early, early_ids = _taxonomy(
        panel=feature, anchors=EARLY_ANCHORS, expected_count=expected_count, name="feature_common_pgd"
    )
    late, late_ids = _taxonomy(
        panel=outcome, anchors=(LATE_ANCHOR,), expected_count=expected_count, name="outcome_common_pgd"
    )
    teacher = feature_meta["teacher"]
    if not isinstance(teacher, Mapping) or not isinstance(teacher.get("registry_id"), str):
        raise H4aTaxonomyError("teacher lineage lacks registry ID")
    return {
        "schema_version": 1,
        "contract": "h4a_schema_v2_common_pgd_taxonomy_v1",
        "diagnostic_only": True,
        "no_routes_or_thresholds": True,
        "input_identity": {
            "run_id": feature_meta["run_id"],
            "config_hash": feature_meta["config_hash"],
            "teacher_registry_id": teacher["registry_id"],
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "feature_attack_domain": feature_meta["feature_protocol"],
            "outcome_attack_domain": outcome_meta["outcome_protocol"],
        },
        "early": early,
        "late": late,
        "analysis_provenance": dict(
            _tracked_clean_provenance() if analysis_provenance is None else analysis_provenance
        ),
        "_group_ids": {"early": early_ids, "late": late_ids},
        "_feature_panel": feature,
        "_outcome_panel": outcome,
    }


def analyze_h4a_collection(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": "h4a_schema_v2_common_pgd_collection_v1",
        "diagnostic_only": True,
        "no_routes_or_thresholds": True,
        "reports": {
            label: {key: value for key, value in report.items() if not key.startswith("_")}
            for label, report in reports.items()
        },
        "cross_seed_jaccard": _cross_seed_jaccard(reports),
        "blinded_panel": _blinded_panel(reports),
    }


def write_h4a_outputs(*, output_dir: Path, collection: Mapping[str, Any]) -> dict[str, Path]:
    paths = {"report": output_dir / "taxonomy.json", "blinded_manifest": output_dir / "blinded-panel.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite H4a taxonomy outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in collection.items() if key != "blinded_panel"}
    paths["report"].write_bytes(canonical_json(report) + b"\n")
    paths["blinded_manifest"].write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "contract": "h4a_blinded_id_manifest_v1",
                "diagnostic_only": True,
                "contains_images": False,
                "contains_label_corrections": False,
                "rows": collection["blinded_panel"],
            }
        )
        + b"\n"
    )
    return paths
