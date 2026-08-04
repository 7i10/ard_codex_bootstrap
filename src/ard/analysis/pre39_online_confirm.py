"""Exact-online epoch-34 PRE39 confirmation against frozen replay outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ard.analysis.h4a_taxonomy import _domain_panel, _lineage
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS, canonical_json, repository_root_from_source
from ard.analysis.signal_audit import SignalAuditError, _bootstrap_indices, binary_metrics, sha256_file


class Pre39OnlineConfirmError(ValueError):
    """The exact-online candidate or its frozen replay comparison drifted."""


ANCHOR = 34
OUTCOME_WINDOW = (99, 104, 109)
Q = 0.10
BOOTSTRAP_SEED = 2026080501
BOOTSTRAP_REPLICATES = 2000
ONLINE_COLUMNS = {
    "namespace",
    "sample_id",
    "class_id",
    "anchor_epoch",
    "robust_correct_count",
    "robust_correct_frequency_inclusive",
    "margin_ema",
    "last_margin",
    "current_robust_correct",
}
MODEL_NAMES = ("exact_online_student", "instantaneous_margin", "teacher_entropy")


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except Exception as exc:  # pragma: no cover - Arrow error details vary
        raise Pre39OnlineConfirmError("PRE39 input parquet is unreadable") from exc


def _provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "pre39_online_confirm": Path(__file__).resolve(),
        "pre39_online_confirm_cli": root / "src/ard/cli/pre39_online_confirm.py",
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
        raise Pre39OnlineConfirmError("online confirmation requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise Pre39OnlineConfirmError("online confirmation requires a tracked-clean revision")
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": {name: sha256_file(path) for name, path in paths.items()},
    }


def _finite(value: object, *, name: str, lower: float = -1, upper: float = 1) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lower <= float(value) <= upper
    ):
        raise Pre39OnlineConfirmError(f"{name} is outside contract")
    return float(value)


def _midrank(values: Mapping[int, float]) -> dict[int, float]:
    if not values:
        raise Pre39OnlineConfirmError("cannot score an empty candidate population")
    ordered = sorted((float(value), int(sample_id)) for sample_id, value in values.items())
    result: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank = (start + (end - start) / 2) / len(ordered)
        for _, sample_id in ordered[start:end]:
            result[sample_id] = rank
        start = end
    return result


def _top_q(scores: Mapping[int, float]) -> set[int]:
    if not scores:
        return set()
    count = max(1, math.floor(Q * len(scores)))
    return {sample_id for sample_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:count]}


def _metrics(scores: Mapping[int, float], labels: Mapping[int, int]) -> dict[str, Any]:
    ordered = sorted(labels)
    selected = _top_q({sample_id: scores[sample_id] for sample_id in ordered})
    positives = sum(labels.values())
    result: dict[str, Any] = {
        "count": len(ordered),
        "prevalence": positives / len(ordered) if ordered else None,
        "q": Q,
        "top_q_count": len(selected),
        "top_q_sample_ids_sha256": _hash(sorted(selected)),
        "precision_at_q": sum(labels[sample_id] for sample_id in selected) / len(selected) if selected else None,
        "recall_at_q": sum(labels[sample_id] for sample_id in selected) / positives if positives else None,
    }
    if not ordered or positives in {0, len(ordered)}:
        result.update({"auroc": None, "auprc": None})
    else:
        result.update(binary_metrics([labels[item] for item in ordered], [scores[item] for item in ordered]))
        result.pop("log_loss", None)
    return result


def _overlap(scores: Mapping[str, Mapping[int, float]], eligible: set[int]) -> dict[str, Any]:
    selected = {name: _top_q({sample_id: score[sample_id] for sample_id in eligible}) for name, score in scores.items()}
    return {
        "q": Q,
        "masks": {name: {"count": len(ids), "sample_ids_sha256": _hash(sorted(ids))} for name, ids in selected.items()},
        "jaccard": {
            left: {
                right: len(selected[left] & selected[right]) / len(selected[left] | selected[right])
                if selected[left] | selected[right]
                else None
                for right in MODEL_NAMES
            }
            for left in MODEL_NAMES
        },
    }


def _online_rows(
    path: Path, lineage_path: Path, *, expected_count: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    try:
        meta = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Pre39OnlineConfirmError("online-state lineage is unreadable") from exc
    required = {
        "schema_version",
        "contract",
        "run_id",
        "config_hash",
        "world_size",
        "scientific_git_sha",
        "seed",
        "teacher",
        "dataset_identity",
        "attack_identity",
        "anchor_epoch",
        "expected_count",
        "row_count",
        "checkpoint_sha256",
        "feature_observations_sha256",
        "feature_lineage_sha256",
        "analysis_provenance",
        "observations_sha256",
    }
    if (
        not isinstance(meta, dict)
        or set(meta) != required
        or meta.get("schema_version") != 1
        or meta.get("contract") != "pre39_online_state_candidate_v1"
    ):
        raise Pre39OnlineConfirmError("online-state lineage schema/contract drifted")
    if (
        meta.get("anchor_epoch") != ANCHOR
        or meta.get("expected_count") != expected_count
        or meta.get("row_count") != expected_count
    ):
        raise Pre39OnlineConfirmError("online-state anchor/count contract drifted")
    if not isinstance(meta.get("observations_sha256"), str) or meta["observations_sha256"] != sha256_file(path):
        raise Pre39OnlineConfirmError("online-state observations hash drifted")
    rows = _read_parquet(path)
    if len(rows) != expected_count:
        raise Pre39OnlineConfirmError("online-state row count drifted")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if set(row) != ONLINE_COLUMNS:
            raise Pre39OnlineConfirmError("online-state observation schema drifted")
        sample_id, class_id = row["sample_id"], row["class_id"]
        if (
            row["namespace"] != "train"
            or isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or sample_id < 0
            or sample_id in result
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 10
            or row["anchor_epoch"] != ANCHOR
            or isinstance(row["robust_correct_count"], bool)
            or not isinstance(row["robust_correct_count"], int)
            or not 0 <= row["robust_correct_count"] <= ANCHOR + 1
            or not isinstance(row["current_robust_correct"], bool)
        ):
            raise Pre39OnlineConfirmError("online-state stable-ID/temporal contract drifted")
        frequency = _finite(
            row["robust_correct_frequency_inclusive"], name="online correctness frequency", lower=0, upper=1
        )
        if frequency != row["robust_correct_count"] / (ANCHOR + 1):
            raise Pre39OnlineConfirmError("online-state inclusive count/frequency contract drifted")
        result[sample_id] = {
            **row,
            "robust_correct_frequency_inclusive": frequency,
            "margin_ema": _finite(row["margin_ema"], name="online margin EMA"),
            "last_margin": _finite(row["last_margin"], name="online last margin"),
        }
    return result, meta


def _validated_replay(
    *,
    feature_observations: Path,
    feature_lineage: Path,
    outcome_observations: Path,
    outcome_lineage: Path,
    expected_count: int,
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, dict[int, dict[str, Any]]], dict[str, Any], dict[str, Any]]:
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
        feature_raw, outcome_raw = _read_parquet(feature_observations), _read_parquet(outcome_observations)
        feature = _domain_panel(feature_raw, epochs=FEATURE_EPOCHS, expected_count=expected_count, name="feature")
        outcome = _domain_panel(outcome_raw, epochs=OUTCOME_EPOCHS, expected_count=expected_count, name="outcome")
    except ValueError as exc:
        raise Pre39OnlineConfirmError(str(exc)) from exc
    keys = ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher")
    if any(feature_meta[key] != outcome_meta[key] for key in keys):
        raise Pre39OnlineConfirmError("feature/outcome lineage identity drifted")
    entropy: dict[int, float] = {}
    for row in feature_raw:
        if row["epoch"] == ANCHOR:
            entropy[int(row["sample_id"])] = _finite(
                row.get("teacher_entropy_normalized"), name="teacher entropy", lower=0, upper=1
            )
    if set(entropy) != set(feature[ANCHOR]):
        raise Pre39OnlineConfirmError("feature entropy stable-ID join drifted")
    for sample_id, value in entropy.items():
        feature[ANCHOR][sample_id]["teacher_entropy_normalized"] = value
    return feature, outcome, feature_meta, outcome_meta


def analyze_pre39_online_confirm(
    *,
    online_observations: Path,
    online_lineage: Path,
    feature_observations: Path,
    feature_lineage: Path,
    outcome_observations: Path,
    outcome_lineage: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the immutable epoch-34 online state without future leakage."""
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise Pre39OnlineConfirmError("expected_count must be positive")
    provenance = dict(_provenance() if analysis_provenance is None else analysis_provenance)
    online, online_meta = _online_rows(online_observations, online_lineage, expected_count=expected_count)
    feature, outcome, feature_meta, outcome_meta = _validated_replay(
        feature_observations=feature_observations,
        feature_lineage=feature_lineage,
        outcome_observations=outcome_observations,
        outcome_lineage=outcome_lineage,
        expected_count=expected_count,
    )
    if online_meta["feature_observations_sha256"] != sha256_file(feature_observations) or online_meta[
        "feature_lineage_sha256"
    ] != sha256_file(feature_lineage):
        raise Pre39OnlineConfirmError("online-state feature hash lineage drifted")
    keys = ("run_id", "config_hash", "scientific_git_sha", "teacher", "dataset_identity", "attack_identity")
    if any(online_meta[key] != feature_meta[key] for key in keys):
        raise Pre39OnlineConfirmError("online-state/feature lineage identity drifted")
    if (
        set(online) != set(feature[ANCHOR])
        or set(online) != set(outcome[99])
        or any(
            online[item]["class_id"] != feature[ANCHOR][item]["class_id"]
            or online[item]["class_id"] != outcome[99][item]["class_id"]
            for item in online
        )
    ):
        raise Pre39OnlineConfirmError("online/replay stable-ID/class join drifted")
    frequency = _midrank({item: 1 - float(row["robust_correct_frequency_inclusive"]) for item, row in online.items()})
    margin = _midrank({item: -float(row["margin_ema"]) for item, row in online.items()})
    scores = {
        "exact_online_student": {item: (frequency[item] + margin[item]) / 2 for item in online},
        "instantaneous_margin": _midrank({item: -float(feature[ANCHOR][item]["student_margin"]) for item in online}),
        "teacher_entropy": {item: float(feature[ANCHOR][item]["teacher_entropy_normalized"]) for item in online},
    }
    pf_ids = {item for item, row in online.items() if bool(row["current_robust_correct"])}
    nr_ids = set(online) - pf_ids
    labels = {
        "PF": {
            item: int(sum(not bool(outcome[epoch][item]["robust_correct"]) for epoch in OUTCOME_WINDOW) >= 2)
            for item in pf_ids
        },
        "NR": {
            item: int(all(not bool(outcome[epoch][item]["robust_correct"]) for epoch in OUTCOME_WINDOW))
            for item in nr_ids
        },
    }
    reports: dict[str, Any] = {}
    bootstrap_rows: dict[str, list[dict[str, Any]]] = {}
    for stratum, target in labels.items():
        reports[stratum] = {
            "definition": "online_current_correct_and_wrong_at_least_two_of_99_104_109"
            if stratum == "PF"
            else "online_current_wrong_and_wrong_all_of_99_104_109",
            "models": {name: _metrics(score, target) for name, score in scores.items()},
            "top_q_overlap": _overlap(scores, set(target)),
        }
        bootstrap_rows[stratum] = [
            {
                "sample_id": item,
                "class_id": online[item]["class_id"],
                "outcome": target[item],
                **{name: score[item] for name, score in scores.items()},
            }
            for item in sorted(target)
        ]
    return {
        "schema_version": 1,
        "contract": "pre39_exact_online_confirm_v1",
        "cpu_only": True,
        "anchor_epoch": ANCHOR,
        "outcome_window": list(OUTCOME_WINDOW),
        "score_contract": "equal_midrank_of_one_minus_inclusive_robust_correct_frequency_and_negative_margin_ema_v1",
        "model_predictors": {
            "exact_online_student": ["1 - robust_correct_frequency_inclusive", "-margin_ema"],
            "instantaneous_margin": ["-student_margin_epoch34"],
            "teacher_entropy": ["teacher_entropy_normalized_epoch34"],
        },
        "input_identity": {
            "run_id": feature_meta["run_id"],
            "config_hash": feature_meta["config_hash"],
            "scientific_git_sha": feature_meta["scientific_git_sha"],
            "teacher_registry_id": feature_meta["teacher"].get("registry_id"),
            "online_observations_sha256": sha256_file(online_observations),
            "online_lineage_sha256": sha256_file(online_lineage),
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "feature_attack_domain": feature_meta["feature_protocol"],
            "outcome_attack_domain": outcome_meta["outcome_protocol"],
        },
        "strata": reports,
        "analysis_provenance": provenance,
        "_bootstrap_rows": bootstrap_rows,
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def write_pre39_online_confirm(*, output: Path, report: Mapping[str, Any], overwrite: bool = False) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite PRE39 exact-online confirmation")
    _write_atomic(output, {key: value for key, value in report.items() if not key.startswith("_")})
    return output


def _bootstrap_task(rows: object, *, stratum: str, baseline: str) -> list[dict[str, Any]]:
    if (
        stratum not in {"PF", "NR"}
        or baseline not in {"instantaneous_margin", "teacher_entropy"}
        or not isinstance(rows, list)
    ):
        raise Pre39OnlineConfirmError("bootstrap task identity is invalid")
    required = {"sample_id", "class_id", "outcome", *MODEL_NAMES}
    parsed: list[dict[str, Any]] = []
    ids: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise Pre39OnlineConfirmError("bootstrap row schema drifted")
        sample_id, class_id, outcome = row["sample_id"], row["class_id"], row["outcome"]
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or sample_id in ids
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 10
            or outcome not in {0, 1}
        ):
            raise Pre39OnlineConfirmError("bootstrap sample/class/outcome contract drifted")
        ids.add(sample_id)
        parsed.append(
            {**row, **{name: _finite(row[name], name=f"bootstrap {name}", lower=0, upper=1) for name in MODEL_NAMES}}
        )
    if not parsed or not any(row["outcome"] for row in parsed) or all(row["outcome"] for row in parsed):
        raise Pre39OnlineConfirmError("bootstrap outcome requires both classes")
    return parsed


def run_pre39_online_bootstrap(
    *,
    report: Mapping[str, Any],
    stratum: str,
    baseline: str,
    output: Path,
    progress: Path,
    max_replicates: int | None = None,
) -> dict[str, Any]:
    """Run one class-stratified paired student-minus-comparator AUROC bootstrap."""
    if output.exists():
        raise FileExistsError("refusing to overwrite PRE39 exact-online bootstrap")
    rows = _bootstrap_task(report.get("_bootstrap_rows", {}).get(stratum), stratum=stratum, baseline=baseline)
    limit = BOOTSTRAP_REPLICATES if max_replicates is None else min(BOOTSTRAP_REPLICATES, max_replicates)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise Pre39OnlineConfirmError("bootstrap replicate limit must be positive")
    fingerprint = _hash(
        {
            "contract": "pre39_exact_online_paired_bootstrap_v1",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "stratum": stratum,
            "baseline": baseline,
            "rows": rows,
        }
    )
    state: dict[str, Any] = {"fingerprint": fingerprint, "completed": {}}
    if progress.exists():
        try:
            state = json.loads(progress.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Pre39OnlineConfirmError("bootstrap progress is unreadable") from exc
        if state.get("fingerprint") != fingerprint or not isinstance(state.get("completed"), dict):
            raise Pre39OnlineConfirmError("bootstrap progress fingerprint mismatch")
    completed = {int(key): value for key, value in state["completed"].items()}
    for replicate in range(limit):
        if replicate in completed:
            continue
        selected = _bootstrap_indices(rows, seed=BOOTSTRAP_SEED, replicate=replicate, cluster=False)
        target = [int(rows[index]["outcome"]) for index in selected]
        try:
            completed[replicate] = (
                binary_metrics(target, [float(rows[index]["exact_online_student"]) for index in selected])["auroc"]
                - binary_metrics(target, [float(rows[index][baseline]) for index in selected])["auroc"]
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
    deltas = sorted(float(value) for replicate, value in completed.items() if replicate < limit and value is not None)
    result = {
        "schema_version": 1,
        "contract": "pre39_exact_online_paired_bootstrap_v1",
        "anchor_epoch": ANCHOR,
        "stratum": stratum,
        "candidate": "exact_online_student",
        "baseline": baseline,
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
