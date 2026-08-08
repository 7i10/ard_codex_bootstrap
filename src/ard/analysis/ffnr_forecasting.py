"""Read-only FF/NR forecasting analysis on hash-bound trajectory panels.

The module deliberately separates the deployable, training-time
``SampleStateStore`` domain from the five-epoch common-PGD replay domain.  It
is a development analysis tool: it enumerates plateau ground-truth candidates
and reports score diagnostics, but never chooses a target definition or a
predictor from their measured accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.history_cohort import HistoryCohortError, bind_reports_to_cohort, load_cohort_inventory
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS, PANEL_EMA_BETA, repository_root_from_source
from ard.analysis.signal_audit import canonical_json, sha256_file


class FFNRForecastingError(ValueError):
    """Raised when a FF/NR input violates its frozen development contract."""


CONTRACT = "ffnr_online_forecasting_v1"
OBSERVATION_COLUMNS = {
    "namespace",
    "sample_id",
    "class_id",
    "epoch",
    "teacher_entropy_normalized",
    "student_probability_margin",
    "student_margin_risk",
    "robust_correct",
}
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FFNRForecastingError(f"unreadable JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise FFNRForecastingError("JSON input must be an object")
    return value


def _int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FFNRForecastingError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value: object, name: str, *, lo: float = -math.inf, hi: float = math.inf) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lo <= float(value) <= hi
    ):
        raise FFNRForecastingError(f"{name} is outside its finite contract")
    return float(value)


def _tracked_clean_provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "ffnr_forecasting": Path(__file__).resolve(),
        "ffnr_forecasting_cli": root / "src/ard/cli/ffnr_forecasting.py",
        "history_online_state": root / "src/ard/analysis/history_online_state.py",
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
        raise FFNRForecastingError("FF/NR analysis requires tracked source and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise FFNRForecastingError("FF/NR analysis requires a tracked-clean analysis revision")
    return {"git": {"sha": sha, "dirty": False}, "source_files": {k: sha256_file(v) for k, v in paths.items()}}


def deterministic_midranks(values: Mapping[int, float]) -> dict[int, float]:
    """Return deterministic centred ranks in ``(0, 1)`` with exact tie handling."""
    if not values:
        return {}
    ordered = sorted((float(value), sample_id) for sample_id, value in values.items())
    result: dict[int, float] = {}
    begin = 0
    while begin < len(ordered):
        end = begin + 1
        while end < len(ordered) and ordered[end][0] == ordered[begin][0]:
            end += 1
        rank = ((begin + 1 + end) / 2.0) / len(ordered)
        for _, sample_id in ordered[begin:end]:
            result[sample_id] = rank
        begin = end
    return result


def equal_rank_score(components: Mapping[str, Mapping[int, float]]) -> dict[int, float]:
    """Combine same-domain components only after deterministic within-population ranking."""
    if not components:
        raise FFNRForecastingError("an equal-rank score requires at least one component")
    ids = None
    ranks: list[Mapping[int, float]] = []
    for name, values in components.items():
        if ids is None:
            ids = set(values)
        elif set(values) != ids:
            raise FFNRForecastingError(f"score component {name} stable-ID coverage differs")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise FFNRForecastingError(f"score component {name} is non-finite")
        ranks.append(deterministic_midranks(values))
    assert ids is not None
    return {sample_id: sum(rank[sample_id] for rank in ranks) / len(ranks) for sample_id in sorted(ids)}


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:
        raise FFNRForecastingError(f"unreadable Parquet input: {path}") from exc
    if not OBSERVATION_COLUMNS.issubset(table.column_names):
        raise FFNRForecastingError("replay Parquet observation schema is incomplete")
    return [dict(row) for row in table.to_pylist()]


def _lineage(path: Path, observations: Path, key: str, expected_count: int) -> dict[str, Any]:
    value = _json(path)
    if (
        value.get("schema_version") != 1
        or value.get("observation_schema_version") != 2
        or value.get("train_expected_count") != expected_count
        or value.get(key) != sha256_file(observations)
    ):
        raise FFNRForecastingError("replay lineage byte/count contract drifted")
    for name in ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher"):
        if name not in value:
            raise FFNRForecastingError("replay lineage identity is incomplete")
    is_outcome = key == "outcome_observations_sha256"
    seed = value.get("seed")
    if not is_outcome and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise FFNRForecastingError("feature replay lineage seed is incomplete")
    if is_outcome and "seed" in value and seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise FFNRForecastingError("outcome replay lineage seed is invalid")
    if not isinstance(value["teacher"], Mapping) or not isinstance(value["dataset_identity"], Mapping):
        raise FFNRForecastingError("replay lineage teacher/dataset identity is invalid")
    return value


def _validate_kl_teacher_clean_pgd10(attack: object) -> None:
    """Fail closed before calling an existing replay a KL-PGD10 sensitivity."""
    if not isinstance(attack, Mapping):
        raise FFNRForecastingError("replay attack identity is incomplete")
    required: dict[str, object] = {
        "norm": "linf",
        "input_domain": "pixel_0_1",
        "epsilon": "8/255",
        "step_size": "2/255",
        "steps": 10,
        "random_start": True,
        "loss": "kl",
        "kl_target": "teacher_clean",
        "temperature": 1.0,
        "temperature_squared": True,
        "student_mode": "eval",
        "teacher_mode": "eval",
    }
    if any(attack.get(name) != value for name, value in required.items()):
        raise FFNRForecastingError("five-epoch sensitivity requires exact KL teacher_clean PGD10 attack identity")
    if not math.isclose(_float(attack.get("epsilon_value"), "attack epsilon", lo=0), 8.0 / 255.0, abs_tol=1e-12):
        raise FFNRForecastingError("five-epoch sensitivity attack epsilon drifted")
    if not math.isclose(_float(attack.get("step_size_value"), "attack step size", lo=0), 2.0 / 255.0, abs_tol=1e-12):
        raise FFNRForecastingError("five-epoch sensitivity attack step-size drifted")


def _online_panel(
    path: Path, lineage_path: Path, expected_count: int
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[str, Any]]:
    lineage = _json(lineage_path)
    if (
        lineage.get("contract") != "h5_online_state_anchor_v1"
        or lineage.get("expected_count") != expected_count
        or lineage.get("observations_sha256") != sha256_file(path)
    ):
        raise FFNRForecastingError("online-state lineage is not hash-bound to the requested panel")
    for name in (
        "run_id",
        "config_hash",
        "scientific_git_sha",
        "seed",
        "attack_identity",
        "dataset_identity",
        "teacher",
    ):
        if name not in lineage:
            raise FFNRForecastingError("online-state lineage identity is incomplete")
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise FFNRForecastingError("online-state observations are unreadable") from exc
    panel: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row.get("namespace") != "train":
            raise FFNRForecastingError("official-test or non-train sample state is forbidden")
        epoch = _int(row.get("anchor_epoch"), "online anchor epoch")
        sample_id = _int(row.get("sample_id"), "online sample ID")
        label = _int(row.get("true_label"), "online class ID")
        if label >= 10 or sample_id in panel.setdefault(epoch, {}):
            raise FFNRForecastingError("online stable-ID/class contract drifted")
        seen = _int(row.get("robust_correct_count"), "online robust correct count")
        frequency = _float(row.get("robust_correct_frequency_inclusive"), "online correctness frequency", lo=0, hi=1)
        if seen > epoch + 1 or not math.isclose(frequency, seen / (epoch + 1), abs_tol=1e-7):
            raise FFNRForecastingError("online inclusive frequency contract drifted")
        if not isinstance(row.get("previous_robust_correct"), bool):
            raise FFNRForecastingError("online current correctness must be boolean")
        panel[epoch][sample_id] = {
            "class_id": label,
            "current_correct": row["previous_robust_correct"],
            "frequency_risk": 1.0 - frequency,
            "margin_risk": (1.0 - _float(row.get("margin_ema"), "online margin EMA", lo=-1, hi=1)) / 2.0,
            "last_margin_risk": (1.0 - _float(row.get("last_margin"), "online last margin", lo=-1, hi=1)) / 2.0,
        }
    if not panel or any(len(rows_by_id) != expected_count for rows_by_id in panel.values()):
        raise FFNRForecastingError("online panel lacks exact stable-ID coverage")
    reference = next(iter(panel.values()))
    if any(
        set(rows_by_id) != set(reference)
        or any(rows_by_id[sample_id]["class_id"] != reference[sample_id]["class_id"] for sample_id in reference)
        for rows_by_id in panel.values()
    ):
        raise FFNRForecastingError("online panel stable-ID/class join drifted")
    return panel, lineage


def _replay_panel(
    rows: Sequence[Mapping[str, Any]], epochs: Sequence[int], expected_count: int, name: str
) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = {epoch: {} for epoch in epochs}
    for row in rows:
        if row.get("namespace") != "train":
            raise FFNRForecastingError("official-test or non-train replay input is forbidden")
        epoch = _int(row.get("epoch"), f"{name} epoch")
        sample_id = _int(row.get("sample_id"), f"{name} sample ID")
        class_id = _int(row.get("class_id"), f"{name} class ID")
        if epoch not in result or class_id >= 10 or sample_id in result[epoch]:
            raise FFNRForecastingError(f"{name} replay epoch/ID contract drifted")
        margin = _float(row.get("student_probability_margin"), f"{name} probability margin", lo=-1, hi=1)
        risk = _float(row.get("student_margin_risk"), f"{name} margin risk", lo=0, hi=1)
        if not math.isclose(risk, (1.0 - margin) / 2.0, abs_tol=1e-7) or not isinstance(
            row.get("robust_correct"), bool
        ):
            raise FFNRForecastingError(f"{name} student signal contract drifted")
        result[epoch][sample_id] = {
            **dict(row),
            "class_id": class_id,
            "margin": margin,
            "correct": row["robust_correct"],
        }
    if any(len(by_id) != expected_count for by_id in result.values()):
        raise FFNRForecastingError(f"{name} replay lacks exact stable-ID coverage")
    reference = next(iter(result.values()))
    if any(
        set(by_id) != set(reference)
        or any(by_id[sample_id]["class_id"] != reference[sample_id]["class_id"] for sample_id in reference)
        for by_id in result.values()
    ):
        raise FFNRForecastingError(f"{name} replay stable-ID/class join drifted")
    return result


def _identity_matches(*lineages: Mapping[str, Any]) -> None:
    keys = ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher")
    first = lineages[0]
    if any(any(item.get(key) != first.get(key) for key in keys) for item in lineages[1:]):
        raise FFNRForecastingError("online/feature/outcome lineage identity drifted")


def _load_validation(path: Path) -> dict[int, float]:
    try:
        if path.suffix == ".jsonl":
            raw: object = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FFNRForecastingError("validation PGD history is unreadable") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("records", raw.get("metrics", raw))
    if not isinstance(raw, list):
        raise FFNRForecastingError("validation PGD history must be a list or JSONL")
    result: dict[int, float] = {}
    for row in raw:
        if not isinstance(row, Mapping) or "epoch" not in row or "val_pgd_accuracy" not in row:
            continue
        epoch = _int(row["epoch"], "validation epoch")
        value = _float(row["val_pgd_accuracy"], "validation PGD accuracy", lo=0, hi=1)
        if epoch in result:
            raise FFNRForecastingError("validation PGD history has a duplicate epoch")
        result[epoch] = value
    if set(result) != set(range(200)):
        raise FFNRForecastingError("validation PGD history must contain each exact epoch 0..199 once")
    return result


def _validation_manifest(path: Path, history: Path, replay: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the validation time series to the completed baseline trajectory."""
    if path.parent.resolve() != history.parent.resolve():
        raise FFNRForecastingError("validation manifest must be a sibling of validation history")
    value = _json(path)
    git = value.get("git")
    teacher = value.get("teacher")
    if value.get("status") != "completed" or not isinstance(git, Mapping) or not isinstance(teacher, Mapping):
        raise FFNRForecastingError("validation manifest must be completed with git and teacher identity")
    observed = {
        "run_id": value.get("run_id"),
        "config_hash": value.get("config_hash"),
        "scientific_git_sha": git.get("sha"),
        "seed": value.get("seed"),
        "teacher_registry_id": teacher.get("registry_id"),
    }
    expected = {
        "run_id": replay.get("run_id"),
        "config_hash": replay.get("config_hash"),
        "scientific_git_sha": replay.get("scientific_git_sha"),
        "seed": replay.get("seed"),
        "teacher_registry_id": (
            replay.get("teacher", {}).get("registry_id") if isinstance(replay.get("teacher"), Mapping) else None
        ),
    }
    if observed != expected:
        raise FFNRForecastingError("validation manifest and replay identity drifted")
    return value


def plateau_candidates(
    validation_pgd: Mapping[int, float],
    *,
    saved_epochs: Sequence[int],
    scheduler_stages: Sequence[Sequence[int]],
    deltas_pp: Sequence[float] = (0.25, 0.5, 1.0),
    window_sizes: Sequence[int] = (3, 5, 7),
    thresholds: Sequence[str] = ("majority", "two_thirds", "all"),
) -> list[dict[str, Any]]:
    """Enumerate centred same-stage plateau candidates without selecting one.

    A candidate is censored when a full symmetric checkpoint window is not
    available.  In particular this prevents the terminal ERT case from being
    silently moved left or shortened.
    """
    saved = tuple(sorted(set(saved_epochs)))
    if tuple(saved_epochs) != saved or not saved:
        raise FFNRForecastingError("saved checkpoint epochs must be sorted and unique")
    stages = []
    for stage in scheduler_stages:
        if len(stage) != 2:
            raise FFNRForecastingError("scheduler stage must contain [start, end]")
        start, end = _int(stage[0], "stage start"), _int(stage[1], "stage end")
        if end < start:
            raise FFNRForecastingError("scheduler stage end precedes start")
        stages.append((start, end))
    if not stages or any(stages[index][1] >= stages[index + 1][0] for index in range(len(stages) - 1)):
        raise FFNRForecastingError("scheduler stages must be strictly non-overlapping")
    raw_best_value = max(validation_pgd.values())
    raw_best_epoch = min(epoch for epoch, value in validation_pgd.items() if value == raw_best_value)
    available = {epoch: validation_pgd[epoch] for epoch in saved if epoch in validation_pgd}
    if not available:
        raise FFNRForecastingError("validation history has no replay-inventory checkpoint epoch")
    # The five-epoch sensitivity panel cannot define a label at a checkpoint it
    # did not replay.  Its grid best is reported separately from the raw best.
    best_value = max(available.values())
    best_epoch = min(epoch for epoch, value in available.items() if value == best_value)
    matching = [stage for stage in stages if stage[0] <= best_epoch <= stage[1]]
    if len(matching) != 1:
        raise FFNRForecastingError("earliest validation best does not belong to exactly one scheduler stage")
    stage_start, stage_end = matching[0]
    stage_saved = [epoch for epoch in saved if stage_start <= epoch <= stage_end]
    result: list[dict[str, Any]] = []
    for delta in deltas_pp:
        delta_value = _float(delta, "plateau delta", lo=0, hi=100) / 100.0
        eligible = {
            epoch
            for epoch in stage_saved
            if epoch in validation_pgd and validation_pgd[epoch] >= best_value - delta_value
        }
        center_index = stage_saved.index(best_epoch) if best_epoch in stage_saved else None
        # Component is in checkpoint cadence, not a raw validation-epoch run.
        component: list[int] = []
        if center_index is not None and best_epoch in eligible:
            left = center_index
            while left > 0 and stage_saved[left - 1] in eligible:
                left -= 1
            right = center_index
            while right + 1 < len(stage_saved) and stage_saved[right + 1] in eligible:
                right += 1
            component = stage_saved[left : right + 1]
        for window_size in window_sizes:
            if window_size < 1 or window_size % 2 == 0:
                raise FFNRForecastingError("plateau window sizes must be positive odd integers")
            half = window_size // 2
            censor_reason: str | None
            censor_direction: str | None
            if center_index is None:
                window: list[int] = []
                censor_reason, censor_direction = "best_not_saved_checkpoint", None
            elif not component:
                window = []
                censor_reason, censor_direction = "best_outside_delta_component", None
            elif center_index - half < 0 or center_index + half >= len(stage_saved):
                window = []
                censor_reason = "symmetric_window_crosses_scheduler_stage"
                censor_direction = "left" if center_index - half < 0 else "right"
            else:
                window = stage_saved[center_index - half : center_index + half + 1]
                censor_reason = (
                    None if all(epoch in component for epoch in window) else "symmetric_window_leaves_delta_component"
                )
                censor_direction = None
                if censor_reason:
                    window = []
            for threshold in thresholds:
                if threshold not in {"majority", "two_thirds", "all"}:
                    raise FFNRForecastingError("unknown future-failure threshold")
                required = (
                    window_size // 2 + 1
                    if threshold == "majority"
                    else math.ceil(2 * window_size / 3)
                    if threshold == "two_thirds"
                    else window_size
                )
                result.append(
                    {
                        "raw_validation_best_epoch": raw_best_epoch,
                        "raw_validation_best_accuracy": raw_best_value,
                        "best_available_replay_epoch": best_epoch,
                        "best_available_replay_validation_pgd_accuracy": best_value,
                        "replay_grid_cadence_epochs": saved[1] - saved[0] if len(saved) > 1 else None,
                        "scheduler_stage": [stage_start, stage_end],
                        "delta_pp": float(delta),
                        "window_size": window_size,
                        "threshold": threshold,
                        "required_wrong_checkpoints": required,
                        "component_saved_epochs": component,
                        "window_epochs": window,
                        "censored": bool(censor_reason),
                        "censor_reason": censor_reason,
                        "censor_direction": censor_direction,
                    }
                )
    return result


def split_ff_nr(
    *, future_failure: Mapping[int, int], online_current_correct: Mapping[int, bool]
) -> tuple[dict[int, int], dict[int, int]]:
    """Partition future failure by exact online correctness.

    The second output is current-wrong future failure, not generic
    non-recovery: a plateau-window failure alone does not prove a sample could
    never recover before that window.
    """
    if set(future_failure) != set(online_current_correct):
        raise FFNRForecastingError("future-failure and online eligibility stable IDs differ")
    ff = {
        sample_id: int(bool(outcome) and online_current_correct[sample_id])
        for sample_id, outcome in future_failure.items()
    }
    current_wrong_future_failure = {
        sample_id: int(bool(outcome) and not online_current_correct[sample_id])
        for sample_id, outcome in future_failure.items()
    }
    if any(
        ff[sample_id] + current_wrong_future_failure[sample_id] != int(bool(future_failure[sample_id]))
        for sample_id in future_failure
    ):
        raise FFNRForecastingError("FF/current-wrong future-failure partition is not disjoint and exhaustive")
    return ff, current_wrong_future_failure


def _tie_inclusive_selection(
    scores: Mapping[int, float], ranked: Sequence[int], nominal_count: int
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if not ranked or nominal_count < 1 or nominal_count > len(ranked):
        raise FFNRForecastingError("tie-inclusive selection requires a valid nominal count")
    boundary = scores[ranked[nominal_count - 1]]
    selected = tuple(sample_id for sample_id in ranked if scores[sample_id] >= boundary)
    tie_count = sum(scores[sample_id] == boundary for sample_id in ranked)
    return selected, {
        "nominal_count": nominal_count,
        "realized_count": len(selected),
        "realized_fraction": len(selected) / len(ranked),
        "boundary_score": boundary,
        "boundary_tie_count": tie_count,
    }


def _metric(
    scores: Mapping[int, float], targets: Mapping[int, int], *, ranked: Sequence[int] | None = None
) -> dict[str, Any]:
    if set(scores) != set(targets):
        raise FFNRForecastingError("metric scores and targets stable IDs differ")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0 for value in scores.values()):
        raise FFNRForecastingError("metric scores must be finite ranks in [0, 1]")
    count = len(targets)
    positives = sum(targets.values())
    base: dict[str, Any] = {
        "count": count,
        "positive_count": positives,
        "prevalence": positives / count if count else None,
    }
    if not count or positives in {0, count}:
        return {**base, "auroc": None, "auprc": None, "oracle_m": None, "top_percent": {}}
    ordered = tuple(sorted(scores, key=lambda sample_id: (-scores[sample_id], sample_id))) if ranked is None else ranked
    if len(ordered) != count or set(ordered) != set(scores):
        raise FFNRForecastingError("cached metric ranking stable IDs differ")

    def summary(nominal_count: int) -> dict[str, float | int]:
        chosen, selection = _tie_inclusive_selection(scores, ordered, nominal_count)
        hits = sum(targets[sample_id] for sample_id in chosen)
        precision = hits / len(chosen)
        recall = hits / positives
        return {**selection, "precision": precision, "recall": recall, "lift": precision / base["prevalence"]}

    negatives = count - positives
    concordant = 0.0
    processed_negatives = 0
    cumulative_positives = 0
    auprc = 0.0
    start = 0
    while start < count:
        end = start + 1
        score = scores[ordered[start]]
        while end < count and scores[ordered[end]] == score:
            end += 1
        group_positives = sum(targets[sample_id] for sample_id in ordered[start:end])
        group_negatives = end - start - group_positives
        negatives_below = negatives - processed_negatives - group_negatives
        concordant += group_positives * negatives_below + 0.5 * group_positives * group_negatives
        cumulative_positives += group_positives
        auprc += group_positives * (cumulative_positives / end) / positives
        processed_negatives += group_negatives
        start = end
    top = {str(fraction): summary(max(1, math.ceil(fraction * count))) for fraction in TOP_FRACTIONS}
    return {
        **base,
        "auroc": concordant / (positives * negatives),
        "auprc": auprc,
        "oracle_m": summary(positives),
        "top_percent": top,
    }


def _linear_slope(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise FFNRForecastingError("slope requires at least two aligned observations")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def _replay_components(
    panel: Mapping[int, Mapping[int, Mapping[str, Any]]], anchor: int
) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    epochs = tuple(epoch for epoch in FEATURE_EPOCHS if epoch <= anchor)
    if anchor not in panel or len(epochs) < 2:
        raise FFNRForecastingError("anchor does not have five-epoch replay history")
    ids = set(panel[anchor])
    margins = {sample_id: [float(panel[epoch][sample_id]["margin"]) for epoch in epochs] for sample_id in ids}
    rank_by_epoch = {
        epoch: deterministic_midranks({sample_id: margins[sample_id][index] for sample_id in ids})
        for index, epoch in enumerate(epochs)
    }
    values: dict[str, dict[int, float]] = {
        "L_probability_margin_risk": {sample_id: -margins[sample_id][-1] for sample_id in ids},
        "T_five_epoch_margin_drop": {
            sample_id: -(margins[sample_id][-1] - margins[sample_id][-2]) for sample_id in ids
        },
        "T_recent_margin_slope_risk": {
            sample_id: -_linear_slope(epochs[-min(5, len(epochs)) :], margins[sample_id][-min(5, len(epochs)) :])
            for sample_id in ids
        },
        "T_rank_slope_risk": {
            sample_id: -_linear_slope(
                epochs[-min(5, len(epochs)) :],
                [rank_by_epoch[epoch][sample_id] for epoch in epochs[-min(5, len(epochs)) :]],
            )
            for sample_id in ids
        },
        "S_replay_correctness_frequency_risk": {
            sample_id: 1.0 - sum(bool(panel[epoch][sample_id]["correct"]) for epoch in epochs) / len(epochs)
            for sample_id in ids
        },
        "S_replay_flip_rate": {
            sample_id: sum(
                bool(panel[left][sample_id]["correct"]) != bool(panel[right][sample_id]["correct"])
                for left, right in zip(epochs, epochs[1:])
            )
            / (len(epochs) - 1)
            for sample_id in ids
        },
        "S_margin_variance": {
            sample_id: sum((value - sum(margins[sample_id]) / len(epochs)) ** 2 for value in margins[sample_id])
            / len(epochs)
            for sample_id in ids
        },
    }
    fast_beta = 0.5

    def ema(sequence: Sequence[float], beta: float) -> float:
        answer = sequence[0]
        for value in sequence[1:]:
            answer = beta * answer + (1 - beta) * value
        return answer

    values["T_fast_minus_slow_ema_risk"] = {
        sample_id: ema(margins[sample_id], PANEL_EMA_BETA) - ema(margins[sample_id], fast_beta) for sample_id in ids
    }
    unavailable = {
        "L_logit_margin_risk": "not_recorded_in_historical_replay",
        "L_adversarial_cross_entropy": "not_recorded_in_historical_replay",
        "T_one_epoch_difference": "five_epoch_replay_only",
        "S_current_streak": "not_present_in_common_online_anchor_export",
        "D_teacher_js_response": "full_probability_vectors_not_recorded_in_historical_replay",
    }
    current = panel[anchor]
    required_columns = {
        "D_teacher_nonresponse_risk": ("teacher_clean_to_adversarial_margin_delta",),
        "D_teacher_adversarial_confidence": ("teacher_adversarial_entropy_normalized",),
        "D_student_teacher_response_gap": (
            "student_clean_probability_margin",
            "teacher_clean_probability_margin",
            "teacher_adversarial_probability_margin",
        ),
    }
    for name, columns in required_columns.items():
        absent = [column for column in columns if not all(column in current[sample_id] for sample_id in ids)]
        if absent:
            unavailable[name] = "column_absent:" + ",".join(absent)
            continue
        if name == "D_teacher_nonresponse_risk":
            # The attached hypothesis treats little teacher response as a
            # disagreement risk: with stored adv-clean deltas, -0.1 ranks
            # above -0.5.  This is deliberately not teacher-fragility risk.
            values[name] = {
                sample_id: _float(current[sample_id][columns[0]], columns[0], lo=-2, hi=2) for sample_id in ids
            }
        elif name == "D_teacher_adversarial_confidence":
            values[name] = {
                sample_id: 1.0 - _float(current[sample_id][columns[0]], columns[0], lo=0, hi=1) for sample_id in ids
            }
        else:
            values[name] = {
                sample_id: (
                    _float(current[sample_id]["student_clean_probability_margin"], "student clean margin", lo=-1, hi=1)
                    - margins[sample_id][-1]
                    - (
                        _float(
                            current[sample_id]["teacher_clean_probability_margin"], "teacher clean margin", lo=-1, hi=1
                        )
                        - _float(
                            current[sample_id]["teacher_adversarial_probability_margin"],
                            "teacher adversarial margin",
                            lo=-1,
                            hi=1,
                        )
                    )
                )
                for sample_id in ids
            }
    return values, unavailable


def _online_components(
    panel: Mapping[int, Mapping[int, Mapping[str, Any]]], anchor: int
) -> dict[str, dict[int, float]]:
    if anchor not in panel:
        raise FFNRForecastingError("requested anchor is absent from exact online state")
    return {
        "online_L_last_margin_risk": {sample_id: row["last_margin_risk"] for sample_id, row in panel[anchor].items()},
        "online_S_frequency_risk": {sample_id: row["frequency_risk"] for sample_id, row in panel[anchor].items()},
        "online_S_margin_ema_risk": {sample_id: row["margin_risk"] for sample_id, row in panel[anchor].items()},
    }


def _score_specs(
    replay: Mapping[str, Mapping[int, float]], online: Mapping[str, Mapping[int, float]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Return a bounded, explicit ablation matrix without a hidden all-T/S average."""
    level = "L_probability_margin_risk"
    ts = (
        "T_five_epoch_margin_drop",
        "T_recent_margin_slope_risk",
        "T_rank_slope_risk",
        "T_fast_minus_slow_ema_risk",
    )
    ss = ("S_replay_correctness_frequency_risk", "S_replay_flip_rate", "S_margin_variance")
    ds = (
        "D_teacher_nonresponse_risk",
        "D_teacher_adversarial_confidence",
        "D_student_teacher_response_gap",
    )
    specs: dict[str, tuple[str, ...]] = {
        "online_L": ("online_L_last_margin_risk",),
        "online_S_frequency": ("online_S_frequency_risk",),
        "online_S_margin_ema": ("online_S_margin_ema_risk",),
        "online_LS": ("online_L_last_margin_risk", "online_S_frequency_risk", "online_S_margin_ema_risk"),
        "replay_L": (level,),
    }
    specs.update({f"replay_{name}": (name,) for name in (*ts, *ss, *ds)})
    specs.update({f"replay_L_plus_{name}": (level, name) for name in ts})
    specs.update({f"replay_L_plus_{name}": (level, name) for name in ss})
    specs.update({f"replay_{t}_plus_{s}": (t, s) for t in ts for s in ss})
    specs.update({f"replay_L_plus_{t}_plus_{s}": (level, t, s) for t in ts for s in ss})
    # A fixed representative is useful only as an explicitly named reference;
    # it is not selected from the reported results.
    representative = (level, "T_recent_margin_slope_risk", "S_replay_correctness_frequency_risk")
    specs["replay_LTS_representative"] = representative
    specs.update({f"replay_LTS_representative_plus_{name}": (*representative, name) for name in ds})
    all_components = {**replay, **online}
    available = {
        name: names for name, names in specs.items() if all(component in all_components for component in names)
    }
    unavailable = {
        name: "component_absent:" + ",".join(component for component in names if component not in all_components)
        for name, names in specs.items()
        if name not in available
    }
    return available, unavailable


def _conditional_scores(
    *,
    replay: Mapping[str, Mapping[int, float]],
    online: Mapping[str, Mapping[int, float]],
    score_specs: Mapping[str, Sequence[str]],
    sample_ids: set[int],
) -> dict[str, dict[int, float]]:
    """Rank components only after restricting to a FF/current-wrong risk set."""
    components = {**replay, **online}
    return {
        name: equal_rank_score(
            {
                component: {sample_id: components[component][sample_id] for sample_id in sample_ids}
                for component in names
            }
        )
        for name, names in score_specs.items()
    }


def _stability(
    masks: Mapping[int, Mapping[str, set[int]]], *, include_partition_transitions: bool = True
) -> dict[str, Any]:
    epochs = sorted(masks)
    pairs = []
    transitions: dict[str, int] = {}
    for left, right in zip(epochs, epochs[1:]):
        for name in sorted(set(masks[left]) & set(masks[right])):
            before, after = masks[left][name], masks[right][name]
            union = before | after
            pairs.append(
                {
                    "from_anchor": left,
                    "to_anchor": right,
                    "score": name,
                    "jaccard": len(before & after) / len(union) if union else None,
                    "retention": len(before & after) / len(before) if before else None,
                    "entry": len(after - before),
                    "exit": len(before - after),
                }
            )
        if include_partition_transitions:
            ff_before, cwff_before = (
                masks[left].get("FF", set()),
                masks[left].get("current_wrong_future_failure", set()),
            )
            ff_after, cwff_after = (
                masks[right].get("FF", set()),
                masks[right].get("current_wrong_future_failure", set()),
            )
            for previous, current, count in (
                ("FF", "FF", len(ff_before & ff_after)),
                ("FF", "current_wrong_future_failure", len(ff_before & cwff_after)),
                ("current_wrong_future_failure", "FF", len(cwff_before & ff_after)),
                ("current_wrong_future_failure", "current_wrong_future_failure", len(cwff_before & cwff_after)),
            ):
                transitions[f"{left}->{right}:{previous}->{current}"] = count
    result: dict[str, Any] = {"consecutive_masks": pairs}
    if include_partition_transitions:
        result["ff_current_wrong_future_failure_transitions"] = transitions
    return result


def analyze_ffnr_run(
    *,
    label: str,
    feature_observations: Path,
    feature_lineage: Path,
    outcome_observations: Path,
    outcome_lineage: Path,
    online_states: Path,
    online_lineage: Path,
    validation_history: Path,
    validation_manifest: Path,
    expected_count: int,
    scheduler_stages: Sequence[Sequence[int]],
    anchors: Sequence[int] = (39, 59, 79),
    deltas_pp: Sequence[float] = (0.25, 0.5, 1.0),
    window_sizes: Sequence[int] = (3, 5, 7),
    thresholds: Sequence[str] = ("majority", "two_thirds", "all"),
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build every censored plateau candidate and non-selective FF/NR score report."""
    if label not in {"L1", "L2", "L3", "L4"}:
        raise FFNRForecastingError("FF/NR development reports require an L1--L4 label")
    if expected_count < 1 or tuple(sorted(set(anchors))) != tuple(anchors):
        raise FFNRForecastingError("anchors must be sorted unique positive epochs")
    fmeta = _lineage(feature_lineage, feature_observations, "feature_observations_sha256", expected_count)
    ometa = _lineage(outcome_lineage, outcome_observations, "outcome_observations_sha256", expected_count)
    online, online_meta = _online_panel(online_states, online_lineage, expected_count)
    _validate_kl_teacher_clean_pgd10(fmeta["attack_identity"])
    _validate_kl_teacher_clean_pgd10(ometa["attack_identity"])
    _identity_matches(fmeta, ometa, online_meta)
    if online_meta.get("seed") != fmeta.get("seed") or ometa.get("seed") not in {None, fmeta.get("seed")}:
        raise FFNRForecastingError("online/feature/outcome seed identity drifted")
    _validation_manifest(validation_manifest, validation_history, fmeta)
    feature = _replay_panel(_read_parquet(feature_observations), FEATURE_EPOCHS, expected_count, "feature")
    outcome = _replay_panel(_read_parquet(outcome_observations), OUTCOME_EPOCHS, expected_count, "outcome")
    ids = set(feature[FEATURE_EPOCHS[-1]])
    if any(anchor not in online or anchor not in feature for anchor in anchors):
        raise FFNRForecastingError("each requested anchor must exist in exact online and replay feature panels")
    if ids != set(outcome[OUTCOME_EPOCHS[0]]) or ids != set(next(iter(online.values()))):
        raise FFNRForecastingError("online/feature/outcome stable-ID sets differ")
    for sample_id in ids:
        class_id = feature[FEATURE_EPOCHS[-1]][sample_id]["class_id"]
        if (
            outcome[OUTCOME_EPOCHS[0]][sample_id]["class_id"] != class_id
            or next(iter(online.values()))[sample_id]["class_id"] != class_id
        ):
            raise FFNRForecastingError("online/feature/outcome class join drifted")
    validation = _load_validation(validation_history)
    candidates = plateau_candidates(
        validation,
        saved_epochs=OUTCOME_EPOCHS,
        scheduler_stages=scheduler_stages,
        deltas_pp=deltas_pp,
        window_sizes=window_sizes,
        thresholds=thresholds,
    )
    score_rows: list[dict[str, Any]] = []
    score_masks: dict[int, dict[str, set[int]]] = {}
    score_mask_selection: dict[int, dict[str, dict[str, Any]]] = {}
    anchor_scores: dict[int, dict[str, dict[str, dict[int, float]]]] = {}
    anchor_rankings: dict[int, dict[str, dict[str, tuple[int, ...]]]] = {}
    availability: dict[int, dict[str, Any]] = {}
    for anchor in anchors:
        replay, unavailable = _replay_components(feature, anchor)
        exact = _online_components(online, anchor)
        specs, unavailable_scores = _score_specs(replay, exact)
        availability[anchor] = {
            "available_scores": {name: list(parts) for name, parts in specs.items()},
            "unavailable": {**unavailable, **unavailable_scores},
        }
        current_correct = {sample_id for sample_id in ids if online[anchor][sample_id]["current_correct"]}
        current_wrong = ids - current_correct
        anchor_scores[anchor] = {
            "FF": _conditional_scores(replay=replay, online=exact, score_specs=specs, sample_ids=current_correct),
            "current_wrong_future_failure": _conditional_scores(
                replay=replay, online=exact, score_specs=specs, sample_ids=current_wrong
            ),
        }
        anchor_rankings[anchor] = {
            stratum: {
                name: tuple(sorted(values, key=lambda sample_id: (-values[sample_id], sample_id)))
                for name, values in stratum_scores.items()
            }
            for stratum, stratum_scores in anchor_scores[anchor].items()
        }
        score_masks[anchor] = {}
        score_mask_selection[anchor] = {}
        for stratum, stratum_scores in anchor_scores[anchor].items():
            for sample_id in sorted(next(iter(stratum_scores.values()), {})):
                score_rows.append(
                    {
                        "run": label,
                        "anchor_epoch": anchor,
                        "stratum": "online_current_correct" if stratum == "FF" else "online_current_wrong",
                        "sample_id": sample_id,
                        "class_id": online[anchor][sample_id]["class_id"],
                        **{name: values[sample_id] for name, values in stratum_scores.items()},
                    }
                )
            for name, values in stratum_scores.items():
                nominal_count = max(1, math.ceil(0.10 * len(values))) if values else 0
                selected, selection = _tie_inclusive_selection(
                    values, anchor_rankings[anchor][stratum][name], nominal_count
                )
                key = f"{stratum}:{name}"
                score_masks[anchor][key] = set(selected)
                score_mask_selection[anchor][key] = selection
    candidate_reports: list[dict[str, Any]] = []
    candidate_masks: dict[str, set[int]] = {}
    candidate_transitions: dict[str, dict[int, dict[str, set[int]]]] = {}
    for candidate in candidates:
        candidate_id = "d{delta:g}_k{k}_{threshold}".format(
            delta=candidate["delta_pp"], k=candidate["window_size"], threshold=candidate["threshold"]
        )
        window = tuple(candidate["window_epochs"])
        report: dict[str, Any] = {**candidate, "candidate_id": candidate_id, "anchors": {}}
        if candidate["censored"]:
            candidate_reports.append(report)
            continue
        future = {
            sample_id: int(
                sum(not outcome[epoch][sample_id]["correct"] for epoch in window)
                >= candidate["required_wrong_checkpoints"]
            )
            for sample_id in ids
        }
        candidate_masks[candidate_id] = {sample_id for sample_id, value in future.items() if value}
        report["future_failure_count"] = sum(future.values())
        report["future_failure_prevalence"] = sum(future.values()) / len(future)
        transitions: dict[int, dict[str, set[int]]] = {}
        for anchor in anchors:
            if anchor not in anchor_scores or not window or anchor >= min(window):
                report["anchors"][str(anchor)] = {
                    "status": "ineligible",
                    "reason": "anchor_not_strictly_before_plateau_or_missing",
                }
                continue
            current = {sample_id: online[anchor][sample_id]["current_correct"] for sample_id in ids}
            ff, current_wrong_future_failure = split_ff_nr(future_failure=future, online_current_correct=current)
            transitions[anchor] = {
                "FF": {sample_id for sample_id, value in ff.items() if value},
                "current_wrong_future_failure": {
                    sample_id for sample_id, value in current_wrong_future_failure.items() if value
                },
            }
            strata = {
                "FF": (
                    {sample_id: future[sample_id] for sample_id in ids if current[sample_id]},
                    "online_current_correct",
                ),
                "current_wrong_future_failure": (
                    {sample_id: future[sample_id] for sample_id in ids if not current[sample_id]},
                    "online_current_wrong",
                ),
            }
            score_report = {
                stratum: {
                    "eligibility": eligibility,
                    "future_failure_partition": {
                        "ff_count": sum(ff.values()),
                        "current_wrong_future_failure_count": sum(current_wrong_future_failure.values()),
                        "future_failure_count": sum(future.values()),
                    },
                    "scores": {
                        name: _metric(values, targets, ranked=anchor_rankings[anchor][stratum][name])
                        for name, values in anchor_scores[anchor][stratum].items()
                    },
                }
                for stratum, (targets, eligibility) in strata.items()
            }
            report["anchors"][str(anchor)] = {
                "status": "eligible",
                "feature_domain": "five_epoch_replay",
                "online_domain": "exact_online_sample_state",
                "score_availability": availability[anchor],
                "strata": score_report,
            }
        candidate_transitions[candidate_id] = transitions
        report["ff_current_wrong_future_failure_stability"] = _stability(transitions)
        candidate_reports.append(report)
    equivalence: dict[str, str] = {}
    representative_by_digest: dict[str, str] = {}
    for report in candidate_reports:
        candidate_id = report["candidate_id"]
        if report["censored"]:
            report["ground_truth_equivalence_class"] = None
            report["ground_truth_representative"] = False
            continue
        positives = candidate_masks[candidate_id]
        digest = hashlib.sha256(canonical_json(sorted(positives))).hexdigest()
        equivalence[candidate_id] = digest
        representative = representative_by_digest.setdefault(digest, candidate_id)
        report["ground_truth_equivalence_class"] = f"gt_{digest[:16]}"
        report["ground_truth_representative"] = candidate_id == representative
        report["ground_truth_equivalent_to"] = representative
    representative_candidate_masks = {
        candidate_id: candidate_masks[candidate_id]
        for candidate_id in candidate_masks
        if representative_by_digest[equivalence[candidate_id]] == candidate_id
    }
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "scientific_status": "development_sensitivity_only_no_automatic_gt_or_predictor_selection",
        "label": label,
        "input_identity": {
            "run_id": fmeta["run_id"],
            "config_hash": fmeta["config_hash"],
            "scientific_git_sha": fmeta["scientific_git_sha"],
            "seed": fmeta["seed"],
            "teacher_registry_id": fmeta["teacher"].get("registry_id"),
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "online_states_sha256": sha256_file(online_states),
            "validation_history_sha256": sha256_file(validation_history),
            "feature_lineage_sha256": sha256_file(feature_lineage),
            "outcome_lineage_sha256": sha256_file(outcome_lineage),
            "online_lineage_sha256": sha256_file(online_lineage),
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "feature_attack_identity": fmeta["attack_identity"],
            "outcome_attack_identity": ometa["attack_identity"],
            "online_attack_identity": online_meta["attack_identity"],
            "feature_protocol": fmeta.get("feature_protocol"),
            "outcome_protocol": ometa.get("outcome_protocol"),
            "outcome_seed_compatibility": "declared" if "seed" in ometa else "historical_missing_seed",
            "stable_id_class_universe_sha256": hashlib.sha256(
                canonical_json(
                    [
                        {"sample_id": sample_id, "class_id": feature[FEATURE_EPOCHS[-1]][sample_id]["class_id"]}
                        for sample_id in sorted(ids)
                    ]
                )
            ).hexdigest(),
        },
        "ground_truth_attack_status": "five_epoch_KL_PGD10_development_sensitivity_not_primary_CE_PGD20",
        "candidates": candidate_reports,
        "score_mask_stability": {
            **_stability(score_masks, include_partition_transitions=False),
            "selection": score_mask_selection,
        },
        "analysis_provenance": dict(
            _tracked_clean_provenance() if analysis_provenance is None else analysis_provenance
        ),
        "_score_rows": score_rows,
        # Formula masks preserve every uncensored candidate for same-formula
        # cross-seed sensitivity.  Representative masks are only for
        # deduplicated within-run summaries and GT-Parquet materialization.
        "_formula_candidate_positive_masks": candidate_masks,
        "_representative_candidate_positive_masks": representative_candidate_masks,
        "_candidate_equivalence": equivalence,
    }


def write_ffnr_report(
    *, output_dir: Path, reports: Mapping[str, Mapping[str, Any]], config_path: Path, cohort_inventory: Path
) -> dict[str, Path]:
    """Write a single hash-bound collection; never overwrite a prior analysis."""
    paths = {
        "report": output_dir / "ffnr-report.json",
        "score_rows": output_dir / "ffnr-score-rows.parquet",
        "ground_truth_rows": output_dir / "ffnr-ground-truth-rows.parquet",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite FF/NR analysis output")
    try:
        inventory, cohort_digest = load_cohort_inventory(cohort_inventory)
        bind_reports_to_cohort(inventory=inventory, reports=reports)
    except HistoryCohortError as exc:
        raise FFNRForecastingError(str(exc)) from exc
    for left, right in (("L1", "L3"), ("L2", "L4")):
        if left not in reports or right not in reports:
            continue
        if reports[left].get("input_identity", {}).get("stable_id_class_universe_sha256") != reports[right].get(
            "input_identity", {}
        ).get("stable_id_class_universe_sha256"):
            raise FFNRForecastingError(
                "same-teacher cross-seed Jaccard requires identical stable-ID/class universe hash"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        score_writer = None
        gt_writer = None
        for label in sorted(reports):
            score_rows = reports[label].get("_score_rows", [])
            if not isinstance(score_rows, list):
                raise FFNRForecastingError("FF/NR score-row internal contract drifted")
            for start in range(0, len(score_rows), 50_000):
                table = pa.Table.from_pylist(score_rows[start : start + 50_000])
                if score_writer is None:
                    score_writer = pq.ParquetWriter(paths["score_rows"], table.schema)
                score_writer.write_table(table)
            class_by_id = {row["sample_id"]: row["class_id"] for row in score_rows}
            masks = reports[label].get("_representative_candidate_positive_masks", {})
            if not isinstance(masks, Mapping):
                raise FFNRForecastingError("FF/NR candidate-mask internal contract drifted")
            for candidate_id, positives in sorted(masks.items()):
                if not isinstance(positives, set):
                    raise FFNRForecastingError("FF/NR candidate-mask type drifted")
                materialized = [
                    {
                        "run": label,
                        "candidate_id": candidate_id,
                        "sample_id": sample_id,
                        "class_id": class_by_id[sample_id],
                        "future_failure": int(sample_id in positives),
                    }
                    for sample_id in sorted(class_by_id)
                ]
                table = pa.Table.from_pylist(materialized)
                if gt_writer is None:
                    gt_writer = pq.ParquetWriter(paths["ground_truth_rows"], table.schema)
                gt_writer.write_table(table)
        if score_writer is None or gt_writer is None:
            raise FFNRForecastingError("FF/NR report has no eligible score or ground-truth rows")
        score_writer.close()
        gt_writer.close()
    except Exception as exc:
        raise FFNRForecastingError("unable to write FF/NR Parquet report") from exc
    public = {
        label: {
            key: value
            for key, value in report.items()
            if key
            not in {
                "_score_rows",
                "_formula_candidate_positive_masks",
                "_representative_candidate_positive_masks",
                "_candidate_equivalence",
            }
        }
        for label, report in reports.items()
    }
    gt_sensitivity: dict[str, Any] = {"within_run_candidate_jaccard": {}, "same_teacher_seed_jaccard": {}}
    for label in sorted(reports):
        masks = reports[label].get("_representative_candidate_positive_masks", {})
        if not isinstance(masks, Mapping):
            raise FFNRForecastingError("FF/NR candidate-mask internal contract drifted")
        pairs = []
        for left in sorted(masks):
            for right in sorted(masks):
                if left >= right:
                    continue
                before, after = masks[left], masks[right]
                if not isinstance(before, set) or not isinstance(after, set):
                    raise FFNRForecastingError("FF/NR candidate-mask type drifted")
                union = before | after
                pairs.append(
                    {"left": left, "right": right, "jaccard": len(before & after) / len(union) if union else None}
                )
        gt_sensitivity["within_run_candidate_jaccard"][label] = pairs
    for left, right in (("L1", "L3"), ("L2", "L4")):
        if left not in reports or right not in reports:
            continue
        left_masks, right_masks = (
            reports[left].get("_formula_candidate_positive_masks", {}),
            reports[right].get("_formula_candidate_positive_masks", {}),
        )
        rows_for_teacher = []
        for candidate_id in sorted(set(left_masks) & set(right_masks)):
            before, after = left_masks[candidate_id], right_masks[candidate_id]
            union = before | after
            rows_for_teacher.append(
                {"candidate_id": candidate_id, "jaccard": len(before & after) / len(union) if union else None}
            )
        gt_sensitivity["same_teacher_seed_jaccard"][f"{left}_vs_{right}"] = rows_for_teacher
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "reports": public,
        "gt_sensitivity": gt_sensitivity,
        "score_rows_sha256": sha256_file(paths["score_rows"]),
        "ground_truth_rows_sha256": sha256_file(paths["ground_truth_rows"]),
        "config_sha256": sha256_file(config_path),
        "cohort_inventory_sha256": cohort_digest,
        "input_hash": hashlib.sha256(
            canonical_json({label: report["input_identity"] for label, report in public.items()})
        ).hexdigest(),
    }
    paths["report"].write_bytes(canonical_json(payload) + b"\n")
    return paths
