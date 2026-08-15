"""Offline history-only replay for the ERT S3 routing predicate.

This module deliberately consumes only the saved KL-PGD10 state trajectories.
It never loads a checkpoint, creates an attack, or reads endpoint metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class HistoryReplayError(RuntimeError):
    """The offline trajectory contract cannot be established."""


REQUIRED_COLUMNS = (
    "epoch",
    "sample_id",
    "class_id",
    "student_adv_correct",
    "teacher_adv_correct",
)
RULES: tuple[str, ...] = (
    "Instant",
    "Consecutive-2",
    "Consecutive-3",
    "Majority-3",
    "Majority-5",
    "Loose-5",
)
ASYMMETRIC_RULES: tuple[str, ...] = (
    "Majority-5_exit-2-correct",
    "Majority-5_exit-3-correct",
)
MAJORITY3_PERSISTENCE_RULES: tuple[str, ...] = (
    "Majority-3_exit-2-correct",
    "Majority-3_exit-3-correct",
    "Majority-3_min-dwell-2",
    "Majority-3_min-dwell-3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rolling_count(flags: np.ndarray, window: int) -> np.ndarray:
    counts = np.zeros(flags.shape, dtype=np.int16)
    for epoch_index in range(flags.shape[1]):
        start = max(0, epoch_index - window + 1)
        counts[:, epoch_index] = flags[:, start : epoch_index + 1].sum(axis=1)
    return counts


def _full_window_signal(flags: np.ndarray, *, window: int, threshold: int) -> np.ndarray:
    signal = _rolling_count(flags, window) >= threshold
    if window > 1:
        signal[:, : window - 1] = False
    return signal


def rule_signal(wrong: np.ndarray, name: str) -> np.ndarray:
    """Build a causal signal using visits through the current epoch only."""
    if name == "Instant":
        return wrong.copy()
    if name == "Consecutive-2":
        return _full_window_signal(wrong, window=2, threshold=2)
    if name == "Consecutive-3":
        return _full_window_signal(wrong, window=3, threshold=3)
    if name == "Majority-3":
        return _full_window_signal(wrong, window=3, threshold=2)
    if name == "Majority-5":
        return _full_window_signal(wrong, window=5, threshold=3)
    if name == "Loose-5":
        return _full_window_signal(wrong, window=5, threshold=2)
    raise HistoryReplayError(f"unknown history rule: {name}")


def _entry_exit_state(wrong: np.ndarray, *, entry_rule: str, exit_correct_visits: int) -> np.ndarray:
    """Apply a history entry rule with a consecutive-correct exit."""
    if exit_correct_visits not in {2, 3}:
        raise HistoryReplayError("exit_correct_visits must be 2 or 3")
    entry = rule_signal(wrong, entry_rule)
    state = np.zeros_like(wrong, dtype=bool)
    for sample_index in range(wrong.shape[0]):
        active = False
        correct_streak = 0
        for epoch_index in range(wrong.shape[1]):
            if not active:
                if entry[sample_index, epoch_index]:
                    active = True
                    correct_streak = 0
            elif wrong[sample_index, epoch_index]:
                correct_streak = 0
            else:
                correct_streak += 1
                if correct_streak >= exit_correct_visits:
                    active = False
                    correct_streak = 0
            state[sample_index, epoch_index] = active
    return state


def asymmetric_state(wrong: np.ndarray, *, exit_correct_visits: int) -> np.ndarray:
    """Apply Majority-5 entry with a consecutive-correct exit state machine."""
    return _entry_exit_state(wrong, entry_rule="Majority-5", exit_correct_visits=exit_correct_visits)


def minimum_dwell_state(wrong: np.ndarray, *, entry_rule: str, dwell_visits: int) -> np.ndarray:
    """Keep an entered action active for at least ``dwell_visits`` visits."""
    if dwell_visits not in {2, 3}:
        raise HistoryReplayError("dwell_visits must be 2 or 3")
    entry = rule_signal(wrong, entry_rule)
    state = np.zeros_like(wrong, dtype=bool)
    for sample_index in range(wrong.shape[0]):
        active = False
        age = 0
        for epoch_index in range(wrong.shape[1]):
            if not active:
                if entry[sample_index, epoch_index]:
                    active = True
                    age = 1
            elif age < dwell_visits:
                age += 1
            elif not entry[sample_index, epoch_index]:
                active = False
                age = 0
            else:
                age += 1
            state[sample_index, epoch_index] = active
    return state


def load_trajectory(path: Path, *, expected_count: int, expected_epochs: tuple[int, ...]) -> dict[str, Any]:
    """Load and validate one sparse-ID trajectory into sample-major arrays."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise HistoryReplayError("history replay requires pyarrow") from exc
    if not path.is_file():
        raise HistoryReplayError(f"trajectory is missing: {path}")
    table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    if table.num_rows != expected_count * len(expected_epochs):
        raise HistoryReplayError(
            f"trajectory row count mismatch: {path} has {table.num_rows}, "
            f"expected {expected_count * len(expected_epochs)}"
        )
    columns = {name: np.asarray(table[name]) for name in REQUIRED_COLUMNS}
    order = np.lexsort((columns["epoch"], columns["sample_id"]))
    sample_ids = columns["sample_id"][order].astype(np.int64, copy=False)
    epochs = columns["epoch"][order].astype(np.int64, copy=False)
    classes = columns["class_id"][order].astype(np.int64, copy=False)
    expected_epoch_array = np.asarray(expected_epochs, dtype=np.int64)
    matrix_shape = (expected_count, len(expected_epochs))
    sample_matrix = sample_ids.reshape(matrix_shape)
    epoch_matrix = epochs.reshape(matrix_shape)
    if not np.all(sample_matrix == sample_matrix[:, :1]):
        raise HistoryReplayError(f"sample IDs are not contiguous by epoch: {path}")
    if not np.all(epoch_matrix == expected_epoch_array[None, :]):
        raise HistoryReplayError(f"epoch coverage/order mismatch: {path}")
    if np.unique(sample_ids).size != expected_count:
        raise HistoryReplayError(f"sample ID duplication/universe mismatch: {path}")
    class_matrix = classes.reshape(matrix_shape)
    if not np.all(class_matrix == class_matrix[:, :1]):
        raise HistoryReplayError(f"class label changes for a stable sample ID: {path}")
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "sample_ids": sample_matrix[:, 0].copy(),
        "classes": class_matrix[:, 0].copy(),
        "epochs": expected_epoch_array.copy(),
        "student_adv_correct": columns["student_adv_correct"][order].reshape(matrix_shape).astype(bool),
        "teacher_adv_correct": columns["teacher_adv_correct"][order].reshape(matrix_shape).astype(bool),
    }
    payload = [[int(sample_id), int(label)] for sample_id, label in zip(result["sample_ids"], result["classes"])]
    result["stable_id_class_sha256"] = hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def _scalar(value: float | int) -> float | int:
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": int(max(values)),
    }


def _run_lengths(state: np.ndarray) -> list[int]:
    lengths: list[int] = []
    for row in state:
        current = 0
        for active in row:
            if bool(active):
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        if current:
            lengths.append(current)
    return lengths


def _entry_delays(state: np.ndarray, wrong: np.ndarray) -> list[int]:
    delays: list[int] = []
    for state_row, wrong_row in zip(state, wrong):
        wrong_indices = np.flatnonzero(wrong_row)
        active_indices = np.flatnonzero(state_row)
        if wrong_indices.size and active_indices.size:
            delays.append(int(active_indices[0] - wrong_indices[0]))
    return delays


def _exit_delays(state: np.ndarray, wrong: np.ndarray) -> list[int]:
    delays: list[int] = []
    for state_row, wrong_row in zip(state, wrong):
        exits = np.flatnonzero(state_row[:-1] & ~state_row[1:]) + 1
        for exit_index in exits:
            previous_wrong = np.flatnonzero(wrong_row[:exit_index])
            if previous_wrong.size:
                delays.append(int(exit_index - previous_wrong[-1]))
    return delays


def summarize_rule(
    *,
    state: np.ndarray,
    wrong: np.ndarray,
    teacher_correct: np.ndarray,
    future_persistent: np.ndarray,
    epochs: np.ndarray,
    name: str,
) -> dict[str, Any]:
    """Summarize state/action stability without using endpoint performance."""
    action = state & teacher_correct
    state_switch = state[:, 1:] != state[:, :-1]
    teacher_flip = teacher_correct[:, 1:] != teacher_correct[:, :-1]
    action_switch = action[:, 1:] != action[:, :-1]
    student_only = action_switch & state_switch & ~teacher_flip
    teacher_only = action_switch & ~state_switch & teacher_flip
    mixed = action_switch & state_switch & teacher_flip
    entries = state[:, 1:] & ~state[:, :-1]
    exits = ~state[:, 1:] & state[:, :-1]
    # A re-entry after exactly one inactive visit is state[t-2]=1,state[t-1]=0,state[t]=1.
    one_visit = np.zeros_like(entries)
    if state.shape[1] >= 3:
        one_visit[:, 1:] = state[:, :-2] & ~state[:, 1:-1] & state[:, 2:]
    valid_future = np.zeros_like(future_persistent)
    valid_future[:, : max(0, future_persistent.shape[1] - 3)] = True
    future_count = int((future_persistent & valid_future).sum())
    active_future = int((state & future_persistent & valid_future).sum())
    action_future = int((action & future_persistent & valid_future).sum())
    transient = wrong & ~future_persistent & valid_future
    transient_count = int(transient.sum())
    active_transient = int((state & transient).sum())
    lengths = _run_lengths(state)
    delays = _entry_delays(state, wrong)
    exit_delays = _exit_delays(state, wrong)
    result: dict[str, Any] = {
        "rule": name,
        "epochs": [int(epoch) for epoch in epochs],
        "state_active_fraction": float(state.mean()),
        "action_active_fraction": float(action.mean()),
        "unique_active_ids": int(state.any(axis=1).sum()),
        "unique_action_ids": int(action.any(axis=1).sum()),
        "state_switch_count": int(state_switch.sum()),
        "action_switch_count": int(action_switch.sum()),
        "switches_per_sample": float(action_switch.sum() / state.shape[0]),
        "teacher_correctness_flip_count": int(teacher_flip.sum()),
        "switch_attribution": {
            "student_history_only": int(student_only.sum()),
            "teacher_correctness_only": int(teacher_only.sum()),
            "student_and_teacher": int(mixed.sum()),
        },
        "entry_count": int(entries.sum()),
        "exit_count": int(exits.sum()),
        "reentry_count": int(max(0, entries.sum() - state.any(axis=1).sum())),
        "one_visit_reentry_count": int(one_visit.sum()),
        "one_visit_reentry_rate": float(one_visit.sum() / max(1, entries.sum())),
        "active_duration": _summary(lengths),
        "entry_delay_from_first_wrong": _summary(delays),
        "exit_delay_from_last_wrong": _summary(exit_delays),
        "near_future_reference": {
            "definition": "next 3 visits contain at least 2 adversarial-wrong visits",
            "eligible_positions": int(valid_future.sum()),
            "positive_positions": future_count,
            "state_capture_rate": float(active_future / future_count) if future_count else None,
            "action_capture_rate": float(action_future / future_count) if future_count else None,
        },
        "transient_failure_activation_rate": float(active_transient / transient_count)
        if transient_count
        else None,
    }
    result["longest_active_run"] = int(max(lengths, default=0))
    return result


def analyze_run(*, label: str, trajectory: dict[str, Any]) -> dict[str, Any]:
    wrong = ~trajectory["student_adv_correct"]
    teacher_correct = trajectory["teacher_adv_correct"]
    future_counts = np.zeros_like(wrong, dtype=np.int16)
    if wrong.shape[1] > 3:
        for epoch_index in range(wrong.shape[1] - 3):
            future_counts[:, epoch_index] = wrong[:, epoch_index + 1 : epoch_index + 4].sum(axis=1)
    future_persistent = future_counts >= 2
    rules: dict[str, np.ndarray] = {name: rule_signal(wrong, name) for name in RULES}
    rules[ASYMMETRIC_RULES[0]] = asymmetric_state(wrong, exit_correct_visits=2)
    rules[ASYMMETRIC_RULES[1]] = asymmetric_state(wrong, exit_correct_visits=3)
    rules[MAJORITY3_PERSISTENCE_RULES[0]] = _entry_exit_state(
        wrong, entry_rule="Majority-3", exit_correct_visits=2
    )
    rules[MAJORITY3_PERSISTENCE_RULES[1]] = _entry_exit_state(
        wrong, entry_rule="Majority-3", exit_correct_visits=3
    )
    rules[MAJORITY3_PERSISTENCE_RULES[2]] = minimum_dwell_state(
        wrong, entry_rule="Majority-3", dwell_visits=2
    )
    rules[MAJORITY3_PERSISTENCE_RULES[3]] = minimum_dwell_state(
        wrong, entry_rule="Majority-3", dwell_visits=3
    )
    return {
        "label": label,
        "trajectory": {
            "path": trajectory["path"],
            "sha256": trajectory["sha256"],
            "row_count": int(wrong.size),
            "sample_count": int(wrong.shape[0]),
            "epoch_count": int(wrong.shape[1]),
            "epochs": [int(epoch) for epoch in trajectory["epochs"]],
            "stable_id_class_sha256": trajectory["stable_id_class_sha256"],
        },
        "rules": {
            name: summarize_rule(
                state=state,
                wrong=wrong,
                teacher_correct=teacher_correct,
                future_persistent=future_persistent,
                epochs=trajectory["epochs"],
                name=name,
            )
            for name, state in rules.items()
        },
    }


def write_report(
    *, output: Path, reports: dict[str, Any], config_path: Path, source_git_sha: str
) -> dict[str, Any]:
    if output.exists():
        raise HistoryReplayError(f"refusing to overwrite existing report: {output}")
    report = {
        "schema_version": 1,
        "contract": "ert_s3_history_replay_v1",
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "source_git_sha": source_git_sha,
        "future_reference_is_descriptive_only": True,
        "runs": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_sha256"] = _sha256(output)
    return report
