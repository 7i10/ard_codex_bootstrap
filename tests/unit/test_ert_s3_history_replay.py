from __future__ import annotations

import numpy as np

from ard.analysis.ert_s3_history_replay import (
    _entry_exit_state,
    asymmetric_state,
    minimum_dwell_state,
    rule_signal,
    summarize_rule,
)


def test_history_rules_are_causal_and_distinct() -> None:
    wrong = np.array([[False, True, True, False, True, False]], dtype=bool)
    instant = rule_signal(wrong, "Instant")
    consecutive_two = rule_signal(wrong, "Consecutive-2")
    majority_three = rule_signal(wrong, "Majority-3")
    majority_five = rule_signal(wrong, "Majority-5")
    assert instant.tolist() == [[False, True, True, False, True, False]]
    assert consecutive_two.tolist() == [[False, False, True, False, False, False]]
    assert majority_three.tolist() == [[False, False, True, True, True, False]]
    assert majority_five.tolist() == [[False, False, False, False, True, True]]


def test_asymmetric_exit_requires_consecutive_correct_visits() -> None:
    wrong = np.array([[True, True, False, True, False, False, False, False]], dtype=bool)
    exit_two = asymmetric_state(wrong, exit_correct_visits=2)
    exit_three = asymmetric_state(wrong, exit_correct_visits=3)
    assert exit_two.tolist() == [[False, False, False, False, True, True, False, False]]
    # Majority-5 is entered at the fifth visit; one correct visit does not exit.
    assert exit_three.tolist() == [[False, False, False, False, True, True, True, False]]


def test_switch_attribution_separates_teacher_flips() -> None:
    state = np.array([[False, True, True, False]], dtype=bool)
    wrong = np.array([[False, True, True, False]], dtype=bool)
    teacher_correct = np.array([[True, True, False, False]], dtype=bool)
    future = np.zeros_like(state)
    result = summarize_rule(
        state=state,
        wrong=wrong,
        teacher_correct=teacher_correct,
        future_persistent=future,
        epochs=np.arange(4),
        name="fixture",
    )
    assert result["action_switch_count"] == 2
    assert result["switch_attribution"]["student_history_only"] == 1
    assert result["switch_attribution"]["teacher_correctness_only"] == 1


def test_majority3_exit_and_dwell_are_distinct_persistence_controls() -> None:
    wrong = np.array([[True, False, True, False, False, False, False]], dtype=bool)
    exit_two = _entry_exit_state(wrong, entry_rule="Majority-3", exit_correct_visits=2)
    exit_three = _entry_exit_state(wrong, entry_rule="Majority-3", exit_correct_visits=3)
    dwell_two = minimum_dwell_state(wrong, entry_rule="Majority-3", dwell_visits=2)
    dwell_three = minimum_dwell_state(wrong, entry_rule="Majority-3", dwell_visits=3)
    # Majority-3 first enters at visit 3; the exit rules count correct visits,
    # while dwell rules only guarantee a minimum active duration.
    assert exit_two.tolist() == [[False, False, True, True, False, False, False]]
    assert exit_three.tolist() == [[False, False, True, True, True, False, False]]
    assert dwell_two.tolist() == [[False, False, True, True, False, False, False]]
    assert dwell_three.tolist() == [[False, False, True, True, True, False, False]]

    # A later Majority-3 re-entry separates “correct streak” exit from a
    # minimum dwell: the dwell rule exits as soon as its minimum is met,
    # whereas the exit rule keeps waiting for consecutive correct visits.
    reentry_wrong = np.array([[True, False, True, False, True, False, False]], dtype=bool)
    exit_two_reentry = _entry_exit_state(reentry_wrong, entry_rule="Majority-3", exit_correct_visits=2)
    dwell_two_reentry = minimum_dwell_state(reentry_wrong, entry_rule="Majority-3", dwell_visits=2)
    assert exit_two_reentry.tolist() == [[False, False, True, True, True, True, False]]
    assert dwell_two_reentry.tolist() == [[False, False, True, True, True, False, False]]
