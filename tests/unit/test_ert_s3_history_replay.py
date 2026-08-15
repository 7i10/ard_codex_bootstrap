from __future__ import annotations

import numpy as np

from ard.analysis.ert_s3_history_replay import asymmetric_state, rule_signal, summarize_rule


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
