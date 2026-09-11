"""Formula-level tests for plan 0098's gap-adaptive lambda schedule.

Fixture-free, no Trainer required -- pure hand-computed sequences, matching
this project's existing schedule-primitive test style
(tests/unit/test_cosine_value_schedule.py).
"""

from __future__ import annotations

import pytest

from ard.schedules.gap_adaptive import GapAdaptiveState, gap_adaptive_step, lambda_from_state

LOW, HIGH = 0.7, 0.95


def test_first_epoch_has_no_prior_state_and_gap_ema_equals_the_raw_gap() -> None:
    state, lambda_value = gap_adaptive_step(
        train_robust_accuracy=0.60,
        val_pgd_accuracy=0.40,
        previous_state=None,
        beta=0.9,
        lambda_low=LOW,
        lambda_high=HIGH,
    )
    assert state.gap_ema == pytest.approx(0.20)
    # No history yet: running_max equals this epoch's own gap, so severity
    # saturates at 1.0 and lambda hits its ceiling.
    assert state.running_max == pytest.approx(0.20)
    assert lambda_value == pytest.approx(HIGH)


def test_hand_computed_three_epoch_sequence_matches_the_formula_exactly() -> None:
    """Eq. from docs/plans/0098: gap_ema_e = beta*gap_ema_{e-1} + (1-beta)*raw_gap_e;
    running_max_e = max(running_max_{e-1}, gap_ema_e);
    severity_e = clip(gap_ema_e / max(eps, running_max_e), 0, 1);
    lambda_e = low + (high-low)*severity_e."""
    beta = 0.5
    # Epoch 0: raw_gap = 0.6 - 0.4 = 0.2
    state0, lambda0 = gap_adaptive_step(
        train_robust_accuracy=0.6, val_pgd_accuracy=0.4, previous_state=None, beta=beta, lambda_low=LOW, lambda_high=HIGH
    )
    assert state0.gap_ema == pytest.approx(0.2)
    assert state0.running_max == pytest.approx(0.2)
    assert lambda0 == pytest.approx(HIGH)  # severity 1.0 (no history to fall short of)

    # Epoch 1: raw_gap = 0.7 - 0.3 = 0.4; gap_ema = 0.5*0.2 + 0.5*0.4 = 0.3
    state1, lambda1 = gap_adaptive_step(
        train_robust_accuracy=0.7,
        val_pgd_accuracy=0.3,
        previous_state=state0,
        beta=beta,
        lambda_low=LOW,
        lambda_high=HIGH,
    )
    assert state1.gap_ema == pytest.approx(0.3)
    assert state1.running_max == pytest.approx(0.3)  # 0.3 > previous 0.2
    assert lambda1 == pytest.approx(HIGH)  # still the worst gap so far -> severity 1.0

    # Epoch 2: raw_gap = 0.5 - 0.4 = 0.1; gap_ema = 0.5*0.3 + 0.5*0.1 = 0.2
    # severity = 0.2 / max(eps, 0.3) = 2/3
    state2, lambda2 = gap_adaptive_step(
        train_robust_accuracy=0.5,
        val_pgd_accuracy=0.4,
        previous_state=state1,
        beta=beta,
        lambda_low=LOW,
        lambda_high=HIGH,
    )
    assert state2.gap_ema == pytest.approx(0.2)
    assert state2.running_max == pytest.approx(0.3)  # unchanged: 0.2 < 0.3
    expected_severity = 0.2 / 0.3
    assert lambda2 == pytest.approx(LOW + (HIGH - LOW) * expected_severity)


def test_flat_zero_gap_does_not_divide_by_zero_and_yields_the_lambda_floor() -> None:
    """A run with zero train/val gap from the start (no overfitting signal
    at all) must not raise ZeroDivisionError and must read as zero severity."""
    state, lambda_value = gap_adaptive_step(
        train_robust_accuracy=0.5,
        val_pgd_accuracy=0.5,
        previous_state=None,
        beta=0.9,
        lambda_low=LOW,
        lambda_high=HIGH,
    )
    assert state.gap_ema == pytest.approx(0.0)
    assert state.running_max == pytest.approx(0.0)
    assert lambda_value == pytest.approx(LOW)


def test_negative_gap_val_ahead_of_train_also_yields_the_lambda_floor_without_error() -> None:
    state, lambda_value = gap_adaptive_step(
        train_robust_accuracy=0.4,
        val_pgd_accuracy=0.5,
        previous_state=None,
        beta=0.9,
        lambda_low=LOW,
        lambda_high=HIGH,
    )
    assert state.gap_ema == pytest.approx(-0.1)
    assert lambda_value == pytest.approx(LOW)


def test_severity_is_clipped_at_one_even_if_gap_ema_exceeded_the_prior_running_max_arithmetically() -> None:
    # running_max is updated to include the current gap_ema before the
    # division, so severity can never exceed 1.0 by construction; this pins
    # that invariant rather than trusting the clip alone.
    state, lambda_value = gap_adaptive_step(
        train_robust_accuracy=1.0, val_pgd_accuracy=0.0, previous_state=None, beta=0.9, lambda_low=LOW, lambda_high=HIGH
    )
    assert lambda_value == pytest.approx(HIGH)


def test_lambda_from_state_is_a_pure_function_reusable_for_checkpoint_resume() -> None:
    state = GapAdaptiveState(gap_ema=0.15, running_max=0.3)
    assert lambda_from_state(state, lambda_low=LOW, lambda_high=HIGH) == pytest.approx(LOW + (HIGH - LOW) * 0.5)


def test_state_round_trips_through_its_plain_dict_form() -> None:
    state = GapAdaptiveState(gap_ema=0.123, running_max=0.456)
    assert GapAdaptiveState.from_dict(state.to_dict()) == state


def test_beta_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="beta"):
        gap_adaptive_step(
            train_robust_accuracy=0.5, val_pgd_accuracy=0.4, previous_state=None, beta=1.0, lambda_low=LOW, lambda_high=HIGH
        )
    with pytest.raises(ValueError, match="beta"):
        gap_adaptive_step(
            train_robust_accuracy=0.5, val_pgd_accuracy=0.4, previous_state=None, beta=0.0, lambda_low=LOW, lambda_high=HIGH
        )
