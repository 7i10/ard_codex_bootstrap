from __future__ import annotations

import pytest

from ard.schedules.epsilon_warmup import epsilon_warmup_value


def test_epsilon_warmup_epoch_zero_is_the_first_ramp_fraction_not_zero() -> None:
    """Matches warmup_multistep_multiplier's convention: never exactly the
    zero endpoint at epoch 0 (a fully-unattacked epoch would be a training-
    time confound, scientific review P1-1/P2-8)."""
    target = 4 / 255
    assert epsilon_warmup_value(0, warmup_epochs=10, target_epsilon=target) == pytest.approx(target / 10)


def test_epsilon_warmup_reaches_target_at_the_last_warmup_epoch() -> None:
    target = 4 / 255
    assert epsilon_warmup_value(9, warmup_epochs=10, target_epsilon=target) == pytest.approx(target)


def test_epsilon_warmup_matches_hand_computed_midpoint() -> None:
    target = 4 / 255
    assert epsilon_warmup_value(4, warmup_epochs=10, target_epsilon=target) == pytest.approx(0.5 * target)


def test_epsilon_warmup_stays_at_target_past_warmup_epochs() -> None:
    target = 4 / 255
    assert epsilon_warmup_value(10, warmup_epochs=10, target_epsilon=target) == pytest.approx(target)
    assert epsilon_warmup_value(49, warmup_epochs=10, target_epsilon=target) == pytest.approx(target)


def test_epsilon_warmup_disabled_reproduces_todays_behavior_exactly() -> None:
    """warmup_epochs=0 is the default/unset case: every epoch is the target, unchanged."""
    target = 4 / 255
    for epoch in (0, 1, 25, 49):
        assert epsilon_warmup_value(epoch, warmup_epochs=0, target_epsilon=target) == pytest.approx(target)


def test_epsilon_warmup_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="warmup_epochs must be non-negative"):
        epsilon_warmup_value(0, warmup_epochs=-1, target_epsilon=4 / 255)
    with pytest.raises(ValueError, match="epoch must be non-negative"):
        epsilon_warmup_value(-1, warmup_epochs=10, target_epsilon=4 / 255)
