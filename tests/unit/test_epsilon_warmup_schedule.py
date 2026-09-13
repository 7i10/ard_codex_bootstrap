from __future__ import annotations

import pytest

from ard.schedules.epsilon_warmup import epsilon_warmup_value


def test_epsilon_warmup_starts_at_zero() -> None:
    assert epsilon_warmup_value(0, warmup_epochs=10, target_epsilon=4 / 255) == pytest.approx(0.0)


def test_epsilon_warmup_reaches_target_exactly_at_warmup_epochs() -> None:
    assert epsilon_warmup_value(10, warmup_epochs=10, target_epsilon=4 / 255) == pytest.approx(4 / 255)


def test_epsilon_warmup_matches_hand_computed_midpoint() -> None:
    target = 4 / 255
    assert epsilon_warmup_value(5, warmup_epochs=10, target_epsilon=target) == pytest.approx(0.5 * target)


def test_epsilon_warmup_stays_at_target_past_warmup_epochs() -> None:
    target = 4 / 255
    assert epsilon_warmup_value(11, warmup_epochs=10, target_epsilon=target) == pytest.approx(target)
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
