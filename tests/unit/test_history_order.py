from __future__ import annotations

import pytest

from ard.policies.history_order import stable_risk_order, verify_h2_rank_equivalence


def test_stable_risk_order_breaks_ties_by_source_id() -> None:
    assert stable_risk_order({4: 1.0, 2: 1.0, 9: 0.0}) == [2, 4, 9]


def test_h2_rank_gate_accepts_negative_coefficient_and_rejects_positive() -> None:
    margins = {10: -0.2, 2: 0.3, 7: 0.1, 3: 0.3}
    accepted = verify_h2_rank_equivalence(margins, coefficient=-0.5, mean=0.0, std=1.0)
    assert accepted["exact"] is True
    rejected = verify_h2_rank_equivalence(margins, coefficient=0.5, mean=0.0, std=1.0)
    assert rejected["exact"] is False


def test_h2_rank_gate_rejects_invalid_standard_deviation() -> None:
    with pytest.raises(ValueError, match="standard deviation"):
        verify_h2_rank_equivalence({1: 0.0}, coefficient=-1.0, mean=0.0, std=0.0)
