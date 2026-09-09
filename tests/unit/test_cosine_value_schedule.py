from __future__ import annotations

import math

import pytest

from ard.schedules.cosine_value import cosine_anneal


def test_cosine_anneal_starts_at_start_value() -> None:
    assert cosine_anneal(start=2.5, end=2.0, iteration=0, total_iterations=100) == pytest.approx(2.5)


def test_cosine_anneal_approaches_end_value_at_the_final_iteration() -> None:
    total = 100
    value = cosine_anneal(start=2.5, end=2.0, iteration=total - 1, total_iterations=total)
    # The paper's own indexing (0..N-1) never exactly reaches the endpoint;
    # replicate that rather than rounding it away.
    assert value != pytest.approx(2.0)
    assert value == pytest.approx(2.0, abs=1e-2)


def test_cosine_anneal_matches_hand_computed_midpoint() -> None:
    total = 200
    iteration = total // 2
    expected = 2.0 + 0.5 * (2.5 - 2.0) * (1 + math.cos(math.pi * iteration / total))
    assert cosine_anneal(start=2.5, end=2.0, iteration=iteration, total_iterations=total) == pytest.approx(expected)


def test_cosine_anneal_supports_an_increasing_schedule() -> None:
    # lambda anneals 0.7 -> 0.95, i.e. start < end.
    first = cosine_anneal(start=0.7, end=0.95, iteration=0, total_iterations=100)
    last = cosine_anneal(start=0.7, end=0.95, iteration=99, total_iterations=100)
    assert first == pytest.approx(0.7)
    assert last > first
    assert last == pytest.approx(0.95, abs=1e-2)


def test_cosine_anneal_rejects_out_of_range_iteration_and_non_positive_total() -> None:
    with pytest.raises(ValueError, match="total_iterations must be positive"):
        cosine_anneal(start=1.0, end=0.0, iteration=0, total_iterations=0)
    with pytest.raises(ValueError, match="iteration must lie in"):
        cosine_anneal(start=1.0, end=0.0, iteration=10, total_iterations=10)
    with pytest.raises(ValueError, match="iteration must lie in"):
        cosine_anneal(start=1.0, end=0.0, iteration=-1, total_iterations=10)
