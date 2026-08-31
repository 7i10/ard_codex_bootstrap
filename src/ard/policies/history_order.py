"""Frozen history-risk rank helpers used by the first ordering intervention."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def stable_risk_order(scores: Mapping[int, float]) -> list[int]:
    """Return descending-score IDs with source-ID tie breaking."""
    return [sample_id for sample_id, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), int(item[0])))]


def verify_h2_rank_equivalence(
    margin_ema: Mapping[int, float],
    *,
    coefficient: float,
    mean: float,
    std: float,
) -> dict[str, Any]:
    """Check frozen one-feature Ridge risk against direct ``-margin_ema`` rank.

    The intercept is irrelevant to rank and standardization is monotone when
    ``std > 0``.  We nevertheless compute the complete frozen score and use
    the same stable-ID tie break on both sides, so this is an executable gate
    rather than an assumption about the sign.
    """
    if not margin_ema:
        raise ValueError("H2 rank gate requires at least one sample")
    if std <= 0.0:
        raise ValueError("H2 predictor standard deviation must be positive")
    if any(isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0 for sample_id in margin_ema):
        raise ValueError("H2 sample IDs must be non-negative integers")
    values = {int(sample_id): float(value) for sample_id, value in margin_ema.items()}
    if any(not (value == value and abs(value) != float("inf")) for value in values.values()):
        raise ValueError("H2 margin_ema values must be finite")
    ridge_scores = {
        sample_id: float(coefficient) * ((value - float(mean)) / float(std))
        for sample_id, value in values.items()
    }
    direct_scores = {sample_id: -value for sample_id, value in values.items()}
    ridge_order = stable_risk_order(ridge_scores)
    direct_order = stable_risk_order(direct_scores)
    return {
        "exact": ridge_order == direct_order,
        "n": len(values),
        "coefficient": float(coefficient),
        "mean": float(mean),
        "std": float(std),
        "ridge_order_sha256": _order_sha256(ridge_order),
        "direct_order_sha256": _order_sha256(direct_order),
        "tie_count": len(values) - len({value for value in values.values()}),
    }


def _order_sha256(order: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(value) for value in order], separators=(",", ":")).encode()).hexdigest()
