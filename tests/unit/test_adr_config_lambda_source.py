"""Schema tests for plan 0098's AdrConfig.lambda_source / gap_smoothing_beta.

Existing plan-0097 configs never set these fields; the whole point of this
plan is that they must not need to -- the default must reproduce plan
0097's exact validated behavior.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ard.config.schema import AdrConfig


def test_default_adr_config_is_unchanged_cosine_behavior() -> None:
    config = AdrConfig()
    assert config.lambda_source == "cosine"
    assert config.gap_smoothing_beta == 0.9


def test_a_plan_0097_style_adr_config_that_never_mentions_the_new_fields_still_validates() -> None:
    config = AdrConfig(ema_decay=0.995, temperature_high=2.5, temperature_low=2.0, lambda_low=0.7, lambda_high=0.95)
    assert config.lambda_source == "cosine"


def test_gap_smoothing_beta_is_rejected_when_lambda_source_is_cosine() -> None:
    with pytest.raises(ValidationError, match="gap_smoothing_beta is only meaningful"):
        AdrConfig(lambda_source="cosine", gap_smoothing_beta=0.8)


def test_gap_adaptive_lambda_source_accepts_a_custom_smoothing_beta() -> None:
    config = AdrConfig(lambda_source="gap_adaptive", gap_smoothing_beta=0.8)
    assert config.lambda_source == "gap_adaptive"
    assert config.gap_smoothing_beta == 0.8


def test_gap_adaptive_lambda_source_keeps_the_default_smoothing_beta_if_unset() -> None:
    config = AdrConfig(lambda_source="gap_adaptive")
    assert config.gap_smoothing_beta == 0.9


@pytest.mark.parametrize("beta", [0.0, 1.0, -0.1, 1.1])
def test_gap_smoothing_beta_must_lie_strictly_inside_the_unit_interval(beta: float) -> None:
    with pytest.raises(ValidationError):
        AdrConfig(lambda_source="gap_adaptive", gap_smoothing_beta=beta)
