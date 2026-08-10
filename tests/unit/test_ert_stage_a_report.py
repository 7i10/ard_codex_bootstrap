from __future__ import annotations

from ard.analysis.ert_stage_a_report import _paired


def test_paired_reports_rescue_harm_and_margin_delta() -> None:
    control = {
        1: {
            "robust_correct": False,
            "clean_correct": True,
            "clean_probability_margin": 0.1,
            "adversarial_probability_margin": -0.2,
        },
        2: {
            "robust_correct": True,
            "clean_correct": True,
            "clean_probability_margin": 0.2,
            "adversarial_probability_margin": 0.3,
        },
    }
    treatment = {
        1: {
            "robust_correct": True,
            "clean_correct": True,
            "clean_probability_margin": 0.2,
            "adversarial_probability_margin": 0.1,
        },
        2: {
            "robust_correct": False,
            "clean_correct": False,
            "clean_probability_margin": 0.0,
            "adversarial_probability_margin": 0.0,
        },
    }
    result = _paired(control, treatment, {1, 2})
    assert result["rescue_count"] == 1
    assert result["harm_count"] == 1
    assert result["net_rescue_count"] == 0
    assert abs(result["adversarial_margin_delta"]) < 1e-12
