from __future__ import annotations

from ard.analysis.ert_stage_a_modifiers import _effect, _tertiles


def test_tertiles_are_deterministic_and_cover_ids() -> None:
    rows = {i: {"x": float(i)} for i in range(6)}
    groups = _tertiles(set(rows), rows, "x")
    assert set.union(*groups.values()) == set(rows)
    assert all(groups[name] for name in groups)


def test_effect_reports_paired_rescue_and_harm() -> None:
    control = {
        1: {"robust_correct": False, "clean_correct": True, "adversarial_probability_margin": 0.0},
        2: {"robust_correct": True, "clean_correct": True, "adversarial_probability_margin": 0.2},
    }
    treatment = {
        1: {"robust_correct": True, "clean_correct": True, "adversarial_probability_margin": 0.1},
        2: {"robust_correct": False, "clean_correct": False, "adversarial_probability_margin": 0.0},
    }
    result = _effect(control, treatment, {1, 2})
    assert result["rescue_rate"] == 0.5
    assert result["harm_rate"] == 0.5
    assert result["net_rescue_rate"] == 0.0
