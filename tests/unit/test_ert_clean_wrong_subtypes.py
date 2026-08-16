from __future__ import annotations

from ard.analysis.ert_clean_wrong_subtypes import GROUPS, _effect, _group, _summary


def _row(*, clean: bool, robust: bool, treatment_clean: bool, treatment_robust: bool) -> dict[str, object]:
    return {
        "clean_correct": clean,
        "robust_correct": robust,
        "student_clean_correct": clean,
        "student_adv_correct": robust,
        "student_clean_margin": 0.1,
        "student_adv_margin": -0.1,
        "student_clean_true_probability": 0.4,
        "student_adv_true_probability": 0.2,
        "teacher_clean_correct": True,
        "teacher_adv_correct": False,
        "teacher_clean_margin": 0.2,
        "teacher_adv_margin": -0.2,
        "teacher_clean_true_probability": 0.6,
        "teacher_adv_true_probability": 0.3,
        "delta_teacher_margin": 0.4,
        "true_label": 0,
    }


def test_rescue_groups_are_mutually_exclusive() -> None:
    base = _row(clean=False, robust=False, treatment_clean=False, treatment_robust=False)
    assert _group(base, {**base, "clean_correct": True, "robust_correct": True}) == GROUPS[0]
    assert _group(base, {**base, "clean_correct": True}) == GROUPS[1]
    assert _group(base, {**base, "robust_correct": True}) == GROUPS[2]
    assert _group(base, base) == GROUPS[3]


def test_summary_reports_teacher_rates_and_quantiles() -> None:
    row = _row(clean=False, robust=False, treatment_clean=True, treatment_robust=True)
    result = _summary([row, {**row, "teacher_adv_correct": True, "student_clean_margin": 0.3}])
    assert result["n"] == 2
    assert result["teacher_clean_correct_rate"] == 1.0
    assert result["teacher_adv_correct_rate"] == 0.5
    assert result["student_clean_margin"]["mean"] == 0.2


def test_effect_separates_accuracy_and_margin_delta() -> None:
    base = [
        {
            "sample_id": 1,
            "clean_correct": False,
            "robust_correct": False,
            "clean_probability_margin": -0.2,
            "adversarial_probability_margin": -0.3,
        },
        {
            "sample_id": 2,
            "clean_correct": True,
            "robust_correct": True,
            "clean_probability_margin": 0.2,
            "adversarial_probability_margin": 0.3,
        },
    ]
    treatment = [
        {
            "sample_id": 1,
            "clean_correct": True,
            "robust_correct": False,
            "clean_probability_margin": -0.1,
            "adversarial_probability_margin": -0.4,
        },
        {
            "sample_id": 2,
            "clean_correct": False,
            "robust_correct": True,
            "clean_probability_margin": 0.1,
            "adversarial_probability_margin": 0.2,
        },
    ]
    clean = _effect(base, treatment, "clean_probability_margin")
    robust = _effect(base, treatment, "adversarial_probability_margin")
    assert clean["accuracy_delta"] == 0.0
    assert clean["margin_delta"] == 0.0
    assert robust["accuracy_delta"] == 0.0
    assert abs(float(robust["margin_delta"]) + 0.1) < 1e-12
    assert clean["accuracy_delta"] == clean["rescue_rate"] - clean["harm_rate"]
