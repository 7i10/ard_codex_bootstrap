from __future__ import annotations

# ruff: noqa: E501
import pytest

from scripts.analyze_ffnr_human_review import HumanReviewAnalysisError, _argmax, _full_stats, _panel_stats


def _row(sample_id: int, class_id: int, *, robust: bool, clean: bool, adv: int, teacher_clean: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "student_robust_correct": robust,
        "student_clean_correct": clean,
        "teacher_adv_pred": adv,
        "teacher_clean_pred": teacher_clean,
    }


def test_argmax_rejects_ties() -> None:
    assert _argmax([1.0] + [0.0] * 9) == 0
    with pytest.raises(HumanReviewAnalysisError, match="tie"):
        _argmax([0.5, 0.5] + [0.0] * 8)


def test_full_stats_keeps_student_as_correctness_only() -> None:
    rows = {
        1: _row(1, 0, robust=False, clean=True, adv=8, teacher_clean=0),
        2: _row(2, 0, robust=True, clean=True, adv=0, teacher_clean=0),
    }
    report = _full_stats(rows)
    assert report["class_stats"]["0"]["student_robust_error_rate"] == 0.5
    assert report["teacher_adv_confusion"][0][8] == 1
    assert "student_predicted_class" not in report


def test_panel_stats_splits_clear_easy_and_hard() -> None:
    rows = [
        {**_row(1, 0, robust=False, clean=True, adv=8, teacher_clean=0), "classification": "clear_match", "commented": False},
        {**_row(2, 0, robust=True, clean=True, adv=0, teacher_clean=0), "classification": "clear_match", "commented": True},
        {**_row(3, 0, robust=False, clean=False, adv=8, teacher_clean=8), "classification": "ambiguous", "commented": True},
    ]
    report = _panel_stats(rows)
    assert report["groups"]["clear_easy"]["count"] == 1
    assert report["groups"]["clear_hard"]["count"] == 1
    assert report["groups"]["ambiguous"]["count"] == 1
    assert report["by_class"]["0"]["ambiguous"] == 1
