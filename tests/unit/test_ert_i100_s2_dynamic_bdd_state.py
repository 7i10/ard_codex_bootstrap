from __future__ import annotations

from ard.analysis.ert_i100_s2_dynamic_bdd_state import _matches_sha256, canonical_state_summary


def _row(sample_id: int, student_correct: bool, student_margin: float, teacher_correct: bool, teacher_margin: float):
    return {
        "sample_id": sample_id,
        "student_ce20_adv_correct": student_correct,
        "student_ce20_adv_margin": student_margin,
        "teacher_ce20_adv_correct": teacher_correct,
        "teacher_ce20_adv_margin": teacher_margin,
    }


def test_canonical_state_summary_uses_stable_id_to_break_q10_ties() -> None:
    rows = [
        _row(3, True, 0.1, True, 0.3),
        _row(1, True, 0.1, True, 0.3),
        _row(2, True, 0.2, True, 0.4),
        _row(4, True, 0.3, True, 0.5),
        _row(5, True, 0.4, True, 0.6),
        _row(6, True, 0.5, True, 0.7),
        _row(7, True, 0.6, True, 0.8),
        _row(8, True, 0.7, True, 0.9),
        _row(9, True, 0.8, True, 1.0),
        _row(10, False, -0.1, False, -0.2),
    ]
    result = canonical_state_summary(rows)
    # ceil(.1 * 9) = 1; the lower stable ID wins the equal-margin Student tie.
    assert result["state_by_id"][1] == {"student": "S2", "teacher": "T2", "joint": "S2xT2"}
    assert result["state_by_id"][3] == {"student": "S1", "teacher": "T1", "joint": "S1xT1"}
    assert result["state_by_id"][10] == {"student": "S3", "teacher": "T3", "joint": "S3xT3"}
    assert result["joint_counts"]["S2xT2"] == 1
    assert result["joint_counts"]["S3xT3"] == 1


def test_canonical_state_summary_preserves_joint_count_total() -> None:
    rows = [_row(i, True, float(i), True, float(i + 1)) for i in range(1, 11)]
    result = canonical_state_summary(rows)
    assert result["row_count"] == 10
    assert sum(result["joint_counts"].values()) == 10


def test_teacher_sha_check_accepts_a_path_object(tmp_path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"frozen teacher")
    assert _matches_sha256(checkpoint, "ba0c97e201a2ce05f581d231a550aa246286c3ac9ac618b482e9a4ac2629b68b")
