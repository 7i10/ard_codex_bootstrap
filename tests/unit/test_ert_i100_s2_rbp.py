from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_ert_i100_s2_rbp import canonical_s2_t1, write_mask


def _train_rows() -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for sid in range(10):
        positive = sid < 8
        rows[sid] = {
            "sample_id": sid,
            "class_id": sid % 2,
            "student_ce20_adv_correct": positive,
            "student_ce20_adv_margin": 0.1 + 0.1 * sid if positive else -0.1,
            "teacher_ce20_adv_correct": positive,
            # Student S2 is sid 0; make it Teacher T1 while sid 1 is T2.
            "teacher_ce20_adv_margin": 0.8 if sid == 0 else 0.1 if sid == 1 else 0.2 + 0.1 * sid,
        }
    return rows


def test_canonical_s2_t1_uses_positive_q10_and_teacher_t1() -> None:
    selected, counts = canonical_s2_t1(_train_rows())
    assert selected == {0}
    assert counts["student_s2_count"] == 1
    assert counts["teacher_t2_count"] == 1
    assert counts["teacher_t1_count"] == 7


def test_mask_schema_supports_validation_field_names(tmp_path: Path) -> None:
    train = _train_rows()
    validation = {
        sid: {
            "sample_id": sid,
            "true_label": int(row["class_id"]),
            "student_adv_correct": bool(row["student_ce20_adv_correct"]),
            "student_adv_margin": float(row["student_ce20_adv_margin"]),
            "teacher_adv_correct": bool(row["teacher_ce20_adv_correct"]),
            "teacher_adv_margin": float(row["teacher_ce20_adv_margin"]),
        }
        for sid, row in train.items()
    }
    output = tmp_path / "mask.json"
    metadata = write_mask(run="dev-1", train_rows=train, val_rows=validation, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["contract"] == "ert_rslad_i100_s2_rbp_masks_v1"
    assert payload["masks"]["s2_t1"]["selected_ids"] == [0]
    assert payload["masks"]["validation_s2_t1"]["selected_ids"] == [0]
    assert metadata["sha256"]
