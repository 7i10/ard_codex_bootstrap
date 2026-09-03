from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/aggregate_ert_i100_s2_dynamic_bdd_recovery.py"
SPEC = importlib.util.spec_from_file_location("aggregate_dynamic_bdd", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(clean: bool, robust: bool, clean_margin: float, robust_margin: float) -> dict[str, object]:
    return {
        "clean_correct": clean,
        "robust_correct": robust,
        "clean_probability_margin": clean_margin,
        "adversarial_probability_margin": robust_margin,
    }


def test_paired_effect_keeps_accuracy_and_rescue_harm_semantics_distinct() -> None:
    control = {
        1: _row(False, False, -0.2, -0.3),
        2: _row(True, True, 0.2, 0.3),
        3: _row(True, False, 0.1, -0.1),
    }
    treatment = {
        1: _row(True, True, 0.1, 0.2),
        2: _row(False, False, -0.1, -0.2),
        3: _row(True, True, 0.3, 0.2),
    }
    result = MODULE.paired_effect(control, treatment, [1, 2, 3])
    assert result["clean"] == {
        "accuracy_delta": 0.0,
        "margin_delta": pytest.approx(1 / 15),
        "rescue_count": 1,
        "harm_count": 1,
        "net_rescue_count": 0,
        "rescue_rate": pytest.approx(1 / 3),
        "harm_rate": pytest.approx(1 / 3),
        "net_rescue_rate": 0.0,
    }
    assert result["robust"]["accuracy_delta"] == pytest.approx(1 / 3)
    assert result["robust"]["net_rescue_rate"] == pytest.approx(1 / 3)
    assert result["robust"]["margin_delta"] == pytest.approx(0.1)


def test_state_replay_metadata_rejects_wrong_train_split(tmp_path: Path) -> None:
    rows = tmp_path / "state-rows.parquet"
    rows.write_bytes(b"immutable rows")
    metadata = {
        "contract": "ert_rslad_i100_s2_dynamic_bdd_state_replay_v1",
        "attack_identity_sha256": MODULE.ATTACK_SHA,
        "teacher_checkpoint_sha256": MODULE.TEACHER_SHA,
        "checkpoint_epoch": 114,
        "row_count": 45_000,
        "rows_sha256": MODULE.sha256(rows),
        "split_identity": {"name": "train", "count": 45_000, "sample_id_label_sha256": "wrong"},
    }
    with pytest.raises(MODULE.AggregationError, match="train split identity"):
        MODULE.validate_state_replay_metadata(metadata, rows_path=rows, expected_epoch=114)
