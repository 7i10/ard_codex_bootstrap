from __future__ import annotations

import pytest

import ard.cli.ffnr_state_mechanism as state_cli
from ard.analysis.ffnr_state_mechanism import (
    EXPECTED_ONLINE_ATTACK,
    FFNRStateMechanismError,
    _endpoint,
    _margin_rows,
    _model_rows,
    _risk_curves,
    _state_summaries,
    _surface,
    _threshold_tables,
    _validate_online_attack,
)

pytestmark = pytest.mark.unit


def _raw_row(*, correct: bool = False) -> dict[str, object]:
    return {
        "class_id": 1,
        "student_clean_probability_margin": 0.4,
        "student_adversarial_probability_margin": 0.2 if correct else -0.2,
        "student_clean_correct": True,
        "student_robust_correct": correct,
        "student_clean_to_adversarial_probability_margin_delta": -0.2 if correct else -0.6,
        "teacher_clean_probabilities": [0.05, 0.7, 0.25, 0, 0, 0, 0, 0, 0, 0],
        "teacher_adversarial_probabilities": [0.1, 0.2, 0.7, 0, 0, 0, 0, 0, 0, 0],
    }


def test_margin_signs_endpoint_and_compact_state_tables_fail_closed() -> None:
    raw = {39: {10: _raw_row(correct=False), 11: _raw_row(correct=True)}}
    rows = _margin_rows(raw, anchor=39)
    assert rows[10]["DeltaS"] == pytest.approx(0.6)
    assert rows[10]["DeltaT"] == pytest.approx(0.95)
    target = {10: 1, 11: 0}
    assert len(_risk_curves(rows, target)) == 60
    assert len(_surface(rows, target)) == 25
    assert len(_threshold_tables(rows, target)) == 8
    states = _state_summaries(rows, target, 0.3)
    assert set(states) == {"two_state", "three_state", "three_state_deltaS_threshold"}
    bad = _raw_row(correct=False)
    bad["student_robust_correct"] = True
    with pytest.raises(FFNRStateMechanismError, match="sign disagree"):
        _margin_rows({39: {10: bad}}, anchor=39)
    outcome = {
        epoch: {10: {"student_robust_correct": epoch == 189}, 11: {"student_robust_correct": False}}
        for epoch in (189, 194, 199)
    }
    assert _endpoint(outcome, "majority") == {10: 1, 11: 1}
    assert _endpoint(outcome, "all") == {10: 0, 11: 1}


def test_cross_seed_models_keep_response_terms_out_of_exact_dependence() -> None:
    def records(offset: float) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
        values = {
            item: {
                "mS_adv": offset + (item - 10) / 20,
                "mT_adv": offset + (item - 9) / 25,
                "mT_clean": offset + (item - 8) / 24,
                "DeltaS": item / 30,
                "DeltaT": item / 40,
                "online_margin_risk": item / 50,
            }
            for item in range(20)
        }
        return values, {item: int(item >= 10) for item in values}

    fit, fit_target = records(0.0)
    evaluate, evaluate_target = records(0.1)
    rows = _model_rows("L2", "L4", fit, evaluate, fit_target, evaluate_target)
    assert [row["model"] for row in rows] == [
        "M0",
        "M0_history",
        "M1_student_plus_teacher_clean",
        "M2_student_plus_teacher_response",
        "M3_student_plus_both_teacher_parts",
        "M4_student_plus_teacher_adv",
    ]
    assert rows[-1]["fields"] == ["mS_adv", "mT_adv"]
    assert "delta_auroc_vs_M0" in rows[-1]


def test_intermediate_payload_tampering_is_rejected(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("frozen\n", encoding="utf-8")
    inputs = {}
    for name in (
        "feature_observations",
        "feature_lineage",
        "outcome_observations",
        "outcome_lineage",
        "online_states",
        "online_lineage",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs[name] = path
    monkeypatch.setattr(state_cli, "_tracked_clean_provenance", lambda: {"git": {"sha": "test", "dirty": False}})
    report = {
        "contract": "ffnr_state_mechanism_v1",
        "label": "L2",
        "input_identity": {"input_sha256": {name: state_cli.sha256_file(path) for name, path in inputs.items()}},
    }
    bound = state_cli._bind_intermediate(report, config_path=config_path)
    config = {"runs": {"L2": inputs}}
    state_cli._validate_intermediate(bound, label="L2", config=config, config_path=config_path)
    bound["input_identity"]["tampered"] = True
    with pytest.raises(FFNRStateMechanismError, match="payload hash drifted"):
        state_cli._validate_intermediate(bound, label="L2", config=config, config_path=config_path)


def test_online_attack_identity_is_frozen() -> None:
    _validate_online_attack({"attack_identity": dict(EXPECTED_ONLINE_ATTACK)})
    mutated = dict(EXPECTED_ONLINE_ATTACK)
    mutated["steps"] = 20
    with pytest.raises(FFNRStateMechanismError, match="KL-PGD10"):
        _validate_online_attack({"attack_identity": mutated})
