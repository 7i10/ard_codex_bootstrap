from __future__ import annotations

from pathlib import Path

import pytest

from ard.analysis.ffnr_causal_ce20 import (
    ARMS,
    CONTRACT_ID,
    HORIZON_TO_CHECKPOINT_EPOCH,
    CausalCE20Error,
    ce_pgd20_attack_identity,
    parse_causal_config,
    summarize_endpoint,
)

pytestmark = pytest.mark.t2


def _config() -> dict[str, object]:
    entry = {
        "manifest": "manifest.json",
        "resolved_config": "resolved.yaml",
        "validation_metrics": "metrics.parquet",
    }
    return {
        "schema_version": 1,
        "contract": CONTRACT_ID,
        "expected_count": 45_000,
        "stable_id_class_universe_sha256": "9a1a7929e47196ca4cb49a7c2bea5029170ecdb1c18f9f38c05ea14d9913bf60",
        "checkpoint_inventory": "inventory.json",
        "horizons": [84, 89, 94],
        "replay_batch_size": 128,
        "attack_seed": 20260810,
        "replay_device_type": "cuda",
        "bootstrap": {"replicates": 10, "seed": 7, "strata": "class_id"},
        "s2_overlay": None,
        "runs": {label: {arm: dict(entry) for arm in ARMS} for label in ("L2", "L4")},
    }


def _row(sample_id: int, *, robust: bool, clean: bool, margin: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "class_id": sample_id % 2,
        "student_robust_correct": robust,
        "student_clean_correct": clean,
        "student_clean_probability_margin": margin,
        "student_adversarial_probability_margin": margin - 0.1,
    }


def test_ce_pgd20_identity_is_complete_pixel_eval_contract() -> None:
    attack = ce_pgd20_attack_identity()
    assert attack["input_domain"] == "pixel_0_1"
    assert attack["epsilon"] == "8/255"
    assert attack["step_size"] == "2/255"
    assert attack["steps"] == 20
    assert attack["loss"] == "ce"
    assert attack["random_start"] is True
    assert attack["student_mode"] == attack["teacher_mode"] == "eval"
    assert HORIZON_TO_CHECKPOINT_EPOCH == {84: 84, 89: 89, 94: 93}


def test_config_rejects_unregistered_checkpoint_paths_and_s2_guessing() -> None:
    assert parse_causal_config(_config())["horizons"] == [84, 89, 94]
    with pytest.raises(CausalCE20Error, match="keys"):
        parse_causal_config({**_config(), "extra": True})
    invalid = _config()
    invalid["s2_overlay"] = "infer-from-final-state"
    with pytest.raises(CausalCE20Error, match="S2"):
        parse_causal_config(invalid)


def test_paired_report_keeps_rescue_harm_spillover_and_seeded_bootstrap(tmp_path: Path) -> None:
    control = [
        _row(0, robust=False, clean=True, margin=0.1),
        _row(1, robust=True, clean=True, margin=0.2),
        _row(2, robust=False, clean=False, margin=-0.1),
        _row(3, robust=True, clean=True, margin=0.3),
    ]
    rows = {
        "C79": control,
        "RA": [
            _row(0, robust=True, clean=True, margin=0.3),
            _row(1, robust=True, clean=True, margin=0.2),
            _row(2, robust=False, clean=False, margin=-0.1),
            _row(3, robust=True, clean=True, margin=0.3),
        ],
        "RAR": [
            _row(0, robust=False, clean=True, margin=0.1),
            _row(1, robust=False, clean=True, margin=0.0),
            _row(2, robust=False, clean=False, margin=-0.1),
            _row(3, robust=True, clean=True, margin=0.3),
        ],
        "RB": [
            _row(0, robust=False, clean=True, margin=0.1),
            _row(1, robust=True, clean=True, margin=0.2),
            _row(2, robust=True, clean=True, margin=0.1),
            _row(3, robust=True, clean=True, margin=0.3),
        ],
        "RBR": [
            _row(0, robust=False, clean=True, margin=0.1),
            _row(1, robust=True, clean=True, margin=0.2),
            _row(2, robust=False, clean=False, margin=-0.1),
            _row(3, robust=False, clean=True, margin=0.1),
        ],
    }
    report = summarize_endpoint(
        label="L2",
        horizon=94,
        rows_by_arm=rows,
        masks={"RA": {0, 1}, "RAR": {0, 1}, "RB": {2, 3}, "RBR": {2, 3}},
        bootstrap={"replicates": 10, "seed": 7, "strata": "class_id"},
        validation_paths={arm: tmp_path / f"{arm}.missing" for arm in ARMS},
    )
    route_a = report["routes"]["A"]
    assert route_a["selected"]["rescue_count"] == 1
    assert route_a["random"]["harm_count"] == 1
    assert route_a["selected_minus_random"]["net_rescue_rate"] == 1.0
    assert route_a["selected_nonselected_spillover"]["n"] == 2
    assert route_a["bootstrap_selected_minus_random"]["seed"] == 7
    assert report["s2_overlay"]["available"] is False
