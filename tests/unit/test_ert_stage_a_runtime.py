from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ard.analysis.ert_stage_a_runtime import (
    StageARuntimeError,
    StageATreatment,
    _epoch80_equivalence,
    _epoch80_gate,
    _mask_from_overlay,
)


def test_stage_a_treatment_requires_explicit_clean_wrong_mode() -> None:
    with pytest.raises(StageARuntimeError, match="clean-wrong"):
        StageATreatment(arm="CW1", mask_key="student_clean_wrong", kind="clean_wrong")
    assert StageATreatment(
        arm="CW1",
        mask_key="student_clean_wrong",
        kind="clean_wrong",
        beta_cleance=0.2,
        clean_wrong_mode="clean_ce_only",
    ).clean_wrong_mode == "clean_ce_only"


def test_stage_a_overlay_mask_keeps_stable_ids_and_class_counts(tmp_path: Path) -> None:
    path = tmp_path / "masks.json"
    path.write_text(
        json.dumps(
            {
                "anchor_epoch": 79,
                "masks": {
                    "s3_t1_q10": {"selected_ids": [2, 4], "selected_class_counts": {"0": 1, "1": 1}},
                },
            }
        ),
        encoding="utf-8",
    )
    mask = _mask_from_overlay(path, "s3_t1_q10")
    assert mask.selected_ids == frozenset({2, 4})
    assert mask.class_counts == {0: 1, 1: 1}


def test_confirmatory_treatment_coefficients_are_explicit() -> None:
    t1 = StageATreatment(arm="T1WCONF", mask_key="s3_t1_q10", kind="advce", beta_advce=0.075)
    assert t1.beta_advce == 0.075
    t3 = StageATreatment(
        arm="T3LP05CONF",
        mask_key="s3_t3_q10",
        kind="advkd_advce",
        beta_advce=0.075,
        advkd_multiplier=0.5,
    )
    assert t3.advkd_multiplier == 0.5


def test_horizon_contract_rejects_duplicate_or_pre_parent_epochs() -> None:
    from ard.analysis.ert_stage_a_runtime import StageARuntimeError, _validate_horizons

    with pytest.raises(StageARuntimeError, match="horizon"):
        _validate_horizons((79, 84), 94)
    with pytest.raises(StageARuntimeError, match="unique"):
        _validate_horizons((84, 84), 94)
    _validate_horizons((84, 89, 94), 94)


def test_epoch80_gate_requires_full_state_parity_and_capture_identity(tmp_path: Path) -> None:
    payload = {
        "model": {"x": torch.tensor([1])},
        "optimizer": {"state": {}},
        "scheduler": {},
        "scaler": None,
        "rng": [{"torch_cpu": torch.tensor([2])}],
        "sampler_epoch": [80],
        "sampler_state": [{"epoch": 80}],
        "sample_state": {"records": {}},
        "global_step": 10,
    }
    components = _epoch80_equivalence(payload)
    assert set(components) == set(payload)
    own_dir, peer_dir = tmp_path / "own", tmp_path / "peer"
    own_dir.mkdir()
    peer_dir.mkdir()
    capture = {"selected_ids_sha256": "a" * 64}
    (own_dir / "routing-capture-mask.json").write_text(json.dumps(capture), encoding="utf-8")
    (peer_dir / "routing-capture-mask.json").write_text(json.dumps(capture), encoding="utf-8")
    peer = peer_dir / "epoch80-routing-state.json"
    peer.write_text(json.dumps({"components": components}), encoding="utf-8")
    _epoch80_gate(
        own={"components": components, "capture_path": str(own_dir / "routing-capture-mask.json")},
        peer_path=peer,
        timeout_seconds=0.01,
    )
