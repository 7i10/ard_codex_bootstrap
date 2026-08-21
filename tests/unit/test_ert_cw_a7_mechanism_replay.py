from pathlib import Path

import torch

from ard.analysis.ert_cw_a7_mechanism_replay import _regime, _targets, _treatment


def test_regime_boundaries_are_disjoint() -> None:
    floor, cap = 0.03221710026264191, 0.13952550292015076
    assert [_regime(x, floor=floor, cap=cap) for x in (-0.1, 0.0, 1e-6, floor, cap, cap + 1e-6)] == [
        "R0",
        "R0",
        "R1",
        "R2",
        "R2",
        "R3",
    ]


def test_frozen_target_policies() -> None:
    margin = torch.tensor([-0.2, 0.01, 0.05, 0.2])
    common = {"margin_cap": 0.14, "margin_floor": 0.03, "margin_gamma": 0.07}
    target, active = _targets(margin, {**common, "margin_target_mode": "teacher_zero"})
    assert torch.allclose(target, torch.tensor([0.0, 0.01, 0.05, 0.14]))
    assert bool(active.all())
    target, active = _targets(margin, {**common, "margin_target_mode": "teacher_floor"})
    assert torch.allclose(target, torch.tensor([0.03, 0.03, 0.05, 0.14]))
    assert bool(active.all())
    target, active = _targets(margin, {**common, "margin_target_mode": "teacher_abstain"})
    assert torch.allclose(target, torch.tensor([0.0, 0.01, 0.05, 0.14]))
    assert active.tolist() == [False, True, True, True]
    target, active = _targets(margin, {**common, "margin_target_mode": "fixed"})
    assert torch.allclose(target, torch.full_like(margin, 0.07))
    assert bool(active.all())


def test_a7_configs_have_frozen_policy() -> None:
    root = Path(".cache/analysis/ert-cw-margin-screen-v1-r3/L2")
    assert _treatment(root / "A5/resolved_config.yaml")["margin_target_mode"] == "fixed"
    assert _treatment(root / "A6/resolved_config.yaml")["margin_target_mode"] == "teacher_zero"
    assert _treatment(root / "A7/resolved_config.yaml")["margin_target_mode"] == "teacher_floor"
    assert _treatment(root / "A8/resolved_config.yaml")["margin_target_mode"] == "teacher_abstain"
