from __future__ import annotations

import pytest

import ard.analysis.ert_online_routing_proxy as proxy

pytestmark = pytest.mark.unit


def test_frozen_majority_topk_and_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = (0, 1, 2, 3)
    feature = {epoch: {item: {"class_id": item % 2} for item in ids} for epoch in proxy.ANCHORS}
    outcome = {
        epoch: {item: {"student_robust_correct": not (item == 0 or (item == 1 and epoch != 199))} for item in ids}
        for epoch in proxy.TERMINAL_EPOCHS
    }
    online = {
        epoch: {
            item: {
                "class_id": item % 2,
                "current_correct": item != 0,
                "margin_risk": item / 10,
                "frequency_risk": item / 20,
            }
            for item in ids
        }
        for epoch in proxy.ANCHORS
    }

    def margins(_feature: object, *, anchor: int) -> dict[int, dict[str, object]]:
        return {
            item: {
                "class_id": item % 2,
                "mS_adv": -item / 10,
                "mT_adv": -item / 20,
                "mT_clean": 0.5 - item / 20,
                "DeltaT": item / 30,
                "student_robust_correct": item != 0,
                "student_clean_correct": item != 1,
                "teacher_adv_correct": item != 2,
                "teacher_clean_correct": item != 3,
            }
            for item in ids
        }

    monkeypatch.setattr(proxy, "_margin_rows", margins)
    report = proxy.diagnose(feature=feature, outcome=outcome, online=online)
    assert report["target"] == "majority_future_failure_over_189_194_199"
    assert report["anchors"]["79"]["top_k"]["teacher_signed_dominance"][0]["k"] == 1
    gt_row = report["anchors"]["79"]["top_k"]["teacher_signed_dominance"][-1]
    assert gt_row["selection"] == "gt_count"
    assert "jaccard" in gt_row
    assert report["state_definition"]["S2"] == "robust_correct fragile q10"
    strong_cells = report["anchors"]["79"]["state_cells"]["strong_oracle"]
    assert any(row["student_state"] == "S2" for row in strong_cells)
    assert any(row["student_state"] == "S3" for row in strong_cells)
    assert report["one_epoch_delayed"]["available"] is False
    assert report["transitions"]
    assert report["anchors"]["79"]["ce20_vs_online"]["state_confusion"]


def test_proxy_rejects_missing_terminal_epoch() -> None:
    with pytest.raises(proxy.ERTOnlineRoutingProxyError, match="terminal"):
        proxy._future_failure({189: {}, 194: {}})


def test_factorial_lineage_rejects_identity_drift() -> None:
    condition = "ce_pgd10"
    attack = proxy.factorial_attack(condition)
    metadata = {
        "condition": condition,
        "run_id": "run",
        "requested_epochs": list(proxy.TERMINAL_EPOCHS),
        "attack_seed_base": proxy.FACTORIAL_ATTACK_SEED,
        "train_expected_count": 45_000,
        "attack_identity": attack.identity(),
        "attack_identity_sha256": attack.identity_sha256(),
    }
    proxy._validate_factorial_lineage(metadata, condition=condition, expected_count=45_000, expected_run_id="run")
    with pytest.raises(proxy.ERTOnlineRoutingProxyError, match="factorial condition lineage"):
        proxy._validate_factorial_lineage(
            {**metadata, "run_id": "other"},
            condition=condition,
            expected_count=45_000,
            expected_run_id="run",
        )
    with pytest.raises(proxy.ERTOnlineRoutingProxyError, match="factorial condition lineage"):
        proxy._validate_factorial_lineage(
            {**metadata, "attack_seed_base": proxy.FACTORIAL_ATTACK_SEED + 1},
            condition=condition,
            expected_count=45_000,
            expected_run_id="run",
        )
