from __future__ import annotations

import pytest

from ard.analysis.ffnr_attack_factorial import (
    CONDITIONS,
    FactorialReplayError,
    factorial_attack,
    parse_factorial_config,
)

pytestmark = pytest.mark.t2


def test_factorial_attack_contract_separates_loss_and_steps() -> None:
    assert CONDITIONS == ("ce_pgd10", "ce_pgd20", "kl_pgd10", "kl_pgd20")
    assert factorial_attack("ce_pgd10").identity()["loss"] == "ce"
    assert factorial_attack("ce_pgd10").steps == 10
    assert factorial_attack("ce_pgd20").steps == 20
    assert factorial_attack("kl_pgd10").kl_target == "teacher_clean"
    assert factorial_attack("kl_pgd20").identity()["loss"] == "kl"
    assert factorial_attack("kl_pgd20").steps == 20
    assert len({factorial_attack(condition).identity_sha256() for condition in CONDITIONS}) == 4


def test_factorial_attack_rejects_unknown_condition() -> None:
    with pytest.raises(FactorialReplayError, match="unknown factorial condition"):
        factorial_attack("ce_pgd30")


def test_factorial_config_is_strict_and_ordered() -> None:
    valid = {
        "schema_version": 1,
        "contract": "ffnr_attack_factorial_v1",
        "run_id": "run",
        "manifest": "manifest.json",
        "checkpoint_inventory": "inventory.json",
        "epochs": [189, 194, 199],
        "train_expected_count": 45000,
        "replay_batch_size": 128,
        "attack_seed": 20260808,
        "replay_device_type": "cuda",
        "output_root": "out",
    }
    assert parse_factorial_config(valid)["epochs"] == [189, 194, 199]
    with pytest.raises(FactorialReplayError, match="keys"):
        parse_factorial_config({**valid, "extra": True})
    with pytest.raises(FactorialReplayError, match="sorted"):
        parse_factorial_config({**valid, "epochs": [194, 189]})
