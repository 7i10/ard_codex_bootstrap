from __future__ import annotations

import json

from scripts.build_attack_seed_probe_registry import DOMAIN, attack_seed, build_registry


def test_attack_seed_registry_is_deterministic_and_unique() -> None:
    values = [attack_seed(index) for index in range(8)]
    assert len(set(values)) == 8
    assert values == [attack_seed(index) for index in range(8)]
    assert DOMAIN == "ert_rslad_attack_seed_probe_v1"


def test_registry_freezes_only_attack_seed_values() -> None:
    registry = build_registry(
        source_sha="a" * 40,
        parent_hashes={"seed1": "b" * 64, "seed2": "c" * 64},
    )
    assert registry["status"] == "frozen_before_training"
    assert registry["invariant_streams"] == [
        "model_init",
        "data_order",
        "augmentation",
        "evaluation_attack",
        "python_numpy_torch_other_rng",
    ]
    rows = registry["seeds"]
    assert [row["index"] for row in rows] == list(range(8))
    assert [row["value"] for row in rows] == [attack_seed(index) for index in range(8)]
    json.dumps(registry, sort_keys=True)
