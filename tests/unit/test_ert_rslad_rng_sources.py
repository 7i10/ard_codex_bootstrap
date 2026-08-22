from __future__ import annotations

from ard.analysis.ert_rslad_rng_sources import (
    MAX_SEED,
    RNGSourceSeeds,
    derive_seed,
    run_seed_isolation_canary,
)


def test_rng_source_seed_triplet_is_strict_and_serializable() -> None:
    seeds = RNGSourceSeeds(data_seed=1, attack_seed=2, other_seed=3)
    assert seeds.as_dict() == {"data_seed": 1, "attack_seed": 2, "other_seed": 3}
    assert (
        0
        <= derive_seed(
            experiment_name="ert_rslad_rng_source_decomposition_v1",
            teacher="L2",
            source_name="D0",
            replicate="1",
        )
        < MAX_SEED
    )


def test_rng_source_canary_separates_data_and_attack_streams() -> None:
    result = run_seed_isolation_canary(batches=1)
    assert result["status"] == "passed"
    assert all(result["assertions"].values())
