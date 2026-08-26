from __future__ import annotations

from ard.analysis.ert_rslad_rng_sources import (
    MAX_SEED,
    RNGSourceSeeds,
    ShuffleAugmentationSeeds,
    derive_seed,
    run_seed_isolation_canary,
    run_shuffle_augmentation_canary,
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


def test_shuffle_augmentation_seed_quadruplet_is_strict_and_serializable() -> None:
    seeds = ShuffleAugmentationSeeds(shuffle_seed=1, augmentation_seed=2, attack_seed=3, other_seed=4)
    assert seeds.as_dict() == {
        "shuffle_seed": 1,
        "augmentation_seed": 2,
        "attack_seed": 3,
        "other_seed": 4,
    }


def test_rng_source_canary_separates_data_and_attack_streams() -> None:
    result = run_seed_isolation_canary(batches=1)
    assert result["status"] == "passed"
    assert all(result["assertions"].values())


def test_shuffle_augmentation_canary_separates_order_and_views() -> None:
    result = run_shuffle_augmentation_canary()
    assert result["status"] == "passed"
    assert all(result["assertions"].values())
