"""Independent RNG streams and a bounded source-isolation canary."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ard.attacks import AttackRequest, LinfPGD
from ard.config.schema import AttackConfig
from ard.data import (
    EpochShuffleSampler,
    EpochSourceTransform,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    data_loader_generator,
    seed_data_loader_worker,
)

MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class RNGSourceSeeds:
    """Seeds whose post-parent effects are intentionally independent."""

    data_seed: int
    attack_seed: int
    other_seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("data_seed", self.data_seed),
            ("attack_seed", self.attack_seed),
            ("other_seed", self.other_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SEED:
                raise ValueError(f"{name} must be an integer in [0, {MAX_SEED}]")

    def as_dict(self) -> dict[str, int]:
        return {
            "data_seed": self.data_seed,
            "attack_seed": self.attack_seed,
            "other_seed": self.other_seed,
        }


def derive_seed(*, experiment_name: str, teacher: str, source_name: str, replicate: str) -> int:
    """Derive a preregistration-safe seed without observing outcomes."""
    key = f"{experiment_name}|{teacher}|{source_name}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % MAX_SEED


def reseed_other_stream(seed: int) -> None:
    """Reseed only global Python/NumPy/Torch sources after parent restore."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("other seed must be a non-negative integer")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def reseed_data_stream(
    *,
    dataset: Any,
    sampler: EpochShuffleSampler,
    loader_generator: torch.Generator,
    seed: int,
) -> None:
    """Rebind future order, augmentation, and worker draws to ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("data seed must be a non-negative integer")
    sampler.reseed(seed)
    loader_generator.manual_seed(seed)
    if hasattr(dataset, "set_augmentation_seed"):
        dataset.set_augmentation_seed(seed)


def _generator_clone(generator: torch.Generator) -> torch.Generator:
    device = getattr(generator, "device", torch.device("cpu"))
    clone = torch.Generator(device=device)
    clone.set_state(generator.get_state())
    return clone


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).flatten().tolist()
    return hashlib.sha256(bytes(raw)).hexdigest()


def _batch_identity(batch: Any) -> str:
    payload = {
        "sample_ids": [int(value) for value in batch.sample_ids.tolist()],
        "labels": [int(value) for value in batch.labels.tolist()],
        "images_sha256": _tensor_sha256(batch.images),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _raw_attack_draw_hash(shape: torch.Size, *, device: torch.device, generator: torch.Generator) -> str:
    clone = _generator_clone(generator)
    draw = torch.empty(shape, device=device, dtype=torch.float32)
    draw.uniform_(-1.0, 1.0, generator=clone)
    return _tensor_sha256(draw)


def _canary_arm(*, data_seed: int, attack_seed: int, batches: int) -> dict[str, list[str]]:
    raw = SyntheticCIFAR(size=32, num_classes=3, image_size=32, seed=1717)
    dataset = IndexedDataset(raw, EpochSourceTransform(augmentation_seed=data_seed))
    sampler = EpochShuffleSampler(len(dataset), seed=data_seed, shuffle=True)
    loader_generator = data_loader_generator(data_seed)
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_indexed,
        generator=loader_generator,
        worker_init_fn=seed_data_loader_worker,
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 3))
    attack = LinfPGD(
        AttackConfig(
            epsilon="1/255",
            step_size="1/255",
            steps=1,
            random_start=True,
            loss="ce",
            student_mode="eval",
            teacher_mode="eval",
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(attack_seed)
    data_hashes: list[str] = []
    attack_hashes: list[str] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        data_hashes.append(_batch_identity(batch))
        attack_hashes.append(_raw_attack_draw_hash(batch.images.shape, device=torch.device("cpu"), generator=generator))
        attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=model,
                teacher=None,
                generator=generator,
            )
        )
    if len(data_hashes) != batches:
        raise RuntimeError("RNG canary did not observe the requested number of batches")
    return {"data_hashes": data_hashes, "attack_draw_hashes": attack_hashes}


def run_seed_isolation_canary(*, batches: int = 2) -> dict[str, Any]:
    """Exercise REF/ATTACK/DATA isolation and fail closed on coupling."""
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
        raise ValueError("canary batches must be a positive integer")
    reference = _canary_arm(data_seed=101, attack_seed=202, batches=batches)
    attack_changed = _canary_arm(data_seed=101, attack_seed=303, batches=batches)
    data_changed = _canary_arm(data_seed=404, attack_seed=202, batches=batches)
    if reference["data_hashes"] != attack_changed["data_hashes"]:
        raise AssertionError("attack seed changed data-side identity")
    if reference["attack_draw_hashes"] == attack_changed["attack_draw_hashes"]:
        raise AssertionError("attack seed did not change the PGD random-start stream")
    if reference["data_hashes"] == data_changed["data_hashes"]:
        raise AssertionError("data seed did not change data-side identity")
    if reference["attack_draw_hashes"] != data_changed["attack_draw_hashes"]:
        raise AssertionError("data seed changed the PGD random-start stream")
    return {
        "schema_version": 1,
        "status": "passed",
        "batches": batches,
        "arms": {"REF": reference, "ATTACK": attack_changed, "DATA": data_changed},
        "assertions": {
            "attack_change_preserves_data": True,
            "attack_change_changes_random_start": True,
            "data_change_changes_data": True,
            "data_change_preserves_random_start": True,
        },
    }
