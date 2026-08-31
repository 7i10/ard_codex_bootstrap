"""Stable sample identity and deterministic epoch-aware sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator, Sequence, Sized
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast, overload

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class IndexedBatch:
    images: torch.Tensor
    labels: torch.Tensor
    sample_ids: torch.Tensor
    state_update_mask: torch.Tensor | None = None
    multiplicity: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> IndexedBatch:
        return IndexedBatch(
            self.images.to(device),
            self.labels.to(device),
            self.sample_ids.to(device),
            None if self.state_update_mask is None else self.state_update_mask.to(device),
            None if self.multiplicity is None else self.multiplicity.to(device),
        )


@dataclass(frozen=True)
class SampleRef:
    """One epoch position, retaining the source ID and DDP-padding status."""

    index: int
    state_update_mask: bool
    multiplicity: int


IndexedItem: TypeAlias = tuple[torch.Tensor, int, int] | tuple[torch.Tensor, int, int, bool, int]


class SourceIdTransform(Protocol):
    """Transform whose deterministic output is keyed by immutable source ID."""

    source_id_keyed: bool

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor: ...


IndexedTransform: TypeAlias = Callable[[Any], torch.Tensor] | SourceIdTransform


class IndexedDataset(Dataset[IndexedItem]):
    """Attach the immutable source index after the wrapped transform executes."""

    def __init__(self, dataset: Dataset[Any], transform: IndexedTransform | None = None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.content_identity: dict[str, object] | None = None

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]: ...

    @overload
    def __getitem__(self, index: SampleRef) -> tuple[torch.Tensor, int, int, bool, int]: ...

    def __getitem__(self, index: int | SampleRef) -> IndexedItem:
        reference = index if isinstance(index, SampleRef) else None
        if reference is None:
            assert isinstance(index, int)
            source_index = index
        else:
            source_index = reference.index
        item = self.dataset[source_index]
        if not isinstance(item, Sequence) or len(item) < 2:
            raise TypeError("wrapped dataset items must contain image and label")
        image, label = item[0], item[1]
        if self.transform is not None:
            if getattr(self.transform, "source_id_keyed", False):
                image = cast(SourceIdTransform, self.transform)(image, source_id=source_index)
            else:
                image = cast(Callable[[Any], torch.Tensor], self.transform)(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError("dataset transform must produce a torch.Tensor")
        if reference is None:
            return image, int(label), int(source_index)
        return image, int(label), int(source_index), reference.state_update_mask, reference.multiplicity

    def set_epoch(self, epoch: int) -> None:
        if self.transform is not None and hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)

    def set_augmentation_seed(self, seed: int) -> None:
        if self.transform is not None and hasattr(self.transform, "set_seed"):
            self.transform.set_seed(seed)


def collate_indexed(items: list[tuple[Any, ...]]) -> IndexedBatch:
    if not items:
        raise ValueError("cannot collate an empty indexed batch")
    images = tuple(item[0] for item in items)
    labels = tuple(item[1] for item in items)
    sample_ids = tuple(item[2] for item in items)
    masks = tuple(bool(item[3]) if len(item) > 3 else True for item in items)
    multiplicities = tuple(int(item[4]) if len(item) > 4 else 1 for item in items)
    return IndexedBatch(
        images=torch.stack(images),
        labels=torch.tensor(labels, dtype=torch.long),
        sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        state_update_mask=torch.tensor(masks, dtype=torch.bool),
        multiplicity=torch.tensor(multiplicities, dtype=torch.long),
    )


class EpochShuffleSampler(Sampler[SampleRef]):
    """Deterministic single-server sampler with explicit epoch/world-size state."""

    def __init__(self, size: int, *, seed: int, rank: int = 0, world_size: int = 1, shuffle: bool = True) -> None:
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid sampler size/rank/world_size")
        self.size, self.seed, self.rank, self.world_size = size, seed, rank, world_size
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def reseed(self, seed: int) -> None:
        """Change only future sample-order draws after an exact parent restore."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampler seed must be a non-negative integer")
        self.seed = seed

    def __iter__(self) -> Iterator[SampleRef]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
        else:
            indices = list(range(self.size))
        total_size = math.ceil(self.size / self.world_size) * self.world_size
        padded_indices = [indices[position % self.size] for position in range(total_size)]
        multiplicities = Counter(padded_indices)
        positions = range(self.rank, total_size, self.world_size)
        return iter(
            SampleRef(
                index=padded_indices[position],
                state_update_mask=position < self.size,
                multiplicity=multiplicities[padded_indices[position]],
            )
            for position in positions
        )

    def __len__(self) -> int:
        return math.ceil(self.size / self.world_size)

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("seed", "rank", "world_size", "shuffle"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"sampler {key} mismatch")
        self.set_epoch(int(state["epoch"]))


def _ordering_mix(value: int) -> int:
    """Stable 64-bit mixer for the ordering sampler's named RNG streams."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


class HistoryBalancedSampler(Sampler[SampleRef]):
    """History-conditioned order with exact-once 20/60/20 strata.

    The sampler consumes only a caller-provided pre-epoch ``margin_ema``
    lookup.  It never changes exposure counts: after the deterministic
    HIGH/MID/MID/LOW/MID interleave, every source ID occurs exactly once before
    DDP padding.  The state is intentionally compact; the permutation is
    derived again from the frozen epoch and seed on iteration.
    """

    policy_id = "history_balanced_v1"
    _HIGH = 0
    _MID = 1
    _LOW = 2

    def __init__(
        self,
        size: int,
        *,
        sample_ids: Sequence[int],
        margin_ema_provider: Callable[[int], float],
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid sampler size/rank/world_size")
        if len(sample_ids) != size:
            raise ValueError("sample_ids must match sampler size")
        normalized = [int(sample_id) for sample_id in sample_ids]
        if len(set(normalized)) != size or any(sample_id < 0 for sample_id in normalized):
            raise ValueError("sample_ids must be unique non-negative stable IDs")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampler seed must be a non-negative integer")
        self.size = size
        self.sample_ids = tuple(normalized)
        self._position_by_source = {source_id: position for position, source_id in enumerate(normalized)}
        self.margin_ema_provider = margin_ema_provider
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0
        self.last_metadata: dict[str, Any] = {}

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def reseed(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampler seed must be a non-negative integer")
        self.seed = seed

    def _shuffled(self, values: list[int], *, stratum: int) -> list[int]:
        generator = torch.Generator().manual_seed(
            _ordering_mix(self.seed ^ _ordering_mix(self.epoch) ^ _ordering_mix(stratum + 0x51ED))
        )
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[position] for position in order]

    def _global_indices(self) -> list[int]:
        scored = []
        for source_id in self.sample_ids:
            margin = float(self.margin_ema_provider(source_id))
            if not math.isfinite(margin):
                raise ValueError("history sampler margin_ema must be finite")
            # High risk means low margin EMA.  Source ID is the deterministic
            # tie-break and is part of the scientific ordering contract.
            scored.append((-margin, source_id))
        scored.sort(key=lambda item: (item[0], item[1]))
        high_count = self.size // 5
        low_count = self.size // 5
        strata = {
            self._HIGH: [source_id for _, source_id in scored[:high_count]],
            self._MID: [source_id for _, source_id in scored[high_count : self.size - low_count]],
            self._LOW: [source_id for _, source_id in scored[self.size - low_count :]],
        }
        for stratum, values in strata.items():
            strata[stratum] = self._shuffled(values, stratum=stratum)
        cursors = {self._HIGH: 0, self._MID: 0, self._LOW: 0}
        interleaved: list[int] = []
        pattern = (self._HIGH, self._MID, self._MID, self._LOW, self._MID)
        while len(interleaved) < self.size:
            progressed = False
            for stratum in pattern:
                cursor = cursors[stratum]
                values = strata[stratum]
                if cursor < len(values):
                    interleaved.append(values[cursor])
                    cursors[stratum] = cursor + 1
                    progressed = True
            if not progressed:
                break
        if len(interleaved) != self.size or set(interleaved) != set(self.sample_ids):
            raise RuntimeError("history-balanced order is not an exact source-ID permutation")
        self.last_metadata = {
            "policy": self.policy_id,
            "epoch": self.epoch,
            "seed": self.seed,
            "strata_counts": {"high": high_count, "mid": len(strata[self._MID]), "low": low_count},
            "strata_pattern": ["high", "mid", "mid", "low", "mid"],
            "risk_definition": "-margin_ema",
        }
        return [self._position_by_source[source_id] for source_id in interleaved]

    def __iter__(self) -> Iterator[SampleRef]:
        indices = self._global_indices()
        total_size = math.ceil(self.size / self.world_size) * self.world_size
        padded_indices = [indices[position % self.size] for position in range(total_size)]
        multiplicities = Counter(padded_indices)
        positions = range(self.rank, total_size, self.world_size)
        return iter(
            SampleRef(
                index=padded_indices[position],
                state_update_mask=position < self.size,
                multiplicity=multiplicities[padded_indices[position]],
            )
            for position in positions
        )

    def __len__(self) -> int:
        return math.ceil(self.size / self.world_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy_id,
            "epoch": self.epoch,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "sample_ids_sha256": hashlib.sha256(json.dumps(self.sample_ids).encode()).hexdigest(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # An epoch-99 I100 parent was produced by the canonical shuffle
        # sampler.  Accept that exact legacy state only at the fork boundary;
        # all later checkpoints carry and require this sampler's policy tag.
        if "policy" not in state:
            for key in ("seed", "rank", "world_size"):
                if state.get(key) != getattr(self, key):
                    raise ValueError(f"sampler {key} mismatch")
            if state.get("shuffle") is not True:
                raise ValueError("history-balanced fork requires a shuffled parent sampler")
            self.set_epoch(int(state["epoch"]))
            return
        if state.get("policy") != self.policy_id:
            raise ValueError("sampler policy mismatch")
        for key in ("seed", "rank", "world_size"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"sampler {key} mismatch")
        expected_ids = hashlib.sha256(json.dumps(self.sample_ids).encode()).hexdigest()
        if state.get("sample_ids_sha256") != expected_ids:
            raise ValueError("sampler source-ID universe mismatch")
        self.set_epoch(int(state["epoch"]))
