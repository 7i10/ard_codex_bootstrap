"""Stable sample identity and deterministic epoch-aware sampling."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterator, Sequence, Sized
from dataclasses import dataclass
from typing import Any, TypeAlias, cast, overload

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


class IndexedDataset(Dataset[IndexedItem]):
    """Attach the immutable source index after the wrapped transform executes."""

    def __init__(self, dataset: Dataset[Any], transform: Callable[[Any], torch.Tensor] | None = None) -> None:
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
                image = self.transform(image, source_id=source_index)
            else:
                image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError("dataset transform must produce a torch.Tensor")
        if reference is None:
            return image, int(label), int(source_index)
        return image, int(label), int(source_index), reference.state_update_mask, reference.multiplicity

    def set_epoch(self, epoch: int) -> None:
        if self.transform is not None and hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)


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
