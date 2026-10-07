"""Training batches that carry the crop key of every image.

A crop key is ``(epoch, top, left, height, width, flip)``: the epoch the
training transform was set to and the RandomResizedCrop box (original-image
pixels) plus the horizontal flip it actually drew for that image.  The key is
read from :class:`ard.data.datasets.EpochImageNetTransform` immediately after
it produced the image, in the same worker, so it describes the pixels in the
batch rather than a recomputation of what they should have been.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from ard.data.datasets import EpochImageNetTransform, SourceIndexedSubset
from ard.data.indexed import IndexedBatch, SampleRef

CROP_KEY_FIELDS = ("epoch", "top", "left", "height", "width", "flip")


@dataclass(frozen=True)
class CropKeyedBatch(IndexedBatch):
    """An :class:`IndexedBatch` plus one ``int64 [batch, 6]`` crop key per image."""

    crop_keys: torch.Tensor | None = None

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> CropKeyedBatch:
        base = super().to(device, non_blocking=non_blocking)
        return CropKeyedBatch(
            base.images,
            base.labels,
            base.sample_ids,
            base.state_update_mask,
            base.multiplicity,
            None if self.crop_keys is None else self.crop_keys.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> CropKeyedBatch:
        base = super().pin_memory()
        return CropKeyedBatch(
            base.images,
            base.labels,
            base.sample_ids,
            base.state_update_mask,
            base.multiplicity,
            None if self.crop_keys is None else self.crop_keys.pin_memory(),
        )


def training_transform(subset: SourceIndexedSubset) -> EpochImageNetTransform:
    transform = getattr(subset.dataset, "transform", None)
    if not isinstance(transform, EpochImageNetTransform):
        raise TypeError("crop keys are defined only for the ImageNet EpochImageNetTransform training view")
    if transform.heavy_augmentation:
        raise ValueError(
            "crop keys do not determine the training view under imagenet_heavy_augmentation "
            "(RandAugment/RandomErasing draw from the global RNG)"
        )
    return transform


class CropKeyedSubset(Dataset[tuple[Any, ...]]):
    """Wrap the ImageNet training view so each item also returns its crop key."""

    def __init__(self, subset: SourceIndexedSubset) -> None:
        self.subset = subset
        self.transform = training_transform(subset)
        self.indices = subset.indices
        self.dataset = subset.dataset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int | SampleRef) -> tuple[Any, ...]:
        self.transform.last_crop_key = None
        item = self.subset[index]
        key = self.transform.last_crop_key
        if key is None:
            raise RuntimeError("training transform produced an image without recording its crop key")
        return (*item, key)

    def set_epoch(self, epoch: int) -> None:
        self.subset.set_epoch(epoch)

    def set_augmentation_seed(self, seed: int) -> None:
        self.subset.set_augmentation_seed(seed)


def collate_crop_keyed(items: list[tuple[Any, ...]]) -> CropKeyedBatch:
    from ard.data.indexed import collate_indexed

    if not items:
        raise ValueError("cannot collate an empty crop-keyed batch")
    keys = [item[-1] for item in items]
    if any(not isinstance(key, tuple) or len(key) != len(CROP_KEY_FIELDS) for key in keys):
        raise TypeError("crop-keyed items must end with a 6-field crop key")
    base = collate_indexed([item[:-1] for item in items])
    return CropKeyedBatch(
        base.images,
        base.labels,
        base.sample_ids,
        base.state_update_mask,
        base.multiplicity,
        torch.tensor(keys, dtype=torch.int64),
    )
