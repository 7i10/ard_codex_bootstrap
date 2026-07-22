"""Dataset adapters; attacks always receive float tensors in pixel space [0, 1]."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from ard.config.schema import DatasetConfig

from .indexed import IndexedDataset, IndexedItem, SampleRef


class SourceIndexedSubset(Dataset[IndexedItem]):
    """Subset whose returned ID remains the original train-set ID."""

    def __init__(self, dataset: IndexedDataset, indices: list[int]) -> None:
        self.dataset, self.indices = dataset, indices

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]: ...

    @overload
    def __getitem__(self, index: SampleRef) -> tuple[torch.Tensor, int, int, bool, int]: ...

    def __getitem__(self, index: int | SampleRef) -> IndexedItem:
        reference = index if isinstance(index, SampleRef) else None
        if reference is None:
            assert isinstance(index, int)
            subset_index = index
        else:
            subset_index = reference.index
        image, label, source_id = self.dataset[self.indices[subset_index]]
        if reference is None:
            return image, label, source_id
        return image, label, source_id, reference.state_update_mask, reference.multiplicity


class SyntheticCIFAR(Dataset[tuple[torch.Tensor, int]]):
    """CIFAR-shaped deterministic fixture; each sample depends only on seed and index."""

    def __init__(self, *, size: int, num_classes: int, image_size: int = 32, seed: int = 0) -> None:
        self.size, self.num_classes, self.image_size, self.seed = size, num_classes, image_size, seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator, dtype=torch.float32)
        return image, index % self.num_classes


class TinyImageNetDataset(Dataset[tuple[Image.Image, int]]):
    """Read the standard Tiny-ImageNet train or annotated validation layout."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self._resolved_root = root.resolve()
        classes_file = root / "wnids.txt"
        if not classes_file.is_file():
            raise FileNotFoundError(f"Tiny-ImageNet class list missing: {classes_file}")
        classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        if split == "train":
            samples = [
                (path, self.class_to_index[class_name], class_name)
                for class_name in classes
                for path in sorted((root / "train" / class_name / "images").glob("*"))
                if path.is_file()
            ]
        elif split == "val":
            annotations = root / "val" / "val_annotations.txt"
            if not annotations.is_file():
                raise FileNotFoundError(f"Tiny-ImageNet validation annotations missing: {annotations}")
            labels = {
                parts[0]: parts[1]
                for line in annotations.read_text(encoding="utf-8").splitlines()
                if len(parts := line.split()) >= 2
            }
            samples = [
                (root / "val" / "images" / name, self.class_to_index[label], label)
                for name, label in sorted(labels.items())
            ]
        elif split == "test":
            raise ValueError("Tiny-ImageNet has no implicit test-to-validation alias; provide an official test adapter")
        else:
            raise ValueError(f"unsupported Tiny-ImageNet split: {split}")
        if not samples or any(not path.is_file() for path, _, _ in samples):
            raise FileNotFoundError(f"Tiny-ImageNet {split} images are missing under {root}")
        self.samples = [(path, label) for path, label, _ in samples]
        self.content_identity = self._content_identity(split, classes, samples)

    def _content_identity(
        self, split: str, classes: list[str], samples: list[tuple[Path, int, str]]
    ) -> dict[str, object]:
        digest = hashlib.sha256()
        header = {"algorithm": "tiny-imagenet-visible-v1", "split": split, "classes": classes}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        for path, label, class_name in sorted(samples, key=lambda item: item[0].relative_to(self.root).as_posix()):
            if path.is_symlink():
                raise ValueError(f"Tiny-ImageNet content digest rejects symlink: {path}")
            resolved = path.resolve()
            if self._resolved_root not in resolved.parents:
                raise ValueError(f"Tiny-ImageNet content digest rejects out-of-root path: {path}")
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            with Image.open(path) as image:
                dimensions = image.size
            record = {
                "path": path.relative_to(self.root).as_posix(),
                "label": label,
                "class": class_name,
                "size": path.stat().st_size,
                "sha256": file_digest.hexdigest(),
                "dimensions": dimensions,
            }
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        return {
            "algorithm": "tiny-imagenet-visible-v1",
            "observed_sha256": digest.hexdigest(),
            "verification": "computed",
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), label


def build_dataset(config: DatasetConfig, *, transform: Callable[[Any], torch.Tensor] | None = None) -> IndexedDataset:
    tensor_transform = transform or transforms.ToTensor()
    if config.name == "synthetic_cifar":
        return IndexedDataset(
            SyntheticCIFAR(
                size=config.num_samples, num_classes=config.num_classes, image_size=config.image_size, seed=config.seed
            ),
            transform,
        )
    if config.root is None:
        raise ValueError(f"{config.name} requires an explicit dataset root")
    if config.name in {"cifar10", "cifar100"}:
        factory = datasets.CIFAR10 if config.name == "cifar10" else datasets.CIFAR100
        base = factory(root=str(config.root), train=config.split == "train", download=config.download)
        return IndexedDataset(base, tensor_transform)
    if config.name == "tiny_imagenet":
        base = TinyImageNetDataset(config.root, config.split)
        if len(base.class_to_index) != config.num_classes:
            raise ValueError(
                f"Tiny-ImageNet class count mismatch: config={config.num_classes}, layout={len(base.class_to_index)}"
            )
        observed = base.content_identity["observed_sha256"]
        if config.content_sha256 is not None and config.content_sha256 != observed:
            raise ValueError("Tiny-ImageNet content_sha256 does not match adapter-visible content")
        if config.content_sha256 is not None:
            base.content_identity = {
                **base.content_identity,
                "expected_sha256": config.content_sha256,
                "verification": "computed-and-matched",
            }
        indexed = IndexedDataset(base, tensor_transform)
        indexed.content_identity = base.content_identity
        return indexed
    raise ValueError(f"unknown dataset: {config.name}")


def stratified_train_validation_split(
    dataset: IndexedDataset, *, validation_fraction: float, seed: int
) -> tuple[SourceIndexedSubset, SourceIndexedSubset]:
    """Split one official train dataset deterministically, retaining source IDs.

    The validation subset is sampled independently inside each class with a
    seed-fixed generator.  A class with at least two examples always retains a
    training example; singleton classes remain training-only.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    by_label: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        _, label, source_id = dataset[index]
        by_label[label].append(source_id)
    generator = torch.Generator().manual_seed(seed)
    validation_ids: list[int] = []
    for label in sorted(by_label):
        members = by_label[label]
        if len(members) < 2:
            continue
        count = min(len(members) - 1, max(1, round(len(members) * validation_fraction)))
        ordering = torch.randperm(len(members), generator=generator).tolist()
        validation_ids.extend(members[position] for position in ordering[:count])
    validation_set = set(validation_ids)
    training_ids = [index for index in range(len(dataset)) if index not in validation_set]
    if not training_ids or not validation_ids:
        raise ValueError("stratified validation split requires at least two samples in one class")
    return SourceIndexedSubset(dataset, training_ids), SourceIndexedSubset(dataset, sorted(validation_ids))
