"""Dataset adapters; attacks always receive float tensors in pixel space [0, 1]."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional

from ard.config.schema import DatasetConfig

from .indexed import IndexedDataset, IndexedItem, IndexedTransform, SampleRef


class EpochSourceTransform:
    """Deterministic CIFAR augmentation keyed by seed, epoch, and source ID."""

    def __init__(self, *, augmentation_seed: int) -> None:
        self.augmentation_seed = augmentation_seed
        self.epoch = 0
        self.source_id_keyed = True

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("augmentation epoch must be non-negative")
        self.epoch = epoch

    def set_seed(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("augmentation seed must be a non-negative integer")
        self.augmentation_seed = seed

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor:
        # This is deliberately independent of worker order, sampler order,
        # rank, and process RNG state.  It makes a resumed epoch reproduce the
        # same train view for every immutable official source ID.
        generator = torch.Generator().manual_seed(
            self.augmentation_seed + 1_000_003 * self.epoch + 10_007 * source_id
        )
        padded = transform_functional.pad(image, padding=4, fill=0)
        top = int(torch.randint(0, 9, (), generator=generator).item())
        left = int(torch.randint(0, 9, (), generator=generator).item())
        cropped = transform_functional.crop(padded, top=top, left=left, height=32, width=32)
        if bool(torch.randint(0, 2, (), generator=generator).item()):
            cropped = transform_functional.hflip(cropped)
        return _to_tensor(cropped)


class EpochCropshiftTransform:
    """Source/epoch-keyed CropShift augmentation from the pinned reference.

    The upstream IDBH CIFAR weak pipeline applies a random horizontal flip,
    then ``CropShift(0, 11)``.  This implementation keeps that ordering while
    routing every draw through a per-source generator, so worker and sampler
    order cannot change a sample's view.  ``high`` is exclusive, matching the
    reference implementation; it is clamped only when an image is smaller
    than the configured maximum shift.
    """

    def __init__(self, *, augmentation_seed: int, high: int = 11) -> None:
        if isinstance(high, bool) or not isinstance(high, int) or high < 1:
            raise ValueError("CropShift high must be a positive integer")
        self.augmentation_seed = augmentation_seed
        self.high = high
        self.epoch = 0
        self.source_id_keyed = True

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("augmentation epoch must be non-negative")
        self.epoch = epoch

    def set_seed(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("augmentation seed must be a non-negative integer")
        self.augmentation_seed = seed

    @staticmethod
    def _sample_top(generator: torch.Generator, x: int, y: int) -> tuple[int, int]:
        return (
            int(torch.randint(0, x + 1, (), generator=generator).item()),
            int(torch.randint(0, y + 1, (), generator=generator).item()),
        )

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.augmentation_seed + 1_000_003 * self.epoch + 10_007 * source_id)
        image = _apply_cropshift(image, generator=generator, high=self.high)
        return _to_tensor(image)


def _apply_cropshift(image: Any, *, generator: torch.Generator, high: int) -> Any:
    width, height = transform_functional.get_image_size(image)
    max_strength = min(high - 1, width - 1, height - 1)
    strength = int(torch.randint(0, max_strength + 1, (), generator=generator).item())

    # Match IDBH ordering: RandomHorizontalFlip before CropShift.
    if bool(torch.randint(0, 2, (), generator=generator).item()):
        image = transform_functional.hflip(image)
    crop_x = int(torch.randint(0, strength + 1, (), generator=generator).item())
    crop_y = strength - crop_x
    crop_width, crop_height = width - crop_x, height - crop_y

    def sample_top(x: int, y: int) -> tuple[int, int]:
        return (
            int(torch.randint(0, x + 1, (), generator=generator).item()),
            int(torch.randint(0, y + 1, (), generator=generator).item()),
        )

    top_x, top_y = sample_top(crop_x, crop_y)
    image = transform_functional.crop(image, top=top_y, left=top_x, height=crop_height, width=crop_width)
    image = transform_functional.pad(image, padding=[crop_x, crop_y], fill=0)
    top_x, top_y = sample_top(crop_x, crop_y)
    return transform_functional.crop(image, top=top_y, left=top_x, height=height, width=width)


def _source_layer_seed(*, augmentation_seed: int, epoch: int, source_id: int, layer: str) -> int:
    """Derive a deterministic named substream without touching global RNG state."""
    payload = f"ard-augmentation-v1|{augmentation_seed}|{epoch}|{source_id}|{layer}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _layer_generator(*, augmentation_seed: int, epoch: int, source_id: int, layer: str) -> torch.Generator:
    return torch.Generator().manual_seed(
        _source_layer_seed(augmentation_seed=augmentation_seed, epoch=epoch, source_id=source_id, layer=layer)
    )


def _random_erase_with_generator(
    image: torch.Tensor,
    *,
    generator: torch.Generator,
    p: float = 0.5,
    scale: tuple[float, float] = (0.02, 0.33),
    ratio: tuple[float, float] = (0.3, 3.3),
    value: float = 0.0,
) -> torch.Tensor:
    """Torchvision RandomErasing defaults with an explicit deterministic RNG."""
    if float(torch.rand((), generator=generator).item()) >= p:
        return image
    channels, height, width = image.shape[-3:]
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio, dtype=torch.float32))
    for _ in range(10):
        erase_area = area * (scale[0] + (scale[1] - scale[0]) * float(torch.rand((), generator=generator).item()))
        aspect_ratio = float(
            torch.exp(log_ratio[0] + (log_ratio[1] - log_ratio[0]) * float(torch.rand((), generator=generator).item()))
        )
        erase_height = int(round((erase_area * aspect_ratio) ** 0.5))
        erase_width = int(round((erase_area / aspect_ratio) ** 0.5))
        if not (erase_height < height and erase_width < width):
            continue
        top = int(torch.randint(0, height - erase_height + 1, (), generator=generator).item())
        left = int(torch.randint(0, width - erase_width + 1, (), generator=generator).item())
        fill = torch.full((channels, erase_height, erase_width), value, dtype=image.dtype, device=image.device)
        return transform_functional.erase(image, top, left, erase_height, erase_width, fill, False)
    return image


def _idbh_color_with_generator(image: Any, *, generator: torch.Generator) -> Any:
    """Upstream IDBH ``ColorShape('color')`` distribution with named RNG draws."""
    operations: tuple[tuple[str, float, float], ...] = (
        ("color", 0.1, 1.9),
        ("brightness", 0.5, 1.9),
        ("contrast", 0.5, 1.9),
        ("sharpness", 0.1, 1.9),
        ("autocontrast", 0.0, 0.0),
        ("equalize", 0.0, 0.0),
        ("shear", 0.05, 0.15),
        ("rotate", 1.0, 11.0),
    )
    index = min(int(float(torch.rand((), generator=generator).item()) * len(operations)), len(operations) - 1)
    operation, lower, upper = operations[index]
    if operation == "color":
        strength = lower + (upper - lower) * float(torch.rand((), generator=generator).item())
        return transform_functional.adjust_saturation(image, strength)
    if operation == "brightness":
        strength = lower + (upper - lower) * float(torch.rand((), generator=generator).item())
        return transform_functional.adjust_brightness(image, strength)
    if operation == "contrast":
        strength = lower + (upper - lower) * float(torch.rand((), generator=generator).item())
        return transform_functional.adjust_contrast(image, strength)
    if operation == "sharpness":
        strength = lower + (upper - lower) * float(torch.rand((), generator=generator).item())
        return transform_functional.adjust_sharpness(image, strength)
    if operation == "autocontrast":
        return transform_functional.autocontrast(image)
    if operation == "equalize":
        return transform_functional.equalize(image)
    if operation == "shear":
        strength = lower + (upper - lower) * float(torch.rand((), generator=generator).item())
        if bool(torch.randint(2, (), generator=generator).item()):
            strength *= -1
        degrees = math.degrees(strength)
        axis_x = bool(torch.randint(2, (), generator=generator).item())
        shear = [degrees, 0.0] if axis_x else [0.0, degrees]
        return transform_functional.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=shear,
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0,
        )
    strength = int(torch.randint(int(lower), int(upper), (), generator=generator).item())
    if bool(torch.randint(2, (), generator=generator).item()):
        strength *= -1
    return transform_functional.rotate(
        image, angle=strength, interpolation=transforms.InterpolationMode.NEAREST, fill=0
    )


class EpochCropReTransform(EpochCropshiftTransform):
    """CROPSHIFT followed by upstream-default Random Erasing."""

    policy_id = "crop_re"

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor:
        spatial_generator = torch.Generator().manual_seed(
            self.augmentation_seed + 1_000_003 * self.epoch + 10_007 * source_id
        )
        image = _apply_cropshift(image, generator=spatial_generator, high=self.high)
        return _random_erase_with_generator(
            _to_tensor(image),
            generator=_layer_generator(
                augmentation_seed=self.augmentation_seed, epoch=self.epoch, source_id=source_id, layer="erase"
            ),
        )


class EpochIdbhWeakTransform(EpochCropshiftTransform):
    """Upstream IDBH CIFAR weak distribution with isolated deterministic layers."""

    policy_id = "idbh_weak"

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor:
        spatial_generator = torch.Generator().manual_seed(
            self.augmentation_seed + 1_000_003 * self.epoch + 10_007 * source_id
        )
        image = _apply_cropshift(image, generator=spatial_generator, high=self.high)
        image = _idbh_color_with_generator(
            image,
            generator=_layer_generator(
                augmentation_seed=self.augmentation_seed, epoch=self.epoch, source_id=source_id, layer="colorshape"
            ),
        )
        tensor = _to_tensor(image)
        return _random_erase_with_generator(
            tensor,
            generator=_layer_generator(
                augmentation_seed=self.augmentation_seed, epoch=self.epoch, source_id=source_id, layer="erase"
            ),
        )


def _to_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        if not image.is_floating_point():
            return image.to(dtype=torch.float32).div(255)
        return image
    return transforms.ToTensor()(image)


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

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def set_augmentation_seed(self, seed: int) -> None:
        self.dataset.set_augmentation_seed(seed)


class SyntheticCIFAR(Dataset[tuple[torch.Tensor, int]]):
    """CIFAR-shaped deterministic fixture; each sample depends only on seed and index."""

    def __init__(self, *, size: int, num_classes: int, image_size: int = 32, seed: int = 0) -> None:
        self.size, self.num_classes, self.image_size, self.seed = size, num_classes, image_size, seed
        self.targets = tuple(index % num_classes for index in range(size))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator, dtype=torch.float32)
        return image, self.targets[index]


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
        self.targets = tuple(label for _, label in self.samples)
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


def build_raw_dataset(config: DatasetConfig) -> Dataset[Any]:
    if config.name == "synthetic_cifar":
        return SyntheticCIFAR(
            size=config.num_samples, num_classes=config.num_classes, image_size=config.image_size, seed=config.seed
        )
    if config.root is None:
        raise ValueError(f"{config.name} requires an explicit dataset root")
    if config.name in {"cifar10", "cifar100"}:
        factory = datasets.CIFAR10 if config.name == "cifar10" else datasets.CIFAR100
        return factory(root=str(config.root), train=config.split == "train", download=config.download)
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
        return base
    raise ValueError(f"unknown dataset: {config.name}")


def build_dataset(config: DatasetConfig, *, transform: Callable[[Any], torch.Tensor] | None = None) -> IndexedDataset:
    """Build one indexed view; callers needing train/validation use views below."""
    base = build_raw_dataset(config)
    indexed = IndexedDataset(base, transform or _to_tensor)
    if isinstance(base, TinyImageNetDataset):
        indexed.content_identity = base.content_identity
    return indexed


def build_train_validation_views(
    config: DatasetConfig,
    *,
    validation_fraction: float,
    split_seed: int,
    augmentation_seed: int,
) -> tuple[SourceIndexedSubset, SourceIndexedSubset]:
    """Create independently transformed train/validation views over one raw set."""
    if config.split != "train":
        raise ValueError("train/validation views require the official train split")
    raw = build_raw_dataset(config)
    split_view = IndexedDataset(raw, _to_tensor)
    split_train, split_validation = stratified_train_validation_split(
        split_view, validation_fraction=validation_fraction, seed=split_seed
    )
    train_transform: IndexedTransform
    if config.name in {"cifar10", "cifar100"}:
        if config.augmentation_policy == "canonical":
            train_transform = EpochSourceTransform(augmentation_seed=augmentation_seed)
        elif config.augmentation_policy == "cropshift":
            train_transform = EpochCropshiftTransform(
                augmentation_seed=augmentation_seed, high=config.augmentation_crop_shift_high
            )
        elif config.augmentation_policy == "crop_re":
            train_transform = EpochCropReTransform(
                augmentation_seed=augmentation_seed, high=config.augmentation_crop_shift_high
            )
        elif config.augmentation_policy == "idbh_weak":
            train_transform = EpochIdbhWeakTransform(
                augmentation_seed=augmentation_seed, high=config.augmentation_crop_shift_high
            )
        else:  # pragma: no cover - DatasetConfig rejects unknown literals
            raise ValueError(f"unsupported CIFAR augmentation policy: {config.augmentation_policy}")
    else:
        train_transform = _to_tensor
    train_view = IndexedDataset(raw, train_transform)
    validation_view = IndexedDataset(raw, _to_tensor)
    return (
        SourceIndexedSubset(train_view, list(split_train.indices)),
        SourceIndexedSubset(validation_view, list(split_validation.indices)),
    )


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
    raw = dataset.dataset
    targets = getattr(raw, "targets", None)
    if not isinstance(targets, (list, tuple)) or len(targets) != len(dataset):
        raise TypeError("stratified validation split requires a label-only targets sequence matching dataset length")
    by_label: dict[int, list[int]] = defaultdict(list)
    for source_id, label in enumerate(targets):
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError("dataset targets must contain integer class labels")
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
