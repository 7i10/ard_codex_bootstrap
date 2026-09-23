"""Dataset adapters; attacks always receive float tensors in pixel space [0, 1]."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, overload

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional

from ard.config.schema import DatasetConfig
from ard.policies.fixed_mask import selected_ids_sha256

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


class EpochStagewiseAugmentationTransform:
    """Hard switch from CropShift to one fixed late augmentation policy.

    Both component transforms derive their spatial prefix from the same
    source/epoch-keyed stream.  The wrapper therefore changes only the late
    policy at ``switch_epoch``; it never resets augmentation or global RNG.

    ``late_mask`` restricts the switch to the listed sample ids.  Because every
    call builds its generator from ``(augmentation_seed, epoch, source_id)`` and
    nothing is shared between samples, an image outside the mask gets exactly the
    CropShift view -- bit for bit, and identical across any two arms in which
    that image is untreated.  It does **not** match a run with no mask at all:
    there, every image is switched, so an untreated image would get the late
    policy.  See docs/plans/0096-hardness-allocation-direction.md.
    """

    policy_id = "stagewise"

    def __init__(
        self,
        *,
        augmentation_seed: int,
        switch_epoch: int,
        late_policy: str,
        high: int = 11,
        late_mask: frozenset[int] | None = None,
    ) -> None:
        if isinstance(switch_epoch, bool) or not isinstance(switch_epoch, int) or switch_epoch < 1:
            raise ValueError("stagewise switch epoch must be a positive integer")
        if late_policy not in {"crop_re", "idbh_weak"}:
            raise ValueError("stagewise late policy must be crop_re or idbh_weak")
        self.switch_epoch = switch_epoch
        self.late_policy = late_policy
        self.prefix = EpochCropshiftTransform(augmentation_seed=augmentation_seed, high=high)
        self.late = (
            EpochCropReTransform(augmentation_seed=augmentation_seed, high=high)
            if late_policy == "crop_re"
            else EpochIdbhWeakTransform(augmentation_seed=augmentation_seed, high=high)
        )
        if late_mask is not None and not late_mask:
            raise ValueError("stagewise late mask must be non-empty; omit it to switch every sample")
        self.late_mask = late_mask
        self.epoch = 0
        self.source_id_keyed = True

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("augmentation epoch must be non-negative")
        self.epoch = epoch
        self.prefix.set_epoch(epoch)
        self.late.set_epoch(epoch)

    def set_seed(self, seed: int) -> None:
        self.prefix.set_seed(seed)
        self.late.set_seed(seed)

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor:
        switched = self.epoch >= self.switch_epoch and (self.late_mask is None or source_id in self.late_mask)
        transform = self.late if switched else self.prefix
        return transform(image, source_id=source_id)


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


class ImageNetDataset(Dataset[tuple[Image.Image, int]]):
    """Read the standard ImageNet-1k ``train/<wnid>/*.JPEG`` / ``val/<wnid>/*.JPEG`` layout.

    Structurally this follows ``TinyImageNetDataset`` above (root, split,
    class-to-index map, ``samples``, ``targets``, ``content_identity``), but
    its content identity is deliberately **not** a full byte-level SHA-256 of
    every image, unlike ``TinyImageNetDataset``. Plan 0099's "Design
    question" (docs/plans/0099-imagenet-stage0-prep.md) worked out why: at
    Tiny-ImageNet's ~100k images, hashing every file's bytes on every dataset
    construction (i.e. every job launch) is already a real cost; at
    ImageNet-1k's ~1.28M images / 146GB, doing that on every process start is
    not viable. Instead this adapter hashes a **manifest** -- one
    ``(relative_path, label, file_size)`` record per sample, from a cheap
    ``os.stat`` per file, no file bytes read and no image opened -- which
    takes seconds rather than minutes-to-tens-of-minutes over the full set.
    This is a deliberate change to what ``content_sha256`` means for this one
    dataset (a manifest digest, not a full-content digest); the full
    byte-level digest plan 0099 recommends pinning once, offline, alongside
    the manifest digest is a separate, future maintenance command, not
    implemented here -- this adapter never computes it on the hot launch
    path. This is intentional, not an oversight: do not "fix" it by adding a
    full re-hash back onto the construction path.

    Class-to-index mapping: the standard convention (and this dataset's own
    ``class_names.json``, when present at the dataset root) orders ImageNet's
    1000 synsets by ascending WordNet ID (wnid) string -- confirmed by
    inspecting the real on-disk ``class_names.json`` during this plan's
    implementation, whose order is byte-identical to ``sorted()`` of its own
    wnids. So ``class_names.json`` is not load-bearing for index correctness:
    this adapter derives the canonical class order directly from the sorted
    class-directory names actually present under ``root/<split>/`` (mirroring
    ``TinyImageNetDataset``'s plain ``wnids.txt`` listing, and letting a
    synthetic test fixture omit ``class_names.json`` entirely). When
    ``class_names.json`` is present, its human-readable names are attached to
    ``content_identity`` purely as provenance metadata and are not otherwise
    used.
    """

    def __init__(self, root: Path, split: str, *, image_size: int) -> None:
        self.root = root
        self._resolved_root = root.resolve()
        if split == "test":
            raise ValueError("ImageNet has no implicit test-to-validation alias; provide an official test adapter")
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported ImageNet split: {split}")
        if image_size < 1:
            raise ValueError("ImageNet image_size must be a positive integer")
        self.image_size = image_size
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"ImageNet {split} directory missing: {split_dir}")
        classes = sorted(entry.name for entry in split_dir.iterdir() if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"no ImageNet class directories found under {split_dir}")
        self.class_to_index = {wnid: index for index, wnid in enumerate(classes)}
        samples = [
            (path, self.class_to_index[wnid])
            for wnid in classes
            for path in sorted((split_dir / wnid).glob("*"))
            if path.is_file()
        ]
        if not samples:
            raise FileNotFoundError(f"ImageNet {split} images are missing under {split_dir}")
        self.samples = samples
        self.targets = tuple(label for _, label in self.samples)
        self.class_names = self._read_class_names(root / "class_names.json", classes)
        self.content_identity = self._content_identity(split, classes, samples)

    @staticmethod
    def _read_class_names(path: Path, classes: list[str]) -> dict[str, str] | None:
        """Best-effort, provenance-only human-readable names; never required."""
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        by_wnid: dict[str, str] = {}
        for entry in payload.values():
            if isinstance(entry, list) and len(entry) == 2 and all(isinstance(value, str) for value in entry):
                by_wnid[entry[0]] = entry[1]
        return {wnid: by_wnid[wnid] for wnid in classes if wnid in by_wnid} or None

    def _content_identity(
        self, split: str, classes: list[str], samples: list[tuple[Path, int]]
    ) -> dict[str, object]:
        digest = hashlib.sha256()
        header = {"algorithm": "imagenet-manifest-v1", "split": split, "classes": classes}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        for path, label in sorted(samples, key=lambda item: item[0].relative_to(self.root).as_posix()):
            if path.is_symlink():
                raise ValueError(f"ImageNet content manifest rejects symlink: {path}")
            resolved = path.resolve()
            if self._resolved_root not in resolved.parents:
                raise ValueError(f"ImageNet content manifest rejects out-of-root path: {path}")
            record = {
                "path": path.relative_to(self.root).as_posix(),
                "label": label,
                "size": path.stat().st_size,
            }
            digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        return {
            "algorithm": "imagenet-manifest-v1",
            "observed_sha256": digest.hexdigest(),
            "verification": "computed",
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        # Plan 0100: deliberately NOT resized here (plan 0099's original
        # direct-squash resize was found, in scientific review, to both
        # forbid any real crop-based augmentation and to disagree with the
        # pretrained weights' own preprocessing convention). Native,
        # variable resolution is preserved; fixed-size, aspect-ratio-
        # preserving resize/crop is the transform pipeline's job now (see
        # EpochImageNetTransform for training, ImageNetEvalTransform for
        # validation/evaluation), matching every other adapter's own
        # separation of "read the sample" from "shape it for a batch".
        return rgb, label


class ImageNetEvalTransform:
    """Deterministic ImageNet-standard preprocessing: resize the shorter side
    to ``image_size * 256/224`` (preserving aspect ratio), then center-crop
    to ``image_size``. No randomness, no epoch/source dependency -- used for
    validation-during-training and for the official evaluation split, so it
    intentionally does not implement ``set_epoch``/``set_seed``
    (``IndexedDataset`` only calls those when present, per
    ``ard.data.indexed``). Matches the preprocessing convention pretrained
    torchvision ImageNet-1k weights were themselves trained under -- plan
    0100's scientific review found the plain squash-resize this replaces
    left pretrained initialization starting several points of clean
    accuracy below its published number.
    """

    def __init__(self, *, image_size: int) -> None:
        if image_size < 1:
            raise ValueError("ImageNet eval transform image_size must be a positive integer")
        self.image_size = image_size

    def __call__(self, image: Any) -> torch.Tensor:
        resize_size = round(self.image_size * 256 / 224)
        resized = transform_functional.resize(image, resize_size)
        cropped = transform_functional.center_crop(resized, self.image_size)
        return _to_tensor(cropped)


class EpochImageNetTransform:
    """Deterministic ImageNet RandomResizedCrop + horizontal flip, keyed by
    seed, epoch, and source ID -- same discipline as ``EpochSourceTransform``'s
    CIFAR augmentation above (a resumed epoch must reproduce the same
    augmented view for every source ID, independent of worker/sampler
    order). torchvision's own ``RandomResizedCrop.get_params`` draws from the
    global RNG with no injectable generator, so the crop-parameter search
    (up to 10 attempts, log-uniform aspect ratio in [3/4, 4/3], uniform area
    fraction in [0.08, 1.0], falling back to a centered square crop of the
    shorter side if no attempt fits) is reimplemented here against a local
    ``torch.Generator``, matching torchvision's own algorithm
    (``torchvision.transforms.RandomResizedCrop``) parameter-for-parameter.

    **Exception (scientific review, 2026-09-21, P2 finding 5)**: when
    ``heavy_augmentation=True`` (``DatasetConfig.imagenet_heavy_augmentation``),
    the reproduce-per-source-ID guarantee above no longer holds.
    ``torchvision.transforms.RandAugment``/``RandomErasing`` consume the
    *global* PyTorch RNG, not this class's local generator, by deliberate
    human decision (chat, 2026-09-21) rather than reimplementing
    RandAugment's ~14 operations against a local generator. A resumed
    epoch still reproduces the same crop and flip, but not the same
    RandAugment/RandomErasing draw.
    """

    _SCALE = (0.08, 1.0)
    _LOG_RATIO = (math.log(3.0 / 4.0), math.log(4.0 / 3.0))
    _ATTEMPTS = 10

    def __init__(self, *, augmentation_seed: int, image_size: int, heavy_augmentation: bool = False) -> None:
        if image_size < 1:
            raise ValueError("ImageNet augmentation image_size must be a positive integer")
        self.augmentation_seed = augmentation_seed
        self.image_size = image_size
        self.epoch = 0
        self.source_id_keyed = True
        # Plan 0102 Workstream A: Singh/Croce/Hein 2023's own RandAugment(2
        # layers, magnitude 9) + RandomErasing(p=0.25) (arXiv:2303.01870,
        # Appendix A.2). Deliberately global-RNG-based, not keyed by this
        # class's own local generator -- see DatasetConfig.imagenet_heavy_augmentation's
        # docstring for why (human decision, chat, 2026-09-21).
        self.heavy_augmentation = heavy_augmentation
        self._rand_augment = transforms.RandAugment(num_ops=2, magnitude=9) if heavy_augmentation else None
        self._random_erasing = transforms.RandomErasing(p=0.25) if heavy_augmentation else None

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("augmentation epoch must be non-negative")
        self.epoch = epoch

    def set_seed(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("augmentation seed must be a non-negative integer")
        self.augmentation_seed = seed

    def _crop_box(self, image: Any, generator: torch.Generator) -> tuple[int, int, int, int]:
        width, height = transform_functional.get_image_size(image)
        area = width * height
        for _ in range(self._ATTEMPTS):
            target_area = area * (
                self._SCALE[0] + (self._SCALE[1] - self._SCALE[0]) * torch.rand((), generator=generator).item()
            )
            aspect_ratio = math.exp(
                self._LOG_RATIO[0] + (self._LOG_RATIO[1] - self._LOG_RATIO[0]) * torch.rand((), generator=generator).item()
            )
            candidate_width = int(round(math.sqrt(target_area * aspect_ratio)))
            candidate_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < candidate_width <= width and 0 < candidate_height <= height:
                top = int(torch.randint(0, height - candidate_height + 1, (), generator=generator).item())
                left = int(torch.randint(0, width - candidate_width + 1, (), generator=generator).item())
                return top, left, candidate_height, candidate_width
        # Fallback (matches torchvision's own): a centered square crop of the shorter side.
        side = min(width, height)
        top = (height - side) // 2
        left = (width - side) // 2
        return top, left, side, side

    def __call__(self, image: Any, *, source_id: int) -> torch.Tensor:
        # Independent of worker order, sampler order, rank, and process RNG
        # state -- see EpochSourceTransform's identical rationale above.
        generator = torch.Generator().manual_seed(
            self.augmentation_seed + 1_000_003 * self.epoch + 10_007 * source_id
        )
        top, left, height, width = self._crop_box(image, generator)
        cropped = transform_functional.resized_crop(
            image, top, left, height, width, [self.image_size, self.image_size]
        )
        if bool(torch.randint(0, 2, (), generator=generator).item()):
            cropped = transform_functional.hflip(cropped)
        if self._rand_augment is not None:
            # Global RNG from here on -- see __init__'s heavy_augmentation
            # comment; the deterministic local generator above still governs
            # the crop and flip.
            cropped = self._rand_augment(cropped)
        tensor = _to_tensor(cropped)
        if self._random_erasing is not None:
            tensor = self._random_erasing(tensor)
        return tensor


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
    if config.name == "imagenet":
        imagenet_base = ImageNetDataset(config.root, config.split, image_size=config.image_size)
        if len(imagenet_base.class_to_index) != config.num_classes:
            raise ValueError(
                "ImageNet class count mismatch: "
                f"config={config.num_classes}, layout={len(imagenet_base.class_to_index)}"
            )
        observed = imagenet_base.content_identity["observed_sha256"]
        if config.content_sha256 is not None and config.content_sha256 != observed:
            raise ValueError("ImageNet content_sha256 does not match adapter-visible manifest")
        if config.content_sha256 is not None:
            imagenet_base.content_identity = {
                **imagenet_base.content_identity,
                "expected_sha256": config.content_sha256,
                "verification": "computed-and-matched",
            }
        return imagenet_base
    raise ValueError(f"unknown dataset: {config.name}")


def load_stagewise_late_mask(path: Path) -> frozenset[int]:
    """Read an allocation mask and check it against itself.

    Only self-consistency is checked here: the schema, that the IDs are sorted
    and unique, and that the recorded digest matches the IDs.  Whether the mask
    is the one the run declares is checked in ``build_train_validation_views``
    against the config, and whether its IDs belong to the training partition is
    checked there too, because only that function knows the partition.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "namespace",
        "num_classes",
        "selected_ids",
        "selected_ids_sha256",
        "selected_count",
        "selected_class_counts",
        "provenance",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"allocation mask has unexpected or missing fields: {path}")
    if payload["schema_version"] != 1 or payload["namespace"] != "train":
        raise ValueError(f"allocation mask is not a version-1 train-namespace mask: {path}")
    ids = payload["selected_ids"]
    if not isinstance(ids, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in ids):
        raise ValueError(f"allocation mask IDs must be integers: {path}")
    if list(ids) != sorted(set(ids)):
        raise ValueError(f"allocation mask IDs must be sorted and unique: {path}")
    if payload["selected_count"] != len(ids):
        raise ValueError(f"allocation mask count does not match its own IDs: {path}")
    if selected_ids_sha256(tuple(ids)) != payload["selected_ids_sha256"]:
        raise ValueError(f"allocation mask digest does not match its own IDs: {path}")
    return frozenset(ids)


def build_dataset(config: DatasetConfig, *, transform: Callable[[Any], torch.Tensor] | None = None) -> IndexedDataset:
    """Build one indexed view; callers needing train/validation use views below."""
    base = build_raw_dataset(config)
    if transform is not None:
        default_transform: Callable[[Any], torch.Tensor] = transform
    elif config.name == "imagenet":
        # Unlike CIFAR/Tiny-ImageNet, ImageNetDataset no longer hands out a
        # fixed-size image (plan 0100) -- a caller that doesn't supply its
        # own transform needs the deterministic eval-standard resize/crop,
        # not bare _to_tensor, which would receive variable-resolution
        # images and fail to batch.
        default_transform = ImageNetEvalTransform(image_size=config.image_size)
    else:
        default_transform = _to_tensor
    indexed = IndexedDataset(base, default_transform)
    if isinstance(base, (TinyImageNetDataset, ImageNetDataset)):
        indexed.content_identity = base.content_identity
    return indexed


def build_train_validation_views(
    config: DatasetConfig,
    *,
    validation_fraction: float,
    split_seed: int,
    augmentation_seed: int,
    stagewise_late_mask: frozenset[int] | None = None,
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
        elif config.augmentation_policy == "stagewise":
            assert config.stagewise_switch_epoch is not None
            assert config.stagewise_late_policy is not None
            # The config carries the mask's identity and the caller carries the
            # mask, so the two are checked against each other here rather than
            # trusted: a run whose config declares one allocation and whose
            # loader supplies another would be indistinguishable in the record.
            declared_digest = config.stagewise_late_mask_selected_ids_sha256
            if (stagewise_late_mask is None) != (declared_digest is None):
                raise ValueError(
                    "the stagewise late mask and the identity declared in the config "
                    "must both be present or both absent"
                )
            if stagewise_late_mask is not None:
                assert config.stagewise_late_mask_selected_count is not None
                if len(stagewise_late_mask) != config.stagewise_late_mask_selected_count:
                    raise ValueError("stagewise late mask size differs from the count declared in the config")
                if selected_ids_sha256(tuple(sorted(stagewise_late_mask))) != declared_digest:
                    raise ValueError("stagewise late mask IDs differ from the digest declared in the config")
                # Without this an ID outside the training partition is simply never
                # looked up: the run treats fewer images than the record claims and
                # nothing reports it.
                outside = stagewise_late_mask.difference(split_train.indices)
                if outside:
                    raise ValueError(
                        f"stagewise late mask contains {len(outside)} IDs outside the training partition; "
                        f"the first is {min(outside)}"
                    )
            train_transform = EpochStagewiseAugmentationTransform(
                augmentation_seed=augmentation_seed,
                switch_epoch=config.stagewise_switch_epoch,
                late_policy=config.stagewise_late_policy,
                high=config.augmentation_crop_shift_high,
                late_mask=stagewise_late_mask,
            )
        else:  # pragma: no cover - DatasetConfig rejects unknown literals
            raise ValueError(f"unsupported CIFAR augmentation policy: {config.augmentation_policy}")
    elif config.name == "imagenet":
        train_transform = EpochImageNetTransform(
            augmentation_seed=augmentation_seed,
            image_size=config.image_size,
            heavy_augmentation=config.imagenet_heavy_augmentation,
        )
    else:
        train_transform = _to_tensor
    validation_transform: IndexedTransform = (
        ImageNetEvalTransform(image_size=config.image_size) if config.name == "imagenet" else _to_tensor
    )
    train_view = IndexedDataset(raw, train_transform)
    validation_view = IndexedDataset(raw, validation_transform)
    return (
        SourceIndexedSubset(train_view, list(split_train.indices)),
        SourceIndexedSubset(validation_view, list(split_validation.indices)),
    )


def train_probe_ids(indices: Sequence[int], targets: Sequence[int], *, size: int, seed: int) -> list[int]:
    """Fixed, sorted, class-stratified sample of training-partition source IDs
    for the per-epoch train probe (``size // n_classes`` per class, remainder
    filled at random). Depends only on the partition, labels and seed."""
    if not 1 <= size <= len(indices):
        raise ValueError(f"train probe size must be in [1, {len(indices)}], got {size}")
    generator = torch.Generator().manual_seed(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for source_id in indices:
        by_label[int(targets[source_id])].append(int(source_id))
    per_class = size // len(by_label)
    chosen: list[int] = []
    leftover: list[int] = []
    for label in sorted(by_label):
        members = by_label[label]
        order = torch.randperm(len(members), generator=generator).tolist()
        take = min(per_class, len(members))
        chosen.extend(members[position] for position in order[:take])
        leftover.extend(members[position] for position in order[take:])
    remainder = size - len(chosen)
    if remainder:
        fill = torch.randperm(len(leftover), generator=generator).tolist()[:remainder]
        chosen.extend(leftover[position] for position in fill)
    return sorted(chosen)


def build_train_probe_view(
    train_dataset: SourceIndexedSubset, validation_dataset: SourceIndexedSubset, *, size: int, seed: int
) -> SourceIndexedSubset:
    """Training-partition probe IDs viewed through the validation transform,
    so seen-vs-unseen accuracy is measured under identical conditions."""
    targets = getattr(validation_dataset.dataset.dataset, "targets", None)
    if not isinstance(targets, (list, tuple)):
        raise TypeError("train probe requires a label-only targets sequence on the raw dataset")
    probe_ids = train_probe_ids(train_dataset.indices, targets, size=size, seed=seed)
    return SourceIndexedSubset(validation_dataset.dataset, probe_ids)


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
