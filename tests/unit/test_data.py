from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ard.config.schema import DatasetConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SyntheticCIFAR,
    build_dataset,
    build_train_validation_views,
    collate_indexed,
    stratified_train_validation_split,
)

pytestmark = pytest.mark.t1


def test_stable_ids_survive_augmentation_and_epoch_shuffle() -> None:
    base = SyntheticCIFAR(size=9, num_classes=3, image_size=4, seed=11)
    indexed = IndexedDataset(base, transform=lambda image: image.flip(-1))
    sampler = EpochShuffleSampler(len(indexed), seed=5)
    sampler.set_epoch(2)
    loader = DataLoader(indexed, batch_size=3, sampler=sampler, collate_fn=collate_indexed)
    seen = []
    for batch in loader:
        seen.extend(batch.sample_ids.tolist())
        for image, sample_id in zip(batch.images, batch.sample_ids.tolist(), strict=True):
            expected, _ = base[sample_id]
            assert torch.equal(image, expected.flip(-1))
    assert sorted(seen) == list(range(len(indexed)))


def test_synthetic_samples_depend_only_on_seed_and_index() -> None:
    first = SyntheticCIFAR(size=3, num_classes=2, seed=19)
    second = SyntheticCIFAR(size=3, num_classes=2, seed=19)
    torch.manual_seed(999)
    assert torch.equal(first[2][0], second[2][0])
    assert first[2][1] == second[2][1] == 0


def test_seed_fixed_stratified_validation_keeps_original_ids_and_train_only_samples() -> None:
    indexed = IndexedDataset(SyntheticCIFAR(size=12, num_classes=3, image_size=4, seed=19))
    train_first, validation_first = stratified_train_validation_split(indexed, validation_fraction=0.25, seed=7)
    train_second, validation_second = stratified_train_validation_split(indexed, validation_fraction=0.25, seed=7)
    train_ids = {train_first[index][2] for index in range(len(train_first))}
    validation_ids = {validation_first[index][2] for index in range(len(validation_first))}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == set(range(len(indexed)))
    assert [validation_first[index][2] for index in range(len(validation_first))] == [
        validation_second[index][2] for index in range(len(validation_second))
    ]
    assert [train_first[index][2] for index in range(len(train_first))] == [
        train_second[index][2] for index in range(len(train_second))
    ]


def test_ddp_padding_marks_repeated_source_ids_as_non_updating_and_records_multiplicity() -> None:
    samplers = [EpochShuffleSampler(2, seed=3, rank=rank, world_size=4, shuffle=False) for rank in range(4)]
    references = [reference for sampler in samplers for reference in sampler]
    assert [reference.index for reference in references] == [0, 1, 0, 1]
    assert [reference.state_update_mask for reference in references] == [True, True, False, False]
    assert [reference.multiplicity for reference in references] == [2, 2, 2, 2]
    indexed = IndexedDataset(SyntheticCIFAR(size=2, num_classes=2, image_size=4, seed=1))
    batch = collate_indexed([indexed[reference] for reference in references])
    assert batch.sample_ids.tolist() == [0, 1, 0, 1]
    assert batch.state_update_mask is not None and batch.state_update_mask.tolist() == [True, True, False, False]


def test_ddp_padding_repeats_single_sample_enough_for_world_size() -> None:
    references = [
        reference
        for rank in range(4)
        for reference in EpochShuffleSampler(1, seed=3, rank=rank, world_size=4, shuffle=False)
    ]
    assert [reference.index for reference in references] == [0, 0, 0, 0]
    assert [reference.state_update_mask for reference in references] == [True, False, False, False]
    assert [reference.multiplicity for reference in references] == [4, 4, 4, 4]


def test_large_sampler_padding_has_linear_construction_contract() -> None:
    size, world_size, rank = 100_003, 8, 7
    references = list(EpochShuffleSampler(size, seed=1, rank=rank, world_size=world_size, shuffle=False))
    assert len(references) == 12_501
    assert sum(reference.state_update_mask for reference in references) == 12_500
    assert references[-1].index == 4
    assert references[-1].multiplicity == 2


def test_tiny_imagenet_validation_layout_adapter(tmp_path: Path) -> None:
    (tmp_path / "wnids.txt").write_text("class_a\nclass_b\n", encoding="utf-8")
    images = tmp_path / "val" / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(images / "sample.JPEG")
    (tmp_path / "val" / "val_annotations.txt").write_text("sample.JPEG\tclass_a\t0\t0\t4\t4\n", encoding="utf-8")
    config = DatasetConfig(name="tiny_imagenet", root=tmp_path, split="val", num_classes=2, image_size=4)
    dataset = build_dataset(config)
    image, label, sample_id = dataset[0]
    assert image.shape == (3, 4, 4)
    assert label == 0 and sample_id == 0
    assert image.min() >= 0 and image.max() <= 1


def _tiny_validation_layout(root: Path, *, label: str = "class_a", color: tuple[int, int, int] = (10, 20, 30)) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "wnids.txt").write_text("class_a\nclass_b\n", encoding="utf-8")
    images = root / "val" / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(images / "sample.JPEG")
    (root / "val" / "val_annotations.txt").write_text(f"sample.JPEG\t{label}\t0\t0\t4\t4\n", encoding="utf-8")


def test_tiny_content_digest_is_root_independent_and_detects_visible_mutations(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    _tiny_validation_layout(first)
    _tiny_validation_layout(second)
    config = DatasetConfig(name="tiny_imagenet", root=first, split="val", num_classes=2, image_size=4)
    first_dataset = build_dataset(config)
    second_dataset = build_dataset(config.model_copy(update={"root": second}))
    assert first_dataset.content_identity == second_dataset.content_identity
    observed = first_dataset.content_identity
    assert observed is not None
    assert observed["verification"] == "computed"

    _tiny_validation_layout(second, color=(99, 20, 30))
    mutated = build_dataset(config.model_copy(update={"root": second}))
    assert mutated.content_identity != observed
    _tiny_validation_layout(second, label="class_b")
    relabeled = build_dataset(config.model_copy(update={"root": second}))
    assert relabeled.content_identity != observed

    expected = observed["observed_sha256"]
    assert isinstance(expected, str)
    matched = build_dataset(config.model_copy(update={"content_sha256": expected}))
    assert matched.content_identity is not None
    assert matched.content_identity["verification"] == "computed-and-matched"
    with pytest.raises(ValueError, match="does not match"):
        build_dataset(config.model_copy(update={"content_sha256": expected, "root": second}))


@pytest.mark.parametrize(("name", "classes", "attribute"), (("cifar10", 10, "CIFAR10"), ("cifar100", 100, "CIFAR100")))
def test_cifar_adapters_respect_explicit_root_split_and_no_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, classes: int, attribute: str
) -> None:
    calls = []

    def factory(*, root: str, train: bool, download: bool) -> SyntheticCIFAR:
        calls.append((root, train, download))
        return SyntheticCIFAR(size=2, num_classes=classes, image_size=4)

    monkeypatch.setattr(f"ard.data.datasets.datasets.{attribute}", factory)
    config = DatasetConfig(name=name, root=tmp_path, split="train", download=False, num_classes=classes)
    dataset = build_dataset(config, transform=lambda image: image)
    assert calls == [(str(tmp_path), True, False)]
    assert dataset[1][2] == 1


def test_cifar_train_and_validation_views_are_independent_epoch_keyed_and_keep_official_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool, bool]] = []

    class FakeCIFAR(SyntheticCIFAR):
        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32) / (3 * 32 * 32)
            return image + index / 100_000, index % 2

    def factory(*, root: str, train: bool, download: bool) -> FakeCIFAR:
        calls.append((root, train, download))
        return FakeCIFAR(size=12, num_classes=10, image_size=32)

    monkeypatch.setattr("ard.data.datasets.datasets.CIFAR10", factory)
    config = DatasetConfig(name="cifar10", root=tmp_path, split="train", download=False, num_classes=10)
    train, validation = build_train_validation_views(
        config, validation_fraction=0.25, split_seed=7, augmentation_seed=11
    )
    assert calls == [(str(tmp_path), True, False)]
    assert train.dataset is not validation.dataset
    train_ids = {train[index][2] for index in range(len(train))}
    validation_ids = {validation[index][2] for index in range(len(validation))}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == set(range(12))
    sample_id = train[0][2]
    train.set_epoch(0)
    epoch_zero = train[0][0]
    train.set_epoch(1)
    epoch_one = train[0][0]
    alternate_train, _ = build_train_validation_views(
        config, validation_fraction=0.25, split_seed=7, augmentation_seed=12
    )
    alternate_train.set_epoch(0)
    assert sample_id == train[0][2]
    assert not torch.equal(epoch_zero, epoch_one)
    assert not torch.equal(epoch_zero, alternate_train[0][0])
    assert torch.equal(validation[0][0], validation[0][0])


def test_cifar_split_reads_targets_without_loading_or_transforming_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ImageCountingCIFAR(SyntheticCIFAR):
        def __init__(self) -> None:
            super().__init__(size=12, num_classes=10, image_size=32)
            self.image_reads = 0

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            self.image_reads += 1
            return super().__getitem__(index)

    raw = ImageCountingCIFAR()
    monkeypatch.setattr("ard.data.datasets.datasets.CIFAR10", lambda **_: raw)
    config = DatasetConfig(name="cifar10", root=tmp_path, split="train", download=False, num_classes=10)
    train, validation = build_train_validation_views(
        config, validation_fraction=0.25, split_seed=7, augmentation_seed=11
    )
    assert raw.image_reads == 0
    train_ids = [train[index][2] for index in range(len(train))]
    validation_ids = [validation[index][2] for index in range(len(validation))]
    assert raw.image_reads == len(train) + len(validation)
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(range(len(raw)))


def test_official_test_is_never_part_of_train_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def factory(*, root: str, train: bool, download: bool) -> SyntheticCIFAR:
        calls.append(train)
        return SyntheticCIFAR(size=12, num_classes=10, image_size=32)

    monkeypatch.setattr("ard.data.datasets.datasets.CIFAR10", factory)
    config = DatasetConfig(name="cifar10", root=tmp_path, split="train", download=False, num_classes=10)
    build_train_validation_views(config, validation_fraction=0.25, split_seed=7, augmentation_seed=11)
    assert calls == [True]
