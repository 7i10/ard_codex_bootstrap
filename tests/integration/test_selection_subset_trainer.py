"""``training.selection_subset_size`` in the Trainer (plan 0103 Phase 2), on CPU.

* Training is unchanged: last.pt's model / optimizer / scheduler / RNG / step equal a
  full-split-selection run's bit for bit.
* Per-epoch selection reads only the subset loader; the full held-out loader is read only
  at the final epoch.
* At the final epoch, last AND best.pt (and best-ema.pt) are evaluated on the full split,
  under distinct keys, with exactly the numbers a direct evaluation of the saved weights
  gives, and the live weights are restored.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config.schema import AttackConfig, ModelConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SourceIndexedSubset,
    SyntheticCIFAR,
    build_selection_subset_view,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import PGDATObjective
from ard.policies import selected_ids_sha256

pytestmark = pytest.mark.t3

SUBSET_SIZE = 3
WALL_CLOCK = {"train_seconds", "train_images_per_second"}
SUBSET_KEYS = {"val_selection_subset_size", "val_selection_subset_sha256"}
FULL_KEYS = {
    "val_full_num_examples",
    "val_full_clean_accuracy",
    "val_full_pgd_accuracy",
    "best_val_full_epoch",
    "best_val_full_clean_accuracy",
    "best_val_full_pgd_accuracy",
}
FULL_EMA_KEYS = {
    "val_full_clean_accuracy_ema",
    "val_full_pgd_accuracy_ema",
    "best_ema_val_full_epoch",
    "best_ema_val_full_clean_accuracy",
    "best_ema_val_full_pgd_accuracy",
}


def _loader(dataset: SourceIndexedSubset, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(dataset), seed=4, shuffle=shuffle),
        collate_fn=collate_indexed,
    )


def _loaders() -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    dataset = IndexedDataset(SyntheticCIFAR(size=24, num_classes=3, image_size=4, seed=4))
    train, validation = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=4)
    subset = build_selection_subset_view(validation, size=SUBSET_SIZE, seed=4)
    metadata = {
        "size": len(subset),
        "ids_sha256": selected_ids_sha256(tuple(subset.indices)),
        "full_split_size": len(validation),
        "seed": 4,
        "sampling": "class_stratified",
        "source": "held_out_validation_split",
    }
    return _loader(train, shuffle=True), _loader(validation, shuffle=False), _loader(subset, shuffle=False), metadata


def _trainer(output: Path, *, selection_subset: dict[str, Any] | None, ema: bool = False) -> Trainer:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    selection = AttackConfig(
        epsilon="2/255", step_size="2/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=1, gamma=0.8),
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="2/255", step_size="2/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(selection),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="a" * 64,
        seed=4,
        tracker_run_id="offline-fixture",
        weight_ema_decay=0.5 if ema else None,
        selection_subset=selection_subset,
    )


def _seed() -> None:
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)


def _equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    return payload


def test_training_is_unchanged_and_selection_reads_only_the_subset(tmp_path: Path) -> None:
    train_a, full_a, _, _ = _loaders()
    train_b, full_b, subset_b, metadata = _loaders()
    plain = _trainer(tmp_path / "plain", selection_subset=None)
    subset = _trainer(tmp_path / "subset", selection_subset=metadata)
    calls: list[str] = []
    original = subset.validate_epoch

    def recording(loader: DataLoader, *, model: torch.nn.Module | None = None) -> dict[str, float]:
        calls.append({id(subset_b): "subset", id(full_b): "full"}[id(loader)])
        return original(loader, model=model)

    subset.validate_epoch = recording  # type: ignore[method-assign]
    _seed()
    plain_rows = plain.fit(train_a, validation_loader=full_a, epochs=3)
    _seed()
    subset_rows = subset.fit(train_b, validation_loader=subset_b, full_validation_loader=full_b, epochs=3)
    # Selection every epoch on the subset; the full split only at the final epoch (last; best
    # reuses or reloads -- one or two passes).
    assert calls[:3] == ["subset"] * 3 and set(calls[3:]) == {"full"} and 1 <= len(calls[3:]) <= 2

    # Training identical: every training-side checkpoint key of last.pt, and the train_* rows.
    first, second = _load(tmp_path / "plain" / "last.pt"), _load(tmp_path / "subset" / "last.pt")
    for key in ("model", "optimizer", "scheduler", "rng", "global_step", "sampler_state", "sample_state", "epoch"):
        assert _equal(first[key], second[key]), key
    for plain_row, subset_row in zip(plain_rows, subset_rows, strict=True):
        for key, value in plain_row.items():
            if key.startswith("train_") and key not in WALL_CLOCK or key.endswith("learning_rate"):
                assert subset_row[key] == value, key

    # Labelled rows: subset size/digest on every row; full-split keys only on the final row.
    for epoch, row in enumerate(subset_rows):
        assert row["val_selection_subset_size"] == SUBSET_SIZE
        assert row["val_selection_subset_sha256"] == metadata["ids_sha256"]
        assert FULL_KEYS.isdisjoint(row) if epoch < 2 else FULL_KEYS <= set(row)
    assert SUBSET_KEYS.isdisjoint(set().union(*plain_rows)) and FULL_KEYS.isdisjoint(set().union(*plain_rows))
    assert subset_rows[-1]["val_full_num_examples"] == len(full_b.dataset)

    # The selection record states the subset and holds the subset numbers.
    selection = second["selection_metadata"]
    assert selection["selection_subset"] == metadata
    selected = selection["selected_epoch"]
    assert selection["selected_pgd_accuracy"] == subset_rows[selected]["val_pgd_accuracy"]
    assert selection["last_pgd_accuracy"] == subset_rows[-1]["val_pgd_accuracy"]
    assert "selection_subset" not in first["selection_metadata"]
    full = selection["full_split_final"]
    assert full["best_epoch"] == selected == subset_rows[-1]["best_val_full_epoch"]
    assert full["last_pgd_accuracy"] == subset_rows[-1]["val_full_pgd_accuracy"]
    assert full["best_pgd_accuracy"] == subset_rows[-1]["best_val_full_pgd_accuracy"]
    # Live weights after the final full-split passes are the last weights.
    for name, tensor in subset.model.state_dict().items():
        assert torch.equal(tensor, second["model"][name]), name


def _force_early_best(trainer: Trainer, subset_loader: DataLoader) -> None:
    """Subset numbers that fall every epoch, so best.pt stays at epoch 0 and the final
    full-split pass must reload it from disk; full-split passes stay real."""
    original = trainer.validate_epoch
    falling = iter([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    def fake(loader: DataLoader, *, model: torch.nn.Module | None = None) -> dict[str, float]:
        if loader is subset_loader:
            value = next(falling)
            return {"clean_accuracy": value, "pgd_accuracy": value}
        return original(loader, model=model)

    trainer.validate_epoch = fake  # type: ignore[method-assign]


def _direct(trainer: Trainer, path: Path, key: str, loader: DataLoader) -> dict[str, float]:
    """Full-split metrics of the saved weights, evaluated directly on a fresh model."""
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    model.load_state_dict(_load(path)[key])
    return Trainer.validate_epoch(trainer, loader, model=model)


@pytest.mark.parametrize("ema", [False, True], ids=["student", "weight-ema"])
def test_final_full_split_evaluates_saved_best_and_last_on_the_same_images(tmp_path: Path, ema: bool) -> None:
    train, full, subset_loader, metadata = _loaders()
    trainer = _trainer(tmp_path, selection_subset=metadata, ema=ema)
    _force_early_best(trainer, subset_loader)
    _seed()
    rows = trainer.fit(train, validation_loader=subset_loader, full_validation_loader=full, epochs=3)
    best, last = _load(tmp_path / "best.pt"), _load(tmp_path / "last.pt")
    assert best["epoch"] == 0 and last["epoch"] == 2
    # best.pt was written at epoch 0, before any full-split pass; it is never rewritten.
    assert "full_split_final" not in best["selection_metadata"]
    assert best["selection_metadata"]["selection_subset"] == metadata
    final = rows[-1]
    expected_best = _direct(trainer, tmp_path / "best.pt", "model", full)
    expected_last = _direct(trainer, tmp_path / "last.pt", "model", full)
    assert final["best_val_full_epoch"] == 0
    assert (final["best_val_full_clean_accuracy"], final["best_val_full_pgd_accuracy"]) == (
        expected_best["clean_accuracy"],
        expected_best["pgd_accuracy"],
    )
    assert (final["val_full_clean_accuracy"], final["val_full_pgd_accuracy"]) == (
        expected_last["clean_accuracy"],
        expected_last["pgd_accuracy"],
    )
    assert last["selection_metadata"]["full_split_final"] == {
        "num_examples": len(full.dataset),
        "last_epoch": 2,
        "last_clean_accuracy": expected_last["clean_accuracy"],
        "last_pgd_accuracy": expected_last["pgd_accuracy"],
        "best_epoch": 0,
        "best_clean_accuracy": expected_best["clean_accuracy"],
        "best_pgd_accuracy": expected_best["pgd_accuracy"],
    }
    # Subset numbers stay the selection numbers, distinct from the full-split ones.
    # (with EMA the student and EMA passes alternate on the falling sequence)
    assert final["val_pgd_accuracy"] == pytest.approx(0.5 if ema else 0.7)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(tensor, last["model"][name]), name
    if ema:
        best_ema = _load(tmp_path / "best-ema.pt")
        assert best_ema["epoch"] == 0
        expected = _direct(trainer, tmp_path / "best-ema.pt", "ema", full)
        expected_last_ema = _direct(trainer, tmp_path / "last.pt", "ema", full)
        assert FULL_EMA_KEYS <= set(final)
        assert final["best_ema_val_full_pgd_accuracy"] == expected["pgd_accuracy"]
        assert final["val_full_pgd_accuracy_ema"] == expected_last_ema["pgd_accuracy"]
        assert last["selection_metadata_ema"]["full_split_final"]["best_epoch"] == 0
        assert trainer.ema_model is not None
        for name, tensor in trainer.ema_model.state_dict().items():
            assert torch.equal(tensor, last["ema"][name]), name
    else:
        assert FULL_EMA_KEYS.isdisjoint(final)


def test_subset_and_full_loader_must_come_together(tmp_path: Path) -> None:
    train, full, subset_loader, metadata = _loaders()
    with pytest.raises(ValueError, match="together"):
        _trainer(tmp_path / "a", selection_subset=metadata).fit(train, validation_loader=subset_loader, epochs=1)
    with pytest.raises(ValueError, match="together"):
        _trainer(tmp_path / "b", selection_subset=None).fit(
            train, validation_loader=full, full_validation_loader=full, epochs=1
        )
    with pytest.raises(ValueError, match="lacks"):
        _trainer(tmp_path / "c", selection_subset={"size": 1})
    with pytest.raises(ValueError, match="full_split_size"):
        _trainer(tmp_path / "d", selection_subset={**metadata, "size": metadata["full_split_size"]})


def test_a_best_checkpoint_from_another_epoch_is_refused(tmp_path: Path) -> None:
    """The final pass evaluates exactly the file selection recorded, never a stale one."""
    train, full, subset_loader, metadata = _loaders()
    trainer = _trainer(tmp_path, selection_subset=metadata)
    _force_early_best(trainer, subset_loader)
    trainer.fit(train, validation_loader=subset_loader, full_validation_loader=full, epochs=2)
    trainer.selection_metadata["selected_epoch"] = 1
    with pytest.raises(RuntimeError, match="holds epoch 0, but selection recorded epoch 1"):
        trainer._validate_saved_weights(full, tmp_path / "best.pt", key="model", model=trainer.model, expected_epoch=1)


def test_a_best_checkpoint_from_another_run_is_refused(tmp_path: Path) -> None:
    train, full, subset_loader, metadata = _loaders()
    trainer = _trainer(tmp_path, selection_subset=metadata)
    _force_early_best(trainer, subset_loader)
    trainer.fit(train, validation_loader=subset_loader, full_validation_loader=full, epochs=2)
    trainer.config_hash = "f" * 64
    with pytest.raises(RuntimeError, match="different config hash"):
        trainer._validate_saved_weights(full, tmp_path / "best.pt", key="model", model=trainer.model, expected_epoch=0)
