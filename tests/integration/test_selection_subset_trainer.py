"""``training.selection_subset_size`` in the Trainer (plan 0103 Phase 2), on CPU.

* Training is unchanged: last.pt's model / optimizer / scheduler / RNG / step equal a
  full-split-selection run's bit for bit.
* Per-epoch selection reads only the subset loader and is named val_subset_*; the full
  held-out loader is read only at the final epoch.
* At the final epoch, last AND best.pt (and best-ema.pt) are evaluated on the full split and
  on its complement (held-out minus subset), under distinct keys, with exactly the numbers a
  direct evaluation of the saved weights gives, and the live weights are restored.
* Resume continues the same subset (an interrupted run equals an uninterrupted one), refuses a
  different subset or a missing / stale best.pt, and a fork child adopts its own subset (or
  none) and never inherits the parent's final full-split record.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.analysis.intervention_fork import _fork_selection_metadata
from ard.attacks import LinfPGD
from ard.cli.train import _selection_subset_summary
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
EPOCHS = 3
WALL_CLOCK = {"train_seconds", "train_images_per_second"}
FULL_KEYS = {
    "val_full_num_examples",
    "val_complement_num_examples",
    "val_full_clean_accuracy",
    "val_full_pgd_accuracy",
    "val_complement_clean_accuracy",
    "val_complement_pgd_accuracy",
    "best_val_full_epoch",
    "best_val_full_clean_accuracy",
    "best_val_full_pgd_accuracy",
    "best_val_complement_clean_accuracy",
    "best_val_complement_pgd_accuracy",
}
FULL_EMA_KEYS = {
    "val_full_clean_accuracy_ema",
    "val_full_pgd_accuracy_ema",
    "val_complement_clean_accuracy_ema",
    "val_complement_pgd_accuracy_ema",
    "best_ema_val_full_epoch",
    "best_ema_val_full_clean_accuracy",
    "best_ema_val_full_pgd_accuracy",
    "best_ema_val_complement_clean_accuracy",
    "best_ema_val_complement_pgd_accuracy",
}


class Data:
    def __init__(self, *, subset_seed: int = 4) -> None:
        dataset = IndexedDataset(SyntheticCIFAR(size=48, num_classes=3, image_size=4, seed=4))
        train, validation = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=4)
        subset = build_selection_subset_view(validation, size=SUBSET_SIZE, seed=subset_seed)
        self.ids = list(subset.indices)
        self.metadata = {
            "size": len(subset),
            "ids_sha256": selected_ids_sha256(tuple(subset.indices)),
            "full_split_size": len(validation),
            "seed": subset_seed,
            "sampling": "class_stratified",
            "source": "held_out_validation_split",
        }
        self.train = _loader(train, shuffle=True)
        self.full = _loader(validation, shuffle=False)
        self.subset = _loader(subset, shuffle=False)


def _loader(dataset: SourceIndexedSubset, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(dataset), seed=4, shuffle=shuffle),
        collate_fn=collate_indexed,
    )


def _trainer(output: Path, data: Data | None, *, ema: bool = False, config_hash: str = "a" * 64) -> Trainer:
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
        config_hash=config_hash,
        seed=4,
        tracker_run_id="offline-fixture",
        weight_ema_decay=0.5 if ema else None,
        selection_subset=None if data is None else data.metadata,
        selection_subset_ids=None if data is None else data.ids,
    )


def _fit(trainer: Trainer, data: Data, *, subset: bool = True, **kwargs: Any) -> list[dict[str, Any]]:
    if subset:
        return trainer.fit(data.train, validation_loader=data.subset, full_validation_loader=data.full, **kwargs)
    return trainer.fit(data.train, validation_loader=data.full, **kwargs)


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


def _without_wall_clock(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in WALL_CLOCK}


# =========================================================================== training and selection


def test_training_is_unchanged_and_selection_reads_only_the_subset(tmp_path: Path) -> None:
    plain_data, data = Data(), Data()
    plain = _trainer(tmp_path / "plain", None)
    subset = _trainer(tmp_path / "subset", data)
    calls: list[str] = []
    original = subset.validate_epoch

    def recording(loader: DataLoader, *, model: torch.nn.Module | None = None) -> dict[str, float]:
        calls.append({id(data.subset): "subset", id(data.full): "full"}[id(loader)])
        return original(loader, model=model)

    subset.validate_epoch = recording  # type: ignore[method-assign]
    _seed()
    plain_rows = _fit(plain, plain_data, subset=False, epochs=EPOCHS)
    _seed()
    subset_rows = _fit(subset, data, epochs=EPOCHS)
    # Selection every epoch on the subset (the final held-out passes do not select).
    assert calls == ["subset"] * EPOCHS

    # Training identical: every training-side checkpoint key of last.pt, and the train_* rows.
    first, second = _load(tmp_path / "plain" / "last.pt"), _load(tmp_path / "subset" / "last.pt")
    for key in ("model", "optimizer", "scheduler", "rng", "global_step", "sampler_state", "sample_state", "epoch"):
        assert _equal(first[key], second[key]), key
    for plain_row, subset_row in zip(plain_rows, subset_rows, strict=True):
        for key, value in plain_row.items():
            if key.startswith("train_") and key not in WALL_CLOCK or key.endswith("learning_rate"):
                assert subset_row[key] == value, key

    # Distinct names: val_subset_* replace val_* when the option is on; Phase-1 names otherwise.
    for epoch, row in enumerate(subset_rows):
        assert {"val_clean_accuracy", "val_pgd_accuracy"}.isdisjoint(row)
        assert {"val_subset_clean_accuracy", "val_subset_pgd_accuracy"} <= set(row)
        assert row["val_selection_subset_size"] == SUBSET_SIZE
        assert row["val_selection_subset_sha256"] == data.metadata["ids_sha256"]
        assert FULL_KEYS.isdisjoint(row) if epoch < EPOCHS - 1 else FULL_KEYS <= set(row)
    for row in plain_rows:
        assert {"val_clean_accuracy", "val_pgd_accuracy"} <= set(row)
        assert not any(key.startswith(("val_subset", "val_selection", "val_full", "val_complement")) for key in row)
    final = subset_rows[-1]
    assert final["val_full_num_examples"] == len(data.full.dataset) == 12
    assert final["val_complement_num_examples"] == 12 - SUBSET_SIZE

    # The selection record states the subset and holds the subset numbers.
    selection = second["selection_metadata"]
    assert (
        selection["metric"] == "val_subset_pgd_accuracy" and first["selection_metadata"]["metric"] == "val_pgd_accuracy"
    )
    assert selection["selection_subset"] == data.metadata
    selected = selection["selected_epoch"]
    assert selection["selected_pgd_accuracy"] == subset_rows[selected]["val_subset_pgd_accuracy"]
    assert selection["last_pgd_accuracy"] == final["val_subset_pgd_accuracy"]
    assert "selection_subset" not in first["selection_metadata"]
    full = selection["full_split_final"]
    assert full["best_epoch"] == selected == final["best_val_full_epoch"]
    assert full["last_pgd_accuracy"] == final["val_full_pgd_accuracy"]
    assert full["best_complement_pgd_accuracy"] == final["best_val_complement_pgd_accuracy"]
    summary = _selection_subset_summary(selection, second.get("selection_metadata_ema"))
    assert summary["last_full_pgd_accuracy"] == final["val_full_pgd_accuracy"]
    assert summary["last_complement_clean_accuracy"] == final["val_complement_clean_accuracy"]
    assert summary["robust_overfit_gap_complement"] == pytest.approx(
        final["best_val_complement_pgd_accuracy"] - final["val_complement_pgd_accuracy"]
    )
    # Live weights after the final held-out passes are the last weights.
    for name, tensor in subset.model.state_dict().items():
        assert torch.equal(tensor, second["model"][name]), name


def _force_early_best(trainer: Trainer, subset_loader: DataLoader) -> None:
    """Subset numbers that fall every call, so best.pt stays at epoch 0 and the final
    held-out pass must reload it from disk; the held-out passes stay real."""
    original = trainer.validate_epoch
    falling = iter([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    def fake(loader: DataLoader, *, model: torch.nn.Module | None = None) -> dict[str, float]:
        if loader is subset_loader:
            value = next(falling)
            return {"clean_accuracy": value, "pgd_accuracy": value}
        return original(loader, model=model)

    trainer.validate_epoch = fake  # type: ignore[method-assign]


def _direct(trainer: Trainer, path: Path, key: str, loader: DataLoader) -> dict[str, float]:
    """Held-out metrics of the saved weights, evaluated directly on a fresh model."""
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    model.load_state_dict(_load(path)[key])
    return trainer._held_out_metrics(loader, model)


def _expected_record(last: dict[str, float], best: dict[str, float], *, best_epoch: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "num_examples": 12,
        "complement_num_examples": 12 - SUBSET_SIZE,
        "last_epoch": EPOCHS - 1,
        "best_epoch": best_epoch,
    }
    for which, values in (("last", last), ("best", best)):
        for part in ("", "complement_"):
            for metric in ("clean", "pgd"):
                record[f"{which}_{part}{metric}_accuracy"] = values[f"{part}{metric}_accuracy"]
    return record


@pytest.mark.parametrize("ema", [False, True], ids=["student", "weight-ema"])
def test_final_held_out_passes_evaluate_saved_best_and_last_on_the_same_images(tmp_path: Path, ema: bool) -> None:
    data = Data()
    trainer = _trainer(tmp_path, data, ema=ema)
    _force_early_best(trainer, data.subset)
    _seed()
    rows = _fit(trainer, data, epochs=EPOCHS)
    best, last = _load(tmp_path / "best.pt"), _load(tmp_path / "last.pt")
    assert best["epoch"] == 0 and last["epoch"] == EPOCHS - 1
    # best.pt was written at epoch 0, before any held-out pass; it is never rewritten.
    assert "full_split_final" not in best["selection_metadata"]
    assert best["selection_metadata"]["selection_subset"] == data.metadata
    final = rows[-1]
    expected_best = _direct(trainer, tmp_path / "best.pt", "model", data.full)
    expected_last = _direct(trainer, tmp_path / "last.pt", "model", data.full)
    assert expected_best["complement_num_examples"] == 12 - SUBSET_SIZE
    assert last["selection_metadata"]["full_split_final"] == _expected_record(
        expected_last, expected_best, best_epoch=0
    )
    assert final["best_val_full_epoch"] == 0
    assert final["best_val_full_pgd_accuracy"] == expected_best["pgd_accuracy"]
    assert final["best_val_complement_clean_accuracy"] == expected_best["complement_clean_accuracy"]
    assert final["val_full_clean_accuracy"] == expected_last["clean_accuracy"]
    assert final["val_complement_pgd_accuracy"] == expected_last["complement_pgd_accuracy"]
    # Subset numbers stay the selection numbers (with EMA the student and EMA passes
    # alternate on the falling sequence).
    assert final["val_subset_pgd_accuracy"] == pytest.approx(0.5 if ema else 0.7)
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(tensor, last["model"][name]), name
    if ema:
        assert _load(tmp_path / "best-ema.pt")["epoch"] == 0
        expected = _direct(trainer, tmp_path / "best-ema.pt", "ema", data.full)
        expected_last_ema = _direct(trainer, tmp_path / "last.pt", "ema", data.full)
        assert FULL_EMA_KEYS <= set(final)
        assert {"val_subset_clean_accuracy_ema", "val_subset_pgd_accuracy_ema"} <= set(final)
        assert last["selection_metadata_ema"]["full_split_final"] == _expected_record(
            expected_last_ema, expected, best_epoch=0
        )
        assert final["best_ema_val_complement_pgd_accuracy"] == expected["complement_pgd_accuracy"]
        assert trainer.ema_model is not None
        for name, tensor in trainer.ema_model.state_dict().items():
            assert torch.equal(tensor, last["ema"][name]), name
    else:
        assert FULL_EMA_KEYS.isdisjoint(final)


def test_mismatched_subset_arguments_are_refused(tmp_path: Path) -> None:
    data = Data()
    with pytest.raises(ValueError, match="together"):
        _trainer(tmp_path / "a", data).fit(data.train, validation_loader=data.subset, epochs=1)
    with pytest.raises(ValueError, match="together"):
        _trainer(tmp_path / "b", None).fit(
            data.train, validation_loader=data.full, full_validation_loader=data.full, epochs=1
        )
    base = _trainer(tmp_path / "c", None)
    common = {key: getattr(base, key) for key in ("model", "optimizer", "scheduler", "objective")}
    kwargs = dict(
        **common,
        scaler=None,
        attack=None,
        selection_attack=base.selection_attack,
        device=torch.device("cpu"),
        output_dir=tmp_path / "c",
        config_hash="a" * 64,
        seed=4,
    )
    with pytest.raises(ValueError, match="together"):
        Trainer(**kwargs, selection_subset=data.metadata)
    with pytest.raises(ValueError, match="lacks"):
        Trainer(**kwargs, selection_subset={"size": 1}, selection_subset_ids=[0])
    with pytest.raises(ValueError, match="full_split_size"):
        Trainer(**kwargs, selection_subset={**data.metadata, "size": 12}, selection_subset_ids=data.ids)
    with pytest.raises(ValueError, match="digest"):
        Trainer(**kwargs, selection_subset=data.metadata, selection_subset_ids=[*data.ids[:-1], data.ids[-1] + 1])


def test_a_best_checkpoint_from_another_epoch_or_run_is_refused(tmp_path: Path) -> None:
    """The final pass evaluates exactly the file selection recorded, never a stale one."""
    data = Data()
    trainer = _trainer(tmp_path, data)
    _force_early_best(trainer, data.subset)
    _fit(trainer, data, epochs=2)
    with pytest.raises(RuntimeError, match="holds epoch 0, but selection recorded epoch 1"):
        trainer._saved_selection_weights(tmp_path / "best.pt", key="model", expected_epoch=1)
    trainer.config_hash = "f" * 64
    with pytest.raises(RuntimeError, match="different config hash"):
        trainer._saved_selection_weights(tmp_path / "best.pt", key="model", expected_epoch=0)


# =========================================================================== resume and forks


class _Interrupt(Exception):
    pass


def _interrupt_after_first_epoch(metrics: Any, improved: bool) -> None:
    raise _Interrupt


def test_a_mid_run_resume_equals_an_uninterrupted_run(tmp_path: Path) -> None:
    whole_data, data = Data(), Data()
    whole = _trainer(tmp_path / "whole", whole_data)
    _seed()
    whole_rows = _fit(whole, whole_data, epochs=EPOCHS)

    interrupted = _trainer(tmp_path / "split", data)
    _seed()
    with pytest.raises(_Interrupt):
        _fit(interrupted, data, epochs=EPOCHS, on_epoch_end=_interrupt_after_first_epoch)
    resumed_data = Data()
    resumed = _trainer(tmp_path / "split", resumed_data)
    start = resumed.resume(tmp_path / "split" / "last.pt", sampler=resumed_data.train.sampler).next_epoch
    assert start == 1
    resumed_rows = _fit(resumed, resumed_data, epochs=EPOCHS, start_epoch=start)
    assert [_without_wall_clock(row) for row in resumed_rows] == [_without_wall_clock(row) for row in whole_rows[1:]]
    for name in ("best.pt", "last.pt"):
        left, right = _load(tmp_path / "whole" / name), _load(tmp_path / "split" / name)
        for key in ("model", "optimizer", "epoch", "selection_metadata", "best_metric", "global_step"):
            assert _equal(left[key], right[key]), (name, key)


def test_resume_refuses_a_different_subset_and_a_missing_best_checkpoint(tmp_path: Path) -> None:
    data = Data()
    trainer = _trainer(tmp_path, data)
    _seed()
    with pytest.raises(_Interrupt):
        _fit(trainer, data, epochs=EPOCHS, on_epoch_end=_interrupt_after_first_epoch)
    other = Data(subset_seed=5)
    assert other.metadata["ids_sha256"] != data.metadata["ids_sha256"]
    changed = _trainer(tmp_path, other)
    with pytest.raises(ValueError, match="selection subset .* differs"):
        changed.resume(tmp_path / "last.pt", sampler=other.train.sampler)
    without = _trainer(tmp_path, None)
    with pytest.raises(ValueError, match="selection subset .* differs"):
        without.resume(tmp_path / "last.pt", sampler=data.train.sampler)
    (tmp_path / "best.pt").unlink()
    same = _trainer(tmp_path, Data())
    with pytest.raises(FileNotFoundError, match="best.pt does not exist"):
        same.resume(tmp_path / "last.pt", sampler=data.train.sampler)


def _fork(parent_last: Path, child_dir: Path, *, config_hash: str) -> Path:
    """What the fork builders do: copy the parent's checkpoint, reset selection with a
    scope, and give it the child's config hash."""
    payload = copy.deepcopy(_load(parent_last))
    payload["selection_metadata"] = _fork_selection_metadata(payload["selection_metadata"])
    payload["config_hash"] = config_hash
    payload["best_metric"] = float("-inf")
    child_dir.mkdir(parents=True)
    path = child_dir / "fork.pt"
    torch.save(payload, path)
    return path


def test_a_fork_from_a_subset_run_into_a_full_split_child_drops_the_subset(tmp_path: Path) -> None:
    data = Data()
    _seed()
    _fit(_trainer(tmp_path / "parent", data), data, epochs=2)
    parent = _load(tmp_path / "parent" / "last.pt")["selection_metadata"]
    assert "full_split_final" in parent and "selection_subset" in parent
    fork = _fork(tmp_path / "parent" / "last.pt", tmp_path / "child", config_hash="b" * 64)
    child_data = Data()
    child = _trainer(tmp_path / "child", None, config_hash="b" * 64)
    assert child.resume(fork, sampler=child_data.train.sampler).next_epoch == 2
    assert "selection_subset" not in child.selection_metadata
    assert "full_split_final" not in child.selection_metadata
    assert child.selection_metadata["metric"] == "val_pgd_accuracy"
    rows = _fit(child, child_data, subset=False, epochs=EPOCHS, start_epoch=2)
    assert {"val_clean_accuracy", "val_pgd_accuracy"} <= set(rows[0])
    assert not any(key.startswith(("val_subset", "val_selection", "val_full", "val_complement")) for key in rows[0])
    last = _load(tmp_path / "child" / "last.pt")
    assert "selection_subset" not in last["selection_metadata"] and "full_split_final" not in last["selection_metadata"]
    assert _selection_subset_summary(last["selection_metadata"], last.get("selection_metadata_ema")) == {}


def test_a_fork_from_a_full_split_run_into_a_subset_child_adopts_its_subset(tmp_path: Path) -> None:
    parent_data = Data()
    _seed()
    _fit(_trainer(tmp_path / "parent", None), parent_data, subset=False, epochs=2)
    fork = _fork(tmp_path / "parent" / "last.pt", tmp_path / "child", config_hash="b" * 64)
    data = Data()
    child = _trainer(tmp_path / "child", data, config_hash="b" * 64)
    assert child.resume(fork, sampler=data.train.sampler).next_epoch == 2
    assert child.selection_metadata["selection_subset"] == data.metadata
    assert child.selection_metadata["metric"] == "val_subset_pgd_accuracy"
    rows = _fit(child, data, epochs=EPOCHS, start_epoch=2)
    assert FULL_KEYS <= set(rows[-1]) and "val_subset_pgd_accuracy" in rows[-1]
    final = _load(tmp_path / "child" / "last.pt")["selection_metadata"]["full_split_final"]
    assert final["best_epoch"] == 2 and final["last_epoch"] == 2
    assert final["best_pgd_accuracy"] == final["last_pgd_accuracy"] == rows[-1]["val_full_pgd_accuracy"]
