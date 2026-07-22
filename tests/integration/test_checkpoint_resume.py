from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.attacks.base import AttackResult
from ard.config.schema import AttackConfig, ModelConfig, SchedulerConfig
from ard.data import (
    EpochShuffleSampler,
    EpochSourceTransform,
    IndexedBatch,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.checkpoint import REQUIRED_KEYS, load_checkpoint, save_checkpoint
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import ObjectiveTerms, PGDATObjective
from ard.schedules import build_scheduler
from ard.state import SampleStateStore
from ard.tracking import NullTracker, coordinated_tracker_action
from ard.tracking.diagnostics import TrainingDiagnostics


def test_training_diagnostics_are_observational_for_full_checkpoint_state(tmp_path: Path) -> None:
    import random

    import numpy as np

    def equal(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    plain = make_trainer(tmp_path / "plain-diagnostics")
    observed = make_trainer(tmp_path / "observed-diagnostics")
    loader_a, validation_a, _ = make_loaders()
    loader_b, validation_b, _ = make_loaders()
    observed.diagnostics = TrainingDiagnostics.for_ids(list(range(8)), seed=4, size=2)
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    plain.fit(loader_a, validation_loader=validation_a, epochs=1)
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    observed.fit(loader_b, validation_loader=validation_b, epochs=1)
    for name in ("best.pt", "last.pt"):
        first = torch.load(tmp_path / "plain-diagnostics" / name, map_location="cpu", weights_only=False)
        second = torch.load(tmp_path / "observed-diagnostics" / name, map_location="cpu", weights_only=False)
        assert REQUIRED_KEYS.issubset(first) and REQUIRED_KEYS.issubset(second)
        for key in REQUIRED_KEYS:
            assert equal(first[key], second[key]), key


pytestmark = pytest.mark.t3


def make_loaders(seed: int = 4) -> tuple[DataLoader, DataLoader, EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_indexed)
    validation_loader = DataLoader(
        validation_dataset, batch_size=4, sampler=validation_sampler, collate_fn=collate_indexed
    )
    return loader, validation_loader, sampler


def make_trainer(output: Path, *, seed: int = 4) -> Trainer:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255",
                step_size="1/255",
                steps=1,
                random_start=True,
                student_mode="eval",
                teacher_mode="eval",
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="a" * 64,
        seed=seed,
        tracker_run_id="offline-fixture",
    )


def test_checkpoint_is_complete_and_best_last_are_distinct(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.sample_state = {"placeholder_version": 1}
    loader, validation_loader, _ = make_loaders()
    callback_metrics: list[dict[str, float]] = []
    history = trainer.fit(
        loader,
        validation_loader=validation_loader,
        epochs=2,
        on_epoch_end=lambda metrics, _: callback_metrics.append(dict(metrics)),
    )
    best, last = tmp_path / "best.pt", tmp_path / "last.pt"
    assert best.is_file() and last.is_file() and best != last
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert REQUIRED_KEYS.issubset(payload)
    assert payload["epoch"] == 1 and payload["epoch_boundary"] == "end"
    assert payload["sample_state"] == {"placeholder_version": 1}
    assert payload["tracker_run_id"] == "offline-fixture"
    assert payload["world_size"] == 1
    assert set(history[0]) == {
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "train_valid_examples",
        "train_seconds",
        "train_images_per_second",
        "train_cuda_peak_allocated_bytes",
        "train_teacher_clean_forward_calls",
        "val_clean_accuracy",
        "val_pgd_accuracy",
    }
    assert history[0]["train_valid_examples"] == float(len(cast(Sized, loader.dataset)))
    assert history[0]["train_teacher_clean_forward_calls"] == 0.0
    assert history[0]["train_cuda_peak_allocated_bytes"] == 0.0
    assert callback_metrics == history
    assert payload["selection_metadata"]["metric"] == "val_pgd_accuracy"
    assert payload["selection_metadata"]["tie_break"] == "earliest_epoch"


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    uninterrupted = make_trainer(tmp_path / "full")
    full_loader, full_validation_loader, _ = make_loaders()
    uninterrupted_history = uninterrupted.fit(full_loader, validation_loader=full_validation_loader, epochs=2)

    first_leg = make_trainer(tmp_path / "resumed")
    first_loader, first_validation_loader, _ = make_loaders()
    first_leg_history = first_leg.fit(first_loader, validation_loader=first_validation_loader, epochs=1)
    resumed = make_trainer(tmp_path / "resumed")
    resumed_loader, resumed_validation_loader, resumed_sampler = make_loaders()
    state = resumed.resume(tmp_path / "resumed" / "last.pt", sampler=resumed_sampler)
    assert state.next_epoch == 1
    resumed_history = resumed.fit(
        resumed_loader, validation_loader=resumed_validation_loader, epochs=2, start_epoch=state.next_epoch
    )

    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    assert uninterrupted.global_step == resumed.global_step
    assert uninterrupted.best_metric == resumed.best_metric
    deterministic_metrics = (
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "train_valid_examples",
        "train_teacher_clean_forward_calls",
        "val_clean_accuracy",
        "val_pgd_accuracy",
    )
    combined_history = first_leg_history + resumed_history
    for uninterrupted_epoch, resumed_epoch in zip(uninterrupted_history, combined_history, strict=True):
        assert {key: uninterrupted_epoch[key] for key in deterministic_metrics} == {
            key: resumed_epoch[key] for key in deterministic_metrics
        }
        assert uninterrupted_epoch["train_valid_examples"] == float(len(cast(Sized, full_loader.dataset)))
        assert uninterrupted_epoch["train_teacher_clean_forward_calls"] == 0.0


def _advance_optimizer_and_schedule(optimizer: SGD, scheduler: object, *, completed_epochs: int) -> None:
    for _ in range(completed_epochs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()  # type: ignore[union-attr]


def _state_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


@pytest.mark.parametrize("completed_epochs", (99, 100, 149, 150))
def test_multistep_optimizer_scheduler_checkpoint_roundtrip_matches_uninterrupted(
    tmp_path: Path, completed_epochs: int
) -> None:
    schedule = SchedulerConfig(id="multistep", milestones=(100, 150), gamma=0.1, step_at="epoch_end")

    def components() -> tuple[nn.Linear, SGD, object, EpochShuffleSampler]:
        torch.manual_seed(71)
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        return model, optimizer, build_scheduler(optimizer, schedule), EpochShuffleSampler(2, seed=3)

    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler, _ = components()
    _advance_optimizer_and_schedule(
        uninterrupted_optimizer, uninterrupted_scheduler, completed_epochs=completed_epochs + 1
    )

    saved_model, saved_optimizer, saved_scheduler, saved_sampler = components()
    _advance_optimizer_and_schedule(saved_optimizer, saved_scheduler, completed_epochs=completed_epochs)
    saved_sampler.set_epoch(completed_epochs)
    checkpoint = tmp_path / f"boundary-{completed_epochs}.pt"
    save_checkpoint(
        checkpoint,
        epoch=completed_epochs - 1,
        model=saved_model,
        optimizer=saved_optimizer,
        scheduler=saved_scheduler,
        scaler=None,
        sampler=saved_sampler,
        sample_state={},
        global_step=completed_epochs,
        best_metric=0.0,
        selection_metadata={},
        tracker_run_id="scheduler-roundtrip",
        config_hash="a" * 64,
    )

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_sampler = components()
    state = load_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
        sampler=resumed_sampler,
        expected_config_hash="a" * 64,
        device=torch.device("cpu"),
    )
    assert state.next_epoch == completed_epochs
    assert _state_equal(resumed_optimizer.state_dict(), saved_optimizer.state_dict())
    assert resumed_scheduler.state_dict() == saved_scheduler.state_dict()  # type: ignore[union-attr]
    assert resumed_optimizer.param_groups[0]["lr"] == saved_optimizer.param_groups[0]["lr"]

    _advance_optimizer_and_schedule(resumed_optimizer, resumed_scheduler, completed_epochs=1)
    assert _state_equal(resumed_model.state_dict(), uninterrupted_model.state_dict())
    assert _state_equal(resumed_optimizer.state_dict(), uninterrupted_optimizer.state_dict())
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler.state_dict()  # type: ignore[union-attr]
    assert resumed_optimizer.param_groups[0]["lr"] == uninterrupted_optimizer.param_groups[0]["lr"]


def test_epoch_keyed_augmentation_view_at_resumed_epoch_matches_uninterrupted() -> None:
    raw = SyntheticCIFAR(size=8, num_classes=2, image_size=32, seed=13)
    uninterrupted = IndexedDataset(raw, EpochSourceTransform(augmentation_seed=17))
    resumed = IndexedDataset(raw, EpochSourceTransform(augmentation_seed=17))
    uninterrupted.set_epoch(100)
    resumed.set_epoch(100)
    for source_id in range(len(raw)):
        first, _, first_id = uninterrupted[source_id]
        second, _, second_id = resumed[source_id]
        assert first_id == second_id == source_id
        assert torch.equal(first, second)


def test_rng_consuming_recording_callback_is_scientifically_observational(tmp_path: Path) -> None:
    plain = make_trainer(tmp_path / "plain")
    recorded = make_trainer(tmp_path / "recorded")
    loader_a, validation_a, _ = make_loaders()
    loader_b, validation_b, _ = make_loaders()
    plain.fit(loader_a, validation_loader=validation_a, epochs=2)
    tracker = NullTracker("recording-only")

    def callback(_: object, __: bool) -> None:
        def consume(_: object) -> None:
            import random

            random.random()
            torch.rand(31)

        coordinated_tracker_action(tracker, phase="recording parity", action=consume)

    recorded.fit(loader_b, validation_loader=validation_b, epochs=2, on_epoch_end=callback)
    for key, value in plain.model.state_dict().items():
        assert torch.equal(value, recorded.model.state_dict()[key])
    for name in ("best.pt", "last.pt"):
        first = torch.load(tmp_path / "plain" / name, map_location="cpu", weights_only=False)
        second = torch.load(tmp_path / "recorded" / name, map_location="cpu", weights_only=False)
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "best_metric",
            "global_step",
            "selection_metadata",
            "sample_state",
        ):
            assert (
                first[key] == second[key]
                if not isinstance(first[key], dict)
                else first[key].keys() == second[key].keys()
            )


def test_checkpoint_resume_restores_student_sample_store_exactly(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.sample_store = SampleStateStore(ema_decay=0.9)
    trainer.sample_state = trainer.sample_store.state_dict()
    loader, validation_loader, _ = make_loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    expected = trainer.sample_store.state_dict()
    assert expected["records"] and expected["pending"] == []

    resumed = make_trainer(tmp_path)
    resumed.sample_store = SampleStateStore(ema_decay=0.9)
    _, _, sampler = make_loaders()
    resumed.resume(tmp_path / "last.pt", sampler=sampler)
    assert resumed.sample_store.state_dict() == expected


def test_resume_rejects_world_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = make_trainer(tmp_path)
    loader, validation_loader, _ = make_loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    target = make_trainer(tmp_path)
    _, _, sampler = make_loaders()
    monkeypatch.setattr("ard.engine.checkpoint.get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world size"):
        target.resume(tmp_path / "last.pt", sampler=sampler)


@pytest.mark.parametrize("world_size", (0, -1, True, "1"))
def test_resume_rejects_invalid_checkpoint_world_size(tmp_path: Path, world_size: object) -> None:
    trainer = make_trainer(tmp_path)
    loader, validation_loader, _ = make_loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    checkpoint = tmp_path / "last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["world_size"] = world_size
    torch.save(payload, checkpoint)
    target = make_trainer(tmp_path)
    _, _, sampler = make_loaders()

    with pytest.raises(ValueError, match="positive integer"):
        target.resume(checkpoint, sampler=sampler)


def test_best_selection_uses_post_update_validation_and_keeps_earliest_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(tmp_path)
    trainer.scheduler = None
    loader, validation_loader, _ = make_loaders()
    train_results = iter(
        [
            {"loss": 0.0, "clean_accuracy": 0.0, "robust_accuracy": 1.0},
            {"loss": 0.0, "clean_accuracy": 0.0, "robust_accuracy": 0.0},
        ]
    )
    validation_results = iter(
        [
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.25},
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.25},
        ]
    )
    monkeypatch.setattr(trainer, "train_epoch", lambda _: next(train_results))
    monkeypatch.setattr(trainer, "validate_epoch", lambda _: next(validation_results))
    trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 0
    assert last["selection_metadata"]["selected_epoch"] == 0
    assert trainer.best_metric == pytest.approx(0.25)


def test_padded_rows_are_excluded_from_training_loss_and_accuracy(tmp_path: Path) -> None:
    class IdentityAttack:
        def generate(self, request):
            return AttackResult(request.inputs, torch.zeros_like(request.inputs), (), 0.0)

    class LabelObjective:
        def __call__(
            self, *, student_logits: torch.Tensor, labels: torch.Tensor, teacher_logits=None
        ) -> ObjectiveTerms:
            hard = student_logits[:, 0] * 0 + labels.to(torch.float32) + 1
            return ObjectiveTerms(hard, torch.zeros_like(hard), torch.zeros_like(hard))

    trainer = make_trainer(tmp_path)
    trainer.attack = IdentityAttack()
    trainer.objective = LabelObjective()
    for parameter in trainer.model.parameters():
        parameter.data.zero_()
    batch = IndexedBatch(
        images=torch.rand(4, 3, 4, 4),
        labels=torch.tensor([0, 1, 0, 1]),
        sample_ids=torch.tensor([0, 1, 0, 1]),
        state_update_mask=torch.tensor([True, True, False, False]),
        multiplicity=torch.tensor([2, 2, 2, 2]),
    )
    metrics = trainer.train_epoch([batch])
    assert {key: metrics[key] for key in ("loss", "clean_accuracy", "robust_accuracy")} == {
        "loss": pytest.approx(1.5),
        "clean_accuracy": pytest.approx(0.5),
        "robust_accuracy": pytest.approx(0.5),
    }
    assert metrics["valid_examples"] == 2.0
    assert metrics["teacher_clean_forward_calls"] == 0.0
    assert metrics["cuda_peak_allocated_bytes"] == 0.0
    assert metrics["seconds"] > 0.0
    assert metrics["images_per_second"] == pytest.approx(2.0 / metrics["seconds"])


def test_validation_attack_preserves_batchnorm_state_and_modes(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Flatten(),
        nn.Linear(3 * 4 * 4, 3),
    )
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(
            AttackConfig(
                epsilon="1/255",
                step_size="1/255",
                steps=1,
                random_start=False,
                student_mode="train",
            )
        ),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255",
                step_size="1/255",
                steps=1,
                random_start=False,
                student_mode="eval",
                teacher_mode="eval",
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="b" * 64,
        seed=5,
    )
    trainer.model.train()
    before_state = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    before_modes = {name: module.training for name, module in trainer.model.named_modules()}
    batch = IndexedBatch(
        images=torch.rand(2, 3, 4, 4),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([0, 1]),
        state_update_mask=torch.tensor([True, True]),
        multiplicity=torch.ones(2, dtype=torch.long),
    )

    trainer.validate_epoch([batch])

    assert before_modes == {name: module.training for name, module in trainer.model.named_modules()}
    for name, expected in before_state.items():
        assert torch.equal(expected, trainer.model.state_dict()[name]), name


def test_validation_random_stream_advances_repeats_and_separates_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingAttack:
        def __init__(self) -> None:
            self.records: list[torch.Tensor] = []

        def generate(self, request):
            draw = torch.rand(request.inputs.shape, generator=request.generator, device=request.inputs.device)
            self.records.append(draw.detach().clone())
            return AttackResult(request.inputs, torch.zeros_like(request.inputs), (), 0.0)

    batch = IndexedBatch(
        images=torch.rand(2, 3, 4, 4),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([0, 1]),
        state_update_mask=torch.tensor([True, True]),
        multiplicity=torch.ones(2, dtype=torch.long),
    )

    def sequence(rank: int) -> list[torch.Tensor]:
        monkeypatch.setattr("ard.engine.trainer.get_rank", lambda: rank)
        trainer = make_trainer(tmp_path / f"rank-{rank}")
        recorder = RecordingAttack()
        trainer.selection_attack = recorder
        trainer.global_step = 7
        trainer.validate_epoch([batch, batch])
        return recorder.records

    rank_zero = sequence(0)
    repeated_rank_zero = sequence(0)
    rank_one = sequence(1)
    assert not torch.equal(rank_zero[0], rank_zero[1])
    assert all(torch.equal(left, right) for left, right in zip(rank_zero, repeated_rank_zero, strict=True))
    assert not torch.equal(rank_zero[0], rank_one[0])
