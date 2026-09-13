"""Plan 0101: epsilon warmup wiring, end to end at fixture scale.

Fixture-scale (``fixture_cnn`` on ``SyntheticCIFAR``), CPU-only, no GPU
required -- mirrors ``tests/integration/test_adr_trainer.py``'s shape.
Proves two things the schedule-function unit tests
(``tests/unit/test_epsilon_warmup_schedule.py``) cannot: that the trainer
actually threads the ramped values into the *training* attack call only,
and that Option 1's step:epsilon coupling never trips
``LinfPGD.generate``'s own ``step_size <= epsilon`` guard.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.attacks.base import AttackRequest, AttackResult
from ard.config.schema import AttackConfig, ModelConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import PGDATObjective

import pytest

pytestmark = pytest.mark.t3


def _loaders(seed: int = 7) -> tuple[DataLoader, DataLoader]:
    dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_indexed)
    validation_loader = DataLoader(
        validation_dataset, batch_size=4, sampler=validation_sampler, collate_fn=collate_indexed
    )
    return loader, validation_loader


def _recording_attack(real_attack: LinfPGD, sink: list[AttackRequest]) -> LinfPGD:
    """Wrap a real LinfPGD so every request it receives is captured, while
    the real (guard-checking) ``generate`` still runs -- a mock would hide
    exactly the guard violation this test needs to prove doesn't happen."""
    original_generate = real_attack.generate

    def recording_generate(request: AttackRequest) -> AttackResult:
        sink.append(request)
        return original_generate(request)

    real_attack.generate = recording_generate  # type: ignore[method-assign]
    return real_attack


def _pgd_at_trainer(
    output: Path, *, epsilon_warmup_epochs: int | None, seed: int = 7
) -> tuple[Trainer, list[AttackRequest], list[AttackRequest]]:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    # A real, non-trivial epsilon/step ratio (2/3, Salman et al. 2020's own
    # convention) so the guard has something real to check against.
    training_attack = LinfPGD(
        AttackConfig(epsilon="4/255", step_size="2.6667/255", steps=1, random_start=True)
    )
    selection_attack = LinfPGD(
        AttackConfig(
            epsilon="4/255", step_size="2.6667/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
        )
    )
    training_requests: list[AttackRequest] = []
    selection_requests: list[AttackRequest] = []
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=_recording_attack(training_attack, training_requests),
        selection_attack=_recording_attack(selection_attack, selection_requests),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="c" * 64,
        seed=seed,
        tracker_run_id="offline-fixture-epsilon-warmup",
        epsilon_warmup_epochs=epsilon_warmup_epochs,
    )
    return trainer, training_requests, selection_requests


def test_epsilon_warmup_overrides_only_the_training_attack_at_epoch_zero(tmp_path: Path) -> None:
    trainer, training_requests, selection_requests = _pgd_at_trainer(tmp_path / "warmup", epsilon_warmup_epochs=5)
    loader, validation_loader = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    assert training_requests, "the training attack must have been invoked"
    for request in training_requests:
        assert request.epsilon_override is not None
        assert request.step_size_override is not None
        # Epoch 0 of a 5-epoch warmup: the schedule's own starting value (0).
        assert torch.all(request.epsilon_override == 0)
        assert torch.all(request.step_size_override == 0)
        # The guard LinfPGD.generate itself enforces -- proven here to
        # actually hold for Option 1's coupling, not just asserted by
        # construction.
        assert bool((request.step_size_override <= request.epsilon_override).all())

    assert selection_requests, "the selection attack must have been invoked during validation"
    for request in selection_requests:
        assert request.epsilon_override is None
        assert request.step_size_override is None


def test_epsilon_warmup_mid_ramp_keeps_step_at_or_below_epsilon(tmp_path: Path) -> None:
    """Same proof one epoch into a longer warmup, where both values are
    strictly between 0 and the target -- the case most likely to trip the
    guard if the coupling were implemented wrong."""
    trainer, training_requests, _ = _pgd_at_trainer(tmp_path / "warmup-mid", epsilon_warmup_epochs=4)
    loader, validation_loader = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=2)

    epoch_zero = [r for r in training_requests[: len(training_requests) // 2]]
    epoch_one = [r for r in training_requests[len(training_requests) // 2 :]]
    assert epoch_zero and epoch_one
    for request in epoch_one:
        assert request.epsilon_override is not None and request.step_size_override is not None
        target = 4 / 255
        assert 0 < float(request.epsilon_override[0]) < target
        assert bool((request.step_size_override <= request.epsilon_override).all())


def test_epsilon_warmup_disabled_never_overrides_the_training_attack(tmp_path: Path) -> None:
    """Default (None) reproduces today's exact behavior for every existing config."""
    trainer, training_requests, _ = _pgd_at_trainer(tmp_path / "no-warmup", epsilon_warmup_epochs=None)
    loader, validation_loader = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    assert training_requests
    for request in training_requests:
        assert request.epsilon_override is None
        assert request.step_size_override is None
