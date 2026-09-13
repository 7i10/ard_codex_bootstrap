"""Plan 0101: epsilon warmup wiring, end to end at fixture scale.

Fixture-scale (``fixture_cnn`` on ``SyntheticCIFAR``), CPU-only, no GPU
required -- mirrors ``tests/integration/test_adr_trainer.py``'s shape.
Proves what the schedule-function unit tests
(``tests/unit/test_epsilon_warmup_schedule.py``) cannot: that the trainer
actually threads the ramped values into the *training* attack call only,
that Option 1's step:epsilon coupling never trips ``LinfPGD.generate``'s
own ``step_size <= epsilon`` guard at any ramp point, and that the
threat model returns to exactly the configured target once the ramp ends
(scientific review P2-2, P2-3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
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

pytestmark = pytest.mark.t3

TARGET_EPSILON = 4 / 255
TARGET_STEP_SIZE = 2.6667 / 255  # a real, non-2/3-simplified ratio, so the guard has something real to check


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


def _recording_attack(
    real_attack: LinfPGD, sink: list[tuple[AttackRequest, AttackResult]]
) -> LinfPGD:
    """Wrap a real LinfPGD so every request/result pair it handles is
    captured, while the real (guard-checking) ``generate`` still runs -- a
    mock would hide exactly the guard violation this test needs to prove
    doesn't happen."""
    original_generate = real_attack.generate

    def recording_generate(request: AttackRequest) -> AttackResult:
        result = original_generate(request)
        sink.append((request, result))
        return result

    real_attack.generate = recording_generate  # type: ignore[method-assign]
    return real_attack


def _pgd_at_trainer(
    output: Path, *, epsilon_warmup_epochs: int | None, seed: int = 7
) -> tuple[Trainer, list[tuple[AttackRequest, AttackResult]], list[tuple[AttackRequest, AttackResult]]]:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    training_attack = LinfPGD(AttackConfig(epsilon="4/255", step_size="2.6667/255", steps=1, random_start=True))
    selection_attack = LinfPGD(
        AttackConfig(
            epsilon="4/255", step_size="2.6667/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
        )
    )
    training_calls: list[tuple[AttackRequest, AttackResult]] = []
    selection_calls: list[tuple[AttackRequest, AttackResult]] = []
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=_recording_attack(training_attack, training_calls),
        selection_attack=_recording_attack(selection_attack, selection_calls),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="c" * 64,
        seed=seed,
        tracker_run_id="offline-fixture-epsilon-warmup",
        epsilon_warmup_epochs=epsilon_warmup_epochs,
    )
    return trainer, training_calls, selection_calls


def test_epsilon_warmup_epoch_zero_uses_the_first_ramp_fraction_exactly(tmp_path: Path) -> None:
    warmup_epochs = 5
    trainer, training_calls, selection_calls = _pgd_at_trainer(tmp_path / "warmup", epsilon_warmup_epochs=warmup_epochs)
    loader, validation_loader = _loaders()
    metrics = trainer.fit(loader, validation_loader=validation_loader, epochs=1)[0]

    expected_epsilon = (1 / warmup_epochs) * TARGET_EPSILON  # epoch 0 -> (0+1)/warmup_epochs
    expected_step = (TARGET_STEP_SIZE / TARGET_EPSILON) * expected_epsilon

    assert training_calls, "the training attack must have been invoked"
    for request, result in training_calls:
        assert request.epsilon_override is not None
        assert request.step_size_override is not None
        assert torch.allclose(request.epsilon_override, torch.full_like(request.epsilon_override, expected_epsilon))
        assert torch.allclose(request.step_size_override, torch.full_like(request.step_size_override, expected_step))
        # The guard LinfPGD.generate itself enforces -- proven here to
        # actually hold for Option 1's coupling, not just asserted by
        # construction.
        assert bool((request.step_size_override <= request.epsilon_override).all())
        # The realized perturbation actually stayed inside the ramped ball,
        # not the full target ball (scientific review P2-2).
        assert result.max_abs_delta <= expected_epsilon + 1e-6

    assert selection_calls, "the selection attack must have been invoked during validation"
    for request, _ in selection_calls:
        assert request.epsilon_override is None
        assert request.step_size_override is None

    # The realized budget is reported, self-describing regardless of config
    # (scientific review P2-1) -- robust_accuracy this epoch was measured
    # under this reduced threat, not the configured 4/255.
    assert metrics["train_attack_epsilon"] == pytest.approx(expected_epsilon)
    assert metrics["train_attack_step_size"] == pytest.approx(expected_step)


def test_epsilon_warmup_mid_ramp_keeps_step_at_or_below_epsilon(tmp_path: Path) -> None:
    """Same proof one epoch into a longer warmup, where both values are
    strictly between the first ramp fraction and the target -- the case
    most likely to trip the guard if the coupling were implemented wrong."""
    warmup_epochs = 4
    trainer, training_calls, _ = _pgd_at_trainer(tmp_path / "warmup-mid", epsilon_warmup_epochs=warmup_epochs)
    loader, validation_loader = _loaders()
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=2)

    per_epoch_batches = len(training_calls) // 2
    epoch_one_calls = training_calls[per_epoch_batches:]
    expected_epsilon = (2 / warmup_epochs) * TARGET_EPSILON  # epoch 1 -> (1+1)/warmup_epochs
    for request, result in epoch_one_calls:
        assert request.epsilon_override is not None and request.step_size_override is not None
        assert torch.allclose(request.epsilon_override, torch.full_like(request.epsilon_override, expected_epsilon))
        assert bool((request.step_size_override <= request.epsilon_override).all())
        assert result.max_abs_delta <= expected_epsilon + 1e-6
    assert history[1]["train_attack_epsilon"] == pytest.approx(expected_epsilon)


def test_epsilon_warmup_returns_to_the_exact_target_after_the_ramp_ends(tmp_path: Path) -> None:
    """The whole "temporarily reduced threat" claim rests on this: an epoch
    at or past warmup_epochs must be indistinguishable from a config with no
    warmup at all (scientific review P2-2's highest-priority missing case)."""
    warmup_epochs = 1
    trainer, training_calls, _ = _pgd_at_trainer(tmp_path / "warmup-then-full", epsilon_warmup_epochs=warmup_epochs)
    loader, validation_loader = _loaders()
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=2)

    per_epoch_batches = len(training_calls) // 2
    epoch_one_calls = training_calls[per_epoch_batches:]  # epoch index 1, >= warmup_epochs=1
    for request, result in epoch_one_calls:
        # Past the ramp, the elif branch that sets an explicit override is
        # simply never taken -- None falls through to LinfPGD's own
        # AttackConfig-resolved budget, which is numerically identical to
        # the target. Either representation is correct; what matters is
        # the *realized* perturbation, proven directly below.
        if request.epsilon_override is not None:
            assert torch.allclose(request.epsilon_override, torch.full_like(request.epsilon_override, TARGET_EPSILON))
        if request.step_size_override is not None:
            assert torch.allclose(request.step_size_override, torch.full_like(request.step_size_override, TARGET_STEP_SIZE))
        assert result.max_abs_delta <= TARGET_EPSILON + 1e-6
    assert history[1]["train_attack_epsilon"] == pytest.approx(TARGET_EPSILON)
    assert history[1]["train_attack_step_size"] == pytest.approx(TARGET_STEP_SIZE)


def test_epsilon_warmup_disabled_never_overrides_the_training_attack(tmp_path: Path) -> None:
    """Default (None) reproduces today's exact behavior for every existing config."""
    trainer, training_calls, _ = _pgd_at_trainer(tmp_path / "no-warmup", epsilon_warmup_epochs=None)
    loader, validation_loader = _loaders()
    metrics = trainer.fit(loader, validation_loader=validation_loader, epochs=1)[0]

    assert training_calls
    for request, _ in training_calls:
        assert request.epsilon_override is None
        assert request.step_size_override is None
    # Still reported (always-on telemetry, scientific review P2-1), and
    # equal to the plain configured budget when no warmup is active.
    assert metrics["train_attack_epsilon"] == pytest.approx(TARGET_EPSILON)
    assert metrics["train_attack_step_size"] == pytest.approx(TARGET_STEP_SIZE)


def test_epsilon_warmup_and_mixed_selected_attack_budget_are_rejected_together(tmp_path: Path) -> None:
    """Scientific review P2-3: the exclusivity guard exists and actually fires."""
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    attack = LinfPGD(AttackConfig(epsilon="4/255", step_size="2.6667/255", steps=1, random_start=True))
    selection_attack = LinfPGD(
        AttackConfig(
            epsilon="4/255", step_size="2.6667/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
        )
    )
    with pytest.raises(ValueError, match="not specified together"):
        Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            attack=attack,
            selection_attack=selection_attack,
            objective=PGDATObjective(),
            device=torch.device("cpu"),
            output_dir=tmp_path / "rejected",
            config_hash="d" * 64,
            seed=7,
            tracker_run_id="offline-fixture-mutual-exclusion",
            epsilon_warmup_epochs=5,
            selected_attack_epsilon=8 / 255,
            selected_attack_step_size=4 / 255,
        )
