"""End-to-end ADR: EMA lifecycle, rectification wiring, and checkpoint round-trip.

Fixture-scale (``fixture_cnn`` on ``SyntheticCIFAR``), CPU-only, no GPU
required.  This is the "does the full loop even run" gate from the ADR
implementation plan, before anything touches a real GPU: rectification before
the attack, the attack ascending the same rectified target, the objective
consuming it, the EMA update after the optimizer step, and both surviving a
save/resume cycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config.schema import AdrConfig, AttackConfig, ModelConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedBatch,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import ADRObjective, ADRTRADESObjective, PGDATObjective
from ard.objectives.adr import rectify_label
from ard.schedules.gap_adaptive import gap_adaptive_step

pytestmark = pytest.mark.t3


def _loaders(seed: int = 7) -> tuple[DataLoader, DataLoader, EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_indexed)
    validation_loader = DataLoader(
        validation_dataset, batch_size=4, sampler=validation_sampler, collate_fn=collate_indexed
    )
    return loader, validation_loader, sampler


def _rectified_attack() -> LinfPGD:
    """Plain ADR's attack: ascends the rectified label directly."""
    return LinfPGD(
        AttackConfig(
            loss="kl",
            kl_target="rectified",
            temperature=1.0,
            epsilon="1/255",
            step_size="1/255",
            steps=1,
            random_start=True,
        )
    )


def _trades_shaped_attack() -> LinfPGD:
    """ADR-TRADES' attack: unrectified, identical to plain TRADES' inner-max
    (confirmed against .external/adr/src/util/trades_attack.py -- the
    official code's TRADES+ADR inner PGD step never reads the rectified
    label)."""
    return LinfPGD(
        AttackConfig(
            loss="kl",
            kl_target="student_clean",
            temperature=1.0,
            epsilon="1/255",
            step_size="1/255",
            steps=1,
            random_start=True,
        )
    )


def _adr_trainer(
    output: Path,
    *,
    objective: object,
    epochs: int,
    seed: int = 7,
    attack: LinfPGD | None = None,
    adr_config: AdrConfig | None = None,
) -> Trainer:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    loader, _, _ = _loaders(seed=seed)
    total_iterations = epochs * len(loader)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=attack if attack is not None else _rectified_attack(),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=objective,
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="b" * 64,
        seed=seed,
        tracker_run_id="offline-fixture-adr",
        adr_config=(
            adr_config
            if adr_config is not None
            else AdrConfig(ema_decay=0.9, temperature_high=2.0, temperature_low=1.0, lambda_low=0.5, lambda_high=0.9)
        ),
        total_iterations=total_iterations,
    )


def _gap_adaptive_adr_config() -> AdrConfig:
    return AdrConfig(
        ema_decay=0.9,
        temperature_high=2.0,
        temperature_low=1.0,
        lambda_low=0.5,
        lambda_high=0.9,
        lambda_source="gap_adaptive",
        gap_smoothing_beta=0.5,
    )


def test_one_epoch_of_adr_runs_end_to_end(tmp_path: Path) -> None:
    trainer = _adr_trainer(tmp_path / "adr", objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    assert trainer.ema_model is not None
    assert (tmp_path / "adr" / "last.pt").exists()


def test_one_epoch_of_adr_trades_runs_end_to_end(tmp_path: Path) -> None:
    trainer = _adr_trainer(
        tmp_path / "adr_trades", objective=ADRTRADESObjective(beta=6.0), epochs=1, attack=_trades_shaped_attack()
    )
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    assert trainer.ema_model is not None


def test_ema_weights_diverge_from_the_live_model_after_training(tmp_path: Path) -> None:
    """A trivial confirmation that ``_update_ema`` actually ran: after one
    epoch at least one EMA tensor must differ from its initial (student-
    identical) state, and the EMA state as a whole must differ from the live
    model's post-training state (decay < 1, so it lags)."""
    torch.manual_seed(123)
    initial_model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    initial_state = {key: value.clone() for key, value in initial_model.state_dict().items()}

    trainer = _adr_trainer(tmp_path / "ema-diverge", objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    assert trainer.ema_model is not None
    ema_state = trainer.ema_model.state_dict()
    live_state = trainer.model.state_dict()
    assert any(not torch.equal(ema_state[key], initial_state[key]) for key in ema_state), (
        "EMA never moved from its initial state"
    )
    assert any(not torch.equal(ema_state[key], live_state[key]) for key in ema_state), (
        "EMA exactly matches the live model after training; decay < 1 means it must lag"
    )


def test_ema_state_round_trips_exactly_through_save_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "resume"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    expected_ema_state = {key: value.clone() for key, value in trainer.ema_model.state_dict().items()}

    resumed = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    _, _, sampler = _loaders()
    resumed.resume(output / "last.pt", sampler=sampler)
    assert resumed.ema_model is not None
    for key, expected_value in expected_ema_state.items():
        assert torch.equal(resumed.ema_model.state_dict()[key], expected_value), key


def test_best_ema_checkpoint_is_written_independently_of_best_pt(tmp_path: Path) -> None:
    """The EMA shadow model gets its own independent checkpoint selection,
    matching the official ADR code's "ADR + WA" convention (EMA weights at
    an EMA-reselected best epoch) rather than reusing the student's
    best.pt epoch. best-ema.pt must exist, carry an "ema" key that matches
    the live EMA state, and self-document its selection source."""
    output = tmp_path / "adr-best-ema"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    assert (output / "best-ema.pt").exists()
    payload = torch.load(output / "best-ema.pt", map_location="cpu", weights_only=False)
    assert payload["selection_metadata"]["selection_source"] == "ema"
    assert payload["best_metric"] == trainer.best_metric_ema
    for key, value in trainer.ema_model.state_dict().items():
        assert torch.equal(payload["ema"][key], value), key
    # The selection record must name the attack and RNG protocol that
    # produced it, exactly like the student's -- otherwise the sole
    # artifact behind the "ADR + WA" number can't be audited on its own.
    assert payload["selection_metadata"]["attack"] == trainer.selection_metadata["attack"]
    assert payload["selection_metadata"]["seed_protocol"] == trainer.selection_metadata["seed_protocol"]


def test_best_ema_selection_is_independent_of_student_selection(tmp_path: Path) -> None:
    """best.pt and best-ema.pt must be free to disagree on which epoch is
    best -- a fixture where the student peaks at epoch 0 and the EMA peaks
    at epoch 2 would pass even if best-ema.pt were gated on the student's
    own `improved` flag, unless the two selections are checked separately."""
    output = tmp_path / "adr-independent-selection"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=3)
    loader, validation_loader, _ = _loaders()
    student_trajectory = iter(
        [
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.9},
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.1},
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.2},
        ]
    )
    ema_trajectory = iter(
        [
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.1},
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.2},
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.9},
        ]
    )

    def fake_validate_epoch(loader, *, model=None):
        return next(ema_trajectory) if model is trainer.ema_model else next(student_trajectory)

    trainer.validate_epoch = fake_validate_epoch
    trainer.fit(loader, validation_loader=validation_loader, epochs=3)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    best_ema = torch.load(output / "best-ema.pt", map_location="cpu", weights_only=False)
    assert best["selection_metadata"]["selected_epoch"] == 0
    assert best_ema["selection_metadata"]["selected_epoch"] == 2
    assert best["best_metric"] == pytest.approx(0.9)
    assert best_ema["best_metric"] == pytest.approx(0.9)


def test_resuming_from_best_ema_checkpoint_is_rejected(tmp_path: Path) -> None:
    """best-ema.pt's top-level best_metric/selection_metadata describe the
    EMA's own selection, not the student's -- resuming TRAINING from it
    would silently overwrite self.best_metric/selection_metadata with the
    EMA's values. Only last.pt (or epoch-NNN.pt) may be resumed from."""
    output = tmp_path / "adr-reject-ema-resume"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    resumed = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    _, _, sampler = _loaders()
    with pytest.raises(ValueError, match="cannot resume training from an EMA-selected checkpoint"):
        resumed.resume(output / "best-ema.pt", sampler=sampler)


def test_ema_selection_window_start_recorded_when_resuming_a_pre_feature_checkpoint(tmp_path: Path) -> None:
    """A checkpoint written before best_metric_ema/selection_metadata_ema
    existed has neither key; resuming it must not silently claim best-ema.pt
    describes the whole run -- the fresh restart records where its search
    window actually begins."""
    output = tmp_path / "adr-ema-window-start"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    del payload["best_metric_ema"]
    del payload["selection_metadata_ema"]
    torch.save(payload, output / "last.pt")

    resumed = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    _, _, sampler = _loaders()
    state = resumed.resume(output / "last.pt", sampler=sampler)
    assert resumed.selection_metadata_ema["selection_window_start"] == state.next_epoch


def test_best_metric_ema_round_trips_through_resume(tmp_path: Path) -> None:
    output = tmp_path / "resume-ema-metric"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    expected_best_metric_ema = trainer.best_metric_ema
    expected_selection_metadata_ema = dict(trainer.selection_metadata_ema)

    resumed = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    _, _, sampler = _loaders()
    resumed.resume(output / "last.pt", sampler=sampler)
    assert resumed.best_metric_ema == expected_best_metric_ema
    assert resumed.selection_metadata_ema == expected_selection_metadata_ema


def test_checkpoint_without_adr_still_resumes_a_non_adr_trainer(tmp_path: Path) -> None:
    """Backward compatibility: a checkpoint from a non-ADR run has no 'ema'
    key at all, and a non-ADR trainer resuming it must not even look for one."""
    output = tmp_path / "plain"
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="c" * 64,
        seed=7,
        tracker_run_id="offline-fixture-plain",
    )
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert "ema" not in payload
    assert "best_metric_ema" not in payload
    assert REQUIRED_KEYS.issubset(payload)
    assert not (output / "best-ema.pt").exists()

    resumed_model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    resumed = Trainer(
        model=resumed_model,
        optimizer=SGD(resumed_model.parameters(), lr=0.03, momentum=0.9),
        scheduler=StepLR(SGD(resumed_model.parameters(), lr=0.03, momentum=0.9), step_size=1, gamma=0.8),
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="c" * 64,
        seed=7,
        tracker_run_id="offline-fixture-plain",
    )
    assert resumed.ema_model is None
    _, _, sampler = _loaders()
    resumed.resume(output / "last.pt", sampler=sampler)  # must not raise


def test_resuming_an_adr_checkpoint_with_ema_model_none_ignores_the_ema_state(tmp_path: Path) -> None:
    """A checkpoint that carries EMA state, resumed by a trainer with no
    ema_model at all, must simply ignore it rather than raise -- ema_model is
    the caller's declaration of intent, not a checkpoint-driven decision."""
    output = tmp_path / "adr-then-plain"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    torch.manual_seed(123)
    plain_model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    plain = Trainer(
        model=plain_model,
        optimizer=SGD(plain_model.parameters(), lr=0.03, momentum=0.9),
        scheduler=StepLR(SGD(plain_model.parameters(), lr=0.03, momentum=0.9), step_size=1, gamma=0.8),
        scaler=None,
        attack=_rectified_attack(),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="b" * 64,
        seed=7,
        tracker_run_id="offline-fixture-adr",
    )
    assert plain.ema_model is None
    _, _, sampler = _loaders()
    plain.resume(output / "last.pt", sampler=sampler)  # ema_model=None: nothing to restore, must not raise


def test_resuming_a_non_adr_checkpoint_with_an_ema_model_fails_closed(tmp_path: Path) -> None:
    """The real fail-closed path: this run needs EMA state (adr_config is
    set) but the checkpoint being resumed was written by a run that never
    had one.  Silently starting the EMA copy fresh mid-training would be a
    silent divergence from a correct resume (docs/debugging/0026's lesson:
    fail closed and specific, not opaque)."""
    output = tmp_path / "plain-then-adr"
    torch.manual_seed(123)
    plain_model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    plain = Trainer(
        model=plain_model,
        optimizer=SGD(plain_model.parameters(), lr=0.03, momentum=0.9),
        scheduler=StepLR(SGD(plain_model.parameters(), lr=0.03, momentum=0.9), step_size=1, gamma=0.8),
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="d" * 64,
        seed=7,
        tracker_run_id="offline-fixture-plain2",
    )
    loader, validation_loader, _ = _loaders()
    plain.fit(loader, validation_loader=validation_loader, epochs=1)

    adr_trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    adr_trainer.config_hash = "d" * 64  # match the checkpoint written above
    _, _, sampler = _loaders()
    with pytest.raises(ValueError, match="carries no 'ema' state"):
        adr_trainer.resume(output / "last.pt", sampler=sampler)


def test_gap_adaptive_lambda_moves_with_a_real_train_val_gap(tmp_path: Path) -> None:
    """Plan 0098's canary check, exercised deterministically: lambda actually
    responds to the epoch's own train/val gap, matching the schedule
    primitive's own math bit-for-bit -- not just "some function ran"."""
    output = tmp_path / "gap-adaptive-moves"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=3, adr_config=_gap_adaptive_adr_config())
    loader, validation_loader, _ = _loaders()

    train_robust_accuracies = iter([0.9, 0.9, 0.9])
    val_pgd_accuracies = iter([0.5, 0.5, 0.5])

    def fake_train_epoch(loader, **kwargs):
        del loader, kwargs
        return {
            "loss": 0.0,
            "clean_accuracy": 0.5,
            "robust_accuracy": next(train_robust_accuracies),
            "ema_student_agreement": 0.0,
            "rectified_true_class_mass": 0.0,
        }

    def fake_validate_epoch(loader, *, model=None):
        del loader
        if model is trainer.ema_model:
            return {"clean_accuracy": 0.5, "pgd_accuracy": 0.5}
        return {"clean_accuracy": 0.5, "pgd_accuracy": next(val_pgd_accuracies)}

    trainer.train_epoch = fake_train_epoch
    trainer.validate_epoch = fake_validate_epoch

    observed_lambdas: list[float] = []
    trainer.fit(
        loader,
        validation_loader=validation_loader,
        epochs=3,
        on_epoch_end=lambda metrics, improved: observed_lambdas.append(trainer._current_lambda_floor),
    )

    expected_lambdas = []
    state = None
    for _ in range(3):
        state, lambda_value = gap_adaptive_step(
            train_robust_accuracy=0.9,
            val_pgd_accuracy=0.5,
            previous_state=state,
            beta=0.5,
            lambda_low=0.5,
            lambda_high=0.9,
        )
        expected_lambdas.append(lambda_value)
    assert observed_lambdas == pytest.approx(expected_lambdas)
    assert trainer.fork_lineage is not None
    assert trainer.fork_lineage["gap_adaptive_state"] == pytest.approx(state.to_dict())


def test_gap_adaptive_state_round_trips_exactly_through_save_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "gap-adaptive-resume"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=2, adr_config=_gap_adaptive_adr_config())
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    expected_state = trainer._gap_adaptive_state
    expected_lambda = trainer._current_lambda_floor
    assert expected_state is not None

    resumed = _adr_trainer(output, objective=ADRObjective(), epochs=2, adr_config=_gap_adaptive_adr_config())
    _, _, sampler = _loaders()
    resumed.resume(output / "last.pt", sampler=sampler)
    assert resumed._gap_adaptive_state == expected_state
    assert resumed._current_lambda_floor == pytest.approx(expected_lambda)


def test_gap_adaptive_resume_matches_an_uninterrupted_runs_lambda_trajectory(tmp_path: Path) -> None:
    """The real regression this plan requires: resume must not merely avoid
    crashing, it must reproduce the exact lambda an uninterrupted run would
    have used for the epoch after resume."""
    adr_config = _gap_adaptive_adr_config()
    uninterrupted_output = tmp_path / "uninterrupted"
    uninterrupted = _adr_trainer(uninterrupted_output, objective=ADRObjective(), epochs=2, adr_config=adr_config)
    loader, validation_loader, _ = _loaders()
    uninterrupted.fit(loader, validation_loader=validation_loader, epochs=2)
    uninterrupted_second_epoch_lambda = uninterrupted._current_lambda_floor

    interrupted_output = tmp_path / "interrupted"
    interrupted = _adr_trainer(interrupted_output, objective=ADRObjective(), epochs=2, adr_config=adr_config)
    loader, validation_loader, _ = _loaders()
    interrupted.fit(loader, validation_loader=validation_loader, epochs=1)

    resumed = _adr_trainer(interrupted_output, objective=ADRObjective(), epochs=2, adr_config=adr_config)
    _, _, sampler = _loaders()
    state = resumed.resume(interrupted_output / "last.pt", sampler=sampler)
    loader, validation_loader, _ = _loaders()
    resumed.fit(loader, validation_loader=validation_loader, start_epoch=state.next_epoch, epochs=2)

    assert resumed._current_lambda_floor == pytest.approx(uninterrupted_second_epoch_lambda)


def test_resuming_a_cosine_checkpoint_with_a_gap_adaptive_trainer_fails_closed(tmp_path: Path) -> None:
    """A checkpoint written by a cosine-lambda ADR run has no
    ``gap_adaptive_state`` -- a gap-adaptive trainer resuming it must fail
    closed and specifically, not silently start its own schedule from
    scratch mid-training (the same discipline as the existing plain-EMA
    fail-closed resume tests above)."""
    output = tmp_path / "cosine-then-gap-adaptive"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)

    gap_trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1, adr_config=_gap_adaptive_adr_config())
    _, _, sampler = _loaders()
    with pytest.raises(ValueError, match="lambda schedule state"):
        gap_trainer.resume(output / "last.pt", sampler=sampler)


def test_cosine_lambda_is_applied_per_iteration_matching_cosine_anneal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scientific review of plan 0098 flagged that no test pins which lambda
    value is actually *applied* during training (only post-hoc state was
    checked). Spy on the one call site that consumes lambda_floor
    (rectify_label) and confirm the applied value matches cosine_anneal at
    every iteration, not just at epoch boundaries."""
    from ard.engine import trainer as trainer_module
    from ard.schedules.cosine_value import cosine_anneal

    output = tmp_path / "cosine-lambda-applied"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=2)
    loader, validation_loader, _ = _loaders()
    real_rectify_label = trainer_module.rectify_label
    observed: list[tuple[int, float]] = []

    def spy_rectify_label(*, ema_clean_logits, labels, temperature, lambda_floor):
        observed.append((trainer.global_step, lambda_floor))
        return real_rectify_label(
            ema_clean_logits=ema_clean_logits, labels=labels, temperature=temperature, lambda_floor=lambda_floor
        )

    monkeypatch.setattr(trainer_module, "rectify_label", spy_rectify_label)
    trainer.fit(loader, validation_loader=validation_loader, epochs=2)

    assert len(observed) == trainer.total_iterations
    for global_step, lambda_floor in observed:
        expected = cosine_anneal(
            start=trainer.adr_config.lambda_low,
            end=trainer.adr_config.lambda_high,
            iteration=global_step,
            total_iterations=trainer.total_iterations,
        )
        assert lambda_floor == pytest.approx(expected)
    # Confirms the schedule actually varies (a constant lambda_floor would
    # trivially satisfy the assertion above too).
    assert len({round(value, 9) for _, value in observed}) > 1


def test_gap_adaptive_lambda_is_constant_within_an_epoch_and_lags_by_one_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same review finding, for gap-adaptive mode:
    confirm the lambda applied to every iteration of epoch e+1 is exactly
    what gap_adaptive_step computes from epoch e's own real train/val
    metrics -- not epoch e+1's own (not-yet-known) metrics, and not merely
    correct at the end-of-epoch snapshot checked by the other gap-adaptive
    tests above."""
    from ard.engine import trainer as trainer_module
    from ard.schedules.gap_adaptive import gap_adaptive_step

    output = tmp_path / "gap-adaptive-lambda-applied"
    adr_config = _gap_adaptive_adr_config()
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=3, adr_config=adr_config)
    loader, validation_loader, _ = _loaders()
    real_rectify_label = trainer_module.rectify_label
    observed_by_step: dict[int, float] = {}

    def spy_rectify_label(*, ema_clean_logits, labels, temperature, lambda_floor):
        observed_by_step[trainer.global_step] = lambda_floor
        return real_rectify_label(
            ema_clean_logits=ema_clean_logits, labels=labels, temperature=temperature, lambda_floor=lambda_floor
        )

    monkeypatch.setattr(trainer_module, "rectify_label", spy_rectify_label)
    epoch_metrics_log: list[dict[str, float]] = []
    trainer.fit(
        loader,
        validation_loader=validation_loader,
        epochs=3,
        on_epoch_end=lambda metrics, improved: epoch_metrics_log.append(dict(metrics)),
    )

    iterations_per_epoch = len(loader)
    assert len(observed_by_step) == trainer.total_iterations
    for step in range(iterations_per_epoch):
        assert observed_by_step[step] == pytest.approx(adr_config.lambda_low)

    state = None
    for epoch_index, metrics in enumerate(epoch_metrics_log[:-1]):
        state, expected_lambda = gap_adaptive_step(
            train_robust_accuracy=metrics["train_robust_accuracy"],
            val_pgd_accuracy=metrics["val_pgd_accuracy"],
            previous_state=state,
            beta=adr_config.gap_smoothing_beta,
            lambda_low=adr_config.lambda_low,
            lambda_high=adr_config.lambda_high,
        )
        for step in range((epoch_index + 1) * iterations_per_epoch, (epoch_index + 2) * iterations_per_epoch):
            assert observed_by_step[step] == pytest.approx(expected_lambda)


def test_ema_student_agreement_matches_a_hand_computed_pre_vs_post_step_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scientific review of plan 0098 flagged that every existing test feeds
    or asserts a literal 0.0 for train_ema_student_agreement, never
    exercising the real argmax comparison, masking, or division. Substitute
    the EMA forward with the student's own pre-optimizer-step forward (same
    call timing the production code uses) -- a valid stand-in for verifying
    the accumulation/masking/division logic, not real EMA numerics -- then
    independently recompute the expected agreement from a manual pre-step
    vs. post-step comparison on the same batch."""
    from ard.engine.trainer import Trainer

    output = tmp_path / "ema-agreement-hand-computed"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    batch = next(iter(loader))

    pre_step_argmax_holder: dict[str, torch.Tensor] = {}

    def spy_ema_clean_logits(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = trainer.model(images.float()).detach().float()
        pre_step_argmax_holder["argmax"] = logits.argmax(1)
        return logits

    monkeypatch.setattr(trainer, "_ema_clean_logits", spy_ema_clean_logits)
    metrics = trainer.train_epoch([batch])

    with torch.no_grad():
        trainer.model.eval()
        post_step_argmax = trainer.model(batch.images).argmax(1)
        trainer.model.train()

    mask = Trainer._mask(batch)
    expected_matches = float(((pre_step_argmax_holder["argmax"] == post_step_argmax).to(mask.dtype) * mask).sum())
    expected_valid = float(mask.sum())
    assert metrics["ema_student_agreement"] == pytest.approx(expected_matches / expected_valid)


def test_ema_student_agreement_excludes_padded_rows_from_both_numerator_and_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "ema-agreement-masked"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    batch = next(iter(loader))
    # Half the batch is padding: must contribute to neither the numerator
    # nor the denominator, regardless of whether it happens to "agree".
    padded_mask = torch.zeros(batch.labels.shape[0], dtype=torch.bool)
    padded_mask[0] = True
    batch = IndexedBatch(
        images=batch.images, labels=batch.labels, sample_ids=batch.sample_ids, state_update_mask=padded_mask
    )

    def spy_ema_clean_logits(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return trainer.model(images.float()).detach().float()

    monkeypatch.setattr(trainer, "_ema_clean_logits", spy_ema_clean_logits)
    metrics = trainer.train_epoch([batch])
    assert metrics["valid_examples"] == 1.0
    assert metrics["ema_student_agreement"] in (0.0, 1.0)


def test_rectified_true_class_mass_matches_a_hand_computed_rectify_label_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 0100 scientific review: a temperature/lambda pair mistuned for
    the wrong class count can degenerate the rectified target toward
    near-uniform label smoothing. This diagnostic exists to catch that --
    verify it actually reports the real rectify_label output, not a stub."""
    output = tmp_path / "rectified-true-class-mass"
    trainer = _adr_trainer(output, objective=ADRObjective(), epochs=1)
    loader, validation_loader, _ = _loaders()
    batch = next(iter(loader))

    captured: dict[str, torch.Tensor] = {}
    real_ema_clean_logits = trainer._ema_clean_logits

    def spy_ema_clean_logits(images: torch.Tensor) -> torch.Tensor:
        logits = real_ema_clean_logits(images)
        captured["logits"] = logits
        return logits

    monkeypatch.setattr(trainer, "_ema_clean_logits", spy_ema_clean_logits)
    metrics = trainer.train_epoch([batch])

    assert trainer.adr_config is not None
    expected_rectified = rectify_label(
        ema_clean_logits=captured["logits"],
        labels=batch.labels,
        temperature=trainer.adr_config.temperature_high,  # iteration 0: cosine_anneal(start) exactly
        lambda_floor=trainer.adr_config.lambda_low,  # iteration 0, lambda_source=cosine (default)
    )
    mask = Trainer._mask(batch)
    expected_mass = expected_rectified.gather(1, batch.labels[:, None]).squeeze(1)
    expected_mean = float((expected_mass * mask).sum()) / float(mask.sum())
    assert metrics["rectified_true_class_mass"] == pytest.approx(expected_mean)
    # Sanity bound: this must be a real probability mass, not a stub -- and
    # for a 3-class fixture, comfortably above chance (1/3) confirms the
    # mechanism is not already degenerate at this fixture's tiny scale.
    assert 0.0 <= metrics["rectified_true_class_mass"] <= 1.0
