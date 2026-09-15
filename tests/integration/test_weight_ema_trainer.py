"""Plan 0102 Workstream A: plain weight-EMA wiring, end to end at fixture
scale, mirroring ``tests/integration/test_adr_trainer.py``'s EMA tests but
for a plain ``pgd_at`` run with ``training.weight_ema_decay`` set instead of
``method.adr``.

Fixture-scale (``fixture_cnn`` on ``SyntheticCIFAR``), CPU-only, no GPU
required. Proves the trainer-level half of decoupling ``Trainer.ema_model``
from ADR specifically: it is method-agnostic already (gated on ``ema_model
is not None`` throughout ``_update_ema`` and ``ard.engine.checkpoint``), so
this only needed a second construction path and a second decay source, not
new EMA machinery.
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
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import PGDATObjective

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


def _pgd_at_attack() -> LinfPGD:
    return LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True))


def _pgd_at_trainer(
    output: Path, *, weight_ema_decay: float | None, seed: int = 7, adr_config: AdrConfig | None = None
) -> Trainer:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=_pgd_at_attack(),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="d" * 64,
        seed=seed,
        tracker_run_id="offline-fixture-weight-ema",
        weight_ema_decay=weight_ema_decay,
        adr_config=adr_config,
        total_iterations=(1 if adr_config is not None else None),
    )


def test_weight_ema_model_is_none_by_default(tmp_path: Path) -> None:
    """Every existing non-ADR config never mentions weight_ema_decay --
    confirms this plan's addition changes nothing for them."""
    trainer = _pgd_at_trainer(tmp_path / "no-ema", weight_ema_decay=None)
    assert trainer.ema_model is None


def test_weight_ema_and_adr_config_together_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="weight_ema_decay cannot be combined with adr_config"):
        _pgd_at_trainer(
            tmp_path / "conflict",
            weight_ema_decay=0.9,
            adr_config=AdrConfig(ema_decay=0.9, temperature_high=2.0, temperature_low=1.0, lambda_low=0.5, lambda_high=0.9),
        )


def test_weight_ema_weights_diverge_from_the_live_model_after_training(tmp_path: Path) -> None:
    torch.manual_seed(123)
    initial_model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    initial_state = {key: value.clone() for key, value in initial_model.state_dict().items()}

    trainer = _pgd_at_trainer(tmp_path / "ema-diverge", weight_ema_decay=0.9)
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


def test_weight_ema_state_round_trips_through_save_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "resume"
    trainer = _pgd_at_trainer(output, weight_ema_decay=0.9)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    assert trainer.ema_model is not None
    expected_ema_state = {key: value.clone() for key, value in trainer.ema_model.state_dict().items()}

    resumed = _pgd_at_trainer(output, weight_ema_decay=0.9)
    _, _, sampler = _loaders()
    resumed.resume(output / "last.pt", sampler=sampler)
    assert resumed.ema_model is not None
    for key, expected_value in expected_ema_state.items():
        assert torch.equal(resumed.ema_model.state_dict()[key], expected_value), key


def test_weight_ema_best_ema_checkpoint_is_written_independently_of_best_pt(tmp_path: Path) -> None:
    output = tmp_path / "best-ema"
    trainer = _pgd_at_trainer(output, weight_ema_decay=0.9)
    loader, validation_loader, _ = _loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    assert (output / "best-ema.pt").exists()
    payload = torch.load(output / "best-ema.pt", map_location="cpu", weights_only=False)
    assert payload["selection_metadata"]["selection_source"] == "ema"
    assert payload["best_metric"] == trainer.best_metric_ema
    assert trainer.ema_model is not None
    for key, value in trainer.ema_model.state_dict().items():
        assert torch.equal(payload["ema"][key], value), key
