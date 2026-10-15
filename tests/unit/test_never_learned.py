"""Never-learned guard (debugging 0030): a run that never leaves a constant output
stops after epoch 1 instead of running to completion at chance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config import load_config
from ard.config.loader import resolved_config_dict
from ard.config.schema import AttackConfig, ModelConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.never_learned import (
    NEVER_LEARNED_MIN_IMAGES_FOR_DISTINCT,
    NEVER_LEARNED_MIN_LOGIT_STD,
    NeverLearnedError,
    NonFiniteValidationLogitsError,
    epoch_is_degenerate,
    never_learned_reason,
    observations_from_epoch_rows,
)
from ard.engine.checkpoint import config_digest
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import PGDATObjective

HEALTHY = (37.0, 0.42)  # many classes predicted, input-dependent logits
CONSTANT = (1.0, 3e-7)  # the collapsed ConvStem epoch-0 signature


# (a) The guard rule.


N = 5000  # a production-size validation set (CIFAR-10 held-out, ImageNet subset)


def test_epoch_verdict_is_distinct_at_most_one_or_std_below_floor() -> None:
    assert epoch_is_degenerate(1.0, 0.5, validation_images=N)
    assert epoch_is_degenerate(0.0, 0.5, validation_images=N)
    assert epoch_is_degenerate(5.0, NEVER_LEARNED_MIN_LOGIT_STD / 2, validation_images=N)
    assert epoch_is_degenerate(5.0, math.nan, validation_images=N)
    assert not epoch_is_degenerate(2.0, NEVER_LEARNED_MIN_LOGIT_STD, validation_images=N)
    assert not epoch_is_degenerate(*HEALTHY, validation_images=N)


def test_distinct_criterion_needs_a_minimum_validation_size_but_std_does_not() -> None:
    floor = NEVER_LEARNED_MIN_IMAGES_FOR_DISTINCT
    assert epoch_is_degenerate(1.0, 0.5, validation_images=floor)
    # A tiny set: an input-dependent untrained net predicting one class is not evidence.
    assert not epoch_is_degenerate(1.0, 0.0123, validation_images=floor - 1)
    assert not epoch_is_degenerate(1.0, 0.0123, validation_images=None)
    # Input independence is caught at every size.
    assert epoch_is_degenerate(1.0, 3e-7, validation_images=12)
    assert epoch_is_degenerate(3.0, 0.0, validation_images=None)


def test_guard_fires_only_when_epochs_zero_and_one_are_both_degenerate() -> None:
    def reason(observations: dict) -> str | None:
        return never_learned_reason(observations, validation_images=N)

    assert reason({0: CONSTANT, 1: CONSTANT}) is not None
    assert reason({0: (1.0, 0.3), 1: (40.0, 1e-6)}) is not None  # either criterion, per epoch
    # One degenerate epoch alone never fires.
    assert reason({0: CONSTANT}) is None
    assert reason({0: CONSTANT, 1: HEALTHY}) is None
    assert reason({0: HEALTHY, 1: CONSTANT}) is None
    # Epoch 1 without an observed epoch 0 (e.g. a resume without persisted metrics).
    assert reason({1: CONSTANT}) is None
    assert reason({0: HEALTHY, 1: HEALTHY}) is None
    assert reason({}) is None
    # The fixture signature (one class, clearly input-dependent) on 12 images: silent.
    assert never_learned_reason({0: (1.0, 0.0123), 1: (1.0, 0.0124)}, validation_images=12) is None


def test_resume_seeding_reads_only_guarded_epochs_with_both_metrics() -> None:
    rows = [
        {"epoch": 0, "val_subset_distinct_predictions": 1.0, "val_subset_logit_std_over_images": 0.0},
        {"epoch": 1, "val_subset_clean_accuracy": 0.1},  # pre-metric row: ignored
        {"epoch": 2, "val_subset_distinct_predictions": 1.0, "val_subset_logit_std_over_images": 0.0},
    ]
    assert observations_from_epoch_rows(rows, prefix="val_subset") == {0: (1.0, 0.0)}
    assert observations_from_epoch_rows(rows, prefix="val") == {}


# Trainer fixtures.


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = IndexedDataset(SyntheticCIFAR(size=24, num_classes=3, image_size=4, seed=4))
    train, validation = stratified_train_validation_split(dataset, validation_fraction=0.5, seed=4)
    loader = DataLoader(
        train, batch_size=4, sampler=EpochShuffleSampler(len(train), seed=4), collate_fn=collate_indexed
    )
    validation_loader = DataLoader(
        validation,
        batch_size=4,
        sampler=EpochShuffleSampler(len(validation), seed=4, shuffle=False),
        collate_fn=collate_indexed,
    )
    return loader, validation_loader


class _ConstantOutput(nn.Module):
    """Logits = bias: the frozen zero weight keeps the graph connected to the
    input (PGD gets an exactly-zero gradient), so the output can never depend
    on the image however long it trains -- the collapse of debugging 0030."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 3)
        nn.init.zeros_(self.linear.weight)
        self.linear.weight.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images.flatten(1))


def _trainer(model: nn.Module, output: Path) -> Trainer:
    attack = AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)
    selection = AttackConfig(
        epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
    )
    return Trainer(
        model=model,
        optimizer=SGD([p for p in model.parameters() if p.requires_grad], lr=0.05, momentum=0.9),
        scheduler=None,
        scaler=None,
        attack=LinfPGD(attack),
        selection_attack=LinfPGD(selection),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="a" * 64,
        seed=4,
        tracker_run_id="offline-fixture",
    )


# (b) A constant-output student stops after epoch 1 through the failure path.


def test_constant_output_trainer_stops_after_epoch_one(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = _trainer(_ConstantOutput(), tmp_path)
    loader, validation_loader = _loaders()
    rows: list[dict] = []
    with pytest.raises(NeverLearnedError, match="never-learned guard"):
        trainer.fit(
            loader,
            validation_loader=validation_loader,
            epochs=5,
            on_epoch_end=lambda metrics, _: rows.append(dict(metrics)),
        )
    # Epochs 0 and 1 ran and were recorded (checkpoint + metric row), nothing after.
    assert [trainer.current_epoch, len(rows)] == [1, 2]
    for row in rows:
        assert row["val_distinct_predictions"] == 1.0
        assert row["val_logit_std_over_images"] < NEVER_LEARNED_MIN_LOGIT_STD
    assert torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 1
    marker = json.loads((tmp_path / "never-learned.json").read_text(encoding="utf-8"))
    assert set(marker["observations"]) == {"0", "1"} and "never-learned guard" in marker["reason"]
    assert marker["validation_images"] == 12.0


def test_constant_output_resumed_at_epoch_one_uses_persisted_epoch_zero(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = _trainer(_ConstantOutput(), tmp_path)
    loader, validation_loader = _loaders()
    trainer.seed_never_learned_observations(
        [{"epoch": 0, "val_distinct_predictions": 1.0, "val_logit_std_over_images": 0.0}]
    )
    with pytest.raises(NeverLearnedError):
        trainer.fit(loader, validation_loader=validation_loader, epochs=5, start_epoch=1)
    assert trainer.current_epoch == 1


def test_constant_output_resumed_at_epoch_one_without_epoch_zero_does_not_fire(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = _trainer(_ConstantOutput(), tmp_path)
    loader, validation_loader = _loaders()
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=3, start_epoch=1)
    assert len(history) == 2  # epochs 1 and 2; epoch >= 2 never evaluates the guard


class _OneClassInputDependent(nn.Module):
    """Input-dependent logits (std across images well above the floor) whose
    argmax is always class 0, frozen so it cannot learn: the distinct-only case."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 3)
        with torch.no_grad():
            self.linear.weight.normal_(0.0, 0.01)
            self.linear.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))
        self.linear.requires_grad_(False)
        self.unused = nn.Parameter(torch.zeros(1))  # SGD needs one parameter

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images.flatten(1)) + 0.0 * self.unused


def _validation_loader(size: int) -> DataLoader:
    dataset = IndexedDataset(SyntheticCIFAR(size=size, num_classes=3, image_size=4, seed=9))
    return DataLoader(
        dataset,
        batch_size=250,
        sampler=EpochShuffleSampler(len(dataset), seed=9, shuffle=False),
        collate_fn=collate_indexed,
    )


def test_one_class_output_fires_at_production_size_only(tmp_path: Path) -> None:
    loader, _ = _loaders()
    large = _trainer(_OneClassInputDependent(), tmp_path / "large")
    with pytest.raises(NeverLearnedError):
        large.fit(loader, validation_loader=_validation_loader(NEVER_LEARNED_MIN_IMAGES_FOR_DISTINCT), epochs=3)
    observed = large.never_learned_observations
    assert observed[0][0] == observed[1][0] == 1.0 and observed[1][1] > NEVER_LEARNED_MIN_LOGIT_STD
    small = _trainer(_OneClassInputDependent(), tmp_path / "small")
    history = small.fit(loader, validation_loader=_validation_loader(24), epochs=3)
    assert [row["val_distinct_predictions"] for row in history] == [1.0, 1.0, 1.0]


# (c) The metrics appear in the epoch rows of a normal run.


def test_normal_run_rows_carry_the_metrics_and_the_guard_stays_silent(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    trainer = _trainer(model, tmp_path)
    loader, validation_loader = _loaders()
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    assert len(history) == 2
    for row in history:
        assert 1.0 <= row["val_distinct_predictions"] <= 3.0
        assert float(row["val_distinct_predictions"]).is_integer()
        assert row["val_logit_std_over_images"] > NEVER_LEARNED_MIN_LOGIT_STD
    assert not (tmp_path / "never-learned.json").exists()


def test_logit_std_matches_a_direct_computation(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    trainer = _trainer(model, tmp_path)
    _, validation_loader = _loaders()
    metrics = trainer.validate_epoch(validation_loader)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch.images) for batch in validation_loader]).double()
    assert metrics["distinct_predictions"] == float(logits.argmax(1).unique().numel())
    assert metrics["logit_std_over_images"] == pytest.approx(
        float(logits.std(0, unbiased=False).mean()), rel=1e-9, abs=1e-12
    )


# (d) No config surface: the guard is hard-coded, so config hash / resolved config are untouched.


_HASH_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "scientific" / (
    "imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml"
)
# config_digest of _HASH_CONFIG under the environment below, computed at 8a0ef4f
# (before the guard existed) and again with it: identical.
_PINNED_DIGEST = "3bd2e011ae2827ba0eb20d4d1e83c79764ce0e51334f853442443b3ce264455d"


def test_guard_adds_no_config_field_and_leaves_the_config_hash_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "ARD_SEED": "0",
        "ARD_IMAGENET_ROOT": "/data/imagenet",
        "ARD_NUM_WORKERS": "0",
        "ARD_JOB_OUTPUT_DIR": "/out/job",
        "ARD_RUN_ID": "never-learned-hash",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
    }.items():
        monkeypatch.setenv(key, value)
    resolved = resolved_config_dict(load_config(_HASH_CONFIG))
    serialized = json.dumps(resolved, sort_keys=True)
    assert "never_learned" not in serialized and "distinct_predictions" not in serialized
    assert config_digest(resolved) == _PINNED_DIGEST


# Review P2-2: a non-finite clean validation logit fails explicitly, never as a NaN row.


def test_non_finite_validation_logit_at_a_later_epoch_raises_a_clear_error(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    trainer = _trainer(model, tmp_path)
    loader, validation_loader = _loaders()
    in_validation = False
    original = trainer._validation_totals

    def flagged(*args, **kwargs):
        nonlocal in_validation
        in_validation = True
        try:
            return original(*args, **kwargs)
        finally:
            in_validation = False

    def poison(module, inputs, output):
        if in_validation and trainer.current_epoch == 2 and not torch.is_grad_enabled():
            output = output.clone()
            output[0, 0] = float("inf")
            return output
        return None

    trainer._validation_totals = flagged  # type: ignore[method-assign]
    model.register_forward_hook(poison)
    rows: list[dict] = []
    with pytest.raises(NonFiniteValidationLogitsError, match="clean validation logits are non-finite"):
        trainer.fit(
            loader,
            validation_loader=validation_loader,
            epochs=4,
            on_epoch_end=lambda metrics, _: rows.append(dict(metrics)),
        )
    assert trainer.current_epoch == 2
    assert [len(rows)] == [2]  # epochs 0 and 1 recorded; no row for the poisoned epoch
    assert all(math.isfinite(row["val_logit_std_over_images"]) for row in rows)
