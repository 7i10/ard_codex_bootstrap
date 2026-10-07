"""Crop keys, the FKD soft-label bank, online-vs-bank RSLAD target parity, and rslad_advt.

Plan 0103 Phase 2 batch D.  CPU only, on a tiny real-JPEG ImageNet-layout fixture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.cli.build_soft_label_bank import bank_directory, build_banks
from ard.config.schema import AttackConfig, ExperimentConfig, ModelConfig, NormalizationConfig
from ard.data import EpochShuffleSampler, collate_indexed
from ard.data.datasets import EpochImageNetTransform, build_train_validation_views
from ard.distillation.crop_keys import CropKeyedBatch, CropKeyedSubset, collate_crop_keyed
from ard.distillation.soft_label_bank import (
    SoftLabelBank,
    SoftLabelBankError,
    SoftLabelBankTeacher,
    bank_identity,
    compress_probabilities,
    pseudo_logits,
    read_bank_manifest,
    reconstruct_probabilities,
    truncate_like_bank,
)
from ard.distillation.trainer_hooks import DistillationTargetHooks
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.models.imagenet_teacher_registry import IMAGENET_TEACHER_SPECS
from ard.models.registry import FixtureCNN
from ard.models.teacher import TeacherAdapter, TeacherMetadata
from ard.objectives import RSLADObjective
from ard.policies import RSLADBaselinePolicy

NUM_CLASSES = 4
IMAGE_SIZE = 16
REGISTRY_ID = "salman2020_resnet50_linf_eps4"


def _make_imagenet(root: Path) -> None:
    generator = np.random.default_rng(0)
    for class_index in range(NUM_CLASSES):
        directory = root / "train" / f"n{class_index:08d}"
        directory.mkdir(parents=True)
        for image_index in range(5):
            height, width = 30 + 3 * image_index, 44 - 2 * class_index
            pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(directory / f"img_{image_index}.JPEG", quality=90)


def _config(
    root: Path, *, method: str = "rslad", bank: dict[str, Any] | None = None, seed: int = 4
) -> ExperimentConfig:
    spec = IMAGENET_TEACHER_SPECS[REGISTRY_ID]
    attack = {"loss": "kl", "kl_target": "teacher_clean", "epsilon": "4/255", "step_size": "8/765", "steps": 2}
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {"id": "imagenet_stage0_dev_v1"},
        "tier": "dev",
        "seeds": {
            "split": 1,
            "model_init": 2,
            "data_order": 3,
            "augmentation": seed,
            "train_attack": 5,
            "evaluation_attack": 0,
            "qualitative_panel": 6,
        },
        "dataset": {"name": "imagenet", "root": str(root), "num_classes": NUM_CLASSES, "image_size": IMAGE_SIZE},
        "student": {
            "architecture": "fixture_cnn",
            "num_classes": NUM_CLASSES,
            "normalization": {"profile": "imagenet_standard"},
        },
        "teacher": {
            "source": "imagenet_registry",
            "registry_id": REGISTRY_ID,
            "architecture": spec.architecture,
            "num_classes": NUM_CLASSES,
            "normalization": {"profile": "imagenet_standard"},
            "checkpoint": str(root / spec.checkpoint_filename),
            "checkpoint_sha256": spec.checkpoint_sha256,
            "threat_epsilon": "4/255",
        },
        "method": {"id": method, "version": 1, "attack": attack},
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 2, "per_rank_batch_size": 4, "global_batch_size": 4, "validation_fraction": 0.25},
        "distillation": (
            {"target_source": "online_teacher"} if bank is None else {"target_source": "soft_label_bank", "bank": bank}
        ),
    }
    return ExperimentConfig.model_validate(payload)


def _teacher(seed: int = 11) -> TeacherAdapter:
    torch.manual_seed(seed)
    metadata = TeacherMetadata(
        architecture="fixture_cnn",
        num_classes=NUM_CLASSES,
        normalization=NormalizationConfig(profile="imagenet_standard"),
        checkpoint_sha256="f" * 64,
    )
    return TeacherAdapter(FixtureCNN(NUM_CLASSES), metadata)


def _views(config: ExperimentConfig) -> Any:
    train, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    return train


@pytest.fixture()
def imagenet_root(tmp_path: Path) -> Path:
    root = tmp_path / "imagenet"
    _make_imagenet(root)
    return root


def _build(config: ExperimentConfig, teacher: torch.nn.Module, out: Path, *, top_k: int, epochs: list[int]) -> str:
    assert config.teacher is not None
    digests = build_banks(
        config,
        {REGISTRY_ID: config.teacher},
        output_root=out,
        top_k=top_k,
        epochs=epochs,
        batch_size=3,
        num_workers=0,
        device=torch.device("cpu"),
        finalize=True,
        teacher_modules={REGISTRY_ID: teacher},
        log=lambda *_: None,
    )
    digest = digests[REGISTRY_ID]
    assert digest is not None
    return digest


def _bank_config(root: Path, out: Path, digest: str, *, top_k: int, method: str = "rslad") -> ExperimentConfig:
    base = _config(root)
    path = bank_directory(out, base, REGISTRY_ID, top_k)
    return _config(root, method=method, bank={"path": str(path), "manifest_sha256": digest, "top_k": top_k})


def _open(config: ExperimentConfig, train: Any) -> SoftLabelBank:
    assert config.distillation is not None and config.distillation.bank is not None
    bank = config.distillation.bank
    return SoftLabelBank.open(
        bank.path,
        expected_manifest_sha256=bank.manifest_sha256,
        expected_identity=bank_identity(config),
        expected_top_k=bank.top_k,
        required_epochs=config.training.epochs,
        train_ids=list(train.indices),
    )


# --------------------------------------------------------------------------
# Crop keys.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("draft", [False, True])
def test_crop_key_describes_the_drawn_view_and_changes_nothing(imagenet_root: Path, draft: bool) -> None:
    path = next((imagenet_root / "train").rglob("*.JPEG"))
    transform = EpochImageNetTransform(augmentation_seed=7, image_size=IMAGE_SIZE, jpeg_draft_decode=draft)
    transform.set_epoch(3)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    source = path if draft else rgb
    first = transform(source, source_id=9)
    key = transform.last_crop_key
    assert key is not None and key[0] == 3
    # Replay the generator exactly as the transform keys it.
    generator = torch.Generator().manual_seed(7 + 1_000_003 * 3 + 10_007 * 9)
    box = transform._crop_box_for_size(rgb.width, rgb.height, generator)
    flip = int(torch.randint(0, 2, (), generator=generator).item())
    assert key == (3, *box, flip)
    assert torch.equal(transform(source, source_id=9), first)


@pytest.mark.parametrize("workers", [0, 2])
def test_crop_keyed_batches_and_device_move(imagenet_root: Path, workers: int) -> None:
    train = _views(_config(imagenet_root))
    keyed = CropKeyedSubset(train)
    keyed.set_epoch(1)
    loader = DataLoader(keyed, batch_size=4, collate_fn=collate_crop_keyed, num_workers=workers)
    batch = next(iter(loader))
    assert isinstance(batch, CropKeyedBatch) and batch.crop_keys is not None
    assert batch.crop_keys.shape == (4, 6) and bool((batch.crop_keys[:, 0] == 1).all())
    moved = batch.to("cpu")
    assert isinstance(moved, CropKeyedBatch) and torch.equal(moved.crop_keys, batch.crop_keys)  # type: ignore[arg-type]
    plain = next(iter(DataLoader(train, batch_size=4, collate_fn=collate_indexed)))
    assert torch.equal(plain.images, batch.images)


# --------------------------------------------------------------------------
# Storage format.
# --------------------------------------------------------------------------


def test_full_k_round_trip_is_fp16_exact_and_pseudo_logits_reproduce_target() -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, NUM_CLASSES) * 3
    probabilities = F.softmax(logits, dim=1)
    index, values, residual = compress_probabilities(probabilities, NUM_CLASSES)
    assert bool((residual == 0).all())
    restored = reconstruct_probabilities(index, values, residual, num_classes=NUM_CLASSES)
    torch.testing.assert_close(restored, probabilities, rtol=2 * 2**-11, atol=1e-7)
    # softmax(log p) == p for every temperature-1 consumer.
    torch.testing.assert_close(F.softmax(pseudo_logits(restored), dim=1), restored, rtol=0, atol=1e-6)


def test_marginal_smoothing_truncation() -> None:
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.7, 0.1, 0.1, 0.1]])
    index, values, residual = compress_probabilities(probabilities, 2)
    restored = reconstruct_probabilities(index, values, residual, num_classes=4)
    torch.testing.assert_close(restored.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(restored[0], torch.tensor([0.5, 0.3, 0.1, 0.1]), rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(truncate_like_bank(probabilities, 2), restored, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Bank build / verification / online parity.
# --------------------------------------------------------------------------


def test_bank_full_k_reproduces_the_online_rslad_target(imagenet_root: Path, tmp_path: Path) -> None:
    teacher = _teacher()
    digest = _build(_config(imagenet_root), teacher, tmp_path / "banks", top_k=NUM_CLASSES, epochs=[0, 1])
    config = _bank_config(imagenet_root, tmp_path / "banks", digest, top_k=NUM_CLASSES)
    train = _views(config)
    bank_teacher = SoftLabelBankTeacher(_open(config, train))
    keyed = CropKeyedSubset(train)
    objective = RSLADObjective()
    for epoch in (0, 1):
        keyed.set_epoch(epoch)
        for batch in DataLoader(keyed, batch_size=4, shuffle=True, collate_fn=collate_crop_keyed):
            online = teacher(batch.images).float()
            banked = bank_teacher.clean_logits(batch, epoch=epoch)
            torch.testing.assert_close(F.softmax(banked, dim=1), F.softmax(online, dim=1), rtol=2 * 2**-11, atol=1e-6)
            student = torch.randn(batch.images.shape[0], NUM_CLASSES)
            clean = torch.randn_like(student)
            a = objective(
                student_logits=student, labels=batch.labels, teacher_logits=online, clean_student_logits=clean
            )
            b = objective(
                student_logits=student, labels=batch.labels, teacher_logits=banked, clean_student_logits=clean
            )
            torch.testing.assert_close(a.kd, b.kd, rtol=1e-3, atol=1e-5)
    manifest = json.loads((config.distillation.bank.path / "manifest.json").read_text())  # type: ignore[union-attr]
    assert manifest["epochs"] == [0, 1] and manifest["top_k"] == NUM_CLASSES


def test_bank_refuses_mismatches(imagenet_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "banks"
    digest = _build(_config(imagenet_root), _teacher(), out, top_k=2, epochs=[0, 1])
    config = _bank_config(imagenet_root, out, digest, top_k=2)
    train = _views(config)
    bank = _open(config, train)
    keyed = CropKeyedSubset(train)
    keyed.set_epoch(1)
    batch = next(iter(DataLoader(keyed, batch_size=4, collate_fn=collate_crop_keyed)))
    assert batch.crop_keys is not None
    bank.probabilities(batch.sample_ids, batch.crop_keys, epoch=1)
    with pytest.raises(SoftLabelBankError, match="crop key mismatch"):
        bank.probabilities(batch.sample_ids, batch.crop_keys, epoch=0)
    tampered = batch.crop_keys.clone()
    tampered[0, 1] += 1
    with pytest.raises(SoftLabelBankError, match="crop key mismatch"):
        bank.probabilities(batch.sample_ids, tampered, epoch=1)
    assert config.distillation is not None and config.distillation.bank is not None
    path = config.distillation.bank.path
    with pytest.raises(SoftLabelBankError, match="SHA-256 mismatch"):
        read_bank_manifest(
            path,
            expected_manifest_sha256="0" * 64,
            expected_identity=bank_identity(config),
            expected_top_k=2,
            required_epochs=2,
        )
    other_seed = _config(imagenet_root, seed=9)
    with pytest.raises(SoftLabelBankError, match="training_view"):
        read_bank_manifest(
            path,
            expected_manifest_sha256=digest,
            expected_identity=bank_identity(other_seed),
            expected_top_k=2,
            required_epochs=2,
        )
    with pytest.raises(SoftLabelBankError, match="top_k"):
        read_bank_manifest(
            path,
            expected_manifest_sha256=digest,
            expected_identity=bank_identity(config),
            expected_top_k=3,
            required_epochs=2,
        )
    with pytest.raises(SoftLabelBankError, match="lacks training epochs"):
        read_bank_manifest(
            path,
            expected_manifest_sha256=digest,
            expected_identity=bank_identity(config),
            expected_top_k=2,
            required_epochs=3,
        )
    # An altered array is caught before any row of that epoch is served.
    prob = path / "epoch-000" / "topk_prob.npy"
    array = np.load(prob)
    array[0, 0] = np.float16(0.5) if array[0, 0] != np.float16(0.5) else np.float16(0.25)
    np.save(prob, array)
    fresh = _open(config, train)
    keyed.set_epoch(0)
    batch0 = next(iter(DataLoader(keyed, batch_size=4, collate_fn=collate_crop_keyed)))
    with pytest.raises(SoftLabelBankError, match="altered"):
        fresh.probabilities(batch0.sample_ids, batch0.crop_keys, epoch=0)  # type: ignore[arg-type]
    # Plain rslad bank mode has no pixel forward at all.
    with pytest.raises(SoftLabelBankError, match="no pixel forward"):
        SoftLabelBankTeacher(bank)(batch.images)


def test_build_is_resumable_and_non_overwriting(imagenet_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "banks"
    config = _config(imagenet_root)
    assert config.teacher is not None
    teacher = _teacher()
    common = dict(
        output_root=out,
        top_k=2,
        batch_size=3,
        num_workers=0,
        device=torch.device("cpu"),
        teacher_modules={REGISTRY_ID: teacher},
        log=lambda *_: None,
    )
    build_banks(config, {REGISTRY_ID: config.teacher}, epochs=[0], finalize=False, **common)  # type: ignore[arg-type]
    first = (bank_directory(out, config, REGISTRY_ID, 2) / "epoch-000" / "meta.json").read_bytes()
    digest = build_banks(config, {REGISTRY_ID: config.teacher}, epochs=[0, 1], finalize=True, **common)  # type: ignore[arg-type]
    assert (bank_directory(out, config, REGISTRY_ID, 2) / "epoch-000" / "meta.json").read_bytes() == first
    assert digest[REGISTRY_ID] is not None
    with pytest.raises(SoftLabelBankError, match="non-overwrite"):
        build_banks(config, {REGISTRY_ID: config.teacher}, epochs=[1], finalize=True, **common)  # type: ignore[arg-type]
    with pytest.raises(SoftLabelBankError, match="different identity"):
        build_banks(
            _config(imagenet_root, seed=9).model_copy(update={"seeds": config.seeds.model_copy(update={"split": 2})}),
            {REGISTRY_ID: config.teacher},
            epochs=[0],
            finalize=False,
            **{**common, "output_root": out},
        )  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Trainer: online vs bank, and rslad_advt.
# --------------------------------------------------------------------------


class _Recorder:
    """Wrap an objective to record its inputs (detached copies)."""

    requires_clean_student_logits = True
    requires_teacher_clean_logits = True
    requires_rectified_target_probabilities = False
    rectifies_attack_target = False

    def __init__(self, inner: RSLADObjective) -> None:
        self.inner = inner
        self.temperature = inner.temperature
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **inputs: Any) -> Any:
        self.calls.append({k: v.detach().clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)})
        return self.inner(**inputs)


def _trainer(
    teacher: torch.nn.Module,
    objective: Any,
    hooks: DistillationTargetHooks | None,
    out: Path,
    diagnostics: Any | None = None,
) -> Trainer:
    torch.manual_seed(123)
    model = build_student(
        ModelConfig(
            architecture="fixture_cnn", num_classes=NUM_CLASSES, normalization={"profile": "imagenet_standard"}
        ),
        tier="smoke",
    )
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    attack = AttackConfig(
        loss="kl", kl_target="teacher_clean", epsilon="4/255", step_size="8/765", steps=2, random_start=True
    )
    selection = AttackConfig(epsilon="4/255", step_size="8/765", steps=1, random_start=True)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=1, gamma=1.0),
        scaler=None,
        attack=LinfPGD(attack),
        selection_attack=LinfPGD(selection),
        objective=objective,
        policy=RSLADBaselinePolicy(),
        device=torch.device("cpu"),
        output_dir=out,
        config_hash="a" * 64,
        seed=4,
        tracker_run_id="offline-fixture",
        teacher=teacher,
        distillation_hooks=hooks,
        diagnostics=diagnostics,
    )


def _loader(train: Any, epoch: int) -> DataLoader[Any]:
    keyed = CropKeyedSubset(train)
    keyed.set_epoch(epoch)
    return DataLoader(
        keyed,
        batch_size=4,
        sampler=EpochShuffleSampler(len(keyed), seed=3, shuffle=True),
        collate_fn=collate_crop_keyed,
    )


def test_trainer_bank_mode_matches_online_target_and_runs_no_teacher(imagenet_root: Path, tmp_path: Path) -> None:
    teacher = _teacher()
    digest = _build(_config(imagenet_root), teacher, tmp_path / "banks", top_k=NUM_CLASSES, epochs=[0, 1])
    config = _bank_config(imagenet_root, tmp_path / "banks", digest, top_k=NUM_CLASSES)
    train = _views(config)
    online = _Recorder(RSLADObjective())
    banked = _Recorder(RSLADObjective())
    online_metrics = _trainer(
        teacher, online, DistillationTargetHooks(adversarial_teacher_target=False, temperature=1.0), tmp_path / "o"
    ).train_epoch(_loader(train, 0))
    bank_metrics = _trainer(
        SoftLabelBankTeacher(_open(config, train)),
        banked,
        DistillationTargetHooks(adversarial_teacher_target=False, temperature=1.0),
        tmp_path / "b",
    ).train_epoch(_loader(train, 0))
    # The first step starts from identical weights and inputs: same target up to fp16 storage.
    torch.testing.assert_close(
        F.softmax(banked.calls[0]["teacher_logits"], dim=1),
        F.softmax(online.calls[0]["teacher_logits"], dim=1),
        rtol=2 * 2**-11,
        atol=1e-6,
    )
    assert online_metrics["teacher_clean_forward_calls"] == len(online.calls)
    assert bank_metrics["teacher_clean_forward_calls"] == 0.0
    assert bank_metrics["teacher_adversarial_forward_calls"] == 0.0


def test_rslad_advt_equals_rslad_when_adversarial_target_is_clean_target() -> None:
    torch.manual_seed(1)
    student, clean_student, teacher = (
        torch.randn(5, NUM_CLASSES),
        torch.randn(5, NUM_CLASSES),
        torch.randn(5, NUM_CLASSES),
    )
    labels = torch.randint(0, NUM_CLASSES, (5,))
    objective = RSLADObjective()
    base = objective(student_logits=student, labels=labels, teacher_logits=teacher, clean_student_logits=clean_student)
    hooks = DistillationTargetHooks(adversarial_teacher_target=True, temperature=1.0)
    target = hooks.adversarial_target(
        teacher=None,
        teacher_adversarial_logits=teacher,
        teacher_clean_logits=teacher,
        labels=labels,
        mask=torch.ones(5),
    )
    advt = objective(
        student_logits=student,
        labels=labels,
        teacher_logits=teacher,
        clean_student_logits=clean_student,
        adversarial_target_probabilities=target,
    )
    assert torch.equal(advt.kd, base.kd) and torch.equal(advt.adversarial_kd, base.adversarial_kd)  # type: ignore[arg-type]


def test_rslad_advt_trainer_one_adversarial_teacher_forward_per_step(imagenet_root: Path, tmp_path: Path) -> None:
    teacher = _teacher()
    train = _views(_config(imagenet_root, method="rslad_advt"))
    forwards: list[int] = []
    original = teacher.forward

    def counting(pixels: torch.Tensor) -> torch.Tensor:
        forwards.append(pixels.shape[0])
        return original(pixels)

    teacher.forward = counting  # type: ignore[method-assign]
    recorder = _Recorder(RSLADObjective())
    hooks = DistillationTargetHooks(adversarial_teacher_target=True, temperature=1.0)
    trainer = _trainer(teacher, recorder, hooks, tmp_path / "a")
    metrics = trainer.train_epoch(_loader(train, 0))
    steps = len(recorder.calls)
    # One clean forward + one adversarial forward per step; the KL-PGD attack
    # reuses the clean target and never calls the teacher.
    assert len(forwards) == 2 * steps
    assert metrics["teacher_adversarial_forward_calls"] == steps
    assert metrics["teacher_clean_forward_calls"] == steps
    for call in recorder.calls:
        assert "adversarial_target_probabilities" in call
    assert 0.0 <= metrics["advt_teacher_adversarial_accuracy"] <= 1.0
    assert metrics["advt_teacher_adversarial_clean_kl"] >= 0.0 and math.isfinite(
        metrics["advt_teacher_adversarial_clean_kl"]
    )


def test_rslad_advt_bank_mode_truncates_teacher_adversarial_target_like_the_bank(
    imagenet_root: Path, tmp_path: Path
) -> None:
    teacher = _teacher()
    digest = _build(_config(imagenet_root), teacher, tmp_path / "banks", top_k=2, epochs=[0, 1])
    config = _bank_config(imagenet_root, tmp_path / "banks", digest, top_k=2, method="rslad_advt")
    train = _views(config)
    bank = _open(config, train)
    bank_teacher = SoftLabelBankTeacher(bank, online_teacher=teacher)
    hooks = DistillationTargetHooks(adversarial_teacher_target=True, temperature=1.0)
    hooks.reset_epoch(torch.device("cpu"))
    keyed = CropKeyedSubset(train)
    keyed.set_epoch(0)
    batch = next(iter(DataLoader(keyed, batch_size=4, collate_fn=collate_crop_keyed)))
    # Feeding the clean crop as "x'": the advT target must equal the bank's stored row exactly.
    with torch.no_grad():
        logits = bank_teacher(batch.images).float()
    target = hooks.adversarial_target(
        teacher=bank_teacher,
        teacher_adversarial_logits=logits,
        teacher_clean_logits=bank_teacher.clean_logits(batch, epoch=0),
        labels=batch.labels,
        mask=torch.ones(4),
    )
    stored = bank.probabilities(batch.sample_ids, batch.crop_keys, epoch=0)  # type: ignore[arg-type]
    assert torch.equal(target, stored)
    assert (target > 0).all() and torch.equal(target, truncate_like_bank(F.softmax(logits, dim=1), 2))
    trainer = _trainer(bank_teacher, _Recorder(RSLADObjective()), hooks, tmp_path / "t")
    metrics = trainer.train_epoch(_loader(train, 0))
    assert metrics["teacher_clean_forward_calls"] == 0.0
    assert metrics["teacher_adversarial_forward_calls"] > 0


def test_fit_two_epochs_bank_advt_logs_rows(imagenet_root: Path, tmp_path: Path) -> None:
    from ard.data.datasets import build_train_validation_views as views

    teacher = _teacher()
    digest = _build(_config(imagenet_root), teacher, tmp_path / "banks", top_k=2, epochs=[0, 1])
    config = _bank_config(imagenet_root, tmp_path / "banks", digest, top_k=2, method="rslad_advt")
    train, validation = views(
        config.dataset,
        validation_fraction=0.25,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    keyed = CropKeyedSubset(train)
    loader = DataLoader(
        keyed, batch_size=4, sampler=EpochShuffleSampler(len(keyed), seed=3), collate_fn=collate_crop_keyed
    )
    validation_loader = DataLoader(
        validation,
        batch_size=4,
        sampler=EpochShuffleSampler(len(validation), seed=3, shuffle=False),
        collate_fn=collate_indexed,
    )
    trainer = _trainer(
        SoftLabelBankTeacher(_open(config, train), online_teacher=teacher),
        RSLADObjective(),
        DistillationTargetHooks(adversarial_teacher_target=True, temperature=1.0),
        tmp_path / "fit",
    )
    rows = trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    assert len(rows) == 2
    for row in rows:
        assert "train_advt_teacher_adversarial_accuracy" in row
        assert "train_advt_teacher_adversarial_clean_kl" in row
        assert row["train_teacher_clean_forward_calls"] == 0.0


def test_pixel_sentinel_catches_same_geometry_different_pixels(imagenet_root: Path, tmp_path: Path) -> None:
    digest = _build(_config(imagenet_root), _teacher(), tmp_path / "banks", top_k=2, epochs=[0, 1])
    config = _bank_config(imagenet_root, tmp_path / "banks", digest, top_k=2)
    train = _views(config)
    keyed = CropKeyedSubset(train)
    keyed.set_epoch(0)
    teacher = SoftLabelBankTeacher(_open(config, train), sentinel_view=CropKeyedSubset(train))
    batch = next(iter(DataLoader(keyed, batch_size=4, collate_fn=collate_crop_keyed)))
    teacher.clean_logits(batch, epoch=0)  # sentinel matches
    # Same file sizes / crop geometry, different pixels: crop keys still match, the sentinel must not.
    for path in sorted((imagenet_root / "train").rglob("*.JPEG")):
        with Image.open(path) as image:
            size = image.size
        Image.new("RGB", size, color=(5, 6, 7)).save(path, quality=90)
    keyed.set_epoch(1)
    fresh = SoftLabelBankTeacher(_open(config, train), sentinel_view=CropKeyedSubset(train))
    batch1 = next(iter(DataLoader(keyed, batch_size=4, collate_fn=collate_crop_keyed)))
    with pytest.raises(SoftLabelBankError, match="pixel sentinel mismatch"):
        fresh.clean_logits(batch1, epoch=1)


def test_distillation_runs_pay_no_diagnostics_only_teacher_forward(imagenet_root: Path, tmp_path: Path) -> None:
    from ard.tracking.diagnostics import TrainingDiagnostics

    train = _views(_config(imagenet_root))
    panel = TrainingDiagnostics.for_ids(list(train.indices), seed=1, size=4, mode="panel")
    hooks = DistillationTargetHooks(adversarial_teacher_target=False, temperature=1.0)
    metrics = _trainer(_teacher(), RSLADObjective(), hooks, tmp_path / "d", diagnostics=panel).train_epoch(
        _loader(train, 0)
    )
    assert metrics["teacher_adversarial_forward_calls"] == 0.0
    # Without distillation hooks (every pre-existing run) the diagnostic forward is unchanged.
    legacy = _trainer(_teacher(), RSLADObjective(), None, tmp_path / "l", diagnostics=panel).train_epoch(
        _loader(train, 0)
    )
    assert legacy["teacher_adversarial_forward_calls"] == legacy["teacher_clean_forward_calls"] > 0
