"""``training.cuda_graph`` (plan 0105): the captured step is the eager step, bit for bit.

GPU differentials (skipped without CUDA). Each arm -- eager, graph, graph
resumed mid-run, and a negative control -- runs the real ``Trainer.fit`` in
its OWN subprocess (cold CUDA/cuDNN/allocator state, as in production) with
deterministic algorithms, and returns a fingerprint: epoch rows (timing and
the graph-only audit columns aside), ``last.pt`` / ``best.pt`` state (model,
optimizer, scheduler, every RNG stream, sampler, selection), per-sample
diagnostic rows and panel media, and the final CUDA RNG state. The run has
LR milestones inside it, a partial last batch, a train-probe pass between
epochs, and re-captures every epoch. Students: every allowlisted architecture
(``CUDA_GRAPH_ARCHITECTURES``) at 32 px, plus a BatchNorm + dropout fixture
(so a default-generator draw happens inside the graph), with panel, summary
and no tracking diagnostics. Allowlist candidates (``_CANDIDATE_ARCHITECTURES``, plan 0103
Phase 2 MobileNetV4-S variants) run the same cases only with ``ARD_CUDA_GRAPH_CANDIDATES=1``.

Guard tests: a mid-epoch optimizer change is refused rather than replayed
stale; an epoch that never replays fails loudly; the in-graph device asserts
fire from inside a replay; a pixel-range violation on the graph path kills
the process before anything is checkpointed; and the ``LinfPGD.perturb`` core
never synchronizes with the host.

Plan 0103 Phase 2 batch A (2026-10-08) widened the scope by three options, each covered by the same
differentials (``variant``): a plain weight EMA updated inside the captured step
(``training.weight_ema_decay``), label smoothing in the captured objective, and SGD with a
weight-decay-free group for ndim <= 1 parameters (``optimizer.exclude_norm_bias_from_weight_decay``).

Human decision 2026-10-08 widened it again (``method``): RSLAD and RSLAD-advT distillation from a
soft-label bank (bank rows copied into static buffers, reconstructed inside the step; the advT teacher
forward on x' inside the step), RSLAD from an online frozen teacher, ``method.mixed_batch`` without split
BN, and ``method.awp``. Each has the same bitwise differentials, resume, a method-specific negative
control and the nondeterministic rules; every teacher architecture that may run inside the step
(``CUDA_GRAPH_TEACHER_ARCHITECTURES``) has its own bitwise case.

training.deterministic=false (end of file): bitwise parity is impossible, so
exact RNG streams and audit counts over whole runs, plus a one-step
equivalence test against eager-vs-eager nondeterministic noise.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ard.engine.trainer as trainer_module
from ard.attacks import AttackRequest, LinfPGD
from ard.cli.train import _seed_everything
from ard.config.schema import (
    CUDA_GRAPH_ARCHITECTURES,
    CUDA_GRAPH_TEACHER_ARCHITECTURES,
    AttackConfig,
    AwpConfig,
    MixedBatchConfig,
    ModelConfig,
    NormalizationConfig,
)
from ard.data import EpochShuffleSampler, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.distillation.crop_keys import collate_crop_keyed
from ard.distillation.soft_label_bank import (
    SoftLabelBank,
    SoftLabelBankTeacher,
    _EpochArrays,
    compress_probabilities,
)
from ard.distillation.trainer_hooks import DistillationTargetHooks
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.models.registry import PixelModel
from ard.models.teacher import TeacherAdapter, TeacherMetadata
from ard.objectives import PGDATObjective, RSLADObjective
from ard.policies import RSLADBaselinePolicy
from ard.tracking.diagnostics import TrainingDiagnostics
from tests.integration.test_step_sync_free_parity import _STATE_KEYS, _TIMING_SUFFIXES, _canonical

pytestmark = pytest.mark.t3

_ROOT = Path(__file__).resolve().parents[2]
# 26 training samples at batch 4: six full batches (eager, capture, four
# replays) and a partial batch of 2 per epoch.
_TRAIN_SIZE = 26
_BATCH = 4
_EPOCHS = 3
_FIXTURE = "fixture_bn_dropout"
_FIXTURE_TEACHER = "fixture_teacher"
_AUDIT_KEYS = ("train_cuda_graph_captures", "train_cuda_graph_replays", "train_cuda_graph_eager_steps")
# Plan 0103 Phase 2 MobileNetV4-S architecture variants: allowlist CANDIDATES, not allowlisted. Their
# cases run only on request (ARD_CUDA_GRAPH_CANDIDATES=1, on a free GPU); inside the test subprocess the
# Trainer's allowlist is extended for that one candidate, exactly as for the fixture. Adding an id to
# schema.CUDA_GRAPH_ARCHITECTURES remains a separate, human decision after these pass.
_CANDIDATE_ARCHITECTURES = (
    "mobilenetv4_conv_small_silu_imagenet",
    "mobilenetv4_conv_small_gelu_imagenet",
    "mobilenetv4_conv_small_se_imagenet",
    "mobilenetv4_conv_small_silu_se_imagenet",
    "mobilenetv4_conv_small_se_fullhead_imagenet",
)
_candidate_opt_in = pytest.mark.skipif(
    os.environ.get("ARD_CUDA_GRAPH_CANDIDATES") != "1",
    reason="opt-in allowlist-candidate parity case (needs a free GPU): set ARD_CUDA_GRAPH_CANDIDATES=1",
)


# Plan 0103 ConvNeXt-Atto / DeiT-Tiny students and their batch-B variants (AdamW recipe; human-approved
# 2026-10-09). LayerNorm, GELU, depthwise 7x7 convolutions (ConvNeXt) and SDPA attention (DeiT): every case
# of this file runs for them (SGD and AdamW). DeiT needs its native 224 px input; the ConvNeXt variants run at
# 64 px (stage 4 at 2x2, so the depthwise convolutions see a real spatial extent).
_LAYERNORM_ARCHITECTURES = (
    "convnext_atto_imagenet",
    "convnext_atto_deep_narrow_imagenet",
    "convnext_atto_ols_imagenet",
    "convnext_atto_convstem_imagenet",
    "deit_tiny_imagenet",
    "deit_tiny_convstem_imagenet",
)
_STUDENT_IMAGE_SIZE = {
    **dict.fromkeys(_LAYERNORM_ARCHITECTURES, 64),
    "deit_tiny_imagenet": 224,
    "deit_tiny_convstem_imagenet": 224,
}


def _admit_candidate(kind: str) -> None:
    """Test-only allowlist extension for one candidate, confined to the test subprocess."""
    if kind in _CANDIDATE_ARCHITECTURES or kind in _LAYERNORM_ARCHITECTURES:
        trainer_module.CUDA_GRAPH_ARCHITECTURES = CUDA_GRAPH_ARCHITECTURES | {kind}  # type: ignore[attr-defined]


def _bn_dropout_student(num_classes: int) -> nn.Module:
    inner = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )
    return PixelModel(inner, NormalizationConfig(profile="fixture_unit"))


# Teacher-architecture cases: the fixture student at the teacher's resolution, 1000 classes (ViT-S needs 224 px).
_TEACHER_IMAGE_SIZE = {"vit_s_convstem_imagenet": 224}


def _student(kind: str, *, teacher_kind: str = _FIXTURE_TEACHER) -> tuple[nn.Module, int, int]:
    """Return (student, num_classes, image_size) for a fixture or an allowlisted architecture."""
    if kind == _FIXTURE:
        # A test-only extension of the allowlist, confined to this subprocess.
        trainer_module.CUDA_GRAPH_ARCHITECTURES = CUDA_GRAPH_ARCHITECTURES | {_FIXTURE}  # type: ignore[attr-defined]
        if teacher_kind != _FIXTURE_TEACHER:
            return _bn_dropout_student(1000), 1000, _TEACHER_IMAGE_SIZE.get(teacher_kind, 32)
        return _bn_dropout_student(3), 3, 8
    _admit_candidate(kind)
    config = ModelConfig(
        architecture=kind,  # type: ignore[arg-type]
        num_classes=10,
        pretrained=False,
        normalization=NormalizationConfig(profile="imagenet_standard"),
    )
    return build_student(config, tier="dev"), 10, _STUDENT_IMAGE_SIZE.get(kind, 32)


class _CropKeyed(torch.utils.data.Dataset):
    """Test stand-in for CropKeyedSubset: every item also carries a crop key (0, source_id, 0, 0, 0, 0)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: Any) -> Any:
        item = self.inner[index]
        return (*item, (0, int(item[2]), 0, 0, 0, 0))


def _loaders(
    num_classes: int, image_size: int, *, pin_memory: bool, crop_keyed: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = IndexedDataset(SyntheticCIFAR(size=_TRAIN_SIZE, num_classes=num_classes, image_size=image_size, seed=3))
    validation = IndexedDataset(SyntheticCIFAR(size=6, num_classes=num_classes, image_size=image_size, seed=99))
    # Train-probe pass (production stage 1 runs one every epoch): fixed
    # training-partition images, between the graph epochs.
    probe = IndexedDataset(SyntheticCIFAR(size=6, num_classes=num_classes, image_size=image_size, seed=3))

    def ordered(dataset: Any, *, shuffle: bool, keyed: bool = False) -> DataLoader:
        return DataLoader(
            _CropKeyed(dataset) if keyed else dataset,
            batch_size=_BATCH,
            sampler=EpochShuffleSampler(len(dataset), seed=5, shuffle=shuffle),
            pin_memory=pin_memory,
            collate_fn=collate_crop_keyed if keyed else collate_indexed,
        )

    return (
        ordered(train, shuffle=True, keyed=crop_keyed),
        ordered(validation, shuffle=False),
        ordered(probe, shuffle=False),
    )


# Plan 0103 Phase 2 batch A options a variant ("base" or "+"-joined names) switches on.
_VARIANT_OPTIONS = frozenset({"ema", "label_smoothing", "exclude_norm_bias", "adamw"})
_ALL_OPTIONS = "ema+label_smoothing+exclude_norm_bias"
_EMA_DECAY = 0.9  # large steps, so the EMA visibly moves within a 3-epoch fixture run
_LABEL_SMOOTHING = 0.1


def _variant(variant: str) -> set[str]:
    options = set() if variant == "base" else set(variant.split("+"))
    assert options <= _VARIANT_OPTIONS, variant
    return options


def _optimizer_parameters(student: nn.Module, options: set[str], weight_decay: float) -> Any:
    if "exclude_norm_bias" not in options:
        return student.parameters()
    # What ard.cli.train builds for optimizer.exclude_norm_bias_from_weight_decay=true.
    from ard.cli.train import _weight_decay_parameter_groups

    return _weight_decay_parameter_groups(student.parameters(), weight_decay=weight_decay)


# Variant "adamw" (human-approved 2026-10-09): the plan 0103 ConvNeXt-Atto / DeiT-Tiny recipe as ard.cli.train
# builds it -- betas 0.9/0.999, weight decay 0.05 with norm/bias excluded -- with capturable=True, which
# ard.cli.train sets exactly when training.cuda_graph is on. The eager reference of a parity case uses the
# same capturable optimizer (the graph run's own eager steps do); ARD_CUDA_GRAPH_TEST_ADAMW_CAPTURABLE=0
# builds the capturable=False AdamW of an eager run without the flag (only for the numerics comparison).
_ADAMW_WEIGHT_DECAY = 0.05
_ADAMW_LEARNING_RATE = 2e-3


def _optimizer(
    student: nn.Module,
    options: set[str],
    *,
    sgd_learning_rate: float,
    sgd_weight_decay: float,
    adamw_learning_rate: float = _ADAMW_LEARNING_RATE,
) -> torch.optim.Optimizer:
    if "adamw" in options:
        from ard.cli.train import _weight_decay_parameter_groups

        capturable = os.environ.get("ARD_CUDA_GRAPH_TEST_ADAMW_CAPTURABLE", "1") == "1"
        return AdamW(
            _weight_decay_parameter_groups(student.parameters(), weight_decay=_ADAMW_WEIGHT_DECAY),
            lr=adamw_learning_rate,
            betas=(0.9, 0.999),
            **({"capturable": True} if capturable else {}),
        )
    return SGD(
        _optimizer_parameters(student, options, sgd_weight_decay),
        lr=sgd_learning_rate,
        momentum=0.9,
        weight_decay=sgd_weight_decay,
        nesterov=True,
    )


# Training methods a parity case runs (human decision 2026-10-08 added all but pgd_at).
_METHODS = ("pgd_at", "rslad_bank", "rslad_online", "rslad_advt_bank", "mixed_batch", "awp")
_DISTILLATION_METHODS = ("rslad_bank", "rslad_online", "rslad_advt_bank")
_BANK_METHODS = ("rslad_bank", "rslad_advt_bank")
_MIXED = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3)
# AWP is off in epoch 0 and on afterwards, so a run covers both; gamma large enough to move the weights.
_AWP = AwpConfig(gamma=0.05, warmup_epochs=1)


def _bank_top_k(num_classes: int) -> int:
    """A truncating bank (K < C), so the residual-mass reconstruction and the advT truncation both act."""
    return {3: 2, 10: 3}.get(num_classes, 10)


def _fixture_bank(num_classes: int, size: int, epochs: int) -> SoftLabelBank:
    """An in-memory soft-label bank over source IDs 0..size-1, a different random teacher distribution per
    epoch, stored exactly as the builder stores it (compress_probabilities), crop key (0, id, 0, 0, 0, 0)."""
    top_k = _bank_top_k(num_classes)
    ids = np.arange(size, dtype=np.int64)
    bank = SoftLabelBank(Path("in-memory"), {"top_k": top_k, "num_classes": num_classes, "epoch_records": {}}, ids)
    generator = torch.Generator().manual_seed(77)
    keys = np.zeros((size, 6), dtype=np.int32)
    keys[:, 1] = ids
    for epoch in range(epochs):
        probabilities = (torch.randn(size, num_classes, generator=generator) * 2.0).softmax(dim=1)
        index, prob16, residual16 = compress_probabilities(probabilities, top_k)
        bank._epochs[epoch] = _EpochArrays(
            index=index.numpy().astype(np.uint16), prob=prob16.numpy(), residual=residual16.numpy(), keys=keys
        )
    return bank


def _teacher(teacher_kind: str, num_classes: int) -> TeacherAdapter:
    """A frozen eval-mode teacher: a BatchNorm + dropout fixture (test-only allowlist entry, confined to the
    subprocess) or a randomly initialised allowlisted ImageNet teacher architecture (1000 classes)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4321)
        if teacher_kind == _FIXTURE_TEACHER:
            trainer_module.CUDA_GRAPH_TEACHER_ARCHITECTURES = (  # type: ignore[attr-defined]
                CUDA_GRAPH_TEACHER_ARCHITECTURES | {_FIXTURE_TEACHER}
            )
            model = _bn_dropout_student(num_classes).model
            normalization = NormalizationConfig(profile="fixture_unit")
        else:
            from ard.models.imagenet_teacher_registry import build_imagenet_teacher_architecture

            assert num_classes == 1000
            model = build_imagenet_teacher_architecture(teacher_kind)
            normalization = NormalizationConfig(profile="imagenet_standard")
    metadata = TeacherMetadata(
        architecture=teacher_kind, num_classes=num_classes, normalization=normalization, checkpoint_sha256="0" * 64
    )
    return TeacherAdapter(model, metadata)


def _method_parts(
    method: str, *, num_classes: int, train_size: int, epochs: int, teacher_kind: str, attack: dict[str, Any]
) -> dict[str, Any]:
    """Trainer keyword arguments (attack, objective, teacher, policy, hooks, mixed batch, AWP) of a method."""
    assert method in _METHODS, method
    if method not in _DISTILLATION_METHODS:
        return {
            "attack": LinfPGD(AttackConfig(**attack, random_start=True)),
            "mixed_batch": _MIXED if method == "mixed_batch" else None,
            "awp": _AWP if method == "awp" else None,
        }
    teacher: nn.Module
    if method == "rslad_online":
        teacher = _teacher(teacher_kind, num_classes)
    else:
        bank = _fixture_bank(num_classes, train_size, epochs)
        online = _teacher(teacher_kind, num_classes) if method == "rslad_advt_bank" else None
        teacher = SoftLabelBankTeacher(bank, online_teacher=online)
    return {
        "attack": LinfPGD(AttackConfig(**attack, random_start=True, loss="kl", kl_target="teacher_clean")),
        "objective": RSLADObjective(temperature=1.0, temperature_squared=True),
        "policy": RSLADBaselinePolicy(),
        "teacher": teacher,
        "distillation_hooks": DistillationTargetHooks(
            adversarial_teacher_target=method == "rslad_advt_bank", temperature=1.0
        ),
    }


def build_trainer(
    output: Path,
    *,
    kind: str,
    cuda_graph: bool,
    device: torch.device,
    diagnostics: str = "panel",
    variant: str = "base",
    method: str = "pgd_at",
    teacher_kind: str = _FIXTURE_TEACHER,
) -> tuple[Trainer, Any]:
    # Every RNG stream a checkpoint records (Python, NumPy, torch CPU/CUDA), seeded as ard.cli.train
    # does: each arm runs in a fresh process, so an unseeded stream would differ between arms.
    _seed_everything(1234)
    options = _variant(variant)
    student, num_classes, image_size = _student(kind, teacher_kind=teacher_kind)
    student = student.to(device)
    optimizer = _optimizer(student, options, sgd_learning_rate=0.05, sgd_weight_decay=5e-4)
    parts = _method_parts(
        method,
        num_classes=num_classes,
        train_size=_TRAIN_SIZE,
        epochs=_EPOCHS,
        teacher_kind=teacher_kind,
        attack={"epsilon": "4/255", "step_size": "4/255", "steps": 2},
    )
    parts.setdefault(
        "objective", PGDATObjective(label_smoothing=_LABEL_SMOOTHING if "label_smoothing" in options else 0.0)
    )
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=MultiStepLR(optimizer, milestones=[1, 2], gamma=0.1),
        scaler=None,
        selection_attack=LinfPGD(
            AttackConfig(epsilon="4/255", step_size="1/255", steps=2, student_mode="eval", teacher_mode="eval")
        ),
        **parts,
        weight_ema_decay=_EMA_DECAY if "ema" in options else None,
        device=device,
        output_dir=output,
        config_hash="c" * 64,
        seed=11,
        tracker_run_id="cuda-graph",
        diagnostics=(
            None
            if diagnostics == "none"
            else TrainingDiagnostics.for_ids(list(range(_TRAIN_SIZE)), seed=0, size=5, mode=diagnostics)  # type: ignore[arg-type]
        ),
        step_diagnostics=False,
        cuda_graph=cuda_graph,
        student_architecture=kind,
    )
    return trainer, (num_classes, image_size)


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    _canonical(value, digest)
    return digest.hexdigest()


def _fingerprint(trainer: Trainer, history: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows = [
        {
            key: float(value).hex()
            for key, value in sorted(row.items())
            if not key.endswith(_TIMING_SUFFIXES) and key not in _AUDIT_KEYS
        }
        for row in history
    ]
    result: dict[str, Any] = {"rows": rows}
    for name in ("last.pt", "best.pt"):
        payload = torch.load(output / name, map_location="cpu", weights_only=False)
        for key in (*_STATE_KEYS, "scheduler"):
            if key in payload:
                result[f"{name}:{key}"] = _digest(payload[key])
    if trainer.diagnostics is not None:
        result["diagnostics_rows"] = _digest(trainer.diagnostics.all_rows)
        result["diagnostics_panel"] = _digest(trainer.diagnostics.panel_rows)
    result["global_step"] = trainer.global_step
    result["cuda_rng"] = _digest(torch.cuda.get_rng_state())
    return result


def _install_method_control(method: str) -> None:
    """``graph_control``: a defect confined to the graph path of ``method``; the run must then differ.

    Bank methods: the bank rows copied into the static buffers belong to the next sample (rolled by one).
    rslad_online: the in-step teacher sees a slightly shifted clean batch. mixed_batch: the graph attacks
    k + 1 positions (and weights them as adversarial). awp: the graph never perturbs the weights.
    pgd_at: the graph's random start is negated.
    """
    if method in _BANK_METHODS:
        original_rows = SoftLabelBankTeacher.clean_rows

        def rolled(self: SoftLabelBankTeacher, batch: Any, *, epoch: int) -> Any:
            return tuple(row.roll(1, dims=0) for row in original_rows(self, batch, epoch=epoch))

        SoftLabelBankTeacher.clean_rows = rolled  # type: ignore[method-assign]
    elif method == "rslad_online":
        original_forward = TeacherAdapter.forward

        def shifted_forward(self: TeacherAdapter, pixels: torch.Tensor) -> torch.Tensor:
            if torch.cuda.is_current_stream_capturing():
                pixels = (pixels * 0.99).detach()
            return original_forward(self, pixels)

        TeacherAdapter.forward = shifted_forward  # type: ignore[method-assign]
    elif method == "mixed_batch":
        original_count = Trainer._cuda_graph_mixed_count

        def more(self: Trainer, batch_size: int) -> int | None:
            count = original_count(self, batch_size)
            return None if count is None else count + 1

        Trainer._cuda_graph_mixed_count = more  # type: ignore[method-assign]
    elif method == "awp":
        original_step = Trainer._cuda_graph_train_step

        def without_awp(self: Trainer, batch: Any) -> bool:
            assert self._cuda_graph is not None
            self._cuda_graph.awp_active = False
            return original_step(self, batch)

        Trainer._cuda_graph_train_step = without_awp  # type: ignore[method-assign]
    else:
        original_body = Trainer._cuda_graph_body

        def negated(self: Trainer) -> None:
            assert self._cuda_graph is not None and self._cuda_graph.noise is not None
            self._cuda_graph.noise.neg_()
            original_body(self)

        Trainer._cuda_graph_body = negated  # type: ignore[method-assign]


def arm(
    root: Path,
    kind: str,
    mode: str,
    diagnostics: str,
    variant: str = "base",
    method: str = "pgd_at",
    teacher_kind: str = _FIXTURE_TEACHER,
) -> dict[str, Any]:
    """One arm in this (fresh) process: eager | graph | graph_seed_plus_one | graph_control | resumed."""
    device = torch.device("cuda")
    if mode == "graph_control":
        _install_method_control(method)
    if mode == "graph_seed_plus_one":
        # Negative control: the graph path draws from a different attack stream.
        original = Trainer._attack_generator

        def shifted(self: Trainer) -> torch.Generator:
            generator = original(self)
            return generator.manual_seed(generator.initial_seed() + 1)

        Trainer._attack_generator = shifted  # type: ignore[method-assign]
    if mode == "graph_adamw_step_plus_one":
        # Negative control (AdamW): the captured step also advances every AdamW step counter once more,
        # so the bias corrections of every replay are those of a later step. Confined to the graph path.
        original_body = Trainer._cuda_graph_body

        def advanced(self: Trainer) -> None:
            for state in self.optimizer.state.values():
                state["step"].add_(1.0)
            original_body(self)

        Trainer._cuda_graph_body = advanced  # type: ignore[method-assign]
    cuda_graph = mode != "eager"
    common = {"diagnostics": diagnostics, "variant": variant, "method": method, "teacher_kind": teacher_kind}
    trainer, (num_classes, image_size) = build_trainer(root, kind=kind, cuda_graph=cuda_graph, device=device, **common)
    post_seed_cuda_rng = _digest(torch.cuda.get_rng_state())
    crop_keyed = method in _BANK_METHODS
    loader, validation_loader, probe_loader = _loaders(num_classes, image_size, pin_memory=True, crop_keyed=crop_keyed)
    loaders = {"validation_loader": validation_loader, "probe_loader": probe_loader}
    if mode == "resumed":
        history = trainer.fit(loader, epochs=1, **loaders)
        del trainer
        trainer, _ = build_trainer(root, kind=kind, cuda_graph=True, device=device, **common)
        loader, validation_loader, probe_loader = _loaders(
            num_classes, image_size, pin_memory=True, crop_keyed=crop_keyed
        )
        start = trainer.resume(root / "last.pt", sampler=loader.sampler).next_epoch
        assert start == 1
        history += trainer.fit(
            loader, validation_loader=validation_loader, probe_loader=probe_loader, epochs=_EPOCHS, start_epoch=start
        )
    else:
        history = trainer.fit(loader, epochs=_EPOCHS, **loaders)
    fingerprint = _fingerprint(trainer, history, root)
    fingerprint["cuda_rng_advanced"] = fingerprint["cuda_rng"] != post_seed_cuda_rng
    fingerprint["audit"] = [[row.get(key) for key in _AUDIT_KEYS] for row in history]
    fingerprint["has_probe"] = all("train_probe_pgd_accuracy" in row for row in history)
    fingerprint["has_ema"] = all("val_pgd_accuracy_ema" in row for row in history)
    if trainer.ema_model is not None:
        # The EMA really averaged: it differs from the live student at the end.
        live = trainer.model.state_dict()
        fingerprint["ema_differs_from_model"] = any(
            not torch.equal(value, live[key]) for key, value in trainer.ema_model.state_dict().items()
        )
    fingerprint["optimizer_groups"] = [group["weight_decay"] for group in trainer.optimizer.param_groups]
    fingerprint["optimizer"] = type(trainer.optimizer).__name__
    fingerprint["optimizer_capturable"] = [group.get("capturable") for group in trainer.optimizer.param_groups]
    # Evidence the method's own step ran (method columns of the epoch rows).
    fingerprint["method_rows"] = [
        {
            key: row[key]
            for key in (
                "train_teacher_clean_forward_calls",
                "train_teacher_adversarial_forward_calls",
                "train_awp_active",
                "train_mixed_batch_adversarial_examples",
                "train_mixed_batch_clean_examples",
            )
            if key in row
        }
        | {key: True for key in row if key.startswith("train_advt_")}
        for row in history
    ]
    fingerprint["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return fingerprint


def nondeterministic_arm(
    root: Path, kind: str, mode: str, variant: str = "base", method: str = "pgd_at"
) -> dict[str, Any]:
    """``arm`` under training.deterministic=false, plus the attack-generator stream.

    Records every attack generator the run creates (seed and final philox
    state, i.e. how far it was drawn): kernel nondeterminism must not touch
    any RNG stream, so these, the checkpointed RNG states and the CUDA RNG
    state must match the eager run exactly even here.
    """
    generators: list[torch.Generator] = []
    original = Trainer._attack_generator

    def recorded(self: Trainer) -> torch.Generator:
        generator = original(self)
        generators.append(generator)
        return generator

    Trainer._attack_generator = recorded  # type: ignore[method-assign]
    fingerprint = arm(root, kind, mode, "panel", variant, method)
    fingerprint["attack_generators"] = _digest(
        [(generator.initial_seed(), generator.get_state()) for generator in generators]
    )
    fingerprint["attack_generator_count"] = len(generators)
    fingerprint["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    fingerprint["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    return fingerprint


class _StopEpoch(Exception):
    """Raised from on_batch_start once the single-step comparison has what it needs."""


# Batch indices of epoch 0 at whose start every arm is set to the reference
# run's exact state: batch 1 is the capture step (captured, then replayed) and
# batch 3 a later replay. The state after that one step is read at the start
# of the next batch.
_SYNC_BATCHES = (1, 3)
# Graph arms with a deliberate defect confined to one step, set after the sync
# and before the capture step (so the stale-graph guard does not see it). Each
# must FAIL the equivalence rule: they show it resolves parameter-update errors.
_SINGLE_STEP_CONTROLS = ("attack_seed_plus_one", "lr_zero", "lr_times_1p001", "weight_decay_zero")


def _single_step_trainer(
    spec: dict[str, Any], output: Path, *, cuda_graph: bool
) -> tuple[Trainer, DataLoader, DataLoader]:
    """A trainer (``spec["method"]``, default PGD-AT) and a five-batch synthetic loader for the one-step
    comparison."""
    _seed_everything(1234)
    device = torch.device("cuda")
    kind = spec["kind"]
    if kind == _FIXTURE:
        student, num_classes, _ = _student(kind)
    else:
        _admit_candidate(kind)
        num_classes = spec["num_classes"]
        config = ModelConfig(
            architecture=kind,  # type: ignore[arg-type]
            num_classes=num_classes,
            pretrained=False,
            normalization=NormalizationConfig(profile="imagenet_standard"),
        )
        student = build_student(config, tier="dev")
    if spec.get("checkpoint"):
        student.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)["model"])
    student = student.to(device)
    options = _variant(spec.get("variant", "base"))
    optimizer = _optimizer(
        student,
        options,
        sgd_learning_rate=spec["learning_rate"],
        sgd_weight_decay=spec["weight_decay"],
        adamw_learning_rate=spec.get("adamw_learning_rate", _ADAMW_LEARNING_RATE),
    )
    size = 5 * spec["batch"]
    method = spec.get("method", "pgd_at")
    parts = _method_parts(
        method,
        num_classes=num_classes,
        train_size=size,
        epochs=1,
        teacher_kind=_FIXTURE_TEACHER,
        attack={"epsilon": spec["epsilon"], "step_size": spec["step_size"], "steps": spec["steps"]},
    )
    if parts.get("awp") is not None:
        # Active in the single epoch, at the production (AT-AWP default) gamma.
        parts["awp"] = AwpConfig(gamma=0.01, warmup_epochs=0)
    parts.setdefault(
        "objective", PGDATObjective(label_smoothing=_LABEL_SMOOTHING if "label_smoothing" in options else 0.0)
    )
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        selection_attack=LinfPGD(
            AttackConfig(epsilon=spec["epsilon"], step_size="1/255", steps=1, student_mode="eval", teacher_mode="eval")
        ),
        **parts,
        weight_ema_decay=_EMA_DECAY if "ema" in options else None,
        device=device,
        output_dir=output,
        config_hash="c" * 64,
        seed=11,
        tracker_run_id="single-step",
        diagnostics=TrainingDiagnostics.for_ids(list(range(size)), seed=0, size=5, mode="panel"),
        step_diagnostics=False,
        cuda_graph=cuda_graph,
        student_architecture=kind,
    )

    def loader(dataset_size: int, seed: int, *, keyed: bool = False) -> DataLoader:
        dataset = IndexedDataset(
            SyntheticCIFAR(size=dataset_size, num_classes=num_classes, image_size=spec["image_size"], seed=seed)
        )
        return DataLoader(
            _CropKeyed(dataset) if keyed else dataset,
            batch_size=spec["batch"],
            sampler=EpochShuffleSampler(dataset_size, seed=5, shuffle=True),
            pin_memory=True,
            collate_fn=collate_crop_keyed if keyed else collate_indexed,
        )

    return trainer, loader(size, 3, keyed=method in _BANK_METHODS), loader(spec["batch"], 99)


def _optimizer_state_groups(optimizer: torch.optim.Optimizer) -> dict[str, str]:
    """Tensor group name -> per-parameter optimizer state key compared by the one-step rule."""
    if type(optimizer) is AdamW:
        return {"exp_avg": "exp_avg", "exp_avg_sq": "exp_avg_sq"}
    return {"momentum": "momentum_buffer"}


def _single_step_state(trainer: Trainer) -> dict[str, Any]:
    """Parameters, model buffers, the optimizer state (SGD momentum buffers; AdamW exp_avg, exp_avg_sq and step
    counters), the EMA state (when the variant has one) and the CUDA RNG state, copied to the host."""
    names = {name for name, _ in trainer.model.named_parameters()}
    state = trainer.model.state_dict()
    parameters = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
    saved = {
        "parameters": {key: value.detach().cpu().clone() for key, value in state.items() if key in names},
        "buffers": {key: value.detach().cpu().clone() for key, value in state.items() if key not in names},
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    for group, key in _optimizer_state_groups(trainer.optimizer).items():
        saved[group] = [trainer.optimizer.state[p][key].detach().cpu().clone() for p in parameters]
    if type(trainer.optimizer) is AdamW:
        saved["step"] = [trainer.optimizer.state[p]["step"].detach().cpu().clone() for p in parameters]
    if trainer.ema_model is not None:
        saved["ema"] = {key: value.detach().cpu().clone() for key, value in trainer.ema_model.state_dict().items()}
    return saved


def _load_single_step_state(trainer: Trainer, saved: dict[str, Any]) -> None:
    """Overwrite the trainer's state in place: tensor addresses, and so a captured graph, stay valid."""
    state = trainer.model.state_dict()
    parameters = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
    with torch.no_grad():
        for key, value in state.items():
            value.copy_(saved["parameters"][key] if key in saved["parameters"] else saved["buffers"][key])
        for group, key in _optimizer_state_groups(trainer.optimizer).items():
            for parameter, buffer in zip(parameters, saved[group], strict=True):
                trainer.optimizer.state[parameter][key].copy_(buffer)
        for parameter, step in zip(parameters, saved.get("step", []), strict=False):
            trainer.optimizer.state[parameter]["step"].copy_(step)
        if trainer.ema_model is not None:
            for key, value in trainer.ema_model.state_dict().items():
                value.copy_(saved["ema"][key])
    torch.cuda.set_rng_state(saved["cuda_rng"])


def single_step_arm(root: Path, spec_json: str, name: str) -> dict[str, Any]:
    """One process of the one-step comparison.

    ``reference`` (eager) saves its exact state at the start of each batch in
    _SYNC_BATCHES and its state after that one step. Every other process runs
    two eager arms (``eager_a``, ``eager_b``) and then either two graph arms
    (``graph_a``, ``graph_b``; process name ``pair<i>``) or one graph arm per
    _SINGLE_STEP_CONTROLS entry (process ``controls``). Each arm loads the
    reference's state in place at both sync points and saves its state after
    the step as ``<process>.<arm>_after<sync>.pt``. A control's defect is set
    once, before the capture step, so it is baked into the graph and also
    present in the later replay (changing it there would trip the stale-graph
    guard instead).
    """
    spec = json.loads(spec_json)
    state_dir = Path(spec["state_dir"])
    original_generator = Trainer._attack_generator

    def run_one(arm_name: str, *, cuda_graph: bool, control: str | None = None) -> None:
        trainer, loader, validation_loader = _single_step_trainer(spec, root / arm_name, cuda_graph=cuda_graph)

        def hook(epoch: int, batch_index: int, _batch: Any) -> None:
            if batch_index - 1 in _SYNC_BATCHES:
                torch.save(_single_step_state(trainer), state_dir / f"{arm_name}_after{batch_index - 1}.pt")
            if batch_index in _SYNC_BATCHES and arm_name == "reference":
                torch.save(_single_step_state(trainer), state_dir / f"sync{batch_index}.pt")
            elif batch_index in _SYNC_BATCHES:
                _load_single_step_state(trainer, torch.load(state_dir / f"sync{batch_index}.pt", weights_only=False))
                for group in trainer.optimizer.param_groups:
                    if batch_index == _SYNC_BATCHES[0] and control == "lr_zero":
                        group["lr"] = 0.0
                    elif batch_index == _SYNC_BATCHES[0] and control == "lr_times_1p001":
                        group["lr"] *= 1.001
                    elif batch_index == _SYNC_BATCHES[0] and control == "weight_decay_zero":
                        group["weight_decay"] = 0.0
            if batch_index > max(_SYNC_BATCHES):
                raise _StopEpoch

        if control == "attack_seed_plus_one":

            def shifted(self: Trainer) -> torch.Generator:
                generator = original_generator(self)
                return generator.manual_seed(generator.initial_seed() + 1)

            Trainer._attack_generator = shifted  # type: ignore[method-assign]
        try:
            trainer.fit(loader, validation_loader=validation_loader, epochs=1, on_batch_start=hook)
        except _StopEpoch:
            pass
        finally:
            Trainer._attack_generator = original_generator  # type: ignore[method-assign]
        if cuda_graph:
            assert trainer._cuda_graph is not None
            counts = (trainer._cuda_graph.captures_this_epoch, trainer._cuda_graph.replays_this_epoch)
            assert counts == (1, 3), (arm_name, counts)

    def run_and_release(arm_name: str, *, cuda_graph: bool, control: str | None = None) -> None:
        run_one(arm_name, cuda_graph=cuda_graph, control=control)
        # The arm's trainer, graph pool and buffers are unreachable now: free them before the next arm.
        gc.collect()
        torch.cuda.empty_cache()

    if name == "reference":
        run_and_release(name, cuda_graph=False)
    else:
        run_and_release(f"{name}.eager_a", cuda_graph=False)
        run_and_release(f"{name}.eager_b", cuda_graph=False)
        if name == "controls":
            for control in _SINGLE_STEP_CONTROLS:
                run_and_release(f"{name}.{control}", cuda_graph=True, control=control)
        else:
            run_and_release(f"{name}.graph_a", cuda_graph=True)
            run_and_release(f"{name}.graph_b", cuda_graph=True)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _flat(tensors: Any) -> torch.Tensor:
    values = tensors.values() if isinstance(tensors, dict) else tensors
    floating = [value.double().flatten() for value in values if value.is_floating_point()]
    return torch.cat(floating) if floating else torch.zeros(0, dtype=torch.float64)


_GROUPS = ("parameters", "momentum", "buffers")
# With a weight EMA (variant "ema") its state is a fourth group, held to the same rule. With AdamW the
# optimizer state is two groups (exp_avg, exp_avg_sq) instead of the momentum buffers.
_ALL_STATE_GROUPS = ("parameters", "momentum", "exp_avg", "exp_avg_sq", "buffers", "ema")


def _groups(result: dict[str, Any]) -> tuple[str, ...]:
    return tuple(result["floor"])


def single_step_distances(state_dir: Path, processes: list[str], sync: int) -> dict[str, Any]:
    """Every distance the one-step rule needs, per tensor group (parameters, momentum buffers,
    BatchNorm buffers), each relative to the reference's own update of that group: from every eager
    outcome (each process's eager_a / eager_b) to its nearest other eager outcome, and from every graph
    and control arm to every eager outcome. Also the FP32 floor (one rounding of every element of the
    group's new value, on the same scale), the cross-process difference and reference sanity."""
    before = torch.load(state_dir / f"sync{sync}.pt", weights_only=False)
    states = {"reference": torch.load(state_dir / f"reference_after{sync}.pt", weights_only=False)}
    for path in sorted(state_dir.glob(f"*.*_after{sync}.pt")):
        if path.name.split(".", 1)[0] in processes:
            states[path.name[: -len(f"_after{sync}.pt")]] = torch.load(path, weights_only=False)
    reference = states["reference"]
    present = [group for group in _ALL_STATE_GROUPS if group in reference]
    flat = {name: {group: _flat(state[group]) for group in present} for name, state in states.items()}
    result: dict[str, Any] = {"update": {}, "floor": {}, "relative_update": {}}
    # A group the reference step does not change at all (no floating-point element, or constant buffers such
    # as a LayerNorm model's pixel-normalization constants) has no scale: it must be bitwise equal in every
    # arm instead, and is left out of the distance rule.
    constant = [
        group
        for group in present
        if flat["reference"][group].numel() == 0 or torch.equal(flat["reference"][group], _flat(before[group]))
    ]
    result["constant_groups_equal"] = {
        group: all(torch.equal(flat[name][group], flat["reference"][group]) for name in flat) for group in constant
    }
    groups = [group for group in present if group not in constant]
    # AdamW step counters: exact in every arm (integers stored as float32).
    result["steps_equal"] = all(
        len(state.get("step", [])) == len(reference.get("step", []))
        and all(torch.equal(a, b) for a, b in zip(state.get("step", []), reference.get("step", []), strict=True))
        for state in states.values()
    )
    for group in groups:
        new, old = flat["reference"][group], _flat(before[group])
        update = float((new - old).norm())
        result["update"][group] = update
        result["relative_update"][group] = update / float(old.norm())
        result["floor"][group] = torch.finfo(torch.float32).eps * float(new.norm()) / update

    def distance(left: str, right: str) -> dict[str, float]:
        return {
            group: float((flat[left][group] - flat[right][group]).norm()) / result["update"][group] for group in groups
        }

    # The reference only supplies the state and the scale: it is alone in its process, and a
    # different process can make a different cuDNN algorithm choice.
    eager = [name for name in states if name.split(".")[-1].startswith("eager")]
    others = [name for name in states if name not in eager and name != "reference"]
    result["eager_nearest"] = {
        name: {group: min(distance(name, other)[group] for other in eager if other != name) for group in groups}
        for name in eager
    }
    result["to_eager"] = {name: {other: distance(name, other) for other in eager} for name in others}
    result["cross_process"] = {
        group: max(distance(f"{process}.eager_a", "reference")[group] for process in processes) for group in groups
    }
    result["integer_buffers_equal"] = all(
        torch.equal(state["buffers"][key], value)
        for state in states.values()
        for key, value in reference["buffers"].items()
        if not value.is_floating_point()
    )
    running_vars = [value for key, value in reference["buffers"].items() if key.endswith("running_var")]
    result["max_running_var"] = max((float(value.max()) for value in running_vars), default=0.0)
    result["finite"] = all(
        bool(torch.isfinite(flat["reference"][group]).all()) and bool(torch.isfinite(_flat(before[group])).all())
        for group in groups
    )
    return result


# Stale-graph mutations of the 2026-10-08 scope and the method each needs.
_METHOD_MUTATIONS = {
    "bank_buffer": "rslad_bank",
    "teacher_weight": "rslad_advt_bank",
    "teacher_mode": "rslad_online",
    "awp_proxy_tensor": "awp",
    "awp_gamma": "awp",
    "mixed_weight": "mixed_batch",
}


# Stale-graph mutations of the AdamW scope (2026-10-09): every AdamW hyperparameter the step bakes in (per
# group) and every per-parameter state tensor it reads or writes by address.
_ADAMW_MUTATIONS = (
    "adamw_lr",
    "adamw_beta1",
    "adamw_beta2",
    "adamw_eps",
    "adamw_weight_decay_group1",
    "adamw_exp_avg",
    "adamw_exp_avg_sq",
    "adamw_step_tensor",
    "adamw_load_state_dict",
)


def stale_guard(root: Path, mutation: str) -> str:
    """Mutate the optimizer (or another baked-in input of the step) mid-epoch after capture; return the error."""
    device = torch.device("cuda")
    method = _METHOD_MUTATIONS.get(mutation, "pgd_at")
    variant = "base"
    if mutation == "ema_tensor":
        variant = "ema"
    elif mutation in _ADAMW_MUTATIONS:
        variant = "adamw"
    trainer, (num_classes, image_size) = build_trainer(
        root,
        kind=_FIXTURE,
        cuda_graph=True,
        device=device,
        variant=variant,
        method=method,
    )
    loader, validation_loader, _ = _loaders(
        num_classes, image_size, pin_memory=False, crop_keyed=method in _BANK_METHODS
    )

    def mutate(epoch: int, batch_index: int, _batch: Any) -> None:
        if epoch != 0 or batch_index != 3:  # batch 0 eager, 1 captured and replayed, 2 replayed
            return
        optimizer = trainer.optimizer
        if mutation == "lr":
            optimizer.param_groups[0]["lr"] *= 0.5
        elif mutation == "momentum":
            optimizer.param_groups[0]["momentum"] = 0.8
        elif mutation == "weight_decay":
            optimizer.param_groups[0]["weight_decay"] = 0.0
        elif mutation == "load_state_dict":
            # A deep copy, as a checkpoint load gives: new momentum-buffer tensors.
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        elif mutation == "model_mode":
            trainer.model.eval()
        elif mutation == "ema_tensor":
            # A new EMA tensor (as a non-in-place EMA load would give): the graph writes the old one.
            assert trainer.ema_model is not None
            batchnorm = next(module for module in trainer.ema_model.modules() if isinstance(module, nn.BatchNorm2d))
            batchnorm.register_buffer("running_mean", batchnorm.running_mean.clone())
        elif mutation == "bank_buffer":
            # A reallocated static bank buffer: the graph reads the old one.
            state = trainer._cuda_graph
            assert state is not None and state.bank_prob is not None
            state.bank_prob = state.bank_prob.clone()
        elif mutation == "teacher_weight":
            # A teacher reload that is not in place: the graph reads the old weights.
            assert isinstance(trainer.teacher, SoftLabelBankTeacher) and trainer.teacher.online_teacher is not None
            layer = next(m for m in trainer.teacher.online_teacher.modules() if isinstance(m, nn.Conv2d))
            layer.weight = nn.Parameter(layer.weight.detach().clone(), requires_grad=False)
        elif mutation == "teacher_mode":
            assert trainer.teacher is not None
            nn.Module.train(trainer.teacher, True)  # bypass the adapter's eval-only override
        elif mutation == "awp_proxy_tensor":
            assert trainer._awp is not None
            layer = next(m for m in trainer._awp.proxy.modules() if isinstance(m, nn.BatchNorm2d))
            layer.register_buffer("running_var", layer.running_var.clone())
        elif mutation == "awp_gamma":
            assert trainer._awp is not None
            trainer._awp.gamma *= 2.0
        elif mutation == "mixed_weight":
            assert trainer.mixed_batch is not None
            trainer.mixed_batch = trainer.mixed_batch.model_copy(update={"adversarial_weight": 0.5})
        elif mutation in _ADAMW_MUTATIONS:
            assert type(optimizer) is AdamW
            first = optimizer.param_groups[0]["params"][0]
            if mutation == "adamw_lr":
                optimizer.param_groups[1]["lr"] *= 0.5
            elif mutation == "adamw_beta1":
                optimizer.param_groups[0]["betas"] = (0.8, optimizer.param_groups[0]["betas"][1])
            elif mutation == "adamw_beta2":
                optimizer.param_groups[0]["betas"] = (optimizer.param_groups[0]["betas"][0], 0.99)
            elif mutation == "adamw_eps":
                optimizer.param_groups[0]["eps"] = 1e-6
            elif mutation == "adamw_weight_decay_group1":
                # The norm/bias group (weight decay 0) gets decay: a per-group baked-in scalar.
                optimizer.param_groups[1]["weight_decay"] = 0.05
            elif mutation in ("adamw_exp_avg", "adamw_exp_avg_sq", "adamw_step_tensor"):
                # A reallocated state tensor (same values): the graph would write the old one.
                key = {"adamw_exp_avg": "exp_avg", "adamw_exp_avg_sq": "exp_avg_sq", "adamw_step_tensor": "step"}
                optimizer.state[first][key[mutation]] = optimizer.state[first][key[mutation]].clone()
            else:
                # A deep copy, as a checkpoint load gives: new exp_avg / exp_avg_sq / step tensors.
                optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        else:
            raise AssertionError(mutation)

    try:
        trainer.fit(loader, validation_loader=validation_loader, epochs=1, on_batch_start=mutate)
    except RuntimeError as error:
        assert trainer._cuda_graph is not None and trainer._cuda_graph.replays_this_epoch == 2
        return str(error)
    return "no error"


def never_replays(root: Path) -> str:
    """cuda_graph on, but no capture is ever possible: the epoch must fail, not silently train eagerly."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(root, kind=_FIXTURE, cuda_graph=True, device=device)
    trainer_module.optimizer_state_ready = lambda _optimizer: False  # type: ignore[assignment]
    loader, validation_loader, _ = _loaders(num_classes, image_size, pin_memory=False)
    try:
        trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    except RuntimeError as error:
        return f"{error} | last.pt exists: {(root / 'last.pt').exists()}"
    return "no error"


def poisoned_replay(root: Path, poison: str) -> None:
    """Capture on clean data, then replay with poisoned static inputs, bypassing the host-side guard.

    ``pixel``: the attack clamps its output into [0, 1], so inside the graph
    only the diagnostics clean forward (panel diagnostics here) sees the raw
    batch and trips the model adapter's assert. ``nan``: NaN passes both
    [0, 1] guards (every comparison is false) and trips the finite-loss assert.
    """
    device = torch.device("cuda")
    # Keep the epoch's graph alive past the epoch end (it is normally released there to free its pool).
    Trainer._release_cuda_graph = lambda self: None  # type: ignore[method-assign]
    method = "rslad_bank" if poison == "bank_nan" else "pgd_at"
    trainer, (num_classes, image_size) = build_trainer(
        root, kind=_FIXTURE, cuda_graph=True, device=device, method=method
    )
    loader, validation_loader, _ = _loaders(
        num_classes, image_size, pin_memory=False, crop_keyed=method in _BANK_METHODS
    )
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    state = trainer._cuda_graph
    assert state is not None and state.graph is not None and state.images is not None
    torch.cuda.synchronize()
    print("captured", flush=True)
    if poison == "pixel":
        state.images.fill_(1.5)
    elif poison == "nan":
        state.images.fill_(float("nan"))
    elif poison == "bank_nan":
        # A corrupt bank row: reconstruct_probabilities' check is a device assert inside the step.
        assert state.bank_prob is not None
        state.bank_prob.fill_(float("nan"))
    else:
        raise AssertionError(poison)
    state.graph.replay()
    torch.cuda.synchronize()
    print("replayed without a device assert", flush=True)


class _OutOfRange(torch.utils.data.Dataset):
    """Wrap a dataset; one late sample leaves the [0, 1] pixel box."""

    def __init__(self, inner: Any, bad_index: int) -> None:
        self.inner, self.bad_index = inner, bad_index

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: Any) -> Any:
        image, label, sample_id, *rest = self.inner[index]
        if sample_id == self.bad_index:
            image = image.clone()
            image[0, 0, 0] = 1.25
        return (image, label, sample_id, *rest)


def out_of_range_batch(root: Path) -> None:
    """A replayed batch with an out-of-range pixel: the process must die, nothing checkpointed."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(root, kind=_FIXTURE, cuda_graph=True, device=device)
    loader, validation_loader, _ = _loaders(num_classes, image_size, pin_memory=False)
    order = list(iter(loader.sampler))
    # Position 4*4 is in the fifth batch of epoch 0: a replayed step.
    bad = order[4 * _BATCH].index
    poisoned = DataLoader(
        _OutOfRange(loader.dataset, bad),
        batch_size=_BATCH,
        sampler=loader.sampler,
        collate_fn=collate_indexed,
    )
    try:
        trainer.fit(poisoned, validation_loader=validation_loader, epochs=1)
    finally:
        print("last.pt exists:", (root / "last.pt").exists(), flush=True)
    print("training completed without a device assert", flush=True)


def nan_bank_row(root: Path) -> None:
    """A replayed RSLAD-bank batch with one NaN stored probability (graph pool release active, nothing patched):
    the in-step reconstruction assert must kill the process before anything is checkpointed."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(
        root, kind=_FIXTURE, cuda_graph=True, device=device, method="rslad_bank"
    )
    loader, validation_loader, _ = _loaders(num_classes, image_size, pin_memory=False, crop_keyed=True)
    order = list(iter(loader.sampler))
    # Position 4*4 is in the fifth batch of epoch 0: a replayed step.
    bad = order[4 * _BATCH].index
    assert isinstance(trainer.teacher, SoftLabelBankTeacher)
    arrays = trainer.teacher.bank._epochs[0]
    arrays.prob[bad, 0] = np.float16("nan")
    try:
        trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    finally:
        print("last.pt exists:", (root / "last.pt").exists(), flush=True)
    print("training completed without a device assert", flush=True)


_SCRIPT = r"""
import json, os, sys, tempfile
from pathlib import Path
import torch
_determinism = os.environ.get("ARD_CUDA_GRAPH_TEST_DETERMINISM", "deterministic")
if _determinism == "deterministic":
    torch.use_deterministic_algorithms(True)
else:
    # training.deterministic=false, as ard.cli.train runs it (optionally with cudnn_benchmark=true).
    assert _determinism in ("nondeterministic", "nondeterministic_benchmark"), _determinism
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = _determinism == "nondeterministic_benchmark"
sys.path.insert(0, sys.argv[1])
import tests.integration.test_cuda_graph_training_step as module
name, args = sys.argv[2], sys.argv[3:]
with tempfile.TemporaryDirectory() as root:
    result = getattr(module, name)(Path(root), *args)
    print("RESULT " + json.dumps(result), flush=True)
"""


def _run(
    name: str, *args: str, timeout: int = 900, determinism: str = "deterministic"
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(_ROOT / "src"), str(_ROOT)])
    environment["ARD_CUDA_GRAPH_TEST_DETERMINISM"] = determinism
    # Production runs (ard.cli.train under the launchers) set no cuBLAS
    # workspace override, deterministic or not, and torch 2.11 neither errors
    # nor warns without it; the proof runs in that same setting.
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(_ROOT), name, *args],
        cwd=_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _result(completed: subprocess.CompletedProcess[str]) -> Any:
    assert completed.returncode == 0, completed.stderr[-4000:]
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0].removeprefix("RESULT "))


def _comparable(fingerprint: dict[str, Any], *, diagnostics: bool = True) -> dict[str, Any]:
    ignored = {
        "cuda_rng_advanced",
        "audit",
        "has_probe",
        "has_ema",
        "ema_differs_from_model",
        "optimizer_groups",
        "optimizer",
        "optimizer_capturable",
        "method_rows",
        "peak_reserved_gib",
    }
    return {
        key: value
        for key, value in fingerprint.items()
        if key not in ignored and (diagnostics or not key.startswith("diagnostics_"))
    }


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
_FULL_BATCHES = _TRAIN_SIZE // _BATCH
# Per epoch: one capture, (full - 1) replays, 2 eager steps (warm-up + partial).
_EXPECTED_AUDIT = [[1.0, float(_FULL_BATCHES - 1), 2.0]] * _EPOCHS
_PARITY_CASES = [
    (_FIXTURE, "panel"),
    (_FIXTURE, "summary"),
    (_FIXTURE, "none"),
    *((architecture, "panel") for architecture in sorted(CUDA_GRAPH_ARCHITECTURES)),
    *(pytest.param(architecture, "panel", marks=_candidate_opt_in) for architecture in _CANDIDATE_ARCHITECTURES),
]


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "diagnostics"), _PARITY_CASES)
def test_graph_step_is_bit_identical_to_the_eager_step(kind: str, diagnostics: str) -> None:
    eager = _result(_run("arm", kind, "eager", diagnostics))
    graph = _result(_run("arm", kind, "graph", diagnostics))
    assert eager["has_probe"] and graph["has_probe"]
    assert "last.pt:rng" in eager and "best.pt:model" in eager and "last.pt:scheduler" in eager
    # The graph path really ran, and the run records it for audit.
    assert graph["audit"] == _EXPECTED_AUDIT
    assert eager["audit"] == [[None, None, None]] * _EPOCHS
    assert _comparable(graph) == _comparable(eager)
    if kind == _FIXTURE:
        # Dropout drew from the default CUDA generator inside the graph.
        assert eager["cuda_rng_advanced"] and graph["cuda_rng_advanced"]


# Plan 0103 Phase 2 batch A: each option alone on the fixture, all three together on every allowlisted
# architecture.
_VARIANT_PARITY_CASES = [
    (_FIXTURE, "ema"),
    (_FIXTURE, "label_smoothing"),
    (_FIXTURE, "exclude_norm_bias"),
    *((architecture, _ALL_OPTIONS) for architecture in sorted(CUDA_GRAPH_ARCHITECTURES)),
]


def _assert_variant_ran(result: dict[str, Any], variant: str) -> None:
    options = _variant(variant)
    assert result["has_ema"] is ("ema" in options)
    if "ema" in options:
        assert result["ema_differs_from_model"]
        assert "last.pt:ema" in result and "best.pt:ema" in result
    if "adamw" in options:
        assert result["optimizer"] == "AdamW" and result["optimizer_capturable"] == [True, True]
        assert result["optimizer_groups"] == [_ADAMW_WEIGHT_DECAY, 0.0]
    else:
        assert result["optimizer"] == "SGD"
        assert result["optimizer_groups"] == ([5e-4, 0.0] if "exclude_norm_bias" in options else [5e-4])


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "variant"), _VARIANT_PARITY_CASES)
def test_graph_step_with_phase2_options_is_bit_identical_to_the_eager_step(kind: str, variant: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", variant))
    graph = _result(_run("arm", kind, "graph", "panel", variant))
    _assert_variant_ran(eager, variant)
    _assert_variant_ran(graph, variant)
    assert graph["audit"] == _EXPECTED_AUDIT
    assert _comparable(graph) == _comparable(eager)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "variant"), [(_FIXTURE, "ema"), ("mobilenetv4_conv_small_imagenet", _ALL_OPTIONS)])
def test_graph_run_with_phase2_options_resumed_mid_run_equals_the_eager_run(kind: str, variant: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", variant))
    resumed = _result(_run("arm", kind, "resumed", "panel", variant))
    _assert_variant_ran(resumed, variant)
    assert _comparable(resumed, diagnostics=False) == _comparable(eager, diagnostics=False)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", [_FIXTURE, "mobilenetv4_conv_small_imagenet"])
def test_graph_run_resumed_mid_run_equals_the_uninterrupted_eager_run(kind: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel"))
    resumed = _result(_run("arm", kind, "resumed", "panel"))
    # A resumed run's diagnostics tracker restarts, so compare training state and rows.
    assert _comparable(resumed, diagnostics=False) == _comparable(eager, diagnostics=False)


@pytest.mark.gpu
@requires_cuda
def test_negative_control_a_shifted_attack_stream_is_detected() -> None:
    eager = _result(_run("arm", _FIXTURE, "eager", "panel"))
    shifted = _result(_run("arm", _FIXTURE, "graph_seed_plus_one", "panel"))
    assert shifted["audit"] == _EXPECTED_AUDIT
    assert shifted["last.pt:model"] != eager["last.pt:model"]
    assert shifted["rows"] != eager["rows"]


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("mutation", ["lr", "momentum", "weight_decay", "load_state_dict", "model_mode", "ema_tensor"])
def test_mid_epoch_optimizer_change_is_refused_not_replayed(mutation: str) -> None:
    message = _result(_run("stale_guard", mutation))
    assert "CUDA graph is stale" in message


@pytest.mark.gpu
@requires_cuda
def test_an_epoch_that_never_replays_fails_loudly_before_any_checkpoint() -> None:
    message = _result(_run("never_replays"))
    assert "replayed no CUDA graph" in message
    assert message.endswith("last.pt exists: False")


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(
    ("poison", "expected"),
    [("pixel", "model adapter expects pixels in [0, 1]"), ("nan", "non-finite training loss")],
)
def test_in_graph_device_asserts_fire_from_a_replay(poison: str, expected: str) -> None:
    completed = _run("poisoned_replay", poison)
    assert "captured" in completed.stdout, completed.stderr[-4000:]
    assert "replayed without a device assert" not in completed.stdout
    assert completed.returncode != 0
    assert "device-side assert" in completed.stderr
    assert expected in completed.stderr


@pytest.mark.gpu
@requires_cuda
def test_out_of_range_pixels_on_the_graph_path_abort_before_any_checkpoint() -> None:
    completed = _run("out_of_range_batch")
    assert "training completed without a device assert" not in completed.stdout
    assert completed.returncode != 0
    assert "device-side assert" in completed.stderr
    # The host-launched copy of LinfPGD.generate's pixel check, ahead of the replay.
    assert "attack inputs must lie in pixel domain [0, 1]" in completed.stderr
    assert "last.pt exists: False" in completed.stdout


@pytest.mark.gpu
@requires_cuda
def test_a_nan_bank_row_in_a_replayed_batch_aborts_before_any_checkpoint() -> None:
    """Review of d2e82b2 (P3-2): end to end, with the per-epoch graph-pool release active."""
    completed = _run("nan_bank_row")
    assert "training completed without a device assert" not in completed.stdout
    assert completed.returncode != 0
    assert "device-side assert" in completed.stderr
    assert "bank row reconstructs to a non-finite or empty distribution" in completed.stderr
    assert "last.pt exists: False" in completed.stdout


@pytest.mark.gpu
@requires_cuda
def test_perturb_core_never_synchronizes_with_the_host() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    student = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(64, 3))
    student = student.to(device)
    attack = LinfPGD(AttackConfig(epsilon="8/255", step_size="2/255", steps=3))
    request = AttackRequest(
        inputs=torch.rand(6, 3, 4, 4, device=device),
        labels=torch.zeros(6, dtype=torch.long, device=device),
        student=student,
    )
    noise = torch.empty_like(request.inputs).uniform_(-1.0, 1.0)
    epsilon, step_size = attack.budgets(request)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attack.perturb(request, epsilon=epsilon, step_size=step_size, unit_noise=noise)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


# ======================================= human decision 2026-10-08: RSLAD / RSLAD-advT, mixed batch, AWP
_NEW_METHODS = ("rslad_bank", "rslad_online", "rslad_advt_bank", "mixed_batch", "awp")
_NEW_METHOD_CASES = [
    (kind, method) for kind in (_FIXTURE, *sorted(CUDA_GRAPH_ARCHITECTURES)) for method in _NEW_METHODS
]


def _assert_method_ran(result: dict[str, Any], method: str) -> None:
    """The epoch rows show the method's own step (and the graph replayed it)."""
    rows = result["method_rows"]
    assert len(rows) == _EPOCHS
    steps = float(_FULL_BATCHES + 1)
    for epoch, row in enumerate(rows):
        clean_calls = steps if method == "rslad_online" else 0.0
        adversarial_calls = steps if method == "rslad_advt_bank" else 0.0
        assert row["train_teacher_clean_forward_calls"] == clean_calls, (method, row)
        assert row["train_teacher_adversarial_forward_calls"] == adversarial_calls, (method, row)
        assert ("train_advt_teacher_adversarial_accuracy" in row) is (method == "rslad_advt_bank")
        if method == "awp":
            assert row["train_awp_active"] == (1.0 if epoch >= _AWP.warmup_epochs else 0.0)
        if method == "mixed_batch":
            # k = 2 of every full batch of 4, k = 1 of the partial batch of 2.
            assert row["train_mixed_batch_adversarial_examples"] == 2 * _FULL_BATCHES + 1
            assert row["train_mixed_batch_clean_examples"] == 2 * _FULL_BATCHES + 1


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "method"), _NEW_METHOD_CASES)
def test_graph_step_of_rslad_mixed_batch_and_awp_is_bit_identical_to_the_eager_step(kind: str, method: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", "base", method))
    graph = _result(_run("arm", kind, "graph", "panel", "base", method))
    _assert_method_ran(eager, method)
    _assert_method_ran(graph, method)
    assert eager["has_probe"] and graph["has_probe"]
    assert graph["audit"] == _EXPECTED_AUDIT
    assert _comparable(graph) == _comparable(eager)
    if kind == _FIXTURE and method in ("awp", "rslad_advt_bank", "rslad_online"):
        # Dropout drew from the default CUDA generator inside the graph (student; AWP proxy).
        assert eager["cuda_rng_advanced"] and graph["cuda_rng_advanced"]


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NEW_METHODS)
@pytest.mark.parametrize("kind", [_FIXTURE, "mobilenetv4_conv_small_imagenet"])
def test_graph_step_of_new_methods_with_the_phase2_weight_ema_is_bit_identical(kind: str, method: str) -> None:
    """Every Phase 2 config also sets training.weight_ema_decay: each new method together with the EMA."""
    eager = _result(_run("arm", kind, "eager", "panel", "ema", method))
    graph = _result(_run("arm", kind, "graph", "panel", "ema", method))
    _assert_method_ran(graph, method)
    _assert_variant_ran(eager, "ema")
    _assert_variant_ran(graph, "ema")
    assert graph["audit"] == _EXPECTED_AUDIT
    assert _comparable(graph) == _comparable(eager)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", ["rslad_online", "rslad_advt_bank"])
@pytest.mark.parametrize("teacher_kind", sorted(CUDA_GRAPH_TEACHER_ARCHITECTURES))
def test_in_step_teacher_forward_of_every_allowlisted_teacher_is_bit_identical(teacher_kind: str, method: str) -> None:
    """Every teacher architecture that may run inside the captured step (random weights, 1000 classes; the
    fixture student at 32 px, ViT-S at 224 px)."""
    eager = _result(_run("arm", _FIXTURE, "eager", "panel", "base", method, teacher_kind))
    graph = _result(_run("arm", _FIXTURE, "graph", "panel", "base", method, teacher_kind))
    _assert_method_ran(eager, method)
    assert graph["audit"] == _EXPECTED_AUDIT
    assert _comparable(graph) == _comparable(eager)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NEW_METHODS)
@pytest.mark.parametrize("kind", [_FIXTURE, "mobilenetv4_conv_small_imagenet"])
def test_new_method_graph_run_resumed_mid_run_equals_the_eager_run(kind: str, method: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", "base", method))
    resumed = _result(_run("arm", kind, "resumed", "panel", "base", method))
    _assert_method_ran(resumed, method)
    assert _comparable(resumed, diagnostics=False) == _comparable(eager, diagnostics=False)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NEW_METHODS)
def test_negative_control_a_graph_only_defect_of_each_new_method_is_detected(method: str) -> None:
    """Bank rows of the wrong sample, a shifted online-teacher input, k + 1 attacked positions, AWP off
    inside the graph: each confined to the graph path, each must change the run."""
    eager = _result(_run("arm", _FIXTURE, "eager", "panel", "base", method))
    control = _result(_run("arm", _FIXTURE, "graph_control", "panel", "base", method))
    assert control["audit"] == _EXPECTED_AUDIT
    assert control["last.pt:model"] != eager["last.pt:model"]
    assert control["rows"] != eager["rows"]


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("mutation", sorted(_METHOD_MUTATIONS))
def test_mid_epoch_change_of_a_new_baked_in_input_is_refused_not_replayed(mutation: str) -> None:
    message = _result(_run("stale_guard", mutation))
    assert "CUDA graph is stale" in message


@pytest.mark.gpu
@requires_cuda
def test_a_corrupt_bank_row_fires_the_in_graph_reconstruction_assert() -> None:
    completed = _run("poisoned_replay", "bank_nan")
    assert "captured" in completed.stdout, completed.stderr[-4000:]
    assert "replayed without a device assert" not in completed.stdout
    assert completed.returncode != 0
    assert "device-side assert" in completed.stderr
    assert "bank row reconstructs to a non-finite or empty distribution" in completed.stderr


# ============================================== training.deterministic=false (human decision 2026-10-03)
#
# Bitwise parity is impossible here: two eager nondeterministic runs already
# differ (cuDNN/cuBLAS kernels that accumulate with atomics sum in a
# run-dependent order). Two tests replace it, for every allowlisted student
# and the fixture, with and without cuDNN benchmark:
#
# 1. Whole runs (3 epochs, the parity test's run): everything kernel
#    nondeterminism cannot touch must still be EXACTLY equal between the
#    eager and graph runs -- every RNG stream (checkpointed Python / NumPy /
#    torch CPU / CUDA states, the final CUDA RNG state, the seed and final
#    philox state of every per-step attack generator), global step,
#    scheduler and sampler state -- and the audit counts must show the graph
#    ran. (Model weights are not compared here: PGD's sign step makes whole
#    nondeterministic runs drift apart chaotically, eager vs eager included.)
# 2. One step from one exact state: see run_single_step_check below.
_EXACT_UNDER_NONDETERMINISM = (
    "last.pt:rng",
    "cuda_rng",
    "cuda_rng_advanced",
    "attack_generators",
    "attack_generator_count",
    "global_step",
    "last.pt:scheduler",
    "last.pt:sampler_epoch",
    "last.pt:sampler_state",
    "has_probe",
)
# cuDNN benchmark stays refused with the graph: in this one-step test a captured
# step under benchmark once landed 0.27 of a step away from every eager outcome
# of its process (a different cuDNN algorithm choice). The script keeps the
# "nondeterministic_benchmark" mode so that finding can be reproduced (with the
# Trainer's benchmark refusal lifted locally).
_NONDETERMINISTIC_CASES = [
    (_FIXTURE, "nondeterministic"),
    *((architecture, "nondeterministic") for architecture in sorted(CUDA_GRAPH_ARCHITECTURES)),
    *(
        pytest.param(architecture, "nondeterministic", marks=_candidate_opt_in)
        for architecture in _CANDIDATE_ARCHITECTURES
    ),
]


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "determinism"), _NONDETERMINISTIC_CASES)
def test_nondeterministic_graph_run_keeps_every_rng_stream_exact(kind: str, determinism: str) -> None:
    eager = _result(_run("nondeterministic_arm", kind, "eager", determinism=determinism))
    graph = _result(_run("nondeterministic_arm", kind, "graph", determinism=determinism))
    for result in (eager, graph):
        assert result["deterministic_algorithms"] is False
        assert result["cudnn_benchmark"] is (determinism == "nondeterministic_benchmark")
        assert result["has_probe"]
    assert graph["audit"] == _EXPECTED_AUDIT
    assert eager["audit"] == [[None, None, None]] * _EPOCHS
    assert eager["attack_generator_count"] > 0
    for key in _EXACT_UNDER_NONDETERMINISM:
        assert graph[key] == eager[key], key
    if kind == _FIXTURE:
        # Dropout drew from the default CUDA generator inside the graph, and the stream still matches.
        assert eager["cuda_rng_advanced"]


# 2. One step from one exact state (single_step_arm): a reference eager
#    process saves its exact state (parameters, BatchNorm buffers, SGD
#    momentum buffers, CUDA RNG) before the capture step (batch 1) and before
#    a later replay (batch 3), and its state after each step. Other processes
#    load that state in place and take the same step: two eager arms and two
#    graph arms per ``pair`` process, and two eager arms plus four defective
#    graph arms (controls) in the ``controls`` process. Per tensor group --
#    parameters, momentum buffers, BatchNorm buffers -- distances are relative
#    to the reference's own update of that group, with a floor of one FP32
#    rounding of the group's new value on the same scale.
#    One step's nondeterminism can be bimodal (a summation-order difference
#    that flips the sign of an attack input gradient moves the whole step, in
#    eager and graph arms alike). So each graph outcome is compared with ONE
#    nearest eager outcome (the same eager arm for all groups), and the noise
#    level (spread) is the median over eager outcomes of the distance to their
#    nearest other one, so an eager arm alone in the other mode cannot widen
#    the bound. Rule, every group: graph <= _SAME_ORDER * max(spread, floor);
#    a spread above _MAX_SPREAD_OVER_FLOOR x the floor fails the test (the
#    check could not resolve a defect). Power: the four controls (attack seed
#    + 1, lr = 0, lr * 1.001, weight decay 0; set before the capture step and
#    so baked into the graph) must each be at least _CONTROL_MARGIN x the
#    bound away from every eager outcome, at BOTH sync points. Reference
#    sanity is asserted: finite, BatchNorm running variances <=
#    _MAX_RUNNING_VAR, parameter update <= _MAX_RELATIVE_PARAMETER_UPDATE of
#    the weights. The cross-process difference (eager_a vs the reference) is
#    reported, not used.
_SAME_ORDER = 4.0
_CONTROL_MARGIN = 10.0
_MAX_SPREAD_OVER_FLOOR = 100.0
_SINGLE_STEP_PAIR_PROCESSES = 2
_MAX_RUNNING_VAR = 1e4
_MAX_RELATIVE_PARAMETER_UPDATE = 0.05
# Unit-test-sized regime for every student. The parameter update must clear
# the FP32 floor (eps * |weights| / |update|) by enough that the lr * 1.001
# control is detectable, so the fixture (which keeps its 8 px input) and
# EfficientNet-B0 (smaller gradients) get a larger learning rate; every
# update stays around 1% of the weights.
_SINGLE_STEP_OVERRIDES: dict[str, dict[str, Any]] = {
    _FIXTURE: {"image_size": 8, "learning_rate": 0.05},
    "efficientnet_b0_imagenet": {"learning_rate": 0.015},
    # ConvNeXt-Atto / DeiT-Tiny family with SGD: at lr 0.002 the update stayed so close to the FP32 floor that
    # the lr x 1.001 control sat only 1.1-2.0x the bound (2026-10-09); lr 0.02 brings it to about 1% of the weights
    # (the convstem variants, 8.8x at lr 0.02: lr 0.04).
    **{architecture: {"learning_rate": 0.02} for architecture in _LAYERNORM_ARCHITECTURES},
    "convnext_atto_convstem_imagenet": {"learning_rate": 0.04},
    # DeiT-Tiny takes its native 224 px input only; batch 8 keeps every process small next to a production job.
    "deit_tiny_imagenet": {"image_size": 224, "batch": 8, "learning_rate": 0.02},
    "deit_tiny_convstem_imagenet": {"image_size": 224, "batch": 8, "learning_rate": 0.04},
}
_SINGLE_STEP_REGIME = {
    "image_size": 64,
    "batch": 32,
    "num_classes": 10,
    "learning_rate": 0.002,
    "weight_decay": 5e-4,
    "epsilon": "4/255",
    "step_size": "4/255",
    "steps": 2,
}


def run_single_step_check(state_dir: Path, spec: dict[str, Any], determinism: str) -> dict[str, Any]:
    """Run every process of the one-step comparison; return the per-group distances."""
    spec = {**spec, "state_dir": str(state_dir)}
    processes = [*(f"pair{index}" for index in range(_SINGLE_STEP_PAIR_PROCESSES)), "controls"]
    modes = set()
    peaks = []
    for name in ("reference", *processes):
        result = _result(_run("single_step_arm", json.dumps(spec), name, determinism=determinism))
        modes.add((result["deterministic_algorithms"], result["cudnn_benchmark"]))
        peaks.append(result["peak_reserved_gib"])
    assert modes == {(False, determinism == "nondeterministic_benchmark")}, modes
    distances: dict[str, Any] = {str(sync): single_step_distances(state_dir, processes, sync) for sync in _SYNC_BATCHES}
    for result in distances.values():
        result["peak_reserved_gib"] = max(peaks)
    return distances


def single_step_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Noise level, bound and, for every graph / control arm, its one common nearest eager outcome.

    Spread = the MEDIAN over eager outcomes of the distance to their nearest other eager outcome: an
    eager arm alone in the other mode of a bimodal step has no partner, and must not widen the bound
    to the gap between the modes. The nearest eager outcome is one arm for all groups: the one that
    minimises the largest group distance in units of the bound."""
    spread = {
        group: statistics.median(distances[group] for distances in result["eager_nearest"].values())
        for group in _groups(result)
    }
    bound = {group: _SAME_ORDER * max(spread[group], result["floor"][group]) for group in _groups(result)}
    arms = {}
    for name, to_eager in result["to_eager"].items():
        nearest = min(
            to_eager, key=lambda eager: max(to_eager[eager][group] / bound[group] for group in _groups(result))
        )
        arms[name] = {
            "nearest": nearest,
            **to_eager[nearest],
            "ratio": max(to_eager[nearest][group] / bound[group] for group in _groups(result)),
        }
    return {"spread": spread, "bound": bound, "arms": arms}


def single_step_verdict(distances: dict[str, Any]) -> list[str]:
    """Return every violation of the one-step rule (empty when it holds)."""
    failures = []
    for sync, result in distances.items():
        if not result["finite"] or result["max_running_var"] > _MAX_RUNNING_VAR:
            failures.append(f"sync {sync}: reference step not sane ({result['finite']=}, {result['max_running_var']=})")
        if result["relative_update"]["parameters"] > _MAX_RELATIVE_PARAMETER_UPDATE:
            failures.append(f"sync {sync}: parameter update {result['relative_update']['parameters']} too large")
        if not result["integer_buffers_equal"]:
            failures.append(f"sync {sync}: integer buffers differ between arms")
        if not all(result.get("constant_groups_equal", {}).values()):
            failures.append(
                f"sync {sync}: a group the step leaves unchanged differs ({result['constant_groups_equal']})"
            )
        if not result.get("steps_equal", True):
            failures.append(f"sync {sync}: AdamW step counters differ between arms")
        summary = single_step_summary(result)
        for group in _groups(result):
            if summary["spread"][group] > _MAX_SPREAD_OVER_FLOOR * result["floor"][group]:
                failures.append(
                    f"sync {sync}: eager outcomes too spread to resolve a defect ({group} spread "
                    f"{summary['spread'][group]:.3g} > {_MAX_SPREAD_OVER_FLOOR:g} x floor {result['floor'][group]:.3g})"
                )
        graphs = [name for name in summary["arms"] if name.split(".")[-1].startswith("graph")]
        controls = [name for name in summary["arms"] if name.split(".")[-1] in _SINGLE_STEP_CONTROLS]
        if not graphs:
            failures.append(f"sync {sync}: no graph arm ran")
        if len(controls) != len(_SINGLE_STEP_CONTROLS):
            failures.append(f"sync {sync}: controls missing ({controls})")
        for name in graphs:
            arm = summary["arms"][name]
            if arm["ratio"] > 1.0:
                failures.append(f"sync {sync}: {name} {arm} exceeds bound {summary['bound']}")
        for name in controls:
            arm = summary["arms"][name]
            if arm["ratio"] < _CONTROL_MARGIN:
                failures.append(f"sync {sync}: control {name} not detected (distance/bound {arm['ratio']:.3g})")
    return failures


def _verdict_fixture(eager_nearest: list[float], graph: float, *, controls: bool = True) -> dict[str, Any]:
    """A synthetic distances entry: eager outcomes with the given nearest-other distances, two graph arms at
    ``graph`` from eager_a, and (optionally) the four controls at distance 1."""
    floor = {"parameters": 1.5e-5, "momentum": 1.7e-7, "buffers": 1e-6}
    eager = [f"pair{index // 2}.eager_{'ab'[index % 2]}" for index in range(len(eager_nearest))]
    arms = ["pair0.graph_a", "pair1.graph_a", *(f"controls.{name}" for name in _SINGLE_STEP_CONTROLS if controls)]
    to_eager = {
        name: {
            other: {group: (graph if "graph" in name else 1.0) if index == 0 else 1.0 for group in _GROUPS}
            for index, other in enumerate(eager)
        }
        for name in arms
    }
    return {
        "finite": True,
        "max_running_var": 5.0,
        "relative_update": {"parameters": 0.01},
        "integer_buffers_equal": True,
        "floor": floor,
        "eager_nearest": {
            name: dict.fromkeys(_GROUPS, value) for name, value in zip(eager, eager_nearest, strict=True)
        },
        "to_eager": to_eager,
    }


def test_single_step_verdict_is_not_widened_by_one_eager_arm_in_the_other_mode() -> None:
    """Review 093e654: one eager arm alone in the other mode of a bimodal step must not widen the bound to the
    gap between the modes, and controls must run at every sync point."""
    clean = _verdict_fixture([1e-7] * 6, 2e-7)
    assert single_step_verdict({"1": clean, "3": clean}) == []
    # One eager arm 3e-2 away from all others; a graph defect of 1e-2 of a step must still fail.
    lonely = _verdict_fixture([1e-7] * 5 + [3e-2], 1e-2)
    assert any("pair0.graph_a" in failure for failure in single_step_verdict({"1": lonely, "3": lonely}))
    # A spread far above the floor fails instead of widening the bound.
    spread = _verdict_fixture([3e-2] * 6, 1e-2)
    assert any("too spread" in failure for failure in single_step_verdict({"1": spread, "3": spread}))
    # Controls are required at both sync points.
    no_controls = _verdict_fixture([1e-7] * 6, 2e-7, controls=False)
    assert any("controls missing" in failure for failure in single_step_verdict({"1": clean, "3": no_controls}))


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", [_FIXTURE, "mobilenetv4_conv_small_imagenet"])
def test_nondeterministic_graph_run_with_phase2_options_keeps_every_rng_stream_exact(kind: str) -> None:
    eager = _result(_run("nondeterministic_arm", kind, "eager", _ALL_OPTIONS, determinism="nondeterministic"))
    graph = _result(_run("nondeterministic_arm", kind, "graph", _ALL_OPTIONS, determinism="nondeterministic"))
    _assert_variant_ran(eager, _ALL_OPTIONS)
    _assert_variant_ran(graph, _ALL_OPTIONS)
    assert graph["audit"] == _EXPECTED_AUDIT
    for key in _EXACT_UNDER_NONDETERMINISM:
        assert graph[key] == eager[key], key


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", [_FIXTURE, "mobilenetv4_conv_small_imagenet"])
def test_nondeterministic_graph_step_with_phase2_options_matches_eager_within_one_step_noise(
    tmp_path: Path, kind: str
) -> None:
    """The one-step rule with all three options on; the EMA state is a fourth tensor group."""
    spec = {**_SINGLE_STEP_REGIME, "kind": kind, "variant": _ALL_OPTIONS}
    spec.update(_SINGLE_STEP_OVERRIDES.get(kind, {}))
    distances = run_single_step_check(tmp_path, spec, "nondeterministic")
    assert all("ema" in result["floor"] for result in distances.values())
    failures = single_step_verdict(distances)
    assert not failures, (failures, {sync: single_step_summary(result) for sync, result in distances.items()})


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "determinism"), _NONDETERMINISTIC_CASES)
def test_nondeterministic_graph_step_matches_eager_within_one_step_noise(
    tmp_path: Path, kind: str, determinism: str
) -> None:
    spec = {**_SINGLE_STEP_REGIME, "kind": kind}
    spec.update(_SINGLE_STEP_OVERRIDES.get(kind, {}))
    distances = run_single_step_check(tmp_path, spec, determinism)
    failures = single_step_verdict(distances)
    assert not failures, (failures, {sync: single_step_summary(result) for sync, result in distances.items()})


# AWP is admitted with the graph in deterministic mode only (bitwise parity above). Under deterministic=false
# the one-step check could not resolve it: on MobileNetV4-S (64 px, batch 32, gamma 0.01) the EAGER outcomes of
# one AWP step were 0.012-0.014 of a step apart at one sync point in one of two runs (the proxy-normalized
# perturbation amplifies kernel noise into a bimodal step), above 100x the FP32 floor, so the rule fails as
# designed; the schema and the Trainer therefore refuse method.awp with cuda_graph when deterministic=false.
_NONDETERMINISTIC_NEW_METHODS = tuple(method for method in _NEW_METHODS if method != "awp")


def nondeterministic_awp_refusal(root: Path) -> str:
    """AWP + graph under deterministic=false (this subprocess): the Trainer must refuse it."""
    try:
        build_trainer(root, kind=_FIXTURE, cuda_graph=True, device=torch.device("cuda"), method="awp")
    except ValueError as error:
        return str(error)
    return "no error"


@pytest.mark.gpu
@requires_cuda
def test_nondeterministic_awp_is_refused_with_the_graph() -> None:
    message = _result(_run("nondeterministic_awp_refusal", determinism="nondeterministic"))
    assert "AWP only with deterministic algorithms" in message


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NONDETERMINISTIC_NEW_METHODS)
@pytest.mark.parametrize("kind", [_FIXTURE, "mobilenetv4_conv_small_imagenet"])
def test_nondeterministic_new_method_graph_run_keeps_every_rng_stream_exact(kind: str, method: str) -> None:
    eager = _result(_run("nondeterministic_arm", kind, "eager", "base", method, determinism="nondeterministic"))
    graph = _result(_run("nondeterministic_arm", kind, "graph", "base", method, determinism="nondeterministic"))
    _assert_method_ran(eager, method)
    _assert_method_ran(graph, method)
    assert graph["audit"] == _EXPECTED_AUDIT
    for key in _EXACT_UNDER_NONDETERMINISM:
        assert graph[key] == eager[key], key


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NONDETERMINISTIC_NEW_METHODS)
def test_nondeterministic_new_method_graph_step_matches_eager_within_one_step_noise(
    tmp_path: Path, method: str
) -> None:
    """The one-step rule (same bound, same controls) for each 2026-10-08 method on MobileNetV4-Conv-Small.

    Regime: the KL objective's gradients are smaller than CE's, so at lr 0.002 the lr x 1.001 control sat only
    8-9x the bound (FP32 floor-limited); RSLAD runs at lr 0.006 (update about 1% of the weights, as for every
    other student). AWP runs at the production gamma 0.01: at gamma 0.05 the EAGER outcomes alone were 0.04 of
    a step apart (the check then cannot resolve a defect and fails as designed)."""
    kind = "mobilenetv4_conv_small_imagenet"
    spec = {**_SINGLE_STEP_REGIME, "kind": kind, "method": method}
    spec.update(_SINGLE_STEP_OVERRIDES.get(kind, {}))
    if method in _DISTILLATION_METHODS:
        spec["learning_rate"] = 0.006
    distances = run_single_step_check(tmp_path, spec, "nondeterministic")
    failures = single_step_verdict(distances)
    assert not failures, (failures, {sync: single_step_summary(result) for sync, result in distances.items()})


# ---------------------------------------------------------------- production shapes (opt-in)
#
# The same one-step check at production shapes (batch 128, 1000 classes),
# from trained checkpoints where available. Opt-in: it needs about 15 GB of
# GPU memory per process, so run it on a FREE GPU only, never next to a
# production job:
#
#   ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1 \
#   ARD_CG_CHECK_MNV4S_STAGE1_CKPT=<plan0103 MNv4-S stage-1 last.pt> \
#   ARD_CG_CHECK_MNV4S_FULL_CKPT=<plan0103 MNv4-S stage-2 last.pt> \
#   ARD_CG_CHECK_MNV4M_CKPT=<plan0103 MNv4-M phase-1 last.pt> \
#   CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=src python -m pytest -s \
#     tests/integration/test_cuda_graph_training_step.py -k production_shape
#
# A case whose checkpoint variable is unset is skipped. EfficientNet-B0 starts
# from random initialisation with lr 0.5 so its update (about 1% of the
# weights) clears the FP32 floor. Rerun after a torch or driver upgrade.
_PRODUCTION_SHAPES: dict[str, dict[str, Any]] = {
    # imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s256.yaml
    "stage1": {"image_size": 112, "steps": 1, "step_size": "4/255", "learning_rate": 0.025},
    # imagenet_mobilenetv4_pgd_at_random_init_30ep_cg.yaml
    "full": {"image_size": 224, "steps": 3, "step_size": "8/765", "learning_rate": 0.05},
}
_PRODUCTION_COMMON = {"batch": 128, "num_classes": 1000, "weight_decay": 1e-4, "epsilon": "4/255"}
_PRODUCTION_CHECKPOINT_ENV = {
    ("mobilenetv4_conv_small_imagenet", "stage1"): "ARD_CG_CHECK_MNV4S_STAGE1_CKPT",
    ("mobilenetv4_conv_small_imagenet", "full"): "ARD_CG_CHECK_MNV4S_FULL_CKPT",
    ("mobilenetv4_conv_medium_imagenet", "stage1"): "ARD_CG_CHECK_MNV4M_CKPT",
    ("mobilenetv4_conv_medium_imagenet", "full"): "ARD_CG_CHECK_MNV4M_CKPT",
}
# Random-init cases use lr 0.5 so the one-step update clears the FP32 floor (EfficientNet-B0's
# rationale above); the Phase 2 MobileNetV4-S candidates have no checkpoints, so they do the same.
_PRODUCTION_RANDOM_INIT_LR = {"efficientnet_b0_imagenet": 0.5, **dict.fromkeys(_CANDIDATE_ARCHITECTURES, 0.5)}


@pytest.mark.gpu
@requires_cuda
@pytest.mark.skipif(
    os.environ.get("ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK") != "1",
    reason="opt-in production-shape check (needs a free GPU): set ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1",
)
@pytest.mark.parametrize("shape", sorted(_PRODUCTION_SHAPES))
@pytest.mark.parametrize(
    "kind",
    [
        # The ConvNeXt-Atto / DeiT-Tiny family has its own opt-in production-shape check (AdamW, bitwise:
        # test_production_shape_adamw_graph_run_is_bit_identical).
        *sorted(CUDA_GRAPH_ARCHITECTURES - set(_LAYERNORM_ARCHITECTURES)),
        *(pytest.param(architecture, marks=_candidate_opt_in) for architecture in _CANDIDATE_ARCHITECTURES),
    ],
)
def test_production_shape_graph_step_matches_eager_within_one_step_noise(tmp_path: Path, kind: str, shape: str) -> None:
    spec = {**_PRODUCTION_COMMON, **_PRODUCTION_SHAPES[shape], "kind": kind}
    if kind in _PRODUCTION_RANDOM_INIT_LR:
        spec["learning_rate"] = _PRODUCTION_RANDOM_INIT_LR[kind]
    else:
        variable = _PRODUCTION_CHECKPOINT_ENV[(kind, shape)]
        checkpoint = os.environ.get(variable)
        if not checkpoint:
            pytest.skip(f"{variable} is not set")
        spec["checkpoint"] = checkpoint
    distances = run_single_step_check(tmp_path, spec, "nondeterministic")
    summaries = {sync: single_step_summary(result) for sync, result in distances.items()}
    print(json.dumps({"kind": kind, "shape": shape, "distances": distances, "summary": summaries}, default=str))
    failures = single_step_verdict(distances)
    assert not failures, (failures, summaries)


# Review of d2e82b2 (P3-1): the same opt-in production-shape regime for the 2026-10-08 distillation scope,
# deterministic and BITWISE: MobileNetV4-Conv-Small at 224 px, batch 128, 1000 classes, full-AT attack (PGD-3,
# step 8/765, epsilon 4/255), one epoch of 4 full batches (eager warm-up, capture, 3 replays) plus a partial
# batch of 64 and a validation pass, eager vs graph in separate processes. Teachers load their real checkpoints
# from ARD_EXTERNAL_CHECKPOINT_ROOT when present (same relative paths as the Phase 2 configs, registry
# normalization: imagenet_raw_identity for ConvNeXt-B-cvst, the embedded custom profile for ViT-S-cvst,
# imagenet_standard for ResNet-50), random weights otherwise. FREE GPU only (about 12-16 GB per process):
#
#   ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1 ARD_EXTERNAL_CHECKPOINT_ROOT=<root> \
#   [ARD_CG_CHECK_MNV4S_FULL_CKPT=<plan0103 MNv4-S stage-2 last.pt>] \
#   CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=src python -m pytest -s \
#     tests/integration/test_cuda_graph_training_step.py -k production_shape_distillation
_PRODUCTION_TEACHER_CHECKPOINTS = {
    "convnext_base_convstem_imagenet": (
        "singh2023_convnext_b_convstem",
        "singh2023_revisiting_at/convnext_b_cvst/convnext_b_cvst_robust.pt",
    ),
    "vit_s_convstem_imagenet": ("singh2023_vit_s_convstem", "singh2023_revisiting_at/vit_s_cvst/vit_s_cvst_robust.pt"),
    "resnet50_imagenet": ("salman2020_resnet50_linf_eps4", "madrylab/resnet50_linf_eps4.0.ckpt"),
}
_PRODUCTION_DISTILLATION_CASES = [
    ("rslad_online", "convnext_base_convstem_imagenet"),
    ("rslad_online", "vit_s_convstem_imagenet"),
    ("rslad_bank", "resnet50_imagenet"),
    ("rslad_advt_bank", "resnet50_imagenet"),
]
_PRODUCTION_BATCH = 128
_PRODUCTION_FULL_BATCHES = 4


def _production_teacher(teacher_kind: str) -> tuple[TeacherAdapter, bool]:
    """The registry teacher (real checkpoint when available) with its registry normalization."""
    from ard.models.imagenet_teacher_registry import (
        build_imagenet_teacher_architecture,
        load_imagenet_teacher_network,
        spec_for,
    )

    registry_id, relative = _PRODUCTION_TEACHER_CHECKPOINTS[teacher_kind]
    spec = spec_for(registry_id)
    root = os.environ.get("ARD_EXTERNAL_CHECKPOINT_ROOT")
    path = None if not root else Path(root) / relative
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4321)
        if path is not None and path.is_file():
            model, real = load_imagenet_teacher_network(spec, path), True
        else:
            model, real = build_imagenet_teacher_architecture(spec.architecture), False
    metadata = TeacherMetadata(
        architecture=spec.architecture,
        num_classes=1000,
        normalization=spec.normalization(),
        checkpoint_sha256=spec.checkpoint_sha256,
    )
    return TeacherAdapter(model, metadata), real


def production_distillation_arm(root: Path, method: str, teacher_kind: str, mode: str) -> dict[str, Any]:
    """One eager or graph process of the production-shape distillation check (deterministic)."""
    _seed_everything(1234)
    device = torch.device("cuda")
    kind = "mobilenetv4_conv_small_imagenet"
    student = build_student(
        ModelConfig(
            architecture=kind,  # type: ignore[arg-type]
            num_classes=1000,
            pretrained=False,
            normalization=NormalizationConfig(profile="imagenet_standard"),
        ),
        tier="dev",
    )
    checkpoint = os.environ.get("ARD_CG_CHECK_MNV4S_FULL_CKPT")
    if checkpoint:
        student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    student = student.to(device)
    optimizer = SGD(student.parameters(), lr=0.025, momentum=0.9, weight_decay=1e-4, nesterov=True)
    size = _PRODUCTION_FULL_BATCHES * _PRODUCTION_BATCH + _PRODUCTION_BATCH // 2
    teacher, real_teacher = _production_teacher(teacher_kind)
    if method == "rslad_online":
        distillation_teacher: nn.Module = teacher
    else:
        online = teacher if method == "rslad_advt_bank" else None
        distillation_teacher = SoftLabelBankTeacher(_fixture_bank(1000, size, 1), online_teacher=online)
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(
            AttackConfig(
                epsilon="4/255", step_size="8/765", steps=3, random_start=True, loss="kl", kl_target="teacher_clean"
            )
        ),
        selection_attack=LinfPGD(AttackConfig(epsilon="4/255", step_size="8/765", steps=2)),
        objective=RSLADObjective(temperature=1.0, temperature_squared=True),
        policy=RSLADBaselinePolicy(),
        teacher=distillation_teacher,
        distillation_hooks=DistillationTargetHooks(
            adversarial_teacher_target=method == "rslad_advt_bank", temperature=1.0
        ),
        device=device,
        output_dir=root,
        config_hash="c" * 64,
        seed=11,
        tracker_run_id="production-shape-distillation",
        diagnostics=TrainingDiagnostics.for_ids(list(range(size)), seed=0, size=24, mode="panel"),
        step_diagnostics=False,
        cuda_graph=mode == "graph",
        student_architecture=kind,
    )

    def loader(dataset_size: int, seed: int, *, keyed: bool) -> DataLoader:
        dataset = IndexedDataset(SyntheticCIFAR(size=dataset_size, num_classes=1000, image_size=224, seed=seed))
        return DataLoader(
            _CropKeyed(dataset) if keyed else dataset,
            batch_size=_PRODUCTION_BATCH,
            sampler=EpochShuffleSampler(dataset_size, seed=5, shuffle=True),
            pin_memory=True,
            collate_fn=collate_crop_keyed if keyed else collate_indexed,
        )

    history = trainer.fit(
        loader(size, 3, keyed=method != "rslad_online"),
        validation_loader=loader(_PRODUCTION_BATCH, 99, keyed=False),
        epochs=1,
    )
    fingerprint = _fingerprint(trainer, history, root)
    fingerprint["audit"] = [[row.get(key) for key in _AUDIT_KEYS] for row in history]
    fingerprint["real_teacher"] = real_teacher
    fingerprint["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return fingerprint


@pytest.mark.gpu
@requires_cuda
@pytest.mark.skipif(
    os.environ.get("ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK") != "1",
    reason="opt-in production-shape check (needs a free GPU): set ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1",
)
@pytest.mark.parametrize(("method", "teacher_kind"), _PRODUCTION_DISTILLATION_CASES)
def test_production_shape_distillation_graph_run_is_bit_identical(method: str, teacher_kind: str) -> None:
    eager = _result(_run("production_distillation_arm", method, teacher_kind, "eager", timeout=3600))
    graph = _result(_run("production_distillation_arm", method, teacher_kind, "graph", timeout=3600))
    print(
        json.dumps(
            {
                "method": method,
                "teacher": teacher_kind,
                "real_teacher": graph["real_teacher"],
                "peak_reserved_gib": [eager["peak_reserved_gib"], graph["peak_reserved_gib"]],
            }
        )
    )
    assert graph["audit"] == [[1.0, float(_PRODUCTION_FULL_BATCHES - 1), 2.0]]
    ignored = {"audit", "real_teacher", "peak_reserved_gib"}
    assert {k: v for k, v in graph.items() if k not in ignored} == {k: v for k, v in eager.items() if k not in ignored}


# ============================== human approval 2026-10-09: AdamW, ConvNeXt-Atto / DeiT-Tiny and batch-B variants
#
# AdamW is captured through torch's capturable foreach implementation (device step counter and bias
# corrections); ard.cli.train builds AdamW with capturable=True exactly when training.cuda_graph is on, so the
# graph run's eager steps (first full batch, partial batch) and an eager reference built the same way are
# the comparison. capturable=False (every eager AdamW run without the flag) computes the bias corrections on
# the host in float64 and is numerically different -- shown below, and the reason a deterministic AdamW graph
# run carries cuda_graph in its training_protocol_identity. Every case runs for the fixture and every
# allowlisted student, which includes the ConvNeXt-Atto / DeiT-Tiny family (_LAYERNORM_ARCHITECTURES).
_ADAMW_PHASE2 = "adamw+ema+label_smoothing"
_ADAMW_KINDS = (_FIXTURE, *sorted(CUDA_GRAPH_ARCHITECTURES | set(_LAYERNORM_ARCHITECTURES)))
_ADAMW_PARITY_CASES = [
    (_FIXTURE, "adamw"),
    *((kind, _ADAMW_PHASE2) for kind in _ADAMW_KINDS),
]
# The Phase 2 students with ConvNeXt-Atto / DeiT-Tiny method configs (RSLAD bank/online, mixed batch).
_ADAMW_METHOD_KINDS = (_FIXTURE, "convnext_atto_imagenet", "deit_tiny_imagenet")


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "variant"), _ADAMW_PARITY_CASES)
def test_adamw_graph_step_is_bit_identical_to_the_capturable_eager_step(kind: str, variant: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", variant))
    graph = _result(_run("arm", kind, "graph", "panel", variant))
    _assert_variant_ran(eager, variant)
    _assert_variant_ran(graph, variant)
    assert eager["has_probe"] and graph["has_probe"]
    assert graph["audit"] == _EXPECTED_AUDIT
    assert _comparable(graph) == _comparable(eager)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NEW_METHODS)
@pytest.mark.parametrize("kind", _ADAMW_METHOD_KINDS)
def test_adamw_graph_step_of_new_methods_is_bit_identical(kind: str, method: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", "adamw+ema", method))
    graph = _result(_run("arm", kind, "graph", "panel", "adamw+ema", method))
    for result in (eager, graph):
        _assert_method_ran(result, method)
        _assert_variant_ran(result, "adamw+ema")
    assert graph["audit"] == _EXPECTED_AUDIT
    assert _comparable(graph) == _comparable(eager)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", _ADAMW_METHOD_KINDS)
def test_adamw_graph_run_resumed_mid_run_equals_the_eager_run(kind: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", _ADAMW_PHASE2))
    resumed = _result(_run("arm", kind, "resumed", "panel", _ADAMW_PHASE2))
    _assert_variant_ran(resumed, _ADAMW_PHASE2)
    assert _comparable(resumed, diagnostics=False) == _comparable(eager, diagnostics=False)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", [_FIXTURE, "convnext_atto_imagenet"])
def test_negative_control_an_adamw_step_counter_advanced_in_the_graph_is_detected(kind: str) -> None:
    eager = _result(_run("arm", kind, "eager", "panel", "adamw"))
    shifted = _result(_run("arm", kind, "graph_adamw_step_plus_one", "panel", "adamw"))
    assert shifted["audit"] == _EXPECTED_AUDIT
    assert shifted["last.pt:model"] != eager["last.pt:model"]
    assert shifted["last.pt:optimizer"] != eager["last.pt:optimizer"]
    assert shifted["rows"] != eager["rows"]


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("mutation", _ADAMW_MUTATIONS)
def test_mid_epoch_adamw_change_is_refused_not_replayed(mutation: str) -> None:
    message = _result(_run("stale_guard", mutation))
    assert "CUDA graph is stale" in message


def _run_with_capturable(capturable: bool, *args: str, determinism: str = "deterministic") -> Any:
    previous = os.environ.get("ARD_CUDA_GRAPH_TEST_ADAMW_CAPTURABLE")
    os.environ["ARD_CUDA_GRAPH_TEST_ADAMW_CAPTURABLE"] = "1" if capturable else "0"
    try:
        return _result(_run(*args, determinism=determinism))
    finally:
        if previous is None:
            os.environ.pop("ARD_CUDA_GRAPH_TEST_ADAMW_CAPTURABLE")
        else:
            os.environ["ARD_CUDA_GRAPH_TEST_ADAMW_CAPTURABLE"] = previous


@pytest.mark.gpu
@requires_cuda
def test_capturable_adamw_is_not_bitwise_the_default_adamw() -> None:
    """Why a deterministic AdamW graph run is recorded as such (ard.cli.evaluate._throughput_protocol_identity):
    torch's capturable AdamW computes the bias corrections on the device in float32, the default one on the host
    in float64, so two eager runs that differ only in ``capturable`` diverge. If a torch upgrade makes them
    equal, the identity rule (and plan 0105) can be revisited."""
    default = _run_with_capturable(False, "arm", _FIXTURE, "eager", "panel", "adamw")
    capturable = _run_with_capturable(True, "arm", _FIXTURE, "eager", "panel", "adamw")
    assert default["optimizer_capturable"] == [False, False]
    assert capturable["optimizer_capturable"] == [True, True]
    assert default["last.pt:model"] != capturable["last.pt:model"]


_ADAMW_NONDETERMINISTIC_KINDS = (_FIXTURE, *_LAYERNORM_ARCHITECTURES)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", _ADAMW_NONDETERMINISTIC_KINDS)
def test_nondeterministic_adamw_graph_run_keeps_every_rng_stream_exact(kind: str) -> None:
    eager = _result(_run("nondeterministic_arm", kind, "eager", _ADAMW_PHASE2, determinism="nondeterministic"))
    graph = _result(_run("nondeterministic_arm", kind, "graph", _ADAMW_PHASE2, determinism="nondeterministic"))
    _assert_variant_ran(eager, _ADAMW_PHASE2)
    _assert_variant_ran(graph, _ADAMW_PHASE2)
    assert graph["audit"] == _EXPECTED_AUDIT
    for key in _EXACT_UNDER_NONDETERMINISM:
        assert graph[key] == eager[key], key


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NONDETERMINISTIC_NEW_METHODS)
@pytest.mark.parametrize("kind", ["convnext_atto_imagenet", "deit_tiny_imagenet"])
def test_nondeterministic_adamw_new_method_graph_run_keeps_every_rng_stream_exact(kind: str, method: str) -> None:
    eager = _result(_run("nondeterministic_arm", kind, "eager", "adamw+ema", method, determinism="nondeterministic"))
    graph = _result(_run("nondeterministic_arm", kind, "graph", "adamw+ema", method, determinism="nondeterministic"))
    _assert_method_ran(eager, method)
    _assert_method_ran(graph, method)
    assert graph["audit"] == _EXPECTED_AUDIT
    for key in _EXACT_UNDER_NONDETERMINISM:
        assert graph[key] == eager[key], key


# One-step regime for AdamW: lr sets the per-element update size (about lr / |w| of the weights). The update
# must clear the FP32 floor by enough that the lr x 1.001 control is detectable: at lr 2e-4 ConvNeXt-Atto moved
# 0.3-0.4% of its weights and that control sat only 7-8x the bound (the fixture: 0.07%, 1.5x). lr 6e-4 (the
# fixture, with its larger weights, 3e-3) moves about 1% -- the size every SGD case uses.
_ADAMW_SINGLE_STEP_LR = {_FIXTURE: 3e-3}


def _adamw_single_step(kind: str) -> dict[str, Any]:
    return {"adamw_learning_rate": _ADAMW_SINGLE_STEP_LR.get(kind, 6e-4)}


def _single_step_spec(kind: str, **extra: Any) -> dict[str, Any]:
    spec = {**_SINGLE_STEP_REGIME, "kind": kind, **extra}
    spec.update(_SINGLE_STEP_OVERRIDES.get(kind, {}))
    return spec


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", _ADAMW_NONDETERMINISTIC_KINDS)
def test_nondeterministic_adamw_graph_step_matches_eager_within_one_step_noise(tmp_path: Path, kind: str) -> None:
    """The one-step rule with AdamW (+ the Phase 2 EMA and label smoothing): exp_avg and exp_avg_sq are
    tensor groups held to the same bound, the step counters must be exact."""
    spec = _single_step_spec(kind, variant=_ADAMW_PHASE2, **_adamw_single_step(kind))
    distances = run_single_step_check(tmp_path, spec, "nondeterministic")
    for result in distances.values():
        assert {"exp_avg", "exp_avg_sq", "ema"} <= set(result["floor"]), result["floor"]
        assert "momentum" not in result["floor"]
    failures = single_step_verdict(distances)
    assert not failures, (failures, {sync: single_step_summary(result) for sync, result in distances.items()})


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("method", _NONDETERMINISTIC_NEW_METHODS)
def test_nondeterministic_adamw_new_method_graph_step_matches_eager_within_one_step_noise(
    tmp_path: Path, method: str
) -> None:
    kind = "convnext_atto_imagenet"
    spec = _single_step_spec(kind, variant="adamw+ema", method=method, **_adamw_single_step(kind))
    distances = run_single_step_check(tmp_path, spec, "nondeterministic")
    failures = single_step_verdict(distances)
    assert not failures, (failures, {sync: single_step_summary(result) for sync, result in distances.items()})


# Opt-in production shape, deterministic and BITWISE (as the distillation check above): the plan 0103
# ConvNeXt-Atto / DeiT-Tiny Phase 2 PGD-AT step -- 224 px, batch 128, 1000 classes, PGD-3 (step 8/765,
# epsilon 4/255), AdamW (lr 1e-3, betas 0.9/0.999, weight decay 0.05 without norm/bias) with the 0.9999
# weight EMA -- one epoch of 4 full batches plus a partial batch of 64 and a validation pass, eager vs graph in
# separate processes, random initialisation. FREE GPU only (about 6-8 GB per process):
#
#   ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1 CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=src python -m pytest -s \
#     tests/integration/test_cuda_graph_training_step.py -k production_shape_adamw
def production_adamw_arm(root: Path, kind: str, mode: str) -> dict[str, Any]:
    from ard.cli.train import _weight_decay_parameter_groups

    _seed_everything(1234)
    device = torch.device("cuda")
    _admit_candidate(kind)
    student = build_student(
        ModelConfig(
            architecture=kind,  # type: ignore[arg-type]
            num_classes=1000,
            pretrained=False,
            normalization=NormalizationConfig(profile="imagenet_standard"),
        ),
        tier="dev",
    ).to(device)
    optimizer = AdamW(
        _weight_decay_parameter_groups(student.parameters(), weight_decay=0.05),
        lr=1e-3,
        betas=(0.9, 0.999),
        capturable=True,
    )
    size = _PRODUCTION_FULL_BATCHES * _PRODUCTION_BATCH + _PRODUCTION_BATCH // 2
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="4/255", step_size="8/765", steps=3, random_start=True)),
        selection_attack=LinfPGD(AttackConfig(epsilon="4/255", step_size="8/765", steps=2)),
        objective=PGDATObjective(),
        weight_ema_decay=0.9999,
        device=device,
        output_dir=root,
        config_hash="c" * 64,
        seed=11,
        tracker_run_id="production-shape-adamw",
        diagnostics=TrainingDiagnostics.for_ids(list(range(size)), seed=0, size=24, mode="panel"),
        step_diagnostics=False,
        cuda_graph=mode == "graph",
        student_architecture=kind,
    )

    def loader(dataset_size: int, seed: int) -> DataLoader:
        dataset = IndexedDataset(SyntheticCIFAR(size=dataset_size, num_classes=1000, image_size=224, seed=seed))
        return DataLoader(
            dataset,
            batch_size=_PRODUCTION_BATCH,
            sampler=EpochShuffleSampler(dataset_size, seed=5, shuffle=True),
            pin_memory=True,
            collate_fn=collate_indexed,
        )

    history = trainer.fit(loader(size, 3), validation_loader=loader(_PRODUCTION_BATCH, 99), epochs=1)
    fingerprint = _fingerprint(trainer, history, root)
    fingerprint["audit"] = [[row.get(key) for key in _AUDIT_KEYS] for row in history]
    fingerprint["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return fingerprint


@pytest.mark.gpu
@requires_cuda
@pytest.mark.skipif(
    os.environ.get("ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK") != "1",
    reason="opt-in production-shape check (needs a free GPU): set ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1",
)
@pytest.mark.parametrize("kind", _LAYERNORM_ARCHITECTURES)
def test_production_shape_adamw_graph_run_is_bit_identical(kind: str) -> None:
    eager = _result(_run("production_adamw_arm", kind, "eager", timeout=3600))
    graph = _result(_run("production_adamw_arm", kind, "graph", timeout=3600))
    print(json.dumps({"kind": kind, "peak_reserved_gib": [eager["peak_reserved_gib"], graph["peak_reserved_gib"]]}))
    assert graph["audit"] == [[1.0, float(_PRODUCTION_FULL_BATCHES - 1), 2.0]]
    ignored = {"audit", "peak_reserved_gib"}
    assert {k: v for k, v in graph.items() if k not in ignored} == {k: v for k, v in eager.items() if k not in ignored}
