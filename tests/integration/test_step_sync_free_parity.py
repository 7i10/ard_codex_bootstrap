"""Sync-free training step: bit-identical numerics and a fail-closed loss guard.

The per-step epoch accumulators are kept on the device (no ``float(tensor)``
per step) and the non-finite-loss guard is a ``torch._assert_async`` on CUDA.
These tests pin the exact epoch metrics and final tensor state of three
fixture runs against golden values captured from the pre-change code
(commit 39b9967), so any drift in value, dtype, summation order or RNG
consumption fails here. Fixture-scale and CPU-only except the explicitly
GPU-marked subprocess test.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
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
from ard.engine.trainer import Trainer, _assert_finite_training_loss, _float64_totals
from ard.models import build_student
from ard.objectives import ADRObjective, ObjectiveTerms, PGDATObjective, RSLADObjective
from ard.policies import RSLADBaselinePolicy
from ard.state import SampleStateStore

pytestmark = pytest.mark.t3

FIXTURE = ModelConfig(architecture="fixture_cnn", num_classes=3)
# Wall-clock/device-memory telemetry is not a function of the numerics.
_TIMING_SUFFIXES = ("seconds", "images_per_second", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")


def _loaders(seed: int = 11) -> tuple[DataLoader, DataLoader]:
    dataset = IndexedDataset(SyntheticCIFAR(size=16, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(train_dataset), seed=seed),
        collate_fn=collate_indexed,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False),
        collate_fn=collate_indexed,
    )
    return loader, validation_loader


def _selection_attack() -> LinfPGD:
    return LinfPGD(
        AttackConfig(
            epsilon="2/255", step_size="1/255", steps=2, random_start=True, student_mode="eval", teacher_mode="eval"
        )
    )


def _trainer(output: Path, method: str) -> Trainer:
    torch.manual_seed(123)
    student = build_student(FIXTURE, tier="smoke")
    optimizer = SGD(student.parameters(), lr=0.05, momentum=0.9)
    common: dict[str, Any] = {
        "model": student,
        "optimizer": optimizer,
        "scheduler": StepLR(optimizer, step_size=1, gamma=0.8),
        "scaler": None,
        "selection_attack": _selection_attack(),
        "device": torch.device("cpu"),
        "output_dir": output,
        "config_hash": "c" * 64,
        "seed": 11,
        "tracker_run_id": f"sync-free-{method}",
    }
    if method == "pgd_at":
        return Trainer(
            attack=LinfPGD(AttackConfig(epsilon="2/255", step_size="1/255", steps=2, random_start=True)),
            objective=PGDATObjective(),
            **common,
        )
    if method == "rslad":
        torch.manual_seed(456)
        teacher = build_student(FIXTURE, tier="smoke")
        return Trainer(
            teacher=teacher,
            attack=LinfPGD(
                AttackConfig(
                    loss="kl",
                    kl_target="teacher_clean",
                    temperature=1.0,
                    epsilon="2/255",
                    step_size="1/255",
                    steps=2,
                    random_start=True,
                )
            ),
            objective=RSLADObjective(),
            policy=RSLADBaselinePolicy(),
            # Exercises the teacher-adversarial forward counter (index 5).
            sample_store=SampleStateStore(ema_decay=0.9),
            observation_profile="teacher_response",
            **common,
        )
    if method == "adr":
        # Exercises the EMA-agreement and rectified-mass accumulators (6, 7).
        return Trainer(
            attack=LinfPGD(
                AttackConfig(
                    loss="kl",
                    kl_target="rectified",
                    temperature=1.0,
                    epsilon="2/255",
                    step_size="1/255",
                    steps=2,
                    random_start=True,
                )
            ),
            objective=ADRObjective(),
            adr_config=AdrConfig(
                ema_decay=0.9, temperature_high=2.0, temperature_low=1.0, lambda_low=0.5, lambda_high=0.9
            ),
            total_iterations=2 * 3,
            **common,
        )
    raise AssertionError(method)


def _tensor_digest(named: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named):
        value = named[name].detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.numpy().tobytes() if value.dtype != torch.bfloat16 else value.view(torch.int16).numpy())
    return digest.hexdigest()


def _flatten(prefix: str, value: object, out: dict[str, torch.Tensor]) -> None:
    if isinstance(value, torch.Tensor):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}", item, out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten(f"{prefix}.{index}", item, out)


def _fingerprint(output: Path, method: str) -> dict[str, Any]:
    trainer = _trainer(output, method)
    loader, validation_loader = _loaders()
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    rows = [
        {key: float(value).hex() for key, value in sorted(row.items()) if not key.endswith(_TIMING_SUFFIXES)}
        for row in history
    ]
    tensors: dict[str, torch.Tensor] = {}
    for name in ("last.pt", "best.pt"):
        payload = torch.load(output / name, map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "ema"):
            if key in payload:
                _flatten(f"{name}.{key}", payload[key], tensors)
    return {"rows": hashlib.sha256(repr(rows).encode()).hexdigest(), "tensors": _tensor_digest(tensors), "_rows": rows}


# Captured from the pre-change trainer (commit 39b9967, torch 2.11.0 CPU) with
# ``_fingerprint``; the row digest covers every non-timing epoch metric as an
# exact float hex string, the tensor digest every model/optimizer/EMA tensor
# of last.pt and best.pt.
GOLDEN: dict[str, dict[str, str]] = {
    "pgd_at": {
        "rows": "68bda68ed5654ae527368d441ccf1be8ee1848734c867088a37158467c249c3a",
        "tensors": "78b10c2a5e171525aed9ddd355e52c8ce47186a2fce85a1c037bdaeaaa1b392a",
    },
    "rslad": {
        "rows": "428fa43d5ab8542cba24fcfc29fa77e4571d062b1ca4d1b6416ecc17241643c3",
        "tensors": "a108b7e63d58b68e84c0172917567d9638a9379d3630f2dfe56f26c79b319b07",
    },
    "adr": {
        "rows": "2705eacde1cf1655567db1d65f7b1e7a6f7f20412ebcb78eb2eac5cf5ee183d9",
        "tensors": "c5de89284da246182b5d9ebc67267584b179b7ac6cb3cc9f0c86c09d65ea8770",
    },
}


@pytest.mark.parametrize("method", ["pgd_at", "rslad", "adr"])
def test_epoch_metrics_and_checkpoints_are_bit_identical_to_pre_change_golden(tmp_path: Path, method: str) -> None:
    observed = _fingerprint(tmp_path / method, method)
    assert {"rows": observed["rows"], "tensors": observed["tensors"]} == GOLDEN[method], observed["_rows"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_device_accumulation_equals_host_float_round_trip_exactly(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    old = torch.zeros(3, dtype=torch.float64)
    new = torch.zeros(3, dtype=torch.float64)
    for _ in range(50):
        sums = [(torch.randn(128, generator=generator) * 1e3).to(dtype).sum(), torch.randn(7).to(dtype).sum()]
        old += torch.tensor([float(sums[0]), float(sums[1]), 1.0], dtype=torch.float64)
        new += _float64_totals([sums[0], sums[1], 1.0], device=torch.device("cpu"))
    assert torch.equal(old, new)


class _NaNAtStep(nn.Module):
    """Wrap PGD-AT and make the loss non-finite on the ``bad_step``-th call."""

    def __init__(self, bad_step: int) -> None:
        super().__init__()
        self.inner = PGDATObjective()
        self.calls = 0
        self.bad_step = bad_step

    def forward(self, **inputs: torch.Tensor) -> ObjectiveTerms:
        terms = self.inner(**inputs)
        self.calls += 1
        if self.calls != self.bad_step:
            return terms
        poisoned = terms.hard.clone()
        poisoned[0] = float("nan")
        return ObjectiveTerms(
            hard=poisoned,
            kd=terms.kd,
            regularization=terms.regularization,
            adversarial_kd=terms.adversarial_kd,
            clean_kd=terms.clean_kd,
        )


def test_non_finite_loss_aborts_before_the_update_on_cpu(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path / "nan", "pgd_at")
    trainer.objective = _NaNAtStep(bad_step=2)
    snapshots: list[dict[str, torch.Tensor]] = []
    original_step = trainer.optimizer.step

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        snapshots.append({key: value.clone() for key, value in trainer.model.state_dict().items()})
        return result

    trainer.optimizer.step = recording_step  # type: ignore[method-assign]
    loader, validation_loader = _loaders()
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    # Exactly the first (finite) step was applied; the non-finite step never
    # reached backward or the optimizer, and nothing was persisted.
    assert len(snapshots) == 1
    assert trainer.global_step == 1
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, snapshots[0][key]), key
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    assert all(p.grad is None for p in trainer.model.parameters())
    assert not (tmp_path / "nan" / "last.pt").exists()
    assert not (tmp_path / "nan" / "best.pt").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_loss_guard_rejects_every_non_finite_value_on_cpu(value: float) -> None:
    _assert_finite_training_loss(torch.tensor(1.0))
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        _assert_finite_training_loss(torch.tensor(value))


_CUDA_SCRIPT = r"""
import torch
from ard.engine.trainer import _assert_finite_training_loss
weight = torch.ones((), device="cuda", requires_grad=True)
_assert_finite_training_loss(weight * 2.0)
torch.cuda.synchronize()
print("finite ok", flush=True)
loss = weight * float("nan")
_assert_finite_training_loss(loss)
loss.backward()
with torch.no_grad():
    weight -= 0.1 * weight.grad
print("weight", float(weight.detach().cpu()), flush=True)
print("guard did not fire", flush=True)
"""


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_loss_guard_is_a_fatal_device_side_assert() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", _CUDA_SCRIPT],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert "finite ok" in result.stdout, result.stderr
    # The poisoned update can never be read back to the host.
    assert "weight" not in result.stdout
    assert "guard did not fire" not in result.stdout
    assert result.returncode != 0
    assert "device-side assert" in result.stderr


def test_indexed_batch_pins_every_tensor_and_copies_non_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned: list[int] = []

    def fake_pin(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        pinned.append(id(self))
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    batch = collate_indexed([(torch.rand(3, 4, 4), 1, 7, True, 1), (torch.rand(3, 4, 4), 2, 9, False, 2)])
    # DataLoader's pin-memory thread only pins objects that expose
    # ``pin_memory()``; a plain dataclass would be passed through unpinned.
    from torch.utils.data._utils.pin_memory import pin_memory as loader_pin_memory

    result = loader_pin_memory(batch)
    assert isinstance(result, IndexedBatch)
    assert len(pinned) == 5
    for field in ("images", "labels", "sample_ids", "state_update_mask", "multiplicity"):
        assert torch.equal(getattr(result, field), getattr(batch, field)), field
    moved = batch.to(torch.device("cpu"), non_blocking=True)
    for field in ("images", "labels", "sample_ids", "state_update_mask", "multiplicity"):
        assert torch.equal(getattr(moved, field), getattr(batch, field)), field


def test_loader_pin_memory_only_for_cuda() -> None:
    from ard.cli.train import _loader_pin_memory

    assert _loader_pin_memory(torch.device("cuda", 0)) is True
    assert _loader_pin_memory(torch.device("cuda")) is True
    assert _loader_pin_memory(torch.device("cpu")) is False
