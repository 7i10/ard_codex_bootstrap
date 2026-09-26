"""Sync-free training step: bit-identical numerics and a fail-closed loss guard.

The per-step epoch accumulators are kept on the device (no ``float(tensor)``
per step), the non-finite-loss guard is a ``torch._assert_async`` on CUDA,
and CUDA loaders pin memory with non-blocking copies. The differential tests
run three fixture methods twice in one process -- once through the
pre-change path (host ``float()`` accumulation, host ``isfinite`` raise,
blocking copies, unpinned loaders; reconstructed from commit 39b9967) and
once through the shipped path -- and require identical epoch metrics,
model/optimizer/EMA tensors, RNG state, sampler/sample state and selection
state. Being same-process, the comparison is machine-independent. The CUDA
variant runs in a subprocess with deterministic algorithms.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import ard.engine.trainer as trainer_module
from ard.attacks import LinfPGD
from ard.cli.train import _loader_pin_memory
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
METHODS = ("pgd_at", "rslad", "adr")
# Wall-clock/device-memory telemetry is not a function of the numerics.
_TIMING_SUFFIXES = ("seconds", "images_per_second", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
# Checkpoint entries whose drift would reveal a numeric, RNG-consumption,
# sampler, sample-state or selection change.
_STATE_KEYS = (
    "model",
    "optimizer",
    "ema",
    "rng",
    "sampler_epoch",
    "sampler_state",
    "sample_state",
    "global_step",
    "best_metric",
    "best_metric_ema",
    "selection_metadata",
    "selection_metadata_ema",
    "fork_lineage",
)
_BATCH_FIELDS = ("images", "labels", "sample_ids", "state_update_mask", "multiplicity")


def _loaders(*, pin_memory: bool = False, num_workers: int = 0, seed: int = 11) -> tuple[DataLoader, DataLoader]:
    dataset = IndexedDataset(SyntheticCIFAR(size=16, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(train_dataset), seed=seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_indexed,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_indexed,
    )
    return loader, validation_loader


def _selection_attack() -> LinfPGD:
    return LinfPGD(
        AttackConfig(
            epsilon="2/255", step_size="1/255", steps=2, random_start=True, student_mode="eval", teacher_mode="eval"
        )
    )


def _trainer(output: Path, method: str, device: torch.device | None = None) -> Trainer:
    device = torch.device("cpu") if device is None else device
    torch.manual_seed(123)
    student = build_student(FIXTURE, tier="smoke").to(device)
    optimizer = SGD(student.parameters(), lr=0.05, momentum=0.9)
    common: dict[str, Any] = {
        "model": student,
        "optimizer": optimizer,
        "scheduler": StepLR(optimizer, step_size=1, gamma=0.8),
        "scaler": None,
        "selection_attack": _selection_attack(),
        "device": device,
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
        teacher = build_student(FIXTURE, tier="smoke").to(device)
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


def _canonical(value: object, digest: Any) -> None:
    """Feed an exact, type-tagged serialization of ``value`` into ``digest``."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"T|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        raw = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
        digest.update(raw.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(f"N|{value.dtype}|{value.shape}|".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        digest.update(f"D{len(value)}|".encode())
        for key in sorted(value, key=repr):
            digest.update(f"{key!r}:".encode())
            _canonical(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}{len(value)}|".encode())
        for item in value:
            _canonical(item, digest)
    elif isinstance(value, float):
        digest.update(f"F{value.hex()}|".encode())
    else:
        digest.update(f"{type(value).__name__}:{value!r}|".encode())


def _fingerprint(output: Path, method: str, *, device: torch.device, loaders: tuple[DataLoader, DataLoader]) -> dict:
    trainer = _trainer(output, method, device)
    loader, validation_loader = loaders
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    rows = [
        {key: float(value).hex() for key, value in sorted(row.items()) if not key.endswith(_TIMING_SUFFIXES)}
        for row in history
    ]
    fingerprint: dict[str, Any] = {"rows": hashlib.sha256(repr(rows).encode()).hexdigest()}
    for name in ("last.pt", "best.pt"):
        payload = torch.load(output / name, map_location="cpu", weights_only=False)
        for key in _STATE_KEYS:
            if key in payload:
                digest = hashlib.sha256()
                _canonical(payload[key], digest)
                fingerprint[f"{name}:{key}"] = digest.hexdigest()
    return fingerprint


def _pre_change_float64_totals(values: Sequence[torch.Tensor | float], *, device: torch.device) -> torch.Tensor:
    # Commit 39b9967: one host ``float()`` (a GPU sync) per entry.
    return torch.tensor([float(value) for value in values], dtype=torch.float64, device=device)


def _pre_change_assert_finite(loss: torch.Tensor) -> None:
    # Commit 39b9967: host branch on the loss tensor.
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")


def _pre_change_to(self: IndexedBatch, device: torch.device | str, *, non_blocking: bool = False) -> IndexedBatch:
    # Commit 39b9967: blocking copies, ``non_blocking`` ignored.
    del non_blocking
    return IndexedBatch(
        self.images.to(device),
        self.labels.to(device),
        self.sample_ids.to(device),
        None if self.state_update_mask is None else self.state_update_mask.to(device),
        None if self.multiplicity is None else self.multiplicity.to(device),
    )


@contextmanager
def _pre_change_path() -> Iterator[None]:
    saved = (trainer_module._float64_totals, trainer_module._assert_finite_training_loss, IndexedBatch.to)
    trainer_module._float64_totals = _pre_change_float64_totals  # type: ignore[assignment]
    trainer_module._assert_finite_training_loss = _pre_change_assert_finite  # type: ignore[assignment]
    IndexedBatch.to = _pre_change_to  # type: ignore[method-assign]
    try:
        yield
    finally:
        trainer_module._float64_totals, trainer_module._assert_finite_training_loss = saved[:2]
        IndexedBatch.to = saved[2]  # type: ignore[method-assign]


def differential(method: str, device_type: str, root: Path) -> dict[str, Any]:
    """Run ``method`` through the pre-change and shipped paths; return both fingerprints.

    The worker count is the same on both sides (so DataLoader base-seed draws
    match); only the pinning and the step path differ.
    """
    device = torch.device(device_type)
    num_workers = 2 if device.type == "cuda" else 0
    with _pre_change_path():
        old = _fingerprint(
            root / "old", method, device=device, loaders=_loaders(pin_memory=False, num_workers=num_workers)
        )
    pin = _loader_pin_memory(device)
    new = _fingerprint(root / "new", method, device=device, loaders=_loaders(pin_memory=pin, num_workers=num_workers))
    probe_loader, _ = _loaders(pin_memory=pin, num_workers=num_workers)
    probe_batch = next(iter(probe_loader))
    pinned = {field: bool(getattr(probe_batch, field).is_pinned()) for field in _BATCH_FIELDS}
    return {"old": old, "new": new, "pin_memory": pin, "pinned": pinned}


@pytest.mark.parametrize("method", METHODS)
def test_cpu_shipped_path_is_bit_identical_to_pre_change_path(tmp_path: Path, method: str) -> None:
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        result = differential(method, "cpu", tmp_path)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    assert result["pin_memory"] is False
    assert not any(result["pinned"].values())
    assert set(result["new"]) == set(result["old"])
    assert "last.pt:rng" in result["new"] and "best.pt:model" in result["new"]
    assert result["new"] == result["old"]


_CUDA_DIFFERENTIAL_SCRIPT = r"""
import json, sys, tempfile
from pathlib import Path
import torch
torch.use_deterministic_algorithms(True)
sys.path.insert(0, sys.argv[1])
from test_step_sync_free_parity import differential
with tempfile.TemporaryDirectory() as root:
    print("RESULT " + json.dumps(differential(sys.argv[2], "cuda", Path(root))), flush=True)
"""


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", METHODS)
def test_cuda_shipped_path_is_bit_identical_to_pre_change_path(method: str) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    # Required by deterministic cuBLAS; must be set before CUDA initializes.
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        [sys.executable, "-c", _CUDA_DIFFERENTIAL_SCRIPT, str(Path(__file__).resolve().parent), method],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, completed.stdout
    result = json.loads(lines[0].removeprefix("RESULT "))
    assert result["pin_memory"] is True
    assert result["pinned"] == dict.fromkeys(_BATCH_FIELDS, True)
    assert set(result["new"]) == set(result["old"])
    assert "last.pt:rng" in result["new"]
    assert result["new"] == result["old"]


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
    assert "non-finite training loss" in result.stderr


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
    assert _loader_pin_memory(torch.device("cuda", 0)) is True
    assert _loader_pin_memory(torch.device("cuda")) is True
    assert _loader_pin_memory(torch.device("cpu")) is False
