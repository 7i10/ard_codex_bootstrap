"""``training.cuda_graph`` (plan 0105): the captured step is the eager step, bit for bit.

GPU differentials (subprocess, deterministic algorithms, skipped without CUDA)
run the real ``Trainer.fit`` twice in one process -- eager and with the
captured step -- and require identical epoch rows (timing aside), identical
checkpoint state (model, optimizer, scheduler, every RNG stream, sampler,
selection) in ``last.pt`` / ``best.pt``, identical per-sample diagnostic
rows and panel media, and identical state after a mid-run resume. The run has
a learning-rate milestone inside it, a partial last batch, BatchNorm and
dropout (a default-generator draw inside the graph), and exercises
re-capture every epoch.

Guard tests: a mid-epoch optimizer change is refused rather than replayed
stale; the in-graph device asserts (pixel range, finite loss) fire from inside
a replay; and a pixel-range violation on the graph path kills the process
before anything is checkpointed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config.schema import AttackConfig, ModelConfig, NormalizationConfig
from ard.data import EpochShuffleSampler, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.models.registry import PixelModel
from ard.objectives import PGDATObjective
from ard.tracking.diagnostics import TrainingDiagnostics
from tests.integration.test_step_sync_free_parity import _STATE_KEYS, _TIMING_SUFFIXES, _canonical

pytestmark = pytest.mark.t3

_ROOT = Path(__file__).resolve().parents[2]
# 26 training samples at batch 4: six full batches (eager, capture, four
# replays) and a partial batch of 2 per epoch.
_TRAIN_SIZE = 26
_BATCH = 4


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


def _student(kind: str) -> tuple[nn.Module, int, int]:
    """Return (student, num_classes, image_size)."""
    if kind == "bn_dropout":
        return _bn_dropout_student(3), 3, 8
    if kind == "mobilenetv4":
        config = ModelConfig(
            architecture="mobilenetv4_conv_small_imagenet",
            num_classes=10,
            pretrained=False,
            normalization=NormalizationConfig(profile="imagenet_standard"),
        )
        return build_student(config, tier="dev"), 10, 32
    raise AssertionError(kind)


def _loaders(num_classes: int, image_size: int, *, pin_memory: bool) -> tuple[DataLoader, DataLoader]:
    train = IndexedDataset(SyntheticCIFAR(size=_TRAIN_SIZE, num_classes=num_classes, image_size=image_size, seed=3))
    validation = IndexedDataset(SyntheticCIFAR(size=6, num_classes=num_classes, image_size=image_size, seed=99))
    loader = DataLoader(
        train,
        batch_size=_BATCH,
        sampler=EpochShuffleSampler(len(train), seed=5),
        pin_memory=pin_memory,
        collate_fn=collate_indexed,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=_BATCH,
        sampler=EpochShuffleSampler(len(validation), seed=5, shuffle=False),
        pin_memory=pin_memory,
        collate_fn=collate_indexed,
    )
    return loader, validation_loader


def build_trainer(output: Path, *, kind: str, cuda_graph: bool, device: torch.device) -> tuple[Trainer, Any]:
    torch.manual_seed(1234)
    student, num_classes, image_size = _student(kind)
    student = student.to(device)
    optimizer = SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=MultiStepLR(optimizer, milestones=[1, 2], gamma=0.1),
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", steps=2, random_start=True)),
        selection_attack=LinfPGD(
            AttackConfig(epsilon="4/255", step_size="1/255", steps=2, student_mode="eval", teacher_mode="eval")
        ),
        objective=PGDATObjective(),
        device=device,
        output_dir=output,
        config_hash="c" * 64,
        seed=11,
        tracker_run_id="cuda-graph",
        diagnostics=TrainingDiagnostics.for_ids(list(range(_TRAIN_SIZE)), seed=0, size=5, mode="panel"),
        step_diagnostics=False,
        cuda_graph=cuda_graph,
    )
    return trainer, (num_classes, image_size)


def _canonical_digest(value: object) -> str:
    digest = hashlib.sha256()
    _canonical(value, digest)
    return digest.hexdigest()


def _fingerprint(trainer: Trainer, history: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows = [
        {key: float(value).hex() for key, value in sorted(row.items()) if not key.endswith(_TIMING_SUFFIXES)}
        for row in history
    ]
    result: dict[str, Any] = {"rows": rows}
    for name in ("last.pt", "best.pt"):
        payload = torch.load(output / name, map_location="cpu", weights_only=False)
        for key in (*_STATE_KEYS, "scheduler"):
            if key in payload:
                result[f"{name}:{key}"] = _canonical_digest(payload[key])
    assert trainer.diagnostics is not None
    result["diagnostics_rows"] = _canonical_digest(trainer.diagnostics.all_rows)
    result["diagnostics_panel"] = _canonical_digest(trainer.diagnostics.panel_rows)
    result["global_step"] = trainer.global_step
    result["cuda_rng"] = _canonical_digest(torch.cuda.get_rng_state())
    return result


def parity(root: Path, kind: str) -> dict[str, Any]:
    """Eager, graph and graph-with-resume fingerprints for one student."""
    device = torch.device("cuda")
    epochs = 3
    fingerprints: dict[str, Any] = {}
    graph_counts: dict[str, Any] = {}
    for label, cuda_graph in (("eager", False), ("graph", True)):
        output = root / label
        trainer, (num_classes, image_size) = build_trainer(output, kind=kind, cuda_graph=cuda_graph, device=device)
        loader, validation_loader = _loaders(num_classes, image_size, pin_memory=True)
        replays: list[int] = []
        history = []
        for epoch in range(epochs):
            history += trainer.fit(loader, validation_loader=validation_loader, epochs=epoch + 1, start_epoch=epoch)
            if trainer._cuda_graph is not None:
                replays.append(trainer._cuda_graph.replays_this_epoch)
        fingerprints[label] = _fingerprint(trainer, history, output)
        if trainer._cuda_graph is not None:
            graph_counts = {"captures": trainer._cuda_graph.captures, "replays_per_epoch": replays}
    # Resume: graph run of one epoch, then a new graph trainer resumes last.pt.
    output = root / "resumed"
    trainer, (num_classes, image_size) = build_trainer(output, kind=kind, cuda_graph=True, device=device)
    loader, validation_loader = _loaders(num_classes, image_size, pin_memory=True)
    history = trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    del trainer
    resumed, _ = build_trainer(output, kind=kind, cuda_graph=True, device=device)
    loader, validation_loader = _loaders(num_classes, image_size, pin_memory=True)
    start = resumed.resume(output / "last.pt", sampler=loader.sampler).next_epoch
    # The uninterrupted runs' diagnostics saw every epoch; a resumed tracker
    # only sees epochs after the resume, so compare training state only.
    history += resumed.fit(loader, validation_loader=validation_loader, epochs=epochs, start_epoch=start)
    fingerprints["resumed"] = _fingerprint(resumed, history, output)
    return {"fingerprints": fingerprints, "graph_counts": graph_counts, "resume_start": start}


def stale_guard(root: Path, mutation: str) -> str:
    """Mutate the optimizer mid-epoch after capture; return the error text."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(root, kind="bn_dropout", cuda_graph=True, device=device)
    loader, validation_loader = _loaders(num_classes, image_size, pin_memory=False)

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
        else:
            raise AssertionError(mutation)

    try:
        trainer.fit(loader, validation_loader=validation_loader, epochs=1, on_batch_start=mutate)
    except RuntimeError as error:
        assert trainer._cuda_graph is not None and trainer._cuda_graph.replays_this_epoch == 2
        return str(error)
    return "no error"


def poisoned_replay(root: Path, poison: str) -> None:
    """Capture on clean data, then replay with poisoned static inputs, bypassing the host-side guard."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(root, kind="bn_dropout", cuda_graph=True, device=device)
    loader, validation_loader = _loaders(num_classes, image_size, pin_memory=False)
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    state = trainer._cuda_graph
    assert state is not None and state.graph is not None and state.images is not None
    torch.cuda.synchronize()
    print("captured", flush=True)
    if poison == "pixel":
        state.images.fill_(1.5)
    elif poison == "nan":
        # NaN passes both [0, 1] guards (all comparisons false) and reaches the loss.
        state.images.fill_(float("nan"))
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
    trainer, (num_classes, image_size) = build_trainer(root, kind="bn_dropout", cuda_graph=True, device=device)
    loader, validation_loader = _loaders(num_classes, image_size, pin_memory=False)
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


_SCRIPT = r"""
import json, sys, tempfile
from pathlib import Path
import torch
torch.use_deterministic_algorithms(True)
sys.path.insert(0, sys.argv[1])
import tests.integration.test_cuda_graph_training_step as module
name, args = sys.argv[2], sys.argv[3:]
with tempfile.TemporaryDirectory() as root:
    result = getattr(module, name)(Path(root), *args)
    print("RESULT " + json.dumps(result), flush=True)
"""


def _run(name: str, *args: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(_ROOT / "src"), str(_ROOT)])
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("kind", ["bn_dropout", "mobilenetv4"])
def test_graph_step_is_bit_identical_to_the_eager_step(kind: str) -> None:
    result = _result(_run("parity", kind))
    fingerprints = result["fingerprints"]
    eager, graph, resumed = fingerprints["eager"], fingerprints["graph"], fingerprints["resumed"]
    # The graph path really ran: captured once per epoch, replayed every full
    # batch after the first (one eager warm-up, one capture-and-replay).
    full_batches = _TRAIN_SIZE // _BATCH
    assert result["graph_counts"] == {"captures": 3, "replays_per_epoch": [full_batches - 1] * 3}
    assert "last.pt:rng" in eager and "best.pt:model" in eager and "last.pt:scheduler" in eager
    assert graph == eager
    # Resume from the epoch-1 checkpoint reproduces the uninterrupted run's
    # training state and rows (the per-run diagnostics tracker restarts).
    assert result["resume_start"] == 1
    for key in eager:
        if key.startswith("diagnostics_"):
            continue
        assert resumed[key] == eager[key], key


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("mutation", ["lr", "momentum", "weight_decay", "load_state_dict", "model_mode"])
def test_mid_epoch_optimizer_change_is_refused_not_replayed(mutation: str) -> None:
    message = _result(_run("stale_guard", mutation))
    assert "CUDA graph is stale" in message


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
