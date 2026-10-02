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
and no tracking diagnostics.

Guard tests: a mid-epoch optimizer change is refused rather than replayed
stale; an epoch that never replays fails loudly; the in-graph device asserts
fire from inside a replay; a pixel-range violation on the graph path kills
the process before anything is checkpointed; and the ``LinfPGD.perturb`` core
never synchronizes with the host.

training.deterministic=false (end of file): bitwise parity is impossible, so
exact RNG streams and audit counts over whole runs, plus a one-step
equivalence test against eager-vs-eager nondeterministic noise.
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

import ard.engine.trainer as trainer_module
from ard.attacks import AttackRequest, LinfPGD
from ard.cli.train import _seed_everything
from ard.config.schema import CUDA_GRAPH_ARCHITECTURES, AttackConfig, ModelConfig, NormalizationConfig
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
_EPOCHS = 3
_FIXTURE = "fixture_bn_dropout"
_AUDIT_KEYS = ("train_cuda_graph_captures", "train_cuda_graph_replays", "train_cuda_graph_eager_steps")


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
    """Return (student, num_classes, image_size) for a fixture or an allowlisted architecture."""
    if kind == _FIXTURE:
        # A test-only extension of the allowlist, confined to this subprocess.
        trainer_module.CUDA_GRAPH_ARCHITECTURES = CUDA_GRAPH_ARCHITECTURES | {_FIXTURE}  # type: ignore[attr-defined]
        return _bn_dropout_student(3), 3, 8
    config = ModelConfig(
        architecture=kind,  # type: ignore[arg-type]
        num_classes=10,
        pretrained=False,
        normalization=NormalizationConfig(profile="imagenet_standard"),
    )
    return build_student(config, tier="dev"), 10, 32


def _loaders(num_classes: int, image_size: int, *, pin_memory: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = IndexedDataset(SyntheticCIFAR(size=_TRAIN_SIZE, num_classes=num_classes, image_size=image_size, seed=3))
    validation = IndexedDataset(SyntheticCIFAR(size=6, num_classes=num_classes, image_size=image_size, seed=99))
    # Train-probe pass (production stage 1 runs one every epoch): fixed
    # training-partition images, between the graph epochs.
    probe = IndexedDataset(SyntheticCIFAR(size=6, num_classes=num_classes, image_size=image_size, seed=3))

    def ordered(dataset: Any, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=_BATCH,
            sampler=EpochShuffleSampler(len(dataset), seed=5, shuffle=shuffle),
            pin_memory=pin_memory,
            collate_fn=collate_indexed,
        )

    return ordered(train, shuffle=True), ordered(validation, shuffle=False), ordered(probe, shuffle=False)


def build_trainer(
    output: Path, *, kind: str, cuda_graph: bool, device: torch.device, diagnostics: str = "panel"
) -> tuple[Trainer, Any]:
    # Every RNG stream a checkpoint records (Python, NumPy, torch CPU/CUDA), seeded as ard.cli.train
    # does: each arm runs in a fresh process, so an unseeded stream would differ between arms.
    _seed_everything(1234)
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


def arm(root: Path, kind: str, mode: str, diagnostics: str) -> dict[str, Any]:
    """One arm in this (fresh) process: eager | graph | graph_seed_plus_one | resumed."""
    device = torch.device("cuda")
    if mode == "graph_seed_plus_one":
        # Negative control: the graph path draws from a different attack stream.
        original = Trainer._attack_generator

        def shifted(self: Trainer) -> torch.Generator:
            generator = original(self)
            return generator.manual_seed(generator.initial_seed() + 1)

        Trainer._attack_generator = shifted  # type: ignore[method-assign]
    cuda_graph = mode != "eager"
    trainer, (num_classes, image_size) = build_trainer(
        root, kind=kind, cuda_graph=cuda_graph, device=device, diagnostics=diagnostics
    )
    post_seed_cuda_rng = _digest(torch.cuda.get_rng_state())
    loader, validation_loader, probe_loader = _loaders(num_classes, image_size, pin_memory=True)
    loaders = {"validation_loader": validation_loader, "probe_loader": probe_loader}
    if mode == "resumed":
        history = trainer.fit(loader, epochs=1, **loaders)
        del trainer
        trainer, _ = build_trainer(root, kind=kind, cuda_graph=True, device=device, diagnostics=diagnostics)
        loader, validation_loader, probe_loader = _loaders(num_classes, image_size, pin_memory=True)
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
    return fingerprint


def nondeterministic_arm(root: Path, kind: str, mode: str) -> dict[str, Any]:
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
    fingerprint = arm(root, kind, mode, "panel")
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
_SINGLE_STEP_EAGER_REPLICAS = 3
_SINGLE_STEP_GRAPH_REPLICAS = 2


def _state_vector(trainer: Trainer) -> torch.Tensor:
    state = trainer.model.state_dict()
    return torch.cat(
        [value.detach().double().flatten().cpu() for _, value in sorted(state.items()) if value.is_floating_point()]
    )


def single_step_equivalence(root: Path, kind: str) -> dict[str, Any]:
    """One training step from one exact state: graph vs eager, and eager vs eager.

    PGD's sign step makes PGD-AT training chaotic: a 1e-9 relative state
    difference flips the sign of a large share of the attack's input
    gradients within a step or two, so whole nondeterministic runs drift
    apart to the size of a different seed and say little about one step.
    Here every arm instead takes the SAME step from the SAME state: at the
    start of each batch in _SYNC_BATCHES the arm's parameters, BatchNorm
    buffers, momentum buffers and CUDA RNG state are overwritten in place
    (addresses unchanged, so the captured graph stays valid) with the
    reference eager run's, and the state after that one step is compared
    with the reference's. Eager replicas give the nondeterministic noise of
    one step; graph arms must match it; a graph arm with a shifted attack
    stream is the negative control. Distances are relative to the size of
    the reference step's own update.
    """
    device = torch.device("cuda")
    synced: dict[int, dict[str, Any]] = {}
    after: dict[str, dict[int, torch.Tensor]] = {}

    def run(name: str, *, cuda_graph: bool) -> None:
        trainer, (num_classes, image_size) = build_trainer(
            root / name, kind=kind, cuda_graph=cuda_graph, device=device, diagnostics="panel"
        )
        loader, validation_loader, _ = _loaders(num_classes, image_size, pin_memory=True)
        parameters = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
        after[name] = {}

        def hook(epoch: int, batch_index: int, _batch: Any) -> None:
            if batch_index - 1 in _SYNC_BATCHES:
                after[name][batch_index - 1] = _state_vector(trainer)
            if batch_index in _SYNC_BATCHES:
                if name == "reference":
                    synced[batch_index] = {
                        "model": {key: value.detach().clone() for key, value in trainer.model.state_dict().items()},
                        "momentum": [
                            trainer.optimizer.state[p]["momentum_buffer"].detach().clone() for p in parameters
                        ],
                        "cuda_rng": torch.cuda.get_rng_state(),
                        "vector": _state_vector(trainer),
                    }
                else:
                    state = synced[batch_index]
                    with torch.no_grad():
                        for key, value in trainer.model.state_dict().items():
                            value.copy_(state["model"][key])
                        for parameter, buffer in zip(parameters, state["momentum"], strict=True):
                            trainer.optimizer.state[parameter]["momentum_buffer"].copy_(buffer)
                    torch.cuda.set_rng_state(state["cuda_rng"])
            if batch_index > max(_SYNC_BATCHES):
                raise _StopEpoch

        try:
            trainer.fit(loader, validation_loader=validation_loader, epochs=1, on_batch_start=hook)
        except _StopEpoch:
            pass
        if cuda_graph:
            graph_state = trainer._cuda_graph
            assert graph_state is not None, name
            assert (graph_state.captures_this_epoch, graph_state.replays_this_epoch) == (1, 3), name
        assert sorted(after[name]) == list(_SYNC_BATCHES), name

    run("reference", cuda_graph=False)
    for index in range(_SINGLE_STEP_EAGER_REPLICAS):
        run(f"eager{index}", cuda_graph=False)
    for index in range(_SINGLE_STEP_GRAPH_REPLICAS):
        run(f"graph{index}", cuda_graph=True)
    original = Trainer._attack_generator

    def shifted(self: Trainer) -> torch.Generator:
        generator = original(self)
        return generator.manual_seed(generator.initial_seed() + 1)

    Trainer._attack_generator = shifted  # type: ignore[method-assign]
    run("negative", cuda_graph=True)
    result: dict[str, Any] = {}
    for batch_index in _SYNC_BATCHES:
        reference = after["reference"][batch_index]
        update = float((reference - synced[batch_index]["vector"]).norm())
        assert update > 0
        result[str(batch_index)] = {
            name: float((values[batch_index] - reference).norm()) / update
            for name, values in after.items()
            if name != "reference"
        }
        # One FP32 rounding of every element of the new state, on the same scale.
        result[f"{batch_index}:fp32_rounding"] = torch.finfo(torch.float32).eps * float(reference.norm()) / update
    result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    result["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    return result


def stale_guard(root: Path, mutation: str) -> str:
    """Mutate the optimizer mid-epoch after capture; return the error text."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(root, kind=_FIXTURE, cuda_graph=True, device=device)
    loader, validation_loader, _ = _loaders(num_classes, image_size, pin_memory=False)

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


def never_replays(root: Path) -> str:
    """cuda_graph on, but no capture is ever possible: the epoch must fail, not silently train eagerly."""
    device = torch.device("cuda")
    trainer, (num_classes, image_size) = build_trainer(root, kind=_FIXTURE, cuda_graph=True, device=device)
    trainer_module.momentum_buffers_ready = lambda _optimizer: False  # type: ignore[assignment]
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
    trainer, (num_classes, image_size) = build_trainer(root, kind=_FIXTURE, cuda_graph=True, device=device)
    loader, validation_loader, _ = _loaders(num_classes, image_size, pin_memory=False)
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    state = trainer._cuda_graph
    assert state is not None and state.graph is not None and state.images is not None
    torch.cuda.synchronize()
    print("captured", flush=True)
    if poison == "pixel":
        state.images.fill_(1.5)
    elif poison == "nan":
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
    if determinism == "deterministic":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # A nondeterministic production run sets no cuBLAS workspace override.
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
    ignored = {"cuda_rng_advanced", "audit", "has_probe"}
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
@pytest.mark.parametrize("mutation", ["lr", "momentum", "weight_decay", "load_state_dict", "model_mode"])
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
# 2. One step from one exact state (single_step_equivalence): the graph's
#    capture step and a later replay, each taken from the reference eager
#    run's exact state, must land as close to the reference's result as eager
#    replicas do: at most _SAME_ORDER times the largest eager-vs-eager
#    distance, or _SAME_ORDER times one FP32 rounding of the new state when
#    the eager replicas happen to agree bitwise (below that a difference is a
#    summation-order effect, which is all nondeterministic kernels change). A
#    graph negative control (attack seed + 1) must be at least
#    _NEGATIVE_CONTROL_MARGIN times farther, so the distance resolves any
#    real defect (a missed, extra or stale op moves the step by far more than
#    a rounding).
_SAME_ORDER = 4.0
_NEGATIVE_CONTROL_MARGIN = 100.0
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
_NONDETERMINISTIC_CASES = [
    (_FIXTURE, "nondeterministic"),
    *((architecture, "nondeterministic") for architecture in sorted(CUDA_GRAPH_ARCHITECTURES)),
    *((architecture, "nondeterministic_benchmark") for architecture in sorted(CUDA_GRAPH_ARCHITECTURES)),
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


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize(("kind", "determinism"), _NONDETERMINISTIC_CASES)
def test_nondeterministic_graph_step_matches_eager_within_one_step_noise(kind: str, determinism: str) -> None:
    result = _result(_run("single_step_equivalence", kind, determinism=determinism))
    assert result["deterministic_algorithms"] is False
    assert result["cudnn_benchmark"] is (determinism == "nondeterministic_benchmark")
    for batch_index in map(str, _SYNC_BATCHES):
        distances = result[batch_index]
        eager = [distances[f"eager{index}"] for index in range(_SINGLE_STEP_EAGER_REPLICAS)]
        graph = [distances[f"graph{index}"] for index in range(_SINGLE_STEP_GRAPH_REPLICAS)]
        noise = max(*eager, result[f"{batch_index}:fp32_rounding"])
        summary = f"batch {batch_index}: {distances}, fp32 rounding {result[f'{batch_index}:fp32_rounding']}"
        assert max(graph) <= _SAME_ORDER * noise, summary
        assert distances["negative"] >= _NEGATIVE_CONTROL_MARGIN * _SAME_ORDER * noise, summary
