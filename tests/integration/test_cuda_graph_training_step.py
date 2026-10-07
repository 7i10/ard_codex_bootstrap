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


def _admit_candidate(kind: str) -> None:
    """Test-only allowlist extension for one candidate, confined to the test subprocess."""
    if kind in _CANDIDATE_ARCHITECTURES:
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


def _student(kind: str) -> tuple[nn.Module, int, int]:
    """Return (student, num_classes, image_size) for a fixture or an allowlisted architecture."""
    if kind == _FIXTURE:
        # A test-only extension of the allowlist, confined to this subprocess.
        trainer_module.CUDA_GRAPH_ARCHITECTURES = CUDA_GRAPH_ARCHITECTURES | {_FIXTURE}  # type: ignore[attr-defined]
        return _bn_dropout_student(3), 3, 8
    _admit_candidate(kind)
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
# Graph arms with a deliberate defect confined to one step, set after the sync
# and before the capture step (so the stale-graph guard does not see it). Each
# must FAIL the equivalence rule: they show it resolves parameter-update errors.
_SINGLE_STEP_CONTROLS = ("attack_seed_plus_one", "lr_zero", "lr_times_1p001", "weight_decay_zero")


def _single_step_trainer(
    spec: dict[str, Any], output: Path, *, cuda_graph: bool
) -> tuple[Trainer, DataLoader, DataLoader]:
    """A PGD-AT trainer and a five-batch synthetic loader for the one-step comparison."""
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
    optimizer = SGD(
        student.parameters(),
        lr=spec["learning_rate"],
        momentum=0.9,
        weight_decay=spec["weight_decay"],
        nesterov=True,
    )
    size = 5 * spec["batch"]
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(
            AttackConfig(epsilon=spec["epsilon"], step_size=spec["step_size"], steps=spec["steps"], random_start=True)
        ),
        selection_attack=LinfPGD(
            AttackConfig(epsilon=spec["epsilon"], step_size="1/255", steps=1, student_mode="eval", teacher_mode="eval")
        ),
        objective=PGDATObjective(),
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

    def loader(dataset_size: int, seed: int) -> DataLoader:
        dataset = SyntheticCIFAR(size=dataset_size, num_classes=num_classes, image_size=spec["image_size"], seed=seed)
        return DataLoader(
            IndexedDataset(dataset),
            batch_size=spec["batch"],
            sampler=EpochShuffleSampler(dataset_size, seed=5, shuffle=True),
            pin_memory=True,
            collate_fn=collate_indexed,
        )

    return trainer, loader(size, 3), loader(spec["batch"], 99)


def _single_step_state(trainer: Trainer) -> dict[str, Any]:
    """Parameters, model buffers, SGD momentum buffers and the CUDA RNG state, copied to the host."""
    names = {name for name, _ in trainer.model.named_parameters()}
    state = trainer.model.state_dict()
    parameters = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
    return {
        "parameters": {key: value.detach().cpu().clone() for key, value in state.items() if key in names},
        "buffers": {key: value.detach().cpu().clone() for key, value in state.items() if key not in names},
        "momentum": [trainer.optimizer.state[p]["momentum_buffer"].detach().cpu().clone() for p in parameters],
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def _load_single_step_state(trainer: Trainer, saved: dict[str, Any]) -> None:
    """Overwrite the trainer's state in place: tensor addresses, and so a captured graph, stay valid."""
    state = trainer.model.state_dict()
    parameters = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
    with torch.no_grad():
        for key, value in state.items():
            value.copy_(saved["parameters"][key] if key in saved["parameters"] else saved["buffers"][key])
        for parameter, buffer in zip(parameters, saved["momentum"], strict=True):
            trainer.optimizer.state[parameter]["momentum_buffer"].copy_(buffer)
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
                group = trainer.optimizer.param_groups[0]
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
    return torch.cat([value.double().flatten() for value in values if value.is_floating_point()])


_GROUPS = ("parameters", "momentum", "buffers")


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
    flat = {name: {group: _flat(state[group]) for group in _GROUPS} for name, state in states.items()}
    result: dict[str, Any] = {"update": {}, "floor": {}, "relative_update": {}}
    for group in _GROUPS:
        new, old = flat["reference"][group], _flat(before[group])
        update = float((new - old).norm())
        result["update"][group] = update
        result["relative_update"][group] = update / float(old.norm())
        result["floor"][group] = torch.finfo(torch.float32).eps * float(new.norm()) / update

    def distance(left: str, right: str) -> dict[str, float]:
        return {
            group: float((flat[left][group] - flat[right][group]).norm()) / result["update"][group] for group in _GROUPS
        }

    # The reference only supplies the state and the scale: it is alone in its process, and a
    # different process can make a different cuDNN algorithm choice.
    eager = [name for name in states if name.split(".")[-1].startswith("eager")]
    others = [name for name in states if name not in eager and name != "reference"]
    result["eager_nearest"] = {
        name: {group: min(distance(name, other)[group] for other in eager if other != name) for group in _GROUPS}
        for name in eager
    }
    result["to_eager"] = {name: {other: distance(name, other) for other in eager} for name in others}
    result["cross_process"] = {
        group: max(distance(f"{process}.eager_a", "reference")[group] for process in processes) for group in _GROUPS
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
        for group in _GROUPS
    )
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
        for group in _GROUPS
    }
    bound = {group: _SAME_ORDER * max(spread[group], result["floor"][group]) for group in _GROUPS}
    arms = {}
    for name, to_eager in result["to_eager"].items():
        nearest = min(to_eager, key=lambda eager: max(to_eager[eager][group] / bound[group] for group in _GROUPS))
        arms[name] = {
            "nearest": nearest,
            **to_eager[nearest],
            "ratio": max(to_eager[nearest][group] / bound[group] for group in _GROUPS),
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
        summary = single_step_summary(result)
        for group in _GROUPS:
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
@pytest.mark.parametrize(("kind", "determinism"), _NONDETERMINISTIC_CASES)
def test_nondeterministic_graph_step_matches_eager_within_one_step_noise(
    tmp_path: Path, kind: str, determinism: str
) -> None:
    spec = {**_SINGLE_STEP_REGIME, "kind": kind}
    spec.update(_SINGLE_STEP_OVERRIDES.get(kind, {}))
    distances = run_single_step_check(tmp_path, spec, determinism)
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
        *sorted(CUDA_GRAPH_ARCHITECTURES),
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
