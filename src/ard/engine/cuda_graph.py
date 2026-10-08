"""CUDA-graph training step state for ``training.cuda_graph`` (plan 0105).

The Trainer owns the step body (``Trainer._cuda_graph_body``); this module owns
the buffers the captured graph reads and writes, the capture lifecycle, and the
guard that refuses to replay a graph whose baked-in optimizer values are stale.

Contract, tested in tests/integration/test_cuda_graph_training_step.py:
bitwise equal to the eager step in deterministic mode. With deterministic
algorithms off (cuDNN benchmark stays off) the graph is NOT guaranteed to
replay exactly the eager kernels; what the tests observe is that every RNG
stream stays exactly equal over whole runs, and that one step from one exact
state lands within 4x max(the median spread of the eager outcomes, one FP32
rounding of the new value) of one nearest eager outcome, in every tensor group
(parameters, momentum buffers, BatchNorm buffers, and the weight-EMA state when
``training.weight_ema_decay`` is set). Which term sets the bound
depends on the group: for parameters it is the FP32 rounding, for momentum
buffers at production shapes it is the measured eager spread.

* A graph bakes SGD's ``lr``/``momentum``/``weight_decay``/... in as Python
  scalars and the addresses of every parameter, gradient, momentum buffer,
  BatchNorm buffer and (with a weight EMA) EMA tensor. It is therefore captured again at the start of every epoch
  (the scheduler changes the learning rate only at epoch ends; resume loads
  new optimizer state tensors), and :meth:`CudaGraphStepState.check_replayable`
  compares a fingerprint of all of those before every replay and raises on any
  difference instead of replaying stale values.
* The first full batch of every epoch runs eagerly (it creates momentum
  buffers on a fresh run and warms lazily initialized CUDA libraries before a
  capture); a batch whose shape differs from the captured one (the last
  partial batch) also runs eagerly.
* The random start is drawn outside the graph, into a static buffer, from the
  same per-step generator the eager attack uses.
* Since 2026-10-08 the captured step also covers RSLAD / RSLAD-advT distillation
  (soft-label bank rows copied into static buffers and reconstructed inside the
  step; an online or advT teacher forward runs inside the step when the teacher
  is a frozen eval-mode, allowlisted module), ``method.mixed_batch`` without
  split BN, and ``method.awp`` (proxy reload, proxy SGD step, perturb and
  restore inside the step). The fingerprint then also covers every tensor
  those write or read by address (static buffers, teacher weights, the AWP
  proxy and its optimizer, the advT metric accumulator) and the Python scalars
  they bake in (mixed-batch ``k`` / lambda, AWP gamma and activity).
* Memory: the graph's private pool is released (and the CUDA cache emptied)
  before any eager step that follows a capture (the last partial batch) and at
  the end of every training epoch, so validation, the train probe and eager
  steps never hold their allocations next to the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ard.config.schema import CUDA_GRAPH_ARCHITECTURES

__all__ = [
    "CUDA_GRAPH_ARCHITECTURES",
    "CudaGraphStepState",
    "module_fingerprint",
    "momentum_buffers_ready",
    "optimizer_fingerprint",
    "optimizer_groups_fingerprint",
]


def module_fingerprint(module: nn.Module) -> tuple[Any, ...]:
    """Identity, parameter / buffer addresses and every submodule's mode of a module a captured step uses."""
    return (
        id(module),
        tuple(parameter.data_ptr() for parameter in module.parameters()),
        tuple(buffer.data_ptr() for buffer in module.buffers()),
        tuple(submodule.training for submodule in module.modules()),
    )


def optimizer_fingerprint(
    optimizer: Optimizer, model: nn.Module, *, ema_model: nn.Module | None = None, extra: tuple[Any, ...] = ()
) -> tuple[Any, ...]:
    """Everything a captured step bakes in that the host could change between replays.

    With a weight EMA (``training.weight_ema_decay``) the captured step also
    writes every EMA state tensor in place, so their addresses are included.
    ``extra`` is whatever else the step reads or writes by address or bakes in
    as a Python scalar (static buffers, teacher, AWP proxy and its optimizer,
    mixed-batch ``k``; built by the Trainer).
    """
    groups = _optimizer_groups(optimizer)
    buffers = tuple(buffer.data_ptr() for buffer in model.buffers())
    ema = None if ema_model is None else tuple(value.data_ptr() for value in ema_model.state_dict().values())
    return (id(optimizer), tuple(groups), buffers, model.training, ema, extra)


def optimizer_groups_fingerprint(optimizer: Optimizer) -> tuple[Any, ...]:
    """A second optimizer's hyperparameters and parameter / state addresses (the AWP proxy's SGD)."""
    return (id(optimizer), tuple(_optimizer_groups(optimizer)))


def _optimizer_groups(optimizer: Optimizer) -> list[tuple[Any, ...]]:
    groups = []
    for group in optimizer.param_groups:
        hyperparameters = []
        for key in sorted(group):
            if key == "params":
                continue
            value = group[key]
            if isinstance(value, torch.Tensor):
                # A tensor learning rate takes a different (not bitwise-equal)
                # foreach path; the graph step is defined for Python scalars only.
                raise RuntimeError(f"training.cuda_graph requires a Python-scalar optimizer {key!r}, not a tensor")
            hyperparameters.append((key, value))
        tensors = []
        for parameter in group["params"]:
            buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
            tensors.append((id(parameter), parameter.data_ptr(), None if buffer is None else buffer.data_ptr()))
        groups.append((tuple(hyperparameters), tuple(tensors)))
    return groups


def momentum_buffers_ready(optimizer: Optimizer) -> bool:
    """True once every parameter of a momentum group has its momentum buffer.

    SGD creates the buffer on a parameter's first update; capturing before that
    would record the one-off ``clone`` initialization instead of the update.
    """
    for group in optimizer.param_groups:
        if group.get("momentum", 0) == 0:
            continue
        for parameter in group["params"]:
            if optimizer.state.get(parameter, {}).get("momentum_buffer") is None:
                return False
    return True


@dataclass
class DeferredDiagnostics:
    """One replayed step's per-sample diagnostic inputs, read back at a flush."""

    sample_ids: list[int]
    valid: list[bool]
    labels: list[int]
    clean_predictions: torch.Tensor
    adversarial_predictions: torch.Tensor
    panel_positions: list[int]
    panel_clean_images: torch.Tensor | None
    panel_adversarial_images: torch.Tensor | None
    # RSLAD (policy weights) and RSLAD-advT (teacher on x'); mixed batch: the attacked prefix length.
    kd_weights: torch.Tensor | None = None
    joint_risks: torch.Tensor | None = None
    teacher_predictions: torch.Tensor | None = None
    teacher_entropies: torch.Tensor | None = None
    mixed_count: int | None = None


@dataclass
class CudaGraphStepState:
    device: torch.device
    images: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    # Soft-label bank rows of the batch (int64 top-K indices, fp16 probabilities, fp16 residual mass).
    bank_index: torch.Tensor | None = None
    bank_prob: torch.Tensor | None = None
    bank_residual: torch.Tensor | None = None
    totals: torch.Tensor | None = None
    # method.mixed_batch: the epoch's mixed-batch accumulator (7 slots, as in the eager step).
    mixed_totals: torch.Tensor | None = None
    # method.awp: whether this epoch's step perturbs the weights (constant within an epoch).
    awp_active: bool = False
    graph: torch.cuda.CUDAGraph | None = None
    fingerprint: tuple[Any, ...] | None = None
    # Tensors allocated during capture that stay valid (and are overwritten)
    # at every replay: the adversarial batch and the diagnostic predictions.
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    eager_steps_this_epoch: int = 0
    captures: int = 0
    captures_this_epoch: int = 0
    replays_this_epoch: int = 0
    # Batches that fit the static buffers (full batches) this epoch.
    full_batches_this_epoch: int = 0
    deferred: list[DeferredDiagnostics] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("training.cuda_graph requires a CUDA device")
        self.totals = torch.zeros(9, dtype=torch.float64, device=self.device)

    def invalidate(self) -> None:
        """Drop the captured graph; the next eligible step captures again."""
        self.graph = None
        self.fingerprint = None
        self.outputs = {}

    def begin_epoch(self, *, mixed: bool = False) -> torch.Tensor:
        """Invalidate, zero the epoch accumulator(s), and return the main one (eager steps add into it too)."""
        self.invalidate()
        self.eager_steps_this_epoch = 0
        self.captures_this_epoch = 0
        self.replays_this_epoch = 0
        self.full_batches_this_epoch = 0
        assert self.totals is not None
        self.totals.zero_()
        if mixed:
            if self.mixed_totals is None:
                self.mixed_totals = torch.zeros(7, dtype=torch.float64, device=self.device)
            self.mixed_totals.zero_()
        return self.totals

    def static_fingerprint(self) -> tuple[Any, ...]:
        """Addresses of every static buffer the captured step reads or writes."""
        return tuple(
            None if tensor is None else tensor.data_ptr()
            for tensor in (
                self.images,
                self.labels,
                self.valid,
                self.noise,
                self.bank_index,
                self.bank_prob,
                self.bank_residual,
                self.totals,
                self.mixed_totals,
            )
        )

    def matches(self, images: torch.Tensor, labels: torch.Tensor, *, noise_rows: int | None = None) -> bool:
        """Allocate the static inputs on first use; True iff this batch fits them.

        A non-contiguous batch never fits (the static buffers are contiguous and
        a strided copy is not what the eager step reads); it runs eagerly.
        ``noise_rows`` (default: the batch size) is the number of attacked
        examples whose random start the noise buffer holds (``k`` for a mixed batch).
        """
        if not (images.is_contiguous() and labels.is_contiguous()):
            return False
        if self.images is None:
            if not images.is_floating_point():
                return False
            self.images = torch.empty(images.shape, dtype=images.dtype, device=self.device)
            self.labels = torch.empty(labels.shape, dtype=labels.dtype, device=self.device)
            self.valid = torch.empty(labels.shape, dtype=torch.bool, device=self.device)
            rows = images.shape[0] if noise_rows is None else noise_rows
            self.noise = torch.empty((rows, *images.shape[1:]), dtype=torch.float32, device=self.device)
        assert self.labels is not None
        fits = (
            images.shape == self.images.shape
            and images.stride() == self.images.stride()
            and images.dtype == self.images.dtype
            and labels.shape == self.labels.shape
            and labels.stride() == self.labels.stride()
            and labels.dtype == self.labels.dtype
        )
        if fits:
            self.full_batches_this_epoch += 1
        return fits

    def check_epoch_used_graph(self) -> None:
        """Fail loudly if a cuda_graph epoch with at least three full batches never replayed.

        One eager warm-up batch and one capture-and-replay batch are expected,
        so three full batches must give at least one replay; zero means the
        option silently degraded to eager training.
        """
        if self.full_batches_this_epoch >= 3 and self.replays_this_epoch == 0:
            raise RuntimeError(
                f"training.cuda_graph is enabled but this epoch replayed no CUDA graph "
                f"({self.full_batches_this_epoch} full batches, {self.eager_steps_this_epoch} eager steps)"
            )

    def check_replayable(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        *,
        ema_model: nn.Module | None = None,
        extra: tuple[Any, ...] = (),
    ) -> None:
        if self.graph is None or self.fingerprint is None:
            raise RuntimeError("no captured CUDA graph to replay")
        if optimizer_fingerprint(optimizer, model, ema_model=ema_model, extra=extra) != self.fingerprint:
            raise RuntimeError(
                "CUDA graph is stale: an optimizer hyperparameter, parameter, momentum buffer, model buffer, "
                "EMA tensor, the model mode, a static buffer, the teacher, the AWP proxy or a baked-in step "
                "setting changed since capture; refusing to replay values baked in at capture time"
            )
