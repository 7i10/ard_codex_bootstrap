"""CUDA-graph training step state for ``training.cuda_graph`` (plan 0105).

The Trainer owns the step body (``Trainer._cuda_graph_body``); this module owns
the buffers the captured graph reads and writes, the capture lifecycle, and the
guard that refuses to replay a graph whose baked-in optimizer values are stale.

Contract (bitwise equal to the eager step in deterministic mode):

* A graph bakes SGD's ``lr``/``momentum``/``weight_decay``/... in as Python
  scalars and the addresses of every parameter, gradient, momentum buffer and
  BatchNorm buffer. It is therefore captured again at the start of every epoch
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def optimizer_fingerprint(optimizer: Optimizer, model: nn.Module) -> tuple[Any, ...]:
    """Everything a captured step bakes in that the host could change between replays."""
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
    buffers = tuple(buffer.data_ptr() for buffer in model.buffers())
    return (id(optimizer), tuple(groups), buffers, model.training)


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


@dataclass
class CudaGraphStepState:
    device: torch.device
    images: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    totals: torch.Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    fingerprint: tuple[Any, ...] | None = None
    # Tensors allocated during capture that stay valid (and are overwritten)
    # at every replay: the adversarial batch and the diagnostic predictions.
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    eager_steps_this_epoch: int = 0
    captures: int = 0
    replays_this_epoch: int = 0
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

    def begin_epoch(self) -> torch.Tensor:
        """Invalidate, zero the epoch accumulator, and return it (eager steps add into it too)."""
        self.invalidate()
        self.eager_steps_this_epoch = 0
        self.replays_this_epoch = 0
        assert self.totals is not None
        self.totals.zero_()
        return self.totals

    def matches(self, images: torch.Tensor, labels: torch.Tensor) -> bool:
        """Allocate the static inputs on first use; True iff this batch fits them."""
        if self.images is None:
            if not images.is_floating_point():
                return False
            self.images = torch.empty(images.shape, dtype=images.dtype, device=self.device)
            self.labels = torch.empty(labels.shape, dtype=labels.dtype, device=self.device)
            self.valid = torch.empty(labels.shape, dtype=torch.bool, device=self.device)
            self.noise = torch.empty(images.shape, dtype=torch.float32, device=self.device)
        assert self.labels is not None
        return (
            images.shape == self.images.shape
            and images.dtype == self.images.dtype
            and labels.shape == self.labels.shape
            and labels.dtype == self.labels.dtype
        )

    def check_replayable(self, optimizer: Optimizer, model: nn.Module) -> None:
        if self.graph is None or self.fingerprint is None:
            raise RuntimeError("no captured CUDA graph to replay")
        if optimizer_fingerprint(optimizer, model) != self.fingerprint:
            raise RuntimeError(
                "CUDA graph is stale: an optimizer hyperparameter, parameter, momentum buffer, model buffer or "
                "the model mode changed since capture; refusing to replay values baked in at capture time"
            )
