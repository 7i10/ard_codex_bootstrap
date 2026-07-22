"""Small single-server distributed helpers with safe non-DDP defaults."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

T = TypeVar("T")


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_ready() else 0


def get_world_size() -> int:
    return dist.get_world_size() if distributed_ready() else 1


def is_rank_zero() -> bool:
    return get_rank() == 0


def initialize_from_env(requested_device: str) -> tuple[torch.device, bool]:
    """Initialize torchrun's single-node environment and select the local device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if world_size == 1:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(requested_device), False
    if distributed_ready():
        raise RuntimeError("distributed process group was already initialized")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = requested_device != "cpu" and torch.cuda.is_available()
    if requested_device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return device, True


def barrier() -> None:
    if distributed_ready():
        dist.barrier()


def teardown() -> None:
    if distributed_ready():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> DistributedDataParallel:
    """Wrap a model for single-server DDP with PyTorch's normal buffer sync."""
    device_ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)


@contextmanager
def suspend_ddp_buffer_broadcasts(model: torch.nn.Module) -> Iterator[None]:
    """Suppress buffer mutation during an additional pre-backward forward.

    RSLAD-family objectives retain the adversarial forward graph while doing a
    clean student forward. DDP's pre-forward BatchNorm buffer broadcast would
    mutate a tensor saved by the retained graph. Preserve the ordinary first
    forward and next-step synchronization while making only the additional
    forward exception-safe.
    """
    if not isinstance(model, DistributedDataParallel):
        yield
        return
    broadcast_buffers = model.broadcast_buffers
    require_forward_param_sync = model.require_forward_param_sync
    model.broadcast_buffers = False
    try:
        yield
    finally:
        model.broadcast_buffers = broadcast_buffers
        # A no-grad DDP forward sets this false in its post-forward hook.
        # Restore it so the next ordinary forward performs its expected
        # parameter/buffer synchronization.
        model.require_forward_param_sync = require_forward_param_sync


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def run_rank_zero_phase(operation: Callable[[], None], *, phase: str) -> None:
    """Run a filesystem phase only on rank zero and broadcast its outcome.

    A second broadcast can coordinate the subsequent write phase, so rank-zero
    exceptions never strand peers at a barrier.
    """
    if not distributed_ready():
        operation()
        return
    outcome: list[dict[str, Any] | None] = [None]
    if is_rank_zero():
        try:
            operation()
        except Exception as exc:
            outcome[0] = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
        else:
            outcome[0] = {"ok": True}
    dist.broadcast_object_list(outcome, src=0)
    result = outcome[0]
    if result is None or not result.get("ok", False):
        error_type = "UnknownError" if result is None else result.get("type", "UnknownError")
        message = "rank zero returned no outcome" if result is None else result.get("message", "")
        raise RuntimeError(f"rank-zero {phase} failed ({error_type}): {message}")


def run_rank_zero_value(operation: Callable[[], T], *, phase: str) -> T:
    """Run a read-only rank-zero operation and return one broadcast value."""
    if not distributed_ready():
        return operation()
    outcome: list[dict[str, Any] | None] = [None]
    if is_rank_zero():
        try:
            outcome[0] = {"ok": True, "value": operation()}
        except Exception as exc:
            outcome[0] = {"ok": False, "type": type(exc).__name__, "message": str(exc)}
    dist.broadcast_object_list(outcome, src=0)
    result = outcome[0]
    if result is None or not result.get("ok", False):
        error_type = "UnknownError" if result is None else result.get("type", "UnknownError")
        message = "rank zero returned no outcome" if result is None else result.get("message", "")
        raise RuntimeError(f"rank-zero {phase} failed ({error_type}): {message}")
    return cast(T, result["value"])


def reduce_sums(values: torch.Tensor) -> torch.Tensor:
    if distributed_ready():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def reduce_min(value: torch.Tensor) -> torch.Tensor:
    """Return the cross-rank minimum without mutating the caller's tensor."""
    reduced = value.detach().clone()
    if distributed_ready():
        dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
    return reduced


def reduce_max(value: torch.Tensor) -> torch.Tensor:
    """Return the cross-rank maximum without mutating the caller's tensor."""
    reduced = value.detach().clone()
    if distributed_ready():
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced


def gather_objects(value: Any) -> list[Any]:
    if not distributed_ready():
        return [value]
    gathered: list[Any] = [None] * get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered
