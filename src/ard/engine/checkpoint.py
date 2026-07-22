"""Atomic, complete epoch-boundary checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .distributed import gather_objects, get_rank, get_world_size, run_rank_zero_phase, unwrap_model

REQUIRED_KEYS = {
    "format_version",
    "epoch",
    "epoch_boundary",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "rng",
    "sampler_epoch",
    "sampler_state",
    "sample_state",
    "global_step",
    "best_metric",
    "selection_metadata",
    "tracker_run_id",
    "config_hash",
    "world_size",
}


@dataclass(frozen=True)
class TrainingState:
    next_epoch: int
    global_step: int
    best_metric: float
    selection_metadata: dict[str, Any]
    tracker_run_id: str | None
    sample_state: dict[str, Any]


def config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": None,
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError as exc:
            raise RuntimeError("checkpoint contains NumPy RNG state but NumPy is unavailable") from exc


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Any,
    sampler: Any,
    sample_state: dict[str, Any],
    global_step: int,
    best_metric: float,
    selection_metadata: Mapping[str, Any],
    tracker_run_id: str | None,
    config_hash: str,
) -> None:
    local_sampler_state = sampler.state_dict() if sampler is not None and hasattr(sampler, "state_dict") else {}
    rng_by_rank = gather_objects(capture_rng_state())
    sampler_state_by_rank = gather_objects(local_sampler_state)
    sampler_epoch_by_rank = gather_objects(getattr(sampler, "epoch", epoch))

    def write_atomic_checkpoint() -> None:
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "epoch_boundary": "end",
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "rng": rng_by_rank,
            "sampler_epoch": sampler_epoch_by_rank,
            "sampler_state": sampler_state_by_rank,
            "sample_state": sample_state,
            "global_step": global_step,
            "best_metric": best_metric,
            "selection_metadata": dict(selection_metadata),
            "tracker_run_id": tracker_run_id,
            "config_hash": config_hash,
            "world_size": get_world_size(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    # Every rank enters the same outcome broadcast.  A rank-zero filesystem
    # error therefore propagates instead of stranding peers at a barrier.
    run_rank_zero_phase(write_atomic_checkpoint, phase=f"checkpoint write {path.name}")


def _optimizer_to(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def validate_resume_checkpoint(path: Path, *, expected_config_hash: str) -> None:
    """Check reuse eligibility before a CLI creates or overwrites any output."""
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must be a mapping")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError("checkpoint is incomplete; missing: " + ", ".join(sorted(missing)))
    if payload["epoch_boundary"] != "end":
        raise ValueError("only epoch-boundary checkpoints can be resumed")
    world_size = payload["world_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("checkpoint world size must be a positive integer")
    if world_size != get_world_size():
        raise ValueError(f"checkpoint world size {world_size} does not match {get_world_size()}")
    if payload["config_hash"] != expected_config_hash:
        raise ValueError("checkpoint config hash does not match resolved config")


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Any,
    sampler: Any,
    expected_config_hash: str,
    device: torch.device,
) -> TrainingState:
    validate_resume_checkpoint(path, expected_config_hash=expected_config_hash)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError("checkpoint is incomplete; missing: " + ", ".join(sorted(missing)))
    unwrap_model(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to(optimizer, device)
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint is missing scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint is missing scaler state")
        scaler.load_state_dict(payload["scaler"])
    rank = get_rank()
    rng_by_rank, sampler_state_by_rank = payload["rng"], payload["sampler_state"]
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != get_world_size():
        raise ValueError("checkpoint RNG state is incomplete for the saved world size")
    if not isinstance(sampler_state_by_rank, list) or len(sampler_state_by_rank) != get_world_size():
        raise ValueError("checkpoint sampler state is incomplete for the saved world size")
    if sampler is not None:
        sampler.load_state_dict(sampler_state_by_rank[rank])
    restore_rng_state(rng_by_rank[rank])
    return TrainingState(
        next_epoch=int(payload["epoch"]) + 1,
        global_step=int(payload["global_step"]),
        best_metric=float(payload["best_metric"]),
        selection_metadata=dict(payload["selection_metadata"]),
        tracker_run_id=payload["tracker_run_id"],
        sample_state=dict(payload["sample_state"]),
    )
