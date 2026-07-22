"""Scheduler construction with explicit epoch-boundary semantics."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from ard.config.schema import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler:
    """Build a scheduler stepped exactly once after every completed epoch."""
    if config.step_at != "epoch_end":
        raise ValueError(f"unsupported scheduler step point: {config.step_at}")
    if config.id == "identity":
        return MultiStepLR(optimizer, milestones=(), gamma=1.0)
    if config.id == "multistep":
        return MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    raise ValueError(f"unsupported validated scheduler: {config.id}")


__all__ = ["build_scheduler"]
