"""Scheduler construction with explicit epoch-boundary semantics."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, MultiStepLR

from ard.config.schema import SchedulerConfig


def warmup_multistep_multiplier(
    epoch: int, *, warmup_epochs: int, milestones: tuple[int, ...], gamma: float
) -> float:
    """Pure function: the LR multiplier ``LambdaLR`` applies at a given epoch.

    Linear warmup from ``1/warmup_epochs`` at epoch 0 up to exactly ``1.0``
    at epoch ``warmup_epochs`` (matching Singh/Croce/Hein 2023's own
    pretrained-init recipe -- "peak LR attained at epoch 10" of a 50-epoch
    run), then the usual multistep decay by ``gamma`` at each milestone.
    ``epoch`` is the zero-based epoch index that will run under this LR (the
    same convention ``MultiStepLR`` already uses via ``LRScheduler.last_epoch``).
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    multiplier = 1.0
    for milestone in milestones:
        if epoch >= milestone:
            multiplier *= gamma
    return multiplier


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler:
    """Build a scheduler stepped exactly once after every completed epoch."""
    if config.step_at != "epoch_end":
        raise ValueError(f"unsupported scheduler step point: {config.step_at}")
    if config.id == "identity":
        return MultiStepLR(optimizer, milestones=(), gamma=1.0)
    if config.id == "multistep":
        return MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    if config.id == "warmup_multistep":
        assert config.warmup_epochs is not None
        warmup_epochs, milestones, gamma = config.warmup_epochs, config.milestones, config.gamma
        return LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: warmup_multistep_multiplier(
                epoch, warmup_epochs=warmup_epochs, milestones=milestones, gamma=gamma
            ),
        )
    raise ValueError(f"unsupported validated scheduler: {config.id}")


__all__ = ["build_scheduler", "warmup_multistep_multiplier"]
