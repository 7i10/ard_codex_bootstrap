"""Scheduler construction with explicit epoch-boundary semantics."""

from __future__ import annotations

import math

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


def warmup_cosine_multiplier(epoch: int, *, warmup_epochs: int, total_epochs: int) -> float:
    """Pure function: linear warmup (same shape as ``warmup_multistep_multiplier``),
    then cosine decay from 1.0 at ``epoch == warmup_epochs`` to 0.0 at
    ``epoch == total_epochs - 1`` (the last epoch that runs), matching
    Singh/Croce/Hein 2023's own pretrained-init recipe (arXiv:2303.01870,
    Appendix A.1: linear warmup then cosine decay). ``total_epochs`` must be
    the same value as ``TrainingConfig.epochs`` for this run -- the decay
    curve is defined relative to the run's own length, not an absolute
    epoch count, so a different ``total_epochs`` changes the whole curve,
    not just its tail.
    """
    if warmup_epochs >= total_epochs:
        raise ValueError("warmup_cosine requires warmup_epochs < total_epochs")
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    span = total_epochs - 1 - warmup_epochs
    progress = 1.0 if span <= 0 else min((epoch - warmup_epochs) / span, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig, *, total_epochs: int) -> LRScheduler:
    """Build a scheduler stepped exactly once after every completed epoch.

    ``total_epochs`` is only consumed by ``warmup_cosine`` (its decay curve
    is defined relative to the run's own length); every other scheduler id
    ignores it, matching their existing epoch-count-independent shapes.
    """
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
    if config.id == "warmup_cosine":
        assert config.warmup_epochs is not None
        warmup_epochs = config.warmup_epochs
        return LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: warmup_cosine_multiplier(
                epoch, warmup_epochs=warmup_epochs, total_epochs=total_epochs
            ),
        )
    raise ValueError(f"unsupported validated scheduler: {config.id}")


__all__ = ["build_scheduler", "warmup_multistep_multiplier", "warmup_cosine_multiplier"]
