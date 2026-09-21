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
    then cosine decay from 1.0 at ``epoch == warmup_epochs`` towards (but
    never exactly reaching) 0.0, matching Singh/Croce/Hein 2023's own
    pretrained-init recipe (arXiv:2303.01870, Appendix A.1: linear warmup
    then cosine decay). ``total_epochs`` must be the same value as
    ``TrainingConfig.epochs`` for this run -- the decay curve is defined
    relative to the run's own length, not an absolute epoch count, so a
    different ``total_epochs`` changes the whole curve, not just its tail.

    Plan 0102 scientific review (2026-09-21, P1 finding 2): the post-warmup
    span is ``total_epochs - warmup_epochs`` (the denominator), not
    ``total_epochs - 1 - warmup_epochs`` -- matching
    ``ard.schedules.cosine_value.cosine_anneal``'s own convention
    (``iteration / total_iterations``, endpoint approached but never
    reached) instead of introducing a second, inconsistent convention for
    the same "cosine towards a target" family. This has two effects, both
    intentional: (1) since ``warmup_epochs < total_epochs`` is already
    required below, the post-warmup span is always at least 1 -- there is
    no longer a silent all-zero degenerate case at a short horizon (e.g. a
    12-epoch canary with a 10-epoch warmup no longer collapses to
    multiplier 0.0 for its only two post-warmup epochs); (2) the very last
    epoch of a full run never receives an exact multiplier of 0.0 (a literal
    no-op training step, since AdamW's decoupled weight decay is also
    scaled by the LR) -- it receives a small positive value instead, the
    same way ``cosine_anneal`` never exactly reaches its own endpoint.
    """
    if warmup_epochs >= total_epochs:
        raise ValueError("warmup_cosine requires warmup_epochs < total_epochs")
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    span = total_epochs - warmup_epochs
    progress = (epoch - warmup_epochs) / span
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
