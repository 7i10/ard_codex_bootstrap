"""Gap-adaptive lambda: a reliability-driven replacement for ADR's cosine lambda.

Plan 0098. ADR's own lambda schedule (``ard.schedules.cosine_value``) is a
pure function of epoch index, tuned once on the paper's own architectures.
This module replaces lambda's *source* only -- everything else about ADR
(EMA teacher, per-sample rectification, tau's own cosine anneal) is
unchanged -- with one driven by a live, self-normalizing measurement of how
badly the run is currently robust-overfitting: the gap between train robust
accuracy and held-out validation PGD accuracy, both already computed and
logged every epoch by the existing training loop. See
``docs/plans/0098-gap-adaptive-adr-cifar10.md`` for the full mechanism and
its methodological rationale (in particular, the note on reusing the
held-out validation split as a training-time signal).

A pure function plus a small immutable state, matching this project's
existing schedule-primitive shape (``ard.schedules.cosine_value``) so it
composes with checkpoint resume the same way this project's other
per-run router state does (``Trainer.fork_lineage``): the caller persists
``GapAdaptiveState`` and passes it back in at the next epoch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

# A numerical safety net only (denominator floor so a flat or non-positive
# gap cannot divide by zero) -- not a scientific hyperparameter, never
# tuned, and small enough to be negligible whenever running_max is a real
# positive gap.
_DENOMINATOR_FLOOR = 1e-6


@dataclass(frozen=True)
class GapAdaptiveState:
    """Resumable state: the smoothed gap and the worst gap seen so far."""

    gap_ema: float
    running_max: float

    def to_dict(self) -> dict[str, float]:
        return {"gap_ema": self.gap_ema, "running_max": self.running_max}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> GapAdaptiveState:
        return cls(gap_ema=float(payload["gap_ema"]), running_max=float(payload["running_max"]))


def _severity(state: GapAdaptiveState) -> float:
    return min(1.0, max(0.0, state.gap_ema / max(_DENOMINATOR_FLOOR, state.running_max)))


def lambda_from_state(state: GapAdaptiveState, *, lambda_low: float, lambda_high: float) -> float:
    """The lambda a given state implies -- a pure function of the state.

    Used both by ``gap_adaptive_step`` and by checkpoint resume (recomputed
    from the restored state rather than cached, so a resumed run's lambda
    trajectory is bit-identical to an uninterrupted one by construction).
    """
    if lambda_low > lambda_high:
        raise ValueError("lambda_low must not exceed lambda_high")
    return lambda_low + (lambda_high - lambda_low) * _severity(state)


def gap_adaptive_step(
    *,
    train_robust_accuracy: float,
    val_pgd_accuracy: float,
    previous_state: GapAdaptiveState | None,
    beta: float,
    lambda_low: float,
    lambda_high: float,
) -> tuple[GapAdaptiveState, float]:
    """One epoch-boundary update: the new state and the lambda it implies.

    ``previous_state`` is ``None`` only for the very first epoch this
    schedule governs (``gap_ema_0 = raw_gap_0`` per plan 0098); every later
    call must supply the state this function returned last time.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    raw_gap = train_robust_accuracy - val_pgd_accuracy
    if previous_state is None:
        gap_ema = raw_gap
        running_max = gap_ema
    else:
        gap_ema = beta * previous_state.gap_ema + (1.0 - beta) * raw_gap
        running_max = max(previous_state.running_max, gap_ema)
    state = GapAdaptiveState(gap_ema=gap_ema, running_max=running_max)
    return state, lambda_from_state(state, lambda_low=lambda_low, lambda_high=lambda_high)


__all__ = ["GapAdaptiveState", "gap_adaptive_step", "lambda_from_state"]
