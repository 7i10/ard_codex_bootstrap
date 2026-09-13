"""Linear epsilon warmup for adversarial training's inner-maximization budget.

Plan 0101. Debenedetti, Sehwag, Mittal, "A Light Recipe to Train Robust
Vision Transformers" (arXiv:2209.07399, SaTML 2023) found that ramping the
training attack's epsilon linearly from 0 up to the target budget over the
first several epochs, rather than training at the full budget from epoch 0,
measurably improved both clean and AutoAttack robust accuracy for small
architectures. Mechanism verified against the paper and the authors' own
released code (github.com/dedeswim/vits-robustness-torch, src/attacks.py's
``make_linear_schedule``): epoch-indexed, linear,
``eps(t) = (t/period) * final_eps`` for ``t < period``, else ``final_eps``.

This project's own attack code (``ard.attacks.pgd.LinfPGD.generate``)
asserts ``step_size <= epsilon`` for every sample -- a real invariant
Debenedetti's own mechanism violates during warmup (their step size is
fixed at the *target* epsilon's own ratio throughout, not the currently-
ramped value). Plan 0101 deliberately does not replicate that part: the
caller applies this project's own existing step-to-epsilon ratio (2/3,
Salman et al. 2020) to the *current* ramped epsilon instead of the target,
keeping the existing guard meaningful. This is a documented, deliberate
deviation from Debenedetti's exact recipe (see the plan's "Option 1 vs.
Option 2" discussion), not an oversight.
"""

from __future__ import annotations


def epsilon_warmup_value(epoch: int, *, warmup_epochs: int, target_epsilon: float) -> float:
    """Linear, epoch-indexed ramp from 0 at epoch 0 to ``target_epsilon`` at ``warmup_epochs``.

    ``epoch`` is the zero-based epoch about to run, matching
    ``warmup_multistep_multiplier``'s own convention (``ard.schedules``).
    Returns ``target_epsilon`` unchanged once ``epoch >= warmup_epochs``,
    and always for ``warmup_epochs == 0`` (the "no warmup configured" case,
    identical to every existing config's behavior today).
    """
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return target_epsilon
    return (epoch / warmup_epochs) * target_epsilon


__all__ = ["epsilon_warmup_value"]
