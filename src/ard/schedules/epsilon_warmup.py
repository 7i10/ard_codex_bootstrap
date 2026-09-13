"""Linear epsilon warmup for adversarial training's inner-maximization budget.

Plan 0101. Debenedetti, Sehwag, Mittal, "A Light Recipe to Train Robust
Vision Transformers" (arXiv:2209.07399, SaTML 2023) found that ramping the
training attack's epsilon linearly from a small value up to the target
budget over the first several epochs, rather than training at the full
budget from epoch 0, measurably improved both clean and AutoAttack robust
accuracy for small architectures. This module is an epsilon warmup
*inspired by* that paper, not a faithful reproduction of it -- three
deliberate deviations, none an oversight:

1. **Epoch indexing, not iteration indexing.** The paper's own released
   code (github.com/dedeswim/vits-robustness-torch, src/attacks.py's
   ``make_linear_schedule``) is not pinned in this project's
   ``.external/`` the way every other adopted baseline is, so whether its
   schedule is epoch- or iteration-indexed could not be independently
   re-verified in-repo (scientific review, plan 0101). This implementation
   is epoch-indexed, matching this project's own ``warmup_multistep_multiplier``
   (``ard.schedules``) convention exactly: ``(epoch + 1) / warmup_epochs``,
   reaching exactly ``target_epsilon`` at epoch ``warmup_epochs - 1`` (the
   last warmup epoch) and staying there, never a literal reading of the
   paper's own ``t/period`` (which would hit exactly 0 at epoch 0 -- a
   fully non-adversarial epoch, flagged by scientific review as wasting
   compute and, worse, as a training-time confound when compared against a
   config that has no warmup at all: an unattacked epoch's clean accuracy
   says nothing about the architecture).
2. **Step size couples to the current ramped epsilon, not the target.**
   This project's own attack code (``ard.attacks.pgd.LinfPGD.generate``)
   asserts ``step_size <= epsilon`` for every sample -- a real invariant
   the paper's own mechanism violates during warmup (their step size is
   fixed at the target epsilon's own ratio throughout). The caller applies
   this project's own resolved step-to-epsilon ratio to the *current*
   ramped epsilon instead, keeping the existing guard meaningful (plan
   0101's "Option 1", scientific-reviewer-verified to satisfy the guard
   for every ratio a resolved ``AttackConfig`` already allows).
3. **3-step training PGD, not the paper's 1-step FGSM.** Unchanged from
   plan 0100's own established recipe.

Report this as "an epsilon warmup inspired by Debenedetti et al.", not "we
replicated arXiv:2209.07399's mechanism."
"""

from __future__ import annotations


def epsilon_warmup_value(epoch: int, *, warmup_epochs: int, target_epsilon: float) -> float:
    """Linear, epoch-indexed ramp reaching ``target_epsilon`` at epoch ``warmup_epochs - 1``.

    ``epoch`` is the zero-based epoch about to run. Matches
    ``warmup_multistep_multiplier``'s exact convention
    (``(epoch + 1) / warmup_epochs``): never exactly 0 (epoch 0 already
    gets a real, if small, attack budget), reaches ``target_epsilon``
    exactly at the last warmup epoch and stays there. Returns
    ``target_epsilon`` unchanged once ``epoch >= warmup_epochs``, and
    always for ``warmup_epochs == 0`` (the "no warmup configured" case,
    identical to every existing config's behavior today).
    """
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return target_epsilon
    return ((epoch + 1) / warmup_epochs) * target_epsilon


__all__ = ["epsilon_warmup_value"]
