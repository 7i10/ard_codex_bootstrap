"""Per-iteration cosine annealing for a plain scalar, not an optimizer LR.

``build_scheduler`` (this package's ``__init__``) is hard-wired to
``torch.optim.lr_scheduler`` and only ever drives the learning rate, stepped
once per epoch.  ADR (Wu, Wang & Chen, "Annealing Self-Distillation
Rectification Improves Adversarial Training", ICLR 2024, arXiv:2305.12118)
anneals two unrelated scalars -- the EMA-teacher softening temperature and the
label-trust floor lambda -- on a *per-iteration* cosine schedule, entirely
outside the optimizer.  This is that primitive, matching the paper's own
implementation exactly (``src/util/utils.py`` in the official code,
github.com/yuyuwu5/ADR): a plain function of the current iteration, not a
stateful object, so it composes with resume (the current global step already
determines the value, nothing to checkpoint separately).
"""

from __future__ import annotations

import math


def cosine_anneal(*, start: float, end: float, iteration: int, total_iterations: int) -> float:
    """Cosine-anneal from ``start`` at iteration 0 towards ``end``.

    ``iteration`` and ``total_iterations`` are both zero-based iteration
    counts (not epochs): iteration ``total_iterations - 1`` is the last
    iteration of training, matching the paper's code
    (``iters = np.arange(epochs * niter_per_ep - warmup_iters)``, indexed
    ``0..len(iters)-1``).  Because of that indexing the endpoint is
    approached but never exactly reached (``cos(pi*(N-1)/N) != -1``); this
    mirrors the published implementation rather than "fixing" it, since the
    replication's point is to match the paper, not to improve on it.
    """
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if not 0 <= iteration < total_iterations:
        raise ValueError("iteration must lie in [0, total_iterations)")
    progress = math.cos(math.pi * iteration / total_iterations)
    return end + 0.5 * (start - end) * (1.0 + progress)


__all__ = ["cosine_anneal"]
