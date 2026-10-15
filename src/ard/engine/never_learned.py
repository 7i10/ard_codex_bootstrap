"""Never-learned guard: stop a run whose student never left a constant output.

Debugging 0030: a ConvStem student under PGD-AT from random init collapsed to
an input-independent output in epoch 0 and stayed at chance for all 50 epochs
(about 29 GPU-hours) while the run ended "completed". Loss and accuracy cannot
detect this early: healthy runs start at a training loss of up to 7.15 and a
validation accuracy as low as 0.21%. What a collapsed student cannot fake is
input dependence, so the guard reads two observability metrics of the existing
per-epoch clean validation pass (raw weights, never EMA):

* ``<prefix>_distinct_predictions``: the number of distinct argmax classes over
  the validation images (``<prefix>`` is ``val_subset`` under
  ``training.selection_subset_size``, else ``val``).
* ``<prefix>_logit_std_over_images``: for every class, the population standard
  deviation (ddof 0) of that class's logit across the validation images; the
  metric is the mean of those per-class values. An input-independent network
  has exactly 0 (the collapsed epoch-0 checkpoint of debugging 0030 gave 3e-7).

An epoch is *degenerate* when ``distinct_predictions <= 1`` OR
``logit_std_over_images < 1e-4``. The guard fires only when BOTH epoch 0 and
epoch 1 are degenerate; one degenerate epoch alone never fires. It is not
configurable (there is no config field, so the config hash is unchanged) and
it gates nothing else: no checkpoint, selection, schedule or attack reads it.

Sample-size floor on the distinct-predictions criterion: it applies only when
the validation pass has at least ``NEVER_LEARNED_MIN_IMAGES_FOR_DISTINCT``
(1,000) valid images. On a handful of images an input-dependent but untrained
network routinely predicts one class (the repository's CPU fixtures: 6-24
images, distinct 1, logit std 5e-3 to 0.1, i.e. 50-1000x above the floor), so
"one class" says nothing there. Every production validation set is larger
(CIFAR-10 held-out 5,000; ImageNet subset 5,000; ImageNet-100 about 2,500), so
production behaviour is exactly the OR rule. The logit-std criterion, which is
the direct test of input independence, applies at every size.

Non-finite logits: if any clean validation logit (any epoch, any model passed
to ``Trainer.validate_epoch``) is NaN or inf, validation raises
``NonFiniteValidationLogitsError`` ("clean validation logits are non-finite")
at once, on every DDP rank alike (the count is cross-rank reduced). The guard
never sees such an epoch and no epoch row with a NaN statistic is persisted.

Calibration caveat (review P2-4): the 1e-4 floor is calibrated on CPU only --
the collapsed ConvStem checkpoint (3e-7) against the CPU fixtures and probe
runs (5e-3 and up). Check the epoch-0 and epoch-1 values of the first healthy
GPU runs (bf16/AMP, compiled models) against it. The statistic is a mean over
classes, so one input-dependent class among many constant ones dilutes it by
roughly 1/num_classes (a single live logit with std s over 1,000 classes reads
about s/1000); a run that is constant except for one class could therefore sit
below the floor while not strictly input-independent.

Resume: the trainer evaluates the epochs it observes. A run resumed at epoch
>= 2 never evaluates the guard. A run resumed at epoch 1 is seeded with epoch
0's values from the persisted ``epoch-metrics.jsonl`` row when that row carries
both metrics (see ``observations_from_epoch_rows``); rank zero reads the file
and broadcasts the rows, so every rank holds the same seed, and an unreadable
file fails the resume on every rank. Without the metrics (for example a run
started before they existed) the guard cannot see epoch 0 and stays inactive
for that run. The fire decision is additionally MAX-reduced across ranks, so
all ranks raise together. A run the guard stopped leaves
``never-learned.json`` in its output directory, and ``ard.cli.train`` refuses
``--resume`` while that file exists; deleting it is the deliberate override.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

#: Epochs whose validation must all be degenerate for the guard to fire.
NEVER_LEARNED_EPOCHS: tuple[int, ...] = (0, 1)
#: An epoch is degenerate when at most this many distinct classes are predicted...
NEVER_LEARNED_MAX_DISTINCT_PREDICTIONS = 1
#: ...or when the mean per-class logit std across images is below this value.
NEVER_LEARNED_MIN_LOGIT_STD = 1e-4
#: The distinct-predictions criterion needs at least this many validation images.
NEVER_LEARNED_MIN_IMAGES_FOR_DISTINCT = 1000

DISTINCT_PREDICTIONS_SUFFIX = "distinct_predictions"
LOGIT_STD_SUFFIX = "logit_std_over_images"


class NeverLearnedError(RuntimeError):
    """The student produced a constant (input-independent) output at epochs 0 and 1."""


class NonFiniteValidationLogitsError(FloatingPointError):
    """A clean validation pass produced a NaN or infinite logit."""


def epoch_is_degenerate(
    distinct_predictions: float, logit_std_over_images: float, *, validation_images: float | None
) -> bool:
    """One epoch's verdict. A non-finite std counts as degenerate (no signal).
    ``validation_images`` ``None`` (unknown) disables only the distinct criterion."""
    distinct_applies = validation_images is not None and validation_images >= NEVER_LEARNED_MIN_IMAGES_FOR_DISTINCT
    if distinct_applies and distinct_predictions <= NEVER_LEARNED_MAX_DISTINCT_PREDICTIONS:
        return True
    return not math.isfinite(logit_std_over_images) or logit_std_over_images < NEVER_LEARNED_MIN_LOGIT_STD


def never_learned_reason(
    observations: Mapping[int, tuple[float, float]], *, validation_images: float | None
) -> str | None:
    """Return a failure message when every guarded epoch is observed and degenerate.

    ``observations`` maps an epoch to ``(distinct_predictions,
    logit_std_over_images)``. Missing guarded epochs mean "not evaluable", never
    "degenerate", so the result is ``None`` until both epoch 0 and 1 are present.
    ``validation_images`` is the size of the (fixed) validation set the
    observations were measured on.
    """
    if any(epoch not in observations for epoch in NEVER_LEARNED_EPOCHS):
        return None
    if not all(
        epoch_is_degenerate(*observations[epoch], validation_images=validation_images)
        for epoch in NEVER_LEARNED_EPOCHS
    ):
        return None
    detail = "; ".join(
        f"epoch {epoch}: distinct_predictions={observations[epoch][0]:g}, "
        f"logit_std_over_images={observations[epoch][1]:.3g}"
        for epoch in NEVER_LEARNED_EPOCHS
    )
    return (
        "never-learned guard: the student's clean validation output is constant at epochs "
        f"{', '.join(str(epoch) for epoch in NEVER_LEARNED_EPOCHS)} "
        f"(distinct_predictions <= {NEVER_LEARNED_MAX_DISTINCT_PREDICTIONS} or "
        f"logit_std_over_images < {NEVER_LEARNED_MIN_LOGIT_STD:g} at every one; "
        f"validation images: {validation_images}): {detail}. "
        "See docs/debugging/0030-convstem-dc-offset-pgd-at-collapse.md."
    )


def observation_from_metrics(metrics: Mapping[str, Any], *, prefix: str) -> tuple[float, float] | None:
    """``(distinct, std)`` from an epoch row, or ``None`` when either is absent."""
    distinct = metrics.get(f"{prefix}_{DISTINCT_PREDICTIONS_SUFFIX}")
    std = metrics.get(f"{prefix}_{LOGIT_STD_SUFFIX}")
    if isinstance(distinct, bool) or isinstance(std, bool):
        return None
    if not isinstance(distinct, (int, float)) or not isinstance(std, (int, float)):
        return None
    return float(distinct), float(std)


def observations_from_epoch_rows(rows: Iterable[Mapping[str, Any]], *, prefix: str) -> dict[int, tuple[float, float]]:
    """Guarded-epoch observations from persisted epoch-metric rows (resume seeding)."""
    observations: dict[int, tuple[float, float]] = {}
    for row in rows:
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch not in NEVER_LEARNED_EPOCHS:
            continue
        observation = observation_from_metrics(row, prefix=prefix)
        if observation is not None:
            observations[epoch] = observation
    return observations
