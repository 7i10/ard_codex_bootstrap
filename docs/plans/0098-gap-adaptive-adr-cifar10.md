# 0098 — Gap-adaptive ADR: CIFAR-10 validation of a reliability-adaptive λ schedule

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Depends on: decision packet 0011 (`chosen: B`), plan 0097 (SIGN_CONFIRMED,
  fully closed — this plan reuses its arms rather than re-running them)
- Current milestone: design + implementation, pre-review. No GPU job launched
  yet; no source SHA frozen.

## Goal

Plan 0097 confirmed ADR's capacity-inverse self-distillation benefit
(MobileNetV2 +2.79pp AutoAttack vs ResNet-18 Nesterov-matched +1.22pp,
SIGN_CONFIRMED). ADR's own λ/τ schedule (cosine, epoch-indexed, tuned once at
11-46M params by the original paper) was flagged this session as a real risk
for transfer to smaller capacity or different datasets (`docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md`
§9.1). This plan tests one concrete refinement — **replace the fixed λ
schedule with one driven by a live, self-normalizing measurement of how badly
the run is currently robust-overfitting** — cheaply, at CIFAR-10 scale,
before any ImageNet investment (matching packet 0011's recommendation and
its own preregistered cost/risk reasoning).

## Mechanism: Gap-Adaptive ADR

Everything about ADR stays as implemented and validated in plan 0097 —
EMA teacher, per-sample rectification (Eq. 3-5), τ's own cosine anneal,
γ = 0.995 — **except λ's source**. ADR's original: `λ_e = λ_low + (λ_high -
λ_low) · cosine_fraction(e)`, a pure function of epoch index. This plan's
replacement:

1. `raw_gap_e = train_robust_accuracy_e - val_pgd_accuracy_e` — both already
   computed and logged every epoch by the existing training loop
   (`epoch-metrics.parquet`'s `train_robust_accuracy` / `val_pgd_accuracy`
   columns; no new forward pass, no new held-out data).
2. `gap_ema_e = β · gap_ema_{e-1} + (1-β) · raw_gap_e` (β a new hyperparameter,
   default 0.9; `gap_ema_0 = raw_gap_0`) — smooths the pre-decay noise this
   session's own post-hoc check found in the raw per-epoch gap
   (`docs/decisions/0011-post-cifar-adr-next-step.md`'s discussion; verified
   directly against plan 0097's own logs before this plan was written).
3. `severity_e = clip(gap_ema_e / max(ε, running_max(gap_ema_0..e)), 0, 1)` —
   self-normalizing against the worst overfitting *this specific run* has
   shown so far, so no new dataset- or architecture-specific reference
   constant is needed (directly answers the transfer-risk concern above).
4. `λ_e = λ_low + (λ_high - λ_low) · severity_e`, same `λ_low`/`λ_high`
   bounds ADR's paper already uses (0.7 / 0.95).

Computed once per epoch boundary (the natural cadence — `train_robust_accuracy`
and `val_pgd_accuracy` are themselves only available once per epoch), applied
to every iteration in the following epoch.

**Methodological note on using the held-out validation split this way**,
recorded here since it was a live discussion this session, not an
afterthought: this reuses the held-out validation split (already used for
checkpoint selection) as a continuous input to the training objective, not
only for a one-time post-hoc decision. This is a qualitatively stronger use
of that split than early stopping, but it has real, published precedent
(Population Based Training, Jaderberg et al. 2017; Self-Tuning Networks,
MacKay et al. ICLR 2019; MAML-style bilevel meta-learning) — all of which
treat validation feedback as a legitimate training-time signal, on the
condition that a genuinely separate, single-use set is reserved for the final
reported number. This project already has exactly that structure (the
official CIFAR-10 test set, touched once via AutoAttack, never used during
training or checkpoint selection). No change to that convention is made
here. What this *does* mean: this arm's internal validation accuracy is no
longer a clean generalization estimate in the way plan 0097's arms' was — it
is now partly a training signal — and this should be stated plainly in any
report, not silently glossed over.

**What this plan does not change**: no change to ε, attack steps, step size,
random start, normalization, temperature schedule, EMA decay, or evaluation
attacks (per CLAUDE.md rule 6). Checkpoint selection's own convention
(held-out validation, best/last both kept) is unchanged.

## Diagnostic instrumentation added alongside (not gating anything yet)

Per the user's explicit request: add the cheapest train-only candidate signal
found this session — the EMA teacher's argmax agreement with the student's
own prediction, computed on the **training batch** (not validation) — as a
new logged epoch-metric (`train_ema_student_agreement`), purely for
observation this run. This does not gate λ or anything else in this plan; it
exists so this plan's own runs produce, for the first time, a real
apples-to-apples comparison between the validation-based signal (`raw_gap`/
`severity`, used above) and a train-only candidate, without needing a second
campaign. If it looks promising after this run, a follow-up plan can test it
as an alternative or combined gating signal.

Implementation note (from this session's engine investigation): the EMA's
clean-image forward on the training batch already happens every iteration
inside `Trainer._rectified_target` (`src/ard/engine/trainer.py`) for the
rectification step itself — capturing its argmax and comparing to the
student's own post-update clean argmax (already computed for
`train_clean_accuracy`) adds no new forward pass, only a new accumulator and
one new `epoch_metrics` key, following the exact pattern
`train_clean_accuracy` already uses. Known caveat, disclosed rather than
hidden: for plain ADR (not the TRADES variant), the EMA forward is
pre-optimizer-step and the student's own comparison forward is post-step —
a real temporal mismatch, plausibly small given γ=0.995's slow decay, but
not zero. Record this metric as an observation, not a validated proxy, until
checked against the gap-based signal in this run's own results.

## Experimental design

**Reuse, don't re-run.** Plan 0097's `adr`, `mobilenetv2_adr` (vanilla ADR,
3 seeds each) and their matched baselines `pgd_at_nesterov`, `mobilenetv2_pgd_at`
(3 seeds each) are already recorded (`docs/experiments/adr_cifar10_replication_v1.json`)
under the identical protocol this plan targets. This plan launches only the
new arm:

| arm | protocol | seeds | new? |
|---|---|---|---|
| `adr_gap` (ResNet-18, Gap-Adaptive ADR) | `controlled_cifar10_r18_adr_v1` (λ source changed only) | 0, 1, 2 | new |
| `mobilenetv2_adr_gap` (MobileNetV2, Gap-Adaptive ADR) | `controlled_cifar10_mobilenetv2_adr_v1` (λ source changed only) | 0, 1, 2 | new |

6 new training runs (not 12 — packet 0011's original estimate assumed
re-running vanilla ADR too), each with model + EMA weights evaluation
(AutoAttack, official test, matching plan 0097's exact contract). Estimated
cost, using this project's own measured plan-0097 numbers: ~3h/training run
× 6 ≈ 18 GPU-hours, plus evaluation (~2h/checkpoint-pair × 6 ≈ 12 GPU-hours,
using post-packet-0010-fix throughput, likely faster than plan 0097's
incident-affected numbers) ≈ **30 GPU-hours total**, well under packet
0011's ≈85-100 GPU-hour estimate.

## Preregistered decision rule

For each architecture, compute Gap-Adaptive ADR's mean AutoAttack gain over
its own matched plain-AT baseline (reused from plan 0097) and compare to
vanilla ADR's already-recorded gain for the same pair:

- **IMPROVED** if Gap-Adaptive's mean gain exceeds vanilla ADR's mean gain by
  more than the combined per-arm sd (~0.5pp, matching plan 0097's own arms'
  sd) for at least one architecture, and is not worse by more than that
  margin on the other.
- **NULL** if Gap-Adaptive's gain is within ±0.5pp of vanilla ADR's for both
  architectures.
- **WORSE** if Gap-Adaptive's gain is more than 0.5pp below vanilla ADR's for
  either architecture.

A NULL or WORSE result is a real, reportable finding (the mechanism doesn't
help, or the premise from the post-hoc log analysis doesn't transfer to an
active run) — not a reason to iterate on the formula without a fresh
decision packet.

## Implementation checklist

1. New schedule primitive (e.g. `src/ard/schedules/gap_adaptive.py` or
   alongside the existing `cosine_value.py`): pure function taking the
   epoch's `train_robust_accuracy`, `val_pgd_accuracy`, and running state
   (`gap_ema`, `running_max`), returning the new `gap_ema`/`running_max`/`λ`.
   Unit-testable on hand-computed small sequences, matching this project's
   existing schedule-primitive test style.
2. New `AdrConfig` field (e.g. `lambda_source: Literal["cosine", "gap_adaptive"]`,
   default `"cosine"` so plan 0097's existing configs are untouched) and,
   when `"gap_adaptive"`, a `gap_smoothing_beta: float` field (default 0.9).
3. `Trainer` wiring: compute the new λ at each epoch boundary (after that
   epoch's train/val metrics are known, before the next epoch's training
   iterations start), store `gap_ema`/`running_max` in the epoch state so
   checkpoint resume restores them exactly (matching this project's existing
   resume-correctness discipline for other per-run state).
4. `train_ema_student_agreement` diagnostic metric per the instrumentation
   section above — new accumulator + one new `epoch_metrics` key, no gating
   role.
5. New configs: `configs/scientific/cifar10_r18_adr_gap.yaml`,
   `configs/scientific/cifar10_mobilenetv2_adr_gap.yaml` — copies of the
   existing `adr`/`mobilenetv2_adr` configs with only `method.adr.lambda_source:
   gap_adaptive` (and `gap_smoothing_beta` if non-default) changed.
6. Checkpoint round-trip: `gap_ema`/`running_max` added to the saved/restored
   per-run state (not the model's own `state_dict`) — a regression test
   confirming resume reproduces the exact same λ trajectory as an uninterrupted
   run, matching this project's established checkpoint-resume test pattern.

## Tests

- Formula unit test: hand-computed `gap_ema`/`severity`/`λ` on a small fixed
  sequence of `(train_robust_accuracy, val_pgd_accuracy)` pairs, including
  the edge case of a flat/zero gap (severity must not divide by zero).
- Schedule-only test: confirm `lambda_source: cosine` (the default) produces
  bit-identical λ values to plan 0097's existing runs — this plan must not
  silently change anything for configs that don't opt in.
- `train_ema_student_agreement` correctness: on a tiny fixture model,
  hand-verify the logged agreement rate against a manually computed one.
- Checkpoint resume: train a few epochs, resume, confirm `gap_ema`/
  `running_max` and the resulting λ trajectory match an uninterrupted run
  exactly (same pattern as existing sample-store/EMA resume tests).

## Scientific review

This touches `src/ard/engine/trainer.py` (new epoch-boundary computation,
new resumable state) and `src/ard/config/schema.py` (new `AdrConfig` fields)
— per this project's standing rule, needs the scientific-reviewer agent
before any source SHA is frozen for a GPU campaign, same as the original ADR
implementation did in plan 0097's M0.

## Verification, end to end

1. `scripts/verify.py --changed` after implementation.
2. Fixture-scale CPU test confirming the full loop (gap computation → λ →
   rectification → attack → objective → checkpoint → resume) runs without
   shape errors.
3. One real GPU canary (a few epochs) confirming `λ` actually moves in
   response to a real train-val gap, before committing to the full 6-run
   campaign.
4. Full campaign per the experimental design above once the above pass and
   scientific review is complete.
