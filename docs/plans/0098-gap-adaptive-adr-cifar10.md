# 0098 — Gap-adaptive ADR: CIFAR-10 validation of a reliability-adaptive λ schedule

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Depends on: decision packet 0011 (`chosen: B`), plan 0097 (SIGN_CONFIRMED,
  fully closed — this plan reuses its arms rather than re-running them)
- **Shelved (decision packet 0012, `chosen: C`, 2026-09-11).** Implementation
  and tests are complete and green (see Progress log), but scientific review
  plus an offline replay against plan 0097's own recorded epoch trajectories
  found the λ-severity mechanism as specified saturates almost immediately
  (epoch 15-25/200) on a spurious early signal, then stays pinned near
  λ_high for 43-75% of training — not the continuously-adaptive schedule
  this plan intended. No fix found this session (free or costed) resolves
  the core issue, since real robust overfitting is itself monotonic once it
  starts, defeating any "compare against your own historical worst"
  normalization regardless of measurement cleanliness. No GPU job launched;
  no source SHA frozen. The code (schedule primitive, config fields, Trainer
  wiring, checkpoint round-trip, tests) is left in place, uncommitted, for a
  future session to build on if a magnitude-sensitive (not history-relative)
  severity formulation is found.

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

## Progress log

- 2026-09-11: implementation complete against this plan's checklist:
  `src/ard/schedules/gap_adaptive.py` (new), `AdrConfig.lambda_source`/
  `gap_smoothing_beta` (`src/ard/config/schema.py`), `Trainer` epoch-boundary
  wiring + checkpoint round-trip + `train_ema_student_agreement` diagnostic
  (`src/ard/engine/trainer.py`), two new configs
  (`configs/scientific/cifar10_r18_adr_gap.yaml`,
  `configs/scientific/cifar10_mobilenetv2_adr_gap.yaml`), and unit +
  integration tests (`tests/unit/test_gap_adaptive_schedule.py`,
  `tests/unit/test_adr_config_lambda_source.py`,
  `tests/integration/test_adr_trainer.py`, `tests/unit/test_pilot_observability.py`).
  `scripts/verify.py --changed` green (all impacted T0-T3 tiers). No source
  SHA frozen yet, no GPU touched.
- 2026-09-11: scientific-reviewer ran on the full uncommitted diff.
  **Verdict: conditional fail — do not freeze a SHA for the 6-run campaign
  yet.** No P0. Five P2 items were fixed directly in this session (none
  changed the mechanism): `train_lambda_floor`/`train_gap_ema`/
  `train_gap_running_max` are now logged per epoch (previously only
  reconstructable from a checkpoint's `fork_lineage`); the
  pre-vs-post-optimizer-step temporal-mismatch caveat comment was corrected
  to state it applies to `adr_trades` too, not only plain `adr`; two new
  integration tests pin the *actually applied* per-iteration lambda in both
  `cosine` and `gap_adaptive` modes (closing a real gap — every prior test
  only checked post-epoch state, never what iteration `e` actually used);
  two new tests exercise `train_ema_student_agreement`'s real argmax/mask/
  division logic (every prior test fed or asserted a literal `0.0`);
  `docs/SCIENTIFIC_INVARIANTS.md` and `AdrConfig`'s docstring were corrected
  — they previously asserted lambda *always* anneals by per-iteration
  cosine, which is now conditionally true only for `lambda_source: cosine`.
  **Two P1 items were left open — mechanism-design questions, not code bugs,
  and per CLAUDE.md rule 7 the human decides them, not this session:**
  1. **The severity ratchet.** `running_max` in `gap_adaptive_step` includes
     the *current* epoch's `gap_ema` before dividing (correct per this
     plan's own formula, `running_max(gap_ema_0..e)` inclusive of `e`), so
     `severity == 1.0` (`λ = λ_high`) on every epoch that sets a new
     smoothed-gap high. Since robust overfitting typically widens
     monotonically after the LR decay, the likely real trajectory is
     `λ = λ_low` until the gap first turns positive, then `λ ≈ λ_high` for
     most of the rest of training — closer to a one-time step than a
     continuously adaptive schedule. An IMPROVED result under this plan's
     preregistered rule would then be hard to attribute to *adaptivity*
     specifically, versus simply "λ reaches 0.95 earlier than cosine does."
  2. **What `raw_gap` actually measures.** `train_robust_accuracy` (10-step
     `kl`/`rectified` attack, `model.train()` batch-norm statistics,
     epoch-averaged over updating weights) and `val_pgd_accuracy` (20-step
     `ce` selection attack, eval-mode running-statistics batch-norm,
     post-epoch snapshot weights) differ in threat identity and model mode,
     not only in split. Their difference is not a clean train/val
     generalization gap; it is dominated by "a weaker attack in train-mode
     BN vs. a stronger attack in eval-mode BN," which independently drives
     finding 1's early ratchet.
  Three further P2 items remain open, deferred rather than fixed, since
  none blocks correctness of what is already implemented and two depend on
  how the P1 items resolve: (a) the campaign's own eventual aggregator
  (not yet written — plan 0097's `scripts/aggregate_adr_cifar10_replication.py`
  covers only its own 8 arms) must assert `method.adr.lambda_source` as an
  identity field, since `adr_gap`/`mobilenetv2_adr_gap` are otherwise
  indistinguishable from plan 0097's vanilla-ADR arms by protocol+method id
  alone; (b) adding `AdrConfig.lambda_source`/`gap_smoothing_beta` changed
  `config_hash` for every existing ADR config (defaults are digested too),
  so no plan-0097 ADR checkpoint can be *resumed* at a source SHA including
  this diff — evaluation of already-completed 0097 checkpoints is
  unaffected (it hashes the saved YAML mapping, not this schema); this must
  be stated in this plan's own completion notes so it isn't a mid-campaign
  surprise; (c) a mixed-schema `epoch-metrics.parquet` read is untested
  (harmless today only because (b) already refuses the resume that would
  trigger it).
  **Next action is the human's**: decide how to resolve P1-1/P1-2 (redesign
  `raw_gap`'s inputs to share threat identity and BN mode; redefine
  `severity`'s normalization to not saturate at 1.0 on every new high; run
  the mechanism as specified and treat the interpretability caveat as a
  disclosed limitation; or decide the offline check the plan already asked
  for — replaying real plan-0097 `train_robust_accuracy`/`val_pgd_accuracy`
  columns through this formula — settles it well enough to proceed as-is).
  No GPU canary launches and no source SHA freezes until that decision is
  made.
- 2026-09-11: the human requested the offline check directly. Replayed
  `gap_adaptive_step` (β=0.9, λ_low=0.7, λ_high=0.95) against all six
  already-recorded plan-0097 ADR epoch trajectories
  (`runs/adr-cifar10-campaign-v1/cifar10_{r18,mobilenetv2}_adr-s{0,1,2}/
  train/epoch-metrics.parquet`), zero GPU time. **P1-1 confirmed
  dramatically**: λ reaches λ_high by epoch 15-17/200 (ResNet-18) or
  21-25/200 (MobileNetV2) and stays there 68-75% / 43-46% of the full run.
  Root cause, confirmed by inspecting the per-epoch `train_robust_accuracy`/
  `val_pgd_accuracy` values directly (not just the aggregate schedule
  output): the real run shows two distinct regimes — a small, roughly flat
  gap (~0.02-0.07) through epoch ~15-99, then a sharp, sustained,
  monotonically worsening explosion starting exactly at epoch 100 (the
  first configured LR-decay milestone) through epoch 199 — the classical
  "robust overfitting accelerates right after LR decay" signature. The
  severity mechanism saturates during the small early plateau (epoch
  15-25), not the large post-100 event, because a causal "compare to your
  own historical worst" metric cannot distinguish "a new record because
  history is still short" from "a new record because something dramatic
  happened" — it reacts to the first, tiny rise and has no headroom left
  when the real, large one arrives. The human further asked whether P1-1
  is downstream of P1-2: very likely yes for the *timing* of the false
  trigger (the small pre-100 plateau is a plausible fingerprint of the
  train/val measurement mismatch — real robust overfitting classically
  should not move much before the first LR decay), but P1-1's saturation
  *itself* is independent of P1-2 and does not resolve if only P1-2 is
  fixed: real post-100 robust overfitting is itself monotonic, so any
  running-max-based normalization saturates on it too, cleanly measured or
  not. Two candidate fixes were explored offline against the same six
  trajectories: (a) decaying the running-max reference — **made things
  worse** (ceiling occupancy rose to 85-93%), confirming mathematically
  that lowering the reference only makes the current value exceed it
  sooner; (b) restarting gap/severity tracking at the scheduler's own first
  LR-decay milestone (epoch 100 here, no new hyperparameter) instead of
  epoch 0 — removes the spurious early trigger, but the restarted series'
  first tracked point is trivially "the worst so far" by construction, so
  ResNet-18 saturates within 1 epoch of the restart and MobileNetV2 within
  9, landing at 82-100% ceiling occupancy for the rest of training — i.e.
  close to a two-value step function keyed to the LR-decay epoch, not the
  continuously-adaptive schedule this plan intended, even though it is a
  legitimate, zero-cost, well-motivated design in its own right.
  **Decision (decision packet 0012, `chosen: C`): shelve.** Neither the
  free fix (epoch-100 restart, reduces to a step function) nor the costed
  fix (matching train/val measurement, does not touch the saturation
  itself) resolves what this plan actually wanted to test. No further work
  on this mechanism this session; implementation stays in place,
  uncommitted, for a future session with a magnitude-sensitive (not
  historical-max-relative) severity formulation. See decision packet 0012
  for the full evidence and reasoning.
