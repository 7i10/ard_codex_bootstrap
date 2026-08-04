# Pre-39 forecasting and prescriptive routing

## Status

- Owner: main thread; one research-planner pass, one Terra implementation
  owner, one consolidated scientific review after real reports are stable
- Branch / base SHA: `research/prescriptive-routing-v3` / `2adce67`
- Current milestone: M0 input freeze
- Last updated: 2026-08-05

## Goal

Use only the completed non-intervened L1--L4 trajectories to determine the
earliest pre-39 point at which student history adds prospective information
over both instantaneous student state and teacher state.  Characterize the
teacher/student states inside the predicted PF/NR groups, then freeze at most
two orthogonal interventions whose effect can be separated from selector and
random-mask effects.  Launch no new training until those contracts are fixed.

In parallel, use the completed history-routing v2 control/history/random arms
to distinguish prognosis from treatment utility.  Determine whether selected
samples were rescued, harmed, or unchanged, and whether true-label mixing was
a near no-op or removed useful soft-label structure before choosing another
treatment.

## Non-goals

- Do not retune the closed uniform-softening, fixed KD-downweight, or
  true-label-anchor families on L1/L3 outcomes.
- Do not use official CIFAR-10 test, AutoAttack, unseen Bartoldson seeds, or
  Chen no-harm seeds for development.
- Do not rerun GPU replay: the schema-v2 feature/outcome column union already
  contains the required teacher and student primitives.
- Do not transfer every periodic checkpoint.  Retrieve only an exact online
  candidate anchor after the cheaper replay-domain point screen passes.
- Do not equate failure prognosis with positive treatment response.
- Do not interpret paired Control/intervention sample outcomes as identifiable
  individual causal effects: the intervention changes the shared model.

## Existing state

- L1--L4 feature replay covers checkpoint epochs
  `4,9,14,...,34,39,...,99`; outcome replay covers `99,104,...,199` under
  the fixed common pixel-space PGD contract.
- The raw replay Parquets and lineage files are available under the existing
  hash-bound `h5-matrix-6e5dcf5` analysis root.  No new image inference is
  required for point estimates, teacher/student comparison, or taxonomy.
- L1/L2 versioned checkpoint bytes are local.  L3/L4 pre-39 checkpoint bytes
  remain on Ferret; only a selected candidate anchor will be exported there
  and returned as a compact, hash-bound online-state artifact.
- Exact-online epoch 39 already predicts peak failure on Bartoldson with
  AUROC `.9116/.9055`, and non-recovery with `.7956/.7843`.  The selected
  true-label target mix did not improve Best validation PGD, so prediction
  success is not accepted as intervention utility.

## Scientific contracts affected

- All features are available at or before their named anchor.  Outcomes use
  the independently seeded common-PGD replay at epochs `99/104/109`; official
  test remains sealed.
- PF is anchor-correct and wrong at at least two of `99/104/109`.  NR is
  anchor-wrong and wrong at all three.  The strata remain disjoint.
- Replay-domain and exact-online scores are named separately.  A replay point
  screen can nominate an anchor but cannot make a deployability claim.
- Exact-online score uses inclusive correctness frequency and margin EMA from
  the checkpoint `SampleStateStore`; future labels and teacher primitives do
  not enter the student score.
- Stable sample ID, class, run/config/Git SHA, teacher/checkpoint SHA, attack
  identity, dataset partition, and input/report hashes are exact.
- Any later intervention must preserve epsilon, steps, step size, attack loss,
  normalization, clean branch, temperature, optimizer/scheduler/RNG, best/last
  selection, W&B identity, and sample-state resume unless its frozen method
  identity explicitly changes one item.

## Decisions

### Two-stage anchor search

1. Run a CPU-only replay-domain point screen at anchors
   `4,9,14,19,24,29,34`.  Epoch 4 is diagnostic only because it has no
   preceding trajectory.
2. For each anchor and PF/NR stratum report an inclusive replay-history score
   (all observations through the anchor), instantaneous student margin,
   teacher normalized entropy, additive Student+Teacher, and
   Student+Teacher+interaction models, together with prevalence, AUROC,
   AUPRC, precision/recall at fixed `q=10%`, and actual top-q overlap.  A
   lead-time score ending five epochs before the anchor is diagnostic only.
   Teacher correctness/margin remain routing moderators rather than post-hoc
   predictor alternatives.
3. A candidate requires, on both Bartoldson development seeds, student-history
   AUROC at least `+0.02` above the best instantaneous-student and best
   teacher-only comparator, with precision@10% no more than `0.01` below
   either comparator.  Chen L2/L4 are cross-teacher diagnostics, not the
   Bartoldson gate.
4. Evaluate anchors in chronological order.  Bootstrap only the earliest
   replay point candidate using the fixed 2,000-replicate, class-stratified,
   sample-ID paired contract.  Require both paired AUROC lower bounds above
   zero.
5. Only then retrieve/export that anchor's exact online state for L1--L4 and
   rerun the same point/paired criteria.  If none passes, retain epoch 39 and
   do not consume more checkpoint bytes.

### Prescriptive routing audit

For the selected top-q masks, report without fitting thresholds:

- teacher clean/adversarial correctness and probability margin;
- wrong-confidence and clean-to-adversarial response;
- student clean-correct/robust-wrong state;
- margin trend and correctness frequency;
- PF, NR, persistent-wrong, recovered-stable, and recovered-relapsed mass;
- teacher/seed-specific route prevalence and cross-seed Jaccard.

The audit may nominate at most two orthogonal interventions.  The first
candidate axis is supervision strength: adversarial-KD downweight only for a
persistent-error selector that materially differs from the failed epoch-99
mixed selector.  The optional second axis is per-sample attack easing, and is
admissible only if teacher-correct persistent errors dominate and measured
headroom justifies changing the training-attack identity.  Exact equations,
route predicates, strength, parent epoch, controls, and Go/No-Go thresholds
must be frozen in a follow-up section before training.  No generic
intervention is launched merely because a sample is predicted difficult.

Each admitted treatment uses a shared delayed-RSLAD control `C`, a
history-selected arm `H`, and a class/anchor-state/teacher-state/count-matched
random arm `R`.  Development uses Bartoldson L1/L3 validation only.  The gate
is mean Best PGD at least `+0.50 pp` versus `C`, both seeds non-negative,
positive mean and non-negative per-seed differences versus `R`, clean loss at
most `0.50 pp`, and non-worsening RO gap.  Reusing delayed controls requires
an exact parent/scheduler/RNG identity proof.

### Completed-v2 rescue/harm audit

Compare the completed Bartoldson L1/L3 delayed controls, history arms, and
matched-random arms under one common evaluation attack and fixed checkpoint
epochs.  Best checkpoints from different epochs must not be joined as a
paired trajectory.  The primary paired snapshots are common peak-window
epochs `99/104/109` and last epoch `199`; best checkpoints remain aggregate
validation results only.

For each stable train sample classify `Control -> arm` as rescued
(`wrong -> correct`), harmed (`correct -> wrong`), stable correct, or unchanged
failure.  Report counts and net rescue (`rescued - harmed`) separately for
PF/NR, history/random selection, selected/non-selected, teacher state, the
peak window, and last.  This is exploratory treatment moderation rather than
an individual treatment-effect estimate.

Audit the actual true-label target change on the same selected IDs.  Exact
quantities derivable from stored teacher probabilities apply to all samples;
full-distribution KL/JS and loss-gradient cosine/norm use one immutable
class/state/teacher-state-stratified panel with fixed attack seeds.  This
distinguishes a near no-op from destruction of dark knowledge without changing
the training or evaluation attacks.

The audit may nominate different treatment axes for PF and NR.  PF temporal
stabilization (past-student consistency) and NR input-side learnability
(explicit training-only attack curriculum) enter the candidate pool but do
not launch until exact equations and controls are frozen.

## Milestones

- [x] M0 -- freeze the existing L1--L4 replay/checkpoint inventory and this
  analysis contract.
  - Files: this plan and one hash-bound input inventory outside Git.
  - Tests: input existence, SHA, schema, sparse-ID/class join.
  - Acceptance: all replay inputs exist; missing exact-online bytes are named
    without downloading them.
  - Commit: included with M1.
- [ ] M1 -- implement the CPU-only pre-39 point screen and routing audit.
  - Files: new analysis module/CLI and focused unit tests; do not change the
    trainer, objective, attack, or frozen H5 estimator.
  - Owner: one Terra implementation pass because estimator/lineage code is a
    bounded scientific-core change.
  - Tests: temporal leakage, row permutation, sparse IDs, exact strata,
    deterministic top-q ties, teacher/student comparator direction, report
    non-overwrite, and input-hash drift.
  - Acceptance: one CLI produces a hash-bound L1--L4 point report from the
    real replay matrix without CUDA or network.
  - Commit: `analysis: add pre39 prescriptive routing audit`.
- [ ] M2 -- apply the point gate, run only the admissible paired bootstrap,
  and, if needed, export one candidate exact-online anchor.
  - Tests: deterministic/resumable bootstrap and exact checkpoint state.
  - Acceptance: earliest anchor is fixed or epoch 39 retained; no GPU replay.
- [ ] M2b -- complete the v2 rescue/harm and target-change audit.
  - Files: a reusable paired-checkpoint observation CLI, CPU report CLI,
    focused tests, and hash-bound reports; no training-loop change.
  - Execution: one real checkpoint smoke, then independent fixed-epoch replay
    jobs assigned longest-processing-time-first across available GPUs.
  - Acceptance: identical attack/seed/data/sample-ID contract across arms,
    exhaustive rescue/harm categories, selected/non-selected spillover,
    target-change and fixed-panel gradient diagnostics, and no official test.
- [ ] M3 -- run one consolidated scientific review and freeze at most two
  intervention candidates plus their no-intervention and matched-random
  controls.
  - Acceptance: every proposed run changes a decision-relevant factor and has
    an explicit stop rule; P0/P1 findings are closed.
- [ ] M4 -- implement only the selected intervention boundary, run focused
  formula/gradient/resume tests and one real-parent one-epoch smoke.
  - Acceptance: non-selected samples and unrelated scientific state are exact;
    W&B/artifact lineage is unique and resumable.
- [ ] M5 -- launch the smallest development screen on the common parent.
  Official test/AA and confirmation seeds remain sealed until Development Go.

## Agent and review budget

Use one planner pass, one Terra writer for M1/M2, and one consolidated
scientific review after the real report.  A follow-up review is allowed only
for a concrete P0/P1 delta.  No monitoring agent, replacement planner, or Luna
sync pass is needed before an API is frozen.

## Test plan

- Focused T1 analysis/unit tests first.
- `scripts/verify.py --changed` once after the coherent analysis delta.
- One real L1 sparse-ID CLI smoke through report writing before the L1--L4
  collection.
- Point estimates before bootstrap; bootstrap runs only after its frozen point
  gate and persists deterministic progress.
- No GPU test until an intervention changes target/loss/gradient or exact
  remote checkpoint export is required.
- T4/T5, official PGD, and AutoAttack are deferred.

## Risks and mitigations

- Reusing development outcomes can overfit the anchor.  Use a fixed anchor
  grid/criteria and confirm any method only on untouched seeds.
- Replay state is not exact online state.  Label it diagnostic and require one
  exact-online confirmation before declaring deployability.
- Taking the best of multiple teacher comparators is conservative for the
  student claim, but it is not a learned teacher model.
- Teacher-wrong mass is very small for Bartoldson.  Report prevalence before
  allocating a route; do not amplify a rare subgroup into a global claim.
- A treatment can harm even when selection is accurate.  Matched random and
  no-intervention controls remain mandatory.
- Rescue/harm violates unit-level no-interference.  Use it to find moderators
  and mechanisms, not to claim per-sample causality.
- Host-local paths are provenance, not portable identity.  Transfer compact
  outputs with SHA verification instead of copying all checkpoints.

## Progress log

- 2026-08-05: Plan opened after `teacher_target_true_label_mix@1` failed its
  preregistered Best-PGD development gate.  Read-only inventory found both
  GPUs idle on Hamster and confirmed that pre-39 replay analysis needs no new
  GPU work.  L3/L4 exact pre-39 state remains remote and will be retrieved only
  for an admissible candidate anchor.
- 2026-08-05: Research-planner review fixed the predictor family, kept teacher
  correctness/margin as routing moderators, fixed the delayed-control H/R/C
  screen, and kept official test, AutoAttack, and confirmation seeds sealed.
- 2026-08-05: Added a completed-v2 rescue/harm and target-change audit.  PF
  temporal stabilization and NR input-side curriculum are hypotheses only;
  neither is an experiment until this audit fixes mechanism and headroom.

## Completion report

Pending.
