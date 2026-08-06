# Prescriptive v3 closure and utility-oriented pivot

## Status

- Owner: main thread; one Terra analysis owner; one consolidated scientific
  review after real reports are stable
- Branch / base SHA: `master` / `e6aee0b`
- Current milestone: complete -- no intervention candidate nominated
- Last updated: 2026-08-06

## Goal

Close the completed PF-retention and NR-prefix treatment families without
using official test data, determine from existing artifacts whether they had
selected-sample benefit that was cancelled by model-wide spillover, and test
whether teacher response or validation-gradient utility supplies treatment
information beyond prognostic Student history.  Admit at most two new
intervention candidates; launch no new 200-epoch proposal run until the
offline utility gate and leakage-safe evaluation protocol are frozen.

## Non-goals

- Do not retune v1 uniform softening, KD downweight, v2 true-label mixing, v3
  anchor mixing, attack prefix, masks, durations, or schedule on L1/L3.
- Do not use CIFAR-10 official test, AutoAttack, Chen no-harm, unused
  Bartoldson seeds, MobileNetV2, or CIFAR-100 for development decisions.
- Do not treat rescue/harm under a shared trained model as an individual
  causal treatment effect.
- Do not start bilevel training or second-order meta-learning before a cheap
  first-order offline utility audit shows stable headroom.
- Do not repeat checkpoint or W&B inventory searches after a hash-bound
  inventory has been written.

## Existing state

All eight prescriptive-v3 children completed at immutable Git
`97926553ed6773666df915860460b90c353e721d`, one GPU, per-rank batch 128,
local BatchNorm, delayed milestones `[120,170]`.  Validation CE-PGD20 results
are percentages:

| Arm | Seed | Best clean | Best PGD | Best epoch | Last clean | Last PGD | RO gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 1 | 85.64 | 51.88 | 124 | 86.14 | 47.64 | 4.24 |
| C | 2 | 84.72 | 51.74 | 122 | 86.22 | 47.74 | 4.00 |
| PF-H | 1 | 84.28 | 51.72 | 121 | 86.10 | 47.38 | 4.34 |
| PF-H | 2 | 84.74 | 51.78 | 121 | 86.32 | 47.54 | 4.24 |
| PF-R | 1 | 84.22 | 51.76 | 121 | 85.64 | 47.28 | 4.48 |
| PF-R | 2 | 84.90 | 51.66 | 122 | 86.32 | 47.44 | 4.22 |
| NR-H | 1 | 84.32 | 51.72 | 121 | 86.08 | 47.32 | 4.40 |
| NR-H | 2 | 85.30 | 51.60 | 123 | 86.00 | 47.20 | 4.40 |
| NR-R | 1 | 84.64 | 52.06 | 121 | 85.70 | 47.72 | 4.34 |
| NR-R | 2 | 85.36 | 51.60 | 125 | 86.00 | 48.04 | 3.56 |

PF-H minus C averaged `-0.06 pp` Best PGD; NR-H minus C averaged
`-0.15 pp`.  PF-H minus PF-R averaged only `+0.04 pp` and was negative in
seed 1; NR-H minus NR-R averaged `-0.17 pp`.  Both frozen treatment gates are
No-Go.  Student history remains a prognostic result, not an authorized
prescriptive method.

The repository already contains the completed-v2 hash-bound rescue/harm
replay implementation.  It is restricted to the v2 lineage and must be
generalized explicitly rather than bypassed.  Periodic model artifact history
exists at the configured five-epoch cadence, but checkpoint epoch and bytes
must be extracted once into a new immutable inventory before selecting the
common replay panel.

## Scientific contracts affected

- Replay uses raw CIFAR-10 train pixels, stable sparse source IDs, the saved
  CE-PGD20 pixel-space selection attack, and one fixed evaluation seed.
- Comparisons use common epochs, never each arm's separately selected Best
  checkpoint.  Candidate common epochs are `79,99,119,129,149,199`, subject
  to exact artifact availability in the inventory.  Epoch 79 is the shared
  pre-treatment parent and is not replayed once per child.
- The frozen observation-column union includes Student clean/robust
  predictions and margins; teacher clean/student-adversarial predictions,
  true-class probabilities, margins, entropy and KL response; route/mask;
  intervention active state; anchor-alignment quantities for PF; and applied
  PGD prefix identity for NR.
- All reports bind run/config/Git SHA, parent checkpoint and sample-state SHA,
  mask/bundle SHA, checkpoint SHA, attack identity, dataset partition, source
  IDs, CLI/source hash, and input inventory hash.
- Official test and AutoAttack remain sealed.  Validation is development-only
  and cannot later serve as an untouched confirmation set.
- Any future validation-guided method must reserve a meta-probe disjoint from
  checkpoint-selection validation.  Its matched control uses the same reduced
  method-training partition.

## Decisions

### M1 mechanism question

For each PF/NR H/R child against the delayed control at common epochs, report
selected and non-selected rescued, harmed, stable-correct and
unchanged-failure counts, net rescue, clean/robust transition, margin change,
and model-wide spillover.  PF additionally reports Student-anchor versus
teacher target alignment.  NR reports the immediate epochs-80--99 prefix
response and persistence after all samples return to PGD-10.

This analysis changes the next decision:

- no selected-sample benefit: close the treatment mechanism outright;
- selected benefit with larger non-selected harm: retain only the mechanism
  explanation and test whether gradient utility predicts spillover;
- H not better than R within the selected group: history is not a treatment
  selector even if it remains prognostic.

### M2 utility question

Use existing development checkpoints first.  Compare prognostic Student
history with two prescriptive moderators:

1. teacher difficulty response on the Student adversarial input, including
   true-class probability/margin change, correctness and
   `KL(p_T(clean) || p_T(x_adv^S))`;
2. a first-order, detached gradient-utility diagnostic measuring alignment of
   candidate sample KD gradients with a fixed robust meta-probe gradient.

The primary outcome is rescue versus harm under the completed interventions,
not future failure alone.  Report incremental held-out log loss/AUROC where a
predictive model is used, decile-wise monotonicity, prevalence, and actual
top-q overlap.  Splits are by sample ID; no row from one sample may enter both
fit and evaluation folds.  A moderator must have the same signed effect on
both L1/L3 before it can nominate an intervention.

The first-order audit may compare at most two gradient choices: ordinary
teacher-clean adversarial KD and one teacher-response-aware target.  It does
not update model parameters and does not authorize second-order training.

### Candidate and launch boundary

Admit at most one primary and one orthogonal candidate.  A candidate must have
a stated equation, anchor, active period, selector, matched-random control,
meta-probe partition, computation budget, and stop rule before training.
Candidate families are limited to:

- utility-aware adversarial KD weighting; and
- teacher-response-aware target construction.

If neither M2 moderator is stable across L1/L3, stop sample-wise intervention
development and organize the result as `predictable but not actionable`.

If one candidate passes, the smallest development screen reuses the L1/L3
delayed controls and launches H plus matched R for that candidate: four new
continuations, not another unconstrained matrix.  The frozen treatment gate
remains mean Best PGD `>= +0.50 pp` versus C, per-seed nonnegative H-C and
H-R, Best-clean degradation `<=0.50 pp`, and non-worsening mean RO gap.
Only a Development Go proceeds to untouched Bartoldson seeds, Chen no-harm,
official PGD and AutoAttack in that order.

### Baseline closure

Baseline work is independent of proposal tuning.  Its order is Bartoldson
full-SAAD seed 0, Chen full-SAAD seed 0, controlled PGD-AT seed 0, controlled
TRADES seed 0.  Upstream full SAAD remains an isolated fixed-commit reference,
not a silently controlled-protocol result.  Run one bounded runtime/VRAM
smoke before a long full-SAAD job; do not require a five-epoch pilot for every
baseline.

## Milestones

- [x] M0 -- record all eight terminal manifests, exact validation results and
  the No-Go decision; freeze this plan.
- [x] M1 -- write one hash-bound checkpoint inventory, generalize the existing
  rescue/harm analysis to v3, pass one real-checkpoint/sparse-ID CLI smoke,
  then run only the common-epoch replay and point report.
- [x] M2 -- implement and run the offline teacher-response point audit only.
  The first-order gradient audit is skipped by its frozen M1 launch gate.
- [x] M3 -- run one consolidated scientific review and freeze zero, one, or at
  most two candidate interventions.
- [x] M4 -- if and only if M3 admits a candidate, implement its isolated
  boundary, run focused formula/gradient/resume tests and one real-parent
  smoke, then launch the four-child L1/L3 screen.  No candidate was admitted,
  so no implementation or child was launched.
- [x] M5 -- on Development Go only, execute untouched-seed confirmation,
  Chen no-harm and saved-checkpoint official evaluation.  On No-Go, close the
  sample-wise branch and complete the negative-result report.  The No-Go path
  was taken; official test and AutoAttack remained sealed.

## Agent and review budget

The main thread owns contracts and integration.  One Terra writer owns M1/M2
analysis code.  One consolidated scientific review occurs after real point
reports; re-review is limited to an actual P0/P1 delta.  No monitoring agent,
duplicate planner, or Luna pass is needed until result/document
synchronization.  GPU jobs are shell processes, not reasoning-agent tasks.

## Test plan

- Focused T1: inventory hashing, sparse-ID joins, row permutation,
  selected/non-selected exhaustive categories, common-epoch enforcement,
  v2 backward compatibility, non-overwrite and source-hash drift.
- Focused T2: fixed-attack replay equality, teacher/Student mode and gradient
  source, detached utility quantities, target equation and finite values.
- Real smoke: one L1 checkpoint through public CLI, Parquet schema, lineage,
  stable-ID/class joins and report creation before full replay.
- Run `scripts/verify.py --changed --non-scientific` once after the coherent
  implementation delta.  Reuse unchanged cached passes.
- Production training, official PGD, full AutoAttack and full-SAAD runs remain
  outside the automated suite.

## Risks and mitigations

- Rescue/harm has interference.  Report moderation and spillover, not
  individual causal effect.
- Periodic best/last artifact names are mutable aliases.  Inventory exact
  checkpoint epoch and SHA from bytes; never infer epoch from artifact
  version number.
- Validation-gradient utility can overfit the selection set.  Keep the
  current audit exploratory and require a separate meta-probe plus matched
  control before method training.
- Large-teacher replay may exceed the two-hour target.  Time one real
  checkpoint first, freeze the column union, assign longest jobs first and
  parallelize independent feature/outcome jobs.
- L1/L3 have been used repeatedly for development.  They cannot support a
  confirmation claim; unused seeds and official evaluation remain sealed.
- Full SAAD upstream licensing is unresolved.  Execute the pinned source as
  an external oracle only and do not vendor it.

## Progress log

- 2026-08-06: all eight v3 children completed with exit code 0.  Both PF and
  NR failed the preregistered Best-PGD treatment gate; official test and
  AutoAttack were not used.  The research question moved from future-failure
  prognosis to treatment utility and teacher difficulty response.
- 2026-08-06: read-only upstream audit found full-SAAD seed 0 not yet safe to
  launch.  The pinned checkout's local `autoattack` shadows the package API
  expected by pinned RobustBench, upstream has no bounded smoke or AutoAttack
  disable flag, and even one epoch runs full PGD/CW/AutoAttack.  A separately
  locked compatible environment and staged dataset/teacher paths are required;
  any result remains an isolated upstream oracle because upstream lacks
  best/last, resume, separate evaluation and controlled-protocol parity.
- 2026-08-06: the v3 hash-bound inventory/replay/report implementation passed
  22 focused unit tests and one consolidated scientific review.  Four P1
  findings were fixed before launch: explicit control-parent handling,
  analysis/Git provenance checks, epoch-79 NR phase labeling, and a public
  one-checkpoint smoke path.  Child replay now references the exact shared
  epoch-79 parent without repeating its PGD/teacher inference; only the
  control emits the shared-parent baseline rows.
- 2026-08-06: the real PF-H epoch-99 public-CLI smoke covered all 45,000
  sparse train IDs in 250.18 seconds at batch 128; GPU utilization reached
  100% with 3.8 GiB used.  Ten L1/L3 common-epoch panels and both point
  reports then completed.  Ferret was occupied by unrelated work, so the 25
  decision-relevant L3 checkpoints (2.607 GiB) were transferred at about
  111 MB/s; every checkpoint, config, mask and bundle matched its inventory
  SHA before Hamster replay.
- 2026-08-06: M1 did not support the proposed spillover-cancellation
  mechanism.  At the primary PF epoch 119, selected H net rescue was `-2`
  on L1 and `+20` on L3, versus matched-R `-17` and `+37`; non-selected H
  net rescue was `-225` and `+1243`.  At the primary NR epoch 99, selected H
  was `+14/-4` versus R `-9/+13`.  H benefit and H-over-R direction therefore
  did not reproduce.  Because PF selected benefit was non-positive on L1 and
  spillover cancellation was absent on L3, the preregistered gradient-utility
  audit is a No-Launch.  Only the cheap epoch-79 teacher-response point audit
  remains in M2.
- 2026-08-06: M2a ran from tracked-clean analysis commit
  `e67abc8024d9352b35e3e4fb7985206bf1765f5e`.  Its hash-bound point report is
  `76cbf914dc091246994833696d7cebbab41c6cc5febf38c02f371ae513910c06`.
  For PF-H, adding teacher response to Student history changed held-out AUROC
  from `0.6742` to `0.5985` on L1 and `0.3308` to `0.3077` on L3.  For NR-H,
  it changed `0.8000` to `0.8500` and `0.7778` to `0.8889`, with lower log
  loss on both seeds, but the matched-random sensitivity reversed on L1
  (`0.4599` to `0.4444`).  PF failed the primary incremental and decile gates;
  NR failed the preregistered matched-random gate.  The held-out discordant
  support was also small (PF L1/L3 `23/33`, NR `13/6`).  No route was
  nominated, no bootstrap was authorized, and no new training was launched.
- 2026-08-06: the consolidated review found four P1 input-contract gaps
  (real nested dataset schema, complete CE-PGD20 identity, attack-seed/control
  identity, and SampleState range validation).  Focused regressions fixed all
  four; the delta review found no remaining P0/P1.  The completed result is
  therefore `predictable but not actionable` for the tested PF/NR families.

## Completion report

The completed v3 screen did not improve Best PGD and did not reproduce a
History-over-random treatment effect.  The offline mechanism audit rejected
spillover cancellation, and the teacher-response audit nominated neither PF
nor NR.  Student history remains a strong prognostic signal, but none of the
tested target-retention or attack-prefix interventions converts it into stable
treatment utility.  This plan is closed without using official test data,
AutoAttack, untouched confirmation seeds, Chen no-harm runs, or additional
200-epoch training.  The next research plan must either change the causal
intervention question or return to independent baseline closure; it must not
retune these v3 arms on L1/L3.
