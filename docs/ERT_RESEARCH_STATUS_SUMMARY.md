# ERT / RSLAD research status summary

Status: consolidated navigation readout of completed development experiments
through 2026-09-05, revised on 2026-09-05 after the measurement-design,
coefficient and arm-registry audits and the evidence reclassification built on
them. This document is a navigation and interpretation aid; the linked
reports and hash-bound JSON records remain the sources of truth. Results are
internal validation results unless explicitly stated otherwise. The only official
CIFAR-10 test and AutoAttack result is the last ledger row (`I100` official test,
2026-09-05); every other row is internal validation.

## Executive conclusion

Two global augmentation results are real and replicated: `+1.612 pp` for
`CROPSHIFT` over five seeds, and `+0.864 pp` (`SD 0.247`, `t(4) = 7.82`) for the
frozen epoch-100 switch `I100` (`CROPSHIFT` through epoch 99, then `IDBH_WEAK`),
whose direction also survives on the official 10,000-example CIFAR-10 test set
under standard AutoAttack in all three unseen confirmation seeds. Against them,
every sample-level intervention produced a large, reproducible effect on the
samples it treated -- 13x to 33x the train-split noise -- and no held-out effect
our screens could resolve. That contrast is the finding. At learning rate 0.1 a
forked control differs from its sibling as much as a fresh seed does, so a
five-epoch screen cannot see below about 2.5 pp while the interventions aimed at
0.1--2 pp. We therefore report a **measured ceiling on what this apparatus can
detect**, not a refutation of sample-level intervention. One hypothesis was
genuinely refuted at its own preregistered target; the rest remain open and
untested. `I100` is therefore the global incumbent; history and routing signals
remain mechanism evidence rather than promoted methods.

## Measurement limits

Two runs differing only in their random number stream land `1.14--1.25 pp` apart
before the epoch-100 learning-rate decay and about `0.25--0.50 pp` apart after
it. The pre-decay floor is flat in horizon -- already `1.14 pp` after five extra
epochs -- so a verdict stabilises at the learning-rate decay, not by training
longer. The post-decay bracket is interpolated from two adjacent designs and has
never been measured for this design. Many verdicts below were reached under a
floor larger than the effect they judged, and several arms were run more than
once under different names. See [`measurement design`](MEASUREMENT_DESIGN.md),
[`coefficient audit`](COEFFICIENT_AUDIT.md), [`arm registry`](ARM_REGISTRY.md),
and the row-by-row [`evidence reclassification`](EVIDENCE_RECLASSIFICATION.md).

## Main evidence ledger

| question | result | decision |
|---|---|---|
| Does `CROPSHIFT` improve canonical RSLAD? | Final robust gain `+1.32/+1.16 pp` (seed 1/2); AUC gain `+0.895/+0.899 pp`. | Supported as the fixed-prefix baseline; clean final change is small negative (`-0.32/-0.28 pp`). See [`static trajectory`](ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md). |
| Do stronger static policies help throughout training? | `CROP_RE` and `IDBH_WEAK` improve final robustness, but lose full-trajectory AUC on both seeds (`-0.509` to `-0.794 pp`). | No automatic promotion. See [`static family`](ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md). |
| Can late augmentation recover both properties? | `IDBH_WEAK@100` gains final robustness `+1.20/+1.04 pp`, full AUC `+0.502/+0.313 pp`, and post-switch AUC `+1.010/+0.629 pp` in development. In unseen confirmation, independent epoch-199 robust gain over matched `CROP_SUFFIX` is `+0.78/+0.68/+0.62 pp`. | `I100` is frozen as the current global incumbent; the finite timing screen is closed. See [`stage-wise`](ERT_RSLAD_STAGEWISE_AUGMENTATION.md), [`timing`](ERT_RSLAD_SINGLE_SWITCH_TIMING.md), and [`unseen confirmation`](ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md). |
| Does Student history predict future failure? | Online margin-history ranks correlate strongly with strong replay (`rho` about `.85--.88`); hard-state agreement is about `.79--.81`. | Predictive signal is real, but prediction does not imply intervention utility. See [`online proxy`](ERT_ONLINE_ROUTING_PROXY_RESULTS.md). |
| Does history-smoothed S3 routing improve robustness? | Majority-3 reduces switches by about `52.6%` and re-entry by about `66%`, but is below BASE at epoch 94 in both seeds; exit-2 adds no benefit. | Routing stabilization is supported; robust-accuracy promotion is not. See [`history production`](ERT_S3_HISTORY_PRODUCTION_RESULTS.md). |
| Do fixed Stage-A treatments transfer? | T1 weak AdvCE produces a large, replicated direct effect on its selected training cohort (`+4.409/+2.210 pp`). Its held-out effect was never measurable at this design: 5 paired blocks are needed to see +1.5 pp and 40 to see +0.5 pp; one block per seed was run. The softening and T3 ablations flip sign between campaigns by about the size of the shift between the two campaigns' controls, so they resolve nothing in either direction. | No automatic route promotion. See [`Stage A`](ERT_STAGE_A_TREATMENT_RESULTS.md) and [`T123`](ERT_CONFIRMATORY_T123_RESULTS.md). |
| Can Clean-Wrong treatment solve robustness? | Extra CleanCE improves clean recovery on the direct cohort but leaves a clean/robust trade-off; reliability-gated variants were screened at epoch 94 under a floor of 1.25 pp that exceeds every effect observed; the screen does not distinguish the gate from no gate in either direction. | Teacher-margin gate is not confirmed as a practical intervention selector. See [`broad screen`](ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md) and [`gated CleanCE`](ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md). |
| Is action utility heterogeneous by Teacher margin? | Yes descriptively in the fixed train cohort; action rankings are more aligned in high-margin bins, but this is not held-out router validation. | Keep as a hypothesis map; do not choose Q5/thresholds post hoc. See [`margin action map`](ERT_CW_MARGIN_ACTION_MAP.md). |
| What causes run-to-run divergence? | Shuffle, augmentation, and their interaction can cause local 1--2 pp changes; source dominance differs by seed. | Preserve the frozen RNG contract; do not infer a universal dominant source. See [`RNG decomposition`](ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md) and [`shuffle/augmentation`](ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md). |
| Does dynamic S2 boundary-distance treatment improve the I100 trajectory? | At e114, DPM vs Control is `+0.08/+0.12 pp`; D-BDD vs DPM is `-0.04/+0.08 pp`. Corrected S-BDD became non-finite in both development seeds. | No D-BDD promotion or e199 extension. S-BDD is `NUMERICALLY_UNSUPPORTED`; any stabilized secant redesign needs a new contract. See [`BDD recovery`](ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md) and [`secant forensic`](ERT_RSLAD_I100_SECANT_BOUNDARY_DISTANCE_FORENSIC.md). |
| Does online pre-update S2×T1 preservation (OS-PMP / OS-DBDP) improve the I100 trajectory? | e114 held-out CE-PGD20 vs Control: PMP `+0.14/+0.20 pp`, DBDP `+0.14/+0.06 pp` (both seeds positive); DBDP−PMP `+0.00/-0.14 pp`. The post-decay floor for this design has never been measured; it is bracketed at 0.25--0.50 pp, needing 17--141 paired blocks. Two were run. | Classification: **DIRECTIONAL, UNRESOLVED**, at a single untested coefficient. No promotion, no e199/seed/official-test extension without a new contract. See [`online-state S2`](ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md). |
| Does the I100 advantage survive on the official CIFAR-10 test set under AutoAttack? | Epoch-199 last-checkpoint AutoAttack on the official 10,000-example test set, I100 minus matched `CROP_SUFFIX`: `+0.41/+0.06/+0.46 pp` on three unseen confirmation seeds, mean `+0.31 pp`, against the internal-validation reference these runs were meant to test (`+0.78/+0.68/+0.62 pp` on CE-PGD20). All six seed x checkpoint measurements are positive. Clean cost `-0.11 pp` on average. | CONFIRMED under the preregistered rule, and the only official-test result in the project. The mean sits inside the unmeasured 0.25--0.50 pp post-decay floor bracket, so the direction is established but the magnitude is at the resolution limit; no promotion. See [`official test`](ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md). |

## Interpretation boundaries

- Student history is useful for forecasting and trajectory description, but
  the experiments do not show that forecasting alone identifies a treatment
  that generalizes to held-out samples.
- Direct rescue/net-rescue on a selected training cohort is not a substitute
  for held-out CE-PGD20 accuracy. Several experiments demonstrate this gap.
- The Clean-Wrong action map is heterogeneous and informative for designing
  the next hypothesis, but it does not validate a margin threshold, a Q5-only
  route, or a combined treatment.
- The stage-wise augmentation result is a global trajectory intervention,
  separate from Student-history-conditioned routing. It should not be used to
  claim that dynamic routing has been solved.
- Canonical Student robustness states remain `S1` (adversarial correct,
  non-fragile), `S2` (adversarial correct, fragile), and `S3` (adversarial
  wrong); clean correctness is an independent flag. Historical overlays do not
  replace these predicates.
- The S-BDD result is a numerical/formula finding, not evidence that a
  floor/cap/smoothed reciprocal variant is effective. No such variant has been
  evaluated under the current contract.
- All conclusions above are descriptive two-development-seed evidence. They
  are not population-level seed inference. The screens were also not powered
  for the effects they sought, so a negative reading is not evidence of
  absence.

## Runtime and compute conclusion

The bounded runtime audit used the real Chen configuration and checkpoint. On
Hamster, the eager reference is `679.1 img/s`; `torch.compile` is `21--23%`
slower and fails strict one-step parity, while pinning/non-blocking and worker
changes produce no material gain. On idle Ferret, GPU0/1 reach `599.65/607.05`
img/s, but GPU2 reaches only `375.95 img/s`; NUMA1 binding raises GPU2 to
`424.99 img/s`. GPU2's `SYS`/NUMA path and its `2.54x` slower PGD10 segment
explain the historical `360--370 img/s` rows. No production runtime code was
changed. Schedule long jobs on Hamster, medium independent jobs on Ferret
GPU0/1, and short jobs on GPU2 (prefer NUMA1 binding when appropriate). See
[`runtime audit`](ERT_RSLAD_RUNTIME_PERFORMANCE_AUDIT.md).

## Current next-step discipline

Open hypotheses, ranked by how cheaply a powered test could run
([`evidence reclassification`](EVIDENCE_RECLASSIFICATION.md), Part 3(c)):

1. Measure the post-decay floor itself: 3 untreated control replicates per seed, e101--114, endpoints at e104/109/114 (**2.2 GPU-h**). Prerequisite for every item below.
2. TPFM on the Clean-Wrong cohort, re-run post-decay, 5 seeds, 2 controls per seed (about 1 day).
3. Pair-margin / boundary-distance at a different coefficient: add one neighbour arm at each sampling-band edge to any future post-decay screen (+2 arms).
4. A smaller AdvCE coefficient (0.5x the calibrated `0.070959`), post-decay, 5 seeds (about 1 day).
5. History-informed routing below `+0.5 pp`: needs `k >= 11--31` blocks, so it waits on item 1 (weeks).

Standing discipline, unchanged by the reclassification:

- Keep frozen `I100` as the global augmentation incumbent; do not reopen the
  completed single-switch timing grid from these results.
- Do not promote a Student-history, Teacher-margin, Clean-Wrong, DPM, or D-BDD
  route from direct training effects alone.
- Treat a stabilized indirect/secant BDD proposal as a new scientific contract
  requiring a new calibration and reviewed plan; it is not a technical retry of
  S-BDD.
- For future long jobs, use the measured GPU-specific longest-processing-time-
  first policy; do not wait for another runtime audit unless the hardware or
  software stack changes.
- Preserve metrics-only W&B tracking and local checkpoint/run-bundle storage;
  the runtime benchmark wrote only local JSON and uploaded no model artifacts.

## Provenance

The individual reports link their source Git SHAs, parent/checkpoint hashes,
attack identities, and machine-readable result artifacts. This summary was
written after the idle-Ferret follow-up benchmark and is intentionally not a
replacement for those records.
