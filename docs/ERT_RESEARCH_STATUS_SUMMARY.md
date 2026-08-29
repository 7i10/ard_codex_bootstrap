# ERT / RSLAD research status summary

Status: consolidated readout of the completed development experiments through
2026-08-29. This document is a navigation and interpretation aid; the linked
reports and hash-bound JSON records remain the sources of truth. Results are
internal validation results unless explicitly stated otherwise. No official
CIFAR-10 test or AutoAttack result is included.

## Executive conclusion

The strongest reproducible global result so far is the static `CROPSHIFT`
augmentation baseline. It improves Chen ERT validation CE-PGD20 robustness
over canonical RSLAD in both development seeds and improves the full robust
trajectory AUC. More complex sample-level interventions can predict difficult
examples or improve the treated training cohort, but they have not yet
produced a stable held-out robust-accuracy improvement. The next scientific
step should therefore preserve `CROPSHIFT` as the incumbent while treating
history/routing signals as mechanism evidence, not as a promoted method.

## Main evidence ledger

| question | result | decision |
|---|---|---|
| Does `CROPSHIFT` improve canonical RSLAD? | Final robust gain `+1.32/+1.16 pp` (seed 1/2); AUC gain `+0.895/+0.899 pp`. | Supported as the current incumbent candidate; clean final change is small negative (`-0.32/-0.28 pp`). See [`static trajectory`](ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md). |
| Do stronger static policies help throughout training? | `CROP_RE` and `IDBH_WEAK` improve final robustness, but lose full-trajectory AUC on both seeds (`-0.509` to `-0.794 pp`). | No automatic promotion. See [`static family`](ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md). |
| Can late augmentation recover both properties? | `IDBH_WEAK@100` gains final robustness `+1.20/+1.04 pp`, full AUC `+0.502/+0.313 pp`, and post-switch AUC `+1.010/+0.629 pp`; `CROP_RE@100` is also positive but smaller. | Promising stage-wise evidence, still two-seed internal validation and human-review gated. See [`stage-wise`](ERT_RSLAD_STAGEWISE_AUGMENTATION.md). |
| Does Student history predict future failure? | Online margin-history ranks correlate strongly with strong replay (`rho` about `.85--.88`); hard-state agreement is about `.79--.81`. | Predictive signal is real, but prediction does not imply intervention utility. See [`online proxy`](ERT_ONLINE_ROUTING_PROXY_RESULTS.md). |
| Does history-smoothed S3 routing improve robustness? | Majority-3 reduces switches by about `52.6%` and re-entry by about `66%`, but is below BASE at epoch 94 in both seeds; exit-2 adds no benefit. | Routing stabilization is supported; robust-accuracy promotion is not. See [`history production`](ERT_S3_HISTORY_PRODUCTION_RESULTS.md). |
| Do fixed Stage-A treatments transfer? | T1 weak AdvCE is positive on some selected training cohorts, while held-out effects are not durable; softening and T3 ablations are inconsistent or harmful. | No automatic route promotion. See [`Stage A`](ERT_STAGE_A_TREATMENT_RESULTS.md) and [`T123`](ERT_CONFIRMATORY_T123_RESULTS.md). |
| Can Clean-Wrong treatment solve robustness? | Extra CleanCE improves clean recovery on the direct cohort but leaves a clean/robust trade-off; reliability-gated variants fail to beat BASE consistently at epoch 94. | Teacher-margin gate is not confirmed as a practical intervention selector. See [`broad screen`](ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md) and [`gated CleanCE`](ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md). |
| Is action utility heterogeneous by Teacher margin? | Yes descriptively in the fixed train cohort; action rankings are more aligned in high-margin bins, but this is not held-out router validation. | Keep as a hypothesis map; do not choose Q5/thresholds post hoc. See [`margin action map`](ERT_CW_MARGIN_ACTION_MAP.md). |
| What causes run-to-run divergence? | Shuffle, augmentation, and their interaction can cause local 1--2 pp changes; source dominance differs by seed. | Preserve the frozen RNG contract; do not infer a universal dominant source. See [`RNG decomposition`](ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md) and [`shuffle/augmentation`](ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md). |

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
- All conclusions above are descriptive two-development-seed evidence. They
  are not population-level seed inference.

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

1. Keep `CROPSHIFT` as the incumbent until any single-switch timing or unseen
   seed confirmation satisfies its preregistered gates.
2. Do not promote a Student-history, Teacher-margin, or Clean-Wrong route from
   direct training effects alone.
3. For future long jobs, use the measured GPU-specific longest-processing-time-
   first policy; do not wait for another runtime audit unless the hardware or
   software stack changes.
4. Preserve metrics-only W&B tracking and local checkpoint/run-bundle storage;
   the runtime benchmark wrote only local JSON and uploaded no model artifacts.

## Provenance

The individual reports link their source Git SHAs, parent/checkpoint hashes,
attack identities, and machine-readable result artifacts. This summary was
written after the idle-Ferret follow-up benchmark and is intentionally not a
replacement for those records.
