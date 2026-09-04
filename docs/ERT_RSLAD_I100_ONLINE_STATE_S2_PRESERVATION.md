# I100 Online-State S2×T1 Preservation Screen

## Decision

The registered primary endpoint is e114 held-out CE-PGD20 robust accuracy.
PMP versus Control is **SUPPORTED**; D-BDP versus Control is
**SUPPORTED**; and the D-BDP-specific comparison is
**NOT_SUPPORTED**.  These are two development-seed directional
classifications, not population-level significance claims.

Machine-readable decision: `ONLINE_S2_SUPPORTED_FOR_NEXT_STAGE`;
`OS_PMP_SUPPORTED_FOR_NEXT_STAGE`;
`OS_DBDP_SUPPORTED_FOR_NEXT_STAGE`; and
`DBDP_SPECIFIC_SUPERIORITY_NOT_SUPPORTED`.

The threshold was frozen once from each seed's shared e100 no-action prefix.
It was not recomputed for any child or later epoch.  The online action was
strictly limited to current `S2_T1`; `CW`, `S3`, `S2_T2`, and `S2_T3` remained
baseline.  PMP/D-BDP use the registered pre-update pair margin only after
this detached state decision, whereas the router itself uses global logit
margin.

## Frozen lineage

| seed | e99 parent SHA-256 | e100 Student q10 | e100 Teacher q10 | threshold SHA-256 |
| --- | --- | ---: | ---: | --- |
| dev-1 | 360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835 | 0.140155 | 0.174019 | 0e45334c41d4fac34fc8df3925c9f0a65cfc6d20ab618b65e2251f27b8c4c40a |
| dev-2 | bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7 | 0.138205 | 0.177792 | ccc835415997d077ef763038bf52f6aecfa210b46e41b421cc1d78c00f7ba26f |

The common Teacher SHA-256 is `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.  The exact calibration
artifact SHA-256 is `37bf0a0e1aa6ff12951f1c05f59f6df55700be0e28291c6925670d7b6cb56840`. Training used sample-keyed
KL-PGD10 `97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d` and endpoint evaluation used CE-PGD20
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.

## Held-out endpoints

| seed | epoch | arm | held-out clean | held-out robust | clean Δ vs Control | robust Δ vs Control |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| dev-1 | 104 | CONTROL | 82.94% | 56.06% | +0.00 pp | +0.00 pp |
| dev-1 | 104 | PMP | 82.84% | 56.04% | -0.10 pp | -0.02 pp |
| dev-1 | 104 | DBDP | 82.92% | 56.02% | -0.02 pp | -0.04 pp |
| dev-1 | 109 | CONTROL | 83.48% | 56.94% | +0.00 pp | +0.00 pp |
| dev-1 | 109 | PMP | 83.46% | 56.98% | -0.02 pp | +0.04 pp |
| dev-1 | 109 | DBDP | 83.46% | 56.96% | -0.02 pp | +0.02 pp |
| dev-1 | 114 | CONTROL | 83.64% | 57.32% | +0.00 pp | +0.00 pp |
| dev-1 | 114 | PMP | 83.84% | 57.46% | +0.20 pp | +0.14 pp |
| dev-1 | 114 | DBDP | 83.78% | 57.46% | +0.14 pp | +0.14 pp |
| dev-2 | 104 | CONTROL | 82.76% | 55.80% | +0.00 pp | +0.00 pp |
| dev-2 | 104 | PMP | 82.78% | 55.80% | +0.02 pp | +0.00 pp |
| dev-2 | 104 | DBDP | 82.72% | 55.86% | -0.04 pp | +0.06 pp |
| dev-2 | 109 | CONTROL | 83.58% | 56.40% | +0.00 pp | +0.00 pp |
| dev-2 | 109 | PMP | 83.48% | 56.54% | -0.10 pp | +0.14 pp |
| dev-2 | 109 | DBDP | 83.48% | 56.56% | -0.10 pp | +0.16 pp |
| dev-2 | 114 | CONTROL | 83.72% | 56.96% | +0.00 pp | +0.00 pp |
| dev-2 | 114 | PMP | 83.70% | 57.16% | -0.02 pp | +0.20 pp |
| dev-2 | 114 | DBDP | 83.72% | 57.02% | +0.00 pp | +0.06 pp |

## Paired comparisons

| seed | epoch | comparison | clean Δ | robust Δ | robust rescue | robust harm |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| dev-1 | 104 | pmp_vs_control | -0.10 pp | -0.02 pp | 10 | 11 |
| dev-1 | 104 | dbdp_vs_control | -0.02 pp | -0.04 pp | 13 | 15 |
| dev-1 | 104 | dbdp_vs_pmp | +0.08 pp | -0.02 pp | 15 | 16 |
| dev-1 | 109 | pmp_vs_control | -0.02 pp | +0.04 pp | 17 | 15 |
| dev-1 | 109 | dbdp_vs_control | -0.02 pp | +0.02 pp | 20 | 19 |
| dev-1 | 109 | dbdp_vs_pmp | +0.00 pp | -0.02 pp | 18 | 19 |
| dev-1 | 114 | pmp_vs_control | +0.20 pp | +0.14 pp | 31 | 24 |
| dev-1 | 114 | dbdp_vs_control | +0.14 pp | +0.14 pp | 31 | 24 |
| dev-1 | 114 | dbdp_vs_pmp | -0.06 pp | +0.00 pp | 25 | 25 |
| dev-2 | 104 | pmp_vs_control | +0.02 pp | +0.00 pp | 10 | 10 |
| dev-2 | 104 | dbdp_vs_control | -0.04 pp | +0.06 pp | 14 | 11 |
| dev-2 | 104 | dbdp_vs_pmp | -0.06 pp | +0.06 pp | 19 | 16 |
| dev-2 | 109 | pmp_vs_control | -0.10 pp | +0.14 pp | 22 | 15 |
| dev-2 | 109 | dbdp_vs_control | -0.10 pp | +0.16 pp | 24 | 16 |
| dev-2 | 109 | dbdp_vs_pmp | +0.00 pp | +0.02 pp | 22 | 21 |
| dev-2 | 114 | pmp_vs_control | -0.02 pp | +0.20 pp | 33 | 23 |
| dev-2 | 114 | dbdp_vs_control | +0.00 pp | +0.06 pp | 35 | 32 |
| dev-2 | 114 | dbdp_vs_pmp | +0.02 pp | -0.14 pp | 25 | 32 |

For every paired result, `accuracy_delta = rescue_rate - harm_rate` was
asserted.  e114 raw-train CE-PGD20 state replays are recorded separately as a
canonical audit; they are not relabelled as a fixed causal cohort after the
online policy has changed the trajectory.

## Online-state mechanics and canonical diagnostic

| seed | arm | S2×T1 state epochs | state fraction | action exposure | pair-gated loss |
| --- | --- | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 21173 | 3.361% | 0 | 0 |
| dev-1 | PMP | 21174 | 3.361% | 21174 | 0 |
| dev-1 | DBDP | 20996 | 3.333% | 20996 | 0 |
| dev-2 | CONTROL | 20911 | 3.319% | 0 | 0 |
| dev-2 | PMP | 20699 | 3.286% | 20699 | 0 |
| dev-2 | DBDP | 20712 | 3.288% | 20712 | 0 |

| seed | arm | action fraction | action entries | action re-entry | action switches | precision | recall | Jaccard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 0.000% | 0 | 0 | 0 | 13.55% | 7.47% | 5.06% |
| dev-1 | PMP | 3.361% | 20063 | 4778 | 38675 | 12.96% | 7.18% | 4.84% |
| dev-1 | DBDP | 3.333% | 19891 | 4728 | 38352 | 12.24% | 6.68% | 4.52% |
| dev-2 | CONTROL | 0.000% | 0 | 0 | 0 | 11.79% | 6.77% | 4.49% |
| dev-2 | PMP | 3.286% | 19647 | 4649 | 37812 | 11.81% | 6.68% | 4.46% |
| dev-2 | DBDP | 3.288% | 19667 | 4684 | 37834 | 12.67% | 7.24% | 4.83% |

Precision/recall/Jaccard compare the e114 online augmented KL10 pre-update
state with the arm's raw CE20 canonical state.  This is a proxy-alignment
diagnostic, not a causal subgroup definition.  State persistence and action
exposure are separately preserved in the machine artifact for every e101–e114
epoch.  `router exposure` is the
global-margin Online-S2×T1 action decision.  `pair-gated loss` is the subset
whose reused Student-rival Teacher pair also had positive pair margin inside
the already-frozen PMP/DBDP formula; it is not a second router state.

| seed | arm | state entries | state exits | state re-entry | action exposure | action entries | action re-entry |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 19976 | 20137 | 5585 | 0 | 0 | 0 |
| dev-1 | PMP | 19958 | 20115 | 5512 | 21174 | 20063 | 4778 |
| dev-1 | DBDP | 19790 | 19968 | 5447 | 20996 | 19891 | 4728 |
| dev-2 | CONTROL | 19762 | 19805 | 5439 | 0 | 0 | 0 |
| dev-2 | PMP | 19567 | 19638 | 5355 | 20699 | 19647 | 4649 |
| dev-2 | DBDP | 19583 | 19636 | 5379 | 20712 | 19667 | 4684 |

The first three columns are state persistence, reported for Control, PMP, and
D-BDP alike. The final three columns are action events and are necessarily
zero for Control.

| seed | arm | S2×T1→S1 | →S3 | →CW | S1→S2×T1 | S3→S2×T1 | CW→S2×T1 | S2×T2/T3→S2×T1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 9293 | 4462 | 6172 | 8988 | 4467 | 6309 | 212 |
| dev-1 | PMP | 9398 | 4305 | 6208 | 9033 | 4469 | 6262 | 194 |
| dev-1 | DBDP | 9319 | 4299 | 6150 | 9000 | 4414 | 6174 | 202 |
| dev-2 | CONTROL | 9103 | 4448 | 6017 | 8726 | 4558 | 6257 | 221 |
| dev-2 | PMP | 9079 | 4442 | 5887 | 8706 | 4493 | 6157 | 211 |
| dev-2 | DBDP | 9073 | 4416 | 5923 | 8701 | 4494 | 6188 | 200 |

| seed | arm | e114 Student current q10 − frozen | e114 Teacher current q10 − frozen |
| --- | --- | ---: | ---: |
| dev-1 | CONTROL | +0.012839 | +0.002689 |
| dev-1 | PMP | +0.012714 | +0.003603 |
| dev-1 | DBDP | +0.012571 | +0.001650 |
| dev-2 | CONTROL | +0.010609 | -0.007812 |
| dev-2 | PMP | +0.012036 | -0.007951 |
| dev-2 | DBDP | +0.010660 | -0.006547 |

The q10 values in this table are diagnostics only.  They were not used to
update the frozen e100 thresholds.

| seed | arm | online S2×T1 | canonical S2×T1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 1447 | 2623 | 196 | 1251 | 2427 | 41126 |
| dev-1 | PMP | 1451 | 2620 | 188 | 1263 | 2432 | 41117 |
| dev-1 | DBDP | 1430 | 2620 | 175 | 1255 | 2445 | 41125 |
| dev-2 | CONTROL | 1510 | 2630 | 178 | 1332 | 2452 | 41038 |
| dev-2 | PMP | 1482 | 2620 | 175 | 1307 | 2445 | 41073 |
| dev-2 | DBDP | 1500 | 2624 | 190 | 1310 | 2434 | 41066 |

The e114 canonical audit measures current occupancy and online-proxy
agreement only.  Because this campaign does not replay an earlier canonical
state from the new trajectory, the following table instead uses the already
hash-bound historical e99 canonical S2×T1 IDs as a descriptive fixed reference.
It is not a post-treatment causal subgroup comparison.

| seed | arm | historical e99 S2×T1 | e114 S1 | e114 S3-non-CW | e114 Clean-Wrong | e114 S2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 2212 | 1555 | 297 | 3 | 357 |
| dev-1 | PMP | 2212 | 1560 | 298 | 2 | 352 |
| dev-1 | DBDP | 2212 | 1558 | 284 | 2 | 368 |
| dev-2 | CONTROL | 2141 | 1402 | 357 | 3 | 379 |
| dev-2 | PMP | 2141 | 1396 | 363 | 3 | 379 |
| dev-2 | DBDP | 2141 | 1398 | 349 | 3 | 391 |

## Runtime

| seed | arm | mean train seconds / epoch | mean images/s | mean epoch-time Δ vs Control |
| --- | --- | ---: | ---: | ---: |
| dev-1 | CONTROL | 92.2 | 488.0 | 0.00% |
| dev-1 | PMP | 93.3 | 482.3 | 1.19% |
| dev-1 | DBDP | 99.5 | 452.3 | 7.90% |
| dev-2 | CONTROL | 95.2 | 472.6 | 0.00% |
| dev-2 | PMP | 96.4 | 466.7 | 1.26% |
| dev-2 | DBDP | 102.7 | 438.2 | 7.86% |

Runtime is descriptive because host/GPU placement is operational rather than
scientific identity.  W&B remained metrics-only; checkpoints and 45k-ID state
tables stayed local and hash-bound.

## Interpretation and stop rule

- `D-BDP > PMP` is considered D-BDP-specific only if it is strictly positive
  in both development seeds; the recorded result is **NOT_SUPPORTED**.
- No coefficient, q10 threshold, arm, seed, e199 extension, official test, or
  AutoAttack was added from these results.
- A later Stable Indirect BDD design, if considered, requires a separate
  scientific contract and calibration.  This screen stops here.

## Provenance

Every collected artifact root is recorded twice in the machine artifact: the
declared path under the campaign collection root, and its realpath.  They
differ because a repair campaign collects earlier campaigns by symlink, so the
directory a path resolves into does not attribute the bytes to a producing
campaign.  Producing campaign IDs come from each job's `completion.json`; a
root without one is recorded as unknown rather than assumed.

| artifact class | producing campaign IDs | distinct realpath roots | roots without a completion record |
| --- | --- | ---: | ---: |
| training | ert-i100-online-state-s2-v1, ert-i100-online-state-s2-v1-recovery14 | 6 | 0 |
| endpoints | — | 6 | 6 |
| canonical | — | 6 | 6 |

Scientific source SHA bound to the artifacts: `bcb09a73814e7788026b287309d148b7791ccf75`.
Aggregation code SHA: `66a223daf73f65df4d7dae5dda39ac305f1ded82`.
Record: docs/experiments/ert_rslad_i100_online_state_s2_preservation_v1.json (sha256 in sidecar); campaign root /home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/ert-i100-online-state-s2-v1-recovery18; aggregation source 66a223d; artifact source bcb09a7.
