# ERT I100 canonical S2 robust-boundary preservation screen

Status: complete. The preregistered fixed e100–114 continuation is closed. No e199 extension, History routing, TPFM-as-baseline, new seed, official test, or AutoAttack was run.

## Executive summary

- Recovery6 completed 13/13 jobs: five training continuations, six CE-PGD20 endpoint jobs, and runtime aggregation. No recovery6 job needed a retry.
- At the primary full-validation e114 endpoint, robust deltas versus same-seed control were SBF `+0.160/+0.040 pp` and TPFM `+0.220/+0.040 pp` for dev-1/dev-2. These small effects are not a promotion signal.
- On the fixed validation S2×T1 cohort, e114 robust net rescue was SBF `+0.442/+0.478 pp` and TPFM `-0.442/+0.478 pp`. Retention was SBF `97.62%/98.82%` and TPFM `97.02%/99.41%` (dev-1/dev-2).
- Direct fixed-train S2×T1 robust net rescue was positive for both mechanisms and seeds (SBF `+1.175/+1.261 pp`; TPFM `+1.266/+1.074 pp`), but this direct signal is substantially larger than held-out transfer.

## Lineage and frozen contract

- Production source: `d880bbf18b80912adb8816c2a90da77a3146b4f1`; recovery6 manifest SHA-256 `e2db843a8bfc95bd7328421d54f7789075d35f196d4c4cf835d3749692deb8da`; state SHA-256 `67ef9d69e06d93e1e1adb5d2493eee4b109bb36f42338b250b4e3567865265a7`.
- Epoch-99 parents: dev-1 `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835`; dev-2 `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`.
- Teacher SHA-256: `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.
- Fixed masks: train n=2,212/2,141 and validation n=226/209 for dev-1/dev-2. Mask file hashes are recorded in the machine artifact.
- Training attack: sample-keyed KL-PGD10, `epsilon=8/255`, `step=2/255`, 10 steps, random start, Teacher-clean target. Endpoint: CE-PGD20 with the common `708110...` identity.
- Calibration remained frozen: SBF coefficient `0.23594490117507805`, floors `0.04177670180797577`/`0.03347739577293396`; TPFM coefficient `0.16676844691071563`, floor `0.16590790450572968`, cap `0.32364362478256226`. Achieved median gradient ratios were 0.249 (SBF) and 0.247 (TPFM), target 0.25.

## Full held-out validation endpoints

| seed | epoch | control clean | control robust | SBF clean | SBF robust | TPFM clean | TPFM robust |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev-1 | 104 | 82.940% | 56.060% | 82.980% | 55.980% | 82.920% | 56.040% |
| dev-1 | 109 | 83.480% | 56.940% | 83.480% | 56.880% | 83.340% | 56.680% |
| dev-1 | 114 | 83.640% | 57.320% | 83.760% | 57.480% | 83.860% | 57.540% |
| dev-2 | 104 | 82.760% | 55.800% | 82.800% | 55.880% | 82.780% | 55.800% |
| dev-2 | 109 | 83.580% | 56.400% | 83.520% | 56.440% | 83.500% | 56.360% |
| dev-2 | 114 | 83.720% | 56.960% | 83.700% | 57.000% | 83.640% | 57.000% |

Full-validation e114 clean deltas were SBF `+0.120/-0.020 pp` and TPFM `+0.220/-0.080 pp` for dev-1/dev-2.

## Fixed validation S2×T1 boundary preservation

The cohort is fixed at epoch 99 under the canonical positive-margin q10 state contract. Retention/failure are denominated by rows that are robust-correct in the same-horizon control endpoint. Rescue/harm/net are paired treatment-minus-control endpoint transitions.

| seed | epoch | arm | n | robust net Δ (pp) | rescue/harm | retention | failure | robust margin Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dev-1 | 104 | SBF | 226 | -1.327 | 1/4 | 97.40% | 2.60% | 0.000190 |
| dev-1 | 104 | TPFM | 226 | -0.442 | 2/3 | 98.05% | 1.95% | 0.000620 |
| dev-1 | 109 | SBF | 226 | -0.442 | 2/3 | 98.12% | 1.88% | 0.000496 |
| dev-1 | 109 | TPFM | 226 | -2.212 | 0/5 | 96.88% | 3.12% | 0.000008 |
| dev-1 | 114 | SBF | 226 | 0.442 | 5/4 | 97.62% | 2.38% | 0.001613 |
| dev-1 | 114 | TPFM | 226 | -0.442 | 4/5 | 97.02% | 2.98% | 0.000517 |
| dev-2 | 104 | SBF | 209 | 0.478 | 3/2 | 98.80% | 1.20% | 0.000623 |
| dev-2 | 104 | TPFM | 209 | 1.435 | 3/0 | 100.00% | 0.00% | -0.000003 |
| dev-2 | 109 | SBF | 209 | -0.478 | 0/1 | 99.41% | 0.59% | 0.000417 |
| dev-2 | 109 | TPFM | 209 | -1.435 | 0/3 | 98.22% | 1.78% | 0.000243 |
| dev-2 | 114 | SBF | 209 | 0.478 | 3/2 | 98.82% | 1.18% | 0.000418 |
| dev-2 | 114 | TPFM | 209 | 0.478 | 2/1 | 99.41% | 0.59% | 0.000166 |

The selected cohort is already clean-correct at these endpoint comparisons; e114 selected-cohort clean accuracy is 100% for control and both treatments in both seeds. Thus the relevant question is preservation of the robust boundary, not clean recovery.

## Fixed train S2×T1 direct e114 endpoint

| seed | arm | n | clean net Δ (pp) | robust net Δ (pp) | robust rescue/harm | robust margin Δ |
|---|---|---:|---:|---:|---:|---:|
| dev-1 | SBF | 2212 | 0.271 | 1.175 | 34/8 | 0.002797 |
| dev-1 | TPFM | 2212 | 0.271 | 1.266 | 39/11 | 0.004671 |
| dev-2 | SBF | 2141 | 0.047 | 1.261 | 36/9 | 0.002345 |
| dev-2 | TPFM | 2141 | 0.093 | 1.074 | 33/10 | 0.003798 |

These are direct effects on the fixed training IDs and are not a held-out generalization estimate.

## Decision and engineering notes

- **Scientific decision: MIXED / descriptive.** SBF and TPFM both show positive direct-train robust net rescue, but held-out full-validation and fixed-cohort effects are small, time-varying, and not uniformly replicated by mechanism.
- Neither SBF nor TPFM is promoted. I100 control remains the reference for this screen.
- A prior recovery attempt failed before scientific compute because deterministic W&B run IDs collided. The tracking-only fix in the production source adds campaign/attempt suffixes for orchestrated launches while preserving source, config, parent, seed, attack, and treatment identity. Recovery6 then completed without retry.
- The runtime aggregate and this report do not upload model/run-bundle artifacts; endpoint row paths and SHA-256 values are retained in the machine artifact and runtime aggregate.

## Stop boundary

No coefficient or threshold change, TPFM-as-baseline, History routing, S2 extension, e199 continuation, additional seed, official test, or AutoAttack was started.

Machine artifact: `docs/experiments/ert_rslad_i100_s2_rbp_results_v1.json`.
Runtime aggregate: `docs/experiments/ert_rslad_i100_s2_rbp_runtime_v1.json`.
