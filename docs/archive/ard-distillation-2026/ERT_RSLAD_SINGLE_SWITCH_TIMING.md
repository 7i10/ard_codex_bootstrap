# ERT / RSLAD Single-Switch Augmentation Timing

Status: complete. This final finite timing screen uses fresh I50/I75/I125 suffixes, reuses hash-bound I100/I150 references, and does not use official test or AutoAttack.

## Contract and lineage

- Production source SHA (from every child run manifest): `8083f9c5df9b46a3a02399fbf293ceee6db85083`.
- Teacher SHA-256: `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.
- Prefix: accepted CROPSHIFT control; late policy: frozen IDBH_WEAK.
- Training attack: KL-PGD10, epsilon 8/255, step 2/255, random start, teacher-clean target.
- Endpoint: independent CE-PGD20, epsilon 8/255, step 2/255, 20 steps, random start, eval mode.
- Endpoint table below is fixed internal validation (5,000 samples); train endpoints are retained in the machine artifact.

## Fresh endpoint results

| seed | arm | switch | clean | robust | Δ robust vs CROPSHIFT | Δ clean | full AUC Δ (pp) | post-switch AUC Δ (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | I50 | 50 | 85.660% | 60.080% | +0.68 pp | -0.46 pp | -0.038 | -0.049 |
| 2 | I50 | 50 | 86.020% | 60.180% | +0.92 pp | -0.20 pp | -0.070 | -0.096 |
| 1 | I75 | 75 | 85.880% | 60.360% | +0.96 pp | -0.24 pp | +0.242 | +0.386 |
| 2 | I75 | 75 | 86.180% | 60.280% | +1.02 pp | -0.04 pp | +0.002 | +0.007 |
| 1 | I125 | 125 | 85.860% | 60.600% | +1.20 pp | -0.26 pp | +0.358 | +0.960 |
| 2 | I125 | 125 | 86.340% | 60.300% | +1.04 pp | +0.12 pp | +0.259 | +0.694 |

## Complete timing profile

This descriptive profile combines the fresh I50/I75/I125 suffixes with the hash-bound I0/I100/I150 references. I0 is IDBH_WEAK from scratch; I100 and I150 are prior stage-wise continuations. Values are fixed internal-validation CE-PGD20 endpoint robust accuracy and trajectory AUC.

| seed | arm | switch | source | final clean | final robust | full AUC | post-switch AUC |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | I0 | 0 | reference | 86.160% | 60.520% | 0.509032 | — |
| 2 | I0 | 0 | reference | 86.160% | 59.720% | 0.508120 | — |
| 1 | I50 | 50 | fresh suffix | 85.660% | 60.080% | 0.514227 | 0.541977 |
| 2 | I50 | 50 | fresh suffix | 86.020% | 60.180% | 0.513768 | 0.541885 |
| 1 | I75 | 75 | fresh suffix | 85.880% | 60.360% | 0.517025 | 0.561935 |
| 2 | I75 | 75 | fresh suffix | 86.180% | 60.280% | 0.514488 | 0.558248 |
| 1 | I100 | 100 | reference | 85.820% | 60.600% | 0.519623 | 0.589668 |
| 2 | I100 | 100 | reference | 86.040% | 60.300% | 0.517595 | 0.586220 |
| 1 | I125 | 125 | fresh suffix | 85.860% | 60.600% | 0.518182 | 0.593385 |
| 2 | I125 | 125 | fresh suffix | 86.340% | 60.300% | 0.517058 | 0.591292 |
| 1 | I150 | 150 | reference | 85.780% | 60.160% | 0.516374 | 0.596798 |
| 2 | I150 | 150 | reference | 85.920% | 60.160% | 0.515947 | 0.596347 |

## Shock and throughput

| seed | arm | +1 epoch Δ | +5 epoch Δ | +10 epoch Δ | max negative dip | recovery epoch | throughput Δ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | I50 | -1.680 pp | -5.340 pp | -0.260 pp | -5.340 pp | 52 | +1.5% |
| 2 | I50 | +0.760 pp | -2.320 pp | -1.320 pp | -3.580 pp | 62 | -0.7% |
| 1 | I75 | -0.060 pp | +0.660 pp | -1.080 pp | -3.980 pp | 78 | -40.8% |
| 2 | I75 | -1.540 pp | +0.100 pp | +1.480 pp | -2.840 pp | 78 | -41.4% |
| 1 | I125 | -0.120 pp | -0.160 pp | +0.580 pp | -0.360 pp | 127 | -41.6% |
| 2 | I125 | +0.740 pp | -0.120 pp | +0.440 pp | -0.280 pp | 128 | -40.0% |

## Preregistered gates and I100 replacement

- `I50`: qualifies=False; final=True, full-AUC=False, post-AUC=False, clean=True, throughput=True; replaces I100=False.
- `I75`: qualifies=False; final=True, full-AUC=True, post-AUC=True, clean=True, throughput=False; replaces I100=False.
- `I125`: qualifies=False; final=True, full-AUC=True, post-AUC=True, clean=True, throughput=False; replaces I100=False.

**Freeze decision: `I100`.** The finite search is closed; no additional switch timing or multi-stage schedule was run.

## Interpretation

The timing profile is descriptive over I0/I50/I75/I100/I125/I150. A candidate is not called globally optimal from two development seeds. I100 remains the incumbent because no fresh candidate satisfies the preregistered two-seed replacement rule (final robust dominance plus non-lower full AUC).

## Next stage (not started)

If human review accepts the freeze, the next experiment is three unseen paired seeds for confirmation and full-training stochasticity characterization. Student-History/Ordering work remains separate. No such run was started here.

Machine artifact: `docs/experiments/ert_rslad_single_switch_timing_results_v1.json`.
