# I100 S2×T1 Longitudinal State Audit

## Conclusion

The fixed epoch-99 S2×T1 cohort is highly nonstationary at the four registered CE-PGD20 observations.  Therefore an e114 S2×T1 membership must not be described as continuous membership since e99.  These results are an offline, fixed-cohort description only; they do not validate or instantiate an online selector.

The primary trajectory uses raw, unaugmented train images and sample-keyed CE-PGD20 under the e99 key at e99/e104/e109/e114.  Historical augmented/batch-keyed replays are intentionally excluded from this join.  The KL-PGD10 values below are separate checkpoint no-update runtime proxies, not historical action logs.

## Contract and lineage

- Student branches are mutually exclusive: Clean-Wrong; S3-non-Clean-Wrong; S2; and S1. Teacher T1/T2/T3 is recorded independently.
- Primary fixed cohort is e99 S2×T1: dev-1 $n=2{,}212$, dev-2 $n=2{,}141$.
- Observation attack is the registered sample-keyed CE-PGD20 contract.  Endpoint continuity is observed-only, not continuous-time.
- The replay configs are host-path-rebased copies only: their sole semantic diff from the accepted parent configs is the absolute Teacher checkpoint path, while the frozen Teacher SHA-256 is unchanged.
- Machine artifact: `docs/experiments/ert_rslad_i100_s2_longitudinal_state_audit_v1.json` (SHA-256 `6fee43b4552d6880a10ea878d5673c8024d6837cea4038ff4ff9fb4abaa8fe25`).

## Fixed-cohort current state

Counts below retain the same e99 S2×T1 IDs; S2×T1 is the currently target-matching subset.

| seed | arm | e99 S2×T1 | e104 S2×T1 | e109 S2×T1 | e114 S2×T1 | e114 S1 | e114 S3-non-CW | e114 Clean-Wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 2212 | 379 | 353 | 335 | 1555 | 297 | 3 |
| dev-1 | DBDD | 2212 | 364 | 354 | 329 | 1585 | 275 | 1 |
| dev-1 | DPM | 2212 | 379 | 351 | 344 | 1575 | 272 | 2 |
| dev-2 | CONTROL | 2141 | 401 | 370 | 363 | 1402 | 357 | 3 |
| dev-2 | DBDD | 2141 | 389 | 362 | 362 | 1425 | 335 | 2 |
| dev-2 | DPM | 2141 | 385 | 361 | 353 | 1433 | 336 | 2 |

## Observed membership and re-entry

P1–P5 are intentionally overlapping observed-endpoint indicators; `membership patterns` are the mutually exclusive partition.  P6 (multiple exit/re-entry) is not observable under four endpoints that begin target-active.  An explicit route only means that route at the registered endpoints.

| seed | arm | P1 all observed S2×T1 | P2 terminal S1/outside S2 | P3 observed S3-non-CW | P4 observed Clean-Wrong | P5 observed leave→re-enter | P6 multiple exit/re-entry |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev-1 | CONTROL | 99 | 1532 | 523 | 4 | 343 | not observable |
| dev-1 | DBDD | 86 | 1559 | 517 | 2 | 359 | not observable |
| dev-1 | DPM | 98 | 1549 | 511 | 3 | 357 | not observable |
| dev-2 | CONTROL | 111 | 1365 | 548 | 7 | 365 | not observable |
| dev-2 | DBDD | 101 | 1377 | 532 | 5 | 380 | not observable |
| dev-2 | DPM | 99 | 1388 | 542 | 7 | 367 | not observable |

| seed | arm | P1 / fixed e99 mask | P1 / current e114 S2×T1 target | P5 / fixed e99 mask | S2→S1→S2 | S2→S3-non-CW→S2 | S2→Clean-Wrong→S2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 99/2212 (4.5%) | 99/2623 (3.8%) | 343/2212 (15.5%) | 119 | 118 | 0 |
| dev-1 | DBDD | 86/2212 (3.9%) | 86/2606 (3.3%) | 359/2212 (16.2%) | 119 | 127 | 0 |
| dev-1 | DPM | 98/2212 (4.4%) | 98/2610 (3.8%) | 357/2212 (16.1%) | 118 | 123 | 0 |
| dev-2 | CONTROL | 111/2141 (5.2%) | 111/2630 (4.2%) | 365/2141 (17.0%) | 153 | 116 | 0 |
| dev-2 | DBDD | 101/2141 (4.7%) | 101/2628 (3.8%) | 380/2141 (17.7%) | 150 | 127 | 0 |
| dev-2 | DPM | 99/2141 (4.6%) | 99/2633 (3.8%) | 367/2141 (17.1%) | 143 | 122 | 0 |

## Fixed-mask divergence from current S2×T1

| seed | arm | epoch | current S2×T1 | fraction of fixed mask | current S2×T2/T3 | current S1 | current S3-non-CW | current Clean-Wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 104 | 379 | 17.1% | 9 | 1426 | 397 | 1 |
| dev-1 | CONTROL | 109 | 353 | 16.0% | 12 | 1492 | 354 | 1 |
| dev-1 | CONTROL | 114 | 335 | 15.1% | 22 | 1555 | 297 | 3 |
| dev-1 | DBDD | 104 | 364 | 16.5% | 9 | 1450 | 389 | 0 |
| dev-1 | DBDD | 109 | 354 | 16.0% | 13 | 1512 | 332 | 1 |
| dev-1 | DBDD | 114 | 329 | 14.9% | 22 | 1585 | 275 | 1 |
| dev-1 | DPM | 104 | 379 | 17.1% | 9 | 1434 | 389 | 1 |
| dev-1 | DPM | 109 | 351 | 15.9% | 14 | 1515 | 331 | 1 |
| dev-1 | DPM | 114 | 344 | 15.6% | 19 | 1575 | 272 | 2 |
| dev-2 | CONTROL | 104 | 401 | 18.7% | 16 | 1305 | 415 | 4 |
| dev-2 | CONTROL | 109 | 370 | 17.3% | 24 | 1386 | 357 | 4 |
| dev-2 | CONTROL | 114 | 363 | 17.0% | 16 | 1402 | 357 | 3 |
| dev-2 | DBDD | 104 | 389 | 18.2% | 16 | 1325 | 408 | 3 |
| dev-2 | DBDD | 109 | 362 | 16.9% | 32 | 1406 | 338 | 3 |
| dev-2 | DBDD | 114 | 362 | 16.9% | 17 | 1425 | 335 | 2 |
| dev-2 | DPM | 104 | 385 | 18.0% | 16 | 1324 | 412 | 4 |
| dev-2 | DPM | 109 | 361 | 16.9% | 27 | 1405 | 344 | 4 |
| dev-2 | DPM | 114 | 353 | 16.5% | 17 | 1433 | 336 | 2 |

### Teacher transitions whenever Student is S2 at either adjacent endpoint

| seed | arm | T1→T1 | T1→T2 | T1→T3 | T2→T1 | T2→T2 | T2→T3 | T3→T1 | T3→T2 | T3→T3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 3195 | 82 | 2 | 11 | 31 | 0 | 0 | 0 | 0 |
| dev-1 | DBDD | 3194 | 84 | 2 | 12 | 30 | 1 | 0 | 0 | 0 |
| dev-1 | DPM | 3206 | 81 | 2 | 12 | 32 | 0 | 0 | 0 | 0 |
| dev-2 | CONTROL | 3157 | 114 | 2 | 9 | 48 | 0 | 0 | 0 | 0 |
| dev-2 | DBDD | 3153 | 115 | 2 | 7 | 60 | 0 | 0 | 0 | 1 |
| dev-2 | DPM | 3136 | 113 | 2 | 9 | 54 | 0 | 0 | 0 | 1 |

### New observed S2×T1 entrants

| seed | arm | entrants | e99 S2×T2/T3 | e99 S1 | e99 S3-non-CW | e99 Clean-Wrong | one endpoint | repeated one run | re-entry (≥2 runs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | CONTROL | 4398 | 8 | 1171 | 2901 | 318 | 2510 | 1500 | 388 |
| dev-1 | DBDD | 4416 | 5 | 1182 | 2923 | 306 | 2533 | 1499 | 384 |
| dev-1 | DPM | 4389 | 6 | 1173 | 2900 | 310 | 2504 | 1487 | 398 |
| dev-2 | CONTROL | 4382 | 3 | 1084 | 2911 | 384 | 2573 | 1402 | 407 |
| dev-2 | DBDD | 4365 | 3 | 1077 | 2908 | 377 | 2551 | 1423 | 391 |
| dev-2 | DPM | 4371 | 3 | 1073 | 2923 | 372 | 2529 | 1426 | 416 |

## Checkpoint no-update runtime proxy

The following table evaluates the exact fixed e99 mask at each saved checkpoint under a fresh KL-PGD10 training-view proxy.  It is not a reconstruction of historical minibatch activity.  `extra-loss positive` uses the KL10 Student-selected-rival Teacher-pair gate; it is not CE20 Teacher T1 and may be positive while the Teacher is globally CE20-wrong.  Control has no extra loss.

| seed | arm | epoch | current branch | fixed IDs | proxy extra-loss positive | fraction |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| dev-1 | CONTROL | 104 | Clean-Wrong | 1 | 0 | 0.000 |
| dev-1 | CONTROL | 104 | S1 | 1426 | 0 | 0.000 |
| dev-1 | CONTROL | 104 | S2xT1 | 379 | 0 | 0.000 |
| dev-1 | CONTROL | 104 | S2xT2T3 | 9 | 0 | 0.000 |
| dev-1 | CONTROL | 104 | S3-non-CW | 397 | 0 | 0.000 |
| dev-1 | CONTROL | 109 | Clean-Wrong | 1 | 0 | 0.000 |
| dev-1 | CONTROL | 109 | S1 | 1492 | 0 | 0.000 |
| dev-1 | CONTROL | 109 | S2xT1 | 353 | 0 | 0.000 |
| dev-1 | CONTROL | 109 | S2xT2T3 | 12 | 0 | 0.000 |
| dev-1 | CONTROL | 109 | S3-non-CW | 354 | 0 | 0.000 |
| dev-1 | CONTROL | 114 | Clean-Wrong | 3 | 0 | 0.000 |
| dev-1 | CONTROL | 114 | S1 | 1555 | 0 | 0.000 |
| dev-1 | CONTROL | 114 | S2xT1 | 335 | 0 | 0.000 |
| dev-1 | CONTROL | 114 | S2xT2T3 | 22 | 0 | 0.000 |
| dev-1 | CONTROL | 114 | S3-non-CW | 297 | 0 | 0.000 |
| dev-1 | DBDD | 104 | S1 | 1450 | 766 | 0.528 |
| dev-1 | DBDD | 104 | S2xT1 | 364 | 192 | 0.527 |
| dev-1 | DBDD | 104 | S2xT2T3 | 9 | 5 | 0.556 |
| dev-1 | DBDD | 104 | S3-non-CW | 389 | 192 | 0.494 |
| dev-1 | DBDD | 109 | Clean-Wrong | 1 | 1 | 1.000 |
| dev-1 | DBDD | 109 | S1 | 1512 | 801 | 0.530 |
| dev-1 | DBDD | 109 | S2xT1 | 354 | 177 | 0.500 |
| dev-1 | DBDD | 109 | S2xT2T3 | 13 | 7 | 0.538 |
| dev-1 | DBDD | 109 | S3-non-CW | 332 | 162 | 0.488 |
| dev-1 | DBDD | 114 | Clean-Wrong | 1 | 0 | 0.000 |
| dev-1 | DBDD | 114 | S1 | 1585 | 838 | 0.529 |
| dev-1 | DBDD | 114 | S2xT1 | 329 | 168 | 0.511 |
| dev-1 | DBDD | 114 | S2xT2T3 | 22 | 9 | 0.409 |
| dev-1 | DBDD | 114 | S3-non-CW | 275 | 115 | 0.418 |
| dev-1 | DPM | 104 | Clean-Wrong | 1 | 0 | 0.000 |
| dev-1 | DPM | 104 | S1 | 1434 | 807 | 0.563 |
| dev-1 | DPM | 104 | S2xT1 | 379 | 215 | 0.567 |
| dev-1 | DPM | 104 | S2xT2T3 | 9 | 6 | 0.667 |
| dev-1 | DPM | 104 | S3-non-CW | 389 | 206 | 0.530 |
| dev-1 | DPM | 109 | Clean-Wrong | 1 | 1 | 1.000 |
| dev-1 | DPM | 109 | S1 | 1515 | 853 | 0.563 |
| dev-1 | DPM | 109 | S2xT1 | 351 | 188 | 0.536 |
| dev-1 | DPM | 109 | S2xT2T3 | 14 | 9 | 0.643 |
| dev-1 | DPM | 109 | S3-non-CW | 331 | 175 | 0.529 |
| dev-1 | DPM | 114 | Clean-Wrong | 2 | 1 | 0.500 |
| dev-1 | DPM | 114 | S1 | 1575 | 910 | 0.578 |
| dev-1 | DPM | 114 | S2xT1 | 344 | 182 | 0.529 |
| dev-1 | DPM | 114 | S2xT2T3 | 19 | 7 | 0.368 |
| dev-1 | DPM | 114 | S3-non-CW | 272 | 136 | 0.500 |
| dev-2 | CONTROL | 104 | Clean-Wrong | 4 | 0 | 0.000 |
| dev-2 | CONTROL | 104 | S1 | 1305 | 0 | 0.000 |
| dev-2 | CONTROL | 104 | S2xT1 | 401 | 0 | 0.000 |
| dev-2 | CONTROL | 104 | S2xT2T3 | 16 | 0 | 0.000 |
| dev-2 | CONTROL | 104 | S3-non-CW | 415 | 0 | 0.000 |
| dev-2 | CONTROL | 109 | Clean-Wrong | 4 | 0 | 0.000 |
| dev-2 | CONTROL | 109 | S1 | 1386 | 0 | 0.000 |
| dev-2 | CONTROL | 109 | S2xT1 | 370 | 0 | 0.000 |
| dev-2 | CONTROL | 109 | S2xT2T3 | 24 | 0 | 0.000 |
| dev-2 | CONTROL | 109 | S3-non-CW | 357 | 0 | 0.000 |
| dev-2 | CONTROL | 114 | Clean-Wrong | 3 | 0 | 0.000 |
| dev-2 | CONTROL | 114 | S1 | 1402 | 0 | 0.000 |
| dev-2 | CONTROL | 114 | S2xT1 | 363 | 0 | 0.000 |
| dev-2 | CONTROL | 114 | S2xT2T3 | 16 | 0 | 0.000 |
| dev-2 | CONTROL | 114 | S3-non-CW | 357 | 0 | 0.000 |
| dev-2 | DBDD | 104 | Clean-Wrong | 3 | 1 | 0.333 |
| dev-2 | DBDD | 104 | S1 | 1325 | 720 | 0.543 |
| dev-2 | DBDD | 104 | S2xT1 | 389 | 222 | 0.571 |
| dev-2 | DBDD | 104 | S2xT2T3 | 16 | 8 | 0.500 |
| dev-2 | DBDD | 104 | S3-non-CW | 408 | 206 | 0.505 |
| dev-2 | DBDD | 109 | Clean-Wrong | 3 | 3 | 1.000 |
| dev-2 | DBDD | 109 | S1 | 1406 | 762 | 0.542 |
| dev-2 | DBDD | 109 | S2xT1 | 362 | 183 | 0.506 |
| dev-2 | DBDD | 109 | S2xT2T3 | 32 | 17 | 0.531 |
| dev-2 | DBDD | 109 | S3-non-CW | 338 | 176 | 0.521 |
| dev-2 | DBDD | 114 | Clean-Wrong | 2 | 2 | 1.000 |
| dev-2 | DBDD | 114 | S1 | 1425 | 762 | 0.535 |
| dev-2 | DBDD | 114 | S2xT1 | 362 | 198 | 0.547 |
| dev-2 | DBDD | 114 | S2xT2T3 | 17 | 10 | 0.588 |
| dev-2 | DBDD | 114 | S3-non-CW | 335 | 145 | 0.433 |
| dev-2 | DPM | 104 | Clean-Wrong | 4 | 2 | 0.500 |
| dev-2 | DPM | 104 | S1 | 1324 | 772 | 0.583 |
| dev-2 | DPM | 104 | S2xT1 | 385 | 234 | 0.608 |
| dev-2 | DPM | 104 | S2xT2T3 | 16 | 9 | 0.562 |
| dev-2 | DPM | 104 | S3-non-CW | 412 | 214 | 0.519 |
| dev-2 | DPM | 109 | Clean-Wrong | 4 | 4 | 1.000 |
| dev-2 | DPM | 109 | S1 | 1405 | 835 | 0.594 |
| dev-2 | DPM | 109 | S2xT1 | 361 | 199 | 0.551 |
| dev-2 | DPM | 109 | S2xT2T3 | 27 | 18 | 0.667 |
| dev-2 | DPM | 109 | S3-non-CW | 344 | 194 | 0.564 |
| dev-2 | DPM | 114 | Clean-Wrong | 2 | 1 | 0.500 |
| dev-2 | DPM | 114 | S1 | 1433 | 829 | 0.579 |
| dev-2 | DPM | 114 | S2xT1 | 353 | 215 | 0.609 |
| dev-2 | DPM | 114 | S2xT2T3 | 17 | 9 | 0.529 |
| dev-2 | DPM | 114 | S3-non-CW | 336 | 152 | 0.452 |

## Interpretation boundary

- The fixed-mask causal screen remains evaluable, but its treatment effect should not be narrated as an effect on a persistent S2×T1 population.
- Current S2×T1 entrants are a distinct, untreated-by-fixed-mask descriptive population; their origin mix is recorded above rather than folded into the fixed cohort.
- These data motivate a separate, preregistered online-state contract if an intervention is proposed. This audit does not launch or endorse one.

## Discussion-only next-experiment frame

If human review authorizes a successor, the minimal candidate comparison is I100 Control versus Online-State Pair-Margin Preservation versus Online-State Detached Boundary-Distance Preservation. Its Phase-1 action contract should keep current Clean-Wrong, S3-non-CW, S2×T2/T3, and S1 on baseline RSLAD, and act only on current S2×T1.  This is a proposal boundary, not a launch authorization.
