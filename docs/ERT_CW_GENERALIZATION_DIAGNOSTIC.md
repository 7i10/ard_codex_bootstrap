# ERT Clean-Wrong Generalization Diagnostic

## Conclusion

This is a read-only Direct → train Spillover → Held-out diagnostic. The validation CE-PGD20/KL-PGD10 features were replayed at epoch 79 using the exact L2/L4 parents; no training, threshold tuning, official test, or AutoAttack was run.

The report distinguishes direct train-cohort correction from non-selected train effects and held-out Clean-Wrong effects. Held-out Q1–Q5 use train-derived upper boundaries; validation outcomes never define the bins.

## Frozen lineage

- Analysis source SHA: `7eed6810140bf6e5b6e91d463023703e61e32fea`
- Endpoint attack identity (CE-PGD20): `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`
- Epoch-79 parents: L2 `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`; L4 `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`
- Fixed Clean-Wrong mask hashes: L2 `0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b`; L4 `fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6`
- Validation feature attacks: CE-PGD20 and KL-PGD10; all four replay outputs contain 5,000 stable IDs and are hash-bound in the machine JSON.

## Held-out cohort and boundary transfer

| seed | train CW | held-out CW | held-out CE20 Q1–Q5 counts | held-out KL10 Q1–Q5 counts |
|---|---:|---:|---|---|
| L2 | 8623 | 1090 | {'Q1': 201, 'Q2': 179, 'Q3': 176, 'Q4': 196, 'Q5': 338} | {'Q1': 199, 'Q2': 185, 'Q3': 188, 'Q4': 187, 'Q5': 331} |
| L4 | 8925 | 1105 | {'Q1': 213, 'Q2': 179, 'Q3': 176, 'Q4': 197, 'Q5': 340} | {'Q1': 204, 'Q2': 192, 'Q3': 175, 'Q4': 205, 'Q5': 329} |

## Broad Screen: Direct → non-CW train Spillover → Held-out Clean Wrong

| seed | arm | Direct robust Δ | Spillover robust Δ | Held-out robust Δ | Direct clean Δ | Spillover clean Δ | Held-out clean Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| L2 | C0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| L2 | C4 | +0.406 | -0.223 | -0.642 | +2.598 | +0.181 | +1.468 |
| L2 | C5 | +0.545 | -1.061 | -0.183 | +3.676 | +0.085 | +4.495 |
| L2 | C10 | +1.310 | -1.218 | +0.092 | +5.451 | -0.374 | +4.404 |
| L2 | C11 | +0.696 | -1.204 | +0.092 | +1.508 | -0.613 | +2.752 |
| L2 | C12 | +1.032 | -1.432 | +0.275 | +2.876 | -0.849 | +3.853 |
| L2 | C13 | +0.580 | -0.060 | -0.459 | +0.058 | -0.525 | -1.009 |
| L4 | C0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| L4 | C4 | -0.011 | -0.565 | -0.543 | +2.790 | +0.338 | +0.271 |
| L4 | C5 | -0.381 | -3.005 | -0.633 | +5.490 | +0.804 | +5.520 |
| L4 | C10 | +0.594 | -0.421 | +0.000 | +7.709 | +0.962 | +6.154 |
| L4 | C11 | +0.627 | -0.274 | -0.362 | +1.244 | -0.990 | +1.719 |
| L4 | C12 | +0.975 | -1.258 | +0.543 | +3.765 | -0.690 | +3.258 |
| L4 | C13 | -0.157 | -0.676 | -0.905 | +1.513 | -0.086 | +3.077 |

## Gated experiment: selected / within-CW / non-CW / held-out

| seed | arm | train direct robust Δ | within-CW robust Δ | non-CW robust Δ | held-out selected robust Δ | held-out CW robust Δ |
|---|---|---:|---:|---:|---:|---:|
| L2 | G0_BASE | n/a | +0.000 | +0.000 | n/a | +0.000 |
| L2 | G1_CW_ALL_CE015 | +0.371 | n/a | -3.101 | -0.092 | -0.092 |
| L2 | G2_CW_R_CE20_CE015 | +2.132 | +0.105 | -1.039 | -0.634 | -0.183 |
| L2 | G3_CW_R_KL10_CE015 | +0.695 | +0.077 | -2.592 | -1.544 | -0.826 |
| L4 | G0_BASE | n/a | +0.000 | +0.000 | n/a | +0.000 |
| L4 | G1_CW_ALL_CE015 | +1.602 | n/a | -1.541 | +1.538 | +1.538 |
| L4 | G2_CW_R_CE20_CE015 | +4.168 | +0.258 | -1.256 | +2.236 | +1.176 |
| L4 | G3_CW_R_KL10_CE015 | +2.709 | +0.250 | -1.450 | +1.812 | +0.905 |

All accuracy effects below are percentage points (pp); the machine JSON stores proportions. The JSON also contains paired rescue/harm/net-rescue rates and clean/robust probability-margin deltas.

## Overall held-out validation (dilution check)

This table uses all 5,000 fixed validation IDs, not only the validation Clean-Wrong subset. A positive selected-CW effect can be diluted or reversed by non-CW spillover; that is distinct from failure to transfer within the held-out Clean-Wrong subtype.

| seed | arm | held-out overall robust Δ | held-out overall clean Δ | held-out CW robust Δ | held-out CW clean Δ |
|---|---|---:|---:|---:|---:|
| L2 | G0_BASE | +0.000 | +0.000 | +0.000 | +0.000 |
| L2 | G1_CW_ALL_CE015 | -1.640 | +0.140 | -0.092 | +3.486 |
| L2 | G2_CW_R_CE20_CE015 | +0.440 | -0.400 | -0.183 | +1.835 |
| L2 | G3_CW_R_KL10_CE015 | -1.060 | -0.240 | -0.826 | +1.376 |
| L4 | G0_BASE | +0.000 | +0.000 | +0.000 | +0.000 |
| L4 | G1_CW_ALL_CE015 | -0.640 | +1.220 | +1.538 | +6.697 |
| L4 | G2_CW_R_CE20_CE015 | -0.560 | +0.780 | +1.176 | +4.344 |
| L4 | G3_CW_R_KL10_CE015 | -1.140 | +0.440 | +0.905 | +3.710 |

For the broad-screen arms, the held-out Clean-Wrong table above is the subtype-transfer endpoint; the full-validation effects are available under `broad_effects[*].heldout_validation_overall` in the machine JSON.

## Observed findings

- Broad Screen C10 (CleanCE 0.15) has held-out Clean-Wrong robust effects of +0.092 pp (L2) and +0.000 pp (L4); this is not a consistent robust subtype transfer.
- C12 (MART-inspired adversarial hard-label proxy) is positive on the held-out Clean-Wrong subset in both seeds (+0.275 pp, +0.543 pp), but its non-CW train spillover is negative in both seeds; this is a candidate family signal, not a generalization claim.
- G2 (CE20 reliability gate) selected-CW robust effects are -0.634 pp (L2) and +2.236 pp (L4), while all-validation effects are +0.440 pp and -0.560 pp; the gate is not confirmed as a robust generalization selector.
- G3 (KL10 practical gate) has all-validation robust effects of -1.060 pp and -1.140 pp; it does not provide a reliable online-selector justification.
- The dominant pattern is positive direct train-cohort effects accompanied by negative non-CW spillover and weak or seed-dependent held-out robust effects. This is evidence of distribution/interference sensitivity, not proof that Clean-Wrong treatment is generally ineffective.

## Interpretation rules

- Direct positive with held-out near zero is direct-only evidence.
- Direct and Spillover positive with held-out near zero is train-distribution spillover without held-out transfer.
- Positive held-out effects are evidence of transfer, not proof of generalization from two seeds.
- Direct positive with negative Spillover and/or Held-out is harmful interference.
- No ratio of Held-out/Direct is used as a primary metric.

Held-out subtype transfer is reported only after the fixed epoch-79 validation feature lineage passed. No new intervention is selected automatically.
