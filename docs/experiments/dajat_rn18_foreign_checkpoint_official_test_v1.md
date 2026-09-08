# DAJAT ResNet-18 through this pipeline: does our evaluation agree with the field's?

Contract: `ard_external_checkpoint_evaluation_v1`. Record: `dajat_rn18_foreign_checkpoint_official_test_v1.json`.
Decision: `docs/decisions/0007-dajat-foreign-checkpoint-official-evaluation.md` (option A).

**Question.** Before the thesis compares a distilled ResNet-18 against a teacher-free one, it has
to know that its own official-test AutoAttack agrees with the published one. The cheapest check is
to take a published checkpoint and see whether this pipeline reproduces its published number.

**Subject.** RobustBench model zoo, Addepalli2022Efficient_RN18, gdrive_id 1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f; DAJAT, Addepalli et al., NeurIPS 2022
Checkpoint SHA-256 `a8fd58136e2e1ed78fd87f9fb105d4786b40db4bf5611aaec6db47729bc93f13`. These weights were **not** produced here and
carry no lineage in this repository; the record says so in its `lineage` field rather than
manufacturing one.

**What is ours in this measurement.** The official CIFAR-10 test split, all 10,000 examples,
through this project's own loader; the threat identity (L-infinity, epsilon 8/255); and AutoAttack
standard at the pinned upstream commit `a39220048b3c9f2cca9a4d3a54604793c68eca7e`. The weights and
the architecture are the field's. So this tests the evaluation stack, not the training stack.

## Result

| quantity | this pipeline | published | difference |
| --- | ---: | ---: | ---: |
| clean accuracy | 85.71 % | 85.71 % | -0.00 pp |
| AutoAttack (standard) | 52.45 % | 52.48 % | -0.03 pp |

**The preregistered rule was that AutoAttack within 0.3 pp of 52.48 adopts the pipeline as
calibrated. It lands at -0.03 pp, an order of magnitude inside that band, and clean accuracy
matches to the second decimal.** The evaluation stack agrees with the field's on the one
architecture and threat model the thesis will use it for.

## Why this was not known until 2026-09-08

The run finished on 2026-09-07 at 05:47 and was never imported into `docs/`, so the repository had
no record of it and a search of the repository for the number returned nothing. Two separate
read-only analyses concluded from that search that the run did not exist. It did; results live in
the runtime tree until the import step in the workflow's rule 5 moves them, and that step was
skipped.

It was also invisible in its own log. The script computed `difference_pp` by reading a key named
`robust_accuracy` from a dictionary whose key is `autoattack_accuracy`, so the value defaulted to
NaN and the completion line read "AutoAttack nan% against a published 52.48% (difference +nan pp)".
The one field whose purpose is to say whether the calibration passed could not say it -- and a
*failed* calibration would have printed exactly the same thing. The script now refuses rather than
defaulting. The measured values were correct in the file the whole time and are unaltered here.

## What this does not show

It vouches for the evaluation stack, not for the training stack. It says nothing about whether a
model *trained* in this repository lands where a published training recipe would; the corrected
in-house TRADES sits 1.2 to 1.5 pp under its published range and that gap is untouched by this
result. If anything it sharpens the question, because the evaluation half is now ruled out as the
explanation.

One checkpoint is one point of calibration. Option C of the decision packet -- the same measurement
on IDBH's released checkpoint, about 2 GPU-hours -- would give a second point at the other end of
the 52.3 to 52.8 teacher-free band. It was not chosen.
