# FF/current-wrong future-failure forecasting status

Last updated: 2026-08-08

> **Rejected preliminary report:** the first CPU pass used validation histories
> truncated at epoch 112 (L3) and 192 (L4).  The Best/plateau, cross-seed GT,
> and predictor ranges below are retained only as an audit trail and must not be
> used as scientific evidence.  The rerun is blocked until complete epoch
> `0..199` histories and completed sibling manifests are collected.  The
> implementation now fails closed on both conditions.

The replay lineages already contain checkpoint inventories through epoch 199,
so this is presently a result-collection problem, not evidence that the two
training trajectories must be rerun.  Recollect the completed Ferret run bundle
or export exact W&B history before considering new training.

## Question

This analysis returns to the measurement problem after history-selected loss
interventions failed to improve Best validation PGD.  It asks whether
training-time student state can forecast failure on a later Best-performance
plateau, before defining another intervention.

The four inputs are development trajectories, not untouched confirmation:

| Run | Teacher role | Seed |
| --- | --- | ---: |
| L1 | Bartoldson WRN-94-16 (IRT) | 1 |
| L2 | Chen WRN34-10 (ERT) | 1 |
| L3 | Bartoldson WRN-94-16 (IRT) | 2 |
| L4 | Chen WRN34-10 (ERT) | 2 |

The current result uses the existing five-epoch KL-teacher-clean PGD-10 replay.
It is a development sensitivity analysis, not the intended primary CE-PGD-20
ground truth.  Official CIFAR-10 test and AutoAttack were not read.

## Ground-truth feasibility

The raw validation Best and the best checkpoint available on the five-epoch
replay grid are recorded separately.  Centred windows may not cross an LR
stage boundary.

| Run | Raw Best epoch | Replay-grid Best | Admissible candidates | Future-failure count range | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| L1 | 102 | 104 | 0 / 27 | -- | censored at the LR boundary |
| L2 | 190 | 189 | 12 / 27 | 5,798--8,700 | analyzable sensitivity |
| L3 | 105 | 104 | 0 / 27 | -- | censored at the LR boundary |
| L4 | 185 | 174 | 18 / 27 | 6,039--8,913 | analyzable sensitivity; coarse grid differs materially from raw Best |

For the 12 candidate definitions shared by the two Chen seeds, the positive-mask
Jaccard range is `0.565--0.638`.  This is moderate rather than decisive
cross-seed stability.  No GT candidate was selected from predictor performance,
and bootstrap was not started.

The Bartoldson censoring is structural.  Replaying a stronger attack on the
same checkpoint inventory will not create the missing same-stage checkpoint
before epoch 104.  Before spending GPU time, the study must choose between:

1. retaining the centred-window contract and running a new non-intervened
   trajectory with denser checkpointing around the first LR change; or
2. preregistering a different prospective plateau definition, explicitly
   acknowledging that it answers a different question.

## Exploratory predictor sensitivity

The following ranges are over every admissible KL-PGD-10 GT candidate.  They
are not a selected model result.

### Current-correct to future failure (FF)

| Run / anchor | Current online margin | Online correctness frequency | Online margin EMA |
| --- | ---: | ---: | ---: |
| L2 / 39 | `.740--.758` | `.901--.907` | `.921--.925` |
| L2 / 59 | `.760--.777` | `.913--.917` | `.932--.940` |
| L2 / 79 | `.755--.769` | `.919--.922` | `.930--.937` |
| L4 / 39 | `.763--.786` | `.891--.899` | `.915--.924` |
| L4 / 59 | `.760--.780` | `.911--.916` | `.932--.939` |
| L4 / 79 | `.769--.787` | `.914--.923` | `.933--.941` |

These AUROC ranges indicate that exact online student history, especially
margin EMA, contains substantially more signal than the instantaneous online
margin for the two analyzable Chen trajectories.  This does not yet establish
the result for Bartoldson or under the primary CE-PGD-20 GT.

### Current-wrong to future failure

At anchor 79, online correctness frequency has AUROC `.767--.778` for L2 and
`.766--.777` for L4.  Online margin EMA has `.753--.760` and `.749--.758`.
The label is `current_wrong_future_failure`, not non-recovery: five-epoch
observations do not prove that a sample never recovered between checkpoints.

### Teacher disagreement

Adding teacher margin response to the fixed replay L/T/S representative often
improves that replay-domain representative, while teacher adversarial
confidence and the current response-gap formulation generally reduce AUROC.
None of these fixed D additions exceeds the exact-online student-history FF
signal in this sensitivity analysis.  D is therefore an ablation result, not a
selected final component.

## Mask stability

For the fixed top-10% online student masks, current-correct FF masks have low
consecutive-anchor overlap, whereas current-wrong masks are substantially more
stable.  For example, across L1--L4:

- FF correctness-frequency Jaccard is approximately `.097--.132`;
- current-wrong correctness-frequency Jaccard is approximately `.568--.785`.

These masks are recomputed inside a changing eligibility stratum.  Their
Jaccard therefore mixes score-rank movement with samples entering or leaving
the current-correct/current-wrong population; it is not pure rank stability.

## Reproducible output

Development output (ignored cache):

```text
.cache/analysis/ffnr-cpu-v1-dev-REJECTED-incomplete-validation/
  ffnr-report.json
  ffnr-score-rows.parquet
  ffnr-ground-truth-rows.parquet
```

The report binds the config, cohort, replay/online/validation inputs, and both
Parquet outputs by SHA-256.  It contains 540,000 score rows and 1,350,000 GT
rows.  The full run took 7 minutes 44 seconds on CPU and peaked at about
6.2 GiB RSS.  The implementation now reuses score orderings instead of sorting
the same score separately for every GT candidate.

## Next decision and GPU work

Do not launch the full CE-PGD-20 replay merely because a GPU becomes free.
First inspect the immutable checkpoint inventory and resolve the Bartoldson
centred-window infeasibility.  Once a primary GT contract is frozen, the GPU
path is:

1. freeze one union schema for strong GT and missing L/T/S/D primitives;
2. run one real-checkpoint public-CLI smoke through Parquet, lineage, sparse-ID
   join, and report creation;
3. measure time and teacher-forward overhead;
4. replay independent feature/outcome jobs longest-processing-time-first;
5. compute point estimates before any preregistered bootstrap.

No intervention, new training seed, official PGD, or AutoAttack follows from
the present development result without a separate frozen decision.
