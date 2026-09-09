# FF/current-wrong future-failure forecasting status

Last updated: 2026-08-09

The formal development report was regenerated from complete, unique validation
epochs `0..199` for all four runs.  Each history is bound to its completed
sibling manifest and run/config/Git/seed/teacher identity.  The earlier report
made from truncated L3/L4 histories remains rejected and is not used below.

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
| L4 | 196 | 194 | 6 / 27 | 6,228--8,259 | analyzable sensitivity; terminal three-checkpoint window only |

For the six candidate definitions shared by the two Chen seeds, the positive-mask
Jaccard range is `0.557--0.615`.  This is moderate rather than decisive
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
| L4 / 39 | `.770--.787` | `.891--.893` | `.916--.922` |
| L4 / 59 | `.768--.776` | `.910--.910` | `.932--.933` |
| L4 / 79 | `.777--.784` | `.916--.920` | `.936--.940` |

These AUROC ranges indicate that exact online student history, especially
margin EMA, contains substantially more signal than the instantaneous online
margin for the two analyzable Chen trajectories.  This does not yet establish
the result for Bartoldson or under the primary CE-PGD-20 GT.

### Current-wrong to future failure

At anchor 79, online correctness frequency has AUROC `.767--.778` for L2 and
`.766--.771` for L4.  Online margin EMA has `.753--.760` and `.751--.752`.
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

- FF correctness-frequency Jaccard is approximately `.095--.140`;
- current-wrong correctness-frequency Jaccard is approximately `.723--.869`.

These masks are recomputed inside a changing eligibility stratum.  Their
Jaccard therefore mixes score-rank movement with samples entering or leaving
the current-correct/current-wrong population; it is not pure rank stability.

## Reproducible output

Formal development output (ignored cache):

```text
.cache/analysis/ffnr-cpu-44c7ff9-v1/
  ffnr-report.json
  ffnr-score-rows.parquet
  ffnr-ground-truth-rows.parquet
```

The report binds the config, cohort, replay/online/validation inputs, and both
Parquet outputs by SHA-256.  It contains 540,000 score rows and 315,000
deduplicated GT rows.  The full run took 7 minutes 41 seconds on CPU and peaked
at 6.20 GiB RSS.  Report SHA-256 is
`98a2a2ea4a7c17c55ea604b676e7e58be60c35c9fe2e00a09f2ba9d99ac6faa2`.

## Primary CE-PGD20 result

The authorized Chen-only strong replay is complete.  It used the saved
selection/evaluation attack exactly: pixel-space Linf CE-PGD20, epsilon
`8/255`, step size `2/255`, random start, and student/teacher eval mode.  The
replay ran from clean commit `e5cb442` with deterministic CUDA algorithms,
TF32 disabled, and an identical 45,000-sample sparse-ID/class universe on both
hosts.  The epoch-39 smoke and full L2 replay were exactly equal row for row.

Under CE-PGD20, the admissible future-failure definitions contain
`11,113--12,363` samples for L2 and `11,421--12,348` for L4.  The six common
three-checkpoint formulas have cross-seed positive-mask Jaccard
`0.848--0.858`, substantially above the KL-PGD10 sensitivity result
`0.557--0.615`.  Majority and two-thirds are mathematically equivalent for a
three-checkpoint window; `all` remains a distinct persistent-plateau endpoint.
No endpoint was selected from predictor performance.

For current-correct samples, the main point-estimate AUROC ranges are:

| Run / anchor | Online margin EMA | Strong current logit margin | Teacher signed wrong-class dominance |
| --- | ---: | ---: | ---: |
| L2 / 39 | `.920--.925` | `.937--.945` | `.992--.993` |
| L2 / 59 | `.930--.935` | `.935--.940` | `.992--.993` |
| L2 / 79 | `.929--.933` | `.924--.931` | `.991--.993` |
| L4 / 39 | `.917--.918` | `.927--.931` | `.992--.994` |
| L4 / 59 | `.929--.929` | `.924--.928` | `.991--.992` |
| L4 / 79 | `.930--.932` | `.934--.935` | `.992--.993` |

Teacher signed wrong-class dominance is
`max_{c != y} p_T(c | x_adv) - p_T(y | x_adv)`.  Unlike the rejected
entropy/overconfidence risk, it does not penalize a teacher for being confidently
correct: correct examples have a negative signed value, and larger values mean
higher risk.  Its very high association is a new Chen/ERT development result, not
evidence that downweighting those samples improves training.  It also requires
a strong student attack and teacher forward, whereas online margin EMA is
already available from training state.

High AUROC does not imply a stable intervention set.  Within-run consecutive
anchor top-10% Jaccard is only `.196--.203` for teacher signed dominance,
`.071--.076` for strong logit margin, and `.096--.108` for online margin EMA.
Across the two seeds it is approximately `.187--.191`, `.070--.077`, and
`.093--.096`, respectively.  These masks are conditioned on the changing
online-current-correct population, so the values mix eligibility transitions
with score-rank movement.

Formal output (ignored cache):

```text
.cache/analysis/ffnr-strong-point-6327fd7-v1/
  ffnr-strong-point-report.json
  ffnr-strong-points.parquet
```

The clean analysis commit is `6327fd7`; report SHA-256 is
`cf32dcce02c21617bd7c3322dfa699eec5e2ae11b220ef428b6b99342e68c797`.
The report records path and SHA-256 for every feature, outcome, online-state,
validation, lineage, and checkpoint-inventory input, plus each selected
checkpoint hash and the exact attack identity.
The CPU point analysis took 2 minutes 7 seconds and 1.86 GiB peak RSS.  No
bootstrap, official test, AutoAttack, intervention, or new training was run.

## Chen D3--D5 decomposition

The follow-up analysis uses Chen/ERT L2 and L4 (training seeds 1 and 2),
anchors 39/59/79, and the same train-split CE-PGD20 replay contract.  Its
primary endpoint is majority failure on the common terminal window
`[189, 194, 199]`; wrong on all three is retained as a sensitivity endpoint.
This common-terminal endpoint is deliberately identical across seeds.  It is
not the L2 best-centred `[184, 189, 194]` candidate in the earlier multi-formula
report, so the formal artifact records the endpoint epochs explicitly.

### D3: why teacher signed dominance reaches AUROC near .99

For the primary endpoint, the teacher-adversarial correctness split is almost
binary in both seeds and at every anchor:

| Run / anchor | Teacher correct: future-failure rate | Teacher wrong: future-failure rate |
| --- | ---: | ---: |
| L2 / 39 | 842 / 23,421 (`3.60%`) | 689 / 690 (`99.86%`) |
| L2 / 59 | 858 / 23,952 (`3.58%`) | 665 / 665 (`100.00%`) |
| L2 / 79 | 940 / 24,323 (`3.86%`) | 735 / 737 (`99.73%`) |
| L4 / 39 | 872 / 23,608 (`3.69%`) | 684 / 685 (`99.85%`) |
| L4 / 59 | 942 / 24,220 (`3.89%`) | 722 / 724 (`99.72%`) |
| L4 / 79 | 913 / 24,474 (`3.73%`) | 796 / 800 (`99.50%`) |

The signed teacher score is strongly correlated with both strong current
Student risk (Spearman `.795--.824`) and online margin-EMA risk
(`.846--.878`).  It is nevertheless not redundant.  In deterministic
class-stratified five-fold OOF comparisons, adding teacher dominance to strong
current margin improves AUROC by `.049--.062` and reduces log-loss by
`.029--.035` across all six run/anchor cells.  Adding it after both margin and
history still improves AUROC by `.033--.041`, with lower log-loss in every
cell.

Thus the `.99` association is driven substantially by whether Chen remains
correct on the Student-crafted adversarial input, but teacher state retains
incremental information after conditioning on Student current state and
history.  This is predictive evidence only: it does not establish that
weakening KD on these samples improves training.

### D4: future failure is not one homogeneous event

At anchor 79, the online-history subtypes inside primary future failure are:

| Run | Oscillating | Stable-then-forgotten | Transient-correct | Total |
| --- | ---: | ---: | ---: | ---: |
| L2 | 287 (`17.1%`) | 487 (`29.1%`) | 901 (`53.8%`) | 1,675 |
| L4 | 297 (`17.4%`) | 544 (`31.8%`) | 868 (`50.8%`) | 1,709 |

Only about 30% is the classical “stable, then forgotten” pattern.  Roughly
half consists of samples that are correct only transiently near a decision
boundary.  Same-subtype cross-seed Jaccard is low (`.030` oscillating, `.091`
stable-then-forgotten, `.052` transient-correct), so an intervention must not
assume that the identical sample IDs occupy one FF subtype across seeds.  The
taxonomy uses saved snapshots rather than every minibatch transition and is
therefore a snapshot-dynamics description, not a complete forgetting trace.

### D5: strong current-wrong contains distinct NR mechanisms

Using dense, same-domain CE-PGD20 snapshots from anchor 79 through epoch 199:

| Run / subtype | Count (% of strong-current-wrong) | Student clean-wrong | Teacher adv-wrong |
| --- | ---: | ---: | ---: |
| L2 persistent-wrong | 9,466 (`42.1%`) | `69.6%` | `75.0%` |
| L2 recovered-relapsed | 10,042 (`44.7%`) | `17.1%` | `3.5%` |
| L2 recovered-stable | 2,956 (`13.2%`) | `10.8%` | `0.1%` |
| L4 persistent-wrong | 9,530 (`42.9%`) | `70.2%` | `75.1%` |
| L4 recovered-relapsed | 9,964 (`44.9%`) | `19.3%` | `3.6%` |
| L4 recovered-stable | 2,708 (`12.2%`) | `11.6%` | `0.3%` |

Persistent-wrong is comparatively seed-stable (cross-seed Jaccard `.821`) and
is strongly enriched for clean difficulty and teacher failure.  In contrast,
recovered-relapsed (`.409`) and recovered-stable (`.102`) are less stable and
almost always retain a correct teacher.  This rejects one uniform treatment
for all current-wrong samples: persistent teacher-misaligned cases and
recoverable boundary cases require separate hypotheses.

The analysis also generates a bounded class-matched blinded CIFAR panel for
human label/image-quality review.  Target selection requires persistent strong
wrong, Student clean-wrong, and teacher-adversarial wrong; the public manifest
contains no target/control role, outcome, score, teacher state, or role-coded
row order.  Image inspection remains necessary before making any label-noise
claim.

Formal development output (ignored cache):

```text
.cache/analysis/ffnr-strong-diagnostics-6a90011-v1/
  ffnr-strong-diagnostics.json
  ffnr-strong-diagnostic-points.parquet
  ffnr-strong-blinded-candidates.json
  images/  # 200 bounded blinded PNGs
```

The report was generated from clean analysis commit `6a90011`; its SHA-256 is
`cfdd3f66174562815b7550ee1d7cd7d02752eaaa160d45426c872e8c12eec269`.
The blinded manifest and point Parquet SHA-256 values are respectively
`8d1f21bec9dae9c4d750693374f6b0fd0f752db512491e87e10ac227608a8a02`
and `d99d1f3ab75963f384e1685373a202d2a2bab0b05e4a0a03bdb86b75495dc0ec`.
All five dense replay chunks are bound to source, checkpoint inventory,
checkpoint/result bytes, exact attack, seed, batch, runtime, config, and stable
ID/class identity.  GPU ordinals alone are normalized because independent
chunks intentionally ran on different devices.

## Next decision

- Complete the blinded human image/label review first; this is CPU/manual work
  and does not require another training run.
- Preregister at most two small intervention candidates, separated by causal
  story: (1) a persistent teacher-misaligned NR route that reduces harmful
  adversarial KD and uses a true-label anchor, and (2) a teacher-correct
  transient/oscillating FF route that preserves KD and targets boundary
  stability rather than softening the teacher.
- Screen both from one common parent with a no-intervention continuation and
  class/count-matched random controls.  Prediction quality alone is not a Go
  criterion; the primary goal remains validation Best robust accuracy.
- Bartoldson/IRT remains outside this exact comparison because its current
  checkpoint inventory cannot construct the same preregistered endpoint.
  Checking transfer to IRT requires a prospective denser-checkpoint run or a
  separately frozen endpoint, not silent reuse of a different window.
- Official test and AutoAttack remain untouched until a method is fixed on
  development trajectories and confirmed on unused seeds.

No intervention, new training seed, official test, or AutoAttack was launched
by this diagnostic milestone.
