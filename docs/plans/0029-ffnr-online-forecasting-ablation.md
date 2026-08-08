# FF/NR online forecasting ablation

## Purpose

The previous history-selected interventions were prognostic but did not improve
Best validation PGD.  This milestone therefore returns to the measurement
problem: predict, without changing the loss, which currently correct samples
will fail on the future Best-performance plateau (FF) and which currently wrong
samples will not recover on that plateau (NR).

The attached FF/NR proposal is the research hypothesis for this milestone.  It
does not authorize choosing a ground truth because it gives high AUROC, tuning
weights on the same four development trajectories, opening official test or
AutoAttack, or starting another intervention campaign.

## Existing evidence and inputs

- [x] L1--L4 cover Bartoldson (IRT) and Chen WRN34-10 (ERT), seeds 1 and 2.
- [x] Schema-v2 common-PGD feature observations exist at epochs `4..99` every
  five epochs and outcome observations at `99..199` every five epochs.
- [x] Each epoch has exactly 45,000 unique original CIFAR train IDs; IDs are
  sparse in `[0, 50000)` and must never be joined by row position.
- [x] Exact online checkpoint state is available at anchors 39/59/79 for all
  four runs.  L1/L2 checkpoint bytes are local; L3/L4 replay Parquets are local
  but their source checkpoint bytes are not.
- [x] The two Hamster GPUs are occupied by immutable `52affda` Chen WRN34-20
  upstream/paper-aligned SAAD jobs.  This work must not stop or mutate them.

Canonical replay root:

```text
/home/islab/workspace-local/shunsuke.naito/ard-analysis/h5-matrix-6e5dcf5/
```

## Scientific contracts

### Domains and leakage

- Plateau membership is selected only from validation PGD trajectories.  The
  per-sample GT is then evaluated on the baseline run's 45,000-sample train
  replay.  CIFAR-10 official test is not read.
- Current FF/NR eligibility comes from the deployable online checkpoint state
  (`previous_robust_correct`), not replay correctness.  Replay correctness is an
  outcome/diagnostic domain and cannot redefine the online candidate set.
- Existing observations are five-epoch snapshots.  Reports must say
  `five_epoch_replay`, never claim every-epoch measurements, and must not
  interpolate missing observations.
- L1--L4 are development trajectories already used by earlier H4/H5 work.
  Their results are not unseen confirmation evidence.
- GT-count precision uses future prevalence and is an analysis-only oracle-q
  diagnostic.  Fixed top 1/5/10/20 percent results are the deployable selection
  diagnostics.

### GT candidate family

For each run, record the raw earliest epoch `b_raw` with maximum validation
PGD.  Because the existing replay is available only every five epochs, also
record `b_grid`, the best replayed checkpoint.  The sensitivity analysis forms
its target only around `b_grid`; it must never imply that `b_grid == b_raw`.
Within the scheduler stage containing `b_grid`, form the contiguous
saved-checkpoint component containing `b_grid` whose validation PGD is at
least `best_grid - delta`, for
`delta in {0.25, 0.5, 1.0}` percentage points.  A centred odd window of 3, 5,
or 7 checkpoints is admissible only when the complete window is present in that
component.  A terminal ERT window is reported as right-censored; it is not
silently shrunk, shifted asymmetrically, or allowed to cross an LR boundary.

For each window, define future failure using thresholds:

- `majority`: strictly more than half of plateau checkpoints are wrong;
- `two_thirds`: at least `ceil(2K/3)` are wrong;
- `all`: all plateau checkpoints are wrong.

The existing outcome panel is deterministic KL-PGD10 and is therefore a
cadence-limited development sensitivity analysis, not the primary strong-attack
GT.  The intended primary GT uses the saved selection/evaluation CE-PGD20
identity and requires the queued GPU replay.  These attack domains must remain
separate in every report.

Candidate choice uses prevalence, candidate-to-candidate Jaccard, cross-seed
prevalence, IRT/ERT validity, window-size stability, threshold stability, and
right-censoring only.  Prediction metrics are excluded from GT selection.  If
these criteria do not identify one unambiguous definition, the milestone stops
at a comparison table for human selection.

### FF/NR and score candidates

For a chosen or explicitly requested GT candidate and online anchor `t`:

```text
FF = future_failure AND online_current_correct
current_wrong_future_failure = future_failure AND NOT online_current_correct
```

The second stratum is not called non-recovery unless correctness is observed at
every intermediate point and no recovery is proven.  The present five-epoch
panel cannot establish that stronger statement.

The first CPU pass evaluates only primitives already present in the replay or
online checkpoint states:

- L: negative current probability margin and its within-epoch midrank.  Logit
  margin and adversarial CE are marked unavailable until a GPU replay records
  logits.
- T: 5-epoch difference, recent linear slope, rank slope, and fast-minus-slow
  EMA over the five-epoch replay grid.  A one-epoch difference is unavailable.
- S: online correctness frequency, current streak when checkpoint bytes expose
  it, replay flip rate, and margin variance.  Availability is reported per run;
  missing values are never imputed.
- D: teacher margin response, adversarial entropy, and response gap where
  derivable.  Teacher JS is available from format-v3 checkpoint state but not
  the historical replay Parquet; its availability is reported separately.

Exact-online and replay-domain predictors are reported separately.  The
four-run exact-online comparison uses only fields present in all four compact
anchor exports; richer L1/L2 checkpoint fields are availability diagnostics,
not a stronger score compared unfairly against L3/L4.  Every anchor must be at
least one observed replay interval earlier than the first plateau checkpoint.

All risk directions are explicit.  Multi-component scores are equal-weight
averages of deterministic midranks.  Stage A/B/C/D selection is reported, but
the tool does not automatically declare a final predictor or tune weights.

### Metrics and stability

For FF and NR separately, per run and anchor, report candidate count, positives,
prevalence, AUROC, AUPRC, GT-count precision/recall, and top 1/5/10/20 percent
precision/recall/lift.  Across consecutive available anchors report mask
Jaccard, retention, entry, exit, and FF/NR state transitions.  IRT and ERT stay
separate; means are secondary summaries only.

## Implementation

- [x] Add a typed, read-only analysis module under `src/ard/analysis/` for
  plateau GT candidates, stable-ID joins, L/T/S/D primitives, staged metrics,
  and mask stability.
- [x] Add one public analysis CLI and a frozen YAML configuration binding the
  L1--L4 inputs, validation trajectories, candidate grid, anchors, and output
  directory.
- [x] Write non-overwriting JSON/Parquet outputs with hashes for every input,
  config, source SHA, run/config/teacher/seed identity, attack identity, and
  availability reason for every candidate primitive.
- [x] Add focused tests for plateau component/window/censoring, threshold math,
  sparse-ID joins, FF/NR disjoint completeness, score direction/ties,
  AUROC/AUPRC/top-q, mask stability, domain separation, and no official-test
  input.
- [x] Rerun the CPU point analysis on L1--L4 as a five-epoch KL-PGD10 development
  sensitivity analysis.  Do not bootstrap until the
  preregistered point comparison requiring it is chosen.
- [x] Regenerate a compact GT comparison and feature-availability report for human
  review.  Do not select an ambiguous GT automatically.

## GPU follow-up

GPU work begins only after a Hamster GPU becomes free.

1. Generate one hash-bound checkpoint inventory and reuse it.
2. Freeze the union schema needed by GT, L/T/S/D, and downstream taxonomy.
3. Run one real-checkpoint, full-CLI smoke through Parquet, stable-ID join,
   lineage validation, and non-overwriting report creation.
4. Measure wall time and assign full jobs longest-processing-time-first.
5. Run independent feature and outcome jobs concurrently when two GPUs are
   available.

The primary GPU outcome replay uses the already defined saved-selection
CE-PGD20 attack identity.  The schema should add only primitives unavailable
now and worth measuring: student logit margin, adversarial CE, teacher JS
response/full probabilities needed to compute it, and student
clean-to-adversarial response.  Full images, gradients, attack-step tensors, and
full-logit histories are not persisted.

## Selected tests

```bash
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest \
  tests/unit/test_ffnr_forecasting.py -m unit
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
  -m ard.cli.ffnr_forecasting --help
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
  scripts/verify.py --changed
```

No CUDA test is selected for the CPU-only delta.  A later replay change requires
the existing common-PGD fixed-batch and gradient contracts plus the real
one-checkpoint smoke.

## Risks and stopping conditions

- Validation histories for the four baseline runs may need a one-time W&B
  artifact export before plateau membership can be computed.  Fetch once and
  bind the bytes; do not repeatedly query W&B history.
- ERT Best near epoch 199 can make symmetric windows impossible.  Such cases
  are censored evidence, not a reason to use future information or silently
  change the window.
- Training adversarial current state and deterministic common-PGD outcome state
  differ.  This is intentional deployment-vs-evaluation separation and must be
  reported as a possible noise source.
- If exact online fields are absent from L3/L4 compact exports, restrict the
  first four-run comparison to the common available subset.  Do not compare
  richer L1/L2 scores against poorer L3/L4 scores as if identical.
- Stop before Stage A if no scientifically defensible GT candidate emerges.
- Stop before intervention if prediction is unstable, highly teacher/seed
  specific, or does not beat current-state baselines with useful top-q lift.

## Completion conditions

- [x] CPU analysis is reproducible from one command and creates hash-bound,
  non-overwriting reports for all four development trajectories.
- [x] GT candidate tables expose censoring, prevalence, and overlap without
  using prediction performance to select a target.
- [x] FF and NR are disjoint and exactly partition future failures at every
  analyzed anchor.
- [x] Available L/T/S/D ablations and mask stability metrics are reported per
  teacher and seed; unavailable primitives have explicit reasons.
- [x] Focused tests and impact-selected tests have recorded commands/results.
- [x] One consolidated scientific review has no unresolved P0/P1.
- [x] Current Chen WRN34-20 GPU jobs were not interrupted.

## Progress log

- 2026-08-08: inspected the proposal, repository contracts, L1--L4 replay
  matrix, exact online anchors, and active GPU services.  CPU implementation is
  authorized; GPU replay is queued behind the two active Chen jobs.
- 2026-08-08: focused unit tests passed (`7 passed` after the cached-ranking
  regression was added), Ruff lint/format and focused mypy passed,
  and the real L1 sparse-ID smoke completed in 62 seconds with a 4.8 GiB peak.
  L1's raw/grid Best epochs are 102/104; all centred candidates are correctly
  censored because the preceding replay checkpoint is in the prior scheduler
  stage.  L3 is censored for the same structural reason.  L2 and L4 retain 12
  and 18 admissible development-sensitivity candidates respectively.  The
  full sequential CPU point analysis is in progress; no bootstrap was started.
- 2026-08-08: full L1--L4 point analysis completed in 7m44s (about 6.2 GiB
  peak RSS), producing 540,000 score rows and 1,350,000 GT rows with matching
  report hashes.  L1/L3 have no admissible centred GT; L2/L4 have 12/18.
  Chen cross-seed candidate Jaccard is `0.565--0.638`, so no GT was selected
  and no bootstrap was launched.  Results and the blocked GPU decision are in
  `docs/FFNR_FORECASTING_STATUS.md`.
- 2026-08-08: the first impact gate exposed an overly broad
  `configs/analysis/ -> T3` rule and attempted unrelated two-process training;
  three Gloo cases timed out because the isolated shell rejects localhost
  sockets.  The impact map now binds the FF/NR module, CLI, and config to their
  focused unit test.  The corrected `scripts/verify.py --changed` selected only
  T0/T1 and passed (`7 + 33` tests).  A non-isolated rerun of the unrelated
  Gloo failures was requested but rejected by the execution-usage limit.
- 2026-08-08: consolidated scientific review rejected the preliminary report:
  L3 validation history stops at epoch 112 and L4 at 192, so their Best epochs
  and cross-seed GT comparisons are not valid.  The report remains only as an
  audit artifact.  Required fixes are exact `0..199` validation coverage with
  sibling-manifest identity binding, tie-inclusive top-q masks, complete
  lineage hashes, required-anchor failure, and stable-ID universe binding.
  The D1 direction is intentionally kept as *teacher non-response* risk from
  the attached hypothesis (`adv-clean` closer to zero means more disagreement),
  rather than changing it to the reviewer's distinct teacher-fragility risk.
  Primary CE-PGD20 replay is blocked until complete histories are recovered.
- 2026-08-08: review fixes implemented.  Validation now requires exact unique
  epochs `0..199`, a completed sibling manifest matching run/config/Git/seed/
  teacher identity, and the report binds all lineage/attack identities.  Top-q
  selections are tie-inclusive with realized fractions, requested anchors fail
  closed, D1 is named teacher-nonresponse risk, equivalent GT masks are grouped,
  and cross-seed Jaccard requires an identical stable-ID/class universe hash.
  Ruff, focused mypy, and `43` combined FF/NR + gate unit tests passed.  The
  actual L3 file now fails immediately with the expected incomplete-history
  error; L3/L4 data collection remains the blocker.
- 2026-08-08: fix-delta review found and resolved one final P1: within-run GT
  equivalence representatives had been reused for cross-seed matching and
  could hide a valid same-formula comparison.  Formula masks are now retained
  for cross-seed Jaccard, while representatives only deduplicate within-run
  summaries/Parquet.  The final impact gate selected T0/T1 only and passed
  (`10 + 33` tests).  No unresolved P0/P1 remains in the correctly blocked
  implementation.
- 2026-08-08: collected the complete L3/L4 Ferret bundles and regenerated the
  formal point report from clean immutable `44c7ff9`.  It completed in 7m41s
  with 6.20 GiB peak RSS.  L1/L3 remain censored; L2 has 12 admissible
  candidates and L4 has 6.  Complete L4 raw/grid Best are 196/194, correcting
  the rejected truncated-history values 185/174.  The six formula-matched Chen
  masks have cross-seed Jaccard `0.557--0.615`; no GT or predictor was selected
  and no bootstrap was launched.  Chen-only CE-PGD20 replay is now authorized
  for the union of admissible window epochs, while Bartoldson replay remains
  blocked by missing same-stage checkpoints rather than GPU availability.
- 2026-08-09: committed and pushed deterministic strong-replay fixes at
  `e5cb442`.  A real L2 epoch-39 smoke passed and matched the corresponding
  full-replay epoch exactly.  L2 feature/outcome ran concurrently on Hamster;
  L4 feature/outcome ran concurrently on Ferret.  All four outputs have the
  exact CE-PGD20 identity, clean source SHA, deterministic backend flags, and
  the same 45,000-sample sparse-ID/class hash.
- 2026-08-09: generated the initial point report from clean commit `ca11a4e` in
  2m07s with 1.86 GiB peak RSS.  Strong-GT cross-seed Jaccard is
  `0.848--0.858`.  Teacher adversarial wrong-confidence has FF AUROC
  `0.991--0.994`, online margin EMA `0.917--0.935`, but top-10 masks are much
  less stable (`~0.19` and `~0.09` cross-seed respectively).  No predictor or
  GT was selected, and no bootstrap, intervention, official test, or AA ran.
- 2026-08-09: scientific review found one P1 report-lineage omission and one
  P2 signal-naming ambiguity.  Commit `6327fd7` now binds all eight analysis
  inputs, both checkpoint inventories, selected checkpoint hashes, and the
  exact CE-PGD20 identity.  The signal is named signed teacher wrong-class
  dominance, with its formula and larger-is-higher-risk direction embedded in
  the report.  The regenerated report SHA is
  `cf32dcce02c21617bd7c3322dfa699eec5e2ae11b220ef428b6b99342e68c797`.
