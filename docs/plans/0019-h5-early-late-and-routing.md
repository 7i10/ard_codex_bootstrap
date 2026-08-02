# H5 Early/Late selector and bounded routing plan

## Status

- Owner: main thread; one Sol planning pass, one Terra implementation pass
- Branch / base SHA: `research/h5-early-late` / `d1984cd`
- Current milestone: M0 compact input generation active; M1/M2 implementation complete pending real-data reports
- Last updated: 2026-08-02

## Goal

Use the completed non-intervened L1--L4 trajectories to determine when a
run-local student-history score becomes useful, how its selected top-10% set
differs from the frozen cross-run predictor, and which bounded student/teacher
states are plausible routing targets.  Freeze at most two v2 interventions
before consuming a new confirmation seed.

## Non-goals

- No new softening or fixed `0.5` KD-downweight run.
- No official CIFAR-10 test or AutoAttack use for method selection.
- No automatic label correction, broad curriculum search, or seed-3 launch
  before the score, outcomes, and routing API are frozen.
- No claim that training-sample group prevalence bounds test accuracy gain.

## Existing state

- H2 found history-over-current-state AUROC deltas of `+0.0499`, `+0.0686`,
  `+0.0513`, and `+0.0663` on L1--L4.
- H4 found no meaningful best validation PGD gain from uniform target
  softening or fixed adversarial-KD downweight.  These treatments are closed
  for the primary best-accuracy objective.
- L1--L4 versioned checkpoints contain format-v3 state every five completed
  epochs.  Repository epoch labels are zero-based: completed epochs
  40/60/80/100 correspond to checkpoint epochs `39/59/79/99`.  The previously
  frozen cross-run predictor, however, was fit in the standardized
  saved-checkpoint replay domain, not the online state domain.
- The frozen H2 exporter and predictor are fixed to epoch 99/199 and must not
  be relaxed for H5.

## Scientific contracts affected

- All H5 features must precede their outcome and use only detached training
  observations.  Future correctness/forgetting cannot enter score creation.
- Saved-checkpoint replay uses the existing pixel-space `8/255`, step `2/255`,
  PGD-10 KL training-attack identity unless a separate analysis attack is
  explicitly named and hash-bound.
- Stable sample IDs, partition hash, teacher checkpoint hash, config hash,
  scientific Git SHA, checkpoint SHA, and analysis source hash are required.
- Official test remains sealed until the v2 design is frozen.

## Decisions

- Accept H5 top-q overlap reporting.  At `q=10%`, report Jaccard, overlap,
  common/fixed-only/rank-only/neither counts, forgetting prevalence,
  precision, recall, and lift.  Overlap is diagnostic, not a new gate.
- Compare fixed and rank scores only on the same hash-bound replay feature
  panel.  Applying replay-fitted coefficients to online SampleStateStore
  frequency/EMA would mix feature definitions and is rejected.  H5-Late uses
  the pre-existing seed-0 fit and replay features for both scores, then joins
  the exact epoch-99/199 online state outcome by stable ID.  This explicitly
  tests transfer from the predictor's checkpoint-panel training outcome to
  online future forgetting and avoids redundant outcome-checkpoint replay.
- Split H5 into H5-Late and H5-Early.  H5-Late keeps anchor checkpoint 99 and
  outcome through 199.  H5-Early uses checkpoints 39/59/79 and two distinct
  outcomes: peak-window error and post-peak forgetting.
- Define peak-window error as wrong at at least two of checkpoint epochs
  `99/104/109`.  Define post-peak forgetting on epoch-109-correct samples as a
  later correct-to-wrong transition through epoch 199.  Do not mix these
  saved-state outcomes with validation PGD.
- Keep the run-adaptive score fixed as the equal-weight percentile midrank of
  low robust-correct frequency and negative margin EMA.  Frequency-only and
  margin-only are diagnostics; the fixed cross-run predictor and best
  instantaneous current-state feature are references.
- H4a primary states are mutually exclusive: anchor-correct becomes stable or
  future-forgetting; anchor-wrong becomes persistent-wrong, recovered-stable,
  or recovered-relapsed.  Teacher adversarial correctness and student
  clean-correct/robust-wrong are cross-tabs.  Wrong-confidence and margin trend
  remain continuous until a threshold is frozen.
- Report same-panel oracle headroom only, not “maximum possible overall
  accuracy gain”.
- Reject the proposed epoch-99 delayed-LR launch.  The trainer calls
  `scheduler.step()` before saving epoch 99; `MultiStepLR([100,150])` therefore
  has `last_epoch=100` and LR `0.01`.  A valid delayed `[120,170]` branch must
  start from checkpoint epoch 79 (or reconstruct an explicitly tested
  counterfactual scheduler state); epoch 99 cannot delay the first drop.
- Defer seed-3 common parents.  L1--L4 are development trajectories; after
  freezing H5/H4a, one unseen Bartoldson seed is the first confirmation run.
  Chen follows only as a no-harm check after a Bartoldson candidate passes.

## Milestones

- [ ] M0 -- inventory exact L1--L4 periodic checkpoint bytes/manifests,
  replay panels, and frozen predictor identities.
  - Files: this plan and read-only artifact inventory output outside Git.
  - Tests: checkpoint epoch/state/identity validation only.
  - Acceptance: exact inputs exist for all required epochs or missing inputs
    are named without launching replacement training.
  - Rollback: none; read-only.

- [ ] M1 -- implement H5-Late overlap and score diagnostics without altering
  frozen H2/replay/selector code.
  - Files: new `src/ard/analysis/history_screen.py`, CLI, focused unit tests.
  - Owner: one Terra implementation pass after this contract is frozen.
  - Tests: row-order invariance, deterministic midrank/tie-break, exact set
    partition, zero-size groups, temporal leakage rejection, hash drift.
  - Acceptance: fixed and adaptive metrics use identical replay rows and one
    hash-bound report covers L1--L4; no GPU training.
  - Commit: `analysis: add hash-bound H5 history screen`.

- [ ] M2 -- add parameterized multi-anchor state export and H5-Early outcomes.
  - Files: new module/CLI and focused tests; frozen H2 modules remain unchanged.
  - Tests: epoch-label conversion, exact checkpoint sequence, disjoint risk
    sets, future leakage, stable-ID joins and anchor permutation invariance.
  - Acceptance: separate best-oriented and RO-oriented tables at epochs
    39/59/79/99.

- [~] M3 -- produce bounded H4a taxonomy by reusing the H5 matrix.
  - Files: analysis module/CLI, report schema, tests.
  - Tests: exhaustive/disjoint taxonomy, transition fixtures, lineage joins.
  - Acceptance: primary state counts, teacher/student cross-tabs, continuous
    wrong-confidence/margin trend, cross-seed overlap, and oracle headroom.

- [ ] M4 -- decide schedule control and at most two v2 routes.
  - Schedule control requires a distinct protocol and epoch-79 parent.  Tests
    must match an uninterrupted delayed scheduler reference and allow no other
    checkpoint-state delta.
  - Candidate routing starts with student high-risk plus teacher wrong -> turn
    off adversarial KD and use hard-label/true-label anchoring.  Teacher-correct
    samples retain ordinary KD; stronger KD is not assumed beneficial.
  - Acceptance: design and Go/No-Go thresholds frozen before any unseen seed.

- [ ] M5 -- launch one blinded Bartoldson confirmation parent/screen only after
  M1--M4 and review.  Start Chen only as the predeclared no-harm check.

## Agent and review budget

One completed Sol planning pass produced a single accept/modify/reject table.
Use one Terra writer for M1/M2.  Run one consolidated scientific review only
after the analysis delta and real input report are stable.  No monitoring
agent, repeated review, or GPU job is needed for M0/M1.

## Test plan

- Focused unit tests for the new analysis contracts.
- `scripts/verify.py --changed` once per coherent milestone; cached passes are
  accepted.
- One read-only execution on actual L1--L4 exports after exact input inventory.
- GPU replay only for fields absent from checkpoint state (notably student
  clean correctness), batched once for H5/H4a.
- Production training, official test, and AutoAttack are deferred.

## Risks and mitigations

- Leakage: enforce feature/outcome epoch fields and reject future columns.
- Development reuse: label L1--L4 as development; reserve unseen seed for
  confirmation after freezing.
- Outcome ambiguity: keep train-state outcomes distinct from validation/test
  PGD metrics.
- Schedule confounding: never relabel an already-decayed epoch-99 state as a
  delayed-first-decay control.
- Expensive replay: reuse one hash-bound matrix and transfer checkpoint bytes
  with selective rsync when cheaper than remote recomputation.
- Rare teacher-wrong groups: report mass/headroom before spending a long run.

## Progress log

- 2026-08-02: One Sol planning pass accepted H5 overlap, modified H5-Early and
  H4a, rejected immediate seed-3 launch, and rejected the proposed epoch-99
  schedule-control launch.  Local PyTorch reproduction confirmed 100 scheduler
  steps yield `last_epoch=100` and LR `0.01` for milestones `[100,150]`.
- 2026-08-02: L1/L2 local checkpoint bytes were verified at epochs
  `39/59/79/99/104/109/199`; each has 45,000 format-v3 state records.  Their
  scheduler/LR states are respectively `40/0.1`, `60/0.1`, `80/0.1`,
  `100/0.01`, `105/0.01`, `110/0.01`, and `200/0.001`.  L3/L4 manifests are
  synchronized locally but periodic checkpoint bytes remain on Ferret.  After
  the extraction API is fixed, generate compact state artifacts remotely and
  selectively rsync them instead of copying every checkpoint.
- 2026-08-02: M1 input contract was corrected during implementation review.
  The existing seed-0 fixed predictor was trained on standardized common-PGD
  replay panels, while exact online state frequency/EMA has different
  semantics.  Fixed-versus-rank H5 comparison will therefore use one replay
  panel for both scores; no cross-domain score comparison will be reported.
- 2026-08-02: The first feature-only live smoke exposed a historical-config
  compatibility boundary before any GPU compute: the L1 logging-only metadata
  and later teacher-registry storage metadata were not accepted by the current
  strict runtime schema.  Feature-only replay now performs a fail-closed,
  in-memory migration after checking the original config bytes/hash, objective,
  threat model, normalization, registry ID, and checkpoint hash.  Full replay
  and training remain strict.  Focused verification passed (`76 passed`); the
  L1/L2 exact online epoch-99/199 state exports were generated with 45,000 rows
  each.  GPU feature replay will be restarted from the clean M1 commit.
- 2026-08-02: An observation-first audit stopped the first partial GPU replay
  after only L1/L2/L4 `2/9/3` cached checkpoints.  Completing it would have
  forced a second H4a replay because teacher correctness/wrong-confidence and
  student clean/robust state were not persisted.  Observation schema v2 now
  records detached teacher clean/adversarial probability primitives,
  clean-to-adversarial deltas, and student-clean primitives during the same
  common-PGD replay.  The frozen H5 feature-panel columns/formulas do not
  change; cache/source identity prevents schema-v1 reuse.  M2 focused tests
  passed (`77 passed`), and the impact-selected T0/T1 suites passed
  (`6 + 39 + 33`).  No official test or new training was consumed.
- 2026-08-02: M3 implementation adds a read-only schema-v2 H4a taxonomy and
  blinded stable-ID manifest.  Early taxonomy uses only feature-domain
  anchor-to-99 transitions at epochs 39/59/79; late taxonomy uses only the
  independent outcome-domain 99-to-199 transitions.  It reports exhaustive
  primary groups, continuous teacher confidence/margin summaries, cross-tabs,
  class/ID hashes, same-panel endpoint-error coverage, and within-teacher
  cross-seed Jaccard.  No route or threshold is defined.  Focused unit tests
  passed (`38 passed`); real L1--L4 reports remain pending schema-v2 replay.

## Completion report

Pending.
