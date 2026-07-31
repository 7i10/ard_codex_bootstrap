# Common-trajectory entropy versus student signal comparison

## Status

- Owner: main thread; Terra owns the analysis implementation
- Branch / base SHA: `master` / `6f6f356`
- Current milestone: M1 existing-artifact replay
- Last updated: 2026-07-31

## Goal

Compare teacher entropy and student robust-margin history as predictors of later
robust forgetting on one common, non-intervened RSLAD trajectory. Produce a
fail-closed proxy result from existing periodic checkpoints, then provide an
exact logging-only RSLAD path whose observations do not affect training loss.

## Non-goals

- Do not infer intervention efficacy from predictive association.
- Do not treat checkpoint-panel forgetting as exact online forgetting.
- Do not use official-test outcomes, AutoAttack, Entropy/Student intervention
  trajectories, or a new training run in the artifact-replay stage.
- Do not start a long logging-only run until the implementation, parity gate,
  current GPU queue, and proxy result have been reviewed.

## Existing state

- Canonical seed-0 Chen/RSLAD and Bartoldson/RSLAD runs publish 40 periodic
  `last` checkpoints at epochs `4, 9, ..., 199`.
- RSLAD checkpoints have no `SampleStateStore`; exact online EMA and per-epoch
  forgetting cannot be recovered.
- `teacher_risk_replay.py` already implements strict checkpoint loading, raw
  unaugmented train-partition reconstruction, exact KL PGD-10 replay, stable
  sample IDs, teacher freeze, and lineage checks for one checkpoint.
- Existing Student/Joint formal audits are intervention-confounded and are not
  used for the strict Entropy-versus-Student comparison.
- The worktree also contains an unrelated untracked user document,
  `docs/ARD_RESEARCH_ISSEUES_AND_PROPOSALS.md`; it is outside this plan.

## Scientific contracts affected

- Pixel-space Linf `8/255`, step `2/255`, PGD-10, random start, KL
  `teacher_clean`, temperature 1, student/teacher eval during replay.
- Both signals must be computed from the same adversarial example.
- Feature and future-outcome attack seed panels must be independent.
- Stable source IDs and the 45,000-sample CIFAR-10 train partition are required;
  official test samples are forbidden.
- Source checkpoints, config, teacher, attack, dataset, execution profile, Git
  source, and output bytes are hash-bound.

## Decisions

- Strict scientific conclusion requires a new loss-identical logging-only RSLAD
  trajectory. Existing artifacts yield only a common-trajectory replay proxy.
- Proxy cutoff is epoch 99. Feature checkpoints are epochs `4..99`; outcome
  checkpoints are epochs `99..199`.
- The primary student feature is panel-EMA margin with
  `beta_panel = 0.9^5`; epoch-99 instantaneous margin is a sensitivity result.
- The teacher feature is normalized Shannon entropy `H/log(10)`. The
  batch-relative SAAD weight is not a sample-intrinsic signal and is excluded.
- Primary outcome is any observed robust correct-to-wrong transition after
  epoch 99. Transition count and final robust error are secondary.
- Fixed true-class-stratified 80/20 hash split and identical one-variable
  logistic models are used. Report AUROC, AUPRC, prevalence, and log-loss.
- Paired class-stratified bootstrap uses 1,000 held-out resamples and reports
  Student-minus-Entropy intervals for AUROC, AUPRC, and log-loss. These are
  sample-conditional, not training-seed uncertainty.

## Milestones

- [x] M0: fail-closed 40-checkpoint inventory and frozen replay protocol.
  - Files: new temporal replay analysis/CLI and focused tests.
  - Owner: Terra, because replay semantics and lineage checks are core
    scientific implementation.
  - Acceptance: exact epoch panel, hashes, config/run/Git/world-size identity,
    attack identity, and independent seed panels are validated.
  - Commit: `feat: add common-trajectory RSLAD signal replay`.
- [ ] M1: generate per-checkpoint feature/outcome Parquet on Chen and
  Bartoldson RSLAD.
  - Tests: one checkpoint/one bounded batch GPU smoke; full replay is T5-like
    analysis execution, not an automated test.
  - Acceptance: 45,000 unique stable IDs per checkpoint and canonical scalar
    lineage report; no source artifact mutation.
- [ ] M2: paired predictive audit.
  - Files: analysis/report module, CLI integration, focused deterministic unit
    tests, `docs/SIGNAL_AUDIT.md`.
  - Acceptance: teacher-specific complete Entropy/Student metric table and
    paired intervals from the same held-out samples. Implementation and
    deterministic unit coverage are complete; the teacher-specific table is
    pending M1 artifact replay.
  - Commit: `analysis: compare entropy and margin on RSLAD trajectories`.
- [x] M3: exact logging-only capability.
  - Files: trainer diagnostics/state/config plus parity tests and explicit
    logging-only configs after proxy review.
  - Acceptance: bounded RSLAD and logging-only runs have exact model,
    optimizer, scheduler, RNG, and checkpoint parity while recording both
    pre-update signals from the real augmented training trajectory.
  - Commit: separate from replay to keep the publication core reviewable.
  - Implementation and bounded parity evidence are tracked in
    `0014-future-method-preparation.md`; no long logging-only run starts before
    the proxy review.
- [ ] M4: one consolidated scientific review and result documentation.

## Agent and review budget

One planning pass has completed. Terra is the sole implementation writer for
each scientific delta. Use one consolidated scientific review after focused
tests; re-review only an actual P0/P1 fix delta. Luna is unnecessary unless
multiple frozen configs are later generated mechanically. GPU jobs run as shell
processes, not as reasoning agents.

## Test plan

- New T1/T2 unit tests: exact panel EMA, transition counts, stable-ID joins,
  missing/duplicate epoch rejection, seed-panel independence, split/order
  invariance, paired metric-delta signs, and canonical output bytes.
- Reuse attack contracts rather than rerunning unrelated attack suites.
- New bounded CUDA smoke: one checkpoint and one batch only.
- Full 82-checkpoint replay is intentionally outside `scripts/verify.py`.
- No production training or AutoAttack is selected by this plan.

## Risks and mitigations

- Five-epoch sampling misses within-panel forgetting. Name the outcome
  `checkpoint-panel forgetting` and reserve exact claims for logging-only runs.
- Raw/eval replay differs from augmented, pre-update train-mode observations.
  Keep proxy and exact result namespaces separate.
- Shared random starts could leak attack noise into both feature and outcome.
  Use distinct deterministic seed domains.
- Seed-0 only cannot measure training-seed uncertainty. Never interpret the
  sample bootstrap as a seed confidence interval.
- Bartoldson replay is expensive. Benchmark one bounded batch, then run the two
  teachers on separate free GPUs; do not occupy agents to monitor them.

## Progress log

- 2026-07-31: planning fixed the proxy/exact boundary, epoch-99 cutoff,
  independent seed panels, panel-EMA definition, held-out metrics, and minimum
  acceptance tests. No new training or replay job was started.
- 2026-07-31: implemented and focused-tested the isolated common-trajectory
  replay/audit primitives. The new path rejects any non-exact `4,9,...,199`
  periodic-last panel, binds raw saved resolved-config bytes, checkpoint/Git/
  world-size/attack identity, uses independent fixed hashed feature/outcome
  seed panels (also fixed across checkpoints within a domain), and emits
  atomic hash-verified checkpoint caches, genuine Parquet,
  and canonical JSON lineage. It records epoch-99 entropy, panel-EMA and
  instantaneous-margin sensitivity, checkpoint-panel forgetting, identical
  univariate held-out logistic metrics, and paired Student-minus-Entropy CIs.
  The latest review-delta focused command reported `29 passed`, covering panel
  math, joins, epoch rejection, fixed-domain seeds, all 14 attack-identity
  fields, canonical-config mapping versus file-byte digests, strict-loader
  lineage, tracked semantic source paths, cache reuse/rejection and
  partial-cache recomputation, output determinism, and metric signs. No GPU
  replay or new training was started.
- 2026-07-31: the combined focused integration command reported `66 passed in
  7.22s`; Ruff format/check and `git diff --check` passed. The consolidated
  review and its two bounded P1 fix-delta reviews ended with no remaining
  P0/P1. M4 remains open only for the teacher-specific result documentation.

## Completion report

M0 and the M2 analysis/report primitives are implemented and covered by focused
tests. The bounded CUDA smoke, full Chen/Bartoldson checkpoint replay,
teacher-specific predictive tables, and exact logging-only M3 remain pending;
no operational replay or new training has been run.
