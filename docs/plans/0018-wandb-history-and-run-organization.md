# W&B history and run organization

## Status

- Owner: main thread; core implementation: Terra
- Branch / base SHA: `cleanup/observability-first` / `22e5fa3`
- Current milestone: complete
- Last updated: 2026-08-02

## Goal

Make robust-overfitting analysis fast and reproducible without moving or deleting
existing W&B runs: future train runs publish an exact epoch-metrics Parquet artifact
and terminal RO summaries; analysis reads explicit run cohorts, uses a local success
cache, and validates legacy history coverage; an idempotent tool classifies existing
runs with additive tags.

## Non-goals

- Do not move runs between projects or alter artifact lineage.
- Do not delete, archive, or rewrite historical run data.
- Do not change training, checkpoint selection, attacks, or numerical objectives.
- Do not add a background W&B polling service.

## Existing state

- Rank zero already logs one scalar row per epoch and stores local metrics JSONL.
- Final summaries contain best/last and robust-overfit gap, but not late-window
  means, normalized AUC, or slope.
- Sample statistics are already written as genuine Parquet artifacts.
- Historical W&B `scan_history` is unreliable for these runs; bounded
  `run.history` succeeds but must be coverage-checked and cached.

## Scientific contracts affected

- Tracking and artifact lineage only. Threat model, gradients, normalization,
  training schedule, selection attack, and checkpoint semantics are unchanged.
- RO summaries use `val_pgd_accuracy` from the existing validation protocol and
  must not be presented as official-test accuracy.
- Resume must merge exact epoch rows without silently overwriting conflicting data.

## Decisions

- Preserve artifact-bearing runs in their current project; organize with additive
  tags and saved-view-compatible metadata.
- Exact run IDs are the default analysis input. Whole-project history scans are
  prohibited by the tool interface.
- Future runs use local exact epoch records as the canonical trajectory artifact.
  Legacy W&B history is an explicitly marked fallback and is accepted only when
  epoch coverage is unique and complete for the requested range.
- Semantic classification is an explicit registry and dry-run by default; unknown
  runs remain untouched.

## Milestones

- [x] M0: freeze schemas and cohorts in this plan.
- [x] M1: atomic epoch-metric store, Parquet artifact, and terminal RO summaries.
- [x] M2: exact-ID W&B history analysis with success cache and legacy validation.
- [x] M3: additive, idempotent historical tagger with dry-run/apply modes.
- [x] M4: focused tests, impact-selected verification, one consolidated review,
      and documentation.

Files/modules: `src/ard/analysis/`, `src/ard/cli/train.py`, tracking/config only if
needed for future structural tags, `scripts/`, focused tests, and W&B docs.

Implementation owner: one Terra writer owns all source/tests to avoid overlapping
edits. The main thread owns this plan, documentation integration, external dry-run,
and final commit.

Acceptance criteria:

- A fresh or resumed run creates one full-coverage epoch trajectory artifact.
- Conflicting duplicate epoch rows fail; identical rows are idempotent.
- Canonical 200-epoch summaries expose late means/AUC/slope with tested formulas.
- Repeated unchanged history analysis reports a cached pass and performs no W&B
  history request; `--force` bypasses it.
- Historical tag dry-run is deterministic; apply preserves existing tags and a
  second apply is a no-op.

Rollback point: commit `22e5fa3`. Planned commit: `tracking: cache exact epoch
history and organize wandb runs`.

## Agent and review budget

- No new planning pass: the main thread owns the bounded plan.
- One Terra implementation pass.
- No Luna pass; the documentation delta is small and coupled to the schema.
- One consolidated scientific review after tests; re-review only for a P0/P1.

## Test plan

- Unit: summary math, atomic resume merge/conflict/coverage, cache hit/force,
  cohort validation, tag preservation/idempotence.
- Offline integration: epoch Parquet/artifact/summary from a tiny train run.
- `scripts/verify.py --changed` after focused tests.
- No GPU, production training, AutoAttack, or live-network unit tests.
- Live W&B mutation only after a successful exact-delta dry-run.

## Risks and mitigations

- Sampled legacy history: validate exact unique epoch coverage and label source.
- Resume gaps from pre-feature checkpoints: report partial coverage; never label it
  complete or synthesize missing epochs.
- Tagging the wrong run: exact registry, additive changes, dry-run, unknown no-op.
- API latency/rate limits: summary-first requests, explicit cohort, disk cache.
- Test leakage: these summaries describe validation trajectories and do not choose
  future methods from official-test outcomes.

## Progress log

- 2026-08-02: Plan frozen. Existing runs will not be moved or deleted because
  W&B artifacts do not automatically follow project moves and lineage is primary.
- 2026-08-02: Applied additive tags to 50 exact run IDs after a successful
  dry-run. The second dry-run reported `changed: false` for every entry.
- 2026-08-02: The first live mixed-cohort analysis exposed an incorrect global
  epoch-zero assumption. Read-only diagnosis proved 16 full trajectories at
  `0..199` and five H4 continuations at `100..199`; the per-run range contract
  now validates both history and artifact sources fail-closed.
- 2026-08-02: Explicit legacy source selection reduced the 21-run live fetch
  from about 62 seconds to about 11–14 seconds. A valid finished-run cache hit
  completed in 2.37 seconds without W&B API initialization.
- 2026-08-02: Consolidated review found and closed two P1 issues: missing
  `global_step` caused resume row conflicts, and the cache key omitted the
  epoch-summary implementation. Focused regressions cover both.

## Completion report

- Implemented atomic exact epoch history, Parquet publication, canonical RO
  summaries, explicit mixed-lineage W&B analysis, and additive run tagging.
- Focused final gate: Ruff format/check and mypy passed; epoch/history/tracking
  tests passed (`53 passed` in the final writer gate; main integration subset
  also reported `49 passed`).
- Live W&B: 21/21 trajectory ranges passed; optimized forced fetch exited zero;
  unchanged rerun returned `cached: true`; 50 tag entries are idempotent.
- `scripts/verify.py --changed --non-scientific --dry-run` was executed. Its
  path-only mapping still selects attack/objective/DDP/GPU tests because
  `train.py` changed, although those contracts did not. The broad command was
  intentionally not run as ceremonial duplication; resume/tracking/synthetic
  focused tests were selected instead.
- No GPU training, attack evaluation, AutoAttack, run deletion, project move,
  or artifact rewrite occurred.
