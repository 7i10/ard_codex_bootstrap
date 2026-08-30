# ERT / RSLAD History-Minimality and First History-Conditioned Ordering

## Status

- Owner: Codex root
- Branch / base SHA: `62faf1b7c382a02c58140290f7090803cba0e0e1`
- Current milestone: Stage A complete; Stage B blocked by attack RNG coupling
- Last updated: 2026-08-31

## Goal

Freeze the smallest preregistered subset of the existing P3 History features
using development seeds only, then run the first ordering-only intervention from
the exact I100 epoch-99 parents. Every training sample must appear exactly once
per epoch; no loss, augmentation, attack, optimizer, scheduler, or exposure
count may change. Complete the registered CE-PGD20 endpoints and stop without
launching confirmation seeds.

## Non-goals

No new seed, confirmation training, weighting, oversampling, curriculum,
augmentation or attack change, threshold sweep, official test, AutoAttack,
additional ordering variant, or automatic promotion beyond this development
screen.

## Existing state

The five-seed Student History predictive-validity analysis is recorded at
`62faf1b`. The frozen global method is I100 (CropShift epochs 0--99 followed by
IDBH_WEAK epochs 100--199). Existing epoch-99 I100/CropShift parents and
format-v3 sample state must be hash-verified before a fork. The repository has a
reusable multi-GPU orchestration skill for manifest-driven detached DAG runs.

## Scientific contracts affected

- Add an explicit ordering policy axis while retaining the frozen I100 loss,
  augmentation, attack, optimizer, scheduler, and teacher contracts.
- History scores use only epoch-boundary state available before the ordered
  epoch; no future observations or outcome labels enter the selector.
- Every epoch is a permutation of the canonical 45,000 train IDs exactly once.
- Source-keyed augmentation and attack RNG must remain order-independent; if
  attack randomness is order-coupled, production is blocked.
- Checkpoint/resume must preserve sampler/order identity and sample state.

## Decisions

- Minimality candidates are fixed to H1--H9 in the user protocol. Selection is
  dev-only with Ridge alpha 1.0, dev-only standardization, and the preregistered
  0.01 mean / 0.015 per-seed tolerance to H9.
- Use epoch-99 I100 parents and start ordering at epoch 100. CONTROL and
  HISTORY_BALANCED are fresh suffixes from the same parent per seed.
- HISTORY_BALANCED uses fixed 20/60/20 HIGH/MID/LOW risk strata, stable-ID tie
  breaks, per-stratum order substreams, and HIGH/MID/MID/LOW/MID interleaving.
- Use `multi-gpu-experiment-orchestrator` for production execution and endpoint
  chaining; do not keep the Codex session polling long-running jobs.

## Milestones

- [x] M0 reconcile repository and create immutable manifest/plan
- [x] M1 implement/read-only P3 minimality analysis and freeze artifact
- [x] M2 audit RNG coupling (ordering sampler intentionally not launched)
- [ ] M3 verify exact parents and run bounded CONTROL/HISTORY canaries (blocked)
- [ ] M4 launch four suffixes through the orchestration DAG and validate telemetry (blocked)
- [ ] M5 chain CE-PGD20 endpoints, aggregate results, and write report (blocked)
- [ ] M6 commit/push the verified milestone; stop before confirmation seeds

## Agent and review budget

No subagent is required; one owning writer is sufficient because the ordering
API and scientific contract must be integrated centrally. Use one consolidated
review after implementation and canary evidence is stable.

## Test plan

- Cached existing I100 parent and checkpoint lineage checks.
- New CPU tests for H1--H9 selection, no-confirm leakage, frozen coefficient
  hashes, permutation coverage, deterministic ties, 20/60/20 interleaving,
  order-seed reproducibility, and future-leakage rejection.
- One bounded real checkpoint/public-CLI smoke before any production suffix,
  plus focused changed tests and `scripts/verify.py --changed`.
- Production telemetry validator must require exactly epochs 100--199,
  finite metrics, parent/source/seed identity, and required history snapshots.

## Risks and mitigations

- Attack random-start coupling to DataLoader order: prove source-keyed or stop.
- Resume/config migration drift: retain exact parent lineage and verify model,
  optimizer, scheduler, scaler, RNG, sampler, and sample-state identity.
- GPU/path conflicts: resolve host profiles and reserve UUIDs through the
  orchestrator; never kill external processes.
- W&B storage pressure: keep metrics/lineage only and retain large artifacts
  locally under the existing policy.
- Long-running latency: use detached marker-driven DAG execution and bounded
  launch checks, not Codex polling.

## Progress log

- 2026-08-31: attached protocol read; current HEAD is `62faf1b` and tree is
  clean. Existing `EpochShuffleSampler` and source-keyed augmentation were
  inspected. Exact implementation and RNG coupling audit remain pending.
- 2026-08-31: H1--H9 minimality replay completed from existing state
  checkpoints. H2 (`margin_ema` alone) is the smallest candidate meeting the
  preregistered dev tolerances; confirmation results are descriptive only.
- 2026-08-31: exact dev I100 epoch-99 parents verified at `360910a8...` and
  `bb0c7c1a...`. The read-only attack RNG audit reproduced batch-position
  coupling, so Stage B is fail-closed before canary or GPU launch.

## Completion report

Stage A and the pre-launch gates are complete.  H2 (`margin_ema`) was frozen
with predictor hash `b6be23e7eaa31dd300ed21efb84746bba2270a4a044096c86d3be035c8abebbd`.
The existing PGD random-start generator is seeded per batch/global step and
fills a batched tensor position-wise; reordering changes random-start values
for fixed source IDs.  Therefore the requested ordering-only causal run is
blocked pending human approval of a new source-keyed attack RNG contract and
paired-control redesign.  No production jobs or endpoints were launched.
