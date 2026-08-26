# ERT RSLAD baseline RNG-source decomposition

## Status

- Owner: Codex / repository owner
- Branch / implementation SHA: `master` / `f6f8e11`
- Current milestone: M1 implementation and preregistration complete; GPU campaign pending
- Last updated: 2026-08-22

## Goal

Provide a reproducible, fail-closed protocol that separates PGD random-start,
data-side stochasticity, attack/data interaction, and residual nondeterminism in
the exact L2/L4 epoch-79 BASE RSLAD continuation, then stop after the registered
16 trajectories and fixed-seed endpoint evaluations.

## Non-goals

- No Teacher-adaptive margin treatment, CleanCE, floor/cap, lambda sweep,
  target smoothing, dynamic routing, new Teacher, architecture, or optimizer.
- No official test, AutoAttack, additional PGD restart, or five-seed campaign.
- No automatic stabilization experiment after the decomposition.
- No population-level inference from two perturbations or two REF repeats.

## Existing state

- HEAD and `origin/master` were both at `13a5f60` during reconciliation; the
  first sandboxed `git fetch origin` could not write `.git/FETCH_HEAD`.
- The canonical runtime already had independent local generators for PGD,
  sampler, and source-keyed augmentation, but Stage A exposed only one legacy
  `continuation_seed` and DataLoader worker ownership was implicit.
- L2 parent is the exact `ad43d72…` checkpoint. The recovered L4 fork parent is
  the exact `026a36…` checkpoint; the historical generic `9b51…` binding is not
  substituted.
- Parent config/checkpoint paths and all seed assignments are frozen in
  `docs/experiments/ert_rslad_rng_source_decomposition_v1.json`.

## Scientific contracts affected

- RNG lineage and continuation identity gain explicit `data_seed`,
  `attack_seed`, and `other_seed` fields.
- DataLoader worker seeds are now explicitly descended from a data-owned CPU
  generator. Existing sampler and augmentation distributions are unchanged.
- `EpochShuffleSampler.reseed` and source transform reseeding are permitted only
  after exact parent `load_checkpoint` and before epoch 80 iteration.
- PGD threat identity, RSLAD objective, normalization, model modes, checkpoint
  state, selection attack, and evaluation attack are unchanged.
- Parent SHA is optionally required by the Stage A CLI and is fail-closed when
  supplied.

## Decisions

- Use `RNGSourceSeeds` rather than adding three global schema fields: this is a
  continuation-specific protocol and avoids changing every existing config.
- Keep `--continuation-seed` as a compatibility alias for the historical
  attack/global reseed while preserving parent data order/augmentation; new
  decomposition runs must provide all three explicit flags together.
- Treat sampler order, source-keyed augmentation, and DataLoader worker base
  seeds as one data stream. Split-versus-data composition remains fixed to the
  parent and is not an experimental factor.
- Use the actual public `LinfPGD.generate`, sampler, `EpochSourceTransform`,
  and DataLoader in the CPU canary. Compare raw PGD draw hashes rather than
  adversarial tensor hashes, because data changes legitimately change attack
  inputs.
- Keep endpoint evaluation seed fixed at `0` for every trajectory and run it in
  a separate saved-checkpoint process.
- Do not invent a residual-nondeterminism threshold. Report REF2 minus REF1 and
  compare it descriptively with source perturbations.
- Run both teachers on Hamster for this campaign. This keeps all 16 trajectories
  in one controlled environment and follows the current operational rule that
  Hamster is the preferred faster host; Ferret is excluded.

## Milestones

- [x] M0 audit and reconciliation
  - Files: prior ERT reports/plans, Stage A runtime, attack/data/tracking paths.
  - Owner: main thread; no subagent needed because all evidence is local and
    the repository has no callable research-planner role.
  - Acceptance: parent identities, source ownership, and forbidden changes are
    recorded in the machine audit and this plan.
  - Rollback: no source mutation.
  - Commit boundary: combined with M1 after focused tests.

- [x] M1 independent stream plumbing and canary
  - Files: `src/ard/analysis/ert_rslad_rng_sources.py`, Stage A runtime/CLI,
    `src/ard/data/{rng.py,indexed.py,datasets.py}`, train CLI, focused tests.
  - Owner: main thread; the change is coupled to checkpoint restore and cannot
    be safely split across concurrent writers.
  - Acceptance: explicit triplet is in arm hash, resolved config, fork lineage,
    and result; parent SHA guard fails closed; canary passes all four isolation
    assertions; legacy continuation behavior remains supported.
  - Rollback: revert only the new stream helper and call-site plumbing.
  - Commit boundary: one cohesive implementation commit after review.

- [x] M2 preregistration and report scaffold
  - Files: audit JSON, experiment JSON, this plan, human report.
  - Owner: main thread; deterministic documentation synchronization.
  - Acceptance: seed registry and eight arms per teacher are outcome-independent;
    endpoint/derived metric schemas and stop rule are explicit.
  - Rollback: remove only the new experiment documentation.
  - Commit boundary: same cohesive milestone commit.

- [x] M3 bounded production preflight and GPU canary
  - Files: host/job metadata and canary output outside Git, plus local run
    bundles.
  - Owner: repository owner; requires host/GPU and W&B authority.
  - Acceptance: clean immutable source SHA, exact parent hashes, Hamster-only
    production identity, and canary pass on the production environment.
  - Rollback: do not launch production arms if any gate fails.
  - Commit boundary: none; runtime artifacts are not committed.

- [x] M4 16 trajectories and endpoint collection
  - Files: local run bundles and endpoint results outside Git; update the
    registered JSON/report files after validation.
  - Owner: repository owner / one campaign owner.
  - Acceptance: all L2/L4 arms reached epoch 94 from exact parents, retained
    84/89/94, evaluated fixed-seed CE-PGD20, and had complete lineage.
  - Rollback: exclude incomplete/failed arms; never overwrite a valid arm.
  - Commit boundary: one results/report commit after point-estimate checks.

- [x] M5 point decomposition and human review stop
  - Files: registered machine artifact and human report.
  - Owner: main thread for deterministic aggregation; one consolidated
    scientific review after evidence is stable.
  - Acceptance: the result report answers REF residual, attack/data sensitivity,
    interaction, L2/L4 ranking, trajectory divergence, and the stop decision
    without population claims.
  - Rollback: mark result incomplete rather than filling missing values.
  - Commit boundary: results-only commit.

## Agent and review budget

One owning writer is sufficient. No parallel writer is allowed on the runtime
or experiment artifact. A single consolidated scientific review is required
after M4/M5 evidence, not before GPU execution. No bug-investigator role is
needed because the initial issue is understood and the focused failure was a
mechanical canary hashing bug that was fixed locally.

## Test plan

- [x] `PYTHONPATH=src ...pytest -q tests/unit/test_ert_rslad_rng_sources.py tests/unit/test_ert_stage_a_runtime.py`
  (`11 passed`).
- [x] `PYTHONPATH=src ...python scripts/ert_rslad_rng_source_canary.py --batches 1`
  (`status=passed`; the two-batch command is the production gate).
- [ ] Run `scripts/verify.py --changed` with the repository environment after
  the final diff is stable.
- [ ] Run lint/mypy and the impact-selected DataLoader/checkpoint-resume tests.
- [ ] M3 GPU canary is scientific/operational, not part of the automated suite.
- [ ] M4/M5 are T5 production work and intentionally deferred in this coding
  turn.

## Risks and mitigations

- Scientific drift: exact attack and RSLAD contracts are embedded in the audit
  and experiment JSON; no objective code changed.
- Resume drift: explicit reseeding occurs only after `load_checkpoint`; parent
  config hash and optional parent byte SHA are checked before training.
- Data coupling: sampler, augmentation, and worker generator are separately
  instrumented and checked by the canary.
- Hardware nondeterminism: deterministic flags and same-GPU REF repeats are
  required; no arbitrary cutoff hides a nonzero REF difference.
- DDP: sampler identity remains world-size/rank-aware and the decomposition
  must keep each teacher within one host family; production identity includes
  world size and GPU metadata.
- W&B: production tracking remains mandatory and rank-zero-only; the local run
  bundle remains authoritative.
- Artifact contamination: outputs, checkpoints, W&B data, and canary results
  are not committed.

## Progress log

- 2026-08-22: Reconciled HEAD/origin at `13a5f60`; exact L2/L4 parent files
  verified locally. Initial sandbox fetch could not write `.git/FETCH_HEAD`.
- 2026-08-22: Added independent seed plumbing, explicit DataLoader worker
  ownership, parent SHA guard, and source-isolation canary.
- 2026-08-22: Focused tests passed `11/11`; CPU canary passed. No GPU
  trajectory or endpoint has been launched.
- 2026-08-22: Revised host assignment to Hamster-only so both teachers and all
  matched arms share one controlled environment; Ferret is excluded.
- 2026-08-22: Pinned the reviewed implementation source to `f6f8e11`; the
  subsequent metadata-only pin update does not alter runtime code.
- 2026-08-23: Hamster-only production completed all 16 trajectories through
  epoch 94. REF1/REF2, attack-only, data-only, and both-source arms were
  collected under source SHA `09e627e`.
- 2026-08-23: Independent CE-PGD20 endpoint evaluation completed 96/96 records
  for epochs 84/89/94 on train and fixed validation splits. The endpoint matrix
  passed the complete attack, row-count, source-SHA, and contract checks.
- 2026-08-24: Deterministic point aggregation and source decomposition were
  written to `docs/experiments/ert_rslad_rng_source_decomposition_v1_results.json`
  and `docs/ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md`. The preregistered stop
  boundary was reached; no stabilization run was started.

## Completion report

Complete through M5. The 16 trajectories and 96 endpoint records are present,
the point decomposition is recorded in the result artifact/report, and the
campaign is stopped at the preregistered boundary pending human review.
