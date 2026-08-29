# ERT / RSLAD unseen-seed confirmation and full-training stochasticity

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `9b96c1afc59f8618aa5f46aa9f3f2f93c7ce5941`
- Current milestone: M0 in progress
- Last updated: 2026-08-29

## Goal

Confirm the frozen `I100` policy (CROPSHIFT epochs 0--99, IDBH_WEAK epochs
100--199) against CROPSHIFT on three outcome-independent confirmation seed
bundles, then descriptively characterize global and sample-level stochasticity
over the two development seeds plus those three confirmation seeds.

## Non-goals

No timing or augmentation search, loss or optimizer change, History/Ordering
intervention, seed shopping, official test, AutoAttack, or population-level
significance claim. Development seeds cannot rescue a failed three-seed
confirmation result.

## Existing state

`I100` is the frozen incumbent from the completed finite timing screen. The
canonical stage-wise config and source-keyed augmentation implementation are
already present. The main train CLI accepts per-field seed overrides and full
epoch-boundary resume, but no unseen-seed registry or DAG launcher exists yet.
Hamster has two idle RTX 4090 GPUs but only about 23 GB free; Ferret has three
idle RTX 4090 GPUs and about 878 GB free. W&B production policy is
metrics-only; model checkpoints, row artifacts, and run bundles remain local.

## Scientific contracts affected

- Fixed CIFAR-10 45k/5k split, Chen2021LTD WRN34-10 teacher, canonical RSLAD,
  and KL-PGD10 training attack remain unchanged.
- Confirmation comparison is I100 versus CROPSHIFT on exactly three unseen
  seed bundles. BASE is a secondary comparison.
- CROP_PREFIX (0--99) is computed once per confirmation seed; CROP_SUFFIX and
  I100_SUFFIX fork from its exact epoch-99 end state. BASE is an independent
  200-epoch run.
- Exact resume restores model, optimizer, scheduler, scaler, Python/NumPy/
  Torch/CUDA RNG, sampler/order, augmentation state, attack contract, and
  O(1) sample history.
- Endpoint is independent eval-mode CE-PGD20 on the fixed internal validation
  split at epochs 49, 99, 149, and 199. Official test and AutoAttack remain
  sealed.
- Confirmation seed IDs and component seeds are frozen before any outcome is
  observed; retries cannot change a scientific seed.

## Decisions

- Use deterministic bundle labels `confirm-a`, `confirm-b`, and `confirm-c`,
  selected because repository and local run inventories contain no prior use
  records for these labels or their derived component seeds. Each component
  seed is derived from the preregistration-safe SHA-256 `derive_seed` contract
  and recorded in the registry; no literal `seed3/4/5` choice is made.
- Preserve the current config's explicit split seed and pass the frozen
  component seeds through CLI overrides. Do not silently collapse distinct
  component seeds to one integer.
- Use Ferret for the longer BASE/prefix jobs and Hamster for suffix jobs when
  the measured makespan is lower. Transfer only hash-verified source/checkpoint
  material with `rsync`; record host, GPU UUID, and NUMA binding.
- Write only registered checkpoints and compact validation rows. Never upload
  model or run-bundle artifacts to W&B.
- After stable launch (finite first batches, manifest/run ID, correct output
  paths), stop active polling. Completion aggregation is a later turn.

## Milestones

- [ ] M0 reconcile source, environment, seed-use evidence, and storage
- [ ] M1 freeze seed registry and DAG/lineage plan before training
- [ ] M2 add/verify minimal orchestration and sample-row instrumentation
- [ ] M3 run one resume/parity canary per host, then launch BASE/prefix jobs
- [ ] M4 launch dependent suffixes from exact epoch-99 parents
- [ ] M5 run registered CE-PGD20 endpoints and collect compact rows
- [ ] M6 aggregate confirmation and five-seed descriptive stochasticity
- [ ] M7 write reports/artifacts, review the stable delta, commit and push

## Agent and review budget

No subagent is needed. The root agent owns one implementation path. Use one
focused scientific review only after artifacts and metrics are stable; do not
poll production jobs or invoke per-run reviewers.

## Test plan

- JSON/seed registry schema and deterministic derivation tests (CPU).
- `scripts/verify.py --changed` after the implementation delta.
- One real checkpoint end-to-end smoke per host covering CLI, resolved config,
  lineage, resume, compact row output, and completion marker.
- Exact CROP_PREFIX to suffix resume parity and switch-boundary assertions.
- CE-PGD20 endpoint identity and row hash checks; expensive production runs
  remain outside automated tests.

## Risks and mitigations

- Storage exhaustion: preflight required bytes; keep only registered local
  checkpoints and metrics-only W&B. Stop before launch if the estimate exceeds
  the host budget.
- Seed leakage or reuse: hash-bound registry with inventory evidence and no
  post-outcome replacement.
- Resume drift: require epoch-99 payload/next-epoch/LR/global-step and all
  state fields before suffix launch.
- Host confounding: balance jobs across measured devices and report hardware
  as nuisance metadata, never as a scientific effect.
- Remote divergence: fetch/pin the exact source SHA on Ferret and use hash-
  verified rsync for code, checkpoints, and compact artifacts.

## Progress log

- 2026-08-29: Reconciled `9b96c1a` with `origin/master`; worktree clean.
  Hamster has two idle RTX 4090s and ~23 GB free; Ferret has three idle RTX
  4090s and ~878 GB free with linger enabled. Existing inventories contain
  development seeds 1/2 only; no unseen registry exists.
- 2026-08-29: Derived three outcome-independent bundle component seeds and
  prepared the registry as the pre-training freeze boundary. Extended the
  stage-wise fork utility to accept registry-derived unseen model seeds and to
  bind the materialized parent Git identity to the clean source that created
  the child. Production has not been launched in this milestone.

## Completion report

To be filled after M7 with the frozen registry hash, launch/lineage records,
confirmation results, five-seed descriptive metrics, tests, review findings,
and any blocked or deferred work.
