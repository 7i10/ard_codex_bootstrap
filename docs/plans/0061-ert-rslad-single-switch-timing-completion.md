# ERT / RSLAD single-switch augmentation timing completion

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `960844f032a171edb0963b743ae1f7de5a2b1c0a`
- Current milestone: M0 reconciliation and parent inventory
- Last updated: 2026-08-29

## Goal

Complete the preregistered final single-switch timing screen for
`IDBH_WEAK`: fresh `I50`, `I75`, and `I125` continuations from exact
CROPSHIFT states, reusing accepted `I100` and `I150` results, then freeze the
single-switch timing branch without starting another timing, augmentation, or
official-evaluation experiment.

## Non-goals

No additional switch epoch, multi-stage schedule, ramp, augmentation tuning,
new seed, BASE/CROPSHIFT rerun, Student-History or ordering intervention,
official test, or AutoAttack.

## Existing state

The accepted incumbent and stage-wise source are already on `master`. The
production source for the earlier stage-wise runs is
`bb68afc0ff505248f84c0263179ec24f0b346bcd`; the teacher is Chen2021LTD
WRN34-10 with the hash recorded in the stage-wise result artifact. Accepted
CROPSHIFT controls are local complete runs for seeds 1/2 with sparse
checkpoints at displayed epochs 49/99/149/199. Existing `I100` and `I150`
suffixes, endpoints, and AUC results are hash-bound in
`docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json` and are
not retrained. W&B tracking is metrics-only; model and run-bundle artifacts
remain local.

## Scientific contracts affected

- Only the epoch-based augmentation policy changes: CROPSHIFT before switch,
  frozen IDBH_WEAK after switch.
- RSLAD objective, KL-PGD10 training attack, teacher target, optimizer,
  scheduler, sampler, all RNG ownership, and data order remain unchanged.
- Endpoint evaluation is independent CE-PGD20 on the fixed internal
  validation split; official test and AutoAttack are excluded.
- Parent/resume state must include model, optimizer, scheduler, scaler, RNG,
  sampler, augmentation state, and tracking lineage.

## Decisions

- Candidate set is fixed to `{50,75,100,125}`; existing `I100`/`I150` are
  descriptive references only.
- Exact S50/S75/S125 parents are materialized by deterministic CROPSHIFT-only
  continuation from the sparse checkpoint immediately preceding each boundary.
- Six fresh continuations are run on available Hamster/Ferret GPUs (I50/I75/I125
  × seeds 1/2), with host and GPU UUID recorded for audit.
- I100 is the incumbent for replacement comparisons. No candidate replaces it
  unless both seeds improve final CE-PGD20 robustness and do not reduce full
  AUC, with post-switch AUC, clean, and throughput guardrails reported.
- No automatic promotion is made; the branch closes after the report and
  freeze artifact.

## Milestones

- [ ] M0 repo reconciliation, scheduler/RNG audit, and parent inventory
- [ ] M1 exact S50/S75/S125 parent materialization and parity
- [ ] M2 switch canaries and immutable source/config freeze
- [ ] M3 six Hamster/Ferret continuations through epoch 199
- [ ] M4 CE-PGD20 endpoints and hybrid/post-switch AUC analysis
- [ ] M5 timing profile, I100 replacement decision, and freeze artifact
- [ ] M6 consolidated review, cohesive commit, push only if explicitly requested, and stop

## Test plan

- Static host preflight: Git, CUDA/GPU UUID, disk, dataset, teacher, and W&B
  login/metrics-only mode on Hamster and Ferret.
- Measure one short continuation step per host before assignment; use
  longest-processing-time-first while recording host as a nuisance variable.
- Focused stage-wise transform, parent/resume, scheduler-boundary, and RNG
  continuity tests; run `scripts/verify.py --changed` once after the delta.
- One bounded canary per new switch boundary before production.
- Validate all six child trajectories, 32 new endpoint reports (train and
  fixed internal validation splits), and exact
  endpoint attack identity. Do not add expensive tests to automated suites.

## Risks and mitigations

- Off-by-one switch: bind payload epoch, completed count, next epoch, global
  step, scheduler `last_epoch`, LR, and first IDBH epoch in the parent audit.
- Resume/RNG drift: require complete state and compare sample order/spatial
  prefix identity against CROPSHIFT continuation.
- Endpoint leakage: use only predeclared internal validation and independent
  CE-PGD20; do not inspect official test.
- Storage: W&B metrics-only; keep checkpoints and row artifacts local. Use
  hash-verified `rsync` for any cross-host parent transfer.
- Host/runtime: distribute jobs by measured throughput, retain host/GPU UUID,
  and do not kill other jobs.

## Progress log

- 2026-08-29: Reconciled clean `master` at `960844f`; previous stage-wise
  result commit is present on `origin/master`. No active training or endpoint
  jobs are running. Existing I100/I150 results and exact sparse CROPSHIFT
  checkpoints are available locally.

## Completion report

To be filled after M6 with parent hashes, canary results, six run IDs, endpoint
hashes, timing profile, freeze decision, and remaining uncertainty.
