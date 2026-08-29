# ERT / RSLAD single-switch augmentation timing completion

## Status

- Owner: Codex root
- Branch / base SHA: `master` / `3c4031033dee5c3f728f0d2981dc169c232037e2`
- Current milestone: M6 complete; timing branch closed
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

- [x] M0 repo reconciliation, scheduler/RNG audit, and parent inventory
- [x] M1 exact S50/S75/S125 parent materialization and parity
- [x] M2 switch canaries and immutable source/config freeze
- [x] M3 six Hamster/Ferret continuations through epoch 199
- [x] M4 CE-PGD20 endpoints and hybrid/post-switch AUC analysis
- [x] M5 timing profile, I100 replacement decision, and freeze artifact
- [x] M6 consolidated review, cohesive commit, push only if explicitly requested, and stop

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

- 2026-08-29: Reconciled the completed six fresh suffixes from production
  source `8083f9c`; exact S50/S75/S125 parents were materialized from accepted
  CROPSHIFT controls, with scheduler last-epoch/LR/global-step boundary checks
  and hash-verified cross-host transfer. All I50/I75/I125 seed-1/2 runs reached
  epoch 199 and produced the required train/validation CE-PGD20 endpoint rows.
  W&B remained metrics-only; checkpoints and run bundles stayed local.
- 2026-08-29: Ran the existing read-only aggregator against the six child
  trajectories, accepted CROPSHIFT controls, and hash-bound endpoints. Fixed a
  type-only epoch-key conversion in the aggregator (string JSON keys to integer
  path formatting); no scientific inputs or definitions changed. The report
  and machine artifact now record the timing profile and preregistered freeze.

## Completion report

M3--M6 completed. The six fresh runs are `idbh-s50-s1-prod`,
`idbh-s50-s2-prod`, `idbh-s75-s1-prod`, `idbh-s75-s2-prod`,
`idbh-s125-s1-prod`, and `idbh-s125-s2-prod`. The result report and hash-bound
machine artifact are [ERT_RSLAD_SINGLE_SWITCH_TIMING.md](../ERT_RSLAD_SINGLE_SWITCH_TIMING.md)
and [ert_rslad_single_switch_timing_results_v1.json](../experiments/ert_rslad_single_switch_timing_results_v1.json).
All six parent/child lineages and endpoint attack identities passed aggregation.
I50 did not pass full/post-switch AUC gates; I75 and I125 passed their
development gates but neither replaced I100 under the preregistered
two-seed replacement rule. Freeze decision: `I100`; the finite timing search
is closed. Only two development seeds and internal validation are used, so no
population-level seed claim or official-test claim is made.
