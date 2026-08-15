# ERT Confirmatory T1/T2/T3 fixed-anchor experiment

## Status

- Owner: ARD lead
- Branch / base SHA: `master` / `55d01a9`
- Current milestone: M0 contract and runtime preparation
- Last updated: 2026-08-15

## Goal

Run the preregistered Chen ERT confirmatory screen from the exact epoch-79
L2/L4 parents.  Continue one common parent lineage to epochs 84, 89, and 94
for `C79CONF`, `T1WCONF`, `T2WCONF`, and `T3LP05CONF`, then evaluate every
checkpoint with independent eval-mode CE-PGD20 on train and held-out
validation data and write direct/spillover/held-out effect reports.

## Non-goals

- No dynamic router, threshold or coefficient tuning, new seed, IRT run,
  official test, AutoAttack, Stage B extension, or new loss combination.
- Do not select a winner automatically from the endpoint results.

## Existing state

- Exact epoch-79 parents and fixed Stage-A state masks are present under
  `.cache/analysis/ffnr-causal-pilot-screens-e79-94/` and
  `.cache/analysis/ert-state-overlay-v1/`.
- Shared runtime is `src/ard/analysis/ert_stage_a_runtime.py`; it already
  restores optimizer, scheduler, RNG, sampler, and sample state.
- Stage-A CE-PGD20 endpoint evaluator and effect decomposition are committed;
  endpoint train/validation split identity and attack hashes are fixed.
- Prior calibration measured precise coefficients, but this experiment
  explicitly freezes rounded `beta_advce=0.075` and `advkd_multiplier=0.5`.

## Scientific contracts affected

- Training inner attack remains KL-PGD10 (`8/255`, `2/255`, 10 steps,
  random-start, teacher-clean target, pixel space).
- Endpoint attack remains independent CE-PGD20 with the same budget on every
  arm and horizon.
- Fixed epoch-79 masks are hard, stable-ID keyed, and never use future state.
- Full-batch mean is preserved; teacher parameters remain frozen.
- Epoch-84/89/94 checkpoints are exact copies of the post-epoch `last.pt`.
- Train and validation endpoint identities remain disjoint and are reported
  separately; bootstrap is sample-level uncertainty only.

## Decisions

- Reuse the shared Stage-A runtime rather than duplicate a training loop.
- Add optional horizon checkpoint copies and a confirmatory run label while
  preserving the historical Stage-A defaults.
- Use one no-update coefficient sanity artifact before any GPU continuation.
- Use a one-epoch T3 engineering canary to exercise the new treatment path;
  it is not a scientific arm.
- Run eight trajectories (four arms × two seeds) and stop after the fixed
  CE-PGD20 reports.

## Milestones

- [ ] M0: add plan, frozen config, coefficient sanity CLI/artifact.
- [ ] M1: extend runtime with horizon checkpoint copies and confirmatory IDs;
  add focused contract tests.
- [ ] M2: run no-update sanity, clean-tree canary, then eight continuations.
- [ ] M3: evaluate train/validation CE-PGD20 at 84/89/94 and generate the
  fixed direct/spillover/held-out report with 2,000 class-stratified bootstrap
  replicates per preregistered comparison.
- [ ] M4: update result docs with hashes, actual commands, and evidence summary;
  create one cohesive commit. Do not start another intervention.

## Test plan

- T0/T1: config parsing, treatment coefficient contracts, horizon naming,
  existing Stage-A runtime tests, Ruff and `scripts/verify.py --changed`.
- T2: empty-mask/baseline equivalence, T3 AdvKD multiplier isolation, and
  horizon checkpoint hash/epoch regression.
- GPU: one one-epoch canary, then eight 15-epoch continuations. Endpoint
  evaluation is a separate process and uses only saved checkpoints.
- Deferred: official test and AutoAttack.

## Risks and mitigations

- Parent or mask drift: verify checkpoint/config/mask SHA before launch.
- Missing horizon checkpoint: fail the endpoint collection rather than infer a
  last checkpoint.
- Coefficient drift: require the tracked confirmatory config and sanity artifact.
- Runtime/OOM: schedule one trajectory per GPU and retain all outputs atomically.
- Validation leakage: validation is endpoint-only and never enters training or
  coefficient selection.

## Progress log

- 2026-08-15: Reconciled `55d01a9`; exact parents, masks, calibration artifact,
  and endpoint evaluator located. No GPU continuation started yet.

## Completion report

To be filled with exact commands, cached tests, trajectory and endpoint hashes,
bootstrap status, and remaining scientific uncertainty.
