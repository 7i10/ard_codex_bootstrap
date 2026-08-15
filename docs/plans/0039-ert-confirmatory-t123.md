# ERT Confirmatory T1/T2/T3 fixed-anchor experiment

## Status

- Owner: ARD lead
- Branch / base SHA: `master` / `55d01a9`
- Current milestone: complete; fixed endpoint report recorded
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

- [x] M0: add plan, frozen config, coefficient sanity CLI/artifact.
- [x] M1: extend runtime with horizon checkpoint copies and confirmatory IDs;
  add focused contract tests.
- [x] M2: run no-update sanity, clean-tree canary, then eight continuations.
- [x] M3: evaluate train/validation CE-PGD20 at 84/89/94 and generate the
  fixed direct/spillover/held-out report with 2,000 class-stratified bootstrap
  replicates per preregistered comparison.
- [x] M4: update result docs with hashes, actual commands, and evidence summary;
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
- 2026-08-15: Ran the no-update coefficient sanity check and a corrected
  one-epoch T3 engineering canary. The first canary invocation was rejected
  because an exclusive epoch bound would have produced zero continuation
  epochs; the runtime now fails closed for that case, and the corrected canary
  completed successfully.
- 2026-08-15: Completed eight online W&B continuations (L2/L4 × four arms)
  from the exact epoch-79 parents, saving epochs 84/89/94. Independent
  train/validation CE-PGD20 endpoint evaluation produced all 24 endpoints per
  seed. The fixed 2,000-replicate class-stratified report was generated.

## Completion report

The machine-readable result is
`docs/experiments/ert_confirmatory_t123_results_v1.json` with SHA-256
`5967fa9d6ac03f004b76c89c08eeb719b58e080431256d631570cd9950e4ceae`.
The no-update sanity artifact is
`docs/experiments/ert_confirmatory_t123_calibration_sanity_v1.json` with
SHA-256 `3bcf69110216ce992b6d3e3e25a3894cc8d6a2f66fe266311c1f166c17f57a5c`.
The endpoint interpretation and limitations are in
`docs/ERT_CONFIRMATORY_T123_RESULTS.md`.

Executed verification and run commands:

- `pytest -q tests/unit/test_ert_stage_a_runtime.py tests/unit/test_ert_confirmatory_calibration.py` — 6 passed.
- `pytest -q tests/unit/test_ert_stage_a_runtime.py tests/unit/test_ert_confirmatory_calibration.py tests/unit/test_ert_confirmatory_report.py` — 8 passed.
- Ruff on all affected Python files — passed.
- `scripts/verify.py --changed` after the final documentation-only delta —
  selected T0 and correctly reported no impacted tests.
- The no-update calibration CLI completed before GPU continuation.
- One corrected canary completed, followed by eight W&B-online continuations
  and 48 independent endpoint jobs.

The repository-wide mypy baseline still reports 15 pre-existing errors in
`autoattack.py`, `signal_audit.py`, and `teacher_risk_replay.py`; no new
confirmatory module error was introduced. Official test and AutoAttack were
intentionally not run.
