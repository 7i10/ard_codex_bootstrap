# ERT Clean-Wrong teacher-adaptive margin lambda sensitivity

Status: completed; six production trajectories and all preregistered
CE-PGD20 endpoint evaluations completed. No automatic lambda selection was
performed.

## Frozen question

Test whether the historical A7 teacher-floor, margin-only treatment remains
effective when only its margin coefficient changes.  CleanCE is excluded from
every arm.  The frozen target floor/cap are `0.03221710026264191` and
`0.13952550292015076`; the candidate coefficients are `0`, `0.10`,
`0.2388051152229309`, `0.25`, and `0.50`.

## Lineage and reuse

- Parents are the exact Chen epoch-79 checkpoints for L2 and L4.
- The epoch-79 Clean-Wrong masks are immutable and hash-bound.
- Historical A0 is reused as `L0_BASE`; historical A7 is reused as
  `L2_CAL` only after the reuse audit passes.  New runs are L1/L3/L4 for both
  seeds.
- Training remains the canonical Teacher-clean KL-PGD10 contract.  Evaluation
  is independent CE-PGD20 at epochs 84, 89, and 94 on train and internal
  validation.  No test/AA, new seed, threshold, floor/cap sweep, or extra PGD.

## Execution gates

1. Verify source, parent, mask, calibration, and historical reuse identities.
2. Run changed tests and a fixed-batch/no-update lambda pressure probe.
3. Transfer only the immutable source commit and required inputs to the GPU
   host; use metrics-only W&B retention and local checkpoints.
4. Run a lambda=0.25 one-epoch canary, then launch L1/L3/L4 on L2/L4.
5. Run CE-PGD20 endpoint jobs for the six fresh trajectories, reusing only
   hash-validated historical L0/L2 endpoint rows.
6. Generate the point report with Direct, non-CW spillover, held-out overall,
   held-out CW, CE20/KL10 Q1--Q5, trajectory, and cross-seed summaries.
7. Stop.  Do not auto-select a lambda or start floor/cap sensitivity.

## Execution record

- Source was pinned and pushed before launch. Ferret preflight found three idle
  RTX 4090 GPUs and the exact L2/L4 parent, mask, and calibration artifacts
  were transferred with SHA-256 verification.
- The L2 lambda=.25 one-epoch canary completed with finite metrics and no
  artifact upload. L2 L1/L3/L4 and L4 L1/L3/L4 then completed through epoch
  94 with exit code 0.
- The first endpoint invocation omitted `PYTHONPATH=src` and failed before
  evaluation; no checkpoint or training state was changed. All 18 endpoint
  evaluations were rerun with the corrected import environment and completed
  with the common CE-PGD20 attack identity.
- Endpoint rows were collected without checkpoint files and validated for
  row count, checkpoint SHA, and attack SHA. The point report is
  `docs/ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md` with machine output at
  `docs/experiments/ert_cw_margin_lambda_sensitivity_v1.json`.

## Endpoint result snapshot

Epoch-94 held-out robust-accuracy deltas versus the corresponding BASE were:

| seed | λ=.10 | λ=.238805 (historical A7) | λ=.25 | λ=.50 |
|---|---:|---:|---:|---:|
| L2 | +0.34 pp | +0.82 pp | -0.50 pp | +1.62 pp |
| L4 | -0.02 pp | +1.06 pp | +0.28 pp | +0.38 pp |

These are descriptive sample-level paired effects, not training-seed
confidence intervals. The report contains epoch-84/89/94, direct, spillover,
held-out Clean-Wrong, and CE20/KL10 Q1--Q5 summaries. The campaign stops here;
no floor/cap sweep or automatic coefficient promotion follows.
