# ERT Clean-Wrong teacher-adaptive margin lambda sensitivity

Status: planned; GPU execution is blocked until the immutable source and
required parent artifacts are available on the GPU host.

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

## Current preflight

The local sandbox has no visible NVIDIA driver.  Ferret has three idle RTX
4090 GPUs, but its checkout is behind this source SHA and lacks the required
epoch-79 parent files.  No GPU training has started.
