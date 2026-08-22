# ERT Clean-Wrong Teacher-Adaptive Margin RNG Stability Diagnostic

Status: in progress (read-only trajectory audit and fixed-probe replay)

## Purpose

Use only the completed 0054 matched continuations to separate baseline RSLAD
continuation variance from incremental variance associated with the frozen
Teacher-adaptive margin treatment.  No optimizer or scheduler update, new
training, new seed, attack change, or post-hoc coefficient selection is
allowed.

## Frozen lineage and contracts

- Source parent campaign: `f90afa3` / 0054 valid v2 outputs only.
- Blocks: `L2-R1`, `L2-R2`, `L4-R1`, `L4-R2`.
- Arms: `B0_BASE`, `N95`, `A100`, `N105` (N90/N110 are endpoint-only secondary
  context where available).
- Checkpoints: epochs 84, 89, and 94, with exact checkpoint SHA-256 recorded
  in the machine artifact.
- Training attack contract: pixel-space Teacher-clean KL-PGD10,
  epsilon `8/255`, step `2/255`, random start.
- Margin target: Teacher adversarial probability margin clipped to
  `[0.03221710026264191, 0.13952550292015076]`; CleanCE is zero; coefficients
  are inherited from 0054 and never retuned.
- Fixed probe: first 256 sorted IDs from each registered epoch-79
  Clean-Wrong mask, with ID and class hashes recorded.  Probe attack seeds
  are deterministic and shared between R1/R2 for each teacher/epoch/batch.

## Analysis stages

1. Validate 0054 lineage, manifests, checkpoint hashes, and finite metric
   logs.  Report absolute BASE R1/R2 variance and paired treatment-effect
   variance from the existing endpoint panel.
2. Run one real-checkpoint fixed-probe smoke through the public no-update
   replay path.  Verify schema, stable-ID joins, no-update behavior, lineage,
   and report output before launching the panel.
3. Run fixed-probe no-update replay for N95/A100/N105 at epochs 84/89/94 for
   both replicates and teachers.  Save Student/Teacher margins, clipped
   target, deficit, hinge state, R0--R3 regime, probabilities, and boundary
   windows `0.005/0.01/0.02`.
4. Run the existing no-update gradient probe on a deterministic 128-ID subset
   for N95/A100/N105 at epochs 84/94.  Save base/margin/total norms, cosine
   diagnostics, and weighted margin/base ratios.  If a focused probe cannot
   complete safely, record it as unavailable rather than weakening the
   contract.
5. Aggregate R1/R2 divergence curves, regime/hinge transition matrices,
   boundary concentration, target contraction, optional prediction/parameter
   diagnostics, and descriptive correlations with final endpoint variance.
6. Write the machine artifact and human report.  Rank hypotheses only as
   descriptive evidence; do not start target smoothing, adaptive weighting,
   smooth hinge, additional seeds, or any training.

## Gates and exclusions

- Any parent/checkpoint/mask/attack/hash mismatch is fail-closed.
- Fixed replay must not mutate Student, Teacher, optimizer, scheduler, RNG,
  BN state, or checkpoints.
- Bootstrap and population-level seed inference are not part of this
  diagnostic; point estimates and four-block descriptive comparisons are
  reported.
- The historical A7 result is context only and cannot select an arm.

## Outputs

- `docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md`
- `docs/experiments/ert_cw_margin_rng_stability_diagnostic_v1.json`

After the report is complete, stop for human review.
