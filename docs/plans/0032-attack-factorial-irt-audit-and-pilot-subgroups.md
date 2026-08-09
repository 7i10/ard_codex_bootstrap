# 0032 — CE/KL attack factorial, IRT checkpoint audit, and causal-pilot subgroup evidence

## Scope and frozen decisions

- No new intervention training is launched in this phase.
- Chen L2/L4 use fixed checkpoints `189, 194, 199` and the preregistered
  pixel-space `L∞ ε=8/255`, step `2/255`, random-start matrix:
  `CE-PGD10`, `CE-PGD20`, `KL-PGD10`, `KL-PGD20`.
- The existing hash-bound CE-PGD20 outputs are reused only when their complete
  attack identity, seed, epoch set, stable-ID/class universe, and lineage match;
  otherwise that cell is replayed under the new factorial contract.
- Point estimates precede any bootstrap. Bootstrap count, seed, strata, and
  estimator are fixed before launch and progress is resumable.
- Ferret is used for independent replay cells after a real-checkpoint smoke;
  Hamster receives the longest measured jobs first. Remote artifacts are
  transferred with hash-verified `rsync`.
- Bartoldson dense continuation is an audit decision, not an automatic
  200-epoch retrain: complete-state and resume-parity evidence is required
  before any continuation.

## Checklist

- [x] Add strict factorial attack identities and configs.
- [x] Add public CLI and unit tests for loss/step separation and strict config.
- [x] Add CPU point-report and causal-pilot subgroup analysis scripts.
- [ ] Commit/push the implementation so replay provenance is immutable.
- [ ] Run one real sparse-ID checkpoint smoke through the public CLI; verify
  schema, lineage, stable-ID/class join, and non-overwrite output.
- [ ] Run CE/KL × PGD10/20 replay for Chen L2/L4, reusing exact CE20 cells.
- [ ] Produce prevalence, Jaccard, chance-adjusted Jaccard, Cohen κ, and
  failure-frequency correlation with attack identity in every row.
- [x] Audit available Bartoldson L1 checkpoint inventory and continuation
  feasibility; L3/seed-2 remains blocked and no retraining was launched.
- [x] Compute Route A/B subgroup rescue/harm, class spillover, and fixed-seed
  sample-level bootstrap CIs from the existing causal-pilot tables.
- [ ] Update results docs with explicit exploratory/confirmatory boundaries,
  artifact hashes, and any blocked inputs.

## Risks and completion criteria

- Missing or incomplete validation/checkpoint inventories fail closed; no
  partial replay is interpreted as a result.
- Attack objective and step count must remain separable in lineage; no claim
  about CE versus strength is made until all four cells share the same sample
  universe and epochs.
- Human-review labels are optional covariates and cannot silently define a
  train-time treatment mask.
- Completion means focused tests pass, the smoke reaches report creation, all
  available replay cells have immutable lineage, and CPU point/subgroup reports
  are written. It does not imply a new method has been selected.
