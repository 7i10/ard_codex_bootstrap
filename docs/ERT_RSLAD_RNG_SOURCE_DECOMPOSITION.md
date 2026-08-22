# ERT / RSLAD baseline RNG-source decomposition

Status: preregistered; implementation and CPU isolation canary complete. No
production trajectory or endpoint result is included yet.

## Current answer

The experiment has not run, so the primary scientific question is intentionally
unanswered. The repository audit establishes that the public paths can now be
separated as follows:

- PGD random-start draws use an explicit attack-owned generator.
- Sampler order, source-keyed augmentation, and DataLoader worker seeds use an
  explicit data-owned generator.
- Python, NumPy, and global Torch CPU/CUDA sources are retained as an explicit
  `other_seed` control.
- The CPU canary passed the four preregistered ownership assertions.

This is an implementation result, not evidence that attack-side or data-side
stochasticity dominates the eventual accuracy variance.

## Frozen design

Two Chen2021LTD_WRN34_10 continuation parents (L2/L4), eight BASE arms per
teacher, and fixed endpoint CE-PGD20 are registered in
[`ert_rslad_rng_source_decomposition_v1.json`](experiments/ert_rslad_rng_source_decomposition_v1.json).
The source audit is in
[`ert_rslad_rng_source_audit_v1.json`](experiments/ert_rslad_rng_source_audit_v1.json).

Training remains Teacher-clean KL-PGD10, epsilon `8/255`, step `2/255`, ten
steps, random start, frozen Teacher, and uniform BASE RSLAD. Epoch 94 requires
the exclusive end argument `--epochs 95`; checkpoints are 84, 89, and 94.

## Results table (pending)

| teacher | arm | val robust 84 | val robust 89 | val robust 94 | REF/source status |
|---|---|---:|---:|---:|---|
| L2 | REF1/REF2/ATTACK1/ATTACK2/DATA1/DATA2/BOTH1/BOTH2 | pending | pending | pending | not launched |
| L4 | REF1/REF2/ATTACK1/ATTACK2/DATA1/DATA2/BOTH1/BOTH2 | pending | pending | pending | not launched |

## Interpretation boundary

After collection, the report will answer REF1/REF2 residual difference,
attack-only and data-only effects, descriptive interaction, source ranking per
teacher and pooled, trajectory divergence, and best-versus-last gap. Two
perturbations are descriptive only; no population inference or automatic
stabilization decision will be made.

The campaign stops after decomposition and endpoint validation. A possible
single stabilization experiment or return to A7 precision work is a human
review decision, not an automatic follow-up.
