# 0038 — ERT Stage A direct, spillover, and held-out decomposition

Status: complete
Date: 2026-08-13

## Objective

Reuse the completed Stage A epoch-84 checkpoints without any training or
treatment change. Separate registered selected-cohort direct effects,
non-selected train spillover, and held-out validation effects under one
independent CE-PGD20 endpoint.

## Frozen scope

- Seeds: Chen L2/seed 1 and L4/seed 2.
- Arms: `C79`, `ST1W`, `ST1M`, `ST1S`, `ST2W`, `ST2M`, `ST2S`,
  `ST3K1`, `ST3K05`, `ST3K0`, `CW1`, `CW2`, `CW3`.
- Checkpoint: exact epoch-84 end checkpoint from each existing Stage A arm.
- Training endpoint: reuse existing 45,000-row CE-PGD20 Parquets after
  stable-ID and attack-identity validation.
- Held-out endpoint: deterministic validation view from the parent config's
  `validation_fraction=0.1` and `split_seed`; no official CIFAR-10 test.
- Attack: eval-mode CE-PGD20, pixel `[0,1]`, Linf, epsilon `8/255`, step
  `2/255`, random start, independently generated per checkpoint.
- No new training, coefficient/threshold tuning, Stage B, dynamic routing,
  extra seed, or AutoAttack.

## Acceptance gates

- [x] Reconcile HEAD and required Stage A artifacts.
- [x] Add explicit train/validation scope to the endpoint evaluator.
- [x] Test exact validation IDs, train/validation disjointness, and attack
  identity.
- [x] Run one real checkpoint validation smoke before the full queue.
- [x] Evaluate all 26 held-out checkpoints; fail closed on any partial output.
- [x] Compute direct/spillover from existing train endpoints and verify the
  weighted identity for clean and robust accuracy.
- [x] Compute held-out clean/robust effects, class diagnostics, and fixed
  class-stratified paired bootstrap confidence intervals.
- [x] Write the hash-bound JSON/Markdown report and stop without selecting a
  new treatment.

## Provenance requirements

The report records source Git SHA, endpoint source SHA, checkpoint and parent
hashes, treatment arm, validation split ID/class hash and count, attack
identity/hash, input and output hashes, and bootstrap contract. A dirty
evaluator tree, wrong checkpoint, train/validation overlap, mismatched attack,
or failed weighted identity prevents report generation.

## Interpretation is fixed before evaluation

- `ST1W`: held-out support requires direct and held-out robust effects both
  positive; no claim is made from direct recovery alone.
- `ST2M`: compare direct and spillover to distinguish selected recovery from
  train redistribution; held-out must be positive for generalization.
- T3 arms: direct positive is recovery-like; direct non-positive with positive
  spillover and held-out is low-pressure-like; spillover-only is not
  generalization.
- Clean-Wrong: report clean recovery and robust preservation separately;
  clean improvement with held-out robust harm is not promotion evidence.

No automatic winner selection follows the report.

## Completion record (2026-08-13)

- Validation CE-PGD20 completed for all 26 epoch-84 checkpoints on GPU0/GPU1;
  each produced 5,000 rows and an explicit split identity.
- The first real validation smoke exposed and fixed a sampler-size bug for a
  validation subset. The corrected smoke then passed at L2/C79.
- The report validated 45,000/5,000 train/validation disjointness, all stable
  ID/class joins, arm-specific parent lineage, CE-PGD20 identity, and the
  weighted direct/spillover identity for both clean and robust accuracy.
- Class-stratified paired bootstrap used 2,000 replicates with fixed seed
  `20260813`; no outcome was used to tune a route.
- Outputs:
  `docs/ERT_STAGE_A_EFFECT_DECOMPOSITION.md` and
  `docs/experiments/ert_stage_a_effect_decomposition_v1.json`.
- No new training or automatic follow-up was started.
