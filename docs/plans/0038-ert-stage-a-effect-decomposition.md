# 0038 — ERT Stage A direct, spillover, and held-out decomposition

Status: in progress
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
- [ ] Add explicit train/validation scope to the endpoint evaluator.
- [ ] Test exact validation IDs, train/validation disjointness, and attack
  identity.
- [ ] Run one real checkpoint validation smoke before the full queue.
- [ ] Evaluate all 26 held-out checkpoints; fail closed on any partial output.
- [ ] Compute direct/spillover from existing train endpoints and verify the
  weighted identity for clean and robust accuracy.
- [ ] Compute held-out clean/robust effects, class diagnostics, and fixed
  class-stratified paired bootstrap confidence intervals.
- [ ] Write the hash-bound JSON/Markdown report and stop without selecting a
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
