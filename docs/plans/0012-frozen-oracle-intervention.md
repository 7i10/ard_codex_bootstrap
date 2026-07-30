# Frozen oracle intervention

Status: in progress

## Goal

Test whether the existing uniform target-softening intervention can help when
sample selection is fixed from future Bartoldson/RSLAD train failures, before
changing the student signal or launching more Student/Joint seeds.

## Scientific boundary

- The mask is built from the CIFAR-10 **training split** only. Official test
  labels, correctness, and AutoAttack outputs are forbidden inputs.
- Selection is frozen before each new run and is not recomputed from the
  intervened model.
- The oracle is an upper-bound experiment, not a deployable method.
- Oracle and random controls use identical sample counts per class, identical
  nonzero risk values, optimizer/schedule/attack/teacher, and seed. Only the
  assignment of selected stable sample IDs differs.
- The primary comparison is best official-test AutoAttack after the fixed
  validation-PGD checkpoint selection. Clean accuracy and best-to-last AA gap
  remain separate guard metrics.

## Frozen design

- Source: Bartoldson/RSLAD seed-0 `last:v19` (epoch 99) and `last:v39`
  (epoch 199), with exact W&B/local checkpoint hashes and scientific Git SHA.
- Common source attack: raw unaugmented CIFAR-10 train partition, pixel-space
  Linf KL PGD-10, epsilon `8/255`, step `2/255`, random start with recorded
  deterministic seed protocol.
- Oracle set: samples robustly correct at epoch 99 and wrong at epoch 199
  (future forgetting), plus samples wrong at both checkpoints (persistent
  failure). Equivalently, future epoch-199 robust error, with transition type
  retained for analysis.
- Intervention: fixed risk `1.0` on selected IDs and `0.0` otherwise;
  `teacher_target_uniform_mix@1` therefore applies `rho=0.5` only to selected
  adversarial KD targets. Clean KD targets and sample inclusion are unchanged.
- Controls: three fixed class-matched random masks, each with the same selected
  count per class and the same binary risk multiset.
- Development uses the already-explored seed 0 only for the oracle-vs-random
  upper-bound decision. It is not a confirmatory v2 seed.

## Progress

- [x] Complete the prospective signal audit and retain the student signal.
- [x] Freeze Oracle Go/No-Go before implementation.
- [x] Implement a fail-closed source replay and frozen-mask manifest.
- [x] Implement frozen sample-risk lookup as a new explicit method identity.
- [x] Add formula, stable-ID, hash/lineage, resume, and random-control tests.
- [x] Bind W&B `last:v19/v39` artifact digests and checkpoint byte identities
  to the generated mask.
- [x] Preserve the historical raw resolved-YAML digest across later schema
  defaults; see `docs/debugging/0014-frozen-oracle-historical-config-digest.md`.
- [x] Exercise the real frozen-risk Trainer branch across deterministic
  checkpoint/resume, padding masks, and frozen-teacher gradient contracts.
- [x] Generate one oracle and three class-matched control manifests read-only.
- [x] Launch oracle/control-1/control-2 on Ferret GPUs 0/1/2 and queue
  control-3 behind oracle on GPU 0.
- [ ] Run Bartoldson oracle/control training and saved-checkpoint PGD/AA.
- [x] Apply one consolidated scientific review to the stable implementation
  delta. Three P1 findings were fixed; delta review found no remaining P0/P1.
- [ ] Record the result in the experiment dashboard.

## Go / No-Go

- Oracle Go: oracle best AA is at least `0.5 pp` above the mean of all three
  random controls, exceeds every random control, and clean accuracy drops by at
  most `0.5 pp` relative to the random-control mean.
- Oracle No-Go: oracle best AA is no greater than the random-control mean.
- Otherwise: inconclusive; do not tune the mask on official-test results.

These thresholds allocate compute and are not training-seed confidence
intervals.

## Tests

- Unit: manifest schema/hash, stable-ID lookup, class counts, binary range,
  official-test namespace rejection, random determinism, no missing train IDs.
- Regression: risk 0/1 gives exact identity/rho `0.5`, clean target invariance,
  frozen risk survives checkpoint/resume, teacher gradients remain absent.
- Integration: synthetic mask construction and one-epoch method switch.
- Operational: Bartoldson source replay on one GPU; no new training until mask
  and source identities pass. W&B `last:v19/v39`, file MD5/size/SHA, local
  checkpoint epoch/config/run identity, and common replay protocol are bound.

## Risks

- The oracle uses future training behavior and cannot be presented as an
  online method.
- A binary `rho=0.5` may be too strong; changing it after seeing AA would be
  tuning on the official test and is prohibited for this experiment.
- Random controls are conditional on one mask and one training seed; passing
  this upper-bound gate only justifies a fresh-seed v2 experiment.
- Replay and four 200-epoch runs are GPU-expensive. Run the four independent
  jobs in parallel on free GPUs after a bounded method smoke.

## Completion evidence

Record source checkpoint/replay/mask hashes, exact train/evaluate commands,
W&B run IDs, best/last clean/PGD/AA, resource usage, tests, and review findings.

Implementation gate before commit:

- `ruff format --check ...`: 9 files already formatted.
- `ruff check ...`: pass.
- `pytest -q tests/unit/test_frozen_oracle.py tests/unit/test_config.py
  tests/regression/test_m3_student_aware.py`: 52 passed.
- Focused mypy uses `--follow-imports=skip` because the repository-wide gate
  currently reports pre-existing errors in signal-audit/AutoAttack modules;
  those unrelated errors are not attributed to this delta.

Operational evidence:

- Source W&B `last:v19`/`last:v39`: epoch 99/199 checkpoint SHA-256
  `9071a7af…23f5` / `d373ebf8…102a`, raw config SHA-256 `b105b3da…222a`.
- Replay/mask builder: clean Git `05cd0c66367e399dde266bd898c3ddc4097ca95c`;
  45,000 train IDs, 3,566 selected, identical per-class counts in all masks.
- Mask SHA-256: oracle `6d85afd7…c614`, controls
  `013ba310…ead2`, `2658f67f…8b54`, `e5795e87…8e7d`.
- Active W&B train IDs: `bart-oracle-soft-s0-05cd0c6`,
  `bart-rand1-soft-s0-05cd0c6`, `bart-rand2-soft-s0-05cd0c6`.
  `bart-rand3-soft-s0-05cd0c6` is prepared and predecessor-gated.
