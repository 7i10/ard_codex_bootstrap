# Experiment Taxonomy, Teacher Audit, and Two-GPU Pilot Readiness

## Status

- Base: `f7ec48c`
- Current milestone: P3 config/documentation migration and P4 observability
- Owners: Sol for decisions/review, Terra for core runtime, Luna for configs/docs after API freeze
- Last updated: 2026-07-22

## Goal and decisions

Separate scientific protocol, experiment lifecycle tier, and hardware execution profile so a controlled comparison is not mislabeled as SAAD reproduction and one-GPU/local-BN runs are never aggregated with two-GPU/local-BN runs.

- Remove runnable `configs/reproduction/`; do not replace it with another runnable reference directory. Public SAAD paper/code settings remain non-runnable records in `configs/protocols/` with documented deltas.
- Keep `tier: repro` readable for old resolved configs, but deprecate it and create no new checked-in `repro` template.
- Add `tier: pilot`. A five-epoch pilot uses its own `controlled_cifar10_r18_pilot_v1` identity; it is not a shortened result under the 200-epoch canonical identity.
- Keep ordinary DDP with local BatchNorm. Do not silently introduce SyncBatchNorm. Persist `world_size`, `per_rank_batch_size`, `global_batch_size`, and `batchnorm_mode: local_per_rank` in manifest, evaluation identity, output/group identity, and aggregation keys.
- Canonical production templates are Chen/Bartoldson × RSLAD/entropy/student/joint. The execution profile is two GPUs, per-rank batch 64, global batch 128; all compared runs must use the same profile.
- Add a separate, W&B-free teacher accuracy audit over the official CIFAR-10 test set. Its bounded PGD-20 CE result is a screening measurement, not RobustBench AutoAttack reproduction.
- Prepare five-epoch pilot configs and commands, but do not run them without explicit approval because the current bootstrap execution boundary is at most two epochs.

## Milestones

- [x] P1 — Taxonomy, protocol, and execution identity
  - Core: schema, protocol registry, tracking manifest, evaluation identity, aggregation, focused tests.
  - `pilot` requires real data/teacher and online or offline-sync tracking lineage. Legacy `repro` remains parse-compatible only.
  - `local_per_rank` BatchNorm identity differentiates ws1/prb128 from ws2/prb64 even at global batch 128.
- [x] P2 — Bounded teacher accuracy audit
  - Core: typed audit config, `ard.cli.audit_teacher`, official-test indexed loader with no download, strict teacher registry/SHA/normalization, clean and PGD-20 CE accuracy, stable IDs, result JSON, CUDA peak allocated/reserved bytes and GPU identity.
  - No W&B, AutoAttack, training, or automatic dataset/checkpoint download.
- [x] P3 — Config and documentation migration
  - Delete `configs/reproduction/*.yaml`.
  - Add `configs/audit/{chen,bartoldson}.yaml`, two teacher-specific five-epoch RSLAD pilot configs, and eight teacher-specific canonical production configs.
  - Keep protocol records under `configs/protocols/`; update experiment, implementation, W&B, reproduction, invariant, and test docs without rewriting historical plan evidence.
- [x] P4 — Pilot observability and handoff
  - Add global images/sec, rank-max peak VRAM, teacher-clean forward count, and execution-profile metrics only if not already derivable without hot-loop synchronization.
  - Provide one-GPU teacher-audit and two-GPU pilot/production commands. Do not execute five-epoch pilot, 200-epoch production, or full AutoAttack.

## Test selection

- T0/T1: schema/protocol guards, every checked-in runnable config resolves, no runnable `configs/reproduction`, pilot is five epochs and ws2/prb64/gb128, eight production configs differ only by teacher/method/output identity.
- T1/T2: teacher audit stable IDs/count/accuracy/attack identity/freeze/result lineage; execution identity stored in manifest and evaluation; aggregation rejects ws1/ws2 or BatchNorm-mode mixing.
- T3: run existing bounded single-/two-GPU smoke only if trainer/distributed instrumentation changes. Use GPU file locks and do not run five CIFAR epochs as a test.
- Final: one impact-selected non-scientific gate, one unchanged cached-pass confirmation, `make lint`, and a Sol scientific review. Do not rerun unrelated unchanged upstream/W&B/attack suites manually.

## Risks and acceptance

- Local BatchNorm makes ws1 and ws2 scientifically different despite equal global batch; identity and grouping must make mixing impossible.
- Bartoldson is fully replicated per rank; DDP does not halve teacher parameter memory. Pilot telemetry must report rank-max VRAM.
- Bounded teacher accuracy depends on sample IDs, seed, batch size, and PGD identity; every field must be recorded and the result must not be compared as if it were AutoAttack.
- Pilot milestones 100/150 do not fire in five epochs. The pilot preserves canonical optimizer/attack settings without pretending to validate the full schedule.

Completion means the old runnable reproduction templates are gone, historical `repro` configs remain readable, pilot and production identities cannot mix across execution profiles, both strict teacher-audit configs resolve locally, all bounded tests pass, heavy runs remain unexecuted, and the milestone is committed without datasets, checkpoints, or W&B run directories.

## Verification ledger

- P1 focused schema/tracking/evaluation tests: 159 passed. The post-review single-process evaluation regression for a two-GPU checkpoint passed; Ruff and mypy passed on the owned files.
- P2 teacher-audit tests: 27 passed; Ruff and mypy passed on the owned files. Scientific review required exact teacher metadata, backend restoration, transactional artifacts, and untracked-file content hashes in Git lineage; all four corrections are implemented.
- P3 config taxonomy: 23 focused config tests passed; two audit, two pilot, and eight production configs resolve under a controlled environment. Runnable reproduction configs were removed.
- P4 observability/review: focused tracking/evaluation/observability selection passed 118 tests; one invalid test mutation failed before reaching the intended guard, then the corrected last-failed regression passed. Final delta review reported no open P0/P1 finding.
- Final changed-path non-scientific gate in the GPU-visible shell: 337 passed, 1 skipped across 20 selected commands. The first sandbox attempt failed only three Gloo localhost-socket tests; the same last-failed tests passed 3/3 outside the socket-restricted sandbox.
- Final lint/import/CLI gate: Ruff format/check passed, mypy passed for 60 source files, import tests passed 2/2, and train/evaluate/audit CLI help resolved.
- Test-gate overhead fix: `--dry-run` no longer collects markers or fingerprints, and exact cached passes skip repeat marker collection. The focused verify-gate suite passed 31/31.
- CIFAR-10 acquisition: the official torchvision archive matched MD5 `c58f30108f718f92721af3b95e74349a`; torchvision verified 50,000 train and 10,000 test examples under `/home/shunsukenaito/workspace-local/datasets/ard/torchvision`.
- Post-commit operation: both 1000-sample CIFAR-10 teacher audits completed in parallel on GPUs 0/1 at clean HEAD `56610ea`; results are recorded in `docs/REPRODUCTION_STATUS.md`.
- Not run: five-epoch pilot, 200-epoch production, or full AutoAttack.
