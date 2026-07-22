# Single-Teacher ARD Bootstrap and Core

## Status

- Owner: primary agent with Sol/Terra/Luna role separation
- Branch / base SHA: `master` / unborn
- Current milestone: complete
- Last updated: 2026-07-22

## Goal

Deliver a reproducible single-teacher adversarial robustness distillation platform whose train/evaluate CLIs, RSLAD/entropy/student/joint ablations, sample state, checkpoint resume, W&B offline integration, upstream lock, and impact-aware verification are exercised without production training or full AutoAttack.

## Non-goals

- CIFAR 300-epoch or Tiny-ImageNet production training
- Full AutoAttack during bootstrap
- ImageNet-1K or dual-teacher implementation
- Vendoring or copying license-unclear upstream code
- Mid-epoch exact resume or cross-server DDP in v1

## Existing state

- The repository is an unborn Git repository containing only the bootstrap instructions and documentation; all initial files are untracked and must be preserved.
- The package, dependency definition, test gate, synthetic core, and external-management scripts now exist. No real teacher checkpoint or dataset cache has been supplied.
- Official SAAD is locked at `https://github.com/HongsinLee/saad.git` commit `295121c5d2eed827b5b2d6aa42307de809bdfada`; the checkout is clean and no root license file was found, so source is not copied or vendored.
- The restricted sandbox hides CUDA and forbids localhost TCPStore sockets. A narrowly permissioned inventory found two RTX 4090 GPUs; focused CUDA and CPU/Gloo tests were run outside that sandbox.

## Scientific contracts affected

- Pixel `[0,1]` attack domain, explicit normalization ownership, and rational epsilon/step-size specifications
- Teacher parameter freeze versus optional teacher input gradients
- KL direction, temperature, `T^2`, clean/adversarial logit pairing, and pre-reduction weights
- Stable sample identity, DDP padding masks, EMA update timing, and state serialization
- Epoch-boundary checkpoint/resume including optimizer, scheduler, scaler, RNG, sampler, tracking identity, and best/last selection
- Separate saved-checkpoint evaluation and rank-zero-only W&B tracking

## Decisions

- Use Pydantic v2 and PyYAML with strict unknown-key rejection and resolved YAML output.
- Keep attack inputs in pixel space; normalization lives in explicit model/teacher adapters.
- `rslad_entropy` uses upstream-exact entropy weighting; hard-label fallback is limited to student/joint policies.
- Robust learnability is probability-margin EMA with decay `0.9`, first-observation initialization, an epoch-zero exact baseline-RSLAD warmup, and pre-update detached FP32 logits.
- Student risk is `(1-margin_ema)/2`; teacher overconfidence is `1-H/log(C)`; joint risk is their product.
- Exact resume is supported at epoch boundaries only; repro/production reject mid-epoch checkpoint claims and world-size changes.
- The core RSLAD threat model is `epsilon=8/255`, `step_size=2/255`; the apparent upstream argument inversion is documented, not copied.
- The agent will not commit or push. Dev verification supports an unborn branch; repro/production require a real HEAD.

## Milestones

- [x] M0 Repository/bootstrap
  - Files/modules: packaging, ignore rules, CLI/package scaffold, external lock/bootstrap/verify, impact/cache gate, plan/docs
  - Owner: Terra for scripts/testing; Luna only for non-overlapping scaffold/config/docs
  - Tests: T0 imports/config/CLI; offline external and cache units
  - Acceptance: atomic external handling, unborn/untracked diff support, cache hit/invalidation, CLI help
  - Rollback point: reviewed M0 diff and test ledger
- [x] M1 Scientific core
  - Files/modules: typed config, indexed data, model/teacher registries, PGD contracts, trainer/checkpoint/distributed
  - Owner: Terra
  - Tests: config rejection, stable indices, model/teacher contracts, PGD projection/gradient/mode, deterministic epoch resume, CPU and conditional GPU smoke
  - Acceptance: distinct best/last and synthetic end-to-end train checkpoint
  - Rollback point: reviewed core interfaces and fixed scientific contracts
- [x] M2 Baseline reproduction
  - Files/modules: PGD-AT/TRADES/RSLAD, entropy signal/policy, upstream wrapper, regression fixtures, reproduction docs
  - Owner: Terra; upstream inspection remains read-only
  - Tests: formula/gradient/fixed-batch regressions and conditional upstream differential
  - Acceptance: method switch by config and evidence-backed SAAD weight
  - Rollback point: reviewed baseline parity report
- [x] M3 Student-aware components
  - Files/modules: sample store, robust margin/correctness/forgetting, student/joint policies, ablation configs
  - Owner: Terra; Luna may add configs after API freeze
  - Tests: update/range/monotonicity/serialization/resume/DDP duplicate contracts
  - Acceptance: four required RSLAD ablations use one trainer and never delete samples
  - Rollback point: reviewed policy/state integration
- [x] M4 W&B and evaluation
  - Files/modules: tracker adapters/guard/manifest/artifacts, offline sync, clean/PGD/AutoAttack evaluation, analysis
  - Owner: Terra; Luna may synchronize docs/configs afterward
  - Tests: offline W&B, rank-zero, resume ID, production guard, saved-checkpoint evaluation
  - Acceptance: separate evaluation CLI and complete local lineage/artifact manifest
  - Rollback point: reviewed tracking/evaluation lifecycle
- [x] M5 Verification and handoff
  - Files/modules: final impact map/cache, Make targets, usage/reproduction/test/W&B docs
  - Owner: Terra for verification behavior; Luna for final documentation
  - Tests: one impact-selected non-scientific milestone run, cache-hit self-test, conditional GPU/DDP smoke
  - Acceptance: commands/results/cached/deferred tests and remaining scientific risks are fully reported
  - Rollback point: final scientific review

## Test plan

- T0: format, lint, type, config, import, CLI dry-run
- T1: changed-module unit tests for data/models/state/policies/tracking
- T2: fixed-batch numerical, gradient, and conditional upstream regression
- T3: at most two epochs and a few batches for CPU/GPU smoke, checkpoint/resume, and W&B offline
- T4: limited scientific verification is command-only unless explicitly approved
- T5: production training and full AutoAttack are never part of bootstrap tests
- Successful fingerprints include exact command, relevant file hashes, Python/PyTorch/CUDA/GPU, external SHA, fixture version, and seed. T4/T5 results are never pass-cached.

## Risks and mitigations

- Upstream license absence: verify after clone, record evidence, do not copy code, use subprocess/test oracle only.
- Threat-model drift: typed rational quantities plus projection/clamp and fixed-batch tests.
- Upstream RSLAD epsilon/step mismatch: keep paper/main contract and document upstream-code behavior separately.
- DDP duplicate state: mask padded samples and deterministically reduce sparse updates.
- Resume nondeterminism: exact epoch boundary, per-rank RNG/sampler state, matching-world-size guard.
- W&B duplication/offline loss: stable run ID, rank-zero ownership, pending/synced markers, no cleanup before sync.
- Missing CUDA: run CPU/fixed-batch/offline validation and explicitly defer GPU/DDP completion.

## Progress log

- 2026-07-22: Sol planning and read-only upstream exploration completed; implementation authorized; M0 started.
- 2026-07-22: Added direct resolution coverage for all checked-in experiment configs and verified configs impact selection.
- 2026-07-22: M0 implemented and reviewed. Focused verification reached 20 passing tests, followed by three valid cache hits; `make lint` and live `verify_external.py` passed. SAAD was cloned at the locked SHA, remote/clean state verified, and absent license-file evidence recorded. Reviewer-identified impact-selection, stale-cache, T4/T5 caching, invalid-base, and license-evidence defects were fixed with regressions. M1 started.
- 2026-07-22: M1 implemented and iteratively reviewed. Scientific review and `$ard-bug-hunt` found and fixed validation/test leakage, pre-update best selection, frozen-teacher BatchNorm mode leakage, DDP padding and `size < world_size`, output races/collisions, user-bypassable production tracking, implicit real-dataset normalization, checkpoint I/O deadlocks, selection-time BN mutation, correlated validation random starts, and selection-attack threat-model drift. Final impact-selected M1 gate passed outside the socket-restricted sandbox: checkpoint 7, synthetic/Gloo 5, CUDA smoke 1, config 13, data 9, distributed 1, external 10, imports 2, teacher 4, PGD 3, verify-gate 9. A deliberately over-broad test failure injection was narrowed after it intercepted collective pickling; the corrected real two-rank checkpoint success/failure regression passed. Scientific reviewer signed off with no remaining P0-P2 findings. M2 started.
- 2026-07-22: M2 added PGD-AT/TRADES/RSLAD/RSLAD-entropy composition to the one Trainer path, explicit KL target direction and optional `T^2`, frozen-teacher RSLAD, Shannon entropy signal, and visible pre-reduction policy weights. The approved entropy ablation weights the complete RSLAD sample loss exactly by `5*(H-min_batch(H))`. Fixed-batch and one-epoch method-switch tests passed; the opt-in upstream differential skipped because the pinned clone's optional RobustBench/AutoAttack dependency stack is incomplete. The upstream launcher only verifies and runs the locked clone as a subprocess.
- 2026-07-22: M2 scientific review and `$ard-bug-hunt` found rank-local entropy minima, padding poisoning, and a configurable coefficient that violated the exact factor-five method identity. Policies now receive validity/reduction context, every rank takes the same global valid-batch minimum, padding has zero detached weight, and `rslad_entropy` rejects a configurable scale. Independent frozen scalar/gradient regressions cover TRADES/RSLAD with `T^2` on/off. A real two-rank Gloo oracle confirmed weights `[0, 1.5]` and gradient `2.25` for unequal entropy shards with a padded low-entropy row. Final impact gate passed; the pinned upstream primitive oracle remained a documented dependency skip after remote/SHA/clean verification. M2 reviewer reported no P0-P2 findings. M3 started.
- 2026-07-22: M2 review corrected entropy weighting under DDP padding: invalid rows are excluded before a global minimum reduction, all ranks use the same finite valid minimum, invalid weights are detached zero, and the final masked `world_size/global_count` reduction is preserved. The coefficient five is now an immutable method identity. Frozen scalar objective and sharded loss/gradient regressions were added.
- 2026-07-22: M3 added detached FP32 robust-margin signals, replicated stable-ID sample state (EMA=0.9, correctness frequency, forgetting, sparse pending observations), epoch-boundary deterministic rank merge before checkpoints, and exact sample-state resume. RSLAD now exposes its complete 5/6 adversarial + 1/6 clean KD total separately from the adversarial CE fallback: baseline/entropy retain `hard=0`, while student/joint policies blend `kd=1-risk`, `hard=risk`; epoch zero is exact baseline RSLAD (`hard=0`, `kd=valid`, `joint_risk=0`) while collecting state for epoch one. Canonical student/joint method IDs fix EMA decay at 0.9 and warmup at one epoch. Deterministic two-epoch CPU resume coverage and a two-rank Gloo state-merge oracle were added; the Gloo node must run outside the restricted sandbox because TCPStore sockets are denied there.
- 2026-07-22: M4 added rank-zero-only `ExperimentTracker` adapters with stable checkpoint/manifest resume identity, durable local lineage bundles, offline pending/sync handling, W&B public artifact/table boundaries, and post-atomic-checkpoint metric/artifact publication. Saved-checkpoint evaluation now validates the sibling resolved training config digest before loading a student-only model, reports best/last clean and explicit PGD accuracy in a separate evaluation tracker/job, and writes fixed sparse panels plus opt-in genuine Parquet sample statistics. Focused tracking/evaluation/config tests passed with mock/offline W&B; the restricted sandbox prevents W&B service sockets, so offline smoke preserves the local pending bundle in that environment.
- 2026-07-22: M4 corrective lifecycle pass made tracking phases RNG-observational on every rank and routed init, attach/resume, epoch publication, normal finish, and failed-manifest writes through coordinated rank-zero phases. Fresh and resumed W&B intent are now explicit, generated IDs incorporate Git lineage, offline versus offline-sync state is distinguished, artifact hashes are retained, and evaluation result records carry checkpoint and threat hashes. Focused tracker/config/evaluation/checkpoint-resume tests passed.
- 2026-07-22: M4 follow-up made AutoAttack reachable only from the explicit saved-checkpoint evaluation CLI opt-in, maps the validated Linf contract to AutoAttack's `Linf`, records seed/version metadata, and added an injected fake-adapter regression without running the real attack. Offline-sync manifests now persist concrete W&B segment paths and the sync command validates identities before segment-aware `--append` synchronization. Focused evaluation/tracking integration and lint passed.
- 2026-07-22: M4 final review closed transactional artifact-history rollback, failed-finalization no-op resume, strict canonical aggregation, training/evaluation dataset and protocol identity, world-size/effective-batch lineage, and training/evaluation seed recording. A `$ard-bug-hunt` diagnosis then found first-checkpoint world-size laundering, partial attack identity serialization, and an unverified Tiny-ImageNet content assertion. The evaluator now rejects mixed best/last world sizes before output creation; one canonical 14-field attack identity is shared by Trainer metadata, evaluation comparison, protocol, and threat hash; and Tiny adapter-visible content is deterministically hashed with computed/matched provenance. Dedicated regressions cover same-name artifact preservation, failed terminal resume, strict W&B retry/exit behavior, mixed world size, temperature-squared-only attack drift, raw-byte/label mutation, and root-independent Tiny content. Sol signed off M4 with no P0-P2 findings; observed Tiny training identity persistence remains explicitly deferred before Tiny T5 or paper aggregation. M5 started.
- 2026-07-22: M5's first visible-GPU gate attempt failed before pytest because PyTorch's `_CUuuid` crossed the JSON cache boundary. A second gate reached the synthetic integration and exposed an obsolete production-test assertion plus an all-rank shared Parquet temporary-file race. `$ard-bug-hunt` isolated both causes; explicit primitive runtime canonicalization now preserves GPU UUID identity, and sample statistics are written in an outcome-broadcasting rank-zero phase. Focused regressions passed (`test_verify_gate.py`: 24; production guard: 1; two-rank Parquet/checkpoint: 2), with details in `docs/debugging/0004-m5-test-gate-and-ddp-artifact-races.md`.
- 2026-07-22: Final `make lint` passed (Ruff format/check, mypy 51 source files, import 2, train/evaluate help). Final `make verify-milestone` / `scripts/verify.py --changed --force --non-scientific` completed 22 impact-selected test-file commands with 213 passed and one optional upstream-oracle skip, including real CUDA and two-GPU DDP smoke, CPU/Gloo, checkpoint/resume, and offline/mock W&B. With identical source and physical `PYTHONPATH`, the non-forced command reported 22 `cached pass` entries. A one-epoch, one-step synthetic `ard.cli.train` run and a separate saved-best/last `ard.cli.evaluate` run both exited zero and produced checkpoints, Parquet sample statistics, panels, and local lineage bundles; their fixture accuracies are not research results. T4/T5, CIFAR/Tiny full training, live online W&B, dependency-complete upstream SAAD, and real full AutoAttack remain deferred.

## Completion report

### Delivered

- M0-M5 are implemented under `src/ard/`, `configs/`, `scripts/`, `tests/`, and the linked protocol documents. The runtime keeps one Trainer while config/registries exchange attacks, objectives, signals, policies, sample state, teacher adapters, tracking, and saved-checkpoint evaluation.
- Required method configs exist for RSLAD, upstream-exact entropy weighting, student robust-margin EMA, and joint risk. Best/last checkpointing, epoch-boundary resume, stable sample IDs, single-server DDP, W&B/offline lineage, qualitative panels, Parquet statistics, and impact-aware pass caching are present.
- Official SAAD is a clean external oracle at `https://github.com/HongsinLee/saad.git`, commit `295121c5d2eed827b5b2d6aa42307de809bdfada`. `scripts/verify_external.py` passed; no root license file was found, so `external.lock.yaml` records `license_status: absent` and no upstream source was copied.

### Verification ledger

- `make lint`: passed after deterministic formatting; Ruff format/check covered 87 files, mypy passed 51 source files, imports passed 2 tests, and both CLI help commands exited zero.
- `make verify-milestone` (`scripts/verify.py --changed --force --non-scientific`): 22 test-file commands, 213 passed and one skipped optional upstream oracle, exit zero. This included real CUDA PGD, two-RTX-4090 DDP smoke, CPU/Gloo DDP, fixed-batch objective/gradient regressions, deterministic resume, best/last separation, W&B offline/mock lifecycle, and saved-checkpoint evaluation.
- The identical non-forced command with the same physical `PYTHONPATH` emitted 22 `cached pass` lines and ran no pytest items. A mismatched ambient `PYTHONPATH` correctly changed the fingerprint; that exploratory retry was interrupted after two redundant passes and is not counted as cache evidence.
- A bounded actual CLI smoke used `python -m ard.cli.train --config configs/experiments/synthetic_pgd_at.yaml --output /tmp/ard-final-cli-train-0722-a1 method.attack.steps=1`, followed by a separate `python -m ard.cli.evaluate` on both saved checkpoints. Both exited zero and wrote best/last, resolved configs, Parquet statistics, panels, and lineage bundles. Its clean/PGD accuracy `0.125` is a synthetic fixture check, not a research result.
- T4/T5, full CIFAR/Tiny training, live online W&B and production sync operations, real full AutoAttack, dependency-complete full SAAD reproduction/differential, and a monolithic single-command pytest run were not executed.

### W&B and review outcome

- Network-free offline/mock coverage verified stable resume IDs, rank-zero ownership, public artifact/Table paths, content-addressed local history, same-name rollback preservation, failed initialization/publication manifests and exit codes, pending offline-sync state, best/last model artifacts, sample Parquet, and evaluation grouping. Live online upload remains operationally unverified.
- Sol's final scientific implementation review approved with no P0-P2 findings. Confirmed bugs fixed during review include DDP-global entropy minima/padding, teacher mode and input-gradient contracts, selection attack drift, checkpoint collective failures, tracking RNG drift, artifact rollback/no-op completion, evaluation identity laundering, partial attack identity, Tiny content assertion labeling, CUDA UUID cache serialization, and the all-rank Parquet race. Details are under `docs/debugging/`.
- Remaining scientific risks: CIFAR schedule/augmentation/public-number parity is not established; no non-fixture accuracy or upstream parity is claimed; observed Tiny training identity must be persisted before Tiny T5/paper aggregation; hardware-level nondeterminism beyond the recorded runtime identity remains possible; W&B emits a deprecation warning and one test-only resume fixture emits a scheduler-order warning.

### Reproduction and production commands

After creating the first Git commit and exporting the dataset, teacher, W&B, seed, schedule, and output variables documented in `docs/REPRODUCTION_STATUS.md`:

```bash
PYTHONPATH=src python -m ard.cli.train \
  --config configs/reproduction/cifar10_r18_rslad.yaml
PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/reproduction/cifar10_r18_rslad.yaml \
  --checkpoint-dir "${ARD_OUTPUT_ROOT}/cifar10-r18-rslad-repro-s${ARD_SEED}"

PYTHONPATH=src python -m ard.cli.train \
  --config configs/production/cifar10_r18_rslad.yaml
PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/production/cifar10_r18_rslad.yaml \
  --checkpoint-dir "${ARD_OUTPUT_ROOT}/cifar10-r18-rslad-production-s${ARD_SEED}"
```

The entropy, student, and joint variants use the neighboring config files only; no training-loop copy is required. Full AutoAttack remains a separate explicit saved-checkpoint process with `--allow-autoattack evaluation.autoattack=true` and was not run during bootstrap.
