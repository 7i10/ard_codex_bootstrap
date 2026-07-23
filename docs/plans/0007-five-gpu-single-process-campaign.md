# Five-GPU Single-Process Campaign

## Status

- Owner: main thread integration; Sol planning/review; Terra core implementation; Luna config/docs synchronization
- Branch / base SHA: `master` / `9a45747`
- Current milestone: C4 focused verification and consolidated review
- Last updated: 2026-07-23

## Goal

Run an immutable single-GPU CIFAR-10 campaign across Hamster GPUs 0/1 and Ferret GPUs 0/1/2 without a foreground
Codex monitor. Each static job must progress durably through training, saved-checkpoint clean/PGD-20 evaluation,
optional lower-priority AutoAttack, and terminal lineage validation without duplicate launch after controller restart.

## Non-goals

- No multi-host DDP, general cluster scheduler, dashboard, database service, automatic hyperparameter changes, or
  accuracy-driven method invention.
- Do not cancel, restart, mutate, or aggregate the existing
  `chen-rslad-production-s0-0ca90ad` two-GPU run with the new single-GPU cohort.
- Do not automatically extend beyond the preregistered seed-0 core. Seed 1/2 and other stretch work require a later
  scientific decision.

## Existing state

- HEAD `9a45747` is clean and pushed. Future production configs use W&B online; the already-running fixed-SHA run
  remains offline-sync.
- Ferret's fixed-SHA detached execution and selective collection scripts are reusable, but their launch path rejects
  any busy GPU and has no durable multi-phase queue.
- The existing training/evaluation manifests already store world size, per-rank/effective global batch, BatchNorm
  identity, Git/external/teacher/config hashes, best/last checkpoints, and W&B identity.
- All five RTX 4090s have ample free VRAM at inspection time, but campaign-external EEG processes currently keep
  Hamster GPUs at 98/88% and Ferret GPUs at 98/88/36% utilization. Shared launches therefore require explicit
  evidence and must not use throughput as a scientific comparison.
- Hamster and Ferret share driver 595.71.05, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and W&B 0.28.0;
  Python is 3.11 on Hamster and 3.12 on Ferret and remains recorded environment lineage.

## Scientific contracts affected

- The new cohort is `ws1_prb128_gb128_localbn_v1`; it is never pooled with
  `ws2_prb64_gb128_localbn`.
- Dataset, student, optimizer, schedule, KL PGD-10 training attack, CE PGD-20 selection/evaluation attack,
  normalization, seeds, best/last selection, and deterministic mode remain unchanged.
- Training remains a direct one-process invocation with one visible physical GPU. It is never represented as
  world-size five.
- Evaluation remains a separate saved-checkpoint process and constructs no teacher. AutoAttack is opt-in, runs
  separately, and is lower priority than preregistered training and PGD evaluation.

## Decisions

- Use one common Python worker and thin Hamster/Ferret wrappers. Persist atomic JSON plus append-only events and
  `flock` locks; do not introduce SQLite.
- A checked-in campaign template does not embed its own Git SHA. `campaign-start --sha <full SHA>` atomically fixes
  the immutable SHA in runtime state, avoiding the impossible self-referential commit hash.
- Assign jobs deterministically to hosts before launch. Each host owns a static queue and reconciles only its jobs,
  avoiding cross-host claim races.
- Explicit `allow_with_memory_gate` permits campaign-external processes but never duplicate campaign claims.
  Admission uses free VRAM plus pilot `max_memory_reserved`/observed process peak, not allocated memory alone.
- Pilot lengths are one epoch for Hamster Chen RSLAD, three epochs for Hamster Chen Joint, and one epoch for Ferret
  Bartoldson RSLAD. These are engineering checks, not efficacy measurements.
- After pilots pass, the preregistered seed-0 core may start without another code change. After the eight core jobs,
  workers stop at `awaiting_scientific_review`; metric-driven seed extension is not automatic.
- Within each host, pending training and PGD evaluation outrank AutoAttack so long AA runs cannot starve the core
  training queue.

## Milestones

- [x] C0 — Plan, current-run snapshot, and exact campaign contracts
  - Files: this plan and read-only host/run evidence.
  - Tests: none.
  - Acceptance: current run remains alive/terminal without mutation; five GPU UUIDs and external processes recorded.
  - Rollback: no runtime mutation.
  - Commit: included with C1.
- [x] C1 — Execution identity and campaign schema/state
  - Files: `src/ard/campaign/`, config/protocol identity, unit tests.
  - Owner: Terra because state/idempotency and scientific identity are coupled core logic.
  - Tests: strict schema, unique IDs, profile equality, safe paths, transitions, atomic state, ws1/ws2 aggregation
    rejection.
  - Acceptance: duplicate IDs/GPU claims/output/W&B identities and SHA/profile drift fail closed.
  - Rollback: new modules/configs are additive.
  - Commit: `Add durable single-GPU campaign state`.
- [x] C2 — Detached launcher, adoption, GPU sharing gate, and phase pipeline
  - Files: campaign worker/launcher/GPU inspection, scripts, focused unit/integration/remote tests.
  - Owner: Terra; Luna may later synchronize wrappers/docs without touching core modules.
  - Tests: process identity/adoption, orphan evidence, duplicate prevention, external-process snapshot, memory wait,
    train-to-PGD-to-AA transition, failure isolation, current Ferret run reservation.
  - Acceptance: controller restart cannot duplicate a live phase; no scientific setting is changed on failure.
  - Rollback: controllers are unarmed by default and cannot start a real job.
  - Commit: `Add detached campaign workers`.
- [x] C3 — WS1 pilot/production configs and host wrappers
  - Files: `configs/campaigns/`, `configs/pilot/single_gpu/`, `configs/production/single_gpu/`,
    `scripts/campaign/`, documentation.
  - Owner: Luna after C1/C2 APIs freeze.
  - Tests: every config resolves; attack/schedule equality; profile/group/output separation; shell syntax.
  - Acceptance: eight seed-0 cells are preregistered, host-balanced where possible, and distinct from ws2 lineage.
  - Rollback: no launch until explicit arming.
  - Commit: `Configure the five-GPU seed-zero campaign`.
- [x] C4 — Focused verification and consolidated scientific review
  - Tests: `scripts/verify.py --changed`, failed tests only, lint/type on changed modules, synthetic detached lifecycle.
  - Reviewer: one consolidated scientific review; repeat only for unresolved P0/P1 or a changed scientific delta.
  - Acceptance: no P0/P1, clean pushed full SHA, no unchanged full-suite reruns.
  - Commit: corrective delta only if required.
- [ ] C5 — Real pilots and acceptance
  - Runs: Hamster Chen RSLAD 1 epoch, Hamster Chen Joint 3 epochs, Ferret Bartoldson RSLAD 1 epoch.
  - Acceptance: finite train/eval metrics; best/last; full 10k clean/PGD-20; process adoption; no duplicate launch;
    W&B/local terminal lineage; measured memory gate; Joint active after warmup.
  - Failure: preserve evidence and do not arm affected production jobs. Any source/config fix creates a new campaign
    SHA and invalidates only pilots whose execution path changed.
- [ ] C6 — Seed-0 production handoff
  - Runs: Chen/Bartoldson × RSLAD/entropy/student/joint, one single-GPU job per claimed slot.
  - Acceptance before Codex stops: workers detached on both hosts; correct SHA/profile/GPU; first epoch finite;
    durable state advances; W&B identity visible. Workers continue autonomously for about five days.
  - Completion: after core PGD/selected AA, state becomes `awaiting_scientific_review`, not an automatic seed sweep.

## Agent and review budget

Use one Sol plan, one Terra owning core writer, one bounded Luna config/docs pass after APIs freeze, and one consolidated
scientific review. GPU monitoring and routine state collection use scripts, not reasoning agents. Do not repeat review
for closed findings or wait synchronously on long experiments.

## Test plan

- T0/T1: campaign schema, state transitions, identity, path/SHA validation, config resolution, static wrappers.
- T2: detached fixture process adoption, duplicate launch, memory wait, orphan/failure evidence, phase ordering,
  terminal manifest validation, ws1/ws2 separation.
- T3: synthetic detached controller lifecycle on both hosts, then the three bounded real pilots.
- T4/T5: the real 200-epoch campaign and full AutoAttack are experiments, not automated tests.
- Use the success cache. Restricted-sandbox localhost failures are rerun only for the failed nodes in a GPU/network
  capable shell.

## Risks and mitigations

- External GPU compute contention: record process snapshot and shared flag; never compare throughput across shared and
  exclusive runs; wait on memory rather than changing batch size.
- Bartoldson batch-128 OOM: block that teacher/profile only; never reduce batch or add accumulation automatically.
- Host/environment confounding: retain environment hashes and balance final seeds across hosts; seed-0 is exploratory.
- Disk/W&B artifact growth: enforce disk floor, sparse publication, and no cleanup before verified online artifact
  completion. Current checkpoints are about 89.5 MB.
- Online W&B failure: record failed evidence and do not silently fall back to disabled/offline tracking.
- Controller/process loss: corroborate PID, start time, cwd, argv digest, run ID, and SHA before adoption or stop.

## Progress log

- 2026-07-23: Feasibility confirmed. No new experiment started. Existing two-GPU run remains protected.
- 2026-07-23: Added strict campaign schema/state, argv-only detached phase adoption, shared-memory admission,
  host-local controller wrappers, three bounded pilot cells, and the eight-cell production queue. Production remains
  unarmed pending focused verification, review, commit, and pilot evidence.
- 2026-07-23: Focused verification passed. Review findings were closed with expected AutoAttack lineage validation,
  generator-derived hash-bound pilot evidence, and an exact protected-run release contract. Broad tests were not
  rerun after these bounded deltas.
- 2026-07-23: Final bounded delta checks: Ruff passed; mypy passed for 11 campaign/evaluation scripts/modules;
  `pytest -q tests/unit/test_campaign.py tests/unit/test_campaign_management.py tests/unit/test_evaluation.py
  tests/unit/test_config.py tests/unit/test_pilot_observability.py` passed 78 tests. Consolidated scientific delta
  review reported no remaining P0/P1.
- 2026-07-23: First real pilot attempt at `23968e7` exposed an orchestration namespace collision before training:
  phase control pre-created the guarded job output. All three attempts exited nonzero without scientific metrics.
  Control records moved under campaign state, a focused regression was added, and all three pilots are required again
  from the corrective SHA. See `docs/debugging/0009-campaign-control-output-collision.md`.

## Completion report

Pending.
