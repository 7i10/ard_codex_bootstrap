# Baseline Readiness and Target Semantics v2

## Status

- Owner: primary agent; one owning Terra per milestone; Luna only after each API freeze; Sol scientific review at each milestone boundary
- Branch / base SHA: `master` / `937decf2415dc5827df7f06f46316688a1f51507`
- Current milestone: M1 in progress; M0 approved
- Last updated: 2026-07-22

## Goal

Make the repository ready for comparable CIFAR baseline pilots without starting a production experiment: replace the main student/joint hard-label fallback with adversarial-branch teacher-target softening, establish schema/protocol v2, separate student/teacher preprocessing ownership, add a clean-room SAAD-compatible CIFAR ResNet-18 and exact schedule/data contracts, pin RobustBench and define verified local teacher acquisition, remove avoidable teacher/diagnostic overhead, and pass bounded T0–T3 verification at at most two epochs.

## Non-goals

- Downloading RobustBench teacher weights during this implementation
- Teacher clean/AutoAttack accuracy audit, five-epoch pilot, 200-epoch training, multi-seed matrices, or full AutoAttack
- CIFAR-100/Tiny-ImageNet training or ImageNet work
- Copying license-unclear SAAD source into this repository
- Claiming paper/source/public-number parity from schema, fixtures, or synthetic smoke
- Preserving schema-v1 config hashes, checkpoints, method semantics, or W&B runs; no research runs exist

## Existing state

- HEAD `937decf` is clean. The platform has raw-pixel attacks, stable sample state, checkpoint/resume, separated saved-checkpoint evaluation, and W&B lineage, but all checked-in configs use schema-v1 semantics.
- `rslad_student` and `rslad_joint` currently blend complete RSLAD KD with adversarial hard-label CE. The main v2 methods must instead soften only the adversarial KD target; the old behavior remains only as explicit `rslad_hard_fallback`.
- Train and validation subsets currently share one transform-bearing dataset. The current `resnet18_cifar` is a torchvision variant with adapter normalization. The trainer uses identity `StepLR` and can repeat teacher clean forwards and collect PGD step losses through CPU synchronization.
- The installed `robustbench 1.1` is an editable checkout outside this repository at commit `78fcc9e48a07a861268f295a777b975f25155964`; it is useful evidence but not acceptable production lineage. Neither required teacher checkpoint is present.
- Dataset roots exist under `/home/shunsukenaito/workspace-local/datasets/ard/`. Synthetic smoke remains W&B disabled; the future five-epoch pilot, not this milestone, will use project `single-teacher-ard` with offline-sync.

## Scientific contracts affected

- Method identity, target distribution, stop-gradient behavior, KL branch decomposition, warmup, and risk-to-softening transform
- Stable sample state and DDP-valid risk values without sample removal
- Raw pixel `[0,1]` attack domain with exactly-once, independently owned student and teacher preprocessing
- Train-only augmentation, deterministic validation/test views, stable source IDs, and fixed split seed
- Architecture/state-dict identity for clean-room SAAD ResNet-18 versus torchvision CIFAR ResNet-18
- Global/per-rank batch identity, exact zero-based MultiStep LR schedule, scheduler checkpoint/resume
- RobustBench repository/model/checkpoint/preprocessing/threat lineage and no production auto-download
- Teacher clean-forward count, PGD trace synchronization, diagnostics modes, and W&B artifact cadence

## Decisions

- Require `schema_version: 2`; fail closed on v1. Add explicit `protocol.id`, `method.id/version`, structured seeds, optimizer/scheduler, global batch, preprocessing ownership, and target-policy identity to resolved config.
- Add `TeacherTargetPolicy` under `src/ard/targets/`. It consumes detached teacher target information plus detached risk and returns normalized detached probabilities and `rho`; it never owns loss weighting.
- `rslad_student` uses student risk; `rslad_joint` uses student risk times teacher overconfidence. During epoch zero, `rho=0` while state accumulates. Default v2 uses `rho=rho_max*clip(risk,0,1)`, uniform mixing, and adversarial KD only; clean KD is unchanged.
- Keep ablations explicit: `rslad_joint_downweight` downweights KD without hard fallback, while `rslad_hard_fallback` preserves the former joint KD/CE blend. Do not silently reuse either identity for the main methods.
- Preserve raw-pixel attack input. Student and teacher adapters independently declare `preprocessing_owner`; matching normalization profiles are no longer required, but exactly one owner must be validated.
- Add `saad_resnet18_cifar_v1` as a clean-room architecture specification. Retain and rename the current torchvision path as `torchvision_resnet18_cifar_norm_v1`; do not import `.external/saad`.
- Define three immutable protocol IDs: `saad_paper_reproduction_v1`, `saad_code_295121c_audit_v1`, and `controlled_cifar10_r18_v1`. Paper/control use SGD 0.1, momentum 0.9, weight decay 5e-4, global batch 128, milestones 100/150, and crop/flip train augmentation. Code-audit records its 2e-4 and source deviations separately.
- Pin RobustBench at `78fcc9e48a07a861268f295a777b975f25155964`. Registry IDs are `chen2021_ltd_wrn34_10` and `bartoldson2024_adversarial_wrn94_16`; production only loads an already present hash-verified checkpoint and never calls a downloader.
- Synthetic/unit smoke uses disabled/offline W&B only. No live upload occurs. W&B cadence changes retain local last each epoch, local best on improvement, sparse panels, and configured periodic/final artifact publication.

## Milestones

- [x] M0 — Target semantics and schema v2
  - Files/modules: `src/ard/config/`, new `src/ard/targets/`, RSLAD objective/policies/trainer composition, method configs, focused unit/regression tests, affected invariants
  - Owner: one Terra for schema-to-trainer semantics; one Luna batch for YAML/docs after API freeze
  - Tests: config v1 rejection and method identity; risk 0/1, probability sum/finite/stop-gradient; adversarial-only target change; clean branch exact invariance; warmup; explicit downweight/fallback; DDP-valid masking
  - Acceptance: main student/joint methods contain no hard CE fallback; resolved config uniquely reconstructs the equation; fixed-batch gradient contract passes
  - Rollback point: reviewed schema/target/objective diff
  - Planned commit: `feat: introduce target-softening method semantics v2`
- [ ] M1 — Protocol, data, student, and schedule readiness
  - Files/modules: protocol/schedule config, data views/transforms, clean-room student registry, scheduler construction/resume, protocol configs and tests
  - Owner: one Terra; Luna synchronizes standalone YAMLs only after schema is stable
  - Tests: stable IDs across stochastic/deterministic views; validation has no random augmentation; architecture shapes/count/identity; LR at 99/100/149/150 and resumed equivalence; global/per-rank batch validation
  - Acceptance: controlled/paper/code-audit protocols resolve distinctly; raw-pixel contract remains; exact epoch schedule survives checkpoint resume
  - Rollback point: reviewed M1 delta
  - Planned commit: `feat: add versioned CIFAR baseline protocols`
- [ ] M2 — Pinned RobustBench teacher registry
  - Files/modules: `external.lock.yaml`, `.external/robustbench`, `teachers.lock.yaml`, teacher registry/adapter, bootstrap/verify scripts, teacher configs and tests
  - Owner: one Terra; main thread performs the explicit external clone/license verification; no weight acquisition
  - Tests: exact model IDs/constructor metadata/parameter counts; Chen identity preprocessing; Bartoldson embedded preprocessing without double normalization; lock/hash failure; production download rejection; frozen/eval contract
  - Acceptance: repository SHA/license and two teacher specifications are locked; missing weight fails clearly; constructor metadata can be audited offline without loading a checkpoint
  - Rollback point: reviewed M2 delta
  - Planned commit: `feat: add pinned RobustBench teacher registry`
- [ ] M3 — Runtime efficiency and tracking cadence
  - Files/modules: attack request/trace config, trainer teacher-logit reuse, diagnostics modes, tracker publication cadence, manifests/tests
  - Owner: one Terra; Luna only updates stable config/docs
  - Tests: exactly one teacher clean forward per batch and numerical parity; trace disabled means no PGD-step CPU scalar collection; diagnostics off constructs/runs nothing; panel/summary modes; sparse artifact cadence; checkpoint/resume unchanged
  - Acceptance: production defaults avoid repeated teacher clean work and per-step/per-sample synchronization while preserving loss, state, best/last, and tracking lineage
  - Rollback point: reviewed M3 delta
  - Planned commit: `perf: bound teacher diagnostics and artifact overhead`
- [ ] M4 — Bounded integration and handoff
  - Files/modules: impact map/cache inputs, bounded configs/tests, all affected docs and execution ledger
  - Owner: one Terra for integration/impact; one final Luna docs batch
  - Tests: targeted T0–T2; synthetic at most two epochs/one train batch; conditional single-GPU and two-GPU DDP only if available; offline W&B; checkpoint/resume; saved-checkpoint PGD; constructor-only teacher smoke without weights
  - Acceptance: one engine switches required v2 methods/configs; all selected non-scientific tests pass or an environmental skip is explicit; scientific reviewer has no P0/P1; no download/training/full-AA/live-W&B occurred
  - Rollback point: final reviewed diff and clean local commits
  - Planned commit: `test: verify baseline readiness v2`

## Agent and review budget

- One planner pass was requested but its parent turn stalled and was interrupted; one independent read-only code/RobustBench map supplied the concrete integration evidence. The primary agent owns this canonical plan.
- Each milestone has exactly one owning Terra. Write agents never overlap on the same files. Luna is used at most once per milestone after API freeze for mechanical YAML/docs synchronization.
- Each milestone receives one consolidated `scientific_reviewer` pass under `scientific-change-review`. Re-review is delta-only and only for remaining P0/P1 or new evidence.
- `bug_investigator` plus `$ard-bug-hunt` is reserved for unexplained scientific/runtime failures, not known mechanical corrections.

## Test plan

- T0: schema-v2 strict resolution, protocol/method/teacher IDs, production guards, CLI imports, impact map.
- T1: target policies, separate dataset views, model/teacher registries, exact scheduler, diagnostics and cadence.
- T2: target distributions/branches/gradients, preprocessing exactly once, RSLAD regression, teacher-forward parity, resume LR sequence.
- T3: bounded synthetic CPU; CUDA/DDP/offline W&B only when directly affected and available. No network is required.
- T4/T5, teacher accuracy audit, five-/200-epoch CIFAR, multi-seed, full AutoAttack, live W&B, and checkpoint downloads are deferred and never cached as tests.
- After a failure, rerun only the failed node. At each milestone end, run one `scripts/verify.py --changed` selected gate; demonstrate cache only when the same final fingerprint is intentionally checked.

## Risks and mitigations

- Target branch ambiguity: expose adversarial and clean KD separately and assert only the adversarial target changes.
- Detach/gradient drift: calibrated probabilities and risk are detached; student adversarial/clean gradients retain fixed float64/FP32 regressions; teacher parameter gradients remain `None`.
- Schema migration: v1 fails with an actionable error; no compatibility shim or misleading resume is provided.
- Data leakage: train and validation wrappers share raw data/source IDs, not transforms; test never participates in selection.
- Double normalization: raw pixel is the only shared interface; adapters validate preprocessing ownership; RobustBench internal normalization is never wrapped again.
- Upstream licensing: SAAD architecture is clean-room from public structure and independently tested; no source is copied. RobustBench license evidence is recorded after the local exact clone.
- Huge teacher memory: tests inspect registry metadata and lightweight constructors selectively; no Bartoldson checkpoint/AutoAttack or simultaneous teacher matrix is loaded.
- Schedule off-by-one/resume: assert the complete boundary LR sequence and interrupted/resumed equivalence.
- W&B pollution/cost: synthetic smoke is disabled/offline; artifact and media publication is sparse; future pilot uses the dedicated `single-teacher-ard` project only after readiness.

## Progress log

- 2026-07-22: Read repository contracts and the external review at HEAD `937decf`; worktree was clean. Created shared dataset/teacher directories outside Git. Preflight found PyTorch 2.11, installed RobustBench/W&B/PyArrow, and no sandbox-visible CUDA.
- 2026-07-22: Read-only mapping confirmed current fallback semantics, shared train/validation transform, torchvision student, identity scheduler, repeated teacher-forward sites, and external RobustBench commit evidence. No teacher checkpoint was found and no download was attempted.
- 2026-07-22: M0 implemented fail-closed schema v2, structured seeds, explicit method identity, detached adversarial-only teacher-target softening, and explicit downweight/hard-fallback ablations. All checked-in configs and fixtures were migrated.
- 2026-07-22: M0 review found and fixed five P1 classes: FP32 target renormalization/parity, malformed repository YAML and incomplete method fragments, unsupported preprocessing ownership, DDP global-batch identity, and collapsed seed lineage. A delta review then found and fixed the two-GPU smoke batch declaration. Final scientific review approved M0 with no P0/P1.
- 2026-07-22: M0 evidence includes config gate `21 passed`, focused numerical/lineage regressions `4 passed`, impact-selected checkpoint/resume `14 passed`, and six-method synthetic switch `1 passed`. Additional focused suites reported `92 passed` and offline tracking/evaluation `2 passed`. CUDA remains unavailable and a restricted-sandbox Gloo subprocess timed out; neither was retried.

## Completion report

Pending M0–M4 implementation, focused tests, scientific reviews, bounded smoke, documentation, and local commits. No production experiment has started.
