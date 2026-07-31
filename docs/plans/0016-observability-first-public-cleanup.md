# Observability-first public cleanup

## Status

- Owner: main thread; one Terra writer for scientific/runtime code, Luna only for the later mechanical archive/docs delta
- Branch / base SHA: `cleanup/observability-first` / `0fdfaeb5809e3b08a0825e2e5caf0bebfa215047`
- Active-run constraint: the Bartoldson logging-only seed-1 run continues from the clean base SHA and is not modified or restarted
- Last updated: 2026-08-01

## Goal

Make future experiments collect reusable, intervention-independent student-history and teacher-response primitives on their first run, while removing one-experiment launch logic and completed campaign recovery machinery from the publishable `ard` package. Replace manually maintained live state with a small read-only status view derived from the run bundle that training already writes.

## Decisions

- Observation is a separate config axis, not a method ID or a loss policy. Raw detached primitives are stored; proposed risks, thresholds, and gates remain offline derivations.
- Profiles are explicit about cost: `off`, `student_history` (no teacher forward), and `teacher_response` (student history plus clean/adversarial teacher primitives; at most one reusable adversarial teacher forward per batch).
- Canonical research configs opt in before launch. We do not silently enable the expensive profile for every dev/smoke run.
- The public trainer keeps generic production guards only. The frozen seed-1 preregistration and attestation remain immutable provenance documents, but their hashes, run ID, artifact list, and one-shot allocation claim do not belong in `ard.cli.train`.
- Exact resume of the currently active run remains bound to `0fdfaeb`; the cleanup branch must not claim checkpoint compatibility with a changed runtime.
- Completed Hamster/Ferret campaign controllers, watchdogs, recovery CLIs, host wrappers, configs, and tests move together under `tools/internal/legacy_campaign/`. They remain available as development history but are excluded from the installed package and default test suite.
- `run-bundle/manifest.json` gains an atomically written latest-progress snapshot at each existing epoch-level metric log. A read-only status CLI aggregates local manifests without GPU polling, PID mutation, a daemon, or network access. W&B remains the live cross-host view.
- One consolidated scientific review follows the stable delta. A second review is allowed only for an actual P0/P1 fix.

## Milestones

- [x] O0 — protect the active run and inventory coupling.
  - Separate worktree created at `/tmp/ard-observability-cleanup`.
  - Acceptance: active worktree remains clean at `0fdfaeb`; no process, checkpoint, W&B run, or output directory is touched.
- [x] O1 — decouple observation from intervention.
  - Files: config schema, trainer/build path, sample state, canonical scientific configs, focused config/state/parity tests.
  - Acceptance: ordinary RSLAD can select either observation profile without changing objective, attack, gradients, optimizer/scheduler/RNG, or model checkpoint tensors; resume restores observation state; DDP padding remains excluded.
  - Efficiency: teacher-adversarial logits are reused across observation, policy, and qualitative diagnostics; focused tests assert forward-call counts.
- [x] O2 — remove one-run launch machinery from the public trainer.
  - Files: `src/ard/cli/train.py`, schema/config tests, provenance docs/config archive.
  - Acceptance: no frozen run IDs, report hashes, attestation hashes, allocation claims, or research-specific artifact validation remain in the training entry point; normal output collision, config, Git, teacher, W&B, and resume guards remain.
- [x] O3 — make run state self-updating and readable.
  - Files: tracking adapter, `ard.cli.status`, focused tracking/status tests, Make target.
  - Acceptance: every epoch-level metric update atomically refreshes latest epoch/step/time in the local manifest; completed/failed/sync-pending/stale states render deterministically from fixtures; no CUDA or network is used.
- [x] O4 — isolate completed campaign operations.
  - Files: `src/ard/campaign/`, `scripts/campaign/`, `configs/campaigns/`, their dedicated tests and operational docs move under `tools/internal/legacy_campaign/`; impact map and documentation index are updated.
  - Acceptance: `import ard` and public CLI/tests do not import campaign code; the archive clearly states it is unsupported historical operational tooling and is not a scientific method dependency.
- [x] O5 — verify and review the complete delta; commit and feature-branch publication follow this ledger update.
  - Run the smallest focused tests first, then one `scripts/verify.py --changed --non-scientific` selection. Reuse cache hits.
  - One consolidated `scientific_reviewer` checks observation parity, teacher forward reuse, state/resume, evaluation separation, and lineage.
  - Commit only source/docs/tests. Do not commit outputs, datasets, checkpoints, W&B data, `.external`, or caches.
  - Push the feature branch only; fast-forward `master` after the active run is terminal.

## Test selection

- Config: strict profile validation, teacher requirement, canonical production declarations, removal of the one-off research-design contract.
- Numerical/runtime: RSLAD `off` versus observed model/optimizer/scheduler/RNG parity; teacher parameter gradients remain `None`; detached observations are finite; one adversarial teacher forward is shared.
- State/resume: history and teacher primitives serialize exactly, stable sample IDs and DDP padding contracts remain intact.
- Tracking/status: epoch progress atomically updates the manifest; JSON/Markdown aggregation covers running, stale, failed, sync-pending, and completed fixtures without process/GPU inspection.
- Public boundary: installed package has no `ard.campaign`; public `train.py` contains no frozen allocation identity.
- GPU: no new long experiment or full AutoAttack. Run a bounded CUDA parity test only if the selected delta requires it and a non-isolated GPU shell is available.

## Risks

- Observation can be computationally expensive. Mitigation: explicit profile, recorded profile in resolved config/manifest, reuse of teacher logits, and forward-count regression.
- Removing the legacy method/gate can break resume at the new SHA. Mitigation: current run and any exact resume stay on `0fdfaeb`; this is documented rather than hidden behind compatibility code.
- A committed Markdown “live dashboard” becomes stale by construction. Mitigation: keep scientific results static, derive live status from manifests/W&B, and label timestamps and roots explicitly.
- Moving operational code can erase useful incident history. Mitigation: archive it intact with its tests and protocol rather than deleting it.

## Completion conditions

- Future canonical RSLAD-family runs declare reusable observation profiles before launch.
- No observation tensor reaches loss, attack, policy, or optimizer unless a method explicitly consumes a separately configured policy signal.
- Public training contains no seed-1-specific evidence gate or allocation claim.
- Public package/test path excludes legacy campaign recovery/watchdog machinery.
- Local status updates with training metrics and can be rendered without manual GPU checks.
- Focused and impact-selected tests pass from actual commands; scientific review has no open P0/P1.

## Progress log

- 2026-08-01: the requested planning subagent did not return a conclusion within a useful bounded window. It was
  stopped without treating silence as approval; the main thread froze the minimal plan from repository contracts and
  the two ARD research documents instead of launching replacement planners.
- 2026-08-01: all completed campaign/controller/watchdog/recovery source, configs, dedicated tests, and protocol
  documents moved intact to `tools/internal/legacy_campaign/`. `git diff --check` passed for the mechanical move.
- 2026-08-01: observation became the explicit `off|student_history|teacher_response` config axis. The separate
  `rslad_logging_only` method and `ResearchDesignConfig` were removed; all 16 canonical production RSLAD-family
  configs opt into `teacher_response`. One teacher adversarial forward is shared per batch and its count is logged.
- 2026-08-01: the public trainer no longer contains the seed-1 hashes, evidence evaluator, W&B/output allocation, or
  claim file. Frozen documents moved to `tools/internal/history_replication/provenance/`; exact resume remains on
  `0fdfaeb`.
- 2026-08-01: `LocalTracker.log_metrics` atomically writes `latest_progress`; `ard.cli.status` renders local JSON or
  Markdown without a network, PID, GPU query, or daemon. Focused integration/config/status checks reported
  `102 passed`; impact/import checks reported `34 passed` and `2 passed`. Ruff and focused mypy passed. An initial
  command used two stale node IDs and collected no tests (exit 4); the corrected command is the recorded pass.
- 2026-08-01: consolidated scientific review found no P0, one P1 and one P2. The P1 showed that changing the same
  schema-v2 normalized config would block additional evaluation of old checkpoints. Evaluation now hashes untouched
  saved YAML before an evaluation-only migration and reports source/runtime method identities. The P2 dropped
  `latest_progress` on running resume and preferred old progress time for terminal rows; both were fixed. The focused
  fix gate reported `37 passed`, with Ruff, focused mypy, and diff check passing. Fix-delta review is pending.
- 2026-08-01: fix-delta review found no remaining P0/P1. Its one P2 on resume freshness was fixed by deriving
  running freshness from the latest progress/resume event while retaining the actual metric timestamp; the focused
  status gate reported `4 passed`. Final `PYTHONPATH=src .../python scripts/verify.py --changed --non-scientific`
  completed with exit 0. It selected only non-scientific affected tests; T4/T5 and long GPU experiments were not run.
