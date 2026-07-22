# Ferret Fixed-SHA Remote Execution

## Status

- Base: `dd7c54b`
- Current milestone: F6 blocked on Ferret SSH authentication
- Owners: Sol for plan/review, Terra for scripts/tests, main thread for integration and remote operations
- Last updated: 2026-07-23

## Goal and decisions

Provide an explicitly invoked repo-local `$run-on-ferret` skill that lets Hamster prepare, launch, observe, collect, cancel, and clean bounded jobs on Ferret without editing remote code or running an unfixed branch.

- Treat GitHub as the source transport and require a full 40-character commit SHA plus a detached worktree per run.
- Keep remote paths configurable. The prompt's Ferret paths are examples until read-only preflight verifies them.
- Accept commands after `--` as argv; do not accept an eval-able command string. Generate launch metadata through a deterministic helper and quote argv safely.
- Refuse busy GPUs, invalid GPU sets, duplicate runs, dirty source assumptions, and destructive cleanup without explicit execution flags.
- Default collection excludes checkpoints, W&B offline data, caches, and Python bytecode. Never auto-commit collected results.
- Use GNU screen as requested, but isolate every run under a validated run root and record its process group, screen name, Git SHA, GPUs, timestamps, and exit status.
- The current Hamster has no working `Ferret` alias. Direct IP reaches SSH but key authentication is not configured, so remote mutation and smoke remain blocked until the user registers a key.

## Milestones

- [x] F0 — Repository, instruction, and Hamster inspection
- [x] F1 — Skill scaffold, shared validation/transport library, example configuration
- [x] F2 — Read-only preflight and fixed-SHA prepare
- [x] F3 — screen launch, normalized status, bounded logs
- [x] F4 — selective rsync collect, targeted cancel, dry-run cleanup
- [x] F5 — local mocked/static tests and skill validation
- [ ] F6 — Ferret read-only, prepare-only, CPU screen smoke, collect, cleanup dry-run
- [ ] F7 — delta review, documentation, cohesive commit, non-force push

## Changed surfaces

- `.agents/skills/run-on-ferret/`: explicit skill metadata and executable workflow.
- `configs/remote/ferret.example.env`: non-secret path and host examples only.
- `tests/remote/`: local validation and mocked transport/status tests.
- `docs/FERRET_EXECUTION_PROTOCOL.md`, `docs/README.md`: operating contract and index.
- `docs/plans/0006-ferret-remote-execution.md`: progress and evidence ledger.

## Test selection

- Static: Bash syntax, skill `quick_validate.py`, Ruff if Python helpers are added, ShellCheck when installed.
- Local focused: run-id/SHA/GPU/path validation, argv quoting, busy-GPU refusal, duplicate run refusal, state normalization, safe collection/cleanup command construction.
- Remote read-only: BatchMode identity, GPU inventory, tools, disk, repo/remote/status/worktrees, screens/runs, dataset/checkpoint accessibility.
- Remote bounded: prepare-only, CPU screen command, status/log/collect, cleanup dry-run. Do not start production, downloads, AutoAttack, or live W&B.

## Risks and completion

- SSH host-key acceptance and public-key registration require human confirmation. Never expose or replace private keys.
- `screen` can disappear while child processes survive; status/cancel must corroborate screen, recorded process group, command path, and ownership.
- A shell command is inherently powerful on the user's account; invocation must be explicit and preserved as argv without `eval`.
- Three-GPU execution changes the existing global-batch-128 protocol because 128 is not divisible by three. The skill may transport a three-GPU command but must not claim protocol equivalence or rewrite batch/LR settings.

Completion requires local safety tests and skill validation, verified BatchMode SSH, Ferret GPU/repo discovery, fixed-SHA prepare, CPU screen lifecycle, selective collection, dry-run cleanup, documentation, review, commit, and push. Until SSH authentication is configured, report partial readiness rather than claiming remote validation.

## Verification ledger

- Hamster inspection: `ssh -G Ferret` resolved only the unconfigured hostname `ferret`; no `~/.ssh/config` exists. Direct `shunsukenaito@192.168.100.18` reached SSH but BatchMode authentication failed with `Permission denied (publickey,password)`.
- Static/local: Bash syntax passed for all nine scripts; skill `quick_validate.py` passed; focused remote-controller tests passed `16 passed in 0.08s`; `git diff --check` passed. ShellCheck is not installed.
- Review: the scientific-change checklist found no threat-model/runtime changes. Safety corrections added exact-origin verification, safe absolute roots, full-SHA/run-ID validation, argv quoting without `eval`, launch locking, busy-GPU refusal, process identity checks before TERM, selective collection, and dry-run/collected-evidence cleanup.
- Deferred: Ferret read-only preflight, fixed-SHA prepare, CPU screen lifecycle, collection, cleanup dry-run, CUDA/DDP smoke, production, AutoAttack, and live W&B. No remote state was changed.
