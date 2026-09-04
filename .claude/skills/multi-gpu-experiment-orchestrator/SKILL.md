---
name: multi-gpu-experiment-orchestrator
description: Execute an immutable multi-host experiment campaign from a manifest with host-aware GPU scheduling, completion-marker DAGs, technical-only retries, endpoint chaining, and resumable lineage. Use for independent runs, parent forks, and train-to-evaluation pipelines; not for a single short local command.
---

# Multi-GPU Experiment Orchestrator

Full contract: `docs/MULTI_GPU_EXPERIMENT_ORCHESTRATION_SKILL.md`. Manifest schema:
`.agents/skills/multi-gpu-experiment-orchestrator/references/manifest.md`.
Script: `.agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py` (do not move; tests import by path).

## When to use

A scientific prompt that already fixes method, seeds, parents, attacks, evaluation, and stop rule, and needs
execution only: preflight, resource placement, detached jobs, dependency transitions, retries, collection, audit
trail. It must not edit model/loss/attack/dataset/sampler/scientific configuration. For campaign authoring and
source/parent/config validation, use `production-launch-gate` first — it delegates here after its preflight/canary.

## Command sequence

```bash
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  validate --manifest campaign.json
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  preflight --manifest campaign.json
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  plan --manifest campaign.json --dry-run
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  run --manifest campaign.json                 # detached controller; do not poll
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  status --manifest campaign.json              # state read only, no side effects
```

`validate`/`preflight`/`plan` are read-only. `run` starts a detached controller and returns; use it again to resume
an interrupted controller (completed/running jobs are not relaunched). `--foreground` is only for a bounded
CPU/dummy test. `--once` runs one reconciliation tick for diagnostics, not for keeping this session alive.

## Outputs to check

`status --manifest campaign.json` reads the persisted state: per-job `status` ∈ {pending, running, completed,
failed, blocked, orphaned}, plus `controller_spawned`/`host_confirmed_started` for external jobs, and a terminal
timing summary (parent-completion-to-child-launch delay, worker duration, declared-work rate in the manifest's
`work_unit`). Put every terminal node (training, endpoint evaluation, collection, aggregation, report) in the same
manifest before the first `run`. Use `docs/decisions/*.md` for the human decision that follows a terminal state, per
`CLAUDE.md` rule 7.

## Hard limits

- **One GPU per job.** Multi-GPU/DDP work must be one externally managed job (normally `run-on-ferret` remotely).
- Do not poll — no `sleep`/`watch`/W&B polling/GPU queries in-session. End the turn; the detached controller and
  `ardx-watch.service` are the data plane.
- Commands are argv arrays, never shell strings; wrap in a non-executable `['bash', 'scripts/wrapper.sh', ...]`.
- Freeze a full Git source SHA; run only from a pinned worktree (`scripts/ardx/pin_source.py <sha>`).
- Only a job-emitted `failure_class: technical, retryable: true` marker may trigger a retry (new attempt ID, same
  scientific identity); a scientific outcome is never a retry reason.
- Do not use `launch_ledger.py`, `reconcile_experiment.py`, `publish_experiment_terminal_event.py`, or the
  fast-path/runtime-signature registry — frozen Codex-era control plane per `CLAUDE.md`.
- Remote commands go through `run-on-ferret`; require host confirmation and a terminal completion probe first.
- Not for a single short command or choosing a scientific winner; does not discover checkpoints, infer parent
  equivalence, evaluate metrics, or select thresholds.
