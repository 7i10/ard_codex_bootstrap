---
paths: ["scripts/**", ".agents/**", "tools/**", "configs/workspace/**", "configs/operational/**"]
---
# Execution plane
GPU work is durable and LLM-free; `docs/CLAUDE_CODE_WORKFLOW.md` §3 is the design.
## Launch discipline
- Every GPU job runs from a pinned worktree: `python3 scripts/ardx/pin_source.py <sha>` →
  `<runtime>/worktrees/source-<sha12>`. Job `command`, `cwd` and `PYTHONPATH` point there, never at this
  checkout. (attempt-11 lost 8 jobs in 2.7 s because the live tree moved under a running DAG.)
- Each launch attempt gets a fresh gate `--output-dir` and a fresh canary output dir. Never reuse one.
- Collect **all** review findings, fix them together, then re-freeze once. No serial one-finding re-freezes.
- If infrastructure blocks a ready campaign for 30 minutes, run the science by the simplest sanctioned path
  (hand-run from the pinned worktree under the run-bundle contract) and open a separate plan for the defect.
- Path: `launch_gate.py` (preflight → dry-run → canary → launch) → `orchestrate.py` (detached DAG), or
  `run-on-ferret` remotely. `scripts/ardx/` adds pin, watch, status, postrun.
## The single completion contract
- **campaign**: terminal iff every job status ∈ `TERMINAL={completed,failed,blocked,orphaned}`; success iff all
  `completed`. `failure_class` aggregates the failed/orphaned/blocked jobs' last attempt
  (`attempts[-1].failure_class` / `.retryable`): any `scientific` → scientific; all technical AND retryable →
  `technical_retryable`; else `unknown`. The campaign-level `status` field is derived — log it, do not trust it.
- **hand-run**: success iff `run-bundle/completion.json` exists ∧ `manifest.status ∈ {completed, sync_pending}`
  ∧ `error-marker.txt` == `no application error recorded`. `failed` iff `manifest.status == failed`. `running`
  with `latest_progress.timestamp` older than `--stale-seconds` (default 3600) is `stale` — not terminal, not
  failed. Postrun additionally checks the manifest's `expected_outputs` exist before saying "completed".
- Do not invent a second completion signal, and never infer completion from a PID, GPU reading or log tail.
## Waiting
Never `sleep`, `watch` or poll GPUs in-session. Use `Monitor`, or end the turn and let `ardx-watch.service` →
`postrun_hook.sh` → headless `claude -p /experiment-postrun` do it. Both paths are idempotent (per-campaign
lock + non-overwriting `docs/experiments`).
## Frozen control plane — do not call, extend or depend on
`scripts/reconcile_experiment.py`, `scripts/publish_experiment_terminal_event.py` and the PR #1 event bus,
the fast path / runtime signature registry, `launch_ledger.py`, `task_context.py`. None ever ran in a real
campaign. Retire a rule in the same commit as the subsystem it describes.
## Hosts
- **Hamster** (local, 2×4090): `loginctl` Linger=**yes**, so `systemd --user` units survive logout — this is
  where the watcher lives.
- **Ferret** (ssh alias `Ferret`, 3×4090): Linger=**no** (`docs/debugging/0024`), so a `systemd --user` job
  there dies at logout; launch detached with `nohup`/`setsid` and collect by `rsync`. GPU0/1 ≈ 600 img/s but
  GPU2 is `SYS`/remote-NUMA at ≈ 376 img/s (≈ 425 bound to NUMA1): use GPU2 for short jobs or bind it.
