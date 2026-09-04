# ARD research platform — Claude Code contract

Single-teacher Adversarial Robustness Distillation (RSLAD family, CIFAR-10, robust WRN teachers).
Scientific correctness, traceability and fair evaluation outrank speed and headline numbers.
Design rationale and runbook: `docs/CLAUDE_CODE_WORKFLOW.md`. Full invariants: `docs/SCIENTIFIC_INVARIANTS.md`.
Path-scoped rules load automatically from `.claude/rules/`. The Codex-era rulebook (`AGENTS.md`, `docs/CODEX_WORKFLOW.md`,
`IMPLEMENTATION_PROMPT.md`, `.codex/`) was removed on 2026-09-05; read it from git history before commit d1c0053 if needed.

## Three planes

- **Scientific core**: `src/ard/` (engine, objectives, attacks, policies, signals, state, tracking), `configs/`. One YAML per run.
- **Execution plane** (durable, no LLM): `launch_gate.py` → `orchestrate.py` (detached DAG) on Hamster (local, 2×4090)
  and `run-on-ferret` (ssh alias `Ferret`, 3×4090). Completion is read from `state.json` / `run-bundle`, never inferred
  from PIDs or GPU load. `scripts/ardx/` adds pinned-worktree creation, the campaign watcher, status and the postrun hook.
- **Agent plane**: this session, `.claude/agents/*`, `.claude/skills/*`. Frozen and NOT to be used or extended:
  `scripts/reconcile_experiment.py`, `scripts/publish_experiment_terminal_event.py` (PR #1 event bus), fast path /
  runtime signatures, `launch_ledger.py`, `task_context.py`.

## Daily commands (prefer the skills; they encode the checked procedure)

- `/experiment-status` — hosts, campaigns, hand-run bundles, pending decisions.
- `/experiment-launch <plan>` — spec → gate preflight/dry-run/canary/launch from a pinned worktree, then arm the watcher.
- `/experiment-postrun <campaign>` — verify completion, aggregate, import record + report, close plan, commit, write decision packet.
- `/experiment-decide` — turn results into `docs/decisions/NNNN-*.md` and stop.
- `/verify` — `scripts/verify.py --changed` with the project Python. `/ard-bug-hunt` — unexplained failures.
- Python is `/home/shunsukenaito/.conda/envs/adv/bin/python` (bare `python` does not exist). `PYTHONPATH=src`.
- First session in a checkout: accept the trust dialog once interactively, or `.claude/settings.json`
  `permissions.allow` is ignored and every Bash call prompts. Trust is keyed on the exact repo-root spelling,
  so the `/home/shunsukenaito/...` symlink path needs its own accepted entry.
  Check: `grep -A2 ard_codex_bootstrap ~/.claude.json | grep hasTrustDialogAccepted`.

## Hard rules (short form; details in `.claude/rules/`)

1. GPU jobs run only from a pinned worktree (`scripts/ardx/pin_source.py <sha>`), never from this checkout.
2. Each launch attempt gets a fresh gate `--output-dir` and canary output. Batch all review fixes before one re-freeze.
3. If infrastructure blocks a ready campaign for 30 minutes, run the science by the simplest sanctioned path
   (hand-run from the worktree with the run-bundle contract) and open a separate plan for the infra defect.
4. Never `sleep`/poll in-session. Use `Monitor`, or end the turn and let `ardx-watch.service` + the headless postrun act.
5. The first action after any campaign ends is: import the record, close the plan, commit, write the decision packet.
   Results docs are generated — the aggregator emits the JSON record and the Markdown report from one dict;
   only decisions and interpretation are hand-written.
6. Never silently change epsilon, steps, step size, random start, normalization, temperature, schedule, checkpoint selection
   or evaluation attacks. Never weaken an attack, a tolerance or a guard to make something pass. Clean and robust accuracy
   are reported separately; best and last checkpoints are both kept; AutoAttack runs only from a saved checkpoint in a
   separate process with `--allow-autoattack`.
7. Scientific decisions (new arm, seed, epoch horizon, official test, promotion) are the human's. Write a decision packet and stop.
8. Do not add a rule here without removing one; retire rules together with the subsystem they describe.

## Models and effort

Default main thread: Opus / xhigh. Switch to Fable / xhigh only for new scientific contracts, the consolidated scientific
review of a milestone, and unexplained failures. Subagents: reviewer and bug-investigator on Opus, mechanical work on Sonnet.
Headless postrun: Opus / high with bounded turns. Claude chat and Claude Code share one quota; separate sessions only for
context hygiene (new campaign → new session, `/compact` when long).

## Git

Commit at milestone boundaries with the plan updated; never commit `.external/`, outputs, caches, W&B data, checkpoints.
Push and history rewrites only on explicit request. Commit messages end with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (or the model in use).
