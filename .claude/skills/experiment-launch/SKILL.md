---
name: experiment-launch
description: Take a plan from docs/plans/ to a running detached campaign — refuse an underspecified plan, pin the source worktree, author the campaign spec against it, then run the launch gate preflight → dry-run → canary → launch and hand off to the watcher. Use when the user says launch/start/run a campaign, arm, or plan number.
---

# experiment-launch

`$ARGUMENTS` is a plan path (`docs/plans/0087-*.md`) or a plan number (`0087`).
Resolve a number by globbing `docs/plans/$ARGUMENTS-*.md`. If it is ambiguous or
missing, ask once and stop.

## 1. Read the plan and refuse if it is underspecified

Require all three, quoted from the plan text: a **frozen scientific contract**
(arms, seeds, epochs, attack/augmentation identity, parents), a **preregistered
decision rule**, and a **stop rule**. Missing any → say which, and stop.

## 2. Pin the source

```bash
git status --porcelain          # must be empty
git rev-parse HEAD
python3 scripts/ardx/pin_source.py <sha> --json
```

Dirty tree → list the dirty files and stop. Record the printed worktree `path`
as `$WT`; every later path points there.

## 3. Author or refresh the campaign spec

For every job: `env` carries `WANDB_ENTITY` and `WANDB_PROJECT` from `configs/tracking/production.env` plus the
campaign's `WANDB_GROUP_*` (a detached controller inherits nothing from your shell); `command[1]`, `cwd` and `env.PYTHONPATH` resolve inside `$WT`;
`source.repo_path` and every `hosts.*.repo_path` also point at `$WT`. Declare
`operational_profile: FULL_NEW_INTEGRATION` unless the plan says otherwise. Put
every terminal node — training, endpoint evaluation, collection, aggregation,
report — in this one spec before the first gate run.

## 4. Gate, in order, with fresh directories

`GATE=<runtime>/runs/<campaign>-attemptN/launch-gate` (N = next unused attempt), plus a fresh canary output dir.

```bash
G=.agents/skills/production-launch-gate/scripts/launch_gate.py
python3 $G --campaign-spec <spec> --preflight-only --output-dir "$GATE"
python3 $G --campaign-spec <spec> --dry-run       --output-dir "$GATE"
python3 $G --campaign-spec <spec> --canary-only   --output-dir "$GATE"
python3 $G --campaign-spec <spec> --launch        --output-dir "$GATE"
```

On any gate error: collect **all** errors from that run, fix them together in
one edit pass, bump to attempt N+1 with a fresh `$GATE` and canary dir, and
restart from `--preflight-only`. Never fix one error and re-run immediately.

## 5. Confirm the handoff, then stop

```bash
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  status --manifest "$GATE/resolved-manifest.json"
systemctl --user is-active ardx-watch.service
```

The controller plus the root jobs must appear. If the watcher is inactive, tell
the user to run `scripts/ardx/install_units.sh --dry-run` then `--install`; do not
install it. If the session stays open, arm `Monitor` on `python3
scripts/ardx/campaign_watch.py --follow --interval 60 --roots <runtime>/runs`,
acting only on `kind=campaign` lines. Otherwise end the turn.

## 6. Record

Append one dated line to the plan's Progress log: campaign id, source SHA,
resolved-manifest SHA, gate dir. Commit **only** the plan file:
`Arm <campaign> from <sha12>`.

## Stop conditions

- Plan lacks a frozen contract, decision rule or stop rule.
- Working tree dirty at step 2, or `pin_source.py` exits 2.
- Gate still red after one batched fix cycle → report every error and stop.
- Infrastructure blocks a ready campaign for 30 minutes → write the sanctioned
  hand-run fallback command (from `$WT`, run-bundle contract, per CLAUDE.md
  rule 3), open a separate infra plan, and stop.
- A campaign for this plan is already `LAUNCHING` or running.

## Never

- Never run a job from this checkout; only from `$WT`.
- Never use `--fast-launch`, `--foreground`, or reuse an attempt's gate/canary dir.
- Never call `reconcile_experiment.py`, `publish_experiment_terminal_event.py`,
  `launch_ledger.py` or `task_context.py`.
- Never `sleep`, `watch` or poll GPUs/W&B; never start a second attempt while
  one is live; never weaken a gate check, tolerance or attack to make it pass.
