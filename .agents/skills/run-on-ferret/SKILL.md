---
name: run-on-ferret
description: Safely prepare, launch, monitor, collect, cancel, and clean fixed-commit GPU experiments on the Ferret host. Use only when explicitly asked to operate a bounded Ferret run through SSH, nohup/setsid, and rsync.
---

# Run On Ferret

Use this skill only through `$run-on-ferret`. Keep planning, edits, Git commits, and analysis on the local host; Ferret receives a detached worktree for one full SHA and executes the supplied argv only.

Run scripts from the repository root. Configuration precedence is CLI arguments, environment, then the safe built-in defaults in `scripts/ferret-common`. Do not put credentials in command arguments or manifests.

```bash
.agents/skills/run-on-ferret/scripts/ferret-preflight
.agents/skills/run-on-ferret/scripts/ferret-prepare --sha <40-lowercase-hex> --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-launch --run-id <id> --gpus 0,1 -- python -m ard.cli.train --config <config>
.agents/skills/run-on-ferret/scripts/ferret-status --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-logs --run-id <id> --tail 200 --both
.agents/skills/run-on-ferret/scripts/ferret-collect --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-cancel --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-cleanup --run-id <id>          # dry run
.agents/skills/run-on-ferret/scripts/ferret-cleanup --run-id <id> --execute
```

## Required sequence

1. Run `preflight`; stop if its JSON reports `ready: false`.
2. Require an already-pushed, full 40-character SHA and run `prepare`.
3. Launch only an explicit argv following `--`; the skill does not parse or `eval` a command string.
4. Use `status` and bounded `logs`; collect small results before any cleanup.
5. Use `cancel` only for the named run. Cleanup is dry-run unless `--execute` is explicit.

The scripts reject unsafe run IDs, non-full SHAs, invalid/duplicate GPU sets, duplicate runs, and selected GPUs with active compute processes. They never change a branch checkout, auto-commit results, or remove a run outside the configured run root.

## GPU and experiment integrity

Pass physical indices such as `0`, `0,1`, or `0,1,2`. The launch manifest records physical indices and world size. Do not silently alter batch size, learning rate, schedule, seed semantics, attack settings, or DDP behavior; a three-GPU command is not automatically protocol-equivalent to the canonical two-GPU run.

`collect` excludes checkpoints, W&B offline data, caches, and bytecode by default. Use an explicit include option only after assessing storage and lineage needs.
