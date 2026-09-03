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
.agents/skills/run-on-ferret/scripts/ferret-launch --run-id <id> --gpus 0,1 -- \
  /usr/bin/env PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.train --config <config>
.agents/skills/run-on-ferret/scripts/ferret-status --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-host-confirm \
  --run-id <id> --campaign-id <campaign> --job-id <job> \
  --identity-hash <64-hex> --source-sha <40-hex> --host ferret --gpu-index 0 \
  --expected-command-json '["/remote/python", "-m", "ard.cli.train"]'
.agents/skills/run-on-ferret/scripts/ferret-logs --run-id <id> --tail 200 --both
.agents/skills/run-on-ferret/scripts/ferret-collect --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-cancel --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-cleanup --run-id <id>          # dry run
.agents/skills/run-on-ferret/scripts/ferret-cleanup --run-id <id> --execute
```

## Required sequence

1. Run `preflight`; stop if its JSON reports `ready: false`.
2. Require an already-pushed, full 40-character SHA and run `prepare`. Git
   mutation in `prepare` is serialized by a per-repository lock, while
   independent prepared worktrees may run concurrently afterward.
3. Launch only an explicit argv following `--`; the skill does not parse or `eval` a command string.
4. For an external orchestrated job, use `ferret-host-confirm` once during the
   bounded launch window. It converts `ferret-status` into the identity-bound
   live-process/GPU/argv payload required for `host_confirmed_started`. Use
   `status` and bounded `logs`; collect small results before any cleanup.
5. Use `cancel` only for the named run. Cleanup is dry-run unless `--execute` is explicit.

The scripts reject unsafe run IDs, non-full SHAs, invalid/duplicate GPU sets, duplicate runs, and selected GPUs with active compute processes. They never change a branch checkout, auto-commit results, or remove a run outside the configured run root.

`ferret-status` records the fixed source SHA, PID, physical GPU indices/UUIDs,
recorded argv, and remote manifest path. These fields are evidence for the
production gate/orchestrator contract, not a substitute for terminal
completion. `ferret-collect` only transfers local result bytes; a production
collection node must SHA-verify them into the canonical local inventory before
aggregation consumes them.

For a parameter matrix, prefer one tracked wrapper at the prepared SHA or one
run bundle per cell. Do not embed a generated `bash -lc` loop in a launcher
argument: local expansion can silently change `$` variables before the command
is recorded. Validate config overrides with `--dry-run` before reserving a GPU.

## Placement and transfer decision

Before choosing a host, compare input locality and bytes, measured throughput,
idle GPUs, and expected runtime. Do not treat the current location as fixed:
use selective `rsync --partial --safe-links` when transfer time is smaller than
the expected compute saving. Transfer only required checkpoints/configs, verify
their SHA-256 at the destination, and record the execution host and transferred
artifact identity. Avoid duplicate computation when the same immutable input
can be transferred safely.

Do not add a training parity run ceremonially. First name the failure it could
detect and the decision its result would change. Prefer static Git/config,
checkpoint, teacher, environment, RNG, and hardware identity checks when a
short continuation cannot predict long-horizon equivalence. For cross-host
screens, keep primary contrasts within a host and replicate only a promising
effect if host sensitivity remains material.

## GPU and experiment integrity

Pass physical indices such as `0`, `0,1`, or `0,1,2`. The launch manifest records physical indices and world size. Do not silently alter batch size, learning rate, schedule, seed semantics, attack settings, or DDP behavior; a three-GPU command is not automatically protocol-equivalent to the canonical two-GPU run.

`collect` excludes checkpoints, W&B offline data, caches, and bytecode by default. Use an explicit include option only after assessing storage and lineage needs.

For this repository's CIFAR-10 single-GPU workload on Ferret, use the pinned
`/home/shunsukenaito/.conda/envs/adv/bin/python` environment (the system
`/usr/bin/python3` does not contain Torch), and begin with
`ARD_NUM_WORKERS=4`. A bounded 2026-08-01 teacher-response profile measured
387.4 images/s with 4 workers versus 338.4 with 8; see
`tools/internal/performance/provenance/ferret_workers_2026-08-01.yaml`. Treat
this as host/workload execution metadata, not a scientific hyperparameter, and
never change it inside an active run.
