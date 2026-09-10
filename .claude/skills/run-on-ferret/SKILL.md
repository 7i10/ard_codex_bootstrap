---
name: run-on-ferret
description: Safely prepare, launch, monitor, collect, cancel, and clean fixed-commit GPU experiments on the Ferret host over SSH. Use only when explicitly asked to operate a bounded Ferret run; do not use for local Hamster jobs.
---

# Run On Ferret

Full protocol: `docs/FERRET_EXECUTION_PROTOCOL.md`.
Scripts live under `.agents/skills/run-on-ferret/scripts/` (do not move them; tests import by path):
`ferret-preflight`, `ferret-prepare`, `ferret-launch`, `ferret-host-confirm`, `ferret-status`, `ferret-logs`,
`ferret-collect`, `ferret-cancel`, `ferret-cleanup`, `ferret-common`.

## Command sequence

```bash
.agents/skills/run-on-ferret/scripts/ferret-preflight                       # stop if ready:false
.agents/skills/run-on-ferret/scripts/ferret-prepare --sha <40-hex> --run-id <id> \
  [--campaign-id <c> --job-id <j> --identity-hash <64-hex> \
   --execution-host ferret --expected-origin-host <remote-hostname>]
.agents/skills/run-on-ferret/scripts/ferret-launch --run-id <id> --gpus 0 -- \
  /usr/bin/env PYTHONPATH=src /home/shunsukenaito/.conda/envs/ard-v2/bin/python -m ard.cli.train --config <config>
.agents/skills/run-on-ferret/scripts/ferret-host-confirm --run-id <id> --campaign-id <c> --job-id <j> \
  --identity-hash <64-hex> --source-sha <40-hex> --host ferret --gpu-index 0 \
  --expected-origin-host <remote-hostname> --expected-command-json '["/remote/python", "-m", "ard.cli.train"]'
.agents/skills/run-on-ferret/scripts/ferret-status --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-logs --run-id <id> --tail 200 --both
.agents/skills/run-on-ferret/scripts/ferret-collect --run-id <id>
.agents/skills/run-on-ferret/scripts/ferret-cancel --run-id <id>            # only the named run
.agents/skills/run-on-ferret/scripts/ferret-cleanup --run-id <id>          # dry run
.agents/skills/run-on-ferret/scripts/ferret-cleanup --run-id <id> --execute
```

## Outputs to check

- `preflight`: JSON `ready` flag.
- `prepare`: remote manifest path; must record the same identity/origin values passed in.
- `launch`: recorded fixed source SHA, PID, physical GPU indices/UUIDs, argv, remote manifest path — evidence, not a
  substitute for terminal completion.
- `host-confirm`: converts `status` into the identity-bound live-process/GPU/argv payload required for
  `host_confirmed_started`.
- `collect`: only transfers local result bytes; SHA-verify into the canonical local inventory before aggregation
  consumes them. `collect` excludes the code tree, checkpoints, W&B offline data, caches, and bytecode by default —
  target the run's `outputs` area, never `repo/outputs`.

## Hard limits

- Require an already-pushed, full 40-character SHA. `prepare` mutates Git under a per-repo lock.
- Launch only an explicit argv after `--`; never a shell string.
- **`ferret --preflight` first, every time** — it does not check `loginctl Linger` (Ferret's Linger is `no`).
- Use `nohup`/`setsid` semantics via `ferret-launch`; never leave a process tied to an interactive shell.
- Collect results before running cleanup.
- Never silently change batch size, learning rate, schedule, seed semantics, attack settings, or DDP behavior; a
  3-GPU command is not automatically protocol-equivalent to a 2-GPU one.
- Use the pinned `/home/shunsukenaito/.conda/envs/ard-v2/bin/python` on Ferret (`/usr/bin/python3` has no Torch).
- For a parameter matrix, prefer one tracked wrapper at the prepared SHA or one run bundle per cell; do not embed a
  generated `bash -lc` loop in a launcher argument.
- For lower-level scheduling, use `multi-gpu-experiment-orchestrator`; for campaign freezing, use
  `production-launch-gate`. This skill does not implement a DAG controller.
