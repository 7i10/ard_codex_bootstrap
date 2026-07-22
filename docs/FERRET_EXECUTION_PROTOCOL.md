# Ferret execution protocol

Hamster remains the planning, editing, review, and Git integration node. Ferret is an execution node: it runs an immutable Git commit in a detached worktree and returns artifacts over SSH/rsync. Do not edit source on Ferret or run another Codex there.

```text
Hamster repository --commit/push--> GitHub --fetch SHA--> Ferret detached worktree
       ^                                                   |
       +---------------- selective rsync ------------------+
```

## Prerequisites and configuration

Hamster needs OpenSSH and rsync; Ferret needs Git, `nohup`, `setsid`, Python, CUDA, and the experiment dependencies. BatchMode public-key authentication and host-key verification must work before mutation. Do not enable agent forwarding or store keys, tokens, checkpoints, datasets, or W&B offline data in Git.

Configuration precedence is CLI option, environment variable, then built-in default. Source [the example environment](../configs/remote/ferret.example.env) only after checking every path. The current intended source repository is `/home/shunsukenaito/workspace-local/ard_codex_bootstrap` on Ferret, but preflight remains authoritative.

Run layout:

```text
$FERRET_RUN_ROOT/<run-id>/
  repo/                 detached fixed-SHA worktree
  outputs/              experiment output root
  logs/                 stdout.log and stderr.log
  control/              launch metadata, timestamps, exit code, locks
  manifest.json
  status.json
```

Run IDs match `^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$`; `..` is rejected. A full 40-hex Git SHA is mandatory. Existing runs are never overwritten.

## Explicit skill workflow

Invoke `$run-on-ferret` explicitly; implicit invocation is disabled. The skill maps each operation to the executable of the same name in `.agents/skills/run-on-ferret/scripts/`.

1. `ferret-preflight` performs read-only identity, tools, GPU, disk, repository, worktree, run-root, W&B-variable-name, dataset, and checkpoint checks and emits JSON. A false readiness result blocks prepare/launch.
2. `ferret-prepare --sha <40-hex> --run-id <id>` fetches origin, verifies the commit, creates a new run directory and detached worktree, and writes the initial manifest. It does not launch.
3. `ferret-launch --run-id <id> --gpus 0,1 --launcher direct -- <argv...>` records argv and GPUs, refuses busy GPUs and duplicate launch, then starts one detached `nohup setsid` process group. User argv is never passed through `eval`.
4. `ferret-status --run-id <id>` combines the manifest, verified process identity, exit code, GPU state, logs, and output inventory into a normalized state.
5. `ferret-logs --run-id <id> --tail 200 --both` returns bounded logs. Follow mode is explicit and intended for a human terminal, not indefinite agent monitoring.
6. `ferret-collect --run-id <id>` rsyncs small lineage/results by default. Checkpoints, W&B offline files, or all files require their explicit include flag.
7. `ferret-cancel --run-id <id>` targets only the recorded process group after PID, owner, start marker, cwd, and command checks; it sends TERM first and never kills unrelated user processes.
8. `ferret-cleanup --run-id <id>` is a dry-run by default. Execution requires a terminal run, collection evidence, and an explicit execute option; paths must resolve below the configured run root.

Use `--help` on each executable for the exact supported arguments. Commands are argv after `--`, not one shell string. Put environment variables in the experiment config or an intentionally prepared wrapper; never interpolate credentials into command arguments or logs.

## GPU profiles

One independent GPU run selects one physical index. Two-GPU DDP for the current global-batch-128 protocol uses per-rank batch 64:

```bash
ferret-launch --run-id chen-s0 --gpus 0,1 --launcher direct -- \
  /home/shunsukenaito/.conda/envs/adv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module ard.cli.train \
  --config configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml
```

Three GPUs are supported mechanically, including three independent single-GPU runs. They are not scientifically interchangeable with the existing two-GPU protocol: global batch 128 is not divisible by three, local BatchNorm statistics differ, and the skill will not alter batch size, learning rate, scheduler, attack, seeds, or accumulation to compensate. A three-GPU DDP protocol requires a separately reviewed config.

## Failure recovery and safety

- `prepared` has no launch; `running` requires live PID/start-marker evidence; an absent process without exit code is `orphaned`; exit zero/nonzero maps to `completed`/`failed`.
- For an orphan, inspect status, bounded logs, process ownership/start time, and run-directory command before canceling. Never act on a bare PID.
- Busy GPUs block launch. Do not stop another user's process or bypass the check.
- Collection excludes `*.pt`, `*.pth`, `*.ckpt`, `wandb/`, caches, and bytecode by default. Validate the collected manifest SHA/run ID before analysis. Collection never commits automatically.
- Cleanup never targets running jobs and never defaults to deletion. Do not clean until required checkpoints and W&B offline data are durably collected or synchronized.
- Production, checkpoint downloads, full AutoAttack, and live W&B are outside remote-skill validation. The bounded validation sequence is read-only preflight, prepare-only, detached CPU smoke, status/logs/collect, cleanup dry-run, then optional short CUDA/DDP smoke.

## Verified host

On 2026-07-23 BatchMode SSH authenticated through the `Ferret` alias to `islab-3gpu` as `shunsukenaito`. Read-only preflight verified three idle RTX 4090 GPUs, the exact GitHub origin, the synchronized repository, `/usr/bin/python3`, `nohup`, `setsid`, `flock`, and more than 1 TB free disk.
