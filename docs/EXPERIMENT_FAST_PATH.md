# Experiment Execution Fast Path

## Purpose

This is the operational path for a short scientific campaign whose public
runtime already exists.  It shortens request-to-controller latency by removing
duplicate preparation work; it does not reduce source, lineage, attack, seed,
or evaluation validation.

The implementation is a mode of the existing Production Launch Gate, not a
second gate or controller.

## The only two operational profiles

| Profile | Use when | Launch target |
| --- | --- | ---: |
| `FAST_EXISTING_RUNTIME` | Existing public CLI, objective/trainer path, dataset, Teacher, parent form, attack, one-GPU job contract, and known Hamster/Ferret executor. | 30 min |
| `FULL_NEW_INTEGRATION` | A new objective/runtime, trainer path, DDP model, dataset loader, remote executor, checkpoint serialization, or artifact schema is involved, or the integration is genuinely uncertain. | 90 min |

Existing runtime defaults to Fast.  A declared integration change forces Full;
the gate rejects an attempt to launch it with `--fast-launch`.

## Fast sequence

One preparation pass resolves and records:

1. task context and the complete input inventory;
2. source, parent/config/Teacher/mask/calibration, dataset/split, attack,
   output, and W&B identities;
3. every host × job cell and planned GPU slot;
4. static CLI checks and representative exact smoke groups;
5. one immutable resolved manifest;
6. the existing detached controller.

Run the thin gate mode:

```bash
python3 .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --fast-launch
```

It performs `resolve → validate → static CLI → exact smoke → freeze →
detached controller launch`.  It leaves long-running completion, endpoint,
collection, aggregation, and report dependencies to the existing DAG.  Codex
does not poll a stable controller to advance its children.

## Required Fast proof

Fast still requires the following exact scientific identity inputs for every
job: source SHA, parent bytes, config SHA, Teacher SHA, dataset/split,
attack, seed, arm, output identity, and applicable mask/calibration SHA.
The resolved host × job matrix is written to `preflight.json`; job-specific
host-local path rebases are validated before any GPU reservation.

Each Fast spec must set `canary.require_exact_smoke: true` and define bounded
`static_cli` checks.  A missing path, source/config/parent drift, conflicting
GPU slot, or failed exact smoke fails the whole matrix before freeze.

## Smoke groups

`smoke_group` is operational coverage, not a scientific equivalence claim.
It may cover seed variants only when all members declare the same
`smoke_equivalence` object:

```json
{
  "public_cli": "ard.cli.train/v3",
  "output_semantics": "non-overwrite-attempt-scoped",
  "config_schema": "rslad-v5",
  "checkpoint_load_path": "resume-checkpoint-v2",
  "treatment_branch": "clean-wrong-a7"
}
```

The group must have one execution class.  Mixing Hamster/local and
Ferret/external jobs in a group fails closed.  Every job is still independently
source/parent/config/Teacher/input validated; the representative smoke proves
the shared executable branch only.

An exact external smoke may set `subsumes_remote_lifecycle: true` only when its
bounded JSON proof includes remote process confirmation, source, remote
manifest, completion marker, and an artifact that passes the existing staged
SHA-verified collection path.  In that case the gate skips the duplicate
generic lifecycle canary for that host.  It does not skip remote preflight,
identity, or collection validation.

## Manifest annotations

At the campaign root, optional operational declarations are:

```json
{
  "operational_profile": "FAST_EXISTING_RUNTIME",
  "integration_changes": {
    "new_trainer_execution_path": false
  }
}
```

At each equivalent job:

```json
{
  "smoke_group": "a7-local",
  "smoke_equivalence": {
    "public_cli": "ard.cli.train/v3",
    "output_semantics": "non-overwrite-attempt-scoped",
    "config_schema": "rslad-v5",
    "checkpoint_load_path": "resume-checkpoint-v2",
    "treatment_branch": "a7"
  }
}
```

These fields are operational metadata and do not change scientific identity.

## Timing ledger

The Fast report writes `fast-path-summary.json` beside the immutable manifest.
It records:

- `request_to_ready_seconds`;
- `request_to_controller_seconds`;
- static-check, smoke, preflight, and manifest durations;
- manifest freeze cycles; and
- controller launch attempts.

The standalone `launch_ledger.py` records the same duration fields when a
request needs a user-visible SLO ledger:

```bash
python3 .agents/skills/multi-gpu-experiment-orchestrator/scripts/launch_ledger.py \
  init --output <runtime>/orchestration/<campaign>/launch-ledger.json \
  --campaign-id <campaign> --requested-at <ISO-8601-with-offset> \
  --requested-at-precision exact --request-evidence <request-reference> \
  --operational-profile FAST_EXISTING_RUNTIME --strict-critical-path
```

If the 30-minute target is at risk, state the blocker, owning layer, smallest
resolution, and whether Full is actually required.  Do not silently expand
the campaign into infrastructure work.

Preparation timestamps are deliberately kept outside the immutable manifest.
They are operational evidence in `preflight.json` and
`fast-path-summary.json`, so an unchanged ordinary
`preflight -> canary -> launch` sequence can reuse its scientific freeze
instead of failing because wall-clock timestamps changed.

## Bounded verification

The operational Fast Path verification is recorded in
[`ard_experiment_execution_fast_path_v1.json`](experiments/ard_experiment_execution_fast_path_v1.json).
Its CPU-only bounded public-CLI dummy compared the former three-command
workflow to `--fast-launch` on the same fixture:

| Measure | Baseline `db7cca9` | Fast |
| --- | ---: | ---: |
| Operator commands | 3 | 1 |
| Preparation passes | 3 | 1 |
| Representative smoke executions | 2 generic | 1 exact |
| Freeze cycles | 3 | 1 |
| Controller launches | 1 | 1 |
| CPU dummy request-to-controller overhead | 0.356 s | 0.222 s |

This measures control-plane overhead only; it is not evidence that a real
scientific campaign will launch in under 30 minutes.  The latter still depends
on a complete known-input inventory, the host × job matrix, and the exact
smoke for the frozen campaign identity.

The bounded Hamster and Ferret checks used the final runtime source SHA,
including a Ferret live-process confirmation and collected-artifact SHA
round-trip.  They did not run a model, dataset, checkpoint, endpoint, or W&B
job.

## What Fast deliberately does not do

- It does not create campaign-specific shell wrappers.
- It does not use one worktree per seed, arm, or GPU.  Use a worktree only for
  actual source divergence or concurrent source work after a scientific SHA
  has been frozen.
- It does not add a new SSH executor, controller, or polling loop.
- It does not turn a failed scientific outcome into a retry.
- It does not auto-launch a follow-on scientific experiment after reporting.

For remote runners whose collector excludes the prepared repository worktree,
public commands must write to the registered run output root, not to a path
inside the code checkout.  Collection is evidence only after those bytes enter
the canonical local staging path and pass SHA verification.

See [Experiment Launch Discipline](EXPERIMENT_LAUNCH_DISCIPLINE.md) for the
incident evidence, escalation behavior, and estimate contract.
