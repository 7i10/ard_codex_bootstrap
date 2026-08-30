---
name: multi-gpu-experiment-orchestrator
description: Execute immutable multi-host experiment campaigns from a manifest with host-aware GPU scheduling, completion-marker DAGs, technical-only retries, endpoint chaining, and resumable lineage. Use for independent runs, parent forks, and train-to-evaluation pipelines; do not use for a single short local command.
---

# Multi-GPU Experiment Orchestrator

Use this skill when a scientific prompt already defines the method, seeds,
parents, attacks, evaluation, and stop rule. This skill owns execution only:
preflight, resource placement, detached jobs, dependency transitions, retries,
collection/finalization, and an audit trail. It must not edit model, loss,
attack, dataset, sampler, or scientific configuration.

## Workflow

1. Freeze a full Git SHA and a JSON manifest. Put all scientific identity in
   each job's `scientific_identity`; the controller hashes it and preserves it
   across retries.
2. Run `validate`, then `preflight`, then `plan` (all are read-only). Resolve
   host-local Python, data, teacher, output, and GPU paths before reserving a
   resource.
3. Run `run` once. By default it starts a detached controller and returns; use
   `--foreground` only for a bounded CPU/dummy test. The controller launches
   every independent ready job, then advances children only after a validated
   completion marker. It also chains endpoint, aggregate, and report jobs when
   those are represented as dependencies.
4. After launch, perform at most a bounded stable check. Do not keep the Codex
   session alive with `sleep`, `watch`, W&B polling, or repeated GPU queries.
   The detached controller is the data plane and may reconcile its own worker
   state. `status` reads the persisted state without polling.
5. Use `run` again to resume an interrupted controller. Existing completed or
   running jobs are not relaunched. Collect remote artifacts through the
   existing `run-on-ferret` skill; do not duplicate its SSH/rsync logic here.

## Manifest contract

See [references/manifest.md](references/manifest.md). A manifest contains a
campaign ID, full source SHA, host profiles, jobs, dependencies, output paths,
and retry policy. Commands are argv arrays, never shell strings. Host profiles
may expose measured throughput, GPU UUIDs, and required paths; values are
metadata, not scientific hyperparameters. `external_probe` jobs wrap an
existing launcher (for example `run-on-ferret`) and provide a probe argv that
returns zero only when the remote run is complete. The remote skill remains the
authority for remote lifecycle and safety.

## Scheduling and safety

- Ready jobs are sorted longest-processing-time first. Candidate slots are
  scored by `transfer_seconds + estimated_work / throughput`; explicit host or
  GPU constraints are honored. GPU UUIDs are recorded when supplied.
- A campaign lock and in-state reservations prevent two controller processes
  from claiming one host/GPU. External processes are never killed; a busy
  resource must be rejected by the host executor/preflight.
- Completion markers include campaign, job, attempt, source SHA, and identity
  hash. Stale or foreign markers do not unblock a dependency.
- Only a job-emitted JSON marker with `failure_class: technical` and
  `retryable: true` can trigger a retry. Accuracy, loss, or a scientific
  outcome is never a retry reason. Retries get a new attempt ID and retain the
  exact scientific identity; the first valid completion wins.
- State and events are atomically written. `status` values include `pending`,
  `running`, `completed`, `failed`, `blocked`, and `orphaned`. An incomplete
  dependency blocks descendants rather than silently skipping it.
- W&B settings are passed through the scientific command. The orchestrator
  does not enable model/run-bundle uploads or alter tracking policy.

## Commands

```bash
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  validate --manifest campaign.json
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  preflight --manifest campaign.json
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  plan --manifest campaign.json --dry-run
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  run --manifest campaign.json                 # detached controller
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  status --manifest campaign.json              # state read only
```

`run --foreground --poll-interval 0.5` is intended for a bounded local test.
`--once` executes one reconciliation tick and is useful for diagnostics, not
for a Codex completion loop.

## Do not use / known limits

Do not use this skill for a single short command or to choose a scientific
winner. It does not discover checkpoints, infer parent equivalence, evaluate
metrics, or select thresholds. A remote command must be made safe by its
existing executor (normally `run-on-ferret`), and a remote completion probe
must be supplied explicitly. This initial implementation supports one GPU per
job; multi-GPU/DDP jobs should be represented by the existing fixed-SHA remote
launcher as one externally managed job.
