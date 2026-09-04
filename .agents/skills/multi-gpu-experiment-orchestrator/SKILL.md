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

For a human-authored campaign that needs source, parent, dataset, and config
validation plus an immutable resolved manifest before scheduling, use the
`production-launch-gate` wrapper. The gate delegates here after its preflight
and bounded canary; this skill remains authoritative for reservations,
detached workers, completion markers, retries, and resume.

## Workflow

1. Freeze a full Git SHA and a JSON manifest. Put all scientific identity in
   each job's `scientific_identity`; the controller hashes it and preserves it
   across retries.
   New production manifests must opt into the tracked workspace contract:
   `workspace_contract: {registry: <.../ard_workspace_v1.json>,
   enforce_future_writes: true}`. State, reservation, controller sidecar, and
   every future output path then fail closed unless they are below the
   canonical runtime root. Historical manifests may remain read-only without
   this opt-in.
2. Run `validate`, then `preflight`, then `plan` (all are read-only). Resolve
   host-local Python, data, teacher, output, and GPU paths before reserving a
   resource.
3. Put every known terminal node in the same manifest before launch:
   training, endpoint evaluation, collection, aggregation, and report. Then
   run `run` once. By default it starts a detached controller and returns; use
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

For campaigns launched through the production gate, the runtime-bound
schema-v2 `experiment-state.json` is the lightweight control-plane bridge.
The reconciler uses orchestrator state and registered job IDs as authority
for training/downstream status. It does not inspect a local PID or GPU
utilization to infer a remote result, and never launches an endpoint already
assigned to the orchestrator DAG.

## Rapid-launch discipline

For a short campaign whose scientific implementation and inputs already
exist, target controller launch within 30 minutes of a recorded request.  A
new runtime integration may need more time, but state the estimate and the
specific blocker before the target is exceeded; never hide the delay behind
routine status updates.

The production launch gate labels this `FAST_EXISTING_RUNTIME`; it is the
default for a known public CLI and known input forms.  A new runtime/trainer,
DDP, dataset loader, remote mechanism, checkpoint serialization, artifact
schema, or genuine uncertainty is `FULL_NEW_INTEGRATION` and uses the
90-minute integration target.  These are the only two operational profiles.
Fast is not a weaker profile: it retains exact source/input identity, static
CLI validation, representative exact smoke coverage, immutable freeze, and
this controller's preflight/run handoff.

Create a compact timing ledger before implementation with
`scripts/launch_ledger.py`.  Before controller launch, record evidence for:

1. the complete input inventory (parent, config, Teacher, mask, attack, and
   output roots);
2. the host × job config matrix, including every host-local path rebase;
3. the frozen source SHA;
4. host/GPU throughput, `estimated_work`, transfer cost, and a declared
   `work_unit` for every schedulable job; and
5. the immutable manifest.

`ready` rejects a ledger missing any of those records.  This is not a second
scientific gate; it prevents a late discovery of a host-only path mismatch.
Use `summary` in the launch handoff to report request-to-controller time and
the estimate's assumptions.

If one host/config cell fails, do not repair and launch the remaining cells
serially.  Revalidate the entire equivalent host × job matrix first, then
make one technical retry with a new attempt/run namespace.  Keep unrelated
infrastructure work out of the scientific critical path: improve it after a
stable launch or as a separate operational task.

## Manifest contract

See [references/manifest.md](references/manifest.md). A manifest contains a
campaign ID, full source SHA, host profiles, jobs, dependencies, output paths,
and retry policy. Commands are argv arrays, never shell strings. Host profiles
may expose measured throughput, GPU UUIDs, and required paths; values are
metadata, not scientific hyperparameters. `external_probe` jobs wrap an
existing launcher (for example `run-on-ferret`), a bounded
`host_confirm_probe`, an exact `remote_command`, and a completion probe. The
remote skill remains the authority for remote lifecycle and safety. The worker
distinguishes `controller_spawned` from `host_confirmed_started`: the latter
requires a matching live remote PID, source, scientific identity, GPU
index/UUID, remote manifest, and argv before the terminal probe can run.

Because argv is executed without a shell, write a non-executable shell wrapper
as `['bash', 'scripts/wrapper.sh', ...]`; do not put the wrapper itself at
argv position zero. Manifest validation rejects a locally present non-executable
`*.sh` command or completion probe before any GPU reservation.

Production manifests resolved by the launch gate carry
`production_schema_version: 2` and must declare an explicit `job_type` for
every node. Allowed roles are `training`, `evaluation`, `collection`,
`inventory`, `aggregation`, `report`, `finalization`, and `publish`. The gate
derives required training and terminal result sets from these roles when they
are not explicitly declared; unknown, missing, overlapping, or contradictory
declarations fail before launch.

The gate derives a runtime fingerprint from the resolved public CLI shape,
runtime/config/checkpoint contracts, output and marker semantics, execution
class, job roles, dependency topology, and dependency-output bindings. A Fast
signature is valid only when this fingerprint exactly matches a tracked entry
promoted by a passed exact bounded public-CLI smoke; an authored runtime ID
alone never grants Fast eligibility.

## Scheduling and safety

- Ready jobs are sorted longest-processing-time first. Candidate slots are
  scored by `transfer_seconds + estimated_work / throughput`; explicit host or
  GPU constraints are honored. GPU UUIDs are recorded when supplied. Put long
  parents on the fastest compatible GPU first; use a slower device for a short
  endpoint/materialization job only when that lowers its expected finish time.
- A campaign lock and in-state reservations prevent two controller processes
  from claiming one host/GPU. External processes are never killed; a busy
  resource must be rejected by the host executor/preflight.
- Completion markers include campaign, job, attempt, source SHA, and identity
  hash. Stale or foreign markers do not unblock a dependency.
- The controller does not create or write its own `orchestration/` directory
  inside a scientific `output_dir`. Logs, worker results, and default
  completion markers live in a manifest-hash-bound state sidecar, so a public
  CLI may safely require an initially absent output path.
- For a non-overwriting public CLI that can leave partial output on a
  technical failure, declare `attempt_scoped_output: {enabled: true}` and put
  `{attempt_output_dir}` in argv or environment. Each attempt writes a fresh
  staging namespace; only a successful attempt is atomically promoted to the
  canonical output. The canonical path is never overwritten.
- An external confirmation must be derived from the prepared remote manifest,
  not from local values injected into the response. Declare and validate
  `expected_origin_host` when the remote executor supports it; the controller
  rejects a mismatching independently observed remote hostname.
- Only a job-emitted JSON marker with `failure_class: technical` and
  `retryable: true` can trigger a retry. Accuracy, loss, or a scientific
  outcome is never a retry reason. Retries get a new attempt ID and retain the
  exact scientific identity; the first valid completion wins.
- State and events are atomically written. `status` values include `pending`,
  `running`, `completed`, `failed`, `blocked`, and `orphaned`. External event
  evidence additionally records `controller_spawned` and
  `host_confirmed_started`; an incomplete dependency blocks descendants rather
  than silently skipping it.
- At terminal state, the controller adds a timing summary to its state:
  parent-completion-to-child-launch delay, worker execution duration, planned
  throughput, and declared-work rate. The rate retains the manifest's
  `work_unit`; it is not implicitly an image/s claim. Read it with `status`;
  this is a completed-state read, not active monitoring.
- W&B settings are passed through the scientific command. The orchestrator
  does not enable model/run-bundle uploads or alter tracking policy.
- Terminal reporting may use the generic result publisher on the dedicated
  `experiment-results` branch. It writes one compact, revision-keyed event;
  canonical scientific result commits remain authoritative and are never
  merged by the publisher.

The launch bridge records `LAUNCHING` before invoking the detached controller
and changes to `TRAINING` only after a structured controller PID/start proof is
returned. A failed or malformed launch becomes `LAUNCH_FAILED` or
`NEEDS_TECHNICAL_RECOVERY`; a PID guessed from GPU activity is not evidence.

Campaign failure classification aggregates every failed, blocked, orphaned,
and required downstream node. Scientific evidence dominates unknown or
non-retryable technical evidence; retry is permitted only when all observed
failures are explicitly technical and retryable. The terminal publisher
verifies canonical result reachability and declared blob digests on push, is
collision-safe by full payload, and resumes a locally committed event after an
interrupted push or PR step. The normal DAG should contain its final `publish`
node; the reconciler is only a fallback for an already-registered owner.

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

Before asking the repository changed-test gate to run, preview its selection
once with `scripts/verify.py --changed --non-scientific --dry-run`. Some
non-scientific integration fixtures intentionally start synthetic training
subprocesses. A second full invocation is not a status check and must not be
used to make validation faster; run at most one selected full gate, or record
why focused tests are the smallest sufficient verification.

For request-to-launch evidence, use `launch_ledger.py init` with
`--strict-critical-path`. Existing runtime integrations default to a
30-minute controller-launch SLO; a new runtime/objective uses 90 minutes.
The ledger records `launch_slo_breached` automatically once the target is
missed, without weakening any source/lineage gate.
It also records request-to-ready, request-to-controller, static-check, smoke,
preflight, and manifest durations plus the number of freezes and controller
launch attempts.  One unchanged Fast preparation should have one freeze and
one launch attempt.

## Do not use / known limits

Do not use this skill for a single short command or to choose a scientific
winner. It does not discover checkpoints, infer parent equivalence, evaluate
metrics, or select thresholds. A remote command must be made safe by its
existing executor (normally `run-on-ferret`), and needs both host confirmation
and a terminal completion probe. This initial implementation supports one GPU
per job; multi-GPU/DDP jobs should be represented by the existing fixed-SHA
remote launcher as one externally managed job.
