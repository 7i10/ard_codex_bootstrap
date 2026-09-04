# Experiment reconciler and single-owner postprocessing

`scripts/reconcile_experiment.py` is a small control-plane helper for a
detached experiment. It is not a second orchestrator and it does not replace
the production launch gate or the multi-GPU orchestrator.

## State location and authority

The canonical navigation record is
`<runtime>/runs/<experiment-id>/experiment-state.json`. It is derived from the
existing orchestrator state and terminal evidence; checkpoints, metrics,
completion markers, and scientific manifests remain authoritative evidence.
New campaigns use `schema_version: 2`, `mode: orchestrator_campaign`, and a
registered `orchestrator_state_path`.  The v2 bridge also records the manifest
SHA, required training job IDs, terminal result job IDs, recovery policy, and
postprocess owner.  Legacy states with `schema_version: 1` remain readable as
`single_process` states.  The state must contain `experiment_id`, `source_sha`,
`scientific_identity_hash`, and a `state` from the registered lifecycle.

For `orchestrator_campaign`, the orchestrator state is authoritative for
training status.  The reconciler does not inspect a local PID, GPU utilization,
or an inferred output directory to decide whether a remote job is complete.
When an orchestrator-owned downstream DAG is active, the reconciler records
that ownership and exits; it never launches a duplicate evaluator.

Training success is accepted only when the training process is terminal, the
launcher exit evidence is successful, the completion marker matches the
experiment/source/identity, and every registered expected output exists. A
dead PID alone is never success. GPU utilization is not a success criterion.
A launch begins in `LAUNCHING` and is promoted to `TRAINING` only by a
structured controller start proof; failed or malformed starts remain
`LAUNCH_FAILED`/`NEEDS_TECHNICAL_RECOVERY`.

## Scheduled wake

```bash
python scripts/reconcile_experiment.py \
  --experiment-id <id> --scheduled
```

The command reads one state file and its registered evidence only. On
`TRAINING` with a live PID, an active postprocess lease, or a terminal state,
it returns a fast `NO_OP`. It does not scan Git, download W&B data, evaluate a
checkpoint, edit scientific code, or poll until completion.

## Single-owner handoff

After valid training success, the command takes a short OS `flock` and a
durable expiring lease (`lease_id`, owner, start, expiry). It launches the
already-registered `postprocess_command` as a detached process exactly once.
The command receives the state path, lease ID, and completion/failure marker
paths through `ERT_*` environment variables. The existing postprocess DAG must
emit a matching marker after evaluation, aggregation, validation, and any
approved commit/push. A second scheduled wake sees the live lease or process
and returns `NO_OP`.

If the postprocessor crashes, an expired lease permits at most the registered
bounded retry (default two attempts) of the same command and identity. A
technical failure is never repaired by changing a seed, parent, attack,
coefficient, threshold, Teacher, or method. Exhausted or scientific failures
become `NEEDS_RESEARCH_DECISION` with compact evidence instead of an automatic
experiment change.

## Technical failure bridge

Only an orchestrator failure marker classified as a retryable technical
failure may invoke the pre-registered `recovery.command`.  The bridge passes
the experiment ID, state path, lease ID, and attempt number through `ERT_*`
environment variables and preserves source, parent, seed, attack, Teacher,
threshold, coefficient, and method identity.  Scientific, unknown, or
unregistered failures become `NEEDS_RESEARCH_DECISION`; they are never silently
repaired. Failure classification aggregates all required training and
downstream result jobs: scientific evidence wins over unknown/non-retryable
technical evidence, and retry is eligible only when every observed failure is
explicitly technical and retryable. Downstream evaluation/aggregation/report/
publish jobs remain owned by the orchestrator DAG; the reconciler does not
launch a duplicate.

## Example state fragment

```json
{
  "schema_version": 2,
  "experiment_id": "example-v1",
  "mode": "orchestrator_campaign",
  "scientific_identity_hash": "...",
  "source_sha": "...",
  "state": "TRAINING",
  "orchestrator_state_path": "/runtime/orchestration/example/state.json",
  "required_training_jobs": ["train"],
  "terminal_result_jobs": ["endpoint", "aggregate"],
  "training": {
    "pid": 12345,
    "completion_marker": "training/completion.json",
    "exit_evidence": "training/exit.json",
    "expected_outputs": ["training/checkpoints/last.pt"]
  },
  "postprocess": {
    "command": ["bash", "scripts/existing_postprocess.sh"],
    "completion_marker": "postprocess/completion.json",
    "failure_marker": "postprocess/failure.json"
  }
}
```

The postprocess command is an existing experiment-specific DAG/finalizer; the
reconciler intentionally does not implement evaluation, aggregation, Git, or
W&B behavior itself. A successful postprocess marker defaults to
`AWAITING_RESEARCH_REVIEW`; it may explicitly use `PUSHED` when that is the
registered terminal contract.
