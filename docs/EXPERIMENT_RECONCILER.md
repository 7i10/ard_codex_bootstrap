# Experiment reconciler and single-owner postprocessing

`scripts/reconcile_experiment.py` is a small control-plane helper for a
detached experiment. It is not a second orchestrator and it does not replace
the production launch gate or the multi-GPU orchestrator.

## State location and authority

The canonical navigation record is
`<runtime>/runs/<experiment-id>/experiment-state.json`. It is derived from the
existing orchestrator state and terminal evidence; checkpoints, metrics,
completion markers, and scientific manifests remain authoritative evidence.
The state must contain `schema_version: 1`, `experiment_id`, `source_sha`,
`scientific_identity_hash`, and a `state` from the registered lifecycle.

Training success is accepted only when the training process is terminal, the
launcher exit evidence is successful, the completion marker matches the
experiment/source/identity, and every registered expected output exists. A
dead PID alone is never success. GPU utilization is not a success criterion.

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

## Example state fragment

```json
{
  "schema_version": 1,
  "experiment_id": "example-v1",
  "scientific_identity_hash": "...",
  "source_sha": "...",
  "state": "TRAINING",
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
