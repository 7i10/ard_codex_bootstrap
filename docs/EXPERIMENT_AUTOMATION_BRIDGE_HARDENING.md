# Experiment Automation Bridge Hardening

This milestone hardens the control plane without changing any scientific
training code or the active I100 campaign.

## Evidence-backed failure modes

The incident records show that wall time was lost to serial path discovery,
late host-path validation, manual parent-to-child handoff, endpoint launch
gaps, and repeated status checks after a process was already stable.  A Ferret
wrapper permission error and host-specific Python/data/Teacher paths were
technical orchestration failures, not scientific outcomes.  These facts are
recorded in `docs/EXPERIMENT_LAUNCH_DISCIPLINE.md` and the existing postmortem;
this change does not infer new timestamps.

## Capacity and instruction diagnosis

The repository is scientifically complex (multiple hosts, immutable
checkpoint lineage, endpoint attacks, and dependency graphs), but the incident
evidence does not show that the scientific method exceeded execution capacity.
The avoidable delay came from mixing three layers in one interactive turn:
scientific decisions, host/runtime preparation, and terminal result
postprocessing.  Long prompts did include the right safety constraints, yet
their operational steps were easy to execute serially and to revisit after a
late path discovery.  Codex also made operator errors: it assumed a wrapper
was executable, failed to resolve every host path before fan-out, manually
bridged parent to child, and continued status checks after stable launch.

Future prompts should keep the scientific section immutable and add a compact
operational header containing: request timestamp, existing-vs-new runtime
classification, complete host × job matrix, exact runtime signature, output
roots, and the stop condition.  The desired handoff is one sentence such as
“freeze the manifest, run the bounded exact smoke, launch the detached DAG,
then stop monitoring after stability.”  This reduces interpretation and
context churn without weakening any scientific gate.

## Implemented controls

- The launch gate carries operational postprocess/recovery/publish declarations
  into one immutable manifest and writes a runtime-bound schema-v2 bridge only
  for a real controller launch.
- `scripts/reconcile_experiment.py` accepts v1 and v2, uses the orchestrator
  state as authority for multi-job campaigns, avoids local-PID success claims,
  serializes leases with `flock`, and delegates only bounded technical retry.
- Fast Path requires a tracked validated runtime signature and binds the
  dependency topology to exact-smoke evidence.  The initial registry is empty;
  signatures must be added only after evidence exists.
- `scripts/publish_experiment_terminal_event.py` writes one compact,
  idempotent terminal pointer on the dedicated `experiment-results` branch.
  It never changes the scientific branch or automatically merges a PR.
- The orchestration state records failure classification on attempts so the
  bridge can distinguish technical retry from scientific failure.

## Minimal invocation

An existing-runtime campaign declares the operational fields once, then uses
the gate and detached controller:

```json
{
  "campaign_id": "example-v1",
  "operational_profile": "FAST_EXISTING_RUNTIME",
  "runtime_signature": {"id": "validated-runtime-id"},
  "workspace_contract": {"registry": "configs/workspace/ard_workspace_v1.json", "enforce_future_writes": true},
  "postprocess": {"owner_kind": "orchestrator_dag"},
  "result_publish": {"branch": "experiment-results"}
}
```

The signature must already be present in the tracked registry; the snippet is
illustrative and is intentionally rejected until that evidence exists.  After
the campaign's canonical result manifest is terminal, a result event can be
published with:

```bash
python3 scripts/publish_experiment_terminal_event.py \
  --result-manifest /runtime/results/terminal-result.json \
  --worktree /path/to/experiment-results-worktree --push
```

## Remaining limits

The bridge does not invent missing completion markers, recover an unregistered
postprocess command, or decide a scientific retry.  A runtime signature still
needs a project-specific exact smoke before production Fast launch can be
accepted.  Long-running data-plane work remains owned by the detached
orchestrator; the bridge intentionally performs no completion polling.

No active scientific process, checkpoint, W&B run, or source worktree was
modified by this milestone.
