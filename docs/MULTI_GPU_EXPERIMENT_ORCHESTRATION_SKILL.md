# Multi-GPU Experiment Orchestration Skill

## Summary

`multi-gpu-experiment-orchestrator` is a generic execution-layer skill. It
does not know about RSLAD, a dataset, a teacher, an attack, or an augmentation
method. The scientific prompt supplies those immutable job commands and
identity fields; the skill schedules and records them.

## What it reuses

Remote Ferret lifecycle remains delegated to
[`run-on-ferret`](../.agents/skills/run-on-ferret/SKILL.md), including fixed-SHA
worktrees, SSH/rsync, GPU checks, cancellation, and collection. The new
controller only supplies a manifest/DAG and can invoke that existing launcher
as an `external_probe` job. It does not duplicate remote execution logic.

## Failure modes addressed

The prior orchestration audit recorded wrong Python and host paths, missing
environment variables, GPU double-booking, manual parent-to-child delay, and
training completion without endpoint evaluation. The skill addresses these by
preflight validation, cross-campaign host/GPU lock reservations, immutable manifests,
marker-triggered dependencies, endpoint jobs represented in the same DAG, and
technical-only retries.

Measured Hamster/Ferret speeds are not hard-coded. Host profiles may carry
throughput and transfer metadata, and placement minimizes
`transfer_seconds + estimated_work / throughput`. GPU UUIDs are recorded when
available. Host/job `required_paths` and `required_env` are checked during
local preflight. External user processes are never killed.

## DAG and lifecycle

Independent jobs are launched in longest-processing-time order. A successful
worker atomically writes a marker containing campaign, job, attempt, source
SHA, and scientific identity hash. A child is eligible only after every
dependency has a valid marker. Endpoint, aggregation, and report jobs are
ordinary DAG nodes, so training completion alone does not complete a campaign.

State is locked and atomically updated. Re-running the same manifest recognizes
completed/running jobs and does not launch duplicates. A detached controller
survives the Codex session; `status` is a read-only state view and `run` resumes
the controller when needed.

Controller metadata is deliberately separated from scientific output. Logs,
worker results, host-confirmation records, and default completion markers are
written to a state-sidecar keyed by campaign and manifest SHA. The controller
does not pre-create `output_dir`, so a public CLI may own a fresh,
non-overwriting scientific output namespace.

Only an explicit JSON failure marker with `failure_class: technical` and
`retryable: true` permits a retry. A retry receives a new attempt ID but the
same source, config, seed, parent, attack, and method identity. Accuracy or a
scientific result never triggers a retry or winner selection.

## Bounded monitoring

Codex performs validation and a bounded launch/stability check, then returns.
The detached controller reconciles its own worker state; there is no Codex
`sleep`/`watch`/W&B completion loop. Existing `run-on-ferret` commands remain
the authority for remote status and process safety.

## Timing and complete host configuration

For an existing short runtime, the operational target is controller launch
within 30 minutes of a recorded request.  The skill's `launch_ledger.py`
records request, complete input inventory, host × job config matrix, frozen
source, manifest, controller, and host-confirmed timestamps.  `ready` fails
if the pre-launch evidence is incomplete, and `summary` reports the actual
request-to-controller delay.

The matrix is intentionally job-specific: a generic remote host preflight
does not prove that every resolved config can open its Teacher, parent, data,
and output path on that host.  After any host/config mismatch, the whole
equivalent matrix is revalidated before a fresh technical retry.  This avoids
serial one-checkpoint fixes and keeps infrastructure changes off the active
scientific critical path.

## Usage

```bash
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  validate --manifest campaign.json
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  preflight --manifest campaign.json
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  plan --manifest campaign.json
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  run --manifest campaign.json
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py \
  status --manifest campaign.json
```

`run --foreground` is reserved for a bounded CPU/dummy integration test.
Manifest details are in
[`references/manifest.md`](../.agents/skills/multi-gpu-experiment-orchestrator/references/manifest.md).

## Validation and limitations

The skill has CPU-only dummy tests covering independent roots, dependency
forks, endpoint/aggregation/report chaining, idempotency, technical retry,
cycle/missing-dependency rejection, unavailable GPU constraints, stale result
isolation, and output-namespace ownership. The repository's changed
non-scientific gate also passes.

It does not infer parent equivalence, discover checkpoints, evaluate metrics,
or select a scientific treatment. One-GPU-per-job is supported directly;
multi-GPU/DDP remains an externally managed fixed-SHA launcher job. A remote
completion probe must explicitly distinguish successful terminal state from a
failed run; `probe_timeout_seconds` can bound an external probe. Remote lock
domains and remote path checks remain the responsibility of the existing
remote executor.
