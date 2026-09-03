# Orchestrator scientific-output ownership

## Incident

During the 2026-09-04 I100 S2 forensic replay recovery, the controller created
`<scientific-output>/orchestration/` before invoking the public replay CLI.
The replay correctly treats a pre-existing output directory as a non-overwrite
violation, so both initial Hamster replays stopped after input/model loading
and before producing scientific rows.

This was a technical orchestration failure, not a checkpoint, attack, model,
or numerical failure. No optimizer, scheduler, model parameter, checkpoint,
or scientific artifact was mutated.

## Root cause

The generic controller used `job.output_dir` for two different ownership
domains:

1. the scientific command's output namespace; and
2. controller logs, result records, and default completion markers.

That implicit assumption conflicts with valid public CLIs that fail closed
unless they themselves create a fresh output directory.

## Fix

The controller now stores logs, worker results, host-confirmation records, and
default completion markers under a sidecar rooted next to the immutable state
file:

```text
<state-parent>/orchestration/<campaign>/<manifest-sha>/<job-hash>/
```

It no longer creates `output_dir` before executing the argv. Explicit
technical-failure markers remain configurable in the scientific output
namespace so a job can emit them on failure; the controller does not precreate
that path either.

This also makes controller metadata manifest-specific, rather than merely
output-path-specific.

For an interrupted campaign created by the pre-sidecar controller, the new
controller still verifies and accepts its legacy output-root completion marker
and worker-result path. A sidecar upgrade therefore does not require
relaunching a scientifically valid completed job.

## Regression

`test_controller_never_precreates_or_pollutes_scientific_output` executes a
dummy public CLI that exits nonzero if its output directory exists before the
command starts. The test requires successful completion, confirms that the
scientific payload is the only content under that directory, and verifies that
the controller log is stored in the sidecar.

Existing stale-result protection remains covered by
`test_stale_result_from_prior_campaign_cannot_release_gpu_slot`, now against
the manifest-hash-bound sidecar result path.

## Operational rule

Before a fan-out replay, a real one-checkpoint public-CLI smoke must be run
from the final source and exact controller invocation. `--help`, unit imports,
or an earlier-source smoke cannot prove output-ownership compatibility.
