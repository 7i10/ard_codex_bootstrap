# I100 Online-State S2 Launch Postmortem

## Scope and honesty boundary

This is an operational postmortem for the corrected attempt-11 launch of the
I100 Online-State S2 Preservation Screen. It does not change the scientific
campaign, its seeds, parents, attacks, or treatment arms. The production
manifest remains bound to source commit `bcb09a7` and manifest SHA
`5c9220f3ff246f156d7bd075eac2035e21663c358bd33297722a492d21c5130c`.

The original user-turn timestamp was not captured by the launch ledger. It is
therefore not valid to claim a full request-to-GPU latency. The numbers below
are the exact intervals recoverable from the immutable preflight and
orchestrator records. The recorded request time was written after repository
reconciliation and is explicitly not the original user request time.

## Measured launch latency

| Interval | UTC seconds | Interpretation |
| --- | ---: | --- |
| Recorded request marker → controller launch | 294.471 | 4 min 54.5 s; lower-bound operational latency after the ledger began |
| Gate start → controller launch | 69.699 | 1 min 9.7 s for preflight, checks, canary, freeze, and spawn |
| Public canary | 65.986 | 1 min 6.0 s; real checkpoint/data/teacher lifecycle canary |
| Controller launch → first root spawn | 0.055 | Prefix jobs became eligible immediately |

The recorded request marker is `2026-09-04T04:56:39.862570Z`; the controller
launch is `2026-09-04T05:01:34.333353Z`. Gate start was
`2026-09-04T05:00:24.634761Z`, and the successful canary ran from
`05:00:28.165416Z` to `05:01:34.151298Z`. The real user-request-to-launch
interval is **unknown**, not estimated.

## Timeline and evidence

| Event | Time (UTC) | Evidence |
| --- | --- | --- |
| Recorded request marker | 04:56:39.862570 | `launch-gate/preflight.json` timing ledger |
| Gate/preflight start | 05:00:24.634761 | same ledger |
| Static checks passed | 05:00:28.165383 | same ledger |
| Successful public canary started | 05:00:28.165416 | `launch-gate/canary.json` |
| Successful public canary finished | 05:01:34.151298 | `launch-gate/canary.json` |
| Manifest frozen | 05:01:34.154847 | manifest SHA above |
| Controller launched (PID 58117) | 05:01:34.333353 | preflight ledger / orchestrator state |
| Prefix roots spawned | 05:01:34.388671 / 05:01:34.389920 | orchestrator state |

After launch, only one bounded early verification was performed. The campaign
was left in its detached marker-driven DAG; no completion polling was used.

## What caused the delay

### 1. Attempt-10 contract bug (scientific compute did not start)

The child-arm command omitted the canonical `--prefix-state` argument. After
the orchestrator promoted the shared prefix, the child still referenced the
staging path sealed in the checkpoint. All six arm jobs failed immediately
with `shared e100 online state path/hash does not match checkpoint lineage`.

This was a real orchestration/lineage defect, not a model result. It was fixed
in `bcb09a7`, and the focused regression suite now asserts that every child
command contains the canonical promoted prefix-state path (`18 passed`).

### 2. Canary output collision on the first launch invocation

The first `--launch` invocation reran the canary into an already populated
canary output directory and was rejected by the runtime's non-overwrite guard.
The existing canary output was moved to a uniquely named archive directory,
then the same immutable manifest was launched successfully. No production GPU
job was started by the failed invocation.

### 3. One operator path typo

An initial gate command used `/home/islab/workspace-local/ard-runtime` instead
of the actual `/home/islab/workspace-local/shunsuke.naito/ard-runtime` path.
It failed before compute. This was an operator/environment resolution error,
not a scientific retry.

### 4. Source freeze correctly forced fresh prefix materialization

The child-path fix changed the source SHA, so the old attempt-10 prefix could
not be reused under the new immutable contract. Rebuilding the two Hamster
prefixes was slower than an unsafe reuse, but was the correct lineage decision.

## What went well

- The corrected source was committed before the attempt-11 manifest was built.
- The full public canary exercised prefix promotion, threshold generation, and
  all three child treatment paths before fan-out.
- Hamster GPU0 and GPU1 were reserved without overlap; the bounded snapshot
  showed 93–96% utilization.
- Dependency launches were automatic: prefix → threshold → D-BDD/PMP/Control
  child delays observed in the state ledger were 0.30–1.57 s, not hours.
- The campaign remains detached; Codex did not hold an active completion wait.

## Improvements already applied

1. Child commands bind the canonical promoted dependency output, not a staging
   path (`bcb09a7`).
2. A focused unit regression covers the prefix-state materialization contract.
3. The launch ledger records request marker, source readiness, canary, manifest
   freeze, controller launch, and root-spawn times.
4. The production manifest records an immutable source and manifest SHA.

## Improvements still needed (separate operational patch)

These are recorded for the reusable orchestrator and are intentionally not
changed while this campaign is running:

- Generate a unique canary namespace per gate invocation, or let a successful
  canary be explicitly adopted by `--launch`; this prevents the output-collision
  retry.
- Resolve runtime roots from one canonical host profile instead of hand-typed
  shell paths; preflight should print and validate the resolved path once.
- Validate every dependency-produced path in the child argv against the
  producer's promoted output before GPU reservation.
- Record the actual user-request event at the control-plane boundary, with
  precision, so future request-to-launch latency is complete rather than a
  lower bound.
- Keep the bounded early stable check, but never add routine completion polling
  to compensate for missing timestamps.

## Honest conclusion

For the corrected attempt-11 launch, the measurable post-reconciliation
request-marker-to-controller time was **4 min 54.5 s**. The gate itself took
**1 min 9.7 s**, dominated by the required **1 min 6.0 s** public canary. The
full delay from the user's original prompt cannot be reconstructed from the
available records. The principal avoidable delay was the attempt-10 missing
canonical dependency argument; the remaining extra time was a deliberate
lineage-safe rebuild plus one non-scientific canary/path mistake.

The production campaign is running under the corrected immutable manifest.
This document is an operational record only; it does not authorize a restart,
retry, endpoint change, or scientific decision.
