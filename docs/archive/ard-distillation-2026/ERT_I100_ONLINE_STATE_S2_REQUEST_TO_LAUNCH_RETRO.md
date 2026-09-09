# I100 Online-State S2: Request-to-Launch Retrospective

## Scope

This is a separate operational retrospective for the I100 Online-State S2
campaign. It does not alter the scientific manifest, running jobs, parents,
seeds, attacks, or treatment arms. The corrected production attempt remains
bound to source `bcb09a7` and manifest
`5c9220f3ff246f156d7bd075eac2035e21663c358bd33297722a492d21c5130c`.

## Honest elapsed-time statement

The user reported issuing the request at approximately **07:00 JST**. The
original user-turn timestamp was not captured by the control-plane ledger, so
this is not an exact machine timestamp. The corrected controller launch was
recorded at **14:01:34.333 JST**. On that user-reported basis, the
request-to-controller interval was approximately **7 h 1 min 34 s**.

The first machine-recorded repository evidence was commit `3352c41` at
07:05:45 JST. From that evidence to controller launch was **6 h 55 min 49 s**.
The launch ledger's later marker (13:56:39.863 JST) to controller launch was
**4 min 54.5 s**, but that marker was created after much of the work and must
not be presented as the full user-request latency.

## Evidence timeline (JST)

| Time | Evidence | Interpretation |
| --- | --- | --- |
| 07:05:45 | `3352c41` — fast existing-runtime launch path | First machine-visible evidence in this task |
| 07:15:43 | `46329e0` — immutable manifest reuse | Operational preparation continued |
| 07:23:02 | `39a6e7c` — fast-path verification | Verification/ledger work |
| 07:50:59 | `fcbebc6` — parallelize safe gates | Gate/DAG preparation |
| 09:40:25 | `265855c` — implement S2 screen | Scientific runtime implementation completed |
| 09:50:29–10:12:45 | `b122f27`, `cce21fe`, `8321e66`, `2332ef8`, `f810696`, `196b88b` | Prefix/static CLI/telemetry/environment/canary repairs |
| 13:27:01 | `7096a03` — preserve promoted prefix lineage | Lineage correction |
| 13:56:17 | `bcb09a7` — bind child to canonical prefix state | Required child-path fix |
| 14:00:28 | static CLI checks completed | Final preflight evidence |
| 14:01:34.333 | attempt-11 controller launched | First corrected production launch |

The interval 10:12:45–13:27:01 is **not fully timestamped** in the available
ledger. It is recorded as an unaccounted operational interval, not silently
classified as compute or user waiting time.

## What actually consumed the time

The delay was not one long GPU training phase. It combined:

1. A new/repair-heavy integration path was treated as if it were an already
   validated fast path. The runtime, public CLI, telemetry, prefix nodes, and
   canary contracts were repaired incrementally.
2. Attempt 10 launched before the child command was proven to bind the
   promoted prefix output. All six arm jobs then failed on the missing
   canonical `--prefix-state` lineage argument; no scientific compute began.
3. The first attempt-11 launch invocation reused a populated canary output and
   was rejected by the non-overwrite guard. The canary was archived and the
   unchanged manifest was launched again.
4. One preflight command used a hand-typed runtime path without the
   host-specific `.shunsuke.naito` component and failed before compute.
5. The source fix required fresh prefix materialization under the immutable
   source contract. Reusing the old prefix would have been scientifically
   unsafe, but the need was discovered late.

The corrected attempt itself was fast after the gate began: the public canary
took 65.99 s, the gate-to-controller interval was 69.70 s, and the two root
prefixes were spawned within 0.06 s of controller start. Dependency launches
in the corrected attempt were 0.30–1.57 s after their parent markers.

## Revert decision

**Do not revert `bcb09a7`.** It fixes a real causal-lineage defect and the
running attempt-11 manifest is intentionally bound to that corrected source.
Reverting it would reintroduce the attempt-10 failure and invalidate the
child lineage contract. `d438642` is documentation only. The correct action is
to retain both commits, add this explicit request-to-launch retrospective, and
push them; no history rewrite is justified.

## Improvements to apply before the next launch

These are operational safeguards, not scientific changes:

- Record a request event at the control-plane boundary before reconciliation;
  expose exact request-to-controller latency in the final handoff.
- Classify the campaign as `FAST_EXISTING_RUNTIME` or
  `FULL_NEW_INTEGRATION` before editing. A new public CLI, checkpoint
  serialization, or dependency node must use the longer integration budget.
- Resolve all runtime, dataset, Teacher, and output roots from a canonical
  host profile; reject hand-typed path variants in preflight.
- Build and validate the complete host × job matrix, including every producer
  output path, before reserving any GPU.
- Make canary namespaces invocation-unique, or explicitly adopt a successful
  canary instead of rerunning into an existing directory.
- Exercise an exact promoted dependency path in the canary and assert every
  child argv binds that path before launch.
- Freeze source and manifest once, then launch the detached DAG once. Technical
  retries receive a new attempt namespace but retain scientific identity.
- Keep bounded early verification (launch plus one early-progress check); do
  not replace missing timestamps with completion polling.
- Use the 30-minute controller-launch SLO for a known runtime and report the
  breached interval immediately with the concrete blocker.

## Prompt-writing contract for future requests

Every production prompt should state, near the top:

```text
Use the existing multi-GPU orchestrator and production launch gate.
Record request time immediately. Classify FAST_EXISTING_RUNTIME vs
FULL_NEW_INTEGRATION, freeze source/manifest before GPU launch, validate all
host-local paths and dependency outputs, launch the detached DAG once, perform
at most one bounded stable check, and do not poll until completion.
Report request-to-controller time, blockers, retries, and the completion
marker/endpoint chain. Do not change scientific identity on technical retry.
```

The prompt should separately identify the scientific contract (method, seed,
parent, attack, endpoint, stop rule) and the orchestration contract (host,
GPU, paths, dependencies, retry policy, completion conditions). It should not
ask for manual child launches or leave the endpoint/report outside the DAG.

## Actionable target

For the next known-runtime campaign, the target is:

- request marker immediately at receipt;
- complete inventory and host matrix before implementation;
- one source freeze and one launch attempt;
- controller launch within 30 minutes unless a named integration blocker is
  recorded;
- no routine post-stable polling by Codex.

This retrospective records the failure honestly; it does not claim that the
unaccounted interval has been explained or that future launches are already
perfect.
