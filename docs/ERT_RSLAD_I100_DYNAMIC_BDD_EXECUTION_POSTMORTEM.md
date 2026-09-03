# I100 Dynamic Boundary Distance Execution Postmortem

## Outcome

The scientific result was completed and recorded, but the campaign took much
longer than its compute budget required.  The user reported the initial prompt
at approximately 2026-09-03 14:00 JST.  The first production controller was
launched at 16:53:13 JST, a delay of about 2 hours 53 minutes.  The final
result commit was at 23:54:56 JST, about 9 hours 55 minutes after the reported
request time.

The request timestamp is user-reported and only hour-precision.  All other
times below come from Git commits, immutable controller state, or launcher
logs.  This document evaluates execution only; it does not change the Dynamic
BDD scientific interpretation.

## Evidence-backed timeline

| JST | Event | Evidence | Operational reading |
| --- | --- | --- | --- |
| ~14:00 | User requested the screen | user report | Start of the user-visible wall clock; not a repository timestamp. |
| 16:25 | Intervention implementation committed | `9b5cf132` | Core implementation was not yet ready during the first 2h25m. |
| 16:33 | Launch contract committed | `78ad9895` | Immutable launch material became available. |
| 16:53:13 | First production controller launched | `orchestration.state.json` | 2h53m after reported request. |
| 16:53–17:36 | Initial campaign failed/orphaned jobs | controller state and job logs | Direct non-executable shell wrappers, missing remote parent material, and a shared Git ref-lock race prevented a clean first wave. |
| 18:10 | First retry controller launched | `orchestration-retry.state.json` | A recovery was started after technical fixes. |
| 20:12–22:34 | Hash-bound recovery campaign | `recovery-436c920-r2` controller/logs | Valid remaining jobs and endpoints were recovered; S-BDD became numerically unsupported. |
| 22:50–23:01 | State-replay smoke iterations | state-replay controller logs | Real-checkpoint public-CLI smoke was correctly required, but needed three attempts to resolve output/lineage details. |
| 23:30 | First full state-replay campaign launched | state-replay controller state | It used a stale source SHA locally and directly invoked non-executable Ferret wrappers. |
| 23:33–23:47 | Corrected Hamster/Ferret state replays completed | recovery controller states | Both hosts completed the required checkpoint-only replays. |
| 23:54:56 | Final results committed | `5a45a41` | Report, machine artifact, and decision published. |

## What went wrong

| Finding | Evidence | Impact | Classification |
| --- | --- | --- | --- |
| No explicit launch-time target | First controller started 2h53m after the reported request. | Large unmeasured pre-launch delay. | Process gap |
| Direct non-executable shell wrappers | Two initial `PermissionError` traces and the first Ferret state-replay failure. | Controllers marked jobs stable although no remote GPU process existed. | Preventable technical failure |
| Incomplete remote preflight | Recovery audit records absent resolved config and a shared Git ref-lock race. | First remote campaign could not form a reliable parallel wave. | Preventable technical failure |
| Source freeze did not precede replay manifest creation | State-replay jobs expected `772a3ef` while the workspace was `21c4e1e`. | Hamster replay jobs failed before GPU computation. | Preventable lineage failure |
| Remote artifact locality was not normalized before aggregation | Endpoint summaries referenced Ferret's absolute paths. | Aggregation stopped although matching local rows had been collected. | Preventable analysis failure |
| State-replay inventory was incomplete | dev1 Control e104/e109 and e114 came from distinct registered artifacts. | Aggregator initially looked for all three under the smoke directory. | Preventable inventory failure |
| S-BDD became non-finite in both seeds | Corrected v2 run evidence in the results artifact. | No causal endpoint for S-BDD. | Scientific outcome, not an orchestration retry |

## Closure status

The original postmortem was documentation-heavy: it named the failures but did
not yet bind each one to a machine-enforced contract and executable regression.
The generic hardening recorded at the follow-up infrastructure commit closes the
operational findings below. `regression_protected` means the failure now blocks
before expensive compute or aggregation; it does not claim that scientific
inputs can never be missing.

| Failure | Root cause | Automatic prevention | Regression | Status |
| --- | --- | --- | --- | --- |
| Non-executable wrapper | Direct remote `*.sh` argv had no execute bit. | Remote launcher/probe metadata requires an executable wrapper or explicit interpreter. | R13 `test_remote_wrapper_metadata_rejects_non_executable_and_accepts_bash` | `regression_protected` |
| Incomplete Ferret preflight | Remote parent/config/Teacher/data/output evidence was partial. | Every assigned external host returns a complete source/Python/artifact/GPU/disk/output/wrapper record; unknown fails campaign preflight. | R14 `test_external_host_preflight_requires_every_declared_binding` | `regression_protected` |
| Local spawn mistaken for remote start | Controller observed a local wrapper process only. | Bounded remote PID/source/identity/GPU/argv/manifest confirmation is required before terminal probing. | R15 `test_external_probe_requires_host_confirmation_before_completion` | `regression_protected` |
| Stale source SHA | Manifest expectations were created before final source freeze. | Freeze/recheck source before external launch; drift requires a new manifest. | R16 `test_remote_source_drift_is_rechecked_before_launch` | `regression_protected` |
| Ferret ref-lock race | Concurrent prepare mutated one remote Git repository. | Per-repository prepare `flock` serializes Git mutation. | R19 `test_prepare_lock_serializes_concurrent_mutations` | `regression_protected` |
| Remote absolute artifact path | Aggregator consumed Ferret-only paths. | Collect into canonical local path, then verify SHA before inventory/aggregation. | R17 `test_collection_stages_remote_metadata_to_canonical_local_path` | `regression_protected` |
| Incomplete replay inventory | Required e104/e109/e114 cells were distributed across artifacts. | Required identity-bound matrix must be complete before aggregation. | R18 `test_inventory_rejects_missing_required_endpoint_cell` | `regression_protected` |
| Repeated replay smoke | Exit zero did not establish public replay/output/collection usability. | External campaigns require a bounded lifecycle canary with process, completion, collection, and SHA round trip before fan-out. | R20 `test_remote_lifecycle_canary_requires_status_and_hash_roundtrip` | `regression_protected` |
| Collection/hash uncertainty | A local copy could differ from remote bytes. | SHA equality is validated for every collected artifact. | R21 `test_inventory_rejects_collected_hash_mismatch` | `regression_protected` |
| Host/campaign artifact contamination | Artifacts lacked complete identity binding. | Inventory binds campaign/source/job/seed/arm/epoch/split/attack and rejects foreign cells. | R22 `test_inventory_rejects_foreign_campaign_or_source_identity` | `regression_protected` |

S-BDD remains **not applicable** to this closure table: it was a reproducible
scientific/numerical outcome, not a technical launch failure and not a retry
candidate.

## Agent-controlled changes

1. The orchestrator now rejects a locally present, non-executable `*.sh`
   command or completion probe at manifest validation.  A wrapper must either
   be executable or be invoked explicitly as `bash wrapper.sh`.
2. Every long campaign will have a compact time ledger: request time,
   source-freeze time, preflight completion, controller launch, first
   host-confirmed process, first completion marker, and aggregate/report time.
3. A remote job is not described as launched merely because the local
   controller spawned a wrapper.  The bounded stable check must confirm the
   remote run manifest/process and its expected GPU assignment.
4. The immutable manifest must be built only after the final production source
   commit.  Any later source change requires a new manifest and an explicit
   source-delta proof; it cannot reuse the old expected SHA by accident.
5. Remote endpoint rows must be collected into a canonical local location and
   SHA-verified before aggregation.  Aggregation must not depend on a remote
   host's absolute path.
6. One real checkpoint end-to-end replay remains mandatory, but its output
   path/schema/lineage check must be completed before the full replay manifest
   is generated.
7. Stable long-running work will remain in a detached completion-marker DAG.
   Codex will perform only launch and bounded stability checks, not completion
   polling.

## Prompt and launch contract for future experiments

These are operational additions, not requests to weaken scientific gates.

### Default Codex responsibility

For every production request, I will state in the launch response:

- the measured/requested start timestamp and its precision;
- source freeze, manifest, preflight, controller, and first actual-process
  timestamps as they become known;
- an end estimate based on one measured representative job plus endpoint and
  aggregation work;
- whether the estimate excludes a known scientific gate or external blocker.

If implementation and known inputs already exist, the working target is a
controller launch within 30 minutes.  If a new objective/runtime integration
is required, the target is within 90 minutes after reconciliation.  Missing
parents, unresolved scientific contracts, or failed canaries are not reasons
to bypass safety; they must instead be reported immediately as a launch-SLO
breach with a concrete blocker.

### Useful optional prompt fields

Future prompts can add the following compact block:

```text
Operational launch contract:
- Request a detached manifest/DAG; do not wait or poll after stable launch.
- Report request → source-freeze → preflight → controller launch → actual
  host-process timestamps, plus an end estimate that includes endpoints and
  aggregation.
- Preflight every assigned host for wrapper invocation, Python, data, Teacher,
  parent checkpoint/config SHA, output writability, GPU UUID, and disk.
- Freeze the source before creating the manifest.  If source changes, rebuild
  the manifest rather than retrying under its old SHA.
```

The science prompt should still own methods, parents, seeds, attacks,
endpoints, and STOP conditions.  The orchestration layer owns placement,
technical retries, dependency chaining, timing, and reporting.

## Next-campaign checklist

- [ ] Exact parent/config/Teacher and remote paths have one preflight record.
- [ ] Every launcher and completion probe uses an interpreter or has its
      executable bit checked.
- [ ] A source-frozen manifest contains every train, endpoint, aggregation,
      and report node before controller launch.
- [ ] One bounded remote run-manifest/process check verifies actual host/GPU
      startup; local wrapper spawn alone is insufficient.
- [ ] Endpoint row collection and aggregation consume canonical local,
      hash-verified paths.
- [ ] Launch response includes the end estimate and its assumptions.

## Remaining limits

No operational process can make an unimplemented method, an unresolved
scientific contract, missing parent artifact, or reproducible numerical
instability disappear.  The purpose of this protocol is to discover those
conditions before GPU reservation and to avoid adding manual delay once a
valid campaign is ready.
