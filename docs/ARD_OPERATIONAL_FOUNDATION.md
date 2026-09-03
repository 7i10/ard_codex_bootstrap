# ARD operational foundation

## Scope and outcome

This operational-only milestone was completed after the I100 S2 forensic
result had already been recorded at `a8b9d4e089bf45977485b085d826fc06928c8959`.
It standardizes where future runtime state is written, hardens the generic
launch path, and records a small amount of durable recovery context. It did
not change a model, objective, attack, checkpoint, seed, experiment result,
or W&B retention policy.

The implementation was frozen at
`311efd9949a9d46d0723018fbcac967b6222bca8`. A bounded CPU-only operational
verification then exercised that exact source on Hamster and Ferret. No
scientific training or endpoint evaluation was started.

## Forensic incident: cause and corrected gate

The forensic replay exposed four separate operational failures:

1. A new public CLI had an import-path error that reached launch preparation.
2. The controller pre-created `scientific_output/orchestration/` before a
   public CLI that correctly requires a fresh output directory could start.
3. The first local remedy made the scientific CLI tolerate controller metadata
   instead of correcting the controller that created it.
4. A stale orchestration result from a prior campaign/attempt could be seen in
   a reused namespace.

The former gate checked important scientific lineage but did not bind the
final public command, its source, output ownership, remote observed identity,
and attempt namespace tightly enough. The repaired ordering is:

```text
source implementation → source commit → static CLI check →
exact public-CLI smoke → frozen manifest → complete host preflight → fan-out
```

Any source, command, config, parent, orchestrator, schema, execution-class,
or fan-out-group change invalidates the exact smoke. A host-only mismatch
requires complete host × job revalidation before one technical retry; it is
not a reason to weaken an attack, parent, output rule, or scientific CLI.

## Canonical workspace contract

The tracked authority is
[`configs/workspace/ard_workspace_v1.json`](../configs/workspace/ard_workspace_v1.json).
It defines one logical repository, dataset, and future runtime root. Future
runtime writes are allowed only under the registered runtime root:

```text
runs/  analysis/  staging/  worktrees/  orchestration/
task-context/  locks/  tmp/
```

Use `ard.workspace.load_workspace_contract()` in Python and:

```bash
ARD_PYTHON="$(python3 scripts/workspace_paths.py python)"
"$ARD_PYTHON" scripts/workspace_doctor.py --json
```

for a compact check of the current host realization. The doctor intentionally
warns about historical roots; it does not reinterpret them as migration or
deletion candidates.

Historical roots remain read-only until an inventory proves otherwise. The
cleanup record retained all referenced `ard-runs`, `ard-analysis`,
`ard-campaign-runs`, and historical Ferret collection roots. It removed only
a clean detached worktree, six unreferenced temporary wrappers, and empty
legacy lock roots after process, cwd, open-file, lock, and worktree checks.
See
[`ard_workspace_cleanup_v1.json`](experiments/ard_workspace_cleanup_v1.json).

## Layer ownership and first remediation

An execution failure is fixed first in the layer that owns the failed
contract. This is deliberately operational rather than a reason to relax a
scientific CLI.

| Failure | First owner | Correct first action |
| --- | --- | --- |
| Source, config, parent, Teacher, mask, or attack identity mismatch | production launch gate | Fail before manifest freeze; repair the registered input. |
| Host path, Python, data, Teacher cache, output root, or GPU mismatch | host profile / remote executor | Revalidate the complete host × job matrix before a technical retry. |
| Controller creates a path the public CLI requires to be absent | orchestrator | Put controller state in the sidecar; keep the scientific output namespace CLI-owned. |
| Partial retry output exists | orchestrator attempt layer | Write to a fresh attempt namespace and atomically promote only after success. |
| Remote identity or origin cannot be independently observed | `run-on-ferret` | Block terminal completion until the remote prepared manifest and observed host agree. |
| Exact public CLI smoke is invalid | production launch gate | Re-run the same public CLI with the frozen inputs; do not weaken its validation. |

The original public-CLI issue was an output-ownership mismatch: controller
metadata was pre-created in a path owned by a non-overwriting scientific CLI.
Relaxing the CLI would have weakened a useful safety property and hidden
partial-output reuse. The generic repair is controller-sidecar ownership,
not an experiment-specific bypass.

## R23–R35 guards

The historical R1–R22 regression registry now continues through R35 in
`tests/skills/test_production_launch_gate.py`. R33 is a documented ownership
policy whose observable boundary is exercised by R29.

| Guard | Protection |
| --- | --- |
| R23 | Remote confirmation reads identity and origin from the prepared remote manifest, not local reinjection. |
| R24 | A registered remote origin must be independently observed before confirmation. |
| R25 | Collection stages data locally before canonical promotion. |
| R26 | Launch ledger rejects incomplete prelaunch evidence and measures request-to-launch time. |
| R27 | Critical-path launch-SLO breaches are recorded automatically. |
| R28 | Future writes outside the registered runtime root fail closed. |
| R29 | Controller logs, state, and default markers never pollute a scientific output directory. |
| R30 | A static public-CLI failure blocks manifest freeze. |
| R31 | An exact public-CLI smoke binds source, argv, config, parent, execution class, and fan-out group. |
| R32 | A stale prior-campaign result cannot release a current GPU reservation. |
| R33 | Remediation starts with the owner of the violated layer, not by weakening another layer. |
| R34 | Retries use attempt-scoped output and keep partial data outside the canonical namespace. |
| R35 | Equivalent fan-out cannot launch without an exact source-bound public-CLI smoke. |

## Launch and recovery workflow

1. Create a runtime-only task context with `scripts/task_context.py init`.
2. Freeze the source and complete input inventory, then create the immutable
   manifest through the production launch gate.
3. Record the host × job path/config matrix in `launch_ledger.py`; do not
   repair hosts serially after the first failed cell.
4. Run the orchestrator `validate`, `preflight`, and `plan`, then launch its
   detached controller once.
5. Do one bounded stable check. Completion, child jobs, endpoints,
   aggregation, and report nodes advance through completion markers rather
   than Codex polling.
6. Use a technical retry only when the marker explicitly classifies the
   failure as retryable and preserves scientific identity. The first valid
   completion wins.

For an existing short runtime, the operational target is a detached controller
within 30 minutes of the recorded request. A new runtime integration may use
a 90-minute target, but the concrete blocker and estimate must be recorded
before the target is exceeded. See
[`EXPERIMENT_LAUNCH_DISCIPLINE.md`](EXPERIMENT_LAUNCH_DISCIPLINE.md).

## Bounded real-host verification

The verification used the frozen implementation SHA above and a public,
CPU-only non-overwriting dummy CLI. It did not call a training or evaluation
CLI.

| Host | Verified facts |
| --- | --- |
| Hamster | Registry resolution, source cleanliness, actual Teacher and known-parent SHA reads, workspace enforcement, sidecar ownership, attempt-scoped output, and a successful bounded local controller lifecycle. |
| Ferret | Fixed-SHA remote worktree, remote doctor, prepared-manifest identity, independently observed origin host, argv, live PID, assigned GPU UUID, staged collection, SHA equality, atomic promotion, and cleanup of the temporary remote run. |

Ferret did not contain the Hamster I100 stage-wise parent used in the local
check. No parent was substituted. The remote verification instead read a
different checkpoint that was already hash-bound by its own Ferret-local
`horizon-checkpoints.json`; it proves remote operational lineage handling,
not scientific parent equivalence.

The temporary verification output remains only below the canonical runtime
root. Staging and canonical payload hashes were equal before promotion.

## Task context

Task context is intentionally runtime-only and navigation-oriented:

```bash
ARD_PYTHON="$(python3 scripts/workspace_paths.py python)"
"$ARD_PYTHON" scripts/task_context.py init --task-id <id> --goal '<goal>' \
  --source-sha "$(git rev-parse HEAD)" \
  --authoritative-file docs/ERT_RESEARCH_STATUS_SUMMARY.md \
  --pending-milestone '<milestone>' --stop-rule '<stop rule>'
"$ARD_PYTHON" scripts/task_context.py append --task-id <id> \
  --field completed_milestones --value '<completed milestone>'
"$ARD_PYTHON" scripts/task_context.py replace --task-id <id> \
  --field pending_milestones --value '[]'
"$ARD_PYTHON" scripts/task_context.py show --task-id <id>
```

It cannot replace a report, machine artifact, checkpoint hash, or workspace
fact. When the context disagrees with a real artifact, the artifact wins and
the context must be corrected.

## Validation and residual limitations

The focused CPU-only suite validates the operational invariants, including
the dummy DAG/retry lifecycle. It is not evidence that every future
scientific CLI can run: a real production campaign must still pass R30/R31
with its exact source, config, parent, command, and execution class before
fan-out. The workspace registry also does not authorize deletion of an
unknown path; new roots remain an inventory-first, fail-closed decision.

This milestone is complete. It creates no new scientific campaign and does
not select, extend, or reinterpret an existing experiment.
