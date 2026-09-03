# Reusable Production Launch Gate

The launch gate is the validation and freeze layer in front of the existing
[`multi-gpu-experiment-orchestrator`](../.agents/skills/multi-gpu-experiment-orchestrator/SKILL.md).
It does not train a model, choose a method, or implement remote SSH/rsync.

## Normal lifecycle

```text
campaign spec
    -> local + every-assigned-host preflight
    -> source freeze and deterministic resolved manifest
    -> atomic freeze + manifest SHA
    -> local + remote lifecycle canary
    -> detached orchestrator DAG
    -> host-confirmed remote start
    -> completion-marker endpoints
    -> canonical collection + hash verification
    -> complete aggregation inventory validation
    -> aggregation/report + post-run lineage validator
```

Use the gate script from the repository root:

```bash
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --preflight-only
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --dry-run
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --canary-only
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --launch
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --validate-run \
  --resolved-manifest .launch-gate/example/resolved-manifest.json
```

The first successful resolution creates `resolved-manifest.json`, `freeze.json`,
and `preflight.json` in the gate output directory. A different resolution may
not overwrite an existing frozen manifest. `--launch` rechecks the manifest
SHA and source checkout before delegating to the detached controller. A stable
long-running process is not polled by Codex; later status/resume uses the
orchestrator state file. A controller spawning a local remote wrapper is not a
remote launch success: external jobs must emit bounded, identity-bound host
confirmation before their completion probe is eligible.

## Campaign spec boundary

See [campaign-spec.md](../.agents/skills/production-launch-gate/references/campaign-spec.md)
for the complete v1 example. A spec supplies scientific facts and logical
identities; host profiles supply local paths:

```json
{
  "campaign_id": "example-v1",
  "source": {"git_sha": "<40-hex>", "repo_path": "/work/repo"},
  "dataset": {
    "identity": "cifar10_controlled_split_v1",
    "host_paths": {"hamster": "/data/cifar10"}
  },
  "training": {"scientific_final_epoch": 114},
  "hosts": {
    "hamster": {
      "backend": "local",
      "repo_path": "/work/repo",
      "python": "/opt/env/bin/python",
      "dataset_paths": {"cifar10_controlled_split_v1": "/data/cifar10"},
      "gpus": [{"index": 0, "uuid": "GPU-...", "throughput": 679.0}]
    }
  }
}
```

Jobs use structured argv rather than shell strings. `scientific_final_epoch` is
inclusive; the gate binds it to the exclusive CLI `--epochs` value (`114` →
`115`) and records both values. Parent, mask, calibration, configuration, and
teacher descriptors can bind SHA-256 and metadata. `kind: dependency_output`
inputs are checked against a declared producer and are intentionally deferred
until that producer's marker is valid.

### External-host contract

Every host assigned an `external_probe` job must declare `remote_preflight`.
Its bounded command receives the expected frozen source, resolved artifact
bindings, output root, GPU inventory, launcher, and completion probe through
`ARD_LAUNCH_GATE_REMOTE_EXPECTED`, then returns schema-v1 JSON proving:

- frozen source SHA and usable Python;
- every resolved dataset, Teacher, parent, config, mask, and calibration
  binding required on that host;
- output-root writability, minimum free disk, and assigned GPU index/UUID;
- valid remote launcher and completion-probe invocation.

Unknown is failure: a passing Hamster record cannot substitute for a missing
Ferret record. Remote launcher metadata is `{ "argv": [...], "executable":
true|false }`; direct non-executable `foo.sh` is rejected, while `bash foo.sh`
is valid. The gate re-runs this remote source/host proof when `--launch` is
requested, after manifest-source validation; a changed source requires a newly
resolved manifest rather than an old-manifest retry.

For external jobs, `remote_command`, `host_confirm_probe`, and bounded
`host_confirm_timeout_seconds`/`host_confirm_interval_seconds` are required.
The confirmation payload binds campaign, job, scientific identity, source,
host, GPU index/UUID, live PID, remote manifest path, and exact remote argv.

## What fails before GPU launch

- missing/dirty/wrong Git source, registered source-file hash, or frozen-source
  drift;
- missing interpreter, repository, dataset, teacher, parent, mask,
  calibration, or configuration, including exact SHA/metadata mismatches;
- logical dataset identity resolved through the wrong host profile;
- attack identity fields/hash mismatch, epoch off-by-one, or forbidden retry
  mutation;
- dependency cycles, missing producers, producer-path/root mismatch, output
  or completion-marker collisions;
- duplicate W&B execution IDs.
- a direct non-executable remote wrapper/probe, incomplete remote
  parent/config/Teacher/data binding, inaccessible output root, insufficient
  disk, or wrong remote GPU UUID/index;
- source drift between freeze and external launch, a controller spawn without a
  matching live remote process, or an external campaign without collection and
  inventory DAG nodes.

Errors include job, field, expected value, observed value, and remediation.
Preflight is campaign-wide: one error launches zero jobs.

If a gate directory must live inside the checkout, list its non-scientific
prefixes in `source_policy.generated_paths`; otherwise keep gate outputs
outside the source tree. Unlisted tracked or untracked changes still fail the
clean-tree check.

## Identity and retry

The scientific identity hash covers source, arm, seed, scientific config,
dataset/split, parent bytes/metadata, attack, augmentation, RNG, masks, and
calibration. Host, GPU, technical attempt, and W&B execution IDs are not part
of it. The orchestrator receives an attempt-aware W&B template, so a technical
retry gets a new W&B execution ID while retaining the same scientific identity.
Accuracy and endpoint outcomes are never retry reasons.

## Canary, collection, and completion

Canaries are explicit, bounded, non-scientific commands under the same resolved
job context and a separate temporary gate output. They are not a shortened
training command and cannot overwrite production outputs. The existing
orchestrator supplies atomic completion markers and dependency chaining. The
post-run validator additionally requires every job to be completed, every
marker to match campaign/source/identity, and every declared output/hash/final
epoch to be present. Exit code zero alone is insufficient.

External campaigns add one bounded remote lifecycle canary per assigned host.
It demonstrates host process evidence, a completion marker, a local staging
copy, canonical collection, and matching SHA-256 bytes. It is a technical
canary, not scientific training or a W&B run.

Aggregation never consumes a remote absolute path. An external campaign must
declare `artifact_collection` with a schema-v1 inventory manifest plus explicit
`collection` and `inventory` job IDs. The DAG order is:

```text
train -> endpoint -> collection + SHA verify -> inventory validate -> aggregate -> report
```

[`artifact_inventory.py`](../.agents/skills/production-launch-gate/scripts/artifact_inventory.py)
copies already-collected local staging bytes to canonical local paths and
rejects missing or duplicate required cells, byte/hash mismatch, or foreign
campaign/source/job/seed/arm/epoch/split/attack identity before aggregation.
An aggregator may consume only these validated canonical paths; it must not
search alternate directories heuristically.

## Timing ledger

The gate records a best-evidence ledger in `preflight.json` and the frozen
manifest: `request_received` (including declared precision if available),
`gate_started`, source/remote-preflight resolution, `preflight_passed`, and
manifest freeze/SHA. The controller and collection/finalization nodes append
their own evidence for `controller_spawned`, `host_confirmed_started`, training
completion, endpoints, collection, aggregation, and report commit. Missing
evidence stays `unknown`; the ledger never fabricates minute-level precision.

## Regression coverage

`tests/skills/test_production_launch_gate.py` keeps the historical launch
failure IDs explicit. It also runs CPU-only canaries, a detached success DAG,
a technical retry, artifact-inventory fixtures, and fake external lifecycle
coverage. The existing orchestrator tests remain authoritative for GPU
reservations, marker DAG transitions, detached resume, technical retry state,
and host-confirmed external starts.

The requested regression names are kept explicit so future failures remain
traceable. Their originating historical event is recorded in the unseen
confirmation orchestration audit (`48edebc` is the source identifier recorded
there); the prevention is exercised by the corresponding gate test.

| ID | Historical event | Prevention |
| --- | --- | --- |
| R1 | inclusive/exclusive epoch confusion | bind inclusive final to exclusive `--epochs` and reject `runtime_epochs` drift |
| R2 | parent alias pointed at different bytes | exact parent and alias SHA check |
| R3 | dependency endpoint input treated as missing | deferred producer input and path binding |
| R4 | deterministic W&B retry collision | attempt-specific run-ID template |
| R5 | host dataset path leaked across machines | logical identity → host-local path resolution |
| R6 | registered mask bytes/IDs drifted | artifact SHA and metadata check |
| R7 | endpoint/report root crossed campaigns | producer output/dependency path validation |
| R8 | two jobs wrote one output | canonical output/marker collision check |
| R9 | source/config changed after freeze | manifest SHA and source/config revalidation |
| R10 | dependency checkpoint required before producer ran | `dependency_output` is deferred until marker completion |
| R11 | technical retry changed scientific identity | forbidden retry-mutation check and stable identity hash |
| R12 | exit 0 mistaken for completion | marker, state, output, hash, and final-epoch validator |
| R13 | direct non-executable remote wrapper/probe | remote wrapper argv/interpreter metadata validator |
| R14 | one host passed while Ferret inputs were unknown | required per-host remote preflight with complete bindings |
| R15 | local wrapper spawn called a remote launch | bounded live remote process/argv/GPU identity confirmation |
| R16 | frozen manifest was launched from a newer source | source recheck before external launch; new manifest on drift |
| R17 | aggregation read a Ferret absolute path | canonical local staging/collection with SHA verification |
| R18 | endpoint cells were split across directories | fail-closed required-cell inventory before aggregation |
| R19 | parallel Ferret prepare hit Git ref locks | per-remote-repo `flock` around Git mutation |
| R20 | replay smoke exited but did not prove the full path | mandatory remote lifecycle/collection/hash canary |
| R21 | collection could alter artifact bytes | collected SHA-256 equality validator |
| R22 | foreign host/campaign rows entered an aggregate | campaign/source/job/seed/arm/epoch/split/attack-bound inventory |

## Limitations

The gate validates checkpoint metadata supplied as JSON/sidecar descriptors; a
format-specific model loader is intentionally not embedded. `run-on-ferret`
remains the SSH/rsync lifecycle authority and the orchestrator remains the
detached DAG controller; the gate validates their contracts instead of
duplicating either. W&B network access is not performed by the gate. A campaign
author must provide bounded remote preflight/canary commands and expected output
descriptors; scientific commands remain responsible for their metrics-only
tracking policy.
