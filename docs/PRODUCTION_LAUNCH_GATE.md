# Reusable Production Launch Gate

The launch gate is the validation and freeze layer in front of the existing
[`multi-gpu-experiment-orchestrator`](../.agents/skills/multi-gpu-experiment-orchestrator/SKILL.md).
It does not train a model, choose a method, or implement remote SSH/rsync.

## Normal lifecycle

```text
campaign spec
    -> source/host/input preflight
    -> deterministic resolved manifest
    -> atomic freeze + manifest SHA
    -> bounded non-scientific canary
    -> detached orchestrator DAG
    -> completion-marker endpoints/aggregation
    -> post-run output and lineage validator
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
orchestrator state file.

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

## Canary and completion

Canaries are explicit, bounded, non-scientific commands under the same resolved
job context and a separate temporary gate output. They are not a shortened
training command and cannot overwrite production outputs. The existing
orchestrator supplies atomic completion markers and dependency chaining. The
post-run validator additionally requires every job to be completed, every
marker to match campaign/source/identity, and every declared output/hash/final
epoch to be present. Exit code zero alone is insufficient.

## Regression coverage

`tests/skills/test_production_launch_gate.py` covers the twelve historical
launch failures: exclusive epoch bounds, parent aliases, deferred dependency
outputs, W&B retry IDs, host dataset mapping, mask hashes, output collisions,
source drift, retry scientific mutations, and false completion. It also runs a
CPU-only canary, a detached success DAG, and a technical retry. The existing
orchestrator tests remain authoritative for GPU reservations, marker DAG
transitions, detached resume, and technical retry state.

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

## Limitations

The gate validates checkpoint metadata supplied as JSON/sidecar descriptors; a
format-specific model loader is intentionally not embedded. Remote host
lifecycle remains delegated to `run-on-ferret`/the orchestrator's external
executor. W&B network access is not performed by the gate. A campaign author
must provide a bounded canary and expected output descriptors; scientific
commands remain responsible for their own metrics-only tracking policy.
