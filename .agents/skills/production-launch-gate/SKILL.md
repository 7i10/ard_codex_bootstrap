---
name: production-launch-gate
description: Freeze and fail-closed validate a scientific campaign before handing its immutable manifest to the multi-GPU orchestrator.
---

# Production Launch Gate

Use this skill for a multi-job scientific campaign when source, parents,
host-local inputs, dependencies, endpoints, and retry lineage must be checked
before GPU work starts. It is an execution guard, not a scientific method
selector and not a replacement scheduler.

## Workflow

1. Author a JSON (or YAML with PyYAML) campaign spec containing logical dataset
   and teacher identities, host profiles, scientific epoch bounds, attack and
   augmentation contracts, parent/mask/calibration descriptors, jobs, and a
   bounded non-scientific canary.  Include every known endpoint, collection,
   aggregation, and report node in that DAG before the first launch.
2. Run the gate in `--preflight-only` or `--dry-run` mode. It resolves host
   paths, validates Git/source/config/parent/input hashes and metadata, checks
   dependencies/output collisions, and requires a bounded `remote_preflight`
   record for every external host. A single locally passing host never clears an
   unknown remote host.
3. Run `--canary-only` from the same spec. The canary uses the resolved job
   cwd/environment and a temporary output area; it never overwrites a parent
   or production output. External campaigns additionally prove one bounded
   remote process -> completion -> local collection -> SHA round trip per host.
4. Run `--launch`. The gate revalidates the frozen manifest, source, and every
   external-host preflight, invokes the existing
   `multi-gpu-experiment-orchestrator`, and returns after the detached
   controller is launched. Do not poll a stable long-running job from Codex.
5. After the detached DAG is done, run `--validate-run
 --resolved-manifest <path>` to require valid completion markers, expected
 outputs, hashes, and final epochs.

For an existing, already-audited public runtime, use the same gate's one-command
Fast mode instead of running separate preparation commands:

```bash
python3 .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --fast-launch
```

`FAST_EXISTING_RUNTIME` is the default when no new integration mechanism is
declared.  It still performs complete resolution, static CLI checks, exact
public-CLI smoke, one immutable freeze, frozen revalidation, and detached
orchestrator handoff.  `FULL_NEW_INTEGRATION` is required for a new
 objective/runtime, trainer, DDP, dataset loader, remote executor, checkpoint
 serialization, artifact schema, or genuine integration uncertainty; Fast
 refuses it rather than silently weakening validation.

Fast production/workspace campaigns must also declare a runtime signature
registered in `configs/operational/validated_runtime_signatures_v1.json`.
Unknown signatures fail closed until an exact bounded smoke validates the
public CLI, checkpoint path, output/artifact semantics, executor class, and
dependency-output topology. The signature and topology digest are bound into
the immutable manifest and smoke proof.

## Launch timing and host matrix

For a bounded screen with an existing runtime, initialize the orchestrator
skill's `launch_ledger.py` at request receipt and target controller launch in
30 minutes.  Before `--launch`, the ledger must bind a complete host × job
matrix: each resolved config's SHA, its permitted host-local path rebases, the
frozen Teacher byte SHA, parent/checkpoint paths, output root, execution
class, estimated work, work unit, transfer cost, and selected GPU throughput.
A generic host preflight alone is not evidence that a job-specific
resolved config is valid on that host.

If a single cell exposes a path or environment mismatch, stop the fan-out,
repair the whole matrix, and create a fresh manifest/attempt namespace.  Do
not discover equivalent host errors one checkpoint at a time.

The implementation is [scripts/launch_gate.py](scripts/launch_gate.py). It
emits the orchestrator schema-v1 `resolved-manifest.json`, `freeze.json`,
`preflight.json`, and `canary.json` under an immutable gate directory.
[`scripts/artifact_inventory.py`](scripts/artifact_inventory.py) stages and
validates canonical local aggregation inputs. The orchestrator remains
responsible for GPU reservations, host-aware scheduling, detached workers,
completion-marker dependency transitions, endpoint chaining, and
technical-only retries.

On a real `--launch`, the gate creates a runtime-bound schema-v2
`experiment-state.json` bridge. It is not emitted by dry-run or preflight-only;
the scheduled reconciler reads orchestrator state as the authority for
multi-job training and never treats a local PID or GPU sample as remote
success.

## Exact-command smoke contract

For a new or changed production/replay/forensic runtime, set
`canary.require_exact_smoke: true`. Before a controller can reserve a GPU, the
gate runs `canary.static_cli` argv entries (compile/import/`--help` as
applicable). Then each equivalent fan-out group needs at least one
`kind: exact_public_cli` representative smoke. The successful smoke record is
bound to source SHA, production argv hash, smoke argv hash, config SHA, parent
SHA, orchestrator source SHA, manifest schema, and execution class.

Local and external execution classes require separate exact smoke coverage.
The binding is recomputed immediately before launch, so a source/controller,
command, parent, or config change invalidates the smoke and requires a new
manifest/canary. This is a safety gate, not a shortened scientific run: use a
registered bounded public interface rather than silently adding an ad-hoc
one-epoch override.

Use `smoke_group` only when every member declares an identical
`smoke_equivalence` descriptor covering public CLI, output semantics, config
schema, checkpoint-load path, and treatment branch.  This avoids duplicate
per-seed smokes while keeping every job's source/parent/config/Teacher/input
identity independently validated.  A group cannot mix local and external
execution classes.  A successful exact external smoke may set
`subsumes_remote_lifecycle: true` only when it also proves process, source,
remote manifest, completion, and staged SHA-verified collection; that stronger
proof suppresses the duplicate generic lifecycle canary for that host only.

Distinct external-host preflights run concurrently. Static checks remain
serial unless their entry declares `parallel_safe: true`; commands inside one
entry always remain ordered. An exact public-CLI smoke may overlap another
only when it also declares `parallel_safe: true`, a unique
`parallel_resource_key`, and a distinct fixed host/GPU. These declarations
mean the author has verified isolated output, GPU, remote-runner, and mutable
resource ownership. Unmarked checks preserve serial behavior.

New production specs should also carry the workspace-contract opt-in passed to
the orchestrator. Future runtime writes outside the tracked canonical runtime
root are rejected; historical inputs remain readable.

## Safety rules

- Scientific identity hashes include source, arm, seed, config, dataset/split,
  parent bytes, attacks, augmentation, RNG, masks, and calibration. Host, GPU,
  attempt, and W&B execution IDs are excluded.
- Runtime training bounds are exclusive: `scientific_final_epoch=114` resolves
  `--epochs 115` and records expected final epoch 114. A prefix or other
  bounded training DAG node may declare a narrower job-local `epoch_binding`;
  it must stay within the campaign envelope and is identity-bound separately.
- Dependency-produced inputs are deferred until the producer completes;
  existing inputs must already exist and match their hashes.
- Technical retries get an attempt-specific W&B ID but may not change any
  scientific identity field. Accuracy or endpoint outcomes are never retry
  reasons.
- A missing path, source/config drift, parent alias mismatch, dataset mapping
  mismatch, marker/output mismatch, or output collision fails the whole gate;
  no partial campaign is launched.
- An external job requires a `remote_command` and a bounded
  `host_confirm_probe`. The remote status must bind the live PID, source,
  campaign/job/identity, GPU index/UUID, remote manifest, and exact argv before
  the orchestrator labels it `host_confirmed_started`.
- Remote aggregation inputs require an explicit collection node and complete,
  identity-bound inventory. Remote absolute paths are provenance only; an
  aggregator consumes canonical local paths only after SHA validation.
- Frozen source drift requires a new manifest. A technical retry retains the
  existing scientific source/identity; a technical code fix needs a new source
  SHA, a new manifest, and an explicit equivalence record.
- W&B uploads remain governed by the scientific command and repository policy;
  the gate never enables model or run-bundle uploads.

## Commands

```bash
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --preflight-only
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --dry-run
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --canary-only
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --launch
python3 .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --campaign-spec campaign.json --fast-launch
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --validate-run --resolved-manifest .launch-gate/example/resolved-manifest.json
```

For lower-level scheduling or remote Ferret lifecycle, use
`multi-gpu-experiment-orchestrator` and `run-on-ferret` respectively. This gate
does not implement SSH, rsync, or a second DAG controller.
