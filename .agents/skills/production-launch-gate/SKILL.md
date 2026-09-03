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
   bounded non-scientific canary.
2. Run the gate in `--preflight-only` or `--dry-run` mode. It resolves host
   paths, validates Git/source/config/parent/input hashes and metadata, checks
   dependencies and output collisions, and writes no production state.
3. Run `--canary-only` from the same spec. The canary uses the resolved job
   cwd/environment and a temporary output area; it never overwrites a parent
   or production output.
4. Run `--launch`. The gate revalidates the frozen manifest and source, invokes
   the existing `multi-gpu-experiment-orchestrator`, and returns after the
   detached controller is launched. Do not poll a stable long-running job from
   Codex.
5. After the detached DAG is done, run `--validate-run
   --resolved-manifest <path>` to require valid completion markers, expected
   outputs, hashes, and final epochs.

The implementation is [scripts/launch_gate.py](scripts/launch_gate.py). It
emits the orchestrator schema-v1 `resolved-manifest.json`, `freeze.json`,
`preflight.json`, and `canary.json` under an immutable gate directory. The
orchestrator remains responsible for GPU reservations, host-aware scheduling,
detached workers, completion-marker dependency transitions, endpoint chaining,
and technical-only retries.

## Safety rules

- Scientific identity hashes include source, arm, seed, config, dataset/split,
  parent bytes, attacks, augmentation, RNG, masks, and calibration. Host, GPU,
  attempt, and W&B execution IDs are excluded.
- Runtime training bounds are exclusive: `scientific_final_epoch=114` resolves
  `--epochs 115` and records expected final epoch 114.
- Dependency-produced inputs are deferred until the producer completes;
  existing inputs must already exist and match their hashes.
- Technical retries get an attempt-specific W&B ID but may not change any
  scientific identity field. Accuracy or endpoint outcomes are never retry
  reasons.
- A missing path, source/config drift, parent alias mismatch, dataset mapping
  mismatch, marker/output mismatch, or output collision fails the whole gate;
  no partial campaign is launched.
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
python .agents/skills/production-launch-gate/scripts/launch_gate.py \
  --validate-run --resolved-manifest .launch-gate/example/resolved-manifest.json
```

For lower-level scheduling or remote Ferret lifecycle, use
`multi-gpu-experiment-orchestrator` and `run-on-ferret` respectively. This gate
does not implement SSH, rsync, or a second DAG controller.
