---
name: production-launch-gate
description: Freeze and fail-closed validate a scientific campaign before handing its immutable manifest to the multi-GPU orchestrator. Use before any multi-job GPU campaign starts; not a scientific method selector or scheduler.
---

# Production Launch Gate

Full contract: `docs/PRODUCTION_LAUNCH_GATE.md`. Spec fields: `.agents/skills/production-launch-gate/references/campaign-spec.md`.
Script: `.agents/skills/production-launch-gate/scripts/launch_gate.py` (do not move; tests import by path).
Companion: `.agents/skills/production-launch-gate/scripts/artifact_inventory.py`.

## When to use

A multi-job scientific campaign where source, parents, host-local inputs, dependencies, endpoints, and retry
lineage must be validated before any GPU work starts. Jobs must run from a pinned worktree
(`scripts/ardx/pin_source.py <sha>`), never from a live checkout.

## Command sequence — always a fresh `--output-dir` per attempt

`GATE=<runtime>/runs/<campaign>-attemptN/launch-gate` (N = next unused attempt), plus a fresh canary
output dir. Without `--output-dir` the gate writes `.launch-gate/` next to the spec, which dirties this
checkout — the exact failure that lost attempt-11's 8 jobs (`CLAUDE.md` rules 1-2).

```bash
G=.agents/skills/production-launch-gate/scripts/launch_gate.py
python3 $G --campaign-spec campaign.json --preflight-only --output-dir "$GATE"
python3 $G --campaign-spec campaign.json --dry-run        --output-dir "$GATE"
python3 $G --campaign-spec campaign.json --canary-only    --output-dir "$GATE"
python3 $G --campaign-spec campaign.json --launch         --output-dir "$GATE"
python3 $G --validate-run --resolved-manifest "$GATE/resolved-manifest.json"
```

1. `--preflight-only`: resolves host paths, validates Git/source/config/parent/input hashes, checks dependencies
   and output collisions, requires a bounded `remote_preflight` record per external host.
2. `--dry-run`: same validation, no side effects.
3. `--canary-only`: proves a bounded non-scientific run end-to-end (remote hosts also get one process →
   completion → local collection → SHA round trip) without touching production output.
4. `--launch`: revalidates the frozen manifest and every external-host preflight, hands off to
   `multi-gpu-experiment-orchestrator`, returns once the detached controller starts. Do not poll — end the turn.
5. `--validate-run`: after the DAG finishes, requires valid completion markers, expected outputs, hashes, epochs.

## Outputs to check

Under the immutable gate directory: `resolved-manifest.json`, `freeze.json`, `preflight.json`, `canary.json`. On a
real `--launch`, also a schema-v2 `experiment-state.json` bridge (not emitted by dry-run/preflight-only).

## Hard limits

- **Never `--fast-launch`.** Fast mode and its runtime-signature registry are frozen Codex-era control plane —
  not to be used or extended per `CLAUDE.md`.
- Never weaken an attack, tolerance, or guard to make a canary/smoke pass.
- A missing path, source/config drift, parent alias mismatch, dataset mapping mismatch, marker/output mismatch, or
  output collision fails the whole gate; no partial campaign launches.
- Batch all review fixes before one re-freeze; do not repair and relaunch cells one at a time.
- Runtime training bounds are exclusive: `scientific_final_epoch=114` resolves `--epochs 115`.
- Technical retries may not change any scientific identity field; accuracy/outcome is never a retry reason.
- Remote aggregation inputs require an explicit collection node and SHA-validated canonical local paths only. The
  gate never enables model/run-bundle W&B uploads.
- For lower-level scheduling or remote lifecycle, use `multi-gpu-experiment-orchestrator` and `run-on-ferret`
  respectively; this gate implements neither SSH/rsync nor a second DAG controller.
