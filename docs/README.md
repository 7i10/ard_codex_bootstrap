# Documentation index

The research direction changed on 2026-09-09: four months of Adversarial
Robustness Distillation (ARD) and state-conditional-intervention work on
CIFAR-10 were set aside after a clean-slate literature review. The current
direction is mobile-scale ImageNet adversarial robustness via teacher-free
self-distillation, starting with a CIFAR-10-scale replication of one published
result (ADR, arXiv:2305.12118). See `docs/decisions/0009-*.md` (the live
authorisation) and `docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md` (the plan).

**Everything from the ARD/distillation line has been moved to
`docs/archive/ard-distillation-2026/`, not deleted.** It is git-tracked and
readable; it is just out of the way of day-to-day work on the new direction.
That archive holds ~370 files and 18MB — every ERT/FFNR result document, the
evidence audit stack (`ARM_REGISTRY.md`, `EVIDENCE_LEDGER_V2.md`,
`COEFFICIENT_AUDIT.md`, etc.), the research-essay history
(`RESEARCH_DECISIONS.md`, `PRIOR_ART_STATE_CONDITIONED_AUGMENTATION.md`), the
superseded direction-pivot documents (`RESEARCH_RESTART.md`,
`THESIS_FRAME.md`, `CLEAN_SLATE_DIRECTIONS.md`, `METHOD_DIRECTIONS*.md`,
`ARD_VERSUS_AT_ASSESSMENT.md`, `SMALL_MODEL_AT_2024_2026.md`, `RED_TEAM.md`),
and every hash-bound experiment record under
`docs/archive/ard-distillation-2026/experiments-full/`. Nothing there is
needed to work on the current direction; consult it only if a specific old
number or decision needs tracing.

`docs/plans/`, `docs/decisions/`, and `docs/debugging/` were **not** archived
— their numbering is a live sequence and several individual files remain
directly relevant (named below).

## 0. Start here

- [`../CLAUDE.md`](../CLAUDE.md) — the operating contract: the three planes, daily commands, hard rules, model policy. Direction-agnostic.
- [Claude Code workflow](CLAUDE_CODE_WORKFLOW.md) — why the repository is split into a scientific/execution/agent plane, and how a session runs day to day.
- [Scientific invariants](SCIENTIFIC_INVARIANTS.md) — the attack, gradient, checkpoint, and evaluation rules that never change regardless of method.
- [`docs/decisions/0009-mobile-robustness-direction-and-first-step.md`](decisions/0009-mobile-robustness-direction-and-first-step.md) — the live, pending decision: what to run first.
- [Mobile robustness method proposal](MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md) — the current plan, including the exact engine gap-analysis for the ADR replication.

## 1. Contract and invariants (direction-agnostic — read once, applies regardless of what is being studied)

- [Measurement standard v1](MEASUREMENT_STANDARD.md) — pre-registration rules: arm naming, parent/replicate counts, horizon, power, coefficient calibration. Answers "what must a new plan do before it runs?" Written after the ARD line's own failures, but stated as a forward rule for all plans.
- [Implementation specification](IMPLEMENTATION_SPEC.md) — repository structure; a new method is an (attack, objective, signal, policy) combination, never a duplicated training loop.
- [Test strategy](TEST_STRATEGY.md) — test tiers, change-impact selection, pass caching, numeric tolerances, GPU exclusivity.
- [W&B protocol](WANDB_PROTOCOL.md) — tier/state naming, group/job-type conventions, artifacts, the fixed sample table.
- [Experiment protocol](EXPERIMENT_PROTOCOL.md) — run tiers (audit/pilot/canonical production), baselines, seeds, evaluation.
- [Upstream baselines](UPSTREAM_BASELINES.md) — pinned SAAD/TRADES commits, license evidence, known differences from upstream. Section 8 (TRADES) records the detached-target defect; read it before writing any new objective that borrows the KL primitive.
- [Reproduction status](REPRODUCTION_STATUS.md) — what is implemented, what has been run, the actual CLI commands to launch a CIFAR-10 job today.
- [Artifact retention policy](ARTIFACT_RETENTION_POLICY.md) — what gets kept vs. regenerated, classified by regeneration cost now that determinism is confirmed.
- [Backlog](BACKLOG.md) — the autonomy tier rule (A-tier: start unasked; B-tier: decision packet first). Items A3-A6 (ruff format, impact-map coverage, stale worktrees, the still-physically-present frozen control-plane code) are engine hygiene the new direction inherits regardless.

## 2. Execution plane — running GPU jobs (direction-agnostic)

- [Production Launch Gate](PRODUCTION_LAUNCH_GATE.md) — the validation/freeze layer in front of the orchestrator: preflight, dry-run, canary, launch.
- [Multi-GPU Experiment Orchestration Skill](MULTI_GPU_EXPERIMENT_ORCHESTRATION_SKILL.md) — the generic, method-unaware scheduling and DAG-chaining layer.
- [Ferret execution protocol](FERRET_EXECUTION_PROTOCOL.md) — Hamster-as-planning-node / Ferret-as-execution-node split, and the SSH/rsync boundary.
- [ARD workspace contract](WORKSPACE_CONTRACT.md) — canonical runtime paths on Hamster and Ferret. Will need an ImageNet path entry once that dataset is wired in.
- [ARD operational foundation](ARD_OPERATIONAL_FOUNDATION.md) — completed milestone that standardized runtime-state location; still cited by the launch-gate test suite.

## 3. Plans, decisions, debugging (live numbered sequences — not archived)

- [`plans/`](plans/) — one plan per file. Most are ARD-era and closed; two are directly useful templates for the new direction: `0027-controlled-teacherless-baselines.md` (the existing teacherless CIFAR-10 recipe, PGD-AT/TRADES official numbers) and `0091`/`0092` (a preregistered confirmation and a floor-measurement worked example).
- [`decisions/`](decisions/) — human decisions from imported results. `chosen: null` means no decision has been made and no new scientific job should start on that question. `0009` is the only currently open one. `0003` and `0004` document engine defects (undiagnosable failures; a W&B run-ID collision) worth knowing before the first new campaign.
- [`debugging/`](debugging/) — one record per resolved bug or operational incident, never deleted. Most are ARD-era but several are engine-general and will matter again: `0007` (DDP crashes on an extra forward pass — directly relevant to adding an EMA-of-student forward), `0026` (CUDA RNG restore ignoring device count), `0024` (long GPU services die on logout without `loginctl enable-linger`), `0027` (unattended-upgrades silently killing sshd on a long run), `0028` (the TRADES detach defect and its anti-pattern: documented as intentional, pinned by a test, cost never measured — the mistake a new objective must not repeat).

## 4. Archive

- [`archive/ard-distillation-2026/`](archive/ard-distillation-2026/) — everything described at the top of this file. See its own contents for the full ARD/state-conditional-intervention research record, four months of experiment reports, and the direction-pivot documents that led to the current plan.

## CLI entry points

```bash
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml>
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml> --resume <output>/last.pt
PYTHONPATH=src python -m ard.cli.evaluate --config <experiment.yaml> --checkpoint-dir <output>
PYTHONPATH=src python -m ard.cli.status --root <output-root> --format markdown
python scripts/verify.py --changed
```

`evaluate` reads only saved checkpoints; with `--checkpoint-dir` it follows the config's
`evaluation.checkpoints` setting (default `both`) and evaluates `best.pt` and `last.pt`
separately. Full AutoAttack never starts from ordinary train/test; it must be enabled explicitly
in the evaluation config and run as a separate process with `--allow-autoattack`.

Running epoch, step, update time, and terminal state are derived by the status CLI from
`run-bundle/manifest.json`. Cross-host live viewing uses W&B; the Git-tracked dashboard body is
not hand-edited for process monitoring.
