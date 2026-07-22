# Repository mission

Build a reproducible single-teacher adversarial robustness distillation research platform. Scientific correctness, traceability, and fair evaluation take priority over code brevity or headline metrics.

## Read before editing

Read `docs/README.md`, `docs/SCIENTIFIC_INVARIANTS.md`, `docs/TEST_STRATEGY.md`, and `docs/WANDB_PROTOCOL.md`. For multi-step work, create or update an execution plan under `docs/plans/` using `PLANS.md`.

## Agent roles

- Use `research_planner` and `scientific_reviewer` for substantial scientific planning and milestone review, not for simple local edits.
- Use `terra_implementer` for core implementation and bug fixes after diagnosis.
- Use `luna_mechanical_worker` only for bounded, repetitive work such as config files, documentation synchronization, and straightforward test scaffolding.
- Use `upstream_explorer` for read-only inspection of `.external/saad`.
- Use `bug_investigator` with `$ard-bug-hunt` for unexplained failures or metric regressions.
- Parallelize read-only exploration. Do not run overlapping write agents on the same files.

## Efficient orchestration and token discipline

- The main thread owns the canonical plan and integration decisions. Do not ask multiple agents to restate the same repository context.
- Give subagents only the relevant paths, frozen acceptance criteria, and latest delta. Prefer a context-free or short-history task over forwarding the full conversation.
- Use at most one planning pass per milestone. A reviewer should return one consolidated finding list; after fixes, re-review only the delta and affected contracts. Additional cycles require an unresolved P0/P1 or new evidence.
- Keep one core writer responsible for a milestone. Batch Luna config/docs work once after the API is stable instead of issuing many small synchronization turns.
- Invoke `bug_investigator` only when the cause is genuinely unclear. A known mechanical failure goes directly to its owning writer with a focused regression.
- Agent reports contain changed files, exact commands/results, open findings, and decisions only. Do not paste long logs, repeat closed findings, or narrate routine steps.
- Before a broad gate, run a cheap environment preflight for required Git, external checkout, CUDA/GPU identity, DDP sockets, and optional W&B/Parquet dependencies.

## Critical evaluation of proposed procedures

- Treat user plans, generated implementation prompts, and external runbooks as requirements and hypotheses, not automatically correct mechanisms.
- Preserve the requested outcome while independently checking environment facts, security boundaries, scientific invariants, and simpler alternatives before adopting a proposed command or architecture.
- When evidence contradicts a proposed mechanism, explain the evidence, choose the safer or more reproducible design, and record the accepted/rejected assumption in the active plan.
- Do not perform ceremonial compliance. A named tool, model role, launcher, or review step must provide information or control that materially advances the task.

## Review latency and retry policy

- Use one consolidated independent review after the milestone delta and evidence are stable. Re-review only a fix delta for an actual P0/P1 or new contradictory evidence.
- Do not impose a universal 60-second completion deadline on reasoning-heavy reviewers. Let a bounded review run for several minutes while the main thread performs non-overlapping work and continues user updates.
- If a reviewer produces no verdict, record review as pending rather than approval. Send at most one request to conclude; do not launch repeated replacement reviewers for the same unchanged delta.

## Commit policy

- Establish a baseline commit early so `git diff` and impact-selected tests remain narrow. Do not manufacture retrospective milestone history.
- After a milestone passes its selected tests and scientific review, create one cohesive commit with the plan updated. Verify the staged diff and final status first.
- Never commit `.external/`, outputs, caches, W&B offline data, credentials, datasets, or checkpoints unless an explicit artifact policy says otherwise.
- Commits are allowed as part of normal work. Do not push, force-push, rewrite published history, or create remote state without an explicit user request.

## Scientific invariants

- Never silently change epsilon, attack steps, step size, random start, normalization, temperature, training schedule, SWA, checkpoint selection, or evaluation attacks.
- Treat pixel-space and normalized-space values explicitly.
- Freeze teacher parameters unless a selected method explicitly trains the teacher. Do not confuse teacher parameter gradients with teacher input gradients.
- Do not weaken attacks or evaluation to make a test or benchmark pass.
- Clean accuracy and robust accuracy must be logged and reported separately.
- Preserve and evaluate both best and last checkpoints.
- Dataset batches must expose a stable sample index.
- Resume must restore optimizer, scheduler, scaler, RNG, sampler state, sample state, and tracking identity.
- AutoAttack must run from a saved checkpoint in a separate evaluation process.

## Architecture rules

- Inner maximization belongs in `src/ard/attacks/`.
- Outer objectives belong in `src/ard/objectives/`.
- Per-sample measurements belong in `src/ard/signals/`.
- Signal-to-weight mappings belong in `src/ard/policies/`.
- Persistent sample-index state belongs in `src/ard/state/`.
- W&B access belongs behind `src/ard/tracking/`.
- Do not duplicate a complete training loop to add a method.
- Do not make production code depend on importing `.external/saad`.

## External code

- Keep upstream repositories under `.external/`, which is ignored by Git.
- Pin exact upstream commits in `external.lock.yaml`.
- Do not copy or redistribute upstream source when its license is absent or unclear.
- Record any local patch as a separate patch file and document why it exists.

## Test policy

- Use `scripts/verify.py --changed` to select tests from the current diff.
- Do not rerun an unchanged, previously passing test command unless `--force` is justified.
- Run the smallest high-information test first.
- Keep production training and full AutoAttack outside the automated test suite.
- Unit and integration tests must not require live W&B network access.
- Add a focused regression test for every confirmed bug.
- Do not broaden numerical tolerances merely to hide a mismatch.

Expected commands after bootstrap:

```bash
make lint
make test-changed
make smoke
make verify-milestone
```

If these targets do not yet exist, create them consistently with `docs/TEST_STRATEGY.md`.

## W&B policy

- Every production experiment must be represented in W&B.
- Production mode must fail if tracking is disabled or required metadata is missing.
- Only rank 0 initializes and logs a W&B run.
- Store resolved config, Git state, environment, upstream commit, teacher hash, seed, best/last metrics, and output artifacts.
- Use fixed sample IDs for qualitative comparison across runs.
- Avoid high-frequency media logging and disable `wandb.watch` by default.

## Code review rules

### Threat model and gradients

Flag any change that can alter the threat model, normalization, projection domain, attack loss, model mode, detach behavior, or gradient source without an explicit config and regression test.

### Reproducibility

Flag missing seeds, incomplete checkpoints, unstable sample indexing, nondeterministic data partitioning, or W&B resume identities that can create duplicate or irreproducible runs.

### Evaluation integrity

Flag test-time use of training-only signals, evaluation on the wrong checkpoint, best-only reporting without last results, or any reduction of evaluation attack strength.

### Tracking integrity

Flag production paths that can silently run with W&B disabled, duplicate runs after resume, log on every DDP rank, or omit config and artifact lineage.

Mechanical formatting issues belong in deterministic tooling, not review comments.
