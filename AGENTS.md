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
- Before launching a long-lived `systemd --user` job, require `loginctl show-user "$USER" -p Linger --value` to report `yes`. A transient user service is not detached from logout when linger is disabled; do not treat `systemd-run --user` alone as persistence.
- Classify bounded work before orchestration. Pure run tagging, read-only reporting, and UI/view organization use an operational fast lane: one owner, focused tests, and no scientific reviewer unless metric semantics, resume, artifacts, or result lineage change.
- Freeze lineage, resume, cache invalidation, and external-API inputs before handing work to the writer. Aim for one implementation turn; a follow-up requires new live evidence or a concrete review finding.
- A bounded non-GPU operational change should normally reach focused verification within 15 minutes. If it does not, stop adding scope, report the serial bottleneck, and choose the smallest remaining completion path.
- Parallelize only independent read-only discovery or independent jobs. Extra writers/reviewers on one dependency chain increase integration time and token use and are not a latency strategy.
- Classify a campaign as `FAST_EXISTING_RUNTIME` by default when its public runtime and inputs already exist; use `FULL_NEW_INTEGRATION` only for a genuinely new runtime/trainer/DDP/dataset/remote/checkpoint/artifact mechanism. Fast retains every scientific identity check and the exact public-CLI smoke.
- Resolve the complete host × job matrix once before launch. Reuse an unchanged artifact hash within that preparation pass; if one cell is wrong, repair and revalidate the equivalent matrix rather than discovering host mismatches serially.
- Use one representative `smoke_group` only for an explicitly equivalent public CLI, execution class, output semantics, config schema, checkpoint-load path, and treatment branch. A stronger exact remote smoke may subsume its duplicate generic lifecycle canary, never source/identity/collection proof.
- Keep non-critical infrastructure work outside a frozen scientific campaign's critical path, and batch one complete review finding set before the correction wave instead of serial one-finding reruns.

## Critical evaluation of proposed procedures

- Treat user plans, generated implementation prompts, and external runbooks as requirements and hypotheses, not automatically correct mechanisms.
- Preserve the requested outcome while independently checking environment facts, security boundaries, scientific invariants, and simpler alternatives before adopting a proposed command or architecture.
- When evidence contradicts a proposed mechanism, explain the evidence, choose the safer or more reproducible design, and record the accepted/rejected assumption in the active plan.
- Do not perform ceremonial compliance. A named tool, model role, launcher, or review step must provide information or control that materially advances the task.
- Before adding a parity run or repeated validation, state the failure it can detect and the decision it can change. If a short run cannot resolve the long-horizon risk, use static identity checks and a blocked/randomized experiment design instead.
- Before assigning remote compute, compare artifact locality and transfer size with measured host throughput. Use hash-verified `rsync` when transfer is cheaper than leaving a long job on the slower host.
- After a real operational inefficiency, update the narrowest reusable skill or runbook, add a regression only when code caused the failure, and avoid adding campaign-specific checks to the scientific core.

## Large replay analysis protocol

- Before a full replay, run one checkpoint end to end using a real run, real checkpoint, and real sparse source sample IDs. The smoke must invoke the public CLI and reach report creation.
- The smoke must validate the Parquet schema, lineage hashes, stable-ID/class joins, and non-overwriting report output. Unit fixtures with dense IDs are not a substitute.
- Freeze the union of columns needed by feature, outcome, and downstream taxonomy analyses before GPU launch. Do not extend the observation schema after replay begins.
- Measure one-checkpoint wall time per teacher/job first. Assign jobs by longest-processing-time-first; large-teacher feature and outcome jobs get separate GPUs before shorter jobs.
- Feature and outcome replay are independent unless the code proves otherwise. Launch them concurrently from the start.
- Build one hash-bound checkpoint inventory JSON per run and reuse it for local analysis, remote execution, and collection. Do not repeat filesystem or W&B inventory searches for the same lineage.
- Compute point estimates before bootstrap. Start bootstrap as a separate process only for preregistered point gates that pass.
- Keep bootstrap replicate count, seed, strata, and estimator fixed. Reduce latency only by deterministic multiprocessing, never by weakening the scientific contract.
- Persist bootstrap progress at anchor/run/replicate granularity with source and contract hashes, and resume only matching incomplete work.
- A steady-state large replay analysis should target at most two hours including smoke, replay, collection, point estimates, and resumable bootstrap. If the estimate exceeds this, redesign scheduling or analysis execution before launch.

## Review latency and retry policy

- Use one consolidated independent review after the milestone delta and evidence are stable. Re-review only a fix delta for an actual P0/P1 or new contradictory evidence.
- Do not impose a universal 60-second completion deadline on reasoning-heavy reviewers. Let a bounded review run for several minutes while the main thread performs non-overlapping work and continues user updates.
- If a reviewer produces no verdict, record review as pending rather than approval. Send at most one request to conclude; do not launch repeated replacement reviewers for the same unchanged delta.

## Commit policy

- Establish a baseline commit early so `git diff` and impact-selected tests remain narrow. Do not manufacture retrospective milestone history.
- After a milestone passes its selected tests and scientific review, create one cohesive commit with the plan updated. Verify the staged diff and final status first.
- Keep integration branches short-lived. Merge a coherent, verified milestone back to `master`; do not wait for an entire experiment campaign or accumulate unrelated milestones on one branch.
- Before merging, verify both worktrees are clean, the branch is pushed, `master` has not diverged, and active experiments are pinned to immutable SHAs. Prefer `--ff-only` when `master` is an ancestor.
- Detached experiment worktrees do not require delaying an otherwise safe merge. Retain the remote integration branch until other hosts have fetched the merged `master`, then delete it separately.
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

## Operational orchestration

- New production campaigns must use the schema-v2 experiment-state bridge and
  an orchestrator-authoritative `orchestrator_campaign` mode; legacy v1 states
  remain read-only compatible.
- Treat host/runtime signatures and dependency topology as validated inputs.
  Unknown or changed signatures fail closed until an exact bounded smoke is
  recorded in the tracked runtime registry.
- Scheduled reconciliation is bounded and marker-driven.  Do not poll stable
  jobs, infer success from a PID/GPU, or launch duplicate endpoint/report work.
- Technical recovery may retry only a registered command with unchanged
  scientific identity and a finite lease/attempt bound; scientific decisions
  remain human-owned.
- Production job roles are explicit (`training`, `evaluation`, `collection`,
  `inventory`, `aggregation`, `report`, `finalization`, `publish`); never infer
  a missing role as training in a schema-v2 campaign.
- The launch lifecycle is `LAUNCHING` until a structured controller start proof
  is recorded; GPU utilization or a guessed PID is not launch evidence.
- Aggregate all required and downstream failures before retrying; retry only
  when every observed failure is technical and explicitly retryable.
- Publish terminal events only after canonical remote commit/blob verification;
  resume the same revision after a partial push rather than creating a new one.
- Keep the final publish node in the campaign DAG; the reconciler is a bounded
  fallback and must not become a second post-processing owner.
