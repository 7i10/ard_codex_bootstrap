# Experiment Automation Last-Mile Hardening

Status: complete

Scope: operational control-plane only. No scientific campaign, model, loss,
attack, checkpoint, seed, or result is changed.

## Consolidated issue list

1. Fast eligibility trusts an authored signature and the registry is empty.
2. Schema-v2 jobs can inherit a training role when `job_type` is omitted.
3. Failure classification returns on the first failed job.
4. The gate writes `TRAINING` before detached controller launch is proven.
5. Downstream failures use training-oriented classification.
6. Terminal publication needs remote canonical verification, revision collision
   handling, and resumable push/PR phases.
7. The event branch/PR is not registered as a durable operational resource.
8. Publication is not an explicit final DAG role; reconciliation can become the
   normal success path.

## Implementation sequence

- derive and promote resolved-manifest runtime fingerprints;
- enforce explicit schema-v2 job roles and terminal-job derivation;
- harden aggregate failure and launch lifecycle state transitions;
- make terminal publication remotely verifiable and resumable;
- register the event bus and document event-driven finalization;
- run focused unit tests plus CPU-only success/failure DAG fixtures;
- record closure evidence and commit/push the operational-only change.

## Safety boundary

The active Online-State S2 source/worktree/manifest/checkpoints are outside this
worktree and are read-only evidence. Runtime writes for this task remain under
the canonical runtime root. A source-changing recovery is never performed by
the scheduled reconciler.

Verification: focused operational suite passed (92 tests), Python compilation
passed, and `scripts/verify.py --changed --non-scientific --dry-run` was
inspected. No scientific training, endpoint evaluation, or W&B mutation was
performed.
