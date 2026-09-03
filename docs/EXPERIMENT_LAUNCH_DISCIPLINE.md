# Experiment Launch Discipline

## Why this exists

The I100 Dynamic-BDD screen and its subsequent forensic audit exposed a
failure of execution discipline, not a need for weaker scientific checks.
Short, already-specified campaigns should spend their wall clock on the
registered GPU work, not on serial path discovery, source churn, or manual
recovery.

This document is operational only.  It changes neither a scientific result nor
an active campaign.

## Evidence from the incident

The completed Dynamic-BDD screen was requested at approximately 2026-09-03
14:00 JST.  Its first controller launched at 16:53:13 JST: **about 2 h 53 m
before any production data-plane work began**.  The results commit followed at
23:54:56 JST: **about 9 h 55 m** after the user-reported request.  The
evidence-backed detail is in [the execution postmortem](ERT_RSLAD_I100_DYNAMIC_BDD_EXECUTION_POSTMORTEM.md).

For the follow-on forensic audit, the first implementation commit was
01:59:06 JST on 2026-09-04.  The first valid canonical replay fan-out started
at 04:00:12 JST and completed at 04:23:24 JST.  Thus a checkpoint-only
read-only audit spent roughly **2 h 1 m before its first valid GPU launch**.
Later P1 lineage checks correctly rejected a dev2/DPM proxy: the first
technical retry used a Hamster-only Teacher path on Ferret and failed before
meaningful GPU work.  This was a correct fail-closed outcome, but it should
have been prevented by a complete host × job config check before launch.

These elapsed times are not an acceptable baseline for this class of work.

## Root causes owned by Codex

1. **Infrastructure was put on the scientific critical path.**  I added and
   revised generic orchestration pieces while a small bounded campaign should
   have been made runnable first.
2. **Inputs were rediscovered serially.**  Parent, config, Teacher path,
   output root, and remote execution details were not frozen in one inventory
   before writing launchers.
3. **Host-specific config semantics were checked too late.**  A config that
   was valid on Hamster was copied to Ferret without proving the complete
   Ferret path rebase and registry compatibility for every planned job.
4. **Review fixes were applied one finding at a time.**  The P1 forensic
   corrections were individually sound, but multiple source/manifest cycles
   multiplied latency.  The full delta should have been batched before the
   one required re-run.
5. **Estimates omitted operational work.**  Startup, remote worktree
   preparation, exact-command validation, collection, aggregation, and report
   generation were not counted explicitly enough.
6. **Verification selection was not previewed before execution.**  The
   changed-test gate can legitimately select an integration test that starts a
   synthetic training subprocess.  Starting it twice in parallel is neither a
   faster verification strategy nor a scientific safeguard.

None of these justify skipping source, parent, Teacher, attack, or output
ownership validation.  They require performing those validations once, early,
and for the complete job matrix.

## Operating standard going forward

For a short campaign using an existing implementation and known inputs:

| Deadline from recorded request | Required outcome |
| --- | --- |
| T+5 min | Compact task context: exact scientific contract, source/parent/config/Teacher/mask inventory, likely host placement. |
| T+15 min | Complete host × job config matrix; unresolved path or lineage issue is reported as a concrete blocker. |
| T+25 min | Source frozen, static/CLI checks complete, immutable manifest and realistic estimate ready. |
| T+30 min | Detached controller launched, or a `launch_slo_breached` blocker is recorded with the smallest resolution path. |

The 30-minute target applies when the requested runtime already exists.  A new
objective/runtime integration may require up to 90 minutes, but the expected
duration and blocking verification must be stated at the outset—not discovered
after multiple failed launch attempts.

### Required pre-launch evidence

Before a controller is launched, record all of the following in a compact
ledger:

- complete input inventory: source SHA, parent/checkpoint, config, Teacher
  SHA, mask/calibration, attack, output roots;
- every host × job resolved config, including permitted path-only rebases;
- frozen source SHA;
- immutable manifest and planned DAG.

The reusable command is:

```bash
python .agents/skills/multi-gpu-experiment-orchestrator/scripts/launch_ledger.py \
  init --output <runtime>/orchestration/<campaign>/launch-ledger.json \
  --campaign-id <campaign> --requested-at <ISO-8601-with-offset> \
  --requested-at-precision exact --request-evidence <reference>
```

`mark`, `ready`, and `summary` maintain evidence and calculate the actual
request-to-controller time.  This ledger is operational metadata, not
scientific evidence.

### Failure handling

- One host/config mismatch stops the **entire equivalent matrix** for a
  complete revalidation.  Do not repair one epoch or one GPU at a time.
- A technical retry gets a fresh attempt/run namespace.  It retains its
  scientific identity and never reuses partially written scientific output.
- A source change after a canary or manifest freeze invalidates that canary;
  rebuild the source-bound manifest instead of treating a prior pass as proof.
- Once a job is host-confirmed stable, the detached DAG owns completion.  Do
  not spend context on routine polling.
- Before a broad changed-test gate, run `scripts/verify.py --changed
  --non-scientific --dry-run` once and inspect the selected commands.  Start
  at most one full gate; never use a second invocation as a status check.
  Prefer already-passing focused tests when the preview shows an unrelated
  long integration fixture and it cannot change the operational decision.

### Estimate contract

When I launch a long job, I will state one estimate that includes:

$$
T_{total}=T_{startup}+T_{compute}+T_{endpoint}+T_{collection}+T_{aggregate}+T_{report}.
$$

It must be based on a matching prior command or a bounded representative run;
otherwise it will be labelled an estimate with its uncertainty.  I will not
quote only raw epoch time as campaign completion time.

## What the user needs to provide

Usually, nothing beyond the scientific request, stop rule, and any real
deadline.  Codex owns the timing ledger, path resolution, frozen source,
complete host matrix, and launch estimate.  An optional prompt line is useful
when time matters:

```text
Operational target: existing-runtime campaign; target detached controller
launch within 30 minutes.  Report a concrete blocker before exceeding it.
```

That is a service-level target, not permission to bypass a scientific gate.
