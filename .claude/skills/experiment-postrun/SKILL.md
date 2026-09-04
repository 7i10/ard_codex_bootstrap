---
name: experiment-postrun
description: Close out a finished campaign or hand-run — re-derive terminal status, verify outputs, import the record and report into docs/, append the evidence ledger row, close plan milestones, run /verify, make one results commit, then hand off to /experiment-decide. Use when a campaign ends, when the watcher fires, or when the user says postrun/import/record results.
---

# experiment-postrun

`$ARGUMENTS` = `<campaign-or-run id> [--status S] [--kind campaign|run] [--state-path P]`.
Treat `--status` / `--kind` as hints only; re-derive the truth below.

## 1. Re-derive terminal status (the only completion signal)

`--state-path` (the watcher passes it) is the `state.json` / `run-bundle/manifest.json`
that fired the event — use it and do not guess a path. Without it, ask the user or
find the file; the campaign id is **not** a directory name and is not unique
(`ert-i100-online-state-s2-v1` has three attempt dirs).

```bash
# campaign: --roots is the directory containing the campaign run dir
python3 scripts/ardx/campaign_watch.py --once --emit-existing \
  --roots <run dir's parent> --state /tmp/ardx-postrun-<slug>.json
# hand-run: the bundle half needs --include-hand-run, or nothing is emitted
python3 scripts/ardx/campaign_watch.py --once --emit-existing --include-hand-run \
  --roots <bundle's parent dir> --state /tmp/ardx-postrun-<slug>.json
```

Keep the line whose `path` is `--state-path` (fall back to `id` == `<id>` only when
exactly one line matches). If `terminal` is false → say so and stop. Take `success`
and `failure_class` from that line, never from the campaign-level `status` field.

## 2. Idempotency check

If `docs/experiments/<contract>.json` already exists and its sha256 matches the
committed `.sha256`, report `already imported` and stop. Records are never
overwritten.

## 3. Success path

1. Verify every job's `expected_outputs` exists on disk.
2. The DAG usually already ran the aggregation: verify its outputs and that
   their SHAs match. Only if the aggregate node did not run, run the aggregator
   from a pinned worktree at the campaign source SHA, e.g.
   `PYTHONPATH=src <python> scripts/aggregate_<contract>.py --campaign <root>
   --expected-source-sha <sha>` and let `--result` / `--report` default.
3. Import at the aggregator's default paths: record → `docs/experiments/<contract>.json`,
   report → `docs/<REPORT>.md`. Write `docs/experiments/<contract>.json.sha256`
   containing the bare hash (`sha256sum ... | cut -d' ' -f1`).
4. Add a `Provenance` line to the report footer linking the record path and the
   source SHA.
5. Append exactly one row to the evidence ledger table in
   `docs/ERT_RESEARCH_STATUS_SUMMARY.md`: `question | result | decision`.
6. Tick only the plan milestones the evidence actually closes, and write the
   plan's Completion report: source SHA, lineage, decision, caveats.
7. Run `/verify`. Fix nothing scientific to make it pass; if it is red, stop.
8. One commit, `Record <campaign> results`, containing exactly: record, `.sha256`,
   report, plan, ledger.
9. Write the decision packet via `/experiment-decide <campaign>`, then
   `timeout 5 notify-send "ARD postrun <id>: recorded"` if available (a bare
   `notify-send` blocks ~60-75 s here waiting on D-Bus activation).

## 4. Failure path

Read `failure_class` from step 1 and the last attempt's logs, which live under
`<run dir>/orchestration/<campaign>/<manifest-hash>/<job-hash>/*.log`
(plus `<run dir>/orchestration/state.json.controller.log`), where `<run dir>` is
the parent of the `--state-path` `orchestration/` directory.

- `technical_retryable` → write a diagnosis and **one** proposed retry command
  in the plan's Progress log. Do not run it.
- `scientific` or `unknown` → write the decision packet with the evidence and stop.

## Stop conditions

- Not terminal, or no `state.json` / `run-bundle/manifest.json` was located.
- Record already imported with a matching sha256.
- A declared `expected_output` is missing, or an aggregate SHA disagrees.
- `/verify` is red.
- Any number in the generated report disagrees with the record.

## Never

- Never launch or retry training, evaluation or any GPU job from this skill.
- Never overwrite an existing record, report or `.sha256`.
- Never hand-edit a number, table or conclusion in a generated report; fix the
  aggregator and regenerate, or stop.
- Never run the aggregator from this live checkout — use the pinned worktree.
- Never call `reconcile_experiment.py` or the terminal-event publisher.
- Never `git push`, rewrite history, or commit outputs, caches or checkpoints.
- Never choose the next experiment; that is `/experiment-decide` and the human.
