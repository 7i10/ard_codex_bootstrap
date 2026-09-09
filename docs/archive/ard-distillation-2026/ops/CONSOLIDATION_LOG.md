# Documentation consolidation log — 2026-09-07

What changed in `docs/` during this pass, why, and how to undo it. No number, hash, date,
verdict, or scientific claim was changed anywhere. `docs/NUMERIC_CONSISTENCY_AUDIT.md` was read
first and neither its text nor anything it points at was touched.

## What was done, and what was not

Zero files were deleted. Zero files were merged. A repository-wide scan for duplicated prose
(every line over 90 characters, compared across all 112 top-level `docs/*.md` files) found no
duplicated explanatory paragraphs — only a handful of expected, short, identically-worded attack-
identity lines (e.g. "Training attack: KL-PGD10, epsilon 8/255, ...") that two sibling reports
correctly both state for standalone readability. This project's documentation is unusually
disciplined about scope: most documents that look alike by title (for example the three FF/NR
"status" documents, or the three FFNR human-review documents) turned out, on reading, to be
distinct stages of one pipeline (protocol / results / analysis, or successive research phases),
not restatements of each other. Nothing in that category was touched.

Two real changes were made instead: a full rewrite of the navigation index, and five superseded-
document headers for a cluster of Codex-era operational documents that this project's own current
design document (`docs/CLAUDE_CODE_WORKFLOW.md`) and `CLAUDE.md` already declare frozen.

## Table

| file | action | reason | content now lives at |
|---|---|---|---|
| `docs/README.md` | **kept, rewritten** | The old index covered about 35 of 112 documents, filed "Experiment protocol" under two different headings pointing at the same file, and had no grouping by purpose. Rewritten to index every top-level document, grouped as: start here, measurement method, evidence and verdicts, experiment records (by family), plans/decisions, debugging, execution plane (current vs. superseded), foundations/reference. | n/a — nothing was removed, only the index's own structure changed. To undo: `git checkout HEAD -- docs/README.md` restores the prior version (see `git log -- docs/README.md` for the exact pre-change commit). |
| `docs/EXPERIMENT_RECONCILER.md` | **retired (header added)** | Documents `scripts/reconcile_experiment.py`, which `CLAUDE.md` and `docs/CLAUDE_CODE_WORKFLOW.md` (line 24-25: "凍結（呼ばない・依存しない）") both list as frozen. Not cited by any other document or script (checked with `grep -rl` before touching it). | Unchanged — the file's body is untouched; only a blockquote header was added below the title pointing to `docs/CLAUDE_CODE_WORKFLOW.md` (`launch_gate.py` → `orchestrate.py`, `scripts/ardx/`) as the current design. To undo: delete the four added header lines (they are the only diff). |
| `docs/EXPERIMENT_AUTOMATION_BRIDGE_HARDENING.md` | **retired (header added)** | Documents the same reconciler plus `scripts/publish_experiment_terminal_event.py` (the "PR #1 event bus") and the fast-path runtime-signature registry — all three named frozen in `CLAUDE.md`. Not cited anywhere else (`grep -rl` found zero hits, not even the old `docs/README.md`). | Unchanged body; header added. Undo: delete the added header lines. |
| `docs/EXPERIMENT_FAST_PATH.md` | **retired (header added)** | "fast path / runtime signature" is named frozen in `CLAUDE.md`. Cited by `docs/EXPERIMENT_LAUNCH_DISCIPLINE.md` and `docs/plans/0085-ard-experiment-execution-fast-path.md` (a plan; not edited, per the task's plan/decision exclusion), so the document was retired in place rather than deleted. | Unchanged body; header added, pointing to `launch_gate.py` and the `.claude/skills/experiment-launch` skill. Undo: delete the added header lines. |
| `docs/TASK_CONTEXT_PROTOCOL.md` | **retired (header added)** | `scripts/task_context.py` is named frozen in `CLAUDE.md`. Cited by `docs/WORKSPACE_CONTRACT.md` (one link), so retired rather than deleted. | Unchanged body; header added, pointing to `state.json` / `run-bundle` per `docs/CLAUDE_CODE_WORKFLOW.md`. Undo: delete the added header lines. |
| `docs/EXPERIMENT_LAUNCH_DISCIPLINE.md` | **retired (header added)** | Its launch-ledger procedure is built on `launch_ledger.py` (named frozen in `CLAUDE.md`) and it points to `EXPERIMENT_FAST_PATH.md`, itself superseded. Cited by `docs/EXPERIMENT_AUTOMATION_BRIDGE_HARDENING.md`, `docs/ARD_OPERATIONAL_FOUNDATION.md`, `docs/EXPERIMENT_FAST_PATH.md`, and the old `docs/README.md`, so retired rather than deleted. | Unchanged body; header added, pointing to `docs/CLAUDE_CODE_WORKFLOW.md` and the `.claude/skills/experiment-launch` skill. Undo: delete the added header lines. |
| all other `docs/*.md`, `docs/plans/*`, `docs/decisions/*`, `docs/debugging/*`, `docs/reviews/*`, `docs/experiments/*` | **kept, unchanged** | Reviewed (by title, and by full read for every document discussed below) and found to be either in current use, a distinct stage of a multi-document pipeline, or a completed record that later documents still cite. Indexed in the rewritten `docs/README.md`. | n/a |

## Investigated and deliberately left alone

- **`docs/ARD_OPERATIONAL_FOUNDATION.md`** also documents parts of the frozen Codex-era launch
  path (`task_context.py`, `launch_ledger.py` usage in its own walkthrough). It was *not* retired,
  because `tests/skills/test_production_launch_gate.py` cites its "layer ownership and first
  remediation" section by anchor as live evidence for a still-current gate behaviour (requirement
  `R33`). Marking the whole document superseded would misstate that a currently-tested behaviour
  is obsolete. Left untouched.
- **`docs/ERT_CLEAN_WRONG_RESCUE_SUBTYPES.md` vs. `docs/ERT_CLEAN_WRONG_RELIABILITY_STRATIFIED.md`**
  share an identical H1 title ("ERT Clean-Wrong Rescue Subtype Analysis" — the second document's
  own H1 differs slightly but was clearly built as a superset of the first) and the same C0/C10/
  C12/C13 subtype table shape. A line-by-line diff shows the underlying numbers differ from the
  third decimal up (for example C10's `clean_and_robust_rescue` mean robust-margin value is
  `0.2043` in one document and `0.2298` in the other). This is exactly the kind of disagreement
  the task instructions require preserving rather than reconciling, so both documents were left
  fully intact and are indexed separately in the new README with a note pointing at the
  discrepancy for whoever owns `docs/NUMERIC_CONSISTENCY_AUDIT.md`-style follow-up.
- **`docs/ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV.md` vs. `..._V2.md`** look like a draft/final
  pair by name. Reading both: v1 documents a launch that was *blocked* pre-launch by a fail-closed
  RNG audit; v2 documents the actual completed run under the corrected contract. They are
  sequential history, not duplicates — v1 explains why the design in v2 changed. Both kept.
- **The measurement/evidence cluster** (`MEASUREMENT_DESIGN.md`, `MEASUREMENT_STANDARD.md`,
  `EVIDENCE_RECLASSIFICATION.md`, `ERT_RESEARCH_STATUS_SUMMARY.md`, `COEFFICIENT_AUDIT.md`,
  `ARM_REGISTRY.md`) restates related concepts (the noise floor, MDE, the post-decay bracket) in
  several places, but each restatement is scoped differently on inspection: `MEASUREMENT_STANDARD.md`
  itself states its relationship to `MEASUREMENT_DESIGN.md` ("`MEASUREMENT_DESIGN.md` が「過去に
  何が起きたかの監査」、本書が「これから何をするかの規則」" — one is the audit of the past, the
  other is the rule for the future), and `EVIDENCE_RECLASSIFICATION.md` states in its own header
  that it is built on the other three and does not re-derive what they establish. This is
  deliberate layering, not sprawl, so none of the six were merged or retired.

## Net effect

- Files edited: **6** (`docs/README.md` plus five superseded-document headers).
- Files deleted: **0**.
- Files merged: **0**.
- Every edit outside `docs/README.md` is a 4-6 line blockquote inserted immediately after the
  document's title; no other line in any of those five files was touched.
