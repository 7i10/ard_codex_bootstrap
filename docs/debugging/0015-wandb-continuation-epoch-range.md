# W&B continuation epoch-range contract

## Failure signature

The first live run of `scripts/analyze_wandb_ro.py` rejected the explicit
21-run cohort as incomplete. Sixteen runs had exact epochs `0..199`; each H4
factorial arm had exact epochs `100..199`.

## Evidence and root cause

A read-only W&B SDK 0.28.0 comparison fetched both
`[epoch, val_pgd_accuracy]` and
`[epoch, val_pgd_accuracy, val_clean_accuracy]`. Both forms returned 200
unique rows for the first 16 runs and 100 unique rows for every H4 arm, with
no metric nulls. H4 intentionally resumes an epoch-99 common parent, so its
child run owns only epochs 100 through 199.

The failure was not W&B sampling or missing upload. The initial analyzer
incorrectly imposed `0..199` on every run and did not represent continuation
lineage in the cohort schema.

## Fix and regression

Each cohort entry now has an optional inclusive `epoch_start` / `epoch_end`.
Defaults remain `0..199`; H4 declares `100..199`. Artifact and legacy-history
paths share one fail-closed exact-range validator. Analysis summaries include
the requested and observed range and name continuation AUC as
`val_pgd_normalized_auc_requested_range`.

Focused regressions cover a mixed full/continuation cohort, a continuation
without its declared start, an incomplete continuation, and identical
coverage enforcement for artifact/history sources. No attack, objective,
training, checkpoint, or evaluation semantics changed.
