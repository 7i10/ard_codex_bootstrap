# 0034 — FF/NR causal CE-PGD20 endpoint

## Status

- Owner: causal CE20 endpoint writer
- Current milestone: complete (2026-08-10)

## Goal

Replay the five frozen C79/RA/RAR/RB/RBR child checkpoints at each registered
L2/L4 horizon under one common eval-mode pixel CE-PGD20 endpoint and write
non-overwriting Parquet, report, and lineage artifacts.

## Frozen contracts

- Train split stable-ID/class universe: 45,000 rows, fixed SHA-256.
- Attack: Linf pixel `[0,1]`, epsilon `8/255`, step `2/255`, 20 steps,
  random start, hard CE, Student/Teacher eval mode.
- Registered W&B inventory is the only checkpoint authority.  Its payload
  epochs are 84/89/93 for requested horizons 84/89/94; no sibling `last.pt`
  is substituted.
- Fixed epoch-79 Route A/B selected and matched-random masks are loaded from
  the resolved arm config and verified byte-for-byte.

## Non-goals

No training, test split, AutoAttack, selector recomputation, treatment change,
or S2 source reconstruction.  S2 is reported unavailable until a frozen
source schema is registered.

## Verification

- Focused unit tests cover strict config, fixed attack identity, paired rescue
  / harm / spillover computations, bootstrap determinism, and overwrite guard.
- Public-CLI smoke completed for L2 h84, followed by the full six-endpoint
  sweep: L2/L4 × horizons 84/89/94. Every bundle has five arms × 45,000
  rows (225,000 rows), the expected endpoint epoch (84/89/93), 45,000 unique
  stable IDs, common CE-PGD20 attack identity, and a non-overwriting
  report/lineage pair.
- Focused tests:

  ```text
  PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest -q tests/unit/test_ffnr_causal_ce20.py
  3 passed
  /home/shunsukenaito/.conda/envs/adv/bin/ruff check src/ard/analysis/ffnr_causal_ce20.py src/ard/cli/ffnr_causal_ce20.py tests/unit/test_ffnr_causal_ce20.py
  pass
  ```

- Post-sweep validation checked all six reports, Parquet row counts, stable-ID
  uniqueness, finite values, expected endpoint mapping, shared attack and
  inventory hashes, and S2 fail-closed status.
- Results and per-bundle hashes are recorded in
  `docs/FFNR_CAUSAL_HORIZON_CE20_RESULTS.md`.

## Risks

The L4 local run-bundle manifests were stale at the start of the task. Matching
W&B run-bundle manifests, validation Parquets, and resolved configs were
recovered before the sweep and bound to the fixed inventory. The evaluator
therefore did not substitute local checkpoint files. S2 remains unavailable
because no frozen current-ERT state schema is registered; no S2 claim is made.

## Frozen decisions and handoff

- The masks, q, route definitions, and parent lineage were not changed.
- Bootstrap was fixed at 2,000 class-stratified resamples with seed 20260810;
  its intervals are sample-conditional, not training-seed uncertainty.
- No new intervention is launched automatically. Route A has no stable
  selected-minus-random gain, and Route B changes sign across horizons/seeds;
  both remain diagnostic rather than a confirmed treatment.
- Official test and AutoAttack remain outside this endpoint task.
