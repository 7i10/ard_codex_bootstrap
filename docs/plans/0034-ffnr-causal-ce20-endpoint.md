# 0034 — FF/NR causal CE-PGD20 endpoint

## Status

- Owner: causal CE20 endpoint writer
- Current milestone: implementation

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
- Public-CLI smoke is one label/horizon after a tracked-clean commit.  A full
  L2/L4 sweep remains blocked if any manifest, parent, validation, or
  inventory byte fails lineage validation.

## Risks

Local L4 run-bundle manifests are absent/stale relative to the recovered W&B
checkpoint inventory.  The evaluator fails closed and must not use these
checkpoint substitutes until matching run-bundle/validation artifacts are
recovered.
