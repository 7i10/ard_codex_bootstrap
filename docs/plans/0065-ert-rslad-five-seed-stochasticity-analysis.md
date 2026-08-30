# ERT / RSLAD Five-Seed Global & Sample-Level Stochasticity Analysis

## Status

- Owner: Codex root
- Status: complete (read-only analysis)
- Source HEAD: `17cd2d3bb4ef28b1ec567cac49def2a37c023d3f`
- Seeds: `dev-1`, `dev-2`, `confirm-a`, `confirm-b`, `confirm-c`
- Arms: `BASE`, `CROPSHIFT`, `I100`

## Scope and non-goals

Use existing training metrics and CE-PGD20 endpoint rows only. No training,
new seed, RNG fork, endpoint attack change, augmentation change, History or
Ordering intervention, official test, or AutoAttack is permitted.

## Frozen contracts

- Endpoint: CE-PGD20, pixel `[0,1]`, $\epsilon=8/255$, step `2/255`, 20
  steps, random start, eval mode, hard-label CE.
- Endpoint attack identity:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- Validation stable-ID/label identity:
  `16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4`.
- Registered sample epochs: `49, 99, 149, 199`.
- I100 before epoch 100 is the shared CROPSHIFT prefix; it is not counted as
  an independent trajectory there.

## Execution record

1. Repository and lineage reconciled at the source HEAD above.
2. Inventory built from the existing static-trajectory, stage-wise, and
   unseen-confirmation artifacts. All 5 seeds × 3 conceptual arms × 4
   endpoint epochs have 5,000-row stable-ID artifacts.
3. Global metrics were assembled without imputation. BASE has dense five-seed
   metrics; CROPSHIFT and I100 have dense post-100 confirmation suffixes and
   development prefixes. Confirmation prefix dense metrics were not retained
   and remain explicitly sparse.
4. Sample-level robust/clean correctness, margin dispersion, sign agreement,
   rank correlation, Jaccard, entropy, class summaries, rescue/loss, and
   cross-seed rescue consistency were computed. Dependency-free SVGs and the
   underlying CSV/NPZ data were written under the ignored analysis cache.
5. Machine-readable tracked summaries and the human report are produced after
   inspecting the outputs. No endpoint regeneration was needed.

## Acceptance / interpretation guardrails

- Endpoint row count and stable-ID/label joins must match for all cells.
- Accuracy deltas are kept separate from margin deltas; rescue minus loss is
  checked against paired accuracy change.
- Cross-seed results are descriptive (`n=5`), not population-level inference.
- Missing dense confirmation prefixes are not reconstructed from endpoints.
- Global variance reduction and treatment mean shift are reported separately.

## Outputs

- `docs/experiments/ert_rslad_five_seed_artifact_inventory_v1.json`
- `docs/experiments/ert_rslad_five_seed_global_stochasticity_v1.json`
- `docs/experiments/ert_rslad_five_seed_sample_stochasticity_v1.json`
- `docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md`

## Decision

The read-only analysis is complete. Existing artifacts support a full
sample-level comparison and a partial-but-explicit dense global comparison;
no training or endpoint regeneration is required. Any History/Ordering
follow-up requires a separate human-reviewed experiment prompt.
