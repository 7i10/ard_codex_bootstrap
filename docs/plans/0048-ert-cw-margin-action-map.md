# ERT Clean-Wrong Teacher-Margin × Action Map

Status: completed read-only analysis.

## Frozen scope

- Use only the completed Broad Screen C0–C15 L2/L4 epoch-84 train
  CE-PGD20 endpoint artifacts.
- Use the existing hash-bound epoch-79 CE-PGD20 and KL-PGD10 Teacher-margin
  feature replays; do not start a new replay or load a checkpoint.
- Partition each fixed epoch-79 Clean-Wrong train cohort into five independent
  pre-treatment quantiles, sorted by `(teacher_adv_margin, sample_id)`.
- Report paired stable-ID clean/robust accuracy, rescue, harm, net rescue, and
  probability-margin effects versus C0, plus factorial contrasts, Pareto sets,
  and CE20/KL10 action-ranking agreement.
- Do not select a threshold, winner, coefficient, new seed, or follow-up run.

## Inputs and lineage

- Broad Screen source: `cbe03a7b3be0b11fa1555b573c6f453a3d10f27b`.
- L2/L4 epoch-79 parent and fixed Clean-Wrong mask hashes are recorded in the
  machine report.
- Endpoint attack: independent eval-mode CE-PGD20, identity
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- CE20 and KL10 feature metadata, row hashes, attack identities, class joins,
  and quantile ID hashes are recorded in the machine report.

## Outputs

- `docs/ERT_CW_MARGIN_ACTION_MAP.md`
- `docs/experiments/ert_cw_margin_action_map_v1.json`
- `scripts/analysis/ert_cw_margin_action_map.py`

Held-out subtype transfer is intentionally unavailable because the existing
pre-treatment feature replays are train-only. The result is a descriptive
action map, not a validated online router or a causal held-out claim.
