# ERT state-conditioned mechanism pilot — M0 overlay

更新日: 2026-08-11  
Status: `M0 complete; Stage A GPU training not started`

## What was executed

This is the CPU-only first stage of the state-conditioned mechanism pilot. It
does not train, select a new route, tune a threshold, or evaluate the official
test/AutoAttack.

- Teacher: Chen2021LTD_WRN34_10 (ERT)
- Runs: L2/seed 1 and L4/seed 2
- Anchor: exact epoch-79 parent state
- Existing endpoint overlay: train-split CE-PGD20 at horizons 84, 89, 94
- State quantiles: fixed 10% primary and 20% sensitivity overlay
- Endpoint attack: pixel `[0,1]`, Linf `8/255`, step `2/255`, 20-step random-start
  hard-label CE, eval mode

The final public command was run from tracked-clean commit `5bc0a1b`:

```bash
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
  -m ard.cli.ert_state_overlay \
  --config configs/analysis/ert_state_overlay_v1.yaml \
  --output-dir .cache/analysis/ert-state-overlay-v1
```

It completed successfully and wrote two 45,000-row Parquet state tables, two
fixed-mask JSON files, the overlay report, and a hash-bound lineage file. The
final implementation rejects endpoint-to-parent child-run/config/epoch drift.

## Anchor populations

`S3` is the pilot treatment state: student clean-correct and adversarial-wrong.
For completeness, the state table also partitions the student universe into
`S1` (adversarial-correct), `S2` (clean-wrong and adversarial-wrong), and `S3`.
Teacher `T1` is adversarial-correct outside the fixed lower positive-margin
quantile, `T2` is the fixed lower positive-margin quantile, and `T3` is
adversarial-wrong. Teacher clean correctness remains a separate modifier.

| run | S3 | S3×T1 q10 | S3×T2 q10 | S3×T3 q10 | clean-wrong | clean-wrong×T-clean-correct |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 13,841 | 9,889 | 2,162 | 1,790 | 8,623 | 5,538 |
| L4 | 13,277 | 9,368 | 2,138 | 1,771 | 8,925 | 5,826 |

The 20% counts are in the machine mask files. The state table contains the
complete anchor S1/S2/S3 partition, while the endpoint overlay intentionally
reports only the registered pilot S3×T1/T2/T3 cohorts and clean-wrong cohorts;
the old endpoint bundles do not contain registered S1/S2 treatment masks.
No endpoint correctness or future failure label is a column in the state table
or used to construct a mask. The registered teacher dominance signal is
`max_wrong_probability - true_probability = -mT_adv`.

## Existing CE-PGD20 overlay (exploratory)

The overlay joins the fixed anchor states to the already completed old
Route-A/Route-B endpoint observations. The numbers below are old `RB` selected
treatment-minus-C79 paired robust deltas within the new state cohorts; they are
not causal effects of the new state masks.

| run | cohort | h84 | h89 | h94 | interpretation |
|---|---|---:|---:|---:|---|
| L2 | S3×T1 q10 | +2.23 pp | +0.14 pp | +0.20 pp | not stable across horizons |
| L2 | S3×T2 q10 | +1.43 pp | +0.42 pp | −0.05 pp | sign/size weakens |
| L2 | S3×T3 q10 | +0.00 pp | +0.11 pp | −0.17 pp | no stable benefit |
| L4 | S3×T1 q10 | −0.43 pp | +0.29 pp | +4.23 pp | horizon-dependent |
| L4 | S3×T2 q10 | +0.65 pp | +0.19 pp | −0.94 pp | sign changes |
| L4 | S3×T3 q10 | −0.40 pp | +0.62 pp | +0.40 pp | no stable benefit |

Clean harm is nonzero in every listed cohort/horizon (for example, L2 S3×T2
is 9.67%, 8.28%, and 7.68% at h84/89/94; L4 S3×T3 is 7.00%, 9.71%, and
14.12%). This confirms why the old Route-B result cannot be promoted directly
to a new route. Rescue/harm and probability-margin deltas are retained in the
machine report for all arms/cohorts, including old Route A and clean-wrong
strata.

## Lineage and artifacts

The overlay binds each run to its exact epoch-79 parent fork lineage and
state/replay inputs. Parent checkpoint hashes are:

- L2: `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`
- L4: `9b51bca767871ada6c80c75ad92997f9b7f246c0c1e35f3edad35d4e787a4a9c`

Final output hashes (generated from `5bc0a1b`):

| artifact | SHA-256 |
|---|---|
| L2 state table | `8200a9a429ff98576c1dfc138fae7bd85cf695aa002e24bd505bc160f5d251da` |
| L4 state table | `808c6cca72490379005ef6d7533ee81fffbec4163422d5dbac4d92a1809e7c` |
| L2 mask | `af6d150b709a7c1ad55f40d2a822888b8f126ca0ad20a4039573bb674d1eb656` |
| L4 mask | `49fb0a3b9a5ccb0176e90bdfe183bbac1dab2e1a02e7104ebc78dbd5e61d1031` |
| report | `267fec3d8d249f61a64a729bd3a7363f5c9277bf8a3542b320f36ec60965fea4` |
| lineage | `f8c68d7d92c034d4382905489685e6c300c865d75851498e71bf46e2a38228c5` |

Machine output is intentionally ignored by Git at
`.cache/analysis/ert-state-overlay-v1/`.

## Next step

Stage A is not automatically launched from these overlay values. Before GPU
training, the component configs must freeze the numeric moderate KD-softening
temperature (the attachment did not specify it; the current engineering
assumption is `T=2.0`) and the exact clean-CE/Clean-KD branch coefficients.
After that contract is represented and tested, one canary arm is launched from
the same epoch-79 parent; the remaining 12 unique treatments per seed are
submitted only after the canary path and lineage pass.
