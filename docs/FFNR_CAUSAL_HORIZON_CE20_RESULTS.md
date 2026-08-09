# FF/NR causal horizon: common CE-PGD20 endpoint

## Scope and status

This report closes the extended Chen ERT causal pilot for both development
seeds. The registered epoch-79 masks were evaluated at horizons 84, 89, and
94 (the horizon-94 W&B payload is checkpoint epoch 93, because the training
epoch index is zero based). No training, selector/q/loss/routing change,
official test evaluation, or AutoAttack was run.

The six endpoint bundles are:

| seed | horizons | output bundle |
|---|---|---|
| L2 / Chen seed 1 | 84, 89, 94 | `.cache/analysis/ffnr-causal-ce20-results/{smoke-L2-h84,L2-h89,L2-h94}` |
| L4 / Chen seed 2 | 84, 89, 94 | `.cache/analysis/ffnr-causal-ce20-results/{L4-h84,L4-h89,L4-h94}` |

The L2 h84 directory is the public end-to-end smoke output. Its analysis
source-file hashes are byte-identical to the final revision used by the other
five bundles; its lineage records the preceding config-only commit
`0c5c859a`. The final source revision is `b461e23ce0b40ab7508ecddb6b75774e8f3e2fc3`.

## Frozen endpoint and cohorts

- pixel domain `[0,1]`, `L_inf`, `epsilon=8/255`, step `2/255`;
- 20-step random-start hard-label CE, with student and teacher in eval mode;
- every arm generated its own adversarial examples; no control adversarial
  examples were reused;
- control `C79`, Route A `RA`/`RAR`, and Route B `RB`/`RBR`;
- masks were registered at epoch 79 and were not re-selected at a later
  horizon;
- Route A cohort sizes are 5,667 (L2) and 5,755 (L4); Route B sizes are
  1,124 (L2) and 1,138 (L4);
- rescue/harm are paired robust-correctness transitions relative to the
  same-horizon `C79` continuation. They are not individual causal effects of
  a treatment on an independently retrained model.

The bootstrap uses 2,000 class-stratified resamples with seed `20260810`.
Its interval is conditional on the observed training seed and is not a
training-seed uncertainty interval.

## Paired endpoint results

Robust and clean accuracies are percentages. `S-C` and `R-C` are treatment
minus same-cohort control differences. `S-R` is the selected-minus-random
treatment effect. `margin Δ` is the adversarial probability-margin change
(treatment minus control, shown in percentage points); this is not a logit
margin. The final column is validation CE-PGD20 accuracy for the selected arm
and matched-random arm, respectively.

The `S/R rescue/harm/net` columns are percentages. `spillover` is the
selected-treatment effect on the complement of the selected mask, shown as
robust-accuracy delta / clean-accuracy delta / robust harm rate.

| seed | h | route | selected robust | random robust | selected control | random control | S-C | R-C | S-R (95% CI) | S/R rescue/harm/net | selected/random margin Δ | non-selected spillover | validation S/R |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| L2 | 84 | A | 0.688% | 2.982% | 0.424% | 2.612% | +0.265 pp | +0.371 pp | −0.106 pp [−0.547,+0.371] | 0.494/0.229/0.265 / 1.341/0.971/0.371 | +1.6597 / +1.2597 | −2.431 / −1.269 / 6.870% | 45.600/46.320% |
| L2 | 84 | B | 56.050% | 65.925% | 52.224% | 60.320% | +3.826 pp | +5.605 pp | −1.779 pp [−5.249,+1.512] | 10.854/7.028/3.826 / 11.477/5.872/5.605 | +1.5421 / +1.5396 | +0.273 / −0.666 / 4.162% | 46.940/47.580% |
| L2 | 89 | A | 0.424% | 2.470% | 0.247% | 1.553% | +0.176 pp | +0.918 pp | −0.741 pp [−1.147,−0.318] | 0.388/0.212/0.176 / 1.465/0.547/0.918 | +0.8590 / +0.3775 | −0.913 / −0.351 / 5.769% | 45.280/43.440% |
| L2 | 89 | B | 57.028% | 61.833% | 51.690% | 59.786% | +5.338 pp | +2.046 pp | +3.292 pp [−0.267,+6.940] | 11.655/6.317/5.338 / 10.943/8.897/2.046 | +1.7825 / +0.6451 | +0.169 / −0.793 / 4.137% | 46.480/45.240% |
| L2 | 94 | A | 0.671% | 3.265% | 0.653% | 3.088% | +0.018 pp | +0.176 pp | −0.159 pp [−0.653,+0.300] | 0.459/0.441/0.018 / 1.341/1.165/0.176 | +0.4756 / +0.8089 | −0.318 / +1.688 / 5.649% | 46.520/46.100% |
| L2 | 94 | B | 53.203% | 64.413% | 51.601% | 57.740% | +1.601 pp | +6.673 pp | −5.071 pp [−8.185,−1.868] | 8.452/6.851/1.601 / 12.011/5.338/6.673 | +1.8654 / +2.5688 | −0.144 / +0.189 / 4.754% | 46.420/46.140% |
| L4 | 84 | A | 0.295% | 3.023% | 0.382% | 2.884% | −0.087 pp | +0.139 pp | −0.226 pp [−0.695,+0.209] | 0.174/0.261/−0.087 / 1.407/1.268/0.139 | +0.0817 / +0.4152 | +1.363 / +0.966 / 4.704% | 46.300/46.740% |
| L4 | 84 | B | 57.118% | 63.533% | 49.033% | 55.536% | +8.084 pp | +7.996 pp | +0.088 pp [−3.427,+3.515] | 14.148/6.063/8.084 / 12.830/4.833/7.996 | +1.9676 / +2.3311 | +0.488 / +0.182 / 3.976% | 45.340/46.440% |
| L4 | 89 | A | 0.539% | 4.240% | 0.295% | 2.919% | +0.243 pp | +1.321 pp | −1.077 pp [−1.581,−0.608] | 0.434/0.191/0.243 / 2.189/0.869/1.321 | +1.8655 / +1.9215 | −0.775 / −0.910 / 5.820% | 46.320/46.060% |
| L4 | 89 | B | 55.097% | 63.005% | 52.988% | 60.105% | +2.109 pp | +2.900 pp | −0.791 pp [−4.130,+2.636] | 9.578/7.469/2.109 / 11.072/8.172/2.900 | +0.9395 / +1.6883 | −0.666 / −0.388 / 4.676% | 46.160/47.000% |
| L4 | 94 | A | 0.365% | 2.433% | 0.139% | 1.668% | +0.226 pp | +0.765 pp | −0.539 pp [−0.938,−0.139] | 0.295/0.070/0.226 / 1.442/0.678/0.765 | +1.0474 / +0.7176 | +0.624 / −0.148 / 5.122% | 45.560/45.120% |
| L4 | 94 | B | 53.251% | 56.766% | 47.276% | 52.724% | +5.975 pp | +4.042 pp | +1.933 pp [−1.757,+5.536] | 13.445/7.469/5.975 / 12.302/8.260/4.042 | +2.9750 / +1.7889 | +1.190 / −0.128 / 4.307% | 46.200/44.460% |

For example, the first rescue/harm/net triplet is selected Route A and the
second is matched-random Route A; Route B is interpreted the same way. The
large Route B cohort-level accuracy values coexist with small or negative
selected-minus-random effects because the two fixed cohorts have different
baseline difficulty.

## Clean-harm audit

The endpoint schema stores clean correctness and clean accuracy deltas. For
this report, clean harm is additionally derived from the same paired Parquet:
control clean-correct and treatment clean-wrong. This is a CPU-only stable-ID
join, not a new evaluation. Values are selected/random clean harm rates; the
corresponding clean rescue rates are shown for completeness.

| seed | h | A selected harm/rescue | A random harm/rescue | B selected harm/rescue | B random harm/rescue |
|---|---:|---:|---:|---:|---:|
| L2 | 84 | 2.894/7.800% | 4.853/5.841% | 2.758/1.957% | 1.512/1.423% |
| L2 | 89 | 3.017/8.135% | 5.594/9.864% | 1.335/1.512% | 0.890/1.157% |
| L2 | 94 | 3.088/6.158% | 2.876/8.805% | 1.957/3.114% | 1.157/1.779% |
| L4 | 84 | 3.997/5.995% | 5.178/6.881% | 1.670/2.197% | 0.879/1.318% |
| L4 | 89 | 3.197/5.960% | 4.709/9.018% | 2.021/2.109% | 1.142/2.285% |
| L4 | 94 | 2.954/7.281% | 4.709/8.601% | 1.845/2.724% | 0.615/1.757% |

## Validation-wide CE-PGD20 accuracy

This table is the full validation aggregate, not a selected-cohort estimate.

| seed | h | C79 | RA | RAR | RB | RBR |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 84 | 46.760% | 45.600% | 46.320% | 46.940% | 47.580% |
| L2 | 89 | 46.360% | 45.280% | 43.440% | 46.480% | 45.240% |
| L2 | 94 | 46.240% | 46.520% | 46.100% | 46.420% | 46.140% |
| L4 | 84 | 45.280% | 46.300% | 46.740% | 45.340% | 46.440% |
| L4 | 89 | 46.560% | 46.320% | 46.060% | 46.160% | 47.000% |
| L4 | 94 | 44.980% | 45.560% | 45.120% | 46.200% | 44.460% |

## Interpretation bounded to this endpoint

1. Route A selected-minus-random is negative at every horizon for both seeds;
   L2 h89 and L4 h89/h94 have class-stratified intervals entirely below zero.
   This does not support Route A selection as a robust-treatment improvement.
2. Route B changes sign across horizons and seeds. L2 h89 is positive but its
   interval crosses zero; L2 h94 is negative with an interval below zero. The
   L4 intervals all cross zero. There is no stable Route B advantage.
3. Robust rescue and harm are both non-negligible. Clean-harm rates show that
   the intervention can alter clean correctness, so robust gains must not be
   reported without the clean-harm audit.
4. These are fixed-mask, common-endpoint diagnostics. They do not select a
   new q, threshold, selector, route, or intervention. The next intervention
   remains blocked pending a preregistered redesign; no automatic launch is
   triggered by this report.

## S2 exploratory overlay

S2 was not computed. No frozen, hash-bound current-ERT state schema was
registered for this endpoint bundle, so the CLI reports
`s2_overlay.available=false` and fails closed rather than reconstructing state
from an unregistered source. This is an analysis-input limitation, not a
negative S2 result.

## Reproducibility and hashes

- final analysis source Git SHA: `b461e23ce0b40ab7508ecddb6b75774e8f3e2fc3`;
  source-file digest: `78d2bda00dc9f03fe671cd2e0b5cb1b9aaf173b41e6808486d60f13eb447957c`;
- checkpoint inventory SHA: `a16fc9f9412bb19ce7e6b084713a7c4101b9b70a1a7bab80565d60973a132950`;
- CE-PGD20 attack identity SHA (all six):
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`;
- stable train partition universe: 45,000 IDs with the hash recorded in each
  lineage file;
- each endpoint bundle contains `causal-ce20-observations.parquet`,
  `report.json`, and `lineage.json`. Their SHA-256 values are:

| bundle | observations | report | lineage |
|---|---|---|---|
| L2 h84 (smoke) | `02701aded7f262f952b789c96cdab6e967c56b60b659bec9497f30e20739937b` | `0e3aea05a1bf517b3852bbfc1fab0dd43c9ade4ac2f8d1548c4982091e06f90` | `371da834cb3d6ad60473d028aec0b883393fc1f91fc484b66f22f8d3dfbdcb55` |
| L2 h89 | `c4322e035e75cba5480201143eed440bf29f5d3b1659faefcde09846414d6146` | `c728d16bafc59b4e4f89162d1f35b4beebb26885e4097577e56faf84d06bbc6c` | `f5fdae5198f7c3f288fa6980d9d8088cb7797ea75f56859d586d2100156878e3` |
| L2 h94 | `3c464c6ba12517fde1e004ebf196ff8a19e12573ae890f489c0f4e018e08fc8d` | `878779de2adfecc1f27a62a25bd87e4b44346c1d5e6199e72ca6ca01756be6a0` | `d0d689640cd6e64fab3e69e0c544990acbc37a4eb5dd91b96a92b5365449ced5` |
| L4 h84 | `6871e6045e784e57c2c92b12645e51b145e55c9c723d0c8991b765f586850f1e` | `6932bc9b11d276bf412526d377b90cce0b57395d488cc21c87a08c971c697c3a` | `fe770150e70d21f0202ff8b82aa033e6bde19d7a541342c1b8f3d8b35245fa35` |
| L4 h89 | `cbd0c81f15235fb8172ff9fb4f2c4e7406b5a86677fda17deb4e53764b2ab257` | `fa500dbb2694230dd79ac3da2fe3fd2ce613d2d73e0d99d8d87ef1fc40276e56` | `d32b324d0a44cef3829eb01fc86752391152cecf8566119f50a7332baab0f807` |
| L4 h94 | `fe31cd4786daf879cea897bcbfd69fd0f83cc59f3b1bbd7e19f4a9b3e06a1ac9` | `619b3483b4fe4896b57fd58e9e5120e8b9bdcbc51c0f427fc0d0f9d20aaafa2a` | `6ca6b41363b1caead1ecfa634b5c56c7158442011ff973e74750b1e63f71de5b` |

Per-arm checkpoint SHA-256 values and mask IDs are embedded in each
`lineage.json`; the inventory and mask hashes were checked once and reused by
all six runs.

## Exact execution and verification

The public command was run once per endpoint, with only `--label`,
`--horizon`, `--device`, and output directory varied:

```bash
PYTHONPATH=src python -m ard.cli.ffnr_causal_ce20 \
  --config configs/analysis/ffnr_causal_ce20_v1.yaml \
  --label L2 --horizon 89 --device cuda:0 \
  --output-dir .cache/analysis/ffnr-causal-ce20-results/L2-h89
```

The same command shape was used for L2/L4 and horizons 84/89/94. Focused
verification before the sweep was:

```bash
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
  -m pytest -q tests/unit/test_ffnr_causal_ce20.py
# 3 passed
/home/shunsukenaito/.conda/envs/adv/bin/ruff check \
  src/ard/analysis/ffnr_causal_ce20.py src/ard/cli/ffnr_causal_ce20.py \
  tests/unit/test_ffnr_causal_ce20.py
# pass
```

The post-sweep check verified all six reports, 225,000 rows per bundle,
45,000 unique stable IDs, five arms, expected endpoint epoch, common attack
identity, finite output values, and S2 fail-closed status.
