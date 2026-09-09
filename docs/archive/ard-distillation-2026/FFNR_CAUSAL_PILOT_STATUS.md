# FF/NR causal-pilot preparation status

更新日: 2026-08-09

この段階では、Chen ERT seed 1/2のepoch-79親から、選択情報を固定した短期pilotの準備と一回限りのloss-scale dry-runまでを実施した。将来endpoint、official test、AutoAttackはselector作成・dry-runには使用していない。

## Fixed inputs

- parent: W&B `model-chen-rslad-observed-{s1,s2}-confirm-v2-last:v15`
- parent epoch: `79`, `epoch_boundary=end`
- parent payload: model、optimizer、scheduler、RNG、sampler、SampleStateStoreを含む
- parent SHA-256: seed 1 `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`、seed 2 `9b51bca767871ada6c80c75ad92997f9b7f246c0c1e35f3edad35d4e787a4a9c`
- attack: train KL-PGD10、`L_inf=8/255`、step `2/255`、random start、pixel `[0,1]`
- selector attack observations: CE-PGD20、同じepsilon/step/random-start、train split 45,000 stable IDs

## Selection masks (selection-time information only)

`scripts/prepare_ffnr_causal_pilot.py` が次を生成した。mask bytesは`.cache/analysis/ffnr-causal-pilot-masks-v1/`に保存され、future labelは読み込まない。

| run | Route A selected | Route A matched random | Route B pool | Route B q=5% | Route B q=10% |
|---|---:|---:|---:|---:|---:|
| Chen seed 1 (L2) | 5,667 | 5,667 | 22,472 | 1,124 | 2,247 |
| Chen seed 2 (L4) | 5,755 | 5,755 | 22,755 | 1,138 | 2,276 |

Route AはStudent strong-wrong ∩ clean-wrong ∩ Teacher adversarial-wrong。Route BはStudent strong-correct ∩ Teacher adversarial-correctからstrong margin riskの上位を選ぶ。matched randomはclass、clean/robust state、margin decileを一致させた。

q=5%と10%は事前候補として固定した。候補をvalidation結果で変更しない。

## Coefficient dry-run

`scripts/ffnr_loss_scale_dry_run.py` を各parentで実行した。256件の固定train batch、一回のCE-PGD10、optimizer updateなしで、baseline RSLADと仮想selected-only CE/KD変更のloss/gradient scaleを比較した。

| route | candidate | seed 1 loss ratio / grad ratio | seed 2 loss ratio / grad ratio |
|---|---|---:|---:|
| A | adversarial KD `0.5` + adversarial CE `0.25` | 1.038 / 1.004 | 1.038 / 1.005 |
| A | adversarial KD `0.5` + adversarial CE `0.50` | 1.081 / 1.019 | 1.081 / 1.028 |
| B | KD `1.0` + adversarial CE `0.25`, q=5% | 1.009 / 1.006 | 1.007 / 1.000 |
| B | KD `1.0` + adversarial CE `0.50`, q=5% | 1.017 / 1.012 | 1.014 / 1.001 |

dry-runだけからは、Route A `KD=0.5, CE=0.25`、Route B `KD=1.0, CE=0.25`が最も保守的な候補である。これは性能の採用判定ではない。

## Runtime gate and launch state

Route A/B専用のhash-bound arm schema、selected-only adversarial CE branch、epoch79 common-parent forkを実装した。旧C/HS/RS/HD/RD armは変更していない。Route Bはq=5%をprimary候補として固定し、q=10%は感度候補として保存した。

focused contract test後に、両seedの5-arm screen（C79/RA/RAR/RB/RBR、epoch79→84）を作成し、fork checkpointのresume/lineageを検証してからGPUを起動する。既存のstrong diagnostic cacheは分析の参考に残すが、current clean SHAで再生成されていないものを新規因果結果とは主張しない。

## Horizon-94 continuation (completed)

The same five arms from the same epoch-79 parents were continued to
`training.epochs=94` for both seeds without changing masks, q, coefficients,
or treatment. All ten arms reached metric epoch 93 (the zero-based horizon-94
endpoint) and wrote final sample-state Parquets. Validation trajectory and
training-state diagnostic results are documented in
`docs/FFNR_CAUSAL_HORIZON_EXTENSION_RESULTS.md`. The sample-state contrasts
are explicitly not treated as the preregistered CE-PGD20 causal endpoint;
that endpoint still requires a common eval-mode replay.
