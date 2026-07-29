# 実験ダッシュボード

最終スナップショット: **2026-07-29 14:18 JST**

このページは、人間が現在の研究目的、条件、進捗、結果、W&B上のrunの役割を一か所で確認するための
台帳です。実行中の値は変化するため、論文用の確定表ではありません。

## 1. 実験概要と目的

研究質問は、**教師自身のadversarial robustnessが高いほど、常に優れた蒸留教師になるのか**、そして
teacher overconfidenceとstudentの時変robust learnabilityを分離して観測・制御すると、この関係を
説明または改善できるか、です。

現在のseed-0 core campaignは、次の2教師と4手法の8セルを同一条件で比較します。

| 軸 | 条件 |
|---|---|
| ERT側の教師 | `Chen2021LTD_WRN34_10`。公開AA参考値56.94% |
| IRT側の教師 | `Bartoldson2024Adversarial_WRN-94-16`。公開AA参考値73.71% |
| 手法 | RSLAD、SAAD entropy weighting、student robust-margin、joint risk |
| 主な問い | 教師AAの高さ、entropy、student risk、両者の積が学生の頑健性へどう影響するか |

教師AA値はRobustBenchの公開参考値であり、このリポジトリでfull AutoAttack再現した値ではありません。
ローカルではcheckpoint SHA、normalization、forward、1000例のbounded PGD監査までを実施済みです。

### 手法の実際の意味

| 手法 | 追加する信号・処理 |
|---|---|
| `rslad` | uniformなRSLAD KD baseline |
| `rslad_entropy` | `5 * (H_i - min_batch H)`。clip、平均保存、hard-label fallbackなし |
| `rslad_student` | adversarial student probability marginのEMAからriskを計算 |
| `rslad_joint` | student risk × `1 - H/log(C)` |

schema-v2のstudent/joint主経路は、riskに応じてadversarial KDのteacher targetを一様分布へ最大0.5だけ
softenします。KD weightは1、hard-label weightは0のままで、sampleを削除しません。旧KD/CE fallbackは
別の明示的ablationであり、今回の8セルには含まれません。

## 2. 共通実験条件

| 項目 | 固定値 |
|---|---|
| Scientific Git SHA | `2d54b8230b8d14d13c1ea7472ccba53491b4d38d` |
| Dataset | CIFAR-10、trainの10%をseed `20260722`でvalidationへ固定分割 |
| Official evaluation split | CIFAR-10 test 10,000例。train中には使用しない |
| Student | `saad_resnet18_cifar_v1` |
| Epoch / seed | 200 epochs / seed 0 |
| Execution identity | `ws1_prb128_gb128_localbn_v1` |
| BatchNorm / batch | 1 GPU、local BN、per-rank/global batch 128 |
| Optimizer | SGD、LR 0.1、momentum 0.9、weight decay `5e-4` |
| Scheduler | MultiStep、milestones 100/150、gamma 0.1 |
| Train attack | pixel-space Linf KL PGD-10、epsilon `8/255`、step `2/255`、random start |
| Selection/official PGD | pixel-space Linf CE PGD-20、同じepsilon/step、random start |
| Checkpoints | validation PGD最大のbestとepoch 199のlastを別々に保存・評価 |
| AutoAttack | saved checkpointを別processでstandard suite評価。指定セルのみ |
| Tracking | W&B online、train/evaluationを別run、同一groupへ所属 |

このsingle-GPU cohortと、旧world-size 2 / per-rank batch 64の結果は、local BatchNorm統計が異なるため
同じ集計へ混ぜません。

## 3. 現在の進捗

「完了」は、そのセルに事前登録された全phaseが終わったことを表します。student-onlyはPGDまで、
RSLAD/entropy/jointはPGDとAutoAttackまでが予定phaseです。

| 状態 | Host/GPU | Teacher / method | 現在の証拠 |
|---|---|---|---|
| 完了 | Hamster 0 | Chen / RSLAD | 200 epoch、PGD、AA完了 |
| 完了 | Hamster 0 | Bartoldson / Student | 200 epoch、PGD完了。AAは計画対象外 |
| 評価待ち | Hamster 1 | Bartoldson / Entropy | 200 epoch、PGD完了。AAは同GPUのtrain後に実行 |
| 実行中 | Hamster 1 | Chen / Entropy | epoch 116、val clean/PGD 84.52% / 56.54% |
| 実行中 | Ferret 0 | Bartoldson / RSLAD | epoch 25、val clean/PGD 72.90% / 40.90% |
| 未着手・queue済み | Ferret 0 | Chen / Student | Bartoldson RSLADの後に開始 |
| 実行中 | Ferret 1 | Bartoldson / Joint | epoch 25、val clean/PGD 72.58% / 41.60% |
| 完了 | Ferret 2 | Chen / Joint | 200 epoch、PGD、AA完了 |

Hamster GPU 0が現在idleでも、実行中にjobを別GPUへ移すと静的割当・lineageが変わるため、Hamster
GPU 1のqueueを移動しません。W&Bは数epoch遅れて見える場合があり、phase状態と最新epochはhost-local
manifest/metricsを正とします。

### 現在未着手の研究

- core seed-0ではChen / Studentのみが未開始です。
- seed 1/2、複数seed統計、teacher感度のmean/std/worst/bestは未着手です。
- controlled protocolでのPGD-AT、TRADES、full SAAD直接比較は未着手です。
- CIFAR-100、MobileNetV2、Tiny-ImageNet本訓練は未着手です。
- これらを現在のseed-0結果から自動的に開始する設定にはしていません。

## 4. 現在の正式結果

以下はすべてCIFAR-10 official test **10,000例**です。PGDはCE PGD-20、AAはstandard AutoAttackです。
数値はaccuracy (%)で、validation値とは混ぜていません。

| Teacher / method | Best clean | Best PGD | Best AA | Last clean | Last PGD | Last AA |
|---|---:|---:|---:|---:|---:|---:|
| Chen / RSLAD | 83.18 | 55.65 | 51.90 | 83.22 | 55.44 | 51.78 |
| Chen / Joint | 83.08 | 55.46 | 51.65 | 83.07 | 55.17 | 51.54 |
| Bartoldson / Entropy | 85.24 | 50.09 | 未評価 | 85.11 | 48.72 | 未評価 |
| Bartoldson / Student | 84.71 | 50.53 | 計画外 | 85.00 | 45.98 | 計画外 |

別profileの参考値:

| Execution profile | Teacher / method | Best clean / PGD | Last clean / PGD |
|---|---|---:|---:|
| world size 2、per-rank 64、local BN | Chen / RSLAD | 83.51 / 55.88 | 83.42 / 55.61 |

### 現時点で言えること

- Chenでは、seed 0のJointはRSLADに対してbest PGDで`-0.19 pp`、best AAで`-0.25 pp`です。改善とは
  判定できません。
- Bartoldsonの完了済み2手法では、Studentのbest PGDはEntropyより`+0.44 pp`ですが、best-to-last
  PGD低下はStudent `4.55 pp`、Entropy `1.37 pp`です。
- Bartoldson / RSLADとJoint、およびChen / EntropyとStudentが揃っていないため、教師差と手法差を
  分離した結論はまだ出せません。
- すべてseed 0なので、現在値は探索的結果です。複数seedとfull SAAD/direct baselineなしに論文の主要結論
  とはしません。
- AutoAttack manifestのlibrary versionは現在`unknown`です。checkpoint、attack version、epsilon、seed、
  実行環境は保存されていますが、論文用追加runまでにAutoAttack source/package revisionを明示固定する
  必要があります。

## 5. 出力

### ローカル

各train outputには、少なくとも次が残ります。

- `resolved_config.yaml`
- `best.pt`、`last.pt`
- `run-bundle/manifest.json`、`environment.json`、`metrics.jsonl`、completion marker
- `sample-stats-train.parquet`
- 固定sample panelとartifact manifest

各evaluation outputには、次が残ります。

- `evaluation-results.json`
- `evaluation-lineage.json`
- best/lastのParquet sample statisticsと固定sample panel
- 独立したevaluation run bundle

Host別root:

- Hamster: `/home/shunsukenaito/workspace-local/ard-campaign-runs/ard_codex_bootstrap/c10-r18-ws1-b128-core-s0-v1-2d54b82`
- Ferret: `/home/shunsukenaito/workspace-local/ard-runs/ard_codex_bootstrap/c10-r18-ws1-b128-core-s0-v1-2d54b82`

### W&B

[single-teacher-ard project](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard)には、trainの
epoch/step metrics、best/last summaries、model/run-bundle/sample-stat/Table artifactsと、saved-checkpoint
evaluationが別runとして保存されます。evaluation runはtrain runの重複ではなく、評価分離の証拠です。

## 6. W&B 26 runの整理

APIで確認した時点では、**14 train + 12 evaluation = 26 run**、状態は**23 finished + 3 running**でした。
crashed状態はなく、26件すべてが少なくとも1つのartifactを持っています。

### A. 論文候補cohortの重要run — 13件

この13件は削除対象にしません。正式な結果はevaluation run、再現性とcheckpoint lineageは対応するtrain
runが担うため、両方が必要です。

| セル | Train | PGD evaluation | AutoAttack evaluation |
|---|---|---|---|
| Chen / RSLAD | [`prod-chen-rslad-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-rslad-s0-2d54b82) | [`eval-7008…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-7008ebd671ee70b41990) | [`eval-7ebd…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-7ebd06a00e4f8f2d996d) |
| Chen / Entropy | [`prod-chen-entropy-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-entropy-s0-2d54b82) | 未作成 | 未作成 |
| Chen / Student | 未作成 | 未作成 | 計画外 |
| Chen / Joint | [`prod-chen-joint-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-joint-s0-2d54b82) | [`eval-1eefd…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-1eefd911db6c022c20f6) | [`eval-eafdd…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-eafdd255adb7eb585582) |
| Bartoldson / RSLAD | [`prod-bart-rslad-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-rslad-s0-2d54b82) | 未作成 | 未作成 |
| Bartoldson / Entropy | [`prod-bart-entropy-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-entropy-s0-2d54b82) | [`eval-8ff4…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-8ff463c11843dfdd9b24) | 未作成 |
| Bartoldson / Student | [`prod-bart-student-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-student-s0-2d54b82) | [`eval-3839…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-3839fd906dfc2aa5b318) | 計画外 |
| Bartoldson / Joint | [`prod-bart-joint-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-joint-s0-2d54b82) | 未作成 | 未作成 |

### B. 重要だが別profileのreference — 2件

- Train: [`ard-32a10cb8a2cab31a`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/ard-32a10cb8a2cab31a)
- PGD: [`eval-1ef1…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-1ef1f9bf888a3f2ef1fb)

これはworld-size 2のChen / RSLAD referenceです。削除しませんが、Aのsingle-GPU cohortへ集計しません。

### C. 受理済みengineering pilot — 6件

最終SHAの3 trainと対応する3 PGD evaluationです。精度比較には使いませんが、VRAM、remote detachment、
checkpoint/evaluation遷移、Joint warmupの受入証拠なので保持します。

- Train: `pilot-h-chen-rslad-s0-2d54b82`、`pilot-f-bart-rslad-s0-2d54b82`、
  `pilot-h-chen-joint-s0-2d54b82`
- Evaluation: `eval-9bfe992e255a93963ea5`、`eval-4d63212beaf3ba36872d`、
  `eval-6954e9df43e0a4e6839e`

### D. 論文解析には不要な旧pilot — 5件

- 旧world-size 2 pilot: `ard-cde030a72ddca4b9`、`eval-92f3750eb628e93d6060`
- superseded SHA `712b878`: `pilot-h-chen-joint-s0-712b878`、
  `pilot-h-chen-rslad-s0-712b878`、`eval-6f99576e8fa285a61f12`

`712b878`のtrain metricsは最終受理pilotと一致し、科学解析上は重複です。ただし全5件にartifactがあり、
過去のgroup-length修正とpilot受入経緯を追跡できます。**active campaign中は削除せず**、W&Bの通常viewから
除外するのが安全です。campaign完了後に容量または画面の簡潔さを優先する場合、local manifestを保存した上で
まず`712b878`の3件を削除候補にできます。旧world-size 2 pilotの2件は固有の履歴なので、その次の候補です。

### 推奨するW&B view

現在run tagsは全件空です。campaign中に命名規則を変更せず、まず次のfilterで4 viewへ分けます。

1. **Paper candidate**: `tier=production`かつgroupに`-ws1-`を含む
2. **Official evaluation**: 1に加えて`job_type=evaluation`
3. **Reference ws2**: groupに`localbn-ws2`を含む
4. **Pilots/history**: `tier=pilot`

campaign完了後に、同じ分類を`paper-candidate`、`reference-ws2`、`accepted-pilot`、`superseded` tagとして
付与すると見通しが良くなります。既存evaluation run名はPGD/AAを画面上で判別しにくいため、次campaignでは
display nameへ`pgd`または`autoattack`を含めます。seed-0 campaign途中ではlineageを変えないため実施しません。

## 7. 更新ルール

- statusはhost-local job JSONと最新`metrics.jsonl`から更新する。
- 正式結果は`evaluation-results.json`に10,000例が揃った時だけ記載する。
- W&B countと分類はAPIで再確認し、train/evaluationを重複扱いしない。
- seed追加や別execution profileは同じ表へ無条件に混ぜない。
- runを削除する前に、対応するlocal manifest、checkpoint、evaluation、artifact依存を確認する。
