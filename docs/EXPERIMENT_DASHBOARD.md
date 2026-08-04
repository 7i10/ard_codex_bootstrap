# 実験ダッシュボード

科学結果スナップショット: **2026-08-05 JST**

このページは、人間が研究目的、条件、確定済み進捗、結果、W&B上のrunの役割を一か所で確認するための
台帳です。live process状態は本文へ手入力せず、run bundleとW&Bから導出します。

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

### 手法の区分と実際の意味

`★ Proposed`はこの研究基盤の提案手法または、その効果を分離するための提案ablationです。
RSLAD baselineおよびSAAD由来のentropy-onlyとは表・集計で分けます。

| 区分 | 手法 | 追加する信号・処理 |
|---|---|---|
| Baseline | `rslad` | uniformなRSLAD KD baseline |
| SAAD-derived ablation | `rslad_entropy` | `5 * (H_i - min_batch H)`。clip、平均保存、hard-label fallbackなし |
| ★ Proposed ablation | `rslad_student` | adversarial student probability marginのEMAからriskを計算 |
| ★ Proposed main | `rslad_joint` | student risk × `1 - H/log(C)` |

Studentは、各sampleのadversarial predictionについて
`margin = p_student(y|x_adv) - max_{c != y} p_student(c|x_adv)`を測り、decay 0.9のEMAを保持します。
`student_risk = (1-margin_ema)/2`なので、正解classが他classより十分優勢ならriskは低く、誤分類または
境界付近なら高くなります。

Jointはこのstudent riskへ、teacherのoverconfidence
`teacher_risk = 1 - H(teacher(x_adv))/log(C)`を掛けます。したがって、**studentがそのsampleを頑健に
学びにくく、かつteacherが低entropyで過信している場合だけ**riskが大きくなります。

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
| AutoAttack | saved checkpointを別processでstandard suite評価。全8セル（Student 2セルは事後追加） |
| Tracking | W&B online、train/evaluationを別run、同一groupへ所属 |

このsingle-GPU cohortと、旧world-size 2 / per-rank batch 64の結果は、local BatchNorm統計が異なるため
同じ集計へ混ぜません。

## 3. 現在の進捗

live local statusは次で生成します。epoch metricsと同時にmanifestが更新されるため、通常の確認に
`nvidia-smi`、PID polling、watchdogは不要です。別hostのrunはW&B projectをlive sourceとします。

```bash
PYTHONPATH=src python -m ard.cli.status \
  --root <Hamster-or-collected-output-root> \
  --format markdown
```

以下はseed-0 cohort完了時の科学スナップショットであり、現在のGPU占有を表しません。

「完了」は、そのセルに現在登録された全phaseが終わったことを表します。当初student-onlyはPGDまで、
RSLAD/entropy/jointはPGDとAutoAttackまでが予定phaseでした。空きGPUを利用するユーザー判断により、
2026-07-29 23:33 JSTからStudent 2セルにも同一standard AutoAttackを事後追加しています。訓練条件や
checkpoint selectionは変えていません。

| 状態 | Host/GPU | Teacher / method | 現在の証拠 |
|---|---|---|---|
| 完了 | Hamster 0 | Chen / RSLAD | 200 epoch、PGD、AA完了 |
| 完了 | Ferret 2 | Bartoldson / ★ Student | 200 epoch、PGD、事後追加AA完了 |
| 完了 | Ferret 2 | Bartoldson / Entropy | 200 epoch、PGD、AA完了 |
| 完了 | Hamster 0 | Chen / Entropy | 200 epoch、PGD、AA完了 |
| 完了 | Ferret 0 | Bartoldson / RSLAD | 200 epoch、PGD、AA完了 |
| 完了 | Hamster 1 | Chen / ★ Student | 200 epoch、PGD、事後追加AA完了 |
| 完了 | Ferret 1 | Bartoldson / ★ Joint | 200 epoch、PGD、AA完了 |
| 完了 | Ferret 2 | Chen / ★ Joint | 200 epoch、PGD、AA完了 |

2026-07-29の運用判断として、Ferretへ3本のtrainを集中させず、Hamster GPU 0へChen/Student
train+PGD、Ferret GPU 2へBartoldson/Entropy AutoAttackを再配置しました。両sequenceはexit code 0で
完了しています。移送は未開始phaseだけを対象とし、checkpoint SHA、source SHA、実行GPU UUID、
元job IDを記録しました。
W&Bは数epoch遅れて見える場合があり、phase状態と最新epochはhost-local manifest/metricsを正とします。
Chen/StudentはW&B run `prod-chen-student-s0-2d54b82`として完了しました。Bartoldson/Entropy
evaluationはW&B run `eval-6dcf1b78a77d3258b2e0`として完了しました。

重複起動を防ぐため、ユーザー承認後にHamster/Ferretのwatchdogとcontrollerだけを一時停止しました。
Ferretの2 trainは、限定watcherがexit code 0、completion marker、Git SHA、GPU UUID、GPU lease
ownershipを検証してから同じGPUでPGD→AAだけを起動し、全phaseがexit code 0で完了しました。
seed-0 core campaignとfollow-up解析の全phaseはterminalであり、完了済みjobの重複trainはありません。
このterminal reconciliation時点で対象GPUはidleでした。再配置した5 jobは、immutable result/checkpoint/sequence digestを含むportable
evidenceからowning hostへatomic batch import済みです。再importは両hostでstrict no-opとなり、canonical
campaign stateは両方とも`awaiting_scientific_review`へ到達しました。証跡は
[`docs/experiments/reconciliation/`](experiments/reconciliation/)にあります。

### 次段階の未着手研究

- core seed-0の8 train、8 PGD、8 AutoAttackはすべて完了しました。Student 2セルのAAは事後追加評価です。
- 複数seedの性能評価は未完了です。student-history信号の2教師×2 seed
  confirmatory blockは完了しましたが、これは性能主張用の3-seed cohortではありません。
- controlled protocolでのPGD-AT、TRADES、full SAAD直接比較は未着手です。
- PGD-AT/TRADESのcanonical configとCUDA synthetic smoke、および
  Chen/Bartoldson observed RSLAD config/parity gateは準備済みです。
  PGD-AT/TRADESの200 epoch実験は未開始です。
- CIFAR-100、MobileNetV2、Tiny-ImageNet本訓練は未着手です。
- Best-oriented history-routing v2のdevelopment blockは完了しNo-Goです。停止規則により
  Bartoldson seeds 3/4、Chen no-harm、v2 official PGD/AAは未着手のままです。
- これらを現在のseed-0結果から自動的に開始する設定にはしていません。

### Student-history confirmatory block（完了）

seed-0だけの関連を一般化せず、teacher依存・seed依存・proxyからexact online historyへの移行を同時に
確認するため、lossへ介入しないobserved RSLADを2教師×2 seedで実行します。predictor、feature、anchor
epoch 99、future outcome、split、判定閾値はseed 1/2の結果を見る前に
[`history_confirmatory_block_v2.yaml`](../configs/analysis/history_confirmatory_block_v2.yaml)へ固定しました。

| Cell | Teacher | Seed | Host/GPU | W&B run | Status |
|---|---|---:|---|---|---|
| L1 | Bartoldson | 1 | Hamster 1 | `bart-rslad-logging-only-s1-confirm-v1` | 200 epoch完了、解析採用 |
| L2 | Chen | 1 | Hamster 0 | `chen-rslad-observed-s1-confirm-v2` | 200 epoch完了、解析採用 |
| L3 | Bartoldson | 2 | Ferret 0 | `bart-rslad-observed-s2-confirm-v2` | 200 epoch完了、解析採用 |
| L4 | Chen | 2 | Ferret 1 | `chen-rslad-observed-s2-confirm-v2` | 200 epoch完了、解析採用 |

L2--L4の固定SHAは`8254a8899ae7373c2f541d108593e5c8185b26f5`です。L1の旧
`rslad_logging_only`と新しい`rslad + observation.teacher_response`の直接cross-version CUDA bridgeは、同一
RTX 4090・同一環境の二epoch deterministic fixtureでcheckpoint/sample state equalityを満たしました。このため
L1は旧Git lineageを明示したまま同じ観測解析へ含めます。これは200 epoch性能の同値性を主張するものではありません。
証跡は`tools/internal/history_replication/provenance/observation_bridge_2026-08-01.yaml`です。live epochはこの表へ
追記せずW&Bまたは`ard.cli.status`を参照します。

補正済みH5-Early/H5-LateとH4aも完了しました。exact online historyはcurrent marginを両Bartoldson seedで
上回り、Best目的の最早anchorはepoch 39です。一方、Bartoldsonのteacher-adversarial-wrong例は
future-forgetting群で`6/10110`、`4/8856`しかなく、teacher-wrong-only gateは主routeに使いません。

epoch-79から分岐したdelayed-schedule controlも2 seedとも完了しました。normal RSLAD比の平均差はBest
`+0.12 pp`、Last `+0.70 pp`、RO gap `-0.58 pp`です。Best改善としては小さいためschedule自体の成功主張は
せず、次の介入に対する強いmatched controlとして使います。

### Best-oriented history routing v2（完了、Development No-Go）

[事前登録plan](plans/0020-best-oriented-history-routing-v2.md)どおり、epoch-39 online historyのPF/NR上位10%と
class/state/count-matched randomへ同じtrue-label anchorを適用し、Bartoldson 2 seedで比較しました。

| 比較（2 seed平均） | Best PGD | Last PGD | RO gap | 判定 |
|---|---:|---:|---:|---|
| PF-TA - C | -0.18 pp | -0.38 pp | +0.20 pp | No-Go |
| PF-TA - PF-R | -0.13 pp | -0.18 pp | +0.05 pp | selector優位なし |
| NR-TA - C | -0.17 pp | -0.15 pp | -0.02 pp | No-Go |
| NR-TA - NR-R | +0.15 pp | +0.11 pp | +0.04 pp | randomには小さく一貫して優位 |

NRでhistory選択がrandomを両seedともBestで上回った一方、通常のdelayed RSLAD controlを超えませんでした。
したがってstudent historyの予測力は維持しつつ、今回のtrue-label target mixをBest改善手法として停止します。
official test、AutoAttack、未使用seed、Chen no-harmは開発判断に使っていません。完全なrun表、trajectory AUC、
lineageと次の判断は[History-routing v2 results](HISTORY_ROUTING_V2_RESULTS.md)に固定しました。

Ferret GPU 2で同一1-epoch workloadを比較した結果、teacher-response観測時は4 workersが
387.4 images/s、8 workersが338.4 images/sで、4 workersが14.48%高速でした。loss・accuracyは一致しており、
次回のFerret launchは`ARD_NUM_WORKERS=4`を既定とします。進行中runの設定は変更しません。CIFAR-10は
startup後にmemory residentとなるため、dataset再配置はsteady-state差の主因ではありません。prefetch/pinned
transferは未変更で、worker調整後にも残るhost差をprofileする場合だけ追加します。

### ★ Prescriptive v3 intervention screen（実行中）

v2の失敗分析から、Student historyを予後予測として残し、介入をPFの時間的一貫性とNRの入力側learnabilityへ
分離しました。PFはepoch-79 Student anchorをteacher targetへ25%混合し、NRは同じPGD-10 trajectoryのstep-5
prefixをepoch 80--99だけ選択サンプルへ使います。true-label hardening、KD weight変更、attack budget変更は
行いません。詳細な数式、mask、停止基準は[Plan 0022](plans/0022-prescriptive-v3-intervention-screen.md)です。

| 状態 | Host/GPU | Cell | W&B run ID | 現在の証拠 |
|---|---|---|---|---|
| 実行中 | Hamster 0 | L1 / PF-H | `pv3-bart-s1-9792655-pf_ret_h` | epoch 80 smoke完了、val PGD 41.26% |
| 実行中 | Hamster 1 | L1 / NR-H | `pv3-bart-s1-9792655-nr_pfx_h` | epoch 80 smoke完了、val PGD 38.78% |
| 実行中 | Ferret 0 | L3 / PF-H | `pv3-bart-s2-9792655-pf_ret_h` | fixed-SHA process / W&B online確認 |
| 実行中 | Ferret 1 | L3 / NR-H | `pv3-bart-s2-9792655-nr_pfx_h` | fixed-SHA process / W&B online確認 |
| 実行中 | Ferret 2 | L3 / PF-R | `pv3-bart-s2-9792655-pf_ret_r` | smoke合格後に開始 |
| GPU待ち | Hamster | L1 / PF-R | `pv3-bart-s1-9792655-pf_ret_r` | fork済み、未学習 |
| GPU待ち | Hamster | L1 / NR-R | `pv3-bart-s1-9792655-nr_pfx_r` | fork済み、未学習 |
| GPU待ち | Ferret | L3 / NR-R | `pv3-bart-s2-9792655-nr_pfx_r` | prepared fixed-SHA worktree / fork済み |

全runはGit `97926553ed6773666df915860460b90c353e721d`、1 GPU、batch 128、local BN、
delayed milestones 120/170へ固定しています。以前のFerret 4-worker profileは有望ですが、このpaired screenでは
fork契約を変えないため8 workersを維持します。開発結果はvalidationだけで判定し、official test・AutoAttack・
Chen・未使用seedはGo判定まで封印します。

## 4. 現在の正式結果

以下はすべてCIFAR-10 official test **10,000例**です。PGDはCE PGD-20、AAはstandard AutoAttackです。
数値はaccuracy (%)で、validation値とは混ぜていません。

| Teacher / method | Best clean | Best PGD | Best AA | Last clean | Last PGD | Last AA |
|---|---:|---:|---:|---:|---:|---:|
| Chen / RSLAD | 83.18 | 55.65 | 51.90 | 83.22 | 55.44 | 51.78 |
| Chen / Entropy | 82.87 | 55.12 | 51.06 | 83.03 | 55.06 | 51.00 |
| Chen / ★ Student | 83.35 | 55.44 | 51.46 | 83.59 | 55.21 | 51.41 |
| Chen / ★ Joint | 83.08 | 55.46 | 51.65 | 83.07 | 55.17 | 51.54 |
| Bartoldson / RSLAD | 83.39 | 51.26 | 47.11 | 84.55 | 45.55 | 43.12 |
| Bartoldson / Entropy | 85.24 | 50.09 | 47.37 | 85.11 | 48.72 | 46.16 |
| Bartoldson / ★ Student | 84.71 | 50.53 | 46.89 | 85.00 | 45.98 | 43.07 |
| Bartoldson / ★ Joint | 84.03 | 51.30 | 47.31 | 84.93 | 45.39 | 42.89 |

別profileの参考値:

| Execution profile | Teacher / method | Best clean / PGD | Last clean / PGD |
|---|---|---:|---:|
| world size 2、per-rank 64、local BN | Chen / RSLAD | 83.51 / 55.88 | 83.42 / 55.61 |

### これは何の再現実験か

現在の8セルは、**RSLAD原論文の完全再現ではなく、SAAD論文のteacher分析を基準にしたcontrolled
reproduction + 新規ablation**です。

共通しているのはCIFAR-10、ResNet-18、200 epochs、batch 128、SGD、LR milestones 100/150、PGD-10
training、epsilon `8/255`、step `2/255`、およびSAAD論文でERT/IRT分析に使われたChen/Bartoldson
checkpointです。一方、次が異なります。

- [RSLAD原論文](https://arxiv.org/abs/2108.07969)の主結果はTRADESで訓練した別のWRN-34-10教師と
  300 epochsを使用します。
- 現在はseed 0のみですが、SAADの主要表は3 seed平均です。
- 現在は固定validation splitでbestを選択し、world-size 1 / local BNを実行identityとして固定しています。
- `rslad_entropy`はSAAD entropy weightingをRSLADへ載せた軽量ablationで、AdaAD+IGDM inner、
  teacher input gradient、SWAを含む**full SAADではありません**。
- Student/Jointはこの研究基盤の新規ablationで、RSLAD/SAAD論文に同名の公開目標値はありません。

### 論文・公開記録との比較

| Source / setting | Clean | PGD | AA | 現在値との関係 |
|---|---:|---:|---:|---|
| RSLAD原論文: WRN-34-10教師、RSLAD best、300 epochs | 83.38 | 55.94 | 51.49 | Chen/RSLAD現在値は`-0.20 / -0.29 / +0.41 pp`だが、教師とepochが異なる |
| SAAD Table 1/10: Chen34-10教師、RSLAD | — | — | 52.21 | Chen/RSLAD現在AA 51.90は`-0.31 pp` |
| SAAD Table 4: Bart94教師、RSLAD、3-seed mean | 84.28±0.11 | 47.17±0.33 | 44.42±0.34 | 現在のBart/RSLADは`-0.89 / +4.09 / +2.69 pp`。protocol差がある |
| SAAD Table 4: Bart94教師、full SAAD、3-seed mean | 84.27±0.18 | 53.39±0.23 | 50.34±0.08 | Entropy-only best AA 47.37は`-2.97 pp`。同一手法ではない |
| SAAD Table 16: Gowal28教師、RSLAD + SAAD weighting | 81.74 | 50.54 | 48.53 | Entropy weighting単体の機構参照。教師が異なる |

RSLAD原論文のPGD欄は`PGD_TRADES`であり、現在のofficial CE PGD-20とは攻撃lossが同一ではありません。
この行は実装のsanity referenceであって、厳密な数値差判定には使いません。

Sources:

- [RSLAD paper, CIFAR-10 best/last tables](https://arxiv.org/pdf/2108.07969)
- [SAAD paper, ERT/IRT analysis, three-seed results, and weighting ablation](https://arxiv.org/html/2512.10275)

SAAD Table 1では、teacher AAがChen 56.94%からBartoldson 73.71%へ上がる一方、RSLAD student AAは
52.21%から44.07%へ低下し、Bartoldsonのrobust-overfitting gapは5.44 ppと報告されています。これが
現在の2-teacher比較で再確認したい主要現象です。full SAADではBartoldson/ResNet-18のAAが
50.34±0.08%まで改善したと報告されています。

### 現時点で言えること

- RSLADのbest AAはChen 51.90%、Bartoldson 47.11%で、より頑健なBartoldson教師を使うとstudent AAが
  `-4.79 pp`低下しました。seed 0のcontrolled protocolでも、研究対象のERT/IRT paradoxは方向として
  再現しています。
- ChenではRSLADに対し、Entropy/★ Student/★ Jointのbest AAはそれぞれ`-0.84/-0.44/-0.25 pp`です。
  提案2手法を含め、baseline改善はありません。
- BartoldsonではRSLADに対し、Entropy/★ Student/★ Jointのbest AAは`+0.26/-0.22/+0.20 pp`です。
  Jointの`+0.20 pp`はseed 0単独では成功と判定できず、last AAはRSLADより`-0.23 pp`です。
- best-to-last AA低下はBartoldson RSLAD `3.99 pp`、Entropy `1.21 pp`、★ Student `3.82 pp`、
  ★ Joint `4.42 pp`です。Entropy weightingはrobust overfittingを最も抑えましたが、best AAの改善は
  `+0.26 pp`に留まり、full SAAD報告値50.34%には`-2.97 pp`です。
- **現行のstudent-aware介入が明確に成功した証拠はありません。** Student単独は両教師でbest AAを下げ、
  JointはChenで下げ、Bartoldsonでごく小さく上げただけです。一方、後述のprospective auditではstudent
  signal自体に追加予測情報が確認されました。したがってsignalを捨てず、sample selectionと介入方法を分離して
  検証します。
- すべてseed 0なので、現在値は探索的結果です。複数seedとfull SAAD/direct baselineなしに論文の主要結論
  とはしません。
- 完了済みAutoAttack manifestのlibrary versionは`unknown`のまま改変しません。両hostで事後確認した
  installは同一commit/source digestでしたが、これは歴史的processが読み込んだbytesの同時証明ではありません。
  8 resultのimmutable digestと制約は
  [`0002-autoattack-provenance-amendment.json`](experiments/0002-autoattack-provenance-amendment.json)
  に追補しました。今後のrunは固定commit/source digest不一致でfail closedし、論文用確定評価ではsaved
  best/lastをその契約下で再評価します。

### Seed-0 prospective signal audit

W&Bの全8 train runについてmodel artifact version historyをread-onlyで列挙しました。各runに
`last:v0`から`last:v39`があり、Student/Jointの`last:v19`（epoch 99）と`last:v39`（epoch 199）には
45,000 sampleの`SampleStateStore`が保存されています。epoch-99 teacher riskは、保存checkpointから
raw train splitとexact KL PGD-10 attackを使って決定論的に再計算しました。

主outcomeはepoch 99より後のforgettingです。epoch 199 riskと同時点のerrorは別の探索的関連として扱い、
Go/No-Go判定には使いません。

| Teacher / ★ method | Teacher-only AUROC | Student-augmented AUROC | Delta | 95% CI | 判定 |
|---|---:|---:|---:|---:|---|
| Chen / ★ Student | 0.853 | 0.898 | +0.045 | [0.040, 0.051] | Signal Go |
| Chen / ★ Joint | 0.859 | 0.896 | +0.037 | [0.031, 0.043] | Signal Go |
| Bartoldson / ★ Student | 0.586 | 0.923 | +0.337 | [0.325, 0.349] | Signal Go |
| Bartoldson / ★ Joint | 0.602 | 0.937 | +0.335 | [0.323, 0.347] | Signal Go |

全4セルでheld-out log-lossも改善しました。これはstudent signal、特にBartoldsonでの追加情報を支持しますが、
現行target-softeningの有効性を示すものではありません。CIはclass-stratifiedなsample-level bootstrapで、
seed-0訓練runに条件付けられており、訓練seed間の不確実性ではありません。詳細とreport hashは
[Seed-0 signal audit](SIGNAL_AUDIT.md)に固定しています。

### ★ Frozen outcome-informed mask介入（完了、inconclusive）

signalの良否とtarget-softening介入の良否を分離するupper-bound実験です。Bartoldson/RSLAD seed 0の
W&B周期checkpoint `last:v19`（epoch 99）と`last:v39`（epoch 199）をraw train splitで同じKL PGD-10へ
再投入し、最終robust error 3,566 sampleをoracle maskとして固定しました。official testはmask作成に
使っていません。同数・同一class分布のrandom maskを3つ作り、全4 runでselected sampleだけ
`rho=0.5`のuniform target softeningを適用します。このmaskはpersistent failureを多く含むため、治療効果の
oracleではなく、**outcome-informed failure mask**と呼びます。

この3,566件はendpoint定義で、epoch 99正解からepoch 199誤りが404件、両endpointで誤りが3,162件です。
後者は2 endpointだけでは「全期間で永続的に誤り」と「途中で正解した後に再び誤り」を分離できません。
また、後続のhistory predictorが予測する任意のpost-anchor正解→誤りtransitionとは別outcomeです。
したがって3,566は介入budgetの比較には再利用できますが、forgetting predictorの正解maskや本番閾値としては
扱いません。

| 状態 | Ferret GPU | W&B train ID | Mask |
|---|---:|---|---|
| 完了 | — | `bart-oracle-soft-s0-05cd0c6` | outcome-informed failure mask |
| 完了 | — | `bart-rand1-soft-s0-05cd0c6` | class-matched random 1 |
| 完了 | — | `bart-rand2-soft-s0-05cd0c6` | class-matched random 2 |
| 完了 | — | `bart-rand3-soft-s0-05cd0c6` | class-matched random 3 |

実行SHAは`05cd0c66367e399dde266bd898c3ddc4097ca95c`です。best clean/PGD/AAはmask arm
`83.66/50.77/47.36`、random 1 `84.61/49.82/46.35`、random 2
`83.56/50.83/47.37`、random 3 `83.33/50.51/47.04`でした。random平均AAは`46.92`、mask差は
`+0.44 pp`、clean低下は`0.17 pp`です。maskはrandom平均を上回るためNo-Goではありませんが、
事前固定した`+0.5 pp`と全random超過を満たさないため判定は`inconclusive`です。このarmはdeployableな
提案法ではなく、History replicationの独立した介入結果です。

### Common-trajectory history gate（完了、History Go）

介入を一切変えないcanonical RSLADの周期checkpointを同一attackで再推論し、epoch 99のcurrent state、
student history、teacher entropyから後続checkpoint-panel forgettingを予測します。final-SHA
`d3c59b1`のChen replayは45,000 IDでterminalとなり、history-onlyは最良current-state baselineよりAUROC
`+0.06220`（paired 95% CI `[+0.05470,+0.07001]`）、log-loss `-0.07418`でHistory Goでした。
Bartoldson replayも45,000 IDでterminalとなりました。history-onlyは最良current-state baseline
（instantaneous margin）よりAUROC `+0.05250`（paired 95% CI
`[+0.04549,+0.06027]`）、log-loss `-0.05837`でHistory Goです。両report hashとfrozen-mask 4-arm
terminal結果はhash-bound launch designへ固定済みです。

### H2 confirmatory history block（完了、全L1--L4 History Go）

凍結済みの四trajectory解析では、L1 (Bartoldson seed 1)、L2 (Chen seed 1)、
L3 (Bartoldson seed 2)、L4 (Chen seed 2) の history-only AUROC delta が
それぞれ `+0.0499`、`+0.0686`、`+0.0513`、`+0.0663` となり、全runが
History Goでした。方向は両teacher・両seedで再現し、このblockではChenのdeltaが
Bartoldsonより大きい記述的傾向です。teacher-response と student-history の
product は予測上の記述項であり、介入の因果効果やteacher×signal機構の独立確認ではありません。

各cellは `AUROC / AUPRC / log-loss`（前二つは高いほど、log-lossは低いほど良い）です。

| Run | Teacher-only | Student-only | Main effects | Main + product |
|---|---:|---:|---:|---:|
| L1 Bartoldson s1 | .624 / .801 / .584 | .922 / .939 / .302 | .920 / .934 / .303 | .920 / .933 / .302 |
| L2 Chen s1 | .848 / .831 / .476 | .893 / .825 / .394 | .894 / .827 / .385 | .894 / .829 / .389 |
| L3 Bartoldson s2 | .626 / .800 / .584 | .927 / .946 / .293 | .924 / .938 / .294 | .924 / .937 / .294 |
| L4 Chen s2 | .852 / .832 / .471 | .891 / .825 / .398 | .893 / .827 / .388 | .893 / .828 / .392 |

Bartoldsonではteacher追加がStudent-onlyより一貫して悪化し、Chenではmain
effectsの小さな改善がある一方、productはlog-lossを悪化させました。このため
次の介入selectorはstudent historyを主信号とし、teacher積gateは採用しません。

次の状態は feature-only L3 replay、凍結selector mask生成、続くcommon-state
5-arm screenです。H3は「post-H2 prospective intervention screen v2」であり、
retroactive preregistrationでもindependent confirmationでもありません。現時点で
live replayやH3 launchは開始済みと扱いません。

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

## 6. W&B runの整理

2026-07-30 09:03 JSTのAPI監査では、既存の**15 train + 22 evaluation = 37 run**が全件
`finished`でした。その後、frozen-mask train 4件をonlineで開始し、全armのtrain、PGD、AutoAttackが
terminalになりました。ここでの37件はその追加run前の
スナップショットであり、現在の総数ではありません。削除判断前にはW&B APIで再集計します。

### A. 論文候補cohortの重要run

このcohortは削除対象にしません。正式な結果はevaluation run、再現性とcheckpoint lineageは対応するtrain
runが担うため、両方が必要です。

| セル | Train | PGD evaluation | AutoAttack evaluation |
|---|---|---|---|
| Chen / RSLAD | [`prod-chen-rslad-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-rslad-s0-2d54b82) | [`eval-7008…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-7008ebd671ee70b41990) | [`eval-7ebd…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-7ebd06a00e4f8f2d996d) |
| Chen / Entropy | [`prod-chen-entropy-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-entropy-s0-2d54b82) | [`eval-7b10…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-7b10222b14d776f001ca) | [`eval-d35b…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-d35bce64e04dbdbdd56e) |
| Chen / Student | [`prod-chen-student-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-student-s0-2d54b82) | [`eval-141e…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-141ef2fedeb6cd23c4fe) | [`eval-ae9e…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-ae9e62a7694d143f3fb9) |
| Chen / Joint | [`prod-chen-joint-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-joint-s0-2d54b82) | [`eval-1eefd…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-1eefd911db6c022c20f6) | [`eval-eafdd…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-eafdd255adb7eb585582) |
| Bartoldson / RSLAD | [`prod-bart-rslad-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-rslad-s0-2d54b82) | [`eval-78ff…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-78ffd40168b6d8a1d3c8) | [`eval-ff1c…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-ff1c4b50655af6d7997a) |
| Bartoldson / Entropy | [`prod-bart-entropy-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-entropy-s0-2d54b82) | [`eval-8ff4…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-8ff463c11843dfdd9b24) | [`eval-6dcf…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-6dcf1b78a77d3258b2e0) |
| Bartoldson / Student | [`prod-bart-student-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-student-s0-2d54b82) | [`eval-3839…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-3839fd906dfc2aa5b318) | [`eval-180f…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-180f437c57a483fdf8e7) |
| Bartoldson / Joint | [`prod-bart-joint-s0-2d54b82`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-joint-s0-2d54b82) | [`eval-0970…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-0970af3946e2f896e15d) | [`eval-f9f8…`](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/eval-f9f84e63660ac6825d93) |

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
過去のgroup-length修正とpilot受入経緯を追跡できます。campaignはscientific-review境界へ到達しましたが、
今回もrun削除は行っていません。通常viewから除外するのが安全です。容量または画面の簡潔さを優先する場合、
local manifestを保存した上で
まず`712b878`の3件を削除候補にできます。旧world-size 2 pilotの2件は固有の履歴なので、その次の候補です。

### 推奨するW&B view

2026-08-02に`configs/analysis/wandb_run_tags.yaml`のexact ID registryをdry-run後に適用しました。既存tag、
run ID、project、artifactは変更せず、再dry-runが全件`changed: false`になることを確認しています。

1. **Primary seed-0**: tag `canonical-seed0` + `primary-evidence`
2. **Official evaluation**: tag `official-evaluation`、attack別に`attack-pgd20` / `attack-autoattack`
3. **Research evidence**: tag `research-evidence`（logging-only、outcome-informed、H4 factorialを含む）
4. **Accepted pilots/reference**: tag `accepted-pilot`または`reference`
5. **通常非表示**: tag `superseded`

`canonical-seed0`は主要比較であり、複数seed確認済みという意味ではありません。削除やproject移動はartifact
lineageを壊すため行わず、W&B workspaceでは上記tagのsaved viewを作成します。新しい分類はregistryへexact IDを
追加してからdry-runし、未知runを名前パターンだけで自動分類しません。

## 7. 更新ルール

- live statusは`run-bundle/manifest.json`の`latest_progress`またはW&Bから生成し、本文へ転記しない。
- 正式結果は`evaluation-results.json`に10,000例が揃った時だけ記載する。
- W&B countと分類はAPIで再確認し、train/evaluationを重複扱いしない。
- seed追加や別execution profileは同じ表へ無条件に混ぜない。
- runを削除する前に、対応するlocal manifest、checkpoint、evaluation、artifact依存を確認する。
