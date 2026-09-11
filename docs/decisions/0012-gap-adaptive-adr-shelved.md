---
id: 0012
status: decided
created: 2026-09-11
campaign: null
question: plan 0098（Gap-Adaptive ADR）の実装・レビュー・机上検証の結果、λの
  severity正規化に構造的な欠陥が見つかった。追加コストなしの修正案・追加コスト
  ありの修正案のいずれも根本解決に至らないことを確認した上で、このままGPU
  キャンペーン（30 GPU時間見積り）に進むか、保留するか。
options:
  A: 無料の修正（gap-adaptive追跡をscheduler.milestones[0]、この設定では
    epoch 100から開始）を採用し、「LR decayを境にしたstep関数」であることを
    plan本文に明記した上でGPUキャンペーンに進む。0 GPU時間の追加コスト。
  B: P1-2（train/valの測定不一致・rectified targetを介したλとの循環参照）も
    修正してからGPUキャンペーンに進む。追加の評価パス（epoch境界でtrain split
    のサブサンプルに対する20-step PGD攻撃、val相当のコスト増）が必要。
  C: Gap-Adaptive ADRをこのまま保留し、GPUキャンペーンを走らせない。plan 0098は
    「実装・テストは完了、メカニズムは再設計が必要」の状態で凍結し、
    BACKLOG.mdの他項目に進む。0 GPU時間。
recommendation: C
chosen: C
---

## 決定（2026-09-11、本人）

**C を選択。** Gap-Adaptive ADRの検証はここで一旦保留し、BACKLOG.mdの次の項目に進む。
plan 0098は実装・テスト完了のまま凍結し、GPUキャンペーンは走らせない。

## Evidence

GPU未使用。すべてplan 0097で既に記録済みの実データ（`runs/adr-cifar10-campaign-v1/
cifar10_r18_adr-s{0,1,2}`、`cifar10_mobilenetv2_adr-s{0,1,2}`の
`epoch-metrics.parquet`、200エポック×6本）に対する机上のオフライン再計算。
新規GPU時間はゼロ。

**scientific-reviewerが指摘した2つの設計上の懸念（P1-1, P1-2）**:
- P1-1: `severity_e = clip(gap_ema_e / max(ε, running_max_e), 0, 1)`の
  `running_max_e`が当該エポック自身のgap_emaを含んで更新されるため、
  robust overfitting（RO）が単調悪化する通常の訓練では、gapが一度でも
  過去最大を更新した時点でseverity=1に張り付く。
- P1-2: `raw_gap_e = train_robust_accuracy_e − val_pgd_accuracy_e`の2項は
  異なる脅威モデル（10-step KL攻撃 vs 20-step CE攻撃）・異なるBNモード
  （train-mode vs eval-mode）で測られており、純粋な汎化ギャップではない。
  さらに`train_robust_accuracy`はrectified target（λ依存）を狙った攻撃の
  結果なので、gapがλ自身に間接的に依存する循環がある。

**実データでの確認（β=0.9, λ_low=0.7, λ_high=0.95）**:

| arm | 元の設計での天井到達epoch | 元の設計での天井滞在率 |
|---|---:|---:|
| R18 s0/s1/s2 | 15〜17 / 200 | 68〜75% |
| MobileNetV2 s0/s1/s2 | 21〜25 / 200 | 43〜46% |

エポック単位の実測値（R18 s0）から、epoch 0〜99は小さくほぼ一定のgap
（-0.03→0.06程度、横ばい）、epoch 100（1回目のLR decay）を境に急激に
悪化し199エポック目に0.37まで単調に伸びる、という2つの局面が明確に
確認できた——古典的な「LR decay直後に急にROが起こる」描像と整合する。
天井への張り付きは、この本物のRO発生（epoch 100以降）ではなく、
epoch 15〜25の**小さな序盤の立ち上がり**で起きていた。

**Option Aの検証（epoch 100から追跡を再開）**: 誤発火のタイミングは
epoch 100前後に是正されるが、系列を再開した最初の値は定義上「これまでの
最悪」になるため、R18は再開直後の1エポックで、MobileNetV2も9エポック
以内に天井へ張り付き、以降はepoch 100〜199の82〜100%を天井で過ごす。
実質「epoch 100を境にした2値のstep関数」に近い。

**Option Aが解決しないもの**: 本物のRO（epoch 100以降）自体もほぼ単調に
悪化する現象であるため、「自分の過去最悪との比率」という正規化方式は、
P1-2を直しても直さなくても、本物のROに対して同じ理由で早期飽和する。
P1-2の修正は「いつ誤発火するか」を是正するだけで、「発火したら天井に
張り付いたままになる」というP1-1の本体は解決しない。

## Option A — 無料修正（epoch 100から開始）してGPUキャンペーンへ

**GPU時間**: 0（追加コストなし）。
**確立すること**: 「早期の測定不一致に反応しない、LR decayを境にした
2値スケジュール」がcosineより良いか。
**プリレジ判定規則**: plan 0098のIMPROVED/NULL/WORSE基準をそのまま使える。
**リスク**: 実質的にstep関数であり、plan 0098が意図した「継続的な適応性」
の検証にはならない。何を測っているかを正直に書き直す必要がある。

## Option B — P1-2も修正してGPUキャンペーンへ

**GPU時間**: 追加の評価パス分（val相当、既存の評価コストが増える程度）。
**確立すること**: 測定を揃えた「きれいな」gapでも、Option Aと同じ天井
張り付きが起きるかどうか。
**リスク**: 上記の通り、天井張り付き自体はP1-2と独立の問題であるため、
コストをかけても中心的な懸念が解決しない可能性が高い。

## Option C — 保留（採用）

**GPU時間**: 0。
**確立すること**: なし。plan 0098は「実装・テストは完了、severityの
正規化は再設計が必要」という状態で凍結され、将来別のアプローチ
（例: 絶対的なgapの大きさに感応する写像、RO発生タイミングに依存しない
定式化）が見つかった時に、`gap_adaptive.py`のスケジュール原始関数・
`AdrConfig.lambda_source`・Trainerの配線・checkpoint往復・テスト一式
はそのまま再利用できる。

## 推奨理由

Option A・Bのどちらも、GPU時間の多寡に関わらず「plan 0098が本来
検証したかったもの（継続的な適応性）」を検証できない。Option Aは
無料だが実質的にstep関数の検証にすり替わり、Option Bはコストを
払っても天井張り付き自体は解決しない。30 GPU時間（または0でも）を
使う前に、severityの正規化そのものをもう一段設計し直す必要があり、
それは今回のセッションの範囲を超える。安く検証できるという当初の
利点（decision packet 0011のOption B、CIFAR-scaleでの事前検証）は
最大限活かされた——実際にGPUを一切使わずに、この設計の欠陥を発見できた。
