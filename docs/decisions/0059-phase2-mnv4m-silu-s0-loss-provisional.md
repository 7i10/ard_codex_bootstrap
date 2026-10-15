---
id: 0059
status: pending
created: 2026-10-15
campaign: plan0103-phase2-mnv4m-silu-s0-v1（plan 0103 Phase 2。MobileNetV4-M の ReLU をすべて SiLU に替えたもの。ランダム初期化、SGD lr 0.025、full AT 50 epoch、Phase 2 の共通設定（サブセット 5,000、EMA 0.9999）。Ferret で hand-run、seed 0、50/50 epoch 完走、非決定的＋cudnn_benchmark、source cabd60a）
question: MobileNetV4-S では SiLU で PGD-10 が +2.56pt 上がりましたが、MobileNetV4-M では Phase 1 の ReLU より clean −2.44 / PGD-10 −0.96 と下がりました。正式な比較相手（MobileNetV4-M 基準＋EMA）の完了を待って読み直すだけにするか、seed 1 を回すか、plan 0107 の採用条件に「MobileNetV4-M でも下がらないこと」を足すか。
options:
  A: 追加の GPU は使わない（0 GPU時間）。実行中の MobileNetV4-M 基準＋EMA（あと約 1 日）が終わったら、事前ルールで判定し直して記録する。
  B: SiLU MobileNetV4-M の seed 1 を回す（約 36〜51 GPU時間）。Phase 1 ReLU seed 1（63.71 / 39.30）と比べ、d1 ≤ −0.5pt なら「2本とも下がる」として MobileNetV4-M の SiLU を閉じる。
  C: いまは GPU を使わず、plan 0107 の採用条件を変える。MobileNetV4-S で「効果あり」になった活性化の案は、蒸留のベースにする前に MobileNetV4-M でも1本回し（1案あたり約 36〜51 GPU時間、後で）、基準＋EMA より 0.5pt 以上下がらないことを条件にする。
recommendation: A
chosen: null
---

# 0059: SiLU、MobileNetV4-M seed 0 が完了（Phase 1 の ReLU より clean −2.44 / PGD-10 −0.96、暫定）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-15 (postrun, Phase 2 SiLU, MobileNetV4-M)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `ec31b68`（`Record plan0103-phase2-mnv4m-silu-s0-v1 results`）。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は run-bundle の manifest と `epoch-metrics.jsonl` から読みました）。

この run の設定は MobileNetV4-M の基準＋EMA と同じで、違いはモデルの ReLU をすべて SiLU に替えたことだけです（パラメータ数と計算量は同じ）。
ただし、その基準＋EMA（`plan0103-phase2-mnv4m-baseline-ema-s0-v1`）はまだ実行中です（50 epoch のうち 23 epoch）。
そこで今回は、同じ学習手順の Phase 1 の ReLU の2本（seed 0 と seed 1）と比べました。
Phase 2 の共通設定（サブセット評価と EMA）は評価を足すだけなので、普通の重みの学習は Phase 1 と同じ手順です（ビット単位では確かめていません）。

条件: ImageNet-1k（元画像）、MobileNetV4-Conv-M（SiLU）、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、weight decay 1e-4）、ピーク LR 0.025、multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）、50 epoch、world size 1、バッチ 128。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| SiLU、最後（ep 49）、普通の重み（= best） | **61.26 / 38.33** |
| SiLU、最後（ep 49）、EMA の重み（= best EMA） | 61.37 / 38.69 |
| Phase 1 ReLU seed 1（Hamster、`609e6fa`）、最後 | 63.71 / 39.30 |
| Phase 1 ReLU seed 0（Ferret、`086072b`）、最後 | 未読 / 39.39 |

- **Phase 1 seed 1 との差: clean −2.44 / PGD-10 −0.96**（25,620 枚のうち 626 枚と 247 枚）。seed 0 との差は PGD-10 −1.06 です。
  2本の差の SE（clean 約 0.43pt、PGD-10 約 0.37pt）の約 5.7 倍と 2.6 倍です。PGD-10 の差は、MobileNetV4-M の seed の差（0.09pt）の約 10 倍です。
- **Phase 2 のルール**（1.5pt 以上「効果あり」、0.5〜1.5pt「保留」、0.5pt 未満「効果なし」）では、PGD-10 の差は下がる向きの「保留」の帯に入ります。clean は 1.5pt を超えて下がっています。
  ルール上の比較相手は基準＋EMA なので、**この判定は暫定**です。
- **MobileNetV4-S とは逆の向きです。** MobileNetV4-S の SiLU は +1.60 / +2.56 でした（decision 0044、「効果あり」）。
- **学習データへの当てはまりが悪くなっています。** SiLU は最初の 8 epoch（LR 0.02 まで）は学習の損失が低く、速く学びます。
  epoch 8（LR 0.0225）で逆転し、その後は差が広がり続けます。学習の損失の差は epoch 24 で +0.127、epoch 49 で +0.217、学習バッチの頑健精度は epoch 49 で −2.92pt です。LR を下げても差は縮みません。
  MobileNetV4-S では逆に、SiLU のほうが学習の損失が低く、学習バッチの頑健精度も +1.63pt 高くなっていました。
- 学習に使った画像（2,000 枚）での精度も −4.55 / −3.50 で、見ていない画像との差は縮みました（+7.54 / +8.20 → +5.44 / +5.67）。これは当てはまりが悪くなった結果です。
- EMA は最後には効いていません（+0.11 / +0.36）。ウォームアップ中の EMA の遅れは MobileNetV4-S と同じです。頑健さの過学習はありません（best = 最後）。学習崩壊の見張りは一度も発動していません。
- 実時間 51.38 時間（Ferret。370〜380 枚/秒で、Ferret の GPU2 の実測値とほぼ同じ）。Phase 1 seed 1 は Hamster で 35.39 時間でした。ホストの違いが大きく、SiLU の計算の遅さは切り分けられません。メモリは約 2.4 GB 多く使いました。

**ばらつきの目安（noise floor）**: ImageNet では seed 0 と seed 1 の差が MobileNetV4-S 0.36pt、MobileNetV4-M 0.09pt、ConvNeXt-Atto 0.36pt でした（各1組）。CIFAR の画面では seed だけで 1〜2pt 動くことがありました。
PGD-10 の −0.96 は ImageNet の seed の差よりはっきり大きいですが、CIFAR の目安（1〜2pt）よりは小さい値です。clean の −2.44 は CIFAR の目安も超えています。
今回の比較は seed（0 と 1）・source・ホストもまたいでいます。言えるのは「この seed では、MobileNetV4-S の SiLU の上がり幅は MobileNetV4-M に移らず、clean は下がった」までです。

**関係する決定**: decision 0044（MobileNetV4-S の SiLU を標準にするか）は `chosen` がまだ null です。
plan 0107 は「SiLU 並みの上がり幅が出た活性化の案を蒸留のベースにする」としていますが、その案はすべて MobileNetV4-S で試すもので、Ferret の pull キューの先頭にあります（`8f51192`）。
この run が終わったので、Ferret の GPU が1枚空き、plan 0107 の run が始まる見込みです。

## 選択肢

### A: 追加の GPU は使わず、基準＋EMA の完了を待って判定し直す（おすすめ）

- **GPU時間**: 0（基準＋EMA はすでに実行中。今のペース（1 epoch 約 44 分）で、残り 27 epoch は約 20 時間）。
- **わかること**: 正式な比較相手との差。同時に、基準＋EMA と Phase 1（39.39 / 39.30）の差から、Phase 2 の共通設定の影響もわかります。
- **事前ルール**: Δ = SiLU − 基準＋EMA（どちらも最後の普通の重み、取り分けた画像すべての PGD-10）。
  Δ ≤ −0.5pt → 「MobileNetV4-M では SiLU は下がる（seed 1本）」と記録し、MobileNetV4-M の SiLU は採用しない。
  −0.5 < Δ < +0.5 → 「効果なし」と記録し、採用しない。
  Δ ≥ +0.5pt → Phase 1 との比較と向きが合わないので「判定できない」と記録し、続けるかは改めて packet で決める。
  clean の差も同じ表に並べて記録します。
- **リスク**: seed は1本のままです。基準＋EMA は別の source（`d97bf6e`）で、ホストも違う可能性があります。

### B: SiLU MobileNetV4-M の seed 1 を回す

- **GPU時間**: 約 36〜51 時間（Phase 1 の ReLU seed 1 が Hamster で 35.4 時間、今回の SiLU が Ferret の遅い GPU で 51.4 時間だったことからの見積もり。SiLU 自体の遅さは測れていないので幅を持たせています）。Hamster の GPU は 2 枚とも数日かかる run で埋まっています。
- **やり方**: `imagenet_mobilenetv4_conv_medium_silu_pgd_at_phase2_random_lr0025.yaml` を seed 1 で、seed 0 と同じ `cabd60a` の worktree から回します。比べる相手は Phase 1 ReLU seed 1（63.71 / 39.30）。Hamster で回せば、ホストもそろいます。
- **わかること**: MobileNetV4-M で SiLU が下がることが、seed 1 でも起きるか。
- **事前ルール**: d1 = SiLU seed 1 − Phase 1 seed 1（最後の普通の重み、取り分けた画像すべての PGD-10）。
  d1 ≤ −0.5pt → 「2本とも下がる」として MobileNetV4-M の SiLU を閉じる。
  −0.5 < d1 < +0.5 → 「seed 0 では下がり、seed 1 では効果なし」と記録し、閉じる（採用しない）。
  d1 ≥ +0.5pt → seed 0 と向きが逆なので「判定できない」と記録し、改めて packet で決める。2本の平均やばらつきは出しません。
- **リスク**: どの結果でも「採用しない」に近い結論で、36〜51 時間をその補強に使うことになります。plan 0107 の run と GPU を取り合います。

### C: plan 0107 の採用条件に「MobileNetV4-M でも下がらないこと」を足す（いまは GPU を使わない）

- **GPU時間**: いまは 0。後で、MobileNetV4-S で「効果あり」になった案 1 つにつき約 36〜51 時間。その案の MobileNetV4-M 版のモデルと config を作る手間もかかります。
- **わかること**: MobileNetV4-S で効いた活性化の変更が、大きさの違うモデルにも効くか。蒸留のベースを MobileNetV4-S に限らない結論にできるか。
- **事前ルール**: plan 0107 の案が MobileNetV4-S で「効果あり」（基準＋EMA より PGD-10 +1.5pt 以上）になったら、蒸留のベースにする前に、同じ案を MobileNetV4-M で1本回します。
  MobileNetV4-M の基準＋EMA との差が −0.5pt より大きい → ベースにする。
  −0.5pt 以下 → MobileNetV4-S 限定の効果として記録し、ベースにするかは改めて packet で決める。
- **リスク**: 蒸留の検証の開始が 2 日ほど遅れます。plan 0107 の「SiLU 並みの上がり幅が出たらベースにする」という承認済みの条件を変えるので、plan 0107 への追記が要ります。

## おすすめ: A

基準＋EMA はすでに実行中で、約 1 日で終わります。追加の GPU なしで、正式なルールの判定ができます。
MobileNetV4-S と MobileNetV4-M で向きが逆という結果は、seed 1本で 2.6 SE（PGD-10）、5.7 SE（clean）です。B で 36〜51 時間を使っても、結論は「MobileNetV4-M では SiLU を採用しない」からほとんど動きません。GPU は蒸留の検証と plan 0107 の run に残すのがよいと考えます。

**判断が変わる場合**:
- 論文で「SiLU はモデルの大きさによって効き方が逆になる」と主張したい場合 → B が要ります（seed 1本の逆転では主張できません）。
- 蒸留のベースを MobileNetV4-S 以外にも広げる予定がある場合 → C を A と一緒に選ぶ価値が上がります。今回の結果は、MobileNetV4-S で効いた活性化の変更が、そのまま大きいモデルにも効くとは限らないことを示しています。
- A の判定で Δ ≥ +0.5pt（Phase 1 との比較と向きが逆）になった場合 → Phase 2 の共通設定か source の影響を疑い、改めて packet を出します。

`chosen` は人間が記入します（ファイルを編集するか、チャットで答えてください）。記入されるまで、この packet に関わる新しい学習は始めません。
