---
id: 0053
status: pending
created: 2026-10-14
campaign: plan0103-phase2-convnext-atto-baseline-ema-s0-v1（plan 0103 Phase 2 の「基準＋EMA」。ConvNeXt-Atto・ランダム初期化・AdamW lr 1e-3・full AT 50 epoch。Phase 1 と同じ学習に、Phase 2 の共通設定（評価サブセット 5,000 枚、EMA 0.9999）だけを足したもの。Ferret の pull キューから hand-run、seed 0、50/50 epoch 完走、source SHA 9a16cd6、非決定的＋cudnn_benchmark）
question: 設定の確認は通り（Phase 1 との差 −0.10pt）、cosine はこの基準に対しても +0.98pt で「保留」のままです。ConvNeXt-Atto cosine の seed 1 を回すか、基準＋EMA の seed 1 も合わせて回すか、他モデルの cosine を待つか。
options:
  A: ConvNeXt-Atto cosine の seed 1 を1本だけ回す（約 25〜33 GPU時間）。比べる相手は既にある Phase 1 lr 1e-3 seed 1。d1 = cosine seed 1 − Phase 1 seed 1 が +0.5pt 以上かを見る。
  B: cosine seed 1 と基準＋EMA seed 1 の2本を回す（約 50〜66 GPU時間）。Phase 2 の中だけで、seed ごとに cosine − 基準を2組そろえる。
  C: いまは何も足さず、MobileNetV4-M と EfficientNet-B0 の cosine を待って、decision 0038 / 0052 と合わせて4モデルで判定する（追加 0 GPU時間）。
recommendation: A
chosen: null
---

# 0053: Phase 2 の基準＋EMA、ConvNeXt-Atto seed 0 が完了（設定の確認は合格、cosine は「保留」のまま）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-14 (postrun, Phase 2 baseline+EMA ConvNeXt-Atto)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `77748e0`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は run-bundle の manifest と `epoch-metrics.jsonl` から読みました）。

この run は、Phase 1 の ConvNeXt-Atto AdamW lr 1e-3 seed 0（PGD-10 37.02）と同じ学習です。
違いは、Phase 2 の共通設定（毎 epoch の評価を 5,000 枚のサブセットで行う、EMA を並行して保持する）と、W&B のグループ、source SHA、ホストだけです。
役目は3つです。Phase 2 の比較の基準、EMA の効果、設定の影響の確認。

条件: ImageNet-1k（元画像）、ConvNeXt-Atto（V1）、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px、world size 1、バッチ 128。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
AdamW（weight decay 0.05）、ピーク LR 1e-3、epoch 25 と 38 で 1/10、50 epoch。非決定的＋cudnn_benchmark（Phase 1 と同じ）。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| 基準＋EMA、最後（ep 49）、普通の重み | **59.69 / 36.92** |
| 基準＋EMA、最後（ep 49）、EMA の重み | 59.72 / 36.99 |
| 基準＋EMA、best EMA（ep 45） | 59.52 / 36.92 |
| Phase 1 multistep、seed 0、最後、普通の重み | — / 37.02 |
| Phase 2 cosine、seed 0、最後、普通の重み | 60.70 / 37.90 |

（best（サブセットで選んだ普通の重み）は最後の epoch と同じでした。）

- **設定の確認は合格です。** Phase 1 との差は PGD-10 で **−0.10pt**（25,620 枚のうち約 25 枚）。
  ルール（2026-10-09、decision 0052 B と同じ）は「差が seed の差 0.36pt 以内、または 0.5pt 以内なら影響なし」なので、「設定の影響なし」です。
  ただし MobileNetV4-S のような1ビット一致の確認ではありません。両方とも非決定的で、source とホストも違います。−0.10 は、それらを全部合わせた1組での差です。非決定性だけの揺れは、まだ別には測れていません。
- **cosine はこの基準に対しても「保留」です。** cosine − 基準＋EMA（最後・普通の重み）は **+1.01 clean / +0.98 PGD-10** です（約 251 枚、2本の差の SE 約 0.43pt の約 2.3 倍）。
  取り分けた画像の残り 20,620 枚でも +0.95 / +1.04、EMA の重みでも +0.94 / +0.90 で、同じ向きです。
  Phase 1 と比べたときの +0.88 から、比べる相手を替えても +0.10 しか動きません。
  decision 0052 B の事前ルール（`chosen` はまだ空）では、0.5pt 以上なので「保留」のままで、「A（seed 1）を、比べる相手を基準＋EMA に置き換えた形でもう一度検討する」になります。この packet がその検討です。
  弱点: cosine は Hamster・`5dd329e`、基準＋EMA は Ferret・`9a16cd6` です。plan 0106 によると、2つの SHA の差は `cuda_graph` を使うときだけ効くコードで、ConvNeXt の Phase 2 の設定はどれも使っていません。
- **cosine が勝つのは最後の数 epoch だけです。** サブセットの PGD-10 は、ウォームアップの終わり（epoch 9）では同じです（cosine −0.14）。
  multistep は最初の減衰の後に大きく先行し（epoch 30 で cosine が −5.46、epoch 40 で −1.80）、cosine は終わり近くで追い越します（epoch 47 で +1.44、epoch 49 で +0.70）。
- **EMA は最後には効いていません。** 最後の EMA − 普通の重みは +0.03 / +0.06 で「効果なし」（0.5pt 未満）。MobileNetV4-S（+0.10 / +0.10）、ConvNeXt-Atto cosine（−0.04 / −0.01）と同じです。
  途中では、EMA は epoch 8 で普通の重みを追い越し（cosine と同じ epoch）、学習率が高い間はサブセットで最大 +7.60pt（PGD-10）上回りました。
- 頑健性の過学習はありません（best = last）。崩壊の見張りは一度も作動していません（最大の落ち込み 0.58pt）。
- 実時間は 32.18 時間、487〜591 枚/秒でした（Ferret、混んでいる状態。Hamster の cosine は 713〜716 枚/秒）。

**ばらつきの目安（noise floor）**: ImageNet では seed 0 と seed 1 の差が MobileNetV4-S 0.36pt、MobileNetV4-M 0.09pt、ConvNeXt-Atto 0.36pt でした（各1組）。2本の全体評価の差の SE は約 0.43pt です。CIFAR の画面では seed だけで 1〜2pt 動くことがありました。
cosine の +0.98 は ConvNeXt-Atto の seed の差の約 2.7 倍ですが、CIFAR の目安の下の端に届くかどうかの大きさで、seed 1本では「効果あり」とは言えません。2本そろっても、言えるのは「その2本の seed での向き」までです。

**関係する未決定の packet**: 0052（ConvNeXt-Atto cosine、おすすめは B＝この基準＋EMA を待つ。その B はこの run で実行済みの状態になりました）、0038（MobileNetV4-S cosine、「保留」、おすすめは seed 1）。
この packet を決めるときは、0052 も一緒に閉じるのが自然です（例: 0052 は B、0053 で次の一手を選ぶ）。
どの選択肢も register B（2026-10-13）の「手法は学習不足を補う方向で組み立てる」と矛盾しません。

## 選択肢

### A: ConvNeXt-Atto cosine の seed 1 を1本だけ回す

- **GPU時間**: 約 25〜33 時間（4090 1枚）。根拠は実測で、Hamster の cosine seed 0 が 25.23 時間、Ferret の基準＋EMA が 32.18 時間（どちらも 50 epoch、同じモデル）。
- **やり方**: `imagenet_convnext_atto_pgd_at_phase2_cosine.yaml` を seed 1 で回します（source は seed 0 と同じ `5dd329e` の worktree）。
  比べる相手は、既にある Phase 1 の ConvNeXt-Atto lr 1e-3 seed 1（Ferret で実行済み）です。その clean / PGD-10 は、run の前に Ferret の bundle から読んで plan に書き写します。
  設定の確認が通ったので、Phase 1 の run を基準に使ってよい、という読み方です。
- **わかること**: seed 1 どうしの差 d1（cosine seed 1 − Phase 1 seed 1）。seed 0 の差（基準＋EMA に対して +0.98、Phase 1 に対して +0.88）と合わせて、2本の seed での向き。
- **事前ルール（run の前に固定します）**: 指標は、最後の epoch・普通の重み・取り分けた画像すべての PGD-10。
  - d1 が +0.5pt 以上 → ConvNeXt-Atto で cosine は「効果あり」（seed 0 と 1 での向き。母集団についての主張にはしない）。
  - d1 が 0.5pt 未満 → 「効果なし」。multistep のまま。
  - 2本の平均や標準偏差を「分布の推定」としては報告しません。
- **リスク**: seed 1 の基準は Phase 1 の run なので、Phase をまたぐ弱点が残ります。ただしその大きさは seed 0 の組で −0.10pt と測れています。d1 が +0.5 のすぐ近く（±0.1 程度）に出たときは、その分だけ読みにくくなります。

### B: cosine seed 1 と基準＋EMA seed 1 の2本を回す

- **GPU時間**: 約 50〜66 時間（4090 1枚ずつ、2枚なら同時に約 1.5 日）。
- **わかること**: Phase 2 の中だけで、seed 0 と seed 1 の両方の「cosine − 基準＋EMA」。Phase をまたぐ弱点がなくなります。ついでに、ConvNeXt-Atto の基準＋EMA の seed の差もわかります。
- **事前ルール**: 指標は A と同じ。seed 1 の差 d1' = cosine seed 1 − 基準＋EMA seed 1。
  - d0' = +0.98 と d1' がどちらも +0.5pt 以上 → 「効果あり」（この2本の seed での向き）。
  - d1' が 0.5pt 未満 → 「効果なし」。
- **リスク**: A の2倍の GPU 時間です。設定の確認で Phase をまたぐ差が −0.10pt と小さかったので、増える確かさは小さいと見込まれます。Hamster か Ferret の次の項目が 1〜2 日遅れます。

### C: いまは何も足さず、他の2モデルの cosine を待つ

- **GPU時間**: 追加 0。
- **わかること**: cosine の効果がモデルをまたいで同じ向きに出るか。
- **事前ルール**: decision 0038 / 0052 の C と同じ。4モデル（MobileNetV4-S、ConvNeXt-Atto、MobileNetV4-M、EfficientNet-B0）のうち3つ以上で、cosine − 基準（最後・普通の重み・PGD-10）が +0.5pt 以上なら「効果あり」、2つ以下なら「効果なし」。今のところ 2 / 2 です。
- **リスク**: 2026-10-08 のルールは「同じモデルの2本目の seed」なので、C はルールを変えることになり、0038 と一緒に決める必要があります。MobileNetV4-M と EfficientNet-B0 は遅いモデル（Phase 1 で約 35〜49 時間）で、それぞれの基準＋EMA の後ろにあるので、結果は数日先です。

## おすすめ: A

0052 で B をすすめた理由は、「Phase 1 と Phase 2 をまたいでいる」不確かさを先に測るためでした。それが −0.10pt と小さいとわかったので、Phase 1 の seed 1 を seed 1 の基準として使えます。
すると、2026-10-08 のルール（同じモデルの2本目の seed）に一番安く答えられるのは A です。B は A の2倍の時間で、増える確かさは −0.10pt 分だけです。
cosine の +0.98 は、どちらの基準でも「保留」の帯の真ん中にあり、seed 1 を1本足せば向きが決まる位置にあります。

**判断が変わる場合**:
- Phase 1 の seed 1 の bundle が読めない、または壊れている → B（基準＋EMA seed 1 も回す）に切り替えます。
- 0038 で C（モデルをまたいで判定）が選ばれる → この packet も C にそろえるのが筋です。
- MobileNetV4-S の cosine seed 1（0038 A）も回すなら、2モデル分の seed 1 を同時に回すと、cosine の結論が一度に出ます（ConvNeXt は約 25〜33 時間、MobileNetV4-S は約 16 時間）。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習は始めません。
