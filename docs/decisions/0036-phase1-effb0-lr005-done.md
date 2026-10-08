---
id: 0036
status: pending
created: 2026-10-08
campaign: plan0103-phase1-efficientnet-b0-random-lr005-v1（plan 0103 Phase 1 の LR グリッド。EfficientNet-B0・ランダム初期化・SGD lr 0.05・full AT 50 epoch を Hamster GPU0 で実行。hand-run、seed 0、50/50 epoch 完走、source SHA 609e6fa）
question: EfficientNet-B0 はグリッド上端の lr 0.05 が lr 0.025 と同点（35.88 vs 35.93、差 -0.05pt）でした。「上端が最良なら延長」のルールは厳密には発火しません。用意済みの lr 0.1 を回すか、3点（0.0125 の完了待ち）で LR を確定するか。
options:
  A: 延長しない。lr 0.0125 の完了（あと約3時間）を待ち、3点の argmax（最後の epoch の内部検証 PGD-10）を採用する（追加 GPU 0時間）。
  B: 同点を「上端が最良」とみなして lr 0.1 を1セル足し、4点の argmax を採用する。それ以上は足さない（追加 約 47〜50 GPU時間、Hamster の 4090 1枚）。
recommendation: A
chosen: null
---

# 0036: Phase 1 LR グリッド、EfficientNet-B0 SGD lr 0.05 が完了

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-08 (postrun, EfficientNet-B0 lr 0.05)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `6b99bec`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません。

条件は lr 0.025 のセルと LR 以外すべて同じです（config の差は LR と W&B グループだけ）。
ImageNet-1k（元画像）、EfficientNet-B0、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、momentum 0.9、wd 1e-4）、10 epoch ウォームアップ、epoch 25 と 38 で 1/10、50 epoch。
world size 1、バッチ 128、非決定的モード、torch.compile あり。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack は走っていません。

| LR | 状態 | 最後の epoch の clean / PGD-10 |
|---|---|---:|
| 0.0125 | 未完了（epoch 47/50、Hamster GPU1） | — |
| 0.025 | 完了（Ferret、source `086072b`） | 未確認 / 35.93 |
| 0.05 | 完了（この run） | 57.86 / 35.88 |
| 0.1 | config のみ（コミット `5b6b443`）、未起動 | — |

- **0.05 と 0.025 は同点です。** 差は -0.05pt で、2 run の差の標準誤差（約 0.42pt）の約 0.1 倍です。
  0.025 の clean は plan に記録が無く、Ferret にある run を今回は読み直していません。
- best（ep 48）は 57.79 / 36.00 で、last と区別できません（差 0.11pt、検証の標準誤差は約 0.30pt）。
- 学習は正常です。PGD-10 が直前の最大値から落ちた幅は最大 0.69pt（epoch 18）。catastrophic-overfitting の判定は一度も出ていません。
- 最後の区間（lr 0.0005）でも PGD-10 はわずかに上がり続けています（epoch 38 の 35.13 → epoch 39〜49 は 35.39〜36.00）。
- 時間: 48.62 時間（学習は約 46.0 時間）、373〜379 img/s。

**ほかの BN CNN の傾向（同じ SGD レシピ、最後の epoch の PGD-10）:**
- MobileNetV4-M: 0.0125 / 0.025 / 0.05 = 37.00 / 39.39 / 39.10。0.025 が最良で、0.05 が僅差。
- MobileNetV4-S: 0.0125 / 0.025 / 0.1 = 29.16 / 30.71 / 29.18。0.1 は 0.025 より 1.5pt 低い。
- EfficientNet-B0 の 0.025 ≈ 0.05 は、この2モデルと同じ形です。

**延長ルールとの関係:** lr 0.1 の config には人間の判断「上端が最良なら延長する」（2026-10-08）が書かれています。
最後の epoch の PGD-10 で比べると、上端 0.05 は 0.025 に 0.05pt 届かず、argmax ではありません。
ただし差はノイズよりずっと小さく、「同点」とも読めます。どちらに読むかがこの packet の問いです。

## ノイズの目安

- 検証の標準誤差は 1 run あたり PGD-10 で約 0.30pt、2 run の差では約 0.42pt です。
- ImageNet での seed 間の揺れはまだ測っていません。CIFAR 時代の経験では乱数だけで 1〜2pt 動きました。
- 今回の 0.05pt は、どちらの尺度でも読めない差です。「0.025 の方が良い」とも「0.05 の方が良い」とも言いません。
- B で 0.1 が 0.025 / 0.05 を明確に上回る（1.0pt 以上）ことは、上の傾向からは起きにくいと見ます。
  起きなければ、B で分かるのは「最適が 0.1 より上にはない」ことだけです。

## 費用の前提

- この run は Hamster の 4090 1枚で 48.62 時間でした。LR を変えても速度は変わらないので、0.1 も 47〜50 時間と見ます。
- Hamster GPU0 はいま空いています。GPU1 は 0.0125 があと約3時間で終わります。
- Phase 2 batch C（MobileNetV4-S、augmentation × epoch の 2×2）は config ができていて、未起動です。B はその GPU を約2日使います。

## 選択肢

### A: 延長しない（推奨）

- **GPU時間**: 追加 0。
- **分かること**: 3点（0.0125 / 0.025 / 0.05）の中での EfficientNet-B0 の採用 LR。
- **事前に決めるルール**（0.0125 の結果を見る前に、この packet で記録）:
  - 0.0125 の完了後、最後の epoch の内部検証 PGD-10 が最大の LR を採用する。判定は PGD-10 だけ。
  - 0.025 と 0.05 の差（現在 0.05pt）が 0.42pt 未満のまま、0.025 が argmax なら、論文と best-vs-best の表に
    「0.05 とは区別できない」と書く。
  - 0.0125 が argmax になった場合は、下端の延長をどうするか別の packet で聞く（この packet では決めない）。
- **リスク**: 最適が 0.05 と 0.1 の間にあった場合、EfficientNet-B0 だけ少し低い数字で比べることになるかもしれません。
  ほかの BN CNN の傾向からは、その差は小さいと見ます。

### B: lr 0.1 を1セル足す

- **GPU時間**: 追加 約 47〜50 時間（Hamster の 4090 1枚。0.0125 の結果を待たずに GPU0 で始められる）。
- **分かること**: 上端の外側で PGD-10 が下がるか。つまり最適が 0.1 より上にないこと。
- **事前に決めるルール**（0.1 の結果を見る前に、この packet で記録）:
  - lr 0.1 を1セル足す（`imagenet_efficientnet_b0_pgd_at_phase1_random_lr01.yaml`、ほかは 0.05 のセルと同じ、seed 0）。
  - 4点（0.0125 / 0.025 / 0.05 / 0.1）の argmax（最後の epoch の内部検証 PGD-10）を採用する。
  - 0.1 が勝っても、それ以上は足さない。0.1 が argmax なら「最適はグリッドの上にあるかもしれない」と論文に書く。
- **リスク**: EfficientNet-B0 だけ予算が1セル多くなります。ただし ConvNeXt-Atto と DeiT-Tiny も延長しているので、
  「上端付近が最良なら1回延長」という扱いはそろいます。
  Phase 2 の開始が Hamster 1枚分、約2日遅れます。

## 推奨

A を推奨します。延長ルールは「上端が最良なら」で、今回の上端は argmax ではなく、同点です。
MobileNetV4-M（0.025 > 0.05 僅差）と MobileNetV4-S（0.1 が 1.5pt 低い）を見ると、同じ SGD レシピの BN CNN は
0.025〜0.05 が頂上で、0.1 は下り坂側にあると見るのが自然です。B に約2日を使っても、分かるのはおそらく
「0.1 は低い」ことだけで、採用 LR は変わらない見込みが高いです。その2日は Phase 2 に回せます。
次のどれかなら B に変えます。(1)「上端が最良」をノイズ以内の同点も含むと読みたいとき（ConvNeXt-Atto・DeiT-Tiny と
扱いを完全にそろえたい場合）。(2) EfficientNet-B0 が Phase 1 の結論（BN CNN の順位、EasyRobust の 35% AutoAttack との
比較）を左右し、低めの数字で比べるのを避けたいとき。(3) 0.0125 の完了後も Hamster GPU0 に Phase 2 の予定が入らないとき。
