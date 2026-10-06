---
id: 0034
status: pending
created: 2026-10-06
campaign: plan0103-phase1-deit-tiny-adamw-random-lr5em4-v1（plan 0103 Phase 1 の LR グリッド。DeiT-Tiny・ランダム初期化・AdamW lr 5e-4・full AT 50 epoch を Hamster GPU0 で実行。hand-run、seed 0、50/50 epoch 完走、source SHA 609e6fa）
question: DeiT-Tiny の最初のセル（グリッドの上端 lr 5e-4）が 33.27% で終わりました。承認済みの3点グリッドをそのまま回して argmax を取るだけにするか、それとも「上端が明確に勝ったら 1 点だけ上に足す」ルールを、残りのセルの結果が出る前に事前登録しておくか。
options:
  A: 何も足さない。承認済みのグリッド（DeiT-Tiny・ConvNeXt-Atto とも AdamW 1.25e-4 / 2.5e-4 / 5e-4）をそのまま回し、最後の epoch の内部検証 PGD-10 の argmax を採用する（追加 GPU 0時間）。
  B: 上端ルールをいま事前登録する。3点がそろった時点で 5e-4 が argmax で、かつ 2.5e-4 より 1.0pt 以上高ければ、そのモデルに lr 1e-3 を1セル足し、4点の argmax を採用する（追加 0〜約 65 GPU時間、DeiT-Tiny と ConvNeXt-Atto で最大1セルずつ）。
recommendation: B
chosen: null
---

# 0034: Phase 1 LR グリッド、DeiT-Tiny AdamW lr 5e-4 が完了

ほかの packet（0019〜0033）はどれも `chosen` が空のままですが、この packet はそれらと別の問い（DeiT-Tiny と ConvNeXt-Atto の LR グリッドの上端）を扱うので、どれも置き換えません。

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-06 (postrun, DeiT-Tiny lr 5e-4)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `99635cf`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません。

条件: ImageNet-1k（元画像）、DeiT-Tiny、ランダム初期化、plain PGD-AT、教師なし、seed 0。224 px で学習。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
AdamW（betas 0.9/0.999、wd 0.05）、ピーク lr 5e-4、10 epoch のウォームアップのあと epoch 25 と 38 で lr を 1/10、50 epoch。
world size 1、バッチ 128、非決定的モード（cudnn_benchmark）、compile なし、fp32。
数字はすべて内部検証（学習データから取り分けた 25,620 枚、224 px）で、公式テストではありません。AutoAttack は走っていません。

- **最後の epoch（ep 49）は clean 56.85% / PGD-10 33.27%**、best（ep 45）は 56.71 / 33.35 です。
  差は検証の標準誤差（PGD-10 で約 0.29pt）より小さく、best と last は区別できません。
- **AdamW で DeiT-Tiny は普通に学習できています。** PGD-10 が直前の最大値から落ちた幅は最大 0.16pt で、
  SGD の固定レシピのとき（clean が約 30% で止まる失敗）は起きていません。ただし SGD の DeiT-Tiny run は
  epoch 38 で止めたので結果ではなく、SGD と AdamW の差を数字では言えません。
- lr を下げたときの上がり幅: epoch 24→25 で clean +8.33 / PGD-10 +6.26、epoch 37→38 で +1.04 / +0.88。最後の区間はほぼ平らです（32.89〜33.35%）。
- 参考（まだ best-vs-best ではありません）: MobileNetV4-S 30.71、MobileNetV4-M 39.39、EfficientNet-B0 35.93（いずれも SGD lr 0.025）。
- 時間: 33.00 時間（学習は約 31.2 時間）、557〜559 img/s。隣の GPU1 では lr 2.5e-4 のセルがずっと走っていました。

**グリッドの状況（Hamster のキューのログから）:**
- GPU0: この run の終了直後（2026-10-06 12:51 UTC）に EfficientNet-B0 lr 0.05 が自動で始まりました。
- GPU1: DeiT-Tiny lr 2.5e-4 が epoch 47/50。終わると EfficientNet-B0 lr 0.0125 が始まります。
- DeiT-Tiny lr 1.25e-4 と ConvNeXt-Atto の3セルは、Hamster のキューには入っていません。
  Ferret で走っている／待っているかは、この postrun では確認していません。入っていなければ、承認済みグリッドを完走させるために起動が必要です（どちらの選択肢でも同じ）。

## ノイズの目安

- 検証の標準誤差は 1 run あたり PGD-10 で約 0.29pt、2 run の差では約 0.41pt です。
- ImageNet での seed 間の揺れはまだ測っていません。CIFAR 時代の経験では乱数だけで 1〜2pt 動くことがありました。
- 1 seed のグリッドなので、隣り合う LR の差が 1pt 未満なら「どちらが良いか」は読めません。argmax はその場合でも
  「採用する LR を決める」ためだけに使い、「その LR の方が良い」という主張にはしません。
- B のしきい値 1.0pt は、上の揺れの下端に合わせています。1pt 未満の差では上に足しても同じ理由で読めないので、足しません。

## 費用の前提

- DeiT-Tiny の1セル: この run で 33.0 時間（Hamster の 4090、隣の GPU も使用中）。
- ConvNeXt-Atto の1セル: SGD の固定レシピ run が 25.4 時間（718〜723 img/s）。AdamW でも速度はほぼ同じと見て 25〜30 時間。
- lr 1e-3 を足すのは、上端ルールが発火したモデルだけです。0 セル（0 時間）〜2 セル（約 58〜65 時間）。

## 選択肢

### A: 何も足さない

- **GPU時間**: 追加 0。
- **分かること**: 承認済みの3点グリッドの中での DeiT-Tiny と ConvNeXt-Atto の LR。
- **事前に決めたルール**（plan 0103、2026-10-05）: 最後の epoch の内部検証 PGD-10 が最大の LR を採用し、その LR だけ公式 val で評価する（clean・PGD・別プロセスの AutoAttack）。
- **リスク**: 5e-4（上端）が argmax になると、本当の最適はグリッドの外にあるかもしれず、
  そのモデルだけ実力より低い数字で best-vs-best に入ります。ImageNet-100 では MobileViT-S（random）と
  DeiT-Tiny（pretrained）の argmax がどちらも上端でした（ただし A2 の判断で、ImageNet-100 は ImageNet-1k の LR 選びには使いません）。

### B: 上端ルールをいま事前登録する（推奨）

- **GPU時間**: 追加 0〜約 65 時間。発火したモデルだけ1セル。
- **分かること**: 上端が勝ったときに、最適がグリッドのすぐ外にあるかどうか。
- **事前に決めるルール**（DeiT-Tiny と ConvNeXt-Atto に別々に適用。lr 2.5e-4 のセルの結果を見る前にこの packet で記録）:
  - 3点がそろった時点で、最後の epoch の内部検証 PGD-10 で 5e-4 が argmax、かつ 5e-4 − 2.5e-4 ≥ 1.0pt なら、
    lr 1e-3 のセルを1つ足す（ほかの設定は 5e-4 のセルと同じ、seed 0）。
  - 4点の argmax を採用する。1e-3 が勝っても、それ以上は足さない（2点目の延長はしない）。
  - 5e-4 が argmax でも差が 1.0pt 未満なら足さず、5e-4 を採用する。
  - 判定に使うのは PGD-10 だけ。clean は報告するが判定には使わない。
- **前例**: MobileNetV4-S は SGD のグリッドに 4点目（lr 0.1）があります。なので「モデルごとに3点ちょうど」は今でも厳密ではありません。
- **リスク**: 足したモデルだけ予算が1セル多くなり、「同じ予算で best-vs-best」が少し崩れます（論文ではそのまま書く）。
  lr 1e-3 は、DeiT の公式レシピ（バッチ 512 で 5e-4）をバッチ 128 に線形で直した値（1.25e-4）の 8 倍なので、
  学習が不安定になるかもしれません。その場合も、その結果をそのまま argmax の比較に入れます。
  この packet を読むころに lr 2.5e-4 のセルが終わっていたら、ルールは「その結果を見た後」になります。
  その場合は ConvNeXt-Atto の分だけが事前登録で、DeiT-Tiny の分は事後と明記します。

## 推奨

B を推奨します。ルールを書くだけなら費用は 0 で、上端が 1pt 以上の差で勝ったときだけ GPU を使います。
1pt 以上の差でまだ上り坂なら、最適が外にある可能性は高く、そのまま best-vs-best に入れると DeiT-Tiny と ConvNeXt-Atto を不当に低く比べることになります。
逆に差が 1pt 未満なら、上に足しても 1 seed では差を読めないので、足さないのが筋です。
次のどれかなら A に変えます。(1) Phase 1 の締め切りが近く、最大約 65 時間の追加を待てないとき。
(2) 予算を「モデルごとに3点ちょうど」で厳密にそろえることを優先したいとき。
(3) lr 2.5e-4 のセルの結果を見てから決めたいとき（その場合 B のルールは DeiT-Tiny については事後になります）。
