---
id: 0037
status: pending
created: 2026-10-09
campaign: plan0103-phase1-efficientnet-b0-random-lr00125-v1（plan 0103 Phase 1 の LR グリッド。EfficientNet-B0・ランダム初期化・SGD lr 0.0125・full AT 50 epoch を Hamster GPU1 で実行。hand-run、seed 0、50/50 epoch 完走、source SHA 609e6fa）
question: EfficientNet-B0 の 3点グリッドがそろいました（0.0125 / 0.025 / 0.05 = 34.25 / 35.93 / 35.88）。0036 A の事前ルールどおり lr 0.025 を採用してグリッドを閉じるか、lr 0.1 を1セル足すか。
options:
  A: lr 0.025 を採用し、EfficientNet-B0 の LR グリッドを閉じる。lr 0.1 は回さない（追加 GPU 0時間）。0.05 とは区別できないと記録する。
  B: lr 0.1 を1セル足し、4点の argmax（最後の epoch の内部検証 PGD-10）を採用する。それ以上は足さない（追加 約 47〜50 GPU時間、Hamster の 4090 1枚）。
recommendation: A
chosen: null
---

# 0037: Phase 1 LR グリッド、EfficientNet-B0 SGD lr 0.0125 が完了（3点がそろった）

この packet は 0036（lr 0.05 完了時、`chosen` は未記入）の問いを、0.0125 の結果を入れて置き換えます。
よければ 0036 を `superseded` にしてください。

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-09 (postrun, EfficientNet-B0 lr 0.0125)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `494bb1b`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は `epoch-metrics.jsonl` と run-bundle の manifest から読みました）。

条件は 3セルとも LR 以外は同じです（config の差は LR と W&B グループだけ）。
ImageNet-1k（元画像）、EfficientNet-B0、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、momentum 0.9、wd 1e-4）、10 epoch ウォームアップ、epoch 25 と 38 で 1/10、50 epoch。
world size 1、バッチ 128、非決定的モード、torch.compile あり。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack は走っていません。

| LR | 実行 | 最後の epoch（49）の clean / PGD-10 | best の epoch と clean / PGD-10 |
|---|---|---:|---:|
| 0.0125 | Hamster GPU1、source `609e6fa`（この run） | 56.08 / 34.25 | ep 48: 55.98 / 34.32 |
| 0.025 | Ferret、source `086072b` | 未確認 / **35.93** | 未確認 |
| 0.05 | Hamster GPU0、source `609e6fa` | 57.86 / 35.88 | ep 48: 57.79 / 36.00 |
| 0.1 | config のみ（コミット `5b6b443`）、未起動 | — | — |

- **0.0125 ははっきり低いです。** 0.025 より 1.68pt、0.05 より 1.63pt 低い（2 run の差の標準誤差 約 0.42pt の約 4 倍）。
  clean も 0.05 より 1.78pt 低いです。
- **0.025 と 0.05 は同点のままです**（差 0.05pt）。
- 0036 A の事前ルール（最後の epoch の内部検証 PGD-10 が最大の LR を採用）を当てはめると **0.025** です。
  0.0125 は argmax ではないので、「下端が最良なら別 packet」の条件は発火しません。
- 学習は正常です。PGD-10 が直前の最大値から落ちた幅は最大 0.20pt。catastrophic-overfitting の判定は一度も出ていません。
  ただし lr 0.0125 の区間は epoch 24 でもまだ上がり続けていて（26.25 → 26.55 → 26.91）、学習が足りていない形です。
- best と last は区別できません（差 0.07pt）。
- 時間: 49.30 時間（学習は約 46.6 時間）、ほぼ 374〜376 img/s。

**ほかの BN CNN と同じ形です（同じ SGD レシピ、最後の epoch の PGD-10）:**
- MobileNetV4-M: 0.0125 / 0.025 / 0.05 = 37.00 / 39.39 / 39.10
- EfficientNet-B0: 0.0125 / 0.025 / 0.05 = 34.25 / 35.93 / 35.88
- MobileNetV4-S: 0.0125 / 0.025 / 0.1 = 29.16 / 30.71 / 29.18（0.1 は 0.025 より 1.5pt 低い）

## ノイズの目安

- 検証の標準誤差は 1 run あたり PGD-10 で約 0.30pt、2 run の差では約 0.42pt です。
- ImageNet での seed 間の揺れはまだ測っていません。CIFAR 時代の経験では乱数だけで 1〜2pt 動きました。
- 0.0125 の -1.68pt は検証の標準誤差は大きく超えますが、seed の揺れ（1〜2pt）の上端とは同じくらいです。
  ただ、どう読んでも 0.0125 が argmax にはならないので、採用 LR の判断には影響しません。
- 0.025 と 0.05 の 0.05pt は、どちらの尺度でも読めない差です。seed を足しても、この差を決着させるには
  現実的でない本数が要るので、選択肢に入れていません。

## 費用の前提

- 0.0125 と 0.05 はどちらも Hamster の 4090 1枚で 48.6〜49.3 時間でした。LR で速度は変わらないので、0.1 も 47〜50 時間と見ます。
- いま Hamster は2枚とも使用中です（Phase 2 の `plan0103-phase2-mnv4s-cosine-s0-v1` が走っています。もう1枚の中身は今回確認していません）。
  B はどちらかが空くのを待つか、Phase 2 の GPU を約2日分使います。

## 選択肢

### A: lr 0.025 を採用してグリッドを閉じる（推奨）

- **GPU時間**: 追加 0。
- **分かること**: EfficientNet-B0 の採用 LR が決まり、Phase 1 の best-vs-best にこの LR を使えます。
  plan の手順どおり、公式 val での評価（clean、PGD、別プロセスの AutoAttack）は採用 LR にだけ行います（その起動は別に判断）。
- **事前に決めるルール**（0036 A と同じ。0.0125 の結果を見る前に 0036 に書かれたもの）:
  - 採用は最後の epoch の内部検証 PGD-10 の argmax、つまり 0.025。
  - 0.025 と 0.05 の差が 0.42pt 未満なので、論文と best-vs-best の表に「0.05 とは区別できない」と書く。
- **リスク**: 最適が 0.05 と 0.1 の間にあった場合、EfficientNet-B0 だけ少し低い数字で比べることになります。
  上の 3モデルの形からは、その差は小さいと見ます。
  また 0.025 のセルだけ Ferret・別 source（`086072b`）で走っています。公式 val 評価は保存済み checkpoint から行うので、
  その時点で 0.025 の clean と best も読み直します。

### B: lr 0.1 を1セル足す

- **GPU時間**: 追加 約 47〜50 時間（Hamster の 4090 1枚）。
- **分かること**: 上端の外側で PGD-10 が下がるか。つまり最適が 0.1 より上にないこと。
- **事前に決めるルール**（0.1 の結果を見る前に、この packet で記録）:
  - lr 0.1 を1セル足す（`imagenet_efficientnet_b0_pgd_at_phase1_random_lr01.yaml`、ほかは 0.05 のセルと同じ、seed 0）。
  - 4点（0.0125 / 0.025 / 0.05 / 0.1）の argmax（最後の epoch の内部検証 PGD-10）を採用する。
  - 0.1 が勝っても、それ以上は足さない。0.1 が argmax なら「最適はグリッドの上にあるかもしれない」と論文に書く。
  - 0.1 と 1位の差が 0.42pt 未満なら「区別できない」と書く。
- **リスク**: Phase 2 が Hamster 1枚分、約2日遅れます。
  ConvNeXt-Atto と DeiT-Tiny は延長しているので、扱いをそろえる意味はありますが、
  あちらは上端が明確に最良だった（DeiT-Tiny は +2.02pt）のに対し、EfficientNet-B0 は上端が同点です。

## 推奨

A を推奨します。0036 A のルールは 0.0125 の結果を見る前に書かれていて、それを当てはめると 0.025 です。
0.0125 が 1.7pt 低く、0.025〜0.05 が平らな頂上という形は MobileNetV4-M とほぼ同じで、
MobileNetV4-S では 0.1 が 1.5pt 下がっています。B に約2日を使っても、分かるのはおそらく「0.1 は低い」ことだけで、
採用 LR は変わらない見込みが高いです。Hamster は2枚とも Phase 2 で使っているので、その2日は Phase 2 に回せます。
次のどれかなら B に変えます。(1)「上端が最良」をノイズ以内の同点も含むと読み、ConvNeXt-Atto・DeiT-Tiny と扱いを完全にそろえたいとき。
(2) EfficientNet-B0 の数字が Phase 1 の結論（BN CNN の順位、EasyRobust の 35% AutoAttack との比較）を左右し、低めの数字で比べるのを避けたいとき。
(3) Phase 2 の予定が入らず、Hamster の1枚が2日以上空くとき。
