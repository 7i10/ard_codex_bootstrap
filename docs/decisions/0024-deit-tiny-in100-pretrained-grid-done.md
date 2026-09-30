---
id: 0024
status: pending
created: 2026-09-30
campaign: plan0103-in100-tune-deit-tiny-adamw-pretrained-lr2p5em4-v1（plan 0103 の ImageNet-100 学習率調整、DeiT-Tiny・AdamW・事前学習済み初期化・ピーク LR 2.5e-4。Hamster GPU0 の hand-run、seed 0、50/50 epoch 完走、source SHA 5a5dcba）
question: DeiT-Tiny 事前学習済みの ImageNet-100 学習率グリッドが終わり、Hamster GPU0 が空いた。ImageNet-1k で DeiT-Tiny 事前学習済みの3点グリッドをすぐ回すか、AdamW 系の学習率の決め方（0023 で未決）を先に決めるまで待つか。
options:
  A: ImageNet-1k で承認済みの3点 {6.25e-5, 1.25e-4, 2.5e-4} を Hamster GPU0 で順に回す（約100-120 GPU時間、約4.5-5日）。DeiT-Tiny 事前学習済みの ImageNet-1k 学習率を決める。ルールは最終 epoch の PGD-10 の最大値、1pt 未満は区別できないと記録。
  B: 何もしない。0023（MobileViT-S の ImageNet-1k 学習率）を先に決め、AdamW 系3モデルに同じ決め方を当てはめる。その間に Ferret の DeiT-Tiny ランダム初期化グリッドを記録する。GPU 0時間、Hamster GPU0 は空いたまま。
recommendation: B
chosen: null
---

# 0024: DeiT-Tiny 事前学習済みの ImageNet-100 グリッド完了。ImageNet-1k の学習率をどうするか

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、
2026-09-30 (postrun, DeiT-Tiny pretrained / 6.25e-5, 1.25e-4, 2.5e-4) の3項目。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の3行。結果のコミットは `cd919c1`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません
（plan 0103/0104 の他の hand-run と同じ扱い）。

条件: ImageNet-100 代理データ、DeiT-Tiny、ImageNet-1k 事前学習済み（分類ヘッドは 100 クラスで作り直し）、
plain PGD-AT（教師なし）、seed 0、l_inf 4/255、学習時 PGD-3、評価 PGD-10。
AdamW、wd 0.05、50 epoch、world size 1、バッチ 128。
数字はすべて内部検証（学習用データから取り分けた 2,564 枚）で、公式テストではありません。
AutoAttack はまだ走っていません。

内部検証（%）、学習率 低 → 高:

| 学習率 | best PGD-10 (epoch) | last PGD-10 | best の clean | last の clean | probe − val（last、clean / PGD-10） |
|---|---:|---:|---:|---:|---:|
| 6.25e-5 | 56.51 (36) | 55.93 | 82.37 | 82.64 | +5.7 / +12.0 |
| 1.25e-4 | 57.96 (29) | 57.45 | 83.07 | 83.81 | +8.4 / +17.5 |
| 2.5e-4 | **58.66** (34) | **58.11** | 83.31 | 83.46 | +11.8 / +22.1 |

- 最大値はグリッドの上端（2.5e-4）です。MobileViT-S 事前学習済みと同じ形です。
- 上位2点の差は PGD-10 で +0.70pt（best 同士）、+0.66pt（last 同士）です。
  2つの検証値の差の標準誤差は約 1.4pt なので、約 0.5 SE です。区別できません（seed は各1本）。
- 下端の 6.25e-5 との差は best 同士で +2.15pt（約 1.5 SE）です。
- どの学習率でも best と last の差は 0.6pt 未満で、検証の SE（約 1.0pt）より小さいです。
- 学習に使った画像と未使用の画像の差（probe − val）は、学習率とともに広がります。
  ただし val の PGD-10 は下がっていません。
- 2.5e-4 の run では epoch 19 に PGD-10 が1 epoch だけ 47.50% に落ち、次の epoch で戻りました。
  probe でも同じ epoch に落ちているので、測定の揺れではなく実際の揺れです。最終結果には影響していません。
- DeiT-Tiny の事前学習済みとランダム初期化の比較は、まだできません。
  ランダム初期化の ImageNet-100 グリッドは Ferret で回しましたが、まだ記録されていません。
  このセッションからは Ferret の状態を確認できませんでした（ssh の許可なし）。

守るべき既存の判定:

- **A2（2026-09-29）**: ImageNet-100 の代理学習は、学習率の選択には使えません。
  使えるのは初期化の比較（差の向き）だけです。今回のグリッドの最大値も、ImageNet-1k の学習率選びには使いません。
- **公平性の原則（A2）**: モデルごとに調整の予算をそろえ、best 同士で比べます。
- **0023（pending）**: MobileViT-S の ImageNet-1k 学習率の決め方がまだ決まっていません。
  その選択肢 C は「DeiT-Tiny と ConvNeXt-Atto の ImageNet-100 グリッドがそろってから、
  AdamW 系3モデルの決め方をまとめて決める」です。今回そろったのは DeiT-Tiny の事前学習済みだけで、
  DeiT-Tiny のランダム初期化と ConvNeXt-Atto はまだです。
- 承認済みの ImageNet-1k グリッド（2026-09-28）: AdamW 事前学習済み {6.25e-5, 1.25e-4, 2.5e-4}。
  ImageNet-1k 用の DeiT-Tiny AdamW の config はまだありません（あるのは SGD の survey 用と phase1 random 用だけ）。

## ノイズの目安

- ImageNet-1k の内部検証は 25,620 枚です。PGD-10 が 30% 前後のとき、1回の測定の SE は約 0.3pt です。
- plan 0104 と 0023 は判定の閾値を **1pt** にしました。本パケットでも同じ閾値を使います。
  1pt 未満の差は「区別できない」と記録し、結果としては読みません。
- MobileNetV4-S 事前学習済みの ImageNet-1k グリッドでは、3点の幅が 1.34pt しかありませんでした。
  DeiT-Tiny でも上位2点が 1pt 以内に収まる可能性は高いです。

## コストの前提

- ImageNet-100 で測った DeiT-Tiny の速度は 549-556 img/s です（compile なし、fp32、Hamster の 4090）。
  1 epoch の学習は 229 秒、検証と probe を含めて約 250 秒でした。
- ImageNet-1k の学習画像は約 125.6 万枚（98%）なので、学習だけで 1 epoch 約 0.64 時間です。
  検証（25,620 枚、PGD-10）を足して約 0.67-0.7 時間です。
- 50 epoch で1本 **約 34-40 GPU時間** です。3本で約 100-120 GPU時間になります。
- Hamster GPU1 は MobileNetV4-S の2段階学習（stage 1）で使用中なので、GPU0 1枚で順に回す前提です。
  実時間は約 4.5-5 日です。
- ImageNet-1k の元画像の読み込み速度（Hamster、16 workers で約 4,300 img/s）は、DeiT-Tiny の学習速度より十分速いです。
  読み込みが律速になる心配は小さいです。

## 選択肢

### A: ImageNet-1k で DeiT-Tiny 事前学習済みの3点を回す

- **内容**: DeiT-Tiny、事前学習済み、AdamW、wd 0.05、50 epoch、学習率 {6.25e-5, 1.25e-4, 2.5e-4}。
  レシピとグリッドは承認済みなので、新しい科学的な契約ではありません。
  やることは ImageNet-1k 用の config を3つ書き、source を固定し、`/experiment-launch` で起動することです。
  Hamster GPU0 で1本ずつ回します。
- **GPU時間**: 約 100-120（実時間 約4.5-5日）。
- **わかること**: DeiT-Tiny 事前学習済みの ImageNet-1k の学習率を、MobileNetV4-S と同じ予算・同じ指標で決められます。
  経験則「ImageNet-1k の最適値は ImageNet-100 の最大値以下」が2モデル目でも成り立つかも確かめられます。
- **事前登録ルール**:
  - 採用する学習率は、最終 epoch（49）の内部検証 PGD-10 が最大の点（plan 0104、0023-A と同じ指標）。
  - 上位2点の差が 1pt 未満なら「区別できない」と記録します。best 同士の比較には最大の点を使い、
    同点に近いことを明記します。
  - 最大値が上端（2.5e-4）で、しかも 1pt 以上勝っている場合は、グリッドを足さずに人間に戻します。
- **リスク**: 0023 で人間が「3点グリッド以外の方法」（例: 文献の線形則で1点に固定）を選んだ場合、
  この3本はモデル間で方法がそろわず、参考値にしかなりません。
  また、0023-A を選ぶと MobileViT-S が Hamster を約1週間ほしがるので、GPU の取り合いになります。
  seed は1本だけです。

### B: 何もしない（0023 を先に決める）

- **内容**: 新しい GPU ジョブは入れません。先に 0023 で AdamW 系の ImageNet-1k 学習率の決め方を決め、
  同じ決め方を DeiT-Tiny にも当てはめます。
  その間に、Ferret の DeiT-Tiny ランダム初期化 ImageNet-100 グリッドを `/experiment-postrun` で記録し、
  DeiT-Tiny の「事前学習済み vs ランダム初期化」の差の向きを出します（GPU を使わない作業です）。
- **GPU時間**: 0。Hamster GPU0 は空いたままです。
- **わかること**: 新しい GPU の結果は出ません。ただし、AdamW 系3モデルに同じ方法を当てはめることが保証されます。
  ランダム初期化グリッドの記録で、A2 が代理学習に認めた唯一の用途（初期化の比較）が DeiT-Tiny で完了します。
- **事前登録ルール**: 0023 の `chosen` が埋まった時点で、同じ方法を DeiT-Tiny に当てはめます。
  0023 が A（ImageNet-1k で3点）なら、本パケットは A を選んだのと同じ扱いにして、新しいパケットで起動の順番だけ決めます。
  0023 が C のままなら、ConvNeXt-Atto の ImageNet-100 グリッドがそろってから新しいパケットを書きます。
- **リスク**: 決定が遅れた分だけ Hamster GPU0 が空きます。DeiT-Tiny の3本は約5日かかるので、
  待った日数がそのまま Phase 1 の完了の遅れになります。

### 検討して外した案

- **ImageNet-100 で 1.25e-4 と 2.5e-4 に seed を足し、上位2点を区別する案**: 外しました。
  A2 で、代理学習の順位は ImageNet-1k の学習率選びに使えないと判定済みです。区別できても、使い道がありません。
- **ImageNet-100 のグリッドを 5e-4 まで広げる案**: 外しました。同じ理由で、代理学習の最大値の位置は使いません。
- **ImageNet-1k で2点だけ回す案**（0023-B と同じ形）: 外しました。0023 と同じく、打ち切るとモデル間で予算がそろわず、
  A2 の公平性の原則に反します。

## 推奨の理由

B を推奨します。DeiT-Tiny 固有の新しい判断材料は、今回のグリッドからは出てきません。
A2 により、ImageNet-100 の最大値の位置は ImageNet-1k の学習率選びに使えないからです。
残る問いは「AdamW 系の ImageNet-1k の学習率をどう決めるか」で、これは 0023 ですでに問われています。
ここで A を選ぶと、0023 の答えを先取りすることになります。
一方、0023 が A（ImageNet-1k で3点）に決まれば、本パケットの A はすぐ起動できます。
DeiT-Tiny は1本 約35-40 GPU時間で、MobileViT-S（約80-100）の半分以下なので、
Hamster GPU0 では DeiT-Tiny を先に回すのが自然です。

判断が変わる条件は次の2つです。

- 0023 を A（ImageNet-1k で3点グリッド）に決める場合。そのときは本パケットも A にし、
  GPU0 で DeiT-Tiny を先に回すのが最短です。
- 「GPU を空けておくこと」自体のコストを重く見る場合。DeiT-Tiny の3本は他モデルより安いので、
  0023 の結論を待たずに A を始めても、失うのは最大で約 100-120 GPU時間です。
