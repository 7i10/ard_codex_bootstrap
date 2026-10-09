---
id: 0038
status: pending
created: 2026-10-09
campaign: plan0103-phase2-mnv4s-cosine-s0-v1（plan 0103 Phase 2 の候補①「cosine」。MobileNetV4-S・ランダム初期化・SGD lr 0.025・full AT 50 epoch を Hamster GPU1 で実行。hand-run、seed 0、50/50 epoch 完走、source SHA d836c03）
question: MobileNetV4-S の cosine は multistep より PGD-10 が +0.78pt で、事前ルールでは「保留（2本目の seed で確かめる）」に入りました。cosine の seed 1 を回すか、すでにキューにある他モデルの cosine と seed のばらつきの測定を待ってから決めるか。
options:
  A: MobileNetV4-S cosine の seed 1 を1本回す（約 15〜17 GPU時間、4090 1枚）。Hamster GPU1 の baseline+EMA の後に置き、その設定チェックが通ったときだけ起動する。判定は下の事前ルール。
  B: いまは何も足さない（追加 0 GPU時間）。Phase 1 の MobileNetV4-S seed 1（Ferret で実行中）と baseline+EMA の設定チェックが終わってから、seed のばらつきを見てもう一度決める。
  C: MobileNetV4-S の seed 1 は回さず、キューにある他の3モデルの cosine（ConvNeXt-Atto、MobileNetV4-M、EfficientNet-B0）を確認の代わりにする（追加 0 GPU時間）。事前ルールの「同じモデルの2本目の seed」とは違う確かめ方になる。
recommendation: A
chosen: null
---

# 0038: Phase 2 の cosine、MobileNetV4-S seed 0 が完了（判定は「保留」）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-09 (postrun, Phase 2 cosine)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `5fe9495`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は `epoch-metrics.jsonl` と run-bundle の manifest から読みました）。

基準は Phase 1 の `plan0103-mnv4s-fullat-50ep-lr0025-cg-s0-v1`（source `65369a3`）です。
違いは次の3つだけです。

- 学習率のスケジュール: 10 epoch のウォームアップの後、cosine で 0 まで下げる（基準は epoch 25 と 38 で 1/10）。
- Phase 2 の共通設定: 毎 epoch の評価を 5,000 枚の固定サブセットで行う。EMA（0.9999）を並行して保持する。
- W&B のグループと source SHA。

それ以外は同じです。ImageNet-1k（元画像）、MobileNetV4-S、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、momentum 0.9、wd 1e-4）、ピーク LR 0.025、50 epoch、world size 1、バッチ 128、決定的モード＋CUDA Graphs。
数字はすべて内部検証（学習データから取り分けた 25,620 枚。基準と同じ画像）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| cosine、最後（ep 49）、普通の重み | **55.09 / 31.49** |
| cosine、最後（ep 49）、EMA の重み | 55.11 / 31.58 |
| cosine、best（ep 48、サブセットで選択）、普通の重み | 55.01 / 31.61 |
| 基準 multistep、最後 = best（ep 49） | 53.85 / 30.71 |

- **差は +1.24 clean / +0.78 PGD-10 です**（最後の普通の重みどうし。Phase をまたぐときはこれが同じ条件の比較です）。
  1本ずつの差の標準誤差は約 0.42pt なので、PGD-10 の差は約 1.9 倍、clean の差は約 3 倍です。
  ただし、これは評価画像の数から来る誤差だけです。seed を変えたときの ImageNet でのばらつきは、まだ測っていません。
- **事前ルール（人間、2026-10-08）**: 1.5pt 以上は「効果あり」、0.5〜1.5pt は「保留（2本目の seed で確かめる）」、0.5pt 未満は「効果なし」。
  +0.78pt は **「保留」** です。
- **設定チェックは一部だけ済んでいます。** ウォームアップの 10 epoch は両方のスケジュールで同じです。
  その間の probe の精度と学習損失は、基準と完全に一致しました（epoch 0 と 9 で確認）。
  完全なチェックは `plan0103-phase2-mnv4s-baseline-ema-s0-v1` です。
  これは Hamster GPU1 でラベル平滑化の次に入っていて、基準の 53.85 / 30.71 をビット単位で再現するはずです。
- **EMA は最後には効いていません**（+0.02 / +0.08）。学習率がほぼ 0 なので、予想どおりです。
  途中では、EMA はウォームアップ中に普通の重みよりずっと悪く、学習率が高い間はサブセットで最大 +5.8pt PGD-10 上回りました。
  ウォームアップ中に悪い理由は確かめていません。普通の重みと best.pt には影響しません。
- 学習崩壊の見張りは一度も発動していません。実時間は 15.81 時間でした（基準は 15.45 時間）。

**ばらつきの目安（noise floor）**: CIFAR の画面では、seed を変えるだけで 1〜2pt 動くことがありました。
ImageNet の値はまだありません（最良 LR の5モデルの seed 1 で測る予定。うち MobileNetV4-S は Ferret で実行中）。
+0.78pt はどちらの目安でも、1本だけで「効果あり」と言える大きさではありません。

## 選択肢

### A: MobileNetV4-S cosine の seed 1 を1本回す

- **GPU時間**: 約 15〜17 時間（4090 1枚）。根拠はこの run の 15.81 時間です（同じモデル、同じ設定、Hamster）。
- **置き場所**: Hamster GPU1 の baseline+EMA の後。baseline+EMA が 53.85 / 30.71 を再現しなかったときは起動しません（比較の土台が崩れるため）。
- **わかること**: seed 1 どうしの差 d1（cosine seed 1 − Phase 1 の multistep seed 1）。seed 0 の差 d0 = +0.78 と合わせて、2本での向きがわかります。
- **事前ルール（ここで決めて、run の前に固定します）**: 指標は、最後の epoch・普通の重み・全取り分け画像の PGD-10。
  - d0 と d1 がどちらも +0.5pt 以上 → MobileNetV4-S で cosine は「効果あり」（この2本の seed での向き）。
  - d1 が 0.5pt 未満 → 「効果なし」。multistep のまま。
  - どちらの場合も、2本の平均を「分布の推定」としては扱いません。
- **リスク**: 比較相手の multistep seed 1 が Ferret でまだ走っていて、ホストも source SHA も違います。
  キューの次の Phase 2 項目が約 16 時間遅れます。

### B: いまは何も足さず、ばらつきの測定を待つ

- **GPU時間**: 追加 0。
- **わかること**: Phase 1 の MobileNetV4-S seed 0 と seed 1 の差（seed のばらつき）と、設定チェックの結果。
  2026-10-08 の決定で、この測定が出たら Phase 2 の判定基準を見直すことになっています。
- **事前ルール**: seed 0 と seed 1 の差が 0.78pt 以上なら、cosine の差はばらつきの範囲内とみなし「効果なし」とする。
  0.78pt より小さければ、A をもう一度検討する。
- **リスク**: 結論が出るのが遅れます。ばらつきが小さかった場合は、結局 A の 16 時間が必要になります。

### C: 他の3モデルの cosine を確認の代わりにする

- **GPU時間**: 追加 0（ConvNeXt-Atto、MobileNetV4-M、EfficientNet-B0 の cosine はすでに Hamster のキューにあります）。
- **わかること**: cosine の効果がモデルをまたいで同じ向きに出るか。
- **事前ルール**: 4モデル（MobileNetV4-S を含む）のうち3つ以上で、cosine − 基準（最後・普通の重み・PGD-10）が +0.5pt 以上なら「効果あり」。
  2つ以下なら「効果なし」。
- **リスク**: 2026-10-08 のルールは「同じモデルの2本目の seed」です。C はこれと違う確かめ方なので、採る場合はルールを変えることになります。
  また、他モデルの基準（Phase 1）は非決定的モードで走ったものがあり、設定チェック（各モデルの baseline+EMA）が終わるまで比較に注意が必要です。

## おすすめ: A

事前ルールが「保留なら2本目の seed」と決めていて、A はそれをそのまま実行するものです。
費用は約 16 時間と小さく、Hamster のキューは長いので、待っても GPU は空きません。
baseline+EMA の後に置けば、設定チェックが先に終わります。
チェックに失敗したら起動しない条件をつけているので、無駄打ちになりません。
C の他モデルの cosine は、A とは別にどのみち走ります。モデルをまたいだ一貫性は、その結果で追加で見られます。

**判断が変わる場合**: Ferret の seed 1 が先に終わり、seed 0 と seed 1 の差が 0.78pt 以上あれば、B のルールで「効果なし」と読めます。
その場合は A を回す必要はありません。また、cosine を Phase 2 全体の標準のスケジュールにするかどうかは、この packet の範囲外です。
それはスケジュールの変更なので、決めるときは plan に新しい契約として書きます。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習は始めません。
