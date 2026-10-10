---
id: 0040
status: pending
created: 2026-10-10
campaign: plan0103-phase1-mobilenetv4-conv-medium-random-s1-v1（plan 0103 Phase 1 の seed 1。MobileNetV4-M・ランダム初期化・SGD lr 0.025・full AT 50 epoch を Hamster GPU0 で実行。hand-run、50/50 epoch 完走、source SHA 609e6fa）
question: MobileNetV4-M の seed 1 は seed 0 と PGD-10 で 0.09pt しか違いませんでした。記録だけして既定の流れ（残り4本の seed 1 を待ってから閾値を見直す）を続けるか、seed 1 の checkpoint にも Anteater で公式評価をかけるか。
options:
  A: 記録だけで追加の実行はしない（追加 0 GPU時間）。Phase 2 の閾値（0.5 / 1.5pt）の見直しは、既定どおり5本の seed 1 がそろってから行う。
  B: seed 1 の best と last を、seed 0 と同じ評価設定で Anteater で公式評価する（公式 val 5万枚の clean / PGD-10 と AutoAttack n=500）。2080 Ti で約 3〜10 時間（粗い見積もり）。Anteater の soft-label バンク作成（10/13 頃まで）が終わってからキューに入れる。
recommendation: A
chosen: null
---

# 0040: Phase 1 の seed 1、MobileNetV4-M（lr 0.025）が完了（seed 0 との差は 0.09pt）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-10 (postrun, MobileNetV4-M seed 1)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `5128d55`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は `epoch-metrics.jsonl` と run-bundle の manifest から読みました）。

条件は seed 0 の `plan0103-phase1-mobilenetv4-conv-medium-random-v1` と同じ設定ファイル（`imagenet_mobilenetv4_conv_medium_pgd_at_phase1_random.yaml`）で、違いは学習の seed（`ARD_SEED=1`）だけです。
ただし実行したホストと source SHA も違います（seed 0 は Ferret・`086072b`、seed 1 は Hamster・`609e6fa`）。
ImageNet-1k（元画像）、MobileNetV4-Conv-M、ランダム初期化、plain PGD-AT、教師なし、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step、評価の攻撃 seed は 0）。
SGD（Nesterov、momentum 0.9、wd 1e-4）、ピーク LR 0.025、multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）、50 epoch、world size 1、バッチ 128。
非決定的モード（cudnn_benchmark あり）、compile なし、CUDA Graphs なし、EMA なし。
数字はすべて内部検証（学習データから取り分けた 25,620 枚。2つの seed で同じ画像）で、公式テストではありません。この run の AutoAttack と公式評価はまだです。

| run | clean / PGD-10 |
|---|---:|
| seed 1、最後（ep 49） | **63.71 / 39.30** |
| seed 1、best（ep 48、内部検証の PGD-10 で選択） | 63.59 / 39.34 |
| seed 0、最後（ep 49） | 未読 / 39.39 |

- **seed 0 との差は PGD-10 で −0.09pt です。** 1本ずつの差の標準誤差（評価画像の数から来るもの）は約 0.43pt なので、その約 0.2 倍です。
- seed 0 の clean の値は Ferret にしかなく、docs にも記録されていません。今回は読み直していないので、clean の seed 差は出せません。
- 0.09pt には、seed の違いだけでなくホストと source SHA の違いも混ざっています（どちらの run も非決定的モードです）。
- 学習崩壊の見張りは一度も発動していません（内部検証 PGD-10 の最大の落ち込みは 0.15pt）。最後の段階でもわずかに上がり続けています（38.79 → 39.30）。
- best と last の差は 0.04pt で、誤差の範囲です。
- 実時間は 35.39 時間でした（Hamster GPU0、515〜521 img/s）。

**ImageNet での seed のばらつきについて**: これが「最良 LR の5モデルに seed 1 を足す」（discussion register C、2026-10-08）の最初の1本です。
そこでは「5本の seed 1 で ImageNet での揺れの幅を測ったら、Phase 2 の閾値（0.5 / 1.5pt）を見直す」と決めています。
いまわかっている ImageNet の seed の組は2つだけです。

- MobileNetV4-M（今回）: 0.09pt。
- MobileNetV4-S 事前学習（Arm A、plan 0101）: 30.58 / 30.59 で 0.01pt。

2組では「ばらつきの推定」にはなりません。CIFAR の画面では seed だけで 1〜2pt 動いた（noise floor）ことと比べると、ImageNet の内部検証での揺れはずっと小さい可能性がありますが、まだ確かめられていません。

**他の packet との関係**:
- 0039（ラベル平滑化）の「判断が変わる場合」は、MobileNetV4-S の seed 差が 0.2pt 未満とわかったら C を考え直す、というものでした。今回の 0.09pt は別のモデル（MobileNetV4-M）の1組なので、その条件はまだ満たされていません。MobileNetV4-S の seed 1 は Ferret で実行中です。
- MobileNetV4-M の baseline+EMA の設定チェック（register C、2026-10-09）は、比べる相手の Phase 1 の seed が2本そろいました（39.39 / 39.30）。

**Hamster GPU0 のこの後**: 計画どおり、MobileNetV4-S の候補の CUDA Graphs 一致テストと、EfficientNet-B0 の 224 px・バッチ 128 の CUDA Graphs のメモリ測定のあと、pull キューに戻ります（2026-10-09 の Progress log）。この packet はその順番を変えません。

## 選択肢

### A: 記録だけで、追加の実行はしない

- **GPU時間**: 追加 0。
- **わかること**: 新しいことはありません。seed の組が1つ増えたことを記録し、閾値の見直しは5本そろってから行います。
- **事前ルール**: 既定（register C、2026-10-08）のまま。5本の seed 1 がすべて終わったら、seed 0 と seed 1 の内部検証 PGD-10（最後の epoch）の差を5組並べ、閾値の見直しを人間が決めます。それまで 0.5 / 1.5pt はそのまま使います。
- **リスク**: seed 0 の clean の値が欠けたままになります。Ferret の `epoch-metrics.jsonl` を読むだけで埋まります（GPU 不要。ただし今の権限設定では Ferret のファイルを読めないので、許可が必要です）。

### B: seed 1 の checkpoint にも公式評価をかける

- **GPU時間**: Anteater の 2080 Ti で約 3〜10 時間（粗い見積もり）。根拠: 公式 val 5万枚で clean と PGD-10 を best と last の2つ、加えて AutoAttack（standard）を500枚ずつ。seed 0 の公式評価の所要時間は docs に記録が無いので、学習の速さ（4090 で約 520 img/s、PGD-3 込み）から推しています。2080 Ti は TF32 がないので、4090 より2〜3倍遅いと仮定しました。
- **前提**: Anteater の GPU 2/3 は soft-label バンク作成中で、終わるのは 10/13 頃の見込みです。それまでは入れられません。seed 0 の公式評価の結果も、まだ docs に記録されていません。
- **わかること**: 公式 val での seed 差。内部検証の seed 差（0.09pt）が公式 val でも同じくらい小さいか。
- **事前ルール**: 指標は last の公式 PGD-10。seed 0 と seed 1 の差が 0.5pt 以内なら「公式 val でも seed の揺れは Phase 2 の閾値より小さい」と記録し、内部検証で Phase 2 を判定し続けます。0.5pt を超えたら、内部検証だけで判定してよいかを人間が見直します。AutoAttack n=500 は標準誤差が約 2pt あるので、seed 差の判定には使いません（数字は記録します）。
- **リスク**: seed 0 の公式評価と同じ評価設定（ホスト、データのコピー、評価コード）をそろえる必要があります。「公式評価は選んだ LR にだけ行う」という方針（2026-10-05）の範囲ですが、seed 1 にまで広げるのは初めてなので、人間の判断が要ります。

**除いた選択肢**: いまの 0.09pt を使って Phase 2 の閾値を見直すことは、register C の「5本そろってから」と矛盾するので選択肢に入れていません。

## おすすめ: A

この run が答えるのは「seed で内部検証の PGD-10 がどれだけ動くか」で、その答え（0.09pt）は記録済みです。閾値の見直しは5本そろってからと決まっていて、1組だけで動かす理由はありません。
B は公式 val での seed 差を測れますが、Anteater は 10/13 頃まで空かず、seed 0 の公式評価の結果もまだ記録されていません。いま B を決めても、すぐには何も変わりません。
seed 0 の clean の値は、Ferret を読む許可があれば GPU なしで埋められます。

**判断が変わる場合**: 論文の主要な表で公式評価の seed 差が必要になったとき、または seed 0 の公式評価が記録されて Anteater が空いたときは、B を考え直す価値があります。
残り4本の seed 1 のどれかで差が 0.5pt を超えたら、閾値の見直しを5本を待たずに前倒しするかを、別の packet で尋ねます。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習や評価は始めません。
