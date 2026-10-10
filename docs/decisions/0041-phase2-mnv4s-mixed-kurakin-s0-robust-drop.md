---
id: 0041
status: pending
created: 2026-10-10
campaign: plan0103-phase2-mnv4s-mixed-kurakin-s0-v1（plan 0103 Phase 2 の候補③「Kurakin の mixed batch」。各バッチ 128 枚の半分だけを攻撃し、残り半分はきれいな画像のまま学習する。攻撃画像側の重み 0.3、BN は共有。MobileNetV4-S・ランダム初期化・SGD lr 0.025・50 epoch を Hamster GPU0 で実行。hand-run、seed 0、50/50 epoch 完走、source SHA 5dd329e）
question: MobileNetV4-S の mixed batch（半分だけ攻撃、重み 0.3、BN 共有）は基準と比べて clean が +9.03pt 上がり、PGD-10 が −13.37pt 下がりました。この設定の mixed batch をここで打ち切るか、BN を分ける版を1本だけ MobileNetV4-S で試すか、キューに入っている MobileNetV4-M と EfficientNet-B0 の同じ設定の run もそのまま回すか。
options:
  A: 打ち切る（追加 0 GPU時間）。Hamster のキューから MobileNetV4-M と EfficientNet-B0 の mixed batch（BN 共有）の2行を外す（約 50〜70 GPU時間の節約）。BN を分ける版を試すかどうかは、Ferret のキューにある EfficientNet-B0 の BN 分割版に任せる。
  B: A に加えて、MobileNetV4-S の BN 分割版（imagenet_mobilenetv4_pgd_at_phase2_mixed_kurakin_splitbn.yaml）を seed 0 で1本回す（約 12〜16 GPU時間、4090 1枚）。PGD-10 が下がった原因が BN の共有かどうかを、同じモデルで確かめる。
  C: キューをそのまま回し、MobileNetV4-M と EfficientNet-B0 でも同じ設定の mixed batch を1本ずつ回す（約 50〜70 GPU時間）。clean と PGD-10 の交換がモデルによって変わるかを見る。
recommendation: A
chosen: null
---

# 0041: Phase 2 の mixed batch、MobileNetV4-S seed 0 が完了（clean は上がるが、PGD-10 が大きく下がる）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-10 (postrun, Phase 2 mixed batch)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `22b301f`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は `epoch-metrics.jsonl` と run-bundle の manifest から読みました）。

基準は Phase 1 の `plan0103-mnv4s-fullat-50ep-lr0025-cg-s0-v1`（source `65369a3`）です。違いは次の3つだけです。

- **mixed batch**（Kurakin, Goodfellow & Bengio 2017）。各バッチ 128 枚のうち先頭 64 枚だけを PGD-3 で攻撃し、残り 64 枚はきれいな画像のまま学習に入れます。
  損失は「きれいな側 + 0.3 × 攻撃側」を「64 + 0.3 × 64」で割ったものです。攻撃画像の損失が全体に占める割合は 0.15 / 0.65 = **23%** です（基準の full AT では 100%）。
  BN は1組だけで、きれいな画像と攻撃画像の両方を見ます。
- Phase 2 の共通設定: 毎 epoch の評価を 5,000 枚の固定サブセットで行う。EMA（0.9999）を並行して保持する。
- W&B のグループと source SHA。

それ以外は同じです。ImageNet-1k（元画像）、MobileNetV4-S、ランダム初期化、教師なし、seed 0、224 px。
学習率のスケジュールは multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、momentum 0.9、wd 1e-4）、ピーク LR 0.025、50 epoch、world size 1、バッチ 128、決定的モード＋CUDA Graphs。
数字はすべて内部検証（学習データから取り分けた 25,620 枚。基準と同じ画像）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| mixed batch、最後（ep 49）、普通の重み | **62.88 / 17.34** |
| mixed batch、最後（ep 49）、EMA の重み | 63.04 / 17.59 |
| mixed batch、best（ep 48、サブセットで選択）、普通の重み | 62.77 / 17.49 |
| mixed batch、best EMA（ep 37） | 62.24 / 17.85 |
| 基準 full AT・multistep、最後 = best（ep 49） | 53.85 / 30.71 |
| 参考: Phase 2 の cosine（full AT）、最後、普通の重み | 55.09 / 31.49 |
| 参考: 通常学習（攻撃なし）`plan0103-mobilenetv4-standard-50ep-v1`、clean のみ | 70.86 / — |

- **差は +9.03 clean / −13.37 PGD-10 です**（最後の普通の重みどうし）。1本ずつの差の標準誤差は clean で約 0.43pt、PGD-10 で約 0.37pt なので、どちらも 20 倍以上です。偶然では説明できません。
- **事前ルール（人間、2026-10-08）**: 1.5pt 以上は「効果あり」（seed 1本でよい）、0.5〜1.5pt は「保留」、0.5pt 未満は「効果なし」。
  このルールは PGD-10 が上がる場合を想定して書かれています。今回は PGD-10 が 13.37pt **下がって**おり、1本で判定してよい大きさ（1.5pt）の約 9 倍です。
  つまり、この設定の mixed batch は MobileNetV4-S の頑健性を上げません。clean を取って頑健性を手放す方向に動いています。
- **差は最初から開いています。** ウォームアップの終わり（epoch 9）で、サブセットの値は 40.46 / 8.72 です。同じ時点の cosine（基準とウォームアップが完全に一致した run）は 30.96 / 15.98 なので、すでに +9.50 / −7.26 です。その後も差は縮まりません。
- **原因はまだ調べていません。** 候補は2つです。
  1. 攻撃画像の損失が全体の 23% しかない。
  2. BN の共有。評価（攻撃も含む）は eval モードの BN 統計を使いますが、その統計はきれいな画像と攻撃画像の平均です。
  BN を分ける版（攻撃画像側の BN を本体として保存・評価に使う。AdvProp と同じ考え方）なら 2 を切り分けられます。1 はどちらの版でも残ります。
- 学習崩壊の見張りは一度も発動していません（サブセットでの最大の落ち込みは 0.96pt）。best と last の差は 0.16pt で、誤差の範囲です。
- **速い**: 実時間 10.55 時間（full AT の Phase 2 は 15.4〜15.8 時間）。攻撃するのが半分なので、処理速度は 1,763〜1,769 img/s、full AT の約 1.50 倍です。

**ばらつきの目安（noise floor）**: ImageNet で seed を変えた差は、今のところ MobileNetV4-M で 0.09pt、MobileNetV4-S（事前学習あり）で 0.01pt です。CIFAR では 1〜2pt 動くことがありました。
−13.37pt はどの目安よりはるかに大きいので、seed を増やしても結論は変わりません。

**キューの状況**:
- Hamster の `queues/work_queue.txt` に、同じ設定（半分攻撃、重み 0.3、BN 共有）の `plan0103-phase2-mnv4m-mixed-kurakin-s0-v1`（14行目）と `plan0103-phase2-effb0-mixed-kurakin-s0-v1`（16行目）が残っています。
- Ferret のキューには EfficientNet-B0 の BN 分割版が入っています（plan の 2026-10-09 の記録より）。ここからは中身を確認していません。
- MobileNetV4-S の BN 分割版の設定ファイルはありますが、Hamster のキューには入っていません。
- この run が終わったあと、**Hamster GPU0 は空いています**。worker のログは `QUEUE DONE` で終わっていて、キューの残り 15 行は GPU0 では引かれていません。どの項目をいつ GPU0 に流すかは、この packet とは別に決めてください。

## 選択肢

### A: 打ち切る

- **GPU時間**: 追加 0。Hamster のキューから2行を外すので、約 50〜70 GPU時間の節約になります。
  見積もりの根拠: MobileNetV4-M の full AT は Hamster で 35.4 時間、EfficientNet-B0 は 48.6 時間（どちらも Phase 1 の設定）。mixed batch は MobileNetV4-S で 1.50 倍速かったので、それぞれ約 20〜28 時間と 30〜40 時間です。Phase 2 の決定的モード＋CUDA Graphs で速さが変わる分は幅に含めています。
- **わかること**: 新しいことは何もわかりません。今回の結果をそのまま採ります。
- **事前ルール**: すでに適用済みです（PGD-10 が 1.5pt を大きく超えて下がった → この設定は Phase 2 の組み合わせに入れない）。
- **リスク**: モデルによって交換の具合が違う可能性は確かめません。ただし MobileNetV4-S で −13pt なので、他のモデルで PGD-10 が基準を上回る見込みは低いと考えます。BN の影響は、Ferret の EfficientNet-B0 BN 分割版が残っていれば、そちらで分かります。

### B: A に加えて、MobileNetV4-S の BN 分割版を1本回す

- **GPU時間**: 約 12〜16 時間（4090 1枚）。BN 分割版は CUDA Graphs が使えないので、今回の 10.55 時間より遅くなる前提です。eager での正確な速さは測っていないので、幅を広めに取っています。
- **わかること**: 同じモデル・同じ seed で、BN の共有が PGD-10 の低下にどれだけ効いているか。
- **事前ルール（いまここで固定します）**: 指標は、最後の epoch・普通の重み・全取り分け画像の PGD-10。
  - 採用の判定は Phase 2 のルールを基準 30.71 に当てる: 32.21 以上 → 「効果あり」、31.21〜32.21 → 「保留」、31.21 未満 → 「効果なし」。
  - 原因の説明は記述として残すだけにします。BN 分割版が今回の 17.34 より何 pt 戻ったかを書きますが、それを根拠に新しい候補を足すことはしません。
- **リスク**: 攻撃画像の損失が 23% しかない点は変わらないので、full AT の基準を上回る見込みは低いです。Ferret の EfficientNet-B0 BN 分割版と同じ問いに答えることになり、半分は重複します。

### C: キューをそのまま回す

- **GPU時間**: 約 50〜70 時間（A の節約分をそのまま使う）。
- **わかること**: MobileNetV4-M と EfficientNet-B0 で、clean と PGD-10 の交換がどう変わるか。
- **事前ルール**: モデルごとに、最後の epoch・普通の重み・全取り分け画像の PGD-10 を、そのモデルの基準（baseline+EMA が終わっていればそれ、なければ Phase 1 の run）と比べ、同じ 1.5 / 0.5pt のルールを当てます。
- **リスク**: MobileNetV4-S の結果からは、どちらも PGD-10 が大きく下がる見込みが高いです。50〜70 GPU時間を使っても、組み合わせに入る候補が増える可能性は低いと考えます。clean と頑健性の交換の図を論文に載せたい場合には、意味があります。

## おすすめ: A

Phase 2 の目的は、PGD-10 を上げる学習の工夫を探すことです。今回の設定は PGD-10 を 13.37pt 下げており、これはどの誤差の目安よりはるかに大きい差です。
同じ設定を別のモデルで回して（C）、PGD-10 が逆に基準を上回るとは考えにくく、その 50〜70 GPU時間は残りの候補（TRADES、AWP、IDBH、SiLU など）に回せます。
BN の共有が原因かどうか（B）は興味深い問いですが、Ferret の EfficientNet-B0 BN 分割版が同じ問いに答えます。また、攻撃側の重みが 23% しかない以上、BN を分けても full AT の基準を超える見込みは低いです。

**判断が変わる場合**:
- Ferret の EfficientNet-B0 BN 分割版がキューから外れていると分かったら、B（MobileNetV4-S の BN 分割版）を考え直す価値があります。BN の効果を確かめる run が1本も無くなるためです。
- clean と頑健性の交換（たとえば「clean を 9pt 取る代わりに頑健性を 13pt 手放す」）を論文の図として示したい場合は、C を選ぶ理由になります。
- 攻撃側の重みを 0.3 から変える（たとえば 1.0）のは、Kurakin の設定から外れる新しい候補なので、この packet の範囲外です。採るなら plan に新しい項目として書き、別に決めます。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習は始めず、キューも変更しません。
