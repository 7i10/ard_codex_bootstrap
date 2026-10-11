---
id: 0044
status: pending
created: 2026-10-11
campaign: plan0103-phase2-mnv4s-silu-s0-v1（plan 0103 Phase 2 の「ReLU をすべて SiLU に置き換える」。MobileNetV4-S・ランダム初期化・SGD lr 0.025・full AT 50 epoch。基準＋EMA との違いは活性化関数だけ（パラメータ数と計算量は同じ）。Hamster GPU1、hand-run、seed 0、50/50 epoch 完走、source SHA 5dd329e、CUDA Graphs なし）
question: SiLU は PGD-10 を +2.56pt、clean を +1.60pt 上げ、事前ルールの「効果あり」に入りました。SiLU を MobileNetV4-S の Phase 2 の標準にするか、その前に何を確かめるか。
options:
  A: 0042 の B（5dd329e で基準＋EMA を 2 epoch 回し、Phase 1 の基準とビット単位で一致するか見る。約 0.6〜1.0 GPU時間）を先に回し、一致したら SiLU を MobileNetV4-S の Phase 2 の標準にする。以後の MobileNetV4-S の比較相手は SiLU の 55.45 / 33.27。
  B: A に加えて、キューに入っている SE（ReLU 版、plan0103-phase2-mnv4s-se-s0-v1）を SiLU＋SE（既にある imagenet_mobilenetv4_silu_se_pgd_at_phase2_random_lr0025.yaml）に差し替える（追加は差し替え分だけ、1本 約 20〜24 GPU時間）。
  C: A に加えて、SiLU の seed 1 を1本回してから標準にする（約 19〜20.5 GPU時間）。
  D: 記録だけで閉じ、標準は ReLU のまま（追加 0 GPU時間）。SiLU は「効果あり」として記録するが、元の MobileNetV4 が ReLU を選んだ理由（携帯端末の推論での速さ）を優先する。
recommendation: A
chosen: null
---

# 0044: Phase 2 の SiLU、MobileNetV4-S seed 0 が完了（PGD-10 は +2.56pt）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-11 (postrun, Phase 2 SiLU)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `a75700a`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は `epoch-metrics.jsonl` と run-bundle の manifest から読みました）。

この run は、基準＋EMA（`plan0103-phase2-mnv4s-baseline-ema-s0-v1`、53.85 / 30.71）と同じ学習で、違いは活性化関数だけです。
ReLU をすべて SiLU に置き換えました。パラメータ数と計算量は変わりません。
活性化 × weight decay の 2×2（discussion register B、2026-10-08）の「SiLU＋全部にかける」のマスです。

条件: ImageNet-1k（元画像）、MobileNetV4-S、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、momentum 0.9、weight decay 1e-4 を全パラメータに）、ピーク LR 0.025、multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）、50 epoch、world size 1、バッチ 128、決定的モード。SiLU 版は CUDA Graphs の対象外なので、普通に（eager で）回しました。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| 最後（ep 49）、普通の重み | **55.45 / 33.27** |
| 最後（ep 49）、EMA の重み | 55.60 / 33.53 |
| best（ep 47、サブセットで選択）、普通の重み | 55.30 / 33.40 |
| best EMA（= 最後の EMA、ep 49） | 55.60 / 33.53 |
| 基準＋EMA（ReLU）、最後、普通の重み | 53.85 / 30.71 |

- **基準より PGD-10 が 2.56pt、clean が 1.60pt 高い。** 2本の差の誤差（SE）は PGD-10 で約 0.37pt なので約7倍、MobileNetV4-S の seed 0 と seed 1 の差（0.36pt）の約7倍です。clean も約 3.7 SE 上がっています。clean と頑健さの引き換えはありません。
  取り分けた画像の残り 20,620 枚でも +1.49 / +2.49、EMA の重み同士でも +1.65 / +2.73 で、同じ向きです。
- **事前ルール**（discussion register C、2026-10-08。1.5pt 以上は「効果あり」で seed 1本でよい、0.5〜1.5pt は「保留」、0.5pt 未満は「効果なし」）では、**「効果あり」**です。
  MobileNetV4-S の Phase 2 でいちばん大きい改善です（cosine +0.78、ラベル平滑化 +0.45、weight decay の除外 −1.14、混ぜる方式 −13.37）。
- **差は最初の epoch からあり、学習率のどの段階でも続きます。** サブセットの PGD-10 は epoch 9 で +1.56、epoch 24 で +2.26、epoch 49 で +2.82 です。weight decay の除外のように、学習率を下げたところで急に開く差ではありません。
- **学習データへの当てはまりが良くなり、それが見ていない画像にもそのまま出ています。** 学習損失は 3.608（基準 3.697）、学習バッチの頑健精度は 29.53%（基準 27.90%）。
  probe と取り分けた画像の差は PGD-10 で +4.98（基準 +3.69）と少し広がりましたが、誤差の内側です。「取り分けた画像の精度 ≥ 学習バッチの精度」（学習が足りていない側のしるし）は続いています。
  plan 0103 の Phase 2 の方針でいう「容量・アーキテクチャを変える」側の候補で、それが数字を動かしました。
- EMA はウォームアップ中に遅れます（epoch 9 で 7.60 / 4.06）。基準と同じです。同じ source（`5dd329e`）で weight decay を全部にかけたこの run でも遅れるので、weight decay の除外の run で遅れが消えたのは、source ではなく weight decay の変更のせいである可能性が高くなりました（直接は確かめていません。活性化も違います）。
- 学習崩壊の見張りは一度も発動していません（サブセットの最大の落ち込み 1.02pt）。
- 実時間は 19.34 時間で、基準＋EMA（15.52 時間）の 1.25 倍です。928〜943 枚/秒。遅くなった分には、SiLU の計算と CUDA Graphs が使えないことの両方が混ざっていて、分けて測っていません。

**source と実行方法の違い（注意）**: この run は `5dd329e`・eager、基準＋EMA は `d836c03`・CUDA Graphs です。
`5dd329e` が普通の full AT を `d836c03` と同じに学習するかは、まだ確かめていません（decision 0042 の B、未決定）。
もし違うなら、その差がこの +2.56pt に直接混ざります。ただし、判定を「効果あり」から外すには 1pt を超える差が必要です。

**ばらつきの目安（noise floor）**: ImageNet では、seed 0 と seed 1 の差が MobileNetV4-S 0.36pt、MobileNetV4-M 0.09pt、ConvNeXt-Atto 0.36pt でした（各1組）。CIFAR の画面では seed だけで 1〜2pt 動くことがありました。
+2.56pt は、CIFAR の目安の上限（2pt）も越えています。

**関係する未決定の packet**:
- 0043（SiLU＋除外のマスを回すか）は、「SiLU が大きく下がったら A に変える」という条件でした。SiLU は上がったので、0043 のおすすめ（B）はそのままです。この packet では選択肢に入れていません。
- 0042 の B は、この packet の A にも入っています。どちらで選んでも、回すのは1回です。

**インフラの気づき（科学の判断ではありません）**: GPU1 の worker は 09:09:21 UTC に `QUEUE DONE` を書いて止まりました。一方、`queues/work_queue.txt` にはまだ 11 行（MobileNetV4-S の TRADES と SE、ConvNeXt-Atto、MobileNetV4-M、EfficientNet-B0）が残っています。GPU1 はいま空いています。理由は調べていません。

## 選択肢

### A: 0042 の B で source を確かめてから、SiLU を標準にする（おすすめ）

- **GPU時間**: 約 0.6〜1.0 時間（4090 1枚、0042 の見積もりのまま）。
- **やり方**: `5dd329e` の worktree から、基準＋EMA の設定（ReLU、CUDA Graphs）を 2 epoch だけ回し、Phase 1 の基準の epoch 0〜1 と比べます。
- **わかること**: `5dd329e` で普通の full AT が変わっていないか。一致すれば、この +2.56pt に source の差は入っていません（eager と CUDA Graphs の違いは残りますが、CUDA Graphs は ReLU 版で学習1ステップのビット一致テストを通っています）。
- **事前ルール**: 2 epoch の学習損失・学習バッチの頑健精度・サブセットの clean / PGD-10 が、Phase 1 の基準と最後の桁まで一致 → SiLU を MobileNetV4-S の Phase 2 の標準にし、以後の MobileNetV4-S の比較相手を 55.45 / 33.27 にする。一致しない → 標準にはせず、差の原因を別の packet で調べる（weight decay の除外の −1.14pt の読み方も変わる）。
- **リスク**: SE やその他、キューに入っている MobileNetV4-S の Phase 2 の run は ReLU のままです。それらは ReLU の基準（30.71）と比べることになり、SiLU との組み合わせの効果はわかりません。携帯端末での推論の速さ（SiLU は ReLU より遅い機器がある）は測っていません。

### B: A ＋ キューの SE を SiLU＋SE に差し替える

- **GPU時間**: A の分に加えて、差し替える1本が 約 20〜24 時間（SiLU の実測 19.34 時間に、SE の分を少し上乗せした見積もり。SE 版の実測はまだありません）。ReLU 版の SE を回さない分、キュー全体の時間はほぼ同じです。
- **わかること**: SiLU の上に SE を足すと、さらに上がるか。
- **事前ルール**: 比べる相手は SiLU（33.27、同じ `5dd329e`）。最後の普通の重み、取り分けた画像すべての PGD-10 で、Phase 2 のルール（1.5pt 以上「効果あり」、0.5〜1.5pt「保留」、0.5pt 未満「効果なし」）。
- **リスク**: ReLU＋SE の結果が無くなるので、「SE だけの効果」はわかりません。SE と SiLU を同時に入れた EfficientNet 型の設計に寄せる判断になります。

### C: A ＋ SiLU の seed 1

- **GPU時間**: A の分に加えて 約 19〜20.5 時間（SiLU の seed 0 の実測から）。
- **わかること**: +2.56pt が seed による揺れでないか。
- **事前ルール**: seed 1 の最後の普通の重みの PGD-10 を、基準 30.71 と比べる。+1.5pt 以上 → 2本とも「効果あり」、標準にする。+0.5〜1.5pt → 標準にするかは保留、別の packet。+0.5pt 未満 → 標準にしない。
- **リスク**: ルールの文面では、1.5pt 以上は seed 1本でよいことになっています。結果を変えるには seed の揺れが約 1pt 必要で、これまでの ImageNet の seed の差（0.09〜0.36pt）の3倍以上です。19 時間以上を、ほぼ変わらない判断の補強に使うことになります。

### D: 記録だけで閉じ、標準は ReLU のまま

- **GPU時間**: 0。
- **わかること**: 新しいことはありません。
- **事前ルール**: このルールは「効果あり」の候補を採用しない場合を想定していません。採用しない理由（推論の速さ、元の MobileNetV4 との比較のしやすさ）を、この packet の判断として記録します。
- **リスク**: Phase 2 でいちばん大きい改善を使わないことになります。今回の研究の主題（10M パラメータ未満のモデルの敵対的学習）では、どのアーキテクチャの工夫が頑健さに効くかが結果そのものなので、SiLU の効果は論文の材料になります。

## おすすめ: A

事前ルールでは、+2.56pt は seed 1本で「効果あり」です。結果を揺るがしうる残りの問題は source の違いだけで、それは 1 時間足らずの 0042 の B で確かめられます。
C の seed 1 は、ルールの上では必要がなく、判断を変えるには今までに見た seed の揺れの3倍以上が必要です。
B は魅力がありますが、SE を ReLU で回すか SiLU で回すかは「SE だけの効果を知りたいか」という別の問いなので、A を決めてから選んでも遅くありません（キューで SE の番が来る前に決めれば十分です）。

**判断が変わる場合**:
- 0042 の B で一致しなかった場合 → SiLU を標準にするのは止め、原因を調べる packet を開きます。
- 携帯端末での推論の速さを研究の条件にする場合 → D、または SiLU を「ReLU 版と並べて報告する変種」として扱います。
- 学会の締め切りなどで seed 2本の結果が必要な場合 → C。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習は始めません。
