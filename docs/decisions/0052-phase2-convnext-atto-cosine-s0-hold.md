---
id: 0052
status: pending
created: 2026-10-14
campaign: plan0103-phase2-convnext-atto-cosine-s0-v1（plan 0103 Phase 2 の候補①「cosine」。ConvNeXt-Atto・ランダム初期化・AdamW lr 1e-3・full AT 50 epoch。Hamster GPU0 の pull キューから hand-run、seed 0、50/50 epoch 完走、source SHA 5dd329e、非決定的＋cudnn_benchmark）
question: ConvNeXt-Atto の cosine は Phase 1 の multistep より PGD-10 が +0.88pt で、事前ルールでは「保留」です。すぐ seed 1 を回すか、先に ConvNeXt-Atto の基準＋EMA（同じ Phase 2 の比較相手）を待つか、他モデルの cosine とまとめて決めるか。
options:
  A: ConvNeXt-Atto cosine の seed 1 を1本回す（約 25〜27 GPU時間、Hamster の 4090）。Phase 1 の lr 1e-3 seed 1 と比べ、seed 1 でも +0.5pt 以上かを見る。
  B: いまは何も足さない（追加 0 GPU時間）。Ferret の ConvNeXt-Atto 基準＋EMA が終わったら、設定の影響を確かめ、cosine をその普通の重みと比べ直す。それでも「保留」なら A をもう一度検討する。
  C: いまは何も足さず、キューにある MobileNetV4-M と EfficientNet-B0 の cosine を待って、decision 0038 と合わせて4モデルで判定する（追加 0 GPU時間）。事前ルールの「同じモデルの2本目の seed」とは違う確かめ方になる。
recommendation: B
chosen: null
---

# 0052: Phase 2 の cosine、ConvNeXt-Atto seed 0 が完了（判定は「保留」）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-14 (postrun, Phase 2 cosine ConvNeXt-Atto)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `502d54c`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は run-bundle の manifest と `epoch-metrics.jsonl` から読みました）。

比べる相手は、Phase 1 の ConvNeXt-Atto AdamW lr 1e-3 seed 0（学習率の探索で一番よかった点、PGD-10 37.02）です。
違いは次の3つだけです。

- 学習率のスケジュール: 10 epoch のウォームアップの後、cosine で 0 まで下げる（Phase 1 は epoch 25 と 38 で 1/10）。
- Phase 2 の共通設定: 毎 epoch の評価を 5,000 枚の固定サブセットで行う。EMA（0.9999）を並行して保持する。
- W&B のグループと source SHA。

条件: ImageNet-1k（元画像）、ConvNeXt-Atto（V1）、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
AdamW（weight decay 0.05）、ピーク LR 1e-3、50 epoch、world size 1、バッチ 128。
実行は Phase 1 と同じ非決定的＋cudnn_benchmark です（ConvNeXt-Atto で一番速い設定、2026-10-10 の決定）。
数字はすべて内部検証（学習データから取り分けた 25,620 枚。Phase 1 と同じ画像）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| cosine、最後（ep 49）、普通の重み | **60.70 / 37.90** |
| cosine、最後（ep 49）、EMA の重み | 60.66 / 37.89 |
| cosine、best（ep 47、サブセットで選択）、普通の重み | 60.36 / 37.74 |
| Phase 1 multistep、seed 0、最後、普通の重み | — / 37.02 |

- **差は PGD-10 で +0.88pt です**（最後の普通の重みどうし）。25,620 枚のうち約 224 枚の差です。
  2本の差の誤差（SE）は約 0.43pt なので、約 2 倍です。
  Phase 1 の clean の値はこのリポジトリの文書に残っておらず、bundle も Hamster に無いので、clean の差は読めません。
- **事前ルール（2026-10-08）**: 1.5pt 以上は「効果あり」、0.5〜1.5pt は「保留（2本目の seed で確かめる）」、0.5pt 未満は「効果なし」。
  +0.88pt は **「保留」** です。
  Phase 1 の seed 1 と比べても、seed の差 0.36pt の向きによって +0.52 か +1.24 で、どちらでも「保留」です。
- **MobileNetV4-S と同じ向き・同じ段階です。** MobileNetV4-S の cosine は +0.78pt で「保留」でした（decision 0038、`chosen` はまだ空）。
  2モデルとも seed 1本ずつで、どちらもプラスの「保留」です。
- **この比較の一番大きい弱点は、Phase をまたいでいることです。**
  両方とも非決定的モードなので、同じ設定で回し直しただけでどれだけ動くかは、ConvNeXt-Atto ではまだ測っていません。
  その答えになる ConvNeXt-Atto の基準＋EMA（Phase 2 の比較相手と、設定の影響の確認を兼ねる run）は、Ferret でまだ終わっていません（plan 0106 の移し替えの対象。今の進み具合はこの postrun では見ていません）。
  Phase 1 の seed 0 の run の source SHA とホストも、文書には残っていません。
- **最後まで上がり続けています。** サブセットの PGD-10 は、cosine の前半（epoch 9→30）で +8.74、後半（epoch 30→49）で +8.48 上がり、最後の 10 epoch でも +3.36 です。頭打ちは見えません。
- **EMA のウォームアップ中の遅れがありません。** EMA の重みは epoch 8 で普通の重みを追い越し、学習率が高い間はサブセットで最大 +6.80pt（PGD-10）上回りました。
  MobileNetV4-S（SGD）では、ウォームアップの終わりに EMA がずっと低いままでした。ConvNeXt-Atto は LayerNorm（BN の統計が無い）で AdamW なので、どちらが効いているかは確かめていません。
  最後は EMA と普通の重みの差はほぼ 0（−0.04 / −0.01）です。
- 頑健性の過学習はありません（best と last の差は −0.15pt）。崩壊の見張りは一度も作動していません（最大の落ち込みは 0.48pt）。
- 実時間は 25.23 時間、713〜716 枚/秒でした（Hamster GPU0、隣の GPU は使用中）。

**ばらつきの目安（noise floor）**: ImageNet では seed 0 と seed 1 の差が MobileNetV4-S 0.36pt、MobileNetV4-M 0.09pt、ConvNeXt-Atto 0.36pt でした（各1組）。2本の全体評価の差の SE は約 0.43pt です。CIFAR の画面では seed だけで 1〜2pt 動くことがありました。
+0.88 は ConvNeXt-Atto の seed の差の約 2.4 倍ですが、CIFAR の目安には届かず、1本だけで「効果あり」とは言えません。また、seed の差 0.36pt は1組だけの値で、Phase をまたぐ影響（設定の違い）は含んでいません。

**関係する未決定の packet**: 0038（MobileNetV4-S の cosine、「保留」、おすすめは seed 1）。
どの選択肢も 0038 とは矛盾しません。register B（2026-10-13）の「手法は学習不足を補う方向で組み立てる」とも矛盾しません（cosine は最適化を助ける側の変更です）。
cosine を Phase 2 全体の標準スケジュールにするかは、この packet の範囲外です。スケジュールの変更なので、決めるときは plan に新しい契約として書きます。

## 選択肢

### A: ConvNeXt-Atto cosine の seed 1 を回す

- **GPU時間**: 約 25〜27 時間（Hamster の 4090 1枚）。根拠は今回の実測 25.23 時間（713〜716 枚/秒、非決定的）です。Ferret だと混み具合で2倍近くかかることがあります。
- **やり方**: `imagenet_convnext_atto_pgd_at_phase2_cosine.yaml` を `5dd329e` の worktree から seed 1 で回します。比べる相手は、既にある Phase 1 の ConvNeXt-Atto lr 1e-3 seed 1（Ferret で実行済み）です。その数字は run の前に Ferret の bundle から読んで plan に書き写します。
- **わかること**: seed 1 どうしの差 d1（cosine seed 1 − Phase 1 seed 1）。seed 0 の d0 = +0.88 と合わせて、2本での向きがわかります。
- **事前ルール（run の前に固定します）**: 指標は、最後の epoch・普通の重み・取り分けた画像すべての PGD-10。
  - d0 と d1 がどちらも +0.5pt 以上 → ConvNeXt-Atto で cosine は「効果あり」（この2本の seed での向き）。
  - d1 が 0.5pt 未満 → 「効果なし」。multistep のまま。
  - どちらの場合も、2本の平均を「分布の推定」としては扱いません。
- **リスク**: 比べる相手はまた Phase 1 の run なので、Phase をまたぐ弱点がそのまま残ります。d1 が +0.5 前後に出ると、設定の違いと区別できません。Hamster の次の項目が約 1 日遅れます。

### B: いまは何も足さず、ConvNeXt-Atto の基準＋EMA を待つ

- **GPU時間**: 追加 0（基準＋EMA はすでに Ferret のキューにあり、どのみち回ります）。
- **わかること**: (1) Phase 1 と Phase 2 の設定の違い（非決定性を含む）が ConvNeXt-Atto でどれだけ効くか。(2) 同じ Phase 2 の中での cosine と multistep の差。
- **事前ルール（ここで決めて、基準＋EMA が終わる前に固定します）**: 指標は、最後の epoch・普通の重み・取り分けた画像すべての PGD-10。
  - 設定の確認（2026-10-09 / 10-10 の決定と同じ形）: |基準＋EMA − 37.02| が 0.5pt 以内なら「設定の影響なし」。
  - cosine − 基準＋EMA が 0.5pt 未満 → 「効果なし」（+0.88 は Phase をまたいだ揺れの範囲だったと読む）。
  - 0.5pt 以上 → 「保留」のまま。A を、比べる相手を基準＋EMA に置き換えた形でもう一度検討する。
  - 設定の確認が 0.5pt を超えた場合は、Phase 1 との比較（+0.88）は使わず、基準＋EMA との差だけで上の2行を当てます。
- **リスク**: 結論が出るのは基準＋EMA の終了後で、Ferret の混み具合によっては数日かかります。差が 0.5pt 以上残った場合は、結局 A の約 1 日が必要になります。

### C: 他の2モデルの cosine を待って、4モデルでまとめて判定する

- **GPU時間**: 追加 0（MobileNetV4-M と EfficientNet-B0 の cosine は Hamster の `work_queue.txt` に残っています。それぞれの基準＋EMA の後ろです）。
- **わかること**: cosine の効果がモデルをまたいで同じ向きに出るか。
- **事前ルール**: decision 0038 の C と同じ。4モデル（MobileNetV4-S、ConvNeXt-Atto、MobileNetV4-M、EfficientNet-B0）のうち3つ以上で、cosine − 基準（最後・普通の重み・PGD-10）が +0.5pt 以上なら「効果あり」。2つ以下なら「効果なし」。今のところ 2 / 2 です。
- **リスク**: 2026-10-08 のルールは「同じモデルの2本目の seed」です。C はこれと違う確かめ方なので、採る場合はルールを変えることになり、0038 と一緒に決める必要があります。MobileNetV4-M と EfficientNet-B0 は遅いモデル（Phase 1 の EfficientNet-B0 は約 47 時間）で、基準＋EMA が先に入っているので、結果は数日先です。

## おすすめ: B

ConvNeXt-Atto の +0.88 の一番大きい不確かさは、seed よりも「Phase 1 と Phase 2 をまたいでいること」です。両方とも非決定的モードで、回し直しただけの揺れも測っていません。
B は、どのみち回る基準＋EMA でその不確かさを直接測り、同じ Phase 2 の中で cosine を比べ直せます。追加の GPU 時間は 0 です。
A を先に回すと、比べる相手がまた Phase 1 なので、d1 が境界の近くに出たときに同じ弱点で読めなくなります。
MobileNetV4-S（0038）で A をすすめたのは、そちらでは設定の確認（基準＋EMA が Phase 1 と1ビットまで一致）が済んでいたからで、ConvNeXt-Atto はまだその段階に来ていません。

**判断が変わる場合**:
- Ferret の基準＋EMA がしばらく始まらない（たとえば 3 日以上先）と分かったら、A を Hamster で先に回す方が早く結論が出ます。
- 0038 で C（モデルをまたいで判定）が選ばれたら、この packet も C にそろえるのが筋です。
- cosine を「最終レシピの候補」としてすぐ固めたい（Phase 2 の他の変更と組み合わせて回したい）なら、A で seed を足す方が直接的です。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習は始めません。
