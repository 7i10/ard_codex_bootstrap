---
id: 0061
status: decided
created: 2026-10-17
campaign: plan0103-phase2-effb0-baseline-ema-s0-v1（plan 0103 Phase 2。EfficientNet-B0 の基準＋EMA。Phase 1 と同じ学習手順に Phase 2 の共通設定（サブセット 5,000、EMA 0.9999）を足したもの。ランダム初期化、SGD lr 0.025、full AT 50 epoch。Hamster で hand-run、seed 0、50/50 epoch 完走、非決定的＋cudnn_benchmark＋torch.compile、source d97bf6e）
question: EfficientNet-B0 の Phase 2 の基準が 58.11 / 35.87 に決まり、設定チェックも通りました（Phase 1 seed 0 との差 −0.11 / −0.06）。このまま基準を確定してキューを進めるだけにするか、あわせて Ferret にある Phase 1 seed 1（EfficientNet-B0 と DeiT-Tiny）を docs に取り込み、register C で保留していた「判定の基準の見直し」を始めるか。
options:
  A: 追加の作業はしない（0 GPU時間）。基準を 58.11 / 35.87（EMA 58.30 / 36.07）で確定し、キューの EfficientNet-B0 の案（cosine、ラベル平滑化）はこの基準と Phase 2 の判定ルールで読む。
  B: A に加えて、Ferret の Phase 1 seed 1 の2本（EfficientNet-B0、DeiT-Tiny）を postrun で docs に取り込む（0 GPU時間、セッション2回分）。そろったら下の事前ルールで、モデルごとの「効果なし」の線を決める。
recommendation: B
chosen: B
---

# 0061: EfficientNet-B0 基準＋EMA が完了（58.11 / 35.87）。設定チェックは seed の差の範囲内で合格

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-17 (postrun, Phase 2 baseline+EMA, EfficientNet-B0)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `730befb`（`Record plan0103-phase2-effb0-baseline-ema-s0-v1 results`）。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は run-bundle の manifest と `epoch-metrics.jsonl` から読みました）。

この run には役目が3つあります。Phase 2 の設定で Phase 1 と同じ結果が出るかの確認（設定チェック）、EMA の効果、そして EfficientNet-B0 の Phase 2 の案と比べる正式な基準です。

条件: ImageNet-1k（元画像）、EfficientNet-B0、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、weight decay 1e-4）、ピーク LR 0.025、multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）、50 epoch、world size 1、バッチ 128。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| 基準＋EMA、最後（ep 49）、普通の重み | **58.11 / 35.87** |
| 基準＋EMA、最後、EMA の重み（= best EMA） | 58.30 / 36.07 |
| 基準＋EMA、best（ep 48）、普通の重み | 58.14 / 35.96 |
| Phase 1 seed 0（Ferret、`086072b`）、最後 | 58.22 / 35.93 |
| Phase 1 seed 1（Ferret）、最後 | 58.31 / 35.76 |

Phase 1 の seed 0 の clean と seed 1 の2つの数字は、docs には無く、2026-10-19 のゼミ資料の facts sheet（`reports/biweekly/2026-10-19/facts.md`、Ferret の run から直接読んだもの）から取りました。

- **設定チェックは通りました。** Phase 1 seed 0（同じ学習 seed）との差は clean −0.11 / PGD-10 −0.06（約 15 枚、2本の差の SE 約 0.42pt の 0.14 倍）、seed 1 とは −0.20 / +0.11 です。
  PGD-10 はどちらも EfficientNet-B0 の seed の差（0.17pt）より小さく、ルール（register C、2026-10-09）を seed の差の方で満たします。clean は seed の差（0.09pt）より少し大きく、0.5pt の方で満たします。
  seed 0 同士は seed がそろっていて設定も同じ（非決定的＋compile）なので、−0.06 はほぼ「回し直したときの揺れ」に当たります（register C、2026-10-10 で知りたいとしていた値）。ただしホスト（Ferret と Hamster）と source も違うので、そこは分けられません。
- **EMA は最後には効いていません**（+0.19 / +0.20、「効果なし」）。4モデルの中では一番大きいですが、MobileNetV4-S（+0.10 / +0.10）、ConvNeXt-Atto（+0.03 / +0.06）、MobileNetV4-M（+0.03 / +0.12）と同じく、すべてプラスで 0.5pt 未満です。
- 学習の途中では、EMA はウォームアップの間ほぼ 0% のままで、epoch 14〜15 で普通の重みを追い越し、ピーク LR の間は最大 +5.4pt 上回ります。1回目の LR 下げのあとは差が縮み、最後の段階では ±0.3pt 以内です（MobileNetV4-S・M と同じ形）。
- 頑健さの過学習はなく（best は ep 48、最後との差 0.08pt）、学習崩壊の見張りも発動していません。実時間 47.98 時間（Hamster、376〜378 枚/秒。同じホストの Phase 1 lr 0.05 と同じ速さで、共通設定による遅れは見えません）。

**ばらつきの目安（noise floor）**: ImageNet の seed 0 と seed 1 の差（PGD-10）は MobileNetV4-S 0.36、MobileNetV4-M 0.09、ConvNeXt-Atto 0.36、EfficientNet-B0 0.17pt です（各1組）。DeiT-Tiny は facts sheet の値で 1.37pt と、ほかより大きい値です（docs 未取り込み）。CIFAR の画面では乱数だけで 1〜2pt 動くことがありました。
今回の −0.06 はどの目安よりも小さく、「設定の影響なし」以上のことは言えません。

**関係する決定**: register C（2026-10-11）で人間が「判定の基準は、EfficientNet-B0 と DeiT の seed 1 がそろってから見直す」と決めています。2本とも Ferret で完了していますが、docs への取り込みはまだです。
decision 0049 は A（mixed batch を閉じ、EfficientNet-B0 の BN 分割版はキューから外す）が選ばれているので、この基準で読む EfficientNet-B0 の案は cosine（Hamster のキュー）とラベル平滑化 0.1（Ferret のキュー）です（キューの状態は facts sheet の 2026-10-16 時点。この postrun では読み直していません）。
cosine は decision 0052 / 0053 の C（4モデルで判定）とつながっていて、どちらも `chosen` が null のままです。この packet はそれを決めるものではありません。

## 選択肢

### A: 追加の作業はしない

- **GPU時間**: 0。
- **わかること**: 新しくわかることはありません。EfficientNet-B0 の Phase 2 の基準を 58.11 / 35.87（EMA 58.30 / 36.07）に確定します。
- **事前ルール**: キューの EfficientNet-B0 の案は、最後の普通の重み・取り分けた画像すべての PGD-10 を、この基準（35.87）と比べます。Phase 2 の判定ルールのまま：+1.5pt 以上 →「効果あり」、0.5〜1.5pt →「保留」、0.5pt 未満 →「効果なし」。新しいしきい値は作りません。
- **リスク**: register C の「基準の見直し」が始められないまま残ります。DeiT-Tiny のラベル平滑化が Ferret のキューにあり、DeiT の seed の差（facts sheet で 1.37pt）は「保留」の帯に入る大きさなので、今のルールのまま読むと、seed の揺れだけで「保留」が出るおそれがあります。

### B: A に加えて、Phase 1 seed 1 の2本を取り込み、基準の見直しを始める（おすすめ）

- **GPU時間**: 0。必要なのは postrun のセッション2回分です（`plan0103-phase1-efficientnet-b0-random-s1-v1` と DeiT-Tiny の seed 1、どちらも Ferret）。
- **やり方**: 2本それぞれに `/experiment-postrun` をかけ、Progress log と証拠台帳に1行ずつ記録します。そのあと、下のルールの結果を register C に1行で記録します。
- **わかること**: 5モデルすべての seed の差が docs にそろい、モデルごとに「この差なら揺れの範囲」という線が引けます。
- **事前ルール**（いまここで固定します）: 5モデルそれぞれの seed 0 と seed 1 の差（最後の普通の重み、取り分けた画像すべての PGD-10）を見ます。
  - その差が 0.5pt 以下のモデル → Phase 2 の判定ルール（0.5 / 1.5pt）をそのまま使う。
  - その差が 0.5pt を超えるモデル → そのモデルだけ、「効果なし」の線をその差まで上げる（例: DeiT が 1.37 なら、1.37pt 未満は「効果なし」）。「効果あり」の 1.5pt の線はそのまま。もし差が 1.5pt 以上なら、そのモデルの案は seed 1本では判定せず、改めて packet で決める。
  - 2本の平均やばらつきは出しません。差は1組の値として扱います。
- **リスク**: Ferret の run ファイルを読むには、このセッションの許可にない ssh コマンドが要るので、その都度承認が必要です。DeiT の 1.37 が取り込み後に変わる可能性もあります（facts sheet の値は手で読んだもの）。どちらの場合も GPU は使いません。

## おすすめ: B

A は B の一部なので、B を選んでも失うものはありません。B は GPU を使わず、人間がすでに決めていた「seed 1 がそろってから見直す」をそのまま実行するものです。
特に DeiT-Tiny のラベル平滑化がキューにあるので、その結果が出る前に DeiT の線を決めておくと、結果を見てから線を引くことを避けられます。
EfficientNet-B0 自体については、設定チェックが seed の差の範囲で通り、EMA も効いていないので、追加で GPU を使う理由はありません。

**判断が変わる場合**:
- Ferret の空きがなく、run ファイルを読むのも当面難しい場合 → A を選び、B は DeiT の結果が出る前のどこかで行う。
- DeiT-Tiny の Phase 2 の案を回さないことにした場合 → 見直しを急ぐ理由が弱くなり、A で十分です。

`chosen` は人間が記入します（ファイルを編集するか、チャットで答えてください）。記入されるまで、この packet に関わる新しい学習は始めません。

## 決定（2026-10-18、本人、チャットで回答）

**B を選択（人間、2026-10-18）。** EfficientNet-B0 の Phase 2 基準を 58.11 / 35.87 で確定。あわせて Phase 1 の seed 1（EfficientNet-B0 と DeiT-Tiny）を記録に取り込み、5モデルの seed による揺れから判定の基準（1.5 / 0.5pt）を見直す（CPU だけの作業）。
