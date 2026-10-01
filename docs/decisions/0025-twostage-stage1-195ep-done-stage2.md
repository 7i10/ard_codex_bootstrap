---
id: 0025
status: pending
created: 2026-10-01
campaign: plan0103-mnv4s-twostage-stage1-112-pgd1-s256-s0-v1（plan 0103 の2段階学習パイロット、MobileNetV4-S の stage 1。112 px・PGD-1・195 epoch・S=256 の縮小コピーで学習。Hamster GPU1 の hand-run、seed 0、195/195 epoch 完走、source SHA 8333698）
question: 195 epoch の stage 1 が終わった。事前登録どおり、この last.pt から stage 2（224 px・PGD-3・20 epoch）をすぐ始めるか、100/50 epoch 版の stage 1 が終わるのを待って3本の stage 2 をまとめて始めるか。
options:
  A: 今すぐ、この run の last.pt（epoch 194）から stage 2 を1本始める。既存のレビュー済み config（eager、cuda_graph なし）を使い、Hamster GPU1 に相乗りさせる（約9〜13 GPU時間）。事前登録ルールで 195 epoch 版の2段階学習の判定が出る。ルールは stage 2 後の最終 epoch の val PGD-10 を単段の 30.71 と比べ、+1pt 以上なら良い、±1pt 以内なら同等、-1pt 以下なら悪い。
  B: 何もしない。100 epoch 版（Hamster GPU1）と 50 epoch 版（Ferret GPU1）の stage 1 が終わるまで待ち、3本の stage 2 を同じ時期にまとめて始める（stage 2 自体の費用は3本で約27〜40 GPU時間、A と比べて今は0）。判定ルールは A と同じで、3本それぞれに適用する。
recommendation: A
chosen: null
---

# 0025: 2段階学習の stage 1（195 epoch）完了。stage 2 をいつ始めるか

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、
2026-10-01 (postrun, stage 1 195 epochs)。証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。
結果のコミットは `0af5c31`。この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません
（plan 0103/0104 の他の hand-run と同じ扱い）。

条件: ImageNet-1k、MobileNetV4-Conv-S、ランダム初期化、plain PGD-AT（教師なし）、seed 0。
学習データは短辺 256 に縮小したコピー（S=256）、学習時の画像は 112 px。
学習時の攻撃は PGD-1（l_inf 4/255、step 4/255、random start）、評価は PGD-10（4/255、step 8/765）。
SGD lr 0.025、10 epoch warmup、epoch 98 と 148 で ×0.1、195 epoch、world size 1、バッチ 128、決定的モード。
数字はすべて内部検証（S=256 コピーから取り分けた 25,620 枚、112 px で評価）で、公式テストではありません。
AutoAttack はまだ走っていません。

内部検証（%）:

| 時点 | clean | PGD-10 |
|---|---:|---:|
| epoch 97（lr 0.025 の最後） | 36.67 | 15.55 |
| epoch 98（lr 0.0025 の最初） | 43.73 | 19.15 |
| epoch 147（lr 0.0025 の最後） | 46.28 | 19.52 |
| epoch 148（lr 0.00025 の最初） | 48.06 | 20.92 |
| best（epoch 163） | 48.64 | **21.42** |
| last（epoch 194） | 49.04 | 21.11 |

- 学習は正常で、崩壊はありません。catastrophic overfitting のガード（PGD-10 がそれまでの最大値の半分を下回ったら止める）は一度も発動していません。
  最大値からの落ち込みは最大でも 0.9pt です。
- lr 0.025 の区間は epoch 30 ごろから横ばいです（epoch 30〜97 の PGD-10 は 14.67〜15.79%）。
  ただし、この横ばいの epoch が lr を下げた後に効いているかどうかは、この run だけでは分かりません。
  それを確かめるのが 100/50 epoch 版です。
- best と last の差は PGD-10 で 0.31pt です。検証の標準誤差（SE）は約 0.25pt なので、約 1.2 SE です。
- **事前登録ルールの判定はまだ出ません。** ルールは stage 2（224 px）の後の数字を単段の 30.71 と比べるものです。
  上の数字は 112 px・別の検証セットなので、30.71 とは比べられません。

## 現在の GPU の状況

- Hamster GPU0: MobileNetV4-M random lr 0.0125（約42時間の run の途中）。
- Hamster GPU1: stage 1 の 100 epoch 版（cuda_graph）が単独で動いています。この 195 epoch 版が抜けたので空きがあります。
- Ferret GPU0: full AT 30 epoch。Ferret GPU1: stage 1 の 50 epoch 版、その後ろに元画像版 50 epoch が待っています。
- Anteater は 2080 Ti なので、4090 で走った単段の基準とは数字レベルでしか比べられません。

## 選択肢

### A: 今すぐ stage 2 を1本始める（推奨）

- **内容**: この run の `last.pt`（epoch 194）とその sha256 を init にして、
  `configs/scientific/imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml` をそのまま使う。
  元画像、224 px、PGD-3（step 8/765）、20 epoch、lr 0.0025、決定的モード、seed 0。pinned worktree から Hamster GPU1 に相乗りさせる。
- **GPU時間**: MobileNetV4-S の 224 px・50 epoch が 4090 1枚で約21〜23 GPU時間（0020/0021 の見積もり）なので、20 epoch で約 8.5〜9.5 時間。
  GPU1 の 100 epoch 版と相乗りで遅くなる分を見込んで、約9〜13 時間。
- **分かること**: 計算量をそろえた条件で、195 epoch 版の2段階学習が単段 50 epoch（30.71）より良いか、同等か、悪いか。
- **事前登録ルール**: stage 2 の最終 epoch の内部検証 PGD-10 を 30.71 と比べる。+1pt 以上なら良い、±1pt 以内なら同等（実時間と単純さで選ぶ）、-1pt 以下なら悪い。clean も並べて報告する。
- **ノイズの目安**: 224 px の内部検証は約 2.5 万枚なので、1つの数字の SE は PGD-10 で約 0.3pt、2本の差で約 0.4pt です。
  ただし ImageNet で seed による揺れはまだ測っていません（CIFAR では対照同士で 0.16〜1.88pt 動いた例があります）。
  seed 1本ずつなので、±1pt の境界付近の結果は方向だけの判定として扱います。
- **リスク**: cuda_graph なしなので、cuda_graph 版より遅いです。cuda_graph 版の config を新しく作れば速くなりますが、
  init_checkpoint と cuda_graph の組み合わせはまだ試されていないので、その場合はレビューが先に必要です。
  時間差は数時間なので、既存 config を勧めます。

### B: 何もしない（100/50 epoch 版を待ってまとめて始める）

- **内容**: 今は stage 2 を始めない。100 epoch 版と 50 epoch 版の stage 1 が終わってから、3本の stage 2 を近い時期に始める。
- **GPU時間**: 今は 0。後で3本分（約27〜40 GPU時間）。195 epoch 版の判定は、待つ時間の分だけ遅れます。
- **分かること**: A と同じ判定を3本に対して出す。
- **事前登録ルール**: A と同じ。
- **リスク**: 決定的モードなので、始める時期やホストの混み具合で数字は変わりません（同じ GPU 種類なら）。
  まとめて始める科学的な利点は小さく、Hamster GPU1 の空きを使わないまま判定が遅れます。

## 推奨

A を勧めます。stage 2 は事前に設計・レビュー済みで、config もできています。決定的モードなので、3本を同時に始めても後から始めても数字は同じです。
今 Hamster GPU1 に空きがあるので、待つ理由がありません。195 epoch 版の判定が早く出れば、「full AT か2段階か」の判断（2026-10-01 の決定で stage 2 の判定後とされています）にも早く入れます。
推奨が変わるのは、Hamster GPU1 を別の優先度の高い run（たとえば full AT 30 epoch の追加や MobileNetV4-M のグリッド）に使いたい場合です。そのときは B が妥当です。
