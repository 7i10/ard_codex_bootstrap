---
id: 0026
status: pending
created: 2026-10-01
campaign: plan0103-mnv4s-twostage-stage1-112-pgd1-s256-100ep-cg-s0-v1（plan 0103 の stage 1 の長さチェック。MobileNetV4-S、112 px・PGD-1・100 epoch・S=256 の縮小コピー・cuda_graph。Hamster GPU1 の hand-run、seed 0、100/100 epoch 完走、source SHA 45befac）
question: stage 1 の 100 epoch 版も終わり、Hamster GPU1 が空いた。195 epoch 版と 100 epoch 版の stage 2 を今2本まとめて始めるか、50 epoch 版を待って3本そろえるか、先に cuda_graph 対応の stage 2 config を作るか。
options:
  A: 今すぐ、195 epoch 版と 100 epoch 版それぞれの last.pt から stage 2（既存のレビュー済み config、eager）を2本、Hamster GPU1 に相乗りで始める（合計約17〜19 GPU時間、実時間で約12〜19時間）。50 epoch 版の stage 2 は、その stage 1 の postrun の後に別の packet で決める。各 run に既存の事前登録ルール（単段 30.71 との比較）を当て、さらに長さのルール（100 epoch 版が 195 epoch 版の -1pt 以内なら短い方で十分）を当てる。
  B: 何もしない。Ferret GPU1 の 50 epoch 版の stage 1 が終わるのを待ち（推定で数時間）、3本の stage 2 をまとめて始める（後で合計約26〜29 GPU時間）。ルールは A と同じで、50 epoch 版にも長さのルールを当てる。
  C: 先に cuda_graph と init_checkpoint を組み合わせた stage 2 config を作ってレビューとテストを通し、それから stage 2 を始める（開発とレビューに半日〜1日。stage 2 は1本あたり約9時間から約3.5〜4時間に短くなる見込み）。ルールは A と同じ。
recommendation: A
chosen: null
---

# 0026: stage 1 の 100 epoch 版も完了。stage 2 をどう始めるか

この packet は 0025（195 epoch 版の stage 2 をいつ始めるか、未決）と重なっています。
0026 で A か B を選べば、0025 の問いも同時に決まります（0025 の A は 0026 の A に含まれ、0025 の B は 0026 の B と同じです）。
0026 を決めるときに、0025 も superseded にしてください。

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、
2026-10-01 (postrun, stage-1 length check, 100 epochs)。証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。
結果のコミットは `8104b6a`。この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません。

条件: 195 epoch 版と同じです（ImageNet-1k、MobileNetV4-Conv-S、ランダム初期化、plain PGD-AT、seed 0、
S=256 の縮小コピー、112 px、学習時 PGD-1、評価 PGD-10 l_inf 4/255、SGD lr 0.025、world size 1、バッチ 128、決定的モード）。
違いは3つだけです。epoch 数が 100、lr を下げる epoch が [50, 76]（195 epoch 版と同じ 50% / 76% の位置）、cuda_graph がオン。
数字はすべて内部検証（S=256 コピーから取り分けた 25,620 枚、112 px）で、公式テストではありません。AutoAttack は走っていません。

内部検証（%、clean / PGD-10）:

| 時点 | 100 epoch 版 | 195 epoch 版 |
|---|---:|---:|
| lr 0.025 の最後（ep 49 / 97） | 36.13 / 15.11 | 36.67 / 15.55 |
| lr 0.0025 の最後（ep 75 / 147） | 45.96 / 19.51 | 46.28 / 19.52 |
| best（ep 94 / 163） | 48.38 / 21.00 | 48.64 / 21.42 |
| last（ep 99 / 194） | 48.73 / **20.99** | 49.04 / **21.11** |

- **stage 1 の終わりでは、epoch を半分にしても PGD-10 は見分けられるほど下がっていません。**
  last 同士で clean -0.31pt、PGD-10 -0.12pt です。1つの数字の標準誤差（SE）は clean 約 0.31pt、PGD-10 約 0.25pt で、
  差の SE は多くても約 0.44 / 0.36pt です（同じ検証画像なので、実際はもっと小さい可能性があります）。
  PGD-10 の差は SE の半分以下です。seed は1本ずつです。
- best 同士だと -0.26 / -0.42pt ですが、195 epoch 版の best はより多くの epoch の中から同じ検証セットで選んだ最大値なので、上に偏ります。
  比べるなら last 同士です。
- **epoch 0〜49 は 195 epoch 版と完全に同じでした。** 記録された指標が小数の最後の桁まで一致しています。
  つまり cuda_graph は 50 epoch にわたって eager と同じ学習をしており、2本の違いは epoch 50 以降の lr スケジュールだけです。
- 学習は正常で、崩壊はありません。best と last の差は 0.01pt です。
- **長さの問いの答えはまだ出ません。** 事前に決めたとおり、長さは stage 2（224 px）の後の数字で比べます。
  stage 1 の終わりで差が小さくても、stage 2 の後で差が開く可能性はあります。

## 現在の GPU の状況

- Hamster GPU0: MobileNetV4-M random lr 0.0125（約42時間の run の途中）。
- Hamster GPU1: **空き**（195 epoch 版も 100 epoch 版も終わりました）。
- Ferret GPU0: full AT 30 epoch。Ferret GPU1: stage 1 の 50 epoch 版（S=256）、その後ろに元画像版 50 epoch が待っています。
- 50 epoch 版の終わりの見込み: 100 epoch 版は単独で 1 epoch 約4分（学習約190秒＋検証）だったので、50 epoch で約3.5〜4時間です。
  Ferret 上の実際の進み具合は確認していません。

## 費用の前提

stage 2 は既存の `configs/scientific/imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml`
（元画像、224 px、PGD-3 step 8/765、20 epoch、lr 0.0025、決定的、eager、seed 0）です。
MobileNetV4-S の 224 px・50 epoch が 4090 1枚で約21〜23 GPU時間（0020/0021 の見積もり）なので、
20 epoch で1本約 8.5〜9.5 GPU時間としました。2本を同じ GPU に相乗りさせると、合計の進みは単独の約1.0〜1.4倍と仮定しています
（112 px では相乗りで 1.7 倍程度になりましたが、224 px は GPU の計算が重いので伸びは小さいと見ています）。

## 選択肢

### A: 195 epoch 版と 100 epoch 版の stage 2 を今2本始める（推奨）

- **内容**: 2本とも既存の stage 2 config を使い、init にはそれぞれの `last.pt`（epoch 194 と epoch 99）とその sha256 を使う。
  pinned worktree から Hamster GPU1 に2本相乗りで流す。50 epoch 版の stage 2 は、その stage 1 の postrun の後に別の packet で決める。
- **GPU時間**: 合計約17〜19 GPU時間。相乗りなので実時間は約12〜19時間。
- **分かること**: (1) 195 epoch 版と 100 epoch 版それぞれについて、計算量をそろえていない条件での2段階学習が単段 50 epoch（30.71）と比べてどうか。
  (2) stage 1 を半分にしても stage 2 の後で損がないか。
- **事前登録ルール**:
  - 単段との比較（既存のルール）: stage 2 の最終 epoch の内部検証 PGD-10 を 30.71 と比べる。+1pt 以上なら良い、±1pt 以内なら同等、-1pt 以下なら悪い。clean も並べて報告する。
  - 長さのルール（ここで新しく登録）: stage 2 の最終 epoch の内部検証 PGD-10 で、100 epoch 版 − 195 epoch 版が -1pt より上なら「100 epoch で十分」とし、計算量の少ない方を採る。
    -1pt 以下なら「長い stage 1 に意味がある」とする。clean も並べて報告し、clean が 1pt 以上下がる場合はそれも書く。
- **ノイズの目安**: 224 px の内部検証は約2.5万枚なので、1つの数字の SE は PGD-10 で約 0.3pt、2本の差で多くても約 0.4pt です。
  ただし ImageNet での seed による揺れはまだ測っていません（CIFAR では対照同士で 0.16〜1.88pt 動いた例があります）。
  ±1pt の境界付近の結果は、seed 1本ずつの方向だけの判定として扱います。
- **リスク**: 相乗りで2本とも遅くなります。eager なので、cuda_graph 版の config ができれば速くなったはずの時間を使います。
  stage 2 の config はレビュー済みですが、実際に走らせるのは初めてなので、最初の数 epoch は様子を見る必要があります。

### B: 何もしない（50 epoch 版を待って3本まとめて始める）

- **内容**: 今は stage 2 を始めない。Ferret GPU1 の 50 epoch 版の stage 1 が終わってから、3本の stage 2 を近い時期に始める。
- **GPU時間**: 今は 0。後で3本分、約26〜29 GPU時間。
- **分かること**: A と同じことを、50 epoch 版も含めて3本で。
- **事前登録ルール**: A と同じ。長さのルールは 50 epoch 版 − 195 epoch 版にも当てる。
- **リスク**: 決定的モードなので、始める時期で数字は変わりません（同じ GPU 種類なら）。
  待つ科学的な利点はなく、空いている Hamster GPU1 を数時間以上遊ばせることになります。
  50 epoch 版は Ferret で走っているので、その stage 2 をどこで走らせるか（Hamster に checkpoint を移すか Ferret で走らせるか）は別に決める必要があります。

### C: 先に cuda_graph 対応の stage 2 config を作ってから始める

- **内容**: plan 0105 の範囲で、init_checkpoint と cuda_graph を組み合わせた stage 2 config を作り、
  224 px・PGD-3 で eager と同じ結果になることをテストとレビューで確かめてから stage 2 を始める。
- **GPU時間**: 開発・テスト・レビューに半日〜1日（GPU はテストで少し使うだけ）。その後 stage 2 は1本約3.5〜4時間の見込み
  （112 px で単独 約6,600 img/s 対 eager 約2,640 img/s、約2.5倍。224 px でも同じ倍率と仮定）。2本で約7〜8 GPU時間、A より約10 GPU時間少ない。
- **分かること**: A と同じ。加えて、cuda_graph を init_checkpoint つきの run と 224 px・PGD-3 に広げられるか。
- **事前登録ルール**: A と同じ。
- **リスク**: 開発の分だけ判定が遅れます（A なら約12〜19時間後、C なら開発の後さらに約4〜8時間）。
  cuda_graph が 224 px・PGD-3 で eager と一致しなければ、調査が必要になり、さらに遅れます。

## 推奨

A を勧めます。stage 2 の config はレビュー済みで、Hamster GPU1 は今空いています。決定的モードなので、今始めても後で始めても数字は同じで、待つ（B）利点はありません。
C は GPU 時間を約10時間減らせますが、開発とレビューの時間を考えると判定が出るのはほぼ同じか遅くなり、224 px・PGD-3 での一致が確かめられていない分だけ危険が増えます。
cuda_graph の stage 2 対応は、この判定と並行して plan 0105 で進めれば、50 epoch 版の stage 2 や今後の run に使えます。
推奨が変わるのは次の場合です。Hamster GPU1 を別の優先度の高い run（MobileNetV4-M のグリッドや full AT の追加）に使いたいなら B。
今後も stage 2 を何本も走らせる見込みが高く、判定を半日遅らせてもよいなら C。
