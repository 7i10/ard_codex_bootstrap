---
id: 0010
status: pending
created: 2026-09-10
campaign: adr-cifar10-campaign-v1（plan 0097、`cifar10_r18_adr-s2` の
  model 重み評価 `eval-5709dc2e3701675af39e` が AutoAttack 内の CUDA OOM で失敗。
  同じ落ち方が `r18_trades_adr-s1` と `r18_adr-s1` にもある）
question: AutoAttack の最後にある「10,000 枚を一度に流す」forward を先に
  バッチ化してから 3 本を取り直すか、コードには触れず GPU を専有して
  そのまま取り直すか、走行中の 7 本が終わるまで何もしないか。
options:
  A: `autoattack.py:213` の forward を `autoattack_batch_size` でバッチ化し、回帰テストで数値が完全一致することを示してから、失敗した 3 本を取り直す。GPU 約 5.5–10 時間（取り直しぶんのみ、修正自体は 0 時間）。決まること = 残り約 21 本の AutoAttack 評価からこの OOM 経路が消える。事前規則 = バッチ版と非バッチ版が同一テンソルに対して**予測ラベル列まで完全一致**することをテストで示すこと。一致しなければ A は却下。
  B: コードは変えず、空いた 4090 を 1 本ずつ専有して 3 本を取り直す。GPU 約 5.5–10 時間＋ GPU 専有の待ち時間。決まること = その 3 本の数字だけ。残り 18 本の脆弱性はそのまま。事前規則 = 1 本目が完走したら残り 2 本も同じ専有条件で流す。1 本目がまた OOM したら B を捨てて A へ。
  C: 何もせず、走行中の 7 本が terminal になるまで待ってから判断し直す。GPU 追加 0 時間。決まること = この OOM が「3 本の事故」なのか「同時実行しているかぎり必ず出る」のかが分かる。事前規則 = 7 本のうち 1 本でも同じ OOM で落ちたら A を必須とする。全部完走したら B で足りる。
recommendation: A
chosen: null
---

## 追記（2026-09-10 11:07Z）— Option C の事前規則が発火した

この packet を書いた時点で「走行中だった 7 本」のうちの 1 本、
`cifar10_mobilenetv2_adr-s1` の model 重み評価
（`eval-4615ede05d4f96adb48f`）が、**まったく同じ場所で同じように**落ちた。
Option C の事前規則は「7 本のうち 1 本でも同じ OOM で落ちたら A を必須と
する」だった。したがって **C は選択肢から外れる**。`chosen` は null のまま、
決めるのは人間である。

新しく分かったことは 2 つある。

1. **これは ResNet-18 だけの話ではなく、MobileNetV2 のほうが 1.5 倍危ない。**
   失敗した allocation は **3.66 GiB**、これは
   `10000 × 96 × 32 × 32 × 4 B = 3.662 GiB` とバイト単位で一致する。
   `src/ard/models/registry.py:126` が `mobilenet_v2_cifar` の stem stride を
   `(1,1)` にしているので解像度が 32×32 のまま保たれ、`features[2]` の
   expansion conv（expand_ratio 6）が全 10,000 枚ぶんの **96**×32×32 を
   一度に確保しようとする。ResNet-18 の 2.44 GiB
   （`10000 × 64 × 32 × 32 × 4 B`）と同じ機構で、係数だけが大きい。
   残っている MobileNetV2 の評価は、この欠陥に対して最も脆い部類に入る。

2. **落ちているのは冗長な再計算だけで、AutoAttack 自体は完走している。**
   lane log では Square が `34/34` まで進み、AutoAttack 自身が
   `robust accuracy after SQUARE: 42.84% (total time 7656.5 s)` を出力して
   いる（traceback がログ上で先に見えるのは stderr が先に flush された
   だけ）。つまり `run_standard_evaluation` は正常に返り、その次の 213 行で
   死んでいる。**この 42.84 % は完走しなかった run のログ片であって結果では
   なく、どの record にも report にも表にも入れてはならない。**
   `evaluation-results.json`・`autoattack-best.json`・`completion.json` は
   いずれも書かれておらず、`last.pt` には到達していない。2 時間 8 分
   （08:59:09Z → 11:07:25Z）ぶんの GPU 時間から取り込めるものは何もない。

落ちたときの GPU 0 は 23.52 GiB 中 **2.93 GiB** 空き、プロセスが 5 本
（13.12 + 1.74 + 1.74 + 1.71 GiB ＋ この run の 2.25 GiB）同居していた。

**レーンドライバの second defect も再現した。** lane B は `exit=1` の
3 秒後（20:07:26+09:00）にそのまま `--weights=ema`
（`eval-74a8f5d0042ebea0b0d1`）を開始している。これは前述のとおり別 plan の
インフラ案件で、この packet では決めない。

証拠の全文は plan 0097 の Progress log（2026-09-10 11:07Z のエントリ）。
これで同一 traceback による失敗は **4 本**（`r18_adr-s2`、
`r18_trades_adr-s1`、`r18_adr-s1`、`mobilenetv2_adr-s1`）になった。
取り直しの見積もりは A・B とも 3 本ではなく 4 本ぶん、
**約 7–13 GPU 時間**に増える。

## この packet は結果についてではない

plan 0097 の M2/M3 はまだ開いており、`docs/experiments/` に取り込むべき
record は存在しない。実際に何も取り込んでいないし、milestone も一つも
tick していない。この packet が扱うのは `/experiment-postrun` が
`eval-5709dc2e3701675af39e` の失敗を追っている最中に見つかった
**評価コードの欠陥**と、それをどう片付けるかである。

証拠の全文は plan 0097 の Progress log（2026-09-10 11:01Z のエントリ）に
ある。以下は要点だけ。

## 何が起きたか

`src/ard/evaluation/autoattack.py` の 211–213 行はこうなっている。

```python
adversarial = adversary.run_standard_evaluation(images, labels, bs=batch_size)
with torch.no_grad():
    accuracy = model(adversarial).argmax(1).eq(labels).float().mean().item()   # 213
```

`bs=128` は AutoAttack の**内側**では効いているが、その次の行の精度再計算は
それを無視して 10,000 枚を**一度に**student に流す。失敗した allocation は
**2.44 GiB** で、これは `10000 × 64 × 32 × 32 × 4 B = 2.441 GiB`、つまり
ResNet-18 の `layer1` が全件ぶん持つ activation とぴたり一致する
（traceback の最内は `registry.py:75`、`layer1` 内の
`self.bn2(self.conv2(outputs))`）。

つまりピークは 2 時間の仕事の**いちばん最後**、AutoAttack の 4 段階を
全部払い終えたところに来る。しかもその大きさは `autoattack_batch_size` では
なく**テスト集合の枚数**で決まる。落ちた時点で GPU の空きは 23.52 GiB 中
**155 MiB**、同じ 4090 に 3 プロセス（1.73 + 8.46 + 13.13 GiB）が載っていた。
9 レーンの手動評価ドライバが 1 枚の 4090 に評価を複数積むので、この
forward が要求する数 GiB の余裕が取れない。

**失われたもの。** `results.append(...)` は `run_autoattack` が返った後
（`src/ard/cli/evaluate.py:411-421`）、metrics ファイルは checkpoint ループの
後に書かれる。したがって `best.pt` の clean と CE-PGD-20 は計算済みなのに
AutoAttack もろとも捨てられ、`last.pt` には到達すらしていない。取り込める
数字は一つもない。lane log に残っている AutoAttack の途中経過
（initial accuracy 83.32 %、APGD-T 後 48.22 %、FAB-T 後 48.22 %、Square は
2/38 で停止）は**完走しなかった run のログ片であって、結果ではない**。
どの record にも report にも表にも入れてはならない。

**影響範囲は 3 本以上。** `canon-eval-lane-E.log` に
`cifar10_r18_trades_adr-s1` の同一 traceback（`exit=1`、20:01:15+09:00）、
11:05Z の再スキャンでは `cifar10_r18_adr-s1/train/evaluation` も `failed`。

**もう一つの欠陥（別 plan 案件）。** レーンドライバは非 0 終了で止まらず、
`exit=1` の直後にそのまま `--weights=ema` を開始している
（lane-E 20:01:15、lane-F 20:01:19）。両方とも 20:04:24 に SIGTERM
（`exit=143`）で消えた。CLAUDE.md のルール 3 に従い、これはこの packet では
決めず、インフラ欠陥として別 plan に切り出すべきものとして記録するに留める。

**この欠陥が残っている限りの露出。** 現時点で AutoAttack 評価は
model 重みが 13 本（failed 1・実行中 6・未着手 6）、EMA 重みが 8 本
（実行中 1・未着手 7）、合わせて**約 21 本**が未了である。

## コストの前提

このキャンペーンで完走した「両 checkpoint ＋ AutoAttack」1 本ぶんの実測
（すべて他ジョブと同居した状態、run-bundle の `created_at`→`finished_at`）：

| run | 実測 |
|---|---:|
| `r18_pgd_at_nesterov-s0` (model) | 1 h 48 m |
| `r18_pgd_at-s1` (model) | 1 h 50 m |
| `r18_adr-s0` (ema) | 1 h 57 m |
| `r18_adr-s0` (model) | 3 h 15 m |

以下の見積もりはすべて **1 本あたり 1.8–3.3 GPU 時間**から換算した。
専有すれば下限側、あるいはそれより速いはずだが、専有での実測はまだ無い。

## Option A — forward をバッチ化してから 3 本取り直す

- **GPU 見積もり**: 修正 0 時間、取り直し 3 本 × 1.8–3.3 h = **約 5.5–10 GPU 時間**。
- **決まること**: 残り約 21 本の AutoAttack 評価からこの OOM 経路が消える。
  今後の評価が同時実行の込み具合に左右されなくなる。
- **事前規則**: バッチ版と非バッチ版が、同一の adversarial テンソルに対して
  **平均だけでなく予測ラベル列まで完全一致**することを回帰テストで示すこと
  （テストは修正前に落ち、修正後に通ること）。一致しなければ A は却下し、
  B か C に戻る。
- **risks**: これは評価コードへの変更である。ただし epsilon・steps・step size・
  random start・normalization・temperature・schedule・checkpoint 選択・
  評価 attack のどれにも触れない。model は `eval()`、BN は running 統計を
  使うので、バッチ化は数値的に中立であることが期待でき、上の事前規則は
  まさにそれを要求している。**AutoAttack 自身が出す robust accuracy を
  そのまま採用して 213 行を消す**という「もっと安い」案は、記録される量が
  変わるので中立ではない。これは新しい contract であり、この packet の
  選択肢には含めない。

## Option B — コードは変えず、GPU を専有して取り直す

- **GPU 見積もり**: 3 本 × 1.8–3.3 h = **約 5.5–10 GPU 時間**。ただし 1 本ずつ
  4090 を専有するため、他のレーンを止める待ち時間が別途かかる。
- **決まること**: その 3 本の数字だけ。残り 18 本は同じ欠陥を抱えたまま。
- **事前規則**: 1 本目が完走したら、残り 2 本も同じ専有条件で流す。1 本目が
  また OOM したら B を捨てて A に移る。
- **risks**: 2 時間走り切った最後で落ちるので、失敗のコストが高い。専有を
  やめた瞬間に再発する。残り 18 本ぶんの露出（38–60 GPU 時間相当）が
  そのまま残る。

## Option C — 走行中の 7 本が終わるまで何もしない

- **GPU 見積もり**: 追加 **0 時間**（走行中の 7 本は既に走っている）。
- **決まること**: この OOM が「混み合った 3 本の事故」なのか「同時実行して
  いるかぎり必ず出る」のかが分かる。今コードを直しても、走行中の 7 本は
  既に旧コードで動いているので恩恵を受けない、という点でも筋は通る。
- **事前規則**: 7 本のうち 1 本でも同じ OOM で落ちたら A を必須とする。
  7 本とも完走したら B で足りる。
- **risks**: 7 本が終わるまで数時間かかり、その間に落ちた本数だけ 2 時間ずつ
  無駄になる。`r18_adr-s1` が既に 3 本目として落ちている以上、「事故」説は
  すでに弱い。

## なぜ A を推すか

原因が推測ではなく**確定**している点が大きい。要求サイズ 2.44 GiB が
`10000 × 64 × 32 × 32 × 4 B` と桁ではなくバイト単位で一致しており、
traceback の行も特定できている。直し方も一意で、しかも「数値が完全一致
すること」を事前規則にできるので、科学的な中身を一切動かさずに済む。
GPU コストは A も B も同じ 5.5–10 時間で、A だけが残り約 18 本ぶんの
露出（1 本 1.8–3.3 h として 38–60 GPU 時間相当）を同時に消す。

**noise floor について。** ここで測り直そうとしているものは何も無い。
評価は保存済み checkpoint に対して固定 seed（`evaluation_attack: 0`）で
走る決定的な手続きなので、RNG の 1–2 pp という floor はこの判断には
掛からない。むしろ逆で、A の修正後に同じ checkpoint から**違う**
AutoAttack の数字が出たら、それは noise ではなく修正が中立でなかった証拠
であり、上の事前規則がそれを検出する。どの選択肢も「結果」を生まない。

**判断が変わる条件。** 回帰テストで予測ラベル列が一致しなければ A は成立
しないので、その場合は B（専有での取り直し）に落とす。逆に、走行中の 7 本が
全部完走して `r18_adr-s1` の失敗が OOM 以外の原因だったと分かれば、C→B で
十分になる。

## 参照

- plan: `docs/plans/0097-adr-cifar10-replication.md`（Progress log、2026-09-10 11:01Z）
- 証拠ログ: `runs/adr-cifar10-campaign-v1/canon-eval-lane-{D,E,F}.log`
  （失敗した評価ディレクトリ自体は 11:05Z の時点で既に削除されている）
- 該当コード: `src/ard/evaluation/autoattack.py:211-213`、
  `src/ard/cli/evaluate.py:411-421`（いずれも pinned SHA
  `cd0b571e4685` と hash 一致）
