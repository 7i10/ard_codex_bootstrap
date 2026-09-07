---
id: 0004
status: decided
created: 2026-09-07
campaign: alloc-p1-safe-fork（plan 0096 の arm ALLOC_SAFE、親 p1 = parents-v2-cropshift-s1、手動ラン）
question: 同じ run を 12 分の間に 2 回起動してしまい、1 本目は epoch 101 まで進んだところで
  殺され、2 本目は W&B の run ID 衝突で起動 2 秒で落ちた。1 本目を再開するか、
  新しい run ID で取り直すか、そして再発を止める作りを先に入れるか。
options:
  A: ディレクトリ名を `safe-smoketest` → `safe` に戻し、失敗した方のディレクトリを退避してから同じコマンドで再開する。GPU 約 4.3 時間（残り 98 エポック、1 エポック 155 秒）。決まること = この arm を epoch 199 まで進められるか。事前規則 = 再開後に manifest の `wandb_initialized` が true になり、epoch 102 以降の progress が書かれること。ならなければ B へ。
  B: A に加えて、起動前に落とす門を入れる。`tracking.mode: online` かつ設定に `tracking.run_id` が明示されているとき、ローカルの run-bundle に先行 manifest が無いのに W&B 側に同 ID の run が既にある場合は、fork の種づくりより前に preflight で失敗させる。GPU は A と同じ 4.3 時間、作業は半日程度。決まること = 同じ二重起動が今後 GPU を無駄にせず止まるか。事前規則 = 回帰テスト（同 ID の run が既にある状況を模して、preflight が非ゼロで終わり fork 種づくりが 1 度も走らないことを検査）が修正前に落ち修正後に通ること。
  C: 1 本目を捨て、新しい `tracking.run_id`（例 `alloc-p1-safe-fork-a2`）で epoch 100 から取り直す。GPU 約 4.4 時間。決まること = この arm の結果だけ。ただし epoch 100–101 を持つ W&B run が親無しで残り、arm と run が 1 対 1 でなくなる。
  D: 原因も作りもそのままに、同じコマンドをもう一度打つ。GPU 0 時間で必ず同じ 2 秒で落ちる。決まること = 何も。推奨しない。
recommendation: B
chosen: B
---

## 何が起きたか

plan 0096 の最初の arm（`ALLOC_SAFE`、親 p1）を、**同じ設定・同じ run ID で 12 分違いに
2 回起動している**。両方の `config_hash` は `fa0b4a47…` で完全に一致するので、
別 arm でも別条件でもなく、同じ 1 本を 2 回起動した。

| | 1 本目 | 2 本目（今回 postrun が呼ばれた方） |
| --- | --- | --- |
| バンドル | `arms/p1/safe-smoketest/run-bundle/manifest.json` | `arms/p1/safe/run-bundle/manifest.json` |
| 開始 | 2026-09-07T04:01:28Z | 2026-09-07T04:13:28Z |
| 終了 | （記録なし） | 2026-09-07T04:13:30Z、failed |
| W&B | 初期化成功、run `alloc-p1-safe-fork` | **初期化失敗** |
| 進んだところ | epoch 101 まで（`last.pt` / `best.pt` / `epoch-metrics.jsonl` あり） | 学習 0 エポック |
| 現在の manifest 状態 | `running`（終端になっていない） | `failed` / `failure_class: unknown` |

どちらのプロセスも今は生きていない（`pgrep -af ard.cli.train` に残るのは
`trades-fix-v1/seed0` だけ）。1 本目は**終端記録を書かずに死んでいる**ので、
ウォッチャから見ると永久に「実行中」のままの stale バンドルになる。

科学的な損失はほとんど無い。捨てることになるのは fork 後の epoch 100–101 の
2 エポック、GPU にして約 5 分である。

## 2 本目が落ちた理由ははっきりしている

W&B の内部ログに 1 行で出ている。

```
13:13:30.694 ERROR runupserter: failed to init run  error="data but cannot resume"
```
（`arms/p1/safe/wandb/wandb/run-20260907_131330-alloc-p1-safe-fork/logs/debug-internal.log:8`、時刻は JST）

サーバ側の run `alloc-p1-safe-fork` には 1 本目が既に 101 エポック分を書き込んでいる。
そこへ 2 本目が `resume="never"` で同じ ID を作りに行ったので、W&B が拒否した。
`tier: production` なので `wandb.init` の例外はそのまま `TrackingError` になって
落ちる（`src/ard/tracking/adapter.py:637-640`）。manifest の
`wandb_initialized: false` / `sync_state: failed` はその結果である。

**なぜ `resume="never"` になったか**が本題である。resume するかどうかは
**出力ディレクトリの中の先行 manifest があるかどうか**だけで決まる。

```python
self._is_resume = prior is not None            # adapter.py:470、prior は output_dir/run-bundle/manifest.json
"resume": "must" if self._is_resume and self.manifest["wandb_initialized"] else "never"   # adapter.py:627
```

一方 W&B の run ID は設定に書かれた固定の文字列で、ディレクトリとは無関係である
（両方の `resolved_config.yaml:145` が `run_id: alloc-p1-safe-fork`）。

**つまり、再開の根拠はローカルのディレクトリに、同一性はリモートの ID にある。**
ディレクトリを動かすと前者だけが消え、後者は残る。そのとき次の起動は
必ずこの形で落ちる。

## ディレクトリは実行中に改名されている

1 本目の manifest は自分の W&B 出力先をこう記録している。

```
wandb_segments[0].path = .../arms/p1/safe/wandb/wandb/run-20260907_130129-alloc-p1-safe-fork
```

ところが実際にそのディレクトリがあるのは `safe-smoketest/` の下である。
1 本目の `resolved_config.yaml:198` も `output_dir` を `.../arms/p1/safe` と書いている。
`safe-smoketest/` には `best.pt` と `epoch-metrics.jsonl` があり `safe/` には無いので、
コピーではなく**改名**である。

順番はこうなる。1 本目が `safe/` で走っている最中に `safe/` が `safe-smoketest/` へ
改名され、空になった `safe/` に対して 2 本目が起動した。2 本目は fork の種づくりまでは
成功しており（`safe/stagewise-fork-complete.json` が `status: complete`、`safe/last.pt` あり）、
その直後の W&B 初期化で落ちている。1 本目はこの前後で殺された。

なお 1 本目は改名後も、開いたままの inode 経由で `safe-smoketest/run-bundle/manifest.json`
を更新し続けていた。最後の progress は 04:07:25Z、epoch 101 である。

## `failure_class: unknown` の意味

ウォッチャが `unknown` と分類したのは、決定 0003 に書いたとおり**失敗バンドルが
例外を保存しないため**である（残るのは `error-marker.txt` の
`application failure recorded` の 1 行だけ）。今回は W&B 側の内部ログに証拠が
残っていたので原因まで辿れたが、それは幸運であって仕組みではない。
決定 0003 の option B（バンドルに traceback を残す）は今回も有効である。

## 記録されている数字（1 本目、参考値であって結果ではない）

`arms/p1/safe-smoketest/epoch-metrics.jsonl` の全内容。学習時 PGD であって
公式テストではない。

| epoch | train clean | train robust | val clean | val PGD |
| --- | --- | --- | --- | --- |
| 100 | 0.7574 | 0.5216 | 0.8272 | 0.5466 |
| 101 | 0.7637 | 0.5522 | 0.8302 | 0.5524 |

`train_valid_examples: 45000` なので学習は全訓練分割で回っている。
`resolved_config.yaml:18` の `num_samples: 16` は決定 0003 の「ついでに見つかったこと 1」
と同じ来歴の嘘であり、**実行には効いていない**。

固定同一性: CIFAR-10 / `saad_resnet18_cifar_v1` / RSLAD / 教師
`chen2021_ltd_wrn34_10`（SHA `fc398a48…`）/ 学習シード 1 / 評価シード 0 /
linf eps 8/255 step 2/255 / world size 1 / effective global batch 128 /
ソース SHA `815dabd` / 親 `parents-v2-cropshift-s1` の epoch 99（SHA `03feadbb…`）/
switch epoch 100 / late policy `idbh_weak` / mask 15,317 枚（SHA `31912f40…`）。

## 推奨

**B。** A だけでも今日の arm は進むが、plan 0096 は同じ形の fork をこれから
**4 arm × 6 親 × 2 反復**動かす計画で、run ID はすべて設定に固定文字列で書く。
今回の二重起動は fork の種づくりまで走ってから落ちたので、同じことが起きるたびに
出力ディレクトリが 1 つ壊れる。門を preflight に置けば、GPU も種づくりも消費せずに
その場で止まる。

C は 1 本目を捨てるだけでなく、epoch 100–101 だけを持つ孤児の W&B run を残し、
arm と run の 1 対 1 対応を壊すので採らない。

**判断を待たずに直してよいもの**（科学的判断を含まない）:

1. `safe-smoketest/run-bundle/manifest.json` が `running` のまま終端になっていない。
   死んだプロセスのバンドルを終端にする手順が無く、`/experiment-status` は
   これを永久に stale として数え続ける。
2. サーバ側の W&B run `alloc-p1-safe-fork` も終了記録を受け取っていない。

どちらも指示があれば別プランを立てる。

## 参照

- plan: `docs/plans/0096-hardness-allocation-direction.md`（M2、arm `ALLOC_SAFE`）
- 失敗バンドル: `.../runs/alloc-direction-v1/arms/p1/safe/run-bundle/manifest.json`
- 生き残っている 1 本目: `.../runs/alloc-direction-v1/arms/p1/safe-smoketest/`
- 関連決定: `docs/decisions/0003-trades-fix-v1-failure-not-diagnosable.md`

## 決定（2026-09-08、本人）

**B を選択。** ただし A の再開部分は不要である。該当アーム `alloc-v1-p1-safe-fork` は
2026-09-07 に epoch 199 まで完走しており（`arms/p1/safe/epoch-199.pt`）、この論点は
実行としては既に解決している。残るのは作りの方で、preflight 門を入れる作業だけが残件となる。

門の条件は元の記載どおり。`tracking.mode: online` かつ設定に `tracking.run_id` が明示されている
とき、ローカルの run-bundle に先行 manifest が無いのに W&B 側に同 ID の run が既にある場合は、
fork の種づくりより前に preflight で失敗させる。回帰テストが修正前に落ち修正後に通ることを条件と
する。

この衝突は本セッション中に 3 回再発している（plan 0092、TRADES 確認ラン、plan 0096）。
根は `_tracking_run_id` が `continuation_seed` を含まないことで、その都度 `--run-namespace` や
接頭辞で手当てしてきた。門はその手当てを忘れたときに GPU を無駄にしないためのものである。
