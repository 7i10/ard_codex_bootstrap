---
id: 0003
status: decided
created: 2026-09-07
campaign: trades-fix-v1-s0（手動ラン、plan 0027 / docs/debugging/0028 の確認ラン）
question: TRADES 修正の確認ラン trades-fix-v1-s0 は起動 2 秒で例外終了したが、その例外は
  どこにも記録されていない。原因が分からないまま再実行するか、記録を直してから再実行するか。
options:
  A: 同じコマンドを端末で再実行し、stderr を捨てずに保存して traceback を取る。GPU 約 0 時間（失敗は 5 秒で再現する）。決まること = 実際の例外。事前規則 = traceback が取れたら通常の修正に回し、取れなければ B へ。
  B: A に加えて、失敗バンドルが例外を保存するよう train.py の finally を直す（tracker.finish の前に traceback をバンドルへ書く）。GPU 0 時間、作業半日程度。決まること = 今後どの失敗も端末を見ずに診断できるか。事前規則 = 回帰テスト（わざと落とし、バンドルに traceback が残ることを検査）が修正前に落ち修正後に通ること。
  C: 原因不明のまま 200 エポックの確認ランを再投入する。1 GPU x 200 エポック（plan 0027 B2 と同じ予算）。決まること = 一過性の失敗だったかどうかだけ。事前規則 = なし（推奨しない）。
recommendation: B
chosen: B
---

## 何が起きたか

`docs/debugging/0028-trades-clean-target-detached.md` で直した TRADES（clean 側の
KL 標的を detach しない）を確認するための seed 0 ランが、**起動 2 秒で終了コード 1
で落ちた**。学習指標は 1 つも記録されていない。

| 項目 | 値 |
| --- | --- |
| run id | `trades-fix-v1-s0` |
| 状態 | terminal / failed（`success: false`、`failure_class: unknown`） |
| バンドル | `.../runs/trades-fix-v1/seed0/run-bundle/manifest.json` |
| 開始 → 終了 | 2026-09-06T20:49:48Z → 20:49:53Z（W&B runtime 2 秒） |
| ソース SHA | `6eec21f`（master の祖先） |
| 設定 | `configs/scientific/cifar10_r18_trades.yaml`、200 エポック、tier production |

修正そのものは**このランに入っている**。`6eec21f` と現 HEAD で
`src/ard/objectives/` に差分はなく、修正コミット `7666d77` はその祖先にある。
つまり「古いコードを走らせた」という筋ではない。

## 原因は分かっていない。そして分からないのはプラットフォームの欠陥のため

`main()` には非ゼロ復帰の経路が無く、唯一の非ゼロ終了は**例外が外まで抜けること**
（`src/ard/cli/train.py:1217`）である。したがって例外は確実に発生した。
その traceback がどこにも無い。

理由は 2 つ重なっている。

**第一に、失敗バンドルは例外を保存しない。** 失敗時に書かれるのは
`error-marker.txt` の固定文字列 1 行だけで、例外の種類も本文も入らない
（`src/ard/tracking/adapter.py:928`）。バンドルの中身は
`diff.patch` / `environment.json` / `error-marker.txt` / `external.lock.yaml` /
`resolved_config.yaml` の 5 つで、例外を運ぶファイルは 1 つも無い。

**第二に、W&B のログにも入らない。** `train.py:1202-1211` の `finally` は、
例外が外へ抜ける**前に** `tracker.finish(status="failed")` を呼ぶ。W&B は
その中で `_restore()` を実行して stdout/stderr の差し替えを外す。順序は
デバッグログの時刻でそのまま確認できる。

```
05:49:53.114  wandb _finish()
05:49:53.115  got exitcode: 1
05:49:53.115  _restore()      ← ここでストリームが素に戻る
（この後で Python が traceback を印字する）
```

`output.log` に残っているのは差し替えが効いていた間に出た torchvision の
DeprecationWarning 1 件だけで、以降は空である。**traceback は起動した端末の
stderr に出て、そこにしか無い。**

つまり `failure_class` が `unknown` なのは、調べ方が足りないからではなく、
**証拠が保存されない作りになっているため**である。

## 環境は原因ではない

このランの環境は、直前に成功した `parents-v2/seed2` と**完全に一致する**
（torch 2.11.0+cu128、CUDA 12.8、cuDNN 91900、Python 3.11.15、同一ホスト）。
`ard-v2` 環境への移行を疑う筋は、これで消える。

同じ設定ファイルは以前 `f0c3ace` で 200 エポック完走している（plan 0027 の
2026-08-07 の記録）。**同じ設定・同じ環境で、コードだけが変わって落ちるように
なった。** 疑うべき範囲は `f0c3ace` から `6eec21f` までの差分だが、
traceback が無いのでそれ以上は絞れない。

## ついでに見つかったこと（今回の失敗の原因ではない）

**1. 記録される `num_samples` が実態と違う。** `DatasetConfig.num_samples` の
既定値は 16（`src/ard/config/schema.py:263`）で、`cifar10_r18_trades.yaml` は
この値を書いていない。結果、**production の 200 エポック本番ランの resolved config に
`num_samples: 16` が学習側・評価側の両方に記録された**。実行への影響は無い
（`build_raw_dataset` は `synthetic_cifar` のときしか参照せず、cifar10 は
torchvision に丸ごと委ねる。`src/ard/data/datasets.py:471-480`）。
影響が無いのは実行だけで、**来歴の記録は嘘になっている**。合成データ用の
スモーク値が本番の記録に混ざる形なので、既定値を持たせない（未指定を明示的に
「全件」とする）ほうが安全である。

**2. headless postrun が動いていない。** ウォッチャは正しく発火したが、
`orchestration/ardx/claude-runs/20260906T205012Z-trades-fix-v1-s0_seed0.log`
の中身は次の 1 行だけで、postrun は 1 手も進んでいない。

```
Ignoring 44 permissions.allow entries from .claude/settings.json:
this workspace has not been trusted.
```

CLAUDE.md が警告しているとおり、信頼ダイアログはリポジトリルートの綴りごとに
必要で、`/home/islab/...` の実体パスが未承認である。**自動 postrun は現状
一度も成功しない。**

## 推奨

**B。** A だけでも今回の原因は分かるが、同じことが次も起きる。失敗が
5 秒で再現する今が、失敗バンドルに traceback を残す修正を入れる最も安い機会である。
C は、何が壊れているか分からないまま 200 エポック分の GPU を賭けることになるので
推奨しない。

なお 1（`num_samples` の既定値）と 2（信頼ダイアログ）は科学的判断を含まないので、
この決定を待たずに直してよい。指示があれば別プランを立てる。

## 決定（2026-09-08、本人）

**B を選択。** A の traceback 採取に加えて、`train.py` の `finally` を直し、失敗バンドルが例外を
保存するようにする。回帰テスト（わざと落とし、バンドルに traceback が残ることを検査）が修正前に
落ち修正後に通ることを条件とする。GPU 0 時間。
