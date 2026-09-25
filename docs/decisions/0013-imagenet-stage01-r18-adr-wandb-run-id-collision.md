---
id: 0013
status: decided
created: 2026-09-13
campaign: imagenet-stage01-r18-mobilenetv3-adr-v2（plan 0100 stage 1、job `r18_adr-s0`）
question: stage 1 の `r18_adr-s0` が起動 5 秒で落ちた。原因は、中止した v1 attempt2 が同じ W&B run ID
  `imagenet-stage01-r18_adr-s0` に既に書き込んでいたこと。この 1 本をどの run ID で、どの手順で取り直すか。
options:
  A: 中止した v1 の W&B run `imagenet-stage01-r18_adr-s0` と `imagenet-stage01-r18_adr-s1` を W&B 上で削除し、`r18_adr-s0` だけを同じ run ID・同じ identity hash（`62f8fd07e5e7…`）の 1 job キャンペーンとして gate から再起動する。GPU 約 28 時間（plan の 338 GPU 時間 / 12 本からの目安）。決まること = stage 1 の 4 本目が揃うか。事前規則 = 新しい manifest で `wandb_initialized: true` になり、epoch 0 の progress が書かれること。ならなければ止めて報告。削除は元に戻せない。
  B: v1 の W&B run は残し、新しい run ID（例 `imagenet-stage01-v2-r18_adr-s0`、spec の `ARD_RUN_ID`）で 1 job キャンペーンを gate から起動する。GPU は A と同じ約 28 時間。決まること = A と同じ。ただし stage 1 の 4 本のうち 1 本だけ命名が違い、同じ W&B group `imagenet-r18-adr-controlled` に v1 の途中停止 run 2 本が混ざったまま残る。identity hash が変わるかは gate の resolved manifest で確認が要る。
  C: A（または B）の再起動の前に、決定 0004 で選ばれたまま未実装の preflight 門を入れる（`tracking.mode: online` で run ID が明示され、ローカルに先行 manifest が無いのに W&B に同 ID の run がある場合、GPU を使う前に失敗させる）。作業半日程度 + 回帰テスト、GPU は A と同じ。決まること = stage 2（seed 1/2 の 8 本、うち `r18_adr-s1` は同じ衝突を必ず起こす）を安全に出せるか。事前規則 = 回帰テストが修正前に落ち修正後に通ること。
  D: 何も変えずに同じ job を再投入する。GPU 0 時間で必ず同じ 5 秒で落ちる。推奨しない。
recommendation: A を今すぐ、C の門は別プランで stage 2 の起動前までに
chosen: A/B とは別に hand-run で解決（2026-09-13 追記を参照）
---

## 何が起きたか

stage 1 キャンペーン `imagenet-stage01-r18-mobilenetv3-adr-v2` の 4 本のうち、
`mobilenetv3_baseline-s0` が 2026-09-12T20:26:16Z に完走した。空いた GPU 1 に
コントローラが `r18_adr-s0` を入れたが、**5 秒で exit code 1** になった。
コントローラは `failure_class: unknown`、`retryable: false` と記録し、再試行はしていない。
続けて同じ GPU 1 に `r18_baseline-s0` が入り、こちらは正常に走っている。

学習は 0 ステップも進んでいない。失ったのは GPU 5 秒だけで、科学的な損失は無い。

| job | 状態（2026-09-12T20:28Z 時点） |
| --- | --- |
| `mobilenetv3_baseline-s0` | completed（epoch 49、68,275 秒） |
| `mobilenetv3_adr-s0` | running（GPU 0、epoch 43） |
| `r18_baseline-s0` | running（GPU 1、20:26:23Z 開始） |
| `r18_adr-s0` | **failed**（W&B 初期化で停止） |

## 原因

attempt ログの最後にそのまま出ている。

```
wandb.errors.errors.UsageError: You provided an invalid value for the `resume` argument.
The value 'never' is not a valid option for resuming a run (imagenet-stage01-r18_adr-s0) that already exists.
...
ard.tracking.adapter.TrackingError: requested W&B tracker could not initialize
```

W&B 内部ログにも `runupserter: failed to init run error="data but cannot resume"` がある
（`r18_adr-s0/train/wandb/wandb/run-20260913_052621-imagenet-stage01-r18_adr-s0/logs/debug-internal.log:8`）。

順番はこうである。

1. 2026-09-12 に v1 attempt2 が `r18_adr-s0` を約 50 分走らせ、W&B run
   `imagenet-stage01-r18_adr-s0` に epoch 0 の途中（global step 9,809）まで書いた。
   その出力は `runs/imagenet-stage01-r18-mobilenetv3-adr-v1/r18_adr-s0/train` にある。
2. plan 0100 の判断で attempt2 を中止し、**新しいキャンペーン ID と新しい出力ディレクトリ**
   （`...-adr-v2/`）で stage 1 を出し直した。identity hash を attempt2 と一致させたので、
   run ID も同じ文字列のままになった。
3. v2 の出力ディレクトリには先行 manifest が無いので、アダプタは新規 run と判断して
   `resume="never"` で W&B を初期化した（`src/ard/tracking/adapter.py:627`）。
   W&B 側には同 ID の run が既にあるので拒否された。`tier: production` なので例外は
   そのまま学習を止める（`adapter.py:639-640`）。

**決定 0004 と全く同じ形である。** 再開の根拠はローカルのディレクトリ、同一性は W&B の ID にあり、
ディレクトリだけを新しくすると必ずこうなる。0004 では option B（preflight 門）が選ばれたが、
`src/` と `scripts/` にその門は実装されていない（W&B 側の既存 run を確かめるコードが無い）。
plan 0100 の中止記録も「新しいキャンペーン ID なので出力パスは再利用されない」とだけ確認しており、
W&B の run ID が再利用されることは見落としていた。

`mobilenetv3_*` と `r18_baseline-s0` が無事なのは、attempt2 で実際に走ったのが
`r18_adr-s0` と `r18_adr-s1` の 2 本だけだったからである。

## この先に同じことが起きる場所

- **stage 2 の `r18_adr-s1`**: v1 attempt2 が同じ ID に書いているので、何も変えなければ同じく落ちる。
- `r18_adr-s0` を同じ ID で出し直す場合（A）、v1 の W&B run を消さない限り落ちる。

## 判断を待たずに直してよいもの（科学的判断を含まない）

1. **キャンペーンの `state.json` が /tmp にある。**
   `/tmp/claude-1001/.../027b5ec8-.../scratchpad/.orchestration/imagenet-stage01-r18-mobilenetv3-adr-v2.state.json`。
   前セッションの scratchpad なので、ウォッチャの走査対象（`ard-runtime/.../runs`）に入っていない。
   今回 postrun が呼ばれたのは run-bundle 側の検知であって、キャンペーンとしては検知されていない。
   再起動や /tmp の掃除で消えると、キャンペーン全体の完了がどこにも残らない。
   残り 2 本が完走したときキャンペーン単位の postrun が自動で起きない可能性が高い。
2. v1 attempt2 の 2 つの run-bundle は `running` のまま stale で残っている（中止時の判断どおり）。

どちらも指示があれば別プランを立てる。

## GPU の空き

GPU 1 は `r18_baseline-s0` が使っている。GPU 0 の `mobilenetv3_adr-s0` は 20:08Z に epoch 43 で、
残り 7 エポック（1 エポック約 23 分）なので 2026-09-12T23:00Z 前後に空く見込みである。
そこから先は、このキャンペーンに GPU 0 へ入れる job が残っていない。判断がそれより遅れると、
GPU 0 がその分だけ遊ぶ。

## 推奨

**A を今すぐ、C の門は別プランで stage 2 の前までに。**

v1 の W&B run 2 本は、中止して「再開しない」と決めた途中停止 run（epoch 0 の途中まで）であり、
記録にも報告にも使われない。消すことで失う情報は無い。一方で残すと、stage 1 の本番 run と
同じ group に混ざり続ける。A なら run ID も identity hash も attempt2 との一致を保てる。

B は削除という元に戻せない操作を避けられるが、4 本のうち 1 本だけ命名がずれ、
group の汚れも残る。削除に抵抗があるなら B でよい。

C の門は stage 1 の 1 本のためには間に合わせる必要は無いが、stage 2 は 8 本あり、
同じ衝突はこれで少なくとも 4 回目である（決定 0004 の時点で 3 回）。

削除（A）は W&B 上の外部操作なので、選ばれてから行う。

## 追記（2026-09-13、対話セッション内、実施済み）

このpacketの`r18_adr-s0`の再取り直しは、案A/Bを待たずに**hand-runで解決済み**。
中止済みv1のrun 2本は削除していない（案A非採用）。新しいrun ID
（`imagenet-stage01-r18-mobilenetv3-adr-v2-r18_adr-s0-handrun1`、案Bの意図と
同じ「新IDで出す」形）でGPU 0にhand-runし、現在進行中。詳細は
`docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md`のProgress log
（2026-09-13付）を参照。このpacket自体は`chosen: null`のまま残す——
stage 1個別ジョブの再取り直しはもう論点でなくなったが、**案Cの事前チェック**
（stage 2の`r18_adr-s1`が同じ衝突を必ず起こす、という指摘）は生きている。

案Cの事前チェックは実装済み: `src/ard/tracking/adapter.py`の
`validate_tracking_guard`に`_reject_stale_remote_run_id`を追加し、
`tracking.mode: online`かつ`tracking.run_id`が明示されていて、ローカルに
先行run-bundle manifestが無いのにW&B側に同IDのrunが既にある場合、fork種
づくりより前（`ard.cli.train`/`ard.cli.evaluate`双方のoutput guardフェーズ）
に`TrackingError`で止まるようにした。回帰テスト3本
（`tests/unit/test_tracking.py::test_online_guard_*`）を追加し、`scripts/verify.py --changed`
はT0/T1/T3すべて緑。**ただしこれは`src/ard/tracking/`への変更であり、
CLAUDE.mdの標準規則どおりscientific-reviewerのレビューを経てから、
stage 2用にソースSHAを新しく凍結する。**

もう1つ、このpacketの本題ではないが同じセッションで見つかった関連問題:
このv2キャンペーンのオーケストレーターstate.jsonが、対話セッションの
scratchpad（`/tmp/claude-1001/.../scratchpad/.orchestration/...`）に
置かれており、`campaign_watch.py`のデフォルト走査対象
（`run_root`と`orchestration_root`、`configs/workspace/ard_workspace_v1.json`）
に入っていない。原因はlaunch_gate.pyの`state_path`デフォルトが
「specファイル自身の置き場所」からの相対パスであること（specを
scratchpadに書いたため、そこが基準になった）。**恒久対策として
`.claude/skills/experiment-launch/SKILL.md`のstep 3に、`state_path`を
`orchestration_root`配下の絶対パスで必ず明示する、という注記を追加した**
（stage 2以降のspec作成から効く）。**今動いているv2キャンペーン自体は
直せない**（生きているコントローラのstate_pathは変更できない）ため、
残り2本（`r18_baseline-s0`、`r18_adr-s0`のhand-run）が終わったら、
`/experiment-postrun`は`--state-path`にこのscratchpadのstate.jsonを
明示的に渡す必要がある。自動検知に任せると永久に拾われない。

## 参照

- plan: `docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md`（Progress log 2026-09-12 の中止と stage 1 起動）
- 失敗バンドル: `ard-runtime/ard_codex_bootstrap/runs/imagenet-stage01-r18-mobilenetv3-adr-v2/r18_adr-s0/train.attempt1-wandb-collision-failed/run-bundle/manifest.json`
  （元は `train/`。再投入に備えて改名済みで、今は `r18_adr-s0/` の下に新しい `train/` は無い。
  改名後にウォッチャが同じ失敗をもう一度検知したが、状態は同じ `failed` / `unknown` で新しい情報は無い）
- attempt ログ: `<state dir>/orchestration/imagenet-stage01-r18-mobilenetv3-adr-v2/99465e2c…/d4fc2d99…/r18_adr-s0.attempt-1.log`
- 衝突相手: `ard-runtime/ard_codex_bootstrap/runs/imagenet-stage01-r18-mobilenetv3-adr-v1/r18_adr-s0/train`（W&B `imagenet-stage01-r18_adr-s0`）
- gate: `ard-runtime/ard_codex_bootstrap/runs/imagenet-stage01-r18-mobilenetv3-adr-v2-attempt3/launch-gate/`（resolved manifest SHA `99465e2c…`）
- ソース SHA `ae4dd7c82d80`、worktree `source-ae4dd7c82d80`
- 関連決定: `docs/decisions/0004-alloc-p1-safe-fork-run-id-collision.md`、`docs/decisions/0003-trades-fix-v1-failure-not-diagnosable.md`

## 2026-09-25 追記：状態の整理

判断と実施は上の追記のとおり完了していたため、人間の了承（チャット、2026-09-25）を得て status を decided に更新した。
