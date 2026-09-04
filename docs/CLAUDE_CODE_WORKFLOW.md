# Claude Code 実験ワークフロー v1

Status: design + runbook. 2026-09-05 に Codex 運用から移行するにあたり、リポジトリ全体
（503 commits, docs 384 files, runtime root 上の全 campaign 状態）を再調査して設計した。
本書は「なぜこの形か」と「日常どう回すか」だけを書く。科学的契約は
`docs/SCIENTIFIC_INVARIANTS.md`、エージェント向け短縮版は `CLAUDE.md` と `.claude/rules/` にある。

## 0. 結論

1. **3 プレーン分離**。科学コード（`src/ard`, `configs`）／実行プレーン（GPU job の起動・完了検知。
   LLM 不要、耐久）／エージェントプレーン（設計・実装・集計・報告。Claude Code）。
   前回の最大の損失は実行プレーンの修理が科学の critical path に乗ったこと
   （request→launch 7h01m に対し gate 本体は 69.7 s。`docs/ERT_I100_ONLINE_STATE_S2_REQUEST_TO_LAUNCH_RETRO.md`）。
2. **完了検知→次タスクは常駐 watcher + ヘッドレス Claude**。Claude セッションを開きっぱなしにしない。
   Hamster は `loginctl` Linger=yes なので `systemd --user` サービスが logout 後も生きる。
   `claude -p` はサブスクのログインで systemd ユニット内からも動作することを 2026-09-05 に実測済み。
3. **人間の介入点は decision packet に一本化**。Claude は結果を「選択肢＋証拠＋コスト」に整形して止まる。
   人間は選ぶだけ。選ばれた案の起動は Claude が行う。
4. **ソースは必ず pinned worktree から実行**。attempt-11 の 8 job は、稼働中に作業ツリーが動いた
   ことで `online-state runtime requires a clean scientific source tree` により 2.7 s で全滅した。
   recovery14–17 が worktree `source-bcb09a7` から実行して成功したのがその証明。
5. **Codex 時代の制御プレーンは「使う部分」を限定**。使う: `launch_gate.py`
   （preflight / dry-run / canary / launch）、`orchestrate.py`、`run-on-ferret`。
   凍結（呼ばない・依存しない）: `reconcile_experiment.py`、`publish_experiment_terminal_event.py`
   と PR #1 event bus、fast path / runtime signature、`launch_ledger.py`、`task_context.py`。
   いずれも実キャンペーンで一度も動いていない（`experiment-state.json` は runtime root に存在しない、
   event bus のイベント数 0、signature registry はダミー 1 件）。
6. **毎キャンペーンは「記録のコミット」で終わる**。0087 は完了して肯定的判定が出たのに、
   結果は runtime root に取り残され、直後の 13 commit は全部インフラだった。

## 1. 前回運用で実際に起きたこと（証拠は各文書）

| 事象 | 数字 | 出典 |
|---|---|---|
| request→controller | 7h01m（gate 69.7 s、canary 66.0 s） | REQUEST_TO_LAUNCH_RETRO |
| 1 スクリーンの launch 試行 | 17 回（attempt1–11, recovery12–17） | runtime `runs/` |
| 稼働中 DAG の dirty-tree kill | 8 job × 2.7 s | attempt11 `state.json` |
| 連続リカバリ | 6 回、毎回別原因 | recovery12–16 `state.json` |
| 結果の未コミット | 0087 の JSON/MD が runtime に放置 | recovery17 `aggregate/` |
| ルール累積 | AGENTS.md 113→179 行、gate R1–R35 | git log |
| コミット内訳 | fix/harden/record 系 29%、docs のみ 315/503 | git log |
| 検証ゲート | `make lint` 赤（ruff format 78 files, ruff 252, mypy 150） | 実測 |
| 未使用の自動化 | reconciler / publisher / fast path 実行 0 回 | runtime root, branch |

教訓は 5 つに圧縮する。(a) インフラ修理を科学の待ち行列に入れない。(b) 不変ソースから走らせる。
(c) 修正はまとめてから再 freeze する。(d) 完了の定義を 1 つにする。(e) ルールはサブシステムと一緒に引退させる。

## 2. アーキテクチャ

```
 human ◀──── decision packet (docs/decisions/NNNN-*.md) ────▶ Claude (interactive, Hamster)
   ▲                                                              │ /experiment-launch
   │ notify-send / phone (Remote Control)                         ▼
   │                        spec JSON ──▶ launch_gate.py ──▶ orchestrate.py (detached)
   │                                        (pinned worktree)         │ writes
   │                                                                  ▼
   │        ardx-watch.service ──── polls ────▶ <runtime>/runs/*/orchestration/state.json
   │        (systemd --user, linger=yes)                       run-bundle/manifest.json (hand-run)
   │                │ terminal event
   │                ▼
   └──── claude -p "/experiment-postrun <campaign>"  (opus, bounded turns, allowlisted tools)
                    │
                    ▼
        verify artifacts → aggregate → docs/experiments + docs/*.md + .sha256
        → plan milestones/completion → git commit → decision packet → notify
```

セッションが開いていれば同じ event を `Monitor` で受け取って対話的に処理してもよい。
どちらが先に処理しても二重処理にならないよう、postrun は campaign 単位の lock と
`docs/experiments` の非上書き規則で冪等にする。

## 3. コンポーネント

### 3.1 実行プレーンへの追加（`scripts/ardx/`、依存は標準ライブラリのみ）

| ファイル | 役割 |
|---|---|
| `pin_source.py` | `<runtime>/worktrees/source-<sha>` を作成/再利用（clean 検証、`.external`・`teacher_cache` symlink）。全 job の `command[1]`/`cwd`/`PYTHONPATH` はここを指す |
| `campaign_watch.py` | `state.json`（campaign）と `run-bundle/manifest.json`（hand-run）を走査し、遷移ごとに 1 行 JSON を出力。cursor は `<runtime>/orchestration/ardx/watch-state.json`。`--once` と `--follow`。`--on-event CMD` で終端イベント時にコマンド実行 |
| `postrun_hook.sh` | watcher から呼ばれ、`claude -p` を起動。モデル・turn 上限・allowedTools を固定。ログは `<runtime>/orchestration/ardx/claude-runs/` |
| `status.py` | 両ホストの GPU、campaign 状態、hand-run 状態、pending decision、直近 postrun を 1 画面の Markdown に |
| `systemd/ardx-watch.service` + `install_units.sh` | 常駐 watcher。`Restart=always`、Linger 確認付き |

完了の定義（唯一の契約）:

- campaign: `state.json` の `jobs[*].status` がすべて `TERMINAL={completed,failed,blocked,orphaned}`
  になった時点で終端。成功は全 job `completed`。失敗種別は各 job の最終 attempt の
  `failure_class`/`retryable` から `technical_retryable | scientific | unknown` を集約
  （`reconcile_experiment.py:293-327` と同じ優先順位。scientific が優先）。campaign の `status` だけを見ない。
- hand-run（orchestrator 外）: `run-bundle/completion.json` が存在 ∧ `manifest.status ∈ {completed, sync_pending}`
  ∧ `error-marker.txt == "no application error recorded"`。`running` が `--stale-seconds`（既定 3600）を超えたら
  `stale`（`failed` ではない）。
- postrun は上記に加え manifest の `expected_outputs` の存在を確認してから「完了」と言う。

### 3.2 エージェントプレーン（`.claude/`）

| パス | 内容 |
|---|---|
| `CLAUDE.md` | 80 行以内。使命、3 プレーン、日常コマンド、決定プロトコル、モデル方針、コミット方針 |
| `.claude/rules/scientific-core.md`（paths: frontmatter 参照） | 攻撃/正規化/checkpoint/評価の不変条件の短縮版 |
| `.claude/rules/execution-plane.md`（paths: frontmatter 参照） | pinned worktree、fresh attempt dir、単一完了契約、セッション内ポーリング禁止 |
| `.claude/rules/results-records.md`（paths: frontmatter 参照） | record/report/plan/decision の書式と非上書き |
| `.claude/agents/scientific-reviewer.md` | opus, read-only。差分の科学的正しさを 1 回でまとめて指摘 |
| `.claude/agents/bug-investigator.md` | opus, read-only + Bash。`ard-bug-hunt` skill を使う |
| `.claude/agents/mechanical-worker.md` | sonnet, medium。config/docs 同期、fixture、定型 |
| `.claude/agents/upstream-explorer.md` | sonnet。`.external/` の読み取り専用調査 |
| `.claude/skills/experiment-status/` | `status.py` を実行して要約 |
| `.claude/skills/experiment-launch/` | plan → spec → gate（preflight/dry-run/canary/launch）→ watcher 確認。pinned worktree を強制 |
| `.claude/skills/experiment-postrun/` | 完了検証 → 集計 → docs 取り込み → plan 更新 → commit → decision packet → 通知。新しい科学 job は起動しない |
| `.claude/skills/experiment-decide/` | 結果から decision packet を作って止まる |
| `.claude/skills/ard-bug-hunt/` | `.agents/skills/ard-bug-hunt` の移植（references 同梱） |
| `.claude/skills/run-on-ferret/`, `production-launch-gate/`, `multi-gpu-experiment-orchestrator/` | 既存 `.agents/skills/*/scripts` を指す薄い SKILL.md。スクリプト本体は移動しない（テストが参照） |
| `.claude/skills/verify/` | `scripts/verify.py --changed` を正しい Python で実行し要約 |
| `.claude/settings.json` | 許可リスト（adv env の python、verify.py、git 読み取り、ssh Ferret 読み取り、systemctl --user 読み取り）、`SessionStart` hook で `status.py --brief` を注入 |

`.agents/skills/` は Claude Code に読まれない（Codex/agentskills 形式）。SKILL.md は `.claude/skills/` に置く。

### 3.3 人間ループ（`docs/decisions/`）

```markdown
---
id: 0001
status: pending          # pending | decided | superseded
created: 2026-09-05
campaign: ert-i100-online-state-s2-v1
question: 次の一手
options:
  A: ...（何を、何 GPU 時間、何が分かるか、preregistered 判定規則）
  B: ...
recommendation: A
chosen: null             # 人間が A/B/... を書く（またはチャットで指示）
---
（本文: 証拠の要約、数値、リンク。Claude が生成）
```

Claude は `chosen` が埋まるまで新しい科学 job を起動しない。人間はファイル編集でもチャットでも指示できる。

## 4. 運用ルール

正典は `CLAUDE.md` の **Hard rules**（8 条、これ以上増やすときは 1 条減らす）だけ。ここには再掲しない
（二重管理で番号がずれ、skill の「`CLAUDE.md` rule N」引用が壊れた）。ルール番号を引くときは必ず
`CLAUDE.md` の番号を使う。本書で設計した「結果文書は生成物」は `CLAUDE.md` rule 5 に取り込んだ。

## 5. モデル・effort・セッション方針

| 用途 | モデル / effort | 理由 |
|---|---|---|
| 新しい科学的契約の設計、consolidated scientific review、原因不明の失敗 | Fable 5.1 / xhigh | 判断の質が結果を左右する。頻度は週数回 |
| 日常の実装・集計・launch・postrun（対話） | Opus 5 / xhigh（Claude Code 既定） | 単価は Fable の約 1/2。ほとんどの作業はこれで足りる |
| ヘッドレス postrun hook | Opus 5 / high, `--max-turns` 上限あり | 無人。手順が skill で固定されている |
| docs/config 同期、fixture、定型テスト | Sonnet 5 / medium（subagent） | 単価は Opus の約 0.4 |
| 大量読解（コード地図、文献棚卸し） | Opus または Sonnet の並列 subagent | 主スレッドの文脈を汚さない |

- Claude チャットと Claude Code は同じ枠。**壁打ちを別セッションに分ける理由はコストではなく文脈衛生**。
  同じ campaign の議論は同じセッションで続け、campaign が変わったら新しいセッションを開く。長くなったら `/compact`。
- 主スレッドは `/model opus` を既定にし、上表の場面だけ `/model fable` に切り替える。
  `~/.claude/settings.json` の既定は現在 `claude-fable-5-1[1m]` / `xhigh` なので、日常は `opus` に下げる。
- 外出先からは Remote Control（claude.ai/code またはスマホ）で同じセッションに入る。
  クラウドの Routines はローカル GPU に届かないので使わない。
- `/loop` と `CronCreate` はセッションが閉じると消える（最長 7 日）。耐久が必要な監視は watcher に置く。

## 6. 導入順序

- **W-1（初回のみ、手作業）**: リポジトリルートで Claude Code を対話起動し trust dialog を承認する。
  未承認だと `.claude/settings.json` の `permissions.allow` は丸ごと無視され、全 Bash 呼び出しで確認が出る
  （`claude -p` は起動ごとに `Ignoring N permissions.allow entries ...` を stderr に出す）。trust はパス表記
  ごとなので `/home/shunsukenaito/...` 側から入るセッションは別途承認が要る。確認は
  `grep -A2 ard_codex_bootstrap ~/.claude.json | grep hasTrustDialogAccepted`。
  hook と deny ルールは trust 前でも効く。
- **W0（本セッション）**: `.claude/` 一式、`scripts/ardx/` 一式、本書、0087 の記録取り込み
  （aggregator の 2 バグ修正 + 出自記録）、GPU 結合の RNG 復元バグ修正、plan 0087 の完了。
  0087 の取り込みは完了（be29a93）。
- **W1**: ダミー campaign を watcher → headless postrun → decision packet まで通す（CPU のみ、数分）。
- **W2**: 人間が選んだ次の科学 campaign を新ループで実行。
- **後続（別 plan）**: 集計/レポートの共通化（`aggregate_*.py` 13 本が各自 markdown を組み立てている）、
  lint 方針（`ruff format` を一括適用するか、対象を絞るか）、impact map の穴（scripts 84% が全件 fallback）、
  ワークスペース掃除（prunable worktree 12、`.cache/analysis` 146 GB、`a7-mechanism-diagnostic` 213 commits 先行）、
  凍結した制御プレーンの物理削除。

## 7. ユーザー確認が必要な項目

1. §0-5 の凍結対象を実際に削除してよいか（本セッションでは削除しない。ルールからは外す）。
   凍結した制御プレーンは W1 パイロットの完了後に物理削除する予定。
2. 決定済み: 2026-09-05 に `AGENTS.md` / `docs/CODEX_WORKFLOW.md` / `IMPLEMENTATION_PROMPT.md` / `.codex/`
   を削除（git 履歴 d1c0053 以前で参照可）。
3. `ruff format` を全 78 ファイルに一括適用する 1 コミットを許可するか。
4. ワークスペース掃除（worktree prune、`.cache/analysis` の整理）の可否。
5. 0087 の次の一手（decision packet 0001 として提示する）。
