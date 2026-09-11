# バックログ

**この文書の目的は、GPU と私の手が空いたときに「次に何をするか」を人間に聞かずに
決められるようにすることである。**

そのために、項目を 2 段に分ける。**この区別が本文書の全体である。**

- **A 段：断りなく着手してよい。** 基盤整備、検証、記録の整理、既に事前登録済みの
  作業。科学的な判断を含まない。
- **B 段：決定パケットが先に要る。** 新しい arm・seed・地平・official test・採用。
  `CLAUDE.md` 規則 7 により、これらは人間のものである。**空き時間を埋めるために
  B 段を始めてはならない。**

A 段が尽きたときは、B 段を始めるのではなく**決定パケットを書いて止まる**。

## 運用規則

1. 上から順に、着手条件を満たす最初の A 段項目を取る。
2. 1 件終わるごとに記録・報告・計画を閉じてコミットする。半端な状態を積まない。
3. GPU が空いていて A 段に GPU 作業が無ければ、A 段の机上作業をする。
4. 項目が完了したら**この表から消す**。完了記録は git 履歴と当該文書にある。
5. 新しく分かったことで項目が無意味になったら、消して理由を 1 行残す。

---

## A 段 — 断りなく着手してよい

| # | 項目 | 効果 | 費用 | 着手条件 |
|---|---|---|---|---|
| A1 | `docs/archive/ard-distillation-2026/NUMERIC_CONSISTENCY_AUDIT.md` の残りを修正する。**#1・#2 は対処済み**（plan 0093 の訂正、+0.864 pp の 4 文書への注記）。#3 以降が残り | 意思決定を駆動する文書の数値が記録と一致する | 机上 半日 | なし |
| A2 | ~~C0/C10/C12/C13 表の食い違い~~ **完了 2026-09-07。** 片方に replay の来歴が無いことが原因と判明。stratified 側を正典とし、subtypes 側に注記。再生成すれば確定するが急がない | — | — | — |
| A3 | `ruff format` を 78 ファイルに適用する | 差分ノイズが減り、レビューが読みやすくなる | 机上 1 時間 | 走行中のキャンペーンが無いこと（フォーマットはソース SHA を変える） |
| A4 | impact map の穴を塞ぐ。現在スクリプトの 84% が全テスト実行に落ちている | 検証時間が短くなり、`/verify --changed` が意味を持つ | 机上 半日 | なし |
| A5 | 作業領域の掃除。刈れる worktree が 23 個、`.cache/analysis` が 146 GB でうち約 10 GiB は明白な一時物 | ディスクと、成果物の所在の見通し | 机上 1 時間 | 走行中のキャンペーンがその worktree を使っていないこと |
| A6 | 凍結済み制御面サブシステムの物理削除（reconciler、PR #1 イベントバス、fast path、`launch_ledger.py`、`task_context.py`）。文書は退役済みだがコードは残っている | 呼ばれてはならないものが呼べる状態を解消する | 机上 半日 | A4 の後（impact map が参照している可能性） |
| A7 | 再生成不能な成果物の W&B 退避。対象は「ソース SHA・環境世代・親系譜のいずれかを特定できない過去の checkpoint」に限る | 単一ディスク依存の解消 | 転送のみ | A5 の後 |
| A8 | `src/ard/cli/evaluate.py`（best/last チェックポイント間、可能なら AutoAttack の各段間）に `torch.cuda.empty_cache()` を追加する。原因は `docs/debugging/0029-adr-cifar10-eval-oom-cascade.md` — PyTorch のキャッシュアロケータが断片化し、AutoAttack評価が実行時間とともに（アーキテクチャ非依存で）13〜15GB まで肥大化し、plan 0097 で複数回 OOM クラッシュを起こした。**訂正 2026-09-11**: `docs/decisions/0010-autoattack-unbatched-forward-oom.md` が実測で示した通り、**これだけでは今回のクラッシュ経路を塞げない**——落ちたプロセス自身の「解放済みだが未割当」領域は 58〜110 MiB しかなく、必要な 2.44〜3.66 GiB（`autoattack.py:213` の一括 forward、`10000×channels×32×32×4B`）には二桁足りない。A8 が縮めるのは自分自身の断片化（供給側）、パケット 0010 の Option A（同じ行をバッチ化）が縮めるのは一括確保そのもの（需要側）——**両方揃って初めて閉じる可能性がある。片方だけで直ると読んではいけない** | 計算結果（数値・RNG・攻撃挙動）に一切影響しない——キャッシュ解放のみ。単独では今回の OOM 経路を防げないが、断片化由来の別の劣化は減らせる | 机上 1 時間 + 単体テストでの確認 | 走行中のキャンペーンが無いこと（評価コードの変更はソース SHA を変える）。plan 0097 の評価フェーズ完了後。**decision packet 0010（`chosen: null`）と対で判断すること** |

## B 段 — 決定パケットが先に要る

| # | 項目 | 何を決める必要があるか | 費用 | 前提 |
|---|---|---|---|---|
| B1 | `docs/archive/ard-distillation-2026/METHOD_DIRECTIONS.md` 方向 1 の段階 2（状態条件付き処置の特異性、確定実験） | 走らせるか。段階 1（plan 0095）の結果で内容が変わる | 13 GPU 時間 | plan 0095 の完了 |
| B2 | 同 方向 2（I150 を親 6 本で。再分類 A3 行の再測定） | 走らせるか。**k=2 では既に測ってあり**、当時の床 0.40 pp に埋もれていた | 31 GPU 時間 | plan 0093 の e199 床（副産物 1） |
| B3 | 同 方向 3（後期方策の分解：測光操作か erasing か「変えること」か） | 走らせるか。B 自身が「なぜ今か」が最も弱いと認めている | 31〜92 GPU 時間 | plan 0093 の e199 床 |
| B4 | 決定パケット 0001 の選択肢 D。8-cell コホート表の 3 seed 化 | 走らせるか。**論文の主表を 1 seed のまま載せる危険が残っている** | 40 GPU 時間/seed | なし。いつでも判断できる |
| B5 | `/online` コホートのプラセボ | 走らせるか。online ルータにランダムコホート機能が無いので実装も要る | 3 GPU 時間 + 実装 | plan 0093 の報告 |

## 現在走っているもの

| 何 | 状態 |
|---|---|
| 親 6 本（CROPSHIFT 200 エポック、env v2、seed 1–6） | 2026-09-07 00:31 開始、5 枚並列、約 8 時間 |
| plan 0093 | 親の完了待ち。事前登録は副産物を含めて確定済み |
| plan 0095（コホート・プラセボ） | 実装中（選択器 seed の引数化、マスク配線、arm 登録） |

## B-tier item added 2026-09-09: rename the `ard` package/paths

The research direction pivoted away from Adversarial Robustness Distillation
to mobile-scale ImageNet robustness via teacher-free self-distillation. The
package name `src/ard/` (and every `from ard.x import y` across the codebase,
roughly 460+ files) is a naming relic of the abandoned direction and will
read as noise to anyone reading the eventual published research artifact.

This needs a decision packet before it starts (a rename this size touches
every Python file's imports, every config, every script, every doc still
live at the top level, and every CI/test invocation) -- not a drive-by change
folded into an unrelated implementation commit. Do it as its own dedicated
pass: pick a new name, one mechanical find-and-replace commit, full test
suite green before and after, nothing else in the same commit.

Cheap and separate: `CLAUDE.md`'s own self-description ("Single-teacher
Adversarial Robustness Distillation (RSLAD family...)") is now stale
regardless of the rename question and should be corrected to describe the
current direction.

Not in scope for this item: `docs/archive/ard-distillation-2026/` is named
that deliberately, to label it as the ARD-era historical record -- it should
keep the name it has.

## B-tier item added 2026-09-10: retire `adv`, standardize on `ard-v2`

Both hosts' default conda environment (`adv`, referenced by CLAUDE.md,
`docs/FERRET_EXECUTION_PROTOCOL.md`, `docs/TEST_STRATEGY.md`, and every
hand-run driver script) has drifted from its Hamster/Ferret parity: a live
`pip freeze` diff (2026-09-10) found `timm` at 1.0.27 on Hamster vs 1.0.9 on
Ferret, and `robustbench` installed two different ways (editable git checkout
vs a plain pip package). Core numerics-critical packages (`torch`,
`torchvision`, `numpy`, `autoattack`'s pinned commit) still match exactly, and
neither `timm` nor `robustbench` is imported anywhere under `src/ard/`, so
this drift did not affect the ADR CIFAR-10 replication campaign (plan 0097) --
but it is exactly the failure mode `ard-v2` (`requirements/environment.v2.freeze.txt`,
created 2026-09-06, plan 0094) exists to prevent, and it will recur in `adv`
indefinitely since `environment.lock` only pins direct dependencies.

`ard-v2` is already installed on both hosts and independently confirmed
bit-identical between them (a live `pip freeze` diff found nothing but a
`packaging` metadata-representation quirk, same version both sides). Plan
0094 already established that switching from `adv` to `ard-v2` does not
change training determinism (bit-identical checkpoint component hashes).

**Do not switch mid-campaign.** `docs/MEASUREMENT_STANDARD.md` forbids a
comparison spanning two environment generations; plan 0097's evaluation
phase is still using `adv`. Switch after plan 0097 reaches M3, as a single
dedicated commit: update every `python` path reference (CLAUDE.md,
`docs/FERRET_EXECUTION_PROTOCOL.md`, `docs/TEST_STRATEGY.md`, `.claude/skills/`,
scratch/production driver scripts) from `envs/adv/` to `envs/ard-v2/`, mark
`adv` deprecated (not deleted immediately, in case of rollback need), and
remove it once nothing has needed it for a while. Natural pairing: bundle
this with the ImageNet Stage 0 prep work already gated behind the same
plan-0097 M3 decision.

Also on 2026-09-10: found and removed an entirely unreferenced `saad-oracle`
conda environment (zero hits anywhere in the repo). Its near-namesake,
`saad-oracle-py311`, is real and load-bearing -- it's the Python-3.11
runtime for the pinned upstream SAAD reproduction configs
(`configs/upstream/saad_*.yaml`) and is asserted directly in
`tests/unit/test_run_saad_upstream.py:112` -- but had no documentation
anywhere explaining what it is or why it's separate from the main engine
environment, which is how it looked like more of the same clutter at first
glance. Worth a one-line mention somewhere discoverable (this entry is that,
for now) rather than leaving it to be rediscovered by grep again later.
