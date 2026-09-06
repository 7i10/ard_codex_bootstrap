# 成果物の所在表

- 状態: 発効（2026-09-06）
- 棚卸し時刻: **2026-09-06 21:09 JST**、リポジトリ HEAD `069ac07`
- 分類は `docs/ARTIFACT_RETENTION_POLICY.md` に従う
- 対象: Hamster（`islab`、2×4090）と Ferret（`islab-3gpu`、3×4090）
- 作業は読み取り専用。削除も移動も行っていない

## この文書の読み方

数値には次の印を付けた。

- **VERIFIED** — この棚卸しで実際に `du` / `sha256sum` / `git` を実行して得た値
- **INFERRED** — 実測値からの推定。特に GPU 時間はすべて推定

保存方針の中心は「再生成できるか」ではなく「**保存する価値があるか**」である。
訓練は決定論的で環境は固定されているので、手順が残っていれば checkpoint は作り直せる。
だから軸はコストであり、この表の要点は「どれが手順ごと失われうるか」に絞られる。

処分区分は 5 つ。

| 区分 | 意味 |
|---|---|
| `KEEP-IRREPLACEABLE` | 生成手順そのものが無い。未 push のコミット、来歴不明の旧成果物、人が書いた記録 |
| `KEEP-CHEAP-TO-REBUILD` | 来歴が揃っている。手順がバックアップなので、消えても作り直せる（再構築コストを併記） |
| `ARCHIVE-TO-WANDB` | 本当に再生成不能で、かつ git に入っていないもの **だけ** |
| `DELETE-CANDIDATE` | 上書きされた、重複している、または純粋な作業くず |
| `DECIDE` | 判断できない。何が分かれば決まるかを明記する |

### 重要な注意（この棚卸しの最中に起きていたこと）

**plan 0092 の M1 が 21:02 に再起動され、棚卸し中も両 GPU で走っていた。**
`runs/post-decay-floor-v1/arms/dev-{1,2}/r1` はいま増え続けている。
この表の `ard-runtime` の数値は 21:09 時点のスナップショットであり、
plan 0092 が終わるまで増える。プロセスには一切触れていない。

---

## 先に結論

| 問い | 答え |
|---|---|
| いま 1 本のディスクにしか無いものはあるか | **ある。Hamster の git コミット 17 本だけ**（`master` 1 本、`a7-mechanism-diagnostic` 16 本）。ほかに別プロジェクト `repo4` の 1 本 |
| dev-1 / dev-2 の親はどこにあるか | 両ホストに 1 部ずつ。**4 ファイルすべて sha256 一致（VERIFIED）**。正典は Hamster |
| `.cache/analysis` 146 GB の中身は | 96% が checkpoint バイト。docs から名前を参照されているのが 124 GiB、参照ゼロが 21 GiB |
| 35 worktree / `/tmp` の 4 本は | `/tmp` は再起動で全滅。**Hamster の worktree 登録 43 件中 23 件が prunable**（うち `/tmp/ard-attack-source*` 4 本すべて）。Ferret は 165 件で prunable 0 |
| plan 0091 / 0092 の成果物は無事か | **両方とも無事。** 0091 は記録が git に入っており出力も残る。0092 の prefix と threshold は生き残り、いま再利用されて走っている |
| このディスクがいま死んだら何が失われるか | **Hamster: 上記 17 コミット（9,260 行の src / tests / docs）と、いま走っている plan 0092 の 2.2 GPU 時間。checkpoint は全部作り直せる。Ferret: 何も失われない（git は全部 push 済み）。** |

---

## 1. ホスト別の総量（VERIFIED）

### Hamster — `/home/islab/workspace-local/shunsuke.naito`

ディスク `/dev/nvme0n1p2` 3.6 TB、使用 1.3 TB、空き 2.2 TB。

| 用途 | パス | 容量 |
|---|---|---|
| ARD リポジトリ本体 | `ard_codex_bootstrap` | **204 G** |
| 過去の run 群 | `ard-runs` | 38 G |
| 過去のキャンペーン群 | `ard-campaign-runs` | 28 G |
| 実行時ルート | `ard-runtime` | 16 G |
| Ferret からの回収物 | `ferret-results` | 1.8 G |
| 解析作業ルート | `ard-analysis` | 1.7 G |
| Codex 期の鑑識用 worktree | `ard_codex_forensic_9fdc5c1` | 25 M |
| **ARD 小計** | | **≈ 290 G** |
| 別プロジェクト | `repo4` | 202 G |
| 別プロジェクト | `repo3` | 155 G |
| データセット（ImageNet 含む） | `datasets` | 149 G |
| **合計** | | **793 G** |

`ard_codex_bootstrap` 204 G の内訳。

| 中身 | 容量 | 備考 |
|---|---|---|
| `.cache` | 147 G | うち `.cache/analysis` が **146 G** |
| `outputs` | 55 G | うち `outputs/scientific` が 55 G |
| `teacher_cache` | 2.3 G | robustbench の教師重み |
| `.mypy_cache` | 52 M | 純粋な作業くず |
| `docs` | 18 M | **人が書いた記録。ここが本体** |
| `.external` | 15 M | 上流 4 リポジトリのピン留め |
| `.git` | 12 M | |
| `src` / `tests` / `scripts` / `configs` | 合計 19 M | |

### Ferret — `/home/islab/workspace-local/shunsuke.naito`（`/home/shunsukenaito/workspace-local` はここへの symlink）

ディスク `/dev/nvme0n1p2` 3.6 TB、使用 1.7 TB、空き 1.8 TB。

| 用途 | パス | 容量 |
|---|---|---|
| run 群（本体） | `ard-runs` | **117 G** |
| ARD リポジトリ本体 | `ard_codex_bootstrap` | 23 G |
| 解析入力 | `ard-analysis-inputs` | 5.1 G |
| 解析出力 | `ard-analysis` | 377 M |
| 解析用 worktree | `ard-analysis-worktrees` | 16 M |
| worktree | `ard-worktrees` | 3.8 M |
| 実行時ルート | `ard-runtime` | 40 K（ほぼ空） |
| 回収物 | `ferret-results` | 16 K（ほぼ空） |
| キャンペーン群 | `ard-campaign-runs` | 0 |
| **ARD 小計** | | **≈ 146 G** |
| データセット | `datasets` | 260 G |
| 別プロジェクト | `repo3` / `repo4` | 44 G / 5.5 G |
| **合計** | | **454 G** |

`ard_codex_bootstrap` 23 G の内訳: `outputs` 12 G、`.cache` 8.9 G、`teacher_cache` 1.6 G、`.git` 33 M、`docs` 18 M。

### 決定的な数値: 研究ルートの 95〜99% は checkpoint バイト（VERIFIED）

| ルート（Hamster） | 全体 | `*.pt` | checkpoint 比率 |
|---|---|---|---|
| `.cache/analysis` | 145 G | 141 G | **96%** |
| `outputs/scientific` | 54 G | 53 G | **99%** |
| `ard-runs` | 37 G | 36 G | **95%** |
| `ard-campaign-runs` | 27 G | 27 G | **99%** |
| `ard-runtime` | 15 G | 15 G | **97%** |
| `ferret-results` | 1 G | 1 G | 81% |

**Hamster の ARD 290 G のうち、約 273 G が checkpoint である。**
残る約 17 G が metrics・parquet・manifest・ログ、つまり来歴と結果を運んでいる部分にあたる。
方針が言う「保存対象」はこの 17 G 側であり、その中核（`docs/` 18 M と `configs/` 600 K）は
すでに git に入っている。

---

## 2. いま 1 本のディスクにしか無いもの

この棚卸しが存在する理由の問い。答えは **git コミット 17 本だけ** である。

| # | 対象 | ホスト | 内容 | 区分 |
|---|---|---|---|---|
| 1 | `master` `069ac07` | Hamster | 「Record the true state of plan 0092 after the host outage」。**リモートは `b656c6b`**、1 本進んでいる | `KEEP-IRREPLACEABLE` |
| 2 | `a7-mechanism-diagnostic` `193485a` 以下 16 本 | Hamster | **upstream ブランチが存在しない**。`git diff` で 11 ファイル・9,260 行追加。`src/ard/cli/ert_cw_a7_*.py` 3 本、`tests/unit/test_ert_cw_a7_mechanism_replay.py`、A7 機構診断のプラン・レポート一式 | `KEEP-IRREPLACEABLE` |
| 3 | `repo4` の未 push 1 本 | Hamster | `ff8f0d0 fix: run_18スクリプト`。ARD とは別プロジェクト | `KEEP-IRREPLACEABLE`（範囲外だが同じ危険） |

検証の方法と結果（VERIFIED）。

- Hamster `git log --all --not --remotes` = **17**
- Hamster の worktree 43 件の HEAD をすべて remote と照合 → **リモートに無いのは main checkout の `069ac07` のみ**
- Hamster に stash なし、未コミットの変更なし
- **Ferret `git log --all --not --remotes` = 0。stash なし。作業ツリーはクリーン。**
  HEAD は `research/measurement-standard` の `7bcf211`（引き継ぎ文書のコミット）で、
  これも `origin/research/measurement-standard` に含まれている。**Ferret は git 的に完全に安全。**

引き継ぎ文書が「約 12 コミットが未 push」と書いていた分は本日 push 済みで、
残ったのがこの 17 本である。**`a7-mechanism-diagnostic` は push 先のブランチすら無いので、
今日の push では拾われていない。これが最大の穴。**

### W&B は remote コピーとして数えられるか — 数えられる（ただし metrics のみ）

`ard-runtime/runs` 配下の `environment.json` を持つ run 55 件、および現在走っている
plan 0092 の run はいずれも **online モード**で `wandb.ai/shunsuke-n-waseda-university/single-teacher-ard`
に同期している。`offline-run-*` ディレクトリは Hamster の `ard-runtime` / `ard-runs` を
探索して **0 件**（VERIFIED）。したがってエポック指標は雲側にも存在する。
ただし W&B にあるのは指標であって checkpoint ではない。

---

## 3. 開発用親 dev-1 / dev-2

4 ファイルすべて `sha256sum` を実行して確認した（**VERIFIED**）。

| 親 | ホスト | パス | サイズ | sha256 | 一致 |
|---|---|---|---|---|---|
| dev-1 | Hamster | `ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed1/s100/epoch-100.pt` | 104,259,237 B | `360910a8…7630835` | ✔ |
| dev-1 | Ferret | `ard-runs/ard_codex_bootstrap/ert-i100-action-transfer-v3/inputs/parent-dev1.pt` | 104,259,237 B | `360910a8…7630835` | ✔ |
| dev-2 | Hamster | `ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed2/s100/epoch-100.pt` | 104,259,301 B | `bb0c7c1a…f7aaf7` | ✔ |
| dev-2 | Ferret | `ard-runs/ard_codex_bootstrap/ert-i100-action-transfer-v3/inputs/parent-dev2.pt` | 104,259,301 B | `bb0c7c1a…f7aaf7` | ✔ |

- **4 部すべてバイト単位で一致している。** 引き継ぎ文書の記載（Ferret に dev-1 がある）は正しく、
  実際には dev-2 も同じディレクトリにある。
- **正典は Hamster 側**（`ert-rslad-stagewise-v1` が生成元のキャンペーン、plan 0087 系譜の起点）。
  Ferret 側は `ert-i100-action-transfer-v3` が入力として複製したもの。
- Ferret の `ard-runs` に `ert-rslad-stagewise-v1` は存在しない（VERIFIED）。
  つまり生成キャンペーンは Hamster にしか無い。
- 同サイズの `.pt` を全ルートから 10 本見つけたが、ハッシュを取った結果 **重複は無かった**
  （`ert-rslad-static-trajstab-v1` と `unseen-confirm-*-prefix-r2` の e099 は別物）。
- 来歴: 生成 SHA・親系譜・seed が `docs/ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV.md` と
  `docs/experiments/ert_rslad_stagewise_augmentation_parent_audit_v1.json` に記録済み。
  **来歴は完全に特定できる。**
- 再構築コスト: **1 本 3.5 GPU 時間**（100 エポック × 2.08 分、VERIFIED な実測レート）。
- 処分: **`KEEP-CHEAP-TO-REBUILD`**。2 ホストに冗長化されており、片方が死んでも残る。
  W&B に退避する必要は無い。
- なお plan 0093 はこの 2 本を使わない（旧環境世代のため）。**新しい親 6 本 = 21 GPU 時間**が
  別途必要で、それはこの表の外側の話である。

---

## 4. 成果物グループ一覧

GPU 時間はすべて **INFERRED**。単価は I100/RSLAD ResNet-18 on 4090 で
**2.08 分/エポック = 0.0347 GPU 時間/エポック**（Ferret 実測、VERIFIED）、
Bartoldson 系は 4.37 分/エポック。run 数は `resolved_config.yaml` の個数で数えた。

### Hamster

| # | パス | 容量 | 中身 | 生成元 | 来歴 | 再構築コスト | 処分 |
|---|---|---|---|---|---|---|---|
| H1 | `ard_codex_bootstrap/docs` `configs` `src` `tests` `scripts` | 37 M | 記録・設定・コード | 全期間 | — | 再生成不能 | `KEEP-IRREPLACEABLE`（うち未 push 17 本が §2） |
| H2 | `ard_codex_bootstrap/.cache/analysis` | **146 G** | 543 run。解析・スクリーン・canary | 2026-07〜09 の全 ERT/FFNR 系 | ディレクトリ名に campaign 名、多くに `resolved_config.yaml` | ≈ 280 GPU 時間 | §5 で分割 |
| H3 | `ard_codex_bootstrap/outputs/scientific` | 55 G | 44 run。history-routing v2、prescriptive v3、PGD-AT/TRADES 対照、Bartoldson 系 | plan 0050 台以前 | 名前に source SHA（`2337b9d` `9792655` `c2220f1` `f0c3ace`） | ≈ 220 GPU 時間 | `KEEP-CHEAP-TO-REBUILD`（大半は `DELETE-CANDIDATE`、§5 と同じ判断） |
| H4 | `ard-runs/ard_codex_bootstrap` | 38 G | 266 run。stagewise、ordering probes、single-switch、unseen-confirm、static trajstab | plan 0060–0087 系 | `ert-rslad-stagewise-v1` 等 8 件に run-bundle。**run-bundle が無い 22 件も `fork-lineage.json` に `fork_git_sha` / `parent_git_sha` / `parent_checkpoint_sha256` を持つ**（確認済み） | ≈ 300–550 GPU 時間 | `KEEP-CHEAP-TO-REBUILD`。ただし dev-1/dev-2 の親（§3）と unseen-confirm 一式は保持 |
| H5 | `ard-campaign-runs/ard_codex_bootstrap` | 28 G | 4 キャンペーン、29 run。`c10-r18-ws1-*`（2026-07-23） | Codex 期の WS1 パイロット | ディレクトリ名に source SHA。`repo/` に当時のソースツリーを同梱 | ≈ 30–100 GPU 時間 | **`DELETE-CANDIDATE`**。`c10-r18-ws1-b128-core-s0-v1` のみ docs から 6 箇所参照。`c10-r18-ws1-pilots-v1-*` 3 本（2.2 G）は参照 0 |
| H6 | `ard-runtime/ard_codex_bootstrap/runs` | 16 G | 116 run。plan 0087 の 18 回の attempt/recovery、plan 0091、plan 0092 | plan 0087 / 0091 / 0092 | 全 run に `environment.json` と source SHA | §7 参照 | §7 で分割 |
| H7 | `ard-runtime/.../worktrees` | 204 M | 固定ソース worktree 7 本（`source-ed3b77daa1de` ほか） | 各キャンペーン | detached HEAD、全部リモートにある | 0（`git worktree add` で復元） | `KEEP-CHEAP-TO-REBUILD`。`source-ed3b77daa1de` は **いま使用中、消すな** |
| H8 | `ferret-results/ard_codex_bootstrap` | 1.8 G | 59 件の Ferret 回収物。model バイナリは既定で除外されている | Ferret 側キャンペーン | 各ディレクトリに source SHA | 0（Ferret に原本あり） | `DELETE-CANDIDATE`（Ferret に原本が残る限り） |
| H9 | `ard-analysis` | 1.7 G | `common-replay-d3c59b1`、`h5-late-d1984cd`、`h5-matrix-6e5dcf5` | H5 期の解析 | 名前に source SHA | ≈ 5 GPU 時間 | `DELETE-CANDIDATE`。`h5-matrix-6e5dcf5` のみ docs 3 箇所参照、他 2 件は参照 0 |
| H10 | `ard_codex_bootstrap/teacher_cache/robustbench` | 2.3 G | 教師 checkpoint（`Chen2021LTD_WRN34_10` ほか） | — | `teachers.lock.yaml` に sha256、robustbench から再取得可能 | **0 GPU 時間**（ダウンロードのみ） | `KEEP-CHEAP-TO-REBUILD` |
| H11 | `ard_codex_bootstrap/.external` | 15 M | 上流 4 本（saad / trades / robustbench / DA-Alone-Improves-AT） | — | `external.lock.yaml` に commit と license 証跡。`da_alone_improves_at` は `38b740ae`、license verified | 0 | `KEEP-CHEAP-TO-REBUILD`。**Hamster にも取得済み**（`DA-Alone-Improves-AT` への symlink） |
| H12 | `ard_codex_forensic_9fdc5c1` | 25 M | Codex 期の鑑識用 worktree、`55fe703` detached | 2026-09-04 | リモートにある | 0 | `DELETE-CANDIDATE` |
| H13 | `ard_codex_bootstrap/.mypy_cache` `.ruff_cache` `.pytest_cache` | 52 M | 型・lint キャッシュ | — | — | 0 | `DELETE-CANDIDATE`（純粋な作業くず） |
| H14 | `.cache/` の analysis 以外（`prescriptive-v3` `history-routing-v2` `schedule-control` ほか） | 約 500 M | 旧解析キャッシュ | H5 / pv3 期 | 名前に SHA | 少 | `DELETE-CANDIDATE` |
| H15 | `repo3` / `repo4` / `datasets` | 155 G / 202 G / 149 G | 別プロジェクトと ImageNet | ARD 外 | `repo3` は完全 push 済み、`repo4` は 1 本未 push | — | **`DECIDE`**。ARD の範囲外だが Hamster の 506 G を占める。**決め手: この 2 プロジェクトを今後使うか。使わないなら `repo4` の 1 コミットを push したうえで両方削除すると 357 G が空く** |

### Ferret

| # | パス | 容量 | 中身 | 生成元 | 来歴 | 再構築コスト | 処分 |
|---|---|---|---|---|---|---|---|
| F1 | `ard-runs/ard_codex_bootstrap` | **117 G** | 160 個の固定ソース worktree 付き run。`c10-r18-ws1-b128-core` 25 G、`history-routing-v2-L3-v7` 16 G、Bartoldson 系 4.4 G ×4、chen-rslad 系、`ert-i100-action-transfer-v{3,5,6}`、`ert-i100-cw-long-horizon-v1` | 全期間の Ferret 実行分 | **各 run に `repo/` としてソースツリーそのものが同梱**。来歴は最も強い | ≈ 600–900 GPU 時間 | `KEEP-CHEAP-TO-REBUILD`。ただし §3 の親と F2 は保持 |
| F2 | `ard-runs/.../determinism-realdata-v1` | 996 M | **実データ訓練決定論性の検証出力。** runA/runB、各 e104/e109/e114 + best/last、`epoch-metrics.jsonl`、run-bundle | plan 外の検証（2026-09-06） | `run-bundle/manifest.json`、source `b928dc7`（runA）/ `ee6364f`（runB）、親 sha256 記録済み | 約 1.2 GPU 時間（35 分 ×2） | **`KEEP-CHEAP-TO-REBUILD`**。結論は `docs/ERT_RSLAD_REAL_DATA_TRAINING_DETERMINISM.md` として git に入っており、**保存方針全体がこの 1 件に依存している**。バイナリ自体は不要だが、再実行が安いので消して構わない |
| F3 | `ard-runs/.../ert-i100-action-transfer-v3/inputs` | 200 M | dev-1 / dev-2 親の複製 + `dev{1,2}.yaml` | plan 0087 系譜 | sha256 一致確認済み（§3） | 3.5 GPU 時間 ×2 | `KEEP-CHEAP-TO-REBUILD` |
| F4 | `ard-runs/.../unseen-confirm-c-*` | 約 1.1 G | plan 0091 の confirm-c 系 | plan 0091 | source SHA あり。Hamster に mirror 済み | 3.5 GPU 時間/本 | `KEEP-CHEAP-TO-REBUILD` |
| F5 | `ard-analysis-inputs/chen-rslad-s0` | 5.1 G | 解析入力の複製 | chen-rslad 系 | 元データが F1 にある | 0 | `DELETE-CANDIDATE`（重複） |
| F6 | `ard-analysis` | 377 M | `rslad-common-trajectory` ほか 4 件 | H5 期 | 名前に SHA | 少 | `DELETE-CANDIDATE` |
| F7 | `ard-analysis-worktrees` / `ard-worktrees` | 20 M | worktree 4 本 | — | 全部リモートにある | 0 | `DELETE-CANDIDATE` |
| F8 | `ard_codex_bootstrap` | 23 G | リポジトリ + `outputs` 12 G + `.cache` 8.9 G + `teacher_cache` 1.6 G | — | git は完全 push 済み | — | `KEEP-IRREPLACEABLE`（git 部分）+ `DELETE-CANDIDATE`（outputs / cache） |
| F9 | `datasets` 260 G / `repo3` 44 G / `repo4` 5.5 G | 310 G | ARD 外 | — | — | — | `DECIDE`（H15 と同じ） |

### `ARCHIVE-TO-WANDB` に該当するもの: **なし**

方針が定める W&B 退避の条件は「ソース SHA・環境世代・親系譜のいずれかを特定できない過去の checkpoint」である。
両ホストを見た限り、**この条件を満たす checkpoint は 1 つも見つからなかった。**

- `ard-runs` の run-bundle 非保有 22 件も `fork-lineage.json` で親系譜と 2 つの git SHA を持つ
- `ard-campaign-runs` と `ard-analysis` はディレクトリ名に source SHA を含む
- Ferret の `ard-runs` は各 run に `repo/` としてソースツリーが同梱されている

したがって **W&B のクォータを checkpoint 退避に使う必要はない。** ただし §8 の環境世代の問題があり、
「環境世代を特定できる」という条件は形式上しか満たされていない点に注意。

---

## 5. `.cache/analysis` 146 GB の内訳

166 個のエントリ。**96% が checkpoint バイト**（141 G / 145 G、VERIFIED）。

判定方法: 各ディレクトリ名（ハイフン形と `_` 置換形の両方）を `docs/` 全体で `grep` し、
1 箇所でも出現すれば「参照あり」とした。

| 区分 | 容量 | 件数 |
|---|---|---|
| docs から名前を参照されている | **124 GiB** | 61 |
| 参照ゼロ | **21 GiB** | 105 |

**この判定は保守的すぎる方向にも甘い方向にも外れうる。** 名前が出てこなくても、
そのキャンペーンを扱った結果文書は存在しうる（例: `ffnr-attack-factorial` は参照 0 だが
`docs/FFNR_ATTACK_FACTORIAL_RESULTS.md` は存在する）。逆に参照があっても、
それは「結果が git に入っている」という意味であって「バイナリが要る」という意味ではない。

### 参照ゼロで 500 MB を超えるもの（合計 18.6 GiB）

| 容量 | ディレクトリ | 性質 |
|---|---|---|
| 4,119 M | `ffnr-causal-pilot-screens-v6` | FFNR 因果パイロットの第 6 版。v1/v2/v5 も並存 |
| 3,281 M | `ert-cw-margin-canary` | **canary** |
| 2,982 M | `ffnr-causal-ce20-checkpoints-wandb` | W&B から引き戻した checkpoint |
| 1,307 M | `ffnr-causal-pilot-screens-v2` | 旧版 |
| 1,161 M | `ffnr-strong-dense` | |
| 994 M | `ffnr-causal-pilot-screens-v1` | 旧版 |
| 780 M | `ffnr-attack-factorial` | 結果文書はあるが名前の参照なし |
| 597 M ×4 | `ert-clean-wrong-broad-v1-canary-C0/C1/C14/C15` | **canary** |
| 596 M | `ert-confirmatory-t123-canary-v2` | **canary** |
| 596 M | `ert-i100-s2-rbp-canary` | **canary** |
| 497 M | `ffnr-causal-pilot-screens-v5` | 旧版 |

### 迷いなく消せる部分（VERIFIED）

名前に `canary` / `smoke` / `REJECTED` を含むディレクトリ **22 件、合計 10 GiB**。
canary は起動ゲートの通過確認であって科学的成果物ではない。
`ffnr-cpu-v1-dev-REJECTED-incomplete-validation`（243 M）は名前が自ら不採用と述べている。

| 処分 | 対象 | 容量 |
|---|---|---|
| `DELETE-CANDIDATE` | canary / smoke / REJECTED 22 件 | **10 GiB** |
| `DELETE-CANDIDATE` | 参照ゼロの残り（旧版スクリーン v1/v2/v5 など） | **11 GiB** |
| `DECIDE` | 参照ありの 124 GiB | — |

`DECIDE` の決め手: **修論に載る図表がどのキャンペーンから来るかを確定すること。**
それが決まれば、そこに寄与しない参照ありディレクトリも checkpoint だけ落とせる。
方針どおり checkpoint は再生成低コストなので、
`epoch-metrics.jsonl` / `*.parquet` / `manifest.json` を残して `*.pt` だけ消すと
**146 G が約 5 G になる**（96% が `.pt` なので）。これがいちばん効く操作である。

---

## 6. git worktree と `/tmp`

### Hamster: 登録 43 件、うち **23 件が prunable**（VERIFIED）

2026-09-05 の監査は 35 件・prunable 12 件だった。
**2026-09-06 の再起動で `/tmp` が消えたため、prunable が 12 → 23 に増えた。**

| 場所 | 件数 | 状態 |
|---|---|---|
| `/tmp/ard-*` | 23 | **全件 prunable**。実体はディスク上に存在しない |
| `ard-runtime/.../worktrees` | 7 | 生存。`source-ed3b77daa1de` は plan 0092 が使用中 |
| `ard-campaign-runs/.../repo` | 4 | 生存（run に同梱されたソース） |
| `ard-runs/.../repo` ほか | 2 | 生存 |
| `.cache/**` | 4 | 生存 |
| `ard_codex_forensic_9fdc5c1` | 1 | 生存 |
| main checkout | 1 | 生存 |
| その他（experiment-results ほか） | 1 | 生存 |

**`/tmp/ard-attack-source*` 4 本（`ard-attack-source`、`-v2`、`-v2-clean`、`-v3`）は
ディスク上から消滅している。** `ls -d /tmp/ard*` は「No such file or directory」を返す。
残っているのは `.git/worktrees/` 側の登録メタデータだけで、
その HEAD（`2cc3ae7` `7f8a13f` `1818adc`）は**すべてリモートに存在する**ので、
失われた作業は無い。

処分: **`DELETE-CANDIDATE`**。`git worktree prune` を 1 回実行すれば 23 件の
登録が消える。実体が無いのでバイトは減らないが、`git worktree list` が読めるようになる。

### Ferret: 登録 165 件、prunable **0 件**（VERIFIED）

内訳は `ard-runs` 配下 160、`ard-analysis-worktrees` 3、`ard-worktrees` 1、main 1。
これは Ferret の run が 1 本ごとにソースツリーを同梱する設計の結果であり、異常ではない。
**全 165 件が生存しており、prune 対象は 1 つも無い。**

---

## 7. plan 0091 と plan 0092

### plan 0091（I100 公式テスト + AutoAttack）— **安全**

| 項目 | 所在 | 容量 | 状態 |
|---|---|---|---|
| 結果記録 | `docs/experiments/ard_i100_official_test_autoattack_v1.json` + `.sha256` | 8.4 K | **git 済み**（`cdaddef`）、リモートにも存在 |
| 報告書 | `docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md` | — | git 済み |
| 証跡台帳の行 | `5260c2d` | — | git 済み |
| 評価出力 | `ard-runtime/.../runs/i100-official-test-v1/eval/` 6 本 | 6.5 M | Hamster のみ |
| mirror した arm | 同 `/arms/` 3 本 | 599 M | Hamster（原本は Ferret） |
| smoke | 同 `/smoke/confirm-b-i100` | 580 K | Hamster のみ |
| ログ | 同 `/logs/` 7 本 | 252 K | Hamster のみ |

**結論: 数値も来歴も git に入っており、ディスクが死んでも結果は失われない。**
評価出力 6.5 MB は saved checkpoint から再実行できる（訓練 0 GPU 時間、
AutoAttack 評価のみ）。処分は `KEEP-CHEAP-TO-REBUILD`。
M4（決定パケット）が未着手なので、それを書くまでは `eval/` を消さないこと。

### plan 0092（減衰後床較正）— **prefix と threshold は生き残り、いま使われている**

| 項目 | 所在 | 容量 | 状態 |
|---|---|---|---|
| e100 prefix（dev-1 / dev-2） | `runs/post-decay-floor-v1/prefix/dev-{1,2}/training/` | 各 301 M | **生存。9/5 18:03 生成、いま再利用中** |
| 凍結 threshold | 同 `/thresholds/dev-{1,2}` + `.sha256` + `.summary.json` | 28 K | **生存。sha256 サイドカーあり**（`49237fc2…` / `b47bee61…`） |
| online-state parquet | `prefix/dev-{1,2}/training/online-state/epoch-100.parquet` | 各 789 K | 生存 |
| 対照複製の arm | `arms/dev-{1,2}/r1/` | 各 222 M | **21:02 起動、実行中** |
| 記録 | `docs/plans/0092-post-decay-floor-calibration.md` の completion report | — | git 済み（`069ac07`、**ただし未 push**） |

**プランの完了報告が「prefix と threshold は生き残り再利用可能」と書いているとおりで、
実際に生きている。** さらに、その再生成が plan 0087 の prefix 計算を
online-state parquet レベルでビット単位再現している（`0bb0701f…70441a` が一致）ことも
記録済みで、これは決定論性の独立した確認になっている。

処分: **`KEEP-CHEAP-TO-REBUILD`（再構築 4 分）。ただし実行中なので触るな。**
キャンペーン全体でも 2.2 GPU 時間しかない。

---

## 8. 見つかった問題: 環境世代が 2 つある

方針が成立する 3 つの前提のうち「環境の固定」に穴がある。

| ホスト | conda env | Python | 使われた場所 |
|---|---|---|---|
| Hamster | `adv` | **3.11.15**（conda-forge） | `ard-runtime/runs` の全 55 run（plan 0087 / 0091 / 0092） |
| Hamster | `ard-v2` | 3.11.15 | 構築中（別セッションが本日 torch を導入中） |
| Ferret | `adv` | **3.12.13**（Anaconda） | 決定論性検証 runA/runB |

**同じ `adv` という名前の環境が、2 ホストで別々の Python である。**
`torch 2.11.0+cu128` と `cuda 12.8` は一致するが、インタプリタ世代が違う。
さらに `platform` は全 run が `Linux-7.0.0-28-generic` を記録しているのに対し、
Hamster の現在のカーネルは `7.0.0-31-generic`（再起動後）である。

含意は 2 つ。

1. **ホストをまたいだ決定論性は未検証のまま**であり（引き継ぎ §3 の記載どおり）、
   この環境差はその検証が必要な具体的な理由になる。
2. 過去の checkpoint はすべて「環境世代を記録している」が、
   その世代を**再現できる保証は無い**（`environment.lock` はまだどのホストでも受け入れ試験を通っていない）。
   方針の言う「記録が欠ければ再生成低コストは即座に再生成不能に落ちる」という
   いちばん脆い前提が、ここで実際に揺れている。

これは処分の判断を変えるものではない（checkpoint はどのみち保存対象ではない）が、
**`environment.lock` の受け入れ試験（引き継ぎ 0 章 (5)）が終わるまで、
この表の `KEEP-CHEAP-TO-REBUILD` はすべて「たぶん安い」であって「確実に安い」ではない。**

---

## 9. 優先行動リスト

| 順 | やること | 効果 | 所要 |
|---|---|---|---|
| 1 | **`a7-mechanism-diagnostic` を push する。** upstream が無いので `git push -u origin a7-mechanism-diagnostic`。次に `master` の `069ac07` を push | 単一ディスク依存が **完全に消える** | 1 分 |
| 2 | `repo4` の未 push 1 本を push する | ARD 外だが同じ危険が消える | 1 分 |
| 3 | plan 0092 の終了を待ち、記録を import してコミットし、push する | 走っている 2.2 GPU 時間が git に固定される | 数時間（放置） |
| 4 | `git worktree prune` を Hamster で実行 | 23 件の死んだ登録が消え、`git worktree list` が読めるようになる | 1 分 |
| 5 | `.cache/analysis` の canary / smoke / REJECTED 22 件を削除 | **10 GiB** 回収。科学的損失ゼロ | 数分 |
| 6 | `.cache/analysis` の参照ゼロ 105 件のうち旧版スクリーンを削除 | さらに **11 GiB** 回収 | 数分 |
| 7 | `environment.lock` の受け入れ試験（0 章 (5)）を通す | §8 の穴が閉じ、この表の `KEEP-CHEAP-TO-REBUILD` が本物になる | 数時間 |
| 8 | 修論に載るキャンペーンを確定し、`.cache/analysis` と `outputs/scientific` から `*.pt` だけを一括削除（metrics と manifest は残す） | **約 190 GiB** 回収。最大の効果 | 半日 |
| 9 | `repo3` / `repo4` / 旧 `datasets` を使うか決める | Hamster 506 G、Ferret 310 G の処遇が決まる | 本人判断 |

**1 と 2 は今日中にやること。それ以外は待てる。**

### 残る最大のリスク

行動 1〜8 をすべて終えたあとに残るのは、**バイトの問題ではなく手順の問題**である。

方針は「記録を git に置くことがバックアップ戦略そのもの」と言い、checkpoint を捨てる
根拠を「手順があれば作り直せる」に置いている。ところがその手順の一部である
**`requirements/environment.lock` は、まだどのホストでも実訓練を起動できていない**
（引き継ぎ §2 で timm 欠落が修正されたばかり）。さらに実際に使われている環境は
2 ホストで Python 世代が違う（§8）。

つまり、**checkpoint を消したあとで lock から環境を再構築できないと判明した場合、
「再生成低コスト」に分類した 273 GB が一括で「再生成不能」に落ちる。**
そのときには消えているので、落ちたことに気づくのは作り直そうとした瞬間になる。

したがって行動 7（受け入れ試験）は **行動 8（大量削除）より前に置かなければならない。**
上の順序はそのように組んである。

---

## 付録: この棚卸しで使ったコマンド

- `python scripts/ardx/status.py --inventory --no-remote` — `--inventory` は実在する。
  ただし出力は「ルートごとの最上位エントリ数と最新 mtime」だけで、
  容量は出さない（ルートが数十 GB あるため意図的に `du` を避けている）。
  したがって本表の容量はすべて手作業の `du -sb` / `du -h -d 1` による。
- `sha256sum` — dev-1 / dev-2 の 4 ファイル、および同サイズ候補 4 ファイル
- `git worktree list` / `git worktree prune --dry-run -v` / `git log --all --not --remotes`
- `find <root> -name '*.pt' -printf '%s\n'` — checkpoint バイト比率
- `find <root> -name resolved_config.yaml | wc -l` — run 数
- Ferret へは `ssh -o BatchMode=yes Ferret` で読み取りのみ
