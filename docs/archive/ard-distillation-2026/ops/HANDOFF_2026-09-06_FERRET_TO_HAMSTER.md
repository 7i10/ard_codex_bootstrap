# 引き継ぎ 2026-09-06 — Ferret → Hamster

Hamster 停止中に Ferret 側で行った作業の引き継ぎ。
**Hamster 側の作業を始める前に、この文書の「最初にやること」を上から順に実行すること。**

## 0. 最初にやること（順序を守る）

### (1) Hamster の未 push コミットを push する ← 最優先

2026-09-05 の約 12 コミットが Hamster のローカルディスクにしか存在しない。
`CLAUDE.md`、`.claude/`、`docs/plans/0091`、`docs/plans/0092`、`docs/ARM_REGISTRY.md`、
`docs/MEASUREMENT_DESIGN.md`、`docs/COEFFICIENT_AUDIT.md`、
`docs/EVIDENCE_RECLASSIFICATION.md` などが該当する。

**これらは checkpoint と違い再生成できない。** ディスク 1 本に依存している状態を
まず解消する。認証は次で通る。

```
git -c credential.helper='!gh auth git-credential' push origin <branch>
```

### (2) Ferret 側のブランチを取り込む

```
git fetch origin
git log --oneline origin/research/measurement-standard
```

構成は次のとおり。**すべて新規ファイルなので衝突しない**
（`requirements/environment.lock` のみ変更、これは Ferret で作った
`env/pin-interpreter` 上のファイル）。

```
origin/master (ba767a7)
  └─ env/pin-interpreter (f1bb4ab)   Python インタプリタのピン留め
       └─ research/measurement-standard (1e2331b)  ← 本引き継ぎの成果 7 コミット
```

| コミット | 内容 |
|---|---|
| `4dfd99f` | 測定基準 v1 の制定 |
| `1776b83` | スクリーン判定を既存 validation 分割に変更 |
| `a434e11` | 橋渡し検証を確定ランの副産物に変更 |
| `044e4cc` | plan 0093（`PM(online)` 主要確定）の事前宣言 |
| `b928dc7` | biweekly-review スキルと pptx 生成器 |
| `ee6364f` | environment.lock の教師依存欠落を修正 |
| `1e2331b` | 実データ訓練決定論性の検証記録 |

### (3) Hamster 停止の原因究明と対策 ← 未着手

**この作業はまだ何も行われていない。** 判明している事実のみ記す。

- ping は通り、SSH は Mac からも Ferret からも拒否された
- rpcbind ポートは開いていた
- Hamster → Ferret 方向の SSH は 2026-09-05 17:03 / 17:06 に成功していた
- BMC 経路は未確認

### (4) plan 0092（減衰後床較正）の結果を確認する

2026-09-05 18:01 に Hamster 上で `nohup setsid` で起動した。状態不明（おそらく完了）。
6 本の未処置対照複製、e101–114、endpoint は e104/e109/e114。

**この結果が出るまで、測定基準 §5 の数値と plan 0093 は暫定値のままである。**
σ_d の暫定値 0.35 pp を確定値に置き換え、plan 0093 §3 の検出可能効果を再計算すること。

### (5) `ard-v2` を Hamster で構築し、受け入れ試験を通す

`requirements/environment.lock` から構築し、`pip freeze` を照合したうえで、
**実訓練ジョブを 1 本起動する**。起動すれば合格。

理由は §2 に記す欠陥のため。**この受け入れ試験は、まだどのホストでも通っていない。**

### (6) 成果物の棚卸し

両ホストで `scripts/ardx/status.py --inventory` を取り、差分を
`docs/ARTIFACT_LOCATION_MAP.md` に記録する。分類は
`docs/ARTIFACT_RETENTION_POLICY.md` に従う（再生成不能 / 高コスト / 低コスト）。

W&B へ退避するのは**再生成不能なものだけ**、すなわち
ソース SHA・環境世代・親系譜のいずれかを特定できない過去の checkpoint に限る。

### (7) 親 6 本を env v2 で生成する

plan 0093 の着手前提。1 本 3.5 GPU 時間、計 21 GPU 時間。
既存の dev-1 / dev-2 は旧環境世代なので**使わない**。

---

## 1. 決まったこと

### 命名規則（`docs/MEASUREMENT_STANDARD.md` §1）

```
LOSS(params) @STATE /SELECTOR
```

- 例: `PM(λ=0.054) @S2×T1 /online`（旧 OS-PMP）、`PM(λ=0.054) @S2×T1 /fixed@e99`（旧 DPM）
- 係数は**有効数字 2 桁に丸めて凍結し、丸めた値で走らせる**
- **短縮名 = 正式名から、その比較の中で変化しない欄を落としたもの**
- 「dynamic」は廃止。判定時点は `/online` と `/fixed@eN` でのみ表す
- **登録簿への登録が走らせる前の関門**。既に同じ行があればそれは再実行

### 測定基準（`docs/MEASUREMENT_STANDARD.md`）

- 推論の単位は**親**。標準は親 6 本 × 複製 2 本、親レベル対比較 6 対
- 判定は LR 減衰後
- スクリーンは **validation 5000**（既存の層化分割）、確定は **official test 10000**
- `validation_fraction` と `split_seed` は 1 つの比較の中で必ず同一
- **test を見る回数は事前に宣言**。標準は 1 知見につき 1 回
- 検出したい最小効果を事前宣言し、そこから親と複製の数を導く
- 主要 arm は 1 本だけ指名。副次の陽性は結論ではなく次プランの候補
- 主要 arm のみ λ/2 と 2λ の両側を走らせる
- 非主要 arm が較正点で効かないときは、**機構が作動したか**を見てから捨てる。
  作動していなければ用量不足として 2λ を 1 点だけ追試（工程変数が引き金なので
  「結果を見ての再調整なし」に反しない）
- **スクリーンは確定ランの途中経過**。（スクリーン判定, 確定判定）の対が
  符号一致で 2 組貯まるまで、スクリーンを単独の根拠にしない。**現在 0 組**

### 成果物の保存（`docs/ARTIFACT_RETENTION_POLICY.md`）

「再生成可能／不可能」という分類は誤りだった。訓練は決定論的で環境は固定されているので、
手順が残っている限り checkpoint はすべて再生成できる。軸は**コスト**、判断は**保存対象か否か**。

- 再生成不能 = git 未 push のコミット、来歴を特定できない旧成果物、人が書いた記録
- 再生成高コスト = **該当なし**
- 再生成低コスト = 親 checkpoint（3.5 GPU 時間）、確定ラン（3.5 GPU 時間）
- **記録を git に置くことがバックアップ戦略そのもの**

### 進め方

3 ヶ月の詳細計画は立てない。**常に「今日打ち切っても書ける状態」を保つ**ラチェット方式。
確定が 1 件終わるたびに、その場で記録に閉じてから次へ進む。

| 線 | 日付 |
|---|---|
| 新規実験の投入停止 | 12 月中旬 |
| 最終再現＋成果の固定 | 12 月後半 |
| 修論執筆 | 1 月前半（提出は 1 月中旬） |

最低線: **親が固定環境で揃い、床が測れており、`PM(online)` が宣言した検出力の下で決着している。**
新基準の下では null も結果になるので、ここまで来れば書ける。

隔週レビューは `.claude/skills/biweekly-review/SKILL.md`。発表開始は 10 月頃、日付未定。
**cron は使わない**（quota 共有のため）。スライドは `scripts/ardx/build_slides.py` で
Markdown → pptx。内容は git、見た目は本人所有のテンプレート。

### plan 0093 の主要 arm を PM にした理由

**効果量ではない。** PM の観測値はすべて減衰後の床（0.25–0.50 pp）の内側にある。

| 変種 | 対 Control |
|---|---|
| `PM /fixed@e99` | `+0.08 / +0.12 pp`（e104/e109 では符号反転） |
| `PM /online` | `+0.14 / +0.20 pp`（2 seed, e114, validation CE-PGD20） |

> **訂正 2026-09-06。** この行には当初 `+0.41 / +0.06 / +0.46 pp` と書かれていた。
> それは plan 0091 の **I100 official test AutoAttack** の値（3 seed）であって、
> PM のものではない。数が 3 つある時点で 2 seed 設計と合っていない。
> PM `/online` の実測値は `docs/EVIDENCE_RECLASSIFICATION.md` E1 行の
> `+0.14 / +0.20 pp`。主要 arm の選定理由は効果量ではないと本文が明記して
> いるので結論は変わらないが、検出力を再計算する人を誤らせる。

効果量で選べば `ST1W` の `+1.50 pp` を選ぶことになり、それは再現しなかった。
選定基準は ①エポック時間増分 +1.2%（D-BDD は +7.9%）②反証可能な予測を持つ
③実装が検証済み ④競合が壊れている（S-BDD 非有限、D-BDD 符号割れ）。

**PM は「本命」ではなく「新基準の最初のテストケース」。null は想定された結果。**

過去の状態条件付き処置は 1 つも床を超えていないが、これは「効かない」ではなく
「測っていない」である。2 seed・σ_d = 0.35 pp での検出可能効果は
`0.35 × √(7.85/2) ≈ 0.69 pp`。唯一検出された I100 の `+0.62〜0.78 pp` は
ちょうどこの限界の上にある。0.3〜0.6 pp の帯域は原理的に見えていなかった。

---

## 2. 今日わかったこと

### `requirements/environment.lock` に重大な欠落があった（修正済み）

`ard-v2` で実訓練を起動したら `import timm` で落ちた。`.external/robustbench/data.py`
が要求しているのに lock に 1 行もなかった。教師読み込みの依存が丸ごと抜けていた。

**この lock から作った環境では、どのホストでも 1 つもジョブが動かなかった。**

追加: `pillow` `scipy` `pandas`（runtime）、新設 `[teacher]` に
`timm` `tqdm` `requests` `Jinja2` `gdown` `geotorch` `torchdiffeq`。

受け入れ条件を lock 内に明記した。

> **lock ファイルは、それ単独で構築した環境で実訓練が起動するまで未検証である。**

### 実データ訓練は決定論的（`docs/ERT_RSLAD_REAL_DATA_TRAINING_DETERMINISM.md`）

`I100_CONTROL` を dev-1 親から e100→e114 まで同一 GPU で 2 回。
**model と optimizer の状態が 3 地平すべてでビット単位一致**。
エポック指標は 36 中 33 が完全一致（`train_loss` は最終桁まで同一）。
異なったのは `train_seconds` / `train_images_per_second` /
`train_cuda_peak_reserved_bytes` の 3 つだけで、いずれも計算結果ではない。

**checkpoint のファイルハッシュを再現性の判定に使ってはならない。**

### 実行中のコミットが run 識別子を変える（規則を追加）

2 ランのファイルハッシュは異なった。原因は run A 実行中にリポジトリへコミットしたこと
（run A は `b928dc7`、run B は `ee6364f`）。差分は lock の 25 行のみで
`src/` `scripts/` `configs/` は 0 行だったため重みは一致した。

来歴機構は正しく働いたが、手続きとしては誤り。

> **キャンペーンの実行中は、そのソースツリーにコミットしない。**
> 実行前にソース SHA を固定し、全アームが同じ SHA を記録したことを完了時に確認する。

### Ferret 側の状態

- `.external/da_alone_improves_at` を取得済み（ピン留め `38b740ae`、license verified）。
  これは**出自検証用であって実行依存ではない**（IDBH はリポジトリ内に自前実装）
- **I100 e99 親は Ferret に存在する。** `parent-dev1.pt` の sha256 は
  `360910a8…7630835` で plan 0087 の正典 dev-1 親と完全一致
- I100 系の実測エポック時間は **2.08 分**（361 img/s）。
  Bartoldson 系は 4.37 分（172 img/s）
- `ard-v2`（Python 3.11.15）は timm 欠落のため訓練を起動できない。
  `adv`（Python 3.12.13）は動く
- 決定論性検証の出力は
  `/home/shunsukenaito/workspace-local/ard-runs/ard_codex_bootstrap/determinism-realdata-v1/`

---

## 3. 未検証・未解決のまま残っているもの

| 項目 | 状態 |
|---|---|
| Hamster 停止の原因と対策 | **未着手** |
| Hamster の約 12 コミット | **未 push。最優先** |
| plan 0092 の床 σ_d | 未確認（暫定 0.35 pp） |
| `ard-v2` の受け入れ試験 | どのホストでも未通過 |
| 100 エポック級の決定論性 | 未検証（今回は 15 エポック） |
| ホストをまたいだ決定論性 | 未検証 |
| 成果物の所在表 | 未作成 |
| 親 6 本 | 未生成 |
| スクリーンの妥当性 | 対 0 組。最初の 2 arm は全訓練で判断する |
| `docs/ERT_RESEARCH_STATUS_SUMMARY.md` | 新基準を反映していない |

## 4. 実験は開始しない

親 6 本の生成を含め、**plan 0093 の着手前提がすべて埋まるまで実験を開始しない**
（本人と合意済み）。0 章の (1)〜(6) が終わってから (7) に進む。
