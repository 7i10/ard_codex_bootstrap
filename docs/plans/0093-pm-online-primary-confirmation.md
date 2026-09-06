# 0093 — `PM(online)` 主要確定（新測定基準の第1号）

## Status

- Owner: Claude
- Base SHA: `1776b83`（`research/measurement-standard`）
- Current milestone: 事前宣言。未着手
- Last updated: 2026-09-06
- 準拠: [`docs/MEASUREMENT_STANDARD.md`](../MEASUREMENT_STANDARD.md)

## Goal

新しい測定基準の下で `PM(λ=0.054) @S2×T1 /online` を決着させる。
本プランは**この処置が効くことを示すため**ではなく、**効くか効かないかを、
宣言した検出力の下で確定させるため**に走らせる。

## 主要 arm を PM にした理由（効果量ではない）

過去の PM の観測値は次のとおりで、**いずれも減衰後の床の内側**にある。

| 変種 | 出典 | 対 Control |
|---|---|---|
| `PM /fixed@e99`（旧 DPM） | BDD recovery, e114 | `+0.08 / +0.12 pp`（e104/e109 では符号が反転） |
| `PM /online`（旧 OS-PMP） | plan 0087 experiment B | `+0.14 / +0.20 pp` |

> **訂正 2026-09-07。** この行には `+0.41 / +0.06 / +0.46 pp` と書かれていた。
> それは plan 0091 の I100 official test AutoAttack の値（3 seed）で、PM のもの
> ではない。plan 0087 は 2 seed 設計なので、数値が 3 つある時点で成立しない。
> 実測値は `docs/EVIDENCE_RECLASSIFICATION.md` E1 行の `+0.14 / +0.20 pp`。
> 引き継ぎ文書は 02f0bb3 で訂正済みだったが、事前登録である本文書が残っていた。

したがって **PM は効果量で選ばれたのではない。** 効果量で選べば
`ST1W` の `+1.50 pp` を選ぶことになり、それは再現しなかった。
観測値による選択は、過去に偽陽性を生んだ手続きそのものである。

事前登録可能な基準で選んでいる。

1. **計算コスト**: エポック時間の増分 `+1.2%`（`D-BDD` は `+7.9%`）。
   主張が「計算コストを維持・改善しつつ」である以上、これは適格性の要件である。
2. **反証可能な予測を持つ**: 機構が「教師の pair margin まで生徒の margin を
   引き上げる」であるなら、`@S2×T1` で最大、`@S2×T2` で減衰、`@S1` で無効ないし有害、
   `@S3` で挙動が変わる、と予測される。格子探索ではなく予測の検証になる。
3. **実装が既に存在し、オンライン router 経路が検証済みである**（plan 0087）。
4. **競合の状態**: `S-BDD` は `NUMERICALLY_UNSUPPORTED`（非有限）、
   `D-BDD` は 2 seed で符号が割れている。

**null は想定された結果であり、失敗ではない。**
現行の証拠は「効かない」ではなく「測れていない」である（§付録）。

## 事前宣言（測定基準 §7）

### 1. arm 一覧

| 役割 | 正式名 |
|---|---|
| 対照 | `I100_CONTROL`（処置なし） |
| **主要** | **`PM(λ=0.054) @S2×T1 /online`** |
| 近傍 | `PM(λ=0.027) @S2×T1 /online` |
| 近傍 | `PM(λ=0.11) @S2×T1 /online` |

λ は較正値 `0.05380932585058825` を有効数字 2 桁に丸めた `0.054` を凍結する。
近傍は `λ/2` と `2λ` を同じ丸め規則で `0.027` と `0.11` とする。

### 2. 主要 arm の指名

主要は `PM(λ=0.054) @S2×T1 /online` の 1 本のみ。
近傍 2 本は**係数頑健性の検査**であって独立の仮説ではなく、
単独で採択判定の対象にしない。

### 3. 検出したい最小効果

**0.30 pp**（official test AutoAttack, robust accuracy）。

根拠: 減衰後の床の測定値（plan 0092）を σ_d とし、両側 α=0.05・検出力 80% で
`n = 7.85·σ_d²/δ²`。

**再計算済み 2026-09-07。** plan 0092 の確定値は **σ_d = 0.0924 pp**（e114、
validation CE-PGD20、自由度 4）。親 6 本での検出可能効果は
`0.0924·√(7.85/6)` = **0.106 pp**。宣言した 0.30 pp を満たすので
**親を増やす必要はない**。

参考までに、暫定値 0.35 pp のままなら 0.400 pp となり宣言を満たさず、
親 11 本（38.5 GPU 時間）が必要だった。床を測ったことで 5 本分が不要になった。

**未解決の設計上の問い（人間の判断が要る）。** 宣言している最小効果 0.30 pp は、
実測されている PM の効果 `+0.14 / +0.20 pp`（平均 0.17 pp）より**大きい**。
つまり宣言どおりなら、効果がこれまで観測されたとおりであっても「検出したい最小効果に
届かない」ことになる。到達可能な分解能は 0.106 pp なので 0.17 pp は解像できる。
0.30 pp を 0.15 pp 前後に引き下げるかどうかは、着手前に決めること。

**この σ_d は e114・validation・CE-PGD20 で測ったものであり、本計画が判定する
e199・official test・AutoAttack の床ではない。** ただし外挿の向きは保守的である。
地平が延びれば床は縮み（0.159 → 0.124 → 0.092 pp）、official test は 5000 ではなく
10000 例なので標本誤差も小さい。どちらも床を下げる向きなので、0.092 pp を使うことは
検出力を過小に見積もる側に働く。e199 と AutoAttack の床は本計画の対照枝 12 本が
副産物として produce する。

### 4. 親と複製

- **親 6 本を env v2 で新規生成**する（既存 dev-1 / dev-2 は旧環境世代のため使わない）。
  生成コストは 1 本あたり 3.5 GPU 時間、計 21 GPU 時間。
- 親ごとに処置枝と対照枝を対で分岐し、`continuation_seed` を枝間で共有する。
- 親ごとの複製は 2 本。
- 検定は親レベルの対比較 6 対（自由度 5）。

### 5. 判定指標

| 段階 | 指標 | データ |
|---|---|---|
| 途中記録 | PGD-20 | validation 5000（既存の層化分割） |
| 確定 | AutoAttack | official test 10000 |

- `validation_fraction` と `split_seed` は全 arm で同一に固定する。
- **official test を見る回数は 1 回**。主要 arm と対照の 24 ランに対してのみ。
  近傍 2 本は validation で判定し、test を見ない。
- 標準精度と実測エポック時間を必ず併記する。
- スクリーンの妥当性はまだ 0 組なので、**主要 arm と対照は e199 まで走らせる**。
  その過程で（e114 判定, e199 判定）の対を 1 組記録する（測定基準 §3.1）。

### 6. 採択規則

主要 arm は次を**同時に**満たすとき採択する。

- 親レベル対比較の平均差 > 0、かつ 95% 信頼区間の下限 > 0
- 標準精度の低下が 0.5 pp 以内
- エポック時間の増分が 2% 以内

いずれかを欠けば不採択。効果量と信頼区間を必ず併記し、p 値のみの報告はしない。
近傍 2 本で符号が保たれない場合、採択されても「係数に対して脆い」と明記する。

### 7. 機構作動の測定量と閾値

測定量: **選択サンプル（online S2×T1 分岐）上の生徒 adv pair-margin の中央値**を、
処置枝と対照枝で比較した差。

- 差が対照の分布の中央値から **0.02 以上**動いていれば「機構は作動した」
- 動いていなければ「用量不足」とみなし、`2λ` を 1 点だけ追試する

主要 arm は近傍を既に走らせるため、この門は `@S2×T2` 以降の非主要 arm に適用する。
閾値 0.02 は plan 0092 の対照複製から得る margin 中央値のばらつきに対して
再検討し、着手前に確定させる。

### 8. 副産物として必ず記録すること（2026-09-07 追加）

以下はいずれも**訓練を追加せず、評価の数分で得られる**。着手後に思いついても
取り直しになるので、事前登録に含める。

1. **対照 12 本を e114 / e129 / e149 / e154 / e174 / e199 の 6 地平で評価し、
   per-sample の行を保存する。** 宣言している 2 地平だけにしない。
   これは自由度 5 の **fork 床を各地平で** 与える唯一の機会であり、
   **第 2 減衰（e150）をまたいで床が縮み続けるのか戻るのか**が分かる。
   plan 0092 が測ったのは e104–e114 だけで、e199 の床は誰も持っていない。
2. **e149 のチェックポイントを保存する。**
3. **全 arm の `continuation_seed` を記録に書く。** plan 0087 の記録には無く、
   そのせいで後から複製の対応を数字の一致から推測する羽目になっている。
4. **各地平の標準精度の床を同じ run から計算する。** 採択条件の
   「標準精度の低下 0.5 pp 以内」に、初めて実測の分母がつく。
5. **処置枝と対照枝の行から、S2 部分集合に限った保留効果を計算する。**
   部分集合は親ごとの凍結閾値が定義している。

これらは方向性の提案ではなく、**同じ GPU 時間から取り漏らさないための指定**である。

## 禁止事項

- 結果を見ての係数再調整
- 親の追加・除外を結果を見てから行うこと
- official test の 2 回目以降の参照
- 本プラン内での `@S2×T2` `@S1` `@S3` `@CW` への拡張
- 環境世代 v2 以外での実行

## 着手前提

| 前提 | 状態 |
|---|---|
| Hamster 復旧と未 push コミットの push | 済（2026-09-06。master 25 件に加え、上流を持たない `a7-mechanism-diagnostic` 16 件も push。どのリモートにも無いコミットは 0 件） |
| plan 0092 の床確定（σ_d） | 実行中（2026-09-06 21:02 起動、未処置対照 6 本、e101–114） |
| env v2 を Hamster で構築し `pip freeze` を `requirements/environment.lock` と照合 | 済（2026-09-06。21 項目一致、受け入れ試験通過。Ferret では未構築） |
| ランタイム root v2 の作成 | 未 |
| 親 6 本の生成 | 進行中（2/6 完了。§Progress log 参照。**待ち受けのパス欠陥は修正済みだが、`materialize_chain.sh:51` が呼ぶスクリプトを間違えており 6/6 到達後に全滅する。`parent.sh:27` も未修正。seed 3/4/5 は未確認**） |

## Progress log

### 2026-09-07 — 親 seed1 完了、ただし材料化は現状のままでは必ず失敗する

**seed1 は正常終了した。** `parents-v2-cropshift-s1`、200/200 エポック、
source SHA `6ab179d`、env v2（`ard-v2`）、world size 1、global batch 128。
`epoch-049/099/149/199.pt`、`best.pt`、`last.pt`、`epoch-metrics.parquet`、
`sample-stats-train.parquet` がすべて存在する。
best 5000-validation CE-PGD20 = 59.40 %（e198）、last = 59.30 %、
clean は best 86.12 % / last 86.26 %、robust overfit gap 0.10 pp。
**これは validation の途中記録であり official test ではない。**

**阻害欠陥（実行系。科学的な欠陥ではない）。**
材料化スクリプトが探すチェックポイントのパスが、実際に書かれるパスと一致しない。

| 参照箇所 | 探しているパス | 実際に存在するパス |
|---|---|---|
| `materialize_chain.sh:21`（ローカル seed 1/2/6） | `<out>/checkpoints/epoch-99.pt` | `<out>/epoch-099.pt` |
| `materialize_chain.sh:23`（Ferret seed 3/4/5） | `<out>/checkpoints/epoch-99.pt` | 同上 |
| `parent.sh:27`（完了スキップ判定） | `<out>/checkpoints/epoch-199.pt` | `<out>/epoch-199.pt` |

ずれは 2 か所ある。`checkpoints/` という中間ディレクトリは存在せず、
ファイル名は 3 桁ゼロ詰めである。根拠は `src/ard/engine/trainer.py:1538`
（`self.output_dir / f"epoch-{epoch + 1:03d}.pt"`）と、seed1 の実ファイル一覧
（`checkpoints/` は空、`epoch-099.pt` は存在）。VERIFIED。

**帰結。** `have()` は 6 本すべてについて空を返し続ける。したがって
`materialize_chain.sh` は 12 時間の期限（2026-09-07 13:17 JST）まで 0/6 のまま
待ち、`gave up after twelve hours (0/6)` を出して exit 1 する。
epoch-99 のチェックポイントがディスク上に揃っていても、親は 1 本も材料化されない。
**plan 0096 の M0 と本プランの着手は、訓練が全部終わっても自動では解けない。**

**未確認。** seed 3/4/5 は Hamster では走っていない（GPU0 が seed 1→6、GPU1 が
seed 2）。`materialize_chain.sh` はこの 3 本を Ferret に期待しているが、
Ferret への ssh は本セッションで承認されず確認できていない。
仮に Ferret で走っていなければ、パスを直しても 6/6 には到達しない。
**着手前提の表に「Ferret では env v2 未構築」とあるため、ここは実際に確認が要る。**

**提案する修正（未実行。実行は人間の判断）。**
実行中の bash はスクリプトを再読するため `parent.sh` は走行中に編集しない。
`materialize_chain.sh` は関数を読み込み済みなので、編集だけでは効かず再起動が要る。

```bash
CAMP=/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/parents-v2
kill 111500 \
  && sed -i 's#/checkpoints/epoch-99\.pt#/epoch-099.pt#g' "$CAMP/materialize_chain.sh" \
  && nohup "$CAMP/materialize_chain.sh" "$CAMP" \
       /home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap/worktrees/parents-6ab179d4d76d \
       /home/shunsukenaito/.conda/envs/ard-v2/bin/python >/dev/null 2>&1 &
```

`parent.sh:27` は別件として、次に走らせる前に `<out>/epoch-199.pt` へ直す。
現状ではスキップ判定が一度も成立しないため、`parent.sh` を再実行すると
**完了済みの seed1 を先頭から訓練し直して上書きする。**

**GPU ジョブはこのセッションからは一切起動していない。**

### 2026-09-07 — 親 seed2 完了、待ち受けは直ったが材料化は呼ぶ先を間違えている

**seed2 は正常終了した。** `parents-v2-cropshift-s2`、200/200 エポック、
source SHA `6ab179d`（作業ツリーの差分なし）、env v2、world size 1、
per-rank batch 128、global batch 128、teacher `chen2021_ltd_wrn34_10`
（SHA `fc398a48…`、宣言値と実測値が一致）。
`epoch-049/099/149/199.pt`、`best.pt`、`last.pt` がすべて存在する。
run-bundle の 2 成果物（`epoch-metrics.parquet` = `a68556ba…`、
`sample-stats-train.parquet` = `78c5e15e…`）も両方あり、
バンドルは内容アドレス方式の控えを同じハッシュ名で持っている。
所要時間は 4 時間 38 分（15:31:42Z → 20:10:24Z）。

| | seed1 | seed2 |
|---|---|---|
| best PGD | 59.40 %（e198） | **59.50 %**（e178） |
| last PGD | 59.30 % | **59.12 %** |
| best clean | 86.12 % | **85.88 %** |
| last clean | 86.26 % | **86.30 %** |
| robust overfit gap | 0.10 pp | **0.38 pp** |

**これは 5000 枚 validation の CE-PGD20（ε=8/255、step 2/255、20 step、
random start、batch keying）の途中記録であり、official test ではない。**
seed は 2 本しかないので、これは 2 本ぶんの方向の記録であって、
分布の推定でも母集団の主張でもない。

**材料化の待ち受けは直っている（新情報）。** `materialize_chain.sh:21-23` の
`have()` は現在 `$CAMP/seed$1/epoch-099.pt` を見ており、前回記録した
`checkpoints/epoch-99.pt` ではない。ログは
`2026-09-07T05:07:45+09:00 waiting for six epoch-99 checkpoints` で再起動を示す。
**12 時間の新しい期限は 2026-09-07 17:07:45 JST。**

**`parent.sh:27` は未修正のまま。** いまも
`[ -f "$OUT/checkpoints/epoch-199.pt" ]` を見ており、`checkpoints/` は空なので
スキップ判定は一度も成立しない。**`parent.sh` を再実行すると、完了済みの
seed1 と seed2 を先頭から訓練し直して上書きする。** 次に走らせる前に
`$OUT/epoch-199.pt` へ直すこと。

**しかし待ち受けの先に第 2 の閉塞欠陥がある。`materialize_chain.sh:51` は
呼ぶスクリプトを間違えている。** 6/6 が揃った直後に、6 本すべてが例外で落ちる。

`epoch-099.pt` という名前のファイルは、**payload epoch 98**（global_step
34848 = 99×352）の状態を持つ。トレーナが `epoch-{epoch + 1:03d}.pt` で書く
ためで（`src/ard/engine/trainer.py:1538`）、
`scripts/analysis/materialize_stagewise_parents.py:130` の
`expected_source_payload = source_label - 1` も同じ規約を独立に述べている。
一方 switch=100 のフォークが必要とするのは **payload epoch 99**（global_step
35200）である。

chain が呼ぶ `scripts/create_stagewise_augmentation_forks.py` は、この 1 エポック
の差を埋めない。埋めないどころか明示的に拒否する。

```
:116  expected_epoch = args.switch - 1                      # = 99
:117  if parent.get("epoch") != expected_epoch or parent.get("epoch_boundary") != "end":
:118      raise ValueError("parent payload epoch does not match the requested switch boundary")
:122  if parent.get("world_size") != 1 or parent.get("global_step") != (args.switch * 352):
:123      raise ValueError("parent world size/global step is inconsistent ...")
```

`epoch-099.pt` は epoch=98、global_step=34848 なので、:117 で必ず落ちる。
**このツールはラベルを付け替えるだけで、継続訓練をしない。**

隙間を埋めるのは `scripts/analysis/materialize_stagewise_parents.py` のほうで、
これは最も近い疎チェックポイントから CropShift を境界の直前まで継続する
（同ファイル :4-7、:49-51）。コミット `d7c05d7` がこのツールを
**任意の seed・任意の source run へ拡張済み**であり、
`--source-root` と、バンドル外に置かれた `resolved_config.yaml` への
フォールバックが入っている。**閉塞の解除に必要な変更は、すでに master にある。**

**必要な修正（未実行。実行は人間の判断）。** `materialize_chain.sh:51-54` を
差し替える。出力先 `$CAMP/parents/seed$s/s100/epoch-100.pt` は変わらないので、
:49 のスキップ判定と :62 の SHA 出力はそのまま使える。

```bash
PYTHONPATH="$WT/src" "$PY" scripts/analysis/materialize_stagewise_parents.py \
  --seed "$s" --boundary 100 \
  --source-root "$CAMP/seed$s" \
  --output-root "$CAMP/parents" >"$CAMP/logs/materialize-seed$s.log" 2>&1
```

`--device` の既定は `cuda` で、材料化は 1 エポックぶんの継続訓練を行うため
**GPU を使う**。いま空いている Hamster GPU1 で賄える規模である。
seed2 の `resolved_config.yaml` は run 直下にもバンドル内にも存在するので、
上記フォールバックはどちらでも解決する。

**残る不確実性は 2 つ。**

1. **seed 3/4/5（Ferret）は今回も未確認。** ssh は本セッションでも承認されず、
   実行していない。セッション開始時のフックが Ferret の 3 GPU を
   41〜52 %・2319〜2533 MiB と報告しており、seed2 の実測ピーク
   （reserved 1.91 GB）と矛盾しない規模ではあるが、**走っているジョブの同定には
   ならない。** 3 本が実在しなければ、期限までに 6/6 には到達しない。
2. **seed6 は間に合う見込み。** 05:09 JST 時点で epoch 2 / global_step 1056。
   seed2 の実測 83.6 秒/エポックで外挿すると epoch 99 到達は **07:25 JST 前後**で、
   17:07 の期限には余裕がある。これは外挿であって観測ではない。

**Hamster の GPU1 は空いた。** GPU1 の `parent.sh` は seed2 だけを渡されており、
後続がない。GPU0 は seed6 を走らせている。

**このセッションからは GPU ジョブを起動も再試行もしていない。**

## 付録 — 過去の null が「効かない」を意味しない理由

過去の状態条件付き処置はすべて 2 development seed で判定されていた。
σ_d = 0.35 pp、n = 2 のとき検出可能効果は

    δ = 0.35 × √(7.85 / 2) ≈ 0.69 pp

**つまり過去のパイプラインは 0.7 pp 未満の効果を原理的に見つけられなかった。**
そして実際に検出された唯一の効果は `I100` の `+0.62 〜 +0.78 pp`（unseen 確認）で、
ちょうどこの検出限界の上にある。

0.3〜0.6 pp の効果は、存在していたとしても過去の設計では見えない。
本プランはその帯域を初めて可視化する。
