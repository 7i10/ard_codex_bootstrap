---
id: 0006
status: decided
created: 2026-09-07
campaign: alloc-direction-v1（plan 0096、arm `ALLOC_SAFE` と `ALLOC_FRAGILE` が epoch 199 到達、`ALLOC_RANDOM` 実行中）
question: 凍結済みの設定は epoch 114 の checkpoint を保存しない。plan 0096 が求める
  三つの horizon のうち e114 を、これから走る 21 本の arm に足すか、諦めるか、
  それとも終わった 2 本を取り直して 6 親すべてを揃えるか。
options:
  A: e114 を plan から落とし、e149 と e199 の二つの horizon で判定する。GPU 0 時間。決まること = 何も新しくは決まらない。事前規則 = 主判定は元のまま e199 の親ごと対応差、閾値 0.25 pp。
  B: まだ始まっていない 21 本の `checkpoint_epochs` に 114 を足し、終わった p1 の 2 本はそのままにする。GPU 追加 0 時間、endpoint 評価が 21 本ぶん増えて約 2–4 時間。決まること = p2–p6 でだけ e114 が見られるか。事前規則 = 先に「114 を足しても学習が bit 単位で変わらない」ことを回帰テストで示すこと。示せなければ A。
  C: p1 の `ALLOC_SAFE` と `ALLOC_FRAGILE` を 114 込みで取り直し、6 親すべてを同じ checkpoint 表に揃える。GPU 約 4.5–5 時間（学習）＋ endpoint 評価。決まること = 揃うということだけ。科学的な問いは一つも進まない。
recommendation: A
chosen: A
---

## この packet は結果についてではない

plan 0096 の結果はまだ一つも出ていない。比較相手である `ALLOC_RANDOM` が
まだ走っており、held-out CE-PGD20 の endpoint 評価はどの arm でも一度も
走っていない。したがって `docs/experiments/` に取り込むべき record は存在せず、
実際に取り込んでいない。

この packet が扱うのは、`/experiment-postrun` で出力を検証している最中に
見つかった**設計と凍結設定の食い違い**である。

## 何が食い違っているか

plan 0096 の design はこう書いている。

> **Judged at e199, recorded at e149 and e114.**
> Endpoint: held-out CE-PGD20 at all three horizons.

ところが凍結された設定は checkpoint を三つの epoch でしか保存しない。

```yaml
# runs/alloc-direction-v1/arms/p1/fragile/resolved_config.yaml:130-133
training:
  epochs: 200
  checkpoint_epochs:
  - 99
  - 149
  - 199
```

epoch 199 まで終わった 2 本の arm が持っているのは `epoch-149.pt`、
`epoch-199.pt`、`best.pt`、`last.pt` の四つだけで、**`epoch-114.pt` は無い**。
endpoint 評価は保存済み checkpoint を別プロセスで読んで走らせる仕組みなので、
評価すべき重みが無い以上、この 2 本で e114 を測る方法は無い。取り戻すには
その arm を学習し直すしかない。

e199 の主判定と e149 は無傷である。失われるのは三つのうち一番早い horizon
だけで、これは効果の時間的な形を見るために入っていたものである。

## 手元にある数字（結果ではない、checkpoint 選択用の validation）

| arm | mask SHA | best epoch | val clean | val PGD | last epoch | val clean | val PGD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALLOC_SAFE` | `31912f40…` | 178 | 0.8618 | 0.6004 | 199 | 0.8640 | 0.5968 |
| `ALLOC_FRAGILE` | `8cc7f286…` | 188 | 0.8592 | 0.6006 | 199 | 0.8602 | 0.5980 |

これは学習時の validation split の値であって plan の endpoint ではない。
**この表から arm 間の差を読んではいけない。** 親は 1 つ、閾値の基準になる床は
まだ測られておらず、比較相手の `ALLOC_RANDOM` も無い。

固定同一性: CIFAR-10 / `saad_resnet18_cifar_v1` / RSLAD / 教師
`chen2021_ltd_wrn34_10`（SHA `fc398a48…`）/ 学習シード 1 / 評価シード 0 /
linf eps 8/255 step 2/255 / world size 1 / effective global batch 128 /
ソース SHA `815dabd` / 親 `parents-v2-cropshift-s1` の epoch 99
（SHA `03feadbb…`）/ switch epoch 100 / late policy `idbh_weak` /
mask 15,317 枚（両 arm 同数）。

2 本の `resolved_config.yaml` は行単位で比べて三行しか違わない
（mask の digest、`tracking.run_id`、`output_dir`）。dose も class 構成も
設計どおり揃っている。

## e114 はそもそも判定できたのか

ここが推奨の根拠である。

- 軌跡の途中での対照どうしの揺れは **1–2 pp** である（RNG だけで動く幅）。
- I100 自身の効果は e149 で `+0.24` pp、e199 で `+0.69` pp
  （`docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:51,55`）。plan 0096 の design は
  同じ資料を引いて **e114 では効果が無い** と書いている。
- plan 0096 が測るのは I100 そのものではなく**配り方の差**であり、しかも主判定
  `ALLOC_SAFE − ALLOC_RANDOM` は重なり約 47 % のぶん**さらに半分程度に減衰する**
  と plan 自身が事前に宣言している。

つまり e114 で探すことになるのは、**ほぼゼロと予想される差を、1–2 pp の床に対して**
測る作業である。2 seed では原理的に決着しない。e114 の値が仮に取れていても、
それを結果として読むことは事前規則が禁じている。

## 各選択肢

### A — e114 を落とす（推奨）

- **GPU**: 0 時間。
- **決まること**: 新しくは何も決まらない。plan の主判定 e199 と副次の e149 は
  そのまま残る。
- **事前規則**: 変更なし。親ごとの対応差、閾値 0.25 pp（plan 0093 の 12 本の
  対照が床を出せばその値に差し替え、結果を読む前に確定する）。
- **リスク**: 効果の時間的な形を示す材料が一つ減る。ただし上のとおり e114 は
  床を越えない見込みなので、失うのは「決着しない測定」である。

### B — これから走る 21 本にだけ 114 を足す

- **GPU**: 学習の追加は 0 時間（checkpoint の保存は数秒）。endpoint 評価が
  21 本ぶん増え、1 checkpoint 5–10 分として **約 2–4 時間**。
- **決まること**: p2–p6 でだけ e114 が見られる。p1 では見られない。
- **事前規則**: **先に、`checkpoint_epochs` に 114 を足しても学習が bit 単位で
  変わらないことを回帰テストで示すこと。** 示せなければ B は成立せず A に落ちる。
  示せた場合でも、e114 の差は 1–2 pp の床を越えたときにのみ読む。plan の
  事前宣言（e114 では効果が無い見込み）からして越えないほうが自然なので、
  「越えなかった」を結果として書かないこと。
- **リスク**: p1 の 2 本と p2–p6 で checkpoint 表が違うので、`config_hash` が
  親をまたいで揃わなくなる。判定は親ごとの対応差なので分析は壊れないが、
  「全 arm は mask 以外同一」という説明が親をまたいでは言えなくなり、
  査読で一行の言い訳が要る。床を越えない測定のためにその負債を負う。

### C — p1 の 2 本を取り直して 6 親を揃える

- **GPU**: 観測値 81.7 秒/epoch（この campaign の実測、`ALLOC_FRAGILE` は
  06:35:05Z→08:51:17Z で 100 epoch）から、1 本あたり約 2.3 時間、2 本で
  **約 4.5–5 時間**。endpoint 評価が別途。
- **決まること**: checkpoint 表が揃うということだけ。科学的な問いは一つも進まない。
- **事前規則**: B と同じ回帰テストが前提。加えて、取り直した 2 本の e149 / e199 が
  既存の 2 本と床の内側で一致すること（一致しなければ、再現性のほうが
  はるかに大きい問題として別に扱う）。
- **リスク**: 終わった 2 本を捨てるか重複として残すかという整理が要る。
  4.5 時間の GPU を、決着しない horizon の見た目の統一に使う。

## 推奨の理由と、それが変わる条件

**A。** e114 は、この plan の効果量と既知の床を突き合わせると、seed 2 本では
決着しない horizon である。plan 自身が「e114 では効果が無い」と事前に書いており、
配り方の差は I100 の効果よりさらに小さく減衰すると宣言している。決着しない
測定のために、B は親をまたいだ設定の不揃いを、C は 4.5 時間の GPU を払う。
どちらも払う先が測定精度ではない。

**A を変える条件**は二つある。

1. plan 0093 の 12 本の対照が、軌跡途中の床を 1–2 pp ではなく**明確に小さい値**で
   出した場合。そのときは e114 が判定可能な horizon になり、B の価値が変わる。
   結果を読む前に閾値を確定するという plan の手順は保たれる。
2. e199 の主判定が出たあとで「いつ効果が立ち上がったか」が独立した問いとして
   要求された場合。ただしそれは新しい問いであり、新しい plan と新しい contract
   に属する。

なお B の「学習が bit 単位で変わらない」は**未検証の仮定**であり、ここで
真だと主張していない。B か C を採るなら、その回帰テストが最初の作業になる。

## 決定 0004 との関係

決定 0004 も plan 0096 の campaign について `status: pending` のまま残っているが、
問いが違う。0004 は run ID 衝突を起こした `alloc-p1-safe-fork` を再開するか
取り直すか、という問いで、plan の Progress log に書いたとおり**取り直しは
事実としてすでに起きている**（成功した run の ID は `alloc-v1-p1-safe-fork`）。
0004 が今も開いているのは、そこで勧めた preflight の門が未着手だからである。
この packet はその門とは無関係の、checkpoint 表の話である。重複ではない。

## 参照

- plan: `docs/plans/0096-hardness-allocation-direction.md`（Design、M3、Progress log 2026-09-07 `alloc-v1-p1-fragile-fork`）
- 凍結設定: `runs/alloc-direction-v1/arms/p1/fragile/resolved_config.yaml:130-133`
- 終端バンドル: `runs/alloc-direction-v1/arms/p1/{safe,fragile}/run-bundle/manifest.json`
- I100 の horizon 別効果: `docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:51,55`
- 関連決定: `docs/decisions/0004-alloc-p1-safe-fork-run-id-collision.md`

## 決定（2026-09-08、本人）

**A を選択。** e114 を plan 0096 から落とし、e149 と e199 の二つの horizon で判定する。
GPU 追加 0 時間。主判定は元のまま、e199 の親ごと対応差、閾値 0.25 pp。

これにより plan 0096 の horizon 表は e149 / e199 の 2 点となり、既に完走した p1 の 4 アームと
これから終わる 20 アームが同じ表に載る。取り直しは行わない。
