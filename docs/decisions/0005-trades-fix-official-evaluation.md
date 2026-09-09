---
id: 0005
status: decided
created: 2026-09-07
campaign: trades-fix-v1-s0-attempt2（手動ラン、plan 0027 B4 / docs/debugging/0028 の確認ラン）
question: 修正後 TRADES の validation は欠陥版より robust `+3.14 pp` 上がった。
  ダッシュボードの official TRADES 行（AA 45.14%）を置き換えるための official 評価を
  いま走らせるか、seed をもう 1 本足してから走らせるか、当面やらないか。
options:
  A: 修正 run の best.pt / last.pt に対し official saved-checkpoint 評価（clean + CE-PGD-20）と、別プロセスの AutoAttack（`--allow-autoattack`）を実行する。約 3--6 GPU 時間（推定）。決まること = 修正後 TRADES が文献帯 49.0--49.4% AA に入るか、つまり detach 欠陥が 4 pp の全額だったか。事前規則 = 下記 A の規則。
  B: A に加えて、修正後 TRADES を seed 1 でもう 1 本回してから両方を official 評価する。A + 約 3 GPU 時間（実測）+ 評価分。決まること = `+3 pp` が修正の効果か run 間ばらつきかを 2 seed で切り分ける。事前規則 = 下記 B の規則。
  C: 当面 official 評価をしない。validation の証拠だけを残し、ダッシュボードの行は「superseded だが未置換」のまま据え置く。0 GPU 時間。決まること = 何も決まらない。事前規則 = なし。
recommendation: A
chosen: A
---

## 何が分かったか

修正後の TRADES seed 0（`trades-fix-v1-s0-attempt2`、SHA `ee9ced0`、clean checkout）が
200/200 エポックを完走した。証拠は
[`plan 0027 の B4 completion report`](../plans/0027-controlled-teacherless-baselines.md)
にある。集約レコード（`docs/experiments/*.json`）は無い。この手動ランには contract 名も
aggregator も宣言されていなかったためで、plan 0027 の baseline は以前から plan の
progress log とダッシュボードで記録されている。

**すべて validation の値であり、official test ではない。** 同一 config、同一 protocol、
同一 seed で、違うのは目的関数だけ。

| validation | 欠陥版 (`f0c3ace`) | 修正版 (`ee9ced0`) | 差 |
| --- | ---: | ---: | ---: |
| best epoch | 154 | 150 | |
| best clean | 82.00% | 83.32% | +1.32 pp |
| best CE-PGD-20 | 48.62% | 51.76% | **+3.14 pp** |
| last clean | 83.10% | 83.28% | +0.18 pp |
| last CE-PGD-20 | 45.74% | 48.66% | **+2.92 pp** |
| best-to-last robust gap | 2.88 pp | 3.10 pp | +0.22 pp |

方向も大きさも、`docs/debugging/0028` が勾配差 58% から予想したとおりである。
定性的な特徴も戻っている。修正後 TRADES は PGD-AT の best validation robust
（`51.76%` 対 `51.80%`）に並びながら、best-to-last gap は `3.10 pp` と
PGD-AT の `8.54 pp` よりはるかに小さい。これは文献の TRADES の性質であり、
欠陥版では成り立っていなかった。

## ノイズ床について

正直に書くと、**この比較の床は測られていない**。

- プロジェクトで測ってある `1--2 pp` の RNG 床は I100 の e114 アームのもので、
  200 エポックの baseline run のものではない。そのまま流用できない。
- 上の比較は 1 run 対 1 run である。seed は両方 0 だが、損失が変われば以降の
  RNG ストリームは分岐するので、強い意味での対応比較ではない。
- したがって `+3.14 pp` は「床の上にあると推定される」までしか言えない。
  既知のどの床の見積もりより大きく、独立した定性的整合性チェック
  （PGD-AT との gap 比較）とも一致するので、単なる seed ノイズである可能性は低い。
  だが 1 seed で確定はしない。これを外せるのは B だけである。

一方 **AutoAttack 側に評価ノイズは無い**。checkpoint を固定すれば AutoAttack は
決定的なので、official 値のばらつきは訓練側の run 間ばらつきだけである。

## 選択肢

### A. 修正 run の official 評価（推奨）

**GPU 時間**: 推定 3--6 時間。内訳は official CE-PGD-20 が checkpoint あたり数分
（10,000 例）、AutoAttack が best/last の 2 checkpoint で大半を占める。
AutoAttack の実測時間はこのリポジトリに記録が無いため、これは推定であり、
plan 0027 で PGD-AT の best/last AutoAttack が訓練と並行して同日中に終わった
という事実から置いている。訓練側は実測がある（この run は 200 エポックで 2 時間 58 分）。

**決まること**: 修正後 TRADES の official AA が文献帯 `49.0--49.4%` に入るか。
入れば detach 欠陥が 4 pp の全額だったことになり、ダッシュボードの
`45.14%` 行を正当に置き換えられる。

**事前規則**:

- best checkpoint の AA が `48.0%` 以上 → 修正は確認。欠陥が主因。TRADES 行を差し替え、
  `docs/archive/ard-distillation-2026/ARD_VERSUS_AT_ASSESSMENT.md` の比較を進めてよい。
- AA が欠陥版 `45.14%` に対し `+2.0 pp` 以上だが `48.0%` 未満 → detach は実在の欠陥だが
  4 pp の全額ではない。残差の別原因を新しい調査として開く。行は差し替えるが
  「文献帯に未達」と明記する。
- AA の改善が `+2.0 pp` 未満 → detach 欠陥が 4 pp を説明するという
  `docs/debugging/0028` の**大きさの主張は支持されない**。同文書を訂正し、
  validation と official が食い違う理由を先に調べる。

`48.0%` は文献帯 `49.0--49.4%` の下端から 1 pp 下、`2.0 pp` は
validation で観測された `+2.92/+3.14 pp` の下側に取った。どちらもこの結果を
見る前に決めた値であり、結果を見てから動かさない。

**リスク**: seed は 1 本のままなので、帯に入っても「seed 0 でそうだった」以上の
主張はできない。論文レベルの主張には plan 0027 が既に書いているとおり seed の追加が要る。

### B. seed 1 を足してから両方 official 評価

**GPU 時間**: A に加えて訓練 約 3 時間（この run の実測 2 時間 58 分から）と、
2 checkpoint 分の official 評価 + AutoAttack を追加。合計おおよそ 9--14 時間。

**決まること**: `+3 pp` が修正の効果か run 間ばらつきかを 2 seed で切り分ける。
A が答えない唯一の問いがこれである。

**事前規則**: 両 seed とも欠陥版 seed 0 に対し best AA `+2.0 pp` 以上、かつ
2 seed の best AA が両方 `48.0%` 以上 → 修正を確認。片方だけなら方向のみとし、
確認としない。

**リスク**: 2 seed は平均・標準偏差を出す根拠にならない（`.claude/rules/results-records.md`
の claims discipline）。得られるのは「2 本とも同じ向きだった」までである。
欠陥版の seed 1 は存在しないので、比較相手は欠陥版 seed 0 のままであり、
比較の非対称は残る。

### C. 当面やらない

**GPU 時間**: 0。

**決まること**: 何も。

**リスク**: `docs/archive/ard-distillation-2026/ARD_VERSUS_AT_ASSESSMENT.md` が論文の問いに据えようとしている
「adversarial training 対 adversarial distillation」の公平な比較は、teacher-free
baseline が正しいことに依存している。ダッシュボードには現在、公式値としては
欠陥由来の `45.14%` しか無く、それは**教師を 4 pp よく見せる向き**に効く。
訂正済みの official 値が無いまま比較を進めると、その誤りを引き継ぐ。
一方、比較を当面進めないのであれば、これは急がない。

## 推奨とその理由

**A。** 欠陥の代償は AA で 4 pp と主張されている（`docs/debugging/0028`）。
その主張を閉じられるのは AA だけで、validation では閉じられない。
B の seed 追加は価値があるが、**順序が逆である**。まず A で帯に入るかを見れば、
入った場合の B は「seed 0 で正しかったものを 2 本目で確かめる」通常の複製になり、
入らなかった場合は seed を増やすより残差原因の調査が先になる。
どちらの結果でも A が B の設計を決めるので、A を先に走らせるのが安い。

C を採るのは、TRADES 比較を当面使わないと決めた場合だけである。

**推奨が変わる条件**: 近いうちに ARD 対 AT の比較を論文の主結果として回す予定が
無いなら C でよい。逆に、複数 seed の baseline cohort を組む予定が既にあるなら、
A を単独で走らせず B として一度に組んだほうが GPU の段取りは減る。

## この packet が扱わないこと

decision 0003 が提起した**プラットフォーム欠陥**——失敗した run bundle が
traceback を保存しないため、例外が端末以外のどこにも残らない——は未解決のままである。
今回の attempt 1 の原因（非 detach で `log(0)` に落ち、mixed precision で非有限になった）は
`ee9ced0` のコミットメッセージに記録されており、その意味で 0003 の
「原因が分からない」という状態は解消した。しかし 0003 の option B が指していた
記録の欠陥そのものは直っていない。科学的判断を含まないので、この decision を
待たずに別プランで直してよい。

## 来歴

- run bundle: `<runtime>/runs/trades-fix-v1/seed0/run-bundle/manifest.json`
- run id `trades-fix-v1-s0-attempt2`、W&B `single-teacher-ard/trades-fix-v1-s0-attempt2`
- source SHA `ee9ced07d78214e7507f2bdb3a355690fbf7b56a`（working diff 空）
- config hash `cbed20a1330c49e9deb9efb84856550e7e47872e1157019c6e979d3111cc8299`、
  protocol `controlled_cifar10_r18_v1`、seed 0、world size 1、global batch 128
- checkpoints: `best.pt`（epoch 150）、`last.pt`（epoch 199）、epoch-049/099/149/199
- 結果コミット `917447d`

## 決定（2026-09-08、本人）

**A を選択。** ただしこの評価は既に実施済みである。修正後 TRADES の official saved-checkpoint
評価と AutoAttack は 2026-09-07 に走り、best checkpoint で **AutoAttack 47.87%** を得た
（欠陥版 45.14% に対し +2.73 pp）。本パケットが pending のまま評価が走った点は手続き上の誤りで、
ここに記録する。

**残る論点は 47.87 が文献帯 49.0–49.4% より 1.2–1.5 pp 低いことである。** これは本パケットの
問いではなく、`docs/archive/ard-distillation-2026/THESIS_FRAME.md` §3 の同点崩し条件に直結する別の問いなので、そちらで追う。
`docs/archive/ard-distillation-2026/THESIS_FRAME.md` §7 項目 2 は、9 本の run を投じる前に攻撃初期化の差
（ローカルは一様 [-eps, eps]、公式は Gaussian scale 0.001）を固定バッチ上で GPU 0 時間で
測れと指摘しており、それを先に行う。
