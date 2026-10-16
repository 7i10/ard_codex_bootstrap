---
id: 0060
status: pending
created: 2026-10-16
campaign: plan0103-phase2-mnv4m-baseline-ema-s0-v1（plan 0103 Phase 2。MobileNetV4-M の基準＋EMA。Phase 1 と同じ学習手順に Phase 2 の共通設定（サブセット 5,000、EMA 0.9999）を足したもの。ランダム初期化、SGD lr 0.025、full AT 50 epoch。Hamster で hand-run、seed 0、50/50 epoch 完走、非決定的＋cudnn_benchmark、source d97bf6e）
question: MobileNetV4-M の Phase 2 の基準が 63.59 / 39.10 に決まり、SiLU はこれより clean −2.33 / PGD-10 −0.77 でした。decision 0059 の A のルールどおり「MobileNetV4-M では SiLU を採用しない」と記録して閉じるか、SiLU の seed 1 を回すか、plan 0107 に「MobileNetV4-M でも下がらないこと」という条件を足すか。
options:
  A: 追加の GPU は使わない（0 GPU時間）。基準を 63.59 / 39.10 で確定し、decision 0059 の A のルールで「MobileNetV4-M では SiLU は PGD-10 を下げる（seed 1本）、採用しない」と記録して閉じる。
  B: SiLU MobileNetV4-M の seed 1 を Hamster で回す（約 36〜40 GPU時間）。Phase 1 ReLU seed 1（63.71 / 39.30、Hamster）と比べ、d1 ≤ −0.5pt なら「2本とも下がる」として閉じる。
  C: いまは GPU を使わず、plan 0107 の採用条件に「MobileNetV4-S で効果ありになった活性化の案は、蒸留のベースにする前に MobileNetV4-M でも1本回し、この基準より 0.5pt 以上下がらないこと」を足す（後で 1 案あたり約 36〜51 GPU時間）。
recommendation: A
chosen: null
---

# 0060: MobileNetV4-M 基準＋EMA が完了（63.59 / 39.10）。SiLU は正式な基準に対しても −2.33 / −0.77

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-16 (postrun, Phase 2 baseline+EMA, MobileNetV4-M)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `146b675`（`Record plan0103-phase2-mnv4m-baseline-ema-s0-v1 results`）。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は run-bundle の manifest と `epoch-metrics.jsonl` から読みました）。

この run には役目が3つあります。Phase 2 の設定で Phase 1 と同じ結果が出るかの確認（設定チェック）、EMA の効果、そして MobileNetV4-M の Phase 2 の案（いまは SiLU）と比べる正式な基準です。

条件: ImageNet-1k（元画像）、MobileNetV4-Conv-M（ReLU）、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、weight decay 1e-4）、ピーク LR 0.025、multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）、50 epoch、world size 1、バッチ 128。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| 基準＋EMA、最後（ep 49）、普通の重み（= best） | **63.59 / 39.10** |
| 基準＋EMA、最後、EMA の重み | 63.62 / 39.23 |
| 基準＋EMA、best EMA（ep 37） | 62.38 / 39.06 |
| Phase 1 seed 0（Ferret、`086072b`）、最後 | 63.79 / 39.39 |
| Phase 1 seed 1（Hamster、`609e6fa`）、最後 | 63.71 / 39.30 |
| SiLU seed 0（Ferret、`cabd60a`）、最後、普通の重み | 61.26 / 38.33 |
| SiLU seed 0、最後、EMA の重み | 61.37 / 38.69 |

- **設定チェックは通りました（0.5pt の条件で）。** Phase 1 seed 0（同じ学習 seed）との差は clean −0.20 / PGD-10 −0.29（約 74 枚、2本の差の SE 0.43pt の約 0.7 倍）、seed 1 とは −0.12 / −0.20 です。
  ルール（register C、2026-10-09：「差が seed の差以下、または 0.5pt 以下なら設定の影響なし」）は 0.5pt の方で満たします。MobileNetV4-M の seed の差（0.09pt）よりは大きい差です。
  3本とも非決定的で、source とホストも違うので、−0.29 は「非決定性・source・ホスト・共通設定」を合わせた大きさです。どれの分かは分けられません。
- **EMA は最後には効いていません**（+0.03 / +0.12、「効果なし」）。MobileNetV4-S（+0.10 / +0.10）、ConvNeXt-Atto（+0.03 / +0.06）と同じです。
- **SiLU をこの基準と比べると clean −2.33 / PGD-10 −0.77**（約 197 枚、SE の約 5.4 倍と 1.8 倍）。取り分けた画像のうちサブセットを除いた分では −2.33 / −0.86、EMA の重み同士では −2.25 / −0.54 です。
  Phase 1 seed 1 と比べた暫定値（−2.44 / −0.96）から、PGD-10 の差は 0.19pt 縮みましたが、Phase 2 のルールでは同じ「保留」（0.5〜1.5pt）の帯で、下がる向きです。clean は 1.5pt を超えて下がっています。
- **decision 0059 の A の事前ルール**（Δ ≤ −0.5pt なら「MobileNetV4-M では SiLU は下がる（seed 1本）、採用しない」）では、Δ = −0.77 なので「採用しない」になります。0059 の `chosen` はまだ null です。
- **SiLU の当てはまりの悪さは、同じ seed の基準でも再現しました。** 学習の損失は epoch 4 では SiLU が低く（−0.077）、epoch 8 から高くなり、差は広がり続けます（epoch 49 で +0.218、学習バッチの頑健精度 −2.91pt）。seed 1 と比べたときの値（+0.217 / −2.92）とほぼ同じなので、seed の違いのせいではありません。
- 頑健さの過学習はなく（best = 最後）、学習崩壊の見張りも発動していません。実時間 35.83 時間（Hamster、500〜507 枚/秒。同じホストの Phase 1 seed 1 より約 2〜3% 遅い）。共通設定によるメモリの増加は約 0.03 GB で、SiLU の +2.4 GB は SiLU 自体によるものです。

**ばらつきの目安（noise floor）**: ImageNet の seed 0 と seed 1 の差は MobileNetV4-S 0.36pt、MobileNetV4-M 0.09pt、ConvNeXt-Atto 0.36pt でした（各1組）。CIFAR の画面では乱数だけで 1〜2pt 動くことがありました。
SiLU の PGD-10 −0.77 は ImageNet の seed の差よりは大きく、CIFAR の目安（1〜2pt）よりは小さい値です。clean −2.33 は CIFAR の目安も超えています。
今回の比較は seed と実行モードはそろっていますが、source とホストはまたいでいます。言えるのは「この seed では、MobileNetV4-M の SiLU は clean を下げ、PGD-10 も下げる向き」までです。

**関係する決定**: decision 0059（SiLU MobileNetV4-M、暫定）は `chosen` が null のままです。この packet は 0059 の問いに正式な基準の数字で答え直すものなので、ここで選ばれたら 0059 は superseded として扱います（0059 は人間が直すか、指示があれば私が直します）。
decision 0044（MobileNetV4-S の SiLU を標準にするか）も未決です。plan 0107 の MobileNetV4-S の活性化の案（hard-swish、勾配だけ滑らかにする）は、この結果とは別に進んでいます。
2026-10-14 の人間の決定で、今のキューが空いたら5枚の GPU で蒸留の検証を回す予定です。B と C はその GPU と取り合います。

## 選択肢

### A: 追加の GPU は使わず、基準を確定して SiLU MobileNetV4-M を閉じる（おすすめ）

- **GPU時間**: 0。
- **わかること**: 新しくわかることはありません。0059 の A の事前ルールの判定を記録し、MobileNetV4-M の Phase 2 の基準を 63.59 / 39.10（EMA 63.62 / 39.23）に確定します。
- **事前ルール**: 0059 の A のルールをそのまま使います。Δ = SiLU − 基準＋EMA（最後の普通の重み、取り分けた画像すべての PGD-10）= −0.77 ≤ −0.5 → 「MobileNetV4-M では SiLU は PGD-10 を下げる（seed 1本）」と記録し、MobileNetV4-M の SiLU は採用しない。新しいしきい値は作りません。
- **リスク**: seed は1本のままです。−0.77 は CIFAR のばらつきの目安より小さいので、「SiLU はモデルの大きさで効き方が逆になる」とは主張できません。主張できるのは「この seed では MobileNetV4-M に移らなかった」までです。

### B: SiLU MobileNetV4-M の seed 1 を Hamster で回す

- **GPU時間**: 約 36〜40 時間（この run と Phase 1 seed 1 が Hamster で 35.4〜35.8 時間。SiLU の計算の遅さは Ferret では切り分けられなかったので、上に幅を持たせています）。
- **やり方**: `imagenet_mobilenetv4_conv_medium_silu_pgd_at_phase2_random_lr0025.yaml` を seed 1 で、seed 0 と同じ `cabd60a` の worktree から Hamster で回します。比べる相手は Phase 1 ReLU seed 1（63.71 / 39.30、Hamster）。ホストもそろいます。
- **わかること**: MobileNetV4-M で SiLU が下がることが seed 1 でも起きるか。
- **事前ルール**（0059 の B と同じ）: d1 = SiLU seed 1 − Phase 1 seed 1（最後の普通の重み、取り分けた画像すべての PGD-10）。
  d1 ≤ −0.5pt → 「2本とも下がる」として閉じる。−0.5 < d1 < +0.5 → 「seed 0 では下がり、seed 1 では効果なし」と記録して閉じる（採用しない）。d1 ≥ +0.5pt → 「判定できない」と記録し、改めて packet で決める。2本の平均やばらつきは出しません。
- **リスク**: どの結果でも「採用しない」に近く、36〜40 時間をその補強に使います。蒸留の検証の開始が Hamster 1 枚分遅れます。

### C: plan 0107 に MobileNetV4-M での確認を条件として足す（いまは GPU を使わない）

- **GPU時間**: いまは 0。後で、MobileNetV4-S で「効果あり」になった案 1 つにつき約 36〜51 時間（Hamster なら約 36〜40、Ferret なら約 51）。その案の MobileNetV4-M 版のモデルと config を作る手間もかかります。
- **わかること**: MobileNetV4-S で効いた活性化の変更が、大きさの違うモデルにも効くか。
- **事前ルール**: plan 0107 の案が MobileNetV4-S で「効果あり」（基準＋EMA より PGD-10 +1.5pt 以上）になったら、蒸留のベースにする前に同じ案を MobileNetV4-M で1本回します。
  この基準（39.10）との差が −0.5pt より大きい → ベースにする。−0.5pt 以下 → MobileNetV4-S だけの効果として記録し、ベースにするかは改めて packet で決める。
- **リスク**: plan 0107 の承認済みの条件を変えるので、plan 0107 への追記が要ります。蒸留の検証のベースが決まるのが 2 日ほど遅れます。A と一緒に選べます。

## おすすめ: A

0059 で事前に決めたルールが、そのまま「採用しない」を出しています。正式な基準に替えても、暫定の判定（下がる向きの「保留」）は動きませんでした。
学習の当てはまりの悪さも、同じ seed の基準で再現しました。B に 36〜40 時間を使っても、どの結果でも結論は「MobileNetV4-M では SiLU を採用しない」からほとんど動きません。GPU は蒸留の検証に残すのがよいと考えます。

**判断が変わる場合**:
- 論文で「SiLU の効き方がモデルの大きさで逆になる」と主張したい場合 → B が要ります（seed 1本では主張できません）。
- 蒸留のベースを MobileNetV4-S 以外にも広げる予定がある場合 → C を A と一緒に選ぶ価値が上がります。
- 設定チェックの差（−0.29）が気になる場合: これは 0.5pt の条件は満たしていますが、seed の差（0.09pt）より大きい値です。ただし、SiLU との差（−0.77）は Phase 1 と比べても基準＋EMA と比べても同じ帯に入るので、この判定には効きません。

`chosen` は人間が記入します（ファイルを編集するか、チャットで答えてください）。記入されるまで、この packet に関わる新しい学習は始めません。
