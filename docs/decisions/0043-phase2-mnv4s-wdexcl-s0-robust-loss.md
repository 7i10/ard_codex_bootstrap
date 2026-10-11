---
id: 0043
status: pending
created: 2026-10-11
campaign: plan0103-phase2-mnv4s-wdexcl-s0-v1（plan 0103 Phase 2 の「weight decay を正規化層とバイアスから外す」。MobileNetV4-S（ReLU）・ランダム初期化・SGD lr 0.025・full AT 50 epoch。基準＋EMA との違いは exclude_norm_bias_from_weight_decay だけ。Hamster GPU0、hand-run、seed 0、50/50 epoch 完走、source SHA 5dd329e）
question: weight decay の除外は PGD-10 を −1.14pt 下げました。記録だけで閉じるか、活性化 × weight decay の 2×2 の残り1マス（SiLU＋除外）を回すか、ReLU＋除外の seed 1 を回すか。
options:
  A: 記録だけで閉じる（追加 0 GPU時間）。MobileNetV4-S の標準は今のまま（weight decay を全パラメータにかける）。2×2 は3マスで止め、SiLU＋除外は回さない。
  B: 2×2 を完成させる。SiLU＋除外（imagenet_mobilenetv4_pgd_at_phase2_silu_wdexcl.yaml、既にある）を seed 0 で1本、5dd329e から回す（約 19〜20.5 GPU時間、4090 1枚。SiLU は CUDA Graphs なし）。SiLU で除外の効果の向きが変わるかを見る。
  C: ReLU＋除外の seed 1 を1本回す（約 15〜16.5 GPU時間）。−1.14pt が seed による揺れでないかを確かめる。
recommendation: B
chosen: null
---

# 0043: Phase 2 の weight decay 除外、MobileNetV4-S seed 0 が完了（PGD-10 は −1.14pt）

## 結果のまとめ

記録: `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の Progress log、2026-10-11 (postrun, Phase 2 weight-decay exclusion)。
証拠台帳 `docs/RESEARCH_STATUS_SUMMARY.md` の最後の行。結果のコミットは `f9e68b6`。
この契約には集計スクリプトが無いので、`docs/experiments/` の record はありません（数字は `epoch-metrics.jsonl` と run-bundle の manifest から読みました）。

この run は、基準＋EMA（`plan0103-phase2-mnv4s-baseline-ema-s0-v1`、53.85 / 30.71）と同じ学習で、違いは1つだけです。
BatchNorm の scale と shift、バイアス（1次元以下のパラメータ）に weight decay をかけません。畳み込みと全結合の重みには、今までどおり 1e-4 をかけます。
活性化 × weight decay の 2×2（discussion register B、2026-10-08）の「ReLU＋除外」のマスです。「ReLU＋全部にかける」は基準＋EMA、「SiLU＋全部にかける」はいま GPU1 で走っています。「SiLU＋除外」は設定ファイルはありますが、キューに入っていません。

条件: ImageNet-1k（元画像）、MobileNetV4-S、ランダム初期化、plain PGD-AT、教師なし、seed 0、224 px。
学習時の攻撃は PGD-3（l_inf 4/255、step 8/765、ランダムスタート）、評価は PGD-10（同じ eps と step）。
SGD（Nesterov、momentum 0.9）、ピーク LR 0.025、multistep（10 epoch のウォームアップ、epoch 25 と 38 で 1/10）、50 epoch、world size 1、バッチ 128、決定的モード＋CUDA Graphs。
数字はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではありません。AutoAttack も公式評価もまだです。

| チェックポイント | clean / PGD-10 |
|---|---:|
| 最後（ep 49）、普通の重み | **53.47 / 29.57** |
| 最後（ep 49）、EMA の重み | 53.65 / 29.72 |
| best（ep 47、サブセットで選択）、普通の重み | 53.61 / 29.69 |
| best EMA（ep 48） | 53.64 / 29.74 |
| 基準＋EMA（全部にかける）、最後、普通の重み | 53.85 / 30.71 |

- **基準より PGD-10 が 1.14pt 低い**（clean は −0.38）。2本の差の誤差（SE）は PGD-10 で約 0.37pt なので約3倍、MobileNetV4-S の seed 0 と seed 1 の差（0.36pt）の約3倍です。clean の差は誤差の内側です。
  取り分けた画像の残り 20,620 枚でも −1.12、EMA の重み同士でも −1.08 で、同じ向きです。
- **事前ルール**（1.5pt 以上「効果あり」、0.5〜1.5pt「保留」、0.5pt 未満「効果なし」）は、PGD-10 が上がる場合のために書かれています。今回は大きさが「保留」の帯に入りますが、向きは下がる方です。どう読んでも改善ではありません。
- **差が開くのは最初の学習率の低下の後で、取り分けた画像の上だけです。** 学習率が高い間（epoch 24 まで）は、サブセットの PGD-10 が基準と 0.5pt 以内で並んでいます。epoch 37 で −1.40 になり、最後まで −1.0〜−1.3 です。
  一方、学習画像（probe）の精度、学習損失（3.695 と 3.697）、学習バッチの頑健精度（27.87% と 27.90%）は最後に基準と同じです。
  つまり、学習データへの当てはまりは同じで、見ていない画像での頑健さだけが下がっています（probe と取り分けた画像の差は PGD-10 で +3.69 → +4.88）。正規化層への weight decay が、頑健さの汎化を助けていたという読み方ができますが、seed 1本で、仕組みは確かめていません。
- **EMA のウォームアップ中の遅れが消えました。** epoch 9 で、EMA の重みはサブセットで 26.68 / 15.44（普通の重みは 29.70 / 15.86）です。基準＋EMA では 5.30 / 2.84 でした。
  原因の候補は、weight decay の変更と source の違い（`5dd329e`）の2つで、確かめていません。同じ `5dd329e` の mixed batch（全部にかける）では遅れが残っていたので、weight decay の変更の方が疑わしいですが、目的関数が違う run です。
- 学習崩壊の見張りは一度も発動していません。実時間は 15.22 時間、1,197〜1,199 枚/秒でした。

**source の違い（重要な注意）**: この run は `5dd329e`、基準＋EMA は `d836c03` です。`5dd329e` が普通の full AT を `d836c03` と同じに学習するかは、まだ確かめていません（decision 0042 の B、未決定）。もし違うなら、その差がこの −1.14pt に直接混ざります。

**ばらつきの目安（noise floor）**: ImageNet では、seed 0 と seed 1 の差が MobileNetV4-S 0.36pt、MobileNetV4-M 0.09pt、ConvNeXt-Atto 0.36pt でした（各1組）。CIFAR の画面では seed だけで 1〜2pt 動くことがありました。
−1.14pt は ImageNet の1組の差は越えますが、CIFAR の目安の範囲には入ります。ImageNet の判定基準の見直しは、EfficientNet-B0 と DeiT の seed 1 がそろってからです（discussion register C、2026-10-11）。

## 選択肢

### A: 記録だけで閉じる

- **GPU時間**: 追加 0。
- **わかること**: 新しいことはありません。MobileNetV4-S（ReLU）では、除外は PGD-10 を上げなかった（seed 0、−1.14pt）として閉じます。
- **事前ルール**: 改善を探す候補なので、上がらなければ採用しない。今回は下がったので、標準は今のまま（全部にかける）。
- **リスク**: 2×2 の目的（weight decay と SiLU の関係、2026-10-08 の人間の決定）に答えが出ません。SiLU で除外が効く場合を見落とします。

### B: 2×2 を完成させる（SiLU＋除外、seed 0）

- **GPU時間**: 約 19〜20.5 時間（4090 1枚）。SiLU 版は CUDA Graphs の対象外で、決定的モードのまま普通に（eager で）回ります。根拠は、いま走っている SiLU＋全部にかける run の実測 933〜943 枚/秒です（1 epoch 約 1,340 秒の学習＋約 48 秒の評価など、× 50）。
- **やり方**: 既にある `imagenet_mobilenetv4_pgd_at_phase2_silu_wdexcl.yaml` を、`5dd329e` の worktree（`source-5dd329eaf31f`）から seed 0 で回します。設定ファイルは SiLU＋全部にかける run と、除外の1行と W&B のグループだけが違います（どちらも CUDA Graphs なし）。キューのどこに入れるかは人間が決めます。
- **わかること**: SiLU のときに除外の効果の向きが変わるか（交互作用）。比べる相手は同じ source の SiLU＋全部にかける run なので、source の違いの問題がこのマスの比較にはありません。
- **事前ルール（run の前に固定します）**: Δ = (SiLU＋除外) − (SiLU＋全部にかける)、最後の普通の重み、取り分けた画像すべての PGD-10。
  - Δ ≥ +0.5pt → ReLU（−1.14）と向きが逆。交互作用がありうるので、両方の除外のマスの seed 1 を検討する（別の packet）。
  - Δ ≤ −0.5pt → どちらの活性化でも除外は下げる。除外は採用しない。
  - |Δ| < 0.5pt → SiLU では効果なし。ReLU で下がったので、除外は採用しない。
- **リスク**: seed 1本ずつの 2×2 なので、交互作用の大きさが 1pt 前後だと揺れと区別できません。SiLU＋全部にかける run がまだ終わっていない（GPU1、10/11 03:30 UTC 時点で epoch 35/50、09 時台 UTC に終わる見込み）ので、結果が出るのはその後です。

### C: ReLU＋除外の seed 1

- **GPU時間**: 約 15〜16.5 時間（4090 1枚）。
- **わかること**: −1.14pt が seed による揺れでないか。「保留」の帯の大きさなので、ルールの文面どおりなら2本目の seed で確かめる場面です。
- **事前ルール**: seed 1 の最後の普通の重みの PGD-10 を 30.71 と比べる。
  - −0.5pt 以下 → 2本とも下がる。「この2本の seed では除外は下げる」という向きの結論（母集団の主張にはしない）。採用しない。
  - −0.5pt より上 → 結論は出ない。どちらにしても改善の証拠はないので、採用しない。
- **リスク**: どちらの結果でも行動（採用しない）が変わりません。15 時間以上を使って、標準を変えない判断を補強するだけになります。また、source の違いの問題（0042）は残ったままです。

## おすすめ: B

この run だけで「ReLU では除外は採用しない」は決まっていて、C はその判断を変えられません。
残っている問いは、2026-10-08 に人間が 2×2 を決めた理由、つまり「weight decay と SiLU の関係」です。それに答えるのは SiLU＋除外のマスだけで、設定ファイルも既にあります。
比べる相手（SiLU＋全部にかける）が同じ source なので、0042 の source の問題もこのマスの比較には入りません。

**判断が変わる場合**:
- SiLU＋全部にかける run が基準より大きく下がった（たとえば −1.5pt 以上）なら、SiLU 自体を採用しないので、その上の除外を調べる価値は下がります。そのときは A にします。SiLU の結果は今日のうちに出るので、B を選んでも、その結果を見てから起動します。
- 0042 の B（`5dd329e` の2 epoch の一致確認）で一致しなかった場合、ReLU 側の −1.14pt の読み方が変わります。B のマス自体の比較は同じ source なので影響を受けません。

`chosen` は人間が記入します。記入されるまで、この packet に関わる新しい学習は始めません。
