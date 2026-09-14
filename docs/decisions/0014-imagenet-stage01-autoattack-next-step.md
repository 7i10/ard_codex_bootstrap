---
id: 0014
status: pending
created: 2026-09-15
campaign: imagenet-stage01-r18-mobilenetv3-adr-v2（4本とも訓練完了、内訳はorchestrator 3本＋hand-run 1本）
question: plan 0100 stage 1の4本(r18_baseline/r18_adr/mobilenetv3_baseline/mobilenetv3_adr、いずれもseed 0)の訓練が完了した。内部検証(held-out 2%スライス、PGD-10、n=1、公式テストではない)では、ResNet-18のadrはLR減衰後に回復する(best epoch 47)がMobileNetV3-Smallのadrは回復しない(best epoch 3、degenerate)——plan本文の仮説(小型モデルほど自己蒸留が効く)と逆方向の兆候。事前登録された判定ルールはAutoAttack・公式テストを前提にしている。stage 2(seed 1/2、8本、約250 GPU時間)に進む前に、AutoAttackをどう・どこまで実行するか。
options:
  A: 4本のbestチェックポイントだけAutoAttackを実行する(4回)。GPU時間は未計測(plan 0100自身が「ImageNetスケールのAutoAttackコストはこのプロジェクトで未測定」と明記)。決まること = 内部検証で見えた逆方向の兆候が公式テストでも再現するか、stage 2(約250 GPU時間)に進む前の安価なゲートとして機能する。事前規則 = 4本ともAutoAttack robust accuracyを算出し、mobilenetv3_adrがmobilenetv3_baselineを下回り続け、かつr18_adrがr18_baselineを上回るという内部検証と同じ符号関係が出るかを確認する(n=1なので確定的な判定ではなく、次の一手を決めるための参考情報)。
  B(推奨): まずr18_baseline-s0のbestチェックポイント1本だけAutoAttackを実行し、実測GPU時間を測ってから残り3本を判断する。GPU時間 = 測定目的そのもの、上限は事前に不明。決まること = ImageNetスケール・このプロジェクトの評価パイプラインでのAutoAttack実測コスト(初計測)。安ければ即座にA相当(残り3本)に進む、高ければ人が判断する。
  C: AutoAttackは保留し、内部検証の符号だけを根拠にstage 2(seed 1/2、8本)へ進む。0 GPU時間(AutoAttack分)。事前登録された判定ルール(AutoAttackが前提)からの逸脱になるため非推奨——公式テストなしでstage 2の判断をすると、後からAutoAttackを回した際に符号が変わるリスクを抱えたまま大きな投資をすることになる。
  D: 何もしない。0 GPU時間。決まること = 何も。
recommendation: B
chosen: null
---

## 何が完了しているか

plan 0100 stage 1の4本すべてが訓練完了(`docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md`のProgress log参照)。

| run | best epoch | best clean/PGD(内部検証) | last clean/PGD |
|---|---:|---|---|
| `r18_baseline-s0`(pgd_at) | 48 | 55.1% / 32.1% | 55.3% / 31.9% |
| `r18_adr-s0`(adr) | 47 | 50.1% / 33.8% | 50.2% / 33.7% |
| `mobilenetv3_baseline-s0`(pgd_at) | 46 | 45.9% / 25.3% | 46.0% / 25.2% |
| `mobilenetv3_adr-s0`(adr) | 3 | 42.3% / 21.9% | 32.1% / 20.8% |

**注記**: 上記は held-out 2%スライスに対するPGD-10選択攻撃の数値であり、`.claude/rules/results-records.md`が定める通り公式テスト(AutoAttack)ではない。n=1(seed 0のみ)。

ResNet-18は`adr`が`pgd_at`に対して-5.0pt clean/+1.7pt PGD(best)。MobileNetV3-Smallは`adr`が`pgd_at`に対して-3.6pt clean/-3.4pt PGD(best)、-13.9pt clean/-4.4pt PGD(last)。plan 0100の仮説(小型モデルほど自己蒸留の利得が大きい)とは逆方向。

## なぜBを推奨するか

AutoAttackのImageNetスケールでの実コストはこのプロジェクトで一度も測定されていない(plan 0100本文にも明記)。4本×best/last=8回、あるいはbestのみ4回、を先に決め打ちすると、実測コストが想定より大きかった場合に無駄なGPU時間を消費するリスクがある。1本だけ先に測ることで、その後の判断(残り3本、あるいは8回全部)を実測値に基づいて行える。stage 2(seed 1/2、約250 GPU時間)という大きな投資の前段階として、コストが小さいうちに実測しておく価値が大きい。

Cは事前登録された判定ルールから外れるため、内部検証の逆転傾向を確認しないままstage 2に大きな投資をすることになり、リスクが高い。

## 参照

- plan: `docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md`(Progress log、2026-09-14の3エントリ)
- ソースSHA `ae4dd7c82d80`、pinned worktree `source-ae4dd7c82d80`
