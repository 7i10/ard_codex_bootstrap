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
chosen: E
---

## 2026-09-15 追記: 人間の判断でEを実行(BをA/Cから置き換え)

チャットでの人間からの明示的な指示: 「autoattackに関して評価に時間をかけるのは勿体無いが、
Heldoutに対しての防御力はしっかりみるべき。100サンプル以上の短時間だが有効なサンプル数を
選んで本番評価ではなく方向性を決めるための評価を行うべきでは？」

これはBの「1本だけ実測してから残りを決める」という段階的コスト測定ではなく、**4本とも
少サンプルで同時に見る**という異なる設計。新option Eとして記録し、これをchosenとする。

- **E(実行済み)**: 4本すべてのbestチェックポイントに対し、AutoAttackを`autoattack_sample_count=500`
  (heldout評価セットからseed固定の一様ランダム抽出、`EvaluationConfig.autoattack_sample_count`の
  既存メカニズムをそのまま使用、prefixではない)で実行。`evaluation.checkpoints=best`のみ(lastは
  対象外、方向性判定目的でコストを抑える)。使用コマンド:
  ```
  PYTHONPATH=src python -m ard.cli.evaluate \
    --config configs/evaluation/autoattack_saved_checkpoint.yaml \
    --checkpoint-dir <run>/train --output <run>/train/evaluation-direction-n500 \
    --weights model --allow-autoattack \
    evaluation.checkpoints=best evaluation.autoattack_sample_count=500
  ```
  実行元: 新規pinned worktree `source-c2cada7a691a`(SHA `c2cada7a691a`、P1-1/P1-2修正
  commit `1092432`を含む — 旧worktree `source-ae4dd7c82d80`はこの修正より前)。Hamster GPU0/1で
  2本ずつ実行。
  - **訂正(postrunの指摘、2026-09-15)**: 「heldout評価セット」と書いたが、実際に使われた
    `evaluation.dataset`はこのplanの公式評価スプリットである**ImageNet-1k val(5万枚)**で、
    訓練時のチェックポイント選択に使う2%held-outスライスではない。clean/PGD-10はこのval
    5万枚全体で計算され、AutoAttackだけがそこから500枚を抽出している(`autoattack_sample_count`
    は評価対象データセット全体に対する一様ランダム抽出、prefixではない)。
  - 500というサンプル数の根拠: 人間が示した下限(100以上)を上回りつつ、val全体(5万枚)より
    AutoAttackのコストを大幅に抑える。方向性(符号関係)を見るための値であり、公式テストの
    代替ではない。
  - **これは公式テスト(official test)ではない** — `.claude/rules/results-records.md`の record
    フォーマットには載せない。plan本文・decision packetのみに記録し、方向性が確認できた場合に
    初めて、本当に必要な範囲(4本、またはそのうち有望な設定)でフルサンプルのAutoAttackを
    改めて実行し、それを公式記録とする。
  - 決まること: 内部検証(PGD-10、n=1)で見えたmobilenetv3_adrの逆転傾向(§下記)がAutoAttackでも
    同じ符号で出るか。4本同時に見るため、Bのような段階的コスト測定はできないが、Eの時点で
    GPU時間は4本合計でも内部検証よりはるかに小さい(サンプル数1/10未満)。
  - **結果(4本とも完了、2026-09-15)**:

    | run | clean(val 5万枚) | PGD-10(val 5万枚) | AutoAttack(n=500) |
    |---|---:|---:|---:|
    | `r18_baseline-s0`(pgd_at) | 52.0% | — | 22.8% |
    | `r18_adr-s0`(adr) | 47.3% | — | 21.2% |
    | `mobilenetv3_baseline-s0`(pgd_at) | 42.8% | 22.7% | 17.4% |
    | `mobilenetv3_adr-s0`(adr) | 38.8% | 19.7% | 13.6% |

    (r18側のPGD-10列は postrun のログ取り込みが本パケット執筆時点で未完了のため空欄。
    clean/AutoAttackはログから直接確認済み。)

    **両アーキテクチャで`adr`が`pgd_at`に対しclean・AutoAttackとも一貫して劣る**:
    ResNet-18は-4.7pt clean/-1.6pt AutoAttack、MobileNetV3-Smallは-4.0pt clean/-3.8pt
    AutoAttack。**これは内部検証(PGD-10、best epoch)で見えたResNet-18の符号(`adr`が
    `pgd_at`より+1.7pt高いPGD-10)と矛盾する** — AutoAttackではResNet-18も`adr`が
    `pgd_at`を下回る(-1.6pt)。PGD-10(訓練で使う10-step攻撃)がAutoAttack(APGD-CE/T,
    FAB-T, Squareの複合)より弱く、`adr`のself-distillationが勾配マスキング的にPGD-10を
    実際より楽観的に見せていた可能性が高い。n=500・n=1(seed)なので二項標準誤差は約1.6-1.8pt
    (500サンプル、比率15-25%近辺)、符号自体は両アーキテクチャで揃っており偶然とは考えにくい。

    **結論**: plan 0100の仮説(小型モデルほど自己蒸留の利得が大きい)は、内部検証・AutoAttack
    のどちらで見ても支持されない。むしろ**`adr`はこの2アーキテクチャ・このレシピでは
    `pgd_at`に対しclean・robustの両方で一貫して不利**という、より強い逆方向の兆候。
    stage 2(seed 1/2、8本、約250 GPU時間)への投資判断はこの逆方向シグナルを踏まえて
    人間が行うべき — 本パケットはこれ以上の推奨を追加しない(stage 2の可否は本パケットの
    スコープ外、`chosen: E`は方向性判定の実行方法についての決定であり、stage 2着手の
    承認ではない)。

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
