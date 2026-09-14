---
id: 0015
status: pending
created: 2026-09-15
campaign: plan 0101 Stage B（Ferretでのhand-run、3アーム、seed 0、12/50 epoch、全て`completed`）
question: plan 0101 Stage Bの3アーム(A=no-warmup/3-step、B=warmup/3-step、C=warmup/1-step、いずれもMobileNetV4-Conv-Small)が12epochまで完走した。epoch25のLR減衰にはまだ届いていない。Stage C(50epochフル実行)として、どのアームを何本本launchするか。
options:
  A: Arm B(warmup+3-step)だけを50epochフル実行(1本、約35-38 GPU時間)。plan 0100のmobilenetv3_baselineと最も直接比較できる設定。決まること = MobileNetV4アーキテクチャがLR減衰後にmobilenetv3より良く回復するか(plan全体の主眼)。事前規則 = best checkpointのepochが末尾(epoch45以降)ならmobilenetv3_baselineと同様の完全回復、大幅に早ければmobilenetv3_adrと同様の回復不全とみなす。
  B: Arm A(no-warmup)とArm B(warmup)を両方50epochフル実行(2本、約70-76 GPU時間)。決まること = Aと同じ主眼に加えて、ε-warmupの効果がLR減衰後も維持されるか消えるかを確定できる(12epoch時点ではA/Bはほぼ同着だった)。
  C(推奨): 3アーム全部(A/B/C)を50epochフル実行(約35-38×2+19-21 GPU時間 ≒ 90-97 GPU時間)。決まること = Bと同じに加えて、1-stepがLR減衰後も3-stepに対して同じ方向の差(clean高め・PGD低め)を保つか、それとも収束するかを確定できる。Arm Cは1-stepなので他の2本よりコストが低い。
  D: これ以上フル実行せず、12epochの結果だけで暫定判断する。0 GPU時間。事前登録された比較ができないため非推奨。
recommendation: C
chosen: C
---

## 2026-09-15 追記: 人間がCを選択、Stage C起動済み

チャットでの人間の判断: 「mobilenet v4に関して最後まで実行しましょうか」に対しAskUserQuestionで
選択肢A/B/Cを提示し、**Cを選択**(3アーム全部、50epochフル実行)。

起動内容:
- ソースSHA `916535a843b5540f785a8bee4917c8cfdba09717`(Stage Bのソース`3fbf96c`とは
  `src/`・`configs/`・`scripts/`に差分なし——docsのみのコミットが積まれただけで、
  科学的内容は同一と確認済み)。
- Ferret上に新規`ferret-prepare`/`ferret-launch`で3本、GPU0/1/2に1本ずつ:
  | run-id | arm | config | GPU |
  |---|---|---|---:|
  | `plan0101-stagec-arm-a-no-warmup-v1` | A(no-warmup, 3-step) | `imagenet_mobilenetv4_pgd_at_no_warmup.yaml` | 0 |
  | `plan0101-stagec-arm-b-warmup3step-v1` | B(warmup, 3-step) | `imagenet_mobilenetv4_pgd_at.yaml` | 1 |
  | `plan0101-stagec-arm-c-warmup1step-v1` | C(warmup, 1-step) | `imagenet_mobilenetv4_pgd_at_1step.yaml` | 2 |
  いずれも設定ファイル自体が`epochs: 50`(Stage Bはこれを`scripts/run_imagenet_stage01_train.py
  --epochs 12`で上書きしていた canary)なので、今回は`--epochs 50`(上書きなし相当、
  configの値と一致)でフル実行。Stage Bのcheckpoint(epoch 12)からのresumeではなく、
  新しいrun-id・新しい出力ディレクトリでの**ゼロからのフル50epoch再実行**——resumeは
  `--resume`が同一output_dir内のcheckpointしか受け付けない制約と、resumeの
  config_hash/tracker_run_id一致要件を確実に満たす複雑さを避けるため見送った(本パケットの
  GPU時間見積もりはもともとフル実行前提)。
- push: ローカルに溜まっていた6コミット(plan/decisionドキュメントのみ)をorigin/masterへ
  push済み(人間の明示的承認、AskUserQuestion)。
- 起動直後にPID生存・`ferret-status`が`running`であることを確認。GPU使用率の立ち上がり
  ("data/model load中"から実際の計算開始への遷移)はMonitorで別途確認中。

## 参照(追加)

## Stage Bで分かったこと(12/50 epoch、内部検証、n=1、LR減衰前)

| arm | epoch 11 clean | epoch 11 PGD |
|---|---:|---:|
| A(no-warmup, 3-step) | 37.8% | 19.4% |
| B(warmup, 3-step) | 38.2% | 19.2% |
| C(warmup, 1-step) | 44.4% | 15.2% |

- A vs B: ほぼ同着(0.4pt clean, 0.2pt PGD)——ε-warmup終了直後の時点では有意差なし。
- B vs C: 実質的な差(1-stepはclean +6.2pt, PGD -4.0pt)——人が最初に立てた問い(1-stepは3-stepと変わらないか)に対して、この時点では「変わる」という答え。

いずれもepoch25のLR減衰(plan 0100と共有するスケジュール)より前なので、収束した結果ではない。

## なぜCを推奨するか

plan 0101の本来の目的は「MobileNetV4がLR減衰後にmobilenetv3より良く回復するか」(clean精度の底上げ)であり、これは3アームどれか1本でも確認できる。しかし、すでにwarmup(A/B)とstep数(B/C)の両方で12epoch時点の差が測定できているので、同じ50epochの投資でこの2つの副次的な問いにも決着をつけられる。Arm Cは1-stepで訓練が速い(実測643 img/s対441 img/s)ため、3本目の追加コストは相対的に小さい。

Aは最小コストだが、warmupと1-stepの効果がLR減衰後にどうなるかを未確定のまま残す。

## 参照

- plan: `docs/plans/0101-mobile-clean-accuracy-floor.md`(Progress log、Stage B完走エントリ)
- plan 0100: `docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md`(比較対象のmobilenetv3_baseline結果)
- 関連決定: `docs/decisions/0014-imagenet-stage01-autoattack-next-step.md`(plan 0100側の並行する判断)
- ソースSHA `3fbf96c3d089ef62847dcd62ea2922e0c0150a25`
