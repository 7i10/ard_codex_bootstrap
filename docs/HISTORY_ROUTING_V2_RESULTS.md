# Best-oriented student-history routing v2 results

## 結論

2026-08-05時点で、Bartoldson教師を用いたdevelopment blockは完了し、
`teacher_target_true_label_mix@1` route familyを**Development No-Go**と判定した。
exact online student historyの将来失敗予測は2 seedで再現したが、その情報を使った
true-label-anchored target interventionは、事前登録したBest validation PGD改善基準を満たさなかった。

この判定はCIFAR-10 trainから固定分割したvalidation上のCE PGD-20に基づく。
official CIFAR-10 testとAutoAttackは開発判断に使っていない。停止規則に従い、未使用の
Bartoldson seeds 3/4、Chen no-harm、v2 official evaluationは起動していない。

## 科学契約とlineage

- Branch/fork Git SHA: `2337b9d89c8319b39edf7067f08b3d4aa82e0ce0`
- Teacher: `Bartoldson2024Adversarial_WRN-94-16`
- Student/objective: CIFAR-10 ResNet-18 / RSLAD
- Parent: L1 seed 1またはL3 seed 2のepoch-39 checkpoint
- Schedule: delayed milestones `[120, 170]`
- Intervention period: epoch 40--199
- Selection: epoch-39 stateごとにexact online history risk上位10%
- Intervention: selected sampleのadversarial teacher targetだけを
  `0.5 * p_teacher(clean) + 0.5 * one_hot(y)`へ変更
- Unchanged: clean KD、attack、temperature、loss係数、非選択sample、optimizer、RNG

`PF`はepoch 39でrobust-correct、`NR`はrobust-wrongを表す。`TA`はhistory選択、
`R`はclass/state/count-matched randomである。選択数はseed 1でPF 2,160 / NR 2,339、
seed 2でPF 2,149 / NR 2,350だった。全10 W&B runは`finished`である。

| Arm | Seed 1 W&B run | Seed 2 W&B run |
|---|---|---|
| C | `bart-rslad-delayed-s1-dev-v1` | `bart-rslad-delayed-s2-dev-v1` |
| PF-TA | `h2-bart-rslad-logging-only-s1-confirm-v1-pf_ta` | `h2-bart-rslad-observed-s2-confirm-v2-pf_ta` |
| PF-R | `h2-bart-rslad-logging-only-s1-confirm-v1-pf_r` | `h2-bart-rslad-observed-s2-confirm-v2-pf_r` |
| NR-TA | `h2-bart-rslad-logging-only-s1-confirm-v1-nr_ta` | `h2-bart-rslad-observed-s2-confirm-v2-nr_ta` |
| NR-R | `h2-bart-rslad-logging-only-s1-confirm-v1-nr_r` | `h2-bart-rslad-observed-s2-confirm-v2-nr_r` |

## Validation結果

値はaccuracy (%)であり、すべてvalidation CE PGD-20でcheckpointを選択している。
RO gapは`Best PGD - Last PGD`で、小さい方が後半劣化が少ない。

| Arm | Seed | Best clean | Best PGD | Best epoch | Last clean | Last PGD | RO gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 1 | 85.64 | 51.88 | 124 | 86.14 | 47.64 | 4.24 |
| C | 2 | 84.72 | 51.74 | 122 | 86.22 | 47.74 | 4.00 |
| PF-TA | 1 | 84.48 | 51.50 | 121 | 85.64 | 47.36 | 4.14 |
| PF-TA | 2 | 84.80 | 51.76 | 121 | 86.38 | 47.26 | 4.50 |
| PF-R | 1 | 84.40 | 52.20 | 121 | 85.66 | 47.80 | 4.40 |
| PF-R | 2 | 85.46 | 51.32 | 123 | 86.12 | 47.18 | 4.14 |
| NR-TA | 1 | 84.52 | 51.74 | 121 | 86.14 | 47.02 | 4.72 |
| NR-TA | 2 | 85.48 | 51.54 | 125 | 86.74 | 48.06 | 3.48 |
| NR-R | 1 | 84.58 | 51.60 | 121 | 85.62 | 47.16 | 4.44 |
| NR-R | 2 | 84.32 | 51.38 | 122 | 85.94 | 47.70 | 3.68 |

### 2 seed平均差

| 比較 | Best PGD | Last PGD | RO gap | Best clean |
|---|---:|---:|---:|---:|
| PF-TA - C | -0.18 pp | -0.38 pp | +0.20 pp | -0.54 pp |
| PF-TA - PF-R | -0.13 pp | -0.18 pp | +0.05 pp | -0.29 pp |
| NR-TA - C | -0.17 pp | -0.15 pp | -0.02 pp | -0.18 pp |
| NR-TA - NR-R | +0.15 pp | +0.11 pp | +0.04 pp | +0.55 pp |

PFはBest改善、seedごとの非負、matched-random優位、clean低下、ROの全条件を満たさなかった。
NRはhistoryがrandomを両seedでBest `+0.14/+0.16 pp`上回ったが、Cに対しては
`-0.14/-0.20 pp`であり、主基準を満たさなかった。PF-R seed 1の52.20%は全arm中最高だが、
seed 2では51.32%であり、random maskの変動を示す単発値として扱い、採用根拠にしない。

## 学習曲線の確認

W&B epoch-metrics artifactから介入armはepoch 40--199の160行、controlはepoch 80--199の
120行を取得した。共通区間epoch 100--199のnormalized validation-PGD AUCは次の通りである。

| Arm | Seed 1 | Seed 2 | Mean | Mean差 vs C |
|---|---:|---:|---:|---:|
| C | 47.081 | 47.121 | 47.101 | -- |
| PF-TA | 47.006 | 47.145 | 47.075 | -0.025 pp |
| PF-R | 47.207 | 47.088 | 47.148 | +0.047 pp |
| NR-TA | 46.754 | 47.174 | 46.964 | -0.137 pp |
| NR-R | 46.927 | 47.053 | 46.990 | -0.111 pp |

Best epochは全armで121--125に集中し、delayed milestone 120直後だった。したがってpeak位置は
主としてschedule timingに支配され、今回のrouting固有効果とは解釈しない。NR-TA seed 2の
Last `+0.32 pp`、RO gap `-0.52 pp`対Cもseed 1で再現せず、RO改善の確証にはしない。

## ここまでに確定した研究知見

1. **予測信号は再現した。** epoch 39のexact online historyはpeak-window failureに対し、
   Bartoldson seed 1でAUROC `.911550`（instantaneous `.840528`、差`+.071022`、
   95% CI `[.063511,.078699]`）、seed 2で`.905453`（`.848499`、差`+.056955`、
   CI `[.049453,.064472]`）だった。
2. **teacher entropyよりstudent historyが強い。** epoch 99から将来forgettingを予測する
   Student AUROCはBartoldson seed 1/2で`.922/.927`、Teacherは`.624/.626`だった。
3. **teacher状態はERT/IRTで大きく異なる。** epoch 79のteacher-adversarial-wrong率は、
   future-forgetting群でBartoldson `6/10110 = 0.059%`、`4/8856 = 0.045%`、
   Chen `87/9032 = 0.963%`、`96/9481 = 1.013%`、persistent-wrong群で
   Bartoldson `241/13828 = 1.743%`、`260/14138 = 1.839%`、Chen
   `4772/11426 = 41.764%`、`4707/11217 = 41.963%`だった。
   teacher-wrong-only gateはBartoldsonの主routeには疎すぎる。
4. **scheduleはROへ影響する。** delayed scheduleはnormal RSLAD比で2 seed平均Best
   `+0.12 pp`、Last `+0.70 pp`、RO gap `-0.58 pp`だった。ただしBest改善は小さく、
   手法上の成功ではなく強いmatched controlとして扱った。
5. **予測可能性と介入utilityは別である。** NR historyのrandomに対する小さな一貫優位は
   treatment-target alignmentの可能性を残すが、true-label mixはCを超えず、今回の
   intervention familyはBest目的で停止する。

## No-Goの機構仮説（未実証）

この実験が直接示したのは「将来失敗を予測できるsampleへtrue-label mixを適用しても
Best PGDは改善しなかった」ことであり、次の機構はまだ仮説である。

- Student historyは将来最初に崩れるsampleを検出していても、モデル全体を崩す原因や
  正の介入効果を持つsampleを検出しているとは限らない。予後予測と介入utilityを分ける。
- `0.5 * p_teacher + 0.5 * one_hot(y)`はsofteningではなく、教師が正しい場合には
  targetをone-hot側へhardeningする。Bartoldsonでteacher-adversarial-wrongが稀である
  ことを踏まえると、IRTの問題をteacher-wrongとして修正した介入ではない。
- PFはanchor時点ですでに正しいため、必要なのは外部teacher targetのhardeningより
  過去Student状態の保持かもしれない。NRはanchor時点で誤っており、teacherが正しい
  場合はtarget変更よりinput-side learnabilityの改善が適合する可能性がある。
- 固定maskをepoch 40--199へ適用したため、後に回復したsampleへ介入し続ける一方、
  後発の不安定sampleを拾えない。ただし動的maskには介入とsignalのfeedbackがあるため、
  現結果から直ちにオンライン更新を採用しない。
- 高いtrain-sample予測AUROCはvalidation robust accuracyの改善を保証しない。選択sampleの
  rescueと、非選択sampleを含むspillover harmを同時に測る必要がある。

次は新規訓練より先に、既存L1/L3のControl/history/random armを同一attack・同一epochで
再評価し、sampleをrescued / harmed / stable / unchanged failureへ分ける。さらに選択群の
target変更量、teacher clean-to-adversarial response、固定panelのloss-gradient方向を監査する。
これは個別因果効果の推定ではなく、PF temporal stabilizationとNR input-side curriculumの
どちらを次の直交介入候補へ残すかを決めるretrospective moderation分析である。契約は
[Plan 0021](plans/0021-pre39-prescriptive-routing.md)へ固定する。

## 未確認事項と次の研究判断

- `q`、anchor、mix係数を同じL1/L3結果で再調整しない。
- official test、AutoAttack、Bartoldson seeds 3/4、Chen no-harmはDevelopment No-Goにより未実行。
- true-label anchor routeは閉じる。次は既存trajectoryを使い、epoch 39以前のanchor、
  student-vs-teacher比較、persistent-error taxonomyをread-onlyで完了する。
- 次の介入は、teacher-correct/teacher-wrongとpersistent/future-failureを踏まえて最大2つの
  直交候補に固定する。候補例はadversarial KD downweightとteacher-correct時のKD維持だが、
  解析前に実験条件として確定しない。
- full SAAD、TRADES、PGD-ATは提案法と独立したbaseline gapとして残る。

事前登録した設計と停止規則は
[Plan 0020](plans/0020-best-oriented-history-routing-v2.md)を参照する。
