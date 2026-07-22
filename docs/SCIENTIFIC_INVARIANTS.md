# Scientific invariants

この文書は実装・config・review・testで守る契約です。値を変える場合は、新しいmethod/config identity、
根拠、回帰testを同じ変更に含めます。benchmarkを通すためにattackを弱めたりtoleranceを広げたりしません。

## 1. Input domain and normalization

- dataset adapterからattackへ渡す画像はfloat pixel-space `[0,1]`。
- CIFAR normalizationはstudent/teacher model adapterが所有し、attack前後に二重適用しない。
- CIFAR-10 SAAD studentはnamed `cifar10_raw_identity` profile（student adapter所有）を要求する。train augmentationはsource-keyed、validation/testはdeterministicである。
- PGD projectionはpixel-spaceで行い、`Linf` ballへprojectした後`[0,1]`へclampする。
- rational値は文字列`8/255`, `2/255`としてresolved configへ保持し、数値値と照合する。

bootstrapのcanonical CIFAR RSLAD-family budget:

```text
norm: linf
input_domain: pixel_0_1
epsilon: 8/255
step_size: 2/255
steps: 10
random_start: true
```

checkpoint selectionはhard-label CEです。evaluation attackはsaved training configで解決済みのselection attackを
defaultとし、lossを含む全identity fieldのexact equalityを要求します。training attackとのbudget driftと、
saved selection attackからのevaluation driftはschema/CLIで拒否します。

attack identityとthreat hashは`AttackConfig`の全14 field、すなわち`norm`、`input_domain`、`epsilon`、
`epsilon_value`、`step_size`、`step_size_value`、`steps`、`random_start`、`loss`、`kl_target`、
`temperature`、`temperature_squared`、`student_mode`、`teacher_mode`から作ります。comparisonはこのcomplete
mappingのexact equality、hashはcanonical JSONのSHA-256であり、budgetだけの比較やfield省略を認めません。

## 2. Model mode and gradient source

The canonical SAAD CIFAR student has exactly 11,173,962 parameters; a lossless current-PyTorch `state_dict` has 122 keys,
including BatchNorm tracking counters. These counts are identity checks, not interchangeable compatibility aliases.

- attack requestがstudent/teacher modeを明示し、context終了時に元modeへ戻す。
- checkpoint-selection/evaluation attackはstudent/teacherをeval modeに保つ。
- frozen teacherは呼び出し元が`train()`を要求してもnested BatchNormを含め常にeval mode。
- single-teacher RSLAD-familyではteacher parameterを`requires_grad=False`にし、`.grad`を残さない。
- teacher-forward-only pathはteacher input gradientを要求しない。将来teacher input gradientが必要なmethodを
  追加するときは、parameter freezeとinput gradientを別々にtestする。

## 3. Objective and policy identity

- KL directionはteacher/target distributionからstudentへ向ける。
- temperatureはtarget/student logitsへ同じ値を適用し、`temperature_squared=true`では`T^2`を掛ける。
- RSLADはstudent-crafted adversarial inputを使い、complete KD sample objectiveをreduction前に公開する。
- policy weightは必ずper-sample reduction前に掛け、DDP padding maskはloss、signal、state updateから除外する。

4 ablationの追加契約:

- `rslad`: valid sampleのKD weightはuniform、hard-label fallbackは0。
- `rslad_entropy`: frozen teacherのShannon entropyを使い、weightは
  `5 * (H_i - global_min_valid_batch(H))`。係数5はmethod constant。clip、mean preservation、
  hard-label fallbackはない。
- `rslad_student`: student riskは`(1 - margin_ema) / 2`、KD weightは`1-risk`、hard weightは`risk`。
- `rslad_joint`: teacher riskは`1-H/log(C)`、joint riskはstudent riskとの積。KD/hard blendは
  `1-risk`/`risk`。
- student/jointのepoch 0はexact baseline RSLAD（uniform KD、hard=0）としてstateだけを収集する。

sampleをdatasetから削除しません。oracle maskはdev-onlyで、smoke/repro/productionでは禁止です。

## 4. Stable sample state and DDP

- sample IDは元dataset indexであり、subset、augmentation、shuffle、rankで作り直さない。
- robust marginはpre-update detached FP32 logitsから計算する。
- EMA decayはcanonical student/joint methodで`0.9`、first observationで初期化する。
- robust correctness count、observation count、forgetting count、last updateをstable IDごとに保持する。
- rankごとのsparse observationはepoch boundaryで決定論的にmergeし、padding duplicateはstateを更新しない。
- sample stateはcheckpointへ完全保存し、resume後に同じID/stateを復元する。

## 5. Numerical precision

- attack input/gradientとsample signalsはFP32を基準とする。
- non-finite loss/weightはfailし、clampやtoleranceで隠さない。
- config rational equalityは`rtol=0, atol=1e-15`、CPU FP32 fixed-batchは原則
  `rtol=0, atol=1e-7`、distributed FP32は根拠付き`atol=1e-6`までを現在の回帰契約とする。
- PGD境界はFP32 projection roundingだけを許す`epsilon + 1e-7`。
- float64 closed-form scalar/gradient oracleは`abs=1e-14`～`1e-15`。
- checkpoint/hash/state/sample IDsはexact equality。詳細根拠は[TEST_STRATEGY.md](TEST_STRATEGY.md)に記録する。

AMPを有効にする将来configではattack gradient precisionとGradScaler stateを明示し、parity failureを
隠すために既存toleranceを広げません。

## 6. Checkpoint, selection, and resume

- `best.pt`と`last.pt`を別file/別artifactとして必ず保存する。
- bestはvalidation PGD metricで選び、selected epoch、clean/PGD pair、selection attack metadataを保持する。
- checkpointにはmodel、optimizer、scheduler、scaler、Python/NumPy/PyTorch/CUDA RNG、sampler epoch、
  sample state、global step、config hash、tracking run ID、best-selection stateを含める。
- atomic checkpoint write完了後だけtracking artifactを公開する。
- exact resumeはepoch boundaryだけ。config hash、run ID、output directory、world sizeのdriftを拒否する。
- 復元時点で全epochが完了しているno-op resumeはsummary、sample-stat bytes、artifact一覧を変更しない。
- terminal no-op resumeは、prior manifestが`completed`または`sync_pending`であること、completion marker、
  best/last・sample-stats・run-bundle artifactの存在、およびfile artifactのsource/local copy hashを先に検証する。
- mid-epoch exact resumeは実装済みと主張しない。

## 7. Evaluation integrity

- training中のvalidation PGDは正式なtest evaluationではない。
- evaluate CLIは保存済みstudent checkpointと兄弟のresolved training config hashを照合する。
- evaluation processはteacher、training objective/policy、optimizer、sample stateをtest-time defenseに使わない。
- clean accuracy、PGD accuracy、AutoAttack accuracyを分離する。
- `evaluation.seed`はtraining seedと独立し、defaultは`0`。PGD random start/panel selectionとAutoAttackの両方へ使う。
- canonical resultは`training_seed`と`evaluation_seed`を別fieldで保持し、evaluation protocolにはevaluation seed、
  loader batch size、complete attack identity、AutoAttack enabled/batch sizeを含める。
- canonical reportはbest/lastを同じthreat hashで両方評価し、checkpoint filename/alias/SHA-256を残す。
- full AutoAttackはtrain processから実行しない。evaluation configで`autoattack=true`かつCLI
  `--allow-autoattack`を付けた別processだけが、saved checkpointから`Linf` standard evaluationを実行する。
- evaluation config、lineage、results、panel、任意Parquet、run bundleをartifactへ保存する。
- portable dataset identityはname/split/classes/image size/version/content fingerprintで構成し、machine-specific
  rootはprovenanceへ分離する。
- Tiny-ImageNetのobserved split digestは、expected digestなしなら`computed`、configのexpected digestと一致したら
  `computed-and-matched`。training configだけから作るidentityの`expected-unverified`は観測済みという意味ではない。
- 集計ではevaluation/training dataset、student、method、training protocol、evaluation protocol、complete threat、
  evaluation seedを固定する。training protocolにはcheckpoint world size、per-rank batch size、effective global
  batch sizeを含める。training seedとteacherを比較軸として保持し、各training runはbest/lastをexactly oneずつ持つ。

## 8. Reporting boundary

- teacher、student、dataset、seed、checkpoint、threat model、best/lastを省略しない。
- W&B summaryと集計元resultsを一致させ、複数seed/teacherではmean/std/worst/bestを別checkpoint groupで集計する。
- W&B init/artifact failureはlocal manifest/artifactをtransactionalに確定またはrollbackし、failed runには
  failure snapshotとnonzero exit codeを残す。failed `offline_sync` runはupload後もapplication statusを変えない。
- Tiny-ImageNetのT5/paper集計を始める前に、training時にadapterが観測したsplit identityを永続化してevaluation
  resultへ照合できるようにする。現状のtraining config由来`expected-unverified`だけではこの要件を満たさない。
- synthetic smoke、mock W&B、injected AutoAttack adapterをCIFAR reproduction resultとして報告しない。
- T4/T5、CIFAR本訓練、real full AutoAttackが未実行の間はaccuracy/parity成功を主張しない。

## M0 schema v2 target policy

Schema v2 は `teacher_target_uniform_mix@1` を student/joint の adversarial student-KD branch にのみ適用する。teacher probabilities は `softmax(z_t/T)` とし、uniform mixing は `rho_max=0.5`、clean KD target は変更しない。student/joint の main semantics では hard-label fallback は使用しない。旧挙動は明示的な `rslad_hard_fallback@1` ablation としてのみ扱う。
