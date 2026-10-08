# Scientific invariants

この文書は実装・config・review・testで守る契約です。値を変える場合は、新しいmethod/config identity、
根拠、回帰testを同じ変更に含めます。benchmarkを通すためにattackを弱めたりtoleranceを広げたりしません。

## 1. Input domain and normalization

- dataset adapterからattackへ渡す画像はfloat pixel-space `[0,1]`。
- CIFAR normalizationはstudent/teacher model adapterが所有し、attack前後に二重適用しない。
- CIFAR-10 SAAD studentはnamed `cifar10_raw_identity` profile（student adapter所有）を要求する。train augmentationはsource-keyed、validation/testはdeterministicである。
  - **例外(plan 0102, scientific review 2026-09-21)**: `DatasetConfig.imagenet_heavy_augmentation`を有効にしたImageNet訓練では、RandAugment/RandomErasing部分はtorchvisionのグローバル乱数を使うため、source ID単位の再現性(resumeで同じ拡張結果になること)は保証されない(crop/flip部分は引き続きsource-keyedで再現性あり)。人間の明示的判断(chat, 2026-09-21)による、この一実験に限った既知の逸脱。
  - **Exception (plan 0103 loader speedups, human-approved 2026-09-30), ImageNet only, both default off.**
    `training.jpeg_draft_decode: true` decodes the RandomResizedCrop training view at a reduced JPEG scale
    (1/2, 1/4 or 1/8) whenever the crop stays at least the output size, so the resize is still a downscale.
    Crop boxes, flips and random draws are unchanged; training pixels differ slightly. Validation, probe and
    evaluation views are unchanged. `dataset.derived_from` declares a pre-resized copy of the training set
    (`scripts/build_resized_imagenet.py`); training AND in-training validation/probe read the smaller images, so
    more crops are upsampled. The official evaluation always reads the original val set. Stage 2 of the two-stage
    protocol must run on original data.
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

**記録済みの例外（1件、人間承認 2026-09-30）**：`method.selection_step_size_independent: true`。
training attackとselection attackのstep size一致チェックだけを外すフラグです。範囲は次の3条件をすべて
満たす場合に限ります。(1) protocolが`controlled_imagenet_stage02_two_stage_lowres_v1`、(2)
`training.train_image_size`が設定されている（2段階学習のstage 1）、(3) selection attackが参照のidentity
（CE、Linf eps 4/255、step 8/765、10 steps、random start、student/teacherともeval）と完全に一致する。
norm、input domain、epsilon、random startの一致チェックは残ります。理由：1-step PGD（PGD-1）で学習するには
step = eps（4/255）が必要です。一方でselectionとevaluationは参照のPGD-10 step 8/765のままなので、
報告する数値のthreat identityは変わりません。schemaがこの範囲を強制し、テストで確認しています
（`tests/unit/test_two_stage_lowres.py`）。

attack identityとthreat hashは`AttackConfig`の全14 field、すなわち`norm`、`input_domain`、`epsilon`、
`epsilon_value`、`step_size`、`step_size_value`、`steps`、`random_start`、`loss`、`kl_target`、
`temperature`、`temperature_squared`、`student_mode`、`teacher_mode`から作ります。comparisonはこのcomplete
mappingのexact equality、hashはcanonical JSONのSHA-256であり、budgetだけの比較やfield省略を認めません。

`trace_step_losses`はPGD per-step lossの観測/debug専用スイッチで、既定値は`false`です。これはthreat identity
の14 fieldに含めず、既存の攻撃値を変更しません。PGD traceを無効にした通常実行ではper-step device
synchronizationを避けます。RSLADはdetach済みFP32のteacher-clean targetをinner/outerの両方で再利用しますが、
目的関数の式は変更しません。

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
- non-intervened observation baselineは別methodではなく`rslad`と
  `observation.profile=teacher_response`の組合せで表す。attack、objective、KD/hard weight、optimizer updateは
  `rslad`そのものであり、pre-update detached FP32 primitivesをloss、target、sample selectionへ入力しない。
- `rslad_entropy`: frozen teacherのShannon entropyを使い、weightは
  `5 * (H_i - global_min_valid_batch(H))`。係数5はmethod constant。clip、mean preservation、
  hard-label fallbackはない。
- `rslad_student`: student riskは`(1 - margin_ema) / 2`。schema-v2主経路のKD weightは1、hard weightは0で、
  adversarial KD targetだけを`rho=0.5*risk`で一様分布へsoftenする。
- `rslad_joint`: teacher riskは`1-H/log(C)`、joint riskはstudent riskとの積。schema-v2主経路では同じ
  target softeningへjoint riskを使い、KD/hard loss weightは1/0のままにする。
- `rslad_frozen_oracle_softening`: train-onlyのfrozen binary riskを外部manifestのSHA-256で固定し、selected
  sampleだけ`rho=0.5`で同じtarget softeningを行う。これはfuture baseline failureを使うupper-bound実験であり、
  deployable methodではない。実runはproduction lineage guardを通す。
- student/jointのepoch 0はexact baseline RSLAD（target softeningなし、uniform KD、hard=0）としてstateだけを
  収集する。

sampleをdatasetから削除しません。旧online `oracle_mask` flagはdev-onlyです。frozen oracleは別method ID、
train namespace、source W&B version/bytes、builder Git SHAを固定した場合だけguarded production runを許可します。

## 4. Stable sample state and DDP

The controlled protocol fixes global batch 128. A one-GPU per-rank batch of 128 and a two-GPU per-rank batch of 64
are distinct execution identities because ordinary local BatchNorm uses per-rank statistics; they must not be pooled
as one scientific run family. Pilot uses five epochs only; canonical production uses 200 epochs.

- sample IDは元dataset indexであり、subset、augmentation、shuffle、rankで作り直さない。
- robust marginはpre-update detached FP32 logitsから計算する。
- observation stateはmethod/lossから独立し、risk式を固定せず、margin/current EMA、correctness
  frequency、forgetting、teacher entropy、true-class probability、
  max-wrong-class probability、prediction/correctnessをclean/adv別に保持する。
  wrong-confidence、threshold、gate、interventionはofflineで別々に定義する。
- observation profileは`off`、`student_history`、`teacher_response`を明示する。
  `teacher_response`はstudent historyを含み、teacher clean/adversarial primitivesとclean-to-adversarial
  responseを保存する。teacher adversarial forwardはpolicy/diagnosticsとbatch内で共有する。
- observation tensorはpre-update detached FP32であり、attack、loss、policy、gradient、optimizer、scheduler、
  RNGへ流さない。future methodの候補riskは保存済みprimitiveからofflineで構成し、観測のために同じrunを
  再実行しない。
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
- `training.selection_subset_size` (plan 0103 Phase 2, default off): best is selected on a fixed, class-stratified
  subset of the held-out split (seeded by `seeds.split`, at least one image per class). The selection record states
  the subset (`selection_subset`: size, SHA-256 of its sorted source IDs, full-split size, seed) and names its metric
  `val_subset_pgd_accuracy`; epoch rows use `val_subset_*` and the run summary `best_subset_*` / `last_subset_*`,
  never the full-split names. At the final epoch last and `best.pt` (and `best-ema.pt`) are evaluated on the full
  held-out split and on its complement (held-out minus subset), one pass each with shared random starts
  (`val_full_*`, `val_complement_*`, `best_val_*`, `full_split_final`). Resume requires the same subset and checks
  that `best.pt` is the recorded epoch of this run; a fresh fork child adopts its own subset (or none) and drops
  the parent's final record. Refused with ADR's gap-adaptive lambda (which reads the selection metric), so the
  trained weights are identical with and without the option.

## 7. Evaluation integrity

- training中のvalidation PGDは正式なtest evaluationではない。
- evaluate CLIは保存済みstudent checkpointと兄弟のresolved training config hashを照合する。
- 旧schema-v2 resolved configは、保存されたraw YAML mappingのhashをcheckpointへ先に照合してからevaluation-only
  runtime viewへ明示migrationする。migrationは旧`rslad_logging_only`をloss-identicalな`rslad`と
  `teacher_response`へ写し、source/runtime methodと適用変換をevaluation lineageに残す。normal train loaderと
  exact resumeを緩和しない。
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
- Dataset identity also includes `dataset.derived_from` (source digest, short side, JPEG quality, filter, build
  manifest digest) when set; the derived root's own digest is its `content_sha256`, and a derived root cannot be
  loaded without declaring it. `training.jpeg_draft_decode` is in the config hash and in
  `training_protocol_identity`. Both are recorded only when used, so runs with and without them never pool.
- `training.cuda_graph` (plan 0105, default off) replays the PGD-AT training step as one CUDA graph. It is always
  in the config hash. Config validation limits it to the scope tested by
  `tests/integration/test_cuda_graph_training_step.py`: the allowlisted students (`CUDA_GRAPH_ARCHITECTURES`:
  MobileNetV4-Conv-Small/Medium, EfficientNet-B0), an eval-mode training attack, single CUDA device, FP32, method
  `pgd_at`, SGD, no teacher/ADR/policy/mixed batch/AWP — measured on torch 2.11 with an RTX 4090, with no
  `CUBLAS_WORKSPACE_CONFIG` override (as production runs). Since 2026-10-08 (plan 0103 Phase 2 batch A) a plain
  weight EMA (`training.weight_ema_decay`, updated inside the captured step right after the SGD update with the
  eager kernels), `method.label_smoothing` and `optimizer.exclude_norm_bias_from_weight_decay` (two SGD groups) are
  in scope, under the same parity and equivalence tests (the EMA state is a fourth tensor group in the one-step rule).
  Since 2026-10-08 (human decision, speedups worth >= 1 GPU-hour per run) also in scope, under the same tests:
  ImageNet `rslad` / `rslad_advt` distillation with a `distillation` block (KL-to-teacher-clean training attack,
  RSLAD baseline policy) -- from a soft-label bank (the batch's stored top-K rows are copied into static buffers and
  reconstructed inside the captured step with the bank's own kernels; crop-key and pixel-sentinel checks stay on the
  host) or from an online teacher; a teacher whose forward runs inside the step (online target, advT's forward on
  x') must be frozen, in eval mode and on `CUDA_GRAPH_TEACHER_ARCHITECTURES` (the five Phase 2 ImageNet teachers,
  each parity-tested); `method.mixed_batch` without split BN (the last partial batch, with its own `k`, stays
  eager); `method.awp` (proxy reload, proxy SGD step, perturb and restore inside the step) with
  `training.deterministic: true` only (under nondeterministic kernels the eager outcomes of one AWP step were
  too spread for the one-step check to resolve a defect). Split BN stays refused.
  The value checks those paths share with the eager step (bank reconstruction, advT target, KL target validation,
  policy weights, teacher entropy) go through `ard.device_checks.require`: the unchanged host check outside a
  capture, a device assert (fatal before any checkpoint) inside one; never removed or widened.
  - `training.deterministic: true`: **bitwise** equal to the eager step (checkpoints, RNG streams, epoch rows,
    diagnostics). Not in `training_protocol_identity`, like `training.step_diagnostics`, so it pools with eager runs
    of the same arm.
  - `training.deterministic: false` (human decision 2026-10-03; `cudnn_benchmark` stays refused with it): not
    bitwise. What the tests observe (an observation, not a consequence of "same kernels"): (1) over whole runs every
    RNG stream (checkpointed Python/NumPy/torch CPU/CUDA states, the CUDA RNG state, the seed and draw count of every
    attack generator), the global step, scheduler and sampler state are exactly equal; (2) one step from one exact
    state, in every tensor group (parameters, SGD momentum buffers, BatchNorm buffers), lands within 4x max(the
    median spread of the eager outcomes of the same step, one FP32 rounding of the group's new value) of one nearest
    eager outcome; a spread above 100x the floor fails the check. For parameters the FP32 rounding sets the bound;
    for momentum buffers at production shapes the measured eager spread does. Graph controls with lr = 0,
    lr x 1.001, weight decay 0 or attack seed + 1 (baked into the graph) fail that rule by at least 10x, both at the
    capture step and at a later replay. One step's nondeterminism can be bimodal (a summation-order difference flips the sign of an attack input
    gradient and moves the whole step, in eager arms as well). Under cuDNN benchmark a captured step was seen to
    use a different cuDNN algorithm than the eager steps of its process (0.27 of a step away), which is why
    benchmark is refused with the graph. Because the equivalence is not
    bitwise, `cuda_graph: true` IS recorded in `training_protocol_identity` when `deterministic` is false: a
    nondeterministic graph run never pools silently with a nondeterministic eager run.
  - `training.deterministic` itself is always in `training_protocol_identity`, and `cudnn_benchmark` when true, so
    deterministic and nondeterministic runs never pool.
  Rerun the parity and equivalence tests after a torch or driver upgrade and before widening the scope.
- `training.selection_subset_size` is in the config hash and, when set, in `training_protocol_identity`
  (`selection_subset_size`, `selection_subset_sha256` from the checkpoint's selection record, which must agree with
  the config), so subset-selected and full-split-selected runs never pool. Training-time subset/full/complement
  numbers are not official results; cross-run comparisons of "best" use the official evaluation, and `last_full_*`
  is the like-for-like internal number (last weights are identical with and without the option).
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

## Best-oriented history-routing v2

`teacher_target_true_label_mix@1` はepoch 39完了時に固定したbinary train-ID maskへだけ適用する。selected sampleのadversarial RSLAD targetは`0.5 * softmax(z_teacher_clean/T) + 0.5 * one_hot(y)`、unselected sampleは通常RSLAD targetと完全に同一である。clean KD branch、attack、temperature、`T^2` scaling、branch coefficient、reductionは変更しない。selectorは全45,000 train sample上でinclusive online correctness-frequency riskとnegative margin EMAをそれぞれmidrankし、等重み合成後にanchor-correct/anchor-wrongへ分けて各上位10%を固定する。future outcome、official test、teacher correctnessはselectionへ使用しない。

## ADR (EMA自己蒸留) の契約

Wu, Wang & Chen, "Annealing Self-Distillation Rectification Improves Adversarial
Training" (ICLR 2024, arXiv:2305.12118)。公式実装は `.external/adr`
(commit `515da0e0373f9d3de2325ad970f1f9d7e5cdcd3e`) にpin済み。

- **`adr`(PGD-AT base)は内側PGD攻撃を補正ラベル`P(x)`(EMA-of-student softmaxと
  one-hotの per-sample ブレンド、paper Eq.3-5)に対して行う。`adr_trades`はしない**
  ——公式コードの`TRADES.attack`は渡されたラベル引数を一切読まず、素のTRADES内側max
  (学生自身のclean出力とperturbed出力間のKL)のまま。補正ラベルは`adr_trades`でも
  outer natural-CE項にだけ使う。この非対称は`DistillationObjective.rectifies_attack_target`
  (adrのみTrue)で実装されている。攻撃層・目的関数どちらか一方だけをこの規則から
  逸脱させて変更すると、両者の想定するtargetが食い違う。
- **EMA更新は`state_dict()`全体(パラメータ+バッファ、BNのrunning statsと
  `num_batches_tracked`含む)に対し、optimizer.step()の直後、1 iterationに1回**行う
  (timm `ModelEmaV2`と同じ規約)。整数バッファは補間後に丸めて型を戻す。
  epoch単位や一部パラメータのみの更新に変更しない。
- **温度τ(2.5→2.0)は常にper-iteration cosine anneal、warmupなし**
  (公式コードの`cosine_scheduler`はwarmup対応だが、ADRのCIFAR-10設定は
  `warmup_epochs=0`で呼んでいる)。`total_iterations`は`len(loader) * training.epochs`
  から起動のたびに解決済みconfigだけで再計算し、チェックポイントには保存しない
  ——world_size/config_hashの既存drift検知がこの値の一貫性を保証する前提であり、
  それらのチェックを弱めた場合はこの前提も崩れる。
  **λ(0.7→0.95)はデフォルト(`method.adr.lambda_source: cosine`)で同じper-iteration
  cosine anneal だが、plan 0098 の`lambda_source: gap_adaptive`を明示的に選んだ
  configに限り、epoch境界ごとに`ard.schedules.gap_adaptive`で計算した値へ置き換わり
  ——1エポック内では一定値になる(反復ごとの補間ではない)。この場合のλは
  `train_robust_accuracy`/`val_pgd_accuracy`という2つの異なる脅威モデル・BNモードで
  測った量の差に依存する、学習時観測に基づく量である点に注意
  (`docs/plans/0098-gap-adaptive-adr-cifar10.md`参照。この差が本当に
  train/val汎化ギャップを表しているかは、この計画のscientific reviewで
  指摘された未解決の論点)。`lambda_source`を指定しない既存configの挙動は
  この変更で一切変わらない。
- **評価対象の重み(student vs EMA)はデフォルトstudent**。`ard.cli.evaluate --weights
  {model,ema}`で切替可能(公式実装の`--ema`フラグに対応)。**チェックポイント選択は
  studentとEMAで完全に独立**——`ard.engine.trainer`はstudentのvalidation PGD精度で
  `best.pt`を選ぶのと**別に**、EMAシャドウモデル自身のvalidation PGD精度で
  `best-ema.pt`を選ぶ。`--weights=ema`で`best.pt`(studentが選んだepoch)を評価する
  ことは明示的に禁止されており(`ard.cli.evaluate`がファイル名とチェックポイント
  自身の`selection_metadata`の両方をValueErrorで拒否)、`--checkpoint-dir`経由の
  `--weights=ema`評価は自動的に`best-ema.pt`へ読み替えられる。`last.pt`はepoch
  定義上selectionと無関係なので両重みでそのまま評価できる。
  `evaluation-results.json`の`selection_weights`フィールドが、実際に読んだ
  チェックポイントの選択根拠(`"model"`/`"ema"`)を記録する。
  - **公式実装の"ADR"/"ADR + WA"行と厳密には同じ量ではない点に注意**:
    公式コード(`.external/adr/src/advTrainer.py`)はepochごとに**1つのモデル**
    だけをvalidationし(`--ema`はどちらを検証するかを選ぶopt-inフラグ)、選ばれた
    そのepochの`model`と`model_ema`を**同じ**`best_adv_score.pt`に保存する——
    つまり公式の"ADR"行と"ADR + WA"行は**同一epoch**の2つの読み出し方に過ぎない。
    このプロジェクトの`best.pt`(student選択)と`best-ema.pt`(EMA選択)は独立な
    選択なので、一般には**epochが異なる**。公式実装と厳密に対応する同一epoch比較
    をしたい場合は、`best-ema.pt`を`--weights=model`と`--weights=ema`の**両方**で
    評価する(同じファイル、同じepoch、重みだけ切り替え)。

## EMAチェックポイント契約はADR専用ではない(plan 0102)

上記の「評価対象の重み」「チェックポイント選択はstudentとEMAで完全に独立」
「`best-ema.pt`」の契約は、**`method.adr`を使わない、ふつうの`pgd_at`/`trades`
実行が`training.weight_ema_decay`を設定した場合にも同一のまま適用される**
(plan 0102 Workstream A: SWA的な、蒸留ターゲットではない、それ自体のための
weight EMA)。`Trainer.ema_model`・`_update_ema`・`best-ema.pt`書き込み・
`ard.engine.checkpoint`のsave/load・`ard.cli.evaluate --weights ema`の
ゲートは、実装上もともと`self.ema_model is not None`だけで判定しており
`method.id`を見ていなかったため、`adr_config`とは独立な第二の構築経路
(`weight_ema_decay`)を追加しただけで、上記契約を書き換えずに再利用できた
(scientific review、2026-09-15)。`training.weight_ema_decay`と`method.adr`
は**互いに排他**——ADRは既に自分自身のEMAを蒸留ターゲットとして持っているため、
独立な第二のシャドウモデルは併用できない(schema・Trainer両方で拒否)。

## ImageNet distillation (plan 0103 Phase 2 batch D, human-approved 2026-10-08)

- Teachers come only from `ard.models.imagenet_teacher_registry` (`teacher.source: imagenet_registry`): checkpoint
  SHA-256, architecture, parameter count, threat (Linf 4/255, pixel space) and normalization are pinned there and
  restated in the config. Inputs are `[0,1]` pixels at 224 px; the teacher adapter applies the teacher's own
  normalization, compared bit-exactly (FP32) with any normalizer embedded in the checkpoint. Singh et al. ConvNeXt-T/B
  ConvStem are raw-pixel models (`imagenet_raw_identity`); the ViT-S ConvStem embeds a mean that differs from the
  textbook ImageNet mean by <5e-5 and is applied exactly as embedded (`custom`).
- `distillation.target_source` is explicit. `online_teacher` runs the frozen teacher on each clean training batch.
  `soft_label_bank` reads `softmax(T(x))` for the exact training crop from a digest-pinned bank (top-K fp16 + residual
  mass spread uniformly over the other classes, renormalized; RSLAD consumes `log p`). Every batch's crop keys
  `(epoch, top, left, height, width, flip)` must equal the bank's, and the bank identity (dataset digest, partition,
  augmentation seed, view size, `jpeg_draft_decode`, teacher digest) must equal the run's. Bank mode is refused with
  `imagenet_heavy_augmentation` and with a temperature other than 1. With K = class count the two sources give the
  same target up to fp16 storage rounding and FP32/TF32 kernel noise of the teacher forward; the online-vs-bank
  comparison therefore isolates the top-K truncation. A per-epoch pixel sentinel (uint8 hash of a few re-drawn crops)
  refuses a run whose decoder/resize produces different pixels for the same crop keys. The bank digest is per-run
  lineage (`distillation_lineage`), not part of the pooled identity, so seeds of one arm pool.
- `rslad_advt`: RSLAD's attack (student-crafted KL to the clean-teacher target), coefficients (5/6, 1/6), temperature
  and `T^2` are unchanged; only the 5/6 adversarial KL target becomes `softmax(T(x')/tau)` from the one cached teacher
  forward on the training adversarial example. In bank mode `T(x')` runs online and receives the bank's exact top-K
  storage and reconstruction. Do not reuse the CIFAR `iad_inspired` branch for it.
- Distillation configs record `training_protocol_identity.distillation` (target source, teacher registry ID and
  digest, bank storage format and K); runs that differ in any of them never pool. In distillation runs the panel
  diagnostics never trigger a teacher forward of their own.

## Plan 0103 Phase 2 batch A training options (human-approved 2026-10-08)

All three default off and are serialized only when set, so every earlier config keeps a byte-identical resolved
config and hash (tested against the pre-change commit's schema and configs, read from git). When set they are in
the config hash and in the evaluation record: `method.mixed_batch` / `method.awp` through `method_identity`,
`optimizer.exclude_norm_bias_from_weight_decay` through the optimizer entry of `training_protocol_identity`.
None of them touches the attack identity, selection, or evaluation.

- `method.mixed_batch` (`pgd_at` only; Kurakin, Goodfellow & Bengio 2017, arXiv:1611.01236). Of each per-rank
  batch of `m`, the first `k = floor(adversarial_fraction * m)` positions are attacked with the configured training
  attack (random start drawn for those `k` only); the rest enter the training forward clean. Loss
  `(sum_clean L + lambda * sum_adv L) / ((m - k) + lambda * k)` with `lambda = adversarial_weight`, over valid
  examples. World size 1 only (schema and Trainer refuse DDP). The sampler shuffles every epoch, so the attacked
  subset is a seeded random subset that changes per epoch; no extra RNG draw. `train_robust_accuracy` (and
  `_eval_mode`) count the attacked positions only; `train_mixed_batch_*` give the counts, the clean-position
  train-mode accuracy, both branch losses and `train_mixed_batch_weighted_loss` (the optimized objective,
  `sum(lambda_i L_i) / sum(lambda_i)` over the epoch). `train_loss` stays the unweighted per-example mean.
  - `split_batchnorm: true` (Xie & Yuille 2019, arXiv:1906.03787; AdvProp's auxiliary BN): the adversarial sub-batch
    uses the model's own BatchNorm layers (the **main BN = adversarial BN**); the clean sub-batch uses an auxiliary
    copy of every BatchNorm layer's affine parameters and running statistics (`ard.engine.mixed_batch`). Attack,
    validation, selection, the saved `model` weights, EMA and evaluation all use the main (adversarial) BN only; the
    auxiliary BN is checkpointed separately (`auxiliary_batchnorm`, required on resume). BatchNorm students only
    (refused for a student with no BatchNorm layer, e.g. LayerNorm-only ConvNeXt/DeiT); LayerNorm/GroupNorm stay
    shared. No `training.compile`, no `training.init_checkpoint`. Every BN sub-batch must hold at least 2 examples
    (train-mode BN on one example is undefined, e.g. MobileNetV4's head BN after pooling): the schema refuses a
    per-rank batch with `floor(f * m) < 2` or `m - floor(f * m) < 2`, and `ard.cli.train` refuses at startup when the
    epoch's last partial batch (`len(train) % per_rank_batch_size`) would give a 1-example sub-batch. Examples are
    never dropped. The ImageNet-1k Phase 2 split (`validation_fraction` 0.02, `seeds.split` 20260911) has 1,255,547
    training images (the plan 0103 Phase 1 runs' `train_valid_examples`); with per-rank 128 and f = 0.5 every full
    batch splits 64 / 64 and the last batch of 123 splits 61 / 62, so it is admitted. The clean-position train-mode
    accuracy (`train_mixed_batch_clean_accuracy_train_mode`) goes through the auxiliary BN.
- `method.awp` (`pgd_at` only; Wu, Xia & Wang 2020, arXiv:2004.05884; official `csdongxian/AWP` `AT_AWP`, pinned in
  `external.lock.yaml` as `awp` at commit `a7acf5d842fccf1bbb4b91644352ce157d370a26`, vendored under `.external/awp`
  by `scripts/bootstrap_external.py --repository awp`; the parity test imports its `AT_AWP/utils_awp.py`). This is
  **AWP on this project's PGD-AT, not an AT-AWP reproduction**: upstream crafts its PGD examples with the model in
  train mode (except the first batch of each later epoch, which inherits eval mode from the test pass), ours with
  the configured `attack.student_mode` (default eval), on our data, schedule and students. After
  the attack, a proxy copy (train mode) takes one SGD step (lr 0.01) ascending the masked-mean PGD-AT loss (the
  official plain CE when `label_smoothing` is 0); for every state entry with ndim > 1 whose name contains `weight`
  the difference is rescaled to `||w|| / (||d|| + 1e-20) * d`; the student is moved by `gamma * d`, takes its step,
  and `gamma * d` is subtracted after the optimizer step (before the EMA update). Defaults are the AT-AWP code's
  `gamma = 0.01`, `warmup_epochs = 0`. Extra compute: one proxy forward+backward per step and one model copy.
  World size 1, no AMP. Not combinable with `mixed_batch`. The pre-update observability of an AWP step
  (`train_robust_accuracy`, diagnostic rows) is measured at the perturbed weights the training forward used.
- `optimizer.exclude_norm_bias_from_weight_decay` (SGD only): parameters with ndim <= 1 (normalization affine,
  biases) go into a `weight_decay = 0` group, the same split AdamW always uses.
- Mixed batch and AWP also require `observation.profile: off`, no teacher and no `distillation` block (schema, before
  any tracker run; the Trainer repeats the scope check).
- `training.cuda_graph` admits `mixed_batch` (not `split_batchnorm`) and `awp` (deterministic only) since 2026-10-08 (bitwise parity
  tested, plan 0105); only `ard.cli.train` applies the three options and every
  other Trainer builder refuses them (`reject_phase2_batch_a_options`).
