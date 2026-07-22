# W&B experiment tracking protocol

## 1. Boundary and ownership

W&B access is isolated behind `src/ard/tracking/`. Training, objective, attack,
signal, and evaluation code use the small tracker interface and do not import W&B directly.
Only rank 0 initializes a run, logs metrics/tables/artifacts, or finalizes the manifest;
other ranks receive a no-op tracker. Tracking phases preserve Python/NumPy/PyTorch RNG state.

## 2. Tier and state contract

Diagnostics are explicit: `off` writes no diagnostics or sample statistics; `summary` writes scalar metrics and
Parquet sample statistics without image Tables; `panel` writes fixed-ID media plus sample statistics. Production runs
require `panel`; smoke/dev configs may use `off`. `artifact_interval_epochs` controls model artifact publication
(the checked-in cadence is 5 epochs), while local best/last checkpoints are still written at scientific checkpoint
cadence. The final epoch always publishes both best and last model artifacts.

| Tier | Allowed tracking | Completion semantics |
|---|---|---|
| `dev` | disabled/offline/online as explicitly configured | local development only |
| `smoke` | disabled or offline | synthetic/local validation; not a paper result |
| `repro` | online or offline_sync | legacy compatibility only; no new runnable configs |
| `pilot` | online or offline_sync | short engineering check; not a paper result |
| `production` | online or offline_sync | lineage guard plus entity/project/group required |

`offline`と`offline_sync`は別状態です。

- `offline`: local/offline smoke用。完了時は`status=completed`, `sync_state=null`。
- `offline_sync`: W&Bへはoffline modeで書き、開始時`sync_state=running`、正常終了時
  `status=sync_pending`, `sync_state=sync_pending`。この時点では完了扱いにしません。
- `python scripts/sync_wandb.py --root <outputs>`がmanifestに記録されたrun IDと全segmentの
  `.wandb` markerを検証し、最初のsegmentを`wandb sync --id ID DIR`、続きは
  `wandb sync --id ID --append DIR`で同期します。全command成功後だけ
  `sync-complete.json`を作り、`status=completed`, `sync_state=synced`へ遷移します。
- 各segment成功直後に`sync_cursor`をatomic更新します。途中失敗ではmanifestを`sync_pending`のまま
  保持し、次回は成功済みsegmentを送らずcursor位置から`--append`再開します。
- application failureを記録した`offline_sync` runは、全segmentのupload成功後に`sync_state=synced`へ遷移しても
  `status=failed`を保持します。upload成功はapplication成功へ読み替えません。
- segment欠損、ID不一致、command失敗ではmarkerもcompleted stateも作りません。

`--dry-run`は検証とcommand表示だけを行い、subprocess、marker、manifestを変更しません。

## 3. Identity, group, and job type

推奨identity:

```text
project: <WANDB_PROJECT>
entity: <WANDB_ENTITY>
name: <dataset>-<student>-<method>-s<seed>
group: <teacher>-<dataset>-<student>-<shared-budget>
job_type: train | evaluation
```

`group`はseedを除いた比較単位として、4 ablationと対応するsaved-checkpoint evaluationで共有します。
教師ごとに比較baseを分け（例: `WANDB_GROUP_CHEN`, `WANDB_GROUP_BARTOLDSON`）、method/seedはbaseへ含めません。
`job_type`はCLIが設定し、train CLIは`train`、evaluate CLIは`evaluation`です。
analysis専用runは現bootstrapでは実装済みCLIとして主張しません。

run IDは明示値またはconfig hashとGit lineageから安定生成し、checkpointとmanifestへ保存します。
resumeではcheckpoint run ID、config hash、既存manifest identityがすべて一致しなければ停止します。
W&B initの意図はfresh runで`resume=never`、既存manifest resumeで`resume=must`です。

## 4. Manifest lineage

local `run-bundle/manifest.json`は少なくとも次を保持します。

- tier、tracking mode/state、run ID、job type、seed、world size
- resolved config hash
- Git SHA/branch/dirty status/diff SHA-256と`diff.patch`
- environment（Python/PyTorch/CUDA/cuDNN/GPU）
- pinned upstream URL/SHAとlock hash
- teacher architecture/checkpoint SHA-256/normalization
- W&B URLと具体的offline segment path（取得できた場合）
- summary、artifact source path/content-addressed local path/type/alias/SHA-256またはdirectory digest
- resume events

evaluation resultはtraining seedと独立したevaluation seedを持ちます。portable identityはdataset name/split/classes/
image size/version/content fingerprintを保持し、machine-specific dataset rootは`dataset_provenance`へ分離します。
さらにstudent、method、training protocol、evaluation protocol、teacher、resolved selection attackのidentityを
分離して保存します。training protocolにはcheckpoint world size、per-rank batch size、effective global batch size、
evaluation protocolにはevaluation seed、loader batch size、complete attack identity、AutoAttack設定を含めます。

`repro`/`production` guardはreal Git HEAD、valid external lock、一致するclean
`.external/saad`、許可されたtracking modeを要求します。productionはentity/project/groupと、
untracked fileがないことも要求します。tracked dirty stateはnon-empty exact diffを保存できる場合だけ許可されます。

## 5. Actual metric and summary keys

現在のtrain pathはepoch単位で次を記録します。

```text
epoch
global_step
train_loss
train_clean_accuracy
train_robust_accuracy
val_clean_accuracy
val_pgd_accuracy
```

train summary:

```text
best_metric
best_epoch
best_clean_accuracy
best_pgd_accuracy
last_clean_accuracy
last_pgd_accuracy
robust_overfit_gap
```

evaluationはcheckpointごとに`eval_clean_accuracy`と`eval_pgd_accuracy`を記録し、
summaryへ`evaluation_checkpoints`を保存します。clean、PGD、AutoAttackを同じmetric名へ混ぜません。
AutoAttack resultにはexplicit seedと`evaluation.autoattack_batch_size`を保存し、全dataset tensorを
設定batch size以下のchunkへ分割して`run_standard_evaluation`へ渡します。

bootstrap実装はepoch-only loggingです。`tracking.log_every_steps`のnon-null値はschemaで拒否され、
histogramと`wandb.watch`は有効化されません。

## 6. Fixed qualitative sample table

training panelはdataset source IDとseedから固定選択し、疎なepoch（初回、best更新、設定間隔、last）だけ記録します。
列契約は次の16列です。

```text
sample_id
epoch
clean_image
adversarial_image
perturbation_visualization
true_label
student_clean_prediction
student_adv_prediction
teacher_prediction
teacher_entropy
student_robust_margin_ema
student_unlearnability
joint_risk
kd_weight
clean_correct
robust_correct
```

画像tensorはrun bundle内のPNGへ保存され、W&B Tableではちょうど3つのImage列として参照されます。
手法上存在しないteacher/student signalはnullableです。全dataset画像をartifactへ複製しません。
scalar sample statisticsはgenuine Parquetとして別artifactへ保存し、`pyarrow`がない場合は偽の
`.parquet`を作らず明示的に失敗します。

## 7. Artifacts

train run:

- `model-<run-id>-best`（alias `best`, file SHA-256）
- `model-<run-id>-last`（alias `last`, file SHA-256）
- `sample-stats-<run-id>`（Parquet, file SHA-256）
- `run-bundle-<run-id>`（全file hashから作るdirectory digest）

train run bundleにはresolved config、manifest、environment、metrics JSONL、Git diff、external lock、
`completion.json`、`error-marker.txt`、panel JSONL/PNG、artifact copiesが含まれます。
`completion.json`は正常なapplication completionを構造化して記録し、`error-marker.txt`は
application errorが記録されていないことを明示します。stdout/stderrのstub fileは作りません。

正常終了ではsummaryとcompletion/error markerを書いた後に`prepare_finish()`でmanifestを最終stateへ遷移し、
その後でrun bundle artifactを公開してから`finish()`でW&B runを閉じます。run bundleのdirectory digestは
自己参照・後続更新で不安定にならないよう`manifest.json`と`artifacts/`を明示的に除外し、
`digest_excludes: [manifest.json, artifacts/]`をartifact entryへ保存します。

file artifactのlocal copyは`artifacts/<name>/<sha256>/`、external directory artifactは
`artifacts/<name>/<directory_digest>/`へcontent-addressedに保存し、manifest entryの`local_path`から参照します。
同じartifact nameの新contentは既存entry/copyを上書きせずhistoryへappendします。W&B publication失敗時のrollbackは
今回appendしたentryと、今回新規作成したdigest directoryだけに限定し、同名artifactのprior versionや共有済みcopyを
保持します。

evaluation runはresolved evaluation config、evaluation lineage、results JSON、best/last panel、任意の
best/last sample-stat Parquet、evaluation run bundleをそれぞれmanifest付きartifactとして保存します。
resultsはtraining run ID、checkpoint filename/hash/alias、portable evaluation/training dataset identityとroot provenance、
student/method/training/evaluation protocol/teacher identity、training/evaluation seed、config hash、threat hash、
clean accuracy、PGD accuracy、任意のAutoAttack resultを分離して持ちます。PGDとAutoAttackは同じ
`evaluation.seed`を使います。

evaluationは開始前にevaluation datasetのfamily/class count/image sizeと、student architecture/normalizationを
resolved training configへ照合します。evaluation attackは`AttackConfig`の14 fieldすべてについてsaved training
selection attackとのexact equalityを要求し、threat hashも同じcomplete canonical identityから作ります。
複数checkpointは全て同じtraining run ID/config hashでなければ停止します。集計ではevaluation seedを固定identityとし、
evaluation/training dataset、student、method、training/evaluation protocol、complete threatも固定します。
このためworld sizeまたはeffective global batch sizeが異なるrunは混合集計しません。training seedとteacher identityを
比較軸として保持し、各training runのbest/lastをexactly oneずつ要求します。
tracker作成後の評価・artifact処理が失敗した場合は`finish(status="failed")`を試行し、失敗manifestを残して
元の例外を再送出します。正常時はtrainと同じ`prepare_finish → run-bundle publication → finish`順序です。

tracker lifecycleはfailureにもtransactionalです。local manifest作成後のW&B init失敗はmanifestを`failed`へ
遷移させます。artifact publication失敗は未公開entryとlocal artifact copyをrollbackします。failed finishは
`completion.json`を残さず`error-marker.txt`をapplication failureへ更新し、未公開run-bundle entryを除去して、
manifest/artifactsを除いたexact file digestを`failure_snapshot`へ保存します。W&B runは成功時`exit_code=0`、
失敗時`exit_code=1`で閉じます。

全epoch完了後のterminal no-op resumeは、prior terminal statusとcompletion marker、best/last・sample-stats・
run-bundle artifactの存在、file artifactのsource/local content hashを検証してから終了します。summary、artifact
history、sample-stat bytesは再生成しません。

Tiny-ImageNet evaluation adapterのsplit identityは、expected digestなしでは`computed`、expected digest一致時は
`computed-and-matched`です。一方、resolved training configだけから復元するtraining dataset identityは
`expected-unverified`であり、training時のobserved digestではありません。Tiny-ImageNetのT5/paper集計前には、
observed training split identityをtraining lineageへ永続化して評価時に照合する追加実装が必要です。

## 8. Verified scope

Teacher acquisition templates and registry inspection do not initialize W&B.
Until a checkpoint is explicitly registered and a reproducible run approved,
W&B remains offline or disabled; no online lineage or result is claimed for
the missing-weight state.

mock/offline W&B init、rank-zero ownership、failure propagation、Table/Image、file/directory artifacts、
resume identity、transactional init/artifact failure、failure snapshot/exit code、evaluation failed lifecycle、
failed application statusを保持するoffline sync、`sync_cursor`付きmulti-segment retryはfocused testsで
検証されています。diagnosticsがfull checkpoint stateを変えないexact parity regressionも実行済みで、
real two-rank padded-diagnostics dedupのfocused証拠は`1 passed in 4.40s`です。
restricted sandboxではW&B local service socketが使えないため、local pending bundleへのfallbackを確認しました。
final impact-selected non-scientific gateでは、real single-GPU PGD smokeと2基のRTX 4090による2-GPU DDP smokeを
実行しました。このgateはtest-file単位であり、monolithic full pytestではありません。live W&B online upload、
CIFAR production/paper artifact sync、real full AutoAttack、T4/T5、dependency-complete upstream differential、
monolithic full pytestは未実行です。

Resolved schema v2 config metadata includes protocol id, all seven seed identities, and per-rank/global batch sizes. The controlled protocol records exact MultiStepLR semantics (epoch-end boundaries 100 and 150); the old identity-scheduler deferral no longer applies. `resnet18_cifar` remains a compatibility alias, not a canonical student identity.
