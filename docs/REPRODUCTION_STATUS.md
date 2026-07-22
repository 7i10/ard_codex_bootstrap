# Reproduction status

最終更新: 2026-07-22

## 現在の到達点

同一training engine上で、次の8つのschema-v2 methodを選択できます。M4のbounded method-switchは全methodを
one epochのsynthetic fixtureで切り替える構成であり、engineをmethodごとに複製しません。

| Config method | 役割 | 固定された追加契約 |
|---|---|---|
| `pgd_at` | hard-label adversarial training | CE inner/outer objective |
| `trades` | TRADES baseline | student-clean KL inner objective、explicit beta |
| `rslad` | uniform RSLAD baseline | hard-label fallbackなし |
| `rslad_entropy` | teacher entropy weighting | Shannon entropy、係数`5`、clip/mean preservation/fallbackなし |
| `rslad_student` | student-risk target softening | EMA `0.9`、epoch 0はunsoftened warmup、adversarial KD targetのみ変更 |
| `rslad_joint` | joint-risk target softening | student risk × teacher overconfidence、epoch 0はunsoftened warmup |
| `rslad_joint_downweight` | explicit downweight ablation | joint riskでKDをdownweight、hard fallbackなし |
| `rslad_hard_fallback` | legacy fallback ablation | 旧joint KD/CE blendを明示的method IDで保持 |

Baseline-readinessのM0–M4は承認済みです。最終bounded T0–T3 gateは18 commandsで`209 passed, 2 skipped`、
同一の最終fingerprintの再確認は18 commandsすべて`cached pass`でした。
Ruff/mypy/import/CLI gateもpassしました。T4/T5、live W&B、real teacher checkpoint取得・ロード、
teacher accuracy audit、CIFAR本訓練、real full AutoAttackはdeferredです。

全8手法でattackはpixel-space `[0,1]`、`Linf`、`epsilon=8/255`、
`step_size=2/255`、10 steps、random startです。checkpoint selection attackは同じbudgetのhard-label CEです。
attack identity/hashはbudgetだけでなく、resolved quantity、loss/target、temperature、model modeを含む
`AttackConfig`全14 fieldから作ります。これらのいずれかを変更した実験は同一baselineとして扱いません。
`trace_step_losses: false`はper-step PGD lossを収集しない既定の観測設定であり、14-field threat identityから除外されます。

保存対象は`best.pt`と`last.pt`の両方です。resumeはepoch boundaryだけを保証し、optimizer、scheduler、
scaler、RNG、sampler epoch、sample state、global step、tracking run ID、best-selection stateを復元します。
復元時点ですでに全epochが完了しているno-op resumeは、summary、sample statistics、artifact一覧を更新しません。
ただしprior terminal status/completion marker、required artifact集合、file artifactのsource/local hashが完全であることを
検証し、不完全またはdriftしたterminal lineageは拒否します。
mid-epoch resumeとworld-size変更は再現保証の対象外です。

Controlled configs freeze split seed `20260722`, 200 epochs, validation fraction 0.1, global batch 128, and `${ARD_PER_RANK_BATCH_SIZE}` (128 on one GPU, 64 on two). The SAAD student uses raw identity normalization and has 11,173,962 parameters; its lossless current-PyTorch state_dict contains 122 keys including BatchNorm counters. LR is 0.1/0.01/0.001 for epochs 0–99/100–149/150–199. Training uses PGD-10 KL/teacher-clean; selection/evaluation uses explicit PGD-20 CE. Paper/code protocols are audit-only; config checks perform no downloads, GPU, or full training. `resnet18_cifar` is a compatibility alias, not canonical.

## CIFAR template configs and execution status

Runnable taxonomy is now explicit: `configs/audit/` contains exactly two W&B-free RobustBench teacher audits,
`configs/pilot/` contains two 5-epoch RSLAD checks, and `configs/production/` contains eight teacher-explicit
canonical 200-epoch configs (two teachers × four methods). The former reproduction runnable files were
removed; `repro` remains only as a legacy schema tier for old resolved bundles. Public SAAD paper/code records under
`configs/protocols/` are audit-only and are not local reproduction claims.

No 5-epoch pilot, 200-epoch training, or full AutoAttack has been executed. Teacher audit PGD-20 is a bounded screening
measurement and must not be reported as AutoAttack.

Use `ARD_PER_RANK_BATCH_SIZE=64` with `torchrun --nproc_per_node=2` for pilot and canonical production. Execution
identity includes world size, per-rank/global batch, and `local_per_rank` BatchNorm; a 1-GPU batch-128 run is a
scientifically distinct profile. Evaluate saved checkpoints in a separate process and reuse the checkpoint's training
execution identity.

### RobustBench teacher acquisition

Strict `TeacherConfig` fragments under `configs/teachers/` cover the exact IDs
`chen2021_ltd_wrn34_10` and `bartoldson2024_adversarial_wrn94_16`, pinned to
RobustBench commit `78fcc9e48a07a861268f295a777b975f25155964`. Acquisition is
never automatic: the pinned loader was invoked once with
`scripts/acquire_robustbench_teachers.py --allow-network`; the committed lock
now contains both verified checkpoint identities. On a fresh clone,
`bootstrap_teacher.py --install-locked` materializes the ignored runtime cache
without changing that lock. `--update-lock` is reserved for a maintainer
establishing the first identity of a `missing` lock entry. Missing environment
variables fail closed. Preprocessing ownership and the CIFAR-10 `Linf` `8/255`
threat are explicit in each fragment. For a verified entry, acquisition checks
the staged download against the locked SHA before publishing the external file.

```bash
MODEL_DIR=/home/shunsukenaito/workspace-local/datasets/ard/teachers/robustbench
PYTHONPATH=src python scripts/bootstrap_external.py --repository robustbench
PYTHONPATH=src python scripts/acquire_robustbench_teachers.py \
  --registry-id chen2021_ltd_wrn34_10 --model-dir "$MODEL_DIR" \
  --device cuda:0 --allow-network
PYTHONPATH=src python scripts/acquire_robustbench_teachers.py \
  --registry-id bartoldson2024_adversarial_wrn94_16 --model-dir "$MODEL_DIR" \
  --device cuda:1 --allow-network
PYTHONPATH=src python scripts/bootstrap_teacher.py \
  --registry-id chen2021_ltd_wrn34_10 \
  --source "$MODEL_DIR/cifar10/Linf/Chen2021LTD_WRN34_10.pt" --install-locked
PYTHONPATH=src python scripts/bootstrap_teacher.py \
  --registry-id bartoldson2024_adversarial_wrn94_16 \
  --source "$MODEL_DIR/cifar10/Linf/Bartoldson2024Adversarial_WRN-94-16.pt" --install-locked
PYTHONPATH=src python scripts/verify_teacher.py --registry-id chen2021_ltd_wrn34_10
PYTHONPATH=src python scripts/verify_teacher.py --registry-id bartoldson2024_adversarial_wrn94_16
PYTHONPATH=src python scripts/audit_robustbench_teacher.py \
  --registry-id chen2021_ltd_wrn34_10 --model-dir "$MODEL_DIR" --device cuda:0
PYTHONPATH=src python scripts/audit_robustbench_teacher.py \
  --registry-id bartoldson2024_adversarial_wrn94_16 --model-dir "$MODEL_DIR" --device cuda:1
export ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT="$PWD/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt"
export ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256="<SHA_FROM_TEACHERS_LOCK>"
```

The two acquired sources and ignored runtime copies are SHA-locked as follows:

| Teacher / role | Source bytes (SHA-256) | Parameters | Preprocessing |
|---|---:|---:|---|
| `Chen2021LTD_WRN34_10` (ERT) | `184803174`; `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983` | 46,160,474 | adapter-owned raw identity |
| `Bartoldson2024Adversarial_WRN-94-16` (IRT) | `1464289203`; `56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b` | 365,915,610 | model-embedded mean/std |

Sources reside under
`/home/shunsukenaito/workspace-local/datasets/ard/teachers/robustbench/cifar10/Linf/`;
the matching project cache is `teacher_cache/robustbench/` and is Git-ignored.
Fresh-process audits on physical GPUs (`cuda:0` Chen, `cuda:1` Bartoldson) passed
exact pinned-loader/ARD FP32 logit parity (`max_abs_diff=0.0`), finite nonzero
input gradients, frozen teacher parameters, and one-step production PGD with
`Linf=0.007843166589736938` (epsilon `8/255`, step `2/255`). The reported
teacher AutoAttack values (Chen `56.94%`, Bartoldson `73.71%`) are reference
values only and were not locally reproduced. No CIFAR training, AutoAttack, or
W&B run was performed. Chen WRN-34-20 and Gowal WRN-28-10 remain deferred.

### Bounded teacher accuracy audit (2026-07-23)

Clean HEAD `56610ea40d6333c5a98d40f23aa24fd4cc9b11bb`で、official CIFAR-10 testからseed 0で
stratified選択した同一1000 sampleを監査しました。attackはpixel `[0,1]`、Linf `8/255`、step `2/255`、
PGD-20 hard-label CE、random startです。sample-ID digestは両教師とも
`37671f8e336f31778e8f4f2343fc1904ebc981dff6d69b8652ff89e3479fe0ef`、attack digestは
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`でした。

| Teacher | GPU / batch | Clean | PGD-20 | Peak allocated | Result SHA-256 |
|---|---:|---:|---:|---:|---|
| Chen2021LTD_WRN34_10 | visible `0` / 128 | 859/1000 (85.9%) | 630/1000 (63.0%) | 3,214,566,912 B | `716921d89f120d4b238e16438c999595038c045bd3c7be6633a526a21e2fe908` |
| Bartoldson2024Adversarial_WRN-94-16 | visible `1` / 16 | 945/1000 (94.5%) | 771/1000 (77.1%) | 4,357,223,424 B | `39f09e68af426f683d35d2430dca2d25a9b26f311ce7f5b21247da9daf30c716` |

Artifacts are under `/home/shunsukenaito/workspace-local/datasets/ard/audits/56610ea/`. Both jobs ran in parallel;
other GPU processes were present, so this run is not a throughput benchmark. These PGD values are not RobustBench
AutoAttack values and do not replace the reported references.

Audit・pilot・productionを明確に分離します。`configs/audit/`には教師監査2件、`configs/pilot/`には5 epochの
RSLAD動作確認2件、`configs/production/`には2教師×4 methodのcanonical 200 epoch設定8件があります。
旧reproduction directoryのrunnable設定は削除済みで、`repro` tierは旧resolved bundleの互換性だけに残します。
5 epoch pilot、200 epoch本訓練、full AutoAttackはいずれも未実行です。

全テンプレートは未解決値を暗黙defaultにしません。実行前に以下を環境へ設定します。

| Variable | 内容 |
|---|---|
| `ARD_CIFAR10_ROOT` | 既存CIFAR-10 root（configはdownloadしない） |
| `ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT`, `ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256` | Chen fragment用のregistry cache pathとlock SHA |
| `ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT`, `ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256` | Bartoldson fragment用のregistry cache pathとlock SHA |
| `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_GROUP_CHEN`, `WANDB_GROUP_BARTOLDSON` | W&B identity。各教師の4手法と対応evalでgroupを共有する。教師間ではgroupを分ける |
| `ARD_SEED`, `ARD_OUTPUT_ROOT` | seedと出力root |
| `ARD_PER_RANK_BATCH_SIZE` | per-rank batch size（single GPU=128、2 GPU=64） |
| `ARD_NUM_WORKERS`, `ARD_DEVICE` | data-loader worker数とdevice |

W&B remains offline/disabled for this acquisition-only state; no online or
offline-sync experiment is claimed.

Diagnostics policy is explicit: smoke/dev uses `diagnostics_mode: off`; pilot and production templates use fixed-ID
`panel` diagnostics. No W&B online run, CIFAR training run, or real full AutoAttack has been executed in this bootstrap.

Protocol ID, optimizer, scheduler, epoch count (200), validation fraction (0.1), global batch size (128),
and attack identities are fixed in the checked-in YAML. Overrides of these scientific fields are rejected by the schema;
there are no `ARD_TRAIN_EPOCHS`, `ARD_BATCH_SIZE`, `ARD_LEARNING_RATE`, `ARD_MOMENTUM`, `ARD_WEIGHT_DECAY`, or
`ARD_VALIDATION_FRACTION` exports for controlled runs.

比較runでは全4手法へ同じ operator environment を与え、resolved configを保存してください。

## 実際のCLI

例としてChen教師の5 epoch pilotを使う場合:

```bash
export ARD_PER_RANK_BATCH_SIZE=64 ARD_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module ard.cli.train \
  --config configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module ard.cli.train \
  --config configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml \
  --resume "$ARD_OUTPUT_ROOT/cifar10-r18-rslad-chen2021-ltd-wrn34-10-pilot-s$ARD_SEED/last.pt"

PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml \
  --checkpoint-dir "$ARD_OUTPUT_ROOT/cifar10-r18-rslad-chen2021-ltd-wrn34-10-pilot-s$ARD_SEED"
```

Train/resumeは同じ2 GPU execution profileを使います。evaluationはcheckpointへ保存されたtraining identityを
引き継ぐ別の単一processであり、評価process自身を2 GPU DDPにする必要はありません。

別outputへ評価する場合は`--output`、training resolved configがcheckpointの兄弟にない場合は
`--train-config`を指定します。単一checkpointだけを評価するときは`--checkpoint-dir`の代わりに
`--checkpoint`を使います。

full AutoAttackはconfig overrideと二重のCLI opt-inが必要です。

```bash
PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml \
  --checkpoint-dir "$ARD_OUTPUT_ROOT/cifar10-r18-rslad-chen2021-ltd-wrn34-10-pilot-s$ARD_SEED" \
  --allow-autoattack evaluation.autoattack=true
```

このfull AutoAttack commandは未実行です。通常のPGD evaluationと同じ結果として扱いません。
実行時はtraining seedとは独立した`evaluation.seed`（default `0`）をPGDのrandom start/panel selectionと
AutoAttackの両方へ設定し、`evaluation.autoattack_batch_size`以下のbounded chunkで処理します。
seed、batch size、versionはcheckpoint別resultへ保存されます。checked-inの全evaluation-bearing configも
`evaluation.seed: 0`を明示しています。

evaluate CLIはevaluation datasetのfamily/class count/image sizeとstudent architecture/normalizationを
resolved training configへ照合します。dataset rootはprovenanceとして別保存し、portable dataset identityには
含めません。evaluation attackはsaved training configで解決済みのselection attackと14 fieldすべてがexact equalityで
なければ拒否し、best/lastが同じtraining run ID/config hash/world sizeを持つことも確認します。

canonical resultはtraining/evaluation seedを分離し、evaluation protocolへseed、loader batch size、complete attack、
AutoAttack設定を保存します。集計はevaluation/training dataset、student、method、training/evaluation protocol、
complete threat、固定evaluation seedを同一性の軸とします。training protocolにはcheckpoint world sizeとper-rank/
effective global batch sizeが含まれるため、これらが違うrunは混合しません。training seedとteacher identityを比較軸として
保持し、各training runにはbest/lastが1件ずつ必要です。

Tiny-ImageNet split digestはadapter観測時に`computed`、configのexpected digestと一致した場合に
`computed-and-matched`となります。training configだけから再構成するidentityは`expected-unverified`であり、
observed training splitの証明ではありません。Tiny-ImageNetのT5/paper集計は、training時のobserved split identityを
永続化してevaluationへ照合する実装までdeferredです。

local artifact copyはnameとcontent digestでaddressされ、同名artifactのversion historyをmanifestへappendします。
publication failureのrollbackは今回のentry/new digest copyだけに限定され、prior versionを保持します。
tracker作成後に失敗したevaluationはmanifestを`failed`にして元の例外を返します。W&B init/artifact publicationは
transactionalに扱い、failure snapshotと非zero exit codeを残します。`offline_sync`のfailed runはupload成功後も
application statusを`failed`のまま保持します。

## Legacy repro, pilot, and production guard

`repro`（legacy）/`pilot`/`production` trainは、実体のあるmain Git HEAD、一致する`external.lock.yaml`、cleanな
`.external/saad` checkout、`online`または`offline_sync` trackingを要求します。productionはさらに
entity/project/groupを必須とし、untracked main-repository filesを拒否します。tracked dirty stateは
non-empty binary diffをmanifestへ保存できる場合だけ許可されます。したがって、unborn worktreeのまま
production templateを実行すると意図どおり停止します。

teacher checkpointの存在・SHA-256一致とCIFAR dataset availabilityも実行時に検証されます。

## 実行済みの範囲

### TRADES upstream evidence

Official TRADES is bootstrapped from
`https://github.com/yaodongyu/TRADES.git` at
`6e8e11b7c281371c2f027ffadfbaea80361f09de`. The clean checkout is verified;
its root `LICENSE` is MIT with SHA-256
`4b42e38a6899d82801eb6782fe161cccb5d3d685c8bcddc2b877ac9f87161a30`, and the
lock evidence is verified. Use `--repository trades` to bootstrap one checkout
or `--all` to process every lock entry (the default remains SAAD), then run
`python scripts/verify_external.py --all`.

The documented differential contracts are: upstream non-detached outer clean
KL target versus local detached target (same scalar, different clean gradient
and SGD delta); upstream `0.001` Gaussian attack initialization versus local
uniform `[-eps, eps]` initialization immediately clamped; upstream CIFAR
`.031/.007/10/beta6` versus local `8/255,2/255,10,beta6`; and an unreproduced
upstream LR/data/WRN path. Upstream uses `ToTensor()` without normalization;
local attacks operate on the same pixel domain but `PixelModel` applies the
configured normalization once before the architecture. These differences are
not silently normalized away.

Evidence recorded: focused core command `40 passed, 1 skipped` before clone;
after clone, `ARD_TRADES_SOURCE_EVIDENCE=1 ... test_trades_upstream_differential.py`
reported `4 passed`; `verify_external.py --all` passed for both repositories.
Legacy upstream runtime/CIFAR parity, T4/T5, and full training are deferred.

以下はbaseline-readiness M4以前のbootstrapで記録されたfocused test/gateのhistorical evidenceです。
CIFARのaccuracy結果でも、今回のM4 final gate結果でもありません。

- final M5 sourceで`make lint`が完了しました。Ruff format/checkは87 files、mypyは51 source files、
  import testは`2 passed`で、train/evaluate両CLIの`--help`も成功しました。
- `make verify-milestone`が実行する
  `scripts/verify.py --changed --force --non-scientific`は、impact-selectedな22 test-file commandを完了し、
  合計`213 passed, 1 skipped`でした。skipはoptional dependency不足のupstream subprocess oracleです。
- 上記22 commandには、single-GPU PGD smokeと2基のRTX 4090を使う2-GPU DDP smoke、CPU/Gloo/DDP、
  W&B offline/mock、checkpoint/resume、saved-checkpoint evaluationが含まれます。同じsource/environmentで
  `--force`を外した再実行は22件すべてをcached passとして報告しました。
- T0–T3のunit、fixed-batch、synthetic integration、checkpoint/resume、mock/offline W&Bを変更影響ベースで実行。
- M1のimpact-selected gateでcheckpoint、synthetic/Gloo、config、data、distributed、external、imports、
  teacher、PGD、verify-gateのfocused nodesを実行。
- M2でfixed-batch formula/gradient、method switch、real two-rank Gloo entropy reductionを実行。
- M3でdeterministic two-epoch CPU epoch-boundary resumeとreal two-rank Gloo sample-state mergeを実行。
- M4でtracker guard/state、offline bundle、saved best/last PGD evaluation、artifact/panel/Parquet roundtripを
  mockまたはsynthetic dataで実行。AutoAttackはinjected fake adapterだけを実行。
- M4 diagnostics有無で`best.pt`/`last.pt`の全required checkpoint stateがexact parityである回帰を実行。
- real two-rank Gloo padded-diagnostics dedup nodeはfocused実行で`1 passed in 4.40s`。padding sampleを除外し、
  rank/order内部fieldを公開rowから除き、両rankが同じstable-ID row集合を持つことを確認。
- pinned SAAD oracleはremote/SHA/clean stateを検証しましたが、upstreamのoptional dependency不足により
  subprocess differential本体はskipされました。

実CLIのbounded smokeでは`configs/experiments/synthetic_pgd_at.yaml`を使い、PGDを1 step、1 epoch、
16 synthetic samplesとしてtrainをexit 0で完了しました。`best.pt`、`last.pt`、sample Parquet、run bundleを保存しています。
別processのevaluateは両checkpointを8 synthetic samplesで評価し、各checkpointで
`clean_accuracy=0.125`, `pgd_accuracy=0.125`でした。best/last sample Parquet、panel、evaluation run bundleも保存しました。
このaccuracyは小さな決定論的fixtureの配線確認値であり、adversarial robustnessの研究結果ではありません。

最初のM5 gateで検出したCUDA cache serializationとDDP Parquet race、その修正は
[M5 debugging note](debugging/0004-m5-test-gate-and-ddp-artifact-races.md)に記録しています。test-only resume fixtureでは
PyTorchのscheduler/optimizer call-order warningが1件、W&B offline/mock pathsではSDK deprecation warningが出ました。
いずれも上記commandのfailureではありませんが、将来のfixture/API更新で追跡します。

このsuiteは変更影響で選んだtest fileを1 commandずつ実行したものであり、repository全体を単一のmonolithic pytest
invocationで実行したという主張ではありません。22件のcached passも同一source/environment fingerprintの再利用確認です。

## 未実行・結果を主張できない範囲

- T4 limited scientific verification
- T5 production experiment
- 上記CIFAR-10 reproduction/production templateによる本訓練とmulti-seed集計
- CIFAR-100/Tiny-ImageNet本訓練
- real full AutoAttack（best/lastとも未実行）
- dependency-completeなfull upstream SAAD reproduction/differential
- live W&B online uploadと論文用artifact syncの運用確認
- repository全体のmonolithic full pytest suite

従って、現時点ではfixture smoke以外のclean/robust accuracy、upstream parity、論文再現成功を主張しません。

M0 schema v2 config migration is complete. Controlled configs now use the exact M1 MultiStepLR schedule (epoch-end milestones 100 and 150); identity is retained only for synthetic compatibility fixtures. Student/joint target softening is adversarial-branch-only and clean KD remains unchanged.
