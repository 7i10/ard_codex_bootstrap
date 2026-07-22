# Reproduction status

最終更新: 2026-07-22

## 現在の到達点

single-teacherの同一training loop上で、次の4 ablationを選択できます。

| Config method | 役割 | 固定された追加契約 |
|---|---|---|
| `rslad` | uniform RSLAD baseline | hard-label fallbackなし |
| `rslad_entropy` | teacher entropy weighting | Shannon entropy、係数`5`、clip/mean preservation/fallbackなし |
| `rslad_student` | student robust-margin risk | EMA `0.9`、epoch 0はuniform warmup |
| `rslad_joint` | student risk × teacher overconfidence | EMA `0.9`、epoch 0はuniform warmup |

全4手法でattackはpixel-space `[0,1]`、`Linf`、`epsilon=8/255`、
`step_size=2/255`、10 steps、random startです。checkpoint selection attackは同じbudgetのhard-label CEです。
attack identity/hashはbudgetだけでなく、resolved quantity、loss/target、temperature、model modeを含む
`AttackConfig`全14 fieldから作ります。これらのいずれかを変更した実験は同一baselineとして扱いません。

保存対象は`best.pt`と`last.pt`の両方です。resumeはepoch boundaryだけを保証し、optimizer、scheduler、
scaler、RNG、sampler epoch、sample state、global step、tracking run ID、best-selection stateを復元します。
復元時点ですでに全epochが完了しているno-op resumeは、summary、sample statistics、artifact一覧を更新しません。
ただしprior terminal status/completion marker、required artifact集合、file artifactのsource/local hashが完全であることを
検証し、不完全またはdriftしたterminal lineageは拒否します。
mid-epoch resumeとworld-size変更は再現保証の対象外です。

## CIFAR template configs

次の8ファイルはloader/guardに適合する実行テンプレートです。実測精度やupstream-exact scheduleを表す結果ではありません。

- `configs/reproduction/cifar10_r18_{rslad,rslad_entropy,rslad_student,rslad_joint}.yaml`
- `configs/production/cifar10_r18_{rslad,rslad_entropy,rslad_student,rslad_joint}.yaml`

全テンプレートは未解決値を暗黙defaultにしません。実行前に以下を環境へ設定します。

| Variable | 内容 |
|---|---|
| `ARD_CIFAR10_ROOT` | 既存CIFAR-10 root（configはdownloadしない） |
| `ARD_TEACHER_CHECKPOINT` | frozen ResNet-18 teacher checkpoint |
| `ARD_TEACHER_CHECKPOINT_SHA256` | lowercase 64文字のcheckpoint SHA-256 |
| `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_GROUP` | W&B identity。比較する4手法と対応evalでgroupを共有する |
| `ARD_SEED`, `ARD_OUTPUT_ROOT` | seedと出力root |
| `ARD_TRAIN_EPOCHS`, `ARD_BATCH_SIZE`, `ARD_LEARNING_RATE` | 事前登録したschedule |
| `ARD_MOMENTUM`, `ARD_WEIGHT_DECAY`, `ARD_NUM_WORKERS` | optimizer/data loader設定 |
| `ARD_DEVICE`, `ARD_VALIDATION_FRACTION` | deviceと固定validation split率 |

scheduleは現時点でupstream-exactと認定されていないため、テンプレート内に推測値を埋めていません。
比較runでは全4手法へ同じ値を与え、resolved configを保存してください。

## 実際のCLI

例としてRSLAD reproduction templateを使う場合:

```bash
PYTHONPATH=src python -m ard.cli.train \
  --config configs/reproduction/cifar10_r18_rslad.yaml

PYTHONPATH=src python -m ard.cli.train \
  --config configs/reproduction/cifar10_r18_rslad.yaml \
  --resume "$ARD_OUTPUT_ROOT/cifar10-r18-rslad-repro-s$ARD_SEED/last.pt"

PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/reproduction/cifar10_r18_rslad.yaml \
  --checkpoint-dir "$ARD_OUTPUT_ROOT/cifar10-r18-rslad-repro-s$ARD_SEED"
```

別outputへ評価する場合は`--output`、training resolved configがcheckpointの兄弟にない場合は
`--train-config`を指定します。単一checkpointだけを評価するときは`--checkpoint-dir`の代わりに
`--checkpoint`を使います。

full AutoAttackはconfig overrideと二重のCLI opt-inが必要です。

```bash
PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/reproduction/cifar10_r18_rslad.yaml \
  --checkpoint-dir "$ARD_OUTPUT_ROOT/cifar10-r18-rslad-repro-s$ARD_SEED" \
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

## Production/reproduction guard

`repro`/`production` trainは、実体のあるmain Git HEAD、一致する`external.lock.yaml`、cleanな
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

以下はbootstrap中に記録されたfocused test/gateの範囲です。CIFARのaccuracy結果ではありません。

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

M0 schema v2 config migration is complete. Configured schedulers use identity to preserve existing execution semantics; exact schedule migration is deferred to M1. Student/joint target softening is adversarial-branch-only and clean KD remains unchanged.
