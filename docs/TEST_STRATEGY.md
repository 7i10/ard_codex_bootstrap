# Efficient test strategy

## 1. 目的

変更に近い科学的バグを最小コストで先に検出し、無関係なCPU/GPU testを反復しません。
test回数ではなく、単位時間あたりの情報量と再現可能な実行記録を重視します。

## 2. Test tiers

| Tier | 内容 | 実行条件 | pass cache |
|---|---|---|---|
| T0 | import、config、CLI、verification tooling | 関連source/config変更 | 可 |
| T1 | changed-module unit | 対象module変更 | 可 |
| T2 | fixed-batch formula/gradient/upstream regression | attack/objective/model/signal/policy変更 | 可 |
| T3 | tiny smoke、checkpoint/resume、tracking、conditional GPU/DDP | engine/tracking/DDP変更、milestone境界 | 可 |
| T4 | 限定epoch scientific verification | 明示承認されたmilestoneだけ | 不可 |
| T5 | production training/full evaluation | 人間が承認したexperimentだけ | 不可 |

T5は自動test suiteではありません。full AutoAttackとCIFAR本訓練をT0–T3へ入れません。

## 3. Change-impact selection

```bash
python scripts/verify.py --changed
python scripts/verify.py --changed --base <commit>
python scripts/verify.py --tier T2
python scripts/verify.py --failed
python scripts/verify.py --changed --force
```

`--changed`はtracked/staged/untracked pathsを含み、unborn repositoryでも動作します。
不正なbaseはfail closedです。既知pathは`src/ard/testing/impact.py`のmappingでfocused testsへ変換し、
未知source、helper、fixture変更はavailable tests全体へ保守的にfallbackします。

主要mapping:

| Changed path | 必須検証 |
|---|---|
| `configs/**` / `src/ard/config/**` | repository config resolve、独立したevaluation seed default/明示値、schema、関連integration |
| `data/**` | stable ID、split、sampler、checkpoint integration |
| `models/**` | normalization、teacher hash/freeze、attack integration |
| `attacks/**` | projection、gradient、fixed batch、conditional GPU smoke |
| `objectives/**` | formula、KL direction、pre-reduction weights、DDP regressions |
| `signals/**` / `policies/**` / `state/**` | range、EMA/update、serialization、DDP merge |
| `engine/**` | checkpoint/resume、synthetic training |
| `tracking/**` | guard、offline W&B、rank zero、sync、artifact manifest |
| `evaluation/**` | saved-checkpoint load、threat hash、PGD/AA adapter、artifact roundtrip |
| `docs/**` | docs-only T0 selection |

## 4. Pass cache

cacheは`.cache/test-gate/results.jsonl`へappendし、commandと関連入力のfingerprintが完全一致する
successful resultだけを再利用します。fingerprintには次を含めます。

- exact commandと選択されたsource/test/config paths
- Python/platform/PyTorch/CUDA/GPU capability
- external SAAD SHA
- `PYTHONPATH`, `PYTHONHASHSEED`, `CUDA_VISIBLE_DEVICES`, `WANDB_MODE`（存在するもの）

source/test/config/environment/external SHAが変わればcache missです。failureはpassとして再利用せず、
同一fingerprintの後発failureは古いpassを無効化します。`--force`はmatching passも再実行します。
T4/T5 markerを含み得るcommandは収集結果を確認してcache対象外にします。

## 5. Test ordering

1. static inspection、config resolve、tensor shape/domainを確認する。
2. 最小unit/fixed-batch nodeを実行する。
3. 直接関係するintegration/resumeを実行する。
4. CUDA/DDPが必要な変更だけconditional smokeを実行する。
5. milestone終端でimpact-selected non-scientific gateを一度実行する。

同じcommandを再実行する場合は、source変更、environment変更、failure修正、GPU nondeterminism確認など
fingerprintが変わる理由を記録します。

## 6. Numerical tolerance rationale

toleranceは値の由来ごとに固定し、失敗を隠すために広げません。

| Contract | Tolerance | 根拠 |
|---|---:|---|
| config rationalとselection-attack equality | `rtol=0`, `atol=1e-15` | Python floatで同じ文字列分数を再計算する決定論的比較 |
| FP32 fixed-batch logits/loss/weight | `rtol=0`, `atol=1e-7` | 同一CPU演算経路の単精度roundingだけを許容 |
| PGD `Linf` projection | `epsilon + 1e-7` | FP32 subtract/project/clampの境界rounding |
| real distributed FP32 reduction | `rtol=0`, `atol=1e-6` | collective/reduction順序による単精度rounding |
| frozen scalar/gradient oracle（float64） | `abs=1e-14`～`1e-15` | closed-form double-precision oracle |

checkpoint state、sample ID集合、manifest identity/hash、artifact magic bytes、command/state transitionsは
approximateにせずexact equalityで検証します。AMP/GPU toleranceを追加する場合はdtype、演算順、referenceを
test内に記録し、単に数値を大きくしてはいけません。

## 7. GPU and DDP exclusion

CUDA testは`torch.cuda.is_available()`でconditional skipし、GPUを見えないsandboxでpass扱いにしません。
DDP/Gloo testsはTCPStore socketを必要とするため、restricted sandboxで失敗する場合はpermissioned環境で
同じfocused commandを実行し、環境差を記録します。server間DDPは対象外です。

repositoryのpytest fixtureにはprocess間GPU file lockはまだ実装されていません。通常gateはGPU testを
serial実行しますが、独立した複数process/server jobを同じGPUへ向ける場合はoperatorが排他を取ります。
Linux例:

```bash
CUDA_VISIBLE_DEVICES=0 flock /tmp/ard-gpu-0.lock \
  /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest -q tests/smoke/test_gpu_pgd.py
```

lock名と`CUDA_VISIBLE_DEVICES`を一致させます。GPU testを`pytest-xdist`で無条件並列化しません。

## 8. W&B and evaluation tests

- live projectを使わず、mockまたはoffline modeとtemporary directoryを使う。
- rank-zero init、stable ID、resume intent、manifest、Table/Image、artifact hash、`prepare_finish`順序、
  `sync_cursor` partial retryを含むsync state machineを確認する。
- W&B init failureでfailed manifestが残ること、artifact publication failureでentry/local copyがrollbackされること、
  failed finishのfailure snapshotと`exit_code=1`、正常時の`exit_code=0`を確認する。
- content-addressed `local_path`、同名artifactのversion history、publication failure時に今回のentry/new digest copyだけを
  rollbackしてprior versionを保持することを確認する。
- failed `offline_sync` runはupload後に`sync_state=synced`となってもapplication `status=failed`を保持することを確認する。
- Parquet要求時は`PAR1` magicとstable sample IDsを確認し、optional dependency不在では明示failする。
- evaluationはportable dataset identityとroot provenanceの分離、student/method/training protocol/checkpoint identity、
  saved selection attack全14 fieldとのexact equality/canonical hash、failed-manifest lifecycle、best/last threat hashを確認する。
- evaluation resultのtraining/evaluation seed分離と、seed/loader batch/complete attack/AutoAttack設定からなる
  evaluation protocol identityを確認する。
- 集計はevaluation/training dataset、student、method、training/evaluation protocol、complete threat、evaluation seedを
  固定し、world size/effective global batchの混在を拒否する一方、training seed/teacherを比較軸として許可し、
  各training runのbest/lastがexactly oneであることを確認する。
- 独立default `0`のevaluation seedをPGD random start/panel selectionとAutoAttackの両方が使うことを確認する。
- AutoAttack automated testはinjected fake adapterでLinf mapping、explicit seed、bounded batch、mode restorationだけを確認する。
- 完了checkpointからのno-op resumeでprior terminal status/completion marker、required artifact、source/local hashを検証し、
  summary、sample-stat bytes、artifact historyが変わらないことを確認する。
- Tiny-ImageNet split digestの`computed`、`computed-and-matched`、training config-only `expected-unverified`を区別し、
  root差をidentity差にせずcontent差を拒否する。training observed identity永続化まではTiny T5/paper aggregationを行わない。
- full AutoAttackはsaved checkpointから別processで実行するT5 operationであり、自動実行しない。

## 9. 現在の実行境界

final M5 sourceでは次を実行しました。

- `make lint`: Ruff format/check 87 files、mypy 51 source files、import tests `2 passed`、train/evaluateの
  両`--help`が成功。
- `make verify-milestone` / `scripts/verify.py --changed --force --non-scientific`: impact-selectedな
  22 test-file commandが合計`213 passed, 1 skipped`。skipはoptional upstream subprocess oracle。
- 同じsource/environmentから`--force`を外したcommand: 22 commandすべてがcached pass。
- coverageにはreal single-GPU PGD smoke、2基のRTX 4090による2-GPU DDP smoke、CPU/Gloo/DDP、W&B offline/mock、
  checkpoint/resume/evaluationを含む。

これはtest fileごとのimpact-selected executionであり、単一のmonolithic pytest invocationではありません。
bootstrap中のsynthetic checkpoint/resume、diagnostics有無のfull-checkpoint exact parity、real two-rank Gloo
diagnostics dedupのfocused証拠`1 passed in 4.40s`も保持します。

実CLIのbounded smokeでは`configs/experiments/synthetic_pgd_at.yaml`をPGD 1 step、1 epoch、16 samplesでtrainし、
exit 0とbest/last、sample Parquet、run bundleを確認しました。別evaluate processはbest/lastを8 synthetic samplesで
評価し、両方でclean/PGD accuracyが`0.125`、best/last Parquet、panels、evaluation bundleが生成されました。
これはfixture smokeであり、accuracyの科学的主張には使いません。

初回gate failureと修正は[M5 debugging note](debugging/0004-m5-test-gate-and-ddp-artifact-races.md)に記録しています。
test-only resume fixtureのPyTorch scheduler-before-optimizer warning 1件と、W&B offline/mock pathsのSDK deprecation
warningsはpassを妨げませんでしたが、fixture/API maintenance riskとして残ります。

T4、T5、CIFAR本訓練、real full AutoAttack、dependency-complete full SAAD、monolithic full pytest suiteは
実行済みとして扱いません。live W&B online uploadとTiny-ImageNet本訓練/集計も未実行です。詳細は
[Reproduction status](REPRODUCTION_STATUS.md)を参照してください。
