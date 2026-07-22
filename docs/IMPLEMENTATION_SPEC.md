# Implementation specification

## 1. 目的

single-teacher ARDにおけるbaseline再現、per-sample診断、sample-wise policy、評価、W&B可視化を同一基盤で行います。新規methodを追加してもtraining loopを複製せず、attack、objective、signal、policyの組合せとして表現します。

## 2. 推奨リポジトリ構造

```text
.
├── .agents/skills/
├── .codex/agents/
├── .external/                 # gitignore
├── configs/
│   ├── datasets/
│   ├── students/
│   ├── teachers/
│   ├── methods/
│   ├── tracking/
│   └── experiments/
├── docs/
│   ├── plans/
│   └── debugging/
├── scripts/
├── src/ard/
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── attacks/
│   ├── objectives/
│   ├── signals/
│   ├── policies/
│   ├── state/
│   ├── engine/
│   ├── evaluation/
│   ├── tracking/
│   ├── analysis/
│   └── testing/
├── tests/
│   ├── unit/
│   ├── regression/
│   ├── smoke/
│   └── integration/
├── external.lock.yaml
├── AGENTS.md
├── PLANS.md
├── Makefile
└── pyproject.toml
```

## 3. 中核interface

### AttackGenerator

入力、label、student、teacher、attack configを受け、adversarial inputと必要なdiagnosticsを返します。projection domain、gradient source、model modeを実装内で暗黙化せず、configまたは型で表します。

### DistillationObjective

unreduced per-sample lossを返せることを必須にします。policy適用前のhard-label、KD、regularization成分を分離して記録できるようにします。

### SampleSignal

各sampleについて値とvalidity maskを返します。少なくともteacher entropyとrobust-margin EMAを実装します。

### WeightPolicy

signal dictionaryからKD weight、必要ならtemperature/fallback maskを生成します。出力範囲、clipping、reductionはconfig化します。

### SampleStateStore

stable sample IDをキーに、EMA、correctness count、forgetting count、last update等を保持します。CPU residentを基本とし、batch単位で必要部分だけGPUへ移します。DDPの所有・集約方針を明示します。

### ExperimentTracker

`log_metrics`、`log_table`、`log_artifact`、`set_summary`、`finish`等の小さなinterfaceを提供します。W&B固有objectがengineへ漏れないようにします。

## 4. 実装上の重要分離

- Datasetは `(image, label, sample_id)` を返す。
- Teacherのnormalizationはcheckpoint metadataとadapterで管理する。
- attackとobjectiveは別moduleにする。
- sample stateの更新時点を明示する。
- train-time quick PGDと正式evaluationを分離する。
- controlled SAAD studentはraw-pixel identity adapterを使う（canonical architecture id: `saad_resnet18_cifar_v1`）。MultiStepLRはepoch-endにstepし、epoch 0–99/100–149/150–199でLRは0.1/0.01/0.001。controlled attacksはPGD-10 KL (teacher_clean) とselection PGD-20 CE。
- upstream実装はruntime dependencyではなく、oracle/wrapperとする。

## 5. 初期method構成

| Method ID | Attack | Objective | Signals | Policy |
|---|---|---|---|---|
| `pgd_at` | student PGD | hard-label CE | none | uniform |
| `trades` | TRADES inner | TRADES | none | uniform |
| `rslad` | RSLAD-compatible | hard + robust KD | none | uniform |
| `rslad_entropy` | same as RSLAD | same | teacher entropy | entropy weight |
| `rslad_student` | same as RSLAD | same | robust-margin EMA | student weight |
| `rslad_joint` | same as RSLAD | same | entropy + margin EMA | joint-risk gate |

full SAADはupstream wrapperから開始し、必要時に別method IDでclean-room portします。

## 6. 実装済みCLI

現行entry pointとchecked-in reproduction configの使用形は次のとおりです。

```bash
PYTHONPATH=src python -m ard.cli.train \
  --config configs/reproduction/cifar10_r18_rslad.yaml
PYTHONPATH=src python -m ard.cli.train \
  --config configs/reproduction/cifar10_r18_rslad_joint.yaml
PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/reproduction/cifar10_r18_rslad.yaml \
  --checkpoint-dir <training-output>
PYTHONPATH=src python -m ard.cli.evaluate \
  --config configs/reproduction/cifar10_r18_rslad.yaml \
  --checkpoint-dir <training-output> \
  --allow-autoattack evaluation.autoattack=true
python scripts/verify.py --changed
```

evaluationは必ずsaved checkpointを読み、`--checkpoint-dir`ではconfigの`evaluation.checkpoints`に従います。
full AutoAttackはconfig overrideと`--allow-autoattack`の両方を付けた別processだけが実行します。

## 7. 完了時の成果物

- 再現可能なconfig
- fixed-batch parity結果
- smoke checkpoint
- W&B offline integration artifact
- production用launcher
- docs内のreproduction status
- 未実行の重い実験一覧

### M0 schema v2 migration

全 experiment config は `schema_version: 2` と versioned `protocol.id`、7-field `seeds`、top-level optimizer/scheduler、per-rank/global batch identity を保持する。scheduler は M0 では `identity` とし、既存実行の意味を変えない。exact schedule は M1 で導入する。
