# Experiment protocol

## 1. Run taxonomy

### Dev

- 小さなunit/fixed batch
- W&B disabled可
- 結果を論文表に使わない

### Smoke

- 少数sample、最大2 epoch
- end-to-end、checkpoint、resume、trackingを確認
- W&B offline推奨
- 精度比較には使わない

### Audit and pilot

- `configs/audit/` は教師checkpoint・normalization・PGD-20スクリーニング専用。W&Bなし、AA再現ではない。
- `configs/pilot/` は実CIFARを5 epochだけ動かすengineering確認で、canonical paper resultではない。
- 旧`repro` tierはresolved configの互換性だけに残し、新規のrunnable reproduction configは作らない。

### Production

- 事前登録したconfig、seed、teacherで実行
- W&B必須
- best/last保存
- 正式評価は別run
- 論文候補の集計対象

Tierと実行profile（world size/per-rank batch）は別軸である。global batch=128でも、1 GPUのper-rank 128と
2 GPUのper-rank 64は通常のlocal BatchNorm統計が異なるため、同一比較へ混ぜない。推奨順序は1 GPU・W&Bなしの
bounded teacher audit、必要ならmicro smoke、2 GPU・per-rank 64の5 epoch pilot、最後に同じ2 GPU profileで
canonical 200 epoch productionである。

Teacher audit batchはaccuracy定義ではなく実行メタデータであり、Chenは128、Bartoldsonは16から開始する。
BartoldsonはVRAMに余裕がある場合だけ32へ上げ、OOM時だけ8へ下げる。

Example execution profile (pilot or production):

```bash
export ARD_PER_RANK_BATCH_SIZE=64 ARD_DEVICE=cuda
export WANDB_ENTITY=shunsuke-n-waseda-university WANDB_PROJECT=single-teacher-ard WANDB_GROUP_CHEN=chen-cifar10-r18
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module ard.cli.train --config configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml
```

The Bartoldson pilot uses `configs/pilot/cifar10_r18_rslad_bartoldson2024_adversarial_wrn94_16.yaml` and
`WANDB_GROUP_BARTOLDSON` instead. First run the W&B-free teacher screening on one GPU:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ard.cli.audit_teacher \
  --config configs/audit/chen2021_ltd_wrn34_10.yaml
```

Swap the audit config and GPU for Bartoldson. These commands are procedures only; neither audit nor pilot has been run.

Evaluation is a separate saved-checkpoint process and must preserve the checkpoint's training execution identity.

Tracking diagnostics are explicit: `off` emits no diagnostics/sample statistics, `summary` emits scalar metrics and
Parquet statistics without image Tables, and `panel` emits fixed-ID media plus sample statistics. Production requires
`panel`. Local best/last checkpoints remain on scientific checkpoint cadence; model artifact publication follows the
configured interval (5 epochs in checked-in templates) and always publishes both at finalization. PGD step-loss traces
default to off to avoid per-step device synchronization.

## 2. 初期baseline matrix

| Dataset | Student | Methods | Teachers | Seeds |
|---|---|---|---|---|
| CIFAR-10 | ResNet-18 | PGD-AT, TRADES, RSLAD, entropy-only, student-only, joint-risk, full SAAD | good/medium/bad代表 | 探索1、最終3 |
| CIFAR-100 | ResNet-18 | 主要4～6手法 | 代表teacher | 最終3 |
| CIFAR-10 | MobileNetV2 | 主要手法 | 代表teacher | 最終3 |
| Tiny-ImageNet | 選択student | 勝ち残り手法 | 代表teacher | 条件付き |

bootstrapではこのmatrixを実行せず、configとlauncherを作成します。

## 3. Baseline fairness

- dataset、augmentation、epochs、optimizer、scheduler、attack budget、student、teacherを比較内で揃える。
- method固有の必須設定以外を変えない。
- full SAAD等の複合手法は、RSLAD系との計算量差を明示する。
- teacher training costとstudent training marginal costを分けて記録できるようにする。

## 4. Checkpoint reporting

各runについて:

- best clean
- best quick robust
- last clean
- last quick robust
- best-to-last robust gap
- best checkpoint epoch

をW&B summaryへ保存します。正式AutoAttackはbestとlastの両方を評価します。

## 5. Teacher sensitivity

最終分析では、各methodについてteacher間の:

- mean
- standard deviation
- worst case
- best case

を報告できるようにします。単一teacherの最高値だけを主要主張にしません。

## 6. Qualitative analysis

固定sample panelを用い、次を比較します。

- entropy-only vs joint-riskでweightが変わるsample
- robustly learnable/unlearnable proxyの4象限
- forgetting eventが発生したsample
- best epochとlast epochの変化
- good teacherとbad teacherの差

## 7. Evaluation lifecycle

1. train runがbest/last model artifactを生成
2. evaluation runがartifactを入力として取得
3. clean/PGD/AutoAttackを実行
4. evaluation configとlibrary versionを保存
5. train groupへ紐付け
6. analysis runが複数seed/teacherを集計

## 8. Compute placement

- 安定した2-GPUサーバー: final multi-seed、DDP、長時間run
- 不安定な3-GPUサーバー: 独立single-GPU sweep、teacher screening、evaluation、oracle ablation
- サーバー間DDPを行わない
- config、Git SHA、W&B groupでサーバー間の実験を統合する
