# Documentation index

- [Reproduction status](REPRODUCTION_STATUS.md): 実装済み機能、実行済み検証、未実行の重い実験、実際のCLI手順
- [Research decisions](RESEARCH_DECISIONS.md): 今回の研究・実装で固定した方針
- [Implementation specification](IMPLEMENTATION_SPEC.md): リポジトリ構造と責務
- [Scientific invariants](SCIENTIFIC_INVARIANTS.md): attack、gradient、checkpoint、評価の不変条件
- [Test strategy](TEST_STRATEGY.md): tier、変更影響、pass cache、数値tolerance、GPU排他
- [W&B protocol](WANDB_PROTOCOL.md): tier/state、group/job type、artifact、固定sample table
- [Experiment protocol](EXPERIMENT_PROTOCOL.md): run tier、baseline、seed、評価
- [Upstream baselines](UPSTREAM_BASELINES.md): pinned SAAD/TRADES SHA、ライセンス証拠、既知差分
- [Teacher config fragments](../configs/teachers/): strict RobustBench teacher configs; checkpoints are registered explicitly
- [Codex workflow](CODEX_WORKFLOW.md): Sol/Terra/Lunaの役割分担

## CLI entry points

```bash
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml>
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml> --resume <output>/last.pt
PYTHONPATH=src python -m ard.cli.evaluate --config <experiment.yaml> --checkpoint-dir <output>
python scripts/verify.py --changed
```

`evaluate` は保存済みcheckpointだけを読み、`--checkpoint-dir`ではconfigの
`evaluation.checkpoints`（既定は`both`）に従って`best.pt`と`last.pt`を別々に評価します。
full AutoAttackは通常のtestやtrainからは起動せず、evaluation configで明示的に有効化した上で
`--allow-autoattack`を付けた別processだけが実行できます。

実装計画は `docs/plans/`、重大なバグの記録は `docs/debugging/` にあります。CIFAR本訓練を始める前に、
[Reproduction status](REPRODUCTION_STATUS.md) の未実行項目とproduction guardを確認してください。
