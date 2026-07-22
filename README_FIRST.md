# ARD research platform

このrepositoryは、single-teacher Adversarial Robust Distillationの実装済み研究基盤です。同一training loopで
PGD-AT、TRADES、RSLADと3つのRSLAD ablationを切り替え、epoch-boundary resume、saved-checkpoint
evaluation、best/last artifact、sample diagnostics、W&B lineageを一貫して扱います。

## Platform entry points

```bash
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml>
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml> --resume <output>/last.pt
PYTHONPATH=src python -m ard.cli.evaluate --config <experiment.yaml> --checkpoint-dir <output>
python scripts/verify.py --changed
python scripts/sync_wandb.py --root <outputs>
```

実行可能なCIFAR-10 templateと具体的な環境変数、AutoAttack opt-inは
[`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)を参照してください。設計・科学契約の入口は
[`docs/README.md`](docs/README.md)です。

## Operational boundaries

- `repro`/`production`はGit/upstream/teacher/tracking lineage guardを通過しなければ開始しません。
- exact resumeはepoch boundaryだけで、best/last checkpointとtracking identityを保持します。
- full AutoAttackはsaved checkpointを使う別evaluation processでのみ明示的に実行します。
- CIFAR/Tiny-ImageNet本訓練、real full AutoAttack、T4/T5は自動testの対象外です。
- Tiny-ImageNetのpaper aggregationはobserved training split identityの永続化が完了するまでdeferredです。

`AGENTS.md`、`PLANS.md`、`.codex/`、`.agents/`、`IMPLEMENTATION_PROMPT.md`は、継続開発と科学レビューの
規則・履歴としてrepository内に残しています。現時点で主張できる検証範囲と未実行項目は
[`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)を確認してください。
