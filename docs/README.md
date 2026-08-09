# Documentation index

- [Experiment dashboard](EXPERIMENT_DASHBOARD.md): 人間向けの目的、条件、確定結果、W&B run分類
- [History-routing v2 results](HISTORY_ROUTING_V2_RESULTS.md): exact online historyを用いたBest-oriented介入の2-seed結果とDevelopment No-Go判定
- [FF/current-wrong forecasting status](FFNR_FORECASTING_STATUS.md): plateau GTの実現可能性、L/T/S/D CPU ablation、GPU follow-up境界
- [FFNR human image review](FFNR_HUMAN_REVIEW.md): role-blind CIFAR panelのHTMLレビューと分類基準
- [FFNR human review results](FFNR_HUMAN_REVIEW_RESULTS.md): 200枚の人手判定結果と研究上の扱い
- [FFNR human review analysis](FFNR_HUMAN_REVIEW_ANALYSIS.md): クラス別誤り率、教師混同行列、panel条件付き分析
- [FFNR next evidence status](FFNR_NEXT_EVIDENCE_STATUS.md): chance-adjusted GT、cross-seed Teacher情報、Teacher-correct subset、IRT gate
- [Seed-0 signal audit](SIGNAL_AUDIT.md): Student/Joint signalの探索的関連、周期checkpoint、正式判定の境界
- [Reproduction status](REPRODUCTION_STATUS.md): 実装済み機能、実行済み検証、未実行の重い実験、実際のCLI手順
- [Research decisions](RESEARCH_DECISIONS.md): 今回の研究・実装で固定した方針
- [Implementation specification](IMPLEMENTATION_SPEC.md): リポジトリ構造と責務
- [Scientific invariants](SCIENTIFIC_INVARIANTS.md): attack、gradient、checkpoint、評価の不変条件
- [Test strategy](TEST_STRATEGY.md): tier、変更影響、pass cache、数値tolerance、GPU排他
- [W&B protocol](WANDB_PROTOCOL.md): tier/state、group/job type、artifact、固定sample table
- [Experiment protocol](EXPERIMENT_PROTOCOL.md): run tier、baseline、seed、評価
- [Upstream baselines](UPSTREAM_BASELINES.md): pinned SAAD/TRADES SHA、ライセンス証拠、既知差分
- [Teacher config fragments](../configs/teachers/): strict RobustBench teacher configs; checkpoints are registered explicitly
- [Experiment taxonomy](EXPERIMENT_PROTOCOL.md): audit, pilot, and canonical production separation
- [Codex workflow](CODEX_WORKFLOW.md): Sol/Terra/Lunaの役割分担
- [Ferret execution protocol](FERRET_EXECUTION_PROTOCOL.md): fixed-SHA remote GPU runs from Hamster
- [Legacy campaign archive](../tools/internal/legacy_campaign/README.md): 完了済みHamster/Ferret運用コードの非公開・非runtimeアーカイブ

## CLI entry points

```bash
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml>
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml> --resume <output>/last.pt
PYTHONPATH=src python -m ard.cli.evaluate --config <experiment.yaml> --checkpoint-dir <output>
PYTHONPATH=src python -m ard.cli.status --root <output-root> --format markdown
python scripts/verify.py --changed
```

`evaluate` は保存済みcheckpointだけを読み、`--checkpoint-dir`ではconfigの
`evaluation.checkpoints`（既定は`both`）に従って`best.pt`と`last.pt`を別々に評価します。
full AutoAttackは通常のtestやtrainからは起動せず、evaluation configで明示的に有効化した上で
`--allow-autoattack`を付けた別processだけが実行できます。

実行中のepoch、step、更新時刻、terminal stateは`run-bundle/manifest.json`からstatus CLIが導出します。
跨hostのlive viewはW&Bを使い、Git管理されたdashboard本文をprocess監視のために手編集しません。

Teacher audit (W&B-free, PGD screening) should run on one GPU first. After that, use the two-GPU pilot and production
commands documented in `EXPERIMENT_PROTOCOL.md`; set `WANDB_PROJECT=single-teacher-ard` (and teacher-specific group
variables) for pilot/production. Smoke runs may remain disabled and do not upload to W&B; production may not disable it.

実装計画は `docs/plans/`、重大なバグの記録は `docs/debugging/` にあります。CIFAR本訓練を始める前に、
[Reproduction status](REPRODUCTION_STATUS.md) の未実行項目とproduction guardを確認してください。
