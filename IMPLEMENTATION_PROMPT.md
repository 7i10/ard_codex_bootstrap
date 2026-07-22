# Single-Teacher ARD Research Platform 実装プロンプト

## 実行条件

このプロンプトは、リポジトリ直下をVS Codeで開き、Codexを **gpt-5.6-sol / xhigh / Plan mode** にして使用する。

## 貼り付け用プロンプト

あなたは、このリポジトリのリード研究エンジニアとして、single-teacher Adversarial Robust Distillation研究基盤を実装してください。

最初に `AGENTS.md`、`PLANS.md`、`docs/README.md` と、そのリンク先を全て読んでください。既存コードがある場合は、削除・全面書き換えを先に行わず、現在の構造、Git状態、依存関係、未コミット変更を調査してください。

目的は、RSLADを軽量な開発基盤とし、SAADのentropy weightingを再現した上で、学生側の時変robust learnabilityと教師のoverconfidenceを独立に計測・組合せできる、再現性の高い研究コードベースを作ることです。主提案はsingle robust teacher設定で実装し、dual-teacherは主経路に入れません。full SAADは必須比較対象ですが、初期実装の研究基盤はRSLAD中心にします。

### 0. 作業オーケストレーション

作業開始時に、独立したread-only subagentを使って次を並列に実行してください。

1. `research_planner` に、要件、既存リポジトリ、リスク、実装順序、受入基準を整理させる。
2. `.external/saad` が存在する場合は `upstream_explorer` に、公式SAAD実装の実行経路、attack、loss、normalization、checkpoint、evaluation、CLI、ライセンス状況をファイル・シンボル単位で調査させる。存在しない場合は、後述のbootstrap方針だけを調査させる。

両者の結果を待ち、主スレッドで `docs/plans/0001-bootstrap-and-core.md` に統合した実行計画を書いてください。計画には、進捗チェックボックス、変更対象、テスト選択、リスク、完了条件を含めます。

実装は原則として `terra_implementer` に、設定・ドキュメント・単純なボイラープレートは `luna_mechanical_worker` に任せてください。二つのwrite agentが同じファイル群を同時編集してはいけません。各milestoneの後に `scientific_reviewer` でread-onlyレビューを行い、重大な不整合が疑われる場合は `bug_investigator` と `$ard-bug-hunt` を使って原因を特定してから、Terra実装担当に修正させてください。

計画・判断・科学レビューはSol、主要実装はTerra、明確で反復的な作業はLunaという役割分担を維持してください。

### 1. スコープ

以下を実装対象とします。

- Python package: `src/ard/`
- YAMLベースの構成管理。構成は型検証し、解決済みconfigを各runへ保存する。
- CIFAR-10、CIFAR-100。Tiny-ImageNetはdataset adapterとconfigまで用意し、bootstrap中に本訓練しない。
- Student: ResNet-18、MobileNetV2をregistry経由で選択可能にする。
- Teacher: checkpoint metadata、normalization、architecture、threat modelを明示するadapter/registryを実装する。
- Attack: student PGD、評価用PGD、必要に応じてSAAD fast innerを独立したstrategyとして実装する。
- Objectives: PGD-AT、TRADES、RSLAD、RSLAD+SAAD entropy weighting。
- Signals: teacher entropy、student robust-margin EMA、robust correctness frequency、robust forgettingの拡張点。
- Policies: uniform、entropy-only、student-only、joint-risk gate。
- State: stable sample indexをキーとした `SampleStateStore`。checkpoint/resumeで完全復元する。
- Engine: single GPUと同一サーバー内DDP。サーバー間DDPは想定しない。
- Evaluation: clean、PGD、AutoAttackを訓練プロセスから分離してsaved checkpointから実行する。
- Tracking: W&Bを正式backendにし、全production experimentをW&Bへ保存する。
- Upstream: 公式SAADは `.external/saad` に固定commitでcloneし、通常はGit管理対象外とする。
- Tests: 変更影響ベースのtest gate、成功結果cache、fixed-batch differential regression、GPU smoke、W&B offline integration。
- Docs: 実装後の実行方法、再現状況、既知差分、W&B schema、テスト戦略を更新する。

### 2. 非目標

bootstrap実装中には次を行わないでください。

- CIFARの300 epoch本訓練
- Tiny-ImageNetの本訓練
- full AutoAttack評価
- ImageNet-1K対応
- dual-teacherの主実装
- upstreamコードの無断コピーまたはライセンス不明コードのvendoring
- 研究結果を良く見せるためのattack弱化、test-time防御、評価条件変更

最大2 epoch、少数batch、固定seedのsmokeまでに制限してください。重い検証はコマンドと予想コストを提示するだけにします。

### 3. 外部SAAD管理

次の構造を採用してください。

```text
.external/saad/          # gitignore。公式repoの作業コピー
external.lock.yaml       # repo URL、固定commit、取得日、ライセンス確認結果
scripts/bootstrap_external.py
scripts/verify_external.py
```

`bootstrap_external.py` は以下を満たすこと。

- 公式remoteをcloneする。
- 初回にexact commitをlockへ記録し、以後は明示更新なしにHEADを進めない。
- 既存directoryのremoteとcommitを検証する。
- 未コミット変更を破壊しない。
- network失敗時に部分cloneを成功扱いしない。
- ライセンスファイルが見つからない場合は、その事実を `docs/UPSTREAM_BASELINES.md` に記録し、コードを自repoへコピーしない。

upstreamは、再現とdifferential regressionのoracleとして扱います。自作runtimeが常に `.external` をimportする構造にはしません。full SAADについては、まずupstream launcher/wrapperで再現可能にし、その後必要性とライセンスを確認してclean-room portを検討してください。

### 4. 必須アーキテクチャ

最低限、次の責務分離を実装してください。

```text
src/ard/
  cli/
  config/
  data/
  models/
  attacks/
  objectives/
  signals/
  policies/
  state/
  engine/
  evaluation/
  tracking/
  analysis/
```

必須interface:

- `AttackGenerator`: inner maximizationのみを担当
- `DistillationObjective`: outer objectiveのみを担当
- `SampleSignal`: per-sample診断値を計算
- `WeightPolicy`: signalからloss weightを生成
- `SampleStateStore`: sample-index単位のEMA、correctness、forgetting等を保持
- `ExperimentTracker`: W&Bとdisabled/offline test backendを抽象化
- `TeacherAdapter`: preprocessing、logit取得、入力gradient要否、metadataを統一

新しいmethodのためにtraining loop全体を複製してはいけません。attack、objective、signal、policyをconfigで交換可能にしてください。

### 5. 科学的実装要件

`docs/SCIENTIFIC_INVARIANTS.md` を契約として扱ってください。特に以下をテスト可能な形で守ります。

- pixel-space、normalization-space、epsilon、step sizeを混同しない。
- teacher parameterはfreezeする。ただしteacher input gradientが必要な手法では、入力gradientまで無効化しない。
- KLの向き、temperature、`T^2` scaling、clean/adv入力の組合せを明示する。
- attack中のstudent/teacher train/eval modeとBatchNorm挙動を明示する。
- best checkpointとlast checkpointを両方保存する。
- resume時にoptimizer、scheduler、scaler、RNG、sampler epoch、sample state、W&B run IDを復元する。
- AutoAttackはsaved checkpointを読み込む別processで実行する。
- 評価attackを訓練attackと同一視しない。

### 6. SAADとの差分を検証できる実装

最初に次のablationを同一engineで実行可能にしてください。

1. RSLAD
2. RSLAD + teacher entropy weighting
3. RSLAD + student robust learnability signal
4. RSLAD + entropy + student signalのjoint-risk gate

student signalの初期実装はrobust-margin EMAとします。サンプル `i`、時点 `t` について、教師confidenceと学生unlearnabilityを別々に記録し、policyで組み合わせます。サンプルをdatasetから削除せず、高リスク時はteacher KDを弱めてhard-label/AT側へfallbackさせます。

数式・正規化・weight rangeはconfig化し、単体テストを作ってください。SAAD entropy weightingの式は、公式論文とupstream実装を照合してから実装し、推測で補完しないでください。

### 7. W&B要件

W&Bは `src/ard/tracking/` のadapter経由でのみ使用してください。training code内に散在する直接 `wandb.log` を禁止します。

experiment tierを次に分けます。

- `dev`: W&B disabled可
- `smoke`: W&B offlineまたはdisabled
- `repro`: W&B online/offline-sync、`reproduction` tag
- `production`: W&B必須。onlineまたは同期完了が保証されるoffline-syncのみ

production guard:

- W&B entity/projectが設定されていない場合はfail fastする。
- `mode=disabled` を禁止する。
- exact config、Git SHA、dirty diff、external commit、environment、teacher checkpoint hash、seedを保存する。
- rank 0だけがW&B runを初期化する。
- run IDをlocal manifestに保存し、`resume="allow"` または同等の安全な再開を行う。
- pending offline runを検出・同期するscriptを作る。
- production runを同期確認前に削除しない。

全production train runで最低限保存するもの:

- 解決済みconfigとmanifest
- epoch/step metrics
- best/last summary
- best checkpointとlast checkpointをmodel artifactとしてversion管理
- sample statisticsのParquet artifact
- 固定sample panelのW&B Table
- stdout/stderr、環境情報、Git diff

定性的確認用に、固定sample IDを使って次をW&B Tableへ記録してください。

- clean image、adversarial image、可視化用perturbation
- true label
- student clean/adv prediction
- teacher prediction
- teacher entropy
- student robust margin EMA
- joint risk
- applied KD weight
- clean/robust correctness

各epochで画像を大量送信してはいけません。Tableは初期、指定間隔、best、last等の節目だけにし、histogramも疎な頻度で記録します。`wandb.watch` はdebug設定以外では無効にします。

trainとevaluationは同じW&B `group` に置き、`job_type=train|evaluation|analysis` で区別してください。run名にはdataset、student、teacher、method、seed、short SHAを含めます。seedを除く比較単位をgroup名にします。

### 8. 効率的なテストシステム

同じ成功テストを、関連入力が変わっていないのに繰り返してはいけません。次を実装してください。

```text
scripts/verify.py
src/ard/testing/impact.py
src/ard/testing/cache.py
.cache/test-gate/          # gitignore
```

`verify.py` はGit diffと変更pathから必要なtest tierを選びます。成功結果は、少なくとも次のfingerprintでcacheします。

- test command
- 対象testファイルhash
- 関連source/configファイルhash
- Python/PyTorch/CUDA環境fingerprint
- external SAAD commit
- test data fixture version

fingerprintが同一で直前結果がpassなら、実行せず「cached pass」と報告します。`--force` で再実行可能にします。stochasticな本実験結果をunit test cacheとして扱ってはいけません。

テストtier:

- T0: format、lint、type、config schema、import
- T1: changed-module unit tests
- T2: fixed-batch numerical/gradient/differential regression
- T3: tiny subsetでのend-to-end GPU smoke、checkpoint/resume、W&B offline
- T4: milestone scientific verification。限定epoch、限定seed
- T5: production trainingとAutoAttack。自動test suiteには含めない

最低限のmarker:

```text
unit, gpu, slow, upstream, regression, smoke, wandb, scientific
```

ルール:

- 変更直後は最小の高情報量testを実行する。
- 失敗時は `pytest --lf` 相当で失敗箇所だけ再実行する。
- 成功済みのexact commandをコード・環境変更なしに再実行しない。
- full non-scientific suiteはmilestone終了時だけ。
- production experimentをtest代わりにしない。
- W&B unit testはnetworkを使わず、offline/temp directoryまたはmockを使う。
- GPU testはlockを取り、同一GPUで競合させない。

attack/objective/normalization/teacher adapterを変更した場合は、固定batch regressionとgradient contract testを必須にしてください。docsだけの変更でGPU testを走らせてはいけません。

### 9. 必須テスト

少なくとも次を実装してください。

- stable sample indexがshuffle/augmentation/DDP samplerでも元sampleへ対応する
- PGD projection、epsilon bound、input clamp、gradient source
- teacher parameter gradientが不要手法で常に`None`
- teacher input gradientが必要手法で計算可能
- KL direction、temperature、weight range、finite値
- RSLAD fixed-batch regression
- SAAD entropy weighting fixed-batch regression
- upstreamが存在する場合のdifferential regression
- sample stateのupdate、serialization、resume
- checkpoint再開が短い決定論的runと一致する
- best/last checkpointが別々に保存される
- rank 0以外がW&Bを初期化しない
- W&B offline run、resume、artifact manifest
- tiny subsetのsingle-GPU smoke
- CUDAが複数利用可能な場合だけDDP smoke

数値回帰のtoleranceは、dtype、AMP、determinismを考慮して根拠をdocsへ書いてください。広すぎるtoleranceでtestを通してはいけません。

### 10. 実装milestone

次の順序で進めてください。

#### M0: Repository/bootstrap

- package scaffold
- `.gitignore`
- `pyproject.toml`
- environment情報
- `.external` bootstrap/lock
- CLI skeleton
- test gate skeleton
- docsの整合

#### M1: Scientific core

- typed config
- indexed datasets
- models/teacher registry
- PGD attack contracts
- objectives
- trainer/checkpoint/resume
- single-GPU smoke

#### M2: Baseline reproduction

- PGD-AT、TRADES、RSLAD
- SAAD entropy weighting
- upstream wrapper
- fixed-batch differential regression
- reproduction status document

#### M3: Student-aware components

- robust-margin EMA
- sample state store
- entropy-only、student-only、joint-risk policies
- oracle mask extension point
- ablation configs

#### M4: W&B and evaluation

- tracker adapter
- production guard
- qualitative Tables
- model/sample-stat Artifacts
- offline sync
- saved-checkpoint evaluation

#### M5: Verification and handoff

- impact-aware verify commands
- all non-scientific tests
- smoke tests
- scientific review
- docs更新
- userが実行すべきreproduction/production commandsを提示

各milestone後に、同じtest commandを無条件再実行せず、`scripts/verify.py --changed` で必要testを選択してください。reviewで新しい懸念が出た場合だけ追加testを選びます。

### 11. 完了条件

次を全て満たした時だけ完了としてください。

- `python -m ard.cli.train ...` でtiny smokeが動く。
- `python -m ard.cli.evaluate ...` がsaved checkpointから動く。
- RSLAD、entropy-only、student-only、joint-riskをconfig差分だけで切替できる。
- fixed-batch regressionがpassする。
- W&B offline integration testがpassする。
- production設定でW&B無効時にfail fastする。
- best/last checkpointとsample stateがresume可能である。
- `.external` commitがlockされる。
- `scripts/verify.py --changed` が重複実行を避ける。
- docsが実装と一致する。
- bootstrap中に本訓練・full AutoAttackを行っていない。

### 12. 最終報告

最後に、次を簡潔かつ具体的に報告してください。

- 実装したmilestoneと主要ファイル
- upstream commitとライセンス確認結果
- 実行したtest、cached passとして省略したtest、未実行の重いtest
- W&B integrationの確認内容
- 既知の差分、未解決事項、科学的リスク
- reproduction用コマンド
- production train/evaluation用コマンド
- reviewで発見・修正した重大バグ

テストがpassしたと主張する場合は、実際に実行したcommandと結果に基づいてください。推測で「動くはず」と書かないでください。コード変更をpushしないでください。
