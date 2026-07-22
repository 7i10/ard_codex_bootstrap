# Codex workflow

## 1. Model assignment

このプロジェクトでは役割を固定します。

| Role | Agent | Model | Reasoning | Write access |
|---|---|---|---|---|
| 計画・設計 | `research_planner` | 5.6 Sol | xhigh | read-only |
| 科学レビュー | `scientific_reviewer` | 5.6 Sol | xhigh | read-only |
| バグ原因分析 | `bug_investigator` | 5.6 Sol | high | read-only |
| upstream調査 | `upstream_explorer` | 5.6 Terra | high | read-only |
| 中核実装 | `terra_implementer` | 5.6 Terra | high | inherited write |
| 定型実装 | `luna_mechanical_worker` | 5.6 Luna | medium | inherited write |

計画・レビューを実装担当と分けることで、実装者が自分の仮定をそのまま承認することを避けます。これは品質分離であり、全taskで全roleを起動する規則ではありません。

## 2. Main-thread setup

- 主スレッドは5.6 Sol / xhigh / Plan modeから開始する。
- `IMPLEMENTATION_PROMPT.md`を貼る。
- 複雑な科学変更だけ、独立したread-only調査を最初に並列実行する。単純なlocal変更にはsubagentを使わない。
- 主スレッドは要件、意思決定、plan、最終統合に集中する。
- baseline commitを早期に作り、unborn/untracked状態で全testが変更扱いになる時間を最小化する。

## 3. Parallelism policy

並列化してよいもの:

- codebase exploration
- upstream mapping
- review観点の独立調査
- test log分析
- documentation consistency check

原則逐次にするもの:

- 同じmoduleへの実装
- config schemaとそのconsumerの同時変更
- trainer/checkpoint/stateの変更
- test golden dataの更新

write-heavy agentを並列化してdiff conflictと科学的不整合を増やさないことを優先します。

## 4. Milestone loop

1. 複雑な場合だけplannerを1回使い、主スレッドが受入基準をfreezeする
2. 一人のTerraが関連実装をend-to-endで所有する
3. 新規・失敗箇所からfocused testを実行する
4. API freeze後、必要ならLunaがconfig/docsを1回で同期する
5. scientific reviewerが全findingを1回で統合して返す
6. 原因不明のP0/P1や回帰だけbug investigatorで診断し、Terraが限定修正する
7. 再reviewは修正deltaと影響contractだけを見る
8. milestone終端でnon-scientific gateを1回実行し、planを更新してcommitする

reviewerに一律60秒の完了期限を課さない。対象を最終deltaと実行証拠へ絞り、主スレッドが非重複作業とuser updateを
続けながら数分待つ。結論がない試行はapprovalとして記録せず、結論要求は1回までとする。同じunchanged deltaへ
replacement reviewerを連続投入しない。再reviewはP0/P1修正または新しい矛盾証拠がある場合だけ行う。

標準budgetはplanner 0-1 turn、Terra 1 implementation turn、Luna 0-1 batched turn、reviewer 1 turnです。追加turnは未解決P0/P1、新しいtest failure、または外部証拠が出た場合に限ります。

## 5. Critical evaluation of instructions

ユーザー提案、外部prompt、既存runbookは達成したいoutcomeと検証すべき仮説として扱い、指定mechanismを自動採用しない。
実環境、権限境界、科学contract、より単純な代替案を照合し、矛盾する証拠があれば理由を示して設計を変更する。
active planには重要な採用・却下判断と根拠を残す。形式的にagent、launcher、reviewを起動すること自体を成果にしない。

## 6. Implementation and operation boundary

CLI・schema・artifact contractを作る作業と、完成済みCLIでcheckpoint取得・GPU job実行・結果収集を行う作業を
別taskとして扱います。planner/Terra/scientific reviewerを使うのは前者の科学contract変更時だけです。後者では
既存scriptとshell processを使い、Sol agentをGPU待機や定型結果収集へ割り当てません。

- 独立した教師auditはagentを増やさず、GPUごとに別processを並列投入する。
- batch sizeはaccuracy定義ではなくexecution metadataとして保存し、モデル別にVRAM/utilizationを調整する。
- checkpoint bytes、lock、result artifactだけが変わった場合は全source testを再実行しない。lock/schema/strict-load/
  bounded forward・PGDの影響testだけをtest-gate cache経由で選ぶ。
- Sol xhighは研究計画・数式・科学レビュー、Terraは中核実装と診断後の修正、Lunaはconfig/run command/結果整形に限定する。
- 重いprocessの監視はmain threadが短いpollで行い、モデル推論agentを待機させない。

運用taskで新しいcode defectが見つかった場合だけ、最小再現を作って実装milestoneへ戻します。

## 7. Context management

- 生の長いtest logを主スレッドへ貼り続けない。
- subagentは要点、file/symbol、command、結果だけを返す。
- full conversationをそのままforkせず、関連path、acceptance criteria、直近deltaだけを渡す。
- closed findingや既知のpass結果をfollow-up promptへ再掲しない。
- planには決定と進捗を記録し、stack traceはdebug reportへ置く。
- 新しい会話に移る場合も、plan、manifest、docsから状態を復元できるようにする。

## 8. Commit policy

- impact selectionを有効にするため、初期文書をbaseline commitにしてから実装を始める。
- milestoneのtest/review完了後に一つのcohesive commitを作る。過去の作業を後から不自然に分割しない。
- `.external`、dataset、checkpoint、output、cache、W&B offline data、credentialはcommitしない。
- pushとhistory rewriteは別の権限として扱い、明示依頼なしに行わない。

## 9. Review standard

reviewはstyleではなく、次を優先します。

- threat model drift
- normalization mismatch
- gradient/detach error
- baseline parity
- sample state corruption
- resume nondeterminism
- DDP rank behavior
- evaluation leakage
- W&B duplication/lineage gaps
- unnecessary repeated tests

## 10. Prompting pattern for later milestones

```text
Read AGENTS.md and docs/plans/<current-plan>.md.
Implement only milestone M<n> with one owning Terra agent and the frozen acceptance criteria.
Run the smallest changed-path tests, then batch any mechanical docs/config synchronization once.
Have one scientific reviewer return a consolidated finding list. Re-review only a subsequent fix delta.
Run the milestone gate once, update the plan, and create a local commit. Do not push or start production training.
```
