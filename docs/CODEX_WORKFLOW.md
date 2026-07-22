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

標準budgetはplanner 0-1 turn、Terra 1 implementation turn、Luna 0-1 batched turn、reviewer 1 turnです。追加turnは未解決P0/P1、新しいtest failure、または外部証拠が出た場合に限ります。

## 5. Context management

- 生の長いtest logを主スレッドへ貼り続けない。
- subagentは要点、file/symbol、command、結果だけを返す。
- full conversationをそのままforkせず、関連path、acceptance criteria、直近deltaだけを渡す。
- closed findingや既知のpass結果をfollow-up promptへ再掲しない。
- planには決定と進捗を記録し、stack traceはdebug reportへ置く。
- 新しい会話に移る場合も、plan、manifest、docsから状態を復元できるようにする。

## 6. Commit policy

- impact selectionを有効にするため、初期文書をbaseline commitにしてから実装を始める。
- milestoneのtest/review完了後に一つのcohesive commitを作る。過去の作業を後から不自然に分割しない。
- `.external`、dataset、checkpoint、output、cache、W&B offline data、credentialはcommitしない。
- pushとhistory rewriteは別の権限として扱い、明示依頼なしに行わない。

## 7. Review standard

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

## 8. Prompting pattern for later milestones

```text
Read AGENTS.md and docs/plans/<current-plan>.md.
Implement only milestone M<n> with one owning Terra agent and the frozen acceptance criteria.
Run the smallest changed-path tests, then batch any mechanical docs/config synchronization once.
Have one scientific reviewer return a consolidated finding list. Re-review only a subsequent fix delta.
Run the milestone gate once, update the plan, and create a local commit. Do not push or start production training.
```
