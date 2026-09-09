# 0097 — ADR CIFAR-10 self-distillation replication (decision packet 0009, option B)

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Base SHA: `cd0b571e4685fbe02d7655a22b06397c7b8d9a28` (pinned worktree
  `source-cd0b571e4685`, created 2026-09-09)
- Current milestone: M0 and M1a complete; M1c execution under way (8 of the
  20 training runs have run dirs, 6 of them terminal and successful) and its
  checkbox stays unticked until the full 20-job launch is verified; M1b and
  M2/M3 open
- Last updated: 2026-09-10

## Goal

Decision packet `docs/decisions/0009-mobile-robustness-direction-and-first-step.md`
chose option B: validate a teacher-free, EMA-of-student self-distillation
objective (Wu, Wang & Chen, "Annealing Self-Distillation Rectification
Improves Adversarial Training", ICLR 2024, arXiv:2305.12118 — ADR) against
this engine, at CIFAR-10/ResNet-18 scale plus a MobileNetV2 capacity
extension, before any ImageNet investment. This plan runs that campaign.

## Non-goals

- No ImageNet work (option A of packet 0009).
- No zero-held-out full-50k training. This project's own held-out-training-
  split convention is kept; the TRADES-vs-literature gap question is probed
  cheaply instead (see the 49k-validation pilot arm below).
- No AWP or weight-averaging beyond ADR's own EMA.
- No promotion decision. A positive result makes ADR reportable, not adopted
  into any other method's default recipe.

## Frozen scientific contract

Eight configs, each three seeds unless noted, all CIFAR-10, all epoch-199
final (checkpoints: best and last, both evaluated separately):

| # | Config | Protocol | Method | Role |
|---|---|---|---|---|
| 1 | `cifar10_r18_pgd_at.yaml` | `controlled_cifar10_r18_v1` (plain SGD) | pgd_at | existing canonical baseline |
| 2 | `cifar10_r18_pgd_at_nesterov.yaml` | `controlled_cifar10_r18_adr_v1` (Nesterov) | pgd_at | Nesterov-matched baseline for the ADR delta |
| 3 | `cifar10_r18_adr.yaml` | `controlled_cifar10_r18_adr_v1` | adr | replication cell |
| 4 | `cifar10_r18_trades.yaml` | `controlled_cifar10_r18_v1` (plain SGD) | trades | existing canonical baseline |
| 5 | `cifar10_r18_trades_adr.yaml` | `controlled_cifar10_r18_adr_v1` | adr_trades | replication cell |
| 6 | `cifar10_r18_trades_49k_validation.yaml` | `controlled_cifar10_r18_trades_49k_validation_v1` | trades | diagnostic pilot, **1 seed only** (seed 0, matched to #4's seed 0) |
| 7 | `cifar10_mobilenetv2_pgd_at.yaml` | `controlled_cifar10_mobilenetv2_adr_v1` | pgd_at | matched baseline for the MobileNetV2 extension |
| 8 | `cifar10_mobilenetv2_adr.yaml` | `controlled_cifar10_mobilenetv2_adr_v1` | adr | this project's own capacity-extension arm (paper covers R18/PreAct-R18/WRN-34-10 only, not MobileNetV2) |

Every arm: `epochs=200`, `milestones=[100,150]` `gamma=0.1`,
`epsilon=8/255` `step=2/255` `steps=10` (train) / `steps=20` (eval, CE),
`validation_fraction=0.1` except arm 6 (`0.02`), seeds
`{model_init,data_order,augmentation,train_attack,qualitative_panel}` all
set to the same per-arm-seed integer via `ARD_SEED`, `seeds.split=20260722`
fixed. ADR hyperparameters (`ema_decay=0.995`, `temperature 2.5→2.0`,
`lambda 0.7→0.95`) fixed per `docs/config/adr` and confirmed against the
pinned official code at `.external/adr` (commit
`515da0e0373f9d3de2325ad970f1f9d7e5cdcd3e`).

Evaluation for every checkpoint: clean, CE-PGD20, and AutoAttack (standard),
all on the official 10,000-example CIFAR-10 test set, `--allow-autoattack`
in a separate process per this project's evaluation-integrity rule. ADR/
ADR-TRADES checkpoints are additionally evaluated with `--weights=ema`
against `best-ema.pt`/`last.pt` (the EMA's own independently-selected best
epoch, plus the always-unambiguous last epoch) — this is the "ADR + WA"
comparator; see `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section for why
`best.pt`+`--weights=ema` is refused and why `best-ema.pt` evaluated at both
`--weights=model` and `--weights=ema` is the one strictly paper-equivalent
same-epoch comparison.

Seeds: **0, 1, 2** (`ARD_SEED`), matching the seed convention arms #1/#4's
reused historical runs already used for seed 0. Recorded here before any new
job launches.

## Historical reuse — confirmed

Investigated 2026-09-09 (read-only, field-by-field). **Both arm #1 (PGD-AT)
and arm #4 (TRADES) seed 0 are reused, not re-run.**

- PGD-AT seed 0: `outputs/scientific/pgd-at-controlled-s0-c2220f1/` —
  official best clean 0.8201 / CE-PGD20 0.5112 / AA 0.4763; last 0.8446 /
  0.4189 / 0.4036.
- TRADES seed 0 (bug-fixed): `/home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap/runs/trades-fix-v1/seed0/`
  — official best 0.8235 / 0.5066 / 0.4787; last 0.8224 / 0.4749 / 0.4499.
  **Not** `outputs/scientific/trades-controlled-s0-f0c3ace/` or
  `outputs/evaluation/trades-controlled-s0-{pgd20,aa}-f0c3ace` — those are
  the pre-fix defective run (detached clean-target KL, AA best 0.4514 / last
  0.4325) documented in `docs/debugging/0028-trades-clean-target-detached.md`.

Both resolved configs match `configs/scientific/cifar10_r18_{pgd_at,trades}.yaml`
on every contract field (optimizer including `nesterov: false`, attack,
epochs, schedule, `validation_fraction`, student/normalization,
`seeds.split`), the seed convention matches today's `ARD_SEED` templating
exactly, both AutoAttack runs carry real (non-injected) provenance
(`vcs_commit a3922004…`, `count: 10000`, `split: test`), and neither config
file has changed since (`git diff <historical-sha> HEAD -- <config>` empty
for both). `config_hash` will differ from a fresh run (schema has gained
fields since) — reuse is justified by field-by-field contract equality, not
hash equality; noted so no aggregator step treats the hash mismatch as a
rejection.

Two gaps to close as part of M1, not blocking the launch decision itself:
1. Neither run has a committed results record or `.sha256` — the evidence is
   machine-emitted bundles only (`docs/experiments/` is empty except
   `.gitkeep`). Both bundles are also outside the repo and in some cases
   gitignored (`outputs/`) or in the runtime tree, so archive/copy them and
   emit a proper record before they can be pruned.
2. Arms #2/#3/#7/#8 use different protocols (Nesterov, or MobileNetV2) and
   have no historical equivalent — this reuse applies only to #1 and #4.

## Decisions

- **Three seeds for the eight comparison-relevant arms, one seed for the
  diagnostic pilot (#6).** The pilot answers a yes/no question about
  whether a narrower held-out slice moves a already-measured gap in the
  expected direction; it does not need seed-level statistical power the way
  the ADR-vs-baseline comparison does.
- **Nesterov-matched baseline (#2) added, not a Nesterov-dropped ADR arm.**
  Per the human's decision after the first scientific-review round: keep
  ADR faithful to the paper's own hyperparameters (Nesterov included) and
  add a properly matched baseline, rather than deviating from the paper to
  match the pre-existing baseline convention.
- **MobileNetV2 gets its own matched baseline (#7), not reuse of #1/#2.**
  Architecture differs, so the baseline must too; #7 shares protocol
  `controlled_cifar10_mobilenetv2_adr_v1` with #8 (Nesterov included),
  matching #8's optimizer exactly so the MobileNetV2
  ADR-vs-baseline delta isn't confounded either.
- **Both best and last checkpoints, clean/CE-PGD20/AutoAttack all reported
  separately**, per this project's standing evaluation-integrity rule.
  Robust overfitting (best-minus-last gap) is itself part of what a
  self-distillation method might change.

## Preregistered decision rule

Primary quantity (per packet 0009): **AutoAttack accuracy gain of `adr` over
matched plain-AT, best checkpoint**, compared between MobileNetV2 (arm 8 vs
arm 7) and ResNet-18 (arm 3 vs arm 2 — the Nesterov-matched pair, not arm 1).

- **Sign confirmed** if MobileNetV2's gain is positive and exceeds
  ResNet-18's gain (packet 0009's own preregistered threshold: exceeding,
  not just matching, since the two architectures are not identical to the
  paper's own two-point table).
- **Not confirmed** if MobileNetV2's gain is smaller than or equal to
  ResNet-18's, or negative.
- **Mixed / inconclusive** if the two architectures' gains have different
  signs from what either the paper or a null hypothesis would predict, or
  if within-arm seed spread exceeds the measured gap (report both gaps with
  seed spread, do not force a verdict past what three seeds support).

Secondary, reported but not decisive:
- The same comparison on TRADES (arm 5 vs arm 4, no Nesterov-matched TRADES
  baseline exists yet, so this leg is directional only against arm 4).
- `adr`/`adr_trades`'s "ADR + WA" (EMA) numbers alongside "ADR" (student)
  numbers, both checkpoints.
- Arm 6 (49k-validation pilot) read only against arm 4's own seed-0 AA
  number: does validation_fraction 0.1→0.02 move the measured gap between
  this project's TRADES replication and literature (47.87 best / 44.99 last
  at 0.1, per the archived dashboard) toward the literature range
  (AdaAD 49.21 / RSLAD 49.23 / RSLAD-300 49.50, best-vs-best), or not.

## Stop rules

Any protocol-contract violation, missing AutoAttack provenance, attack
identity drift between training and evaluation, or a checkpoint failing
lineage validation blocks that job's result from being reported (technical
failures may retry with the identical scientific identity). A training run
whose EMA validation crashes or whose `best-ema.pt` selection disagrees with
what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section specifies is a scientific
stop for that arm, not a retry. Weak or null results (sign not confirmed)
are reported as such and do not trigger a fourth seed, a new architecture,
or scope expansion without a fresh decision packet.

## GPU-hour budget (measured, not projected)

A real 3-epoch canary of arm 3 (`cifar10_r18_adr.yaml`) ran on Hamster from
this plan's pinned SHA on 2026-09-09: `train_loss` 0.316→0.248→0.240
(monotonic decrease), student clean/PGD accuracy 21.9/16.0 → 29.8/21.5 →
34.9/24.4 %, EMA clean/PGD 10.0/10.0 → 17.4/14.8 → 33.1/25.3 % (expected lag
before catching up, `ema_decay=0.995`), throughput 1136–1144 img/s
(training-loop-only `train_seconds`≈39.4s/epoch), full epoch wall clock
(training + BOTH student and EMA validation passes) ≈54s/epoch — i.e. EMA's
extra validation pass costs ≈15s/epoch, a ≈37% per-epoch overhead versus a
single-validation method. No crash, no NaN, `best-ema.pt` written correctly
with the right selection marker.

Extrapolated at 200 epochs: ADR/ADR-TRADES (arms 3, 5, 8) ≈54s×200 ≈3.0
GPU-hours training per run; plain PGD-AT/TRADES (arms 1, 2, 4, 6, 7)
≈47s×200 ≈2.6 GPU-hours training per run (single validation pass,
extrapolated from the canary's measured single-pass cost, not yet directly
confirmed by a non-ADR canary — the first non-ADR job in this campaign
should be watched to confirm this estimate). AutoAttack, per this project's
own prior measurement, ≈1 GPU-hour per checkpoint (best + last = 2 per run).

This throughput (1140 img/s) is noticeably higher than the workspace
registry's recorded Hamster GPU figure (679 img/s,
`configs/workspace/ard_workspace_v1.json`) — the registry number appears
stale; this plan's budget uses the freshly measured figure.

| Arms | Training runs (new) | Training GPU-h | Eval (AA×2) GPU-h |
|---|---|---|---|
| 1, 4 (existing baselines, seed 0 reused, seeds 1/2 new) | 4 | 10.4 | 8 (seed 0 already evaluated; only 4 new) |
| 2, 7 (Nesterov-matched baselines, all 3 seeds new) | 6 | 15.6 | 12 |
| 3, 5, 8 (ADR family, all 3 seeds new) | 9 | 27.0 | 18 |
| 6 (pilot, 1 seed, new) | 1 | 2.6 | 2 |
| **Total** | **20** | **55.6** | **40** |

**Total ≈ 96 new GPU-hours** (plus the two already-spent, reused seed-0
runs). Across Hamster's 2 local GPUs alone, wall
clock ≈48–53 hours (~2 days) if fully packed; splitting across Hamster (2
GPUs) and Ferret (3 GPUs, `run-on-ferret`) brings wall clock to roughly
19–21 hours if perfectly parallelized. Real scheduling overhead (canary,
retries, sequential AutoAttack passes per host) will push this higher; this
is a multi-day unattended campaign, not a same-session one.

## Milestones

- [x] M0: implement ADR/ADR-TRADES, pass four scientific-review rounds, run
  a real GPU canary confirming throughput and metric sanity (this document's
  base SHA).
- [x] M1a: resolve the historical-reuse question (confirmed above) and
  finalize seeds (0, 1, 2).
- [ ] M1b: archive the two reused historical bundles (PGD-AT seed 0, TRADES
  seed 0 fixed) out of `outputs/`/the runtime tree and into a committed
  record with `.sha256`, so this campaign's aggregation has something
  durable to read instead of only machine-emitted bundles outside the repo.
- [ ] M1c: launch the 20 new training + evaluation jobs (hand-run from the
  pinned worktree under the run-bundle contract — this job set has no
  parent/continuation dependencies, matching the precedent in
  `docs/plans/0091-i100-official-test-autoattack.md`'s Decisions section for
  choosing hand-run over the DAG orchestrator); arm the watcher.
- [ ] M2: all training + evaluation jobs terminal; verify AutoAttack
  provenance and checkpoint lineage for every result.
- [ ] M3: aggregate, write the report, apply the preregistered decision
  rule, write the resulting decision packet, commit.

## Progress log

- 2026-09-09: plan written after M0 (implementation, four scientific-review
  rounds fixing P0/P1/P2 issues, and a successful real 3-epoch GPU canary of
  arm 3 on Hamster). Base SHA `cd0b571e4685fbe02d7655a22b06397c7b8d9a28`,
  worktree `source-cd0b571e4685`.
- 2026-09-09: M1a complete. Historical-reuse investigation confirmed both
  PGD-AT and TRADES seed-0 official-test+AutoAttack results are field-
  identical to the current controlled_cifar10_r18_v1 contract and reusable;
  seeds fixed at 0/1/2. 20 new training runs remain before M1c launch.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_r18_pgd_at_nesterov-s1`
  terminal event. **Nothing imported — no milestone closed.** M1c execution
  is under way but incomplete (6 of the 20 run dirs exist under
  `runs/adr-cifar10-campaign-v1/`), so there is no campaign-level result to
  aggregate, and no evaluation (official test / AutoAttack) job has run yet.
  Watcher scan of the campaign root at 16:40Z: terminal+successful training
  for `cifar10_r18_pgd_at-s1`, `cifar10_r18_pgd_at-s2`,
  `cifar10_r18_pgd_at_nesterov-s0`, `cifar10_r18_pgd_at_nesterov-s1`;
  `cifar10_r18_adr-s0` and `-s1` still running (epoch 0). The M1c checkbox is
  left unticked because the launch of all 20 jobs is not verified from this
  session, and the plan's M1c/M1b state was never updated when execution
  began — close that gap when the launch is next touched.
  Verified for the `pgd_at_nesterov-s1` bundle: `completion.json` present,
  `manifest.status=sync_pending`, error marker "no application error
  recorded", 200/200 epoch rows, both declared artifacts
  (`epoch-metrics.parquet`, `sample-stats-train.parquet`) present with
  content-addressed copies, `best.pt` and `last.pt` on disk, source SHA
  `cd0b571e4685…` clean (empty diff) from worktree `source-cd0b571e4685`,
  seeds all 1 except the fixed `split=20260722`. Held-out **validation**
  diagnostics only (not the official test set, not reportable): best epoch
  101 clean 0.8360 / PGD 0.5188; last clean 0.8524 / PGD 0.4324;
  robust-overfit gap 0.0864. Training throughput 1167 img/s, 38.6 s/epoch
  training-loop time — consistent with the plan's non-ADR ≈47 s/epoch full
  wall-clock budget line, which this first non-ADR run was meant to confirm.
  No aggregator (`scripts/aggregate_*` for this contract) exists yet; it is
  an M3 deliverable. Next terminal events should keep accumulating until all
  training + evaluation jobs are done (M2).
- 2026-09-10: postrun of the `early-check-eval-ema` terminal event
  (`eval-d16d4b5680196136ccd9`, bundle
  `runs/adr-cifar10-campaign-v1/cifar10_r18_adr-s1/train/early-check-eval-ema/`).
  **Nothing imported — no milestone closed.** This is the EMA half of an
  *early check* of one arm, not the contract evaluation: it lives in
  `early-check-eval-ema/`, not the `<run>/evaluation-ema/` path the aggregator
  reads, and it ran with `evaluation.autoattack: false`. The campaign is also
  still mid-flight, so there is no campaign-level result to aggregate either
  way.
  Watcher scan of the campaign root at 19:41Z (`--include-hand-run`), 6 run
  dirs of the planned 20: terminal+successful training for
  `cifar10_r18_pgd_at-s1`, `-s2`, `cifar10_r18_pgd_at_nesterov-s0`, `-s1` and
  now `cifar10_r18_adr-s1` (epoch 199, finished 19:38:26Z);
  `cifar10_r18_adr-s0` still running at epoch 196; the third early-check
  evaluation `early-check-eval-aa-best-model` (`eval-2e106980f84b326999a1`,
  AutoAttack) still running. Arms 4-8 (trades, trades_adr,
  trades_49k_validation, mobilenetv2 pair) have not started.
  Verified for this bundle: `completion.json` present,
  `manifest.status=sync_pending`, error marker "no application error
  recorded", source SHA `cd0b571e4685…` clean (empty diff) from worktree
  `source-cd0b571e4685`, seeds all 1 except the fixed `split=20260722` and the
  fixed `evaluation_attack=0`, `world_size=1`, effective global batch 128,
  protocol `controlled_cifar10_r18_adr_v1` (Nesterov on) — matching the frozen
  contract. Declared-artifact hash re-verification was not possible from this
  session (the sandbox denies hashing outside the repo root); it is the
  aggregator's own check at M3 and no number was imported on the strength of
  it here.

  **These are official CIFAR-10 test-set numbers (`split: test`, `count:
  10000`), CE-PGD-20, `epsilon=8/255 step=2/255 steps=20 random_start=true`,
  `autoattack.enabled=false`.** ADR seed 1 only, one seed, no baseline
  evaluated on the official test set yet, so nothing here is comparable to
  anything and none of it touches the preregistered decision rule.

  | checkpoint | weights | clean | CE-PGD-20 |
  |---|---|---|---|
  | `best.pt` | model (student) | 0.8282 | 0.5334 |
  | `last.pt` | model (student) | 0.8507 | 0.4823 |
  | `best-ema.pt` | ema ("ADR + WA") | 0.8385 | 0.5394 |
  | `last.pt` | ema ("ADR + WA") | 0.8514 | 0.4848 |

  (The first two rows come from the sibling bundle `early-check-eval-model`,
  `eval-1895f30af963e58c697d`, also terminal and successful; recorded together
  because they are the same run's student-vs-EMA comparison.) EMA is ahead of
  the student on both checkpoints and both metrics (+0.60 pp PGD at best,
  +0.25 pp at last), and the robust-overfitting gap (best minus last PGD) is
  0.0511 student / 0.0546 EMA. Single seed, no AutoAttack yet — directional
  for this run only.

  Two things for the human before M2/M3:
  1. **Test-set peeking.** These evaluations are named "early check" and were
     run on the official test set for one ADR seed while 14 of the 20 training
     runs have not started. The preregistered decision rule was written before
     any of it (this plan, 2026-09-09) and is unchanged, so the rule itself is
     not contaminated — but any further mid-campaign official-test evaluation
     should be a deliberate, logged choice rather than a habit.
  2. **Nested bundles.** These three evaluation bundles live *inside* the
     training run's own output dir (`.../cifar10_r18_adr-s1/train/
     early-check-eval-*/run-bundle/`), so the watcher surfaces them nested
     under a training run. They cannot leak into the record:
     `scripts/aggregate_adr_cifar10_replication.py` reads `<run>/evaluation/`
     and `<run>/evaluation-ema/` and refuses any row without an AutoAttack
     block, and `early-check-*` is neither. Keep it that way — the early
     checks are diagnostics and must not be wired into the aggregator.
- 2026-09-10: postrun of the `eval-1895f30af963e58c697d` terminal event
  (`cifar10_r18_adr-s1/train/early-check-eval-model`). **Nothing imported — no
  milestone closed.** This is an *early check* of one arm, not the contract
  evaluation: it lives in `early-check-eval-model/`, not the
  `<run>/evaluation/` path the aggregator reads, and it ran with
  `evaluation.autoattack: false`. The preregistered comparison needs
  AutoAttack, so no comparison is licensed from it.
  Verified for this bundle: `completion.json` `{"status":"completed",
  "results":2}`, `manifest.status=sync_pending`, error marker "no application
  error recorded", all seven declared artifacts present with their
  content-addressed copies under the manifest's own SHA-256 paths, source SHA
  `cd0b571e4685…` clean (`dirty:false`, empty diff) from worktree
  `source-cd0b571e4685`, upstream `adr` pinned at `515da0e0…` as the contract
  requires. Contract fields on both rows match the frozen plan: protocol
  `controlled_cifar10_r18_adr_v1`, Nesterov on, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `validation_fraction=0.1`, ADR
  `ema_decay=0.995` `T 2.5→2.0` `lambda 0.7→0.95`, train attack KL/rectified
  10 steps, eval attack CE 20 steps at `8/255` step `2/255` random start,
  seeds all 1 except `split=20260722` and `evaluation_attack=0`.
  **Official CIFAR-10 test set (10,000 examples), student weights
  (`--weights=model`), no AutoAttack** — clean and PGD reported separately:
  `best.pt` (sha `9181001647cb…`) clean 0.8282 / CE-PGD20 0.5334;
  `last.pt` (sha `a30c7077c0d6…`) clean 0.8507 / CE-PGD20 0.4823;
  best-minus-last PGD gap 0.0511. These are single-seed (seed 1) numbers for
  one arm with no matched baseline evaluated on the same split yet, so they
  orient only and support no ADR-vs-baseline claim. In particular they are
  **not** comparable to the `pgd_at_nesterov-s1` figures logged above, which
  are held-out *validation* diagnostics, not official test.
  Campaign state at this scan: 6 of 20 run dirs exist; training terminal and
  successful for `pgd_at-s1`, `pgd_at-s2`, `pgd_at_nesterov-s0`,
  `pgd_at_nesterov-s1`, `adr-s1`; `adr-s0` still running (epoch 196). Two
  sibling early checks on `adr-s1`: `early-check-eval-ema` is now terminal and
  successful (its own postrun will read it), `early-check-eval-aa-best-model`
  is still running. M1b and M1c stay unticked.
  `scripts/aggregate_adr_cifar10_replication.py` now exists in the working
  tree but is uncommitted and untouched by this postrun; note that it reads
  `<run>/evaluation/` and `<run>/evaluation-ema/`, so the `early-check-*` dirs
  must not be wired into it — they are diagnostics, and it correctly refuses
  any row whose AutoAttack block is absent.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_r18_adr-s1` **training**
  terminal event (`.../cifar10_r18_adr-s1/train/run-bundle/manifest.json`) —
  the parent of the two early-check evaluation entries above, logged after
  them because its headless postrun did not finish. **Nothing imported — no
  milestone closed.** Status re-derived, not taken from the event: a fresh
  `campaign_watch.py --once --emit-existing --include-hand-run` scan of the
  campaign root returns `terminal: true`, `success: true`,
  `failure_class: null` on the line whose `path` is that manifest.
  Verified for the training bundle: `completion.json`
  `{"status":"completed"}`, `manifest.status=sync_pending`, error marker "no
  application error recorded", 200 of 200 epoch rows
  (`epoch_metrics_complete: true`), both declared artifacts present —
  `epoch-metrics.parquet` (`fab2fe25…`) and `sample-stats-train.parquet`
  (`664090fa…`) — each with a content-addressed copy under its own SHA-256
  path. Source SHA `cd0b571e4685…` clean (`dirty: false`, empty diff) from
  worktree `source-cd0b571e4685`, `.external/adr` pinned at `515da0e0…`,
  protocol `controlled_cifar10_r18_adr_v1` with Nesterov on, `world_size=1`,
  effective global batch 128, seeds all 1 except the fixed `split=20260722`
  and `evaluation_attack=0`. Hashes were not recomputed here — this session's
  sandbox refuses to read outside the repo root — so artifact integrity rests
  on the content-addressed paths matching the manifest, and the aggregator
  still owns the real check at M3.
  **ADR stop-rule checks pass.** `best.pt`, `best-ema.pt` and `last.pt` are
  all on disk; EMA validation ran every epoch and never crashed (all 200 rows
  of `run-bundle/metrics.jsonl` carry `val_clean_accuracy_ema` and
  `val_pgd_accuracy_ema`); and `best-ema.pt` was selected independently of the
  student — EMA's best validation epoch is 110, the student's is 103 — which
  is what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section requires.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable): student best epoch 103 clean 0.8392 / PGD 0.5396, last clean
  0.8612 / PGD 0.4988, robust-overfit gap 0.0408; EMA best epoch 110 clean
  0.8524 / PGD 0.5498, EMA last clean 0.8628 / PGD 0.4996. EMA ahead of the
  student at the best checkpoint on validation, the same direction as the
  official-test table above — a consistency check on the selection logic, not
  independent evidence.
  Cost: 1141 img/s, `train_seconds` 39.4 s/epoch — matching the plan's ADR
  canary measurement (1136–1144 img/s, ≈39.4 s/epoch training-loop), so the
  ≈3.0 GPU-h per ADR run budget line stands as written.
  Operational note: the headless postruns for all three `adr-s1` events
  (`orchestration/ardx/claude-runs/20260909T1938*`, `…T1939*`, `…T1940*`) left
  empty `.json` outputs, and the log of the training one contains only
  "Ignoring 44 permissions.allow entries from .claude/settings.json: this
  workspace has not been trusted". Their plan edits were left staged and
  uncommitted; this session commits them together with this entry. The trust
  dialog for the `/home/islab/...` repo-root spelling still needs to be
  accepted once interactively, per CLAUDE.md.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_r18_adr-s0` terminal
  event (bundle `runs/adr-cifar10-campaign-v1/cifar10_r18_adr-s0/train/
  run-bundle/`). **Nothing imported — no milestone closed.** The campaign is
  still mid-flight, this run has produced no `evaluation/` or
  `evaluation-ema/` output at all, and
  `scripts/aggregate_adr_cifar10_replication.py` is still uncommitted, so
  there is no campaign-level result to aggregate and no official-test number
  for this run to import.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, error marker "no application error
  recorded", 200/200 epoch rows (`epoch_metrics_complete: true`, epoch 199,
  `global_step` 70400), both declared artifacts (`epoch-metrics.parquet`,
  `sample-stats-train.parquet`) present with their content-addressed copies
  under the manifest's own SHA-256 paths, `best.pt`, `best-ema.pt` and
  `last.pt` on disk, source SHA `cd0b571e4685…` clean (`dirty: false`, empty
  diff) from worktree `source-cd0b571e4685`, upstream `adr` pinned at
  `515da0e0…`. Contract fields match the frozen plan: protocol
  `controlled_cifar10_r18_adr_v1`, Nesterov on, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `validation_fraction=0.1`, ADR
  `ema_decay=0.995` `T 2.5→2.0` `lambda 0.7→0.95`, train attack KL/rectified
  10 steps at `8/255` step `2/255` random start, selection and evaluation
  attack CE 20 steps, seeds all 0 except the fixed `split=20260722`,
  `world_size=1`, effective global batch 128. Artifact hashes were not
  recomputed — this session's sandbox refuses to hash outside the repo root —
  so integrity rests on the content-addressed paths matching the manifest,
  and the aggregator still owns the real check at M3.
  **ADR stop-rule checks pass.** EMA validation ran every epoch and never
  crashed (all 200 rows of `run-bundle/metrics.jsonl` carry
  `val_clean_accuracy_ema` and `val_pgd_accuracy_ema`), and `best-ema.pt` was
  selected independently of the student — EMA's best validation epoch is 104,
  the student's is 103 — which is what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR
  section requires.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to the official-test table logged above for
  `adr-s1`):

  | weights | checkpoint | epoch | clean | PGD |
  |---|---|---|---|---|
  | student | best | 103 | 0.8402 | 0.5394 |
  | student | last | 199 | 0.8634 | 0.4938 |
  | EMA | best-ema | 104 | 0.8458 | 0.5506 |
  | EMA | last | 199 | 0.8658 | 0.4994 |

  Student robust-overfit gap (best minus last PGD) 0.0456. EMA is ahead of
  the student at the best checkpoint (+1.12 pp validation PGD), the same
  direction as `adr-s1`'s validation and official-test figures — a
  consistency check on the selection logic, not independent evidence.
  Arm 3 (ADR) now has two terminal seeds, and arm 2 (its Nesterov-matched
  baseline) also has two, so a **validation-only** within-arm spread can be
  read for the first time. Student best-checkpoint validation PGD: ADR 0.5394
  (s0) / 0.5396 (s1); Nesterov PGD-AT 0.5184 (s0) / 0.5188 (s1). Student
  robust-overfit gap: ADR 0.0456 / 0.0408; Nesterov PGD-AT 0.0888 / 0.0864.
  Both arms are strikingly tight across their two seeds (≤0.04 pp on best
  PGD), and ADR sits about 2.1 pp above its matched baseline with roughly half
  the robust-overfitting gap. **This is not the preregistered comparison**,
  which is AutoAttack on the official 10,000-example test set at the best
  checkpoint, over three seeds; validation PGD is the model-selection metric
  these runs optimized against, so it is expected to flatter the arm whose
  selection it drove. Two seeds are directional for those seeds only.
  Cost: 1106 img/s, `train_seconds` 40.7 s/epoch — consistent with the plan's
  ADR canary (1136–1144 img/s, ≈39.4 s/epoch training-loop) and with the
  ≈3.0 GPU-h per ADR run budget line.
  Campaign state at the 19:43Z scan (`--include-hand-run`), 7 of the 20 run
  dirs: training terminal and successful for `pgd_at-s1`, `-s2`,
  `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1`; `cifar10_r18_trades-s1` now
  running (the first arm-4 job started); `adr-s1`'s third early check
  `early-check-eval-aa-best-model` (AutoAttack) still running. Arms 5-8
  (trades_adr, trades_49k_validation, mobilenetv2 pair) and seed 2 of arms 2/3
  have not started. M1b and M1c stay unticked.
- 2026-09-10: postrun of the `early-check-eval-aa-best-model` terminal event
  (`eval-2e106980f84b326999a1`, bundle `runs/adr-cifar10-campaign-v1/
  cifar10_r18_adr-s1/train/early-check-eval-aa-best-model/`) — the third and
  last of the `adr-s1` early checks, and the campaign's first AutoAttack job.
  **Nothing imported — no milestone closed.** Same reason as its two siblings:
  it is an *early check*, living in `early-check-eval-aa-best-model/` rather
  than the `<run>/evaluation/` path
  `scripts/aggregate_adr_cifar10_replication.py` reads, so it is a diagnostic
  and must not be wired into the aggregator. The campaign is also still
  mid-flight, so there is no campaign-level result to aggregate.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `failure_class: null` on the line whose `path` is that
  manifest, with `completion.json` present, `manifest.status=sync_pending` and
  error marker "no application error recorded". `docs/experiments/` is still
  empty apart from `.gitkeep`, so there was nothing to collide with on the
  idempotency check.
  **Read limitation — no numbers logged for this bundle.** Unlike the earlier
  postruns in this log, this session's sandbox refuses every read outside the
  repo root (`ls`, `cat`, `find` and ad-hoc `python3 -c` were all blocked on
  `.../ard-runtime/...`), so `evaluation-results.json`, the manifest body and
  the declared-artifact list could not be opened at all. Everything above comes
  from the sanctioned scanner (`scripts/ardx/campaign_watch.py`) and
  `ard.cli.status`, which are on the permission allow-list. The AutoAttack
  accuracy for `adr-s1` `best.pt` (student weights) is therefore **not**
  recorded here. It is a diagnostic that no import depends on; to log it, add
  the runtime tree to the session (`/add-dir /home/islab/workspace-local/
  shunsuke.naito/ard-runtime`) and re-run this postrun. No number was inferred,
  and no contract or artifact check was reported as passed on evidence this
  session could not see.
  Campaign state at the 20:15Z scan (`--include-hand-run`), 8 of the 20 run
  dirs: training terminal and successful for `pgd_at-s1`, `-s2`,
  `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1`; `cifar10_r18_trades-s1`
  running at epoch 34 and `cifar10_r18_trades_adr-s0` (the first arm-5 job)
  running at epoch 0. All three `adr-s1` early checks are now terminal, so that
  diagnostic set is closed. Arms 6-8 (trades_49k_validation, mobilenetv2 pair)
  and seed 2 of arms 2/3 have not started. M1b and M1c stay unticked.
