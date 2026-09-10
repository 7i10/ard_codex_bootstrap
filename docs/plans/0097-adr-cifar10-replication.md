# 0097 — ADR CIFAR-10 self-distillation replication (decision packet 0009, option B)

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Base SHA: `cd0b571e4685fbe02d7655a22b06397c7b8d9a28` (pinned worktree
  `source-cd0b571e4685`, created 2026-09-09)
- Current milestone: M0, M1a, M1b complete. M1c execution under way: **19 of
  the 20 training runs are terminal and successful** in the local campaign
  root. With `cifar10_r18_trades_adr-s2` in, **the entire ResNet-18 leg (arms
  1–6) is training-complete at its full seed count**, as is arm 7; **exactly
  one training run is left in the whole campaign**, `cifar10_mobilenetv2_adr-s0`
  (arm 8 seed 0), which has never been launched. Canonical contract evaluations
  exist for **8 of 20** runs (model weights: `r18_pgd_at-s1/-s2`,
  `r18_pgd_at_nesterov-s0/-s1/-s2`, `r18_adr-s0`, `mobilenetv2_pgd_at-s1/-s2`)
  plus **1 of 9** EMA-weights evaluations (`r18_adr-s0`) — every one checked
  against the literature/consistency expectations in the standing
  autonomous-check instruction, with **no deviation and no bug found in any of
  them**. So **11 terminal training runs still have no terminal contract
  evaluation**. With `mobilenetv2_pgd_at-s1` (12:01Z entry below) the
  MobileNetV2 baseline arm now has **two of its three seeds evaluated**, and
  the two agree; the MobileNetV2 decisive pair still has no treatment seed at
  all. At the 12:02Z full-root scan **five evaluations read `running`**
  (`r18_trades_adr-s2`, `r18_trades-s1`, `-s2`,
  `r18_trades_49k_validation-s0` (retry), and `r18_trades_adr-s0`, a
  human-started lane-K retry that began at 12:02:02Z, outside any postrun),
  **one is terminal-failed and still on disk** (`mobilenetv2_adr-s1`, model
  weights) with five more failure directories deleted or moved aside — all six
  the same AutoAttack unbatched-forward CUDA OOM (`r18_adr-s1`, `r18_adr-s2`,
  `r18_trades_adr-s0`, `r18_trades_adr-s1`, `r18_trades_49k_validation-s0`,
  `mobilenetv2_adr-s1`; see decision packet 0010 and the entries below) — and
  **three `evaluation-ema` bundles read `running` but are dead**, SIGTERM'd
  with no terminal state (`r18_adr-s1`, `r18_trades_adr-s0`,
  `mobilenetv2_adr-s1`). Four training-terminal runs have **no evaluation
  bundle on disk at all**: `mobilenetv2_pgd_at-s0` and `mobilenetv2_adr-s2`
  (never started), plus `r18_adr-s2` and `r18_trades_adr-s1` (failure
  directories deleted or moved aside, no retry started). M1c's checkbox stays
  unticked until the full 20-job launch is verified complete; M2/M3 open. The
  `_load_arm_seed` run-directory-prefix bug in
  `scripts/aggregate_adr_cifar10_replication.py` was found and fixed earlier
  this session (`dir_prefix` per arm) before it could break M3.
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
- [x] M1b: archive the two reused historical bundles (PGD-AT seed 0, TRADES
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
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_r18_trades-s1` **training**
  terminal event (`.../cifar10_r18_trades-s1/train/run-bundle/manifest.json`) —
  the first arm-4 (TRADES) job of this campaign to finish. **Nothing imported —
  no milestone closed.** The campaign is still mid-flight, this run has no
  `evaluation/` output yet, and no arm has a complete official-test result, so
  there is no campaign-level result to aggregate.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py --once
  --emit-existing --include-hand-run` scan of the campaign root returns
  `terminal: true`, `success: true`, `failure_class: null` on the line whose
  `path` is that manifest. `docs/experiments/` is still empty apart from
  `.gitkeep`, so the idempotency check had nothing to collide with.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application error
  recorded", 200/200 epoch rows in both `epoch-metrics.jsonl` and
  `run-bundle/metrics.jsonl` (`epoch_metrics_complete: true`, epoch 199,
  `global_step` 70400), both declared artifacts present —
  `epoch-metrics.parquet` (`5c19a2d9…`) and `sample-stats-train.parquet`
  (`7cafdf05…`) — each with a content-addressed copy under its own SHA-256 path,
  `best.pt` and `last.pt` on disk (and no `best-ema.pt`, correct for a non-ADR
  arm), source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff
  `e3b0c442…`) from worktree `source-cd0b571e4685`. Artifact hashes were not
  recomputed — this session cannot hash outside the repo root — so integrity
  rests on the content-addressed paths matching the manifest; the aggregator
  owns the real check at M3.
  Contract fields match arm 4 of the frozen plan exactly: protocol
  `controlled_cifar10_r18_v1`, method `trades` (`trades_beta 6.0`, `adr: null`),
  SGD with **`nesterov: false`** as arm 4 requires, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `validation_fraction=0.1`, train attack
  KL/`student_clean` 10 steps at `8/255` step `2/255` random start, selection and
  evaluation attack CE 20 steps at the same budget, seeds all 1 except the fixed
  `split=20260722` and `evaluation_attack=0`, `world_size=1`, effective global
  batch 128, identity normalization. `git diff cd0b571e4685 HEAD` on
  `configs/scientific/cifar10_r18_trades.yaml` and `configs/protocols/` is empty,
  so the contract has not drifted since the pin.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to the official-test tables logged above for
  `adr-s1`): best epoch 157 clean 0.8322 / PGD 0.5164; last (epoch 199) clean
  0.8338 / PGD 0.4844; robust-overfit gap 0.0320. Both figures re-read from
  `metrics.jsonl` and they match the manifest summary.
  **Budget finding — the plan's non-ADR cost line does not fit TRADES.** The
  budget table charges arms 1/2/4/6/7 ≈47 s per full epoch (≈2.6 GPU-h per run),
  a figure extrapolated from the ADR canary's single-validation cost and since
  confirmed for PGD-AT (`pgd_at_nesterov-s1`: 1167 img/s, 38.6 s/epoch training
  loop). TRADES is intrinsically more expensive — it adds a clean forward pass
  to the objective — and ran at **998 img/s, ≈45.1 s/epoch training loop** while
  uncontended (epochs 0-123). The training loop alone therefore nearly consumes
  the whole 47 s full-epoch allowance, so the ≈2.6 GPU-h line is too low for the
  three remaining TRADES runs (arm 4 seed 2 and arm 6); ≈3.0 GPU-h, the ADR
  line, is the better estimate. This run's own wall clock was 3 h 27 min
  (19:43:26Z → 23:10:41Z, 62.2 s/epoch averaged), but that figure is inflated by
  GPU contention and is not the uncontended cost: throughput fell to 510-610
  img/s at epochs 124-134, recovered to ≈645 for 135-163 and ≈730 for 164-199 as
  co-scheduled jobs came and went. The per-epoch rows carry no timestamps or
  validation-pass timing, so the uncontended *full*-epoch cost cannot be
  separated out from this run. Contention changes only wall clock, not results:
  `deterministic: true`, fixed seeds and fixed batch size make the numbers
  independent of throughput.
  **First contract evaluations have started.** Two `<arm>-s<seed>/train/
  evaluation/` bundles now exist and are running — `cifar10_r18_adr-s0` and
  `cifar10_r18_pgd_at_nesterov-s1`. These are the real thing, not the
  `early-check-*` diagnostics: their paths are exactly what
  `scripts/aggregate_adr_cifar10_replication.py` reads
  (`run_root/<arm>-s<seed>/train/evaluation/evaluation-results.json`, line 271),
  so the M3 aggregation will find them. `adr-s0` additionally needs an
  `evaluation-ema/` sibling (`has_ema`), which has not appeared yet.
  Campaign state at the 23:11Z scan (`--include-hand-run`), **9 of 20 run
  dirs**: training terminal and successful for `pgd_at-s1`, `-s2`,
  `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1` and now `trades-s1`;
  `trades_adr-s0` running at epoch 145 and `cifar10_mobilenetv2_adr-s1` — the
  first arm-8 job, and the campaign's first MobileNetV2 run — just started. Arms
  6 and 7 (trades_49k_validation, mobilenetv2 pgd_at) and seed 2 of arms 2/3/4
  have not started. M1b and M1c stay unticked.
- 2026-09-10: postrun of the `eval-ef916db6bef00134891b` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_pgd_at_nesterov-s1/train/
  evaluation/`) — **the campaign's first complete contract evaluation**: arm 2
  (Nesterov-matched PGD-AT baseline), seed 1, both checkpoints, clean +
  CE-PGD-20 + AutoAttack on the official test set. **Nothing imported — no
  milestone closed.** The campaign is still mid-flight (9 of 20 run dirs, one
  of 20 contract evaluations terminal), so there is no campaign-level result to
  aggregate; `docs/experiments/` is still empty apart from `.gitkeep`, so the
  idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py --once
  --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `failure_class: null` on the line whose `path` is that
  manifest.
  Verified for this bundle: `completion.json` `{"status": "completed",
  "results": 2}`, `manifest.status=sync_pending`, `error-marker.txt` reads "no
  application error recorded", all seven declared artifacts present on disk with
  their content-addressed copies under the manifest's own SHA-256 paths
  (`resolved_evaluation_config.yaml`, `evaluation-lineage.json`,
  `evaluation-results.json`, `panel-{best,last}.jsonl`,
  `sample-stats-{best,last}.parquet`), plus `autoattack-{best,last}.json`.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`) from
  worktree `source-cd0b571e4685`. Artifact hashes were not recomputed — this
  session's sandbox refuses to hash outside the repo root — so integrity rests
  on the content-addressed paths matching the manifest; the aggregator owns the
  real check at M3.
  Contract fields match arm 2 of the frozen plan exactly: protocol
  `controlled_cifar10_r18_adr_v1` with **`nesterov: true`**, method `pgd_at`
  (`adr: null`), `epochs=200`, `milestones=[100,150] gamma=0.1`,
  `validation_fraction=0.1`, `world_size=1`, effective global batch 128,
  identity normalization, `deterministic: true`. Evaluation identity:
  `evaluation.autoattack: true`, `checkpoints: both`, dataset `split: test`,
  `count: 10000` on both rows, CE-PGD-20 at `epsilon=8/255 step=2/255 steps=20
  random_start=true`, `evaluation_seed=0`, `weights: model`. AutoAttack is the
  **standard** version with real (non-injected) provenance —
  `expected_commit == vcs_commit == a39220048b3c9f2cca9a4d3a54604793c68eca7e`,
  the pinned upstream the aggregator requires — run in its own process at
  `epsilon=8/255`, `Linf`, batch 128, seed 0. Training seeds all 1 except the
  fixed `split=20260722` and `evaluation_attack=0`.

  **Official CIFAR-10 test set (10,000 examples), student weights, arm 2
  (`pgd_at_nesterov`) seed 1 — clean, CE-PGD-20 and AutoAttack reported
  separately, best and last kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best.pt` | `47dbfc493a62…` | 0.8219 | 0.5097 | 0.4741 |
  | `last.pt` | `db5fac975459…` | 0.8406 | 0.4196 | 0.4041 |

  Robust-overfitting gap (best minus last): 9.01 pp on CE-PGD-20, 7.00 pp on
  AutoAttack. This is the campaign's first reportable baseline number, and it
  sits close to the historical plain-SGD PGD-AT seed 0 reused in the "Historical
  reuse" section (best AA 0.4763) — but those are different arms (Nesterov on
  vs off) and different seeds, so that is a sanity check, not a comparison.

  **The `adr-s1` early-check AutoAttack number, previously unreadable, is now
  logged.** The 20:15Z entry above recorded that this session's sandbox could
  not open `early-check-eval-aa-best-model`; it can now.
  `eval-2e106980f84b326999a1` gives arm 3 (`adr`) seed 1, `best.pt`
  (`9181001647cb…`, the same checkpoint as the `early-check-eval-model` rows
  above), student weights, official test, count 10000, standard AutoAttack at
  the same pinned commit: **AutoAttack 0.4865**, clean 0.8282.
  Set beside arm 2 seed 1's `best.pt` AutoAttack 0.4741, ADR is **+1.24 pp**
  over its Nesterov-matched baseline. **This is not the preregistered
  comparison**, which is three seeds per arm, and it mixes a contract
  evaluation (arm 2) with a diagnostic early check (arm 3). It is directional
  for seed 1 only and licenses no ADR-vs-baseline claim; the ResNet-18 leg of
  the decision rule stays open until arms 2 and 3 each have three contract
  evaluations, and the primary MobileNetV2 leg has no data at all yet.

  **Aggregator defect — M3 blocker, found while checking this bundle against
  it.** `scripts/aggregate_adr_cifar10_replication.py:269` builds each run
  directory as `run_root / f"{arm_key}-s{seed}" / "train"`, with `arm_key` taken
  from the `ARMS` dict (`pgd_at`, `pgd_at_nesterov`, `adr`, `trades`,
  `trades_adr`, `trades_49k_validation`, `mobilenetv2_pgd_at`,
  `mobilenetv2_adr`). The campaign writes run dirs named after the config stem
  instead — `cifar10_r18_pgd_at_nesterov-s1`, `cifar10_r18_adr-s0`,
  `cifar10_mobilenetv2_adr-s1`, and so on — so **no arm resolves** and the
  aggregation will fail with "missing run bundle manifest" on the first arm.
  The fix is a directory field per arm (`cifar10_r18_` prefix for arms 1-6,
  `cifar10_` for the two MobileNetV2 arms, i.e. the config file stem), not a
  change to any check. Nothing else about the script is wrong on this bundle:
  its row validation (`count == 10000`, `split == "test"`,
  `runtime_method == "pgd_at"`, `weights == "model"`, AutoAttack block present,
  `attack_version == "standard"`, `expected_commit` equal to the pinned upstream,
  both `best` and `last` aliases present) passes on every field by inspection.
  Left unfixed here — this postrun imports nothing and touches no code; the fix
  belongs to M3 alongside a `--run-root` smoke test against the real layout.
  Cost: this evaluation ran 21:32:29Z → 00:09:35Z, 2 h 37 min wall clock for two
  checkpoints × (clean + CE-PGD-20 + AutoAttack) under GPU contention. The
  plan's budget line charges ≈1 GPU-h per AutoAttack checkpoint, ≈2 GPU-h per
  run; the measured figure is above that but includes the clean/PGD passes and
  contention, so the ≈40 GPU-h evaluation total stands as a lower bound rather
  than a refuted estimate. Watch the next uncontended evaluation before
  revising the line.
  Campaign state at the 00:10Z scan (`--include-hand-run`), 9 of 20 run dirs:
  training terminal and successful for `pgd_at-s1`, `-s2`,
  `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1`, `trades-s1`;
  `trades_adr-s0` running at epoch 184 and `mobilenetv2_adr-s1` at epoch 33.
  Contract evaluations: `pgd_at_nesterov-s1` terminal (this entry),
  `cifar10_r18_adr-s0/train/evaluation` and
  `cifar10_r18_pgd_at-s1/train/evaluation` running. `adr-s0` still needs its
  `evaluation-ema/` sibling (`has_ema`), which has not appeared. Arms 6 and 7
  (trades_49k_validation, mobilenetv2 pgd_at) and seed 2 of arms 2/3/4 have not
  started. M1b and M1c stay unticked.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_r18_trades_adr-s0`
  terminal event (`runs/adr-cifar10-campaign-v1/cifar10_r18_trades_adr-s0/
  train/`) — **the campaign's first arm-5 (ADR-TRADES) run**, seed 0, training
  only. **Nothing imported — no milestone closed.** The campaign is mid-flight
  (10 of 20 run dirs, 1 of 20 contract evaluations terminal), so there is no
  campaign-level result to aggregate; `docs/experiments/` still holds only
  `.gitkeep`, so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of the campaign root returns
  `terminal: true`, `success: true`, `failure_class: null` on the line whose
  `path` is that manifest.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200/200 recorded epochs,
  and `metrics.jsonl` really carries 200 rows ending at epoch 199 /
  `global_step` 70400. Both declared artifacts are present at their
  content-addressed paths under the manifest's own SHA-256
  (`epoch-metrics.parquet` `6a725b7835ea…`, `sample-stats-train.parquet`
  `ed200536ec2d…`); `config_hash` `7d9f2051b774…`. Source SHA
  `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`) from worktree
  `source-cd0b571e4685`, external lock `05cfce4cf8db…` with `.external/adr` at
  the pinned `515da0e0373f…`. Artifact hashes were not recomputed — this
  session's sandbox refuses to hash outside the repo root — so integrity again
  rests on the content-addressed paths matching the manifest; the aggregator
  owns the real check at M3.
  Checkpoint set on disk matches the contract exactly: `epoch-{049,099,149,
  199}.pt` (`checkpoint_epochs`), `best.pt`, `last.pt`, **and `best-ema.pt`** —
  the EMA checkpoint the aggregator's `has_ema: True` for `trades_adr` requires.
  Contract fields match arm 5 of the frozen table exactly: protocol
  `controlled_cifar10_r18_adr_v1` with `nesterov: true`, method `adr_trades`
  v1, `teacher: null` (ADR distils from an internal EMA of the student, not an
  external teacher), `trades_beta=6.0`, ADR block `ema_decay=0.995`,
  `temperature 2.5→2.0`, `lambda 0.7→0.95` — the values fixed in the plan and
  confirmed against the pinned official code. Training attack: KL loss,
  `kl_target=student_clean`, `epsilon=8/255 step=2/255 steps=10
  random_start=true` (the TRADES inner maximization); selection/evaluation
  attack CE-PGD-20 at the same epsilon/step. `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `validation_fraction=0.1` (45,000 train /
  5,000 validation, confirmed by `train_valid_examples: 45000`), `world_size=1`,
  effective global batch 128, identity normalization, `deterministic: true`.
  Training seeds all 0 except the fixed `split=20260722`. ADR is demonstrably
  live: every row carries `val_clean_accuracy_ema` / `val_pgd_accuracy_ema`
  alongside the student columns.

  **Validation-only diagnostics (5,000 held-out *training* images, CE-PGD-20 —
  not the official test set, not AutoAttack):**

  | checkpoint | epoch | clean | CE-PGD-20 |
  |---|---|---|---|
  | best (selection) | 150 | 0.8474 | 0.5442 |
  | last | 199 | 0.8488 | 0.5346 |

  Both re-read from `metrics.jsonl` and they match the manifest summary.
  Robust-overfit gap 0.0096; `val_pgd_slope_epoch_120_199` is +5.0e-05 per
  epoch, i.e. flat, where robust overfitting would show a clear negative slope.
  Set against arm 4 (`trades-s1`, logged at 23:11Z) — best clean 0.8322 / PGD
  0.5164, last 0.8338 / 0.4844, gap 0.0320 — this run is +2.78 pp higher at
  best and has a gap 2.2 pp smaller, exactly the direction ADR claims. **This
  licenses no claim.** It is validation, not the official test; it is a
  different seed (0 vs 1); and arms 4 and 5 differ in optimizer as well as
  method (plain SGD vs Nesterov), which is precisely the confound the plan
  already records as "no Nesterov-matched TRADES baseline exists yet, so this
  leg is directional only". The decisive ResNet-18 leg remains arm 3 vs arm 2.
  **The EMA branch is not redundant for this arm.** At the four epochs sampled
  around the second LR drop and the end, EMA validation was clean/PGD 0.8484 /
  **0.5486** (149), 0.8490 / 0.5448 (150), 0.8516 / 0.5368 (198), 0.8520 /
  0.5372 (199). The EMA weights were therefore above the student's selected
  best (0.5442) at least once, so the `evaluation-ema/` pass this arm still
  owes is a real measurement and not a formality. (Only four epochs were read;
  no claim is made about where the EMA branch actually peaked.)
  **Cost — arm 5 is the most expensive arm measured so far, and the ADR budget
  line is tight for it.** Uncontended (epochs 0-1) this run did **974-981 img/s,
  45.9-46.2 s/epoch training loop**. Ranked against the arms already measured:
  `pgd_at_nesterov` 1167 img/s / 38.6 s, `adr` 1106-1141 / 39.4-40.7, `trades`
  998 / 45.1, `adr_trades` 974 / 46.2. Two readings fall out of that: ADR costs
  a consistent ≈2% of throughput on both objectives (1167→1141 on PGD-AT,
  998→974 on TRADES), and the TRADES objective's extra clean forward pass costs
  ≈15%, which dominates. The budget table charges the ADR family ≈3.0 GPU-h per
  run = 54 s per *full* epoch, leaving ≈8 s for validation — but this arm runs
  **two** validation passes per epoch (student and EMA), so that allowance is
  very likely too small, in the same way the 23:11Z entry found ≈2.6 GPU-h too
  small for TRADES. It cannot be quantified from this run: the per-epoch rows
  still carry no timestamps or validation-pass timing, so the uncontended full-
  epoch cost cannot be separated from the training loop. This run's own wall
  clock was **4 h 16 min 50 s** (20:13:15Z → 00:30:05Z, 77.1 s/epoch averaged),
  but that is heavily inflated by contention — throughput fell from ≈980 img/s
  to ≈724-728 by epoch 149 and stayed there — and is not the uncontended cost.
  Contention changes only wall clock, not results: `deterministic: true`, fixed
  seeds and fixed batch size make the numbers independent of throughput.
  **The M3 aggregator defect logged at 00:10Z has a second instance here, and
  it confirms the fix shape.** Arm 5's run dir is `cifar10_r18_trades_adr-s0`,
  not the `trades_adr-s0` that `aggregate_adr_cifar10_replication.py:269`
  builds from the `ARMS` key. Every run dir seen so far is `<config stem>-s<seed>`
  (`cifar10_r18_` for arms 1-6, `cifar10_` for the two MobileNetV2 arms), so a
  per-arm directory field keyed off the config stem is the right fix, and no
  check in the script needs to change. Still left unfixed — this postrun imports
  nothing and touches no code.
  Campaign state at the 00:30Z scan (`--include-hand-run`), **10 of 20 run
  dirs**: training terminal and successful for `pgd_at-s1`, `-s2`,
  `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1`, `trades-s1` and now
  `trades_adr-s0` (8 of 20); `mobilenetv2_adr-s1` running at epoch 44 and
  `cifar10_mobilenetv2_pgd_at-s0` — **the campaign's first arm-7 job** — newly
  created with no progress row yet. Contract evaluations: `pgd_at_nesterov-s1`
  terminal, `pgd_at-s1` started with no progress yet, and `adr-s0` flagged
  `stale` at 66 min. The stale flag is **not** a failure: that job has already
  written `panel-{best,last}.jsonl`, `sample-stats-{best,last}.parquet` and
  `autoattack-best.json` but not `autoattack-last.json` or
  `evaluation-results.json`, i.e. it is inside the `last.pt` AutoAttack pass,
  which emits no progress rows. Watch it, do not retry it. `adr-s0` also still
  owes its `evaluation-ema/` sibling, and `trades_adr-s0` now owes both
  `evaluation/` and `evaluation-ema/`. Arm 6 (trades_49k_validation) and seed 2
  of arms 2/3/4 have not started. M1b and M1c stay unticked.
- 2026-09-10: postrun of the `eval-a8f0e20f78a4f27ca07d` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_adr-s0/train/evaluation/`) —
  **the campaign's first contract evaluation of an ADR arm**: arm 3 (`adr`),
  seed 0, student weights, both checkpoints, clean + CE-PGD-20 + AutoAttack on
  the official test set. **Nothing imported — no milestone closed.** The
  campaign is mid-flight (10 of 20 run dirs, 2 of 20 contract evaluations
  terminal), so there is no campaign-level result to aggregate;
  `docs/experiments/` still holds only `.gitkeep`, so the idempotency check
  had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `failure_class: null` on the line whose `path` is that
  manifest.
  Verified for this bundle: `completion.json` `{"status": "completed",
  "results": 2}`, `manifest.status=sync_pending`, `error-marker.txt` reads "no
  application error recorded", all seven declared artifacts present on disk
  with their content-addressed copies under the manifest's own SHA-256 paths
  (`resolved_evaluation_config.yaml` `438edcb841b6…`,
  `evaluation-lineage.json` `2c9e66fca042…`, `evaluation-results.json`
  `02c162b2b793…`, `panel-{best,last}.jsonl`,
  `sample-stats-{best,last}.parquet`), plus `autoattack-{best,last}.json`.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`)
  from worktree `source-cd0b571e4685`, external lock `05cfce4cf8db…` with
  `.external/adr` at the pinned `515da0e0373f…`. Artifact hashes were again
  not recomputed — this session's sandbox refuses to hash outside the repo
  root (`sha256sum` on the bundle is blocked) — so integrity rests on the
  content-addressed paths matching the manifest; the aggregator owns the real
  check at M3.
  Lineage ties to the training run: `evaluation-lineage.json` records
  `training_config_hash == training_runtime_config_hash ==
  raw_mapping_hash == 3121716496193f…` with `applied: []` (no config
  migration), and `train_run_id` on both result rows is
  `adr-campaign-v1-cifar10_r18_adr-s0`, the training run whose own terminal
  event was logged above.
  Contract fields match arm 3 of the frozen plan exactly: protocol
  `controlled_cifar10_r18_adr_v1` with **`nesterov: true`**, method `adr` v1,
  `teacher: null`, ADR block `ema_decay=0.995`, `temperature 2.5→2.0`,
  `lambda 0.7→0.95`, `epochs=200`, `milestones=[100,150] gamma=0.1`,
  `validation_fraction=0.1`, `world_size=1`, effective global batch 128,
  identity normalization, `deterministic: true`. Evaluation identity:
  `evaluation.autoattack: true`, `checkpoints: both`, dataset `split: test`,
  `count: 10000` on both rows, CE-PGD-20 at `epsilon=8/255 step=2/255
  steps=20 random_start=true`, `evaluation_seed=0`, `weights: model`, threat
  hash `7081101693340e70…` identical on both rows. AutoAttack is the
  **standard** version with real (non-injected) provenance —
  `expected_commit == vcs_commit == a39220048b3c9f2cca9a4d3a54604793c68eca7e`,
  the pinned upstream the aggregator requires — run in its own process at
  `epsilon=8/255`, `Linf`, batch 128, seed 0. Training seeds all 0 except the
  fixed `split=20260722`.
  Benign field worth naming so it is not misread later: the resolved
  evaluation config carries `evaluation.dataset.num_samples: 16`. That field
  only sizes the `synthetic_cifar` generator (`src/ard/data/datasets.py:471`);
  for `cifar10` the full torchvision split is built, and `count: 10000` on
  both result rows is the proof that the whole official test set was scored.

  **Official CIFAR-10 test set (10,000 examples), student weights, arm 3
  (`adr`) seed 0 — clean, CE-PGD-20 and AutoAttack reported separately, best
  and last kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best.pt` (val epoch 103) | `b5444a7fb8b8…` | 0.8258 | 0.5334 | 0.4875 |
  | `last.pt` (epoch 199) | `6ac8293418d2…` | 0.8489 | 0.4842 | 0.4481 |

  **Robust overfitting is roughly halved, and that is the clearest signal in
  the campaign so far.** Best-minus-last gap for this ADR run: 4.92 pp on
  CE-PGD-20, 3.94 pp on AutoAttack. The two adversarially-trained baselines
  with official numbers both sit near double that — arm 2 (`pgd_at_nesterov`)
  seed 1: 9.01 / 7.00 pp; the reused historical plain-SGD PGD-AT seed 0:
  9.23 / 7.27 pp. Suppressing the late-training robustness decay is exactly
  what ADR claims to do, and this is the first time this campaign can read it
  on the official test set rather than on validation.
  **Accuracy comparisons, in order of how much they license.**
  (a) *Same seed, unmatched optimizer.* Against the reused historical PGD-AT
  seed 0 (best 0.8201 / 0.5112 / 0.4763, last 0.8446 / 0.4189 / 0.4036), this
  run is **+1.12 pp AutoAttack and +2.22 pp CE-PGD-20 at best**, +0.57 pp
  clean, and **+4.45 pp AutoAttack at last**. Seed matches, but arm 1 is plain
  SGD and arm 3 is Nesterov, so the optimizer is confounded — this is the leg
  the plan already calls directional only.
  (b) *Matched optimizer, mismatched seed.* Against arm 2 seed 1's `best.pt`
  AutoAttack 0.4741, this run is **+1.34 pp**. Same protocol
  (`controlled_cifar10_r18_adr_v1`, Nesterov on), different seed.
  (c) *Within-arm spread, first read on AutoAttack.* Arm 3's `best.pt`
  AutoAttack is now 0.4875 (seed 0, this contract evaluation) and 0.4865
  (seed 1, the `early-check-eval-aa-best-model` number logged at 00:10Z) — a
  **0.10 pp** spread on two seeds, with clean 0.8258 vs 0.8282. Both were
  scored on the official test set, `count: 10000`, standard AutoAttack at the
  same pinned commit, student weights, `best.pt`, so they are directly
  comparable even though one came from a diagnostic early check.
  **None of this is the preregistered result.** The decision rule needs three
  contract evaluations per arm on arms 2 and 3, and its primary leg is
  MobileNetV2 (arms 8 vs 7), which has no evaluation data at all — arm 7's
  first training run started today and is at epoch 13. The ADR-side spread in
  (c) being an order of magnitude below the gap in (b) is encouraging, but
  arm 2's own seed spread is still unmeasured (one evaluated seed), so no
  noise floor exists for the difference yet. Report as directional, claim
  nothing.
  **The `evaluation-ema/` sibling this arm owed has now appeared and is
  running** (`eval-897947924c804ba35ea7`, no progress row yet). That is the
  "ADR + WA" half the aggregator requires for every `has_ema` arm, and its
  absence was flagged in the two entries above; the gap is closing on its own,
  no action needed.
  **The M3 aggregator directory-naming defect is unchanged and still the only
  known blocker.** `scripts/aggregate_adr_cifar10_replication.py:269` builds
  `run_root / f"{arm_key}-s{seed}" / "train"`, which for arm 3 gives
  `adr-s0`, not the real `cifar10_r18_adr-s0`. Everything else in the script
  passes on this bundle by inspection: `_load_rows` checks
  `count == 10000`, `split == "test"`, `runtime_method == "adr"`,
  `weights == "model"`, an AutoAttack block on every row,
  `attack_version == "standard"`, `expected_commit` equal to the pinned
  upstream, and both `best` and `last` aliases present — all satisfied here.
  Left unfixed: this postrun imports nothing and touches no code.
  Cost: this evaluation ran 21:32:28Z → 00:47:10Z, **3 h 14 min 43 s** wall
  clock for two checkpoints × (clean + CE-PGD-20 + AutoAttack). It overlapped
  almost exactly with arm 2 seed 1's evaluation (21:32:29Z → 00:09:35Z, 2 h
  37 min) plus running training jobs, so both figures are contention-inflated
  and neither refutes the plan's ≈1 GPU-h per AutoAttack checkpoint line. The
  ≈40 GPU-h evaluation total still stands as a lower bound; an uncontended
  evaluation is still needed before revising it.
  Campaign state at the 00:49Z scan (`--include-hand-run`), **10 of 20 run
  dirs**: training terminal and successful for `pgd_at-s1`, `-s2`,
  `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1`, `trades-s1`,
  `trades_adr-s0` (8 of 20); `mobilenetv2_adr-s1` running at epoch 55 and
  `mobilenetv2_pgd_at-s0` at epoch 13. Contract evaluations:
  `pgd_at_nesterov-s1` and `adr-s0` terminal (2 of 20), `pgd_at-s1` and
  `adr-s0/evaluation-ema` running with no progress row yet. Arm 6
  (trades_49k_validation) and seed 2 of arms 2/3/4 have not started. M1b and
  M1c stay unticked.
- 2026-09-10: postrun of the `eval-0b0343aa17a7a4dbcf78` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_pgd_at-s1/train/evaluation/`) —
  **arm 1 (`pgd_at`, the plain-SGD canonical baseline) seed 1**, student
  weights, both checkpoints, clean + CE-PGD-20 + AutoAttack on the official
  test set. **Nothing imported — no milestone closed.** The campaign is
  mid-flight (10 of 20 run dirs, 3 of 20 contract evaluations terminal), so
  there is no campaign-level result to aggregate; `docs/experiments/` still
  holds only `.gitkeep`, so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `failure_class: null` on the line whose `path` is that
  manifest.
  Verified for this bundle: `completion.json` `{"status": "completed",
  "results": 2}`, `manifest.status=sync_pending`, `error-marker.txt` reads "no
  application error recorded", all seven declared artifacts present on disk
  with their content-addressed copies under the manifest's own SHA-256 paths
  (`resolved_evaluation_config.yaml` `583072673078…`,
  `evaluation-lineage.json` `c8af978e8974…`, `evaluation-results.json`
  `15121fa9be65…`, `panel-{best,last}.jsonl`,
  `sample-stats-{best,last}.parquet`), plus `autoattack-{best,last}.json`.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`)
  from worktree `source-cd0b571e4685`, external lock `05cfce4cf8db…`. Artifact
  hashes were again not recomputed — this session's sandbox refuses to hash
  outside the repo root — so integrity rests on the content-addressed paths
  matching the manifest; the aggregator owns the real check at M3.
  Lineage ties to the training run: `evaluation-lineage.json` records
  `training_config_hash == training_runtime_config_hash == raw_mapping_hash ==
  4da235e699cf…` with `applied: []` (no config migration), and `train_run_id`
  on both rows is `adr-campaign-v1-cifar10_r18_pgd_at-s1`, whose own training
  bundle is terminal with `epoch_metrics_complete: true`, 200/200 epochs, and
  `epoch-{049,099,149,199}.pt` + `best.pt` + `last.pt` on disk (no
  `best-ema.pt`, correct for a non-ADR arm).
  Contract fields match arm 1 of the frozen table exactly: protocol
  `controlled_cifar10_r18_v1` with **`nesterov: false`** (plain SGD, lr 0.1,
  momentum 0.9, weight decay 5e-4), method `pgd_at` v1 (`adr: null`),
  `teacher: null`, `epochs=200`, `milestones=[100,150] gamma=0.1`,
  `validation_fraction=0.1`, `world_size=1`, effective global batch 128,
  identity normalization, `deterministic: true`. Evaluation identity:
  `evaluation.autoattack: true`, `checkpoints: both`, dataset `split: test`,
  `count: 10000` on both rows, CE-PGD-20 at `epsilon=8/255 step=2/255 steps=20
  random_start=true`, `evaluation_seed=0`, `weights: model`, threat hash
  `7081101693340e70…` — the same threat hash as arms 2 and 3, i.e. all three
  evaluated arms were scored under a byte-identical threat model. AutoAttack is
  the **standard** version with real (non-injected) provenance
  (`expected_commit == vcs_commit == a39220048b3c9f2cca9a4d3a54604793c68eca7e`),
  run in its own process at `epsilon=8/255`, `Linf`, batch 128, seed 0.
  Training seeds all 1 except the fixed `split=20260722` and
  `evaluation_attack=0`.

  **Official CIFAR-10 test set (10,000 examples), student weights, arm 1
  (`pgd_at`, plain SGD) seed 1 — clean, CE-PGD-20 and AutoAttack reported
  separately, best and last kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best.pt` (val epoch 104) | `40893eacef81…` | 0.8277 | 0.5045 | 0.4678 |
  | `last.pt` (epoch 199) | `b55709944b11…` | 0.8433 | 0.4247 | 0.4069 |

  **This is the campaign's first noise floor, and it is the reason this entry
  matters.** Arm 1 seed 0 is the reused historical run (best 0.8201 / 0.5112 /
  0.4763; last 0.8446 / 0.4189 / 0.4036), so arm 1 now has **two seeds on the
  official test set under one contract** — the first within-arm repeat this
  campaign can read. Seed-to-seed **range** (two runs give a range, not a
  standard deviation):

  | quantity | seed 0 | seed 1 | range |
  |---|---|---|---|
  | best clean | 0.8201 | 0.8277 | 0.76 pp |
  | best CE-PGD-20 | 0.5112 | 0.5045 | 0.67 pp |
  | **best AutoAttack** | 0.4763 | 0.4678 | **0.85 pp** |
  | last clean | 0.8446 | 0.8433 | 0.13 pp |
  | last CE-PGD-20 | 0.4189 | 0.4247 | 0.58 pp |
  | last AutoAttack | 0.4036 | 0.4069 | 0.33 pp |

  Best-checkpoint quantities move two to three times more than last-checkpoint
  ones, which is what you would expect when `best` is chosen by a 5,000-image
  validation split and `last` is fixed at epoch 199.
  **Read against this floor, the campaign's two directional signals separate
  cleanly.**
  (a) *The accuracy signal is now marginal, not comfortable.* Arm 3 (`adr`)
  seed 0's `best.pt` AutoAttack is 0.4875. Against the three baseline readings
  that exist it is +1.12 pp (arm 1 seed 0), +1.97 pp (arm 1 seed 1) and
  +1.34 pp (arm 2 seed 1, the Nesterov-matched pair that the decision rule
  actually uses). Every one of those exceeds arm 1's own 0.85 pp two-seed
  range, but not by much — and 0.85 pp from two runs is a range, not a σ, so a
  third seed can only widen it. The plan's "mixed / inconclusive if within-arm
  seed spread exceeds the measured gap" clause is live, not theoretical.
  (b) *The robust-overfitting signal survives the floor.* Best-minus-last gap
  for this run: 7.98 pp on CE-PGD-20, **6.09 pp on AutoAttack**. All four
  baseline readings now cluster there — arm 1 seed 0: 9.23 / 7.27 pp; arm 1
  seed 1: 7.98 / 6.09 pp; arm 2 seed 1: 9.01 / 7.00 pp — i.e. a baseline
  AutoAttack-gap range of 6.09–7.27 pp, spanning 1.18 pp within arm 1 alone.
  Arm 3 seed 0's 3.94 pp sits **2.15 pp below the nearest baseline reading**,
  which is larger than the baseline gap's own spread. Suppressed late-training
  robustness decay is therefore the sturdier of the two claims so far.
  (c) *Nesterov versus plain SGD, at matched seed 1, is unresolvable.* Arm 2
  seed 1's `best.pt` AutoAttack 0.4741 against this run's 0.4678 is +0.63 pp
  for Nesterov — the only same-seed, same-method, optimizer-only comparison in
  the campaign, and it is **below** arm 1's 0.85 pp seed range. So the
  optimizer confound the plan added arm 2 to remove is probably small, but this
  cannot show it is nonzero. Adding arm 2 rather than comparing ADR against
  arm 1 remains the right call; it just is not rescuing a large bias.
  **None of (a)-(c) is the preregistered result.** The decision rule needs
  three contract evaluations per arm on arms 2 and 3, and its primary leg is
  MobileNetV2 (arms 8 vs 7), which still has no evaluation data — arm 8 is at
  epoch 99 and arm 7 at epoch 66 of training. Report as directional, claim
  nothing.
  **M1b is now load-bearing, not housekeeping.** The 0.85 pp floor above rests
  on arm 1 seed 0, which exists only as an uncommitted local bundle
  (`outputs/scientific/pgd-at-controlled-s0-c2220f1/`) carried in the
  aggregator as recorded constants
  (`scripts/aggregate_adr_cifar10_replication.py:120-128`, values verified
  here to match the plan's M1a numbers exactly). Two consequences: the seed
  range mixes source SHAs (`c2220f1` vs `cd0b571e4685`), contract-equal by the
  M1a field-by-field audit but not hash-equal, so treat it as an estimate of
  the floor rather than a measurement of it; and if that directory is pruned
  before M1b archives it, the floor disappears with it.
  **The M3 aggregator directory-naming defect has a third instance and needs no
  further evidence.** Arm 1's run dir is `cifar10_r18_pgd_at-s1`, not the
  `pgd_at-s1` that `aggregate_adr_cifar10_replication.py:269` builds. Note that
  arm 1 **seed 0** is unaffected: `_load_arm_seed` returns the reused constants
  before reaching that line, so the defect hits fresh runs only. Everything
  else in the script passes on this bundle by inspection (`count == 10000`,
  `split == "test"`, `runtime_method == "pgd_at"`, `weights == "model"`,
  AutoAttack block on every row, `attack_version == "standard"`,
  `expected_commit` equal to the pinned upstream, both aliases present). Left
  unfixed: this postrun imports nothing and touches no code.
  **Cost — the first evaluation measurement that does not exceed the budget
  line.** This run took 00:09:38.7Z → 01:59:30.9Z, **1 h 49 min 52 s** for two
  checkpoints × (clean + CE-PGD-20 + AutoAttack). The two earlier evaluations
  ran concurrently with each other and came in at 2 h 37 min and 3 h 14 min;
  this one started as the first of them finished, so it was the least contended
  of the three. At under 2 h for two AutoAttack passes plus four cheap passes,
  it is consistent with the plan's ≈1 GPU-h per AutoAttack checkpoint line
  rather than above it. The ≈40 GPU-h evaluation total stands; no revision.
  Campaign state at the 02:05Z scan (`--include-hand-run`, fresh cursor —
  see the operational note below), **10 of 20 run dirs**: training terminal and
  successful for `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `adr-s0`,
  `adr-s1`, `trades-s1`, `trades_adr-s0` (8 of 20); `mobilenetv2_adr-s1`
  running at epoch 99 and `mobilenetv2_pgd_at-s0` at epoch 66. Contract
  evaluations: `pgd_at_nesterov-s1`, `adr-s0` and `pgd_at-s1` terminal (3 of
  20); `pgd_at-s2/evaluation` newly started with no progress row, and
  `adr-s0/evaluation-ema` running. Arm 6 (trades_49k_validation) and seed 2 of
  arms 2/3/4 have not started. M1b and M1c stay unticked.
  **Operational note for future postruns — the watcher cursor is not a scratch
  file.** The first campaign-wide scan in this session used
  `--state /tmp/ardx-postrun-campaign-scan.json` and silently reported only 9
  of 18 bundles, omitting six terminal training runs. That path had already
  been written by a headless postrun, and `scan_run` emits nothing for a bundle
  whose `(status, terminal, success)` triple is unchanged since the cursor was
  last saved (`scripts/ardx/campaign_watch.py:220-222`) — `--emit-existing`
  only covers first sight. Nothing is wrong with the watcher; a reused
  `/tmp/ardx-postrun-*.json` slug just makes a scan look like a partial
  campaign. Use a genuinely unused `--state` path when re-deriving whole-
  campaign state, and treat a bundle count below the number of `run-bundle/
  manifest.json` files on disk as a cursor artefact, not as missing runs.
- 2026-09-10: postrun of the `eval-897947924c804ba35ea7` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_adr-s0/train/evaluation-ema/`) —
  **the campaign's first EMA-weights ("ADR + WA") evaluation**: arm 3 (`adr`),
  seed 0, `--weights=ema` against `best-ema.pt` and `last.pt`, clean +
  CE-PGD-20 + AutoAttack on the official test set. **Nothing imported — no
  milestone closed.** The campaign is mid-flight, `docs/experiments/` still
  holds only `.gitkeep`, so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan over the run's parent
  returns `terminal: true`, `success: true`, `status: completed`,
  `failure_class: null` for exactly the `--state-path` the watcher passed;
  `run-bundle/completion.json` is `{"status": "completed", "results": 2}` and
  `error-marker.txt` reads "no application error recorded".
  Bundle verified against the frozen contract. All seven declared artifacts
  exist at their content-addressed `run-bundle/artifacts/…/<sha256>/` paths
  (`resolved_evaluation_config.yaml` `5e668d12e2e0…`,
  `evaluation-lineage.json` `80cb49d1432e…`, `evaluation-results.json`
  `228ec0ee43b8…`, `panel-best-ema-ema.jsonl`, `panel-last-ema.jsonl`,
  `sample-stats-best-ema-ema.parquet`, `sample-stats-last-ema.parquet`), plus
  `autoattack-best-ema.json` and `autoattack-last.json` beside them. Artifact
  hashes were again not recomputed — this session's sandbox refuses to hash
  outside the repo root — so integrity rests on the content-addressed paths
  matching the manifest; the aggregator owns the real check at M3.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`)
  from worktree `source-cd0b571e4685`, external lock `05cfce4cf8db…`.
  Lineage ties to the training run: `evaluation-lineage.json` records
  `training_config_hash == training_runtime_config_hash == raw_mapping_hash ==
  3121716496193f48…` with `applied: []` (no config migration), which is the
  training bundle's own `config_hash`; `train_run_id` on both rows is
  `adr-campaign-v1-cifar10_r18_adr-s0`, whose training bundle is terminal at
  200/200 epochs with `best.pt`, `best-ema.pt`, `last.pt` and
  `epoch-{049,099,149,199}.pt` all on disk — `best-ema.pt` present as an ADR
  arm requires.
  Contract fields match arm 3 exactly: protocol
  `controlled_cifar10_r18_adr_v1` with `nesterov: true`, method `adr` v1,
  `teacher: null`, ADR `ema_decay 0.995` / `temperature 2.5→2.0` /
  `lambda 0.7→0.95`, `epochs=200`, `milestones=[100,150] gamma=0.1`,
  `validation_fraction=0.1`, `world_size=1`, effective global batch 128,
  identity normalization, `deterministic: true`. Evaluation identity:
  `checkpoints: both`, `split: test`, `count: 10000` on both rows, CE-PGD-20 at
  `epsilon=8/255 step=2/255 steps=20 random_start=true`, `evaluation_seed=0`,
  threat hash `7081101693340e70…` — the **same** threat hash as every other
  evaluated arm in this campaign, so the EMA rows are scored under a
  byte-identical threat model to the student rows. AutoAttack is the
  **standard** version with real provenance (`expected_commit == vcs_commit ==
  a39220048b3c9f2cca9a4d3a54604793c68eca7e`), run in its own process at
  `epsilon=8/255`, `Linf`, batch 128, seed 0. `weights: ema` on both rows;
  `selection_weights` is `"ema"` for `best-ema.pt` and `"model"` for `last.pt`,
  which is correct — `last.pt` is epoch-defined and selection-independent, so
  its selection story is the student's while its read weights are the EMA's.

  **Official CIFAR-10 test set (10,000 examples), EMA weights, arm 3 (`adr`)
  seed 0 — clean, CE-PGD-20 and AutoAttack reported separately, best and last
  kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best-ema.pt` (EMA-selected epoch, not read — see below) | `64854fec10e0…` | 0.8338 | 0.5426 | 0.4961 |
  | `last.pt` (epoch 199) | `6ac8293418d2…` | 0.8514 | 0.4858 | 0.4492 |

  **The one clean model-vs-EMA measurement this campaign produces is nearly
  null, and it is the reason this entry matters.** `last.pt`'s
  `checkpoint_sha256` is `6ac8293418d2…` in **both** this bundle and the
  student-weights `evaluation/` bundle — the same file, the same epoch 199,
  with only the read weights switched. That is the only same-file
  weights-only contrast in the campaign:

  | epoch 199, `last.pt` | `--weights=model` | `--weights=ema` | delta |
  |---|---|---|---|
  | clean | 0.8489 | 0.8514 | +0.25 pp |
  | CE-PGD-20 | 0.4842 | 0.4858 | +0.16 pp |
  | **AutoAttack** | 0.4481 | 0.4492 | **+0.11 pp** |

  +0.11 pp AutoAttack is roughly an eighth of arm 1's 0.85 pp two-seed range.
  At epoch 199, reading the EMA shadow instead of the student buys essentially
  nothing measurable for this seed.
  **The best-checkpoint EMA gain looks eight times larger, and it is
  confounded — the frozen contract cannot separate the two causes.**
  `best-ema.pt` at `--weights=ema` (0.4961 AA) against `best.pt` at
  `--weights=model` (0.4875 AA) is +0.86 pp, with +0.80 pp clean and +0.92 pp
  CE-PGD-20. But per `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section these are
  two **independent selections** — the student's validation PGD accuracy picks
  `best.pt` (val epoch 103) and the EMA shadow's own validation PGD accuracy
  picks `best-ema.pt` — so in general they are different epochs, and the
  +0.86 pp mixes "EMA weights are better" with "the EMA's validation picked a
  different epoch". The invariants doc names the fix: evaluate `best-ema.pt`
  at **both** `--weights=model` and `--weights=ema` (same file, same epoch,
  weights only). **The frozen contract does not schedule that run**, so it does
  not exist for this arm and will not exist for arms 5 or 8 either. Do not fix
  this mid-campaign — the contract is frozen and the numbers above are valid
  for what they measure — but the "ADR + WA" line in the eventual report must
  carry this caveat, and the `last.pt` row is the honest weights-only number
  to quote beside it.
  **Robust-overfitting suppression survives the switch to EMA weights.**
  Best-minus-last gap on this bundle: 5.68 pp CE-PGD-20, **4.69 pp
  AutoAttack**. The baseline AutoAttack-gap band from the four student-weights
  baseline readings is 6.09–7.27 pp, so 4.69 pp still sits below all of it, as
  the student-weights ADR reading (3.94 pp) did. Note the EMA gap is 0.75 pp
  **larger** than the student gap for the same run — expected, not
  contradictory: `best-ema.pt` is the more optimistically selected end of its
  own branch, so its best-minus-last spread is naturally wider.
  **What this does *not* license.** Arm 3 seed 0's headline AutoAttack number
  is now 0.4961 (`best-ema.pt` @ ema). Against arm 2 (`pgd_at_nesterov`) seed 1
  `best.pt` @ model (0.4741) that is +2.20 pp — but that comparison is
  seed-mismatched *and* weights-mismatched, making it the loosest in the
  campaign, not the tightest. The preregistered rule's primary quantity reads
  `best_checkpoint_model_weights`
  (`scripts/aggregate_adr_cifar10_replication.py:353-357`); the EMA delta is a
  named secondary
  (`…:363-365`, `best_checkpoint_ema_weights`, ADR + WA pair only). Neither leg
  has three seeds, and the primary leg is MobileNetV2 (arms 8 vs 7), which
  still has no evaluation data at all. Report as directional; claim nothing.
  **Not read: `best-ema.pt`'s selected epoch.** Recovering it needs
  `val_pgd_accuracy_ema` from the training bundle's `metrics.jsonl`, and both
  attempts to read that file were refused by this session's sandbox. The table
  above therefore names the epoch for `last.pt` only. Nothing in this entry
  depends on the missing number — the load-bearing comparison is the
  same-file `last.pt` row — but a future postrun or the M3 aggregator should
  fill it in, since it is exactly the quantity that would size the selection
  confound described above.
  **The aggregator's EMA leg is confirmed shape-correct against real data for
  the first time.** `_load_arm_seed` reads
  `run_dir / "evaluation-ema" / "evaluation-results.json"` with
  `expect_weights="ema"` and normalizes the `best-ema` alias to `best`
  (`scripts/aggregate_adr_cifar10_replication.py:225-237, 273-278`). Every one
  of those checks passes on this bundle by inspection: `weights == "ema"` on
  both rows, `count == 10000`, `split == "test"`, `runtime_method == "adr"`,
  an AutoAttack block on each row, `attack_version == "standard"`,
  `expected_commit` equal to the pinned upstream, and both aliases present
  after normalization. **The directory-naming defect has a fourth instance and
  is unchanged**: line 269 builds `adr-s0`, the real dir is
  `cifar10_r18_adr-s0`. Left unfixed — this postrun imports nothing and
  touches no code.
  **Budget correction: the evaluation line omits the EMA evaluations
  entirely.** This run took 00:47:14.3Z → 02:44:36.8Z, **1 h 57 min 22 s** for
  two checkpoints × (clean + CE-PGD-20 + AutoAttack), contended with two
  MobileNetV2 training jobs — consistent with the ≈1 GPU-h per AutoAttack
  checkpoint line. But the GPU-hour table above charges arms 3/5/8 only
  `18` evaluation GPU-h, i.e. 9 runs × 2 AutoAttack passes — one evaluation per
  run. The frozen contract requires a **second**, EMA-weights evaluation for
  every ADR-family run, which is 9 more jobs × 2 more AutoAttack passes ≈
  **18 unbudgeted GPU-h**. The campaign's evaluation total is therefore ≈58
  GPU-h, not 40, and the new-work total ≈114 GPU-h, not 96. This is an
  arithmetic omission in the budget, not a contract change; the budget table is
  left as written (it is the record of what was projected) and this line is the
  correction.
  Campaign state at the 02:47Z scan (`--include-hand-run`, fresh unused
  cursor), **10 of 20 run dirs**: training terminal and successful for
  `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `adr-s0`, `adr-s1`,
  `trades-s1`, `trades_adr-s0` (8 of 20); `mobilenetv2_adr-s1` running at
  epoch 124 and `mobilenetv2_pgd_at-s0` at epoch 97. Contract (model-weights)
  evaluations still **3 of 20** terminal — `pgd_at_nesterov-s1`, `adr-s0`,
  `pgd_at-s1` — with `pgd_at-s2/evaluation` and
  `pgd_at_nesterov-s0/evaluation` started and showing no progress row yet.
  EMA-weights evaluations: **1 of 9** terminal (this one). Arm 6
  (`trades_49k_validation`) and seed 2 of arms 2/3/4 have not started. M1b and
  M1c stay unticked.
- 2026-09-10: postrun of the `eval-85c8061a4ec77c7145e5` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_pgd_at-s2/train/evaluation/`) —
  arm 1 (`pgd_at`, plain-SGD PGD-AT baseline) seed 2, `--weights=model`
  against `best.pt` and `last.pt`, clean + CE-PGD-20 + AutoAttack on the
  official CIFAR-10 test set. **Nothing imported — no milestone closed.** The
  campaign is mid-flight and `docs/experiments/` still holds only `.gitkeep`,
  so the idempotency check had nothing to collide with. **This evaluation
  completes arm 1's three-seed set — the first arm in the campaign whose full
  seed set (`ARMS["pgd_at"]["seeds"] == (0, 1, 2)`,
  `scripts/aggregate_adr_cifar10_replication.py:44-51`) has terminal contract
  evaluations.** That is why this entry matters, and also why it is worth
  being precise about what it does *not* do — see (c) below.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan over the run's parent
  returns `terminal: true`, `success: true`, `status: completed`,
  `failure_class: null` for exactly the `--state-path` the watcher passed;
  `run-bundle/completion.json` is `{"status": "completed", "results": 2}` and
  `error-marker.txt` reads "no application error recorded".
  Bundle verified against the frozen contract. All seven declared artifacts
  exist both at their source paths and at their content-addressed
  `run-bundle/artifacts/<name>/<sha256>/` paths, and every one of those
  directory names equals the manifest's own `sha256` field for that artifact
  (`resolved_evaluation_config.yaml` `4a427c3f9df4…`,
  `evaluation-lineage.json` `4cf6c3acfd1b…`, `evaluation-results.json`
  `4d994b20aaf7…`, `panel-best.jsonl` `df151ca3ea7a…`,
  `sample-stats-best.parquet` `00a03038df80…`, `panel-last.jsonl`
  `7fe4b9cd32c8…`, `sample-stats-last.parquet` `7e1a57b37c3b…`), plus
  `autoattack-best.json` and `autoattack-last.json` beside them. Artifact
  bytes were again not re-hashed — this session's sandbox refuses to run a
  hash over paths outside the repo root — so integrity rests on that
  path-equals-hash correspondence; the aggregator owns the real check at M3.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`)
  from worktree `source-cd0b571e4685`, external lock `05cfce4cf8db…`.
  Lineage ties to the training run: `evaluation-lineage.json` records
  `training_config_hash == training_runtime_config_hash == raw_mapping_hash ==
  b5930a4f12f6…` with `applied: []` (no config migration), which is the
  training bundle's own `config_hash`; `train_run_id` on both rows is
  `adr-campaign-v1-cifar10_r18_pgd_at-s2`, whose training bundle is terminal
  at 200/200 epochs (`global_step` 70400) with `best.pt`, `last.pt` and
  `epoch-{049,099,149,199}.pt` on disk and **no `best-ema.pt`**, which is
  correct — arm 1 has no EMA leg (`has_ema: False`).
  Contract fields match arm 1 exactly: protocol `controlled_cifar10_r18_v1`
  with `nesterov: false` (the plain-SGD baseline, not arm 2's Nesterov-matched
  one), method `pgd_at` v1, `teacher: null`, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`,
  `validation_fraction=0.1`, `world_size=1`, effective global batch 128,
  identity normalization, `deterministic: true`. Evaluation identity:
  `split: test`, `count: 10000` on both rows, CE-PGD-20 at `epsilon=8/255
  step=2/255 steps=20 random_start=true`, `evaluation_seed=0`, threat hash
  `7081101693340e70…` — the **same** threat hash as every other evaluated arm
  in this campaign. AutoAttack is the **standard** version with real
  provenance (`expected_commit == vcs_commit ==
  a39220048b3c9f2cca9a4d3a54604793c68eca7e`), run in its own process at
  `epsilon=8/255`, `Linf`, batch 128, seed 0. `weights: model` and
  `selection_weights: model` on both rows.

  **Official CIFAR-10 test set (10,000 examples), student weights, arm 1
  (`pgd_at`) seed 2 — clean, CE-PGD-20 and AutoAttack reported separately,
  best and last kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best.pt` (val epoch 103) | `ab4912bc2569…` | 0.8285 | 0.5103 | 0.4743 |
  | `last.pt` (epoch 199) | `4d758cf35a51…` | 0.8419 | 0.4220 | 0.4019 |

  The selected epoch is read directly from the training bundle's
  `metrics.jsonl`: `val_pgd_accuracy` peaks at 0.5130 on epoch 103
  (`val_clean_accuracy` 0.8396) and never recovers, ending at 0.4394 on epoch
  199 — a 7.36 pp validation decay that tracks the 8.83 pp test CE-PGD-20 gap
  below. Seed 1's selected epoch was 104, four epochs past the first LR drop at
  100, and seed 2's is 103; seed 0's is not on record, since it survives only
  as recorded constants with no bundle.

  **(a) The noise floor is now a three-seed measurement for the baseline arm.**
  Arm 1's three seeds under one contract (seed 0 = the reused historical run
  carried as recorded constants, seeds 1 and 2 = fresh runs at
  `cd0b571e4685`):

  | quantity | seed 0 | seed 1 | seed 2 | mean | range | sd (n=3) |
  |---|---|---|---|---|---|---|
  | best clean | 0.8201 | 0.8277 | 0.8285 | 0.8254 | 0.84 pp | 0.46 pp |
  | best CE-PGD-20 | 0.5112 | 0.5045 | 0.5103 | 0.5087 | 0.67 pp | 0.36 pp |
  | **best AutoAttack** | 0.4763 | 0.4678 | 0.4743 | **0.4728** | **0.85 pp** | **0.44 pp** |
  | last clean | 0.8446 | 0.8433 | 0.8419 | 0.8433 | 0.27 pp | 0.14 pp |
  | last CE-PGD-20 | 0.4189 | 0.4247 | 0.4220 | 0.4219 | 0.58 pp | 0.29 pp |
  | last AutoAttack | 0.4036 | 0.4069 | 0.4019 | 0.4041 | 0.50 pp | 0.25 pp |

  Seed 2 lands inside the seed 0–1 interval on every one of the six
  quantities, so the 0.85 pp best-AutoAttack range measured at two seeds did
  not widen at three. Best-checkpoint quantities still move about twice as
  much as last-checkpoint ones, as expected when `best` is chosen by a
  5,000-image validation split and `last` is fixed at epoch 199.
  **Two floors, not one, and the tighter one is the honest comparator.** Seed
  0 mixes source SHAs (`c2220f1` recorded constants vs `cd0b571e4685` fresh
  runs) — contract-equal by the M1a field-by-field audit, not hash-equal.
  Restricted to the two same-SHA runs (seeds 1 and 2), the best-AutoAttack
  range is **0.65 pp**, and seed 0's 0.4763 sits at the top of the mixed-SHA
  spread. Quote 0.85 pp as the conservative floor and 0.65 pp as the same-SHA
  one; neither is a σ estimate you should extrapolate from, and the 0.44 pp sd
  above is a three-point sample sd, printed because the aggregator's own
  `_summarize` uses exactly that convention, not because three runs pin down a
  distribution.

  **(b) Robust-overfitting suppression is unchanged and remains the sturdier
  claim.** Best-minus-last gap for this run: **8.83 pp CE-PGD-20, 7.24 pp
  AutoAttack**. The four baseline readings that now exist — arm 1 seed 0
  (9.23 / 7.27 pp), seed 1 (7.98 / 6.09 pp), seed 2 (8.83 / 7.24 pp), arm 2
  seed 1 (9.01 / 7.00 pp) — give a baseline AutoAttack-gap band of
  **6.09–7.27 pp**, which seed 2 sits inside rather than widening. Arm 3
  (`adr`) seed 0's 3.94 pp is still 2.15 pp below the nearest baseline
  reading, i.e. further below the band than the band is wide.

  **(c) This does not move the preregistered decision rule at all, and that is
  the point worth recording.** Arm 1 appears in **none** of the three
  `COMPARISON_PAIRS` (`scripts/aggregate_adr_cifar10_replication.py:255-263`):
  the ResNet-18 leg is arm 3 vs **arm 2** (`pgd_at_nesterov`), the primary leg
  is arm 8 vs arm 7 (MobileNetV2), and the TRADES leg is arm 5 vs arm 4. Arm 1
  is the plain-SGD reference the plan deliberately excluded from the rule to
  remove the optimizer confound. So completing arm 1 buys **calibration, not
  evidence**: it tells us how much a baseline seed moves, and nothing about
  whether ADR helps. Against this floor the directional accuracy signal is
  arm 3 seed 0's `best.pt` AutoAttack 0.4875 versus arm 1's three-seed mean
  0.4728 (+1.47 pp), versus arm 1's best individual seed 0.4763 (+1.12 pp),
  and versus the baseline the rule actually uses, arm 2 seed 1's 0.4741
  (+1.34 pp). Every one of those clears 0.85 pp, but **the treatment arm still
  has exactly one seed**: the floor is now measured for the baseline and
  entirely unmeasured for ADR, so a single ADR seed landing 1.3–1.5 pp above a
  0.85 pp baseline spread is suggestive and nothing more. Report as
  directional; claim nothing.
  **Cost.** This run took 01:59:34.2Z → 03:51:01.6Z, **1 h 51 min 27 s** for
  two checkpoints × (clean + CE-PGD-20 + AutoAttack). It started four seconds
  after the arm 1 seed 1 evaluation finished (01:59:30.9Z), i.e. the two ran
  strictly sequentially, and its 1 h 51 min matches that run's 1 h 50 min
  almost exactly. Three of the five evaluations measured so far now cluster at
  1 h 50 min – 1 h 57 min when only two MobileNetV2 training jobs contend; the
  two outliers (2 h 37 min, 3 h 14 min) ran concurrently with each other. The
  ≈1 GPU-h per AutoAttack checkpoint line holds; the ≈58 GPU-h corrected
  evaluation total (see the previous entry's budget correction) stands.
  **The M3 aggregator directory-naming defect has a fifth instance and still
  needs no further evidence.** `_load_arm_seed` builds
  `run_root / f"{arm_key}-s{seed}"` (`…:269`), i.e. `pgd_at-s2`, while the real
  dir is `cifar10_r18_pgd_at-s2`. Everything else in the script passes on this
  bundle by inspection (`count == 10000`, `split == "test"`,
  `runtime_method == "pgd_at"`, `weights == "model"`, an AutoAttack block on
  each row, `attack_version == "standard"`, `expected_commit` equal to the
  pinned upstream, both aliases present). Left unfixed: this postrun imports
  nothing and touches no code.
  Campaign state at the 03:52Z scan (`--include-hand-run`, fresh unused
  cursor — 19 bundles), **10 of 20 run dirs**: training terminal and
  successful for `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `adr-s0`,
  `adr-s1`, `trades-s1`, `trades_adr-s0` (8 of 20); `mobilenetv2_adr-s1`
  running at epoch 162 and `mobilenetv2_pgd_at-s0` at epoch 144. Contract
  (model-weights) evaluations **4 of 20** terminal — `pgd_at_nesterov-s1`,
  `adr-s0`, `pgd_at-s1`, `pgd_at-s2` — with `pgd_at_nesterov-s0/evaluation`
  running and showing no progress row yet. EMA-weights evaluations: **1 of 9**
  terminal. The three `adr-s1/early-check-eval-*` bundles are terminal but are
  not contract evaluations and are not read by the aggregator. Arm 6
  (`trades_49k_validation`) and seed 2 of arms 2/3/4 have not started. M1b and
  M1c stay unticked.
- 2026-09-10: postrun of the `eval-2d9fade14b51c95852f5` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_pgd_at_nesterov-s0/train/
  evaluation/`) — arm 2 (`pgd_at_nesterov`, the Nesterov-matched baseline)
  seed 0, `--weights=model` against `best.pt` and `last.pt`, clean +
  CE-PGD-20 + AutoAttack on the official CIFAR-10 test set. **Nothing
  imported — no milestone closed.** The campaign is mid-flight and
  `docs/experiments/` still holds only `.gitkeep`, so the idempotency check
  had nothing to collide with. **This is the second evaluated seed of arm 2 —
  the first time the baseline that the ResNet-18 decision rule actually uses
  has a seed spread of its own** (`COMPARISON_PAIRS["resnet18_nesterov_
  matched"] = {"treatment": "adr", "baseline": "pgd_at_nesterov"}`,
  `scripts/aggregate_adr_cifar10_replication.py:106-108`). Every earlier
  ADR-vs-baseline number in this log was read against a single arm-2 seed or
  against arm 1, which the rule excludes.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan over the run's parent
  returns `terminal: true`, `success: true`, `status: completed`,
  `failure_class: null` for exactly the `--state-path` the watcher passed;
  `run-bundle/completion.json` is `{"status": "completed", "results": 2}` and
  `error-marker.txt` reads "no application error recorded".
  Bundle verified against the frozen contract. All seven declared artifacts
  exist both at their source paths and at their content-addressed
  `run-bundle/artifacts/<name>/<sha256>/` paths, and every one of those
  directory names equals the manifest's own `sha256` field for that artifact
  (`resolved_evaluation_config.yaml` `d8e1474d41ba…`,
  `evaluation-lineage.json` `59291ebfde13…`, `evaluation-results.json`
  `6df4129d14dd…`, `panel-best.jsonl` `0c40534f75cb…`,
  `sample-stats-best.parquet` `50462a44c8b4…`, `panel-last.jsonl`
  `39075962892a…`, `sample-stats-last.parquet` `8720d9d9694c…`), plus
  `autoattack-best.json` and `autoattack-last.json` beside them. Artifact
  bytes were again not re-hashed — this session's sandbox refuses to run a
  hash over paths outside the repo root — so integrity rests on that
  path-equals-hash correspondence; the aggregator owns the real check at M3.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`)
  from worktree `source-cd0b571e4685`, external lock `05cfce4cf8db…` with
  `.external/adr` at the pinned `515da0e0373f…`.
  Lineage ties to the training run: `evaluation-lineage.json` records
  `training_config_hash == training_runtime_config_hash == raw_mapping_hash ==
  4181914e3e4b…` with `applied: []` (no config migration), which is the
  training bundle's own `config_hash`; `train_run_id` on both rows is
  `adr-campaign-v1-cifar10_r18_pgd_at_nesterov-s0`, whose training bundle is
  terminal (`epoch_metrics_complete: true`, 200 rows in `metrics.jsonl`
  ending at epoch 199 / `global_step` 70400, finished 2026-09-09T16:39:16Z)
  with `best.pt`, `last.pt` and `epoch-{049,099,149,199}.pt` on disk and **no
  `best-ema.pt`**, which is correct — arm 2 has no EMA leg
  (`has_ema: False`), so the aggregator will not look for an
  `evaluation-ema/` sibling here.
  Contract fields match arm 2 exactly: protocol
  `controlled_cifar10_r18_adr_v1` with **`nesterov: true`**, method `pgd_at`
  v1 (`adr: null`), `teacher: null`, `epochs=200`, `milestones=[100,150]
  gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`, `validation_fraction=0.1`,
  `world_size=1`, effective global batch 128, identity normalization,
  `deterministic: true`. Evaluation identity: `split: test`, `count: 10000`
  on both rows, CE-PGD-20 at `epsilon=8/255 step=2/255 steps=20
  random_start=true`, `evaluation_seed=0`, threat hash `7081101693340e70…` —
  the **same** threat hash as every other evaluated arm in this campaign.
  AutoAttack is the **standard** version with real provenance
  (`expected_commit == vcs_commit ==
  a39220048b3c9f2cca9a4d3a54604793c68eca7e`), run in its own process at
  `epsilon=8/255`, `Linf`, batch 128, seed 0. `weights: model` and
  `selection_weights: model` on both rows. Training seeds all 0 except the
  fixed `split=20260722`.

  **Official CIFAR-10 test set (10,000 examples), student weights, arm 2
  (`pgd_at_nesterov`) seed 0 — clean, CE-PGD-20 and AutoAttack reported
  separately, best and last kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best.pt` (val epoch 102) | `0c6a451e1c63…` | 0.8232 | 0.5081 | 0.4719 |
  | `last.pt` (epoch 199) | `aecb953d38cb…` | 0.8424 | 0.4237 | 0.4057 |

  The selected epoch is read directly from the training bundle's
  `metrics.jsonl`: `val_pgd_accuracy` peaks at 0.5184 on epoch 102
  (`val_clean_accuracy` 0.8292) and never recovers, ending at 0.4296 on epoch
  199 — an 8.88 pp validation decay that tracks the 8.44 pp test CE-PGD-20
  gap below. Re-reading arm 2 seed 1's `metrics.jsonl` the same way gives
  epoch **101** (peak 0.5188), a number that entry did not record. Every
  selected epoch measured so far sits in a four-epoch window just past the
  first LR drop at 100: arm 2 seed 1 = 101, arm 2 seed 0 = 102, arm 1 seed 2
  = 103, arm 3 seed 0 = 103, arm 1 seed 1 = 104. Best-checkpoint selection is
  behaving identically across arms and optimizers, which is what makes the
  best-vs-last contrast readable at all.

  **(a) Arm 2's own seed spread is 0.22 pp on best AutoAttack — but it is a
  two-seed range and must not be read as the floor.** Both runs are the same
  source SHA `cd0b571e4685`, same protocol, same evaluation identity;
  only `ARD_SEED` differs.

  | quantity | seed 0 | seed 1 | mean | range |
  |---|---|---|---|---|
  | best clean | 0.8232 | 0.8219 | 0.8226 | 0.13 pp |
  | best CE-PGD-20 | 0.5081 | 0.5097 | 0.5089 | 0.16 pp |
  | **best AutoAttack** | 0.4719 | 0.4741 | **0.4730** | **0.22 pp** |
  | last clean | 0.8424 | 0.8406 | 0.8415 | 0.18 pp |
  | last CE-PGD-20 | 0.4237 | 0.4196 | 0.4217 | 0.41 pp |
  | last AutoAttack | 0.4057 | 0.4041 | 0.4049 | 0.16 pp |

  Arm 1 is the reason to distrust 0.22 pp as a floor: its own two same-SHA
  seeds spanned 0.65 pp on best AutoAttack, and adding the third seed widened
  the (mixed-SHA) range to 0.85 pp. A two-point range is a lower bound on
  spread, not an estimate of it. No mean/sd is quoted here for two runs, per
  the claims-discipline rule. **Quote arm 1's 0.85 pp as the campaign's
  conservative noise floor until arm 2 has its third seed**; arm 2's 0.22 pp
  is one draw of a two-point range and nothing more.

  **(b) The ADR-vs-matched-baseline gap now clears every floor on record, and
  the arithmetic is finally against the right baseline.** Arm 3 (`adr`) seed
  0's contract `best.pt` AutoAttack is 0.4875. Against arm 2's two-seed mean
  0.4730 that is **+1.45 pp**; against arm 2's better seed (seed 1, 0.4741)
  **+1.34 pp**; against its weaker seed (seed 0, 0.4719) **+1.56 pp**. The
  whole +1.34/+1.56 pp band sits above the 0.85 pp conservative floor and
  roughly 6× above arm 2's own two-seed 0.22 pp range. On `last.pt` the gap
  is far larger — arm 3 seed 0's 0.4481 against arm 2's 0.4041/0.4057 is
  **+4.24 to +4.40 pp** — which is the robust-overfitting effect in (c)
  showing up as accuracy, not an independent finding.
  **This still is not the preregistered result, and the missing half is now
  entirely on the treatment side.** The rule needs three contract evaluations
  per arm on arms 2 and 3; arm 2 has two, arm 3 has **one** (plus the
  `early-check-eval-aa-best-model` diagnostic at 0.4865, giving an ADR-side
  two-point spread of 0.10 pp that mixes a contract evaluation with a
  non-contract one). And the *primary* leg of the rule is MobileNetV2 (arm 8
  vs arm 7), which still has **no evaluation data at all** — both its
  training runs are the two jobs currently on the GPUs. Report as
  directional; claim nothing.

  **(c) Robust-overfitting suppression is unchanged and remains the sturdier
  claim.** Best-minus-last gap for this run: **8.44 pp CE-PGD-20, 6.62 pp
  AutoAttack**. The five baseline readings that now exist — arm 1 seed 0
  (9.23 / 7.27 pp), seed 1 (7.98 / 6.09 pp), seed 2 (8.83 / 7.24 pp), arm 2
  seed 1 (9.01 / 7.00 pp), arm 2 seed 0 (8.44 / 6.62 pp) — keep the baseline
  AutoAttack-gap band at **6.09–7.27 pp**, which this run sits inside rather
  than widening, and the CE-PGD-20 band at 7.98–9.23 pp. Arm 3 (`adr`) seed
  0's 3.94 pp AutoAttack gap is still 2.15 pp below the nearest baseline
  reading — further below the band than the band is wide, and now measured
  against a band that includes two seeds of the matched-optimizer arm rather
  than one.
  **Cost.** This evaluation ran 02:44:40.1Z → 04:33:01.9Z, **1 h 48 min 22 s**
  for two checkpoints × (clean + CE-PGD-20 + AutoAttack) — the fastest of the
  six measured so far. It corrects one reading in the previous entry: the arm
  1 seed 2 evaluation (01:59:34Z → 03:51:02Z) was *not* uncontended for its
  second half, because this one started at 02:44:40Z and the two overlapped
  for 1 h 06 min alongside both MobileNetV2 training jobs. Four of the six
  evaluations now cluster at 1 h 48 min – 1 h 57 min and a second concurrent
  evaluation did not move them, so the ≈1 GPU-h per AutoAttack checkpoint
  line holds and the ≈58 GPU-h corrected evaluation total stands.
  **The M3 aggregator directory-naming defect has a sixth instance and needs
  no further evidence.** `_load_arm_seed` builds
  `run_root / f"{arm_key}-s{seed}"` (`…:269`), i.e. `pgd_at_nesterov-s0`,
  while the real dir is `cifar10_r18_pgd_at_nesterov-s0`. Everything else in
  the script passes on this bundle by inspection (`count == 10000`,
  `split == "test"`, `runtime_method == "pgd_at"`, `weights == "model"`, an
  AutoAttack block on each row, `attack_version == "standard"`,
  `expected_commit` equal to the pinned upstream, both aliases present), and
  `has_ema: False` correctly means no `evaluation-ema/` lookup. Left unfixed:
  this postrun imports nothing and touches no code.
  Campaign state at the 04:40Z scan (`--include-hand-run`, fresh unused
  cursor — 20 bundles), **10 of 20 run dirs**: training terminal and
  successful for `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `adr-s0`,
  `adr-s1`, `trades-s1`, `trades_adr-s0` (8 of 20); `mobilenetv2_adr-s1`
  running at epoch 193 and `mobilenetv2_pgd_at-s0` at epoch 175. Contract
  (model-weights) evaluations **5 of 20** terminal — `pgd_at_nesterov-s1`,
  `adr-s0`, `pgd_at-s1`, `pgd_at-s2`, `pgd_at_nesterov-s0`. EMA-weights
  evaluations: **1 of 9** terminal. **No evaluation is running for the first
  time since 2026-09-09**: every evaluable terminal training run has been
  evaluated, so the evaluation queue has drained and the campaign is now
  gated purely on training. The three `adr-s1/early-check-eval-*` bundles are
  terminal but are not contract evaluations and are not read by the
  aggregator. Arm 6 (`trades_49k_validation`) and seed 2 of arms 2/3/4 have
  not started — **10 training runs still have no run dir**, and with the two
  MobileNetV2 jobs finishing within the hour the hosts will go idle unless
  the next batch is launched. That is a launch decision for the human, not
  this postrun. M1b and M1c stay unticked.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_mobilenetv2_adr-s1`
  **training** terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_mobilenetv2_adr-s1/train/run-bundle/manifest.json`) — **the
  campaign's first MobileNetV2 run, and the first job of any kind on arm 8,
  the treatment side of the rule's primary leg**. **Nothing imported — no
  milestone closed.** This is training only: the run has no `evaluation/` or
  `evaluation-ema/` output at all, so it produces no official-test number, and
  `docs/experiments/` still holds only `.gitkeep`, so the idempotency check had
  nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of the campaign root returns
  `terminal: true`, `success: true`, `status: completed`, `failure_class: null`
  on exactly the `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and `epoch-metrics.jsonl` really carries 200 rows ending at epoch 199
  / `global_step` 70400. Both declared artifacts exist at their source paths
  and at their content-addressed `run-bundle/artifacts/<name>/<sha256>/` paths,
  and each directory name equals the manifest's own `sha256` for that artifact
  (`epoch-metrics.parquet` `825545921ef0…`, `sample-stats-train.parquet`
  `4f651d5dd886…`). `best.pt`, `best-ema.pt`, `last.pt` and all four periodic
  checkpoints (`epoch-049/099/149/199.pt`) are on disk. Source SHA
  `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`) from worktree
  `source-cd0b571e4685`, external lock `05cfce4cf8db…` with `.external/adr` at
  the pinned `515da0e0373f…`. Artifact bytes were not re-hashed — this
  session's sandbox refuses to run a hash over paths outside the repo root — so
  integrity rests on that path-equals-hash correspondence; the aggregator owns
  the real check at M3.
  Contract fields match arm 8 exactly: protocol
  `controlled_cifar10_mobilenetv2_adr_v1`, student `mobilenet_v2_cifar`,
  method `adr` v1, `teacher: null`, `nesterov: true`, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`,
  `validation_fraction=0.1`, ADR `ema_decay=0.995` `T 2.5→2.0`
  `lambda 0.7→0.95`, train attack KL/rectified 10 steps at `8/255` step
  `2/255` random start, selection attack CE 20 steps at the same budget,
  `world_size=1`, effective global batch 128, `deterministic: true`, seeds all
  1 except the fixed `split=20260722` and `evaluation_attack=0`.
  `git diff cd0b571e4685 HEAD` is empty for
  `configs/scientific/cifar10_mobilenetv2_{adr,pgd_at}.yaml` and for the
  protocol's own definition (`src/ard/protocols/__init__.py`,
  `src/ard/config/schema.py`), so the contract has not drifted since the pin.
  **Normalization asymmetry, logged deliberately and not a confound for the
  rule.** Both MobileNetV2 arms declare `normalization: {profile:
  cifar10_standard}` (mean/std), while all six ResNet-18 arms declare
  `cifar10_raw_identity`. The preregistered rule compares an ADR-minus-baseline
  *gain* at MobileNetV2 against the same gain at ResNet-18, and within each
  architecture the treatment and its matched baseline share the profile
  (arms 7 and 8 both `cifar10_standard`; arms 2 and 3 both
  `cifar10_raw_identity`), so neither gain is confounded. What it does mean is
  that the two architectures differ in input-normalization convention as well
  as capacity, so no *level* may be read across architectures — only gains.
  **ADR stop-rule checks pass.** EMA validation ran every epoch and never
  crashed: all 200 rows carry `val_clean_accuracy_ema` and
  `val_pgd_accuracy_ema`. And `best-ema.pt` was selected independently of the
  student — EMA's best validation epoch is **141**, the student's is **151** —
  which is what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section requires.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log):

  | weights | checkpoint | epoch | clean | PGD |
  |---|---|---|---|---|
  | student | best | 151 | 0.7666 | 0.4850 |
  | student | last | 199 | 0.7672 | 0.4774 |
  | EMA | best-ema | 141 | 0.7586 | 0.4914 |
  | EMA | last | 199 | 0.7708 | 0.4792 |

  EMA is ahead of the student at each arm's own best checkpoint (+0.64 pp
  validation PGD), the same direction as both ResNet-18 ADR seeds — a
  consistency check on the selection logic, not independent evidence.
  **Two things about this run look different from every ResNet-18 run so far,
  and both are single-seed observations on validation data.**
  1. **The student's best epoch is 151, not ~102.** Every ResNet-18 arm
     measured so far peaked in a four-epoch window just past the *first* LR
     drop at 100 (arm 2 seed 1 = 101, arm 2 seed 0 = 102, arm 1 seed 2 = 103,
     arm 3 seed 0 = 103, arm 3 seed 1 = 103, arm 1 seed 1 = 104). This run
     peaks one epoch past the *second* drop at 150. The EMA peaks at 141, also
     well after the ResNet-18 window.
  2. **The robust-overfitting gap is 0.76 pp, an order of magnitude below
     anything else on record.** Student best-minus-last validation PGD is
     0.4850 − 0.4774 = 0.0076, against 0.0408/0.0456 for ResNet-18 ADR and
     0.0864/0.0888 for the Nesterov-matched ResNet-18 baseline. Validation PGD
     is essentially flat from epoch 150 to 199 (`val_pgd_late_mean_epoch_150_
     199` 0.4802 versus a 0.4850 peak). **This cannot yet be attributed to
     ADR**: arm 7, the matched MobileNetV2 baseline, has no terminal run — its
     seed 0 was at epoch 185 at the 04:46Z scan — so there is no way to
     separate "ADR suppresses robust overfitting at this capacity" from
     "MobileNetV2 at 200 epochs does not robustly overfit much in the first
     place". The arm 7 seed 0 run finishing within the hour is the single
     cheapest observation that would tell those two apart on validation, ahead
     of any AutoAttack number.
  Clean accuracy sits ≈7 pp below the ResNet-18 ADR arms (0.767 versus
  0.839/0.840 validation clean at best), which is the expected capacity cost
  and not a finding.
  **Budget finding — the plan's ADR cost line does not fit MobileNetV2.** The
  budget table charges every ADR-family run ≈54 s per full epoch (≈3.0 GPU-h),
  a figure extrapolated from the ResNet-18 ADR canary and since confirmed for
  ResNet-18 (`adr-s1` 39.4 s, `adr-s0` 40.7 s training-loop). This run's
  training loop alone ran **61.2–87.1 s/epoch**, settling at **61.2–63.5 s for
  its last 37 epochs** (163–199), the least-contended stretch of the run. The
  training loop by itself is therefore ≈62 s/epoch at best observed — already
  above the 54 s full-epoch allowance — so ≈3.0 GPU-h per run is too low for
  the six MobileNetV2 runs (arms 7 and 8); **≈4 GPU-h is the better estimate**,
  which adds roughly 6 GPU-h to the campaign total. Wall clock for this run was
  23:10:45.1Z → 04:43:21.1Z, **5 h 32 min 36 s** (99.8 s/epoch averaged), but
  that figure is inflated by contention — up to two other MobileNetV2/ResNet-18
  training jobs and two concurrent AutoAttack evaluations shared the host — and
  the per-epoch rows carry no validation-pass timing, so the uncontended
  *full*-epoch cost cannot be isolated from this run. Contention changes only
  wall clock, not results: `deterministic: true`, fixed seeds and fixed batch
  size make the numbers independent of throughput.
  **The M3 aggregator directory-naming defect has a seventh instance and needs
  no further evidence.** `_load_arm_seed` builds
  `run_root / f"{arm_key}-s{seed}"` (`scripts/aggregate_adr_cifar10_
  replication.py:269`), i.e. `mobilenetv2_adr-s1`, while the real dir is
  `cifar10_mobilenetv2_adr-s1`. This is the first instance on a MobileNetV2
  arm, which is the case the fix must get right: arms 1-6 need the
  `cifar10_r18_` prefix and arms 7-8 need `cifar10_` (the config file stem),
  so a single global prefix will not do. Left unfixed: this postrun imports
  nothing and touches no code.
  Campaign state at the 04:46Z scan (`--include-hand-run`, fresh unused
  cursor — 19 bundles), **10 of 20 run dirs**: training terminal and
  successful for `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `adr-s0`,
  `adr-s1`, `trades-s1`, `trades_adr-s0` and now `mobilenetv2_adr-s1` (**9 of
  20**); `mobilenetv2_pgd_at-s0` running at epoch 185, the only job on either
  host. Contract (model-weights) evaluations **5 of 20** terminal —
  `pgd_at_nesterov-s1`, `adr-s0`, `pgd_at-s1`, `pgd_at-s2`,
  `pgd_at_nesterov-s0`; EMA-weights evaluations **1 of 9**. No evaluation is
  running, and **four** terminal training runs have no contract evaluation:
  `adr-s1` (its three `early-check-eval-*` bundles are diagnostics the
  aggregator does not read), `trades-s1`, `trades_adr-s0` and this run. Arm 6
  (`trades_49k_validation`) and seed 2 of arms 2/3/4 have not started.
  **Both hosts go idle within the hour**: `mobilenetv2_pgd_at-s0` is 15 epochs
  from done and 10 training runs still have no run dir. Launching the next
  batch, and queueing the missing evaluations, is a decision for the human —
  this postrun launches nothing. M1b and M1c stay unticked.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_mobilenetv2_pgd_at-s0`
  **training** terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_mobilenetv2_pgd_at-s0/train/run-bundle/manifest.json`) — **arm 7 seed
  0, the matched MobileNetV2 baseline, and the second half of the decision
  rule's primary leg**. **Nothing imported — no milestone closed.** Training
  only: the run has no `evaluation/` output, so it produces no official-test
  number, and `docs/experiments/` still holds only `.gitkeep`, so the
  idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `status: completed`, `failure_class: null` on exactly the
  `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and `epoch-metrics.jsonl` really carries 200 rows ending at epoch 199
  / `global_step` 70400. Both declared artifacts exist at their source paths
  and at their content-addressed `run-bundle/artifacts/<name>/<sha256>/` paths,
  each directory name equalling the manifest's own `sha256` for that artifact
  (`epoch-metrics.parquet` `71a71b9ad4de…`, `sample-stats-train.parquet`
  `c5b3b5bdd1de…`). `best.pt`, `last.pt` and all four periodic checkpoints
  (`epoch-049/099/149/199.pt`) are on disk; there is **no** `best-ema.pt`,
  which is correct — `method.adr` is `null` for this arm, so no EMA teacher
  exists to select one. Source SHA `cd0b571e4685…` clean (`dirty: false`,
  empty-diff `e3b0c442…`) from worktree `source-cd0b571e4685`, external lock
  `05cfce4cf8db…`, `config_hash fac5b4eb06c4…`. Artifact bytes were not
  re-hashed — this session's sandbox refuses any Bash read outside the repo
  root — so integrity rests on that path-equals-hash correspondence; the
  aggregator owns the real check at M3.
  Contract fields match arm 7 exactly: protocol
  `controlled_cifar10_mobilenetv2_adr_v1`, student `mobilenet_v2_cifar`,
  method `pgd_at` v1, `teacher: null`, `adr: null`, `nesterov: true`,
  `epochs=200`, `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9
  wd=5e-4`, `validation_fraction=0.1`, train attack CE 10 steps at `8/255`
  step `2/255` random start, selection attack CE 20 steps at the same budget,
  `world_size=1`, effective global batch 128, `deterministic: true`, seeds all
  0 except the fixed `split=20260722` and `evaluation_attack=0`. (The resolved
  config also carries `method.student_ema_decay: 0.9`; that is the schema
  default for policy methods and is inert under `pgd_at`.) `git diff
  cd0b571e4685 HEAD` is empty for
  `configs/scientific/cifar10_mobilenetv2_{pgd_at,adr}.yaml` and for
  `src/ard/protocols/__init__.py` and `src/ard/config/schema.py`, so the
  contract has not drifted since the pin. This run also confirms from the
  baseline side what the arm-8 entry asserted: both MobileNetV2 arms really do
  declare `normalization: {profile: cifar10_standard}`, so the MobileNetV2
  gain is internally matched and only *levels*, never gains, are blocked from
  cross-architecture reading.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log). Both
  rows re-read from `epoch-metrics.jsonl` and they match the manifest summary
  exactly:

  | checkpoint | epoch | clean | PGD |
  |---|---|---|---|
  | best | 175 | 0.7460 | 0.4496 |
  | last | 199 | 0.7480 | 0.4434 |

  **This closes the arm-8 entry's open question, and the answer is the
  deflationary one.** That entry flagged `mobilenetv2_adr-s1`'s 0.76 pp
  robust-overfitting gap as an order of magnitude below every ResNet-18
  reading, and named this run as the cheapest observation able to separate
  "ADR suppresses robust overfitting at this capacity" from "MobileNetV2 at 200
  epochs barely robustly overfits in the first place". The baseline's gap is
  **0.62 pp** (0.4496 − 0.4434) — *smaller* than the ADR arm's 0.76 pp. So the
  tiny gap is a property of MobileNetV2 at this capacity and horizon, not an
  ADR effect. For contrast, at ResNet-18 ADR roughly halves the gap
  (0.0456/0.0408 against the Nesterov-matched baseline's 0.0888/0.0864). Both
  MobileNetV2 readings are single-seed, on validation, and from *different*
  seeds (arm 7 seed 0 versus arm 8 seed 1), so this is directional; but the
  direction is the opposite of an ADR benefit, and a robust-overfitting claim
  for ADR must from here on be stated as a ResNet-18 result, not a general one.
  **The arm-8 entry's late-peaking observation needs correcting, in two ways.**
  It said "every ResNet-18 arm measured so far peaked in a four-epoch window
  just past the first LR drop (101–104)". That is true of the six PGD-AT and
  ADR readings it listed, but **arm 4 (TRADES) seed 1 peaked at epoch 157**,
  logged earlier in this same plan — so late peaking was already visible at
  ResNet-18 and is not a MobileNetV2 signature. This run peaks later still, at
  **175**. The honest summary is that selected epoch tracks the method and
  architecture together: R18 PGD-AT/ADR 101–104, R18 TRADES 157, MobileNetV2
  ADR 151, MobileNetV2 PGD-AT 175.
  Second, **this run's argmax is weak and should not be read as a real peak.**
  Validation PGD over epochs 150–199 is a flat, noisy plateau: the winning
  0.4496 at epoch 175 beats the runner-up (0.4468, epoch 182) by only 0.28 pp,
  and nine separate epochs sit within 0.42 pp of the top (0.4454–0.4496 at
  epochs 154, 159, 161, 162, 167, 175, 178, 182, 186). Selection here is close
  to arbitrary within the plateau, which is also part of why best-minus-last is
  so small — there is little to choose between `best.pt` and `last.pt` on this
  arm. Expect the official-test best-vs-last gap for MobileNetV2 baselines to
  be correspondingly small, and do not treat a sub-1-pp MobileNetV2 gap as
  evidence of anything.
  **Directional read of the primary leg — the first time its shape is visible
  at all, and it must not be quoted as a result.** On validation PGD at each
  arm's own best checkpoint, ADR minus matched baseline is **+3.54 pp** at
  MobileNetV2 (0.4850 arm 8 s1 versus 0.4496 arm 7 s0) against **+2.06 to
  +2.12 pp** at ResNet-18 (0.5394/0.5396 arm 3 versus 0.5184/0.5188 arm 2).
  ADR is also ahead on clean accuracy by more at MobileNetV2 (+2.06 pp, 0.7666
  versus 0.7460) than at ResNet-18 (+0.32 to +0.42 pp). That ordering is the
  "sign confirmed" direction of the preregistered rule — but the rule is
  **AutoAttack on the official 10,000-example test set, best checkpoint, three
  seeds per arm**, and every number here fails that on four counts at once: it
  is validation not test, CE-PGD-20 not AutoAttack, one seed per MobileNetV2
  arm and not even the same seed, and validation PGD is the metric these runs'
  own checkpoint selection optimized against, so it flatters both treatment
  arms. **No verdict is licensed.** The cheapest observation that would make
  the MobileNetV2 leg seed-matched is arm 8 seed 0 or arm 7 seed 1.
  **Budget finding — the plan's non-ADR cost line does not fit MobileNetV2
  either, and the arm-8 entry's correction was arithmetically short.** The
  budget table charges arms 1/2/4/6/7 ≈47 s per full epoch (≈2.6 GPU-h per
  run). This run's training loop alone ran 55.5–79 s/epoch, settling at
  **55.5–56.3 s over its last ~14 epochs** (186–199) at 805–811 img/s — the
  stretch after `mobilenetv2_adr-s1` finished at 04:43:21Z, so the least
  contended. The training loop by itself is therefore ≈55.5 s/epoch at best
  observed, already above the 47 s *full*-epoch allowance. Non-training
  overhead (the single validation pass plus checkpointing and logging) is
  ≈13 s/epoch, from mean wall clock 81.4 s/epoch minus a mean training-loop
  time of ≈68 s/epoch estimated from binned per-epoch values (25 epochs in
  55–59 s, 87 in 60–69 s, 88 in 70–79 s), so that figure carries a ±1–2 s
  error and is not a measurement. Uncontended full epoch is therefore ≈68 s,
  i.e. **≈3.8 GPU-h per arm-7 run, not 2.6**. The arm-8 entry put the
  MobileNetV2 correction at "roughly 6 GPU-h" by charging all six MobileNetV2
  runs the ADR line's 3.0; the table actually charges arm 7 at 2.6 and arm 8
  at 3.0, so the correct increment is **≈+7 GPU-h** (arm 7: 3 runs × (4−2.6) =
  +4.2; arm 8: 3 × (4−3.0) = +3.0), taking the campaign's training total from
  55.6 to ≈63 GPU-h. Wall clock for this run was 00:30:08.9Z → 05:01:28.5Z,
  **4 h 31 min 19.6 s** (81.4 s/epoch averaged, ≈4.5 GPU-h as executed), which
  is inflated by contention with `mobilenetv2_adr-s1` and two AutoAttack
  evaluations. Contention changes only wall clock, not results:
  `deterministic: true`, fixed seeds and fixed batch size make the numbers
  independent of throughput.
  **The M3 aggregator directory-naming defect has an eighth instance.**
  `_load_arm_seed` builds `run_root / f"{arm_key}-s{seed}"`
  (`scripts/aggregate_adr_cifar10_replication.py:269`), i.e.
  `mobilenetv2_pgd_at-s0`, while the real dir is
  `cifar10_mobilenetv2_pgd_at-s0`. Both MobileNetV2 arm keys are now confirmed
  to need the `cifar10_` prefix and arms 1–6 the `cifar10_r18_` prefix, so a
  single global prefix will not do. Left unfixed: this postrun imports nothing
  and touches no code.
  Campaign state at the 05:05Z scan (`--include-hand-run`, fresh unused
  cursor — 19 bundles, every one terminal and successful), **10 of 20 run
  dirs**, all 10 terminal: `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`,
  `adr-s0`, `adr-s1`, `trades-s1`, `trades_adr-s0`, `mobilenetv2_adr-s1` and
  now `mobilenetv2_pgd_at-s0`. Contract (model-weights) evaluations **5 of
  20** — `pgd_at_nesterov-s1`, `adr-s0`, `pgd_at-s1`, `pgd_at-s2`,
  `pgd_at_nesterov-s0`; EMA-weights evaluations **1 of 9**. **Five** terminal
  training runs have no contract evaluation: `adr-s1` (its three
  `early-check-eval-*` bundles are diagnostics the aggregator does not read),
  `trades-s1`, `trades_adr-s0`, `mobilenetv2_adr-s1` and this run. Arm 6
  (`trades_49k_validation`) and seed 2 of arms 2/3/4 have not started.
  **Open question for the human — Ferret.** Nothing is running on Hamster
  (both GPUs at 0 %, 1 and 17 MiB), but Ferret's three GPUs are all occupied
  at 1.5–2.2 GiB and 15–41 %, which is the right size for three concurrent
  training jobs of this campaign. No bundle for them exists under the local
  campaign root and this session may only run `nvidia-smi`/`uptime` on Ferret,
  so their identity is unverified. **Until that is resolved, treat "10
  training runs have no run dir" as a statement about the local root only, not
  as a count of unlaunched work.** Launching anything, and queueing the five
  missing evaluations, is a decision for the human — this postrun launches
  nothing. M1b and M1c stay unticked.

- **2026-09-10, M1b closed + evaluation sweep + Ferret-queue rebalancing +
  aggregator bug fix.** Per the user's standing autonomous-check-and-fix
  authorization:
  - **M1b**: archived both reused historical bundles (PGD-AT
    `pgd-at-controlled-s0-c2220f1`, TRADES `trades-fix-v1-s0-attempt2`) into
    `docs/experiments/historical/{pgd_at_controlled_s0_c2220f1,trades_fix_v1_seed0}/`
    — `evaluation-results-{pgd20,aa}.json`, `resolved_config.yaml`, a
    hand-written `provenance.json`, and a bare-hash `.sha256` next to each
    file. Before copying, independently recomputed `sha256sum` on both
    runs' actual `best.pt`/`last.pt` and confirmed exact equality with the
    `checkpoint_sha256` field already inside each evaluation-results.json
    (not assumed) — both bundles' numbers are genuinely tied to the
    checkpoints they claim to evaluate.
  - **Evaluation sweep**: ran canonical contract evaluations (CE-PGD20 +
    AutoAttack, `checkpoints: both`, model weights and — where the arm has
    one — EMA weights) for every training-terminal, not-yet-evaluated
    seed reachable at the time: `pgd_at-s1/-s2`, `pgd_at_nesterov-s0/-s1`
    (Hamster) and `-s2` (Ferret), `adr-s0` (model + EMA). All 7 came back
    sane against the literature/consistency expectations, no bug found:
    `pgd_at` vs `pgd_at_nesterov` 3-seed mean best-AutoAttack differs by
    0.04pp (contract predicts ~0); `adr-s0` best-AutoAttack 48.75%
    (model) / 49.61% (EMA), both within ~0.1-0.9pp of seed 1's already-known
    48.65%; AutoAttack ≤ CE-PGD20 by a sane margin everywhere; no clean-
    accuracy collapse anywhere. Two Ferret seeds (`trades-s2`,
    `trades_49k_validation-s0`) were left unevaluated — no literature
    baseline was on hand to judge deviation for the `trades` arm in that
    pass, out of scope, not a problem found.
  - **Ferret-queue rebalancing**: found Hamster's own 10-job share complete
    and both its GPUs idle while Ferret's queues still had
    `mobilenetv2_pgd_at` seeds 1 and 2 queued (not missing — confirmed by
    reading all three driver scripts' job lists end-to-end — just later in
    each GPU's sequential queue, behind still-running jobs). Stopped the two
    relevant local Ferret-driver polling loops (PIDs 1616152/1616153 — these
    are local bash processes tracking remote job status, not the remote
    training jobs themselves, which are detached and unaffected) with the
    user's explicit permission, and launched both seeds directly on
    Hamster's now-idle GPUs 0/1 instead, to avoid a redundant duplicate
    launch once Ferret's queue would otherwise have reached them, and to
    finish them sooner than waiting behind Ferret's remaining queue.
  - **Aggregator bug found and fixed** (pre-existing, from this session's
    earlier authoring of `scripts/aggregate_adr_cifar10_replication.py`, not
    from a run — caught while cross-checking an earlier automated postrun's
    note above about an "eighth instance" of a directory-naming defect):
    `_load_arm_seed` built `run_root / f"{arm_key}-s{seed}"`, which is wrong
    for **every** arm, not only MobileNetV2 — actual run directories carry a
    `cifar10_r18_` prefix for arms 1-6 and `cifar10_` for arms 7-8, neither
    of which matches the bare `ARMS` dict key. Fixed by adding a
    `"dir_prefix"` field to every `ARMS` entry and using
    `f"{arm['dir_prefix']}{arm_key}-s{seed}"`; smoke-tested against three
    real run directories (`pgd_at`, `mobilenetv2_pgd_at`, `adr`) to confirm
    the resolved paths now exist on disk. This would otherwise have broken
    M3 for every arm, not just MobileNetV2, the moment the aggregator was
    first run for real.
  - Remaining before M1c can close: Ferret still has `mobilenetv2_adr-s0/-s2`
    and `trades_adr-s2` running, plus `adr-s2`, `trades-s2`,
    `trades_49k_validation-s0`, `trades_adr-s1`, and now-superseded-on-Ferret
    `mobilenetv2_pgd_at-s1/-s2` sitting terminal-but-uncollected — all need
    `ferret-collect` into the canonical local run root and then canonical
    evaluation, matching the same literature/consistency check as above,
    especially the still-unchecked `trades_adr` arm (never validated before,
    teacher-free EMA + TRADES combination) and the `trades_49k_validation`
    pilot (the diagnostic this campaign exists partly to answer).

- 2026-09-10: postrun of the `adr-campaign-v1-ferret-cifar10_r18_trades-s2`
  **training** terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_r18_trades-s2/train/run-bundle/manifest.json`) — **arm 4 seed 2, and
  the first postrun in this campaign of a run that executed on Ferret rather
  than Hamster**. It also completes arm 4's training side: seed 0 is the reused
  historical bundle, seeds 1 and 2 are now both terminal, so TRADES is the
  first arm with all three seeds trained. **Nothing imported — no milestone
  closed.** Training only: the run has no `evaluation/` output, so it produces
  no official-test number, and `docs/experiments/` holds only `.gitkeep` and
  the M1b `historical/` archive, neither of which is this contract's record —
  so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `status: completed`, `failure_class: null` on exactly the
  `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and 200 epoch rows really present in **both** `epoch-metrics.jsonl`
  and `run-bundle/metrics.jsonl`, ending at epoch 199 / `global_step` 70400.
  Both declared artifacts have a content-addressed copy under
  `run-bundle/artifacts/<name>/<sha256>/`, and each directory name equals the
  manifest's own `sha256` for that artifact (`epoch-metrics.parquet`
  `650edbe2cb3d…`, `sample-stats-train.parquet` `a9a655f17a62…`). `best.pt`,
  `last.pt` and all four periodic checkpoints (`epoch-049/099/149/199.pt`) are
  on disk; there is **no** `best-ema.pt`, which is correct — `method.adr` is
  `null` for arm 4. Source SHA `cd0b571e4685…` clean (`dirty: false`,
  empty-diff `e3b0c442…`), external lock `05cfce4cf8db…` with `adr` pinned at
  `515da0e0373f…`, `config_hash dc73a1bf0190…`. Artifact bytes were not
  re-hashed — this session's sandbox refuses any Bash read outside the repo
  root — so integrity rests on that path-equals-hash correspondence; the
  aggregator owns the real check at M3.
  Contract fields match arm 4 exactly: protocol `controlled_cifar10_r18_v1`,
  student `saad_resnet18_cifar_v1` with `cifar10_raw_identity` normalization,
  method `trades` v1 (`trades_beta 6.0`, `adr: null`), `teacher: null`,
  **`nesterov: false`** as arm 4 requires, `epochs=200`, `milestones=[100,150]
  gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`, `validation_fraction=0.1`, train
  attack KL/`student_clean` 10 steps at `8/255` step `2/255` random start,
  selection and evaluation attack CE 20 steps at the same budget,
  `world_size=1`, effective global batch 128, `deterministic: true`, seeds all
  2 except the fixed `split=20260722` and `evaluation_attack=0`. `git diff
  cd0b571e4685 HEAD` is empty for `configs/scientific/cifar10_r18_trades.yaml`,
  `src/ard/protocols/__init__.py` and `src/ard/config/schema.py`, so the
  contract has not drifted since the pin.
  **A second, different M3-blocking aggregator defect — this one caused by
  `ferret-collect`, and it is not the `dir_prefix` bug fixed above.**
  `_verify_bundle` resolves each artifact from the manifest's own absolute
  `path` field (`scripts/aggregate_adr_cifar10_replication.py:201`). For a
  Hamster run that field points at the canonical run dir and the file is there.
  For a **Ferret-collected** run it still points at the host-side staging dir
  the job actually wrote to — here
  `runs/adr-campaign-v1-ferret-cifar10_r18_trades-s2/outputs/train/
  epoch-metrics.parquet` — which **does not exist in the local runtime tree at
  all** (checked: no `runs/adr-campaign-v1-ferret-*` directory exists). The
  aggregator would therefore raise `declared artifact is missing` and abort M3.
  This affects every run whose `run_id` carries the `-ferret-` infix, which is
  already **6 of the 18 terminal training runs**: `r18_trades-s2` (this one),
  `r18_adr-s2`, `r18_pgd_at_nesterov-s2`, `r18_trades_adr-s1`,
  `r18_trades_49k_validation-s0` and `mobilenetv2_adr-s2` — including three
  arms' entire seed-2 leg. The one-line fix is to fall back to the bundle's own
  content-addressed copy when the declared path is absent, i.e. in
  `_verify_bundle`, `if not path.is_file(): path = bundle / entry["local_path"]
  / path.name`. That copy exists locally for this run and travels with the
  bundle by construction. It does **not** weaken the check: the `sha256`
  comparison against the manifest still runs on whichever file is found, so a
  corrupted or substituted artifact still fails. Left unfixed here: this
  postrun imports nothing and touches no code.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log). Both
  rows re-read from `epoch-metrics.jsonl` and they match the manifest summary
  exactly:

  | checkpoint | epoch | clean | PGD |
  |---|---|---|---|
  | best | 102 | 0.8102 | 0.5166 |
  | last | 199 | 0.8432 | 0.4852 |

  **Cross-seed agreement within arm 4 is unusually tight.** Against seed 1
  (Hamster, logged above): best PGD 0.5166 versus 0.5164 (**0.02 pp** apart),
  last PGD 0.4852 versus 0.4844 (0.08 pp), robust-overfit gap 0.0314 versus
  0.0320 (0.06 pp). Only clean accuracy differs materially (best 0.8102 versus
  0.8322, last 0.8432 versus 0.8338), which is what the differing argmax epoch
  below predicts. Two seeds on validation give a directional reading for those
  two seeds and nothing more, but the arm's robust level and its
  robust-overfitting gap both look seed-stable.
  **This corrects the arm-7 entry's summary of when each arm peaks.** That
  entry concluded "selected epoch tracks the method and architecture together:
  R18 PGD-AT/ADR 101–104, R18 TRADES 157, MobileNetV2 ADR 151, MobileNetV2
  PGD-AT 175". The "R18 TRADES 157" cell was one seed. **This seed peaks at
  epoch 102**, squarely inside the 101–104 window every R18 PGD-AT and ADR run
  occupies. So R18 TRADES does not peak late as an arm property — seed 1
  happened to. The mechanism is visible in this run's curve: validation PGD has
  two nearly equal plateaus, one just past the first LR drop (epoch 102 = 0.5166,
  epoch 105 = 0.5162) and one just past the second (epoch 151 = 0.5126, epoch
  157 = 0.5114), separated by only **0.4 pp**. Which plateau wins the argmax is
  within seed noise, so the selected epoch is close to arbitrary between them
  and no method-level or architecture-level story should be built on it. The
  same caution the arm-7 entry raised about its own flat argmax applies here,
  and it now applies to the ResNet-18 TRADES arm too.
  **First recorded environment asymmetry between the two hosts — logged, not a
  new confound, and the human should know it is there.** This run's
  `environment.json` reports Python **3.12.13 (Anaconda)** on kernel
  `7.0.0-28-generic`; arm 4 seed 1, run on Hamster, reports Python **3.11.15
  (conda-forge)** on kernel `7.0.0-31-generic`. `torch 2.11.0+cu128`, CUDA
  12.8, cuDNN 91900 and the GPU model (RTX 4090) are identical on both, and
  every other Ferret run in this campaign carries the same 3.12.13 record
  (checked `r18_adr-s2`). The campaign therefore pools seeds across two hosts
  with different interpreters — arms 2, 3, 4 and 8 each have their seed 2 on
  Ferret and their other seeds on Hamster. Plan 0094 already tested exactly
  these two axes (Hamster versus Ferret, and environment generation versus
  environment generation) and found the epoch-114 checkpoint **component hashes
  bit-identical**, so this is a sanctioned configuration rather than a fresh
  threat; but 0094 tested a 15-epoch continuation, not a 200-epoch run, and it
  is not recorded anywhere in this plan that the two hosts differ at all. The
  0.02 pp seed-1-versus-seed-2 agreement above is a weak consistency check in
  the same direction, not a controlled test. **No number in this campaign
  should be pooled across hosts without stating the host**, and the M3 record
  should carry the environment block per arm-seed so a reader can see it.
  **Budget finding — the plan's cost table is Hamster-derived and a
  Ferret-executed run costs about twice as much wall clock, but the GPU is not
  the reason.** This run's wall clock was 12:16:37.9Z → 18:32:04.0Z, **6 h 15
  min 26 s** (112.6 s/epoch averaged, ≈6.3 GPU-h as executed), against arm 4
  seed 1's 3 h 27 min on Hamster (62.2 s/epoch, ≈3.5 GPU-h) — **1.8×**. The
  training loop alone tells the same story: 191 of 200 epochs ran at **420–490
  img/s** (`train_seconds` ≈ 92–107 s), where Hamster ran the same arm at 998
  img/s / ≈45.1 s. **But nine epochs ran faster, and the fastest, epoch 97, hit
  1026 img/s** — at or above Hamster's uncontended figure for this arm. A GPU
  that can do 1026 img/s for one epoch is not a slower GPU, so the ≈2.1×
  per-epoch penalty is host contention while three training jobs shared Ferret,
  most plausibly dataloader CPU (three jobs × `num_workers: 8`) or the
  remote-NUMA path already recorded for Ferret's GPU2 in
  `.claude/rules/execution-plane.md`. That is an inference from one fast epoch,
  not a measurement of the cause. Practical consequence for the remaining
  campaign: budget Ferret-executed runs at roughly 2× the table's Hamster
  figure at 3-way concurrency, or run two jobs per host instead of three.
  Contention changes only wall clock, not results: `deterministic: true`, fixed
  seeds and fixed batch size make the numbers independent of throughput.
  Campaign state at the 08:59Z scan (`--include-hand-run`, fresh unused
  cursor), and it has moved a long way since the previous entry: **18 of 20 run
  dirs now exist and all 18 are training-terminal and successful** —
  `pgd_at-s1/-s2`, `pgd_at_nesterov-s0/-s1/-s2`, `adr-s0/-s1/-s2`,
  `trades-s1/-s2`, `trades_adr-s0/-s1`, `trades_49k_validation-s0`,
  `mobilenetv2_adr-s1/-s2`, `mobilenetv2_pgd_at-s0/-s1/-s2` (the last two
  finished at 08:53Z). Only **`mobilenetv2_adr-s0` and `trades_adr-s2`** have
  no run dir. Contract (model-weights) evaluations **6 of 20** terminal —
  `pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `-s2`, `adr-s0`; EMA-weights
  evaluations **1 of 9** (`adr-s0`). **Six evaluations are running**
  (`adr-s1`, `adr-s2`, `trades_adr-s0`, `trades_adr-s1`,
  `trades_49k_validation-s0`, `mobilenetv2_adr-s1`) against five busy GPUs
  (Hamster 2 at 100 %, Ferret 3 at 99 %), so at least one is queued rather than
  computing. **Six terminal training runs still have no evaluation of any
  kind**: `trades-s1`, `trades-s2` (this run), `mobilenetv2_adr-s2`,
  `mobilenetv2_pgd_at-s0`, `-s1` and `-s2` — the whole arm-7 leg and both
  trained TRADES seeds.
  Open for the human, none of it actioned here: launch the two missing training
  runs, queue the six missing evaluations, and fix the artifact-path fallback
  before M3 is attempted. This postrun launches nothing and changes no code.
  M1c stays unticked.
- 2026-09-10: postrun of the `adr-campaign-v1-ferret-cifar10_r18_adr-s2`
  **training** terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_r18_adr-s2/train/run-bundle/manifest.json`) — **arm 3 seed 2, which
  completes arm 3 (ResNet-18 ADR) at all three seeds and, together with arm 2's
  three terminal seeds, gives the ResNet-18 half of the primary leg its first
  seed-matched three-seed reading.** **Nothing imported — no milestone
  closed.** Training only: this bundle carries no official-test number, and
  `docs/experiments/` still holds only `.gitkeep` and `historical/`, so the
  idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `status: completed`, `failure_class: null` on exactly the
  `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and `epoch-metrics.jsonl` really carries 200 rows ending at epoch 199
  / `global_step` 70400. Both declared artifacts have a content-addressed copy
  under `run-bundle/artifacts/<name>/<sha256>/`, each directory name equalling
  the manifest's own `sha256` for that artifact (`epoch-metrics.parquet`
  `9663b843988b…`, `sample-stats-train.parquet` `1daccafa8c5a…`), and both also
  sit in the canonical run dir (`train/epoch-metrics.parquet`,
  `train/sample-stats-train.parquet`). `best.pt`, `best-ema.pt`, `last.pt` and
  all four periodic checkpoints (`epoch-049/099/149/199.pt`) are on disk.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`),
  external lock `05cfce4cf8db…` with `.external/adr` at the pinned
  `515da0e0373f…`, `config_hash c0a28b33d36f…`. Artifact bytes were not
  re-hashed — this session's sandbox refuses any Bash read outside the repo
  root — so integrity rests on that path-equals-hash correspondence; the
  aggregator owns the real check at M3.
  **This run is an instance of the `ferret-collect` artifact-path defect the
  `trades-s2` entry above diagnoses, and it is worth stating explicitly for
  this arm.** The manifest's artifact `path` fields still point at the
  Ferret-side staging dir
  `runs/adr-campaign-v1-ferret-cifar10_r18_adr-s2/outputs/train/`, and no
  `runs/adr-campaign-v1-ferret-*` directory exists in the local runtime tree at
  all — so `_verify_bundle`'s declared-path lookup
  (`scripts/aggregate_adr_cifar10_replication.py:201`) would abort M3 on this
  run. The bytes are present; only the pointer is stale. With this run counted,
  the defect covers **arm 3's entire seed-2 leg as well as arm 2's and arm 4's**,
  and the fallback that entry proposes (use the bundle's content-addressed copy
  when the declared path is absent, keeping the `sha256` comparison) resolves it
  here too. Left unfixed: this postrun imports nothing and touches no code.
  Contract fields match arm 3 exactly: protocol `controlled_cifar10_r18_adr_v1`,
  student `saad_resnet18_cifar_v1`, normalization `cifar10_raw_identity`,
  method `adr` v1, `teacher: null`, `nesterov: true`, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`,
  `validation_fraction=0.1`, ADR `ema_decay=0.995` `T 2.5→2.0`
  `lambda 0.7→0.95`, train attack KL/rectified 10 steps at `8/255` step
  `2/255` random start, selection attack CE 20 steps at the same budget,
  `world_size=1`, effective global batch 128, `deterministic: true`, seeds all
  2 except the fixed `split=20260722` and `evaluation_attack=0`. `git diff
  cd0b571e4685 HEAD` is empty for `configs/scientific/cifar10_r18_adr.yaml`,
  `cifar10_r18_pgd_at_nesterov.yaml`, `src/ard/protocols/__init__.py`,
  `src/ard/config/schema.py`, `src/ard/objectives/` and `src/ard/policies/`, so
  the contract has not drifted since the pin.
  **ADR stop-rule checks pass.** EMA validation ran every epoch and never
  crashed: all 200 rows carry `val_clean_accuracy_ema` and
  `val_pgd_accuracy_ema`. `best-ema.pt` was selected independently of the
  student — EMA's best validation epoch is **98**, the student's is **103**.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log). All
  rows re-read from `epoch-metrics.jsonl`; the student rows match the manifest
  summary exactly:

  | weights | checkpoint | epoch | clean | PGD |
  |---|---|---|---|---|
  | student | best | 103 | 0.8436 | 0.5434 |
  | student | last | 199 | 0.8638 | 0.5062 |
  | EMA | best-ema | 98 | 0.8148 | 0.5476 |
  | EMA | last | 199 | 0.8642 | 0.5042 |

  **`best-ema.pt` for this seed is a pre-LR-drop checkpoint, and that is worth
  knowing before the EMA evaluation lands.** The first LR drop is at epoch 100.
  Seeds 0 and 1 put their EMA argmax just after it (epoch 104 at 0.5506, epoch
  109 at 0.5498); this seed puts it at epoch 98, *before* it, and its two
  highest EMA epochs (98 at 0.5476, 96 at 0.5470) are both pre-drop. The margin
  is thin: the best post-drop EMA epoch is 104 at 0.5466, i.e. the selection
  crossed the LR drop on **0.10 pp** of validation PGD. It bought that 0.10 pp
  at a cost of **2.92 pp of EMA clean accuracy** (0.8148 at epoch 98 versus
  0.8440 at epoch 104). The selection rule was applied correctly — argmax
  validation PGD is what the contract specifies — but expect this seed's
  "ADR + WA" official-test clean number to sit well below seeds 0 and 1's, and
  do not read that as an ADR property. It is one seed landing on the far side
  of a near-tie.
  **Arm 3 is now complete at three seeds, and so is arm 2. Seed-matched
  validation comparison, directional only:**

  | seed | arm 2 best PGD | arm 3 best PGD | ADR − baseline | arm 2 gap | arm 3 gap |
  |---|---|---|---|---|---|
  | 0 | 0.5184 (ep 102) | 0.5394 (ep 103) | +2.10 pp | 0.0888 | 0.0456 |
  | 1 | 0.5188 (ep 101) | 0.5396 (ep 103) | +2.08 pp | 0.0864 | 0.0408 |
  | 2 | 0.5184 (ep 105) | 0.5434 (ep 103) | +2.50 pp | 0.0882 | 0.0372 |
  | mean | 0.51853 | 0.54080 | **+2.23 pp** | 0.0878 | 0.0412 |

  Every seed-matched pair is positive and the three differences span 0.42 pp.
  Arm 3's own seed spread is 0.40 pp (0.5394–0.5434) and arm 2's is 0.04 pp
  (0.5184–0.5188), so the +2.23 pp mean is far outside either arm's seed noise
  — on this metric. **The preregistered rule is AutoAttack on the official
  10,000-example test set, best checkpoint; every number in that table is
  validation, CE-PGD-20, and the very metric these runs' checkpoint selection
  optimized against, so it flatters the treatment arm. No verdict is
  licensed.** The same table also shows ADR roughly halving the
  robust-overfitting gap at ResNet-18, 8.78 pp → 4.12 pp on three seeds each,
  with within-arm spreads of 0.24 pp and 0.84 pp.
  **Correction to the `mobilenetv2_pgd_at-s0` entry, two ways.** That entry
  quoted the ResNet-18 leg as "+2.06 to +2.12 pp"; those are the *cross*-paired
  differences (arm 3 seed 0 against arm 2 seed 1 and vice versa). Seed-matched
  they are +2.10 and +2.08, and seed 2 now adds +2.50.
  More substantially, that entry concluded from a single unmatched pair (ADR
  seed 1's 0.76 pp gap against baseline seed 0's 0.62 pp) that MobileNetV2's
  tiny robust-overfitting gap "is a property of MobileNetV2 … not an ADR
  effect". Arms 7 and 8 now have enough terminal seeds to check that pairing
  properly, and the comparison it rested on was inside the baseline's own seed
  noise: arm 7's three gaps are 0.62 / 1.32 / 1.18 pp (seeds 0/1/2), a 0.70 pp
  spread, larger than the 0.14 pp difference the conclusion was drawn from.
  Seed-matched, ADR's gap is the *smaller* one in both available pairs (seed 1
  0.76 vs 1.32 pp, seed 2 0.68 vs 1.18 pp). The claim that survives is weaker
  and in both directions: **MobileNetV2 at this capacity and horizon barely
  robustly overfits at all** (≈0.6–1.3 pp either arm, against 3.7–8.9 pp at
  ResNet-18), and at that scale the ADR-versus-baseline difference (≈0.5 pp) is
  the same size as the baseline's own seed spread, so **no attribution is
  licensed at MobileNetV2 in either direction**. The ResNet-18 halving above,
  three seeds per arm, is the finding that is actually supported.
  **The MobileNetV2 leg is now seed-matched too, which is what the previous
  entry asked for.** Arm 8 minus arm 7 at each seed's own best checkpoint,
  validation PGD: seed 1 0.4850 − 0.4502 = **+3.48 pp**, seed 2
  0.4830 − 0.4494 = **+3.36 pp** (the previous entry's +3.54 pp was arm 8 seed 1
  against arm 7 seed 0, unmatched). Both exceed the ResNet-18 mean of +2.23 pp
  and every individual ResNet-18 pair, which is the "sign confirmed" direction
  of the preregistered rule — **on the wrong metric, the wrong split and two
  seeds, so it is a shape, not a result**. Arm 8 seed 0 is the one training run
  that would make this leg three-seed complete.
  **Budget — the ADR cost line holds on Hamster and does not transfer to
  Ferret, and this run independently confirms the `trades-s2` entry's diagnosis
  that the cause is contention, not a slower GPU.** The plan charges ADR-family
  runs ≈54 s per full epoch (≈3.0 GPU-h), confirmed for ResNet-18 on Hamster
  (`adr-s1` 39.4 s, `adr-s0` 40.7 s training-loop, ≈1140 img/s). This run's
  training loop sat at **90.7–101.9 s/epoch** at **441.8–496.1 img/s** for
  almost every epoch, and its wall clock was 2026-09-09T18:09:29.3Z →
  2026-09-10T01:06:03.6Z, **6 h 56 min 34 s** (125.0 s/epoch averaged, ≈6.9
  GPU-h as executed) — roughly 2.3× Hamster's per-epoch cost for identical
  work. **But six epochs ran far faster, and epoch 22 hit 1048.4 img/s**
  (epoch 37: 987.1), within 8 % of Hamster's uncontended figure for the same
  arm. That is the same signature the `trades-s2` entry found on a different
  arm (its epoch 97 at 1026 img/s), from a different Ferret GPU, so the ≈2.3×
  penalty is host contention while multiple training jobs shared Ferret — not
  hardware, and not the workspace registry's stale per-GPU figures
  (`configs/workspace/ard_workspace_v1.json` lists Ferret at 599.65 / 607.05 /
  424.99 img/s and Hamster at 679.1; both are well below what either host
  actually delivers uncontended). **Budget Ferret-executed ADR runs at ≈2× the
  table's Hamster line at 3-way concurrency, or run two jobs per host.**
  Contention changes only wall clock, not results: `deterministic: true`, fixed
  seeds and fixed batch size make the numbers independent of throughput.
  Campaign state at this postrun's own scan (`--include-hand-run`, fresh unused
  cursor over the campaign root), taken a few minutes after the `trades-s2`
  entry's 08:59Z scan and superseding its evaluation counts. **18 of 20 run
  dirs, and all 18 trainings terminal and successful** — unchanged. Missing
  entirely: **`cifar10_r18_trades_adr-s2` and `cifar10_mobilenetv2_adr-s0`**,
  which have no run dir on either host. Evaluations terminal: **7**
  (`pgd_at-s1/-s2`, `pgd_at_nesterov-s0/-s1/-s2`, `adr-s0` model, `adr-s0`
  EMA) — one more than the entry above, because `pgd_at_nesterov-s2`'s
  evaluation finished at 08:56Z — of which exactly **one** is an EMA-weights
  evaluation. **Eight evaluations are in flight** — `adr-s1`, `adr-s2`,
  `trades_adr-s0/-s1`, `trades_49k_validation-s0`, `mobilenetv2_adr-s1`,
  `mobilenetv2_pgd_at-s1/-s2`, the last two having started since that scan —
  against five busy GPUs (Hamster 2 × 100 %, Ferret 3 × ~100 %), so three are
  queued rather than computing. This run's own contract evaluation
  (`eval-5709dc2e3701675af39e`) is among them. **Four** terminal training runs
  now have no evaluation of any kind: `trades-s1`, `trades-s2`,
  `mobilenetv2_adr-s2` and `mobilenetv2_pgd_at-s0`.
  **Gap to flag for the human: the "ADR + WA" comparator is missing for eight
  of the nine ADR-family runs.** `evaluation-ema/` exists only for `adr-s0`; no
  in-flight job is an EMA evaluation. The frozen contract requires
  `--weights=ema` against `best-ema.pt`/`last.pt` for every `adr` and
  `adr_trades` run, so arms 3, 5 and 8 each still need EMA evaluations before
  M2, on top of the two unlaunched trainings. Launching anything, and queueing
  those, is a decision for the human — this postrun launches nothing. M1b's
  archive is committed (see the entry above) but M1c stays unticked: two
  trainings and most evaluations are outstanding.

- **2026-09-10, postrun of `eval-3bfb6a2f65ef77e308e1` — arm 2
  (`pgd_at_nesterov`) seed 2 evaluation. Arm 2 is now the campaign's first arm
  with all three contract evaluations complete.** Terminal status re-derived
  from the run bundle the watcher pointed at
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_pgd_at_nesterov-s2/train/
  evaluation/run-bundle/manifest.json`): `terminal: true`, `success: true`,
  `failure_class: null`, `status: completed`. `completion.json` reads
  `{"status": "completed", "results": 2}` and `error-marker.txt` reads "no
  application error recorded". The manifest's own `status` field says
  `sync_pending`, which is the offline-W&B sync state and not a completion
  signal — the bundle markers are. **Nothing imported.** The campaign is still
  mid-flight (6 of 20 contract evaluations terminal), so there is no
  campaign-level result to aggregate; `docs/experiments/` has no record for
  this contract and none was created.

  Every declared `expected_output` is present: all seven manifest artifacts
  resolve under `run-bundle/artifacts/`, and the flat copies the M3 aggregator
  actually reads (`train/evaluation/evaluation-results.json` and siblings)
  exist in the local run root with identical numbers to the bundle artifact
  (checked field by field: both `checkpoint_alias` rows, `count: 10000`,
  `split: "test"`, all six accuracies). **The aggregator's directory-naming
  defect fixed in the previous entry is confirmed to resolve this run**:
  `dir_prefix "cifar10_r18_"` + `pgd_at_nesterov` + `-s2` is the real directory,
  and `_load_arm_seed`'s `run_dir / "evaluation" / "evaluation-results.json"`
  exists for all three arm-2 seeds. No new M3 blocker.

  AutoAttack is the **standard** version with real provenance
  (`expected_commit == vcs_commit == a39220048b3c9f2cca9a4d3a54604793c68eca7e`),
  run in its own process at `epsilon=8/255`, `Linf`, batch 128, seed 0.
  `weights: model` and `selection_weights: model` on both rows. Training seeds
  all 2 except the fixed `split=20260722` and `evaluation_attack=0`. Source SHA
  `cd0b571e4685`, `dirty: false`, empty diff.
  **Checkpoint SHA-256 was not independently recomputed this session** — the
  sandbox refuses `sha256sum` outside the repo working directory, so the
  `checkpoint_sha256` values below are the evaluation record's own declared
  values, not a second measurement. Earlier entries in this plan did recompute
  such hashes; this one could not, and says so rather than implying it did.

  **Official CIFAR-10 test set (10,000 examples), student weights, arm 2
  (`pgd_at_nesterov`) seed 2 — clean, CE-PGD-20 and AutoAttack reported
  separately, best and last kept separate:**

  | checkpoint | sha256 (short) | clean | CE-PGD-20 | AutoAttack |
  |---|---|---|---|---|
  | `best.pt` (val epoch 105) | `d3372ac5ff6b…` | 0.8267 | 0.5086 | 0.4737 |
  | `last.pt` (epoch 199) | `418a8ef34b65…` | 0.8415 | 0.4254 | 0.4082 |

  The selected epoch is read directly from the training bundle's
  `metrics.jsonl`: `val_pgd_accuracy` peaks at 0.5184 on epoch 105
  (`val_clean_accuracy` 0.8424) and no other epoch in the run reaches 0.51 at
  all. The runner-up is 0.5128 on epoch 103, so the peak wins by 0.56 pp —
  a real peak, not the near-arbitrary plateau argmax seen on the MobileNetV2
  baseline (0.28 pp). Validation then decays to 0.4302 by epoch 199, an 8.82 pp
  fall that tracks the 8.32 pp test CE-PGD-20 gap below, the same
  validation-tracks-test pattern seed 0 showed (8.88 vs 8.44 pp).
  **This corrects the "101–104 window" claim by one epoch.** Selected epochs
  for the ResNet-18 PGD-AT/ADR readings are now 101, 102, 103, 103, 104 and
  **105**; the window is 101–105, still tight and still just past the first LR
  drop at epoch 100, but the earlier entry's "four-epoch window" is now a
  five-epoch one. Note also that all three arm-2 seeds peak at essentially the
  same validation height — 0.5184 (s0), 0.5188 (s1), 0.5184 (s2) — while
  disagreeing about which epoch gets there. Selection is picking equivalent
  models from different points in the same post-LR-drop ridge.

  **(a) Arm 2's three-seed spread, all same source SHA `cd0b571e4685`, same
  protocol, same evaluation identity, only `ARD_SEED` differing.** This is the
  campaign's first complete three-seed arm and the first same-SHA three-seed
  spread available for the ResNet-18 leg of the preregistered rule.

  | quantity | seed 0 | seed 1 | seed 2 | mean | range |
  |---|---|---|---|---|---|
  | best clean | 0.8232 | 0.8219 | 0.8267 | 0.8239 | 0.48 pp |
  | best CE-PGD-20 | 0.5081 | 0.5097 | 0.5086 | 0.5088 | 0.16 pp |
  | **best AutoAttack** | 0.4719 | 0.4741 | 0.4737 | **0.4732** | **0.22 pp** |
  | last clean | 0.8424 | 0.8406 | 0.8415 | 0.8415 | 0.18 pp |
  | last CE-PGD-20 | 0.4237 | 0.4196 | 0.4254 | 0.4229 | 0.58 pp |
  | last AutoAttack | 0.4057 | 0.4041 | 0.4082 | 0.4060 | 0.41 pp |

  The third seed did **not** widen the best-AutoAttack range: seed 2's 0.4737
  lands between the two existing seeds, leaving the range at 0.22 pp with all
  three points in. The previous entry set the condition "quote arm 1's 0.85 pp
  as the campaign's conservative noise floor until arm 2 has its third seed",
  and that condition is now met. The two floors measure different things and
  both should be kept: **arm 2's 0.22 pp is seed noise alone** (three runs, one
  SHA, one protocol), while **arm 1's 0.85 pp bundles seed noise with SHA
  drift** (its seed 0 is the reused historical run at a different SHA), so it is
  an upper bound on total run-to-run variation rather than a measurement of
  seed spread. The decision rule compares arm 3 against arm 2, so arm 2's own
  0.22 pp is the directly relevant number. Neither is a population estimate:
  a three-point range is still a lower bound on the true spread, and the stop
  rules forbid a fourth seed without a new decision packet, so it will not be
  tightened further in this campaign.

  **(b) The ADR-vs-matched-baseline gap now reads against a complete baseline,
  and the conclusion does not depend on which floor is used.** Arm 3 (`adr`)
  seed 0's contract `best.pt` AutoAttack is 0.4875. Against arm 2's three-seed
  mean 0.4732 that is **+1.43 pp**; against arm 2's strongest seed (seed 1,
  0.4741) **+1.34 pp**; against its weakest (seed 0, 0.4719) **+1.56 pp**. The
  whole +1.34/+1.56 pp band sits above arm 1's conservative 0.85 pp floor and
  roughly 6× above arm 2's own three-seed 0.22 pp range. **This is still not
  the preregistered result.** Arm 3 has one contract evaluation to arm 2's
  three, and the primary MobileNetV2 leg has none at all; the rule needs three
  seeds on both sides of both legs. What changed is only that the ResNet-18
  baseline side is finished, so the remaining uncertainty on that leg is
  entirely on the ADR side — and `adr-s1` and `adr-s2` evaluations are running
  as of this entry.

  **(c) Robust-overfitting suppression is unchanged and remains the sturdier
  claim, now with a complete baseline band.** This run's best-minus-last gap is
  **8.32 pp** on CE-PGD-20 and **6.55 pp** on AutoAttack. Arm 2's three seeds
  therefore span **6.55–7.00 pp** on AutoAttack (mean 6.72 pp) and **8.32–9.01
  pp** on CE-PGD-20 (mean 8.59 pp). Seed 2 sets the low end of the AutoAttack
  band without breaking it. Arm 3 seed 0's 3.94 pp remains 2.61 pp below the
  nearest baseline reading. As established in the arm-7 entry, this is a
  ResNet-18 claim only — MobileNetV2 barely robustly overfits at this horizon
  regardless of method.

  **`environment.json`'s kernel string is a reliable host discriminator for
  this campaign, and it settles where this evaluation ran.** Hamster reports
  `Linux-7.0.0-31-generic`, Ferret `Linux-7.0.0-28-generic`. Arm 2 seed 0's
  evaluation bundle carries -31 (Hamster, as logged); this run's *training*
  bundle carries -28 (Ferret, as logged) and so does its *evaluation* bundle —
  confirming the previous entry's statement that the `-s2` evaluation was
  dispatched to Ferret. Worth reusing: every bundle in this campaign can be
  attributed to a host from a field it already records, without asking either
  machine.

  **Budget finding — Ferret's three-way packing costs more aggregate throughput
  than it buys, and the plan's non-ADR training line holds only on Hamster.**
  Training ran 2026-09-09T12:14:47.5Z → 18:08:46.1Z, **5 h 53 min 58.5 s** for
  200 epochs = **106.2 s per full epoch** (exact, from the manifest's own
  timestamps), i.e. **≈5.9 GPU-h as executed** against the budget table's 2.6
  GPU-h line for arm 2. Sampled per-epoch training-loop times (`train_seconds`,
  16 of 200 epochs at both ends of the run) fall in **80.7–96.5 s** and do not
  trend down over the run, against **38.6 s/epoch at 1167 img/s** already
  recorded on Hamster for *this same arm* at seed 1 — a 2.1–2.5× slowdown on
  identical hardware (both hosts are RTX 4090s). Taking the exact wall-clock
  figures: Ferret running three jobs at once delivers 3 runs / 5.9 h ≈ **0.51
  runs/h**, while Hamster running two at once delivers 2 runs / ≈2.6 h ≈ **0.77
  runs/h**. Ferret has 50% more GPUs and produces about a third *less*
  finished work per hour. That retroactively supports the previous entry's
  decision to pull `mobilenetv2_pgd_at` seeds 1/2 off Ferret's queue onto idle
  Hamster GPUs, and it argues against three-way packing on Ferret in future
  campaigns. **Caveat**: Ferret is a shared lab host and other users' load is
  not observable from these bundles, so "three-way packing" is the most likely
  cause of the slowdown, not a demonstrated one. Contention changes only wall
  clock, never results — `deterministic: true`, fixed seeds and fixed batch
  size make the numbers independent of throughput.
  The *evaluation* half shows the opposite pattern and sharpens the guess. This
  run's evaluation took 21:33:03.0Z → 23:23:32.6Z, **1 h 50 min 29.6 s** on
  Ferret, against **1 h 48 min** for seed 0's evaluation on Hamster — within
  2 minutes of each other for the same two-checkpoint × (clean + CE-PGD-20 +
  AutoAttack) workload. Evaluation is essentially host-insensitive here while
  training is 2.3× slower, which points at dataloader or CPU contention during
  training rather than GPU saturation. The two evaluations' concurrency levels
  are not established from these bundles, so this is a hypothesis worth one
  cheap measurement, not a finding.

  **Campaign state at the 09:0xZ scan** (`--include-hand-run`, fresh unused
  cursor over the campaign root). **18 of 20 training run dirs**, and every one
  of the 18 is terminal and successful — up from 10 at the previous scan:
  `pgd_at-s1/-s2`, `pgd_at_nesterov-s0/-s1/-s2`, `adr-s0/-s1/-s2`,
  `trades-s1/-s2`, `trades_adr-s0/-s1`, `trades_49k_validation-s0`,
  `mobilenetv2_pgd_at-s0/-s1/-s2`, `mobilenetv2_adr-s1/-s2`. The two absent
  from the local root are **`trades_adr-s2`** and **`mobilenetv2_adr-s0`**,
  both of which the previous entry listed as still running on Ferret.
  Contract (model-weights) evaluations **6 of 20** terminal —
  `pgd_at_nesterov-s0/-s1/-s2`, `adr-s0`, `pgd_at-s1`, `pgd_at-s2`;
  EMA-weights evaluations **1 of 9** (`adr-s0`). **Six evaluations are in
  flight right now**: `adr-s1`, `adr-s2`, `trades_adr-s0`, `trades_adr-s1`,
  `trades_49k_validation-s0`, `mobilenetv2_adr-s1`. All five GPUs across both
  hosts are busy (Hamster 4876/3924 MiB at 100%, Ferret 6773/8171/9928 MiB at
  99%), consistent with five of those six running and one queued. The
  evaluation queue that had drained at the previous entry is full again, and
  this postrun launched nothing.
  M1c stays unticked (18 of 20 run dirs). M2 stays unticked (14 of 20 contract
  evaluations outstanding, and the `trades_adr` and `trades_49k_validation`
  arms are still unchecked against any expectation). M3 unopened.

- 2026-09-10: postrun of the
  `adr-campaign-v1-ferret-cifar10_r18_trades_adr-s1` **training** terminal
  event (`runs/adr-cifar10-campaign-v1/cifar10_r18_trades_adr-s1/train/
  run-bundle/manifest.json`) — **arm 5 seed 1, the campaign's second
  ADR-TRADES run and therefore the first cross-seed reading of any ADR-TRADES
  cell**. **Nothing imported — no milestone closed.** This is training only:
  the run has no terminal `evaluation/` or `evaluation-ema/` output, so it
  produces no official-test number, and `docs/experiments/` still holds only
  the `historical/` reuse bundles archived at M1b, so the idempotency check
  had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of the campaign root returns
  `terminal: true`, `success: true`, `status: completed`, `failure_class:
  null` on exactly the `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and `epoch-metrics.jsonl` really carries 200 rows ending at epoch
  199 / `global_step` 70400. Both declared artifacts exist at their
  content-addressed `run-bundle/artifacts/<name>/<sha256>/` paths and each
  directory name equals the manifest's own `sha256` for that artifact
  (`epoch-metrics.parquet` `7acd25981887…`, `sample-stats-train.parquet`
  `379255e25d37…`); `config_hash` `0c3852c48802…`. All seven checkpoints the
  contract requires are on disk: `best.pt`, `best-ema.pt`, `last.pt` and
  `epoch-{049,099,149,199}.pt`. Source SHA `cd0b571e4685…` clean
  (`dirty: false`, empty-diff `e3b0c442…`), external lock `05cfce4cf8db…`
  with `.external/adr` at the pinned `515da0e0373f…`. Artifact bytes were not
  re-hashed — this session's sandbox refuses to run a hash over paths outside
  the repo root — so integrity rests on that path-equals-hash correspondence;
  the aggregator owns the real check at M3.
  **Ferret provenance.** This bundle was `ferret-collect`ed: the manifest's
  artifact `path` fields and `completion.json`'s `output_dir` still name
  Ferret's own run root (`runs/adr-campaign-v1-ferret-cifar10_r18_trades_adr-
  s1/outputs/train/`), not the canonical local dir the bundle now sits in.
  The canonical local copies of both artifacts are present alongside the
  content-addressed ones, so nothing is missing; the stale absolute paths are
  a record of where the job ran, not a broken reference. Consistent with a
  Ferret job, `tracking.mode` resolved to `offline_sync` and `wandb_url` is
  null with `status: sync_pending` — the W&B run has not been synced yet.
  Contract fields match arm 5 exactly: protocol
  `controlled_cifar10_r18_adr_v1`, student `saad_resnet18_cifar_v1` with
  `cifar10_raw_identity` normalization, method `adr_trades` v1,
  `teacher: null`, `trades_beta=6.0`, ADR `ema_decay=0.995` `T 2.5→2.0`
  `lambda 0.7→0.95`, `nesterov: true`, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`,
  `validation_fraction=0.1`, train attack KL/`kl_target=student_clean`
  10 steps at `8/255` step `2/255` random start, selection attack CE 20 steps
  at the same budget, `world_size=1`, effective global batch 128,
  `deterministic: true`, seeds all 1 except the fixed `split=20260722` and
  `evaluation_attack=0`. `git diff cd0b571e4685 HEAD` is empty for
  `configs/scientific/cifar10_r18_trades_adr.yaml`,
  `configs/scientific/cifar10_r18_trades.yaml`, `src/ard/protocols/__init__.py`
  and `src/ard/config/schema.py`, so the contract has not drifted since the
  pin. The 45,000/5,000 split is confirmed arithmetically: epoch 0 reports
  409.74 img/s over 109.83 s, i.e. 45,000 images.
  **ADR stop-rule checks pass.** EMA validation ran every epoch and never
  crashed: all 200 rows carry `val_clean_accuracy_ema` and
  `val_pgd_accuracy_ema`. And `best-ema.pt` was selected independently of the
  student — the EMA's best validation epoch is **121**, the student's is
  **116** — which is what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section
  requires.
  Held-out **validation** diagnostics only (5,000 held-out *training* images,
  CE-PGD-20 — not the official test set, not AutoAttack, not reportable, and
  not comparable to any official-test table in this log):

  | weights | checkpoint | epoch | clean | PGD |
  |---|---|---|---|---|
  | student | best | 116 | 0.8388 | 0.5388 |
  | student | last | 199 | 0.8504 | 0.5262 |
  | EMA | best-ema | 121 | 0.8442 | 0.5500 |
  | EMA | last | 199 | 0.8546 | 0.5252 |

  All four rows re-read from `epoch-metrics.jsonl`; the two student rows match
  the manifest summary exactly. **The EMA branch is again not redundant for
  this arm**: at its own best epoch the EMA is +1.12 pp of validation PGD
  above the student's selected best, the same direction the 00:30Z arm-5
  seed-0 entry found by sampling (+0.44 pp at epoch 149). So the
  `evaluation-ema/` pass this seed still owes is a real measurement.
  **First cross-seed reading of arm 5, and it is stable.** Against seed 0
  (00:30Z entry, same validation protocol):

  | seed | best epoch | best clean | best PGD | last clean | last PGD | gap |
  |---|---|---|---|---|---|---|
  | 0 | 150 | 0.8474 | 0.5442 | 0.8488 | 0.5346 | 0.0096 |
  | 1 | 116 | 0.8388 | 0.5388 | 0.8504 | 0.5262 | 0.0126 |

  Seed-to-seed spread is 0.54 pp of validation PGD and 0.86 pp of clean at the
  best checkpoint — inside the control-vs-control noise floor this project has
  already measured elsewhere, so nothing here looks anomalous. Both seeds are
  flat over the late window (`val_pgd_slope_epoch_120_199` −4.19e-05 for seed 1,
  +5.03e-05 for seed 0, i.e. ≈0.4 pp either way across 80 epochs), and both
  robust-overfitting gaps (1.26 pp, 0.96 pp) are far below arm 4's `trades-s1`
  reading of 3.20 pp. **This licenses no claim.** It is validation, not the
  official test; two seeds give a directional verdict for those two seeds
  only; and arms 4 and 5 still differ in optimizer as well as method (plain
  SGD vs Nesterov), the confound this plan already records. The decisive
  ResNet-18 leg remains arm 3 vs arm 2.
  **The "peaks just past the first LR drop" pattern is an objective effect,
  not an ADR effect.** Earlier entries noted that every ResNet-18 arm measured
  peaked at epoch 101-104, just past the first drop at 100. That holds for
  arms 1-3 (the PGD-AT objective, with and without ADR). Both TRADES-objective
  arms peak much later and much more variably — arm 4 `trades-s1` at 157, arm
  5 at 150 (seed 0) and 116 (seed 1) — so the late peak tracks the TRADES
  objective across both the plain and the ADR variant, and is not evidence
  about ADR. Checkpoint selection is unaffected either way: `best.pt` is
  chosen by validation PGD, not by epoch.
  **No usable cost reading from this run.** Unlike arm 5 seed 0, this run was
  contended from epoch 0 to epoch 199 — throughput sits at 398-438 img/s
  across all 15 epochs sampled (0-3, 48-50, 147-150, 196-199), against the
  974-981 img/s seed 0 measured uncontended, i.e. roughly 40-45 % of the
  uncontended rate for the whole run. Wall clock was 18:32:53.2Z →
  02:15:04.4Z, **7 h 42 min 11.2 s** (138.7 s/epoch averaged). The sampled
  training loop averages ≈106.9 s/epoch, which puts ≈32 s/epoch outside the
  loop — consistent with this arm running two validation passes per epoch
  (student and EMA), and therefore consistent with the seed-0 entry's
  suspicion that the ≈3.0 GPU-h ADR budget line is too small for arm 5 — but
  that ≈32 s is itself contended and estimated from 15 of 200 epochs, so it is
  not a budget number and the budget line is left unchanged. Contention
  changes only wall clock, not results: `deterministic: true`, fixed seeds and
  fixed batch size make the numbers independent of throughput.
  **The M3 aggregator directory-naming defect does not apply to this run.**
  The `dir_prefix` fix committed in `0838d20` resolves arm 5 to
  `cifar10_r18_trades_adr-s{0,1,2}`, which is exactly this run's directory,
  and the arm carries `has_ema: True`, matching the `best-ema.pt` on disk. The
  eight instances logged before that commit are closed; no new one here.
  Campaign state at the 09:02Z scan (`--include-hand-run`, fresh unused
  cursor — 34 bundles, 28 terminal): **18 of 20 training runs terminal and
  successful**, unchanged from the M1b entry above — still outstanding are
  `cifar10_mobilenetv2_adr-s0` and `cifar10_r18_trades_adr-s2` (arm 5's third
  seed), neither with a local run dir. Contract (model-weights) evaluations
  **6 of 20** terminal (`r18_pgd_at-s1/-s2`, `r18_pgd_at_nesterov-s0/-s1/-s2`,
  `r18_adr-s0`) with **6 more running** — `r18_adr-s1`, `r18_adr-s2`,
  `r18_trades_49k_validation-s0`, `r18_trades_adr-s0`,
  `mobilenetv2_adr-s1` and **`r18_trades_adr-s1`, this run's own evaluation**.
  EMA-weights evaluations **1 of 9** terminal (`r18_adr-s0`). **This run's
  evaluation is already in flight — do not launch or retry it.** This postrun
  launches nothing, and the two missing training runs plus the still-unqueued
  EMA evaluations remain the human's call. M1c stays unticked.

- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_mobilenetv2_pgd_at-s1`
  **training** terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_mobilenetv2_pgd_at-s1/train/run-bundle/manifest.json`) — **arm 7
  seed 1, the observation the previous entry named as one of the two cheapest
  ways to make the MobileNetV2 leg seed-matched**. **Nothing imported — no
  milestone closed.** Training only: the run has no `evaluation/` output, so
  it produces no official-test number, and `docs/experiments/` still holds
  only `historical/`, so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `status: completed`, `failure_class: null` on exactly the
  `--state-path` line the watcher passed.
  Verified for this bundle: `run-bundle/completion.json` `{"status":
  "completed"}`, `manifest.status=sync_pending`, `error_marker` reads "no
  application error recorded", `epoch_metrics_complete: true` with 200 of 200
  expected epochs, and `epoch-metrics.jsonl` really carries 200 rows covering
  epochs 0–199 with no gap, ending at `global_step` 70400 (= 200 × 352
  steps, and 352 = ceil(45000/128), with `train_valid_examples` 45000
  confirming the 10% validation split). Both declared artifacts exist at
  their content-addressed `run-bundle/artifacts/<name>/<sha256>/` paths, each
  directory name equalling the manifest's own `sha256` for that artifact
  (`epoch-metrics.parquet` `e8713f4b9de1…`, `sample-stats-train.parquet`
  `c6f316c62695…`). `best.pt`, `last.pt` and all four periodic checkpoints
  (`epoch-049/099/149/199.pt`) are on disk; there is **no** `best-ema.pt`,
  correct because `method.adr` is `null` for this arm. Source SHA
  `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`) from
  worktree `source-cd0b571e4685`, external lock `05cfce4cf8db…` (identical to
  seed 0's), `config_hash d8e8ad98277c…` (differs from seed 0's
  `fac5b4eb06c4…` only because the seeds differ). As in the seed-0 entry,
  artifact bytes were not re-hashed — this session's sandbox refuses any Bash
  read outside the repo root — so integrity rests on that path-equals-hash
  correspondence; the aggregator owns the real check at M3.
  Contract fields match arm 7 exactly: protocol
  `controlled_cifar10_mobilenetv2_adr_v1`, student `mobilenet_v2_cifar` with
  `normalization: {profile: cifar10_standard}`, method `pgd_at` v1,
  `teacher: null`, `adr: null`, `nesterov: true`, `epochs=200`,
  `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9 wd=5e-4`,
  `validation_fraction=0.1`, train attack CE 10 steps at `8/255` step
  `2/255` random start, selection attack CE 20 steps at the same budget,
  `world_size=1`, effective global batch 128, `deterministic: true`. Seeds
  are all **1** (`model_init`, `data_order`, `augmentation`, `train_attack`,
  `qualitative_panel`) with `split=20260722` and `evaluation_attack=0` fixed
  — `evaluation_attack` is deliberately outside the `ARD_SEED` set in the
  frozen contract, so holding it at 0 across seeds is correct and keeps
  evaluation identical between seeds. `git diff cd0b571e4685 HEAD` is empty
  for `configs/scientific/cifar10_mobilenetv2_{pgd_at,adr}.yaml` and for
  `src/ard/protocols/__init__.py` and `src/ard/config/schema.py`, so the
  contract has not drifted since the pin. The LR schedule is visible in the
  data exactly where it should be (validation PGD 0.3458→0.4190 across epoch
  99→100, 0.4268→0.4460 across 149→150; final `learning_rate` 0.001). No
  NaN, no loss blow-up, throughput steady at 783–811 img/s, train-minus-
  validation gap at epoch 199 a modest 3.75 pp clean / 6.54 pp robust. **No
  bug found.**
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log).
  Both rows re-read from `epoch-metrics.jsonl` and they match the manifest
  summary exactly:

  | checkpoint | epoch | clean | PGD |
  |---|---|---|---|
  | best | 173 | 0.7458 | 0.4502 |
  | last | 199 | 0.7424 | 0.4370 |

  **Arm 7 is the campaign's first complete arm (3/3 training seeds).** Seed 2
  also went terminal in this window (05:09:41.62Z → 08:53:26.06Z, best 0.4494
  at epoch 151, last 0.4376) and has its own pending terminal event; it is
  quoted here only as context and is **not** postrun-imported by this entry.
  Arm 8 is now at 2/3 (seed 2 collected from Ferret: best 0.4830 at epoch
  158, last 0.4762). Validation PGD, best checkpoint, per seed:

  | arm | s0 | s1 | s2 | mean |
  |---|---|---|---|---|
  | 7 `mobilenetv2_pgd_at` | 0.4496 | 0.4502 | 0.4494 | 0.4497 |
  | 8 `mobilenetv2_adr` | *running* | 0.4850 | 0.4830 | 0.4840 (2 seeds) |

  **The previous entry's robust-overfitting conclusion was really two claims,
  and they now come apart — separating them is the main finding of this
  postrun.** That entry compared arm 7 seed 0's best-minus-last gap (0.62 pp)
  against arm 8 seed 1's (0.76 pp) and concluded that "MobileNetV2 at 200
  epochs simply does not robustly overfit much — the suppression claim is a
  ResNet-18 result, not a general one".
  **The architecture half survives and strengthens.** All five MobileNetV2
  readings now available span 0.62–1.32 pp, against 4.08–4.56 pp for ADR and
  8.64–8.88 pp for the Nesterov-matched baseline at ResNet-18. Not one
  MobileNetV2 run comes within a factor of three of the *smallest* ResNet-18
  reading, so "MobileNetV2 at this capacity and horizon barely robustly
  overfits, and ADR's gap-halving is a ResNet-18 result" is well supported.
  **The arm-versus-arm half does not survive.** With three arm-7 seeds the
  within-arm spread of that statistic is **0.70 pp** (0.62 / 1.32 / 1.18,
  mean 1.04), five times the 0.14 pp difference the ordering rested on. Arm
  8's two readings are 0.76 and 0.68 (mean 0.72) and sit *inside* arm 7's
  range. The ordering in fact reverses on the means — ADR's gap is smaller,
  and much more tightly clustered — but that difference (0.32 pp) is itself
  inside the baseline's seed spread. **Neither "ADR fails to suppress robust
  overfitting at MobileNetV2" nor its reversal is licensed**: at this capacity
  best-minus-last is too noisy to separate the two arms at all.
  **The mechanism is identifiable, and it is the `last` endpoint, not the
  arms.** Decomposing each gap into (best − late-plateau mean over epochs
  150–199) + (late mean − last):

  | run | best − late mean | late mean − last | gap |
  |---|---|---|---|
  | 7 s0 | +0.77 pp | **−0.15 pp** | 0.62 pp |
  | 7 s1 | +0.62 pp | +0.70 pp | 1.32 pp |
  | 7 s2 | +0.56 pp | +0.62 pp | 1.18 pp |
  | 8 s1 | +0.48 pp | +0.28 pp | 0.76 pp |
  | 8 s2 | +0.55 pp | +0.14 pp | 0.68 pp |

  The first column — the argmax's selection inflation — is stable at
  0.48–0.77 pp across both arms and all five runs. All of the between-run
  variation lives in the second column, i.e. in where a single epoch-199
  happened to land inside a flat plateau. Arm 7 seed 0's small gap was simply
  a lucky endpoint: its epoch 199 finished *above* its own late-plateau mean,
  the only one of the five to do so. **Use best-minus-late-mean, not
  best-minus-last, whenever robust overfitting is discussed for this
  campaign**; the manifest already emits `val_pgd_late_mean_epoch_150_199`,
  so the aggregator needs no change to report it.
  **This run's own argmax is even weaker than seed 0's, in the same way.**
  Validation PGD over epochs 150–199 is a flat, noisy plateau spanning
  0.4354 (epoch 170) to 0.4502 (epoch 173) — a 1.48 pp band. The winning
  0.4502 beats the runner-up (0.4496, epoch 162) by **0.06 pp**, and
  **14** separate epochs sit within 0.42 pp of the top (150, 158, 159, 161,
  162, 164, 168, 169, 173, 174, 181, 185, 188, 189). Meanwhile epoch 199
  lands at 0.4370, the third-lowest of the 50 plateau epochs. Selection here
  is arbitrary within the plateau in both directions.
  **The previous entry's "selected epoch tracks method and architecture"
  reading also needs withdrawing.** It offered "R18 PGD-AT/ADR 101–104, R18
  TRADES 157, MobileNetV2 ADR 151, MobileNetV2 PGD-AT 175" as a pattern.
  Within arm 7 alone the selected epoch is 175 / 173 / 151 across seeds, and
  within arm 8 it is 151 / 158 — so 151 and 175 are the *same* arm. Selected
  epoch on this architecture is plateau argmax noise, not an arm signature.
  **Directional read of the primary leg, now seed-matched.** The previous
  entry's +3.54 pp was a cross-seed estimate (arm 8 s1 against arm 7 s0). The
  seed-matched values are **+3.48 pp** at seed 1 (0.4850 − 0.4502) and
  **+3.36 pp** at seed 2 (0.4830 − 0.4494), so the cross-seed shortcut was
  accurate to 0.06 pp for this quantity — arm 7's best-checkpoint validation
  PGD is remarkably stable across seeds (0.08 pp total spread), even though
  its *gap* statistic is not. Arm means give **+3.43 pp** (0.4840 over 2 ADR
  seeds versus 0.4497 over 3 baseline seeds), or **+3.57 pp** on the
  less selection-contaminated late-plateau mean (0.4789 versus 0.4432), with
  clean accuracy **+2.26 pp** (0.7664 versus 0.7438). Against ResNet-18's
  +2.06 to +2.12 pp, MobileNetV2's ADR gain still looks larger. **The sign is
  confirmed in the direction the preregistered rule cares about, but no
  verdict is licensed**: the rule is **AutoAttack on the official
  10,000-example test set, best checkpoint, three seeds per arm**, and these
  numbers are validation not test, CE-PGD-20 not AutoAttack, two seeds not
  three on the ADR side, and measured with the very metric these runs'
  checkpoint selection optimised against — which flatters both arms.
  **Budget: arm 7's cost line is now measured, not estimated.** Seeds 1 and 2
  started 57 ms apart (05:09:41.62Z and 05:09:41.68Z) on Hamster's two
  otherwise-idle GPUs and finished 28 s apart, so this is a clean
  one-job-per-4090 pair. Seed 1 ran 05:09:41.68Z → 08:53:54.56Z =
  **3 h 44 min 12.9 s** (67.26 s/epoch); seed 2 ran 3 h 43 min 44.4 s
  (67.12 s/epoch). Per-epoch training-loop time is very tight — 200 values
  spanning 55.28–58.78 s, mean ≈56.3 s, with only the epoch-6..17 stretch
  above 57.6 s — leaving ≈11 s/epoch of non-training overhead (validation
  pass, checkpointing, logging). **So an arm-7 run costs ≈3.73 GPU-h**, and
  the previous entry's ≈3.8 estimate was right to within 1 %; the plan's
  budget table figure of ≈2.6 GPU-h (≈47 s/epoch) is wrong. Concretely the
  arm-7 line should be **+3.4 GPU-h** for the campaign (3 × (3.73 − 2.6)),
  which is firmer but smaller than the "+4.2 for arm 7" the previous entry
  derived from a rounded 4.0. **Arm 8's rate remains unmeasured uncontended**
  — `mobilenetv2_adr-s1` ran at 99.8 s/epoch sharing Hamster with
  `mobilenetv2_pgd_at-s0`, and `-s2` at 255 s/epoch sharing Ferret with two
  other jobs — so the previous entry's "+3.0 for arm 8" and its ≈63 GPU-h
  campaign total stay estimates, not measurements. Also worth recording
  operationally: seed 0's 81.4 s/epoch was inflated by sharing the box with
  the heavier ADR job and two AutoAttack evaluations, **not** by having a
  second training job present — two MobileNetV2 trainings on two 4090s cost
  each other essentially nothing. Contention changes only wall clock, not
  results: `deterministic: true`, fixed seeds and fixed batch size make the
  numbers independent of throughput.
  **No new instance of the M3 aggregator directory-naming defect**: the
  `dir_prefix` fix from this session's M1b entry is present in
  `scripts/aggregate_adr_cifar10_replication.py` (per-arm `dir_prefix`, used
  at line 277), so the earlier entries' running count of that defect is
  closed at eight and should not be extended.
  Campaign state at the 08:58Z scan (`--include-hand-run`, fresh unused
  cursor): **18 of 20 run dirs**, every one terminal and successful. Missing:
  `cifar10_mobilenetv2_adr-s0` and `cifar10_r18_trades_adr-s2`, both on
  Ferret and not collected locally; Ferret's three GPUs are at 97–100 % and
  6.8–9.9 GiB, consistent with those two plus one more job, but their
  identity is unverified from here. Hamster is idle (1 MiB / 17 MiB, 0 %).
  Contract (model-weights) evaluations **6 of 20** — `r18_pgd_at-s1`, `-s2`,
  `r18_pgd_at_nesterov-s0`, `-s1`, `-s2`, `r18_adr-s0`; EMA-weights
  evaluations **1 of 9** (`r18_adr-s0`). **Twelve** terminal training runs
  have no contract evaluation: `r18_adr-s1` (its three `early-check-eval-*`
  bundles are diagnostics the aggregator does not read), `r18_adr-s2`,
  `r18_trades-s1`, `-s2`, `r18_trades_49k_validation-s0`,
  `r18_trades_adr-s0`, `-s1`, and all six MobileNetV2 runs.
  **For the human.** Nothing here licenses a scientific decision and this
  postrun launches nothing, but two things are now queued behind a choice:
  (1) twelve terminal runs need canonical evaluation, six of them the entire
  MobileNetV2 leg, and Hamster is idle; (2) `mobilenetv2_pgd_at-s2` is
  terminal with its own unprocessed postrun event. Following the two
  preceding training-only postruns, no decision packet is written — nothing
  was imported and no milestone closed. M1b stays closed; M1c stays unticked.
- 2026-09-10: postrun of the `adr-campaign-v1-cifar10_mobilenetv2_pgd_at-s2`
  **training** terminal event (`.../cifar10_mobilenetv2_pgd_at-s2/train/
  run-bundle/manifest.json`) — arm 7 seed 2, which **completes arm 7's three
  seeds** and makes it the campaign's first arm with all training done.
  **Nothing imported — no milestone closed.** The campaign is still mid-flight
  (18 of 20 run dirs), this run has no `evaluation/` output at all, and
  `docs/experiments/` still holds only the two M1b historical bundles, so there
  is no record to collide with on the idempotency check and no official-test
  number for this run to import.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `failure_class: null` on the line whose `path` is that
  manifest.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", 200/200 epoch rows (`epoch_metrics_complete: true`, epoch
  199, `global_step` 70400, and `epoch-metrics.jsonl` independently counted at
  exactly 200 lines), `train_valid_examples` 45000 on every row so the 10 %
  held-out split was honoured, both declared artifacts present —
  `epoch-metrics.parquet` (`49ca0bf57477…`) and `sample-stats-train.parquet`
  (`01b03f218724…`) — each with a content-addressed copy under its own SHA-256
  path, six checkpoints on disk (`best.pt`, `last.pt`,
  `epoch-{049,099,149,199}.pt`) and **no `best-ema.pt`**, which is correct for
  this arm (`adr: null`, so no EMA stop-rule applies). Source SHA
  `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`) from worktree
  `source-cd0b571e4685`, external lock `05cfce4cf8db…` with `.external/adr`
  pinned at `515da0e0…`. Artifact hashes were not recomputed — this session's
  sandbox refuses `sha256sum` outside the repo root — so integrity rests on the
  content-addressed paths matching the manifest; the aggregator owns the real
  check at M3.
  Contract fields match arm 7 of the frozen plan exactly: protocol
  `controlled_cifar10_mobilenetv2_adr_v1`, student `mobilenet_v2_cifar`,
  teacher `null`, method `pgd_at` (`adr: null`), SGD lr 0.1 / momentum 0.9 /
  wd 5e-4 with **`nesterov: true`** as this protocol requires, multistep
  `milestones=[100,150] gamma=0.1` at epoch end, `epochs=200`,
  `validation_fraction=0.1`, `deterministic: true`, `world_size=1`, effective
  global batch 128, `batchnorm_mode=local_per_rank`, train attack CE/Linf
  `8/255` step `2/255` steps 10 random start, selection and evaluation attack
  CE 20 steps at the same budget, seeds all 2 except the fixed
  `split=20260722` and `evaluation_attack=0`. `git diff cd0b571e4685 HEAD --
  configs/` is empty, so nothing in the contract has drifted since the pin.
  **Normalization checked explicitly, because it differs from the ResNet-18
  arms.** Both MobileNetV2 arms declare `normalization: {profile:
  cifar10_standard}` (mean 0.4914/0.4822/0.4465, std 0.247/0.2435/0.2616)
  where arms 1-6 use identity. That is a per-architecture choice applied
  identically to arm 7 and arm 8, so the MobileNetV2 ADR-vs-baseline delta is
  **not** confounded by it; it does mean MobileNetV2 numbers are not
  input-pipeline-identical to the ResNet-18 ones, which only matters for the
  rule's across-architecture comparison of *gains*, not of absolute accuracies.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable): best epoch 151 clean 0.7396 / PGD 0.4494; last (epoch 199) clean
  0.7564 / PGD 0.4376; robust-overfit gap 1.18 pp; validation PGD mean over
  epochs 150-199 0.44381.
  **Arm 7 is now complete on training, so the MobileNetV2 baseline has its
  first measured noise floor** (all validation, all CE-PGD-20, all
  single-seed-per-row — this is the metric checkpoint selection optimized
  against, so it flatters nothing but itself):

  | seed | best epoch | best clean | best PGD | last clean | last PGD | gap (pp) | PGD mean 150-199 |
  |---|---|---|---|---|---|---|---|
  | 0 | 175 | 0.7460 | 0.4496 | 0.7480 | 0.4434 | 0.62 | 0.44188 |
  | 1 | 173 | 0.7458 | 0.4502 | 0.7424 | 0.4370 | 1.32 | 0.44399 |
  | 2 | 151 | 0.7396 | 0.4494 | 0.7564 | 0.4376 | 1.18 | 0.44381 |

  Best-checkpoint validation PGD spans **0.08 pp** across three seeds
  (0.4494-0.4502) and the flat-window mean spans 0.21 pp — both far tighter
  than any effect this campaign is trying to measure. (Seed 1's numbers are
  read from its own manifest for this spread; its postrun owns that entry.)
  **Three corrections to the arm-7 seed-0 entry above, all from having more
  than one seed.**
  1. **Its robust-overfitting ordering does not survive.** That entry read
     arm 7's 0.62 pp against arm 8 seed 1's 0.76 pp and concluded the baseline
     overfits *less* than the ADR arm. Arm 7's three seeds give 0.62 / 1.32 /
     1.18 pp (mean 1.04), and arm 8's two give 0.76 / 0.68 pp — arm 8's entire
     range sits inside arm 7's, so the ordering is unresolvable at this seed
     count and should not be quoted either way. **The deflationary conclusion
     survives and is now stronger**: every MobileNetV2 reading on either arm is
     ≤1.32 pp, against 4.08-8.88 pp at ResNet-18, so MobileNetV2 at 200 epochs
     barely robustly overfits and no ADR suppression effect is measurable at
     this architecture. ADR's gap-halving stays a ResNet-18 result.
  2. **Its selected-epoch figure was single-seed.** "MobileNetV2 PGD-AT 175"
     should read 151 / 173 / 175 — the argmax wanders 24 epochs within one arm.
  3. **Its plateau warning is confirmed and is sharper here.** This run's
     winning 0.4494 at epoch 151 beats the runner-up (0.4492, epoch 184) by
     **0.02 pp**, and 21 separate epochs sit within 0.42 pp of the top. Best-pt
     selection on this arm is effectively arbitrary inside the plateau, which
     is also why best-minus-last is so small. Do not read a sub-1-pp
     MobileNetV2 best-vs-last gap as evidence of anything.
  **Directional read of the primary leg — first time both sides have a
  measured spread, and still not a result.** On validation PGD at each arm's
  own best checkpoint, arm 8 (ADR) averages 0.4840 over two seeds against arm
  7's 0.44973 over three: **+3.43 pp**, about 17× the larger of the two
  within-arm spreads (0.20 pp). The preregistered rule is **AutoAttack on the
  official 10,000-example test set, best checkpoint, three seeds per arm**, and
  this fails it on three counts — validation not test, CE-PGD-20 not
  AutoAttack, and two seeds on the treatment side. **No verdict is licensed**,
  and the ResNet-18 leg it must be compared against has no three-seed
  official-test pair either. What has changed is that the MobileNetV2
  baseline's own noise floor is no longer a guess.
  **Budget — the seed-0 entry's estimate is now a measurement.** That entry put
  arm 7 at ≈3.8 GPU-h from a contended run, estimating ≈68 s per full epoch.
  This run was uncontended for its whole length (all 200 `train_seconds` fall
  in 55-59 s, mean ≈56.5 s, versus seed 0's 55.5-79 s), and its wall clock was
  05:09:41.6Z → 08:53:26.1Z = **3 h 43 min 44.4 s**, i.e. **67.1 s per full
  epoch** and **≈3.73 GPU-h for the run**. Non-training overhead (one
  validation pass plus checkpointing and logging) is therefore ≈10.6 s/epoch.
  The plan's budget line of ≈47 s / ≈2.6 GPU-h for arm 7 is confirmed wrong by
  direct measurement, the corrected ≈3.7-3.8 GPU-h stands, and so does the
  ≈+7 GPU-h campaign-total correction (55.6 → ≈63 GPU-h training).
  **The M3 aggregator directory-naming defect is resolved for this arm.** The
  committed `scripts/aggregate_adr_cifar10_replication.py` now carries a
  `dir_prefix` per arm and builds `run_root / f"{arm['dir_prefix']}{arm_key}-s{seed}"
  / "train"` (line 277); for arm 7 that resolves to
  `cifar10_mobilenetv2_pgd_at-s2/train`, which is this run's real directory.
  No further action — the eight logged instances are closed.
  Campaign state at the 08:57Z scan (`--include-hand-run`, fresh unused cursor
  — 28 bundles, every one terminal and successful), **18 of 20 run dirs, all 18
  terminal and successful**: `r18_pgd_at-s1/-s2`,
  `r18_pgd_at_nesterov-s0/-s1/-s2`, `r18_adr-s0/-s1/-s2`, `r18_trades-s1/-s2`,
  `r18_trades_49k_validation-s0`, `r18_trades_adr-s0/-s1`,
  `mobilenetv2_adr-s1/-s2`, `mobilenetv2_pgd_at-s0/-s1/-s2`. **Only two run
  dirs are still missing** — `cifar10_mobilenetv2_adr-s0` and
  `cifar10_r18_trades_adr-s2`, both on Ferret, whose three GPUs are at 98-100 %
  at session start, consistent with them still running. Contract
  (model-weights) evaluations **6 of 20**: `r18_pgd_at-s1`, `-s2`,
  `r18_pgd_at_nesterov-s0`, `-s1`, `-s2`, `r18_adr-s0`. EMA-weights evaluations
  **1 of 9** (`r18_adr-s0`). **Twelve terminal training runs have no contract
  evaluation, including all five MobileNetV2 runs** — so the primary leg of the
  preregistered rule still has **zero** official-test data, and queueing those
  evaluations is the one thing standing between this campaign and M2. M1c stays
  unticked until all 20 are launched and verified; this postrun launches
  nothing.
  **Operational blocker, root cause now confirmed in the config — this is the
  highest-value thing for the human to fix.** The headless postrun for this run
  (`orchestration/ardx/claude-runs/20260910T085343Z-…-pgd_at-s2_train.log`)
  contains exactly one line, "Ignoring 44 permissions.allow entries from
  .claude/settings.json: this workspace has not been trusted", and did nothing.
  Read directly from `/home/shunsukenaito/.claude.json` (line 968):
  `projects["/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap"].hasTrustDialogAccepted`
  is **`false`**. Eight further postruns fired at 08:54:13Z and 08:56:45Z —
  `mobilenetv2_pgd_at-s1`, the six Ferret-collected training bundles
  (`r18_adr-s2`, `r18_pgd_at_nesterov-s2`, `r18_trades-s2`,
  `r18_trades_49k_validation-s0`, `r18_trades_adr-s1`, `mobilenetv2_adr-s2`)
  and `eval-3bfb6a2f65ef77e308e1` — and spot-checking their logs shows the same
  single line, so **nine postruns including this one have silently no-opped**
  and every campaign event from here needs a hand-run postrun until it is
  fixed. The fix is one line (set that key to `true`) or accepting the trust
  dialog once interactively at that exact path spelling, per CLAUDE.md. Left
  unchanged here: it is the user's own config, outside the repo, and not this
  skill's to edit.
- 2026-09-10: postrun of the
  `adr-campaign-v1-ferret-cifar10_r18_trades_49k_validation-s0` **training**
  terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_r18_trades_49k_validation-s0/train/run-bundle/manifest.json`) —
  **arm 6, the 49k-validation TRADES pilot, and the only run this arm will
  ever have** (the contract gives it one seed). **Nothing imported — no
  milestone closed.** Training only: `docs/experiments/` holds `.gitkeep` and
  the M1b `historical/` archive and no record for this contract, so the
  idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `status: completed`, `failure_class: null` on exactly the
  `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and `epoch-metrics.jsonl` really carries 200 rows ending at epoch
  199 / `global_step` 76600. Both declared artifacts have a content-addressed
  copy under `run-bundle/artifacts/<name>/<sha256>/` whose directory name
  equals the manifest's own `sha256` (`epoch-metrics.parquet` `1d1cbcbfbe5c…`,
  `sample-stats-train.parquet` `1863bfaac5f3…`). `best.pt`, `last.pt` and all
  four periodic checkpoints (`epoch-049/099/149/199.pt`) are on disk; there is
  **no** `best-ema.pt`, which is correct — `method.adr` is `null` for this arm.
  Source SHA `cd0b571e4685…` clean (`dirty: false`, empty-diff `e3b0c442…`),
  external lock `05cfce4cf8db…`, `config_hash 45b0eaa50b89…`. Artifact bytes
  were not re-hashed (sandbox); the aggregator owns the real check at M3.
  Contract fields match arm 6 exactly: protocol
  `controlled_cifar10_r18_trades_49k_validation_v1`, student
  `saad_resnet18_cifar_v1` with `cifar10_raw_identity` normalization, method
  `trades` v1 (`trades_beta 6.0`, `adr: null`), `teacher: null`, **`nesterov:
  false`**, `epochs=200`, `milestones=[100,150] gamma=0.1`, `lr=0.1
  momentum=0.9 wd=5e-4`, **`validation_fraction=0.02`** — the one and only
  field that distinguishes this arm from arm 4 — train attack KL/`student_clean`
  10 steps at `8/255` step `2/255` random start, selection and evaluation
  attack CE 20 steps at the same budget, `world_size=1`, effective global batch
  128, `deterministic: true`, seeds all 0 except the fixed `split=20260722` and
  `evaluation_attack=0`. `git diff cd0b571e4685 HEAD` is empty for
  `configs/scientific/cifar10_r18_trades_49k_validation.yaml`,
  `cifar10_r18_trades.yaml`, `src/ard/protocols/__init__.py` and
  `src/ard/config/schema.py`, so the contract has not drifted since the pin.
  **The 49k/1k split is confirmed three ways, not assumed**, which matters
  because it is the entire intervention: `train_valid_examples: 49000` on every
  epoch row; `global_step` 76600 = 200 × 383 = 200 × ceil(49000/128); and every
  `val_*` figure in the run is an exact multiple of 0.001, i.e. measured on
  1,000 images. (The resolved config also carries `dataset.num_samples: 16`;
  that is a schema default and is inert here — 49,000 training images were
  actually used.)
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log). Both
  rows re-read from `epoch-metrics.jsonl` and they match the manifest summary
  exactly:

  | checkpoint | epoch | clean | PGD |
  |---|---|---|---|
  | best | 102 | 0.825 | 0.528 |
  | last | 199 | 0.858 | 0.495 |

  **The headline "+1.1 pp over arm 4" that these numbers invite is mostly
  selection noise, and the run's own data shows it.** Cutting validation from
  5,000 to 1,000 images does not just shrink the held-out set — it makes
  `best.pt` a noisier choice. At p≈0.5 the standard error of a validation-PGD
  reading is 1.58 pp at n=1,000 against 0.71 pp at n=5,000, so the argmax over
  ~100 post-drop epochs is pulled further above the truth here than in any
  other arm. Three signs of that in this run: (i) the maximum 0.528 is a
  **two-way tie** between epoch 102 and epoch 151, broken toward the earlier
  epoch, with 0.526 (epoch 159) and 0.525 (epochs 119, 161) one to three
  images behind; (ii) the epoch 100–199 mean is 0.50622 and the 150–199 mean
  0.50852, so the selected 0.528 sits 2.18 pp above its own plateau; (iii) the
  same statistic for arm 4's two fresh seeds is +1.52 pp (seed 1) and +1.53 pp
  (seed 2), i.e. the small validation set inflates the argmax by an extra
  **≈0.65 pp** here. Subtract that from the raw best-checkpoint difference
  (0.528 versus arm 4's 0.5164/0.5166, **+1.15 pp**) and ≈**+0.50 pp** is left
  — which is exactly the plateau-mean difference (0.50622 versus 0.50125 and
  0.501314, **+0.49 pp**). Two independent routes agreeing at 0.01 pp is the
  reason to trust the deflated number rather than the headline one.
  **So the directional read on the pilot's actual question is: the extra 4,000
  training images look worth roughly half a point, not the 1.2–1.5 pp the
  literature gap needs — and even that is inside this arm's own noise.** The
  preregistered comparison (see "Preregistered decision rule") is AutoAttack on
  the official 10,000-image test set against arm 4 seed 0's archived 47.87 best
  / 44.99 last, aiming at the literature range 49.21–49.50. Validation PGD is
  the model-selection metric, on a different set, under a different attack, at
  n=1,000 — **no verdict is licensed and none is offered.** The late-window
  comparison is slightly friendlier to the pilot (150–199 mean 0.50852 versus
  0.500036 and 0.500024, **+0.85 pp**) and the pilot does not decay after the
  second LR drop where arm 4 mildly does (its own 100–149 mean is 0.50392,
  below its 150–199 mean; slope over epochs 120–199 is −1.49e-05/epoch, i.e.
  −0.12 pp across 80 epochs — flat). But a ±1.6 pp standard error swallows all
  of these differences, and **averaging over epochs does not shrink it**: every
  epoch scores the *same* 1,000 images, so the finite-set error is shared and
  never averages away. That is the structural limit of this arm, and it is why
  the official test — 10,000 images, where the pilot and arm 4 are on equal
  footing — is the only thing that can answer it.
  **Robust overfitting is unchanged by the intervention**: best−last gap 3.3 pp
  here against 3.20 pp (arm 4 seed 1) and 3.14 pp (seed 2). Whatever 49k
  training does, it does not move the gap.
  **The evaluation that answers this arm is already running** — verified
  directly, not inferred: `…/train/evaluation/` exists with a live W&B segment
  `eval-7f2c13eb1eb778fd2e2a` opened 2026-09-10T08:59:10Z, `panel-best.jsonl`
  and `sample-stats-best.parquet` already written and **no
  `evaluation-results.json` yet**, so it is mid-flight and not terminal. Its
  `resolved_evaluation_config.yaml` is the right one: `checkpoints: both`, CE
  20 steps at `8/255` step `2/255` random start on the CIFAR-10 **test** split,
  and **`autoattack: true`**, so one job will produce both the CE-PGD-20 and
  the AutoAttack numbers this arm needs. `evaluation-lineage.json` binds it to
  this training run — its `training_config_hash` `45b0eaa50b89…` equals the
  training manifest's `config_hash`. Nothing to do but wait for its own
  terminal event; this postrun launches nothing.
  **Second confirmed instance of the Ferret artifact-path defect** logged in
  the `r18_trades-s2` entry above, so that entry's fix is not
  arm-specific. This run's manifest declares both artifacts at
  `runs/adr-campaign-v1-ferret-cifar10_r18_trades_49k_validation-s0/outputs/
  train/…`, a Ferret-side staging path with **no counterpart in the local
  runtime tree**; only the bundle's content-addressed copies exist here.
  `_verify_bundle` would raise `declared artifact is missing` and abort M3.
  Nothing fixed here — this postrun touches no code.
  **Budget — a measurement of the one cost knob this arm alone can isolate.**
  Wall clock 12:16:39.1Z → 18:11:42.4Z = **5 h 55 min 03 s**, i.e. 106.5 s per
  full epoch and **≈5.92 GPU-h**, against the plan's ≈2.6 GPU-h line for a
  single-validation-pass ResNet-18 run — wrong here by 2.3×, for the Ferret
  contention reason the `r18_trades-s2` entry already established. The new
  information is that this run had a **perfectly concurrent twin**: arm 4 seed
  2 ran 12:16:37.9Z → 18:32:04.0Z on the same host, same method, same
  architecture, alongside the same third job (`pgd_at_nesterov-s2`,
  12:14:47.5Z → 18:08:46.1Z), differing only in 49k/1k versus 45k/5k. The
  pilot finished **5.4 % sooner** (21,303 s versus 22,526 s) *despite training
  on 8.9 % more images* — dropping 4,000 validation images saves more than
  adding 4,000 training images costs, because validation runs a 20-step attack
  where TRADES training runs a 10-step one. Order of magnitude, from this run's
  own numbers (`train_seconds` 96.0–108.8 s, settling at ≈102 s after the first
  dozen epochs, so ≈4 s per epoch outside the training loop for a 1,000-image
  pass plus checkpointing): a validation image costs somewhere between 1.2× and
  1.7× a TRADES training image, the spread coming from not having measured arm
  4 seed 2's own `train_seconds` mean. **Validation-set size is a first-order
  budget knob, not a rounding error** — worth remembering before any ImageNet
  costing.
  Campaign state, from this session's own scans: **18 of 20 run dirs, all 18
  training-terminal and successful**; `mobilenetv2_adr-s0` and `trades_adr-s2`
  still have no run dir. Contract (model-weights) evaluations **6 of 20**
  terminal (`pgd_at-s1`, `-s2`, `pgd_at_nesterov-s0`, `-s1`, `-s2`, `adr-s0`),
  EMA-weights **1 of 9** (`adr-s0`), with this arm's evaluation among those now
  running. The headless postrun path is still dead — re-checked directly,
  `/home/shunsukenaito/.claude.json:968` `hasTrustDialogAccepted` is still
  **`false`**, so this run's own watcher-fired postrun no-opped and this entry
  is a hand-run one. M1c stays unticked.
- 2026-09-10: postrun of the `adr-campaign-v1-ferret-cifar10_mobilenetv2_adr-s2`
  **training** terminal event (`runs/adr-cifar10-campaign-v1/
  cifar10_mobilenetv2_adr-s2/train/run-bundle/manifest.json`) — **arm 8 seed 2,
  the campaign's first cross-seed reading of the MobileNetV2 ADR cell and the
  most expensive single training job it has run.** **Nothing imported — no
  milestone closed.** Training only: the run has no `evaluation/` output of any
  kind, so it produces no official-test number, and `docs/experiments/` still
  holds only M1b's historical archive with no `adr_cifar10_replication_v1.json`,
  so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan over the campaign root returns
  `terminal: true`, `success: true`, `status: completed`, `failure_class: null`
  on exactly the `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application error
  recorded", `epoch_metrics_complete: true` with 200 of 200 expected epochs, and
  `epoch-metrics.jsonl` really carries 200 rows ending at epoch 199 /
  `global_step` 70400. Both declared artifacts exist at their content-addressed
  `run-bundle/artifacts/<name>/<sha256>/` paths, each directory name equalling
  the manifest's own `sha256` (`epoch-metrics.parquet` `eddbb1bb3d38…`,
  `sample-stats-train.parquet` `d61d1f0fc559…`). `best.pt`, `best-ema.pt`,
  `last.pt` and all four periodic checkpoints (`epoch-049/099/149/199.pt`) are on
  disk — seven, the ADR-family count. Source SHA `cd0b571e4685…` clean
  (`dirty: false`, empty-diff `e3b0c442…`), external lock `05cfce4cf8db…` with
  `.external/adr` at the pinned `515da0e0373f…`, `config_hash 15530b2f2f3a…`.
  Artifact bytes were not re-hashed — this session's sandbox refuses any Bash
  read outside the repo root — so integrity rests on that path-equals-hash
  correspondence; the aggregator owns the real check at M3.
  **This run is another instance of the `ferret-collect` artifact-path defect
  already logged for `r18_adr-s2`.** Both declared artifacts' `path` fields, and
  `completion.json`'s `output_dir`, still name Ferret's own run root
  (`runs/adr-campaign-v1-ferret-cifar10_mobilenetv2_adr-s2/outputs/train/…`),
  which does not resolve on this host; the collected copies live at
  `cifar10_mobilenetv2_adr-s2/train/{epoch-metrics,sample-stats-train}.parquet`
  and the content-addressed copies inside the bundle are intact, so nothing is
  lost — but the manifest's absolute paths are not usable as written.
  Contract fields match arm 8 exactly: protocol
  `controlled_cifar10_mobilenetv2_adr_v1`, student `mobilenet_v2_cifar` with
  `normalization: {profile: cifar10_standard}`, method `adr` v1, `teacher: null`,
  `nesterov: true`, `epochs=200`, `milestones=[100,150] gamma=0.1`, `lr=0.1
  momentum=0.9 wd=5e-4`, `validation_fraction=0.1`, ADR `ema_decay=0.995`
  `T 2.5→2.0` `lambda 0.7→0.95`, train attack KL/rectified 10 steps at `8/255`
  step `2/255` random start, selection attack CE 20 steps at the same budget,
  `world_size=1`, effective global batch 128, `deterministic: true`, seeds all
  **2** except the fixed `split=20260722` and `evaluation_attack=0`. `git diff
  cd0b571e4685 HEAD` is empty for `configs/scientific/cifar10_mobilenetv2_
  {adr,pgd_at}.yaml`, `src/ard/protocols/__init__.py` and
  `src/ard/config/schema.py`, so the contract has not drifted since the pin.
  (The resolved config carries `dataset.num_samples: 16`. That is the schema
  default at `src/ard/config/schema.py:301` and is inert on the real CIFAR-10
  path: every epoch row carries `train_valid_examples: 45000.0`, and
  `global_step` 70400 is exactly 200 × 352 batches of 128 — the 45k held-out
  training split, not a 16-sample stub.)
  **ADR stop-rule checks pass.** EMA validation ran every epoch and never
  crashed: all 200 rows carry `val_clean_accuracy_ema` and
  `val_pgd_accuracy_ema`. And `best-ema.pt` was selected independently of the
  student — EMA's best validation epoch is **143**, the student's is **158** —
  which is what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section requires.
  Held-out **validation** diagnostics only (not the official test set, not
  reportable, and not comparable to any official-test table in this log). All
  four rows were re-read from `epoch-metrics.jsonl` and match the manifest
  summary exactly:

  | weights | checkpoint | epoch | clean | PGD |
  |---|---|---|---|---|
  | student | best | 158 | 0.7662 | 0.4830 |
  | student | last | 199 | 0.7714 | 0.4762 |
  | EMA | best-ema | 143 | 0.7618 | 0.4870 |
  | EMA | last | 199 | 0.7700 | 0.4784 |

  EMA is ahead of the student at each's own best checkpoint by **+0.40 pp**
  validation PGD, against +0.64 pp for seed 1 and the same direction as both
  ResNet-18 ADR seeds — a consistency check on the selection logic, not
  independent evidence.
  **Arm 8's first within-arm cross-seed reading, and it is the tightest
  ADR-family cell in the campaign on the student side.** Seed 1 versus seed 2,
  all validation, best checkpoint unless stated:

  | quantity | s1 | s2 | spread |
  |---|---|---|---|
  | student best PGD | 0.4850 | 0.4830 | 0.20 pp |
  | student best clean | 0.7666 | 0.7662 | 0.04 pp |
  | student last PGD | 0.4774 | 0.4762 | 0.12 pp |
  | EMA best PGD | 0.4914 | 0.4870 | 0.44 pp |
  | student best epoch | 151 | 158 | 7 epochs |
  | robust-overfit gap | 0.76 pp | 0.68 pp | 0.08 pp |

  The student-side spread (0.20 pp PGD, 0.04 pp clean) is well inside this
  project's screen noise floor and narrower than arm 3's 0.40 pp at ResNet-18.
  **The EMA side is more than twice as wide (0.44 pp) as the student side**, on
  two seeds — worth watching rather than concluding from, because "ADR + WA" is
  the comparator the ADR paper headlines, and a comparator with a wider seed
  spread than the thing it is compared against needs its own spread reported
  next to every number it produces.
  **This run's argmax is weak, the same way arm 7's are.** Validation PGD over
  the last 70 epochs is a flat plateau: the winning 0.4830 at epoch 158 beats
  the runner-up (0.4814, epoch 185) by only **0.16 pp**, and six epochs sit
  within 0.26 pp of the top (0.4804–0.4830 at epochs 131, 150, 158, 165, 185,
  188). Selection is close to arbitrary within that plateau, which is also why
  best-minus-last is so small. Do not read a sub-1-pp MobileNetV2 best-vs-last
  gap, on either arm, as evidence of anything.
  The cross-arm consequences of this seed — the now seed-matched MobileNetV2
  leg, and the correction to the arm-7-seed-0 entry's robust-overfitting
  claim — are written up in the `r18_adr-s2` and `mobilenetv2_pgd_at-s2` entries
  above and are not restated here.
  **Cost: 14.18 GPU-hours, the campaign's most expensive training job by a wide
  margin, and 3.5× the arm-8 corrected allowance.** Wall clock was
  2026-09-09T18:12:40.3Z → 2026-09-10T08:23:35.6Z, **14 h 10 min 55.3 s**
  (255.3 s/epoch averaged), against the plan's 3.0 GPU-h line for arm 8 and the
  `mobilenetv2_adr-s1` entry's corrected ≈4.0. The training loop alone ran
  165.5–210.2 s/epoch, sitting flat at **184–190 s** for most of the run and
  drifting to 191–210 s over the last 30 epochs; final-epoch throughput was
  **214.1 img/s** against **730.7 img/s** for the identical arm on Hamster
  (seed 1). Contention changes only wall clock, not results: `deterministic:
  true`, fixed seeds and fixed batch size make the numbers independent of
  throughput.
  **Because this run's training loop is so stable, it is the first that lets the
  ADR double-validation overhead be read directly** — mean full-epoch 255.3 s
  minus a ≈189 s mean training loop leaves **≈66 s/epoch** for the two
  validation passes plus checkpointing and logging. That figure is
  Ferret-and-contention-specific and is **not** comparable to the
  `mobilenetv2_pgd_at-s0` entry's ≈13 s single-pass Hamster figure without a
  throughput correction, which this entry does not attempt.
  **Ferret/Hamster wall-clock ratio, five matched arms, adding this one to the
  Ferret-packing finding already logged.** Same arm, same source SHA, same
  contract, Ferret's three-way packing versus Hamster's two-way:
  `r18_pgd_at_nesterov` 5.90 h (s2) vs 2.49/2.62 h → **2.3×**; `r18_adr` 6.94 vs
  2.99/3.06 → **2.3×**; `r18_trades` 6.26 vs 3.45 → **1.8×**; `r18_trades_adr`
  7.70 vs 4.28 → **1.8×**; `mobilenetv2_adr` 14.18 vs 5.54 → **2.6×**.
  MobileNetV2 is the worst-affected arm, which is the opposite of what its
  parameter count suggests and is consistent with depthwise convolutions being
  memory- and host-bound rather than FLOP-bound.
  **Campaign training cost as executed: 88.4 GPU-h for 18 of 20 runs** (Hamster
  41.5 h over 12 runs, Ferret 46.9 h over 6), against the plan's 55.6 GPU-h for
  all 20. Two caveats keep this from being a straight overrun: wall clock equals
  GPU-hours here only because each run held one GPU exclusively for its whole
  duration, and concurrency *inflates* the aggregate — so 55.6 was never wrong
  as a serial estimate, it is wrong as a capacity plan under five-way packing.
  With `r18_trades_adr-s2` and `mobilenetv2_adr-s0` still to run, training is on
  track for **≈95–105 GPU-h**, before the 40 GPU-h evaluation line.
  Campaign state at the 09:0xZ scan (`--include-hand-run`, fresh unused cursor —
  36 bundles), **18 of 20 training run dirs, every one terminal and
  successful**; only `r18_trades_adr-s2` and `mobilenetv2_adr-s0` have no run
  dir. Of the 36 bundles, 28 are terminal and successful and **8 are contract
  evaluations still in flight** (`manifest_status: running`, no
  `completion.json`): `mobilenetv2_adr-s1`, `mobilenetv2_pgd_at-s1`, `-s2`,
  `r18_adr-s1`, `-s2`, `r18_trades_49k_validation-s0`, `r18_trades_adr-s0`,
  `-s1`. All five GPUs are busy (Hamster 6594/6470 MiB at 100 %, Ferret
  7205/8185/9380 MiB at 96–100 %), so none of the eight is obviously stalled —
  but note that **the watcher exposes no progress clock for evaluation jobs**
  (`age_seconds` and `progress_timestamp` are both null on all eight), so
  staleness cannot be judged from the bundle alone. Do not relaunch any of them.
  **Arm 8 is the least-advanced arm in the campaign, and it is the treatment
  side of the primary leg.** Seed 1's evaluation is in flight, **this run has no
  evaluation of any kind**, and seed 0 has never been launched. Queueing this
  run's model-weights and EMA-weights contract evaluations, and launching
  `mobilenetv2_adr-s0`, are decisions for the human — this postrun launches
  nothing. M1c stays unticked.

- 2026-09-10: postrun of the
  `adr-campaign-v1-ferret-cifar10_r18_trades_adr-s2` **training** terminal
  event (`runs/adr-cifar10-campaign-v1/cifar10_r18_trades_adr-s2/train/
  run-bundle/manifest.json`) — **arm 5 seed 2, which completes arm 5 at 3/3
  seeds and with it the whole ResNet-18 leg of the campaign**. **Nothing
  imported — no milestone closed.** Training only: the run has no terminal
  `evaluation/` or `evaluation-ema/` output, so it produces no official-test
  number, and `docs/experiments/` still holds only the M1b `historical/` reuse
  bundles, so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: true`, `status: completed`, `failure_class: null` on exactly the
  `--state-path` line the watcher passed.
  Verified for this bundle: `completion.json` `{"status": "completed"}`,
  `manifest.status=sync_pending`, `error-marker.txt` reads "no application
  error recorded", `epoch_metrics_complete: true` with 200 of 200 expected
  epochs, and `epoch-metrics.jsonl` really carries 200 rows covering epochs
  0–199 with no gap, ending at `global_step` 70400 (= 200 × 352, and 352 =
  ceil(45000/128)). No `NaN`, `Infinity` or `null` appears anywhere in the
  200 rows. Both declared artifacts have a content-addressed copy under
  `run-bundle/artifacts/<name>/<sha256>/` whose directory name equals the
  manifest's own `sha256` (`epoch-metrics.parquet` `3c68312b651f…`,
  `sample-stats-train.parquet` `a30c84b53596…`), and the canonical local
  copies sit alongside them in `train/`. All **seven** checkpoints the contract
  requires are on disk: `best.pt`, `best-ema.pt`, `last.pt` and
  `epoch-{049,099,149,199}.pt` — `best-ema.pt` is present and required here,
  since `method.adr` is non-null for this arm. Source SHA `cd0b571e4685…`
  clean (`dirty: false`, empty-diff `e3b0c442…`), external lock
  `05cfce4cf8db…` with `.external/adr` at the pinned `515da0e0373f…`,
  `config_hash 350881af2c90…`. Artifact bytes were not re-hashed — this
  session's sandbox refuses to hash outside the repo root — so integrity rests
  on that path-equals-hash correspondence; the aggregator owns the real check
  at M3.
  **Ferret provenance, third instance of the artifact-path defect.** The
  manifest's artifact `path` fields and `completion.json`'s `output_dir` still
  name Ferret's own flat run root
  (`runs/adr-campaign-v1-ferret-cifar10_r18_trades_adr-s2/outputs/train/`), not
  the canonical local dir the collected bundle now sits in. Nothing is missing
  — the canonical copies are present — so these are a record of where the job
  ran, not broken references; same shape as the `trades-s2` and
  `trades_49k_validation-s0` entries above. Independent host evidence:
  `environment.json` reports kernel `7.0.0-28-generic` against Hamster's
  `7.0.0-31-generic`, with `torch 2.11.0+cu128` / CUDA 12.8 on an RTX 4090.
  Consistent with a Ferret job, `tracking.mode` resolved to `offline_sync` and
  `wandb_url` is null with `status: sync_pending` — the W&B run is not synced
  yet.
  Contract fields match arm 5 exactly: protocol `controlled_cifar10_r18_adr_v1`,
  student `saad_resnet18_cifar_v1` with `cifar10_raw_identity` normalization,
  method `adr_trades` v1, `teacher: null` (ADR distils from an internal EMA of
  the student, not an external teacher), `trades_beta=6.0`, ADR block
  `ema_decay=0.995` `T 2.5→2.0` `lambda 0.7→0.95`, **`nesterov: true`**,
  `epochs=200`, `milestones=[100,150] gamma=0.1`, `lr=0.1 momentum=0.9
  wd=5e-4`, `validation_fraction=0.1`, train attack KL/`kl_target=student_clean`
  10 steps at `8/255` step `2/255` random start, selection attack CE 20 steps
  at the same budget, `world_size=1`, effective global batch 128,
  `deterministic: true`, seeds all 2 except the fixed `split=20260722` and
  `evaluation_attack=0`. `git diff cd0b571e4685 HEAD` touches no file under
  `configs/` or `src/` at all — only docs, `.claude/`, and the M3 aggregator —
  so the contract has not drifted since the pin. The 45,000/5,000 split is
  confirmed twice: `train_valid_examples` is 45000 on **all 200** rows, and
  `global_step` 70400 is exactly 200 × ceil(45000/128).
  **ADR stop-rule checks pass.** EMA validation ran every epoch and never
  crashed: all 200 rows carry both `val_clean_accuracy_ema` and
  `val_pgd_accuracy_ema`. And `best-ema.pt` was selected independently of the
  student — the EMA's best validation epoch is **118**, the student's is
  **150** — which is what `docs/SCIENTIFIC_INVARIANTS.md`'s ADR section
  requires.
  Held-out **validation** diagnostics only (5,000 held-out *training* images,
  CE-PGD-20 — not the official test set, not AutoAttack, not reportable, and
  not comparable to any official-test table in this log):

  | weights | checkpoint | epoch | clean | PGD |
  |---|---|---|---|---|
  | student | best | 150 | 0.8454 | 0.5428 |
  | student | last | 199 | 0.8492 | 0.5326 |
  | EMA | best-ema | 118 | 0.8428 | 0.5498 |
  | EMA | last | 199 | 0.8468 | 0.5332 |

  All four rows re-read from `epoch-metrics.jsonl`; the two student rows match
  the manifest summary exactly, and the manifest's
  `val_pgd_late_mean_epoch_150_199` (0.535116) reproduces to the last digit
  when the 50 raw rows are averaged by hand.
  **Arm 5 is now complete at 3/3 seeds, and it is the most stable arm measured
  so far.** Same validation protocol throughout:

  | seed | best epoch | best clean | best PGD | last clean | last PGD | gap |
  |---|---|---|---|---|---|---|
  | 0 | 150 | 0.8474 | 0.5442 | 0.8488 | 0.5346 | 0.0096 |
  | 1 | 116 | 0.8388 | 0.5388 | 0.8504 | 0.5262 | 0.0126 |
  | 2 | 150 | 0.8454 | 0.5428 | 0.8492 | 0.5326 | 0.0102 |

  Across three seeds the spread is **0.54 pp** of validation PGD at best
  (0.5388–0.5442), **0.84 pp** at last, and only **0.16 pp** of clean accuracy
  at last (0.8488–0.8504) — inside the control-vs-control noise floor this
  project has already measured elsewhere. Robust overfitting is 0.96–1.26 pp
  in all three seeds, far below arm 4's `trades-s1/-s2` readings of 3.20/3.14
  pp. **This licenses no claim.** It is validation, not the official test; and
  arms 4 and 5 still differ in optimizer as well as method (plain SGD vs
  Nesterov), the confound this plan already records. The decisive ResNet-18 leg
  remains arm 3 vs arm 2.
  **New, and the most useful thing this run adds: ADR's EMA advantage lives
  entirely before the second LR drop, and is gone by the last epoch.** Reading
  the student and EMA validation-PGD curves side by side over the same 5,000
  images (a paired comparison — both branches are scored on the identical
  held-out draw every epoch, so the difference is far better determined than
  either level):

  | window | student mean | EMA mean | EMA − student |
  |---|---|---|---|
  | epochs 100–149 (after 1st LR drop) | 0.53176 | 0.54482 | **+1.31 pp** |
  | epochs 150–199 (after 2nd LR drop) | 0.53512 | 0.53699 | **+0.19 pp** |

  The second drop moves the two branches in *opposite* directions: the student
  gains 0.34 pp across it while the EMA loses 0.78 pp. The EMA peaks at epoch
  118 (0.5498), before the drop, and never recovers that level afterwards; the
  student peaks at epoch 150, the very first epoch after the drop, and holds.
  By epoch 199 the two are tied to within 0.06 pp (0.5332 vs 0.5326).
  **The same pattern is in seed 1**, which makes it a two-seed reading rather
  than a one-run curiosity: seed 1's EMA led by +1.12 pp at the two best
  checkpoints (0.5500 at epoch 121 vs 0.5388 at 116) and ended the run
  0.10 pp *behind* the student (0.5252 vs 0.5262). Seed 0's sampled +0.44 pp
  at epoch 149 is the same sign. So across all three seeds the EMA's
  best-checkpoint lead is positive but modest (+0.44 / +1.12 / +0.70 pp) and
  in both seeds measured to the end it evaporates at the last epoch.
  Two consequences worth carrying into M3, neither of them a decision:
  (i) the `evaluation-ema/` pass this arm still owes is a **real** measurement
  and not redundant — but what it will mostly measure is `best-ema.pt`, i.e.
  the EMA's pre-drop peak, since `last.pt` under `--weights=ema` is where the
  advantage has already vanished; (ii) `best.pt` (epoch 150) and `best-ema.pt`
  (epoch 118) come from **different LR regimes** in this run, which is exactly
  why `docs/SCIENTIFIC_INVARIANTS.md` keeps `best-ema.pt` evaluated at both
  `--weights=model` and `--weights=ema` as the one same-epoch, strictly
  paper-equivalent comparison. Everything above is validation on 5,000 held-out
  *training* images and answers nothing about the official 10,000-image test
  set.
  **A second confirmation that the late peak tracks the objective, not ADR.**
  The PGD-AT arms (1–3) all peak at epoch 101–104, just past the first drop.
  Both TRADES-objective arms peak later and more variably — arm 4 `trades-s1`
  at 157, arm 5 at 150 / 116 / 150 across its three seeds. Two of arm 5's three
  seeds now select epoch **150 exactly**, the first epoch after the second
  drop, where validation PGD jumps 1.68 pp in one epoch (0.5260 → 0.5428).
  Checkpoint selection is unaffected either way: `best.pt` is chosen by
  validation PGD, not by epoch.
  **Cost, and a second Ferret data point for arm 5.** Wall clock 01:07:12.6Z →
  09:08:47.7Z, **8 h 01 min 35.1 s** (144.5 s/epoch averaged). Contended from
  epoch 0 to epoch 199 — throughput 350–445 img/s across the 12 epochs sampled
  (0–3, 96–99, 196–199) against the 974–981 img/s seed 0 measured uncontended
  on Hamster, i.e. 36–45 % of the uncontended rate for the whole run. The
  sampled training loop averages ≈110.2 s/epoch, leaving **≈34 s/epoch outside
  the loop** — which independently reproduces seed 1's ≈32 s and is consistent
  with this arm running two validation passes per epoch (student and EMA).
  Contention changes only wall clock, not results: `deterministic: true`, fixed
  seeds and a fixed batch size make the numbers independent of throughput. This
  extends the Ferret/Hamster packing table: `r18_trades_adr` is now 7.70 h (s1)
  and 8.03 h (s2) on Ferret against 4.28 h (s0) on Hamster — **1.8×/1.9×**,
  the same ratio the previous entry recorded. Campaign training cost as
  executed rises to **≈96.4 GPU-h for 19 of 20 runs** (Ferret 54.9 h over 7),
  against the plan's 55.6 GPU-h for all 20; the caveat in the previous entry
  stands — 55.6 was a serial estimate, and concurrency inflates the aggregate.
  **This run's own evaluation is already in flight — do not launch or retry
  it.** `eval-0f548c3b6b3edadb6524`, opened 09:09:34Z from the pinned worktree
  `source-cd0b571e4685`, `checkpoints: both`, `autoattack: true`, official test
  split, CE-PGD-20 at `8/255`, `evaluation_attack` seed 0, and its lineage is
  hash-bound to this training run: `evaluation-lineage.json` carries
  `training_config_hash 350881af2c90…`, identical to the training manifest's
  own `config_hash`, with an empty migration list and `source_method_id ==
  runtime_method_id == adr_trades`. The **EMA-weights** pass for this seed is
  still owed.
  Campaign state at the 09:12Z scan (`--include-hand-run`, fresh unused cursor
  — 38 bundles, 29 terminal, all successful): **19 of 20 training runs terminal
  and successful**. With this run, arms 1–7 are complete at their full contract
  seed counts and **`cifar10_mobilenetv2_adr-s0` is the only training run left
  in the campaign** — it has never been launched. Contract (model-weights)
  evaluations **6 of 20** terminal, unchanged; **9 running**; and four
  training-terminal runs have no evaluation of any kind at all
  (`r18_trades-s1`, `r18_trades-s2`, `mobilenetv2_pgd_at-s0`,
  `mobilenetv2_adr-s2`). EMA-weights evaluations **1 of 9** terminal
  (`r18_adr-s0`). Note again that **the watcher exposes no progress clock for
  evaluation jobs** (`age_seconds` and `progress_timestamp` are null on all
  nine), so staleness cannot be judged from those bundles alone; do not
  relaunch any of them. Launching `mobilenetv2_adr-s0` and queueing the four
  missing evaluations plus the eight outstanding EMA passes are decisions for
  the human — this postrun launches nothing. M1c stays unticked.
- 2026-09-10 11:01Z: **failure postrun — `eval-5709dc2e3701675af39e`
  (`cifar10_r18_adr-s2`, `train/evaluation`, `--weights=model`) died of CUDA
  OOM inside AutoAttack. Nothing imported, nothing retried, no milestone
  ticked.** Terminal status re-derived from the bundle the watcher named
  (`train/evaluation/run-bundle/manifest.json`): `terminal true`,
  `success false`, `failure_class unknown`. Wall clock 08:59:10Z → 11:01:18Z,
  **2 h 02 min 08 s**, from the pinned worktree `source-cd0b571e4685`.

  **`unknown` here does not mean the cause is unclear.** `campaign_watch.py`
  takes `failure_class` from the orchestrator's per-attempt evidence
  (`scripts/ardx/campaign_watch.py:111`); a hand-run bundle has no attempt
  record, so the hand-run path can only ever emit `unknown`. The lane log
  resolves it unambiguously, and this one is `technical_retryable` in
  substance.

  **Root cause — an unbatched forward over the whole test set, not a capacity
  limit.** `torch.OutOfMemoryError` raised at
  `src/ard/evaluation/autoattack.py:213`:

  ```python
  adversarial = adversary.run_standard_evaluation(images, labels, bs=batch_size)
  with torch.no_grad():
      accuracy = model(adversarial).argmax(1).eq(labels).float().mean().item()   # line 213
  ```

  `bs=128` is honoured *inside* AutoAttack; the accuracy recomputation on the
  next line ignores it and pushes all 10,000 adversarial images through the
  student in **one** forward. The allocation that failed was **2.44 GiB**,
  which is exactly `10000 × 64 × 32 × 32 × 4 B = 2.441 GiB` — ResNet-18
  `layer1`'s activation for the full set (the traceback's innermost frame is
  `registry.py:75`, `self.bn2(self.conv2(outputs))` inside `layer1`). So the
  peak sits at the very end of a two-hour job, after all four AutoAttack
  stages have already been paid for, and it scales with the test-set size
  rather than with `autoattack_batch_size`. Both files are hash-identical
  between this checkout and the pinned SHA (`autoattack.py`
  `4d110410…`, `cli/evaluate.py` `33cff93e…`), so the code read here is the
  code that ran.

  **What tipped it over.** GPU contention from the 9-lane hand-run evaluation
  driver: at the failure the 4090 had 23.52 GiB total and **155 MiB free**,
  across three processes (1.73 + 8.46 + 13.13 GiB). The unbatched forward
  needs several GiB of headroom it cannot get when two other AutoAttack
  evaluations are resident on the same device.

  **Scope — at least three arms, not one.** `canon-eval-lane-E.log` records
  the identical traceback for `cifar10_r18_trades_adr-s1` (`exit=1`,
  20:01:15+09:00), three seconds before lane F's `exit=1` at 20:01:19+09:00.
  At the 11:05Z rescan `cifar10_r18_adr-s1/train/evaluation` (lane D) is
  `failed` as well.

  **Second defect — the lane driver does not stop on a non-zero exit.** Both
  lanes started their `--weights=ema` pass immediately after `exit=1`
  (`[lane-E][eval-start] … weights=ema 20:01:15`, `[lane-F][…] 20:01:19`), so
  a failed model-weights pass silently advances to the next stage instead of
  halting the lane. Both EMA passes were then SIGTERM'd (`exit=143`) at
  20:04:24+09:00.

  **Nothing importable was produced, and the log numbers are not results.**
  `results.append(...)` runs only after `run_autoattack` returns
  (`src/ard/cli/evaluate.py:411-421`) and the metrics file is written after
  the checkpoint loop, so the clean and CE-PGD-20 numbers for `best.pt` were
  computed and then lost together with AutoAttack; `last.pt` was never
  reached. The lane log's AutoAttack intermediates for `best.pt` — initial
  accuracy 83.32 %, 48.22 % after APGD-T, 48.22 % after FAB-T, Square at batch
  2/38 when it died — are log scrapes from a run that never completed and
  **must not be entered into any record, report or table**.

  **The evidence directory has since been removed.** At the 11:05Z rescan
  `cifar10_r18_adr-s2/train/evaluation` and `…/evaluation-ema` no longer
  exist, and neither do either of `cifar10_r18_trades_adr-s1`'s evaluation
  directories; only the training bundles remain. The failure evidence now
  survives only in `canon-eval-lane-{D,E,F}.log` and in this entry, which is
  why it is recorded here in full.

  **One proposed retry command — not run by this postrun.** It must go on an
  otherwise idle 4090, because a retry under the same packing will hit the
  same allocation at the same point after another two hours:

  ```bash
  cd /home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/worktrees/source-cd0b571e4685
  RUNS=/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/adr-cifar10-campaign-v1
  CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src \
    /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.evaluate \
    --config configs/evaluation/autoattack_saved_checkpoint.yaml \
    --checkpoint-dir "$RUNS/cifar10_r18_adr-s2/train" \
    --output "$RUNS/cifar10_r18_adr-s2/train/evaluation" \
    --weights model --allow-autoattack
  ```

  `configs/evaluation/autoattack_saved_checkpoint.yaml` is hash-identical at
  the pinned SHA (`2c94bc05…`), and its `checkpoints: both`, `seed: 0`,
  `autoattack_batch_size: 128` reproduce the failed run's manifest exactly —
  so this re-runs the same measurement, not a weakened one. Whether to retry
  as-is, or to batch the line-213 forward first so the campaign's remaining
  ~13 AutoAttack evaluations stop being contention-fragile, is the human's
  call; see the decision packet. M1c and M2 stay unticked.
- 2026-09-10 11:07Z: postrun of the `eval-4615ede05d4f96adb48f` terminal
  event — the `--weights=model` contract evaluation of
  `cifar10_mobilenetv2_adr-s1` (bundle
  `runs/adr-cifar10-campaign-v1/cifar10_mobilenetv2_adr-s1/train/evaluation/run-bundle/manifest.json`).
  **Nothing imported — no milestone closed.** Watcher rescan of that bundle
  (`--once --emit-existing --include-hand-run`): `terminal=true`,
  `success=false`, `status=failed`, `failure_class=unknown`, error marker
  "application failure recorded". `failure_class` is `unknown` only because a
  hand-run bundle carries no per-attempt classification — the log evidence
  below is unambiguously technical and retryable, and it is the **same defect
  already written up in packet 0010**, not a new one.

  **This is the fourth arm to die at `autoattack.py:213`, and the first
  MobileNetV2 one.** Traceback in `canon-eval-lane-B.log:167-248`, innermost
  frame `torchvision/models/mobilenetv2.py:64 return self.conv(x)` inside an
  `InvertedResidual`, reached from `evaluate.py:411 run_autoattack` →
  `autoattack.py:213`. The failed allocation was **3.66 GiB**, which is exactly
  `10000 × 96 × 32 × 32 × 4 B = 3.662 GiB`: `build_architecture` sets
  `model.features[0][0].stride = (1,1)` for `mobilenet_v2_cifar`
  (`src/ard/models/registry.py:126`), so the stem keeps 32×32, `features[1]`
  emits 16×32×32, and `features[2]`'s expansion conv (expand_ratio 6) emits
  **96**×32×32 for the whole test set at once. Same mechanism as the ResNet-18
  failures, whose 2.44 GiB matched `10000 × 64 × 32 × 32 × 4 B`; MobileNetV2
  simply demands **1.5×** the peak, so this arm is the *most* exposed of the
  eight, not the least.

  **The OOM is in the redundant recompute alone — AutoAttack itself finished.**
  The lane log shows Square running to `34/34` and AutoAttack printing its own
  `robust accuracy after SQUARE: 42.84% (total time 7656.5 s)` (the traceback
  appears earlier in the file only because stderr flushed ahead of buffered
  stdout). So `run_standard_evaluation` returned and the process then died on
  line 213's unbatched `model(adversarial)`. That 42.84 % is **a log scrape
  from a run that never completed and must not enter any record, report or
  table** — no `evaluation-results.json`, no `autoattack-best.json`, no
  `completion.json` was written, and `last.pt` was never reached. On disk the
  evaluation dir holds only `panel-best.jsonl`, `sample-stats-best.parquet`,
  `resolved_evaluation_config.yaml` and `evaluation-lineage.json`; a completed
  evaluation (e.g. `cifar10_r18_pgd_at-s1`) additionally has
  `autoattack-best.json`, `panel-last.jsonl`, `sample-stats-last.parquet`,
  `autoattack-last.json`, `evaluation-results.json`, `run-bundle/metrics.jsonl`
  and `run-bundle/completion.json`. 2 h 08 m of GPU time (created_at
  08:59:09Z → finished_at 11:07:25Z) produced nothing importable.

  **Contention, again from the 9-lane driver.** At the failure GPU 0 had
  23.52 GiB total, **2.93 GiB free**, and *five* resident processes
  (13.12 + 1.74 + 1.74 + 1.71 GiB plus this run's 2.25 GiB).

  **Packet 0010's Option-C trigger has fired.** Option C's preregistered rule
  was: *"if even one of the 7 in-flight evaluations dies of the same OOM,
  Option A becomes mandatory."* This run was one of those seven. Two further
  lanes are also `exit=1` with the identical traceback
  (`canon-eval-lane-{D,E,F}.log`), and lanes E and F's EMA passes were
  SIGTERM'd (`exit=143`) at 20:04:24+09:00 — the second defect (the lane driver
  not halting on a non-zero exit) recurs here too: lane B started
  `--weights=ema` (`eval-74a8f5d0042ebea0b0d1`) at 20:07:26+09:00, three
  seconds after `exit=1`.

  **One proposed retry command — not run by this postrun.** As with the other
  three arms it must go on an otherwise idle 4090; under the present packing it
  would hit the same allocation at the same point after another two hours, and
  for MobileNetV2 it needs 3.66 GiB of headroom rather than 2.44 GiB:

  ```bash
  cd /home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/worktrees/source-cd0b571e4685
  RUNS=/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/adr-cifar10-campaign-v1
  CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src \
    /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.evaluate \
    --config configs/evaluation/autoattack_saved_checkpoint.yaml \
    --checkpoint-dir "$RUNS/cifar10_mobilenetv2_adr-s1/train" \
    --output "$RUNS/cifar10_mobilenetv2_adr-s1/train/evaluation" \
    --weights model --allow-autoattack
  ```

  This reproduces the failed run's manifest (`config_hash`
  `2cab228100f2…`, protocol `controlled_cifar10_mobilenetv2_adr_v1`,
  `evaluation_seed 0`, `world_size 1`, effective global batch 128, source SHA
  `cd0b571e4685` with an empty diff) — the same measurement, not a weakened
  one. Whether to retry as-is or to batch the line-213 forward first is the
  human's call; the question is packet 0010's, and this entry is the evidence
  its Option-C rule asked for. M1c and M2 stay unticked.
- 2026-09-10 11:09Z: postrun of the `eval-e2da1d97fcd3e5932fab` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_adr-s1/train/evaluation/`) — arm 3
  (`adr`) seed 1, model weights, lane D. **Nothing imported — no milestone
  closed.** `docs/experiments/` still holds only `.gitkeep` and `historical/`,
  so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan returns `terminal: true`,
  `success: false`, `status: failed`, `failure_class: unknown` on the line
  whose `path` is that manifest, with `completion_json: false` and error
  marker "application failure recorded". `unknown` carries **no information**
  here: `scripts/ardx/ardx_common.py:357` hard-codes that literal for every
  failed hand-run bundle, because only an orchestrator attempt record can
  carry a real classification. The classification therefore comes from the log,
  and the log is unambiguous — **`technical_retryable`**, the same
  `autoattack.py:213` defect already written up in packet 0010 and in the two
  entries above. Nothing scientific is in question: the checkpoint, the
  protocol and the threat identity are untouched.
  This run's own numbers, for the record: traceback in
  `canon-eval-lane-D.log:167-227`, innermost frame `registry.py:75`
  (`self.bn2(self.conv2(outputs))` inside `layer1`), failed allocation
  **2.44 GiB** = `10000 × 64 × 32 × 32 × 4 B`, with **1.89 GiB free** of
  23.52 GiB and a co-resident 13.13 GiB process (PID 2223692) on the same
  physical 4090. Wall clock `created_at` 08:59:09Z → `finished_at` 11:04:46Z,
  2 h 05 m, `exit=1` at 20:04:47+09:00.
  **AutoAttack finished here too.** `canon-eval-lane-D.log` shows Square
  running to `39/39` and AutoAttack printing its own `robust accuracy after
  SQUARE: 48.65% (total time 7490.2 s)` / `robust accuracy: 48.65%`; the
  traceback sits earlier in the file only because stderr flushed ahead of
  buffered stdout. **That 48.65 % is a log scrape from a run that never
  completed and must not enter any record, report or table** — no
  `autoattack-best.json`, no `evaluation-results.json`, no `completion.json`
  was written, and `last.pt` was never reached.
  Worth noting only as a reproducibility check, not as evidence: it equals the
  `early-check-eval-aa-best-model` AutoAttack figure 0.4865 already logged
  above for the *same* `best.pt` (sha `9181001647cb…`). Two separate processes
  agree to four digits, which is what a fixed `evaluation_attack: 0` on a saved
  checkpoint should give — so a retry is expected to land on the same number,
  and the two hours are being re-paid purely to obtain a written record and the
  missing `last.pt` half.
  **This run's evidence directory still exists**, unlike `r18_adr-s2`'s and
  `r18_trades_adr-s1`'s, which were gone by the 11:05Z rescan. On disk:
  `run-bundle/manifest.json` (status `failed`, `failure_snapshot` with five
  hashed files), `run-bundle/error-marker.txt`, `panel-best.jsonl`,
  `sample-stats-best.parquet`, `resolved_evaluation_config.yaml`,
  `evaluation-lineage.json` and the offline W&B segment
  `offline-run-20260910_175910-eval-e2da1d97fcd3e5932fab`. A retry therefore
  **must not write into that path in place** — move it aside first, both to
  keep the failure evidence and because a record path is written once.
  **One proposed retry command — not run by this postrun.** It must go on an
  otherwise idle 4090, or it will hit the same allocation at the same point
  after another two hours:

  ```bash
  cd /home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/worktrees/source-cd0b571e4685
  RUNS=/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/adr-cifar10-campaign-v1
  mv "$RUNS/cifar10_r18_adr-s1/train/evaluation" \
     "$RUNS/cifar10_r18_adr-s1/train/evaluation.failed-oom-eval-e2da1d97" && \
  CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src \
    /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.evaluate \
    --config configs/evaluation/autoattack_saved_checkpoint.yaml \
    --checkpoint-dir "$RUNS/cifar10_r18_adr-s1/train" \
    --output "$RUNS/cifar10_r18_adr-s1/train/evaluation" \
    --weights model --allow-autoattack
  ```

  This reproduces the failed run's manifest — `config_hash`
  `90da37ed2020d6f2…`, protocol `controlled_cifar10_r18_adr_v1` (Nesterov on),
  `evaluation_seed 0`, training seeds all 1 except the fixed `split=20260722`
  and `evaluation_attack=0`, `world_size 1`, effective global batch 128, source
  SHA `cd0b571e4685` with an empty diff — so it re-runs the same measurement,
  not a weakened one. The rename keeps the aggregator blind to the failed
  attempt (it reads exactly `<run>/train/evaluation/`), but the watcher will
  emit one more terminal-failed event under the new path; that event's answer
  is this entry, not a second postrun.
  Whether to retry as-is or to batch the line-213 forward first is the human's
  call and belongs to packet 0010, whose Option-C trigger this run also fires.
  M1c and M2 stay unticked.
- 2026-09-10 11:21Z: postrun of the `eval-7f2c13eb1eb778fd2e2a` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_trades_49k_validation-s0/train/
  evaluation/run-bundle/manifest.json`) — **arm 6, the 49k-validation TRADES
  pilot**, model weights, lane C. **Nothing imported — no milestone closed.**
  `docs/experiments/` still holds only `.gitkeep` and the M1b `historical/`
  archive, so the idempotency check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of that bundle's parent
  returns `terminal: true`, `success: false`, `status: failed`,
  `failure_class: unknown`, `completion_json: false`, error marker
  "application failure recorded" on the line whose `path` is exactly the
  `--state-path` the watcher passed. As in the two entries above, `unknown`
  carries **no information**: `scripts/ardx/ardx_common.py:357` returns that
  literal for every failed bundle (`"unknown" if terminal and success is False
  else None`), because only an orchestrator attempt record can carry a real
  classification. The classification comes from the log, and the log is
  unambiguous — **`technical_retryable`**, the same `autoattack.py:213` defect
  as packet 0010. Nothing scientific is in question: checkpoint, protocol and
  threat identity are untouched.
  **Fifth identical failure, and the first outside the ADR family.** The four
  before it (`r18_adr-s2`, `r18_trades_adr-s1`, `r18_adr-s1`,
  `mobilenetv2_adr-s1`) all ran an `adr`-family method. This one is plain
  TRADES with `method.adr: null`, which removes the last reason to suspect the
  method: the allocation is set by the **test-set size × channel count**, not
  by the arm. Evidence: traceback in `canon-eval-lane-C.log:167-227`, failed
  allocation **2.44 GiB** = `10000 × 64 × 32 × 32 × 4 B`, identical to the two
  ResNet-18 instances above. The innermost frame here is `registry.py:74`
  (`self.relu(self.bn1(self.conv1(inputs)))`) rather than `registry.py:75`
  (`self.bn2(self.conv2(outputs))`) — the same `layer1` BasicBlock, the same
  allocation size; which of the two batch-norms trips first depends only on
  where the allocator happens to run out, not on anything scientific.
  GPU 0 at the moment of failure: **1.07 GiB free** of 23.52 GiB, this process
  holding 6.02 GiB, with three co-resident processes on the same physical 4090
  (PID 2221244 1.74 GiB, PID 2223691 13.13 GiB, PID 2298430 1.54 GiB). This is
  the tightest of the five (1.07 GiB free here against 1.89 for `r18_adr-s1`
  and 2.93 for `mobilenetv2_adr-s1`). Wall clock `created_at` 08:59:09.846Z →
  `finished_at` 11:20:48.795Z = **2 h 21 m 39 s**, `exit=1` at 20:20:49+09:00.
  **AutoAttack finished here too, and none of its output is usable.** The lane
  log records the whole standard cascade completing: initial accuracy 79.85 %,
  `robust accuracy after APGD-CE: 50.76%` (161.4 s), `after APGD-T: 47.62%`
  (1408.5 s), `after FAB-T: 47.62%` (4538.7 s), `after SQUARE: 47.62%`
  (8459.2 s), then `max Linf perturbation: 0.03137, nan in tensor: 0` and
  `robust accuracy: 47.62%`. The traceback sits earlier in the file only
  because stderr flushed ahead of buffered stdout. **That 47.62 % is a log
  scrape from a run that never completed and must not enter any record, report
  or table** — no `evaluation-results.json`, no `autoattack-best.json`, no
  `completion.json` was written, and `last.pt` was never reached. `best.pt`'s
  clean and CE-PGD-20 numbers were computed and thrown away with it, since
  `results.append(...)` runs only after `run_autoattack` returns
  (`src/ard/cli/evaluate.py:411-421`). Unlike `r18_adr-s1`, this arm has no
  earlier `early-check-eval-*` run to compare the scraped figure against, so
  there is not even a reproducibility cross-check to note.
  **The cost of this one is not interchangeable with the others.** Arm 6 has
  exactly one seed by contract (see the arms table, row 6), so this was the
  only evaluation the 49k-validation pilot will ever have, and the preregistered
  comparison it feeds — official-test AutoAttack against arm 4 seed 0's
  archived 47.87 best / 44.99 last — cannot be formed at all until it is
  re-run. The arm's own training postrun entry above deliberately refused to
  read a verdict out of validation PGD at n=1,000; that refusal stands, and
  this failure is the reason the licensed number is still missing.
  **Capture now: three of the five failure directories are already gone.**
  `r18_adr-s2` and `r18_trades_adr-s1` were gone by the 11:05Z rescan, and
  `cifar10_r18_adr-s1/train/` now contains only `early-check-eval-model/`,
  `early-check-eval-ema/`, `early-check-eval-aa-best-model/` and
  `evaluation-ema/` — its `evaluation/` directory has been **deleted** since
  the 11:09Z entry, and there is no `evaluation.failed-oom-eval-e2da1d97`, so
  it was removed rather than moved aside. The numbers transcribed into that
  entry are now the only surviving record of it. This run's evidence directory
  **does** still exist, and holds `run-bundle/manifest.json` (status `failed`,
  `failure_snapshot` with five hashed files, `directory_digest
  c7fc8e0fa923…`), `run-bundle/error-marker.txt`, `panel-best.jsonl`,
  `sample-stats-best.parquet`, `resolved_evaluation_config.yaml`,
  `evaluation-lineage.json` and the offline W&B segment
  `offline-run-20260910_175910-eval-7f2c13eb1eb778fd2e2a`. Treat it as
  perishable.
  **Evaluation census on disk at 11:21Z** (from the manifests themselves, not
  from the event): 18 evaluation directories exist under the campaign root —
  **7 succeeded** (`manifest.status=sync_pending` with `evaluation-results.json`
  present: `r18_pgd_at_nesterov-s0/-s1/-s2`, `r18_pgd_at-s1/-s2`, `r18_adr-s0`
  model and ema), **2 failed and still present** (`mobilenetv2_adr-s1` model,
  and this run), **9 still `running`** (`mobilenetv2_pgd_at-s0/-s1/-s2`,
  `mobilenetv2_adr-s1` ema, `mobilenetv2_adr-s2`, `r18_trades_adr-s0/-s2`,
  `r18_trades-s2`, `r18_adr-s1` ema). Every one of those 9 is exposed to the
  same defect at the end of its own two-hour run.
  **The lane-driver defect did not bite here, for an arm-specific reason.**
  Lane C started `cifar10_mobilenetv2_pgd_at-s0 weights=model` at
  20:20:49+09:00, the same second as the `exit=1` — so the driver still does
  not stop on a non-zero exit. It cost nothing this time only because arm 6 has
  no EMA (`method.adr: null`, no `best-ema.pt` on disk, confirmed in this arm's
  training postrun entry above), so the next queue item was an unrelated run
  rather than the ema half of the run that had just died. That remains an infra
  matter for a separate plan, per CLAUDE.md rule 3; it is not decided here.
  **One proposed retry command — not run by this postrun.** It must go on an
  otherwise idle 4090, or it will hit the same allocation at the same point
  after another two hours and twenty minutes:

  ```bash
  cd /home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/worktrees/source-cd0b571e4685
  RUNS=/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/adr-cifar10-campaign-v1
  mv "$RUNS/cifar10_r18_trades_49k_validation-s0/train/evaluation" \
     "$RUNS/cifar10_r18_trades_49k_validation-s0/train/evaluation.failed-oom-eval-7f2c13eb" && \
  CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src \
    /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.evaluate \
    --config configs/evaluation/autoattack_saved_checkpoint.yaml \
    --checkpoint-dir "$RUNS/cifar10_r18_trades_49k_validation-s0/train" \
    --output "$RUNS/cifar10_r18_trades_49k_validation-s0/train/evaluation" \
    --weights model --allow-autoattack
  ```

  This reproduces the failed run's manifest — `config_hash e3f5a60d269e…`,
  protocol `controlled_cifar10_r18_trades_49k_validation_v1`, `teacher: null`,
  `evaluation_seed 0`, training seeds all 0 except the fixed `split=20260722`
  and `evaluation_attack=0`, `world_size 1`, effective global batch 128, source
  SHA `cd0b571e4685` with `dirty: false` and empty diff `e3b0c442…`, external
  lock `05cfce4cf8db…` — so it re-runs the same measurement, not a weakened
  one. `evaluation-lineage.json` binds it to the right training run
  (`training_config_hash 45b0eaa50b89…`, equal to that run's manifest
  `config_hash`). The rename keeps the aggregator blind to the failed attempt
  (it reads exactly `<run>/train/evaluation/`) while preserving the perishable
  evidence above; the watcher will emit one more terminal-failed event under
  the new path, and that event's answer is this entry, not a second postrun.
  Whether to retry as-is or to batch the line-213 forward first is the human's
  call and belongs to packet 0010, whose Option-C trigger this run fires for
  the third time. M1c and M2 stay unticked.
- 2026-09-10 11:27Z: postrun of the `eval-6b2c6486fd99df2da5ac` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_r18_trades_adr-s0/train/evaluation/
  run-bundle/manifest.json`) — **arm 5 (ADR-TRADES) seed 0**, model weights,
  lane A. **Nothing imported — no milestone closed.** `docs/experiments/` still
  holds only `.gitkeep` and the M1b `historical/` archive, so the idempotency
  check had nothing to collide with.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of that bundle's parent
  returns `terminal: true`, `success: false`, `status: failed`,
  `failure_class: unknown`, `completion_json: false`, error marker
  "application failure recorded" on the line whose `path` is exactly the
  `--state-path` the watcher passed. As in the four entries above, `unknown`
  carries **no information** — `scripts/ardx/ardx_common.py:357` returns that
  literal for every failed hand-run bundle. The classification comes from the
  log and is unambiguous: **`technical_retryable`**, the same
  `autoattack.py:213` defect as packet 0010. Nothing scientific is in question;
  checkpoint, protocol and threat identity are untouched.
  **Sixth identical failure, and the tightest one yet.** The five before it are
  `r18_adr-s2`, `r18_trades_adr-s1`, `r18_adr-s1`, `mobilenetv2_adr-s1` and
  `r18_trades_49k_validation-s0`. Traceback in `canon-eval-lane-A.log:167-227`,
  innermost frame `registry.py:75` (`self.bn2(self.conv2(outputs))` inside
  `layer1`), failed allocation **2.44 GiB** = `10000 × 64 × 32 × 32 × 4 B` —
  identical to the three other ResNet-18 instances. GPU 0 at the moment of
  failure had **378.56 MiB free** of 23.52 GiB, this process holding 8.46 GiB,
  with two co-resident processes on the same physical 4090 (PID 2223691
  13.13 GiB, PID 2298430 1.54 GiB). Free memory across the six now reads
  0.37 (this) < 1.07 (`49k_validation-s0`) < 1.89 (`r18_adr-s1`) < 2.93 GiB
  (`mobilenetv2_adr-s1`); the failure does not need contention to be extreme,
  it needs only less than ~2.5 GiB of headroom. Wall clock `created_at`
  08:59:09.895Z → `finished_at` 11:24:10.284Z = **2 h 25 m 00 s**, the longest
  of the six, `exit=1` at 20:24:11+09:00.
  **AutoAttack finished here too, and none of its output is usable.** The lane
  log records the whole standard cascade completing: initial accuracy 83.68 %,
  `robust accuracy after APGD-CE: 53.20%` (168.1 s), `after APGD-T: 49.17%`
  (1460.2 s), `after FAB-T: 49.17%` (4641.8 s), `after SQUARE: 49.17%`
  (8660.3 s), then `max Linf perturbation: 0.03137, nan in tensor: 0` and
  `robust accuracy: 49.17%`. The traceback sits earlier in the file only
  because stderr flushed ahead of buffered stdout. **That 49.17 % is a log
  scrape from a run that never completed and must not enter any record, report
  or table** — no `autoattack-best.json`, no `evaluation-results.json`, no
  `completion.json` was written, and `last.pt` was never reached. `best.pt`'s
  clean and CE-PGD-20 numbers were computed and thrown away with it, since
  `results.append(...)` runs only after `run_autoattack` returns
  (`src/ard/cli/evaluate.py:411-421`). Like arm 6 and unlike `r18_adr-s1`, this
  arm has no earlier `early-check-eval-*` run, so there is not even a
  reproducibility cross-check to note.
  **Arm 5 now has zero official-test numbers, at 3/3 training complete.** Seed
  1 died of this same defect (lane E, 20:01:15+09:00) and its evaluation
  directory has since been deleted; seed 0 is this run; seed 2's evaluation
  manifest still reads `running`. So the arm whose training is finished and
  whose validation diagnostics looked most favourable is the arm with the least
  evidence. This does not block the campaign's decisive leg — that is arm 3 vs
  arm 2 on ResNet-18 — but it blocks the arm-5-vs-arm-4 TRADES leg entirely,
  and that leg was already only directional (seed and optimizer both differ,
  per this arm's own training postrun entry above).
  **The lane-driver defect bit here, and it leaves a bundle that will never go
  terminal.** Lane A started `weights=ema` at 20:24:11+09:00 — the same second
  as the `exit=1` — as `eval-77f66adc6ec58702c412` into `evaluation-ema/`, and
  that process was SIGTERM'd (`exit=143`) 56 s later at 20:25:07+09:00. Its
  `run-bundle/manifest.json` is frozen at `"status": "running"` with
  `created_at 11:24:13.628Z`, **no `finished_at` and no `error-marker.txt`**, so
  `campaign_watch.py` reports `terminal: false` for it and will keep doing so
  indefinitely: a SIGTERM writes no terminal state, unlike an exception.
  Verified as a class, not a one-off — `cifar10_mobilenetv2_adr-s1`'s ema
  bundle (`eval-74a8f5d0042ebea0b0d1`, `created_at 11:07:28.531Z`) was
  SIGTERM'd at 20:10:21+09:00 and still read `"status": "running"` at 11:26Z,
  16 minutes dead. **This corrects the 11:21Z census above**: at least two of
  the nine evaluation directories it counted as "still running" are not running
  at all, and a `running` manifest is from here on not sufficient evidence that
  a run is alive — the lane log's `[eval-done] … exit=` line is. Both defects
  (the driver not halting on a non-zero exit, and SIGTERM leaving a
  permanently non-terminal bundle) are infra matters for a separate plan per
  CLAUDE.md rule 3; neither is decided here.
  **This run's evidence directory still exists — treat it as perishable.**
  Three of the six failure directories are already gone (`r18_adr-s2`,
  `r18_trades_adr-s1`, and `r18_adr-s1`, which was deleted rather than moved
  aside). On disk here: `run-bundle/manifest.json` (status `failed`,
  `failure_snapshot` with five hashed files, `directory_digest
  20e54e2be819…`), `run-bundle/error-marker.txt`, `panel-best.jsonl`,
  `sample-stats-best.parquet`, `resolved_evaluation_config.yaml`,
  `evaluation-lineage.json` and the offline W&B segment
  `offline-run-20260910_175910-eval-6b2c6486fd99df2da5ac`.
  **One proposed retry command — not run by this postrun.** It must go on an
  otherwise idle 4090, or it will hit the same allocation at the same point
  after another two and a half hours:

  ```bash
  cd /home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/worktrees/source-cd0b571e4685
  RUNS=/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap/runs/adr-cifar10-campaign-v1
  mv "$RUNS/cifar10_r18_trades_adr-s0/train/evaluation" \
     "$RUNS/cifar10_r18_trades_adr-s0/train/evaluation.failed-oom-eval-6b2c6486" && \
  CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src \
    /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.evaluate \
    --config configs/evaluation/autoattack_saved_checkpoint.yaml \
    --checkpoint-dir "$RUNS/cifar10_r18_trades_adr-s0/train" \
    --output "$RUNS/cifar10_r18_trades_adr-s0/train/evaluation" \
    --weights model --allow-autoattack
  ```

  This reproduces the failed run's manifest — `config_hash ecb8369d1244…`,
  protocol `controlled_cifar10_r18_adr_v1` (Nesterov on), method `adr_trades`
  v1, `teacher: null`, `evaluation_seed 0`, training seeds all 0 except the
  fixed `split=20260722` and `evaluation_attack=0`, `world_size 1`, effective
  global batch 128, source SHA `cd0b571e4685` with `dirty: false` and empty
  diff `e3b0c442…`, external lock `05cfce4cf8db…`. The resolved evaluation
  config confirms the protocol is not weakened: `checkpoints: both`, test
  split, `autoattack: true`, `autoattack_batch_size: 128`, evaluation attack
  CE-PGD-20 at `epsilon 8/255`, `step 2/255`, `random_start: true`, identity
  normalization. `configs/evaluation/autoattack_saved_checkpoint.yaml`
  (`2c94bc05…`), `src/ard/evaluation/autoattack.py` (`4d110410…`) and
  `src/ard/cli/evaluate.py` (`33cff93e…`) are hash-identical between this
  checkout and the pinned SHA, so the code read here is the code that ran.
  `evaluation-lineage.json` binds it to the right training run
  (`training_config_hash 7d9f2051b774…`, equal to that run's manifest
  `config_hash`). The rename keeps the aggregator blind to the failed attempt
  (it reads exactly `<run>/train/evaluation/`) while preserving the perishable
  evidence above; the watcher will emit one more terminal-failed event under
  the new path, and that event's answer is this entry, not a second postrun.
  Whether to retry as-is or to batch the line-213 forward first is the human's
  call and belongs to packet 0010, whose Option-C trigger this run fires for
  the fourth time; `chosen` there is still null. M1c and M2 stay unticked.
- 2026-09-10 11:47Z: postrun of the `eval-e9f181f91ba3fc4a0ca0` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_mobilenetv2_pgd_at-s2/train/evaluation/
  run-bundle/manifest.json`) — **arm 7 (MobileNetV2 PGD-AT) seed 2**, model
  weights, lane H. **This one succeeded.** It is the first contract evaluation
  in this campaign to finish since the six OOM failures above, and the **first
  MobileNetV2 evaluation of any kind**. **Still nothing imported and no
  milestone closed** — see "why no import" below; that is a property of the
  contract, not of this run.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of that bundle's parent
  returns `terminal: true`, `success: true`, `status: completed`,
  `failure_class: null`, `completion_json: true`, error marker "no application
  error recorded" on the line whose `path` is exactly the `--state-path` the
  watcher passed. The lane log agrees: `[lane-H][eval-done]
  cifar10_mobilenetv2_pgd_at-s2 weights=model exit=0 2026-09-10T20:46:45+09:00`.
  Wall clock `created_at` 09:00:58.554Z → `finished_at` 11:46:44.881Z =
  **2 h 45 m 46 s**.
  **The hand-run completion contract is satisfied in full.**
  `run-bundle/completion.json` reads `{"status": "completed", "results": 2}`;
  the manifest reads `status: sync_pending` (an accepted terminal value — the
  offline W&B segment has not been synced, which is a tracking matter and not a
  scientific one); `error-marker.txt` is the clean marker. All **seven declared
  artifacts exist** at their declared paths — `resolved_evaluation_config.yaml`,
  `evaluation-lineage.json`, `evaluation-results.json`, `panel-best.jsonl`,
  `panel-last.jsonl`, `sample-stats-best.parquet`, `sample-stats-last.parquet`
  — each also present under its content-addressed
  `run-bundle/artifacts/<name>/<sha256>/` copy, plus `autoattack-best.json` and
  `autoattack-last.json`. (The per-file re-hash that
  `aggregate_adr_cifar10_replication.py::_verify_bundle` performs was not run
  here: this session's Bash is confined to the repo checkout and cannot hash
  files under the runtime root. The aggregator will do it at M3, from the
  pinned worktree, and that is the check of authority.)
  **The numbers, both checkpoints, official 10,000-example test split.**

  | checkpoint | clean | CE-PGD-20 | AutoAttack (record) | AutoAttack (own print) |
  |---|---|---|---|---|
  | `best.pt` (`8071280f…`) | 73.23 % | 44.40 % | 40.18 % | 40.13 % |
  | `last.pt` (`2baebfb2…`) | 74.71 % | 44.24 % | 39.50 % | 39.46 % |

  Identity, checked field by field against the aggregator's `ARMS` entry rather
  than assumed: protocol `controlled_cifar10_mobilenetv2_adr_v1` ✓, runtime
  method `pgd_at` ✓, `weights: model` ✓, both `best`/`last` aliases present ✓,
  `count 10000` on `split: test` ✓, AutoAttack `version: standard` with
  `expected_commit a39220048b3c…` = the pinned upstream ✓, `has_ema: False` for
  this arm so no `evaluation-ema/` is owed ✓ (and none was started — the
  lane-driver defect from the entries above did not bite here). Source SHA
  `cd0b571e4685` with `dirty: false` and empty diff `e3b0c442…`, external lock
  `05cfce4cf8db…`. `evaluation-lineage.json` binds the evaluation to the right
  training run: `training_config_hash e8b1985b206f…`, equal to that run's own
  manifest `config_hash`. The resolved evaluation config confirms the protocol
  is not weakened: `checkpoints: both`, test split, `autoattack: true`,
  `autoattack_batch_size: 128`, evaluation attack CE-PGD-20 at `epsilon 8/255`,
  `step 2/255`, `steps: 20`, `random_start: true`, `loss: ce`, both modes
  `eval`, images in `pixel_0_1` with `cifar10_standard` normalization owned by
  the student adapter. That 20-step evaluation attack is exactly the training
  config's `selection_attack`, as the evaluation invariant requires. Seeds:
  training 2 throughout, `evaluation_attack 0`, `split 20260722`, AutoAttack
  seed 0. World size 1, per-rank batch 128, effective global batch 128.
  (`evaluation.dataset.num_samples: 16` in the resolved config is a vestigial
  field that the official path ignores — the record's `count: 10000` is what
  actually ran, and it is that field the aggregator gates on.)
  **Consistency checks, all clean.** Against the sibling seed still in flight,
  `mobilenetv2_pgd_at-s1` (lane G, `best.pt` clean 74.34 %, AutoAttack
  39.78 %): within a point of this seed on both axes, so the two MobileNetV2
  baseline seeds agree. Against the Nesterov-matched ResNet-18 baseline
  (`pgd_at_nesterov-s0/-s1` `best.pt`: clean 82.32 / 82.19 %, AutoAttack
  47.19 / 47.40 %): MobileNetV2 sits ≈ 9 pp lower clean and ≈ 7 pp lower under
  AutoAttack, which is the expected cost of the much smaller model and is in
  line with published MobileNetV2 adversarial-training numbers on CIFAR-10. The
  best/last ordering is the usual one — `last.pt` is cleaner (+1.48 pp) and less
  robust (−0.68 pp AutoAttack) than the robustness-selected `best.pt`. No
  deviation, no anomaly, nothing here needs a bug hunt.
  **One new finding, and it bears directly on packet 0010.** The AutoAttack
  number this campaign records is **not** AutoAttack's own reported robust
  accuracy. `src/ard/evaluation/autoattack.py:213` throws away AutoAttack's
  count and recomputes accuracy itself, in a single unbatched forward over all
  10,000 adversarial images, while AutoAttack computed its own number in
  batches of 128. In eval mode BatchNorm uses running statistics, so the two
  are mathematically identical; in practice cuDNN selects different kernels and
  reduction orders at batch 10,000 than at batch 128, and a handful of
  borderline examples flip. Measured across the runs that have both figures:
  `r18_pgd_at-s1` agrees exactly on both checkpoints (46.78 / 40.69 %);
  `r18_adr-s0` records **+2 examples** on each checkpoint (48.75 vs 48.73 %,
  44.81 vs 44.79 %); this run records **+5 and +4** (40.18 vs 40.13 %, 39.50 vs
  39.46 %). **Four out of four disagreements go the same way — the recorded
  number is the higher one**, i.e. the bias flatters the defence. The mechanism
  explains the direction: AutoAttack only banks a point as broken when the
  attack succeeded under batch-128 numerics, so any point that flips back under
  the 10,000-wide forward is added to the recomputed count, while points it
  banked as robust have margin and rarely flip the other way. The magnitude
  here (≤ 0.05 pp) is far below any noise floor this campaign will compare
  against, so **no recorded number is in doubt and nothing needs re-running for
  this reason alone**. What matters is the interaction with packet 0010: the
  fix that packet is weighing is *batching that exact line*, and batching it
  would move recorded AutoAttack accuracies by roughly this much. Runs
  evaluated before the fix and runs evaluated after it would then be measured
  by two slightly different estimators inside one aggregate. Two clean ways out
  — use AutoAttack's own returned robust accuracy instead of recomputing (which
  also removes the OOM entirely and is the standard practice), or batch the
  recomputation and re-evaluate the already-successful runs for uniformity.
  Both are packet 0010's call, not this postrun's; `chosen` there is still null.
  **Why no import, and it is not this run's fault.** `docs/experiments/` holds
  only `.gitkeep` and the M1b `historical/` archive, so the idempotency check
  had nothing to collide with — but `adr_cifar10_replication_v1` is a
  **campaign-level** contract. `aggregate_adr_cifar10_replication.py` takes a
  `--run-root` and walks all eight arms and every seed; it raises
  `AggregationError` on the first missing evaluation. A single arm-seed cannot
  be imported on its own, by design, and there is no per-run record path to
  write. So this run's numbers are verified and safe on disk, and they enter
  `docs/` only at M3.
  **Campaign census at 11:47Z, re-derived from a full-root watcher scan** (this
  supersedes the 11:09Z census in the Status block, which this entry also
  corrects there). Training: **19 of 20 terminal and successful**; the one
  missing is still `cifar10_mobilenetv2_adr-s0`, never launched. Contract
  evaluations, model weights: **7 of 20 complete** — `r18_pgd_at-s1/-s2`,
  `r18_pgd_at_nesterov-s0/-s1/-s2`, `r18_adr-s0`, and now
  `mobilenetv2_pgd_at-s2`. EMA weights: **1 of 9**, still only `r18_adr-s0`.
  Terminal-failed and still on disk: `mobilenetv2_adr-s1` and
  `r18_trades_adr-s0` (the other four failure directories were deleted or moved
  aside). Genuinely running: `mobilenetv2_pgd_at-s1` (lane G),
  `r18_trades-s1` (lane A), `r18_trades-s2` (lane B),
  `r18_trades_49k_validation-s0` (lane J, the retry), `r18_trades_adr-s2`
  (lane I). Reading `running` but **dead**, per the SIGTERM defect recorded in
  the entry above: the `evaluation-ema` bundles of `r18_adr-s1`,
  `mobilenetv2_adr-s1` and `r18_trades_adr-s0`. `mobilenetv2_pgd_at-s0`'s
  evaluation was SIGTERM'd 84 s in (lane C, 20:22:13+09:00) before any manifest
  was written, so it has no bundle at all and no event will ever fire for it.
  **What this does and does not unlock.** The MobileNetV2 leg is one of the two
  decisive comparison pairs (`mobilenetv2`: `mobilenetv2_adr` vs
  `mobilenetv2_pgd_at`). This run supplies the **baseline half, seed 2 only**.
  The pair still needs baseline seeds 0 and 1 and all three treatment seeds
  (model and EMA), and one of those treatment runs has not been trained yet. So
  the leg is no longer at zero, but it is not near a verdict either.
  Nothing was launched, retried or aggregated by this postrun. M1c and M2 stay
  unticked.
- 2026-09-10 12:01Z: postrun of the `eval-ef9c002c70291c816de2` terminal event
  (`runs/adr-cifar10-campaign-v1/cifar10_mobilenetv2_pgd_at-s1/train/evaluation/
  run-bundle/manifest.json`) — **arm 7 (MobileNetV2 PGD-AT) seed 1**, model
  weights, lane G. **Succeeded.** It is the second MobileNetV2 evaluation and
  the second consecutive success after the six OOM failures. **Still nothing
  imported and no milestone closed** — the "why no import" reasoning in the
  11:47Z entry applies unchanged: `adr_cifar10_replication_v1` is a
  campaign-level contract with no per-run record path.
  Status re-derived, not taken from the event: a fresh `campaign_watch.py
  --once --emit-existing --include-hand-run` scan of that bundle's parent
  returns `terminal: true`, `success: true`, `status: completed`,
  `failure_class: null`, `completion_json: true`, error marker "no application
  error recorded" on the line whose `path` is exactly the `--state-path` the
  watcher passed. The lane log agrees: `[lane-G][eval-done]
  cifar10_mobilenetv2_pgd_at-s1 weights=model exit=0 2026-09-10T21:00:38+09:00`.
  Wall clock `created_at` 09:00:58.602Z → `finished_at` 12:00:37.584Z =
  **2 h 59 m 39 s**, of which AutoAttack itself is 10,703.7 s (2 h 58 m 24 s):
  7,177.5 s on `best.pt` and 3,526.2 s on `last.pt`.
  **The hand-run completion contract is satisfied in full.**
  `run-bundle/completion.json` reads `{"status": "completed", "results": 2}`;
  the manifest reads `status: sync_pending` (an accepted terminal value — the
  offline W&B segment has not been synced, a tracking matter, not a scientific
  one); `error-marker.txt` is the clean marker. All **seven declared artifacts
  exist** at their declared paths — `resolved_evaluation_config.yaml`,
  `evaluation-lineage.json`, `evaluation-results.json`, `panel-best.jsonl`,
  `panel-last.jsonl`, `sample-stats-best.parquet`, `sample-stats-last.parquet`
  — each also present under its content-addressed
  `run-bundle/artifacts/<name>/<sha256>/` copy, plus `autoattack-best.json` and
  `autoattack-last.json`. (As in the 11:47Z entry, the per-file re-hash that
  `aggregate_adr_cifar10_replication.py::_verify_bundle` performs was not run
  here — this session's Bash cannot hash files under the runtime root. The
  aggregator will do it at M3 from the pinned worktree, and that is the check
  of authority.)
  **The numbers, both checkpoints, official 10,000-example test split.**

  | checkpoint | clean | CE-PGD-20 | AutoAttack (record) | AutoAttack (own print) |
  |---|---|---|---|---|
  | `best.pt` (`3bef7eec…`) | 74.34 % | 44.28 % | 39.86 % | 39.78 % |
  | `last.pt` (`c73f6f1d…`) | 74.30 % | 43.39 % | 39.26 % | 39.17 % |

  AutoAttack's own `initial accuracy` prints — 74.34 % and 74.30 % — match the
  recorded clean accuracies exactly, so both checkpoints were loaded as
  intended.
  Identity, checked field by field against the aggregator's `ARMS` entry rather
  than assumed: protocol `controlled_cifar10_mobilenetv2_adr_v1` ✓, runtime
  method `pgd_at` ✓, `weights: model` ✓, both `best`/`last` aliases present ✓,
  `count 10000` on `split: test` ✓, AutoAttack `version: standard` with
  `expected_commit a39220048b3c…` = the pinned upstream ✓, `has_ema: False` for
  this arm so no `evaluation-ema/` is owed ✓ (and none was started). Source SHA
  `cd0b571e4685` with `dirty: false` and empty diff `e3b0c442…`, external lock
  `05cfce4cf8db…`. `evaluation-lineage.json` binds the evaluation to the right
  training run: `training_config_hash d8e8ad98277c…`, equal to that run's own
  manifest `config_hash`. `git diff cd0b571e4685 -- src/ard/evaluation/
  src/ard/cli/evaluate.py configs/protocols/` is empty in this checkout, so the
  evaluation code read here is the code that ran. The resolved evaluation config
  confirms the protocol is not weakened: `checkpoints: both`, test split,
  `autoattack: true`, `autoattack_batch_size: 128`, evaluation attack CE-PGD-20
  at `epsilon 8/255`, `step 2/255`, `steps: 20`, `random_start: true`,
  `loss: ce`, both modes `eval`, images in `pixel_0_1` with `cifar10_standard`
  normalization owned by the student adapter. That 20-step evaluation attack is
  exactly the training config's `selection_attack`, as the evaluation invariant
  requires. Seeds: training 1 throughout, `evaluation_attack 0`,
  `split 20260722`, AutoAttack seed 0. World size 1, per-rank batch 128,
  effective global batch 128. (`evaluation.dataset.num_samples: 16` in the
  resolved config is the same vestigial field noted at 11:47Z; the record's
  `count: 10000` is what actually ran and is what the aggregator gates on.)
  **Consistency checks, all clean.** Against the sibling seed
  `mobilenetv2_pgd_at-s2` (`best.pt` clean 73.23 %, CE-PGD-20 44.40 %,
  AutoAttack 40.18 %): the two MobileNetV2 baseline seeds differ by +1.11 pp
  clean, −0.12 pp CE-PGD-20 and −0.32 pp AutoAttack — a spread consistent with
  seed noise, and far too small to suggest a defect in either. Against the
  Nesterov-matched ResNet-18 baseline (`pgd_at_nesterov-s0/-s1` `best.pt`:
  clean 82.32 / 82.19 %, AutoAttack 47.19 / 47.40 %): MobileNetV2 sits ≈ 8 pp
  lower clean and ≈ 7.5 pp lower under AutoAttack, the expected cost of the much
  smaller model and in line with published MobileNetV2 adversarial-training
  numbers on CIFAR-10 (the residual clean deficit is this project's 10 %
  held-out split — training sees 45,000 images, not 50,000). Under AutoAttack
  the usual best/last ordering holds, `best.pt` above `last.pt` by 0.60 pp; the
  clean ordering is the opposite of seed 2's (`last.pt` is 0.04 pp — four
  examples — *below* `best.pt` here, where in seed 2 it was 1.48 pp above), which
  is not an anomaly: `best.pt` is selected on robust validation accuracy and
  places no constraint on clean accuracy, so the clean gap is free to take
  either sign. No deviation, no anomaly, nothing here needs a bug hunt.
  **The packet-0010 estimator bias now has six of six disagreements in one
  direction, and this run is the largest yet.** As established at 11:47Z,
  `src/ard/evaluation/autoattack.py:213` discards AutoAttack's own count and
  recomputes accuracy in a single unbatched forward over all 10,000 adversarial
  images, while AutoAttack computed its own number in batches of 128; in eval
  mode the two are mathematically identical, and the gap is cuDNN kernel and
  reduction-order noise on borderline examples. The tally across every run that
  has both figures, both checkpoints: `r18_pgd_at-s1` agrees exactly (46.78 /
  40.69 %); `r18_adr-s0` records +2 examples on each; `mobilenetv2_pgd_at-s2`
  records +5 and +4; **this run records +8 and +9** (39.86 vs 39.78 %, 39.26 vs
  39.17 %). Six disagreements out of eight comparisons, **all six in the same
  direction — the recorded number is the higher one**, which is what the
  mechanism predicts. The magnitude ceiling rises from 0.05 pp to **0.09 pp**,
  still far below any noise floor this campaign compares against, so no recorded
  number is in doubt and nothing needs re-running for this reason alone. The
  interaction with packet 0010 is unchanged and now slightly sharper: batching
  that line would move recorded AutoAttack accuracies by roughly this much, so
  runs evaluated before and after the fix would be measured by two slightly
  different estimators inside one aggregate. Packet 0010's `chosen` is still
  null; this postrun does not choose.
  **One operational fact this postrun did not cause.** At 12:01:38Z a lane K
  appeared and started a retry of `r18_trades_adr-s0`'s model-weights
  evaluation. Its first attempt died in 3 s on
  `FileExistsError: refusing to overwrite existing evaluation output` (the
  guard in `evaluate.py:299` doing its job against the old failed directory);
  after that directory was cleared the retry restarted at 12:02:02Z and is
  running. Two things follow. First, the lane driver now prints
  `[lane-K][STOPPING] non-zero exit, not auto-continuing to next queued job` —
  the "does not stop on a non-zero exit" defect recorded in the earlier entries
  is fixed in this driver. Second, that retry runs the same unbatched-forward
  code path that OOM'd six times, because packet 0010 has not been decided; if
  it OOMs again that is the expected outcome, not new information. **This
  postrun launched, retried and aggregated nothing.**
  **Campaign census at 12:02Z, re-derived from a full-root watcher scan.**
  Training: **19 of 20 terminal and successful**; the one missing is still
  `cifar10_mobilenetv2_adr-s0`, never launched. Contract evaluations, model
  weights: **8 of 20 complete** — `r18_pgd_at-s1/-s2`,
  `r18_pgd_at_nesterov-s0/-s1/-s2`, `r18_adr-s0`, `mobilenetv2_pgd_at-s2`, and
  now `mobilenetv2_pgd_at-s1`. EMA weights: **1 of 9**, still only `r18_adr-s0`.
  Terminal-failed and still on disk: `mobilenetv2_adr-s1` only. Reading
  `running`: `r18_trades-s1` (lane A), `r18_trades-s2` (lane B),
  `r18_trades_49k_validation-s0` (lane J), `r18_trades_adr-s2` (lane I) and
  `r18_trades_adr-s0` (lane K, the new retry). Reading `running` but **dead**,
  per the SIGTERM defect: the `evaluation-ema` bundles of `r18_adr-s1`,
  `mobilenetv2_adr-s1` and `r18_trades_adr-s0`. No evaluation bundle of any kind
  exists for `mobilenetv2_pgd_at-s0`, `mobilenetv2_adr-s2`, `r18_adr-s2` or
  `r18_trades_adr-s1`.
  **What this does and does not unlock.** The MobileNetV2 decisive pair
  (`mobilenetv2_adr` vs `mobilenetv2_pgd_at`) now has **two of three baseline
  seeds** and **zero treatment seeds**. Two seeds of one arm are a description
  of those two runs, not an estimate of a distribution, and the pair cannot move
  toward a verdict until `mobilenetv2_adr` has evaluations — one of whose three
  training runs (seed 0) has not been launched at all. M1c and M2 stay unticked.
