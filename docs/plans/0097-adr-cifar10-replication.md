# 0097 — ADR CIFAR-10 self-distillation replication (decision packet 0009, option B)

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Base SHA: `cd0b571e4685fbe02d7655a22b06397c7b8d9a28` (pinned worktree
  `source-cd0b571e4685`, created 2026-09-09)
- Current milestone: M0 and M1a complete; M1c execution under way (9 of the
  20 training runs have run dirs, 7 of them terminal and successful; the first
  contract evaluation, arm 2 seed 1, is terminal) and its checkbox stays
  unticked until the full 20-job launch is verified; M1b and M2/M3 open
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
