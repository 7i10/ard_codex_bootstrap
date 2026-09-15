# Clean-accuracy-preserving robustness, and a lightweight ImageNet ARD successor — plan 0102

## Governance note (read first)

This plan runs under a standing autonomous mandate the human gave in chat
on 2026-09-15, verbatim:

> Hamster側は空いていますね、自律的にまずはClean精度の確保。そしてロバスト性を
> Clean精度をできるだけ落とさずに上げていくにはどうしたらいいか、調査と実験を
> 繰り返してください。[...] ADR がImageNetでは再現できないでは弱すぎる。すでに
> ImageNetで有効な手法をさらに軽量にしてどうなるか。そして軽量向けにどのような
> 改善ができるか調査と実験。実際に訓練してどうかってところまでどんどん自律的に
> 調べてほしい。基本的に私には聞かずに実行していいです。長期実行もOKです

This changes CLAUDE.md rule 7's default (new arm / seed / epoch horizon /
official test / promotion decisions normally stop for a human-filled
decision packet) **for this plan's scope only**: choosing which arm to try
next, how many epochs, and when to move on does not need to stop and wait
for `chosen:` to be filled in. What still does **not** change, because the
human did not say it could and CLAUDE.md rule 6 forbids it regardless of
delegation: epsilon, steps, step size, random start, normalization,
temperature, checkpoint selection and evaluation attacks are never
silently edited on an existing frozen config; a new mechanism is always a
new config/protocol id. AutoAttack still runs only from a saved checkpoint
in a separate process with `--allow-autoattack`, is still not run as a
"production" full-sample pass without being named as such, and internal
PGD-10 validation is still never reported as an official result.
Attack/objective/trainer code changes still go through `scientific-reviewer`
before a GPU-hour is spent on them, same as every other plan. Progress is
still logged here as it happens, same discipline as every other plan --
autonomy changes who approves the next step, not whether the step is
recorded.

## Context

Three things converged to prompt this plan, all from chat on 2026-09-15:

1. **Question**: does the pretrained MobileNetV4-Conv-Small checkpoint
   reproduce its published clean accuracy on this project's own pipeline,
   and if the fine-tuned number ends up much lower, why? **Already
   answered by plan 0101's own Stage A** (docs/plans/0101, "Stage A —
   PASSED", 2026-09-13): the bare pretrained checkpoint scores **73.41%**
   top-1 on the real 50k-image ImageNet val split through this project's
   own `build_student`/`ImageNetEvalTransform` pipeline, against the
   paper's published 73.8% -- a 0.4pp residual gap plan 0101 attributes to
   a specific, named, not-yet-fixed cause: `ImageNetEvalTransform` resizes
   with torchvision's default bilinear interpolation, while timm declares
   `interpolation='bicubic'` for this exact checkpoint tag (plan 0101,
   "Known, deliberately-not-fixed gaps", review finding P2-6). So: yes, it
   essentially reproduces (within a well-understood, well-attributed 0.4pp),
   confirming the registry wiring and data pipeline are correct. The much
   larger drop actually observed during training (73.4% pretrained down to
   ~38-52% clean after adversarial fine-tuning, per plan 0100's four stage-1
   runs and plan 0101's Stage B canary) is not a pipeline bug -- it is the
   well-documented clean-accuracy cost of PGD adversarial training itself,
   compounded by fine-tuning from a *non-robust* pretrained backbone
   (exactly the failure mode arXiv:2509.23325 names, see Workstream A
   below), and is not yet known to be permanent: plan 0101's Stage C
   (all three MobileNetV4 arms, full 50 epochs, launched today on Ferret)
   is the test of whether it recovers after the epoch-25/38 LR decay the
   way plan 0100's own `r18_baseline` did (48.4%→55.1% clean across its
   own decay). This plan does not re-run Stage A or Stage C; it starts
   from their answers.
2. **Direction-finding AutoAttack** (decision 0014, same session) found
   that ADR (this project's EMA-of-student self-distillation method)
   underperforms plain PGD-AT on **both** clean and (directionally, not
   yet statistically confirmed for ResNet-18 at n=500) AutoAttack robust
   accuracy, on both ResNet-18 and MobileNetV3-Small, at ImageNet-1k
   scale. The human's own assessment: "ADR がImageNetでは再現できないでは
   弱すぎる" (concluding "doesn't reproduce at ImageNet, full stop" is too
   weak) -- there is real published work on adversarial-robustness
   distillation that *does* work at ImageNet scale (this project's own
   `adr` is a homegrown EMA-of-student variant, not identical to the
   literature's ARD/RSLAD/etc.), and the more interesting, harder question
   is whether one of those methods can be adopted and then made *more*
   lightweight for mobile-scale students, not whether this project's own
   first attempt happened to fail.
3. **Hamster (2×4090, local) is idle** while Ferret (3×4090, remote) runs
   plan 0101's Stage C. The human wants Hamster used for open-ended,
   repeated research-and-experiment cycles on two questions at once:
   raising robust accuracy while minimizing the clean-accuracy cost
   (Workstream A), and finding + lightening an ImageNet-proven robustness-
   distillation method (Workstream B) -- run for real, trained and
   measured, not just surveyed.

## Goal

Two parallel, loosely-coupled workstreams, both scoped to what Hamster's
2×4090 can execute repeatedly without blocking on Ferret:

- **Workstream A (clean-accuracy-preserving robustness recipe)**: identify
  and empirically test techniques that raise robust accuracy (PGD/AutoAttack)
  without costing much clean accuracy, on top of whatever floor plan 0101
  establishes. Candidates to evaluate empirically (not just cite): the
  epsilon-scheduling paper's exact non-robust-pretrained-transfer recipe
  (arXiv:2509.23325, only abstract-read so far -- full read is this plan's
  first concrete task), AWP, TRADES-with-tuned-beta as an alternative
  objective, SWA, and whatever else a literature pass surfaces as tractable
  at this compute scale (~35-40 GPU-hours per full 50-epoch run being the
  existing unit; a technique costing much more than that per run should be
  scaled down -- fewer epochs, a canary -- before committing a full run).
- **Workstream B (lightweight ImageNet-scale ARD successor)**: find a
  published adversarial-robustness-distillation method with a real
  ImageNet-1k validation (not CIFAR/Tiny-ImageNet/ImageNet-100), understand
  what makes it different from this project's own EMA self-distillation,
  assess how it could be made more parameter/compute-lightweight for
  sub-15M-param mobile students without destroying its mechanism, and then
  actually implement and train a canary of the adapted version to see
  whether it clears plain PGD-AT (this project's own baseline) where `adr`
  did not.

Both workstreams report into this plan's Progress log as they produce
results, whether positive, negative, or inconclusive -- a negative result
that narrows the search is still a recorded finding, not a reason to
delete the entry.

## Process

1. **Literature pass** (in progress, backgrounded 2026-09-15): a research
   agent is reading arXiv:2509.23325 in full (not abstract-only), surveying
   AWP/TRADES-beta/SWA/2023-2025 clean-accuracy-tradeoff papers for
   Workstream A, and surveying ImageNet-1k-validated robustness-distillation
   methods plus general small-model distillation tricks for Workstream B.
   Its report becomes this plan's first Progress log entry once it returns.
2. **Design from the literature, not from guessing.** Each new arm this
   plan adds gets its own config file (never edits an existing frozen one),
   a short paragraph here explaining what specific paper/finding motivates
   it, and — if it touches `src/ard/attacks/`, `src/ard/engine/`,
   `src/ard/config/schema.py` objective/policy code — a scientific-reviewer
   pass before its first real GPU-hour, same as plan 0101's own ε-warmup
   work.
3. **Canary first, always.** A new mechanism gets a short (3-5 epoch, or
   whatever this project's existing canary convention uses) run before a
   full 50-epoch commitment, exactly like plan 0101's Stage B. Compare
   against the nearest existing baseline at the same epoch horizon (plan
   0100's or plan 0101's own recorded numbers), not a fresh training-from-
   scratch control, when one already exists at a matching config.
4. **Hamster only, unless stated otherwise.** Ferret is committed to plan
   0101's Stage C; do not launch competing jobs there without first
   confirming (via `ferret-status`/`nvidia-smi` over ssh) that Stage C has
   actually finished or a GPU there is genuinely idle.
5. **Record honestly.** If a technique doesn't help, or actively hurts, or
   is inconclusive at the horizon tested, that is exactly as valid an
   entry in the Progress log as a win -- the goal is a real answer, not a
   positive result.

## Files to touch (grows as workstreams produce concrete designs)

Not yet known -- populated as each workstream's first concrete experiment
is designed from the literature pass. Every new arm gets its own config
under `configs/scientific/`, never an edit to an existing plan 0100/0101
config.

## Progress log

- 2026-09-15 (chat, autonomous): plan opened under the governance note
  above. Confirmed Hamster idle (GPU0/1 both 0% util, 0-17 MiB). Launched
  a backgrounded literature-research agent (general-purpose, read-only --
  web research only, no code) covering: (1) full read of arXiv:2509.23325
  and its exact recipe for the non-robust-pretrained-transfer setting; (2)
  AWP / TRADES-beta / SWA / other clean-accuracy-preserving adversarial
  training techniques tractable at this project's compute scale; (3)
  ImageNet-1k-validated robustness-distillation methods and how they might
  be made more lightweight for mobile-scale students, plus general small-
  model distillation tricks not yet tried in an adversarial context. No
  training launched yet -- waiting on the literature pass before designing
  the first concrete arm, per this plan's own "design from the literature"
  rule above.
- 2026-09-15 (chat, autonomous, continued): literature agent returned. Key
  findings (full report not reproduced here; see this entry's summary):
  - **arXiv:2509.23325 full read**: its schedule is a *trapezoidal*
    (delay-then-ramp) shape -- flat at eps=0 for the first ~24% of epochs
    (T1), linear ramp over the middle ~50% (T1→T2), held at target for the
    rest -- not a pure 0-epoch-start linear ramp like this project's
    current implementation of Debenedetti. It uses APGD (adaptive internal
    step size), not fixed-ratio PGD, so it has **no stated position on
    this project's own step-size-coupling choice** (Option 1) -- that
    remains this project's own deliberate simplification, not something
    the paper confirms or contradicts. Its target task is fine-grained
    transfer datasets (CUB/Cars/Aircraft/etc.) via a robust *pretrained*
    backbone, not ImageNet-1k itself as the fine-tuning target, and no
    mobile-scale architecture -- so its exact numbers don't transfer, but
    its failure-mode framing (fixed-eps fine-tuning from a non-robust
    backbone can collapse to near-chance) independently corroborates that
    this project's large clean-accuracy drop is a known, named phenomenon.
  - **Cheapest, best-evidenced techniques for Workstream A** (full table
    in the agent's report): SWA/weight-averaging (CIFAR evidence: +5.55pp
    robust *and* +0.94pp clean simultaneously, near-zero extra compute,
    but needs per-epoch checkpoint retention this project doesn't do yet);
    AWP (~8% training overhead, TRADES+AWP is the only CIFAR combo found
    that improved both axes together); TRADES with tunable beta (same
    compute as plain PGD-AT, a loss-function swap only, exact beta for
    this regime unknown -- needs its own probe); MixedNUTS-style post-hoc
    logit mixing of the existing pretrained + PGD-AT checkpoints (~1
    GPU-hour, no retraining, diagnostic for how much clean-accuracy loss
    is recoverable at inference time alone). Bag-of-Tricks' "critical"
    weight-decay warning was checked against this project's own ImageNet
    configs (already 1e-4, the standard ImageNet convention, no change
    needed).
  - **Workstream B reframe**: no ARD-family method (this project's own
    EMA self-distillation, the original ARD of Goldblum et al. 2020,
    RSLAD, AdaAD) has a validated true-ImageNet-1k (not CIFAR/Tiny/
    ImageNet-100) result in the literature the agent could find -- the
    absence of a positive result at ImageNet-1k scale is the field-wide
    norm, not a sign this project's implementation is uniquely broken.
    ProARD (arXiv:2506.07666, explicitly targets edge/mobile students) is
    the closest-matching lead but only its abstract was reachable --
    flagged for a dedicated full-text follow-up, not yet actioned.
    Separately, arXiv:2605.21999's "Robustly Unlearnable Set" diagnosis
    (capacity-limited students cannot represent some of a teacher's robust
    features, forcing memorization of spurious noise on exactly those
    samples) gives a concrete, testable hypothesis for *why* ADR's EMA
    target might be hurting rather than helping a small student: an
    overconfident target on hard examples, not "self-distillation doesn't
    work here." Cheap first test: temper/soften the EMA target's
    confidence on high-loss examples, rather than abandoning EMA self-
    distillation outright.
  - **First concrete action taken**: Workstream A's TRADES-beta probe was
    judged cheapest and most directly grounded (TRADES is not new engine
    surface -- it already exists and is exercised by
    `configs/scientific/cifar10_r18_trades.yaml`; only a new config
    combining it with the already-registered MobileNetV4 architecture and
    this project's own ImageNet stage-01 conventions is needed, no
    scientific-reviewer pass required by this plan's own process rule,
    which reserves that gate for attack/engine/schema code changes).
    Added `configs/scientific/imagenet_mobilenetv4_trades_beta1.yaml` and
    `_beta6.yaml` (beta=1.0 and beta=6.0, no epsilon-warmup in either --
    isolating the beta effect alone before combining with warmup),
    registered a new protocol id
    `controlled_imagenet_stage01_mobilenetv4_trades_v1`
    (`src/ard/protocols/__init__.py`, purely additive metadata, same
    pattern as plan 0101's own protocol entry), and a config-identity test
    (`tests/unit/test_config.py::test_mobilenetv4_trades_beta_configs_are_field_identical_except_beta`).
    `scripts/verify.py --changed` run; result recorded in the next entry.
  `scripts/verify.py --changed` green (all suites passed). Pinned a fresh
  worktree at HEAD `3e7224ce0e3f...` and launched both TRADES-beta arms as
  local hand-runs on Hamster's two idle 4090s (GPU0: beta=1.0, GPU1:
  beta=6.0), 12 epochs each -- matching plan 0101 Stage B's own canary
  horizon so the result is directly comparable to plan 0101's recorded
  Arm A (no-warmup PGD-AT) epoch-11 numbers (37.8% clean / 19.4% PGD-10).
  `run_imagenet_stage01_train.py --epochs 12` (confirmatory forward of
  `training.epochs=12`, same wrapper plan 0101 used), `ARD_NUM_WORKERS=8`,
  local ImageNet root. Both processes confirmed launched; GPU utilization
  ramp-up being confirmed separately (Monitor). Queued for later, not yet
  started: the MixedNUTS-style post-hoc logit-mixing diagnostic
  (recommendation 4) on the already-completed plan 0100 pgd_at checkpoints
  -- deferred this round in favor of getting the TRADES canaries running
  first; it needs a small custom PGD loop against the mixed-softmax
  forward pass (this project's `LinfPGD.generate()` is not written for a
  two-model mixture), so it will get its own throwaway script, not a
  scientific-record config.
- 2026-09-15 (`/experiment-postrun`, run-bundle path): `plan0102-trades-beta1-v1`
  completed. Terminal status re-derived with `campaign_watch.py --once
  --include-hand-run`: `terminal: true`, `success: true`, `failure_class: null`.
  All 12 of 12 epoch rows are present. `run-bundle/completion.json` says `completed`.
  Source SHA `3e7224ce0e3f` (clean worktree). World size 1, global batch 128,
  seed 0, protocol `controlled_imagenet_stage01_mobilenetv4_trades_v1`.
  `best.pt` / `last.pt` were not checked from this session, because the
  sandbox blocks reads under `ard-runtime/`.
  **Nothing imported.** No aggregator exists for this contract, no AutoAttack
  has run, and the sibling arm `plan0102-trades-beta6-v1` is still training
  (bundle `running`, epoch 11 in progress at 22:07Z). No record, report,
  ledger row or milestone tick.
  **Internal validation only (held-out slice, PGD-10; not an official
  result, n=1)**, from the run-bundle summary, beta=1.0: best epoch 2, clean /
  PGD 0.531 / 0.108. Last epoch (11): clean / PGD 0.457 / 0.107.
  Next to plan 0101 Stage B Arm A (no-warmup PGD-AT, epoch 11: 37.8% clean /
  19.4% PGD-10), beta=1.0 is about 8 pp higher on clean accuracy and about 9 pp
  lower on PGD accuracy. Robust accuracy has not moved since epoch 2, and
  clean accuracy fell from 53.1% to 45.7% over the same span. So at this
  12-epoch horizon, beta=1.0 buys clean accuracy with robust accuracy; it
  does not raise both. Compare the arms only once beta=6.0 finishes, and
  only after confirming that both runs used the same validation slice and
  attack as Arm A.
- 2026-09-15 (`/experiment-postrun`, run-bundle path): `plan0102-trades-beta6-v1`
  completed. Terminal status re-derived with `campaign_watch.py --once
  --include-hand-run`: `terminal: true`, `success: true`, `failure_class: null`.
  All 12 of 12 epoch rows are present. `run-bundle/completion.json`, `best.pt`,
  `last.pt`, `epoch-metrics.parquet` and `sample-stats-train.parquet` exist.
  Source SHA `3e7224ce0e3f` (clean worktree). World size 1, global batch 128,
  seed 0, protocol `controlled_imagenet_stage01_mobilenetv4_trades_v1`.
  **Nothing imported**, for the same reason as beta=1.0: no aggregator exists
  for this contract, and no AutoAttack has run. No record, report, ledger row
  or milestone tick.
  **Comparability check.** `imagenet_mobilenetv4_trades_beta6.yaml` differs
  from Arm A's `imagenet_mobilenetv4_pgd_at_no_warmup.yaml` only in the header
  comment, `protocol.id`, `tracking.group` and `method`. Inside `method`, the
  `selection_attack` is the same: CE loss, eps 4/255, step 8/765, 10 steps,
  random start, eval mode. So the validation data and attack match Arm A.
  The training objective and the inner attack loss (KL instead of CE) differ
  on purpose. Arm A ran on Ferret, and this session did not re-read its bundle
  to confirm world size 1 / global batch 128. Same config batch, one GPU per arm.
  **Internal validation only (held-out slice, PGD-10; not an official result,
  n=1, 12 of 50 epochs, before the LR decay)**, from the run-bundle summaries:
  | arm | best epoch | best clean / PGD | epoch 11 (last) clean / PGD |
  |---|---:|---:|---:|
  | PGD-AT, Arm A (plan 0101) | — | — | 37.8% / 19.4% |
  | TRADES beta=1.0 | 2 | 53.1% / 10.8% | 45.7% / 10.7% |
  | TRADES beta=6.0 | 2 | 45.4% / 18.7% | 39.5% / 16.7% |
  Reading, for these single runs at this horizon: beta=6.0 is +1.7 pp clean
  and -2.7 pp PGD against Arm A at epoch 11. Neither TRADES arm raised both
  clean and robust accuracy over PGD-AT. Beta moves along the tradeoff:
  going from 1.0 to 6.0 costs 6.2 pp clean and buys 6.0 pp PGD at the last
  epoch. Both TRADES arms picked epoch 2 as their best checkpoint. After
  that, beta=6.0 lost 2.0 pp PGD and 5.9 pp clean by epoch 11 (beta=1.0:
  -0.1 pp PGD, -7.4 pp clean). So through epoch 11, both arms mostly
  lost ground after an early peak while the LR stayed at 0.05. The epoch-25
  decay might reverse this, as it did for plan 0100's `r18_baseline`; these
  12-epoch runs cannot tell. The beta=6.0 gap to Arm A (1.7 / 2.7 pp) is only
  a little larger than the 0.16-1.88 pp control-vs-control spread seen on
  CIFAR screens. No ImageNet noise floor exists yet, and each arm is one
  run. The safe reading is "TRADES beta=6.0 lands near PGD-AT at epoch 11,
  slightly more clean and slightly less PGD". That is not a ranking.
- 2026-09-15 (chat, autonomous, continued): both Hamster GPUs went idle
  once the two TRADES canaries finished (human noticed and asked). Three
  things launched/run immediately:
  1. **Third TRADES-beta probe point** (`imagenet_mobilenetv4_trades_beta3.yaml`,
     beta=3.0, same structure as beta=1/6) added to fill in the curve
     between the two inconclusive points above, plus a 3-way config-
     identity test replacing the pairwise one
     (`test_mobilenetv4_trades_beta_configs_are_field_identical_except_beta`).
     `scripts/verify.py --changed` run; launch follows once green (recorded
     in the next entry).
  2. **MixedNUTS-style post-hoc logit-mixing diagnostic run** (recommendation
     4, deferred from the previous entry) --
     `scripts/analysis/plan0102_mixed_nuts_diagnostic.py` (new, throwaway,
     not a scientific record), a self-contained white-box PGD-10 attack
     (eps=4/255, step=8/765) against the actual softmax mixture of each
     plan 0100 `pgd_at` checkpoint (robust) and its own original pretrained
     backbone (clean), swept over mixing weight lambda in
     {0, 0.25, 0.5, 0.75, 1}, n=1000 random val images, one Hamster GPU,
     finished in well under an hour:

     | arch | lambda | clean_acc | robust_acc (white-box PGD-10 vs the mixture) |
     |---|---:|---:|---:|
     | resnet18 | 0.00 | 69.5% | 0.0% |
     | resnet18 | 0.25 | 70.0% | 0.1% |
     | resnet18 | 0.50 | 69.8% | 3.0% |
     | resnet18 | 0.75 | 66.6% | 17.1% |
     | resnet18 | 1.00 | 50.6% | 28.2% |
     | mobilenetv3 | 0.00 | 66.9% | 0.0% |
     | mobilenetv3 | 0.25 | 66.7% | 0.2% |
     | mobilenetv3 | 0.50 | 65.9% | 1.3% |
     | mobilenetv3 | 0.75 | 62.9% | 9.3% |
     | mobilenetv3 | 1.00 | 39.2% | 21.3% |

     (lambda=1.0's clean/robust numbers are close to, not identical to,
     the earlier full-val/AutoAttack numbers -- expected, this is n=1000
     random images with a different attack (white-box PGD-10 here, not
     AutoAttack) and a different seed's subset, not a discrepancy.)

     **Reading, honestly**: a plain linear softmax mix is not a free
     lunch under a white-box attacker who backpropagates through the
     whole mixture -- robust accuracy stays near zero until lambda is
     already most of the way to 1 (0.5→3.0%, 0.75→17.1% for resnet18),
     by which point most of the clean-accuracy gain is already gone
     (69.8%→66.6% between those same two points). There is no
     lambda where this simple mixture is both close to the clean
     model's accuracy and meaningfully more robust than the plain robust
     model alone. This does not mean MixedNUTS itself doesn't work --
     the actual paper's mixing function is a *nonlinear*, margin-clipped
     combination specifically designed to resist exactly this adaptive-
     attack collapse, which this quick linear approximation does not
     implement. **Correct conclusion: the naive version of this idea
     doesn't help here; the paper's actual (more careful) mechanism has
     not been tested and would need to be implemented properly, not
     assumed from the linear-mix result.** Filed as a checked-and-mostly-
     ruled-out item for the naive form; the real MixedNUTS formulation
     remains a documented, not-yet-tried option if revisited later.
  3. Given (2)'s negative result, recommendation 4 is deprioritized;
     Workstream A continues with the TRADES-beta3 canary (1) and, next,
     starts implementing AWP (recommendation 2) since TRADES alone has not
     yet beaten PGD-AT on both axes at this horizon and AWP is the only
     technique from the literature pass with CIFAR evidence of improving
     both simultaneously (TRADES+AWP combo).
  4. **Reconsidered before starting AWP**: AWP (a nested weight-perturbation
     loop) is a materially bigger, riskier trainer change than first
     estimated. Recommendation 1 (SWA/plain weight-EMA) was rated the
     single best cost/benefit item in the literature review and, on
     inspection of `src/ard/engine/trainer.py`, turned out to be nearly
     free to add: the EMA/checkpoint/best-selection machinery ADR already
     built (`self.ema_model`, `_update_ema`, `best-ema.pt`,
     `best_metric_ema`/`selection_metadata_ema`, `ard.engine.checkpoint`'s
     save/load) was already gated purely on `self.ema_model is not None`,
     never on `method.id` or `adr_config` -- confirmed by grep before
     writing any code, not assumed. Implemented first, ahead of AWP:
     - New `training.weight_ema_decay: float | None` (`src/ard/config/schema.py`),
       mutually exclusive with `method.adr` (adr already tracks its own EMA
       as a distillation target; a second one would need a second shadow
       model this project has never built).
     - `Trainer.__init__` (`src/ard/engine/trainer.py`) constructs
       `self.ema_model` when `weight_ema_decay is not None` OR
       `adr_config is not None` (previously only the latter);
       `_update_ema` resolves decay from whichever is active.
     - `src/ard/cli/train.py` threads the new field through;
       `src/ard/cli/evaluate.py`'s `--weights=ema` preflight gate now also
       accepts a plain-pgd_at-plus-weight_ema_decay checkpoint, not only
       adr/adr_trades.
     - New tests: schema-level mutual-exclusion
       (`test_weight_ema_decay_rejects_combination_with_adr`), trainer-level
       construction/divergence/round-trip/independent-selection
       (`tests/integration/test_weight_ema_trainer.py`, mirroring
       `test_adr_trainer.py`'s own EMA tests), and a CLI-level
       `--weights=ema` acceptance test (`tests/integration/test_tracking_evaluation.py`).
       `scripts/verify.py --changed` green (one own mistake caught and
       fixed along the way: a stray leftover assertion referencing an
       undefined variable in the new CLI test, from a bad edit -- not a
       scientific bug, just sloppy test authoring, fixed before commit).
       Committed as `bc716cf`.
     - Per this plan's own process rule (engine/schema changes need
       scientific-reviewer before a GPU-hour is spent), requested a review
       of this exact change before launching any training with
       `weight_ema_decay` set. Not yet launched; GPU1 is free and reserved
       for it once review clears. AWP itself is deferred, not abandoned --
       revisit after weight-EMA's own result is in, since weight-EMA alone
       might already close much of the gap the literature review flagged.
  5. **Review returned**: 2 blocking (P1), 6 non-blocking (P2) findings, all
     genuine (not false positives) -- notably the review caught that a bad
     edit had *deleted* a pre-existing regression guard (rule 6: never
     weaken a guard to make an edit pass) and that a diagnostic metric
     (`train_ema_student_agreement`) would have silently logged a false
     0.0 instead of the true (>0.95) value for every weight-EMA run, which
     matters because that exact metric has already been read as evidence
     in plan 0100's own write-up. Fixed all 8 findings in one batched pass
     (CLAUDE.md rule 2): restored the guard; re-scoped the metric's gate to
     `adr_config is not None`; added `weight_ema_decay`/`epsilon_warmup_epochs`
     to `evaluate.py`'s `training_protocol_identity` so the aggregator's
     mixed-identity guard can see them (no existing record uses either
     field, so no already-recorded identity changes); documented the
     config-hash/resume caveat (same class as `epsilon_warmup_epochs`
     before it -- any pre-`29decde` checkpoint needs its own pinned SHA to
     resume) and the roughly-doubled per-epoch validation cost; corrected
     four stale ADR-only comments/help text and added a non-ADR-EMA section
     to `docs/SCIENTIFIC_INVARIANTS.md`; replaced a one-epoch test that
     didn't prove "independent selection" with the real independence test.
     Picked and justified `weight_ema_decay: 0.999` (distinct from ADR's
     own 0.995, matching the general per-iteration-EMA convention rather
     than the shorter fixture-test value of 0.9) in a new launch config,
     `configs/scientific/imagenet_mobilenetv4_pgd_at_no_warmup_weight_ema.yaml`
     -- field-identical to plan 0101's Arm A except `weight_ema_decay` and
     `tracking.group`, isolating the weight-EMA effect against Arm A's
     already-recorded epoch-11 baseline (37.8% clean / 19.4% PGD-10).
     `scripts/verify.py --changed` green (confirmed twice, including once
     via explicit exit code after a confusing but benign verify.py
     cache-timing artifact -- not a real failure). Committed as `29decde`.
     Pinned worktree `source-29decdedb4c7`, launched a 12-epoch canary on
     Hamster GPU1 (`plan0102-weight-ema-v1`), GPU0 still running the
     TRADES-beta3 canary.
- 2026-09-15 (`/experiment-postrun`, run-bundle path): `plan0102-trades-beta3-v1`
  completed at 11:01Z. Terminal status re-derived with `campaign_watch.py --once
  --include-hand-run`: `terminal: true`, `success: true`, `failure_class: null`.
  All 12 of 12 epoch rows are present. `run-bundle/completion.json`, `best.pt`,
  `last.pt`, `epoch-metrics.parquet` and `sample-stats-train.parquet` exist.
  Source SHA `e6cf9edf9f9b` (clean worktree). World size 1, global batch 128,
  seed 0, protocol `controlled_imagenet_stage01_mobilenetv4_trades_v1`.
  **Nothing imported**, for the same reason as beta=1.0 and beta=6.0: no
  aggregator exists for this contract, and no AutoAttack has run. No record,
  report, ledger row or milestone tick. The config differs from beta=1/6 only
  in beta. `test_mobilenetv4_trades_beta_configs_are_field_identical_except_beta`
  enforces this, so the validation slice and selection attack match Arm A, as
  checked for beta=6.0 above. One caveat: beta=3.0 ran from `e6cf9ed`, while
  beta=1/6 ran from `3e7224c`. The commits in between change only the plan,
  this config, its test and a throwaway analysis script. None of them touches
  `src/ard/`.
  **Internal validation only (held-out slice, PGD-10; not an official result,
  n=1 per arm, 12 of 50 epochs, before the LR decay)**, from the run-bundle
  summaries:
  | arm | best epoch | best clean / PGD | epoch 11 (last) clean / PGD |
  |---|---:|---:|---:|
  | PGD-AT, Arm A (plan 0101) | — | — | 37.8% / 19.4% |
  | TRADES beta=1.0 | 2 | 53.1% / 10.8% | 45.7% / 10.7% |
  | TRADES beta=3.0 | 2 | 48.5% / 17.2% | 40.9% / 15.3% |
  | TRADES beta=6.0 | 2 | 45.4% / 18.7% | 39.5% / 16.7% |
  Reading, for these single runs at this horizon: beta=3.0 sits between the
  other two TRADES arms on both axes, at best and at last. The step from 1.0
  to 3.0 costs 4.8 pp clean and buys 4.6 pp PGD at the last epoch. The step
  from 3.0 to 6.0 costs 1.4 pp clean and buys 1.4 pp PGD. Both of those last
  gaps are inside the 0.16-1.88 pp CIFAR control-vs-control spread. Against
  Arm A at epoch 11, beta=3.0 is +3.1 pp clean and -4.1 pp PGD. That is a
  point on the same tradeoff, not a gain on both axes. Like the other two arms,
  beta=3.0 peaked at epoch 2 and then lost ground by epoch 11 (-7.6 pp clean,
  -1.9 pp PGD) while the LR stayed at 0.05. So the beta curve is filled in and
  looks smooth: no TRADES beta tested here beats PGD-AT on both clean and PGD
  accuracy before the LR decay. Whether the post-decay recovery changes this
  cannot be read from 12-epoch runs.
- 2026-09-15 (`/experiment-postrun`, run-bundle path): `plan0102-weight-ema-v1`
  completed at 11:22Z. Terminal status re-derived with `campaign_watch.py --once
  --include-hand-run`: `terminal: true`, `success: true`, `failure_class: null`.
  All 12 of 12 epoch rows are present. `run-bundle/completion.json`, `best.pt`,
  `best-ema.pt`, `last.pt`, `epoch-metrics.parquet`/`.jsonl` and
  `sample-stats-train.parquet` exist. Source SHA `29decdedb4c7` (clean
  worktree). World size 1, global batch 128, seed 0, protocol
  `controlled_imagenet_stage01_mobilenetv4_pgd_at_v1`, `weight_ema_decay: 0.999`.
  **Nothing imported**, for the same reason as the TRADES canaries: no
  aggregator exists for this contract, and no AutoAttack has run. No record,
  report, ledger row or milestone tick.
  **How to read this run.** The EMA is a shadow copy. It never feeds back
  into training. So the live weights follow plain PGD-AT, and the EMA weights
  are evaluated on the same held-out slice with the same PGD-10 selection
  attack, run directly against the EMA model. This makes the live-vs-EMA
  comparison a within-run control: same data order, same trajectory, same
  attack. The live weights at epoch 11 (37.82% / 19.43%) agree with plan 0101
  Arm A (37.8% / 19.4%) to the reported precision, as expected for the
  same config and seed. Arm A's bundle is on Ferret and was not re-read, so
  bit-for-bit equality is not confirmed.
  **Internal validation only (held-out slice, PGD-10; not an official result,
  n=1, 12 of 50 epochs, LR warmup 0.005→0.05 over epochs 0-9, before the LR
  decay)**, from `epoch-metrics.jsonl`:
  | epoch | LR | live clean / PGD | EMA clean / PGD | EMA − live (clean / PGD) |
  |---:|---:|---:|---:|---:|
  | 0 | 0.005 | 46.8% / 19.2% | 49.3% / 22.0% | +2.4 / +2.9 pp |
  | 2 | 0.015 | 46.0% / 21.6% | 50.5% / 26.1% | +4.4 / +4.6 pp |
  | 5 | 0.030 | 42.8% / 21.7% | 48.7% / 27.1% | +5.9 / +5.4 pp |
  | 8 | 0.045 | 37.2% / 19.6% | 45.6% / 26.4% | +8.3 / +6.8 pp |
  | 11 (last) | 0.050 | 37.8% / 19.4% | 44.7% / 25.8% | +6.8 / +6.4 pp |
  Best by PGD: live epoch 4 (43.3% / 21.7%); EMA epoch 5 (48.7% / 27.1%).
  Best-checkpoint numbers are selected on this same slice, so the last-epoch
  row is the less biased one.
  Reading, for this single run at this horizon: the EMA weights beat the live
  weights on **both** clean and PGD accuracy at every one of the 12 epochs.
  At epoch 11 the gap is +6.8 pp clean and +6.4 pp PGD. That is well above the
  0.16-1.88 pp CIFAR control-vs-control spread. It is the first arm in this
  plan that is not a point on the clean-vs-robust tradeoff. Against the TRADES
  arms at epoch 11, EMA is ahead on both axes of beta=3.0 and beta=6.0, and
  behind only beta=1.0 on clean (by 1.0 pp) while ahead of it by 15.1 pp PGD.
  Caveats that decide what this is worth:
  1. The gap grew while the LR ramped up (+2.9 pp PGD at LR 0.005, +6.4 pp at
     LR 0.05). A plausible reading is that the EMA smooths out high-LR SGD
     noise. If so, the gap may shrink after the epoch-25/38 LR decay. A
     12-epoch run cannot say.
  2. PGD-10 only. Weight averaging could in principle make the loss surface
     harder for a 10-step attack without real robustness. AutoAttack on the
     EMA weights (from a saved checkpoint, separate process,
     `--allow-autoattack`, `--weights=ema`) is the check. Until then, the PGD
     gain is not a robustness claim.
  3. The decay 0.999 averages over roughly 1,000 steps, about 0.1 epoch at
     9,809 steps per epoch. This is a short average. Averaged BatchNorm
     running statistics are part of the EMA copy and may carry part of the
     effect. Nothing here separates the two.
  4. n=1, seed 0, one GPU, global batch 128.
- 2026-09-15 (chat, autonomous, continued): both Hamster GPUs freed once the
  two canaries above finished (human noticed and asked). Given the
  weight-EMA result's size (well above the CIFAR noise floor) and its two
  open caveats (does it survive the LR decay; is the PGD-10 gain real
  robustness or an attack-strength artifact), launched three jobs at once
  rather than picking one:
  1. **AutoAttack direction-finding (n=500) on `plan0102-weight-ema-v1`'s
     own best.pt AND best-ema.pt** (same run, same epoch pool, only the
     evaluated weights differ -- the cleanest possible apples-to-apples
     check of caveat 2). `--weights model --allow-autoattack` and
     `--weights ema --allow-autoattack`, both `evaluation.checkpoints=best
     evaluation.autoattack_sample_count=500`, both on Hamster GPU0 from the
     same pinned worktree the training run used
     (`source-29decdedb4c7`). Not an official test (reduced sample,
     direction-finding only, same status as decision 0014's option E).
  2. **A fresh full 50-epoch run of the exact same weight-EMA config**
     (`plan0102-weight-ema-full50-v1`, Hamster GPU1) to see whether the gap
     survives the epoch-25/38 LR decay (caveat 1). Attempted a `--resume`
     of the 12-epoch run first, to avoid re-spending the first 12 epochs:
     rejected with "checkpoint config hash does not match resolved
     config" -- `training.epochs` is itself part of the resolved-config
     hash the checkpoint was saved under (it was launched with `--epochs
     12`, i.e. `training.epochs=12` forwarded as an override), so
     resuming under `--epochs 50` is a different identity, not a
     continuation, by this project's own resume-refuses-drift design.
     This empirically confirms the same reasoning that led decision
     0015/plan 0101's Stage C to launch fresh rather than attempt a
     resume-based continuation on Ferret -- not just a theoretical risk,
     a reproduced mechanical fact. Switched to a fresh from-scratch launch
     under a new run-id, same config, same seed.
  **AutoAttack result (addresses caveat 2 directly)**: both checks
  finished. `best.pt` (student-selected, epoch 4): clean 40.28%,
  AutoAttack(n=500) 14.2%. `best-ema.pt` (EMA-selected, epoch 5): clean
  45.14%, AutoAttack(n=500) 19.4%. **The EMA weights beat the live
  weights on both clean (+4.9pp) and real AutoAttack (+5.2pp)** -- this is
  not a PGD-10-specific artifact; a full APGD-CE/APGD-T/FAB-T/Square attack
  shows the same direction. At n=500 each arm's binomial standard error is
  about 1.6-1.8pp, so the 5.2pp AutoAttack gap is roughly 3x the per-arm
  SE -- more distinguishable than the earlier ResNet-18 adr-vs-pgd_at gap
  (1.6pp gap against ~1.8pp SE, indistinguishable), though still n=1 seed
  and not a substitute for a full-sample official test. This is the
  strongest positive result either workstream has produced so far, and the
  first result in this plan that survived a real-attack check, not just an
  internal PGD-10 proxy. Caveats 1, 3 and 4 above are still open (LR-decay
  survival is what `plan0102-weight-ema-full50-v1` is now checking; decay
  value and BN-averaging separation are unaddressed; n=1 seed).
