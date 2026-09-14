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
