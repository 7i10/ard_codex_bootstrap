# 0100 — ImageNet Stage 0/1: does ADR's capacity-inverse effect hold from ResNet-18 to mobile scale?

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Depends on: plan 0099 (ImageNet Stage 0 prep, complete — data loading, AMP,
  architecture registry, real measured throughput), decision packet 0011
  (`chosen: B`, CIFAR-first validation before ImageNet investment — now
  satisfied: plan 0097 confirmed the mechanism on CIFAR-10, plan 0098's
  refinement attempt was shelved per decision packet 0012, and this plan
  runs the *original*, already-validated `adr` mechanism, not the shelved
  gap-adaptive variant).
- Decided this session (chat, not yet a separate decision packet — recorded
  here per this project's convention that a plan's own Decisions section may
  carry a decision made in direct conversation, mirroring plan 0097 §Decisions):
  drop the originally-registered `resnet50_imagenet` reference point (it
  doesn't serve the "does the effect strengthen toward mobile scale"
  question — it is a *larger* model than ResNet-18, and is by far the most
  expensive of the three registered architectures); train from ImageNet-1k
  pretrained (non-robust) initialization, not from scratch; use 50 epochs of
  adversarial fine-tuning, not 100; use 3-step training-time PGD; 3 seeds per
  arm from the start (not staged up from 1-2), the human's explicit call
  after this session raised the seed-count question and recommended staging.
- Current milestone: implementation complete, scientific review complete,
  all findings resolved (see Progress log). No canary run has succeeded
  yet against the corrected recipe — the first canary attempt (killed
  mid-run) used the pre-review recipe and its results are void.

## Goal

Plan 0097 confirmed ADR's capacity-inverse self-distillation benefit at
CIFAR-10 scale (MobileNetV2 +2.79pp AutoAttack vs ResNet-18 Nesterov-matched
+1.22pp, SIGN_CONFIRMED, decision packet 0009). `docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md`
identified two things nobody has measured: (1) whether this capacity-inverse
trend holds or strengthens across a much wider capacity gap than ADR's own
CIFAR-10/WRN-34-10 comparison (~11.2M vs ~46.2M, a ~4x ratio) when moved to
ImageNet-1k and shrunk toward genuinely mobile scale instead of grown toward
larger scale (~11.2M ResNet-18 down to ~2.5M MobileNetV3-Small, a ~4.5x
ratio in the opposite direction); (2) whether a genuinely mobile-scale
(<5M param) model adversarially trained with a modern recipe and evaluated
with AutoAttack has ever been measured at all (it has not, per this
project's own literature survey this session and in the proposal document).
This plan answers both with one campaign.

**The proposed method, as a testable claim** (unchanged from the proposal
document, §2.2): teacher-free EMA self-distillation's robustness margin over
a matched, modern-recipe adversarially trained baseline grows monotonically
as model capacity shrinks, from ResNet-18 to a genuinely mobile architecture,
on ImageNet-1k under AutoAttack.

## Mechanism: unchanged, reusing plan 0097's validated `adr`

Exactly the mechanism plan 0097 already implemented, tested, scientifically
reviewed, and validated on CIFAR-10 — EMA-of-student teacher (γ=0.995),
per-sample rectification (paper Eq. 3-5), τ and λ both annealed by the
existing per-iteration cosine schedule (`lambda_source: cosine`, the
default — **not** plan 0098's shelved `gap_adaptive` variant, which never
reached a working state; decision packet 0012). No new engine code for the
mechanism itself. `method.adr` config block:
`ema_decay: 0.995, temperature_high: 2.0, temperature_low: 1.5, lambda_low: 0.5, lambda_high: 0.9`
— **not** the CIFAR-10 configs' values (`temperature_high: 2.5,
temperature_low: 2.0, lambda_low: 0.7, lambda_high: 0.95`); see the Progress
log's P0-1 entry for why the original draft's assumption that these are
dataset-independent was wrong, and why this plan instead uses the pinned
upstream's own Tiny-ImageNet (200-class) setting as the closest published
anchor to this plan's 1000 classes.

## Recipe: sourced from literature, not invented this session

Every numeric choice below is traced to a specific citation, verified this
session (not recalled), to avoid an arbitrary or silently-changed threat
model per CLAUDE.md rule 6:

| Field | Value | Source |
|---|---|---|
| Threat model | Linf, ε=4/255 | Already this project's own cited convention (Salman et al. 2020, RobustBench) |
| Training PGD steps | 3 | Salman et al. 2020, NeurIPS, Appendix A.1 (arXiv 2007.08489) — verified this session: "3 attack steps and a step size of ε×2/3" for exactly this ε |
| Training PGD step size | ε×2/3 = 8/765 | Same source, same appendix line |
| Initialization | ImageNet-1k pretrained, non-robust (standard torchvision weights) | Singh, Croce & Hein 2023, NeurIPS (arXiv 2303.01870), verified this session from the primary text: main recipe "initialize adversarial training from a fully trained (on ImageNet-1k) standard model"; pretrained source explicitly stated as "standard models available in the timm library or from the original papers" — ordinary public torchvision/timm ImageNet-1k weights are an in-kind substitute, not an approximation |
| Adversarial fine-tune epochs | 50 | Same paper, Section 5 (the actual 1-vs-2-vs-3-step cost/accuracy ablation this session's step-count recommendation rests on) — verified this session to be 50 epochs from pretrained init, not the longer 100-300 epoch numbers on some of that paper's *best released checkpoints*, which are a separate, longer push not used in the ablation itself |
| Evaluation (official test) | AutoAttack, standard, from a saved checkpoint, separate process, `--allow-autoattack` | This project's own standing rule (CLAUDE.md rule 6, `.claude/rules/scientific-core.md`) |
| Batch size | 128 per rank, uniform across every arm/architecture | Measured this session: all three benchmarked architectures (ResNet-18, ResNet-50, MobileNetV3-Small) are GPU-compute-bound at batch 128 on a 4090 — doubling batch size or DataLoader workers produced no measurable throughput change (ResNet-50: 99.5 vs 99.2 img/s at batch 128 vs 192; ResNet-18: 302.8 vs 297.9 img/s at 8 vs 16 workers) — so a uniform batch size costs nothing and matches this project's standing convention that batch size/world size are part of a run's frozen execution identity (`.claude/rules/scientific-core.md`) |

**Not sourced from a citation, this project's own existing convention**:
SGD, Nesterov momentum 0.9. **Changed from the CIFAR configs, not carried
over unchanged as an earlier draft of this table claimed**: weight decay
1e-4 (CIFAR configs use 5e-4 — a real, deliberate ImageNet-convention value,
not an oversight, but flagged since the first draft mischaracterized it as
unchanged). **LR schedule**: `warmup_multistep` (`src/ard/schedules/__init__.py`) —
a linear warmup over the first 10 of 50 epochs (~20%, matching
Singh/Croce/Hein 2023's own pretrained-init recipe shape, see the Progress
log's P1-5 entry), reaching the base LR exactly at epoch 10, then multistep
decay at milestones [25, 38] (both past the warmup window; proportionally
scaled from this project's own CIFAR 200-epoch schedule's 50%/75% points),
gamma 0.1. Base LR **0.05**, the standard linear-scaling-rule value (Goyal
et al. 2017) for this plan's batch size of 128 against the conventional
ImageNet batch-256/LR-0.1 anchor — not the CIFAR configs' 0.1, and not
Singh/Croce/Hein's own 1e-3 (an AdamW/ConvNeXt-family value that does not
transfer numerically to this plan's SGD recipe). **Held-out validation fraction**: 0.02
(≈25,600 images), deliberately smaller than the CIFAR scientific configs'
own 0.1 (an earlier draft of this line misstated that default as 0.25) —
at ImageNet scale even a 10% held-out slice would materially inflate the
per-epoch selection-attack cost for no real gain in selection reliability;
this is a cost engineering choice, not a change to any of CLAUDE.md rule 6's
protected fields. **Selection attack**: PGD-10, hard-label CE, eval mode —
stronger than the 3-step training attack (matching this project's own CIFAR
convention of a materially stronger selection attack than the training
attack), cheaper than full AutoAttack, run only on the small held-out slice.

## Experimental design

2 architectures × 2 arms × 3 seeds = 12 training runs, each with a full
official-test AutoAttack evaluation (best and last checkpoint, matching
this project's standing rule):

| arm | method | architecture | seeds |
|---|---|---|---|
| `r18_baseline` | `pgd_at` | `resnet18_imagenet` | 0, 1, 2 |
| `r18_adr` | `adr` | `resnet18_imagenet` | 0, 1, 2 |
| `mobilenetv3_baseline` | `pgd_at` | `mobilenet_v3_small_imagenet` | 0, 1, 2 |
| `mobilenetv3_adr` | `adr` | `mobilenet_v3_small_imagenet` | 0, 1, 2 |

`resnet50_imagenet` is registered (plan 0099) but deliberately **not** used
in this campaign — see Status.

## Preregistered decision rule

For each architecture, compute `adr`'s mean AutoAttack gain over its own
matched `pgd_at` baseline (best checkpoint, official test):

- **IMPROVED** if MobileNetV3-Small's mean gain exceeds ResNet-18's mean
  gain by more than the combined per-arm standard deviation (n=3 — a
  directional signal, not a population claim, per this project's own claims
  discipline), matching the CIFAR-10 campaign's own ordering
  (MobileNetV2 > ResNet-18).
- **NULL** if the two architectures' gains are within that combined sd of
  each other.
- **REVERSED** if ResNet-18's gain exceeds MobileNetV3-Small's by more than
  that margin — the capacity-inverse trend does not extend to this wider
  gap, itself a genuine, reportable finding per the proposal document's own
  framing (§2.2: "both outcomes are worth reporting").

This project has no established ImageNet-scale seed-to-seed noise floor yet
(unlike CIFAR-10's measured 1-2pp) — the sd figure above will be this
campaign's own first measurement of it, not an assumed prior value. State
this explicitly in the eventual report; do not read n=3 as more decisive
than it is (`.claude/rules/results-records.md`).

## GPU-hour budget (from this session's real measurements, not a guess)

Benchmarked this session on a single Hamster RTX 4090, real ImageNet images,
AMP on: ResNet-18 at 7-step PGD/batch 128 measured 302.8 img/s; scaling to
this plan's 3-step training attack by relative per-image attack cost
(≈1.93x fewer forward/backward passes) gives an estimated **≈580 img/s**.
MobileNetV3-Small measured 612.6 img/s at 7-step, scaling to **≈1,180 img/s**
at 3-step. At 50 epochs (not 100) and ≈1.25M training images/epoch
(1.28M × 0.98, after the 0.02 validation hold-out):

| architecture | est. img/s (3-step) | est. hours/run (50 epochs) | 6 runs (3 seeds × 2 arms) |
|---|---:|---:|---:|
| ResNet-18 | ≈580 | ≈30 | ≈180 GPU-hours |
| MobileNetV3-Small | ≈1,180 | ≈15 | ≈90 GPU-hours |

**Total ≈270 GPU-hours training**, plus AutoAttack (this project's own
existing anchor is ≈1 GPU-hour/checkpoint on a 5k-image subset at CIFAR
scale; ImageNet-scale AutoAttack cost has not been measured by this project
and should be treated as unverified until a real evaluation run measures
it — do not silently reuse the CIFAR anchor across datasets). Across 5 GPUs
(Hamster ×2, Ferret ×3), training alone is roughly ≈54 hours (**≈2.3 days**)
of wall clock if evenly distributed. This is an estimate scaled from a
7-step measurement, not a direct 3-step measurement — a real 3-step canary
(Verification step 2 below) should confirm it before the full campaign is
costed as final.

## Implementation checklist

1. **Pretrained-weight support.** Every ImageNet architecture registered in
   plan 0099 (`src/ard/models/registry.py`) currently hardcodes
   `weights=None`. Add a way to request the standard torchvision ImageNet-1k
   pretrained weights instead (e.g. a `pretrained: bool` field on
   `ModelConfig`, default `False` so every existing CIFAR/dev config is
   unchanged) for `resnet18_imagenet` and `mobilenet_v3_small_imagenet`
   specifically. Note the head/classifier layer must still be
   reinitialized/resized for `num_classes=1000` — for the real ImageNet-1k
   task this is a no-op (1000 classes already), unlike a future
   fewer-classes dev config.
2. **A real (non-`dev`) protocol identity for this campaign** — e.g.
   `controlled_imagenet_stage01_r18_mobilenetv3_adr_v1` — distinct from plan
   0099's `imagenet_stage0_dev_v1` (which is dev-tier-only, unconstrained,
   and explicitly scoped as "never a scientific campaign"). Decide during
   implementation whether this needs the full field-matching strict
   contract `_validate_protocol_contract` gives the CIFAR protocols, or a
   lighter identity-only registration (as `imagenet_stage0_dev_v1` has) is
   acceptable for this first ImageNet campaign — document whichever is
   chosen and why, since it changes how much this config's fields are
   protected against silent drift between the 12 runs.
3. **Real dataset content identity.** `tier: production` with
   `dataset.name: imagenet` requires `dataset.content_sha256`
   (`src/ard/config/schema.py`) — compute it once against the real,
   full 1.28M-image dataset at `/home/shunsukenaito/workspace-local/datasets/imagenet/`
   (read-only) using `ImageNetDataset`'s manifest-based identity (plan 0099
   — cheap, not a full byte hash) and pin the resulting digest in every
   config below.
4. **Four configs** (`configs/scientific/` or wherever this project's
   convention places a first-of-its-kind dataset's configs —
   check/establish this during implementation), one per arm ×
   architecture pairing in the table above, sharing every field except
   `method.id`/`method.adr` and `student.architecture`.

## Tests

- `pretrained: true` actually loads torchvision's pretrained weights (a
  cheap check: compare a few loaded parameter values against a fresh
  `weights=IMAGENET1K_V1` construction, not a full forward-pass equality
  check).
- `pretrained: false`/absent (default) is bit-identical to today's
  `weights=None` behavior — matching this session's own established
  discipline (plan 0098/0099) that a new optional field must not change
  any existing config's behavior.
- New protocol identity resolves and, whichever contract level is chosen in
  checklist item 2, is exercised by at least one test analogous to
  `tests/unit/test_config.py`'s `test_top_level_configs_resolve_under_controlled_environment`.

## Scientific review

This touches `src/ard/models/registry.py` and `src/ard/config/schema.py`
(new `pretrained` field) and adds a new protocol identity — per this
project's standing rule, needs the scientific-reviewer agent before any
source SHA is frozen for this campaign, same discipline as every prior
milestone this session.

## Verification, end to end

1. `scripts/verify.py --changed` after implementation.
2. **One real 3-step GPU canary** (a few epochs, not the full 50) on both
   architectures before committing to the full 12-run campaign — this
   session's cost table above is a 7-step-to-3-step *estimate*, not a
   direct measurement; confirm real 3-step throughput and that pretrained
   initialization loads and trains without error before freezing the
   budget as final.
3. Full 12-run campaign per the experimental design above, launched from a
   pinned worktree, once the above pass and scientific review is complete.
4. AutoAttack official test for all 12 runs (best + last checkpoint), from
   saved checkpoints, separate process, `--allow-autoattack`.
5. Aggregation, decision packet, per this project's standing postrun
   discipline (`.claude/rules/results-records.md`) — a new aggregator
   script for this campaign's own contract, following
   `scripts/aggregate_adr_cifar10_replication.py`'s shape.

## Progress log

- 2026-09-11/12: implementation (`853aa76`) and a first canary attempt.
  scientific-reviewer ran on the full diff before any canary was allowed to
  count, per this project's standing rule. **Verdict: two P0 findings, hold
  on the canary.** The two running canary processes (`imagenet_r18_adr`,
  `imagenet_mobilenetv3_adr`, both mid-flight on the pre-review recipe) were
  killed immediately once the review returned; their partial results are
  void and were never recorded anywhere. All findings below were resolved
  before any further GPU time was spent.
  - **P0-1 (ADR temperature/lambda copied from CIFAR-10 unchanged).** The
    pinned upstream ADR code (`.external/adr/config/`) lowers
    temperature/lambda monotonically as class count rises (CIFAR-10 10cls:
    T=[2.0,2.5] lambda=[0.7,0.95]; CIFAR-100 100cls: T=[1.0,1.5]
    lambda=[0.7,0.95]; Tiny-ImageNet 200cls: T=[?,2.0] lambda=[0.5,0.9] SGD,
    or [1.5,2.0]/[0.3,0.9] with AWP) -- this plan's original configs used
    the 10-class corner (T=[2.0,2.5], lambda=[0.7,0.95]) for a 1000-class
    problem, the least appropriate of the three published settings. Softmax
    entropy at fixed T grows with class count, so this risked degenerating
    the rectified target into near-uniform 999-way label smoothing late in
    training -- exactly the campaign's measured quantity (adr's gain over
    pgd_at) getting silently suppressed by mistuning, not absence of effect.
    **Resolved**: found and fixed a typo in the pinned upstream's own Tiny-
    ImageNet SGD gin config (`resnet18_pgd_sgd_adr.gin:24`,
    `AdvTrainer.tmperature_low = 1.5` -- misspelled key, so gin never
    actually binds it) -- the AWP sibling config sets the same field
    correctly to 1.5, so 1.5 is the best-evidenced reading of the SGD
    variant's real intent. Adopted **temperature_high=2.0,
    temperature_low=1.5, lambda_low=0.5, lambda_high=0.9** (Tiny-ImageNet's
    200-class setting, the closest published anchor to this plan's 1000
    classes) in both `imagenet_r18_adr.yaml` and `imagenet_mobilenetv3_adr.yaml`.
    Also added a permanent, cheap diagnostic (`train_rectified_true_class_mass`,
    `src/ard/engine/trainer.py`) logging the rectified target's own mean
    true-class probability mass every epoch, for every future adr run, not
    just this campaign -- a value near 1/num_classes would mean the
    mechanism has degenerated for whatever class count is in use. The human
    explicitly declined a general CIFAR-side auto-calibration system for
    temperature/lambda (reasoning: the three published points aren't a
    clean function of class count -- Tiny-ImageNet's jump from CIFAR-100
    doesn't fit a simple log(K) extrapolation from the two CIFAR points --
    and CIFAR's own resolution/domain wouldn't de-risk extrapolating to
    ImageNet-1k/224px anyway); the diagnostic plus the closest published
    anchor was judged sufficient, checked at the next real canary.
  - **P0-2 (zero data augmentation for imagenet, `augmentation_policy:
    canonical` recorded but not applied).** `build_train_validation_views`
    had no branch for `config.name == "imagenet"` -- every epoch of all 50
    planned epochs would have trained on the exact same deterministically-
    resized image, no crop, no flip, while the resolved config still claimed
    `canonical`. **Resolved**: `ImageNetDataset.__getitem__` no longer
    resizes at all (returns the native-resolution image); a new
    `EpochImageNetTransform` (`src/ard/data/datasets.py`) implements
    deterministic, per-epoch/per-source-ID-keyed RandomResizedCrop + random
    horizontal flip (torchvision's own algorithm, scale=[0.08,1.0],
    ratio=[3/4,4/3], 10 attempts then a centered-square fallback, reimplemented
    against a local `torch.Generator` since torchvision's own
    `RandomResizedCrop.get_params` has no injectable generator) for training,
    and a new `ImageNetEvalTransform` (deterministic resize-256/224-crop,
    matching the pretrained weights' own preprocessing convention) for
    validation-during-training and the official evaluation split. Both are
    exported from `ard.data`. New tests confirm the training view actually
    varies epoch to epoch (the reviewer's own suggested targeted test),
    reproduces exactly for a fixed epoch/source ID (resume correctness), and
    that the validation view stays unaugmented and epoch-invariant.
  - **P1-3 (`ard.cli.evaluate` had no ImageNet branch)**: `_dataset_identity`
    (`src/ard/cli/evaluate.py`) gained an `imagenet` branch mirroring
    `tiny_imagenet`'s exactly (manifest-based fingerprint, `observed`-vs-
    `expected-unverified` verification shape). Every one of this campaign's
    12 evaluations would otherwise have failed immediately on
    "evaluation dataset requires an explicit portable content fingerprint" --
    discoverable only after ~270 GPU-hours of training. New direct unit tests
    (`tests/unit/test_evaluation.py`) cover both the observed and
    expected-only paths and the still-correct rejection when neither is given.
  - **P1-4 (AutoAttack materializes the full evaluation set on GPU, ~30 GB
    for ImageNet's 50k val images, guaranteed OOM)**: new
    `EvaluationConfig.autoattack_sample_count` (default `None`, preserving
    every existing CIFAR config's exact behavior of attacking every image)
    and a new `_autoattack_loader` helper (`src/ard/cli/evaluate.py`) that
    takes a fixed-seed **uniform random** subset -- not a prefix, since
    `ImageNetDataset`'s own sample ordering is grouped by class and a prefix
    would silently concentrate on the first few classes only. All four
    configs set `autoattack_sample_count: 5000` (RobustBench's own
    ImageNet-scale convention). New unit tests cover the unaffected default,
    the random-not-prefix property, determinism for a fixed seed, and the
    sample-count-exceeds-dataset-size fallback.
  - **P1-5 (LR=0.1 with no warmup applied directly to pretrained weights)**:
    a follow-up literature check found Singh/Croce/Hein 2023's own
    pretrained-init recipe uses a **linear warmup over the first ~20% of
    the run (10 of 50 epochs), reaching peak LR exactly at epoch 10, then
    decay** -- their literal LR (1e-3) is an AdamW/ConvNeXt-family value and
    does not transfer numerically to this plan's SGD/ResNet-18/MobileNetV3
    recipe, but the warmup *shape* does. New `SchedulerConfig` id
    `warmup_multistep` plus `warmup_epochs` field
    (`src/ard/config/schema.py`), implemented via a pure
    `warmup_multistep_multiplier` function and `LambdaLR`
    (`src/ard/schedules/__init__.py`), unit-tested against a hand-computed
    sequence and a resume/state_dict-exactness test mirroring the existing
    `multistep` scheduler's own test. All four configs now use
    `warmup_multistep` with `warmup_epochs: 10`, milestones unchanged at
    `[25, 38]` (both already past the warmup window). Base LR also lowered
    from 0.1 to **0.05**, applying the standard linear-scaling rule (Goyal
    et al. 2017) for this plan's batch size of 128 against the
    literature's conventional batch-256 anchor -- a well-established,
    directly-citable SGD convention, not a guess.
  - **P2-7 (regression: `pretrained=False` no longer bit-identical to
    before `701d3d5` when `num_classes != 1000`)**: `_with_replaced_head`
    was called unconditionally on both the pretrained and non-pretrained
    branches in `build_architecture`; fixed to only run inside the
    `pretrained=True` branch. New regression test confirms `pretrained=False`
    at a non-1000 class count constructs bit-identical initial weights to
    the pre-`701d3d5` behavior (same manual seed, same resulting head
    parameters) -- this had no effect on any of this campaign's own
    committed configs (all `num_classes: 1000`), only on future dev-scale
    use of these architectures.
  - **P2-9 (`amp: true` was a silent no-op)**: confirmed by reading every
    `torch.autocast` call site in `src/` -- all are `enabled=False`
    (deliberately, for detached diagnostic forwards); nothing ever enables
    real mixed-precision compute. `GradScaler` alone without autocast
    provides no speedup and only adds an unbounded-then-halving loss-scale
    walk with a negligible (~1-in-2000-step) skipped-update rate, identical
    in expectation across arms. Real autocast wiring would need its own
    scientific review (confirming no rectified-target/EMA/attack computation
    silently drops to FP16) and is out of scope here; the safe fix taken was
    to set `amp: false` in all four configs, matching what the engine
    actually does today rather than what the flag implies. Plan 0099's own
    throughput numbers, measured under this same (effectively FP32) regime,
    are unaffected and remain the basis for this plan's GPU-hour budget.
  - **P2-12 (no safety net once the strict protocol contract was skipped)**:
    added `test_the_four_imagenet_stage01_configs_are_field_identical_except_architecture_method_and_group`
    (`tests/unit/test_config.py`) -- loads and normalizes all four configs,
    asserting every field agrees except `student.architecture`, `method`,
    and `tracking.group`. A future edit to only one of the four (epochs,
    epsilon, batch size, etc.) now fails `scripts/verify.py --changed`
    instead of silently breaking the controlled comparison.
  - **Not done, deliberately deferred**: P2-8 (pinning a SHA-256 of the
    fetched pretrained-weight state_dict in run lineage) and pre-warming/
    verifying the torch hub weights cache on Ferret before launch --
    real gaps, but neither blocks a canary run on Hamster (which already
    has both architectures' pretrained weights cached from this session's
    canary attempts).
  - `scripts/verify.py --changed` green across every fix above (T0-T3).
  - **Next action**: a real GPU canary against this corrected recipe,
    covering all four arm/architecture combinations (both killed canaries
    this round were `adr` runs; `pgd_at` has never been canaried against
    the fixed augmentation/warmup/LR recipe at all), logging the new
    `train_rectified_true_class_mass` diagnostic to confirm P0-1's fix is
    not itself degenerate, before committing to the full 12-run campaign.
