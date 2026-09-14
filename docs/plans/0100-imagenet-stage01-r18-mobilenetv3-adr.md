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
  all findings resolved, and a real 4-arm GPU canary against the corrected
  recipe succeeded (see Progress log) — pretrained features preserved, no
  sign of the degenerate rectified-target smoothing P0-1 was worried about.
  Proceeding to launch the full 12-run campaign via `/experiment-launch`,
  starting on Hamster's 2 GPUs (Ferret occupied by another user).

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

## Stop rules

A checkpoint failing lineage validation, an attack-identity mismatch between
training and evaluation, or a run whose resolved config diverges from its
three sibling configs (caught by
`test_the_four_imagenet_stage01_configs_are_field_identical_except_architecture_method_and_group`,
since this campaign's protocol identity deliberately carries no strict
field-matching contract of its own) blocks that job's result from being
reported; technical failures may retry with the identical scientific
identity. An `adr` run whose `train_rectified_true_class_mass` diagnostic
decays toward chance level (~1/1000) rather than stabilizing at a level
meaningfully above it is a scientific stop for that arm, not a silent
continuation — it means the temperature/lambda pair chosen in the Progress
log's P0-1 entry has degenerated for this run's actual dynamics, and the
human decides the fix (not a new schedule invented mid-campaign). Weak or
null results (sign not confirmed, or REVERSED per the decision rule above)
are reported as such and do not trigger a fourth seed, a new architecture,
or scope expansion without a fresh decision packet.

## GPU-hour budget (from real 3-step canary measurements, superseding the earlier 7-step-scaled estimate)

The 3-epoch, 4-arm canary run this session (Progress log) measured real
throughput under the *actual* recipe (3-step PGD, real RandomResizedCrop
augmentation, pretrained init, `warmup_multistep`) on a single Hamster
RTX 4090 — lower than the earlier 7-step-scaled estimate, because that
estimate didn't account for augmentation's own per-iteration CPU cost,
which doesn't shrink when attack steps do:

| architecture | method | measured img/s | hours/run (50 epochs, ≈1.256M train images/epoch) | 3 seeds |
|---|---|---:|---:|---:|
| ResNet-18 | `pgd_at` | ≈505 | ≈34.5 | ≈104 GPU-hours |
| ResNet-18 | `adr` | ≈460 | ≈37.9 | ≈114 GPU-hours |
| MobileNetV3-Small | `pgd_at` | ≈910 | ≈19.2 | ≈58 GPU-hours |
| MobileNetV3-Small | `adr` | ≈830 | ≈21.0 | ≈63 GPU-hours |

**Total ≈338 GPU-hours training** (not the earlier ≈270-hour estimate),
plus AutoAttack (this project's own existing anchor is ≈1 GPU-hour/checkpoint
on a 5k-image subset at CIFAR scale; ImageNet-scale AutoAttack cost has not
been measured by this project and should be treated as unverified until a
real evaluation run measures it — do not silently reuse the CIFAR anchor
across datasets). On Hamster's 2 GPUs alone, training alone is roughly
**≈7 days** of wall clock if evenly distributed; adding Ferret's 3 GPUs
(unavailable to this campaign as of the decision to launch — another user
is running work there) would bring this to **≈2.8 days**. The human's
explicit decision (chat, this session): start now on Hamster's 2 GPUs alone
rather than wait for Ferret, and fold Ferret's GPUs into the same campaign's
remaining jobs once it frees up.

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
- 2026-09-12: real 3-epoch, 4-arm canary against the corrected recipe, all
  four succeeded with no crashes (worktree `source-9b8705235327`, sha
  `9b870523532761c09c9d280bb45e641c5c633c36`, Hamster GPUs 0/1, run outputs
  under scratch, never committed). Pretrained features survived the
  warmup+lowered-LR fix (val clean accuracy 37-48% across all four arms by
  epoch 2, nowhere near the ~0.1% chance level a destroyed init would show).
  `train_rectified_true_class_mass` (P0-1's diagnostic): ResNet-18 adr
  0.472/0.307/0.141 across epochs 0-2, MobileNetV3-Small adr
  0.473/0.306/0.139 -- nearly identical trajectories across architectures,
  all epochs comfortably above the ~0.001 chance level for 1000 classes, no
  sign of the degenerate near-uniform smoothing P0-1 was worried about. The
  declining trend across just 3 epochs is explained by the EMA teacher
  rapidly becoming better-calibrated early in fine-tuning (`train_ema_student_agreement`
  rising 0.82->0.89 for both `adr` arms over the same 3 epochs) --
  per-sample lambda_i rising with teacher confidence pulls the blended
  target's true-class mass down from near-1.0 (one-hot-dominated) toward
  the teacher's own (correctly, <1.0) softened value, which is the intended
  mechanism, not a bug. Only 3 epochs of evidence; this diagnostic keeps
  being logged for the full 50-epoch runs and stays a live stop-rule check,
  not a one-time canary box to tick. Real throughput used to correct the
  GPU-hour budget above. Human decision (chat): launch the full 12-run
  campaign now on Hamster's 2 GPUs, not waiting for Ferret (occupied by
  another user), folding Ferret in once free. Proceeding to
  `/experiment-launch`.
- 2026-09-12: launched the full 12-run campaign, `/experiment-launch` steps
  3-6. One mechanical fix was needed before pinning: `ard.cli.train` has no
  `--epochs` flag (only a `training.epochs=N` dot-path override), but the
  launch gate's `_replace_epoch_bound` unconditionally rewrites a literal
  `--epochs` token in every training job's command -- every prior gated
  campaign wrapped its own launcher script for the same reason (e.g.
  `scripts/run_ert_i100_online_state_s2.py`); this is the first plain,
  non-forked `ard.cli.train` campaign through the gate, so it needed the
  same treatment. Added `scripts/run_imagenet_stage01_train.py` (execution-
  plane only, no `src/ard/` change): forwards `--epochs` verbatim as
  `training.epochs=<value>`, confirmatory since all four configs already
  set 50. Committed as `ae4dd7c`; pinned worktree `source-ae4dd7c82d80`
  (sha `ae4dd7c82d801ae39cfe41a89ec5427e59e202ce`). Campaign spec: 12
  training jobs (4 arms x seeds 0/1/2), `teacher: {identity: "none"}` (both
  methods are teacher-free per this plan's own design -- the gate has no
  "no teacher" sentinel, so this is the honest declaration),
  `operational_profile: FULL_NEW_INTEGRATION`, host `hamster` only (2
  GPUs), no evaluation/aggregation nodes (per this project's convention,
  those are `/experiment-postrun`'s job). Attempt1
  (`imagenet-stage01-r18-mobilenetv3-adr-v1-attempt1`): preflight, dry-run
  and a first `--canary-only` all passed, but `--launch` failed -- its own
  internal canary re-run hit `ard.cli.train`'s `_guard_output` ("refusing
  to overwrite existing output directory without --resume"), because the
  canary's `--dry-run` had already written `resolved_config.yaml` into the
  fixed `gate_dir/canary/<job_id>` path during the prior `--canary-only`
  pass, and the gate reruns every canary entry on both `--canary-only` and
  `--launch` against that same path. Fixed by having each canary command
  `mktemp -d` its own scratch output directory (verified idempotent
  standalone) instead of a fixed path; no gate check was weakened.
  Attempt2 (`imagenet-stage01-r18-mobilenetv3-adr-v1-attempt2`): preflight,
  dry-run, canary and launch all passed cleanly. Resolved-manifest SHA-256
  `4d392e360b89c722864c38270ba142fae611ffc45fdb099735a0324630fc946c`, gate
  dir
  `<runtime>/runs/imagenet-stage01-r18-mobilenetv3-adr-v1-attempt2/launch-gate`,
  controller PID 3559429. Confirmed handoff: `orchestrate.py status` shows
  the campaign `running` with `r18_adr-s0`/`r18_adr-s1` already `running`
  (Hamster's 2 GPUs) and the other 10 jobs `pending`; `ardx-watch.service`
  is `active`. `/experiment-postrun` closes this out once the watcher
  reports the campaign terminal.
- 2026-09-12 (chat, ~90 min into attempt2): human reconsidered the pacing
  of this plan against the project's own CIFAR-first iteration philosophy
  (decision packets 0009/0011) and chose to restructure before committing
  the full 7-day/338-GPU-hour budget. Rationale: only a 3-epoch canary had
  been run against the corrected recipe (P0-1's diagnostic, epoch-25/38 LR
  decay milestones, and the full AutoAttack evaluation pipeline against a
  real 50-epoch checkpoint were all still unverified at full horizon), and
  attempt2 had both Hamster GPUs running the *same* arm (`r18_adr` seeds
  0/1) rather than covering all four arm/architecture configs first.
  Decision: abort attempt2 and restructure to a staged design -- **stage 1:
  4 arms x seed 0 only** (one job per arm/architecture, all four configs
  exercised end-to-end once each) on Hamster's 2 GPUs; only after stage 1's
  four jobs each complete a full 50-epoch run with a clean AutoAttack
  evaluation does seed 1/2 (8 more jobs) get launched as a follow-up
  campaign. This mirrors the human's own stated preference going forward:
  new ImageNet-scale questions start at 1 seed / shorter horizon before
  scaling to the full seed count, rather than committing 3 seeds x full
  epochs from the first launch.
  Aborted attempt2 cleanly: `kill -TERM 3559429` (controller only, verified
  dead before touching workers -- killing a worker first risks the live
  controller marking a job `orphaned`/`failed`, a terminal `jobs[*].status`
  that would make `classify_campaign` (`scripts/ardx/ardx_common.py:186`,
  read directly to confirm) declare the whole campaign `failed` and
  fire `postrun_hook.sh`'s headless `/experiment-postrun`, which is exactly
  what must not happen for a deliberate human abort). Then `kill -TERM
  -3559433` / `-3559434` (negative PID = process group, confirmed via `ps
  -o pgid` that each worker and its `run_imagenet_stage01_train.py` child
  share one PGID, no `setsid` in between) to take down both worker+trainer
  trees. Verified clean: no matching PID in `ps`, `nvidia-smi
  --query-compute-apps` empty on both Hamster GPUs. `state.json` for
  attempt2 is left as-is (still shows `r18_adr-s0`/`s1` `running` with a
  dead `controller_pid` -- a known, harmless census artifact per
  `ardx_common.py`; the other 10 jobs stay `pending` forever, so
  `classify_campaign` never marks this campaign terminal and no postrun
  automation fires). `r18_adr-s0/train` and `r18_adr-s1/train` under
  `<runtime>/runs/imagenet-stage01-r18-mobilenetv3-adr-v1/` hold ~50 min of
  partial checkpoints each and are abandoned, not resumed -- the stage-1
  restructuring uses a fresh campaign id so these paths are never reused.
  Total sunk cost: ~3 GPU-hours (2 GPUs x ~1.5h), negligible against the
  full 338 GPU-hour budget. No source or config change; `ae4dd7c` and
  worktree `source-ae4dd7c82d80` remain valid for stage 1. Proceeding to
  `/experiment-launch` again with a new campaign id for stage 1 (4 jobs:
  `r18_baseline-s0`, `r18_adr-s0`, `mobilenetv3_baseline-s0`,
  `mobilenetv3_adr-s0`).
- 2026-09-12: launched stage 1 (4 jobs, seed 0 only) as a fresh campaign,
  `imagenet-stage01-r18-mobilenetv3-adr-v2`, reusing the same source SHA and
  pinned worktree (no re-pin needed -- confirmed no scientific-core files
  changed). Hand-authored a new campaign spec (the original 12-job spec was
  never committed, per the launch skill's own convention of committing only
  the plan file, and was lost with the prior session) by reverse-engineering
  the input shape from attempt2's `resolved-manifest.json` plus
  `.agents/skills/production-launch-gate/references/campaign-spec.md`.
  Verified byte-for-byte before trusting it: each of the four jobs'
  `identity_hash` in the newly resolved manifest matches attempt2's
  corresponding seed-0 job hash exactly (`dcf21aece735...`,
  `62f8fd07e5e7...`, `e11c6aebec78...`, `8c994d18adf9...`), confirming the
  scientific identity (arms, attack fields, augmentation, rng, dataset,
  epoch bounds, source SHA) is unchanged from the original 12-job contract --
  this launch differs only in which subset of jobs runs now. Two spec bugs
  caught and fixed in one batch before the first gate attempt: (1) a job
  needs `config_sha256` explicitly (the gate does not auto-hash a
  `scientific_config` role when the field is omitted, unlike mask/teacher
  roles), confirmed by reading `validate_artifact` directly; (2) per-job
  `rng` is not a real field -- the gate reads `rng_contract` from the
  **top-level spec** (`spec.get("rng_contract", job.get("rng_contract"))`,
  read directly from `launch_gate.py`), not from a job-level `rng` key.
  Attempt1's preflight passed with both fields silently null; caught by
  inspecting the resolved manifest rather than trusting the "pass" status
  alone, fixed both, re-ran clean from `--preflight-only` as attempt2.
  Attempt2's `--canary-only` then failed for an unrelated reason: the
  static-cli `--help` smoke hit `ModuleNotFoundError: No module named
  'ard'` -- `launch_gate.py`'s `_run_static_cli_entry` calls
  `subprocess.run(command, cwd=job["cwd"], ...)` with no `env=` override,
  so it inherits the gate process's own ambient environment rather than the
  job's declared `env` dict; CLAUDE.md's own standing instruction to always
  run this project's Python with `PYTHONPATH=src` turns out to be exactly
  what makes this work (a *relative* `PYTHONPATH=src` resolves against
  the child's cwd at interpreter start, which the gate sets to the
  worktree, landing on `<worktree>/src` correctly) -- this launch simply
  hadn't set it in-session. Fixed by exporting `PYTHONPATH=src` before
  invoking the gate; re-ran clean from `--preflight-only` as attempt3.
  Attempt3: preflight, dry-run, canary and launch all passed.
  Resolved-manifest SHA-256
  `99465e2c596b0c8d651fc4e06f31d872df24ede543384451c03c0be0f755bf2a`, gate
  dir
  `<runtime>/runs/imagenet-stage01-r18-mobilenetv3-adr-v2-attempt3/launch-gate`.
  Confirmed handoff: `orchestrate.py status` shows the campaign `running`
  with `mobilenetv3_adr-s0`/`mobilenetv3_baseline-s0` already `running`
  (Hamster's 2 GPUs) and `r18_adr-s0`/`r18_baseline-s0` `pending` (queued
  for the next free GPU); `ardx-watch.service` is `active`. Once all four
  stage-1 jobs complete with a clean AutoAttack evaluation (via
  `/experiment-postrun`), the human decides whether to launch stage 2
  (seeds 1/2, 8 more jobs) as a follow-up campaign.
- 2026-09-13 (~24h after launch): `mobilenetv3_baseline-s0` completed
  cleanly (`completion.json` present, `expected_outputs` all on disk).
  `r18_adr-s0` (the orchestrated attempt) came back `failed` after only
  ~5 seconds of execution -- root-caused from
  `orchestration/.../r18_adr-s0.attempt-1.log`: `wandb.errors.errors.UsageError:
  ... The value 'never' is not a valid option for resuming a run
  (imagenet-stage01-r18_adr-s0) that already exists.` The hand-authored v2
  spec copied `ARD_RUN_ID` verbatim from the old (aborted) v1 attempt2
  resolved manifest without re-namespacing it to the new campaign id; since
  `r18_adr-s0` (unlike the other three stage-1 jobs) had actually reached
  the tracker-init step under v1 before that campaign was aborted, a W&B
  run named `imagenet-stage01-r18_adr-s0` already existed server-side, and
  `resume=never` refused to reuse it. A purely execution-plane naming bug,
  not a scientific one -- the orchestrator classified it `failure_class:
  unknown`, `retryable: false` (its classifier doesn't pattern-match this
  W&B error, so it did not auto-retry despite `max_attempts: 2` allowing
  one more attempt). Both remaining GPUs were occupied
  (`mobilenetv3_adr-s0` near-finished on GPU 0 at epoch 43/49,
  `r18_baseline-s0` freshly started on GPU 1) with nothing pending in this
  manifest, so no orchestrated retry would occur even if it were
  reclassified -- an idle GPU would sit unused once `mobilenetv3_adr-s0`
  finished. Per CLAUDE.md rule 3 (infra blocking a ready campaign -> hand-run
  from the worktree with the run-bundle contract), fixed and hand-launched
  a replacement rather than waiting for the automated postrun's decision-packet
  path to fire on a bug already root-caused: moved the failed attempt's
  (checkpoint-free, tracker-init-only) output dir aside to
  `r18_adr-s0/train.attempt1-wandb-collision-failed`, then hand-ran the
  identical command (`run_imagenet_stage01_train.py --config
  imagenet_r18_adr.yaml --epochs 50 --output .../r18_adr-s0/train`, same
  worktree, same config, seed 0) with `ARD_RUN_ID` fully namespaced
  (`imagenet-stage01-r18-mobilenetv3-adr-v2-r18_adr-s0-handrun1`) and
  `CUDA_VISIBLE_DEVICES=0` -- deliberately co-located on GPU 0 with the
  near-finished `mobilenetv3_adr-s0` (ample headroom, ~4-6GB used of 24GB)
  rather than GPU 1's `r18_baseline-s0`, which has ~34 hours left; this
  costs at most ~2-3 hours of shared throughput before GPU 0 is exclusive
  again, versus co-locating for the full remaining ~34 hours on GPU 1.
  Confirmed past the tracker-init failure point (alive past 28s, actively
  running) before moving on. **Reconciliation note for
  `/experiment-postrun`**: this campaign's orchestrated manifest
  (`imagenet-stage01-r18-mobilenetv3-adr-v2`) will show `r18_adr-s0` as a
  terminal `failed` job once the other two orchestrated jobs finish,
  classifying the whole campaign `failed` -- that classification is stale.
  The real `r18_adr-s0` result is this hand-run
  (`ARD_RUN_ID=imagenet-stage01-r18-mobilenetv3-adr-v2-r18_adr-s0-handrun1`,
  output at the same `r18_adr-s0/train` path, same `identity_hash` inputs:
  config sha256 `56ed12e5f86fb1c62663d778c99da4e42ae13b2d2a5bf5994d3d046e0c83d951`,
  seed 0, source SHA `ae4dd7c82d801ae39cfe41a89ec5427e59e202ce`) and must be
  picked up via the hand-run path (`campaign_watch.py --include-hand-run`)
  once it completes, not read as a campaign failure.
- 2026-09-13T00:43Z: `mobilenetv3_adr-s0` completed (`/experiment-postrun`,
  run-bundle path). Terminal re-derived via `campaign_watch.py --once
  --include-hand-run`: `terminal: true`, `success: true`, 50/50 epoch rows.
  `completion.json` (its only declared `expected_output`) present with
  `identity_hash` `8c994d18adf9...` and source SHA `ae4dd7c82d80`, matching
  the attempt3 resolved manifest; `best.pt`, `best-ema.pt`, `last.pt` on disk.
  **Nothing imported**: no aggregator for this contract exists yet
  (Verification step 5), no AutoAttack evaluation has run (step 4), and stage 1
  is not complete (`r18_baseline-s0` and the `r18_adr-s0` hand-run still
  training). No record, report, ledger row or milestone tick.
  **Stop-rule check (P0-1 diagnostic)**: `train_rectified_true_class_mass`
  fell smoothly from 0.508 (epoch 0) to 0.105 (epoch 49), flattening over the
  last 5 epochs (0.117 -> 0.105), about 100x the 1/1000 chance level. Not the
  preregistered stop, but see below.
  **Internal validation only (held-out 2% slice, PGD-10; not an official
  result, n=1)**, same arch/seed/recipe, `adr` vs `pgd_at`:
  | run | best epoch | best clean / PGD | last clean / PGD |
  |---|---:|---|---|
  | `mobilenetv3_baseline-s0` (`pgd_at`) | 46 | 0.459 / 0.253 | 0.460 / 0.252 |
  | `mobilenetv3_adr-s0` (`adr`) | 3 | 0.423 / 0.219 | 0.321 / 0.208 |
  Both arms lose clean accuracy during warmup and the peak-LR phase (roughly
  0.45 -> 0.27-0.33) and recover at the epoch-25 decay. `pgd_at` recovers to
  0.46 clean after decay; `adr` only reaches 0.33. `adr`'s best checkpoint is
  therefore a warmup-phase epoch (3), not an end-of-schedule one. A plausible
  but unverified reading is that the late rectified target (~0.1 true-class
  mass) under-fits, a milder form of P0-1's concern. For the human at the
  stage-1 decision, after AutoAttack; no action taken.
- 2026-09-13 (chat): human flagged the `adr` clean-accuracy gap as too large
  to be seed noise and asked for root cause before stage 2. Independently
  corroborated the automated postrun's finding above by diffing
  `mobilenetv3_adr-s0` against `mobilenetv3_baseline-s0`'s full per-epoch
  trajectories: the dip-then-recover shape (clean accuracy crashing during
  the warmup/peak-LR window, epochs ~5-20, then climbing back after the
  epoch-25 LR decay) is **identical in both arms** -- not `adr`-specific, a
  normal consequence of fine-tuning a pretrained model through
  `warmup_multistep`. What *is* `adr`-specific: `pgd_at` recovers essentially
  back to its pre-dip level (45.9% clean) by epoch 45; `adr` does not
  (32.4%, below even its own pre-dip 41-43%). This tracks
  `train_rectified_true_class_mass` and `train_ema_student_agreement`
  directly: from epoch 25 on, agreement rises sharply (0.58 -> 0.96) while
  mass keeps falling (0.30 -> 0.11) through the entire recovery window --
  consistent with a self-reinforcing loop (student/EMA-teacher converge on
  each other's predictions; when that shared prediction under-weights the
  true class, the rectified target's hard-label signal keeps shrinking right
  when LR decay needs a strong corrective push). Verified `total_iterations`
  in `src/ard/engine/trainer.py`/`ard/cli/train.py:909`
  (`len(loader) * epochs`, real per-epoch batch count) is computed correctly
  -- not an implementation bug in the schedule's iteration count. Flagged as
  a real, unverified-but-plausible design interaction instead: the lambda
  cosine anneal spans the *entire* run, so its fastest-change point (cosine
  midpoint) falls at exactly 50% of training -- the same point as this plan's
  first LR-decay milestone (epoch 25/50, itself proportionally scaled from
  CIFAR's 50%/75% convention). Two independently-reasonable schedule choices
  compound at the same epoch. Also flagged: `lambda_low/high` and
  `temperature_high/low` are analogically transferred from the pinned ADR
  upstream's 200-class Tiny-ImageNet setting to this plan's 1000-class
  problem (already noted as an open assumption in the Mechanism section
  above); softmax entropy at a fixed temperature scales with class count, a
  plausible contributing factor to the low true-class mass, separate from
  the schedule-interaction hypothesis. Not yet distinguishable: whether this
  is capacity-specific (the proposal's actual hypothesis, just in the wrong
  direction) or shared across architectures -- `r18_adr-s0`'s hand-run and
  `mobilenetv3_adr-s0` show nearly identical `train_rectified_true_class_mass`
  through epoch 8 (0.479 vs 0.478), so the early trajectory is not yet
  capacity-dependent; whether R18 recovers better than MobileNetV3-Small
  after its own epoch-25 decay is the most direct pending test and needs
  `r18_adr-s0` to reach that epoch. On baseline health: `mobilenetv3_baseline-s0`'s
  46% clean / ~25% training-PGD is in the expected range for this budget
  (Salman et al. 2020's ResNet-18 -- a larger architecture -- reports 25.32%
  official AutoAttack robust accuracy at this same epsilon; a smaller
  MobileNetV3-Small landing near that on a weaker proxy attack is not a
  failure). A literature survey on 2020+ SOTA for adversarially-trained and
  standard-trained mobile-scale ImageNet architectures was launched
  (background) to check whether this project's recipe is leaving known
  accuracy on the table independent of the `adr`/schedule question above;
  findings to follow in a later entry.
- 2026-09-13 (chat): implemented decision 0004's option B (chosen
  2026-09-08, never actually built -- confirmed by grep before this session,
  and independently rediscovered by the automated postrun's decision packet
  `0013` for this exact campaign's `r18_adr-s0` failure, its fourth
  recurrence). Added `_reject_stale_remote_run_id` to
  `validate_tracking_guard` (`src/ard/tracking/adapter.py`): when
  `tracking.mode == "online"` and `tracking.run_id` is explicit and no local
  `run-bundle/manifest.json` exists yet (a fresh start, not a resume), query
  `wandb.Api().run(f"{entity}/{project}/{run_id}")` before any GPU or
  fork-seed work; any successful lookup (remote history exists) raises
  `TrackingError` immediately, matching this project's existing
  dependency-injectable `wandb_module` pattern (`_start_wandb`) for
  testability. Deliberately fails **open** on an inconclusive check (lookup
  raises, e.g. not-found or a transient network error) rather than closed --
  the goal is only to catch the specific known collision shape earlier, not
  to make every online launch depend on W&B's API being reachable at
  preflight time; a missed collision still fails exactly as before, just
  later. Threaded a new required `output_dir` parameter through both call
  sites (`ard/cli/train.py:674`, `ard/cli/evaluate.py:293`) and the two
  existing tests. Added three regression tests
  (`tests/unit/test_tracking.py::test_online_guard_*`): rejects a fresh
  start against a remote-run-found fake wandb module, allows one against a
  not-found fake, and proves a legitimate resume (local manifest present)
  never touches wandb at all (`_ExplodingWandb`, any attribute access
  raises). `scripts/verify.py --changed` green across T0/T1/T3 (61 tracking
  unit tests, all touched integration/regression tiers). **This touches
  `src/ard/tracking/` (W&B lineage) and needs scientific-reviewer per
  CLAUDE.md before its SHA is frozen for stage 2** -- not yet requested.
  Separately, fixed the recurrence vector for a second issue this session
  surfaced (this campaign's own orchestrator `state.json` living under the
  session scratchpad, invisible to `campaign_watch.py`'s default scan
  roots, `docs/decisions/0013-...`'s "何が起きたか" addendum): added an
  explicit instruction to `.claude/skills/experiment-launch/SKILL.md` step 3
  to always set `state_path` to an absolute path under the workspace
  registry's `orchestration_root`, and to never copy `env.ARD_RUN_ID`
  verbatim from a prior campaign's resolved manifest. This campaign's own
  live controller cannot be relocated retroactively; `/experiment-postrun`
  for this campaign must pass `--state-path
  /tmp/claude-1001/-home-islab-workspace-local-shunsuke-naito-ard-codex-bootstrap/027b5ec8-8da5-43a8-b31a-028296996632/scratchpad/.orchestration/imagenet-stage01-r18-mobilenetv3-adr-v2.state.json`
  explicitly once `r18_baseline-s0` and the `r18_adr-s0` hand-run finish --
  automatic discovery will not find it.
- 2026-09-14T08:46Z: `r18_baseline-s0` completed (`/experiment-postrun`,
  run-bundle path). Terminal re-derived via `campaign_watch.py --once
  --include-hand-run`: `terminal: true`, `success: true`, 50/50 epoch rows.
  `completion.json` (its only declared `expected_output`) present with
  `identity_hash` `dcf21aece735...` and source SHA `ae4dd7c82d80`, matching
  the attempt3 resolved manifest; `best.pt`, `last.pt` on disk; world size 1,
  global batch 128. With this, the orchestrated campaign state (scratchpad
  `state.json`) is now terminal and classified `failed` (3 completed, 1
  failed). As recorded above, that classification is stale: the `failed` job
  is the W&B-collision attempt of `r18_adr-s0`, whose real result is the
  hand-run, still training (epoch 42/49 at 08:44Z). A headless postrun fired
  by that campaign-level event must not write a failure packet for it.
  **Nothing imported**, for the same reasons as `mobilenetv3_adr-s0`: no
  aggregator for this contract exists (Verification step 5), no AutoAttack
  evaluation has run (step 4), and stage 1 is not complete (`r18_adr-s0`
  hand-run). No record, report, ledger row or milestone tick.
  **Internal validation only (held-out 2% slice, PGD-10; not an official
  result, n=1)**, from the run-bundle summary: best epoch 48, best clean /
  PGD 0.551 / 0.321; last (epoch 49) clean / PGD 0.553 / 0.319. Best and
  last are almost the same (gap 0.2 pp), so no robust overfitting is visible
  on this slice. The ResNet-18 `pgd_at` arm therefore ends well above the
  MobileNetV3-Small `pgd_at` arm (0.459 / 0.253 best) on the same slice.
  The pending `adr`-vs-`pgd_at` comparison for ResNet-18 (does ResNet-18's
  `adr` clean accuracy recover after the epoch-25 decay, unlike
  MobileNetV3-Small's) needs `r18_adr-s0` to finish.
