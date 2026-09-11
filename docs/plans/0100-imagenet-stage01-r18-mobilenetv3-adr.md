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
- Current milestone: design, this document. No code written for this plan
  yet beyond what plan 0099 already built. No source SHA frozen for this
  plan's own protocol identity.

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
mechanism itself. `method.adr` config block: `ema_decay: 0.995,
temperature_high: 2.5, temperature_low: 2.0, lambda_low: 0.7, lambda_high: 0.95`
— identical values to the CIFAR-10 configs, since nothing about the
mechanism's own hyperparameters is dataset-specific in how it was derived
(the paper's own defaults).

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

**Not sourced from a citation, this project's own existing convention,
carried over from the CIFAR configs unchanged**: SGD, Nesterov momentum 0.9,
weight decay 1e-4. **LR schedule**: multistep, scaled proportionally from
this project's own CIFAR 200-epoch schedule (milestones at 100/150, i.e.
50%/75% of the horizon) to this recipe's 50-epoch horizon: milestones at
[25, 38], gamma 0.1. Base LR 0.1, matching both the CIFAR configs and
standard ImageNet SGD convention. **Held-out validation fraction**: 0.02
(≈25,600 images), deliberately smaller than the CIFAR default (0.25) —
at ImageNet scale a 25% held-out slice would materially inflate the
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
