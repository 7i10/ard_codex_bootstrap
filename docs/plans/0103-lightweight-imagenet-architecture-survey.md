# Plan 0103: adversarial training for lightweight (~1-10M param) ImageNet-1k architectures

## Context

Plan 0102 (`docs/plans/0102-clean-preserving-robustness-and-lightweight-ard.md`)
spent Workstream A trying to import external recipes (TRADES, weight-EMA,
Singh/Croce/Hein 2023's own AdamW/heavy-augmentation ImageNet recipe) onto
MobileNetV4-Conv-Small, and Workstream B trying ADR-style self-distillation
on the same architecture. Every one of six full-50-epoch arms underperformed
plain PGD-AT on at least one axis; one (the revisiting_at recipe transplant)
catastrophically collapsed, and a same-recipe ablation ruled out heavy
augmentation as the cause (decisions 0016-0018).

An independent adversarial review of that plan's own stated priority order
for novel tricks ("data augmentation -> architecture -> loss function",
2026-09-23, background subagent, synthesized into plan 0102's Progress log)
delivered the finding that reframes this whole line of work: **at the
epoch-25 LR decay, validation accuracy exceeds training accuracy on both
clean and robust axes** (`train_clean 41.60% < val_clean 48.40%`,
`train_robust 22.35% < val_pgd 27.19%`), and the ADR-vs-PGD-AT comparison's
`robust_overfit_gap` is ~0.18pp. **This project's own plain-SGD PGD-AT
baseline is underfitting, not overfitting**, at this architecture scale.
That single fact explains why every regularizer-shaped mechanism tried in
plan 0102 (heavy augmentation, weight-EMA, TRADES, self-distillation)
underperformed: they all suppress overfitting that isn't there. It also
reframes the productive levers as capacity/architecture, training-attack
strength, schedule length, and initialization quality -- not augmentation
or loss tricks in isolation.

Separately, a fresh literature pass (2026-09-23, avoiding RobustBench per
standing user instruction -- see the auto-memory feedback note) confirmed
plan 0102's own earlier finding: **no paper adversarially trains a
genuinely lightweight (~1-10M parameter) CNN/hybrid on full ImageNet-1k
with rigorous (AutoAttack/AutoPGD) evaluation.** Singh/Croce/Hein 2023's
own smallest validated architecture is ConvNeXt-T/ViT-S at ~28-29M params;
Peng et al.'s "RobArch" (arXiv:2301.03110), despite being framed as a
"fixed parameter budget" study, has its smallest variant (RobArch-S) at
~26M params; Debenedetti et al. 2022's small-ViT ablations (XCiT-N12,
~3M) are on ImageNet-100 (100 classes), not full ImageNet-1k. This gap is
real and this plan's reason to exist.

## Goal

Establish a rigorous, literature-grounded comparison of several prominent,
well-established lightweight (~1-10M parameter) ImageNet-1k architectures'
adversarial-training behavior: standard (clean-only) pretrained accuracy as
a covariate, then post-AT clean accuracy and post-AT robust accuracy
(PGD-10 internal validation, AutoAttack where budget allows), analyzed for
the gap between them. Per the human's own framing (chat, 2026-09-23), this
comparison alone -- done fairly and rigorously, in a literature gap nobody
else has filled -- has real paper value even before any novel
lightweight-specific method is proposed. A novel method (if the underfitting
diagnosis holds across architectures) is a later, explicitly separate phase.

**Not yet attempting ADR/self-distillation or any other novel loss-level
mechanism.** Plan 0102's Workstream B is not resumed here; if the
underfitting signature turns out to be architecture/capacity-dependent
rather than universal, loss-level or augmentation-level tricks may re-enter
for whichever specific architecture shows a genuine overfitting signature
-- decided later, by mechanism, not by a fixed category order (see Phase 2
below).

## Standing corrections carried over from plan 0102 (human, 2026-09-23)

1. Never cite RobustBench as a literature source for this work.
2. Survey both BatchNorm-based and LayerNorm-based (ConvNeXt/ViT-hybrid)
   architectures, not CNNs only. Weigh FLOPs alongside parameter count for
   ViT/attention-hybrid candidates, since attention cost and (for
   reparameterized architectures like RepViT/FastViT) train-vs-inference
   compute can diverge from what parameter count alone suggests.
3. Model selection must come from an actual survey of recent (2023-2026)
   standard-classification and robust-classification literature, not an
   ad hoc pick.
4. The former "augmentation -> architecture -> loss" priority order for
   novel tricks is retired (see the underfitting finding above); the
   revised structure is the phased plan below.

## Phased plan (per the adversarial review's recommendation)

**Phase 0 (cheap, hours not GPU-days).** Firm up the reference baseline
before building on it.
- Arm A (`imagenet_mobilenetv4_pgd_at_no_warmup.yaml`, plan 0101) already
  has: n=2 seed replication (54.57/30.58 vs 54.33/30.59 clean/PGD-10,
  0.24pp/0.01pp spread -- an explicit ImageNet-scale noise floor, tighter
  than the CIFAR reference 0.16-1.88pp) and an n=500 direction-finding
  AutoAttack pass (decision 0014: 22.8% AutoAttack robust accuracy at
  50.94-51.13% clean, internally consistent with ResNet-18's own 22.8% at
  ~3x the parameters and MobileNetV3-Small's 17.4%). Queued, not yet run:
  a larger-sample (2000-5000) AutoAttack pass to move this closer to
  citable (Ferret is currently saturated by an unrelated user's job;
  Hamster's GPUs are occupied by plan 0102's decision-0017/0018 arms as of
  this plan's creation).
- Random-init control: `imagenet_mobilenetv4_pgd_at_random_init.yaml`
  (plan 0102, commit `89baddb`) -- isolates whether pretrained
  (non-robust) init is a net benefit or a contributing risk factor for
  this architecture scale. Queued to launch on the first free Hamster GPU.

**Phase 1 (the likely paper contribution).** The architecture-selection
screen: several candidate architectures, ALL trained with Arm A's own
pinned, working, plain-SGD PGD-AT recipe (optimizer, scheduler, full
attack identity byte-identical to Arm A) -- never the collapsed AdamW
recipe. Only `student.architecture` (and `tracking.group`) differs per
candidate; same protocol id across the whole survey (one scientific
contract, architecture is the studied variable). Report, per candidate:
pretrained clean top-1 (as published/verified via timm, a covariate not a
result of this project), post-AT clean accuracy, post-AT PGD-10 (internal
validation), and AutoAttack where budget allows -- then analyze the
clean-to-robust gap and whether the underfitting signature (val >= train)
holds or breaks per architecture.

**Phase 2 (residual budget only, gated on Phase 1's own mechanism, not a
fixed category).** If the underfitting signature persists across
architectures, the next candidates are capacity/architecture modification
and inner-attack budget -- not augmentation or loss tricks. If any
architecture in Phase 1 shows a genuine overfitting signature instead
(train > val, growing robust_overfit_gap), augmentation/loss tricks
re-enter specifically for that architecture, not as a blanket next step.

## Candidate architecture shortlist (literature survey, 2026-09-23, background subagent; verified locally against timm 1.0.9 where noted)

All verified to construct cleanly at `num_classes=1000` with this
project's existing `imagenet_standard` normalization profile (mean
0.485/0.456/0.406, std 0.229/0.224/0.225, 224x224) -- confirmed via
`timm.create_model(..., pretrained=False).pretrained_cfg` for the three
new candidates below, matching MobileNetV4/ResNet-18's own convention
exactly; no new normalization profile needed.

| Architecture | Params (verified locally) | Norm | Checkpoint | Status |
|---|---:|---|---|---|
| MobileNetV4-Conv-Small | 3.77M | BatchNorm | `mobilenetv4_conv_small.e1200_r224_in1k` | Already integrated (plan 0101); this plan's own reference point |
| ConvNeXt V2-Atto | 3.71M | LayerNorm (+GRN) | `convnextv2_atto.fcmae_ft_in1k` | New this plan -- see checkpoint-provenance note below |
| XCiT-Nano-12/p16 | 3.05M | LayerNorm, linear (cross-covariance) attention | `xcit_nano_12_p16_224.fb_in1k` (non-distilled) | New this plan |
| GhostNetV2 1.0x | 6.16M | BatchNorm, "cheap operation" design (architecturally distinct from depthwise-separable family) | `ghostnetv2_100.in1k` | New this plan |
| RepViT / FastViT (structural reparameterization) | ~1-8M depending on variant | BatchNorm, train-time multi-branch fused to inference-time single-branch | timm + official repos | **Deferred, not this wave** -- see note below |

**Checkpoint-provenance note (ConvNeXt V2-Atto)**: timm only ships an
FCMAE-pretrained-then-ImageNet-1k-finetuned checkpoint for the Atto/Femto
scale (`convnextv2_atto.fcmae_ft_in1k`); there is no pure
supervised-from-scratch ImageNet-1k checkpoint at this scale in timm.
Judged acceptable for this project's use (the checkpoint is used as an
off-the-shelf non-robust classifier init, not as a claim about which
*training recipe* produced the best clean classifier -- matching how
MobileNetV4's own checkpoint choice, `e1200` vs `e2400`, was already
disclosed rather than treated as a confound) -- disclosed here rather than
silently used.

**Why RepViT/FastViT are deferred rather than included in this first
wave**: both use structural reparameterization -- multiple trainable
branches during training, mathematically fused into a single branch at
inference. Whether PGD/AutoAttack should attack the un-fused multi-branch
training-time graph or the fused inference-time graph is genuinely
unclear and, per the literature survey, unaddressed by any paper found.
Including one now would confound the first wave's architecture comparison
with an open methodological question that deserves its own dedicated
follow-up (a real, reportable finding either way) rather than being
absorbed silently into a cross-architecture comparison. Flagged as a
concrete Phase-1-followup candidate, not dropped.

## Concrete changes

- `src/ard/models/registry.py`: three new `build_architecture` branches
  (`convnextv2_atto_imagenet`, `xcit_nano_imagenet`, `ghostnetv2_imagenet`),
  following `mobilenetv4_conv_small_imagenet`'s exact pattern (lazy timm
  import, `timm.create_model(<tag>, pretrained=pretrained, num_classes=N)`,
  no head-replacement branch needed since this project trains at the
  checkpoint's native 1000-class head).
- `src/ard/config/schema.py`: extend `ModelConfig.validate_pretrained`'s
  and `build_architecture`'s `pretrained=True` allow-lists with the three
  new architecture ids.
- `src/ard/config/schema.py`: one new `ProtocolConfig.id` literal,
  `controlled_imagenet_stage02_lightweight_architecture_survey_v1`, shared
  across every candidate in this survey (one scientific contract,
  architecture is the studied variable) -- registered in
  `src/ard/protocols/__init__.py`.
- `configs/scientific/`: `imagenet_convnextv2_atto_pgd_at.yaml`,
  `imagenet_xcit_nano_pgd_at.yaml`, `imagenet_ghostnetv2_pgd_at.yaml` --
  each identical to `imagenet_mobilenetv4_pgd_at_no_warmup.yaml` except
  `student.architecture`, `protocol.id`, and `tracking.group`.
- Tests: one config-identity test per new architecture (mirrors this
  session's own `test_r18_revisiting_at_recipe_config_changes_only_the_architecture`
  pattern), proving only architecture/protocol/tracking.group differ from
  Arm A -- the full attack identity, optimizer, scheduler and epoch count
  stay byte-identical.

## Verification

`scripts/verify.py --changed` after the registry+schema+config batch,
before any GPU-hour is spent, per this project's standing discipline.

## Progress log

(Continued from plan 0102's 2026-09-23 entries; see that plan for the full
history behind this pivot.)

- 2026-09-24 (chat): **identical-condition generalization-gap check**
  (forward-only, no training). Motivation: the "val > train" reading behind
  the underfitting claim compares logged metrics that are not comparable --
  train_* uses random-resized-crop inputs, train-mode BatchNorm, the 3-step
  attack, and is averaged over a moving model. Here every number uses the
  same eval transform (resize + center crop), eval mode, and the same PGD-10
  selection attack (eps 4/255, step 8/765), on N=2000 images the model
  trained on ("seen", training partition) vs N=2000 from the held-out 2%
  slice of the same split ("unseen"). `last.pt`, seed 0, n=1 per model;
  throwaway script from pinned worktree `source-dd6effad5384`, not committed.
  Binomial SE of each gap is about 1.4-1.5pt.

  | model (all plain-SGD PGD-AT, same recipe) | params / MACs | seen clean / PGD-10 | unseen clean / PGD-10 | gap clean / PGD |
  |---|---:|---:|---:|---:|
  | MobileNetV3-Small (plan 0100) | 2.5M / 0.06G | 45.80 / 25.50 | 45.15 / 24.55 | +0.65 / +0.95 |
  | MobileNetV4-Conv-Small (`plan0102-weight-ema-full50-v1` live weights = Arm A recipe) | 3.8M / 0.19G | 58.50 / 32.60 | 54.40 / 30.10 | +4.10 / +2.50 |
  | ResNet-18 (plan 0100) | 11.7M / 1.81G | 57.20 / 33.30 | 53.75 / 30.70 | +3.45 / +2.60 |
  | MobileNetV4, collapsed AdamW recipe (reference) | 3.8M | 27.25 / 0.00 | 27.40 / 0.15 | -0.15 / -0.15 |

  Reading:
  1. **The logged "val > train" was a measurement artifact.** Under
     identical conditions, seen >= unseen for every healthy model -- there
     is a small, real generalization gap (about 2.5pt robust for
     MobileNetV4-S and ResNet-18; within noise for MobileNetV3-S). The
     adversarial review's "pure underfitting" wording is too strong.
  2. **But the regime is still bias-dominated.** Seen-set PGD-10 accuracy is
     only 25-33%: the models cannot fit their own training images
     adversarially. The gap between seen robust accuracy and a perfect fit
     (~67-75pt) dwarfs the seen-unseen gap (~1-2.6pt). A regularizer can at
     most recover the latter; reducing training error has far more headroom.
     This is the quantitative form of "why regularizers didn't help".
  3. **Against the human's hypothesis** (small models underfit, normal-size
     ones overfit): directionally consistent at the small end (MobileNetV3-S,
     0.06G MACs, shows no detectable gap), but MobileNetV4-S and ResNet-18
     have nearly the same gap despite 3x params and ~10x MACs, and both are
     still far from memorizing. Within <=12M on ImageNet at 50 epochs,
     every model tested is bias-dominated; the overfitting regime where
     heavy augmentation pays off (Singh/Croce/Hein, >=28M) is not reached
     here. Three models, n=1 each -- a hint, not a scaling law.
  4. **The collapse is not memorization.** The collapsed run's robust
     accuracy is 0 on seen and unseen alike (no gap). Its catastrophic
     overfitting is an attack/model interaction, not overfitting to the
     training set in the usual sense.

