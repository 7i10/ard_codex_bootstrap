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

- 2026-09-24 (background literature verification, read from paper PDFs and
  the RobustART repo, no RobustBench): **the literature-gap claim in this
  plan's Context section is false as worded and is superseded here.**
  Adversarial training of <12M-parameter ImageNet-1k models exists, but only
  as isolated data points:
  - RobustART (arXiv:2109.05211, App. F.2): ShuffleNetV2-x2.0 and
    MobileNetV3-x1.4 (~7.5M), PGD-l_inf at eps=16/255 (20 steps), 100 epochs
    from scratch; results only as bar charts, no tables. Its "lightweight
    models don't improve with size" claim is about standard-trained models.
  - Salman et al. 2020 (arXiv:2007.08489): MobileNetV2 / ShuffleNet / MNASNet
    at l2 eps=3 from scratch, clean accuracy only; ResNet-18 at l_inf 4/255
    clean 52.49.
  - Smooth Adversarial Training (arXiv:2006.14536): EfficientNet-B0 at
    eps=4/255, PGD-1 training, 100 epochs from scratch: **65.1% clean /
    37.6% PGD-200** (no AutoAttack).
  - EasyRobust model zoo (arXiv:2503.16975): EfficientNet-B0 **61.83% clean /
    35.06% AutoAttack** (eps 4/255 inferred from checkpoint name; recipe not
    verified).
  - Heuillet et al. 2025 (arXiv:2508.14079): robust fine-tuning of pretrained
    5-10M models (RegNetX-004, EfficientNet-B0, EdgeNeXt-S, DeiT-Tiny,
    CoaT-Tiny, MobileViT-S) at eps=4/255 with AutoAttack -- but on six small
    downstream datasets, not ImageNet. Must be cited and distinguished.
  Re-scoped claim: no work *systematically* studies **modern** lightweight
  architectures (MobileNetV4, ConvNeXt-Atto, MobileViT, DeiT-Ti) on ImageNet-1k
  under l_inf 4/255 with AutoAttack, using **fine-tuning from clean pretrained
  checkpoints**, with a controlled comparison and an analysis of the
  low-capacity regime (capacity, stem, activation, training method incl.
  distillation). Anchor to beat or explain: EfficientNet-B0 at ~35% AutoAttack
  vs this project's MobileNetV4-S 22.8% (n=500) -- the gap is itself
  informative (smooth activation + SE + 2x MACs, and/or our 50-epoch budget).

  Also verified: Singh/Croce/Hein give **no mechanistic explanation** for
  ConvStem (they call it an open question and point to Xiao et al.'s
  trainability argument). Their ConvStem l_inf gain is large for ViT-S
  (+4.1 to +5.2pt AutoAttack) but marginal for ConvNeXt-T (+0.9); the big
  ConvNeXt effect is on unseen l1/l2 threats. Their Table 1 also shows
  pretrained vs random init at ViT-S scale: 39.1 vs 31.8 l_inf AutoAttack
  (+7.3pt) -- a prior for this plan's own pretrained-vs-random-init test.
  CIFAR anchor for distillation: RSLAD gives MobileNetV2 +3.9pt AutoAttack
  over plain AT (arXiv:2108.07969, Table 3), though AdaAD's reproduction of
  RSLAD lands ~3.4pt lower -- a reproducibility warning.
- 2026-09-24 (chat): **decision 0018 option F verdict, epoch 25**
  (`plan0102-revisiting-at-recipe-no-weight-decay-full50-v1`, internal
  validation only, n=1): val clean 46.39% / val PGD 15.63% vs the epoch-9
  reference 15.45% (rule threshold 7.73%; minimum since epoch 9 14.18%);
  `train_robust_overtakes_clean` False through epoch 25;
  `train_robust_accuracy_eval_mode` 12.76% < `train_clean_accuracy` 36.10%.
  Both preregistered conditions hold, so per decision 0018's rule
  **weight_decay 0.05 was a necessary condition for the revisiting_at
  recipe's collapse** on MobileNetV4-S (n=1). Not a healthy run either --
  val PGD drifted from ~17.6% (epoch 17) to ~15.5% and the train/eval-mode
  robust gap widened from 6.0 to 9.5pt. Ended via SIGTERM at epoch 25 per the
  human's prior approval; checkpoints and epoch metrics preserved.
- 2026-09-24 (chat): **Phase 0 budget x init 2x2, first new cell launched**:
  `imagenet_mobilenetv4_pgd_at_pretrained_100ep.yaml` (new protocol
  `controlled_imagenet_stage02_budget_init_v1`): Arm A with epochs 50 -> 100
  and LR milestones scaled [25, 38] -> [50, 76]; 10-epoch warmup kept in
  absolute epochs. Existing Arm A fills the pretrained/50 cell (n=2). This
  cell is common to every variant of the 2x2 under discussion (whether or
  not a random-init/50 cell is also run), so it was launched on the freed
  Hamster GPU0 without waiting for that choice. A 100-epoch run's epoch-49
  checkpoint is *not* a substitute for a 50-epoch run (different LR schedule).
- 2026-09-24 (chat): human stopped option E (answered: no collapse through
  epoch 28; still behind the SGD R18 run at matched epochs, but E differs from
  it in six recipe elements -- optimizer, LR size/shape, weight decay, label
  smoothing, heavy augmentation, weight-EMA -- plus LR-schedule phase, so the
  gap is not attributable to augmentation alone). The first
  pretrained/100-epoch launch (`plan0103-mobilenetv4-pretrained-100ep-v1`,
  single-teacher-ard W&B project, epoch 0 done) was also stopped so both
  100-epoch cells start together from the same code with the train probe and
  in the new W&B project (`configs/tracking/lightweight.env`: production
  `lightweight-imagenet-at`, exploration `lightweight-imagenet-at-dev`).
  EasyRobust EfficientNet-B0 (61.83 / 35.06 AutoAttack): README gives only the
  numbers and "Adversarial Training (Madry)"; its AT example script defaults
  are from scratch, 90 epochs, SGD step decay, PGD-3 eps 4/255 step 2/3 eps
  (no random start); the checkpoint is a bare state_dict with no args, so the
  actual recipe is unverifiable. (The "100 epochs from scratch" figure is
  SAT's EfficientNet-B0, not EasyRobust's.) Plan: re-evaluate that checkpoint
  under this project's own protocol in Phase 1b.
- 2026-09-24 (chat): **train probe** (`training.train_probe_size`, per-epoch
  eval-mode clean/PGD-10 on fixed class-stratified training-partition images
  through the validation transform; `train_probe_*` columns) added, plus
  scientific review of it and of decision 0016's eval-mode robust metric
  (which had been used in options B/E/F *without* the review decision 0016
  itself required -- disclosed). Review verdict: neither change alters
  training. Addressed before launch:
  - P1 (probe added to a running cell): moot -- that run had already been
    stopped; both cells restart from one SHA.
  - P2-2 tests: added a BatchNorm+Dropout bit-identity test (Dropout uses the
    global RNG, BN keeps running stats) and a BN `num_batches_tracked == steps`
    check (no extra train-mode forward, covering change 1 too); the CLI wiring
    moved into a tested helper (`build_train_probe_view`: training IDs only,
    disjoint from validation IDs, same view object as validation).
  - P2-3: `train_robust_accuracy_eval_mode` is post-step on perturbations
    crafted against pre-step weights, so it is biased upward; its gap to
    `train_robust_accuracy` is only a *lower bound* on the pure BatchNorm-mode
    gap. Documented in code. Decision 0018's verdict (12.8 vs 36.1) cannot
    flip. The probe is now the clean seen-set measurement.
  - P2-4: `train_robust_overtakes_clean` compares train-mode pre-step robust
    with eval-mode post-step clean, so it marks the BN-mode divergence, not
    catastrophic overfitting as such; comment corrected, definition kept for
    series continuity.
  - P2-5: parquet writer took its schema from the first row and would drop
    columns first appearing on a resumed run; now writes the union of keys
    (test added). No run in this plan was resumed across that boundary.
  - P2-6: probe sample stratified per class (2000 -> 2 per class).
  - P2-7: probe wrapped in the repo's `capture_rng_state`/`restore_rng_state`
    (all streams) instead of `torch.random.fork_rng`.
  - Config-hash note: the new TrainingConfig field changes every resolved
    config hash; any earlier run must be resumed only from its own worktree.
- 2026-09-24 (chat): **Phase 0 2x2, both 100-epoch cells launched together**
  from pinned worktree `source-b0cfc9e4e2d2` (SHA `b0cfc9e`), seed 0, W&B
  project `lightweight-imagenet-at`, train probe 2000 (2 per class):
  `plan0103-mobilenetv4-pretrained-100ep-v2` (Hamster GPU0) and
  `plan0103-mobilenetv4-random-init-100ep-v1` (Hamster GPU1). Arm A
  (pretrained, 50 epochs, n=2) fills the third cell; the random-init/50 cell
  is deferred until these two land (human, 2026-09-24).
- 2026-09-24 (chat): **final architecture set (human-approved, supersedes the
  earlier shortlist table)**. Replaced the never-launched ConvNeXt V2-Atto,
  XCiT-Nano and GhostNetV2 entries (V2's GRN + FCMAE pretraining are two
  confounds; XCiT-Nano's supervised checkpoint is only 70.0%; GhostNetV2 is
  not prominent enough). All measured locally at 224px (params / GMACs /
  published top-1):

  | family | model | params / GMACs | top-1 | checkpoint | normalization |
  |---|---|---:|---:|---|---|
  | BN CNN | MobileNetV4-Conv-Small (Arm A) | 3.77M / 0.19 | 73.5 | e1200_r224_in1k | imagenet_standard |
  | BN CNN (+SE, SiLU) | EfficientNet-B0 | 5.29M / 0.39 | 77.7 | ra_in1k | imagenet_standard |
  | BN CNN, larger | MobileNetV4-Conv-Medium | 9.72M / 0.83 | 79.1 | e500_r224_in1k | imagenet_standard |
  | LN CNN (patchify) | ConvNeXt-Atto (V1) | 3.70M / 0.55 | 75.7 | d2_in1k | imagenet_standard |
  | ViT (LN) | DeiT-Tiny | 5.72M / 1.07 | 72.2 | fb_in1k (non-distilled) | imagenet_standard |
  | hybrid (BN conv + LN attn) | MobileViT-S | 5.58M / 1.42 | 78.3 @256 | cvnets_in1k | imagenet_raw_identity |

  EfficientNet-B0 uses `ra_in1k` rather than the stronger `ra4_e3600` (78.6%,
  mean/std 0.5) to keep the standard profile. MobileViT-S's checkpoint was
  trained at 256px on raw [0,1] pixels: added a pinned `imagenet_raw_identity`
  profile (schema validator requires it for mobilevit_s_imagenet and forbids
  it elsewhere, with a regression test), and use 224px like every other model
  (uniform input size); the resulting clean-accuracy cost, plus the bicubic
  (timm) vs bilinear (this pipeline) resize difference, is measured in Phase
  1b before any training. MobileNetV3-Small and ResNet-18 stay as existing
  single-seed reference points only. Configs:
  `imagenet_{efficientnet_b0,mobilenetv4_conv_medium,convnext_atto,deit_tiny,mobilevit_s}_pgd_at.yaml`.
- 2026-09-24 (chat): **Phase 1b, pretrained clean top-1 on this project's own
  ImageNet-val pipeline** (full 50k val, 224px, resize 256 + center crop,
  bilinear; forward-only, pinned worktree `source-23f6bf0a3a79`, throwaway
  script not committed):

  | model | ours @224 | published | diff |
  |---|---:|---:|---:|
  | MobileNetV4-Conv-Small | 73.41 | 73.45 | -0.04 |
  | EfficientNet-B0 (ra_in1k) | 77.67 | 77.70 | -0.03 |
  | MobileNetV4-Conv-Medium | 79.09 | 79.09 | -0.00 |
  | ConvNeXt-Atto V1 | 75.62 | 75.67 | -0.05 |
  | DeiT-Tiny | 72.02 | 72.19 | -0.17 |
  | MobileViT-S | 77.03 | 78.30 (@256) | -1.27 |

  Every checkpoint reproduces its published number within 0.2pt (bilinear vs
  bicubic resize does not matter); MobileViT-S pays 1.3pt for running at 224
  instead of its native 256 -- accepted for a uniform input size and recorded
  as a covariate. EasyRobust's adversarially trained EfficientNet-B0: clean
  61.08% on a 10k val subset under imagenet_standard (published 61.83; subset
  SE ~0.5pt), 32.4% under mean/std 0.5, 1.3% under raw pixels -- so it was
  trained with standard normalization and can be re-evaluated (PGD-10 /
  AutoAttack) under this project's protocol as an external anchor.

- 2026-09-24 (chat): **Compute hosts.** Ferret is fully occupied by another
  user's jobs (13 EEG processes, all 3 GPUs at 99%). Crocodile (1x 2080 Ti,
  driver 520 = CUDA 11.8 max) is dropped: a driver upgrade needs lab consent and
  a reboot, and another student actively uses it. **Anteater** (4x RTX 2080 Ti
  11GB, driver 550; we may use GPUs 2,3 only) is set up with the same
  `~/workspace-local/...` layout as Hamster/Ferret. Its lab-shared ImageNet
  copy reproduces both manifest hashes exactly (train `ae033613...`, val
  `abc0ee80...`). Its env mirrors Hamster's pip freeze (torch 2.11+cu128). Its
  repo is a bundle clone of master at `1285358` (GitHub is 34 commits behind,
  not pushed), with a pinned worktree `source-1285358c1b73`. A `gpu-guard`
  wrapper refuses any `CUDA_VISIBLE_DEVICES` outside {2,3} and sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- 2026-09-24 (chat): **2080 Ti step throughput** (throwaway microbenchmark:
  one PGD-AT step = 3-step PGD + CE + SGD, batch 128, 224px, fp32, synthetic
  pixels, no data loading; an upper bound on training img/s):
  MobileNetV4-S 665 img/s (5.3 GiB), MobileNetV4-M 227 (6.8 GiB), ConvNeXt-Atto
  283 (5.4 GiB), DeiT-Tiny 210 (4.1 GiB). **EfficientNet-B0 and MobileViT-S run
  out of memory at batch 128 on 11 GB.** Batch size is part of the recipe (and
  BN statistics), so these two run on 4090s only. Precision caveat: the
  trainer leaves PyTorch defaults, so convolutions use TF32 on the 4090s but
  full fp32 on the 2080 Ti. Keep each comparison set on one GPU type, or
  record the GPU type as a covariate.
- 2026-09-24 (chat): **Step 1c canaries on Anteater** (3 epochs, seed 0, dev
  W&B project, pinned worktree `source-1285358c1b73`, hand-run bundle):
  `plan0103-canary-convnext-atto-anteater-v1` (GPU2) and
  `plan0103-canary-deit-tiny-anteater-v1` (GPU3).
- 2026-09-24 (chat): **Both Anteater canaries stopped after a few minutes.
  Anteater cannot train on ImageNet.** Its ImageNet copy sits on a 5400 rpm
  HDD (`sda`, WD Red 2 TB). Random JPEG reads measured 121 reads/s at 4 MB/s,
  about 40 images/s for both jobs together, against the 200-280 img/s each
  job needs. The 144 GB train set cannot fit in the 92 GB page cache. The only
  SSD is the root disk, which is full. The 50k-image val set (~6.4 GB) and the
  2000-image probe do fit in cache, so **Anteater is an evaluation host**
  (full-val clean/PGD-10 and AutoAttack from saved checkpoints), not a
  training host. The two canary bundles are incomplete by design; they are
  not results.
- 2026-09-24 (chat): **Co-location verdict: two jobs on one 4090 lose total
  throughput. Stopped.** `plan0103-mobilenetv4-pretrained-100ep-v2`'s epoch 3
  was the first epoch fully shared with the EfficientNet-B0 canary on Hamster
  GPU0 and nothing else (the Crocodile rsync had already been stopped). It ran
  at 376 img/s, against 857 alone (epoch 0); the co-located rows are epoch 1:
  791 (partial overlap) and epoch 2: 494 (co-location plus the rsync). The canary
  `plan0103-canary-efficientnet-b0-colocated-v1` had not finished epoch 0 after
  2 h 13 min, i.e. under 157 img/s. So the pair made fewer than 533 img/s
  together, less than one job alone. The canary was stopped; its bundle is
  incomplete and is not a result. One job per GPU from here on. EfficientNet-B0
  still needs its Step 1c canary, on a free 4090.
- 2026-09-24 (chat): **Anteater SSD.** The lab freed ~200 GB on Anteater's
  root SATA SSD (Micron 5200 960 GB). Steps: removed an unused miniforge3,
  cleared pip/uv/wandb caches, ran `conda clean`, and moved `~/.cache` to the
  HDD behind a symlink. ImageNet (train+val, 144 GB) is being copied from the
  HDD copy to `/home/shunsukenaito/datasets-ssd/imagenet`, with a manifest-hash
  check after the copy. Both splits go on the SSD because the dataset adapter
  rejects paths that resolve outside its root. About 56 GB stays free.
- 2026-09-24 (chat): **Anteater now trains from its SSD.** HDD-to-SSD copy
  on Anteater ran at ~6.6 MB/s (seek-bound), so it was stopped. ImageNet was
  rsynced from Hamster's NVMe instead: 150 GB in 21 min at ~112 MB/s,
  `--size-only` on top of the partial copy. Manifest hashes match on both
  splits (train `ae033613...`, val `abc0ee80...`). `~/workspace-local/datasets/imagenet`
  now points at `/home/shunsukenaito/datasets-ssd/imagenet`; 55 GB of the root
  SSD stays free. Hamster's pretrained-100 run was back at 843-845 img/s
  (epochs 7-8) once the co-located canary was stopped.
- 2026-09-24 (chat): **Step 1c canaries, attempt 2 on Anteater** (3 epochs,
  seed 0, dev W&B project, pinned worktree `source-1285358c1b73`, 7 loader
  workers each): `plan0103-canary-convnext-atto-anteater-v2` (GPU2) and
  `plan0103-canary-deit-tiny-anteater-v2` (GPU3). Both GPUs sit at 93-97%
  util with iowait under 1%. Measured training speed, estimated from bytes
  read by the loaders over the first ~1.8 h, is about 120 img/s for
  ConvNeXt-Atto and 157 for DeiT-Tiny, far below the step microbenchmark's 283
  and 210. The microbenchmark let cuDNN pick fast non-deterministic kernels
  (`cudnn.benchmark=True`). The real recipe has `training.deterministic: true`
  (`torch.use_deterministic_algorithms(True)` in `ard.cli.train`), so it
  overstated throughput. The 4090 numbers (e.g. 857 img/s for MobileNetV4-S)
  come from the same deterministic setting and remain the right reference.
  Implied cost on a 2080 Ti: ~2.9 h/epoch for ConvNeXt-Atto, ~2.2 h/epoch for
  DeiT-Tiny.
- 2026-09-24 (chat): **Step 1c canary: DeiT-Tiny completed** on Anteater GPU3.
  The run ended with `completion.json` status `completed`; its
  `error-marker.txt` reads "No application error recorded". Its
  `environment.json` records RTX 2080 Ti, torch 2.11.0+cu128, cuDNN 91900. The
  run is 3 epochs, all inside the 10-epoch LR warmup, so these are pipeline and
  stability checks, not results:

  | epoch | s / img/s | val clean / PGD-10 | probe clean / PGD-10 | train clean / robust |
  |---|---|---|---|---|
  | 0 | 8332 / 151 | 49.44 / 23.09 | 50.75 / 23.30 | 42.62 / 16.27 |
  | 1 | 8346 / 150 | 48.17 / 23.71 | 50.10 / 25.75 | 44.56 / 18.01 |
  | 2 | 8364 / 150 | 46.63 / 22.82 | 49.15 / 23.40 | 44.44 / 18.12 |

  No collapse, and the probe is logged. Seen and unseen accuracy are close
  (probe 1-3 pt above val), as expected this early. Peak memory was 4.5 GB.
  GPU3 then started `plan0103-canary-mobilenetv4-medium-anteater-v1` (same
  worktree, 3 epochs, dev project). ConvNeXt-Atto (GPU2) finished epochs 0-1:
  115 img/s, val 53.28/26.64 then 50.76/26.55, probe 54.70/27.05 then
  52.95/27.90.
- 2026-09-24 (human, chat): **Host policy decided.** The Phase 1 survey runs
  entirely on RTX 4090s, so GPU type (TF32 on Ada vs full fp32 on Turing) is
  not mixed into the architecture comparison. Anteater's 2080 Tis take
  self-contained work only: saved-checkpoint evaluation, extra seeds, and
  comparisons run wholly on that GPU type. **Approved:** re-evaluate
  EasyRobust's adversarially trained EfficientNet-B0 on Anteater under this
  project's protocol (full-val clean/PGD-10, AutoAttack n=500 direction
  setting) as an external anchor. How to stage an external checkpoint for
  `ard.cli.evaluate` is being worked out, and no guard will be weakened to do it.
- 2026-09-24 (chat): **Step 1c canary: ConvNeXt-Atto completed** on Anteater
  GPU2 (`completion.json` completed). Epochs 0-2, all inside the LR warmup:
  10883/10906/10874 s at 115 img/s. val clean/PGD-10 53.28/26.64,
  50.76/26.55, 48.65/26.67. Probe 54.70/27.05, 52.95/27.90, 50.85/27.80. No
  collapse, and the probe is logged. Both Anteater canaries show clean
  accuracy dipping a little each epoch while PGD-10 holds, as expected when
  the LR warms up from a clean-pretrained init.
- 2026-09-24 (chat): **Anteater's ImageNet copy is byte-identical to
  Hamster's.** `rsync -rnc` found 0 differing files in train and val (full
  checksums, not just the size manifest).
- 2026-09-24 (chat): **External anchor code** (commit `f201ac2`).
  `scripts/evaluate_external_checkpoint.py --dataset imagenet` measures
  foreign weights with every setting taken from our own scientific config,
  through the same loop and loader as `ard.cli.evaluate`. It writes a
  FOREIGN-lineage record with git/GPU/TF32 provenance. Scientific review found
  no P1, and all P2/P3 findings were fixed in the same commit. The full
  `verify --changed` set passes except the known pre-existing
  `test_prescriptive_v3::test_v3_generates_four_strict_arms_and_atomically_forks_exact_epoch79_state`.
  **EfficientNet-B0 evaluation does not fit an 11 GB 2080 Ti at batch 128**:
  one PGD-10 batch in eval mode ran out of memory at ~10 GiB. Evaluation PGD
  random starts are drawn per batch, so a smaller batch would change the
  measurement. The EasyRobust anchor therefore waits for a free 4090 (after the
  2x2 cells). Checkpoint on Anteater: `advtrain_efficientnet_b0_ep4.pth`,
  sha256 `d90ccd52...` (from EasyRobust's adversarial-training model zoo).
- 2026-09-24 (chat): **Where the time goes, on a 2080 Ti** (throwaway probe,
  one trainer-shaped PGD-AT step: eval-mode 3-step PGD then a train-mode CE+SGD
  step, batch 128, 224 px, fp32, synthetic pixels; img/s):

  | model | det (current recipe) | non-det | non-det + cudnn.benchmark | det + channels_last | det + torch.compile |
  |---|---:|---:|---:|---:|---:|
  | MobileNetV4-S | 514 | 654 | 657 | 433 | 356 |
  | ConvNeXt-Atto | 141 | 196 | 282 | 141 | 69 |
  | DeiT-Tiny | 191 | 209 | 209 | 188 | 176 |

  `training.deterministic: true` costs 1.1-2.0x, and all of that comes from
  the deterministic-algorithm restriction and the lack of cuDNN autotuning.
  On Turing fp32, channels_last and torch.compile are neutral or harmful. Real
  canary throughput (ConvNeXt 115, DeiT 150) is ~20% below the `det` column
  because of data loading and per-epoch validation/probe. **This is not yet
  evidence for the 4090s:** Ada has TF32 tensor cores and is fast enough for
  launch overhead to dominate, so channels_last and compile may behave
  differently there. Next: repeat this probe on a Hamster 4090 once the 2x2
  cells end, before implementing compile/channels_last flags and before
  proposing any change to `deterministic`. That change is a human decision;
  it would apply uniformly to every survey run, and Arm A and the 2x2 cells
  stay deterministic.
- 2026-09-25 (postrun): **Phase 0 2x2: both 100-epoch cells completed.**
  `plan0103-mobilenetv4-pretrained-100ep-v2` (Hamster GPU0) and
  `plan0103-mobilenetv4-random-init-100ep-v1` (Hamster GPU1) both ran 100/100
  epochs. Each has `completion.json` `completed`, manifest status `completed`,
  and `error-marker.txt` "no application error recorded". The watcher
  re-derived both as terminal and successful. Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract, as for every
  other hand-run in plans 0102/0103.

  Fixed identity for every number below: ImageNet-1k,
  MobileNetV4-Conv-Small, plain PGD-AT (SGD lr 0.05, Nesterov, wd 1e-4,
  10-epoch LR warmup, x0.1 at epochs 50 and 76), no teacher, training and
  evaluation seed 0. Training attack: l_inf 4/255, 3 steps, step 8/765, random
  start. Selection attack: PGD-10, same eps and step, random start, eval mode.
  World size 1, global batch 128, RTX 4090 (TF32), deterministic. Source SHA
  `b0cfc9e` (worktree `source-b0cfc9e4e2d2`), protocol
  `controlled_imagenet_stage02_budget_init_v1`. "val" is **internal
  validation**: the held-out 2% slice of the training split (25,620 images),
  not the official ImageNet val set. "probe" is 2,000 training images (2 per
  class) under the same transform, eval mode and PGD-10. No AutoAttack has run.

  | cell | run | last (ep 99) val clean / PGD-10 | best (by val PGD) | last probe clean / PGD-10 |
  |---|---|---:|---:|---:|
  | pretrained, 100 ep | `plan0103-mobilenetv4-pretrained-100ep-v2` | 55.21 / 30.70 | ep 91: 54.88 / 31.03 | 59.55 / 34.90 |
  | random init, 100 ep | `plan0103-mobilenetv4-random-init-100ep-v1` | 54.56 / 30.58 | ep 97: 54.44 / 30.68 | 60.75 / 35.00 |
  | pretrained, 50 ep (Arm A, plan 0101, seed 0) | | 54.57 / 30.58 (ep 49) | ep 46: 54.57 / 30.74 | not logged |
  | pretrained, 50 ep (Arm A, plan 0101, seed 1) | | 54.33 / 30.59 (ep 49) | | not logged |
  | random init, 50 ep | not run (deferred, human 2026-09-24) | | | |

  Trajectory (val clean / PGD-10), pretrained vs random init: epoch 0
  46.81/19.20 vs 1.36/0.98; epoch 24 38.38/19.58 vs 37.12/19.51; epoch 49
  (end of the lr-0.05 phase) 37.85/20.13 vs 38.02/20.19; epoch 75 (end of the
  lr-0.005 phase) 50.45/27.74 vs 50.09/27.81.

  Reading (n=1 per cell, internal validation, PGD-10 only):
  1. **The pretrained head start is gone by the end of the high-LR phase.**
     Pretrained starts 45pt clean ahead, but the lr-0.05 phase pulls it down
     to where random init climbs to. At every checked epoch from 24 on (24,
     49, 50, 75, 76, 99), the two runs are within 1.3pt clean and 0.3pt
     PGD-10. At the end, pretrained leads by 0.65pt clean
     and 0.12pt PGD-10 (last), or 0.44 / 0.35pt (best).
  2. **Doubling the epochs barely moves robustness.** Pretrained 100 vs 50
     epochs (seed 0 vs seed 0): +0.64pt clean, +0.12pt PGD-10 (last). Random
     init at 100 epochs ties pretrained at 50 epochs (54.56/30.58 vs
     54.57/30.58).
  3. **The 40 epochs at lr 0.05 are a plateau.** Pretrained val clean stays in
     37.4-39.1% and PGD-10 in 18.9-20.5% from epoch 9 to 49. Most of the
     gain arrives right at the two LR decays (epoch 49 to 50: +6.7pt PGD-10;
     epoch 75 to 76: +2.6pt).
  4. **Seen-unseen gap at the end** (probe minus val, same conditions):
     pretrained +4.3 clean / +4.2 PGD-10, random init +6.2 / +4.4. The probe's
     binomial SE is about 1.1pt. Seen-set PGD-10 accuracy is still only ~35%, so
     the regime stays bias-dominated at 100 epochs.

  Caveats: one seed per 100-epoch cell, so no noise floor for this pair. Arm
  A's two seeds differ by 0.24pt clean / 0.01pt PGD-10, but two runs do not
  estimate a spread; the CIFAR control-vs-control floor was 0.16-1.88pt.
  Differences of 0.1-0.4pt PGD-10 here are inside any credible floor. Arm A
  ran from a different source SHA (plan 0101); the train-probe change in
  between was tested to leave training bit-identical. Pretrained epochs 1-4
  shared GPU0 with a canary (throughput only; the run is deterministic).
  Checkpoints (sha256): pretrained `best.pt` `d05be221...`, `last.pt`
  `8f9a0401...`; random init `best.pt` `a26f426f...`, `last.pt` `be64fcaa...`.
  W&B project `lightweight-imagenet-at`, runs under the same ids. Next step:
  decision packet 0019.
- 2026-09-25 (chat): **Step 1c canary: MobileNetV4-Conv-Medium completed**
  on Anteater GPU3 (`completion.json` completed). Epochs 0-2 inside the LR
  warmup: 8637/8583/8586 s at 145-146 img/s, peak 7.45 GB. val clean/PGD-10
  54.66/25.93, 54.25/28.16, 54.24/28.67. Probe 57.30/27.25, 56.35/30.50,
  57.65/32.05. No collapse. Its PGD-10 after 3 warmup epochs (28.7) is already
  close to MobileNetV4-S's end-of-training value (30.7, 100 epochs), consistent
  with capacity-limited underfitting at the small end. This is a canary, not a
  result. Step 1c is now done for ConvNeXt-Atto, DeiT-Tiny and
  MobileNetV4-M; EfficientNet-B0 and MobileViT-S need a 4090 (they do not fit
  11 GB at batch 128).
- 2026-09-25 (human, chat): **Decision 0019: B plus an FT-schedule plan.**
  (1) The fourth 2x2 cell, random init / 50 epochs
  (`imagenet_mobilenetv4_pgd_at_random_init_50ep.yaml`), is launched on Hamster
  GPU0 from pinned worktree `source-987dfb534512` as
  `plan0103-mobilenetv4-random-init-50ep-v1` (seed 0, W&B
  `lightweight-imagenet-at`). The human's stated reason is to learn whether
  pretraining is needed at all; not needing it would be preferred. The
  preregistered rule is unchanged (decision 0019 B). (2) A fine-tuning-shaped
  schedule (Arm A with peak lr 0.005) runs on GPU1 under its own
  contract, plan 0104 (`docs/plans/0104-finetune-lr-schedule.md`).
- 2026-09-25 (chat): **Budget-limited or capacity-limited?** Each LR stage of
  both 100-epoch runs saturates within ~12 epochs. Over the second half of
  each stage, adversarial train loss moves by at most 0.4% relative, and
  seen-set (probe) and val PGD-10 stay flat (lr 0.0005, ep 88-99, pretrained:
  loss 3.659 -> 3.643, probe PGD 35.1 -> 34.9, val PGD 30.81 -> 30.70).
  Doubling the time at every LR (50 -> 100 epochs) moved nothing beyond noise,
  and the model still fits only ~35% of its own training images under PGD-10.
  So under this recipe the limit is what the model can fit (capacity, or
  capacity under this regularization and objective), not epochs. Two caveats
  remain. First, duration is entangled with the LR schedule: a lower final
  LR or a different shape could still reduce training error, which is an
  optimization question; plan 0104's run happens to add a 5e-5 stage.
  Second, this holds for MobileNetV4-S only. Phase 1 must check per
  architecture with the same stage-saturation diagnostic. Supporting
  evidence: the MobileNetV4-M canary reaches probe PGD-10 32% within 3
  warmup epochs, against 21% for MobileNetV4-S at the end of its lr-0.05
  phase.
- 2026-09-25 (chat): **Throughput on a Hamster 4090** (the same throwaway
  trainer-shaped step probe, run on GPU1 while it waited for plan 0104; img/s,
  and in brackets the ratio to the current deterministic recipe; all 42
  settings completed without error):

  | model | det | non-det | +cudnn.benchmark | det+channels_last | det+compile | non-det+bench+cl | non-det+bench+compile |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | MobileNetV4-S | 1358 | 1733 (1.28) | 1739 (1.28) | 1211 (0.89) | 1235 (0.91) | 1457 (1.07) | 1504 (1.11) |
  | EfficientNet-B0 | 320 | 383 (1.19) | 383 (1.19) | 308 (0.96) | 434 (1.35) | 361 (1.13) | 460 (1.44) |
  | MobileNetV4-M | 491 | 588 (1.20) | 593 (1.21) | 469 (0.96) | 527 (1.07) | 558 (1.14) | 572 (1.17) |
  | ConvNeXt-Atto | 698 | 718 (1.03) | 869 (1.25) | 710 (1.02) | 385 (0.55) | 884 (1.27) | 602 (0.86) |
  | DeiT-Tiny | 610 | 630 (1.03) | 631 (1.03) | 613 (1.01) | 572 (0.94) | 635 (1.04) | 579 (0.95) |
  | MobileViT-S | 189 | 213 (1.13) | 213 (1.13) | 177 (0.94) | 219 (1.16) | 204 (1.08) | 235 (1.24) |

  Non-deterministic mode plus cudnn.benchmark gives 1.03-1.28x.
  channels_last never helps. torch.compile helps only EfficientNet-B0 and
  MobileViT-S, and halves ConvNeXt-Atto. A `TORCH_LOGS=recompiles,graph_breaks`
  check on Anteater gave two findings:
  1. Only two expected recompiles occur, one per input `requires_grad`
     variant (attack vs train step).
  2. `PixelNormalization.forward`'s [0,1] range check
     (`if images.amin() < ... or images.amax() > ...`) is a data-dependent
     branch. It breaks the compiled graph at the model entry and forces a
     GPU->CPU sync on every forward, in eager mode too. It is a candidate for
     an equally strict but sync-free form (e.g. an async assert). That change
     needs its own review because it touches a guard.
- 2026-09-25 (human, chat): **Standing design principles for the pretraining
  question.**
  (1) If pretraining turns out unnecessary, Phase 1's main line is random
  init.
  (2) To state that *other* models do not need pretraining, every model gets
  both inits, not only the suspected ViT.
  (3) If random init is enough, whether it needs 50 or 100 epochs must also be
  tested.
  (4) No experiment is designed to steer toward the preferred answer
  ("pretraining unnecessary"). Judge fairly.
  (5) A pretraining comparison is only fair when each init uses a recipe
  suited to it: fine-tuning hyperparameters for pretrained, from-scratch
  hyperparameters for random init. The same Arm A recipe for both is not
  enough.
  Approved: `training.deterministic: false` plus cuDNN autotuning for Phase 1,
  a sync-free form of the pixel-range guard (same strictness), and per-model
  torch.compile where it helps. Implementation is in progress, with a
  scientific review before use. The robust-teacher choice for Phase 2
  distillation is delegated to the assistant (no ImageNet counterpart of the
  CIFAR "ERT" teacher exists).
- 2026-09-26 (chat): **Throughput options merged** (`c4af44c`, `2e0e9dd`). All
  are off by default and behavior-preserving; scientific review found no P1,
  and every P2/P3 finding is fixed.
  - Options: `training.cudnn_benchmark` (requires `deterministic: false`) and
    `training.compile` (requires `deterministic: false`, no DDP, and
    inductor's assert-preserving debug flag; recompile-limit fallback to
    eager is turned into an error).
  - The pixel guard is now sync-free: a `torch._assert_async` with the same
    condition and tolerance.
  - The guard was verified to fire on an RTX 4090 in eager and in inductor,
    in forward, backward and input-grad graphs. On CUDA it is a fatal
    device-side assert that keeps its message.
  - Default `training_protocol_identity` stays byte-identical; enabled runs
    get a distinct identity and are never pooled with eager rows.
  - Every `config_hash` changes, so a run can only resume from its own
    pinned worktree (as with earlier field additions).
- 2026-09-26 (human, chat): **Phase 1 scope approved: every architecture with both
  inits** (pretrained and random), one seed each, and a second seed only for the
  2-3 finalists. AutoAttack runs at n=500 (direction setting). Estimated cost
  is ~440 GPU-h, about 9 days on two 4090s. Each init's recipe (learning rate)
  comes from the MobileNetV4-S per-init LR comparison, which is still awaiting
  approval.
- 2026-09-26 (chat): **Speed-up review against a generic list the human
  supplied** (data loader, BF16, channels_last, compile, larger batch,
  progressive resizing, eval frequency, cached distillation, subset tuning,
  profiling).
  - Measured: Hamster's CPU-side train loader (exact training loader, while
    both 4090 jobs ran) gives 1715 / 2293 / 2693 img/s at 8 / 12 / 16
    workers. That is not the limit today (real training 857 img/s,
    deterministic). It would be at 8 workers once non-deterministic
    MobileNetV4-S reaches ~1740 img/s, so Phase 1 uses 16 workers.
    `EpochImageNetTransform` keys crop/flip on (seed, epoch, source id),
    so the worker count cannot change the data.
  - The training step also runs two eval-mode diagnostic forwards
    (train clean acc, eval-mode robust acc; ~+17% compute) and several
    per-step GPU->CPU syncs (`float()` metric accumulation,
    `if not isfinite(loss)`).
  - Adopting now as engineering-only changes: sync-free step accounting,
    a sync-free non-finite-loss check of equal strictness, and pinned-memory
    non-blocking copies, with bit-identity tests; plus 16 loader workers.
  - Needs a human decision plus one MobileNetV4-S validation run each:
    BF16 autocast for training (evaluation stays fp32), progressive
    resizing, and trimming the per-step diagnostic forwards.
  - Not adopted:
    - larger batch (human decision; changes the recipe);
    - channels_last (measured no gain in fp32);
    - DALI/FFCV (the loader is not the limit; changes decode/resize);
    - pre-resized dataset (changes training pixels; Hamster's 251 GB RAM
      already caches the full set);
    - persistent_workers (set_epoch would not reach persistent workers,
      silently repeating augmentation);
    - evaluating less often (changes best-checkpoint selection);
    - FixRes-style higher evaluation resolution (changes the protocol).
  - For Phase 2: cache robust-teacher outputs once, FKD-style, and reuse
    them across runs. Sample-keyed augmentation makes this exact.
  - ImageNet-100 screening on Anteater is a candidate use of the idle
    2080 Tis.
- 2026-09-26 (chat): **Sync-free training step and pinned loaders merged.**
  Scientific review found no P1, and all P2/P3 findings are fixed.
  - Per-step metric accumulation stays on the device, the non-finite-loss
    check is a sync-free `_assert_async` (still a `FloatingPointError` on
    CPU), and the train/val/probe loaders pin memory with non-blocking copies
    on CUDA.
  - A same-process old-vs-new differential test gives bit-identical epoch
    rows and checkpoint state (weights, optimizer, EMA, RNG, sampler,
    sample state, selection metadata) for PGD-AT, RSLAD and ADR. This holds
    on CPU and on CUDA with pinned loaders, the CUDA case run on an Anteater
    2080 Ti.
  - Remaining per-step syncs are in the attack's guards
    (`src/ard/attacks/pgd.py`: pixel range, step <= eps, random-start
    check, `max_abs_delta`). Removing them would touch attack guards, so it
    needs its own reviewed change.
  - Operational note: pinned host memory is ~2 GB per process at 8
    workers. Check host RSS in the first Phase 1 run.
- 2026-09-26 (postrun): **Phase 0 2x2, fourth cell completed (decision 0019
  B).** `plan0103-mobilenetv4-random-init-50ep-v1` (Hamster, 4090) ran 50/50
  epochs. `completion.json` reads `completed`, the manifest status is
  `completed`, and `error-marker.txt` reads "no application error recorded".
  The watcher re-derived it as terminal and successful. All 50 epoch rows are
  present. Nothing is imported into `docs/experiments/`: no aggregator exists
  for this contract (same as the other three cells).

  Fixed identity: as in the 2026-09-25 postrun entry above (ImageNet-1k,
  MobileNetV4-Conv-Small, plain PGD-AT, no teacher, seed 0, training attack
  l_inf 4/255 3 steps, selection PGD-10, world size 1, global batch 128, RTX
  4090, deterministic, protocol `controlled_imagenet_stage02_budget_init_v1`),
  except `pretrained: false`, 50 epochs, LR x0.1 at epochs 25 and 38, 10-epoch
  warmup. Source SHA `987dfb5` (worktree `source-987dfb534512`, clean). "val"
  is internal validation (held-out 2% of the training split), not the
  official ImageNet val. No AutoAttack has run.

  | cell | last (ep 49 / 99) val clean / PGD-10 | best (by val PGD) | last probe clean / PGD-10 |
  |---|---:|---:|---:|
  | random init, 50 ep (this run) | 53.36 / 30.05 | ep 48: 53.42 / 30.17 | 58.85 / 34.25 |
  | pretrained, 50 ep (Arm A, seed 0) | 54.57 / 30.58 | ep 46: 54.57 / 30.74 | not logged |
  | pretrained, 50 ep (Arm A, seed 1) | 54.33 / 30.59 | | not logged |
  | random init, 100 ep | 54.56 / 30.58 | ep 97: 54.44 / 30.68 | 60.75 / 35.00 |
  | pretrained, 100 ep | 55.21 / 30.70 | ep 91: 54.88 / 31.03 | 59.55 / 34.90 |

  **Preregistered rule (decision 0019 B):** "pretraining helps at 50 epochs"
  only if last PGD-10 <= 29.58% (Arm A's lower seed, 30.58, minus 1pt).
  30.05% > 29.58%, so the verdict is **no PGD-10 difference above 1pt at 50
  epochs either (n=1)**.

  Reading (n=1 per cell, internal validation, PGD-10 only):
  1. Against pretrained/50, random init is behind by 0.53-0.54pt PGD-10 and
     0.97-1.21pt clean (last), and 0.57 / 1.15pt (best vs Arm A seed 0).
     Unlike the 100-epoch pair, all four differences point the same way,
     and the clean gap is at the 1pt scale. The rule is about PGD-10 only;
     the clean gap is reported, not judged.
  2. Random init gains +0.53pt PGD-10 and +1.20pt clean from 50 to 100
     epochs. Pretrained gains +0.12 / +0.64 (seed 0). So the 100-epoch budget
     closes most of random init's shortfall.
  3. Epochs 0-24 are identical to the random-init/100 run to every printed
     digit (epoch 0: 1.36 / 0.98; epoch 24: 37.12 / 19.51). This is expected:
     same seed, deterministic, same warmup and LR up to epoch 24, and it
     shows the `b0cfc9e` -> `987dfb5` changes left training unchanged.
  4. The lr-0.05 phase was cut from 40 to 15 epochs. At its end (epoch 24)
     val was 37.12 / 19.51, vs 38.02 / 20.19 at the end of the 100-epoch
     run's lr-0.05 phase (epoch 49). The lr-0.005 phase ended (epoch 37) at
     49.87 / 27.65, vs 50.09 / 27.81 (epoch 75). The shortfall appears in the
     final lr-0.0005 stage: 30.05 vs 30.58 PGD-10.
  5. The final stage saturated. Over epochs 44-49 the train loss moved
     3.749 -> 3.736 (0.35% relative) and val PGD-10 29.98 -> 30.05.
  6. Seen-unseen gap at the end (probe minus val): +5.5 clean / +4.2 PGD-10.
     The probe fits only ~34% of its own images under PGD-10, so the regime
     stays bias-dominated.

  Caveats: one seed. The CIFAR control-vs-control floor was 0.16-1.88pt, and
  Arm A's two seeds do not estimate a spread. The 0.5pt PGD-10 gap is inside
  that floor, and the ~1.2pt clean gap sits at its lower edge. Arm A ran from
  a different source SHA (plan 0101). Wall clock 21.7 h (2026-09-25 13:08 to
  09-26 10:49 UTC, ~843 img/s), i.e. about 0.43 GPU-h per epoch. Checkpoints
  `best.pt`, `last.pt`, `epoch-049.pt` are on disk; their sha256 was not
  computed in this postrun (the session cannot hash files outside the repo).
  `epoch-metrics.parquet` sha256 `1930be95...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id. Next step: decision
  packet 0020.
- 2026-09-26 (chat): **Decision 0019 B completed:
  `plan0103-mobilenetv4-random-init-50ep-v1`** (Hamster GPU0, source
  `987dfb5`, 50/50 epochs, `completion.json` completed; internal val, seed 0,
  n=1).
  - Last epoch: 53.36 / 30.05 (clean / PGD-10). Best: epoch 48, 30.17. Probe
    at the last epoch: 58.85 / 34.25.
  - Preregistered rule: last PGD-10 <= 29.58% would mean "pretraining helps
    at 50 epochs". 30.05 > 29.58, so **no effect above 1pt is visible at 50
    epochs either** (Arm A seed 0 gives 54.57 / 30.58, a difference of
    -1.21 clean / -0.53 PGD-10).
  - Random init at 100 epochs gave 54.56 / 30.58. Doubling the budget moved
    random init by +1.2 clean / +0.5 PGD-10, inside the noise floor.
  - This is Arm A's recipe for both inits. The fair best-vs-best comparison
    is plan 0104 stage 2, which is running now.
- 2026-09-26 (human-approved, chat): **`training.step_diagnostics` merged.**
  - Default `true` is bit-identical to before, and the field is serialized only
    when false, so config hashes are unchanged.
  - `false` skips the two per-step diagnostic eval-mode forwards (~17%
    compute). The epoch rows then drop train_clean_accuracy,
    train_robust_accuracy_eval_mode, train_robust_overtakes_clean and
    train_ema_student_agreement.
  - Training state is proven identical under `false`: checkpoints, RNG,
    sampler and selection match on CPU for PGD-AT, RSLAD and ADR, and in a
    2-rank DDP test. Under DDP, BN buffers resync at the next DDP forward
    instead of at `train_epoch` return.
  - The field is observability-only and not part of the evaluation pooling
    identity, like `train_probe_size`.
  - Scientific review found no P1, and all findings are fixed.
  - Phase 1 configs set `step_diagnostics: false` and keep
    `train_probe_size: 2000`, the remaining eval-mode seen-set signal.
- 2026-09-27 (human, chat): **Budget question refined.** Lightweight models
  normally do not train with heavy augmentation; the value of long schedules
  there comes from distillation, which gives new information every epoch.
  Squeezing the last 1-2% with a long cosine tail is not a goal, and 300
  epochs is the upper bound worth considering. Hypothesis: the ~50-epoch
  saturation reflects the difficulty of robust versus standard
  classification. Approved:
  - (A) a clean-training control: a new `standard` method (no training
    attack), same recipe, random init, MobileNetV4-S, 50 and 100 epochs, on
    Anteater's 2080 Tis (a comparison run entirely on that GPU type).
    Implementation is in progress.
  - (B) a literature survey on adversarial-training epoch budgets and
    capacity limits for small models.
  Ferret (3x 4090) is free and takes 4090-only work: plan 0104's last
  stage-2 run, and on GPU1 a BF16 / channels_last / compile throughput probe
  with all six models. On GPU2, 3-epoch Phase-1-settings canaries for
  EfficientNet-B0 and MobileViT-S (deterministic off, cudnn autotuning,
  compile, step_diagnostics off, 16 workers; dev W&B project; worktree
  `source-bf37707ee8de`). Early probe reading: MobileNetV4-S reaches only
  ~868 img/s on a Ferret 4090 against ~1739 on Hamster. BF16 adds ~1%. This
  fits a CPU-launch-bound small model on a host with slower cores; to be
  confirmed.
- 2026-09-27 (chat): **Ferret is launch-bound per process; co-location
  doubles throughput there.** Throwaway probe on Ferret GPU1, MobileNetV4-S,
  nondeterministic plus cudnn autotuning, synthetic pixels:
  - one process: 876 img/s unpinned, 740 img/s pinned to the GPU's NUMA
    node 0;
  - two processes on the same GPU, pinned: 740 + 739 = ~1480 img/s, with no
    per-process slowdown.
  Single-job GPU utilization on Ferret is only 30-67%. Hamster has the same
  CPU model (Xeon Gold 6230R, but single-socket) and reaches ~1739 img/s for
  one process. The cause of the per-process gap is still open: NUMA pinning
  did not help, and CPU contention from other jobs on node 0 may be involved.
  This does not contradict the Hamster co-location loss: there one job
  already kept the GPU at ~80%, and the partner was EfficientNet-B0. A
  per-model single / two-concurrent / BF16 probe is running on Ferret GPU1.
  The BF16 probe so far: MobileNetV4-S +1%, so BF16 does not help a
  launch-bound model.
- 2026-09-27 (chat): **Literature on epoch budgets (option B, subagent,
  checked against the paper texts).**
  - **Budgets.** ImageNet adversarial training of CNNs with basic
    augmentation uses 90-110 epochs: Salman 2020 (90), Xie 2020 (110),
    EasyRobust (90, RRC+flip, eps 4/255), ARES ResNets (95), RobustART (100).
    300 epochs appears only with heavy augmentation (ViT/ConvNeXt).
  - **Studies that vary the budget.**
    - Wong 2020: FGSM for 15 epochs vs Free-AT for 92 is ~1pp at 4/255.
    - SAT sec 3.2: ResNet-50 at 100 -> 200 epochs is +2.6 clean but -1.8
      robust.
    - Singh/Croce/Hein 2023: 50 -> 300 epochs helps only with heavy
      augmentation and >=22M-parameter models.
    - No paper varies the budget for ImageNet adversarial training below
      10M parameters. That is an open gap.
  - **Capacity.**
    - Madry 2018 sec 4: capacity is crucial.
    - Xie 2020 sec 5: even ResNet-152 underfits the adversarial
      distribution.
    - SAT: robustness rises from EfficientNet-B0 to B7.
    - Debenedetti 2022: adversarial training needs much more capacity, and
      heavy augmentation costs capacity.
  - **Clean lightweight recipes.** They are long because of heavy
    augmentation or distillation. The official MobileNetV4-Conv-S recipe is
    9600 epochs with RandAug, Mixup, CutMix and EMA (Qin 2024, App. C,
    Table 10). Beyer 2022 shows patient function matching keeps improving.
  - **Reference points.** EasyRobust EfficientNet-B0 (90 epochs): 61.83
    clean / 35.06 AutoAttack. SAT EfficientNet-B0 (PGD-1): 65.1 / 37.6
    PGD-200.
  - **Bottom line.** ~50-epoch saturation is consistent with the literature.
    The mechanism is capacity-limited underfitting of the robust objective.
    Robust overfitting is a large-model phenomenon. The clean-training
    control (option A) tests this directly for MobileNetV4-S.
- 2026-09-27 (chat): **Ferret probe numbers are contaminated.** From ~15:50Z,
  another user's EEG jobs (~36 processes) ran on all three Ferret GPUs. The
  per-model single / two-concurrent / BF16 numbers measured after that time
  are void, and the probe was stopped. Earlier, uncontaminated readings on
  Ferret were MobileNetV4-S single 868-876 img/s and two-concurrent 740+739.
  Why a lone Ferret process runs at ~half of Hamster is still unexplained:
  the GPUs showed no power or clock throttling, so the cause must be
  elsewhere. The plan-0104 lr-0.015 run and the EfficientNet-B0 canary keep
  running on Ferret; they are slower but their numerics are unaffected.
  BF16 and co-location decisions wait for a clean measurement on Hamster.
- 2026-09-27 (chat): **Option A preregistered rule, written before launch.**
  - **Runs.** `imagenet_mobilenetv4_standard_50ep.yaml` and `..._100ep.yaml`
    (method `standard`: clean CE, no training attack; random init; Arm A's
    recipe; deterministic; seed 0; train probe 2000). They run on Anteater
    GPU2/GPU3 in parallel, a comparison made entirely on the 2080 Ti.
  - **Reference.** Adversarial training with the same recipe and random init
    moved from 50 to 100 epochs by +1.20 clean / +0.53 PGD-10 (53.36/30.05
    -> 54.56/30.58, Hamster 4090).
  - **Metric.** Last-epoch internal-val clean accuracy. `best.pt` is chosen by
    val PGD-10, which is ~0 for a clean model, so it is not used.
  - **Rule.** Let G = clean(100) - clean(50) for the standard runs.
    - G >= 2pt: "the 50-epoch saturation is specific to the robust
      objective".
    - G < 1pt: "the recipe itself saturates by 50 epochs; the saturation is
      not specific to adversarial training".
    - Otherwise inconclusive.
  - **Also reported.** Seen-set probe clean accuracy of the standard runs, a
    fit check to compare with ~35% seen-set PGD-10 under adversarial
    training, and the stage-saturation pattern per LR stage. n=1 per cell.
- 2026-09-27 (chat): **Option A launched.** Method `standard` merged as
  `2762b89`. Scientific review found no P0/P1/P2; the P3 follow-ups are in
  progress, none blocking.
  - Anteater GPU2 runs `plan0103-mobilenetv4-standard-50ep-v1` and GPU3
    runs `plan0103-mobilenetv4-standard-100ep-v1`, from pinned worktree
    `source-2762b89ead95` (seed 0, 7 loader workers each, W&B
    `lightweight-imagenet-at`).
  - The review confirmed the standard path shares data order, augmentation,
    optimizer, scheduler, RNG streams and BN update pattern with the PGD-AT
    cells; only the training attack is removed.
  - `best.pt` / W&B best_* for these runs are meaningless (selection by val
    PGD-10 ~0). The analysis reads last-epoch clean, per the preregistered
    rule.
- 2026-09-27 (chat): **Ferret free again; 4090 work resumed.**
  - Step 1c canaries on Ferret GPU2 with Phase-1 options (nondet, cudnn
    autotuning, compile, step_diagnostics off, 16 workers; source
    `bf37707`) both completed rc=0. EfficientNet-B0 at epoch 2: 58.77 /
    30.77, 355 img/s. MobileViT-S at epoch 2: 57.19 / 27.03, 180 img/s.
    Both overlapped partly with another user's jobs, so the throughput is
    indicative only. Step 1c is now complete for all six models.
  - Ferret GPU1 reruns the per-model single / two-concurrent / BF16 probe
    with the host otherwise idle.
  - Ferret GPU2 runs the official evaluation (full ImageNet val clean +
    PGD-10, AutoAttack n=500, best and last, `ard.cli.evaluate`, source
    `b004d51`) of `plan0103-mobilenetv4-{pretrained-100ep-v2,
    random-init-100ep-v1, random-init-50ep-v1}` and
    `plan0104-mobilenetv4-ft-lr0005-v1`. Checkpoints were copied from
    Hamster, with sha256 verified identical. This is followed by the
    EasyRobust EfficientNet-B0 external anchor (published 61.83 clean /
    35.06 AutoAttack).
  - The Ferret stage-2 run (pretrained lr 0.015) was slowed by the
    overlap: epoch 18/50, about 25 h left.
  - Per plan 0104 stage 2's rule, each init's Phase 1 learning rate is its
    own argmax. The random-init side is complete once Hamster's lr 0.1 and
    lr 0.025 runs end (~08:30Z today), so Phase 1 random-init runs start
    then, beginning with ConvNeXt-Atto and DeiT-Tiny. The pretrained side
    waits for lr 0.015.
- 2026-09-27 (human, chat): **Finalist selection (the 2-3 models that get a
  second seed) is decided by the human after Phase 1, from several
  criteria rather than robustness alone.** To avoid post-hoc choice of
  criteria, the dimensions are fixed now and weighted later. Every model
  will be shown on all of them:
  1. robustness: AutoAttack n=500 and full-val PGD-10;
  2. clean accuracy and the clean cost of robustness;
  3. efficiency: robustness per parameter and per MAC, and whether the
     model is Pareto-optimal;
  4. pretraining dependence: pretrained minus random-init, both inits for
     every model;
  5. training state: seen-vs-unseen gap (under- vs overfitting) and
     per-LR-stage saturation;
  6. training cost: GPU-hours per run;
  7. architectural coverage (BN CNN, LN CNN, ViT, hybrid) as rows of the
     Phase 2 method table.
  Finalist selection only allocates second seeds; it makes no claim.
- 2026-09-27 (chat): **Ferret throughput probe rerun on an idle host**
  (06:48Z onward, synthetic trainer-shaped step, non-deterministic plus
  cudnn autotuning). Another user's EEG jobs started on all three GPUs at
  07:01Z. The probe log has no timestamps, so the last rows (MobileNetV4-S)
  may overlap them.
  - **Single process, img/s (Hamster in parentheses).** EfficientNet-B0 386
    (383), MobileNetV4-M 540 (593), ConvNeXt-Atto 850 (869), DeiT-Tiny 650
    (631), MobileViT-S 218 (213). An idle Ferret matches Hamster, so the
    earlier "half speed" readings came from contention.
  - **Two processes on one GPU, total img/s.**
    - Lower than one process for MobileNetV4-M (504), ConvNeXt-Atto (708)
      and DeiT-Tiny (591).
    - EfficientNet-B0 and MobileViT-S run out of memory.
    - Only MobileNetV4-S gains, at ~1.8x (781 + 825), and that may be
      contaminated.
    - Co-location is therefore not used in Phase 1, except possibly for
      MobileNetV4-S.
  - **BF16 autocast gain.** +0-6% for most models, +18% DeiT-Tiny, +14%
    MobileViT-S. Not worth the risk of weaker low-precision attack
    gradients, so BF16 is not recommended for Phase 1 (pending human
    confirmation).
  - The official evaluation of `plan0103-mobilenetv4-pretrained-100ep-v2`
    finished (rc=0) at 07:01Z, before the overlap. The remaining
    evaluations and the plan-0104 lr 0.015 run continue; they are slower but
    their numerics are unaffected.
- 2026-09-27 (chat): **Phase 1 random-init side starts.** Configs are
  `imagenet_{efficientnet_b0, mobilenetv4_conv_medium, convnext_atto,
  deit_tiny, mobilevit_s}_pgd_at_phase1_random.yaml`. Each is its survey
  config with random init, lr 0.025 (plan 0104 stage-2 random argmax) and the
  approved throughput options: nondeterministic plus cudnn autotuning,
  step_diagnostics off, and compile only for EfficientNet-B0 and
  MobileViT-S. BF16 is not used. The config test pins that nothing else
  differs.
  - MobileNetV4-S's random-init Phase 1 cell is the stage-2 lr 0.025 run
    itself (deterministic; its step_diagnostics and determinism differ from
    the other rows only in non-training observability and in
    reproducibility).
  - Order: ConvNeXt-Atto (Hamster GPU0) and DeiT-Tiny (Hamster GPU1) first,
    then MobileNetV4-M, EfficientNet-B0 and MobileViT-S.
- 2026-09-27 (human, chat): **BF16 is not adopted.** All Phase 1 runs stay
  fp32 (TF32 convolutions under PyTorch defaults on the 4090s).
- 2026-09-27 (chat): **Official evaluations of the 2x2 and the FT run**
  (Ferret 4090, full ImageNet val 50k, PGD-10, AutoAttack n=500 seeded
  subset, source `b004d51`, last / best):
  - pretrained 100ep: 51.72 / 27.81 / 23.6 (last); 51.53 / 28.23 / 23.8
    (best).
  - random 100ep: 51.15 / 27.85 / 22.0 (last); 51.25 / 27.99 / 23.6 (best).
  - random 50ep: 50.18 / 27.40 / 22.8 (last); 50.07 / 27.37 / 23.2 (best).
  - FT lr 0.005: 54.17 / 28.49 / 24.2 (last); 54.06 / 28.44 / 23.2 (best).
  - Arm A (pretrained 50ep, plan 0101, best only): clean 50.94,
    AutoAttack 22.8.
  - The AutoAttack SE at n=500 is ~1.9pt, so the 22.0-24.2 spread is not
    resolvable. FT's +2.5-4pt clean advantage reproduces on the official
    split. The EasyRobust anchor's first attempt failed (missing ARD_SEED
    env; the wrapper's `rc=$?` read `date`'s status, now fixed) and is
    rerunning on Ferret GPU2.
- 2026-09-28 (chat): **EasyRobust external anchor reproduced** (Ferret 4090,
  `scripts/evaluate_external_checkpoint.py --dataset imagenet`, protocol
  config `imagenet_efficientnet_b0_pgd_at.yaml`, source `b004d51` clean,
  FOREIGN lineage, not an official test of ours). Full val 50k: clean 61.05
  (published 61.83, -0.78pp), PGD-10 37.59. AutoAttack n=500: 35.0
  (published 35.06, -0.06pp; SE ~2.1pt). So our evaluation stack
  (preprocessing, PGD, pinned AutoAttack) agrees with the field's.
  EfficientNet-B0 trained by EasyRobust (90 epochs from scratch) reaches
  ~35% AutoAttack, against ~23% for our MobileNetV4-S runs.
- 2026-09-28 (human, chat): **Ferret may be used again** (the human watched it
  for 15 min with no new foreign processes). Phase 1 random-init
  `plan0103-phase1-efficientnet-b0-random-v1` (Ferret GPU1) and
  `plan0103-phase1-mobilenetv4-conv-medium-random-v1` (Ferret GPU2)
  launched from Ferret's pinned worktree `source-086072bcde92`, the same SHA
  as the Hamster Phase 1 runs, with 16 workers. If foreign jobs return, the
  runs slow down but their numerics are unaffected.
- 2026-09-28 (chat): **Co-location on Ferret, guided by measured headroom.**
  Real-training GPU utilization: GPU1 EfficientNet-B0 78-100% (10.5 GB),
  GPU2 MobileNetV4-M 54-90% (9.7 GB), GPU0 MobileNetV4-S deterministic lr
  0.015 23-53% (4 GB, launch-bound). The idle-Ferret probe showed that two
  jobs per GPU lower total throughput for MobileNetV4-M and run
  EfficientNet-B0 out of memory, so GPU1/GPU2 stay single. On GPU0, the
  longest Phase 1 cell, `plan0103-phase1-mobilevit-s-random-v1` (~4 days),
  was co-located with the plan-0104 lr 0.015 run (a few hours left). It takes
  the GPU alone once that run ends. Numerics are unaffected; only speed
  changes.
- 2026-09-28 (human, chat): **Paper framing (A1-A5 of the open-issues list).**
  - **A1, the claim.** For each lightweight model: robustness per parameter
    and per MAC, and the training approach that suits it (including whether
    pretraining is needed). Where the model underfits, why, and which remedy
    fixes it. As research questions:
    - RQ1 efficiency: the robustness Pareto frontier.
    - RQ2 prescription: per-model init, recipe and method.
    - RQ3 mechanism: why underfitting happens, and which remedies work and
      why.
  - **A2, fairness.** The human noted that no shared recipe is neutral:
    what works for MobileNet need not work for EfficientNet. Adopted
    principle: "same conditions" means an **equal tuning budget per model,
    compared best-vs-best**, not identical hyperparameters. This is the
    same principle as plan 0104's per-init LR comparison.
    - Fixed-recipe results are kept as a sensitivity reference.
    - Tuning on a cheap proxy (short schedule / ImageNet-100 /
      successive halving) needs a rank-correlation check against the full
      run.
    - Consequence: the current Phase 1 (MobileNetV4-S LR transferred to
      every model) is the biased variant. ConvNeXt-Atto and DeiT-Tiny random
      init reach only ~30% clean, likely a recipe mismatch. Phase 1's design
      is to be revisited (B1/B2).
  - **A3, audience.** Robustness researchers. Since mobile is the premise,
    speed is reported properly at the end: params, MACs, and real-device
    latency (possibly int8). No venue targeting; best effort.
  - **A4, threat models.** l_inf 4/255 now. l2 later, first as evaluation of
    l_inf-trained models, then training if needed.
  - **A5, framing.** "Remedies in the capacity-limited regime", examined
    critically.
    - Deployment-readiness is not claimed: ~51% clean and 23-35%
      AutoAttack.
    - "Capacity" is a hypothesis to test, not an assumption; the
      clean-training control hints at recipe saturation.
    - Every remedy must beat simply spending the same compute on a larger
      model, so the frontier includes scaled-up baselines.
- 2026-09-28 (postrun): **Phase 1 random init, ConvNeXt-Atto completed.**
  `plan0103-phase1-convnext-atto-random-v1` (Hamster GPU0, RTX 4090) ran
  50/50 epochs. `completion.json` reads `completed`, the manifest status is
  `completed`, and `error-marker.txt` reads "no application error recorded".
  The watcher re-derived it as terminal and successful. All 50 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback, NaN
  or error line. `best.pt`, `last.pt` and `epoch-049.pt` are on disk. Nothing
  is imported into `docs/experiments/`: no aggregator exists for this
  contract, as for every other hand-run in plans 0103/0104. The numbers below
  are read from `outputs/train/epoch-metrics.jsonl` and the run-bundle
  manifest summary.

  Fixed identity: ImageNet-1k, ConvNeXt-Atto (V1, `convnext_atto_imagenet`,
  3.70M params / 0.55 GMACs), random init (`student.pretrained: false`), plain
  PGD-AT, no teacher. Training and evaluation-attack seed 0 (split seed
  20260911). Training attack: CE PGD-3, l_inf eps 4/255, step 8/765, random
  start. Selection and validation attack: PGD-10, same eps, eval mode. SGD
  Nesterov, peak LR 0.025, wd 1e-4, warmup_multistep (10-epoch warmup, x0.1 at
  epochs 25 and 38), 50 epochs. World size 1, global batch 128, local
  BatchNorm. Non-deterministic with cuDNN autotuning, no compile,
  step_diagnostics off, fp32 (TF32 convolutions). Protocol
  `controlled_imagenet_stage02_lightweight_architecture_survey_v1`, config
  `imagenet_convnext_atto_pgd_at_phase1_random.yaml`, config hash
  `cf6de143...`. Source SHA `086072bcde9243a4b85a31f009f26249d7bc386f`
  (worktree `source-086072bcde92`, clean). "val" is internal validation (the
  held-out 2% of the training split, 25,620 images), not the official
  ImageNet val. "probe" is 2,000 training images (2 per class), same
  transform, eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.56 / 0.50 | 0.35 / 0.25 | 6.813 |
  | 9 | end of warmup | 8.86 / 4.86 | 8.80 / 5.00 | 6.038 |
  | 24 | end of lr 0.025 | 21.95 / 12.17 | 22.55 / 12.45 | 5.282 |
  | 25 | first epoch at lr 0.0025 | 26.98 / 14.48 | 26.80 / 14.20 | 5.065 |
  | 37 | end of lr 0.0025 | 28.91 / 15.25 | 29.40 / 15.20 | 4.955 |
  | 38 | first epoch at lr 0.00025 | 29.86 / 16.19 | 30.35 / 16.30 | 4.883 |
  | 45 | best (by val PGD-10) | 30.82 / 16.59 | 31.10 / 16.80 | 4.858 |
  | 49 | last | 30.68 / 16.53 | 31.45 / 15.60 | 4.853 |

  For comparison, the MobileNetV4-Conv-Small random-init cell with the same
  LR, epochs and attack (plan 0104 stage 2, `76280eb`, deterministic) ended
  at 53.85 / 30.71 (last = best).

  Reading (n=1, internal validation, PGD-10 only):
  1. **ConvNeXt-Atto trains far worse than MobileNetV4-S under this recipe.**
     At almost the same parameter count (3.70M vs 3.77M) and ~3x the MACs, it
     ends 23.2pt lower in clean and 14.2pt lower in PGD-10. This is far
     outside any noise floor. It confirms the ~30% clean reading behind A2 /
     B1 (recipe mismatch suspected) with the finished run.
  2. **Learning is slow from the start.** By the end of the 10-epoch warmup,
     val clean is only 8.86%. Train loss is 6.81 at epoch 0 (chance is
     ln 1000 = 6.91) and 4.85 at the end. At the same LR, MobileNetV4-S
     random init was at 41.59 / 22.29 by epoch 24; ConvNeXt-Atto is at
     21.95 / 12.17.
  3. **The peak-LR stage did not saturate.** Val PGD-10 rose from 4.86% to
     12.17% over epochs 9-24, and still by about 0.3pt per epoch over the
     last five (10.95 -> 12.17). The first decay then gave +2.31pt PGD-10 and
     +5.03pt clean in one epoch. So the run was cut while it was still
     improving at the peak LR. The later stages do flatten: +0.77pt PGD-10
     over epochs 25-37, and 16.34-16.59% over epochs 44-49 (train loss
     4.859 -> 4.853).
  4. **No seen-unseen gap.** At the last epoch, probe minus val is +0.77
     clean and -0.93 PGD-10, inside the probe's binomial SE (about 1.0 /
     0.8pt). The model fits its own training images as badly as unseen ones.
     This is the underfitting signature, stronger than for MobileNetV4-S
     (+5.9 / +3.7).
  5. No collapse and no instability. `robust_overfit_gap` (best minus last
     PGD-10) is 0.05pt.
  6. What this run does not tell: whether the cause is the LR, the optimizer
     (SGD vs the AdamW that ConvNeXt normally uses, B2), the warmup length,
     the 50-epoch budget, or the random init itself. The Anteater
     pretrained-init canary (2080 Ti, deterministic, lr 0.05, 3 warmup epochs)
     was already at 48.65 / 26.67 at epoch 2, but it differs in init, LR,
     GPU type and determinism, so it is context, not a comparison.

  Caveats: one seed. Non-deterministic mode, so a rerun would not be
  bit-identical. The MobileNetV4-S reference ran deterministic from a
  different SHA; the gap here is far larger than either difference could
  explain. Cost: wall clock 25.4 h (2026-09-27 08:59 to 09-28 10:26 UTC),
  718-723 img/s (the idle-Hamster probe gave 869 for this setting), peak
  allocated memory 5.7 GB. Checkpoint sha256 was not computed in this
  postrun. `epoch-metrics.parquet` sha256 `d73a1df7...`,
  `sample-stats-train.parquet` sha256 `9f1c006b...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id. Hamster GPU0 is now
  idle; DeiT-Tiny random init still runs on GPU1.
- 2026-09-28 (chat): **Literature: official clean recipes vs adversarial-training
  recipes per family** (A2 item 1, subagent, checked against the paper texts;
  items not checked are flagged unverified in the chat record).
  - **Official clean recipes** are co-tuned with heavy augmentation, large
    batches and 300-9600 epochs (MNv4-S: AdamW, 9600 ep). They are not a
    sound adversarial-training baseline wholesale.
  - **Heavy augmentation from random init makes adversarial training of
    ConvNeXt/DeiT fail** (Singh 2023 sec 4.2; Debenedetti 2022 Tab. IV; Bai
    2021 sec 4.1). Keep basic augmentation for all models.
  - **The optimizer follows the normalization type.**
    - LN/attention models (ConvNeXt, DeiT, and MobileViT per its own
      recipe) use AdamW in every ImageNet adversarial-training paper.
      RobustART: "SGD would cause the failure of training". DeiT Tab. 8:
      SGD -7 pts even clean.
    - BN CNNs use SGD everywhere, and AdamW on a BN ResNet collapsed under
      adversarial training (Bai 2021), consistent with our plan-0102
      MobileNetV4 collapse.
    - Our ~30% clean for ConvNeXt-Atto and DeiT-Tiny random init under SGD
      matches this known failure mode.
  - **LR at batch 128.** AdamW ~1.25e-4 (linear rule) to ~2.5e-4-7e-4
    (square-root rule); SGD 0.025-0.05.
  - **Weight decay.** The evidence conflicts: Debenedetti 0.5 helps, Singh
    0.05 best, ARES larger hurts.
  - **Recommended per-family baseline.** 50 ep, 10-ep warmup and the same
    multistep schedule for all:
    - BN CNNs (MNv4-S/M, EfficientNet-B0): SGD, wd 1e-4.
    - ConvNeXt-Atto and DeiT-Tiny: AdamW, wd 0.05, no stochastic depth.
    - MobileViT-S: AdamW, wd 0.01.
    - Tune peak LR (3 points) per model and init.
  - **Implication.** The current SGD Phase 1 random-init runs for
    ConvNeXt-Atto, DeiT-Tiny and MobileViT-S are "fixed-recipe" reference
    runs, not the family-appropriate baseline. Decision pending (B1/B2).
- 2026-09-28 (chat): **Clean-training control, official val.**
  `plan0103-mobilenetv4-standard-50ep-v1`, last checkpoint, full ImageNet val:
  clean 67.47%. Internal val was 70.86% and the seen-set probe 77.70%. Our own
  pipeline gives 73.41% for the public MobileNetV4-S e1200 weights. So 50
  epochs with basic augmentation lands ~6pt below the long heavy-augmentation
  recipe; it does not reproduce it.
- 2026-09-28 (human, chat): **B1/B2 decided: per-family baseline recipe.**
  - BN CNNs (MobileNetV4-S/M, EfficientNet-B0): SGD, wd 1e-4.
  - ConvNeXt-Atto and DeiT-Tiny: AdamW, wd 0.05, no stochastic depth.
  - MobileViT-S: AdamW, wd 0.01.
  - Common to all: 50 epochs, 10-epoch warmup, warmup_multistep, basic
    augmentation.
  - Peak LR is tuned with 3 points per model and init, equal budget,
    best-vs-best. The exact grids are still to be approved.
  - The SGD transfer runs are the "fixed-recipe" reference:
    - `plan0103-phase1-convnext-atto-random-v1` completed. Last 30.68 /
      16.53, probe 31.45 / 15.60: fails to fit even the seen set.
    - `plan0103-phase1-deit-tiny-random-v1` was stopped by the human's
      decision at epoch 38 (DataLoader-worker SIGTERM; bundle
      incomplete).
    - `plan0103-phase1-mobilevit-s-random-v1` was stopped by the human's
      decision early in training (bundle incomplete).
    - Stopped runs are not results.
  - EfficientNet-B0 and MobileNetV4-M random-init SGD lr 0.025 keep running
    on Ferret as the SGD-family reference (their per-model LR tuning is
    still to come).
- 2026-09-28 (human, chat): **LR grids approved; option a (start the
  ImageNet-100 tuning before the proxy verdict) approved.**
  - Grids, 3 points per model x init:
    - SGD random {0.0125, 0.025, 0.05}; SGD pretrained {0.005, 0.015,
      0.05}.
    - AdamW random {1.25e-4, 2.5e-4, 5e-4}; AdamW pretrained {6.25e-5,
      1.25e-4, 2.5e-4}.
    - AdamW betas 0.9/0.999, as in the official ConvNeXt/DeiT/MobileViT
      recipes.
  - ImageNet-100 was built on Hamster and Ferret as well, with identical
    manifest hashes on all three hosts.
  - The 18 AdamW configs are `imagenet100_{convnext_atto,deit_tiny,
    mobilevit_s}_adamw_{random,pretrained}_lr*.yaml`. A config test pins that
    only the dataset, head size, init, optimizer, throughput options,
    protocol and group differ from the survey config.
  - Expected wall time is ~1.5 days on three 4090s; MobileViT-S dominates
    at ~10 h per run.
- 2026-09-29 (postrun): **ImageNet-100 LR tuning, MobileViT-S AdamW random
  init / lr 1.25e-4 completed.** `plan0103-in100-tune-mobilevit-s-adamw-random-lr1p25em4-v1`
  (Hamster, RTX 4090) ran 50/50 epochs. `completion.json` reads `completed`,
  the manifest status is `completed`, and `error-marker.txt` reads "no
  application error recorded". The watcher re-derived it as terminal and
  successful. All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root and
  inductor TF32 warnings). `best.pt`, `last.pt` and `epoch-049.pt` are on
  disk. Nothing is imported into `docs/experiments/`: no aggregator exists for
  this contract, as for the other plan 0103/0104 hand-runs. The numbers below
  are read from `outputs/train/epoch-metrics.jsonl` and the run-bundle
  manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  MobileViT-S (`mobilevit_s_imagenet`, `imagenet_raw_identity` profile),
  random init, plain PGD-AT, no teacher. Training and evaluation-attack seed 0
  (split seed 20260911). Training attack: CE PGD-3, l_inf eps 4/255, step
  8/765, random start. Selection and validation attack: PGD-10, same eps, eval
  mode. AdamW (betas 0.9/0.999), peak LR 1.25e-4, wd 0.01, warmup_multistep
  (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1,
  global batch 128, local BatchNorm. Non-deterministic with cuDNN autotuning,
  `torch.compile`, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_mobilevit_s_adamw_random_lr1p25em4.yaml`, config hash
  `b80c0d42...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation (the
  held-out 2% of the ImageNet-100 training split, 2,564 images), not the
  ImageNet-100 val split. "probe" is 2,000 training images, same transform,
  eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 2.03 / 1.56 | 1.90 / 1.15 | 4.690 |
  | 9 | end of warmup | 15.72 / 10.65 | 17.50 / 12.35 | 4.120 |
  | 24 | end of lr 1.25e-4 | 49.69 / 32.22 | 54.25 / 36.05 | 2.970 |
  | 25 | first epoch at lr 1.25e-5 | 52.77 / 35.37 | 57.35 / 37.95 | 2.877 |
  | 37 | end of lr 1.25e-5 | 54.60 / 35.45 | 59.00 / 39.10 | 2.790 |
  | 38 | first epoch at lr 1.25e-6 | 55.54 / 36.66 | 59.05 / 40.15 | 2.782 |
  | 48 | best (by val PGD-10) | 56.28 / 37.13 | 59.45 / 39.95 | 2.772 |
  | 49 | last | 55.93 / 36.27 | 59.90 / 40.15 | 2.764 |

  Reading (n=1, one of three random-init LR cells, internal validation,
  PGD-10 only):
  1. **The run trains normally.** No collapse, no instability. Val clean ends
     at 55.9% on 100 classes (chance 1%).
  2. **The warmup start is slow.** Over epochs 1-6 (LR 2.5e-5 to 8.75e-5)
     val clean stays at 2.5-5.9%. It takes off only at epochs 7-9, when the
     LR reaches 1.0e-4 to 1.25e-4.
  3. **The peak-LR stage did not saturate.** Val PGD-10 rose by 3.2pt over
     the last five peak-LR epochs (28.98% at ep 19 to 32.22% at ep 24), about
     0.65pt per epoch; clean rose 1.3pt per epoch. The first decay then gave
     +3.2pt PGD-10 and +3.1pt clean in one epoch.
  4. **The two low-LR stages are nearly flat.** Val PGD-10 stays within
     34.91-36.35% over epochs 25-37 and 35.69-37.13% over epochs 38-49. The
     val SE is about 1.0pt (2,564 images), so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.86pt PGD-10
     (`robust_overfit_gap`), under one val SE. Best selection likely picked a
     noise peak.
  6. **Small seen-unseen gap.** At the last epoch, probe minus val is +4.0
     clean and +3.9 PGD-10, about 2.7x the SE of the difference (~1.5pt).
  7. Points 2 and 3 are consistent with 1.25e-4 being below the best peak LR
     for this cell, but only the 2.5e-4 and 5e-4 cells can say that. No
     verdict on the grid until all three random-init cells are done.

  Caveats: one seed. Non-deterministic mode with `torch.compile`, so a rerun
  would not be bit-identical. ImageNet-100 numbers are not comparable to the
  ImageNet-1k Phase 1 runs; the stopped SGD MobileViT-S run is not a result.
  Cost: wall clock 8.9 h (2026-09-28 12:16 to 21:12 UTC), 216-217 img/s
  after epoch 0 (141 img/s at epoch 0, compile warm-up), peak allocated
  memory 18.6 GB at epoch 0 and 14.5 GB at the last epoch. Checkpoint sha256
  was not computed in this postrun. `epoch-metrics.parquet` sha256
  `a6463bd7...`, `sample-stats-train.parquet` sha256 `60fad178...` (from the
  run-bundle manifest). W&B `lightweight-imagenet-at-dev`, same run id. The
  MobileViT-S random / 2.5e-4 and pretrained / 6.25e-5 cells have run
  directories on Hamster.
- 2026-09-29 (postrun): **ImageNet-100 LR tuning, MobileViT-S AdamW
  pretrained / lr 6.25e-5 completed.** `plan0103-in100-tune-mobilevit-s-adamw-pretrained-lr6p25em5-v1`
  (Hamster, RTX 4090) ran 50/50 epochs. The manifest status is `completed`
  and `error-marker.txt` reads "no application error recorded". The watcher
  re-derived it as terminal and successful. All 50 epoch rows are present
  (`epoch_metrics_complete: true`). `train.log` has no traceback, NaN or error
  line (only the wandb git-root and inductor TF32 warnings). `best.pt`,
  `last.pt` and `epoch-049.pt` are on disk. Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract, as for the
  other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  MobileViT-S (`mobilevit_s_imagenet`, `imagenet_raw_identity` profile),
  ImageNet-1k pretrained init (`pretrained: true`, 100-class head), plain
  PGD-AT, no teacher. Training and evaluation-attack seed 0 (split seed
  20260911). Training attack: CE PGD-3, l_inf eps 4/255, step 8/765, random
  start. Selection and validation attack: PGD-10, same eps, eval mode. AdamW
  (betas 0.9/0.999), peak LR 6.25e-5, wd 0.01, warmup_multistep (10-epoch
  warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1, global batch
  128, local BatchNorm. Non-deterministic with cuDNN autotuning,
  `torch.compile`, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_mobilevit_s_adamw_pretrained_lr6p25em5.yaml`, config hash
  `3d991547...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation (the
  held-out 2% of the ImageNet-100 training split, 2,564 images), not the
  ImageNet-100 val split. "probe" is 2,000 training images, same transform,
  eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 31.40 / 0.00 | 31.00 / 0.05 | 4.693 |
  | 1 | warmup | 8.39 / 2.42 | 8.30 / 1.85 | 4.621 |
  | 9 | end of warmup | 75.23 / 46.29 | 77.35 / 49.00 | 2.441 |
  | 24 | end of lr 6.25e-5 | 82.84 / 56.16 | 86.65 / 62.15 | 1.864 |
  | 25 | first epoch at lr 6.25e-6 | 83.42 / 57.02 | 87.10 / 62.05 | 1.839 |
  | 34 | best (by val PGD-10) | 83.62 / 57.88 | 87.15 / 64.05 | 1.816 |
  | 37 | end of lr 6.25e-6 | 84.17 / 57.49 | 87.85 / 64.05 | 1.805 |
  | 38 | first epoch at lr 6.25e-7 | 83.62 / 57.18 | 87.10 / 64.30 | 1.807 |
  | 49 | last | 84.24 / 57.18 | 87.60 / 63.95 | 1.794 |

  Reading (n=1, one of three pretrained LR cells, internal validation,
  PGD-10 only):
  1. **The run trains normally.** No collapse, no instability. Val clean ends
     at 84.2% on 100 classes (chance 1%).
  2. **Clean accuracy dips once at the start.** After epoch 0 (LR 6.25e-6)
     val clean is 31.4% with PGD-10 at 0.0%. After epoch 1 clean falls to
     8.4% while PGD-10 rises to 2.4%. From epoch 2 both climb steadily. The
     train loss stays near ln(100) = 4.6 over these two epochs.
  3. **The peak-LR stage is close to saturation.** Val PGD-10 rose by 1.3pt
     over the last five peak-LR epochs (54.84% at ep 19 to 56.16% at ep 24),
     about 0.26pt per epoch. The random-init 1.25e-4 cell was still rising
     at about 0.65pt per epoch at the same point. The first decay gave only
     +0.86pt PGD-10 and +0.58pt clean.
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     56.47-57.88% over epochs 25-37 and 56.71-57.80% over epochs 38-49. The
     val SE is about 1.0pt (2,564 images), so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.70pt PGD-10
     (`robust_overfit_gap`), under one val SE. Best selection likely picked a
     noise peak.
  6. **The seen-unseen gap is larger on the robust side.** At the last
     epoch, probe minus val is +3.4 clean (about 3x the SE of the difference,
     ~1.0pt) and +6.8 PGD-10 (about 4.6x its SE, ~1.5pt). The random-init
     1.25e-4 cell had +4.0 / +3.9.
  7. Against the random-init 1.25e-4 cell, this run is +28.3pt clean and
     +20.9pt PGD-10 at the last epoch. The two cells differ in both init and
     LR, so this is not the pretrained-vs-random comparison; that one is
     best-vs-best over each grid. No verdict on the LR grid until the 1.25e-4
     and 2.5e-4 pretrained cells are done.

  Caveats: one seed. Non-deterministic mode with `torch.compile`, so a rerun
  would not be bit-identical. ImageNet-100 numbers are not comparable to the
  ImageNet-1k Phase 1 runs. Cost: wall clock 9.1 h (2026-09-28 12:16 to
  21:19 UTC), on Hamster at the same time as the random-init 1.25e-4 cell,
  213-214 img/s after epoch 0 (140 img/s at epoch 0, compile warm-up), peak
  allocated memory 18.6 GB at epoch 0 and 14.5 GB after. Checkpoint sha256
  was not computed in this postrun. `epoch-metrics.parquet` sha256
  `e4d5f562...`, `sample-stats-train.parquet` sha256 `c704562c...` (from the
  run-bundle manifest). W&B `lightweight-imagenet-at-dev`, same run id.
- 2026-09-29 (postrun): **ImageNet-100 LR tuning, MobileViT-S AdamW random
  init / lr 2.5e-4 completed.** `plan0103-in100-tune-mobilevit-s-adamw-random-lr2p5em4-v1`
  (Hamster, RTX 4090) ran 50/50 epochs. `completion.json` reads `completed`,
  the manifest status is `completed`, and `error-marker.txt` reads "no
  application error recorded". The watcher re-derived it as terminal and
  successful. All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` and `epoch-049.pt` are on disk. Nothing is
  imported into `docs/experiments/`: no aggregator exists for this contract,
  as for the other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  MobileViT-S (`mobilevit_s_imagenet`, `imagenet_raw_identity` profile),
  random init, plain PGD-AT, no teacher. Training and evaluation-attack seed 0
  (split seed 20260911). Training attack: CE PGD-3, l_inf eps 4/255, step
  8/765, random start. Selection and validation attack: PGD-10, same eps, eval
  mode. AdamW (betas 0.9/0.999), peak LR 2.5e-4, wd 0.01, warmup_multistep
  (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1,
  global batch 128, local BatchNorm. Non-deterministic with cuDNN autotuning,
  `torch.compile`, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_mobilevit_s_adamw_random_lr2p5em4.yaml`, config hash
  `10861133...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation (the
  held-out 2% of the ImageNet-100 training split, 2,564 images), not the
  ImageNet-100 val split. "probe" is 2,000 training images, same transform,
  eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 2.50 / 2.26 | 2.45 / 2.00 | 4.627 |
  | 9 | end of warmup | 34.09 / 21.88 | 36.90 / 24.50 | 3.481 |
  | 24 | end of lr 2.5e-4 | 60.88 / 40.02 | 65.90 / 45.70 | 2.583 |
  | 25 | first epoch at lr 2.5e-5 | 65.05 / 44.34 | 70.60 / 50.15 | 2.452 |
  | 37 | end of lr 2.5e-5 | 66.54 / 45.94 | 72.75 / 51.55 | 2.335 |
  | 38 | first epoch at lr 2.5e-6 | 66.07 / 46.06 | 72.60 / 52.10 | 2.323 |
  | 46 | best (by val PGD-10) | 66.61 / 46.65 | 73.10 / 53.35 | 2.310 |
  | 49 | last | 66.50 / 45.94 | 73.55 / 53.10 | 2.298 |

  Reading (n=1, two of three random-init LR cells done, internal
  validation, PGD-10 only):
  1. **The run trains normally.** No collapse, no instability. Val clean ends
     at 66.5% on 100 classes (chance 1%).
  2. **The warmup start is faster than at 1.25e-4.** Val clean stays at
     3.4-8.3% over epochs 1-4 (LR 5e-5 to 1.25e-4) and takes off from epoch 5
     (LR 1.5e-4). At 1.25e-4 it took off only at epochs 7-9. In both cells
     the take-off comes once the LR passes about 1e-4.
  3. **The peak-LR stage did not saturate.** Val PGD-10 rose by 2.5pt over
     the last five peak-LR epochs (37.56% at ep 19 to 40.02% at ep 24), about
     0.5pt per epoch; clean rose about 0.8pt per epoch. The first decay then
     gave +4.3pt PGD-10 and +4.2pt clean in one epoch.
  4. **The two low-LR stages are nearly flat.** Val PGD-10 stays within
     44.34-46.18% over epochs 25-37 and 45.36-46.65% over epochs 38-49. The
     val SE is about 1.0pt (2,564 images), so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.70pt PGD-10
     (`robust_overfit_gap`), under one val SE. Best selection likely picked a
     noise peak.
  6. **The seen-unseen gap is larger than at 1.25e-4.** At the last epoch,
     probe minus val is +7.0 clean and +7.2 PGD-10, about 5x the SE of the
     difference (~1.4-1.5pt). The 1.25e-4 cell had +4.0 / +3.9.
  7. **2.5e-4 beats 1.25e-4 by far more than the noise.** Same init, only
     the LR differs. Best vs best: +10.3pt clean (66.61 vs 56.28) and +9.5pt
     PGD-10 (46.65 vs 37.13). Last vs last: +10.6 / +9.7. The SE of a
     difference between two val numbers is about 1.4pt. So far the
     random-init argmax is 2.5e-4, the middle of the grid; the 5e-4 cell
     decides whether it stays there. No verdict on the grid until that cell
     is done.

  Caveats: one seed. Non-deterministic mode with `torch.compile`, so a rerun
  would not be bit-identical. ImageNet-100 numbers are not comparable to the
  ImageNet-1k Phase 1 runs. Cost: wall clock 8.8 h (2026-09-28 21:13 to
  2026-09-29 06:04 UTC), 217 img/s after epoch 0 (193 img/s at epoch 0,
  compile warm-up), peak allocated memory 18.6 GB at epoch 0 and 14.5 GB
  after. Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `56a9e19b...`, `sample-stats-train.parquet`
  sha256 `7df25cab...` (from the run-bundle manifest). W&B
  `lightweight-imagenet-at-dev`, same run id.
- 2026-09-29 (postrun): **ImageNet-100 LR tuning, MobileViT-S AdamW
  pretrained / lr 1.25e-4 completed.** `plan0103-in100-tune-mobilevit-s-adamw-pretrained-lr1p25em4-v1`
  (Hamster, RTX 4090) ran 50/50 epochs. `completion.json` reads `completed`,
  the manifest status is `completed`, and `error-marker.txt` reads "no
  application error recorded". The watcher re-derived it as terminal and
  successful. All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` and `epoch-049.pt` are on disk. Nothing is
  imported into `docs/experiments/`: no aggregator exists for this contract,
  as for the other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  MobileViT-S (`mobilevit_s_imagenet`, `imagenet_raw_identity` profile),
  ImageNet-1k pretrained init (`pretrained: true`, 100-class head), plain
  PGD-AT, no teacher. Training and evaluation-attack seed 0 (split seed
  20260911). Training attack: CE PGD-3, l_inf eps 4/255, step 8/765, random
  start. Selection and validation attack: PGD-10, same eps, eval mode. AdamW
  (betas 0.9/0.999), peak LR 1.25e-4, wd 0.01, warmup_multistep (10-epoch
  warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1, global batch
  128, local BatchNorm. Non-deterministic with cuDNN autotuning,
  `torch.compile`, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_mobilevit_s_adamw_pretrained_lr1p25em4.yaml`, config hash
  `1ce70ed5...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation (the
  held-out 2% of the ImageNet-100 training split, 2,564 images), not the
  ImageNet-100 val split. "probe" is 2,000 training images, same transform,
  eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 34.48 / 0.55 | 34.70 / 0.50 | 4.662 |
  | 1 | warmup | 19.11 / 12.21 | 19.15 / 13.00 | 4.419 |
  | 9 | end of warmup | 79.76 / 51.76 | 82.45 / 55.65 | 2.144 |
  | 24 | end of lr 1.25e-4 | 84.98 / 58.97 | 89.70 / 66.80 | 1.682 |
  | 25 | first epoch at lr 1.25e-5 | 85.41 / 59.83 | 89.90 / 68.45 | 1.633 |
  | 37 | end of lr 1.25e-5 | 85.96 / 59.67 | 90.80 / 69.70 | 1.587 |
  | 38 | first epoch at lr 1.25e-6 | 85.61 / 59.36 | 90.05 / 69.95 | 1.587 |
  | 46 | best (by val PGD-10) | 85.57 / 60.65 | 90.40 / 70.35 | 1.582 |
  | 49 | last | 86.08 / 59.83 | 90.60 / 69.90 | 1.571 |

  Reading (n=1, two of three pretrained LR cells done, internal validation,
  PGD-10 only):
  1. **The run trains normally.** No collapse, no instability. Val clean ends
     at 86.1% on 100 classes (chance 1%).
  2. **The same early clean dip as at 6.25e-5, but shallower.** Val clean is
     34.5% after epoch 0 (LR 1.25e-5) and falls to 19.1% after epoch 1 while
     PGD-10 rises to 12.2%. At 6.25e-5 the dip went to 8.4%. From epoch 2
     both climb steadily.
  3. **The peak-LR stage saturates.** Val PGD-10 moved from 58.78% (ep 19)
     to 58.97% (ep 24), +0.19pt over five epochs, inside noise. The 6.25e-5
     cell was still rising about 0.26pt per epoch at that point. The first
     decay gave +0.86pt PGD-10 and +0.43pt clean.
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     59.36-60.26% over epochs 25-37 and 59.20-60.65% over epochs 38-49. The
     val SE is about 1.0pt (2,564 images), so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.82pt PGD-10
     (`robust_overfit_gap`), under one val SE. Best selection likely picked a
     noise peak.
  6. **The seen-unseen gap on the robust side grows with LR.** At the last
     epoch, probe minus val is +4.5 clean (about 5x the SE of the difference,
     ~0.9pt) and +10.1 PGD-10 (about 7x its SE, ~1.4pt). The 6.25e-5 cell had
     +3.4 / +6.8. The random-init side showed the same direction (1.25e-4:
     +3.9, 2.5e-4: +7.2 PGD-10).
  7. **1.25e-4 is ahead of 6.25e-5, but by a margin close to the noise.**
     Same init, only the LR differs. Best vs best: +1.95pt clean (85.57 vs
     83.62) and +2.77pt PGD-10 (60.65 vs 57.88). Last vs last: +1.84 / +2.65.
     The SE of a difference between two val numbers is about 1.4pt, so the
     PGD-10 lead is about 2 SE and the clean lead about 1.4 SE, from one seed
     each. So far the pretrained argmax is 1.25e-4, the middle of the grid;
     the 2.5e-4 cell decides whether it stays there. No verdict on the grid
     until that cell is done.

  Caveats: one seed. Non-deterministic mode with `torch.compile`, so a rerun
  would not be bit-identical. ImageNet-100 numbers are not comparable to the
  ImageNet-1k Phase 1 runs. Cost: wall clock 9.0 h (2026-09-28 21:20 to
  2026-09-29 06:18 UTC), 214 img/s after epoch 0 (191 img/s at epoch 0,
  compile warm-up), peak allocated memory 18.6 GB at epoch 0 and 14.5 GB
  after. Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `7e67c0b4...`, `sample-stats-train.parquet`
  sha256 `07dc187d...` (from the run-bundle manifest). W&B
  `lightweight-imagenet-at-dev`, same run id.
- 2026-09-29 (chat): **Option A verdict (preregistered): G = 70.95 - 70.86 =
  +0.09pt < 1pt.** The recipe itself saturates by 50 epochs; the saturation
  is not specific to adversarial training.
  - `plan0103-mobilenetv4-standard-100ep-v1` completed.
  - Clean training shows the same per-LR-stage plateau pattern.
  - The robust fit level still differs sharply: seen-set clean ~78% under
    clean training versus seen-set PGD-10 ~35% under adversarial training.
    The capacity account concerns the level of fit, not when it saturates.
- 2026-09-29 (chat): **What the ImageNet-100 AdamW tuning runs can and cannot
  be used for.** Internal-val PGD-10, LR low -> high:
  - ConvNeXt-Atto random 25.1 / 34.8 / 43.9, pretrained 58.9 / 60.5 / 60.4.
  - MobileViT-S random 36.3 / 45.9 / 50.0 (the last value at epoch 49),
    pretrained 57.2 / 59.8 / 61.8 (the last value at epoch 47).
  - DeiT-Tiny random 27.5 / 31.9 (still running).

  **Not usable** to pick the ImageNet-1k LR (stage-3 rule failed). Random
  init prefers the top of the grid for every model, the same bias as the
  proxy. ImageNet-100 also sits in a different regime: pretrained
  ConvNeXt-Atto shows a ~20pt seen-vs-unseen PGD gap (overfitting), while
  ImageNet-1k runs underfit.

  **Usable**, directionally:
  1. AdamW removes the collapse. ConvNeXt-Atto random init fits the seen
     set (probe PGD 55%) where SGD on ImageNet-1k collapsed (seen clean
     31%).
  2. Pretraining helps, with the sign validated by stage 3:
     best-vs-best +16.6 ConvNeXt-Atto, +11.8 MobileViT-S, +4.6
     MobileNetV4-S. The magnitude may be inflated because the random-side
     argmax sits at the grid edge.
  3. Heuristic only: the ImageNet-100 argmax is a likely upper bound for the
     ImageNet-1k optimum, useful for placing ImageNet-1k grids. It rests on
     one SGD model and is not paper evidence.

  A step-matched proxy (ImageNet-100 x 500 epochs) would cost the same as
  ImageNet-1k, so it saves nothing.
- 2026-09-29 (postrun): **ImageNet-100 LR tuning, MobileViT-S AdamW random
  init / lr 5e-4 completed.** `plan0103-in100-tune-mobilevit-s-adamw-random-lr5em4-v1`
  (Hamster, RTX 4090) ran 50/50 epochs. `completion.json` reads `completed`,
  the manifest status is `completed`, and `error-marker.txt` reads "no
  application error recorded". The watcher re-derived it as terminal and
  successful. All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` and `epoch-049.pt` are on disk. Nothing is
  imported into `docs/experiments/`: no aggregator exists for this contract,
  as for the other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  MobileViT-S (`mobilevit_s_imagenet`, `imagenet_raw_identity` profile),
  random init, plain PGD-AT, no teacher. Training and evaluation-attack seed 0
  (split seed 20260911). Training attack: CE PGD-3, l_inf eps 4/255, step
  8/765, random start. Selection and validation attack: PGD-10, same eps, eval
  mode. AdamW (betas 0.9/0.999), peak LR 5e-4, wd 0.01, warmup_multistep
  (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1,
  global batch 128, local BatchNorm. Non-deterministic with cuDNN autotuning,
  `torch.compile`, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_mobilevit_s_adamw_random_lr5em4.yaml`, config hash
  `5f9dd05b...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation (the
  held-out 2% of the ImageNet-100 training split, 2,564 images), not the
  ImageNet-100 val split. "probe" is 2,000 training images, same transform,
  eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 4.29 / 3.08 | 3.40 / 2.95 | 4.562 |
  | 4 | warmup | 23.79 / 15.80 | 25.65 / 16.20 | 3.871 |
  | 9 | end of warmup | 44.70 / 28.98 | 48.15 / 32.25 | 3.163 |
  | 24 | end of lr 5e-4 | 64.47 / 43.88 | 70.85 / 49.60 | 2.383 |
  | 25 | first epoch at lr 5e-5 | 69.34 / 48.71 | 76.05 / 55.70 | 2.229 |
  | 37 | end of lr 5e-5 | 71.96 / 50.08 | 78.25 / 58.50 | 2.096 |
  | 38 | first epoch at lr 5e-6 | 71.41 / 49.69 | 78.40 / 58.35 | 2.080 |
  | 42 | best (by val PGD-10) | 71.92 / 50.51 | 78.65 / 58.50 | 2.069 |
  | 49 | last | 72.00 / 50.16 | 79.20 / 59.50 | 2.055 |

  Reading (n=1, all three random-init LR cells done, internal validation,
  PGD-10 only):
  1. **The run trains normally.** No collapse, no instability at the highest
     LR of the grid. Val clean ends at 72.0% on 100 classes (chance 1%).
  2. **The warmup start is the fastest of the three cells.** Val clean
     climbs from epoch 2 (LR 1.5e-4: 9.3%, then 15.6% and 23.8%). The 2.5e-4
     cell took off at epoch 5 (also LR 1.5e-4), the 1.25e-4 cell at epochs
     7-9. In all three cells the take-off comes once the LR passes about
     1e-4.
  3. **The peak-LR stage did not saturate.** Val PGD-10 rose 2.5pt over the
     last five peak-LR epochs (41.34% at ep 19 to 43.88% at ep 24), about
     0.5pt per epoch; clean rose about 0.4pt per epoch. The first decay then
     gave +4.8pt PGD-10 and +4.9pt clean in one epoch.
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     48.67-50.08% over epochs 25-37 and 49.26-50.51% over epochs 38-49. The
     val SE is about 1.0pt (2,564 images), so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.35pt PGD-10
     (`robust_overfit_gap`), well under one val SE.
  6. **The seen-unseen gap keeps growing with LR.** At the last epoch, probe
     minus val is +7.2 clean and +9.3 PGD-10. The 2.5e-4 cell had +7.0 /
     +7.2, the 1.25e-4 cell +4.0 / +3.9.
  7. **5e-4 beats 2.5e-4 by more than the noise.** Same init, only the LR
     differs. Best vs best: +5.3pt clean (71.92 vs 66.61) and +3.9pt PGD-10
     (50.51 vs 46.65). Last vs last: +5.5 / +4.2. The SE of a difference
     between two val numbers is about 1.3pt (clean) and 1.4pt (PGD-10), so
     the lead is about 4 SE on clean and 2.8 SE on PGD-10, one seed each.
  8. **Grid result: the random-init argmax is 5e-4, the top edge of the
     grid.** Internal-val PGD-10 best by LR: 37.13 / 46.65 / 50.51. The gain
     per doubling shrinks (+9.5pt, then +3.9pt), but the optimum on this
     proxy may lie above 5e-4. This matches the 2026-09-29 (chat) entry
     above: random init prefers the top of the grid for every model. Under
     the A2 verdict (discussion register, 2026-09-29) this argmax is not
     used to pick the ImageNet-1k LR; it only feeds the pretrained vs
     random-init sign comparison.

  Caveats: one seed. Non-deterministic mode with `torch.compile`, so a rerun
  would not be bit-identical. ImageNet-100 numbers are not comparable to the
  ImageNet-1k Phase 1 runs. Cost: wall clock 8.8 h (2026-09-29 06:04 to
  14:55 UTC), 217 img/s after epoch 0 (193 img/s at epoch 0, compile
  warm-up), peak allocated memory 18.6 GB at epoch 0 and 14.5 GB after.
  Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `c17a33ac...`, `sample-stats-train.parquet`
  sha256 `dfba74b4...` (from the run-bundle manifest). W&B
  `lightweight-imagenet-at-dev`, same run id.
- 2026-09-30 (postrun): **ImageNet-100 LR tuning, MobileViT-S AdamW
  pretrained / lr 2.5e-4 completed.** `plan0103-in100-tune-mobilevit-s-adamw-pretrained-lr2p5em4-v1`
  (Hamster, RTX 4090) ran 50/50 epochs. `completion.json` reads `completed`,
  the manifest status is `completed`, and `error-marker.txt` reads "no
  application error recorded". The watcher re-derived it as terminal and
  successful. All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` and `epoch-049.pt` are on disk. Nothing is
  imported into `docs/experiments/`: no aggregator exists for this contract,
  as for the other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  MobileViT-S (`mobilevit_s_imagenet`, `imagenet_raw_identity` profile),
  ImageNet-1k pretrained init (`pretrained: true`, 100-class head), plain
  PGD-AT, no teacher. Training and evaluation-attack seed 0 (split seed
  20260911). Training attack: CE PGD-3, l_inf eps 4/255, step 8/765, random
  start. Selection and validation attack: PGD-10, same eps, eval mode. AdamW
  (betas 0.9/0.999), peak LR 2.5e-4, wd 0.01, warmup_multistep (10-epoch
  warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1, global batch
  128, local BatchNorm. Non-deterministic with cuDNN autotuning,
  `torch.compile`, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_mobilevit_s_adamw_pretrained_lr2p5em4.yaml`, config hash
  `618b1df0...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation (the
  held-out 2% of the ImageNet-100 training split, 2,564 images), not the
  ImageNet-100 val split. "probe" is 2,000 training images, same transform,
  eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 9.32 / 5.97 | 9.05 / 5.45 | 4.615 |
  | 1 | warmup | 42.16 / 25.51 | 43.75 / 27.35 | 3.848 |
  | 9 | end of warmup | 81.28 / 53.90 | 85.00 / 61.15 | 1.963 |
  | 24 | end of lr 2.5e-4 | 85.18 / 60.45 | 90.85 / 70.00 | 1.558 |
  | 25 | first epoch at lr 2.5e-5 | 86.43 / 62.29 | 91.75 / 73.45 | 1.469 |
  | 35 | best (by val PGD-10) | 87.21 / 62.56 | 92.55 / 75.75 | 1.401 |
  | 37 | end of lr 2.5e-5 | 87.52 / 61.66 | 92.75 / 74.65 | 1.396 |
  | 38 | first epoch at lr 2.5e-6 | 86.90 / 61.86 | 92.75 / 75.85 | 1.392 |
  | 49 | last | 87.44 / 61.66 | 93.00 / 75.50 | 1.372 |

  Reading (n=1, all three pretrained LR cells done, internal validation,
  PGD-10 only):
  1. **The run trains normally.** No collapse, no instability at the highest
     LR of the grid. Val clean ends at 87.4% on 100 classes (chance 1%).
  2. **The early clean dip falls in epoch 0 here.** Val clean is 9.3% after
     epoch 0 (LR 2.5e-5), then 42.2% after epoch 1 and 81.3% at the end of
     warmup. In the other two pretrained cells the lowest point came after
     epoch 1 (6.25e-5: 8.4%, 1.25e-4: 19.1%). No pre-training val point is
     logged, so the true bottom of the dip is not observed.
  3. **The peak-LR stage is near its plateau.** Val PGD-10 moved from 58.97%
     (ep 19) to 60.45% (ep 24), +1.48pt over five epochs, about 0.3pt per
     epoch, each step below the val SE. The first decay then gave +1.84pt
     PGD-10 and +1.25pt clean in one epoch (1.25e-4: +0.86 / +0.43).
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     60.96-62.56% over epochs 25-37 and 61.23-62.25% over epochs 38-49. The
     val SE is about 1.0pt (2,564 images), so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.90pt PGD-10
     (`robust_overfit_gap`), under one val SE. The best epoch (35) lies in
     the flat middle stage, so best selection likely picked a noise peak.
  6. **The seen-unseen gap keeps growing with LR.** At the last epoch, probe
     minus val is +5.6 clean and +13.8 PGD-10. The 1.25e-4 cell had +4.5 /
     +10.1, the 6.25e-5 cell +3.4 / +6.8.
  7. **2.5e-4 is ahead of 1.25e-4, but inside two SE.** Same init, only the
     LR differs. Best vs best: +1.64pt clean (87.21 vs 85.57) and +1.91pt
     PGD-10 (62.56 vs 60.65). Last vs last: +1.36 / +1.83. The SE of a
     difference between two val numbers is about 0.95pt (clean, near 87%)
     and 1.4pt (PGD-10), so the lead is about 1.7 SE on clean and 1.4 SE on
     PGD-10 (1.3 SE last vs last), one seed each. This run alone does not
     separate the top two cells.
  8. **Grid result: the pretrained argmax is 2.5e-4, the top edge of the
     grid.** Internal-val PGD-10 best by LR (6.25e-5 / 1.25e-4 / 2.5e-4):
     57.88 / 60.65 / 62.56; last 57.18 / 59.83 / 61.66; clean best 83.62 /
     85.57 / 87.21. The ordering is monotone, the gain per doubling shrinks
     (+2.77pt, then +1.91pt), and the optimum on this proxy may lie above
     2.5e-4. Both MobileViT-S grids (random and pretrained) now put the
     argmax at the top edge. Under the A2 verdict (discussion register,
     2026-09-29) this argmax is not used to pick the ImageNet-1k LR.
  9. **The pretraining sign holds for MobileViT-S.** Best of each grid:
     pretrained 2.5e-4 vs random 5e-4 gives +12.05pt PGD-10 (62.56 vs
     50.51) and +15.29pt clean (87.21 vs 71.92). The 2026-09-29 (chat)
     entry's +11.8 used the in-progress epoch-47 value. The magnitude
     carries the same caveat as there: both argmaxes sit at the grid edge.

  Caveats: one seed. Non-deterministic mode with `torch.compile`, so a rerun
  would not be bit-identical. ImageNet-100 numbers are not comparable to the
  ImageNet-1k Phase 1 runs. Cost: wall clock 9.0 h (2026-09-29 06:18 to
  15:16 UTC), 214 img/s after epoch 0 (191 img/s at epoch 0, compile
  warm-up), peak allocated memory 18.6 GB at epoch 0 and 14.5 GB after.
  Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `73187e0e...`, `sample-stats-train.parquet`
  sha256 `a7796fa2...` (from the run-bundle manifest). W&B
  `lightweight-imagenet-at-dev`, same run id.
- 2026-09-30 (human, chat): **AdvXL-inspired two-stage pilot on MobileNetV4-S,
  compute-matched to the single-stage 50-epoch run.**
  - **AdvXL facts.** Title "Revisiting Adversarial Training at Scale",
    CVPR 2024. The paper has no appendix or hyperparameter table, and the
    repo is evaluation-only.
    - Stated: stage 1 at 112px (anti-aliased bilinear), PGD-1 step 4/255,
      200 ep, AdamW, warmup + cosine, RandAug + MixUp + CutMix; stage 2 at
      224px, PGD-3 step 4/255, 20 ep.
    - ViT-B results: 73.0 / 52.5 at 0.25x compute versus 75.5 / 54.5
      single-stage at 224. Stage 1 alone gives 68.5 / 39.3.
    - Only >=86M-parameter LayerNorm models are covered. Nothing is said
      about catastrophic overfitting.
  - **Compute matching.** Forward FLOPs were measured; torch
    FlopCounterMode overcounts depthwise-conv backward by ~groups x, so
    backward is taken as 2x forward. A 224/PGD-3 step is 9F; a 112/PGD-1
    step is 5 x 0.277F. The ratio is 0.154, so stage 1 gets
    (50 - 20) / 0.154 = 195 epochs.
  - **Design** (the cosine-vs-multistep schedule is deferred as a later
    search candidate).
    - Stage 1: 112px, PGD-1 (eps 4/255, step 4/255, random start), SGD lr
      0.025, warmup_multistep [98, 148], basic augmentation.
    - Stage 2: 224px, PGD-3 with OUR step 8/765 (so it matches the
      reference attack; AdvXL uses 4/255), 20 ep from stage-1 last.pt, lr
      0.0025, 2-ep warmup, [10, 15].
    - Both stages random init, deterministic, seed 0.
  - **Preregistered rule.** Compare last-epoch internal-val PGD-10 after
    stage 2 with the single-stage random-init reference, 30.71 (lr 0.025).
    >= +1pt: two-stage better at equal compute. Within +-1pt: equivalent;
    choose by wall-clock and simplicity. <= -1pt: worse. Clean is reported
    alongside.
  - **Catastrophic-overfitting guard.** Stop and report if stage-1
    val PGD-10 falls below half of its running maximum.
  - Implementation is in progress (train resolution separate from
    evaluation; init from our own checkpoint with sha256 lineage), with
    review before use.
- 2026-09-30 — Two-stage pilot, stage 1 launched (hand-run).
  - Code merged to master as dc0908d + 6d354d5 (review fixes). The
    two-stage and related tests pass (252 passed, 3 skipped).
  - Run id `plan0103-mnv4s-twostage-stage1-112-pgd1-s0-v1`, launched
    from pinned worktree `source-6d354d5a6d1f`.
  - Config `imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml`: 112px
    training, PGD-1, 195 epochs, batch 128, seed 0.
  - Hamster GPU0, 16 workers, W&B project `lightweight-imagenet-at`.
  - Stage 2 is launched from this run's final-epoch `last.pt`, with its
    sha256.
- 2026-09-30 — Stage 1 v1 stopped; relaunched as v2 (human-approved).
  - Why: v1 was loader-bound. Measured CPU read throughput was ~250 MB/s
    (~2300 img/s), with GPU0 at ~35-40%. That implied ~9 min per epoch,
    or ~30 h for 195 epochs, versus ~13 h if compute-bound.
  - v1 (16 workers) was stopped mid-epoch 0 and marked ABANDONED. Its
    W&B run is kept.
  - v2 `plan0103-mnv4s-twostage-stage1-112-pgd1-s0-v2` uses the same
    source (6d354d5), config and seed, with ARD_NUM_WORKERS=32 (option A).
    The preregistered comparison uses v2, on original images.
  - Options B and C were approved. They are being implemented off-master
    and will be reviewed before any use.
    - B: reduced-size JPEG decode (PIL draft), default-off.
    - C: pre-resized derived train copies (short side 160 and 256), with
      a `derived_from` dataset identity and an init-checkpoint rule.
  - B and C change training pixels, so any run using them is a separate,
    disclosed preprocessing condition.
- 2026-09-30 — Hosts reshuffled (human-approved).
  - Stage 1 v2 stopped (Hamster GPU0) and marked ABANDONED. It was slowed
    by CPU contention with the option-C dataset build. It will be replaced
    by a reviewed faster-loader variant, to run on Ferret, which has twice
    Hamster's CPU threads (104 vs 52), with the same CPU model.
  - Ferret GPU0 queue stopped. The running DeiT-Tiny IN-100
    `random-lr5em4-v1` continues to completion.
  - The three remaining DeiT-Tiny IN-100 pretrained items (lr 6.25e-5,
    1.25e-4, 2.5e-4) were moved to Hamster GPU0. They run from the same
    worktree source-5a5dcbaabf0f, keep the same run ids, and go to W&B
    project `lightweight-imagenet-at-dev`.
- 2026-09-30 — Loader speedups B and C merged: bad0dfa, c0a1431, and
  9ad9d31 (scientific-review fixes).
  - Single-process benchmark: decode + 112 training view, img/s,
    2000 images.

    | root     | img/s, draft off | img/s, draft on | crops upsampled to 112 |
    |----------|------------------|-----------------|------------------------|
    | original | 303              | 381             | 5.9%                   |
    | S=256    | 650              | 666             | 16.0%                  |
    | S=160    | 920              | 905             | 51.5%                  |

  - RandomResizedCrop Monte Carlo on a 500x375 image; share of crops
    upsampled in at least one dimension / in both dimensions:
    original 0.2%/0.0%, S=256 12.5%/6.5%, S=160 49.8%/34.6%.
  - Draft decode changes pixels by a mean absolute 0.74 (0-255 scale) on
    original data.
  - Derived roots (built with Q=95 and LANCZOS):
    - `imagenet_train_s160`: content f52345d8…, manifest 2bf01d09….
    - `imagenet_train_s256`: content 65f729c7…, manifest 1b878f7a….
  - Configs: `..._stage1_112_pgd1_resized_s256.yaml` and
    `..._resized_s160.yaml`, each with its own W&B group.
  - Stage 2 refuses to run on derived data.
  - Recommendation: use S=256 with draft off. Awaiting the human's choice
    of S.
  - The S=256 copy is being transferred to Ferret. Ferret worktree
    source-9ad9d318deee is pinned.
- 2026-09-30 (postrun): **ImageNet-100 LR tuning, DeiT-Tiny AdamW
  pretrained / lr 6.25e-5 completed.** `plan0103-in100-tune-deit-tiny-adamw-pretrained-lr6p25em5-v1`
  (Hamster GPU0, RTX 4090) ran 50/50 epochs. `run-bundle/completion.json`
  reads `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 50 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt`, `last.pt`
  and `epoch-049.pt` are on disk. Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract, as for the
  other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-100 proxy (100 classes, pinned train/val hashes),
  DeiT-Tiny (`deit_tiny_imagenet`, `imagenet_standard` profile), ImageNet-1k
  pretrained init (`pretrained: true`, head re-initialised to 100 classes),
  plain PGD-AT, no teacher. Training and evaluation-attack seed 0 (split
  seed 20260911). Training attack: CE PGD-3, l_inf eps 4/255, step 8/765,
  random start. Selection and validation attack: PGD-10, same eps, eval
  mode. AdamW (betas 0.9/0.999), peak LR 6.25e-5, wd 0.05, warmup_multistep
  (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1,
  global batch 128. Non-deterministic with cuDNN autotuning, not compiled,
  step_diagnostics off, fp32. Protocol `controlled_imagenet100_proxy_lr_v1`,
  config `imagenet100_deit_tiny_adamw_pretrained_lr6p25em5.yaml`, config
  hash `d8715612...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation
  (the held-out 2% of the ImageNet-100 training split, 2,564 images), not
  the ImageNet-100 val split. "probe" is 2,000 training images, same
  transform, eval mode and PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 1.01 / 0.98 | 1.00 / 0.90 | 4.800 |
  | 1 | warmup | 1.72 / 1.48 | 1.65 / 1.40 | 4.607 |
  | 4 | warmup | 24.77 / 13.92 | 26.70 / 14.60 | 4.021 |
  | 9 | end of warmup | 71.61 / 43.10 | 74.65 / 46.80 | 2.534 |
  | 24 | end of lr 6.25e-5 | 80.58 / 53.63 | 86.40 / 62.95 | 1.868 |
  | 25 | first epoch at lr 6.25e-6 | 81.86 / 55.54 | 87.05 / 65.85 | 1.757 |
  | 36 | best (by val PGD-10) | 82.37 / 56.51 | 87.90 / 67.20 | 1.684 |
  | 37 | end of lr 6.25e-6 | 82.64 / 55.89 | 88.20 / 67.55 | 1.674 |
  | 38 | first epoch at lr 6.25e-7 | 82.72 / 55.81 | 88.20 / 68.15 | 1.665 |
  | 49 | last | 82.64 / 55.93 | 88.35 / 67.95 | 1.647 |

  Reading (n=1, first of three DeiT-Tiny pretrained LR cells, internal
  validation, PGD-10 only):
  1. **The run trains normally after a slow start.** No instability at the
     end. Val clean ends at 82.6% on 100 classes (chance 1%).
  2. **The early dip reaches chance level and lasts longer than in
     MobileViT-S.** Val clean is 1.01% after epoch 0 (LR 6.25e-6) and 1.72%
     after epoch 1. Train loss on adversarial inputs is 4.80 after epoch 0,
     above ln(100) = 4.61, the loss of a uniform guess. Val clean then
     climbs to 24.8% at epoch 4 and 71.6% at the end of warmup. In the
     MobileViT-S pretrained cell at the same LR, the lowest point was 8.4%
     after epoch 1. No pre-training val point is logged, so the state at
     step 0 is not observed. The final level (below) is far above the
     DeiT-Tiny random-init cells, so the pretrained weights do carry over.
     The next two cells show whether the depth of the dip depends on LR.
  3. **The peak-LR stage is close to its plateau.** Val PGD-10 moved from
     52.85% (ep 19) to 53.63% (ep 24), +0.78pt over five epochs, below one
     val SE. The first decay then gave +1.91pt PGD-10 and +1.29pt clean in
     one epoch.
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     55.27-56.51% over epochs 25-37 and 55.81-56.20% over epochs 38-49. The
     val SE is about 1.0pt for PGD-10 and 0.75pt for clean (2,564 images),
     so these moves are at noise level.
  5. **Best and last are within noise.** Best minus last is 0.59pt PGD-10
     (`robust_overfit_gap`), under one val SE. The best epoch (36) lies in
     the flat middle stage, so best selection likely picked a noise peak.
  6. **Seen-unseen gap.** At the last epoch, probe minus val is +5.7 clean
     and +12.0 PGD-10. The MobileViT-S pretrained cell at the same LR had
     +3.4 / +6.8.
  7. **No grid verdict yet.** Best 82.37 / 56.51, last 82.64 / 55.93. The
     1.25e-4 and 2.5e-4 cells decide the DeiT-Tiny pretrained argmax. The
     pretrained-vs-random comparison for DeiT-Tiny waits for the recorded
     random-init grid (the 2026-09-29 chat values 27.5 / 31.9 were
     in-progress numbers).

  Caveats: one seed. Non-deterministic mode, so a rerun would not be
  bit-identical. ImageNet-100 numbers are not comparable to the ImageNet-1k
  Phase 1 runs. Cost: wall clock 3.5 h (2026-09-29 17:10 to 20:38 UTC),
  542 img/s at epoch 0 and 556 img/s at the last epoch, peak allocated
  memory 4.6 GB. Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `d7ffc681...`, `sample-stats-train.parquet`
  sha256 `3a8090ad...` (from the run-bundle manifest). W&B
  `lightweight-imagenet-at-dev`, same run id.
- 2026-09-30 — Throughput diagnosis for stage 1 (112 px, batch 128),
  measured with throwaway scripts.
  - Loader alone (training view, img/s):
    - Hamster, original data: 4348 / 5406 / 5703 at 16 / 32 / 48 workers.
    - Hamster, S=256: 8722 / 10489 / 11515 at the same worker counts.
    - Ferret, original data: 3099-4180 at 16-64 workers.
    - Ferret, S=256: 6141-8755 at 16-64 workers. Ferret is not faster,
      because it shares the host with other training jobs.
  - GPU-only synthetic step (deterministic, img/s): 3860 at bs128 and
    6669 at bs256; 4994 at bs128 non-deterministic. For reference,
    224 px PGD-3 at bs128 runs at 1319.
  - So at 112 px and bs128 the step is kernel-launch bound (the
    CPU-thread limit), not GPU-compute bound.
  - Human decision:
    - Run stage 1 with S=256 and `training.step_diagnostics: false`,
      which removes per-step host syncs without changing the math
      (sync-free parity-tested). Config changed in 8333698.
    - Batch 256, non-deterministic mode and compile are not used for
      this comparison.
    - Explore CUDA Graphs separately, on Ferret.
  - Launched `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-s0-v1`:
    Hamster GPU1, pinned worktree source-8333698a3de1, 16 workers,
    seed 0.
- 2026-09-30 (postrun): **ImageNet-100 LR tuning, DeiT-Tiny AdamW
  pretrained / lr 1.25e-4 completed.** `plan0103-in100-tune-deit-tiny-adamw-pretrained-lr1p25em4-v1`
  (Hamster GPU0, RTX 4090) ran 50/50 epochs. `run-bundle/completion.json`
  reads `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 50 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt`, `last.pt`
  and `epoch-049.pt` are on disk. Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract, as for the
  other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: identical to the lr 6.25e-5 cell above except the peak
  LR. ImageNet-100 proxy (100 classes, pinned train/val hashes), DeiT-Tiny
  (`deit_tiny_imagenet`, `imagenet_standard` profile), ImageNet-1k
  pretrained init (`pretrained: true`, head re-initialised to 100 classes),
  plain PGD-AT, no teacher. Training and evaluation-attack seed 0 (split
  seed 20260911). Training attack: CE PGD-3, l_inf eps 4/255, step 8/765,
  random start. Selection and validation attack: PGD-10, same eps, eval
  mode. AdamW (betas 0.9/0.999), peak LR 1.25e-4, wd 0.05, warmup_multistep
  (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs. World size 1,
  global batch 128. Non-deterministic with cuDNN autotuning, not compiled,
  step_diagnostics off, fp32. Protocol `controlled_imagenet100_proxy_lr_v1`,
  config `imagenet100_deit_tiny_adamw_pretrained_lr1p25em4.yaml`, config
  hash `54b304cf...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation
  (2,564 held-out training images), "probe" is 2,000 training images, both
  in eval mode with PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 1.52 / 0.78 | 1.80 / 1.00 | 4.714 |
  | 1 | warmup | 5.77 / 4.52 | 5.40 / 4.25 | 4.517 |
  | 4 | warmup | 59.32 / 33.81 | 61.25 / 35.95 | 3.196 |
  | 9 | end of warmup | 75.62 / 48.05 | 79.60 / 53.00 | 2.252 |
  | 24 | end of lr 1.25e-4 | 81.32 / 54.06 | 88.80 / 67.45 | 1.742 |
  | 25 | first epoch at lr 1.25e-5 | 82.64 / 57.33 | 90.15 / 72.00 | 1.570 |
  | 29 | best (by val PGD-10) | 83.07 / 57.96 | 91.20 / 72.80 | 1.486 |
  | 37 | end of lr 1.25e-5 | 83.62 / 57.61 | 92.15 / 73.70 | 1.437 |
  | 38 | first epoch at lr 1.25e-6 | 83.78 / 57.18 | 92.25 / 74.20 | 1.420 |
  | 49 | last | 83.81 / 57.45 | 92.25 / 74.90 | 1.397 |

  Reading (n=1, second of three DeiT-Tiny pretrained LR cells, internal
  validation, PGD-10 only):
  1. **The run trains normally.** No instability at the end. Val clean
     ends at 83.8% on 100 classes (chance 1%).
  2. **The early dip still reaches near chance, but recovery is faster.**
     Val clean is 1.52% after epoch 0 (LR 1.25e-5) and train loss 4.71,
     still above ln(100) = 4.61. After epoch 1 val clean is 5.77% (6.25e-5
     cell: 1.72%), at epoch 4 59.3% (24.8%), at the end of warmup 75.6%
     (71.6%). So doubling the LR did not make the first-epoch dip
     shallower, but it shortened it. No pre-training val point is logged.
  3. **The peak-LR stage is still rising slowly at its end.** Val PGD-10
     moved from 52.81% (ep 19) to 54.06% (ep 24), +1.25pt over five epochs,
     about one val SE (6.25e-5 cell: +0.78pt). The first decay then gave
     +3.28pt PGD-10 and +1.33pt clean in one epoch (6.25e-5: +1.91 / +1.29).
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     56.71-57.96% over epochs 25-37 and 57.18-57.64% over epochs 38-49,
     inside the val SE (~1.0pt PGD-10, ~0.75pt clean).
  5. **Best and last are within noise.** Best minus last is 0.51pt PGD-10
     (`robust_overfit_gap`), under one val SE. The best epoch (29) lies in
     the flat middle stage.
  6. **Against lr 6.25e-5.** Best vs best: +0.70pt clean, +1.45pt PGD-10.
     Last vs last: +1.17 / +1.52. The SE of a difference between two
     unpaired cells is about 1.4pt PGD-10 and 1.05pt clean, so this is about
     one SE on PGD-10, one seed each. Direction favours 1.25e-4; size is at
     noise level.
  7. **The seen-unseen gap grows with LR.** At the last epoch, probe minus
     val is +8.4 clean and +17.5 PGD-10 (6.25e-5: +5.7 / +12.0). The same
     direction was seen in MobileViT-S (1.25e-4: +4.5 / +10.1), and the
     DeiT-Tiny gap is larger at both LRs.
  8. **No grid verdict yet.** Argmax so far is 1.25e-4 (grid middle). The
     2.5e-4 cell decides the DeiT-Tiny pretrained argmax. Pretrained vs
     random for DeiT-Tiny still waits for the recorded random-init grid.

  Caveats: one seed. Non-deterministic mode, so a rerun would not be
  bit-identical. ImageNet-100 numbers are not comparable to the ImageNet-1k
  Phase 1 runs. Cost: wall clock 3.5 h (2026-09-29 20:38 to 2026-09-30
  00:07 UTC), 549 img/s at epoch 0 and 556 img/s at the last epoch, peak
  allocated memory 4.6 GB. Checkpoint sha256 was not computed in this
  postrun. `epoch-metrics.parquet` sha256 `24e26827...`,
  `sample-stats-train.parquet` sha256 `53060009...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at-dev`, same run id.
- 2026-09-30 (postrun): **ImageNet-100 LR tuning, DeiT-Tiny AdamW
  pretrained / lr 2.5e-4 completed.** `plan0103-in100-tune-deit-tiny-adamw-pretrained-lr2p5em4-v1`
  (Hamster GPU0, RTX 4090) ran 50/50 epochs. `run-bundle/completion.json`
  reads `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 50 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt`, `last.pt`
  and `epoch-049.pt` are on disk. Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract, as for the
  other plan 0103/0104 hand-runs. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: identical to the lr 6.25e-5 and 1.25e-4 cells above
  except the peak LR. ImageNet-100 proxy (100 classes, pinned train/val
  hashes), DeiT-Tiny (`deit_tiny_imagenet`, `imagenet_standard` profile),
  ImageNet-1k pretrained init (`pretrained: true`, head re-initialised to
  100 classes), plain PGD-AT, no teacher. Training and evaluation-attack
  seed 0 (split seed 20260911). Training attack: CE PGD-3, l_inf eps 4/255,
  step 8/765, random start. Selection and validation attack: PGD-10, same
  eps, eval mode. AdamW (betas 0.9/0.999), peak LR 2.5e-4, wd 0.05,
  warmup_multistep (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs.
  World size 1, global batch 128. Non-deterministic with cuDNN autotuning,
  not compiled, step_diagnostics off, fp32. Protocol
  `controlled_imagenet100_proxy_lr_v1`, config
  `imagenet100_deit_tiny_adamw_pretrained_lr2p5em4.yaml`, config hash
  `d3157b12...`. Source SHA `5a5dcbaabf0fed62a0c97dc239f11c1365352b6d`
  (worktree `source-5a5dcbaabf0f`, clean). "val" is internal validation
  (2,564 held-out training images), "probe" is 2,000 training images, both
  in eval mode with PGD-10. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 3.04 / 2.34 | 3.30 / 2.25 | 4.645 |
  | 1 | warmup | 13.53 / 8.50 | 14.60 / 9.45 | 4.312 |
  | 4 | warmup | 69.23 / 40.95 | 72.35 / 43.70 | 2.653 |
  | 9 | end of warmup | 75.66 / 48.60 | 80.40 / 54.40 | 2.194 |
  | 19 | one-epoch drop | 77.69 / 47.50 | 85.85 / 57.90 | 1.853 |
  | 24 | end of lr 2.5e-4 | 79.76 / 51.99 | 88.95 / 66.75 | 1.750 |
  | 25 | first epoch at lr 2.5e-5 | 82.45 / 57.72 | 92.00 / 74.75 | 1.501 |
  | 34 | best (by val PGD-10) | 83.31 / 58.66 | 94.25 / 78.45 | 1.322 |
  | 37 | end of lr 2.5e-5 | 83.46 / 58.03 | 94.80 / 79.15 | 1.298 |
  | 38 | first epoch at lr 2.5e-6 | 83.39 / 58.23 | 95.00 / 79.75 | 1.268 |
  | 49 | last | 83.46 / 58.11 | 95.25 / 80.20 | 1.236 |

  Reading (n=1, third of three DeiT-Tiny pretrained LR cells, internal
  validation, PGD-10 only):
  1. **The run trains normally.** No instability at the end. Val clean
     ends at 83.5% on 100 classes (chance 1%).
  2. **The slow start shortens again with LR.** Val clean is 3.04% after
     epoch 0 (LR 2.5e-5; 1.25e-4 cell: 1.52%, 6.25e-5 cell: 1.01%) and
     train loss 4.645, still just above ln(100) = 4.61. After epoch 1 val
     clean is 13.5% (5.8%, 1.7%), at epoch 4 69.2% (59.3%, 24.8%), at the
     end of warmup 75.7% (75.6%, 71.6%). The classifier head is
     re-initialised, so the model starts at chance by construction; what
     changes with LR is how fast it leaves chance.
  3. **The peak-LR stage reached its plateau, with larger swings.** Val
     PGD-10 peaked at 53.43% at epoch 17 and ended the stage at 51.99%
     (ep 24); epochs 20-24 stay within 51.48-53.32%. Epoch 19 dropped to
     47.50% (-4.41pt from ep 18) and recovered the next epoch; probe PGD-10
     dropped at the same epoch (63.6% to 57.9%), so it is a real one-epoch
     dip, not val noise. The first decay then gave +5.73pt PGD-10 and
     +2.69pt clean in one epoch (1.25e-4: +3.28 / +1.33; 6.25e-5:
     +1.91 / +1.29).
  4. **The two low-LR stages are flat.** Val PGD-10 stays within
     57.57-58.66% over epochs 25-37 and 57.92-58.39% over epochs 38-49,
     inside the val SE (~1.0pt PGD-10, ~0.75pt clean).
  5. **Best and last are within noise.** Best minus last is 0.55pt PGD-10
     (`robust_overfit_gap`), under one val SE. The best epoch (34) lies in
     the flat middle stage.
  6. **Against lr 1.25e-4.** Best vs best: +0.24pt clean, +0.70pt PGD-10.
     Last vs last: -0.35 / +0.66. The SE of a difference between two
     unpaired cells is about 1.4pt PGD-10 and 1.05pt clean, so this is
     about half an SE, one seed each: no resolvable difference. Against
     6.25e-5, best vs best is +0.94 / +2.15 (about 1.5 SE on PGD-10).
  7. **The seen-unseen gap keeps growing with LR.** At the last epoch,
     probe minus val is +11.8 clean and +22.1 PGD-10 (1.25e-4: +8.4 /
     +17.5; 6.25e-5: +5.7 / +12.0). Over the last 25 epochs probe PGD-10
     keeps rising (74.75% to 80.20%) while val PGD-10 stays flat, but val
     does not fall either.
  8. **Grid result (DeiT-Tiny pretrained).** Best val PGD-10 56.51 / 57.96
     / 58.66 over 6.25e-5 / 1.25e-4 / 2.5e-4. Argmax 2.5e-4 sits at the top
     edge of the grid, as for MobileViT-S pretrained, and the top two cells
     are within one SE of the difference. Per the A2 verdict this grid is
     not used to pick the ImageNet-1k LR. Pretrained vs random for DeiT-Tiny
     still waits for the recorded random-init grid.

  Caveats: one seed. Non-deterministic mode, so a rerun would not be
  bit-identical. ImageNet-100 numbers are not comparable to the ImageNet-1k
  Phase 1 runs. The argmax at the grid edge means the true optimum may lie
  above 2.5e-4; the grid does not show it. Cost: wall clock 3.5 h
  (2026-09-30 00:07 to 03:36 UTC), 549 img/s at epoch 0 and 556 img/s at
  the last epoch, peak allocated memory 4.6 GB. Checkpoint sha256 was not
  computed in this postrun. `epoch-metrics.parquet` sha256 `ea4cfb15...`,
  `sample-stats-train.parquet` sha256 `a90de400...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at-dev`, same run id.
- 2026-09-30 — MobileNetV4-M Phase 1 LR grid started. These are the
  approved SGD grids: random {0.0125, 0.025, 0.05}, pretrained
  {0.005, 0.015, 0.05}. Random lr 0.025 is the completed
  `plan0103-phase1-mobilenetv4-conv-medium-random-v1`.
  - Configs `imagenet_mobilenetv4_conv_medium_pgd_at_phase1_{random,pretrained}_lr*.yaml`
    (b87f139). Each is identical to the Phase 1 random config except the
    lr, `pretrained` and the tracking group.
  - Launched on Ferret from pinned worktree source-b87f13962747, 16
    workers, seed 0, W&B `lightweight-imagenet-at`:
    - GPU0: `plan0103-phase1-mobilenetv4-conv-medium-pretrained-lr0015-v1`.
    - GPU2: `plan0103-phase1-mobilenetv4-conv-medium-random-lr005-v1`.
  - Still to run: random 0.0125, pretrained 0.005 and pretrained 0.05.
    They go on Hamster GPU0 once the CUDA-graph tests release it, and on
    Ferret GPU1 once EfficientNet-B0 Phase 1 finishes.
  - Expected cost: about 42 h per run on Ferret (the random lr 0.025 run
    averaged 420 img/s).
- 2026-09-30 — CUDA Graphs (human-approved) is separate work under
  plan 0105.
  - A prototype on Ferret showed the graphed step is bitwise identical
    to the eager trainer step in deterministic mode, and about 7x faster
    at 112 px, batch 128.
- 2026-09-30 — MobileNetV4-M grid cell
  `plan0103-phase1-mobilenetv4-conv-medium-random-lr00125-v1` launched on
  Hamster GPU0: worktree source-b87f13962747, 16 workers, seed 0.
  Remaining: pretrained 0.005 and 0.05, to go on Ferret GPU1 after
  EfficientNet-B0.
- 2026-10-01 (human, chat): decisions.
  - CUDA Graphs is the default for new eligible runs: MobileNetV4-S/M or
    EfficientNet-B0, deterministic pgd_at, attack in eval mode.
  - The catastrophic-overfitting guard skips epochs 0-4.
  - No second baseline seed for now.
  - Anteater may be used.
  - Low-utilisation runs get a second co-located job.
  - The human asked whether a ~200-epoch stage 1 is really needed.
- 2026-10-01 — Stage-1 length check launched from 45befac. Configs are
  `..._resized_s256_{100,50}ep_cg.yaml`. Milestones are scaled to the
  same 50% / 76% positions with a 10-epoch warmup, and cuda_graph is on.
  These runs are not compute-matched. Each will get the same stage 2
  (224 px, 20 epochs) for comparison with the 195-epoch run.
  - `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-100ep-cg-s0-v1`:
    Hamster GPU1, co-located with the 195-epoch run, which is
    launch-bound and at about 45% utilisation.
  - `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-50ep-cg-s0-v1`:
    Ferret GPU1. EfficientNet-B0 Phase 1 random completed there.
  - Clean-training schedule control
    `plan0103-mnv4s-standard-50ep-cosine-v1`: Anteater GPU2 (RTX 2080
    Ti), `imagenet_mobilenetv4_standard_50ep_cosine.yaml` (warmup_cosine).
    It compares with the multistep 50-epoch clean run; that run used a
    different GPU type, so compare it at metric level only.
  - Schema byte-identity tests now read the pre-change configs from git,
    so new configs cannot break them.
- 2026-10-01 (human, chat): decisions.
  - **Pretrained init is dropped from the study.** It is either left out
    of the paper or re-run under the final protocol.
    `plan0103-phase1-mobilenetv4-conv-medium-pretrained-lr0015-v1` was
    stopped at epoch 23 of 50 and marked ABANDONED. The queued
    pretrained 0.005 and 0.05 cells are cancelled.
  - Full AT vs two-stage is decided after the stage-2 verdicts.
  - Epoch budget check: full AT 30 epochs, with every phase scaled by
    30/50 (warmup 6, milestones [15, 23]) and cuda_graph on.
    `plan0103-mnv4s-fullat-30ep-cg-s0-v1` runs on Ferret GPU0 from bd5a763.
  - Original-image vs S=256 stage 1, compared at 50 epochs:
    `plan0103-mnv4s-twostage-stage1-112-pgd1-orig-50ep-cg-s0-v1` (32
    workers) is queued on Ferret GPU1 behind the S=256 50-epoch run.
    Same host, so the comparison is clean.
  - MobileViT-S: run only if time allows.
  - Training allocation (epochs, schedule, two-stage) is decided before
    or during Phase 1. Then compare objectives, augmentation,
    distillation (including label smoothing) and architecture
    differences.
- 2026-10-01 (postrun): **Two-stage pilot, stage 1 (195 epochs, S=256)
  completed.** `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-s0-v1`
  (Hamster GPU1, RTX 4090) ran 195/195 epochs. `run-bundle/completion.json`
  reads `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 195 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt`, `last.pt`,
  `epoch-049.pt`, `epoch-099.pt` and `epoch-149.pt` are on disk (the
  configured epoch-199 checkpoint is past the 195-epoch end; `last.pt` is
  epoch 194). Nothing is imported into `docs/experiments/`: no aggregator
  exists for this contract, as for the other plan 0103/0104 hand-runs. The
  numbers below are read from `outputs/train/epoch-metrics.jsonl` and the
  run-bundle manifest summary.

  Fixed identity: ImageNet-1k, MobileNetV4-Conv-S
  (`mobilenetv4_conv_small_imagenet`, `imagenet_standard` profile), random
  init, plain PGD-AT, no teacher. Training data: the derived S=256 copy
  (`imagenet_train_s256`, content `65f729c7...`, short side 256, JPEG Q95,
  LANCZOS, derived from original content `ae033613...`), training view 112 px.
  Training and evaluation-attack seed 0 (split seed 20260911). Training
  attack: CE PGD-1, l_inf eps 4/255, step 4/255, random start, eval mode.
  Selection and validation attack: PGD-10, eps 4/255, step 8/765, random
  start, eval mode. SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.025,
  warmup_multistep (10-epoch warmup, x0.1 at epochs 98 and 148), 195
  epochs. World size 1, global batch 128, local BN. Deterministic, not
  compiled, no cuda_graph, step_diagnostics off, fp32. Protocol
  `controlled_imagenet_stage02_two_stage_lowres_v1`, config hash
  `82dc23ed...`. Source SHA `8333698a3de13d864446d3866a79214b8c6b47d2`
  (worktree `source-8333698a3de1`, clean). "val" is internal validation:
  the held-out 2% of the S=256 training copy (25,620 images), evaluated at
  112 px. "probe" is 2,000 training images, same transform, eval mode and
  PGD-10. Neither is the ImageNet val split. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.69 / 0.50 | 1.05 / 0.80 | 6.830 |
  | 4 | warmup | 12.36 / 5.93 | 12.10 / 5.15 | 5.776 |
  | 9 | end of warmup | 25.66 / 10.99 | 26.15 / 11.25 | 5.005 |
  | 29 | lr 0.025 | 35.25 / 15.05 | 37.10 / 16.70 | 4.489 |
  | 97 | end of lr 0.025 | 36.67 / 15.55 | 40.00 / 15.80 | 4.418 |
  | 98 | first epoch at lr 0.0025 | 43.73 / 19.15 | 48.20 / 21.40 | 4.111 |
  | 110 | peak of lr 0.0025 stage | 45.50 / 20.21 | 50.75 / 23.15 | 3.932 |
  | 147 | end of lr 0.0025 | 46.28 / 19.52 | 51.60 / 22.35 | 3.926 |
  | 148 | first epoch at lr 0.00025 | 48.06 / 20.92 | 55.50 / 24.30 | 3.807 |
  | 163 | best (by val PGD-10) | 48.64 / 21.42 | 55.55 / 25.30 | 3.736 |
  | 194 | last | 49.04 / 21.11 | 56.75 / 25.15 | 3.714 |

  Reading (n=1, internal validation at 112 px, PGD-10 only):
  1. **The run trains normally.** No collapse. The catastrophic-overfitting
     guard (val PGD-10 below half of its running maximum, epochs 0-4
     skipped) never fires: val PGD-10 never falls more than 0.9pt below
     its running maximum (at most 21.42%).
  2. **The peak-LR stage reaches its plateau early.** Val PGD-10 is 15.05%
     at epoch 29 and stays within 14.67-15.79% over epochs 30-97; val clean
     stays within 35.05-37.26%. So 68 of the 89 peak-LR epochs (9-97) add
     nothing visible on val at that LR. The val SE is about 0.25pt PGD-10
     and 0.31pt clean (25,620 images). Whether those epochs still help
     after the decays is what the 100- and 50-epoch length-check runs
     measure; this run alone cannot say.
  3. **The first decay gives the largest gain.** Epoch 97 to 98: +7.06pt
     clean, +3.61pt PGD-10 in one epoch.
  4. **Slight PGD-10 drift inside the lr 0.0025 stage.** Val PGD-10 peaks
     at 20.21% (epoch 110) and ends the stage at 19.52% (epoch 147),
     -0.69pt, while val clean stays within 45.30-46.41% over epochs
     110-147. That is about
     2.7 single-measurement SE, but successive epochs use the same val
     images, so the size is uncertain. The second decay recovers it.
  5. **The second decay gives a smaller gain.** Epoch 147 to 148: +1.78pt
     clean, +1.40pt PGD-10. Over epochs 148-194 val PGD-10 stays within
     20.92-21.42% and val clean within 48.06-49.31%.
  6. **Best and last are close.** Best (epoch 163) 48.64 / 21.42, last
     (epoch 194) 49.04 / 21.11. Best minus last is 0.31pt PGD-10
     (`robust_overfit_gap`), about 1.2 SE. Clean is 0.40pt higher at last.
  7. **Seen-unseen gap is small.** At the last epoch, probe minus val is
     +7.7 clean and +4.0 PGD-10.
  8. **No verdict on the preregistered rule.** The rule compares
     last-epoch val PGD-10 after stage 2 (224 px) with the single-stage
     reference 30.71. These stage-1 numbers are at 112 px on a different
     held-out set (from the S=256 copy), so they are not comparable with
     30.71 or with any 224 px number. Stage 2 starts from this run's
     `last.pt` (epoch 194) with its sha256. AdvXL's ViT-B stage-1 figure
     (68.5 / 39.3) is a different model and evaluation and is not
     compared.

  Caveats: one seed. Training pixels come from the derived S=256 copy (a
  disclosed preprocessing condition, see 2026-09-30 loader entries).
  Cost: wall clock 29.5 h (2026-09-29 23:57 to 2026-10-01 05:29 UTC).
  Throughput about 2,640 img/s over epochs 0-168 and about 1,800 img/s
  over epochs 169-194, after the 100-epoch length-check run was co-located
  on the same GPU. Peak allocated memory 0.79 GB. Checkpoint sha256 was
  not computed in this postrun. `epoch-metrics.parquet` sha256
  `3026bfb5...`, `sample-stats-train.parquet` sha256 `c073a194...` (from
  the run-bundle manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-01 (postrun): **Stage-1 length check, 100 epochs (S=256,
  cuda_graph) completed.**
  `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-100ep-cg-s0-v1` (Hamster
  GPU1, RTX 4090) ran 100/100 epochs. `run-bundle/completion.json` reads
  `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 100 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt`, `last.pt`
  (epoch 99), `epoch-049.pt` and `epoch-099.pt` are on disk. Nothing is
  imported into `docs/experiments/`: no aggregator exists for this
  contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: as the 195-epoch stage-1 entry above, except 100 epochs,
  milestones [50, 76] (same 50% / 76% positions, same 10-epoch warmup) and
  `cuda_graph: true`. Config
  `imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s256_100ep_cg.yaml`,
  config hash `2301f19b...`. Source SHA
  `45befac41a286d1f9326b783681bb087e8f4cef2` (worktree
  `source-45befac41a28`, clean). Training and evaluation-attack seed 0
  (split seed 20260911). World size 1, global batch 128, local BN,
  deterministic, not compiled, fp32. "val" is the same internal validation
  set as the 195-epoch run (25,620 held-out images of the S=256 copy, 112
  px); "probe" is 2,000 training images. Neither is the ImageNet val split.
  No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.69 / 0.50 | 1.05 / 0.80 | 6.830 |
  | 9 | end of warmup | 25.66 / 10.99 | 26.15 / 11.25 | 5.005 |
  | 29 | lr 0.025 | 35.25 / 15.05 | 37.10 / 16.70 | 4.489 |
  | 49 | end of lr 0.025 | 36.13 / 15.11 | 39.10 / 16.55 | 4.443 |
  | 50 | first epoch at lr 0.0025 | 43.53 / 18.83 | 46.95 / 20.25 | 4.137 |
  | 63 | peak of lr 0.0025 stage | 45.85 / 19.82 | 50.50 / 22.25 | 3.952 |
  | 75 | end of lr 0.0025 | 45.96 / 19.51 | 49.90 / 22.35 | 3.944 |
  | 76 | first epoch at lr 0.00025 | 47.74 / 20.60 | 53.20 / 23.75 | 3.840 |
  | 94 | best (by val PGD-10) | 48.38 / 21.00 | 53.90 / 24.85 | 3.769 |
  | 99 | last | 48.73 / 20.99 | 54.15 / 24.75 | 3.765 |

  Against the 195-epoch run at the same schedule positions (val clean /
  PGD-10):

  | position | 100 epochs | 195 epochs |
  |---|---:|---:|
  | end of lr 0.025 (ep 49 / 97) | 36.13 / 15.11 | 36.67 / 15.55 |
  | first epoch at lr 0.0025 (ep 50 / 98) | 43.53 / 18.83 | 43.73 / 19.15 |
  | end of lr 0.0025 (ep 75 / 147) | 45.96 / 19.51 | 46.28 / 19.52 |
  | first epoch at lr 0.00025 (ep 76 / 148) | 47.74 / 20.60 | 48.06 / 20.92 |
  | best (ep 94 / 163) | 48.38 / 21.00 | 48.64 / 21.42 |
  | last (ep 99 / 194) | 48.73 / 20.99 | 49.04 / 21.11 |

  Reading (n=1 per arm, internal validation at 112 px, PGD-10 only):
  1. **Epochs 0-49 are the 195-epoch run's epochs 0-49.** Both runs have
     the same seed, data order and LR up to epoch 49. Val clean, val PGD-10
     and train loss at epochs 47-49 match the eager 195-epoch run to full
     float precision, and epochs 0, 4, 9 and 29 match to two decimals
     including the probe. So `cuda_graph` reproduced the eager trajectory
     over 50 epochs, and the only effective difference between the two
     runs is the schedule from epoch 50 on.
  2. **The run trains normally.** No collapse; the catastrophic-overfitting
     guard never fires. Over epochs 30-49 val PGD-10 stays within
     14.67-15.60% and val clean within 35.05-36.60%.
  3. **First decay.** Epoch 49 to 50: +7.40pt clean, +3.72pt PGD-10 in one
     epoch (195 epochs: +7.06 / +3.61).
  4. **Smaller drift in the lr 0.0025 stage.** Val PGD-10 peaks at 19.82%
     (epoch 63) and ends the stage at 19.51% (epoch 75), -0.31pt (195
     epochs: -0.69pt over a longer stage). Val clean stays within
     45.28-46.13% over epochs 55-75.
  5. **Second decay.** Epoch 75 to 76: +1.78pt clean, +1.09pt PGD-10 (195
     epochs: +1.78 / +1.40). Over epochs 77-99 val PGD-10 stays within
     20.72-21.00% and val clean within 47.73-48.77%.
  6. **Best and last are the same.** Best (epoch 94) 48.38 / 21.00, last
     (epoch 99) 48.73 / 20.99; `robust_overfit_gap` 0.01pt.
  7. **Against 195 epochs, last vs last: -0.31pt clean, -0.12pt PGD-10.**
     Single-measurement val SE is about 0.31pt clean and 0.25pt PGD-10; the
     two runs are evaluated on the same images, so the unpaired SE of a
     difference (about 0.44 / 0.36pt) is an upper bound. The PGD-10
     difference is under half an SE, the clean difference under one SE.
     Best vs best is -0.26 / -0.42, but the 195-epoch best is a maximum
     over more epochs selected on the same val set, so last vs last is
     the fair comparison. At the end of stage 1, halving the epochs
     (51% of the stage-1 compute) costs no resolvable PGD-10, one seed
     each.
  8. **Seen-unseen gap.** At the last epoch probe minus val is +5.4 clean
     and +3.8 PGD-10 (195 epochs: +7.7 / +4.0).
  9. **No verdict on the preregistered rule or on the length question.**
     Both are decided after stage 2 (224 px, PGD-3, 20 epochs) from each
     run's `last.pt`, compared on last-epoch val PGD-10. These 112 px
     numbers are not comparable with the single-stage 30.71.

  Caveats: one seed per arm; the ImageNet seed-to-seed spread is not
  measured. Not compute-matched with the single-stage baseline. Training
  pixels come from the derived S=256 copy. The cuda_graph equivalence in
  point 1 is checked on logged metrics only, not on checkpoint bytes
  (`epoch-049.pt` of both runs could confirm it). Cost: wall clock 7.96 h
  (2026-10-01 00:05 to 08:03 UTC). Throughput about 4,490 img/s over
  epochs 0-59 while co-located with the 195-epoch run, and about 6,600
  img/s over epochs 61-99 alone (the eager 195-epoch run alone: about
  2,640 img/s). Peak allocated memory 0.84 GB. Checkpoint sha256 was not
  computed in this postrun. `epoch-metrics.parquet` sha256 `84a1bd1f...`,
  `sample-stats-train.parquet` sha256 `32e4b0c3...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-02 (postrun): **Phase 1 LR grid, MobileNetV4-M random init /
  lr 0.0125 completed.**
  `plan0103-phase1-mobilenetv4-conv-medium-random-lr00125-v1` (Hamster
  GPU0, RTX 4090) ran 50/50 epochs. `run-bundle/completion.json` reads
  `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 50 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt`, `last.pt`
  (epoch 49) and `epoch-049.pt` are on disk. Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract. The numbers
  below are read from `outputs/train/epoch-metrics.jsonl` and the run-bundle
  manifest summary.

  Fixed identity: ImageNet-1k (original images, content `ae033613...`),
  MobileNetV4-Conv-M (`mobilenetv4_conv_medium_imagenet`,
  `imagenet_standard` profile), random init, plain PGD-AT, no teacher, 224
  px. Training and evaluation-attack seed 0 (split seed 20260911). Training
  attack: CE PGD-3, l_inf eps 4/255, step 8/765, random start, eval mode.
  Selection and validation attack: PGD-10, same eps and step, random start,
  eval mode. SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.0125,
  warmup_multistep (10-epoch warmup, x0.1 at epochs 25 and 38), 50 epochs.
  World size 1, global batch 128, local BN. Non-deterministic, cudnn
  benchmark, not compiled, no cuda_graph, fp32. Config
  `imagenet_mobilenetv4_conv_medium_pgd_at_phase1_random_lr00125.yaml`,
  protocol `controlled_imagenet_stage02_lightweight_architecture_survey_v1`,
  config hash `538a7090...`. Source SHA
  `b87f1396274721ddf6ff81c12b964f7b5ee7f1b3` (worktree
  `source-b87f13962747`, clean). "val" is internal validation: the held-out
  2% of the ImageNet train split (25,620 images). "probe" is 2,000 training
  images, eval mode, PGD-10. Neither is the ImageNet val split. No
  AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.23 / 0.16 | 0.10 / 0.05 | 6.947 |
  | 4 | warmup | 8.13 / 4.55 | 7.40 / 4.25 | 6.204 |
  | 9 | end of warmup | 29.05 / 15.77 | 30.40 / 15.65 | 4.962 |
  | 19 | lr 0.0125 | 47.42 / 26.62 | 51.10 / 27.90 | 4.037 |
  | 24 | end of lr 0.0125 | 50.72 / 28.85 | 54.25 / 31.00 | 3.866 |
  | 25 | first epoch at lr 0.00125 | 56.89 / 34.01 | 61.85 / 38.50 | 3.543 |
  | 37 | end of lr 0.00125 | 60.04 / 35.87 | 66.20 / 42.35 | 3.293 |
  | 38 | first epoch at lr 0.000125 | 60.86 / 36.71 | 67.30 / 43.35 | 3.221 |
  | 42 | best (by val PGD-10) | 61.22 / 37.01 | 67.35 / 43.55 | 3.188 |
  | 49 | last | 61.42 / 37.00 | 67.80 / 44.25 | 3.171 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **The run trains normally.** No collapse. Val PGD-10 never falls more
     than 0.23pt below its running maximum (epoch 33). The val SE is about
     0.30pt for both clean and PGD-10 (25,620 images).
  2. **The peak-LR stage is still rising when it ends.** Over epochs 19-24
     val PGD-10 gains +2.23pt (about 0.45pt per epoch) and val clean
     +3.30pt. So at this LR the 15 peak-LR epochs do not reach a plateau.
  3. **The first decay gives the largest gain.** Epoch 24 to 25: +6.17pt
     clean, +5.16pt PGD-10 in one epoch. The lr 0.00125 stage then adds
     +1.86pt PGD-10 (epochs 25-37).
  4. **The second decay is small, and the last stage is flat.** Epoch 37 to
     38: +0.82 clean / +0.84 PGD-10. Over epochs 38-49 val PGD-10 stays
     within 36.69-37.01% and val clean within 60.86-61.42%.
  5. **Best and last are the same.** Best (epoch 42) 61.22 / 37.01, last
     (epoch 49) 61.42 / 37.00; `robust_overfit_gap` 0.01pt.
  6. **Seen-unseen gap is moderate.** At the last epoch probe minus val is
     +6.4 clean and +7.3 PGD-10, and probe PGD-10 keeps rising over the last
     stage (43.35% to 44.25%) while val stays flat.
  7. **No verdict on the MobileNetV4-M LR grid.** The lr 0.025 cell
     (`plan0103-phase1-mobilenetv4-conv-medium-random-v1`, Ferret) has
     finished but its results are not recorded in docs yet, and the lr
     0.05 cell (Ferret GPU2) is still running. This postrun could not read
     Ferret, so the argmax and the best-vs-best number wait for both.
  8. **Context only, not a grid result.** MobileNetV4-S random init / lr
     0.025 (plan 0104: same training and validation attacks, same schedule
     shape, same val slice, but deterministic) ended at 53.85 / 30.71. This
     MobileNetV4-M cell is +7.57 clean / +6.29 PGD-10 above it at half the
     peak LR. That is not best-vs-best for MobileNetV4-M, so it is not the
     architecture comparison yet.

  Caveats: one seed. Non-deterministic mode, so a rerun would not be
  bit-identical. Cost: wall clock 35.3 h (2026-09-30 05:02 to 2026-10-01
  16:18 UTC), 514-523 img/s throughout (the Ferret lr 0.025 run averaged
  about 420 img/s). Peak allocated memory 9.5 GB at epoch 0, 7.4 GB after.
  Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `b99b62b4...`, `sample-stats-train.parquet`
  sha256 `fc393137...` (from the run-bundle manifest). W&B
  `lightweight-imagenet-at`, same run id.
- 2026-10-02 — Results so far. All figures are internal val, last epoch,
  clean / PGD-10, seed 0.
  - Full AT 30 epochs, cuda_graph: 51.16 / 29.03, against 30.71 PGD-10 at
    50 epochs, i.e. -1.7pt. Human decision: full AT stays at 50 epochs.
  - Stage 1 at 112 px, before stage 2: 195 ep 49.04 / 21.11; 100 ep
    48.73 / 20.99; 50 ep S=256 46.63 / 20.44; 50 ep original images
    46.97 / 20.58. 195 vs 100 epochs differ by -0.1pt PGD-10, and S=256 vs
    original by -0.14pt.
  - MobileNetV4-M random SGD lr 0.0125 / 0.025 / 0.05: 37.00 / 39.39 /
    38.36 PGD-10; the 0.05 run is at epoch 40 of 50. So LR matters for
    SGD models too.
  - Human decisions:
    - LR grid option A: 3 points per model, best-vs-best.
    - The primary cost metric is Hamster wall-clock, measured with
      cuda_graph and no co-location; FLOPs are reported too.
  - Stage 2 was launched late: the session-bound watcher died, so GPUs
    sat idle for about 12 h. All four stage-2 runs start from each
    stage-1 `last.pt`.
    - Hamster: `plan0103-mnv4s-twostage-stage2-224-pgd3-from195-s0-v1`
      (GPU0) and `-from100-s0-v1` (GPU1), worktree source-030be3f7c6a8.
    - Ferret: `-from50s256-s0-v1` (GPU0) and `-from50orig-s0-v1` (GPU1),
      worktree source-bd5a76330e55.
  - MobileNetV4-S grid cell: `plan0103-mnv4s-fullat-50ep-lr00125-cg-s0-v1`
    runs on Ferret GPU0 (4b6780e), co-located with the stage-2 run there
    (about 36% utilisation).
- 2026-10-02 (postrun): **Two-stage pilot, stage 2 from the 195-epoch
  stage 1 completed.**
  `plan0103-mnv4s-twostage-stage2-224-pgd3-from195-s0-v1` (Hamster GPU0,
  RTX 4090) ran 20/20 epochs. `run-bundle/completion.json` reads
  `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 20 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt` (epoch 11)
  and `last.pt` (epoch 19) are on disk; no `epoch-NNN.pt` was written,
  because the configured checkpoint epochs (49, 99, ...) lie beyond 20
  epochs. Nothing is imported into `docs/experiments/`: no aggregator
  exists for this contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: ImageNet-1k (original images, content `ae033613...`),
  MobileNetV4-Conv-S (`mobilenetv4_conv_small_imagenet`,
  `imagenet_standard` profile), plain PGD-AT, no teacher, 224 px training.
  Init: stage-1 195-epoch `last.pt` (epoch 194,
  `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-s0-v1`, sha256
  `b29eb82a...`, strict load, model entry only). Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack: CE
  PGD-3, l_inf eps 4/255, step 8/765, random start, eval mode. Selection
  and validation attack: PGD-10, same eps and step, random start, eval
  mode. SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.0025,
  warmup_multistep (2-epoch warmup, x0.1 at epochs 10 and 15), 20 epochs.
  World size 1, global batch 128, local BN. Deterministic, not compiled,
  no cuda_graph, fp32. Config
  `imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml`, protocol
  `controlled_imagenet_stage02_two_stage_lowres_v1`, config hash
  `b733761b...`. Source SHA `030be3f7c6a8d79fd0bd1b608f09d2ea393d7032`
  (worktree `source-030be3f7c6a8`, clean). "val" is internal validation:
  the held-out 2% of the original ImageNet train split (25,620 images, 224
  px), the same slice as the single-stage reference. "probe" is 2,000
  training images, eval mode, PGD-10. Neither is the ImageNet val split.
  No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup (lr 0.00125) | 51.86 / 28.40 | 57.75 / 32.00 | 3.908 |
  | 1 | first epoch at lr 0.0025 | 50.64 / 27.34 | 55.85 / 32.00 | 3.931 |
  | 5 | lr 0.0025 | 51.26 / 28.29 | 57.00 / 33.10 | 3.876 |
  | 9 | end of lr 0.0025 | 50.72 / 27.30 | 56.60 / 31.15 | 3.849 |
  | 10 | first epoch at lr 0.00025 | 53.57 / 28.70 | 59.75 / 34.10 | 3.684 |
  | 11 | best (by val PGD-10) | 54.31 / 29.11 | 60.60 / 33.80 | 3.646 |
  | 14 | end of lr 0.00025 | 53.73 / 28.54 | 59.95 / 33.70 | 3.605 |
  | 15 | first epoch at lr 0.000025 | 54.63 / 28.59 | 60.95 / 33.90 | 3.566 |
  | 17 | lr 0.000025 | 54.90 / 28.86 | 60.80 / 33.80 | 3.552 |
  | 18 | lr 0.000025 | 54.54 / 27.59 | 60.80 / 32.60 | 3.546 |
  | 19 | last | 54.91 / 28.45 | 61.40 / 33.70 | 3.542 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **Preregistered rule: worse.** Last-epoch val PGD-10 is 28.45%
     against the single-stage random-init reference 30.71% (MobileNetV4-S,
     lr 0.025, 50 epochs, plan 0104: same attacks, same val slice, also
     deterministic). That is -2.26pt, past the -1pt line, so for this seed
     and this stage-1 length the two-stage schedule is worse on PGD-10 at
     the FLOP-matched budget. Clean is +1.06pt (54.91 vs 53.85).
  2. **The verdict does not depend on the last-epoch choice.** No stage-2
     epoch reaches 29.71% (reference minus 1pt); the best is 29.11%
     (epoch 11), -1.60pt. The val SE is about 0.28pt PGD-10 and 0.31pt
     clean, so the unpaired SE of the difference is about 0.40pt PGD-10.
     The only ImageNet seed pair measured (Arm A, seeds 0 and 1) differs
     by 0.01pt PGD-10, but one pair does not give a seed noise floor.
  3. **Stage 2 adds clean accuracy, not PGD-10.** After its first epoch
     (warmup, lr 0.00125) val PGD-10 is already 28.40%. Over the remaining
     19 epochs it peaks at +0.71pt and ends at +0.05pt, while val clean
     rises +3.05pt. In the lr 0.0025 stage (epochs 1-9) val PGD-10 stays
     within 27.30-28.29% and val clean within 50.52-51.67%.
  4. **Decays.** Epoch 9 to 10: +2.85 clean / +1.40 PGD-10. Epoch 14 to
     15: +0.90 / +0.05.
  5. **One-epoch PGD-10 dip in the last stage.** Epoch 17 to 18 val
     PGD-10 falls 1.27pt (28.86% to 27.59%) at lr 0.000025, with clean
     -0.36pt, and recovers to 28.45% at epoch 19. Probe PGD-10 falls by
     1.20pt at the same epoch (33.80% to 32.60%), so it is a change in the
     model, not val sampling. The cause was not investigated. It means a
     single last-epoch reading here can move by about 1pt between
     neighbouring epochs; point 2 shows the verdict survives that.
  6. **Best and last.** Best (epoch 11) 54.31 / 29.11, last (epoch 19)
     54.91 / 28.45; `robust_overfit_gap` 0.66pt.
  7. **Seen-unseen gap.** At the last epoch probe minus val is +6.5 clean
     and +5.2 PGD-10.
  8. **What is still open.** This is one of four stage-2 arms. The
     stage-1 length question (195 vs 100 vs 50 epochs) and the S=256 vs
     original-image question need `-from100-s0-v1` (Hamster GPU1) and
     `-from50s256-s0-v1` / `-from50orig-s0-v1` (Ferret), all still
     running.

  Caveats: one seed. Stage 1 trained on the derived S=256 copy (a
  disclosed preprocessing condition); stage 2 trained on original images.
  The compute match is a FLOP estimate (backward taken as 2x forward), not
  measured wall-clock; the human's primary cost metric (Hamster
  wall-clock with cuda_graph, no co-location) has not been measured for
  either arm. AdvXL's ViT-B result is a different model and recipe and is
  not compared. Cost: wall clock 7.87 h (2026-10-01 18:03 to 2026-10-02
  01:56 UTC), 931-943 img/s throughout, eager. Peak allocated memory
  2.87 GB. Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `5f9d3aa3...`,
  `sample-stats-train.parquet` sha256 `052b0765...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-02 (postrun): **Two-stage pilot, stage 2 from the 100-epoch
  stage 1 completed.**
  `plan0103-mnv4s-twostage-stage2-224-pgd3-from100-s0-v1` (Hamster GPU1,
  RTX 4090) ran 20/20 epochs. `run-bundle/completion.json` reads
  `completed`, the manifest status is `completed`, and
  `run-bundle/error-marker.txt` reads "no application error recorded". The
  watcher re-derived it as terminal and successful. All 20 epoch rows are
  present (`epoch_metrics_complete: true`). `train.log` has no traceback,
  NaN or error line (only the wandb git-root warning). `best.pt` (epoch 0)
  and `last.pt` (epoch 19) are on disk; no `epoch-NNN.pt` was written,
  because the configured checkpoint epochs lie beyond 20 epochs. Nothing
  is imported into `docs/experiments/`: no aggregator exists for this
  contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest summary.

  Fixed identity: identical to the from-195 stage 2 above except the init
  checkpoint. ImageNet-1k (original images, content `ae033613...`),
  MobileNetV4-Conv-S (`mobilenetv4_conv_small_imagenet`,
  `imagenet_standard` profile), plain PGD-AT, no teacher, 224 px training.
  Init: stage-1 100-epoch `last.pt` (epoch 99,
  `plan0103-mnv4s-twostage-stage1-112-pgd1-s256-100ep-cg-s0-v1`, sha256
  `200a94fd...`, strict load, model entry only). Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack: CE
  PGD-3, l_inf eps 4/255, step 8/765, random start, eval mode. Selection
  and validation attack: PGD-10, same eps and step, random start, eval
  mode. SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.0025,
  warmup_multistep (2-epoch warmup, x0.1 at epochs 10 and 15), 20 epochs.
  World size 1, global batch 128, local BN. Deterministic, not compiled,
  no cuda_graph, fp32. Protocol
  `controlled_imagenet_stage02_two_stage_lowres_v1`, config hash
  `ae2411f5...` (differs from the from-195 run's only through the init
  checkpoint). Source SHA `030be3f7c6a8d79fd0bd1b608f09d2ea393d7032`
  (worktree `source-030be3f7c6a8`, clean). "val" is internal validation:
  the held-out 2% of the original ImageNet train split (25,620 images, 224
  px), the same slice as the single-stage reference. "probe" is 2,000
  training images, eval mode, PGD-10. Neither is the ImageNet val split.
  No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup (lr 0.00125); best (by val PGD-10) | 51.70 / 28.40 | 57.45 / 32.90 | 3.943 |
  | 1 | first epoch at lr 0.0025 | 50.77 / 27.60 | 55.35 / 32.60 | 3.939 |
  | 4 | lr 0.0025 | 51.11 / 27.97 | 56.50 / 31.60 | 3.888 |
  | 9 | end of lr 0.0025 | 51.25 / 26.57 | 55.40 / 31.00 | 3.845 |
  | 10 | first epoch at lr 0.00025 | 53.55 / 28.00 | 59.25 / 33.10 | 3.676 |
  | 14 | end of lr 0.00025 | 53.90 / 28.08 | 59.70 / 32.40 | 3.597 |
  | 15 | first epoch at lr 0.000025 | 54.90 / 27.81 | 60.55 / 32.25 | 3.566 |
  | 16 | lr 0.000025 | 54.67 / 28.31 | 60.75 / 33.25 | 3.561 |
  | 17 | lr 0.000025 | 55.03 / 28.26 | 60.85 / 33.00 | 3.558 |
  | 18 | lr 0.000025 | 54.69 / 27.59 | 60.15 / 31.85 | 3.559 |
  | 19 | last | 55.12 / 28.19 | 60.85 / 32.70 | 3.557 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **Preregistered rule: worse.** Last-epoch val PGD-10 is 28.19%
     against the single-stage random-init reference 30.71% (MobileNetV4-S,
     lr 0.025, 50 epochs: same attacks, same val slice, also
     deterministic). That is -2.52pt, past the -1pt line. Clean is +1.27pt
     (55.12 vs 53.85). The rule was written for the compute-matched
     195-epoch arm. This arm uses less compute (FLOP estimate: 100 x 0.154
     + 20 = 35.4 reference-epoch equivalents, about 71% of the reference's
     50), so here "worse" means worse at about 71% of the compute.
  2. **The verdict does not depend on the last-epoch choice.** No stage-2
     epoch reaches 29.71% (reference minus 1pt); the best is 28.40%
     (epoch 0), -2.31pt. The unpaired SE of the difference is about 0.40pt
     PGD-10 (val SE about 0.28pt PGD-10, 0.31pt clean).
  3. **Stage-1 length: no resolvable difference after stage 2.** Against
     stage 2 from the 195-epoch stage 1 (same stage-2 config, same val
     images), last vs last is +0.21 clean / -0.26 PGD-10 (55.12 / 28.19 vs
     54.91 / 28.45), under one SE of the difference; one seed each. Best
     vs best is -0.71 PGD-10, but best is a maximum selected on the same
     val set, so last vs last is the fair comparison. Halving stage 1 (51%
     of the stage-1 compute) cost no resolvable PGD-10 here, as at the end
     of stage 1 (-0.12pt).
  4. **Stage 2 again adds clean accuracy, not PGD-10.** Val PGD-10 after
     the first stage-2 epoch is 28.40%, the same value as in the 195-epoch
     arm. No later epoch exceeds it; the run ends 0.21pt below it while val
     clean rises +3.42pt. In the lr 0.0025 stage (epochs 1-9) val PGD-10
     drifts down from 27.97% (epoch 4) to 26.57% (epoch 9), -1.40pt, with
     clean flat (50.48-51.35%); the 195 arm stayed within 27.30-28.29%
     there.
  5. **Decays.** Epoch 9 to 10: +2.30 clean / +1.43 PGD-10. Epoch 14 to
     15: +1.00 / -0.27.
  6. **One-epoch PGD-10 dip at epoch 18 again.** Epoch 17 to 18 val
     PGD-10 falls 0.67pt (28.26% to 27.59%) with clean -0.34pt, and
     recovers to 28.19% at epoch 19; probe PGD-10 falls 1.15pt at the same
     epoch. The 195-epoch arm dipped at the same epoch (-1.27pt). Both arms
     share seed-0 data order and attack seeds, so the epoch-18 batches are
     a candidate cause; not investigated.
  7. **Best and last.** Best (epoch 0) 51.70 / 28.40, last (epoch 19)
     55.12 / 28.19; `robust_overfit_gap` 0.21pt.
  8. **Seen-unseen gap.** At the last epoch probe minus val is +5.7 clean
     and +4.5 PGD-10 (195 arm: +6.5 / +5.2).
  9. **What is still open.** Two of four stage-2 arms are done. The S=256
     vs original-image question needs `-from50s256-s0-v1` and
     `-from50orig-s0-v1` (Ferret), still running.

  Caveats: one seed. Stage 1 trained on the derived S=256 copy (a
  disclosed preprocessing condition); stage 2 trained on original images.
  The compute figure is a FLOP estimate (backward taken as 2x forward),
  not measured wall-clock; Hamster wall-clock with cuda_graph and no
  co-location has not been measured for either arm. Cost: wall clock
  7.95 h (2026-10-01 18:03 to 2026-10-02 02:01 UTC), 919-931 img/s
  throughout, eager; the from-195 stage 2 ran on GPU0 at the same time.
  Peak allocated memory 2.87 GB. Checkpoint sha256 was not computed in
  this postrun. `epoch-metrics.parquet` sha256 `ce1cc603...`,
  `sample-stats-train.parquet` sha256 `bd8f035e...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-03 — **Stage-2 verdicts (seed 0, internal val, last epoch,
  clean / PGD-10).**
  - Preregistered verdict, 195-epoch two-stage vs single-stage 50-epoch
    full AT (lr 0.025, 30.71): 54.91 / 28.45, i.e. -2.26pt PGD-10.
    **Worse.**
  - Post-hoc variants (not preregistered):
    - stage 1 of 100 epochs: 55.12 / 28.19;
    - stage 1 of 50 epochs on S=256: 53.13 / 29.84 (-0.87);
    - stage 1 of 50 epochs on original images: 53.13 / 29.84.
  - Longer stage 1 gives worse PGD-10 after stage 2. With stage 1 at 195
    or 100 epochs, PGD-10 does not grow during stage 2.
  - S=256 and original images give the same result after stage 2.
  - Clean-training cosine control: 71.28 vs 70.86 for multistep, inside
    noise.
  - MobileNetV4-M random-init LR grid: 0.0125 / 0.025 / 0.05 give
    37.00 / 39.39 / 39.10 PGD-10; argmax 0.025.
- 2026-10-03 (human): measure full AT and two-stage in their fastest
  configurations, and run the generalisation check.
  - **Preregistered rule (two-stage generalisation check, written before
    launch).**
    - Arms: MobileNetV4-M and EfficientNet-B0, random init, seed 0.
      - Stage 1: 112 px on S=256, PGD-1, 50 epochs, lr 0.025,
        milestones [25, 38], cuda_graph on.
      - Stage 2: 224 px original images, PGD-3, 20 epochs from stage-1
        `last.pt`, lr 0.0025, cuda_graph on.
    - References: each model's existing full-AT 50-epoch lr-0.025 run.
      - MobileNetV4-M: 39.39.
      - EfficientNet-B0: 35.93.
      - Both references ran non-deterministic; deterministic vs
        non-deterministic changes only bitwise reproducibility, and this
        is disclosed.
    - Metric: stage-2 last-epoch internal-val PGD-10.
    - Two-stage is adopted as the Phase 1 method for the BN-CNN family
      (MobileNetV4-S/M, EfficientNet-B0) only if BOTH models are within
      1pt of their reference (>= ref - 1.0). Otherwise Phase 1 uses full
      AT for 50 epochs.
    - ConvNeXt-Atto and DeiT-Tiny are not covered by this rule. They are
      not cuda_graph-eligible, and DeiT at 112 px needs
      position-embedding handling.
    - Clean accuracy is reported but not part of the rule.
  - Configs:
    - `imagenet_{mnv4m,effb0}_twostage_stage1_112_pgd1_resized_s256_50ep_cg.yaml`
    - `imagenet_{mnv4m,effb0}_twostage_stage2_224_pgd3_ft_cg.yaml`
    - fastest-config MobileNetV4-S:
      `imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft_cg.yaml` and
      `imagenet_mobilenetv4_pgd_at_random_init_lr0025_cg.yaml`
- 2026-10-03 (human): determinism may be dropped for new runs; AMP and
  BF16 are not used; training PGD stays at 3 steps. Checkpoint and
  validation policy waits for a literature check plus our best-vs-last
  data.
  - Best minus last internal-val PGD-10 over 29 completed ImageNet-1k AT
    runs, excluding the collapsed revisiting-recipe run: median about
    0.1pt, max 0.66pt, mostly under 0.35pt. The val SE is about 0.3pt.
  - **Preregistered rule (stage-2 length check, written before launch).**
    - Arms: from the same stage-1 `last.pt`
      (`plan0103-mnv4s-twostage-stage1-112-pgd1-s256-50ep-cg-s0-v1`,
      sha256 cedf09c5…), run stage 2 at 224 px for 5 and 10 epochs.
      Milestones and warmup are scaled from 20 epochs; deterministic,
      cuda_graph on. Configs are
      `imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft_cg_{5,10}ep.yaml`.
    - Reference: the existing 20-epoch stage 2 from that checkpoint,
      29.84 PGD-10 (last epoch).
    - Rule: adopt the shortest length whose last-epoch internal-val
      PGD-10 is >= 29.34 (reference - 0.5pt); otherwise keep 20 epochs.
      Seed 0 only. Clean accuracy is reported, not ruled on.
- 2026-10-03 (postrun): **Fastest-config timing, MobileNetV4-S stage 1
  (50 epochs, S=256, cuda_graph) on Hamster completed.**
  `plan0103-mnv4s-twostage-stage1-112-s256-50ep-cg-hamster-s0-v1` (Hamster,
  one RTX 4090) ran 50/50 epochs. It is the Hamster re-run of the Ferret
  50-epoch stage 1 (`plan0103-mnv4s-twostage-stage1-112-pgd1-s256-50ep-cg-s0-v1`),
  made to measure two-stage wall-clock in its fastest configuration (human,
  2026-10-03). Its launch was not logged in this plan.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` (epoch 49) and `epoch-049.pt` are on disk.
  Nothing is imported into `docs/experiments/`: no aggregator exists for
  this contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest.

  Fixed identity: as the 195-epoch stage-1 entry (2026-10-01) except 50
  epochs, milestones [25, 38] (10-epoch warmup) and `cuda_graph: true`.
  ImageNet-1k, derived S=256 training copy (`imagenet_train_s256`, content
  `65f729c7...`), 112 px training view, MobileNetV4-Conv-S random init,
  plain PGD-AT, no teacher. Training attack CE PGD-1, l_inf eps 4/255, step
  4/255, random start, eval mode. Selection and validation attack PGD-10,
  eps 4/255, step 8/765, random start, eval mode. SGD (Nesterov, momentum
  0.9, wd 1e-4), peak LR 0.025. Training and evaluation-attack seed 0
  (split seed 20260911). World size 1, global batch 128, local BN.
  Deterministic, not compiled, fp32, 16 workers. Config
  `imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s256_50ep_cg.yaml`,
  protocol `controlled_imagenet_stage02_two_stage_lowres_v1`, config hash
  `e69ebd4c...`. Source SHA `65369a36668b70c789c3e0afbf8f5b3a4870bfc7`
  (worktree `source-65369a36668b`, clean). "val" is internal validation
  (25,620 held-out images of the S=256 copy, 112 px); "probe" is 2,000
  training images. Neither is the ImageNet val split. No AutoAttack has
  run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.69 / 0.50 | 1.05 / 0.80 | 6.830 |
  | 4 | warmup | 12.36 / 5.93 | 12.10 / 5.15 | 5.776 |
  | 9 | end of warmup | 25.66 / 10.99 | 26.15 / 11.25 | 5.005 |
  | 24 | end of lr 0.025 | 34.52 / 14.42 | 36.15 / 16.40 | 4.521 |
  | 25 | first epoch at lr 0.0025 | 41.70 / 18.31 | 43.85 / 20.25 | 4.218 |
  | 37 | end of lr 0.0025 | 44.80 / 19.13 | 48.15 / 21.05 | 4.018 |
  | 38 | first epoch at lr 0.00025 | 46.03 / 19.88 | 50.30 / 21.75 | 3.939 |
  | 49 | last = best (by val PGD-10) | 46.63 / 20.44 | 52.00 / 22.65 | 3.888 |

  Reading (n=1, internal validation at 112 px, PGD-10 only):
  1. **Same result as the Ferret run.** Last-epoch val 46.63 / 20.44 equals
     the Ferret 50-epoch S=256 stage 1 (46.63 / 20.44, 2026-10-02 summary)
     to two decimals. Same config, seed and deterministic mode, different
     host and source SHA. Epochs 0, 4 and 9 also match the eager 195-epoch
     run to two decimals, including the probe. Only these logged values
     were compared; the Ferret per-epoch rows were not read here. The
     `last.pt` files differ in sha256 (`d4e1d8fe...` here, `cedf09c5...` on
     Ferret), but the file holds run metadata, so this is not a weight
     comparison.
  2. **The run trains normally.** Val PGD-10 never falls more than 0.36pt
     below its running maximum (epoch 30); the catastrophic-overfitting
     guard never fires. Val SE is about 0.31pt clean and 0.25pt PGD-10.
  3. **The peak-LR stage is still rising slowly when it ends.** Val PGD-10
     goes 13.56% (epoch 15) to 14.42% (epoch 24), within 14.18-14.69% over
     epochs 20-24. The 195-epoch run reached 15.05% at epoch 29.
  4. **Decays.** Epoch 24 to 25: +7.18 clean / +3.89 PGD-10. Epoch 37 to 38:
     +1.23 / +0.75. No PGD-10 drift in the lr 0.0025 stage (peak 19.36% at
     epoch 32, end 19.13%). In the last stage val PGD-10 stays within
     19.88-20.44% and still creeps up.
  5. **Best and last are the same epoch** (49); `robust_overfit_gap` 0.0.
  6. **Seen-unseen gap.** At the last epoch probe minus val is +5.4 clean
     and +2.2 PGD-10.
  7. **Wall-clock (the purpose of this run).** 3.45 h from start to finish
     (2026-10-02 16:15:58 to 19:42:48 UTC, 12,410 s). Summed per-epoch
     training time is 10,077 s (2.80 h); the remaining 2,333 s (about 47 s
     per epoch) is validation, probe, checkpointing and start-up.
     Throughput is 6,540-6,620 img/s in epochs 0-11 and mostly 6,200-6,570
     img/s afterwards, with two slow stretches: epochs 24-27 (5,353, then
     4,170-4,260 img/s) and epochs 47-49 (6,225 to 5,614 img/s). Their
     cause was not identified. The other Hamster GPU ran
     `plan0103-mnv4s-fullat-50ep-lr0025-cg-s0-v1` (same source, started the
     same second) for the whole run, sharing CPU, disk and data loading. So
     the GPU was not shared, but the host was. At the typical ~194 s per
     epoch, the slow epochs cost about 0.1 h. cuda_graph: 1 capture, 2
     eager steps and 9,807 replays per epoch. Peak allocated memory
     0.84 GB, reserved 2.24 GB.
  8. **No verdict here.** No preregistered rule reads this run on its own.
     The wall-clock comparison with full AT needs the full-AT timing run
     (at epoch 10 of 50 when this run ended) and the Hamster stage 2 from
     this `last.pt`, `plan0103-mnv4s-twostage-stage2-224-cg-from50s256-hamster-s0-v1`,
     which started at 19:43:02 UTC (init sha256 `d4e1d8fe...`).

  Caveats: one seed. Training pixels come from the derived S=256 copy.
  The timing is not a "no co-location" host measurement in the strict
  sense (see point 7). `epoch-metrics.parquet` sha256 `676c1c92...`,
  `sample-stats-train.parquet` sha256 `e73fb33c...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-03 (postrun): **Fastest-config timing, MobileNetV4-S stage 2 on
  Hamster failed at the start of epoch 2 of 20 (CUDA out of memory caused
  by a foreign process).**
  `plan0103-mnv4s-twostage-stage2-224-cg-from50s256-hamster-s0-v1` (hand-run,
  pinned worktree source-65369a36668b, config
  `imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft_cg.yaml`, init
  `plan0103-mnv4s-twostage-stage1-112-s256-50ep-cg-hamster-s0-v1/last.pt`
  sha256 `d4e1d8fe...`, 16 workers, seed 0). The watcher reports
  `terminal: true`, `success: false`, `failure_class: unknown`; the manifest
  status is `failed`, `error-marker.txt` reads "application failure
  recorded", and there is no `completion.json`.
  - Diagnosis (`train.log`): `torch.OutOfMemoryError` in `pgd.py:253`
    (`torch.autograd.grad`), called from `_cuda_graph_body`, right after
    epoch 1 finished (last progress 20:26:18 UTC, failure 20:26:24 UTC). The
    OOM message: the GPU had 32 MiB free of 23.52 GiB; this process held
    3.64 GiB; **process 2384146 held 19.80 GiB**. That is the only other
    process listed, so the device was not the one running the full-AT
    timing run (PID 2201374, physical GPU 0, still running at 4.6 GiB). PID
    2384146 is not in the current GPU process list and was not identified.
    The run's own peak reserved memory was 8.7 GB, so it fits a 24 GB card
    alone. The failure point fits the cuda_graph schedule: the graph is
    re-captured every epoch (1 capture, 2 eager steps per epoch), so the
    run needs fresh memory at each epoch boundary, after validation.
  - Classification: technical (external memory pressure), not scientific.
    The watcher labels it `unknown`, so per the postrun contract this goes
    to decision packet 0031 and nothing is retried here.
  - What the two completed epochs show (internal val at 224 px, PGD-10,
    not an official test): epoch 0 clean 50.12 / PGD-10 27.49, epoch 1
    49.48 / 27.09. `best.pt` and `last.pt` (epoch 1) exist. Training time
    1,165 s and 1,156 s per epoch; wall-clock 2,596 s for two epochs
    (19:43:02 to 20:26:18 UTC), about 21.6 min per epoch, which
    extrapolates to about 7.2 h for 20 epochs (eager was about 7.9 h). This
    is an extrapolation from two epochs, not a measurement.
  - Proposed retry (not run; needs the human's choice in 0031): a fresh
    `-v2` run with its own output directory, same config, init and SHA, on
    the GPU the failed run used (CUDA index 1 assumed, since physical GPU 0
    holds the full-AT run), after checking that GPU is empty:
    `cd <runtime>/worktrees/source-65369a36668b && CUDA_VISIBLE_DEVICES=1
    PYTHONPATH=src ARD_SEED=0 ARD_NUM_WORKERS=16
    ARD_IMAGENET_ROOT=/home/shunsukenaito/workspace-local/datasets/imagenet
    ARD_STAGE1_CHECKPOINT=<runtime>/runs/plan0103-mnv4s-twostage-stage1-112-s256-50ep-cg-hamster-s0-v1/outputs/train/last.pt
    ARD_STAGE1_CHECKPOINT_SHA256=d4e1d8fef8e36d6333a89de96d1e25adba5532ab0ff3bf88cd29b8dccab24753
    WANDB_ENTITY=shunsuke-n-waseda-university
    WANDB_PROJECT=lightweight-imagenet-at
    ARD_RUN_ID=plan0103-mnv4s-twostage-stage2-224-cg-from50s256-hamster-s0-v2
    ARD_JOB_OUTPUT_DIR=<runtime>/runs/plan0103-mnv4s-twostage-stage2-224-cg-from50s256-hamster-s0-v2/outputs/train
    /home/shunsukenaito/.conda/envs/ard-v2/bin/python -m ard.cli.train
    --config configs/scientific/imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft_cg.yaml`,
    where `<runtime>` is
    `/home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap`.
- 2026-10-03 (postrun): **Fastest-config timing, MobileNetV4-S stage 2
  (20 epochs, cuda_graph) on Hamster completed (-v2 rerun).**
  `plan0103-mnv4s-twostage-stage2-224-cg-from50s256-hamster-s0-v2` (hand-run,
  Hamster, one RTX 4090) ran 20/20 epochs. It is option A of decision 0031
  (a fresh rerun of the OOM-failed `-v1`, same config, init and SHA). Its
  launch was not logged in this plan, and 0031's `chosen` field is still
  null. `run-bundle/completion.json` reads `completed`, the manifest status
  is `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 20 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt` (epoch 16) and `last.pt` (epoch 19) are on disk.
  Nothing is imported into `docs/experiments/`: no aggregator exists for
  this contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest.

  Fixed identity: as the from-195 stage 2 (2026-10-02) except the init
  checkpoint and `cuda_graph: true`. ImageNet-1k (original images, content
  `ae033613...`), MobileNetV4-Conv-S, plain PGD-AT, no teacher, 224 px
  training. Init: the Hamster stage-1 50-epoch S=256 `last.pt` (epoch 49,
  `plan0103-mnv4s-twostage-stage1-112-s256-50ep-cg-hamster-s0-v1`, sha256
  `d4e1d8fe...`, strict load, model entry only). Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack CE PGD-3,
  l_inf eps 4/255, step 8/765, random start, eval mode. Selection and
  validation attack PGD-10, same eps and step, random start, eval mode. SGD
  (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.0025, warmup_multistep
  (2-epoch warmup, x0.1 at epochs 10 and 15), 20 epochs. World size 1,
  global batch 128, local BN. Deterministic, not compiled, cuda_graph, fp32.
  Config `imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft_cg.yaml`,
  protocol `controlled_imagenet_stage02_two_stage_lowres_v1`, config hash
  `fdd7454f...`. Source SHA `65369a36668b70c789c3e0afbf8f5b3a4870bfc7`
  (worktree `source-65369a36668b`, clean). "val" is internal validation
  (25,620 held-out original train images, 224 px, the same slice as the
  single-stage reference); "probe" is 2,000 training images. Neither is the
  ImageNet val split. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup (lr 0.00125) | 50.12 / 27.49 | 54.35 / 30.75 | 4.052 |
  | 1 | first epoch at lr 0.0025 | 49.48 / 27.09 | 53.35 / 30.85 | 4.012 |
  | 5 | lr 0.0025 | 50.28 / 27.80 | 54.50 / 31.65 | 3.943 |
  | 9 | end of lr 0.0025 | 50.60 / 27.91 | 54.25 / 31.55 | 3.913 |
  | 10 | first epoch at lr 0.00025 | 52.54 / 29.46 | 57.55 / 34.00 | 3.803 |
  | 14 | end of lr 0.00025 | 52.75 / 29.66 | 57.80 / 33.45 | 3.759 |
  | 15 | first epoch at lr 0.000025 | 53.14 / 29.73 | 58.55 / 33.55 | 3.746 |
  | 16 | best (by val PGD-10) | 53.15 / 29.86 | 58.45 / 34.00 | 3.743 |
  | 17 | lr 0.000025 | 53.08 / 29.77 | 58.45 / 34.05 | 3.740 |
  | 18 | lr 0.000025 | 53.14 / 29.77 | 57.80 / 33.85 | 3.740 |
  | 19 | last | 53.13 / 29.84 | 58.30 / 34.30 | 3.740 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **Reproduces both earlier runs.** Epochs 0 and 1 equal the failed
     `-v1` (50.12 / 27.49 and 49.48 / 27.09) to two decimals, which is the
     check 0031 A asked for. The last epoch, 53.13 / 29.84, equals the
     Ferret stage 2 from the Ferret 50-epoch S=256 stage 1
     (`-from50s256-s0-v1`, 53.13 / 29.84, 2026-10-03 verdict summary) to two
     decimals. Only those logged values were compared; the Ferret per-epoch
     rows were not read here. So the accuracy side of the fastest
     configuration is the already-reported one, and no new accuracy verdict
     comes from this run.
  2. **Wall-clock (the purpose of this run).** 6.30 h from start to finish
     (run-bundle `created_at` 2026-10-02 20:40:59 to `finished_at`
     2026-10-03 02:59:10 UTC, 22,691 s). Summed per-epoch training time is
     21,188 s (5.89 h); the remaining 1,503 s (about 75 s per epoch) is
     validation, probe, checkpointing and start-up. Throughput 1,153 img/s
     in epoch 0, 1,172 in epoch 1, then 1,186-1,188 img/s for every
     remaining epoch; no slow stretch. The 2-epoch extrapolation in 0031
     (7.2 h) was 14% too high: `-v1` trained epochs 0/1 in 1,165 / 1,156 s,
     this run in 1,089 / 1,071 s, presumably because the foreign process
     that later caused the OOM was already sharing `-v1`'s GPU (not
     verified). The full-AT timing run
     (`plan0103-mnv4s-fullat-50ep-lr0025-cg-s0-v1`) held the other Hamster
     GPU for the whole run, so as in stage 1 the GPU was not shared but the
     host was. cuda_graph: 1 capture, 2 eager steps and 9,807 replays per
     epoch. Peak allocated memory 3.04 GB, reserved 8.71 GB (epoch 1).
  3. **Two-stage total on Hamster.** Stage 1 (3.45 h, 12,410 s) plus this
     stage 2 (6.30 h, 22,691 s) is 35,101 s, 9.75 h, start to finish, not
     counting the gap between the two runs. Context only: the eager stage 2
     runs (from195 / from100, source `030be3f`, both Hamster GPUs busy with
     stage 2) took 7.87 / 7.95 h at 919-943 img/s, so cuda_graph here gives
     about 1.26-1.29x their throughput; different source and co-location, so this
     is not a controlled speed-up measurement.
  4. **Stage 2 adds PGD-10 from this stage 1.** Unlike the from-195 and
     from-100 arms, val PGD-10 does not drift down in the lr 0.0025 stage
     (27.09-27.93%, ending at 27.91%) and ends 2.35pt above its epoch-0
     value. Decays: epoch 9 to 10 +1.94 clean / +1.55 PGD-10; epoch 14 to 15
     +0.39 / +0.07. The last stage is flat (29.73-29.86%).
  5. **No epoch-18 dip.** Epoch 17 to 18 val PGD-10 is unchanged (29.77%),
     whereas the from-195 and from-100 arms both dipped at epoch 18 (-1.27 /
     -0.67pt). All three share seed-0 data order and attack seeds, so the
     epoch-18 batches alone do not cause the dip; not investigated further.
  6. **Best and last.** Best (epoch 16) 53.15 / 29.86, last (epoch 19)
     53.13 / 29.84; `robust_overfit_gap` 0.03pt, well under the val SE
     (about 0.28pt PGD-10).
  7. **Seen-unseen gap.** At the last epoch probe minus val is +5.2 clean
     and +4.5 PGD-10.
  8. **No verdict here.** The preregistered wall-clock comparison (0030 A:
     two-stage time / full-AT time in 0.9-1.1 reads "equal") needs the
     full-AT timing run, still running on the other Hamster GPU.

  Caveats: one seed. Stage 1 trained on the derived S=256 copy (a disclosed
  preprocessing condition); stage 2 trained on original images. The timing
  is not a "no co-location" host measurement in the strict sense (point 2).
  Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `0828de5d...`,
  `sample-stats-train.parquet` sha256 `121923a9...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-03 (postrun): **Fastest-config timing, MobileNetV4-S full AT
  (50 epochs, cuda_graph) on Hamster completed.**
  `plan0103-mnv4s-fullat-50ep-lr0025-cg-s0-v1` (hand-run, Hamster, one RTX
  4090) ran 50/50 epochs. It is the full-AT half of the fastest-config
  timing (human, 2026-10-03); it started the same second as the Hamster
  stage 1. Its launch was not logged in this plan.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` and `epoch-049.pt` are on disk (best =
  last = epoch 49). Nothing is imported into `docs/experiments/`: no
  aggregator exists for this contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest.

  Fixed identity: as the single-stage reference
  `plan0104-mobilenetv4-random-init-lr0025-v1` (plan 0104, source
  `76280eb`) except `cuda_graph: true` and the source SHA. ImageNet-1k
  (original images, content `ae033613...`), MobileNetV4-Conv-S random init,
  plain PGD-AT, no teacher, 224 px training. Training and evaluation-attack
  seed 0 (split seed 20260911). Training attack CE PGD-3, l_inf eps 4/255,
  step 8/765, random start, eval mode. Selection and validation attack
  PGD-10, same eps and step, random start, eval mode. SGD (Nesterov,
  momentum 0.9, wd 1e-4), peak LR 0.025, warmup_multistep (10-epoch warmup,
  x0.1 at epochs 25 and 38), 50 epochs. World size 1, global batch 128,
  local BN. Deterministic, not compiled, cuda_graph, fp32, 16 workers.
  Config `imagenet_mobilenetv4_pgd_at_random_init_lr0025_cg.yaml`, protocol
  `controlled_imagenet_stage02_init_lr_grid_v1`, config hash `75d6cd92...`.
  Source SHA `65369a36668b70c789c3e0afbf8f5b3a4870bfc7` (worktree
  `source-65369a36668b`, clean). "val" is internal validation (25,620
  held-out original train images, 224 px, the same slice as the two-stage
  stage 2); "probe" is 2,000 training images. Neither is the ImageNet val
  split. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.81 / 0.60 | 0.85 / 0.75 | 6.831 |
  | 9 | end of warmup | 31.59 / 16.45 | 33.05 / 15.95 | 4.833 |
  | 24 | end of lr 0.025 | 41.59 / 22.29 | 43.75 / 23.20 | 4.330 |
  | 25 | first epoch at lr 0.0025 | 49.11 / 27.95 | 53.15 / 29.65 | 4.016 |
  | 37 | end of lr 0.0025 | 52.19 / 29.38 | 56.30 / 32.85 | 3.827 |
  | 38 | first epoch at lr 0.00025 | 53.25 / 30.19 | 58.95 / 33.85 | 3.744 |
  | 49 | last = best (by val PGD-10) | 53.85 / 30.71 | 59.80 / 34.40 | 3.697 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **Same result as the eager reference.** Last-epoch val 53.85 / 30.71
     equals `plan0104-mobilenetv4-random-init-lr0025-v1` (53.85 / 30.71)
     to two decimals. The other values recorded for the reference also
     match: epoch 24 (41.59 / 22.29), epoch 37 (52.19 / 29.38), the two
     decay jumps (+5.66 and +0.82pt PGD-10) and the last probe
     (59.80 / 34.40). Only these logged values were compared; the
     reference per-epoch rows were not read here. So in deterministic mode
     cuda_graph did not change the logged results of this recipe, and the
     reference 30.71 used by every two-stage verdict stands.
  2. **Wall-clock (the purpose of this run).** 15.45 h from start to
     finish (run-bundle `created_at` 2026-10-02 16:15:59 to `finished_at`
     2026-10-03 07:42:45 UTC, 55,606 s). Summed per-epoch training time is
     52,108 s (14.47 h); the remaining 3,498 s (about 70 s per epoch) is
     validation, probe, checkpointing and start-up. Throughput 1,199-1,207
     img/s in every epoch; no slow stretch, including the hours when the
     Hamster stage 1 slowed down in its epochs 24-27. cuda_graph: 1
     capture, 2 eager steps and 9,807 replays per epoch. Peak allocated
     memory 3.04 GB; reserved 7.62 GB at epoch 0, 8.40 GB at epoch 49.
  3. **Co-location.** The other Hamster GPU ran the stage 1 (16:16-19:43
     UTC), the failed stage 2 `-v1` (19:43-20:26) and the stage 2 `-v2`
     (20:41-02:59), i.e. up to about epoch 34 of this run. What ran there
     after 02:59 UTC was not checked. Throughput in epochs 35-49
     (1,205-1,207 img/s) is within 0.5% of epochs 9-34 (1,201-1,205), so
     the neighbour did not measurably change the speed.
  4. **Wall-clock comparison (rule of decision packet 0030, option A,
     written before this run finished; 0030's `chosen` is still null).**
     Two-stage = Hamster stage 1 (12,410 s) + Hamster stage 2 `-v2`
     (22,691 s) = 35,101 s, 9.75 h. Full AT = 55,606 s, 15.45 h. Ratio
     two-stage / full AT = 0.631, outside 0.9-1.1, so **two-stage is
     faster**: it takes 63% of the full-AT time, 5.70 h less. All three
     runs used source `65369a3`, the same GPU type, cuda_graph and
     deterministic mode, with a neighbour run on the other GPU for most or
     all of their time. The time between stage 1 and stage 2 is not
     counted.
  5. **Accuracy side (already reported, not a new verdict).** The
     fastest two-stage arm ends at 53.13 / 29.84 (Hamster stage 2 `-v2`),
     i.e. -0.72 clean / -0.87 PGD-10 against this run. That arm (stage 1
     of 50 epochs on S=256) is one of the post-hoc variants of
     2026-10-03; the preregistered 195-epoch verdict is "worse". Whether
     two-stage becomes the Phase 1 method is decided by the preregistered
     generalisation check (MobileNetV4-M and EfficientNet-B0 both within
     1pt of their reference), not by this ratio.
  6. **Training is normal.** Val PGD-10 never falls more than 0.60pt
     below its running maximum (epoch 18); the catastrophic-overfitting
     guard never fires. Val SE is about 0.31pt clean and 0.29pt PGD-10.
     Decays: epoch 24 to 25 +7.52 clean / +5.66 PGD-10; epoch 37 to 38
     +1.06 / +0.82. The last stage is almost flat (30.19-30.71%, epochs
     45-49 within 30.61-30.71%).
  7. **Best and last are the same epoch** (49); `robust_overfit_gap` 0.0.
  8. **Seen-unseen gap.** At the last epoch probe minus val is +5.9 clean
     and +3.7 PGD-10.
  9. **Context only, not a controlled speed-up.** The eager reference
     took 21.29 h (76,662 s, 2026-09-26 11:35 to 09-27 08:53 UTC) at
     841-878 img/s on Hamster with source `76280eb`; its co-location was
     not checked. This run's throughput is about 1.40x the reference's, and
     the reference took 1.38x as long.

  Caveats: one seed. The timing is not a "no co-location" host
  measurement in the strict sense (points 3 and 4); the human's primary
  cost metric asks for no co-location. The two-stage time uses stage 1 on
  the derived S=256 copy. Checkpoint sha256 was not computed in this
  postrun. `epoch-metrics.parquet` sha256 `0b797039...`,
  `sample-stats-train.parquet` sha256 `a1e5167b...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-05 — **Verdicts.** Seed 0, internal val, last epoch, clean /
  PGD-10.
  - Fastest-config timing on Hamster, MobileNetV4-S (decisions 0030-0033):
    - full AT, 50 epochs, cuda_graph: 53.85 / 30.71, bitwise equal to the
      eager run; 15.45 h wall.
    - two-stage 50 + 20 epochs, cuda_graph: 53.13 / 29.84, bitwise equal
      to the Ferret run; 3.45 + 6.30 = 9.75 h.
    - Ratio 0.63.
    - Stage 2 -v1 failed from a co-located test's OOM; -v2 was rerun.
  - **Stage-2 length (preregistered rule: shortest length with PGD-10 >=
    29.34):** 5 epochs 51.57 / 28.77 (fails); 10 epochs 52.18 / 29.38
    (passes). So 10 epochs would be adopted if two-stage were used.
  - **Two-stage generalisation (preregistered rule: both models within
    1pt of their full-AT 50-epoch reference).**
    - MobileNetV4-M: 63.04 / 37.99 vs 39.39, i.e. -1.40. **Fails.**
    - EfficientNet-B0 stage 2 OOMed at 224 px with cuda_graph. Peak was
      about 22.6 GB: the graph's private pool plus the eager allocations.
      Stage 1 (112 px) ran: 49.97 / 24.74. The verdict does not depend on
      it, because MobileNetV4-M already fails.
    - **Phase 1 therefore uses full AT, 50 epochs.**
    - Limitation: cuda_graph does not fit EfficientNet-B0 at 224 px,
      batch 128, on 24 GB. EfficientNet-B0 at 224 runs without cuda_graph.
  - **Phase 1 grid launched (option A: 3 LRs per model, full AT 50
    epochs, random init, seed 0).**
    - Best-vs-best uses last-epoch internal-val PGD-10 to pick each
      model's LR. Official-val evaluation (clean, PGD, AutoAttack in a
      separate process) is run only on the chosen LR.
    - Complete already:
      - MobileNetV4-S: 0.0125 / 0.025 / 0.05 / 0.1, giving 29.16 / 30.71
        / (plan 0104) / 29.18.
      - MobileNetV4-M: 0.0125 / 0.025 / 0.05, giving 37.00 / 39.39 /
        39.10.
      - EfficientNet-B0: 0.025, giving 35.93.
    - New, non-deterministic with cudnn_benchmark (determinism dropped,
      human 2026-10-03):
      - ConvNeXt-Atto and DeiT-Tiny, AdamW 1.25e-4 / 2.5e-4 / 5e-4
        (`imagenet_{convnext_atto,deit_tiny}_pgd_at_phase1_adamw_random_lr*.yaml`).
      - EfficientNet-B0 0.0125 / 0.05: same identity as its 0.025 run,
        i.e. compile and non-deterministic.
    - MobileViT-S: only if time allows.
- 2026-10-06 (postrun): **Phase 1 LR grid, DeiT-Tiny AdamW random init /
  lr 5e-4 completed.** `plan0103-phase1-deit-tiny-adamw-random-lr5em4-v1`
  (hand-run, Hamster, one RTX 4090) ran 50/50 epochs.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). `best.pt`, `last.pt` and `epoch-049.pt` are on disk (best =
  epoch 45, last = epoch 49). Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract. The numbers
  below are read from `outputs/train/epoch-metrics.jsonl` and the
  run-bundle manifest.

  Fixed identity: ImageNet-1k (original images, content `ae033613...`),
  DeiT-Tiny (`deit_tiny_imagenet`, `imagenet_standard` profile) random
  init, plain PGD-AT, no teacher, 224 px training. Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack CE PGD-3,
  l_inf eps 4/255, step 8/765, random start. Selection and validation
  attack PGD-10, same eps and step, random start, eval mode. AdamW (betas
  0.9/0.999, wd 0.05), peak LR 5e-4, warmup_multistep (10-epoch warmup,
  x0.1 at epochs 25 and 38), 50 epochs. World size 1, global batch 128,
  local BN mode (DeiT has no BN). Non-deterministic with cudnn_benchmark,
  not compiled, no cuda_graph, fp32, 16 workers. Config
  `imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr5em4.yaml`, protocol
  `controlled_imagenet_stage02_lightweight_architecture_survey_v1`, config
  hash `0430192e...`. Source SHA `609e6fa913d7b0f20ca8e9143f2a8604f68ea77f`
  (worktree `source-609e6fa913d7`, clean). "val" is internal validation
  (25,620 held-out original train images, 224 px); "probe" is 2,000
  training images. Neither is the ImageNet val split. No AutoAttack has
  run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 3.07 / 1.80 | 2.70 / 1.75 | 6.644 |
  | 9 | end of warmup | 29.25 / 14.58 | 30.35 / 15.15 | 5.070 |
  | 24 | end of lr 5e-4 | 43.25 / 23.18 | 45.65 / 24.90 | 4.310 |
  | 25 | first epoch at lr 5e-5 | 51.58 / 29.44 | 54.80 / 31.70 | 3.887 |
  | 37 | end of lr 5e-5 | 55.16 / 32.01 | 59.35 / 36.40 | 3.593 |
  | 38 | first epoch at lr 5e-6 | 56.20 / 32.89 | 60.80 / 37.60 | 3.510 |
  | 45 | best (by val PGD-10) | 56.71 / 33.35 | 61.00 / 38.30 | 3.473 |
  | 49 | last | 56.85 / 33.27 | 61.15 / 38.45 | 3.465 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **First DeiT-Tiny Phase 1 cell; no grid verdict yet.** Last-epoch
     val PGD-10 is 33.27% (the selection metric of the Phase 1 grid).
     The lr 2.5e-4 cell was at epoch 47 of 50 on the other Hamster GPU
     at this postrun; the 1.25e-4 cell has not run. lr 5e-4 is the top
     edge of the approved grid, so if it stays the argmax, the optimum
     may lie above the grid.
  2. **AdamW trains DeiT-Tiny normally.** Val PGD-10 rises every LR
     stage and never falls more than 0.16pt below its running maximum
     (epoch 21); the catastrophic-overfitting guard never fires. There is
     no sign of the ~30%-clean failure seen in the SGD fixed-recipe runs
     (2026-09-28). The SGD DeiT-Tiny run was stopped at epoch 38 and is
     not a result, so no SGD-vs-AdamW number is given.
  3. **Decays.** Epoch 24 to 25: +8.33 clean / +6.26 PGD-10. Epoch 37 to
     38: +1.04 / +0.88. The lr 5e-5 stage still rises by +2.57pt PGD-10
     after its first epoch. The last stage is almost flat: epochs 38-49
     within 32.89-33.35%, epochs 45-49 within 33.21-33.35%.
  4. **Best and last.** Best epoch 45 (56.71 / 33.35), last epoch 49
     (56.85 / 33.27); `robust_overfit_gap` 0.07pt. Val SE is about 0.31pt
     clean and 0.29pt PGD-10, so best and last are not distinguishable.
  5. **Seen-unseen gap.** At the last epoch probe minus val is +4.3 clean
     and +5.2 PGD-10.
  6. **Context, other Phase 1 models (last val PGD-10, each at its best
     LR so far, not yet best-vs-best).** MobileNetV4-S 30.71 (lr 0.025),
     MobileNetV4-M 39.39 (lr 0.025), EfficientNet-B0 35.93 (lr 0.025,
     the other two LRs pending). Those runs use SGD; this one uses AdamW
     per the family recipe.
  7. **Cost.** 33.00 h from start to finish (run-bundle `created_at`
     2026-10-05 03:50:28 to `finished_at` 2026-10-06 12:50:43 UTC,
     118,815 s). Summed per-epoch training time is about 112,460 s
     (31.2 h); the rest (about 127 s per epoch) is validation, probe,
     checkpointing and start-up. Throughput 557-559 img/s in every epoch,
     no slow stretch. The other Hamster GPU ran the lr 2.5e-4 cell for
     the whole run. Peak allocated memory 4.64 GB; reserved 5.02 GB at
     epoch 0, 6.43 GB from epoch 1.

  Caveats: one seed, one LR of three. Non-deterministic mode, so a rerun
  would not reproduce these numbers bit for bit. Checkpoint sha256 was not
  computed in this postrun. `epoch-metrics.parquet` sha256 `78cdce7d...`,
  `sample-stats-train.parquet` sha256 `0e9c2132...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-06 — Phase 1 AdamW grid, interim results (internal val, last
  epoch, clean / PGD-10).
  - ConvNeXt-Atto: 2.5e-4 gives 55.98 / 34.23; 5e-4 gives 58.83 / 36.35.
  - DeiT-Tiny: 1.25e-4 gives 49.18 / 27.17 (epoch 45); 2.5e-4 gives
    54.54 / 31.24 (epoch 49); 5e-4 gives 56.85 / 33.27.
  - For both models the argmax sits at the top grid point, so a 1e-3 cell
    is added for each, on the idle Ferret GPUs 1 and 2. The MobileNetV4-S
    grid also extended past its first range (0.1).
  - Seed 0. The cuda_graph production-shape check: 5/6 cases passed. The
    MobileNetV4-M 224 px non-deterministic case failed: a graph arm landed
    0.026 of a step from every eager outcome. So non-deterministic +
    cuda_graph is not used in production; cuda_graph runs stay
    deterministic (bitwise proven).
- 2026-10-06 (postrun): **Phase 1 LR grid, DeiT-Tiny AdamW random init /
  lr 2.5e-4 completed.** `plan0103-phase1-deit-tiny-adamw-random-lr2p5em4-v1`
  (hand-run, Hamster GPU1, one RTX 4090) ran 50/50 epochs.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning; the wandb internal log has one retried HTTP 503). `best.pt`,
  `last.pt` and `epoch-049.pt` are on disk (best = epoch 45, last =
  epoch 49). Nothing is imported into `docs/experiments/`: no aggregator
  exists for this contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest.

  Fixed identity: identical to the lr 5e-4 cell (2026-10-06 postrun above)
  except the peak LR. ImageNet-1k (original images, content
  `ae033613...`), DeiT-Tiny (`deit_tiny_imagenet`, `imagenet_standard`
  profile) random init, plain PGD-AT, no teacher, 224 px training.
  Training and evaluation-attack seed 0 (split seed 20260911). Training
  attack CE PGD-3, l_inf eps 4/255, step 8/765, random start. Selection
  and validation attack PGD-10, same eps and step, random start, eval
  mode. AdamW (betas 0.9/0.999, wd 0.05), peak LR 2.5e-4,
  warmup_multistep (10-epoch warmup, x0.1 at epochs 25 and 38), 50
  epochs. World size 1, global batch 128, local BN mode (DeiT has no BN).
  Non-deterministic with cudnn_benchmark, not compiled, no cuda_graph,
  fp32, 16 workers. Config
  `imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr2p5em4.yaml`, protocol
  `controlled_imagenet_stage02_lightweight_architecture_survey_v1`, config
  hash `747462db...`. Source SHA `609e6fa913d7b0f20ca8e9143f2a8604f68ea77f`
  (worktree `source-609e6fa913d7`, clean). "val" is internal validation
  (25,620 held-out original train images, 224 px); "probe" is 2,000
  training images. Neither is the ImageNet val split. No AutoAttack has
  run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 2.02 / 1.32 | 1.90 / 1.25 | 6.733 |
  | 9 | end of warmup | 26.81 / 13.19 | 27.40 / 13.60 | 5.201 |
  | 24 | end of lr 2.5e-4 | 44.51 / 23.88 | 47.30 / 25.85 | 4.237 |
  | 25 | first epoch at lr 2.5e-5 | 50.34 / 28.54 | 54.25 / 31.65 | 3.896 |
  | 37 | end of lr 2.5e-5 | 53.57 / 30.61 | 58.40 / 34.40 | 3.659 |
  | 38 | first epoch at lr 2.5e-6 | 54.24 / 30.98 | 58.90 / 35.70 | 3.602 |
  | 45 | best (by val PGD-10) | 54.48 / 31.30 | 59.10 / 35.85 | 3.578 |
  | 49 | last | 54.57 / 31.25 | 59.45 / 36.40 | 3.574 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **Second DeiT-Tiny Phase 1 cell: lr 5e-4 beats lr 2.5e-4 by 2.02pt
     last-epoch val PGD-10** (33.27 vs 31.25; clean 56.85 vs 54.57,
     +2.28). The two-run val SE of the difference is about 0.41pt, so the
     gap is about 5 SE; it is still one seed per cell. The lr 1.25e-4
     cell had not finished at this postrun (49.18 / 27.17 at epoch 45 in
     the interim entry above), so the 3-point argmax is not final.
  2. **Packet 0034 option B condition.** Between the two finished cells,
     5e-4 is the argmax and leads 2.5e-4 by 2.02pt (>= 1.0pt). 0034
     `chosen` is still null; the lr 1e-3 DeiT-Tiny cell was already added
     (interim entry above, commit `b8f9101`). That addition was made after
     the 2.5e-4 cell's epoch-48 numbers were known, so for DeiT-Tiny the
     rule is post hoc, as 0034 B already says it would be in that case.
  3. **Correction to the interim entry above.** It lists 2.5e-4 as
     "54.54 / 31.24 (epoch 49)"; those are the epoch-48 values. The
     epoch-49 (last) values are 54.57 / 31.25. The difference (at most
     0.03pt) does not change any reading.
  4. **Trains normally.** Val PGD-10 rises in every LR stage and never
     falls more than 0.14pt below its running maximum (epoch 34); the
     catastrophic-overfitting guard never fires.
  5. **Decays.** Epoch 24 to 25: +5.83 clean / +4.66 PGD-10. Epoch 37 to
     38: +0.68 / +0.37. The lr 2.5e-5 stage still rises by +2.06pt
     PGD-10 after its first epoch. The last stage is almost flat: epochs
     38-49 within 30.98-31.30%, epochs 39-49 within 31.18-31.30%. Both
     decay jumps are smaller than in the 5e-4 cell (+6.26 and +0.88).
  6. **Best and last.** Best epoch 45 (54.48 / 31.30), last epoch 49
     (54.57 / 31.25); `robust_overfit_gap` 0.05pt. Val SE is about 0.31pt
     clean and 0.29pt PGD-10, so best and last are not distinguishable.
  7. **Seen-unseen gap.** At the last epoch probe minus val is +4.9 clean
     and +5.2 PGD-10 (5e-4 cell: +4.3 / +5.2).
  8. **Cost.** 33.76 h from start to finish (run-bundle `created_at`
     2026-10-05 03:55:43 to `finished_at` 2026-10-06 13:41:01 UTC,
     121,518 s). Summed per-epoch training time is about 115,050 s
     (32.0 h); the rest (about 129 s per epoch) is validation, probe,
     checkpointing and start-up. Throughput 545-547 img/s in every epoch,
     no slow stretch; about 2% below the 5e-4 cell (557-559 img/s) on the
     other GPU of the same host, cause not checked. The other Hamster GPU
     ran the lr 5e-4 cell until 2026-10-06 12:50 UTC, then EfficientNet-B0
     lr 0.05. Peak allocated memory 4.64 GB; reserved 5.02 GB at epoch 0,
     6.43 GB from epoch 1.

  Caveats: one seed, two LRs of three (plus the added 1e-3) finished.
  Non-deterministic mode, so a rerun would not reproduce these numbers
  bit for bit. Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `83121f50...`,
  `sample-stats-train.parquet` sha256 `c9d89771...` (from the run-bundle
  manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-06 (human): the AdamW grid moves up for ConvNeXt-Atto and
  DeiT-Tiny, to {2.5e-4, 5e-4, 1e-3}. If 1e-3 is the best point, 2e-3 is
  added.
  - The queued ConvNeXt-Atto 1.25e-4 cell is cancelled.
  - DeiT-Tiny 1.25e-4 was nearly finished, so it is kept as a reference.
  - ConvNeXt-Atto 2e-3 takes the cancelled cell's slot on Ferret GPU0. It
    starts pre-emptively, because 2.5e-4 -> 5e-4 gained +2.1pt PGD-10.
- 2026-10-08 (chat): **Official evaluation of the Phase 1 chosen-LR cells
  starts on Anteater** (GPUs 2 and 3 only, RTX 2080 Ti), launched
  2026-10-07 15:11Z.
  - Runs: MobileNetV4-S random-init lr 0.025
    (`plan0104-mobilenetv4-random-init-lr0025-v1`, copied from Hamster) on
    GPU2, and MobileNetV4-M random-init lr 0.025
    (`plan0103-phase1-mobilenetv4-conv-medium-random-v1`, copied from
    Ferret through Hamster) on GPU3. `best.pt`, `last.pt`,
    `resolved_config.yaml` and `run-bundle/` were copied, and the sha256 of
    every file matches the source.
  - Source: pinned worktree `source-e1d9c3b1916a` (commit `e1d9c3b`),
    `ard.cli.evaluate` in a separate process with `--allow-autoattack`.
  - Evaluation identity is the same as the 2026-09-27 Ferret official
    evaluation: the training config's own `evaluation:` block (official
    val 50k, eps 4/255, PGD-10, step 8/765, random start), plus
    `configs/evaluation/autoattack_saved_checkpoint.yaml` (best and last,
    seed 0, AutoAttack standard, batch 128), plus
    `autoattack_sample_count: 500`. The only change is
    `evaluation.dataset.root`, which points at Anteater's SSD copy. The
    val manifest digest (`abc0ee80...`) is still checked, and a mismatch
    stops the run. A diff of the resolved evaluation block against the
    earlier Ferret evaluation shows only the root path. The overlay
    carries the full dataset mapping, because the evaluation overlay is
    merged one level deep and a `dataset.root` override alone would drop
    the split and digest.
  - Outputs go to `runs/<run id>/outputs/train/evaluation-official-n500/`
    under Anteater's `~/workspace-local/ard-runtime/ard_codex_bootstrap`.
    W&B project `lightweight-imagenet-at`.
  - The queue is
    `queues/official_eval_queue.sh <gpu> <run id> ...`. It runs one job at
    a time per GPU (a per-GPU lock), skips runs that already have results,
    and logs start, end and rc to `queues/official_eval_gpu<gpu>.log`.
    ConvNeXt-Atto, DeiT-Tiny and EfficientNet-B0 will be added when their
    grids finish.
  - Caveat: the 2080 Ti has no TF32, so convolutions run in full fp32
    rather than TF32 as on the 4090s. Numbers may differ very slightly
    from rows evaluated on Ferret. The evaluation settings are unchanged.
- 2026-10-08 (chat): **Lighter per-epoch validation for Phase 2** (human
  decision). New option `training.selection_subset_size` (default off).
  - What it does: per-epoch selection (clean + PGD-10, so `best.pt` and the
    catastrophic-overfitting check) runs on a fixed subset of the held-out
    split instead of all ~25.6k images. The subset is class-stratified
    (5 images per class at 5000) and seeded by `seeds.split`, so it is the
    same every epoch and for every seed of one split. The training data and
    the held-out split are unchanged; `validation_fraction` keeps its meaning.
    The train probe is unchanged. It cannot be combined with ADR's
    gap-adaptive lambda (which reads the selection metric), so the trained
    weights are the same with or without it.
  - Names: with the option on, the per-epoch numbers are `val_subset_*` in
    the epoch rows and `best_subset_*` / `last_subset_*` /
    `robust_overfit_gap_subset` in the run summary. The Phase-1 names
    (`val_pgd_accuracy`, `best_pgd_accuracy`, ...) are absent, so nothing can
    read a subset number as a full-split one. **The catastrophic-overfitting
    check for a Phase 2 run reads `val_subset_pgd_accuracy`.** Every row and
    the summary also carry the subset size and digest.
  - At the final epoch, last and `best.pt` are both evaluated on the full
    held-out split and on its complement (held-out minus the subset, about
    20.6k images at 5000), with the same random starts: `val_full_*`,
    `val_complement_*`, `best_val_full_*`, `best_val_complement_*` (epoch
    row), `full_split_final` (selection metadata of `last.pt`) and
    `best_full_*`, `last_full_*`, `best_complement_*`, `last_complement_*`
    (run summary). The complement numbers are independent of the images
    that chose `best.pt`. The official evaluation (`ard.cli.evaluate`) is
    unchanged.
  - Comparing across phases: compare "best" between Phase 1 and Phase 2 only
    with the official evaluation, because the two phases choose `best.pt`
    differently. Inside training, `last_full_*` is the like-for-like number
    (last weights do not depend on the option).
  - The option is in the config hash and in the evaluation pooling identity
    (`selection_subset_size` and `selection_subset_sha256`), so runs with and
    without it never pool. Existing configs hash byte-identically.
  - **Phase 2 template:** add `selection_subset_size: 5000` to the
    `training:` block of every Phase 2 config. Basis: Singh/Croce/Hein 2023
    select on 4k held-out images, Gowal 2020 on 1,024, and over 29 of our
    runs best minus last is 0.1pt (median).
- 2026-10-08 (human-approved): **Phase 2 implementation batch A** (code only; nothing launched).
  All options default off; every earlier config resolves and hashes byte-identically (tested against
  `1a36f57`, after batches B and D). Contracts in `docs/SCIENTIFIC_INVARIANTS.md` ("Plan 0103 Phase 2 batch A training options").
  - `method.mixed_batch` (pgd_at): Kurakin et al. 2017 mixed clean+adversarial batch. The first
    `floor(fraction * m)` positions of each batch are attacked; loss weight `adversarial_weight` on them.
    Optional `split_batchnorm`: the clean half uses an auxiliary BN; the main BN (used by attack,
    validation, selection, saved weights and evaluation) is the adversarial BN. BatchNorm students only.
  - `method.awp` (pgd_at): AWP after the official AT-AWP code (gamma 0.01, warmup 0 by default).
  - `optimizer.exclude_norm_bias_from_weight_decay` (SGD): no weight decay on ndim <= 1 parameters.
  - `training.cuda_graph` now also admits `training.weight_ema_decay` (bitwise equal to eager in
    deterministic mode), `method.label_smoothing` and the SGD exclusion; it refuses mixed batch and AWP
    (plan 0105 progress log).
  - Review fixes (scientific review of 5c9e6cf, no P0/P1): split BN refuses any 1-example BN sub-batch
    (schema for the full batch, `ard.cli.train` for the epoch's last batch; never drops examples). The
    ImageNet-1k Phase 2 split has 1,255,547 training images, so per-rank 128 with f 0.5 gives 64/64 and a
    last batch of 123 (61/62): admitted. Mixed batch is single-GPU only. csdongxian/AWP is pinned in
    `external.lock.yaml` (`awp`, a7acf5d8) and the parity test imports its `utils_awp.py`; our AWP is
    "AWP on our PGD-AT" (eval-mode attack by default), not an AT-AWP reproduction. Also: split BN with
    `init_checkpoint` refused; preflight refusals before any tracker run; `train_mixed_batch_weighted_loss`.
- 2026-10-08 (human-approved): **Phase 2 batch C (augmentation x epochs 2x2 on
  MobileNetV4-S), code and configs ready, not launched.**
  - Option `dataset.imagenet_augmentation` (`standard` default and not serialized,
    so every earlier config hashes byte-identically). Values:
    - `idbh_weak_nocropshift` (**the chosen arm**, human 2026-10-08):
      RandomResizedCrop -> flip -> ColorShape('color') -> ToTensor ->
      RandomErasing(p=0.5). Upstream IDBH (Li & Spratling, ICLR 2023) uses
      CropShift *instead of* CIFAR's pad-and-crop. On ImageNet,
      RandomResizedCrop plays that role, so stacking CropShift on top of it has
      no precedent.
    - `idbh_weak` / `idbh_strong` (implemented, not used): the same plus
      CropShift as a *fraction-preserving adaptation*, not upstream itself.
      The level is drawn as upstream (U{0..10}) and becomes round(k x size / 32)
      px, i.e. 0-70 px in steps of 7 at 224. Erasing p = 0.5 / 1.0.
    - Upstream precedent: the code has no ImageNet version. Its Tiny-ImageNet
      (64 px) run reused the 32 px CIFAR PRN18 parameters with CropShift
      unscaled (paper App. C/D, Tab. 12; "not optimized" for TIN).
  - Colour factors, shear (radians), rotation (degrees) and erasing
    area/aspect are kept verbatim. Known deviation: Sharpness (PIL 3x3 kernel)
    and NEAREST shear/rotate act per pixel, so at 224 px they are effectively
    weaker than at 32 px.
  - Determinism: every layer draws from its own substream keyed by
    (augmentation seed, epoch, source id). Crop and flip are bit-identical to
    the standard path, and nothing uses the global RNG, so FKD-style
    precomputation is possible.
  - Identity: in the config hash and in the evaluation `training_protocol_identity`
    (`imagenet_augmentation`). An end-to-end test (train, `ard.cli.evaluate`,
    `summarize_checkpoint_groups`) shows an IDBH run is refused when pooled with a
    standard run.
  - Protocol id: unchanged (`controlled_imagenet_stage02_init_lr_grid_v1`), as
    for Phase 2 batches B and D on the same baseline. The arms share the
    baseline's scientific contract (data split, threat model, evaluation);
    what differs is carried by the config hash and the identity entry above.
    A separate protocol id would put one Phase 2 batch under a different
    contract from the others and from its own baseline.
  - Configs (cuda_graph kept: the augmentation is data-side only;
    `selection_subset_size: 5000` per the Phase 2 template; variant in the W&B
    group):
    - `imagenet_mobilenetv4_pgd_at_random_init_lr0025_idbh_weak_nocropshift_cg.yaml` (50 ep).
    - `imagenet_mobilenetv4_pgd_at_random_init_lr0025_idbh_weak_nocropshift_100ep_cg.yaml`
      (100 ep, milestones [50, 76], warmup 10).
    - `imagenet_mobilenetv4_pgd_at_random_init_lr0025_100ep_cg.yaml` (standard, 100 ep).
  - **Comparing the cells.** The standard-50 cell is the Phase 1 run
    `imagenet_mobilenetv4_pgd_at_random_init_lr0025_cg.yaml`, whose best.pt was
    picked on the full held-out split; the other three pick best.pt on the
    5000-image subset. The clean comparison is therefore last.pt,
    `last_full_*` and the official evaluation (best and last on the official
    val). Training-batch accuracy is measured on augmented images, so it is
    not comparable across augmentation arms; compare fitting with the train
    probe (un-augmented eval-transform views).
  - CPU cost (128 real ImageNet JPEGs, one process, decode + transform, Hamster
    under load): standard 222 img/s, `idbh_weak_nocropshift` 199 img/s (-11%),
    `idbh_weak` 193 img/s. Transform-only numbers are printed by the unit
    benchmark.
- 2026-10-08 (postrun): **Phase 1 LR grid, EfficientNet-B0 random init /
  SGD lr 0.05 completed.** `plan0103-phase1-efficientnet-b0-random-lr005-v1`
  (hand-run, Hamster GPU0, one RTX 4090) ran 50/50 epochs.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning and the inductor TF32 notice); the wandb internal log has no
  error or retry line. `best.pt`, `last.pt` and `epoch-049.pt` are on disk
  (best = epoch 48, last = epoch 49). Nothing is imported into
  `docs/experiments/`: no aggregator exists for this contract. The numbers
  below are read from `outputs/train/epoch-metrics.jsonl` and the
  run-bundle manifest.

  Fixed identity: ImageNet-1k (original images, content `ae033613...`),
  EfficientNet-B0 (`efficientnet_b0_imagenet`, `imagenet_standard`
  profile) random init, plain PGD-AT, no teacher, 224 px training.
  Training and evaluation-attack seed 0 (split seed 20260911). Training
  attack CE PGD-3, l_inf eps 4/255, step 8/765, random start. Selection
  and validation attack PGD-10, same eps and step, random start, eval
  mode. SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.05,
  warmup_multistep (10-epoch warmup, x0.1 at epochs 25 and 38), 50
  epochs. World size 1, global batch 128, local BN. Non-deterministic with
  cudnn_benchmark, compiled (`torch.compile`), no cuda_graph, fp32.
  Config `imagenet_efficientnet_b0_pgd_at_phase1_random_lr005.yaml`
  (identical to the lr 0.025 config except the LR and the W&B group),
  protocol `controlled_imagenet_stage02_lightweight_architecture_survey_v1`,
  config hash `4467459b...`. Source SHA
  `609e6fa913d7b0f20ca8e9143f2a8604f68ea77f` (worktree
  `source-609e6fa913d7`, clean). "val" is internal validation (25,620
  held-out original train images, 224 px); "probe" is 2,000 training
  images. Neither is the ImageNet val split. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.91 / 0.66 | 0.55 / 0.25 | 6.835 |
  | 9 | end of warmup | 35.43 / 19.54 | 35.10 / 19.30 | 4.619 |
  | 24 | end of lr 0.05 | 41.93 / 23.46 | 43.10 / 24.30 | 4.298 |
  | 25 | first epoch at lr 0.005 | 51.83 / 31.44 | 53.45 / 33.60 | 3.862 |
  | 37 | end of lr 0.005 | 53.92 / 32.49 | 56.00 / 35.30 | 3.697 |
  | 38 | first epoch at lr 0.0005 | 56.58 / 35.13 | 59.20 / 38.85 | 3.540 |
  | 48 | best (by val PGD-10) | 57.79 / 36.00 | 61.60 / 38.80 | 3.460 |
  | 49 | last | 57.86 / 35.88 | 61.30 / 39.55 | 3.460 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **lr 0.05 ties lr 0.025 on last-epoch val PGD-10** (35.88 vs 35.93,
     -0.05pt). Val SE is about 0.31pt clean and 0.30pt PGD-10, so the
     two-run SE of the difference is about 0.42pt; the gap is about 0.1
     SE. Only the lr 0.025 PGD-10 (35.93) is recorded in this plan; its
     clean and best-epoch numbers were not re-read in this postrun (the
     run lives on Ferret). The lr 0.025 run used a different host (Ferret)
     and source SHA (`086072b`); the config differs only in LR and group.
  2. **Grid state.** lr 0.0125 (`plan0103-phase1-efficientnet-b0-random-lr00125-v1`,
     Hamster GPU1, same source `609e6fa`) is at epoch 47 of 50, so the
     3-point argmax is not final. Of the two finished cells 0.025 is
     higher by 0.05pt, inside noise.
  3. **Edge-extension rule.** The lr 0.1 config (commit `5b6b443`)
     carries the human rule "extend when the top grid point is best"
     (2026-10-08). On last-epoch val PGD-10 the top point (0.05) is not
     the argmax, by 0.05pt; the two are tied. Whether that tie triggers
     the 0.1 cell is the human's decision (decision packet).
  4. **Trains normally.** Val PGD-10 never falls more than 0.69pt below
     its running maximum (epoch 18, 22.44 after 23.13 at epoch 16); the
     catastrophic-overfitting guard (below half of the running maximum)
     never fires. The lr 0.05 stage flattens: epochs 19-24 stay within
     22.97-23.47%.
  5. **Decays.** Epoch 24 to 25: +9.90 clean / +7.98 PGD-10. Epoch 37 to
     38: +2.66 / +2.64. The lr 0.005 stage is flat after its third epoch
     (epochs 27-37 within 32.29-32.78%). The last stage still creeps up:
     35.13 at epoch 38, 35.39-36.00 over epochs 39-49.
  6. **Best and last.** Best epoch 48 (57.79 / 36.00), last epoch 49
     (57.86 / 35.88); `robust_overfit_gap` 0.11pt, inside one val SE.
  7. **Seen-unseen gap.** At the last epoch probe minus val is +3.4 clean
     and +3.7 PGD-10, smaller than DeiT-Tiny's (+4.3 to +5.2).
  8. **Cost.** 48.62 h from start to finish (run-bundle `created_at`
     2026-10-06 12:51:16 to `finished_at` 2026-10-08 13:28:38 UTC,
     175,042 s). Summed per-epoch training time is about 165,570 s
     (46.0 h); the rest (about 189 s per epoch) is validation, probe,
     checkpointing and start-up. Throughput 409 img/s in epoch 0, then
     373-379 img/s (373-374 in epochs 28-29, cause not checked). The other
     Hamster GPU ran DeiT-Tiny lr 2.5e-4 until 2026-10-06 13:41 UTC, then
     EfficientNet-B0 lr 0.0125. Peak allocated memory 8.56 GB at epoch 0,
     8.03 GB after; reserved 10.29-10.59 GB.

  Caveats: one seed, two LRs of three finished. Non-deterministic mode,
  so a rerun would not reproduce these numbers bit for bit. Checkpoint
  sha256 was not computed in this postrun. `epoch-metrics.parquet` sha256
  `1532f1d8...`, `sample-stats-train.parquet` sha256 `6941081a...` (from
  the run-bundle manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-09 (postrun): **Phase 1 LR grid, EfficientNet-B0 random init /
  SGD lr 0.0125 completed.** `plan0103-phase1-efficientnet-b0-random-lr00125-v1`
  (hand-run, Hamster GPU1, one RTX 4090) ran 50/50 epochs.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`).
  `train.log` has no traceback, NaN or error line (only the wandb git-root
  warning). The wandb internal log has six transient `file_stream` retries
  (2026-10-06 20:08-20:21 and 2026-10-07 21:00 UTC); the local metrics are
  complete, and the W&B history was not checked. `best.pt`, `last.pt` and
  `epoch-049.pt` are on disk (best = epoch 48, last = epoch 49). Nothing is
  imported into `docs/experiments/`: no aggregator exists for this
  contract. The numbers below are read from
  `outputs/train/epoch-metrics.jsonl` and the run-bundle manifest.

  Fixed identity: same as the lr 0.05 cell above except the LR. ImageNet-1k
  (original images, content `ae033613...`), EfficientNet-B0
  (`efficientnet_b0_imagenet`, `imagenet_standard` profile) random init,
  plain PGD-AT, no teacher, 224 px training. Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack CE
  PGD-3, l_inf eps 4/255, step 8/765, random start. Selection and
  validation attack PGD-10, same eps and step, random start, eval mode.
  SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.0125,
  warmup_multistep (10-epoch warmup, x0.1 at epochs 25 and 38), 50
  epochs. World size 1, global batch 128, local BN. Non-deterministic with
  cudnn_benchmark, compiled (`torch.compile`), no cuda_graph, fp32.
  Config `imagenet_efficientnet_b0_pgd_at_phase1_random_lr00125.yaml`
  (differs from the lr 0.05 config only in the LR and the W&B group;
  unchanged since the source SHA), protocol
  `controlled_imagenet_stage02_lightweight_architecture_survey_v1`, config
  hash `433ed1bd...`. Source SHA
  `609e6fa913d7b0f20ca8e9143f2a8604f68ea77f` (worktree
  `source-609e6fa913d7`, clean). "val" is internal validation (25,620
  held-out original train images, 224 px); "probe" is 2,000 training
  images. Neither is the ImageNet val split. No AutoAttack has run.

  | epoch | stage | val clean / PGD-10 | probe clean / PGD-10 | train loss |
  |---|---|---:|---:|---:|
  | 0 | warmup | 0.25 / 0.17 | 0.35 / 0.10 | 6.906 |
  | 9 | end of warmup | 29.20 / 16.09 | 29.10 / 15.35 | 4.944 |
  | 24 | end of lr 0.0125 | 46.55 / 26.91 | 49.20 / 28.85 | 4.094 |
  | 25 | first epoch at lr 0.00125 | 51.99 / 31.55 | 55.20 / 33.65 | 3.803 |
  | 37 | end of lr 0.00125 | 54.64 / 33.14 | 58.50 / 37.70 | 3.640 |
  | 38 | first epoch at lr 0.000125 | 55.32 / 34.00 | 59.45 / 39.10 | 3.574 |
  | 48 | best (by val PGD-10) | 55.98 / 34.32 | 59.50 / 38.85 | 3.539 |
  | 49 | last | 56.08 / 34.25 | 59.85 / 39.65 | 3.539 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **The three-point grid is complete.** Last-epoch val PGD-10:
     0.0125 / 0.025 / 0.05 = 34.25 / 35.93 / 35.88. lr 0.0125 is 1.68pt
     below 0.025 and 1.63pt below 0.05 (about 4 two-run SE, two-run SE of
     a difference about 0.42pt). Clean: 56.08 vs 57.86 at 0.05 (-1.78);
     the 0.025 clean number is still not re-read (Ferret run).
  2. **Argmax.** On last-epoch val PGD-10 the argmax of the three points
     is 0.025, ahead of 0.05 by 0.05pt (inside noise). This is the rule
     preregistered as option A of packet 0036; 0036 `chosen` is still
     null, so the adopted LR and the lr 0.1 question remain the human's.
     The lower edge (0.0125) is clearly the worst point, so no lower-edge
     question arises.
  3. **Same shape as MobileNetV4-M.** 0.0125 / 0.025 / 0.05 =
     37.00 / 39.39 / 39.10 there; here 34.25 / 35.93 / 35.88. In both
     models 0.0125 is about 1.7-2.4pt below a flat 0.025-0.05 top.
  4. **Trains normally, but slower.** Val PGD-10 never falls more than
     0.20pt below its running maximum (epoch 47, 34.11 after 34.31 at
     epoch 45); the catastrophic-overfitting guard (below half of the
     running maximum) never fires. Unlike lr 0.05, the peak-LR stage has
     not flattened by epoch 24 (26.25 / 26.55 / 26.91 over epochs 22-24).
  5. **Decays.** Epoch 24 to 25: +5.44 clean / +4.64 PGD-10 (lr 0.05:
     +9.90 / +7.98). Epoch 37 to 38: +0.68 / +0.86. The lr 0.00125 stage
     creeps up (epochs 27-37 within 32.20-33.17%); the last stage is flat
     (34.00 at epoch 38, 33.98-34.32 over epochs 39-49).
  6. **Best and last.** Best epoch 48 (55.98 / 34.32), last epoch 49
     (56.08 / 34.25); `robust_overfit_gap` 0.07pt, inside one val SE.
  7. **Seen-unseen gap.** At the last epoch probe minus val is +3.8 clean
     and +5.4 PGD-10 (lr 0.05: +3.4 / +3.7).
  8. **Cost.** 49.30 h from start to finish (run-bundle `created_at`
     2026-10-06 13:41:33 to `finished_at` 2026-10-08 14:59:50 UTC,
     177,497 s). Summed per-epoch training time is about 167,930 s
     (46.6 h); the rest (about 191 s per epoch) is validation, probe,
     checkpointing and start-up. Throughput 410 img/s in epoch 0, then
     374-376 img/s, with two slow stretches (epochs 27-29 at 364-370 and
     epochs 39-41 at 346-360 img/s, cause not checked). The other Hamster
     GPU ran EfficientNet-B0 lr 0.05 until 2026-10-08 13:28 UTC. Peak
     allocated memory 8.56 GB at epoch 0, 8.03 GB after; reserved
     9.97 GB at epoch 0, 10.58-10.59 GB after.

  Caveats: one seed per LR; the 0.025 cell ran on a different host
  (Ferret) and source SHA (`086072b`). Non-deterministic mode, so a rerun
  would not reproduce these numbers bit for bit. Checkpoint sha256 was
  not computed in this postrun. `epoch-metrics.parquet` sha256
  `0b628558...`, `sample-stats-train.parquet` sha256 `1db471c4...` (from
  the run-bundle manifest). W&B `lightweight-imagenet-at`, same run id.
- 2026-10-09 — Phase 1 LR grids (internal val, last epoch PGD-10, seed 0).
  - EfficientNet-B0: 0.0125 / 0.025 / 0.05 give 34.25 / **35.93** / 35.88.
    The argmax is interior, so the grid is closed at 0.025.
  - ConvNeXt-Atto AdamW: 2.5e-4 / 5e-4 / 1e-3 / 2e-3 give 34.23 / 36.35 /
    **37.02** / 36.65. The argmax is interior, so the grid is closed at
    1e-3. The 1.25e-4 cell was cancelled.
  - DeiT-Tiny AdamW: 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 give 27.19 / 31.25 /
    33.27 / 35.00. The argmax is at the top edge, so 2e-3 was launched
    automatically on Ferret GPU2 (edge-extension rule, human 2026-10-08).
  - MobileNetV4-S and MobileNetV4-M stay at 0.025.
  - Seed-1 runs:
    - MobileNetV4-S 0.025 on Ferret GPU1 (running), then EfficientNet-B0
      0.025 queued on the same GPU.
    - MobileNetV4-M 0.025 on Hamster GPU0 (running).
    - ConvNeXt-Atto 1e-3 on Ferret GPU0 (launched).
    - DeiT-Tiny after its grid closes.
  - Distillation soft-label banks (K=500, FP16, seed 0, all 5 teachers)
    are being built on Anteater GPUs 2/3; the estimate is about 3 days.
- 2026-10-09 — Phase 2 pull queues armed (human: order group 1
  MobileNetV4-S -> ConvNeXt-Atto -> MobileNetV4-M / EfficientNet-B0).
  - Each host has `queues/work_queue.txt` and one `gpu_worker.sh` per GPU.
    A worker pops the next line only when no other queue script owns its
    GPU.
    - Hamster has 14 items, worktree source-5dd329eaf31f.
    - Ferret has 11 items, same SHA.
    - The items are split between hosts in the human's order.
  - ConvNeXt-Atto Phase 2 configs now carry the closed-grid LR 1e-3.
  - cuda_graph is now on for MobileNetV4-S mixed-batch and AWP.
  - DeiT-Tiny on Ferret GPU2 follows its own rule. After 2e-3 finishes:
    if 2e-3 beats 1e-3 (35.00), run 4e-3; otherwise start DeiT seed 1 at
    1e-3.
  - Distillation waits for the Anteater banks (~10/13); the human says
    there is no hurry.
