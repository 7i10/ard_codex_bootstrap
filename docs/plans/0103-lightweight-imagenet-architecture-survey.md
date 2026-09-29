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
