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
