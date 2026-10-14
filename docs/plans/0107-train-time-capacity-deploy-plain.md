# Plan 0107: train-time capacity, deploy-time plain network (proposed-method direction)

Human-approved 2026-10-15 (discussion register B). This plan holds the proposed-method direction that
grew out of plan 0103 Phase 2. Every experiment that runs under it gets its own config, comparison
and decision packet.

## Context: what plan 0103 Phase 2 established

All numbers: ImageNet-1k, PGD-AT, L_inf eps 4/255, PGD-3 training, from scratch, 50 epochs, batch 128,
full held-out set, last checkpoint, clean / PGD-10.

- MobileNetV4-Conv-S underfits. Its training-probe vs held-out PGD-10 gap is only 4.4 pt, and held-out
  robustness rises until the last epoch.
- Regularizers hurt:
  - AWP: 30.71 -> 17.45.
  - TRADES beta 6: -4.65.
  - IDBH: -1.4 at 50 epochs, -1.6 at 100 epochs.
  - Mixed clean+adversarial batches: large loss, with or without split BN.
- Optimization aids help:
  - SiLU: 55.45 / 33.27 (+2.56).
  - Cosine schedule: +0.78.
  - 100 instead of 50 epochs: 56.20 / 31.88 (+1.17, at 2x compute).
  - SE at matched parameters: +0.4 to +0.6 only.
- Deployment constraint (2026-10-15): avoid special inference-time parts (SE, sigmoid-based
  activations, attention) that are slow on mobile DSPs/NPUs. MobileNetV4 itself avoids SE and GELU
  for this reason (arXiv:2404.10518, Section 6). The efficiency metric stays Hamster RTX 4090
  training time + FLOPs.

## Hypothesis

Lightweight networks underfit adversarial training mainly because optimization fails to find good
solutions that their deployed function class already contains, not because that class is too small.
If this is right, interventions that exist only at training time and leave the deployed network
unchanged should close a large part of the gap to a bigger model at the same compute. That answers
plan 0103 RQ3.

Supporting literature:
- Xie et al. 2020, Smooth Adversarial Training, arXiv:2006.14536, ImageNet ResNet-50, PGD-1 training,
  robust accuracy:
  - SiLU ResNet-50 reaches 42.3, ahead of ReLU ResNet-152 at 39.4.
  - With a ReLU forward pass and a smooth backward pass only, 33.0 rises to 35.8 when the smooth
    gradient is used in the weight update alone, and to 36.9 when it is used in both attack and update.
  - Training cross-entropy falls (2.71 -> 2.59).
- Pang et al. 2021, Bag of Tricks, arXiv:2010.00467: smoothness helps small models more.
- Train-time over-parameterization gains are largest for the smallest networks:
  - MobileOne, arXiv:2206.04040: S0 +1.8.
  - ExpandNets, arXiv:1811.10495.
- Singh et al. 2023, arXiv:2303.01870: at short schedules, initialization and epochs matter more than
  attack strength.

## Method candidates

1. **Smooth-backward surrogate (ReLU forward).** The derivative of a parametric softplus (beta = 10)
   replaces the ReLU derivative in the backward pass.
   - Scope `optimizer`: only the weight update uses it.
   - Scope `attack_and_optimizer`: the training attack also uses it.
   - Selection, validation and evaluation attacks always use true ReLU gradients.
   - Prior work: SAT, ResNet-50 with PGD-1 only. Open: depthwise/MBConv networks, PGD-3, other
     surrogates, and annealing the surrogate back to the true gradient late in training (this plan's
     own variant).
2. **Hard-swish forward.** The DSP-oriented piecewise approximation of SiLU from MobileNetV3. No
   adversarial-training study found.
3. **Structural reparameterization.** MobileOne/RepVGG-style parallel branches during training
   (1x1 conv, identity, BN scale), folded into one conv at inference. RepOpt (arXiv:2205.15242) is the
   zero-cost gradient-multiplier variant. No adversarial-training study found; this is the clearest
   gap. The folded network has exactly the deployed function class, so a gain is direct evidence for
   the optimization hypothesis.
4. **Distillation stage (after the plan 0103 distillation verification): learnability-aware use of
   the precomputed teacher bank.**
   - Per-sample teacher entropy and margin (free from the FKD bank) identify samples that are
     "robustly unlearnable" for the student.
   - Those samples get soft targets only and weak or no attack. Attack compute is concentrated on
     learnable samples, and the saved compute goes to epochs.
   - Closest work:
     - Lee & Chung 2026 (ICML; abstract only read): robustly unlearnable sets.
     - Selective AT, arXiv:2512.22069: student-margin selection, CIFAR.
     - SAAD, arXiv:2512.10275: loss reweighting.
   - No prior work found that uses a precomputed robust-teacher bank to allocate attack compute.
   - This links to the human's CIFAR-10 findings: rescue/relapse cohorts, and capacity-inverse EMA
     self-distillation.
5. **Supporting: compute-reallocation schedule.** Cheap early stage (low resolution, single-step,
   N-FGSM) and strong late stage (224 px, PGD-3), with the savings spent on epochs. Prior work:
   Fast AT arXiv:2001.03994, AdvXL arXiv:2401.04727.

## First experiments (MobileNetV4-Conv-S, seed 0, 50 epochs, non-deterministic fastest setting)

| Run | Config | Compare with |
|---|---|---|
| hard-swish | `imagenet_mobilenetv4_hardswish_pgd_at_phase2_random_lr0025.yaml` | baseline+EMA 30.71, SiLU 33.27 |
| smooth backward, optimizer only | `imagenet_mobilenetv4_pgd_at_phase2_smoothgrad_opt.yaml` | same |
| smooth backward, attack and optimizer | `imagenet_mobilenetv4_pgd_at_phase2_smoothgrad_both.yaml` | same |

- The comparison runs were deterministic. The new runs use the non-deterministic fastest setting
  (human preference). Measured setting effects so far are inside seed noise.
- Decision rule (plan 0103 Phase 2 tiers on PGD-10 vs baseline+EMA): >= 1.5 effect, 0.5-1.5 pending,
  < 0.5 none.
- The share of the SiLU gain (+2.56) that each run recovers is also reported.
- If one reaches SiLU-level gain, it becomes the base model for the distillation verification (with
  its own speed setting). Otherwise SiLU vs ReLU is decided with decision 0044 and the DSP limitation
  wording.

Next, in parallel with the distillation verification: implement structural reparameterization with a
fold-equivalence test (unfolded and folded outputs equal). Run MobileNetV4-S with branches against
MobileNetV4-M at matched GPU-hours.

## Literature collected (2026-10-15 scouting; [A] = abstract only)

- Smooth AT 2006.14536. Bag of Tricks 2010.00467. Singh et al. 2303.01870. Xie & Yuille 1906.03787.
  Ngnawé et al. 2509.23325 (epsilon scheduling, fine-tuning).
- Reparameterization: RepVGG 2101.03697, ACNet 1908.03930, DBB 2103.13425, MobileOne 2206.04040,
  OREPA 2204.00826, RepOpt 2205.15242. ExpandNets 1811.10495. Arora et al. 1802.06509.
- SUGAR (surrogate gradient for ReLU, no AT) 2505.22074.
- Compute:
  - Fast AT 2001.03994, GradAlign 2007.02617, N-FGSM 2202.01181, ATAS 2206.02417.
  - AdvXL 2401.04727.
  - Selective AT 2512.22069 [A].
  - Bartoldson et al. 2404.09349 (scaling, CIFAR). Debenedetti et al. 2312.13131. Wang et al. 2302.04638.
- Distillation:
  - ARD 1905.09747, RSLAD 2108.07969.
  - IBD (NeurIPS 2023; ImageNet ResNet-50, RSLAD +1.7 AA).
  - SAAD 2512.10275 [A]. Lee & Chung 2605.21999 [A].
  - IGDM 2312.03286 [A]. FKD 2112.01528, FerKD 2312.17473 [A].
- Underfitting / capacity: Li et al. 2205.13863 [A]. Dong et al. 2110.03135 [A]. GAIRAT failure
  2103.01914 [A].
- Gap found: no published adversarial-training recipe or AutoAttack table for ImageNet-1k models under
  10M parameters at 4/255. This is itself a plan 0103 RQ1 contribution.

## Progress

- 2026-10-15: direction approved. Hard-swish and smooth-backward (two scopes) implementation started.
