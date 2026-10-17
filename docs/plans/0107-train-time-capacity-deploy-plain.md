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
- Reading rule (review P2-5): in the `attack_and_optimizer` arm, the training attack uses the surrogate
  gradient, so it is weaker against the true network. Its `train_robust_accuracy` and training loss are
  therefore not comparable across arms. Optimization claims across arms use `train_probe_*` (eval-mode
  probe under the true-gradient selection attack) and held-out metrics only.
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
- 2026-10-15: implementation committed (8f51192; scientific review no P1, P2s fixed; verify green). The
  three runs (`plan0107-mnv4s-{hardswish,smoothgrad-opt,smoothgrad-both}-s0-v1`) are at the front of
  Ferret's pull queue, worktree `source-8f511929ed7e`. Hamster's GPUs had just started multi-day
  MobileNetV4-M / EfficientNet-B0 runs, so they start as Ferret GPUs free (from the night of 10/15).
- 2026-10-15 (human): order of decisions. The MobileNetV4-S hard-swish and smooth-backward results come
  first, because DSP-friendly options take priority. If they reach SiLU-level gain, MobileNetV4-M follows
  that direction. If not, MobileNetV4-M gets a SiLU learning-rate search, starting at lr 0.0125. That
  search is needed because the MobileNetV4-M SiLU run, at the ReLU-tuned lr 0.025, gave 61.26 / 38.33
  vs ReLU 63.71 / 39.30, with a higher final training loss.
- 2026-10-18 (postrun): **hard-swish, MobileNetV4-S seed 0 completed.**
  `plan0107-mnv4s-hardswish-s0-v1` (hand-run from the pull queue on Ferret,
  one RTX 4090; W&B metadata host `islab-3gpu`, `environment.json` kernel
  7.0.0-28, driver 595.99.02; the GPU index is not recorded) ran 50/50
  epochs from 2026-10-15 03:44:22 to 2026-10-16 17:01:15 UTC.
  `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application
  error recorded". The watcher re-derived it as terminal and successful.
  All 50 epoch rows are present (`epoch_metrics_complete: true`), one W&B
  segment, no resume. `train.log` has no traceback, NaN, error or warning
  line. The wandb internal log has one transient `file_stream` retry
  (HTTP 500, 2026-10-15 23:06 UTC); the local metrics are complete, and
  the W&B history was not checked. `best.pt` (epoch 47), `best-ema.pt`
  (epoch 42), `last.pt` and `epoch-049.pt` are on disk. Nothing is
  imported into `docs/experiments/`: no aggregator exists for this
  contract. The numbers below are read from the run-bundle manifest and
  `epoch-metrics.jsonl` (via `scripts/ardx/read_bundle.py`); the
  baseline+EMA and SiLU numbers are from their plan 0103 entries
  (2026-10-10 and 2026-10-11 postruns).

  Fixed identity: config
  `configs/scientific/imagenet_mobilenetv4_hardswish_pgd_at_phase2_random_lr0025.yaml`
  (unchanged since the source SHA) = the MobileNetV4-S Phase 2
  baseline+EMA config (`imagenet_mobilenetv4_pgd_at_phase2_baseline_ema_cg.yaml`)
  except (a) `student.architecture: mobilenetv4_conv_small_hardswish_imagenet`
  (every ReLU, stem and head included, replaced by hard-swish; parameters
  and MACs unchanged); (b) `deterministic: false`, `cudnn_benchmark: true`,
  no `cuda_graph` (eager, not compiled); (c) the W&B group. ImageNet-1k
  (original images, content `ae033613...`), MobileNetV4-Conv-S random
  init, plain PGD-AT, no teacher, 224 px training. Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack CE
  PGD-3, l_inf eps 4/255, step 8/765, random start, true gradients.
  Selection and validation attack PGD-10, same eps and step, random
  start, eval mode. SGD (Nesterov, momentum 0.9, wd 1e-4 on all
  parameters), peak LR 0.025, warmup_multistep (10-epoch warmup, x0.1 at
  epochs 25 and 38), no label smoothing, full AT, 50 epochs. Phase 2
  template: selection subset 5,000, EMA 0.9999. World size 1, global
  batch 128, local BN. fp32. Protocol
  `controlled_imagenet_stage02_init_lr_grid_v1`, config hash
  `3d5032be...`. Source SHA `8f511929ed7e67684415fe64f3403ae4aa415b55`
  (worktree `source-8f511929ed7e`, clean). "Full val", "subset",
  "complement" and "probe" mean the same as in plan 0103 (25,620 / 5,000
  / 20,620 held-out original train images, 2,000 training images). None
  is the ImageNet val split. No AutoAttack and no official evaluation has
  run.

  End of training (full val and complement are evaluated at the end, for
  the last and the best checkpoints):

  | checkpoint | full val clean / PGD-10 | complement clean / PGD-10 |
  |---|---:|---:|
  | last (ep 49), raw weights | 55.00 / 32.62 | 55.25 / 32.83 |
  | last (ep 49), EMA weights | 54.82 / 32.70 | 55.14 / 32.90 |
  | best raw (ep 47, by subset PGD-10) | 54.84 / 32.52 | 55.09 / 32.66 |
  | best EMA (ep 42, by EMA subset PGD-10) | 54.49 / 32.58 | 54.75 / 32.74 |
  | baseline+EMA (ReLU, Hamster, `d836c03`), last raw | 53.85 / 30.71 | 54.07 / 30.90 |
  | baseline+EMA, last EMA | 53.95 / 30.80 | 54.20 / 31.00 |
  | SiLU (Hamster, `5dd329e`), last raw | 55.45 / 33.27 | 55.56 / 33.39 |
  | SiLU, last EMA | 55.60 / 33.53 | 55.73 / 33.66 |

  Per-epoch trajectory (subset and probe are evaluated every epoch;
  baseline = baseline+EMA, raw weights):

  | epoch | LR | subset raw clean / PGD-10 | baseline / SiLU subset raw PGD-10 | subset EMA clean / PGD-10 | probe clean / PGD-10 | train loss (baseline / SiLU) | train robust acc |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 0 | 0.0025 | 0.82 / 0.52 | 0.54 / 0.52 | 0.10 / 0.10 | 0.70 / 0.70 | 6.853 (6.831 / 6.861) | 0.29 |
  | 9 | 0.025 (end of warmup) | 31.22 / 16.94 | 15.98 / 17.54 | 7.46 / 4.18 | 32.40 / 17.35 | 4.895 (4.833 / 4.833) | 13.42 |
  | 24 | 0.025 (last) | 41.76 / 23.34 | 21.76 / 24.02 | 46.42 / 28.92 | 45.00 / 24.60 | 4.357 (4.330 / 4.290) | 19.45 |
  | 25 | 0.0025 | 48.62 / 28.70 | 26.94 / 29.66 | 47.64 / 29.24 | 53.20 / 31.80 | 4.017 (4.016 / 3.948) | 23.96 |
  | 37 | 0.0025 (last) | 51.82 / 30.14 | 28.80 / 31.10 | 52.52 / 31.54 | 56.55 / 35.10 | 3.831 (3.827 / 3.761) | 26.44 |
  | 38 | 0.00025 | 52.68 / 31.16 | 29.30 / 32.24 | 52.70 / 31.70 | 58.70 / 35.60 | 3.732 (3.744 / 3.662) | 27.88 |
  | 47 | 0.00025 (best raw) | 53.80 / 31.94 | 29.60 / 33.00 | 53.56 / 31.80 | 59.75 / 36.00 | 3.682 (3.702 / 3.612) | 28.53 |
  | 49 | 0.00025 (last) | 53.96 / 31.78 | 29.98 / 32.80 | 53.52 / 31.86 | 60.10 / 36.45 | 3.678 (3.697 / 3.608) | 28.64 |

  Reading (n=1, internal validation, PGD-10 only):
  1. **Against baseline+EMA, last raw weights on full val:** 55.00 /
     32.62 vs 53.85 / 30.71, i.e. **+1.15 clean / +1.91 PGD-10** (about
     294 / 490 images of 25,620). Complement +1.18 / +1.93; last EMA
     +0.87 / +1.90. With the two-run SE used for SiLU (about 0.43pt
     clean, 0.37pt PGD-10) that is about 2.7 / 5.2 SE, and the PGD-10
     gain is about 5x the MobileNetV4-S seed-0 / seed-1 spread (0.36pt).
     Clean and PGD-10 move up together.
  2. **Preregistered rule** (this plan; the plan 0103 Phase 2 tiers on
     PGD-10 vs baseline+EMA: >= 1.5 effect, 0.5-1.5 pending, < 0.5 none):
     +1.91pt is in the **"effect"** tier, 0.41pt above its lower edge.
  3. **Against SiLU and the share of the SiLU gain.** Hard-swish minus
     SiLU on full val: **-0.45 clean / -0.65 PGD-10** (complement -0.31 /
     -0.56; last EMA -0.78 / -0.83). -0.65 is about 1.8 SE and 1.8x the
     seed spread. Hard-swish recovers **75% of the SiLU PGD-10 gain**
     (1.91 of 2.56) and 72% of the clean gain (1.15 of 1.60); complement
     78% (1.93 of 2.49), EMA 70% (1.90 of 2.73). The plan's "SiLU-level
     gain" condition has no numeric threshold; whether 75% counts is the
     human's call (packet).
  4. **Setting caveat.** The comparison crosses source SHA (`8f51192` vs
     `d836c03` / `5dd329e`), host (Ferret vs Hamster), determinism class
     (non-deterministic + cudnn_benchmark vs deterministic) and execution
     mode (eager vs CUDA Graphs for the baseline). The setting checks so
     far that crossed host / source / determinism moved PGD-10 by 0.06
     (EfficientNet-B0), 0.10 (ConvNeXt-Atto) and 0.29pt (MobileNetV4-M);
     the MobileNetV4-S check matched bit for bit within one mode. A
     setting effect would have to exceed 0.41pt in hard-swish's favour to
     move the result out of the "effect" tier. Not measured here.
  5. **Where the lead appears.** Subset PGD-10 is level with the baseline
     at epoch 0, +0.96 at the end of warmup (SiLU +1.56), +1.58 at the
     end of the peak-LR stage (SiLU +2.26), +1.34 at epoch 37 (SiLU
     +2.30), +1.80 at epoch 49 (SiLU +2.82). Like SiLU, the gap does not
     open at an LR decay; it is smaller than SiLU's at every listed epoch.
  6. **Training fit: the held-out lead does not come with a lower
     training loss until the last stage.** Train loss (train mode,
     augmented, PGD-3) is above the baseline through warmup and the peak
     LR (+0.062 at epoch 9, +0.027 at epoch 24), level in the middle
     stage (+0.001 / +0.004 at epochs 25 / 37) and below only in the last
     stage (-0.019 at epoch 49; SiLU -0.089). Training-batch robust
     accuracy ends at 28.64% vs 27.90% (+0.74; SiLU +1.63). The eval-mode
     probe is ahead throughout the listed epochs: at epoch 24 +1.25 /
     +1.40, at epoch 49 +0.30 / +2.05 against the baseline (SiLU +2.30 /
     +3.85); probe SE is about 1.1pt per run. Probe minus full val at the
     end is +5.10 clean / +3.83 PGD-10 (baseline +5.95 / +3.69, SiLU
     +6.65 / +4.98). The val >= train signature persists (28.64% <
     32.62%). Why train loss and held-out accuracy disagree before epoch
     38 is not explained by these logs.
  7. **EMA.** At the end: EMA minus raw on full val -0.17 clean / +0.08
     PGD-10 (complement -0.11 / +0.07), "no effect". Mid-run: the warmup
     lag is there (epoch 9 subset EMA 7.46 / 4.18 vs raw 31.22 / 16.94),
     as in baseline+EMA and SiLU. EMA passes raw at epoch 13 (PGD-10) and
     14 (clean), as SiLU did; leads by up to +5.16 clean (epoch 22) /
     +6.38 PGD-10 (epoch 21) at the peak LR, +0.54 to +1.54 PGD-10 in the
     middle stage and -0.14 to +0.74 in the last stage.
  8. **Best and last; trains normally.** Best raw (epoch 47 by subset
     PGD-10) minus last on full val: -0.16 / -0.11
     (`robust_overfit_gap_full` -0.11pt, i.e. last is better). Best EMA
     (epoch 42) is 0.12pt PGD-10 below last EMA. Subset raw PGD-10 never
     falls more than 0.64pt below its running maximum (epoch 21, 21.88
     after 22.52 at epoch 20); the catastrophic-overfitting guard never
     fires. Stages on the subset: PGD-10 21.76-23.34 over epochs 17-24,
     28.72-30.14 over epochs 26-37, 31.14-31.94 over epochs 38-49.
     Decays: epoch 24 to 25 +6.86 clean / +5.36 PGD-10 (baseline +7.48 /
     +5.18, SiLU +7.44 / +5.64), epoch 37 to 38 +0.86 / +1.02 (baseline
     +0.80 / +0.50, SiLU +1.26 / +1.14). The subset sits 1.04 / 0.84pt
     below full val at epoch 49.
  9. **Cost (not comparable to the references).** 37.28 h from start to
     finish (134,213 s); summed per-epoch training time about 130,300 s
     (36.2 h), the rest (about 78 s per epoch) is subset validation, the
     final full-val passes, probe, checkpointing and start-up. Throughput
     490-508 img/s in epochs 0-8, 431-470 in epochs 9-27, 470-507 in
     epochs 28-49. That is about half of the idle-Ferret single-process
     MobileNetV4-S rate (868-876 img/s, plan 0103) and of SiLU eager on
     Hamster (928-943). Other jobs on the shared Ferret GPUs, the
     hard-swish kernels and the non-deterministic eager mode are not
     separated, so no speed claim is made. FLOPs and parameters equal
     ReLU. Peak allocated memory 6.58 GB at epoch 0 (cudnn_benchmark
     autotuning is the likely cause; not checked), 3.74 GB after;
     reserved 7.11 / 4.91 GB.

  Caveats: one seed. The comparison crosses source, host, determinism
  class and execution mode (item 3); the baseline+EMA and SiLU runs were
  deterministic, this one is not. Evaluation is CE PGD-10 only; no
  AutoAttack check against gradient masking. On-device (DSP / NPU) speed
  of hard-swish vs SiLU vs ReLU is not measured. Checkpoint sha256 was
  not computed in this postrun. `epoch-metrics.parquet` sha256
  `3b6afc51...`, `sample-stats-train.parquet` sha256 `fbec04b5...` (from
  the run-bundle manifest). W&B `lightweight-imagenet-at`, same run id
  (one segment). Plan status: one of the three first experiments; the
  two smooth-backward runs have their own entries below (optimizer only
  -1.21, attack and optimizer -7.19 PGD-10), so hard-swish is the only
  one of the three that gains. The plan's "base model for the
  distillation verification" choice is the human's (packet).
- 2026-10-18 (postrun): **smooth backward, optimizer only, MobileNetV4-S seed 0 completed.**
  `plan0107-mnv4s-smoothgrad-opt-s0-v1` (hand-run on Ferret, one RTX 4090; `environment.json` records
  kernel 7.0.0-28; the GPU index is not recorded) ran 50/50 epochs from 2026-10-15 03:48:28 to
  2026-10-16 17:31:54 UTC. `run-bundle/completion.json` reads `completed`, the manifest status is
  `completed`, and `run-bundle/error-marker.txt` reads "no application error recorded". The watcher
  re-derived it as terminal and successful. All 50 epoch rows are present (`epoch_metrics_complete:
  true`), one W&B segment, no resume. `train.log` has no traceback, NaN or error line (only the wandb
  git-root warning); the wandb internal log has one transient `file_stream` HTTP 500 retry
  (2026-10-15 23:07 UTC). `best.pt` (epoch 47), `best-ema.pt` (epoch 37), `last.pt` and
  `epoch-049.pt` are on disk. Nothing is imported into `docs/experiments/`: no aggregator exists for
  this contract. The numbers below are read from the run-bundle manifests and `epoch-metrics.jsonl`
  of this run and of the two comparison runs (via `scripts/ardx/read_bundle.py`); the SiLU complement
  and EMA numbers are from its plan 0103 ledger row.

  Fixed identity: config `imagenet_mobilenetv4_pgd_at_phase2_smoothgrad_opt.yaml` (unchanged since the
  source SHA) = the MobileNetV4-S baseline+EMA recipe plus `training.relu_surrogate_gradient: {beta:
  10.0, scope: optimizer}`, i.e. the derivative of softplus (beta 10) replaces the ReLU derivative in
  the backward pass of the training loss only; the forward pass is exactly ReLU, and the training
  attack, selection, probe and validation attacks use true ReLU gradients. ImageNet-1k (original
  images, content `ae033613...`), MobileNetV4-Conv-S (`mobilenetv4_conv_small_imagenet`,
  `imagenet_standard` profile) random init, plain PGD-AT, no teacher, 224 px training. Training and
  evaluation-attack seed 0 (split seed 20260911). Training attack CE PGD-3, l_inf eps 4/255, step
  8/765, random start, eval mode. Selection and validation attack PGD-10, same eps and step, random
  start, eval mode. SGD (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.025, warmup_multistep (10-epoch
  warmup, x0.1 at epochs 25 and 38), no label smoothing, full AT, 50 epochs. Phase 2 template:
  5,000-image selection subset, EMA 0.9999. World size 1, global batch 128, local BN. fp32.
  Non-deterministic with cudnn_benchmark, eager (no `torch.compile`, no cuda_graph). Protocol
  `controlled_imagenet_stage02_init_lr_grid_v1`, config hash `8baf9c56...`. Source SHA
  `8f511929ed7e67684415fe64f3403ae4aa415b55` (worktree `source-8f511929ed7e`, clean). "Full val"
  (25,620), "subset" (5,000), "complement" (20,620) are held-out original train images and "probe"
  is 2,000 training images, as in plan 0103. None is the ImageNet val split. No AutoAttack and no
  official evaluation has run.

  End of training (full val and complement are evaluated at the end, for the last and the best
  checkpoints). Baseline+EMA = `plan0103-phase2-mnv4s-baseline-ema-s0-v1` (Hamster, `d836c03`,
  deterministic, CUDA Graphs); SiLU = `plan0103-phase2-mnv4s-silu-s0-v1` (Hamster, `5dd329e`,
  deterministic, eager):

  | checkpoint | full val clean / PGD-10 | complement clean / PGD-10 | baseline+EMA full val | smooth minus baseline (full val) |
  |---|---:|---:|---:|---:|
  | last (ep 49), raw weights | 54.37 / 29.49 | 54.60 / 29.62 | 53.85 / 30.71 | **+0.52 / -1.21** |
  | last (ep 49), EMA weights | 54.49 / 29.75 | 54.78 / 29.89 | 53.95 / 30.80 | +0.53 / -1.05 |
  | best raw (by subset PGD-10) | 54.31 / 29.72 (ep 47) | 54.65 / 29.82 | 53.74 / 30.57 (ep 43) | +0.57 / -0.85 |
  | best EMA (by EMA subset PGD-10) | 53.61 / 29.85 (ep 37) | 53.90 / 29.98 | 52.75 / 30.60 (ep 36) | +0.87 / -0.75 |
  | SiLU, last raw | 55.45 / 33.27 | 55.56 / 33.39 | | smooth minus SiLU -1.08 / -3.78 |

  Per-epoch trajectory, this run against baseline+EMA (subset and probe are evaluated every epoch;
  subset values for both runs, train loss and training-batch robust accuracy for both runs):

  | epoch | LR | subset raw clean / PGD-10 | baseline subset raw | subset EMA clean / PGD-10 | probe clean / PGD-10 | train loss (baseline) | train robust acc (baseline) |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 0 | 0.0025 | 0.50 / 0.42 | 0.68 / 0.54 | 0.10 / 0.10 | 0.55 / 0.45 | 6.859 (6.831) | 0.28 (0.31) |
  | 4 | 0.0125 | 16.30 / 8.72 | 16.12 / 8.60 | 5.28 / 2.98 | 16.40 / 8.60 | 5.663 (5.673) | 6.17 (6.12) |
  | 9 | 0.025 (end of warmup) | 31.42 / 16.40 | 30.96 / 15.98 | 5.22 / 2.82 | 33.10 / 16.55 | 4.830 (4.833) | 13.87 (13.82) |
  | 14 | 0.025 | 37.76 / 19.84 | 37.18 / 18.96 | 37.20 / 21.86 | 40.45 / 20.50 | 4.505 (4.504) | 17.34 (17.36) |
  | 24 | 0.025 (last) | 41.98 / 21.80 | 40.40 / 21.76 | 46.70 / 27.36 | 46.35 / 23.25 | 4.329 (4.330) | 19.33 (19.34) |
  | 25 | 0.0025 | 48.48 / 27.20 | 47.88 / 26.94 | 47.90 / 27.56 | 54.45 / 29.30 | 4.024 (4.016) | 23.40 (23.51) |
  | 37 | 0.0025 (last; best EMA) | 51.74 / 28.28 | 51.46 / 28.80 | 52.42 / 29.34 | 57.20 / 32.70 | 3.847 (3.827) | 25.69 (26.00) |
  | 38 | 0.00025 | 52.78 / 28.40 | 52.26 / 29.30 | 52.62 / 28.98 | 58.35 / 33.35 | 3.770 (3.744) | 26.88 (27.25) |
  | 47 | 0.00025 (best raw) | 52.94 / 29.26 | 52.86 / 29.60 | 53.18 / 29.20 | 59.25 / 34.05 | 3.732 (3.702) | 27.31 (27.79) |
  | 49 | 0.00025 (last) | 53.42 / 29.02 | 52.96 / 29.98 | 53.28 / 29.10 | 59.50 / 33.80 | 3.728 (3.697) | 27.41 (27.90) |

  Reading (n=1, internal validation, PGD-10 only):
  1. **No gain; PGD-10 is lower.** Last raw weights on full val: **+0.52 clean / -1.21 PGD-10**
     against baseline+EMA (about 132 / 311 images of 25,620; about 1.2 / 3.0 two-run SE of about
     0.44 / 0.40pt; PGD-10 about 3.4x the MobileNetV4-S seed spread of 0.36pt). Complement +0.53 /
     -1.29, last EMA +0.53 / -1.05, best raw +0.57 / -0.85, best EMA +0.87 / -0.75: every checkpoint
     has the same signs. Under the plan's rule (PGD-10 vs baseline+EMA: >= 1.5 effect, 0.5-1.5
     pending, < 0.5 none) the magnitude is in the "pending" band, in the losing direction.
  2. **Share of the SiLU gain recovered: -47%** (-1.21 of +2.56). Against SiLU directly: -1.08 /
     -3.78. SAT's ResNet-50 result (smooth backward in the weight update alone: 33.0 -> 35.8, training
     cross-entropy lower) does not carry over to MobileNetV4-S under PGD-3 here.
  3. **Where it diverges: after the first LR decay, in the training fit.** Through warmup and the
     peak-LR stage the training loss matches the baseline to within 0.010 at every epoch listed
     (epochs 4-24; within 0.003 from epoch 9), and training-batch robust accuracy to within 0.05pt. Held-out subset PGD-10 is
     level or slightly ahead (+0.42 at epoch 9, +0.88 at 14, +0.04 at 24); subset clean leads by up to
     +1.58 (epoch 24). From the first decay on, the training loss falls less than the baseline's (+0.009
     at epoch 25, +0.021 at 37, +0.031 at 49) and training-batch robust accuracy ends 0.49pt lower;
     subset PGD-10 falls behind (-0.52 at epoch 37, -0.90 at 38, -0.96 at 49) and the probe ends
     -0.30 / -0.60. In this arm the training attack uses true gradients, so its training loss and
     training-batch robust accuracy are comparable with the baseline (the review P2-5 caveat applies
     only to `attack_and_optimizer`). So at seed 0 the surrogate fits the training data worse, the
     opposite of the SAT report and of SiLU on MobileNetV4-S (train loss 3.608 vs 3.697). Why the
     surrogate stops helping at low LR (for example, a mismatch between the surrogate direction and the
     true loss landscape that matters once steps are small) is not tested.
  4. **EMA: as in the baseline.** EMA minus raw at the end +0.12 / +0.26 ("no effect", < 0.5pt;
     baseline +0.10 / +0.10). EMA warmup lag as in the baseline: epoch 9 EMA subset 5.22 / 2.82
     (baseline 5.30 / 2.84); EMA passes raw at epoch 14 (PGD-10) and 15 (clean).
  5. **Best and last; trains normally.** Best raw is epoch 47 (subset 29.26 vs last 29.02); on full val
     it is -0.06 / +0.23 against last (`robust_overfit_gap_full` 0.23pt, inside one val SE). Best EMA
     is epoch 37 (the end of the middle stage), 0.10pt above last EMA on full-val PGD-10. Subset raw
     PGD-10 never falls more than 0.94pt below its running maximum (epoch 18, 20.08 after 21.02 at
     epoch 16); the catastrophic-overfitting guard never fires. Probe minus full val at the end: +5.13
     clean / +4.31 PGD-10 (baseline +5.95 / +3.69). Training-batch robust accuracy 27.41% < full-val
     PGD-10 29.49%: the val >= train signature persists.
  6. **Cost: not comparable with the baseline.** 37.72 h from start to finish on Ferret, 438-492 img/s
     (per-epoch training time about 2,600-2,730 s). Baseline+EMA ran 15.52 h at 1,175-1,181 img/s on
     Hamster with CUDA Graphs (deterministic); host, GPU, execution mode and the surrogate all differ,
     so the surrogate's own cost is not separated here.

  Caveats: one seed. The comparison crosses host (Ferret vs Hamster), source SHA (`8f51192` vs
  `d836c03`) and execution mode (non-deterministic eager vs deterministic CUDA Graphs). The setting
  checks measured so far are 0.00 (MobileNetV4-S, same host), -0.06 to -0.29 (other models,
  non-deterministic, some across hosts), all below the 1.21pt here, but none covers this host and mode
  for MobileNetV4-S. No telemetry in the run confirms per step that the surrogate was active; the
  evidence is the resolved config (`relu_surrogate_gradient: {beta: 10.0, scope: optimizer}`), the
  tested code path at the source SHA, and the training loss diverging from the baseline only after the
  first decay. Evaluation is CE PGD-10 only. Checkpoint sha256 was not computed in this postrun.
  `epoch-metrics.parquet` sha256 `ab8856c9...`, `sample-stats-train.parquet` sha256 `72cc2586...`
  (from the run-bundle manifest). W&B `lightweight-imagenet-at`, same run id (one segment).
- 2026-10-18 (postrun): **smooth backward, attack and optimizer, MobileNetV4-S seed 0 completed.**
  `plan0107-mnv4s-smoothgrad-both-s0-v1` (hand-run; `environment.json` records kernel 7.0.0-28, i.e.
  Ferret, where this plan queued the three runs; the GPU index is not recorded) ran 50/50 epochs from
  2026-10-15 08:47 to 2026-10-17 01:11 UTC (40.4 h). `run-bundle/completion.json` and the manifest read
  `completed`, `error-marker.txt` reads "no application error recorded", and the watcher re-derived it
  as terminal and successful. All 50 epoch rows are present, one W&B segment, no resume, no `ERROR` line
  in the wandb internal log. `best.pt` (epoch 26), `best-ema.pt` (epoch 29), `last.pt` and
  `epoch-049.pt` are on disk. Nothing is imported into `docs/experiments/`: no aggregator exists for this
  contract. Numbers are read from the run-bundle manifest and `epoch-metrics.jsonl` via
  `scripts/ardx/read_bundle.py`; the baseline+EMA trajectory is read the same way from
  `plan0103-phase2-mnv4s-baseline-ema-s0-v1`.

  Fixed identity: config `imagenet_mobilenetv4_pgd_at_phase2_smoothgrad_both.yaml` (config hash
  `d0a662f5...`). ImageNet-1k (original images, content `ae033613...`), MobileNetV4-Conv-S
  (`mobilenetv4_conv_small_imagenet`, ReLU forward) random init, plain PGD-AT, no teacher, 224 px.
  `training.relu_surrogate_gradient`: beta 10, scope `attack_and_optimizer` (softplus-derivative
  backward in the training attack and the weight update). Training and evaluation-attack seed 0 (split
  seed 20260911). Training attack CE PGD-3, l_inf eps 4/255, step 8/765, random start. Selection and
  validation attack PGD-10, same eps and step, random start, eval mode, true ReLU gradients. SGD
  (Nesterov, momentum 0.9, wd 1e-4), peak LR 0.025, warmup_multistep (10-epoch warmup, x0.1 at epochs
  25 and 38), 50 epochs. Phase 2 template: 5,000-image selection subset, EMA 0.9999. World size 1,
  global batch 128, local BN, fp32, non-deterministic with cudnn_benchmark, eager. Protocol
  `controlled_imagenet_stage02_init_lr_grid_v1`. Source SHA `650fbed386adc8fbc201c94fc1e079369cf77f5f`
  (worktree `source-650fbed386ad`, clean; it contains the 8f51192 implementation plus the never-learned
  guard). "Full val" is the 25,620 held-out original train images, "subset" the 5,000-image selection
  subset of it, "complement" the other 20,620, "probe" 2,000 training images attacked in eval mode with
  the true-gradient PGD-10.

  Results (internal validation, PGD-10 l_inf 4/255; never an official test result):
  1. **Last (ep 49) raw weights: 55.91% clean / 23.52% PGD-10.** Against baseline+EMA (53.85 / 30.71):
     +2.06 clean / -7.19 PGD-10 (about 1,842 images; about 4.7 / 18 two-run SE; about 20x the
     MobileNetV4-S seed spread of 0.36pt). Against SiLU (55.45 / 33.27): +0.46 / -9.75. Share of the
     SiLU PGD-10 gain recovered: none (the run moves PGD-10 the other way, -2.8x the SiLU gain).
  2. Complement 56.18 / 23.73. Last EMA 55.94 / 23.91 (against baseline EMA 53.95 / 30.80: +1.99 /
     -6.89; EMA minus raw +0.04 / +0.38, "no effect").
  3. Best raw (ep 26, chosen on the subset) 51.71 / 24.94; best EMA (ep 29) 53.25 / 26.31. Best minus
     last on full val PGD-10 is 1.42pt (raw).
  4. **When the gap opens.** Subset PGD-10, this run minus baseline+EMA (raw): -0.62 at epoch 9, -1.86 at
     24, -2.98 at 26, -4.26 at 30, -5.60 at 37, -6.46 at 44, -7.30 at 49. Subset clean is ahead from the
     warmup on (+0.68 at 9, +2.14 at 24, +1.80 at 49). Subset PGD-10 peaks at 24.64% (epochs 26-27) and
     then falls 1.96pt to 22.68% while clean keeps rising; the baseline rises to the end (29.98%).
  5. **The loss is on training images too, so it is not a generalization gap.** Probe PGD-10 (true
     gradient) ends at 27.40% against the baseline's 34.40% (-7.00); probe clean 61.40 vs 59.80
     (+1.60). Probe minus full val PGD-10 is +3.88 (baseline +3.69). Among the epochs read (0, 1, 4, 9, 14, 19, 24-27, 30,
     34, 37-39, 44, 48, 49), probe PGD-10 is highest at 28.0% at epoch 27 and ends at 27.4.
  6. Training loss (3.535 vs 3.697) and training-batch robust accuracy (29.94% vs 27.90%) are measured
     under the surrogate-gradient training attack, so by the reading rule above they are not comparable
     across arms. Within this run, the training-batch robust accuracy under the PGD-3 surrogate attack
     (29.94%, train mode) is above the probe's true-gradient PGD-10 (27.40%, eval mode); in the baseline
     it is below (27.90 vs 34.40). That is consistent with the surrogate training attack being weaker
     against the true network, but the two measurements differ in steps, mode and images, so it is not a
     direct strength measurement.
  7. The run was not stopped by any guard; no collapse (983 distinct subset predictions at the end).
     Throughput 433-499 img/s on shared Ferret (baseline+EMA 1,175-1,181 img/s on Hamster with CUDA
     Graphs, deterministic), so the time cost of the surrogate itself is not separable from host and
     execution mode. Peak allocated 3.74 GB.

  Caveats: one seed. The comparison crosses source SHA (`650fbed` vs `d836c03`), host (Ferret vs
  Hamster) and execution mode (non-deterministic eager vs deterministic CUDA Graphs); measured setting
  effects so far are 0.3pt or less, about 1/20 of this difference. No AutoAttack, no official
  evaluation. The SAT result this arm tests (ResNet-50, PGD-1: 33.0 -> 36.9) did not carry over to
  MobileNetV4-S with PGD-3 at beta 10 on this seed.

  Decision under the rule above: PGD-10 -7.19 is beyond the 1.5pt tier in the losing direction, so this
  arm does not reach SiLU-level gain and does not become the distillation base model. What follows
  (close this scope, try another beta or annealing, or read it together with the optimizer-only and
  hard-swish runs, recorded separately) is the human's decision (packet).

