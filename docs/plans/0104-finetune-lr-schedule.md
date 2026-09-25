# Plan 0104: fine-tuning-shaped LR schedule for adversarial training from a clean checkpoint

## Context

Decision 0019 (human, 2026-09-25) closed plan 0103's budget x init 2x2 with option B plus this plan.
The 2x2 on MobileNetV4-Conv-Small (Arm A recipe, seed 0, internal val) found:

- Pretrained and random init end within 0.65pt clean and 0.35pt PGD-10 at 100 epochs.
  Pretrained at 50 epochs (Arm A) lands in the same place.
- The pretrained head start disappears during the lr-0.05 phase. At epoch 0 (LR still ~0.005 in the warmup), the pretrained run is at
  46.81 / 19.20. As the warmup reaches 0.05, clean falls to ~37-38%, and the random-init run has caught up by epoch ~24.
- The lr-0.05 phase is a plateau (val PGD-10 18.9-20.5% for 40 epochs). Every gain arrives after an LR decay, and each LR stage
  saturates within about 12 epochs.

So Arm A throws the clean-pretrained features away before learning robustness. This plan asks whether a fine-tuning-shaped
schedule keeps them, and whether that pays off in final robustness or in budget.

## Frozen scientific contract

- Config: `configs/scientific/imagenet_mobilenetv4_pgd_at_ft_lr0005.yaml`.
  Protocol id: `controlled_imagenet_stage02_finetune_lr_v1`.
- It is identical to Arm A (`imagenet_mobilenetv4_pgd_at_no_warmup.yaml`) except for:
  - `optimizer.learning_rate`: 0.05 -> 0.005. The schedule shape is unchanged: warmup_multistep, 10-epoch warmup, x0.1 at
    [25, 38]. The run is therefore 0 -> 0.005, then 0.0005 from epoch 25 and 0.00005 from epoch 38.
  - `protocol.id` and `tracking.group`.
  - `training.train_probe_size: 2000`. This is observability only, and it is proven training-neutral by
    `test_train_probe_is_observational_*`.
- Everything else matches Arm A: clean-pretrained init (timm `mobilenetv4_conv_small.e1200_r224_in1k`), 50 epochs, SGD with
  momentum 0.9 and Nesterov, wd 1e-4, batch 128, RandomResizedCrop + flip, and deterministic mode.
  - Training attack: CE PGD-3, eps 4/255, step 8/765, random start, generated in eval mode.
  - Selection and validation attack: PGD-10.
- One run, seed 0, on a Hamster RTX 4090, hand-run from a pinned worktree.
  Run id `plan0104-mobilenetv4-ft-lr0005-v1`, W&B project `lightweight-imagenet-at`.

## Preregistered decision rule

All comparisons use internal-val PGD-10 (the held-out 2% of the training split), as in Arm A. This is not an official test.

- Primary: if last-epoch PGD-10 is >= 31.59% (Arm A's higher seed, 30.59%, + 1pt), this schedule becomes a Phase 1 recipe
  candidate. Otherwise Arm A stays.
- Secondary (budget): if the first epoch at which val PGD-10 is >= 30.58% (Arm A's lower seed) is at or before epoch 25, this is a
  half-budget candidate.
- Clean accuracy is reported next to both.
- Interpretation together with the 2x2's random-init/50 cell (decision 0019 B):

  | random-init/50 vs Arm A | this run vs Arm A | reading |
  |---|---|---|
  | within 1pt | within 1pt | pretraining is not needed |
  | within 1pt | >= 1pt higher | pretraining helps when used with a fine-tuning schedule |
  | >= 1pt lower | any | pretraining already helps under Arm A's recipe |

  These are n=1 readings. Differences under 1pt are below the project's noise floor and are not findings.

## Stop rule

- The run ends at 50/50 epochs, or it is stopped on an application error, NaN loss, or a collapse. A collapse means val clean
  accuracy under 5% for 2 consecutive epochs after the warmup.
- No extension, restart with changed settings, or second seed without a new human decision.

## Stage 2: per-init learning-rate comparison (human-approved 2026-09-26)

Why: a pretraining comparison is only fair when each init uses a recipe suited to it (human, 2026-09-25), and when both inits
get the same number of tries. Before this stage, pretrained had two tries (0.05, 0.005) and random init had one (0.05).

Frozen contract: Arm A's recipe, 50 epochs, seed 0, deterministic, train probe 2000, on Hamster RTX 4090s from pinned worktrees.
Only `optimizer.learning_rate` (and `student.pretrained`) varies. New runs use protocol id
`controlled_imagenet_stage02_init_lr_grid_v1`.

| init | peak lr | run |
|---|---|---|
| pretrained | 0.005 | `plan0104-mobilenetv4-ft-lr0005-v1` (stage 1 above) |
| pretrained | 0.015 | `plan0104-mobilenetv4-pretrained-lr0015-v1` (`imagenet_mobilenetv4_pgd_at_pretrained_lr0015.yaml`) |
| pretrained | 0.05 | Arm A seed 0 (plan 0101; 54.57 / 30.58) |
| random | 0.025 | `plan0104-mobilenetv4-random-init-lr0025-v1` (`imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml`) |
| random | 0.05 | `plan0103-mobilenetv4-random-init-50ep-v1` (decision 0019 B) |
| random | 0.1 | `plan0104-mobilenetv4-random-init-lr01-v1` (`imagenet_mobilenetv4_pgd_at_random_init_lr01.yaml`) |

Preregistered rule (internal-val PGD-10 at the last epoch, seed 0, n=1 per cell):
- For each init, best = the maximum over its three learning rates. D = best(pretrained) - best(random).
- D >= +1pt: "pretraining helps MobileNetV4-S". D <= -1pt: "random init is better". Otherwise: "no effect above 1pt; pretraining
  is not needed for MobileNetV4-S".
- The argmax learning rate of each init becomes that init's Phase 1 recipe. Clean accuracy is reported next to every number.
- Caveats that will be stated: both sides take the maximum of three noisy runs, which biases both upward by about the same amount.
  The optimum for MobileNetV4-S may not transfer to other architectures, and Phase 1 will check this per model through the
  stage-saturation diagnostic.

Stop rule: same as stage 1. No further learning rates, schedules or seeds without a new human decision.

## Progress log

- 2026-09-25: plan created; config, protocol id and config-identity test added.
- 2026-09-25: launched `plan0104-mobilenetv4-ft-lr0005-v1` on Hamster GPU1 from pinned worktree `source-be9fdfeccb93`
  (seed 0, W&B `lightweight-imagenet-at`), hand-run bundle. The auto-launch waiter's `pgrep -f` matched its own command
  line, so the GPU sat idle ~40 min before a manual launch.
- 2026-09-26: stage 2 (per-init LR comparison) approved by the human; three configs, protocol id and config test added.
- 2026-09-26: stage 2 queued from pinned worktree `source-76280eb86100` by a detached queue script
  (`ard-runtime/.../queues/plan0104_stage2_queue.sh`, logs `queues/gpu{0,1}.log`). Each run starts only after the previous
  process exits and its run bundle records `completed`; otherwise the queue stops. GPU0: after
  `plan0103-mobilenetv4-random-init-50ep-v1`, `plan0104-mobilenetv4-random-init-lr01-v1`, then
  `plan0104-mobilenetv4-pretrained-lr0015-v1`. GPU1: after `plan0104-mobilenetv4-ft-lr0005-v1`,
  `plan0104-mobilenetv4-random-init-lr0025-v1`.
