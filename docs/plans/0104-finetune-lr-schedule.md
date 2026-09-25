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

## Progress log

- 2026-09-25: plan created; config, protocol id and config-identity test added.
