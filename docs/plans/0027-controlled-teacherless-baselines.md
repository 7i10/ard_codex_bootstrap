# Controlled teacherless baselines

## Status

- Owner: main thread
- Host: Hamster only; Ferret is forbidden
- Base SHA: `6cd0fb3aef5b2374e208b063e72521c83d930152`
- Current milestone: PGD-AT seed-0 training active
- Last updated: 2026-08-06

## Question and order

Run the already implemented teacherless baselines under
`controlled_cifar10_r18_v1` to separate distillation gains from ordinary
adversarial training:

1. PGD-AT seed 0 on Hamster GPU 1;
2. TRADES seed 0 on the same GPU only after PGD-AT reaches a valid terminal
   state and its saved best/last checkpoints are intact.

These are controlled comparisons, not claims of exact official-paper
reproduction.  TRADES uses the repository's documented clean-CE plus
clean-to-adversarial KL implementation with beta 6.

## Frozen identity

- CIFAR-10, `saad_resnet18_cifar_v1`, seed 0, one GPU, global/per-rank batch
  128, 200 epochs, SGD LR .1, momentum .9, weight decay 5e-4, milestones
  100/150.
- Training PGD-10 uses radius 8/255 and step 2/255.  Checkpoint selection uses
  validation CE-PGD-20; official test and AutoAttack are not consulted during
  training.
- Configs are `configs/scientific/cifar10_r18_pgd_at.yaml` and
  `configs/scientific/cifar10_r18_trades.yaml` without scientific overrides.
- W&B online project `single-teacher-ard`; distinct stable run IDs and output
  directories; production guards remain enabled.

## Gates and tests

- Cheap preflight only: clean committed source, dataset bytes available,
  W&B credentials/entity/project, GPU ownership, resolved-config dry-run, and
  existing CUDA smoke evidence from Plan 0014.  Do not repeat parity runs.
- Confirm the live process, unique W&B identity, finite first batch/epoch,
  learning-rate logging, and atomic best/last checkpoint creation.
- Preserve both best and last; later official PGD and AutoAttack remain
  separate saved-checkpoint evaluation processes.

## Progress

- [x] B0 -- commit/push this frozen launch record and pass the cheap preflight.
- [ ] B1 -- launch PGD-AT seed 0 on Hamster GPU 1 and verify first epoch.
- [ ] B2 -- after PGD-AT terminal validation, launch TRADES seed 0 on GPU 1.
- [ ] B3 -- evaluate both best/last on official clean/PGD-20, then schedule
  AutoAttack only after validation results are recorded without changing the
  frozen training protocol.

## Stop conditions

Stop the affected run on config/lineage drift, non-finite loss, duplicate W&B
identity, wrong GPU, missing checkpoint state, or a resume mismatch.  Do not
change attack strength, batch size, scheduler or selection metric in response
to observed accuracy.

## Progress log

- 2026-08-06: B0 passed from clean SHA
  `c2220f11738e8963b922ae379a047a862ffa5915`.  The resolved production config
  binds CIFAR-10, PGD-10/PGD-20, seed 0, batch 128, workers 8, W&B online,
  entity/project and a unique output/run ID.  W&B API authentication succeeded;
  GPU 1 was idle.  Existing Plan-0014 CUDA/parity evidence was reused rather
  than repeated.
- 2026-08-06: B1 launched as `ard-pgd-at-s0-c2220f1.service`, W&B run
  `pgd-at-controlled-s0-c2220f1`, with config hash
  `4f99a3f46498a162930dfc1b7ac9d16fd7afd6c0c70365da3857ec86a39c63ac`.
  Epoch 0 completed in 38.86 seconds at 1,157.9 images/s; loss `2.25255`,
  validation clean `0.3336`, validation PGD-20 `0.2132`, and both distinct
  best/last checkpoints were written.  By epoch 3, validation PGD-20 was
  `0.2666`; the service remained active on physical GPU 1.
- 2026-08-06: the first-epoch gate exposed an observation-only gap: historical
  epoch rows did not include the optimizer learning rate.  The active run is
  not restarted because its fixed scheduler, checkpointed optimizer/scheduler,
  resolved config and scientific updates are valid; its LR trajectory is
  exactly reconstructible (`0.1`, then `0.01`, then `0.001`).  The trainer now
  records both the rate used during the epoch and the checkpointed next-epoch
  rate for TRADES and all future runs.  A focused two-epoch StepLR regression
  passed.  The impact-selected gate reused cached checkpoint/method-switch
  passes; its three two-process Gloo cases timed out only in the isolated
  localhost-socket environment, then the required `--lf` non-isolated rerun
  passed (`3 passed, 3 deselected`).  The change is observation-only and does
  not alter scheduler timing, optimizer state or checkpoint contents, so no
  second scientific review cycle was added.
