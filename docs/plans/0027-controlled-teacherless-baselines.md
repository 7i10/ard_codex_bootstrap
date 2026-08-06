# Controlled teacherless baselines

## Status

- Owner: main thread
- Host: Hamster only; Ferret is forbidden
- Base SHA: PGD-AT `c2220f11738e8963b922ae379a047a862ffa5915`;
  TRADES must launch from the then-current clean pushed SHA
- Current milestone: PGD-AT training complete; official evaluation and TRADES pending
- Last updated: 2026-08-07

## Question and order

Run the already implemented teacherless baselines under
`controlled_cifar10_r18_v1` to separate distillation gains from ordinary
adversarial training:

1. PGD-AT seed 0 on Hamster GPU 1;
2. evaluate its saved best/last checkpoints on official CIFAR-10 test;
3. run TRADES seed 0 on Hamster in parallel with the PGD-AT evaluation once
   the PGD-AT terminal state and saved checkpoints are verified.

Ferret is not used. The two Hamster GPUs may run independent saved-checkpoint
evaluation and training jobs concurrently; there is no scientific dependency
between them.

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
- [x] B1 -- launch PGD-AT seed 0 on Hamster GPU 1, verify first epoch, and
  validate its successful 200-epoch terminal state.
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
- 2026-08-07: B1 reached a valid terminal state with 200/200 epoch rows and a
  finished W&B run (`pgd-at-controlled-s0-c2220f1`). Best was epoch 102:
  validation clean `83.14%`, validation CE-PGD-20 `51.80%`. Last was epoch
  199: clean `85.08%`, PGD-20 `43.26%`, an `8.54 pp` best-to-last robust gap.
  Mean validation PGD accuracy was `46.326%` over epochs 100--199, `45.325%`
  over 120--199, and `44.698%` over 150--199. Distinct best/last checkpoint
  epochs, complete state, W&B identity, and SHA-256 values were verified.
  These are validation results, not official-test results.

## Hamster-only next execution block

1. **GPU 0:** evaluate PGD-AT best and last on official CIFAR-10 clean and
   CE-PGD-20 from the saved checkpoints. Record the results before starting
   AutoAttack; then run pinned AutoAttack for best and last in the separate
   evaluation process.
2. **GPU 1:** start controlled TRADES seed 0 from the latest clean pushed SHA,
   with `configs/scientific/cifar10_r18_trades.yaml` unchanged. Reuse the
   existing CUDA smoke evidence; only perform the cheap environment, config,
   W&B and GPU preflight.
3. After TRADES finishes, evaluate its best and last with the same official
   clean/PGD-20 and AutoAttack contract.
4. Close the seed-0 baseline screen using official-test Best, Last, and robust
   overfitting gap only. Do not compare PGD-AT validation values with the
   existing official RSLAD/entropy table.

This block changes no method based on official-test observations. It closes
already-frozen baselines. No new student-history intervention is launched:
Plans 0022/0023 found prediction without actionable Best improvement.

## Decision after B3

- If controlled TRADES is at least as strong as the controlled distillation
  methods within the uncertainty of one seed, prioritize robust-overfitting and
  teacher-transfer mechanism analysis rather than another sample-gating v4.
- If RSLAD materially exceeds TRADES while Chen full-SAAD does not improve over
  RSLAD, treat teacher transfer as useful but the present SAAD oracle as
  insufficient evidence for entropy/IGDM gains under the controlled protocol.
- Do not run Bartoldson upstream full-SAAD on Hamster: exact batch 128 exceeded
  the preregistered 24-GB safety bound. Batch reduction or two-GPU execution
  would change the frozen upstream execution identity.
- A paper-level performance claim requires additional seeds. After seed-0
  baseline closure, freeze a narrow replication cohort (PGD-AT, TRADES, and the
  closest controlled distillation baseline) before inspecting any new official
  test results; do not expand all historical 8 cells.
