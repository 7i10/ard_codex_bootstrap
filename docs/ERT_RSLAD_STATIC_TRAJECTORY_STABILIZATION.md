# ERT/RSLAD static trajectory stabilization

Status: protocol and implementation frozen; full Hamster training is pending
the source-freeze commit and GPU launch gate.

## What is fixed

The first screen compares canonical source-keyed RandomCrop(32, padding=4) plus
horizontal flip (`BASE`) with the pinned upstream-order horizontal flip plus
`CropShift(0, 11)` (`CROPSHIFT`). Both use Chen2021 LTD WRN34-10, the canonical
SAAD CIFAR ResNet-18, the same 45k/5k split, seeds 1 and 2, optimizer,
schedule, RSLAD objective, and KL-PGD10 training attack. Only the train-view
augmentation policy differs. The new explicit protocol identity
`controlled_cifar10_r18_cropshift_v1` prevents accidental pooling with the
canonical baseline.

## Retrospective inventory

The inventory is in
`docs/experiments/ert_rslad_trajectory_baseline_inventory_v1.json`. One
complete historical Chen seed-1 BASE is retained as a retrospective diagnostic,
but it is not reused as the primary pair because its augmentation field/source
contract predates this explicit policy and no exact seed-2 companion exists.
Intervention forks, Bartoldson runs, delayed schedules, and other objectives
are excluded rather than pooled.

## Upstream and implementation audit

The reference is TreeLLi/DA-Alone-Improves-AT commit
`38b740aeffe5933c16869a126c6972ef443a8352` (MIT). The local mapping,
algorithm details, source-keyed RNG rule, and canary hashes are recorded in
`docs/experiments/ert_rslad_cropshift_rng_audit_v1.json`. The transform supports
arbitrary image height/width and clamps only the maximum feasible shift; CIFAR
uses the exact exclusive high value 11. No IDBH color/shape, erasing, CutMix,
MixUp, or learned policy is included.

## Decision boundary

After the four fresh full trajectories and fixed validation CE-PGD20 endpoints
at epochs 49/99/149/199, report paired effects, mean/worst seed, trajectory AUC,
RO gap, clean metrics, forgetting, and sample agreement when available. Two
seeds support descriptive paired direction only. A positive mean robust effect
and non-degraded worst seed are required for a strong static candidate; spread
alone cannot promote a lower-accuracy method. No additional seed, dataset,
history intervention, official test, or AutoAttack will be started
automatically.
