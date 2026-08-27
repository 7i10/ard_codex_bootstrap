# 0058 — ERT/RSLAD static trajectory stabilization

## Status

Completed on Hamster. This plan records the first static augmentation screen
after the shuffle/augmentation RNG decomposition (`1a82907`); the result
artifact is `docs/experiments/ert_rslad_static_trajectory_stabilization_results_v1.json`.

## Objective

Measure whether one label-preserving, scalable augmentation (the pinned
TreeLLi/DA-Alone-Improves-AT CropShift) improves the mean CIFAR-10 RSLAD
robust trajectory before any Student-history intervention. Mean robustness is
primary; two-seed spread is descriptive only.

## Frozen design

- Dataset: controlled CIFAR-10 45,000/5,000 split; no official test or
  AutoAttack.
- Teacher: Chen2021 LTD WRN34-10, fixed checkpoint and preprocessing.
- Student: canonical SAAD CIFAR ResNet-18.
- Methods: `BASE` (canonical source-keyed crop+flip) and `CROPSHIFT`
  (upstream-order horizontal flip then CropShift(0,11)).
- Seeds: two paired development seeds (1 and 2), frozen before outcomes.
- Training: fresh epoch 0 through 200, SGD and milestones [100,150],
  deterministic source-keyed augmentation, canonical RSLAD KL-PGD10 inner
  attack (8/255, 2/255, 10 steps, random start, teacher-clean target).
- Endpoints: independent validation CE-PGD20 at epochs 49/99/149/199;
  validation trajectory/AUC uses only one consistent attack identity.
- Checkpoints: at least 49/99/149/199 plus best/last; local bundles are
  authoritative and W&B is metrics-only.

## Gates and order

1. Inventory historical BASE runs without pooling incompatible teacher,
   intervention, schedule, RNG, or transform semantics.
2. Audit the pinned upstream implementation and license; implement only the
   explicit policy field and arbitrary-resolution transform.
3. Run focused tests and a deterministic data/runtime canary. Freeze source,
   configs, seeds, hashes, and W&B metadata in one clean commit.
4. Run a cheap Hamster GPU preflight, then the four full trajectories (BASE /
   CROPSHIFT × seeds 1/2), paired by seed.
5. Evaluate fixed validation endpoints, aggregate dense and checkpoint AUC,
   final/best/last/RO-gap, clean metrics, forgetting and sample agreement when
   available, then perform one consolidated scientific review.

## Execution result

- Four production trajectories completed: BASE/CROPSHIFT × seeds 1/2, each
  with 200 metric rows and registered checkpoints 49/99/149/199.
- All 16 independent validation CE-PGD20 endpoint evaluations completed with
  verified checkpoint and sample-row hashes.
- CROPSHIFT robust accuracy exceeded BASE at every registered endpoint in both
  seeds; mean final gain was +1.24 percentage points and mean trajectory-AUC
  gain was +0.897 pp. Final clean accuracy was -0.30 pp on average.
- Per-sample forgetting transitions are unavailable because the campaign saved
  only final train sample-state tables, not dense sample snapshots.
- No official test, AutoAttack, additional seed, or history intervention was
  launched.

## Decision rule

Report paired seed effects and mean/worst-seed values without population
inference. A strong candidate has same-direction effects, positive mean final
robustness and AUC, no worst-seed degradation or clean collapse; spread or
forgetting improvements strengthen but do not replace mean improvement. Do not
add seeds, tune CropShift, start Tiny-ImageNet, or start history intervention
automatically.

## Lineage / storage

The source Git SHA, protocol IDs, teacher SHA, split identity, all RNG seeds,
upstream SHA/license, transform policy/parameters, checkpoint hashes, W&B run
IDs, and host/GPU identity are recorded in the experiment JSON artifacts. No
model or run-bundle artifact is uploaded to W&B.
