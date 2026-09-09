# ERT/RSLAD static trajectory stabilization

Status: complete (Hamster production and independent validation endpoints).

The four registered trajectories and all 16 validation endpoints completed.
Machine-readable results are in
`docs/experiments/ert_rslad_static_trajectory_stabilization_results_v1.json`.

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

## Completed production results

All runs used source Git SHA
`ffc217dd635462e1f14c93720561208db2d70254`, the Chen2021 LTD WRN34-10
checkpoint SHA
`fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`, and the
same controlled 45,000/5,000 split. The endpoint attack was CE-PGD20 in pixel
`[0,1]`, epsilon `8/255`, step `2/255`, 20 steps, random start, eval mode;
all 16 endpoint reports have attack identity SHA
`7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
The endpoint files and their referenced checkpoint/row hashes were verified.

### Validation CE-PGD20 endpoints

Values are robust accuracy / clean accuracy. The displayed checkpoint label is
the registered epoch; the serialized checkpoint payload is one less because
the trainer stores zero-based epoch indices.

| seed | arm | epoch 49 | epoch 99 | epoch 149 | epoch 199 |
|---:|---|---:|---:|---:|---:|
| 1 | BASE | 0.4642 / 0.7636 | 0.4760 / 0.7914 | 0.5658 / 0.8484 | 0.5808 / 0.8644 |
| 1 | CROPSHIFT | 0.4680 / 0.7698 | 0.4916 / 0.7846 | 0.5706 / 0.8464 | 0.5940 / 0.8612 |
| 2 | BASE | 0.4596 / 0.7566 | 0.4554 / 0.7760 | 0.5640 / 0.8506 | 0.5810 / 0.8650 |
| 2 | CROPSHIFT | 0.4752 / 0.7626 | 0.4560 / 0.7808 | 0.5762 / 0.8512 | 0.5926 / 0.8622 |

Paired CROPSHIFT minus BASE robust effects are positive at every endpoint in
both seeds: seed 1 `+0.38, +1.56, +0.48, +1.32 pp`; seed 2 `+1.56, +0.06,
+1.22, +1.16 pp` (epochs 49/99/149/199). The mean effects are `+0.97,
+0.81, +0.85, +1.24 pp`; the mean final effect is therefore `+1.24 pp` and
the worst-seed final effect is `+1.16 pp`. Clean effects are mixed at earlier
endpoints and are `-0.32` and `-0.28 pp` at epoch 199 (mean `-0.30 pp`).

### Dense trajectory metrics

The trajectory rows contain 200 points with zero-based metric epochs 0--199.
Normalized trapezoidal validation robust AUC, best, last, and the best--last
gap were:

| seed | arm | robust AUC | best (metric epoch) | last | best--last gap | epochs 150--199 mean |
|---:|---|---:|---:|---:|---:|---:|
| 1 | BASE | 0.505652 | 0.5848 (179) | 0.5816 | 0.0032 | 0.580464 |
| 1 | CROPSHIFT | 0.514603 | 0.5940 (198) | 0.5930 | 0.0010 | 0.589564 |
| 2 | BASE | 0.505478 | 0.5864 (195) | 0.5852 | 0.0012 | 0.582380 |
| 2 | CROPSHIFT | 0.514466 | 0.5950 (178) | 0.5912 | 0.0038 | 0.590272 |

The paired AUC gain is `+0.008950` (seed 1) and `+0.008988` (seed 2), mean
`+0.008969` (`+0.897 pp`). The paired best gain is `+0.92` and `+0.86 pp`,
and the paired last gain is `+1.14` and `+0.60 pp`. The robust-overfit-gap
change is mixed: `-0.22 pp` for seed 1 and `+0.26 pp` for seed 2 (mean
`+0.02 pp`), so the improvement should not be described as a consistent
overfitting reduction.

Median training throughput was approximately 635 images/s (BASE seed 1),
623 (CROPSHIFT seed 1), 637 (BASE seed 2), and 622 (CROPSHIFT seed 2). The
lower throughput of CropShift is an operational cost, not a scientific
endpoint.

### Sample-level agreement (secondary)

At epoch 199, same-seed BASE/CROPSHIFT robust-label agreement was 0.9372
(seed 1) and 0.9348 (seed 2), with robust-positive Jaccard 0.8985 and 0.8947.
Cross-seed robust-label agreement at epoch 199 was 0.9354 for BASE and 0.9410
for CROPSHIFT. These are descriptive endpoint agreements; no sample-level
forgetting trajectory is available because only final train sample-state
tables were saved.

## Interpretation and stop decision

The static augmentation screen supports CROPSHIFT as a candidate for a
follow-up: it improves validation CE-PGD20 robustness at all four registered
checkpoints in both seeds, and both the endpoint final effect and full
trajectory AUC effect are positive. The clean change is small and slightly
negative at the final endpoint, so this is not a clean-accuracy win and should
not be overstated. The two-seed design provides paired descriptive evidence,
not training-seed population inference.

The result does not establish official-test or AutoAttack performance, does
not identify a best checkpoint for test reporting, and does not compare any
history intervention. No additional seed, dataset, augmentation tuning,
history intervention, official test, or AutoAttack was started automatically.
