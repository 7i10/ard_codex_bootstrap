# ERT History-Smoothed S3 Production Screen

Status: completed; no Stage B or follow-up training was started.

## Scope and validity

The screen compares `BASE`, `INST075`, `M3_075`, and `M3E2_075` from the
exact Chen ERT epoch-79 parents for L2/seed 1 and L4/seed 2. Each run
continues through epoch 94 and is evaluated at epochs 84, 89, and 94 with an
independent eval-mode CE-PGD20 attack. The endpoint uses pixel `[0,1]`,
Linf, epsilon `8/255`, step `2/255`, 20 steps, random start, and hard-label
CE. Only the fixed internal validation split is used below; no official test
or AutoAttack was run.

An initial non-deterministic screen was retained in the cache for audit but is
not used as evidence. The runtime was corrected to apply the parent
`deterministic: true` CUDA contract, then all eight trajectories and 48
endpoint jobs were rerun. The accepted v2 run passed the epoch-80 parity gate:
BASE/M3/M3E2 model, optimizer, scheduler, scaler, RNG, sampler, sample-state,
global-step hashes and capture-ID hashes agree for both seeds; INST is expected
to differ because it acts at epoch 80.

## Endpoint robust / clean accuracy

Values are validation accuracy. Deltas in parentheses are versus BASE at the
same seed and horizon.

| seed | epoch | BASE | INST075 | M3_075 | M3E2_075 |
|---|---:|---:|---:|---:|---:|
| L2 | 84 | 0.4680 | 0.4724 (+0.0044) | 0.4720 (+0.0040) | 0.4670 (-0.0010) |
| L2 | 89 | 0.4564 | 0.4556 (-0.0008) | 0.4682 (+0.0118) | 0.4660 (+0.0096) |
| L2 | 94 | 0.4614 | 0.4596 (-0.0018) | 0.4526 (-0.0088) | 0.4492 (-0.0122) |
| L4 | 84 | 0.4510 | 0.4546 (+0.0036) | 0.4520 (+0.0010) | 0.4638 (+0.0128) |
| L4 | 89 | 0.4746 | 0.4702 (-0.0044) | 0.4670 (-0.0076) | 0.4644 (-0.0102) |
| L4 | 94 | 0.4754 | 0.4738 (-0.0016) | 0.4708 (-0.0046) | 0.4656 (-0.0098) |

Validation clean accuracy (BASE, INST075, M3_075, M3E2_075 respectively) was:

| seed | epoch 84 | epoch 89 | epoch 94 |
|---|---|---|---|
| L2 | 0.7770, 0.7844, 0.7742, 0.7754 | 0.7950, 0.7786, 0.7832, 0.7848 | 0.7740, 0.7724, 0.7608, 0.7812 |
| L4 | 0.7874, 0.7870, 0.7796, 0.7920 | 0.7870, 0.7872, 0.7866, 0.7946 | 0.7888, 0.7810, 0.7982, 0.7906 |

## Route dynamics

| seed | arm | switches | re-entry | active fraction | median duration | teacher-only share |
|---|---|---:|---:|---:|---:|---:|
| L2 | INST075 | 149,468 | 56,195 | 0.2155 | 1 | 0.0275 |
| L2 | M3_075 | 70,797 | 19,072 | 0.1547 | 2 | 0.0719 |
| L2 | M3E2_075 | 73,801 | 21,181 | 0.1871 | 2 | 0.1321 |
| L4 | INST075 | 149,556 | 56,210 | 0.2140 | 1 | 0.0283 |
| L4 | M3_075 | 71,101 | 19,345 | 0.1546 | 2 | 0.0689 |
| L4 | M3E2_075 | 74,280 | 21,436 | 0.1863 | 2 | 0.1312 |

Compared with Instant, Majority-3 reduced switches by about 52.6–52.7% and
re-entry by about 65.9–66.0% in both seeds. Exit-2 removed one-visit exits in
the trajectory, but increased switches and re-entry relative to plain
Majority-3; it did not provide additional action stability under this
definition.

## Paired rescue / harm diagnostics

These are descriptive paired effects against BASE. `selected_train` uses each
arm's registered epoch-80 capture cohort; `validation_all` uses all 5,000
fixed validation IDs. They are not training-seed uncertainty estimates.
Class-stratified bootstrap confidence intervals were not run for this screen;
the reported point estimates must not be interpreted as seed-level uncertainty.

At the primary epoch 94 endpoint:

| seed | arm | selected-train rescue / harm / net | validation rescue / harm / net |
|---|---|---|---|
| L2 | INST075 | 0.0859 / 0.0836 / +0.0023 | 0.0530 / 0.0548 / -0.0018 |
| L2 | M3_075 | 0.0792 / 0.0821 / -0.0029 | 0.0450 / 0.0538 / -0.0088 |
| L2 | M3E2_075 | 0.0680 / 0.0821 / -0.0141 | 0.0416 / 0.0538 / -0.0122 |
| L4 | INST075 | 0.0843 / 0.0793 / +0.0050 | 0.0418 / 0.0434 / -0.0016 |
| L4 | M3_075 | 0.0812 / 0.0779 / +0.0032 | 0.0422 / 0.0468 / -0.0046 |
| L4 | M3E2_075 | 0.0693 / 0.0787 / -0.0093 | 0.0370 / 0.0468 / -0.0098 |

## Interpretation

- **State smoothing:** supported for reducing flapping, not for endpoint
  robust accuracy. `M3_075` is below `INST075` at epoch 94 for both L2 and
  L4 and below BASE for both seeds.
- **Action persistence:** not supported as a performance or stability gain in
  this screen. `M3E2_075` is below `M3_075` at epoch 94 for both seeds, and
  its switch/re-entry counts are higher than plain Majority-3.
- **Final utility:** neither history arm beats BASE at the primary epoch-94
  validation endpoint in both seeds. The weak `beta_advce=0.075` route should
  not be promoted or tuned automatically.
- The common positive epoch-89 L2 result for M3 is not replicated by L4 and
  therefore is not a confirmatory claim.

## Provenance

- Accepted training source SHA: `ea444179431f982beef95505c4323cbeb722cec9`.
- Training parent SHA: L2
  `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`;
  L4 `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`.
- Chen teacher checkpoint SHA:
  `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.
- Endpoint CE-PGD20 identity SHA:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- Config SHA: `13cf29faf9c1a0170262e3f5db08f368d9e0f432a3dca412ed53a63c3139e536`.
- Machine report SHA:
  `254ed4d45875880e6adf151753b6a0d453a5bad420d9a727064adeef103a71c4`.
- All eight accepted runs were W&B `online` and completed; run IDs use
  `ert-dynamic-s3-recovery-{seed}-{arm}-ea44417`.

No official test, AutoAttack, new seed, coefficient sweep, or follow-up
training was started after this screen.
