# ERT Clean-Wrong Teacher-Adaptive Margin Local Lambda Stability

Status: in progress (preflight/canary before production)

## Question

Test whether the calibrated teacher-adaptive, positive-floor, margin-only
coefficient is locally stable or whether nearby values are dominated by
continuation variance.  The frozen target is

$$
t_i = \operatorname{clip}(m_{T,i}^{adv}, 0.03221710026264191,
0.13952550292015076),
$$

and the selected Clean-Wrong cohort receives no CleanCE.  The margin term is
added to the baseline RSLAD loss with one of the preregistered coefficients:

| arm | relative scale | coefficient |
|---|---:|---:|
| B0_BASE | 0 | 0 |
| N90 | 0.90 | 0.21492460370063782 |
| N95 | 0.95 | 0.22686485946178436 |
| A100 | 1.00 | 0.2388051152229309 |
| N105 | 1.05 | 0.25074537098407745 |
| N110 | 1.10 | 0.262685626745224 |

No coefficient, floor, cap, threshold, or arm is added after observing an
outcome.

## Frozen inputs

- L2 parent: exact SHA
  `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`.
- L4 parent: exact SHA
  `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`.
  The prior recovery audit establishes this as the downstream causal parent;
  the historical upstream `9b51...` byte hash is not substituted.
- L2 Clean-Wrong mask: 8,623 IDs, registered artifact SHA
  `0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b`.
- L4 Clean-Wrong mask: 8,925 IDs, registered artifact SHA
  `fe818e755e4b2da7a5beb7e1a791a52ab92905f01064870237972bb58344a6`.
- Training attack: pixel-space KL-PGD10, $\epsilon=8/255$, step $2/255$,
  random start, Teacher-clean target, eval-mode Teacher.
- Endpoint attack: independent CE-PGD20 with the existing common identity.
- Calibration: `docs/experiments/ert_cw_margin_calibration_v1.json`, SHA
  `a625b43ec12277bbf698270193f27e0e1f62e0a2a9f9a6a49e7fc0702593b2b5`.

The L2 parent config is the hash-bound observed-run config whose digest is
`250f24f...`; the L4 parent config is the hash-bound C79 config whose digest
is `f5c64f...`.  Runtime must reject any other parent/config pair.

## Replicates and execution

Each seed has two continuation replicates, R1 and R2, giving four matched
blocks and 24 trajectories.  A deterministic continuation seed is fixed
before launch.  All six arms in a block share that seed; R1 and R2 differ.
The Stage A runtime records the continuation seed in the child identity and
re-seeds only the post-resume attack stream after restoring the full epoch-79
optimizer/scheduler/sampler/sample state.

The current operational host is Hamster (the local server, two RTX 4090s),
which is the measured faster host.  Jobs are launched in two-GPU waves with
separate output directories.  Ferret is not used for this campaign unless a
later explicit decision changes the host contract.  W&B uses metrics-only
retention; checkpoints and run bundles remain local.

## Gates

1. Verify clean Git tree, parent/config/mask/calibration hashes, dataset and
   Teacher identities, and idle Hamster GPUs.
2. Run changed tests and a one-epoch N105 canary on L2-R1.
3. Confirm finite loss/metrics, exact parent restore, no CleanCE, exact floor,
   cap and coefficient, continuation seed logging, local checkpoint/resume,
   and metrics-only tracking.
4. Launch the 24 frozen trajectories to epoch 94, retaining epoch 84/89/94
   checkpoints.
5. Run independent CE-PGD20 endpoints at 84/89/94; report train direct,
   non-CW spillover, held-out overall and held-out Clean-Wrong effects.
6. Compare A100/N95/N105 for the primary ±5% neighborhood, then N90/N110
   for the exploratory ±10% neighborhood.  Compare between-replicate spread
   with within-block lambda differences descriptively; do not make a
   training-seed population claim from two replicates per seed.
7. Write the machine result and human report, then stop.  Do not auto-select
   a lambda or start another intervention.

## Exclusions

No CleanCE, floor/cap sweep, exact-0.25 re-test, new threshold, new seed,
official test, AutoAttack, dynamic routing, or post-hoc coefficient tuning.
