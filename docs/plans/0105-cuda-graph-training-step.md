# Plan 0105: CUDA-graph training step (`training.cuda_graph`)

## Context

Plan 0103 stage 1 (MobileNetV4-Conv-Small, 112 px, batch 128, PGD-1, deterministic,
`step_diagnostics: false`) is kernel-launch bound. The real trainer runs at about 2600 img/s on a
Hamster 4090 while the GPU has spare capacity (batch 256 reaches 6669 img/s). A throwaway prototype
on Ferret showed that a full training step captured as one CUDA graph gives bit-identical results to
the eager step in deterministic mode, and runs much faster. The human approved this as separate work
on 2026-09-30.

## Goal

A default-off option `training.cuda_graph` that replays the PGD-AT training step as one CUDA graph.
It must give bit-identical checkpoints, RNG streams, epoch rows and diagnostics to the eager trainer
in deterministic mode. With `training.deterministic: false` (allowed since 2026-10-03, see below) it must
keep every RNG stream exactly and one step must match an eager step within eager noise / FP32 rounding.
It must refuse every configuration where that is not proven.

## Design

- **Attack core.** `LinfPGD.generate` is split into a sync-free core, `LinfPGD.perturb` (inputs, labels,
  epsilon/step tensors, raw random-start noise -> adversarial batch), and a validating wrapper
  (`generate`: pixel check, budget checks, random-start draw, `max_abs_delta`). The eager path and the
  graph path call the same core. `generate` is bit-identical to the pre-change version (tested against
  golden output digests produced by the pre-change file, `tests/unit/fixtures/pgd_generate_golden_4fd5ceb.json`).
- **Config.** `training.cuda_graph: bool = False`, serialized only when true, so every existing config
  keeps a byte-identical resolved config and hash (tested against the pre-change schema). It is refused
  unless: `device: cuda`, no AMP, no compile, `step_diagnostics: false`, world size 1
  (`global_batch_size == per_rank_batch_size`), no epsilon warmup, method `pgd_at` without `mixed_batch` / `awp`,
  SGD, no teacher / ADR / target policy / intervention /
  prescriptive route / observation profile, and a CE, eval-mode, batch-keyed, untraced attack with a
  fixed budget (0 < epsilon, step <= epsilon). The student architecture must be on the parity-tested
  allowlist `CUDA_GRAPH_ARCHITECTURES` (MobileNetV4-Conv-Small, MobileNetV4-Conv-Medium, EfficientNet-B0).
  Both `deterministic: true` and (since 2026-10-03) `deterministic: false` are admitted; `cudnn_benchmark`
  stays refused (see Verification). The Trainer repeats the same checks for direct construction (it also
  refuses a `LinfPGD` subclass, deterministic `warn_only` mode and cuDNN benchmark).
  Since 2026-10-08 (plan 0103 Phase 2 batch A) three options are in scope: a plain weight EMA
  (`training.weight_ema_decay`; `_update_ema` runs inside the captured step right after the SGD update, the
  same kernels as the eager step, and the stale-graph fingerprint covers the EMA tensors' addresses),
  `method.label_smoothing` (part of the captured objective) and `optimizer.exclude_norm_bias_from_weight_decay`
  (two SGD parameter groups, each with its own baked-in weight decay).
  Only `ard.cli.train` applies the option; `reject_throughput_options` refuses it everywhere else.
- **Capture lifecycle.** Static device buffers for images, labels, the valid mask and the random-start
  noise. The loader batch is copied into them (non-blocking from pinned memory). The first full batch
  of every epoch runs eagerly (it creates SGD momentum buffers on a fresh run and warms up lazily
  initialized CUDA libraries); the next batch captures the graph and every later full batch replays it.
  The last partial batch, and any batch that is not contiguous in the captured layout, runs eagerly.
  Eager and replayed steps add into one static epoch accumulator, in step order, read once at epoch end.
- **Audit.** With the flag on, each epoch row carries `train_cuda_graph_captures`, `train_cuda_graph_replays`
  and `train_cuda_graph_eager_steps`. An epoch with at least three full batches and no replay raises before
  the epoch's checkpoint, so the option can never silently fall back to eager training.
- **Re-capture guard.** A graph bakes in SGD's `lr`, `momentum`, `weight_decay`, `nesterov`,
  `dampening` as Python scalars (a tensor `lr` is not bitwise equal) and the addresses of parameters,
  momentum buffers and model buffers. The graph is therefore captured again at the start of every
  epoch (the scheduler changes the LR only at epoch ends; resume loads new optimizer tensors and also
  invalidates the graph). Before every replay a fingerprint of all param-group hyperparameters,
  parameter identities and addresses, momentum-buffer addresses, model-buffer addresses and the model
  mode is compared with the one taken at capture. Any difference raises; the trainer never replays
  stale values.
- **Random start.** Drawn outside the graph into the static noise buffer, from the same freshly seeded
  per-step generator (`seed + 1000003 * global_step`) the eager attack uses. Resume therefore keeps the
  stream (the global step drives it).
- **Guards.** `LinfPGD.generate`'s pixel check runs outside the graph on the static input buffer, as a
  `torch._assert_async` (no host sync); this is the guard that catches out-of-range pixels on the graph
  path. Inside the graph, the finite-loss check fires from a replay (tested). The model adapter's pixel
  check sees the raw batch only in the diagnostics clean forward: PGD clamps its output into [0, 1], so
  the attack and training forwards never see an out-of-range pixel. It fires from a replay when
  diagnostics are on (tested) and is not reached when they are off. A fired assert kills the CUDA
  context, so the poisoned update never reaches a checkpoint (tested). The host-side eager guards are
  unchanged.
- **Diagnostics.** Production configs require panel diagnostics, so they are supported: the eval-mode
  clean forward runs inside the graph at the same point as in the eager step, and its predictions are
  read back in one transfer per epoch (and before any eager step, to keep row order).
- **Pooling identity.** `training.cuda_graph` is always in the config hash. Whether it is in
  `training_protocol_identity` depends on the determinism class:
  - Deterministic mode: not in it, like `step_diagnostics`, so graph and eager seeds of one arm pool. This
    holds only because the scope checks confine the flag to what the parity test proves bitwise equal end
    to end (checkpoints, RNG, rows, diagnostics; LR milestones, partial batch, probe pass, resume): the
    allowlisted architectures and an eval-mode training attack, measured on torch 2.11 with an RTX 4090.
    The production-shape check below matched too.
  - Nondeterministic mode (`deterministic: false`): `cuda_graph: true`
    is recorded (2026-10-03 review). The tests show exact RNG streams and a one-step result equal to FP32
    rounding, but not bitwise equality, so a graph run is never
    pooled silently with a nondeterministic eager run. No such run existed before, so no recorded identity
    changes.
  `training.deterministic` is always in `training_protocol_identity`, and `cudnn_benchmark` when true, so
  deterministic and nondeterministic runs never pool. Widening the scope, or upgrading torch or the
  driver, requires rerunning the parity and equivalence tests first.

## Runbook

- The flag is part of the config hash, and resume refuses a config-hash change. A run therefore cannot
  switch between graph and eager on resume. If the graph path fails in a way specific to it, the fallback
  is a restart of the run with `cuda_graph` off (a new run identity), not a resume.
- A capture failure, a stale-graph refusal or an epoch without replays stops the run loudly before that
  epoch's checkpoint; the previous epoch's `last.pt` is intact.

## Verification

- `tests/integration/test_cuda_graph_training_step.py` (GPU, deterministic): real `Trainer.fit`, 3 epochs,
  LR milestones after epochs 1 and 2, partial last batch, a train-probe pass every epoch. Each arm runs in
  its own subprocess (cold caches, as in production). Students: every allowlisted architecture at 32 px
  (panel diagnostics) and a BatchNorm + dropout fixture (panel, summary and no diagnostics; the dropout
  draw advances the CUDA RNG inside the graph, asserted). Eager vs graph: identical epoch rows,
  `last.pt`/`best.pt` state (model, optimizer, scheduler, all RNG streams, sampler, selection), diagnostics
  rows and panel media, CUDA RNG state, and the expected audit counts. A graph run resumed from the epoch-1
  checkpoint matches the uninterrupted eager run (fixture and MobileNetV4-S). A negative control (attack
  seed + 1 on the graph arm) must differ. Guard tests: a mid-epoch change of `lr`, `momentum`,
  `weight_decay`, optimizer state tensors or model mode is refused; an epoch that never replays fails before
  any checkpoint; in-graph pixel and finite-loss asserts fire from a replay; an out-of-range pixel on the
  graph path aborts before any checkpoint; `perturb` makes no host sync.
- Nondeterministic mode, same file (human decision 2026-10-03), for the fixture and every allowlisted
  architecture. Bitwise parity is impossible here, so:
  (1) whole runs (the 3-epoch run above, eager and graph each in their own process): Python/NumPy/torch
  CPU/CUDA RNG states in `last.pt`, the final CUDA RNG state (the fixture's dropout advances it inside the
  graph), the seed and final philox state of every per-step attack generator, global step, scheduler and
  sampler state are exactly equal, and the audit counts show the graph ran. This confirms that
  nondeterministic kernels do not touch any RNG stream. Weights are not compared over whole runs: PGD's
  sign step makes training chaotic. A 1e-9 relative state difference flips the sign of up to a quarter of
  the attack's input gradients one step later, so two eager nondeterministic runs drift apart to the size of
  a different seed within a few steps (measured).
  (2) One step from one exact state (rewritten after the 2026-10-03 review; the first version compared one
  L2 norm over parameters and exploding BatchNorm buffers, so the FP32 floor of the buffers decided it and a
  graph arm with lr = 0 passed). A reference eager process saves its exact state (parameters, BatchNorm
  buffers, SGD momentum buffers, CUDA RNG) before the capture step (batch 1) and a later replay (batch 3),
  and its state after each step. Every other process loads that state in place (addresses unchanged, so
  the graph stays valid) and takes the same step: two `pair` processes with two eager and two graph arms
  each, and a `controls` process with two eager arms and four defective graph arms (attack seed + 1,
  lr = 0, lr x 1.001, weight decay 0, set before the capture step). Per tensor group (parameters, momentum
  buffers, BatchNorm buffers), distances are relative to the reference's own update of that group, with a
  floor of one FP32 rounding of the group's new value on the same scale. One step's nondeterminism can be
  bimodal: a summation-order difference flips the sign of an attack input gradient and moves the whole step,
  in eager arms too (seen at production shapes: some eager arms 4e-3 and 3e-2 of a step away from the
  others). So each graph outcome is compared with one nearest eager outcome (the same eager arm for all
  groups), and the noise level is the eager outcomes' spread: the median over the six eager arms of the
  distance to their nearest other one, so a single eager arm in the other mode cannot widen the bound.
  Rule: graph <= 4 x max(spread, floor) in every group; a spread above 100 x the floor fails the check;
  every control >= 10 x that bound from every eager outcome, at both sync points (the defect is set before
  the capture step, so the later replay carries it too). The cross-process difference (eager_a vs the
  reference) is reported, not used. Regime: 64 px, batch 32, 10 classes, PGD-2, lr 0.002 (EfficientNet-B0
  0.015, fixture 8 px and 0.05). Asserted: finite step, BatchNorm running variances <= 1e4, parameter update
  <= 5% of the weights; measured: running variances <= 10, update about 1% of the weights.
  Measured in this regime: graph arms sit as close to the nearest eager outcome as eager arms sit to each
  other (parameters 1e-7 to 3e-7 of the update, momentum 2e-8 to 2e-7, BatchNorm buffers bitwise equal;
  the fixture is bitwise equal throughout), against FP32 floors of about 1e-5 (parameters) and 2e-7
  (momentum). Controls: lr = 0 gives 1.0, lr x 1.001 gives 1.0e-3, weight decay 0 gives 2e-4 to 6e-3,
  attack seed + 1 gives 0.04 to 1.2 (capture step); every control is at least 10x the bound at both sync
  points (asserted; GPU1 peak with the production job beside it: 5.4 GB in total).
  cuDNN benchmark: with the same rule, a graph arm of MobileNetV4-Conv-Medium (64 px, benchmark on) landed
  0.27 (parameters) and 0.30 (momentum) of a step away from every eager outcome of its process, whose spread
  was 1.3e-7: the captured step had used a different cuDNN algorithm than the eager steps. That is the size
  of the difference between two separate benchmark runs (0.1 to 0.3 of a step at production shapes), not a
  rounding, so `cudnn_benchmark` stays refused together with `cuda_graph` until that is understood. (In 24
  earlier benchmark graph arms at production shapes, the graph matched its process's eager arms to 1e-7.)
  (3) Production shapes: committed as the opt-in test
  `test_production_shape_graph_step_matches_eager_within_one_step_noise` (skipped unless
  `ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1`; checkpoints from `ARD_CG_CHECK_MNV4S_STAGE1_CKPT`,
  `ARD_CG_CHECK_MNV4S_FULL_CKPT`, `ARD_CG_CHECK_MNV4M_CKPT`; the exact command is in the test file). It needs
  about 15 GB of GPU memory per process: run it on a FREE GPU only. The numbers below are from one run on
  2026-10-03, before the median-spread / common-nearest / both-sync-controls rule (the controls then ran at
  the capture step only): the same one-step check, batch 128, 1000 classes, cuDNN benchmark off.
  MobileNetV4-Conv-Small from trained plan-0103 checkpoints (stage 1 at 112 px; stage 2 at 224 px),
  MobileNetV4-Conv-Medium from the trained plan-0103 phase-1 checkpoint (both shapes), EfficientNet-B0 from
  random initialisation (no checkpoint on Hamster; lr 0.5 so its update clears the FP32 floor). Stage 1:
  112 px, PGD-1, step 4/255, lr 0.025; full AT: 224 px, PGD-3, step 8/765, lr 0.05; epsilon 4/255.
  All six cases pass, at both sync points. Distances relative to the reference's update:

  | student, shape | update / weights | graph vs nearest eager (params / momentum) | eager spread (params / momentum) | FP32 floor (params / momentum) | lr x 1.001 | weight decay 0 | attack seed + 1 |
  |---|---|---|---|---|---|---|---|
  | MNv4-S, 112 px | 8.6e-3 | 4.7e-7 / 3.6e-7 | 5.1e-7 / 3.5e-7 | 1.4e-5 / 1.6e-7 | 1.0e-3 | 6.1e-4 | 0.40 |
  | MNv4-S, 224 px | 1.6e-2 | 4.6e-7 / 4.7e-7 | 4.7e-7 / 4.7e-7 | 7.3e-6 / 1.6e-7 | 1.0e-3 | 6.3e-4 | 0.37 |
  | MNv4-M, 112 px | 9.3e-3 | 8.3e-8 / 7.5e-8 | 8.6e-8 / 7.8e-8 | 1.3e-5 / 1.6e-7 | 1.0e-3 | 5.6e-4 | 0.41 |
  | MNv4-M, 224 px | 1.6e-2 | 1.7e-7 / 1.7e-7 | 1.8e-7 / 1.7e-7 | 7.5e-6 / 1.6e-7 | 1.0e-3 | 6.5e-4 | 0.29 |
  | EffB0, 112 px | 9.9e-3 | 3.0e-7 / 3.0e-7 | 2.7e-7 / 2.7e-7 | 1.2e-5 / 1.4e-7 | 1.0e-3 | 1.0e-2 | 0.31 |
  | EffB0, 224 px | 5.1e-3 | 4.6e-7 / 4.8e-7 | 4.6e-7 / 4.9e-7 | 2.3e-5 / 1.6e-7 | 1.0e-3 | 2.0e-2 | 0.08 |

  (Capture step, batch 1; the later replay, batch 3, gives the same picture. Control columns: the
  largest group distance; lr = 0 gives 1.0 everywhere. BatchNorm buffers were bitwise equal in every
  graph arm; maximum running variance 3.9.) For the momentum buffers the eager spread, not the floor,
  sets the bound, so there the graph is shown equal to a measured noise level. The earlier run of this
  check, before the rule was changed to the nearest eager outcome, also covered cuDNN benchmark on for all
  six cases: graph arms matched their process's eager arms to 1e-7 everywhere, separate benchmark processes
  differed by 0.1 to 0.3 of a step, and some eager arms landed in the other mode of a bimodal step (EffB0
  224 px: 4e-3; MNv4-S 224 px: 3e-2), which is what led to the nearest-outcome rule.
- `tests/unit/test_cuda_graph_config.py` (CPU): default off and unserialized; every config that existed at
  4fd5ceb (read from git) resolves and hashes byte-identically; every fail-closed combination (including an
  off-list architecture and a train-mode attack); `deterministic: false` with and without
  the other restrictions (cuDNN benchmark included) still refused; the Trainer's determinism scope (strict
  on and off admitted; `warn_only` and cuDNN benchmark refused);
  the pooling identity in both determinism classes; `reject_throughput_options`; Trainer scope (architecture,
  train-mode attack, `LinfPGD` subclass, `warn_only` determinism); static-buffer admission (contiguity,
  layout); the no-replay check; fingerprint sensitivity; `generate` against the golden digests.
- Throughput on Hamster GPU0 (GPU1 was running the stage-1 production job), stage-1 S=256 config,
  real trainer (`train_epoch` with panel diagnostics), 400-600 steps, img/s after 60 warm-up steps:

  | loader | eager | cuda_graph |
  |---|---|---|
  | 16 workers, real data | 2591 | 6877 |
  | 32 workers, real data | - | 6898 |
  | cached batches (no loader) | 3036 | 6804 |

  So the graph step is about 2.65x faster. It is GPU-bound, not loader-bound: 16 and 32 workers and
  cached batches give the same rate. Peak allocated memory is 0.81 GiB (eager 0.73). The same run
  compared the eager and graph states after 400 and 600 real steps: model, momentum buffers and all
  51,200 / 76,800 diagnostic rows were bitwise equal.
- Throughput, deterministic vs nondeterministic (2026-10-03). Hamster GPU1 only, SHARED with a running
  production job (stage-1 50-epoch run, GPU at 45-100% utilisation), so absolute rates are roughly half a free
  GPU's and noisy; only the ordering is informative. Real trainer (`train_epoch`, panel diagnostics, no
  loader: 8 cached synthetic batches), MobileNetV4-Conv-Small at batch 128, img/s over 200 steps after 40
  warm-up steps, 1 to 3 runs each (range shown). Stage 1 = the S=256 stage-1 config (112 px, PGD-1);
  full AT = `imagenet_mobilenetv4_pgd_at_random_init_30ep_cg.yaml` (224 px, PGD-3).

  | mode | stage 1, 112 px | full AT, 224 px |
  |---|---|---|
  | deterministic, eager | 1580-2297 | 512-732 |
  | deterministic, cuda_graph | 2691-3060 | 553-640 |
  | nondeterministic, eager | 1812 | 613 |
  | nondeterministic, cuda_graph | 3517-3609 | 679-775 |
  | nondeterministic + cudnn_benchmark, eager | 1788 | 617 |
  | nondeterministic + cudnn_benchmark, cuda_graph | 3710-3766 | 780-942 |

  The medians of these contended runs put the nondeterministic graph step about 19% above the
  deterministic graph step at 112 px (3605 vs 3028) and about 23% at 224 px (684 vs 554), and cuDNN
  benchmark about 24% (3750) and 54% (851) (benchmark is refused with the graph since the review). These
  are indicative only: the production job's load varied
  during the runs (the deterministic eager rate alone spans 1580-2297 and 512-732), so a clean
  measurement needs a free GPU.

## Progress log

- 2026-09-30: implemented (attack core split, config flag and scope checks, Trainer capture lifecycle,
  stale-graph guard, deferred diagnostics), tests added, throughput and production-shape parity measured
  on Hamster GPU0. Not merged, not launched; enabling it for a run is the human's decision.
- 2026-09-30: two scientific reviews found no divergence within the tested scope. Applied their batch:
  architecture allowlist and eval-mode attack requirement (pooling exclusion now rests on it), wider
  parity test (all allowlisted students, probe pass, diagnostics modes, separate-process arms, negative
  control, dropout RNG check), exact `LinfPGD` type, contiguity check, audit columns and the no-replay
  failure, strict determinism, git-pinned config test and golden attack digests, runbook. The widened
  GPU tests still have to run on a free GPU.
- 2026-09-30: first GPU run of the widened test failed on the checkpointed `rng` entry only (model,
  optimizer, rows identical). Cause: the test harness, not the graph path. The `rng` entry holds the
  Python and NumPy streams as well as torch CPU/CUDA; the test seeded only torch, and each arm now runs
  in its own process, so the unseeded Python/NumPy states differed between arms (they differed between
  two eager processes too). torch CPU and CUDA RNG states were identical in every arm, with and without
  the probe pass. The earlier same-process test passed because both arms shared the one unseeded state,
  which training never draws from. Fix: the test seeds every stream as `ard.cli.train` does
  (`_seed_everything`); the comparison is unchanged.
- 2026-10-03: human decision: new runs may drop deterministic mode. `training.cuda_graph` now also admits
  `deterministic: false`, optionally with `cudnn_benchmark: true` (schema and Trainer scope); every other
  restriction is unchanged, and deterministic behaviour is unchanged bitwise (the parity tests are untouched).
  Added the nondeterministic tests (exact RNG streams over whole runs, one-step equivalence against
  eager-vs-eager noise) for the fixture and all allowlisted students, with and without cuDNN benchmark; all
  pass on Hamster GPU1. Pooling identity: unchanged rule (`deterministic` and `cudnn_benchmark` stay in the
  identity; `cuda_graph` stays out). Not merged, not launched.
- 2026-10-03, scientific review of 861b288 (P1, P2, P3), all fixed in one batch: the one-step test is
  rewritten (per-group distances with their own floors, sane regime asserted, parameter-only controls
  that must fail, eager arms in separate processes, each graph outcome compared with the nearest eager
  outcome because one step's nondeterminism can be bimodal); production shapes checked once (above);
  `cuda_graph: true` is now recorded in `training_protocol_identity` when `deterministic` is false; docs
  say what is observed rather than implied. The rewritten test found that under cuDNN benchmark a captured
  step can use a different cuDNN algorithm than the eager steps of its process, so `cudnn_benchmark` is
  refused with `cuda_graph` again (schema and Trainer); only `deterministic: false` is newly admitted.
  The parity proof now runs without a `CUBLAS_WORKSPACE_CONFIG` override, as production does (no launcher
  sets it, and torch 2.11 neither errors nor warns without it in deterministic mode); the deterministic
  parity tests pass in that setting.
- 2026-10-03, re-review of 093e654, final batch: spread is the median of the eager outcomes'
  nearest-other distances and a spread above 100 x the floor fails the check (one eager arm in the other
  mode no longer widens the bound; a CPU test pins this); each graph arm has one common nearest eager
  outcome across groups; the controls run at both sync points; the production-shape check is committed as
  an opt-in test; `test_step_sync_free_parity.py` also runs without `CUBLAS_WORKSPACE_CONFIG` (passes).
  Incident: the earlier production-shape run on Hamster GPU1 (about 14 GB per process, next to a production
  job) OOM-killed that production run. Production-shape checks run on a free GPU only.
- 2026-10-08 (plan 0103 Phase 2 batch A, human-approved): scope widened by three options, nothing else
  relaxed. (1) `training.weight_ema_decay`: `_update_ema` runs inside the captured step right after the SGD
  update (the eager step's own kernels and order, decay baked in as the same Python scalar); the stale-graph
  fingerprint now also covers the EMA tensors' addresses (new guard test `ema_tensor`). (2)
  `method.label_smoothing` (it was not refused before but had no parity test). (3)
  `optimizer.exclude_norm_bias_from_weight_decay` (two SGD groups). `method.mixed_batch` and `method.awp`
  are refused (schema and Trainer). Tests on Hamster GPU1 next to the production job, all passing:
  deterministic bitwise parity for each option alone on the fixture and all three together on every
  allowlisted architecture (rows, `last.pt`/`best.pt` incl. `ema`, diagnostics, RNG; the EMA visibly
  differs from the student), resumed-graph = eager (fixture EMA; MobileNetV4-S all three),
  nondeterministic whole-run RNG exactness (fixture, MobileNetV4-S), and the one-step rule with the EMA
  state as a fourth group (fixture: bitwise; MobileNetV4-S: graph at most 0.04 of the bound; EMA group
  graph 2.5e-8 vs eager spread 1.9e-8 and floor 8.7e-6; controls 15x-2e6 x the bound). Peak reserved
  memory per test process under 0.2 GiB. Pooling identity unchanged (`weight_ema_decay` is already in
  `training_protocol_identity`; `cuda_graph` stays out of it in deterministic mode).
