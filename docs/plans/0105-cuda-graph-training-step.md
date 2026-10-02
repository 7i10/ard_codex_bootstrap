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
keep every RNG stream exactly and differ from eager only within nondeterministic noise.
It must refuse every configuration where that is not proven.

## Design

- **Attack core.** `LinfPGD.generate` is split into a sync-free core, `LinfPGD.perturb` (inputs, labels,
  epsilon/step tensors, raw random-start noise -> adversarial batch), and a validating wrapper
  (`generate`: pixel check, budget checks, random-start draw, `max_abs_delta`). The eager path and the
  graph path call the same core. `generate` is bit-identical to the pre-change version (tested against
  golden output digests produced by the pre-change file, `tests/unit/fixtures/pgd_generate_golden_4fd5ceb.json`).
- **Config.** `training.cuda_graph: bool = False`, serialized only when true, so every existing config
  keeps a byte-identical resolved config and hash (tested against the pre-change schema). It is refused
  unless: `device: cuda`, no AMP, no compile, `step_diagnostics: false`, world size 1 (`global_batch_size == per_rank_batch_size`), no epsilon
  warmup, no weight EMA, method `pgd_at`, SGD, no teacher / ADR / target policy / intervention /
  prescriptive route / observation profile, and a CE, eval-mode, batch-keyed, untraced attack with a
  fixed budget (0 < epsilon, step <= epsilon). The student architecture must be on the parity-tested
  allowlist `CUDA_GRAPH_ARCHITECTURES` (MobileNetV4-Conv-Small, MobileNetV4-Conv-Medium, EfficientNet-B0).
  Both `deterministic: true` and (since 2026-10-03) `deterministic: false`, optionally with
  `cudnn_benchmark: true`, are admitted. The Trainer repeats the same checks for direct construction (it also
  refuses a `LinfPGD` subclass, deterministic `warn_only` mode, and cuDNN benchmark with deterministic
  algorithms on).
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
- **Pooling identity.** `training.cuda_graph` is in the config hash but not in
  `training_protocol_identity`, like `step_diagnostics`. This holds only because the scope checks confine
  the flag to what the tests prove equal to the eager step of the same determinism class: the allowlisted
  architectures and an eval-mode training attack, measured on torch 2.11 with an RTX 4090.
  - Deterministic mode: bitwise equal end to end (checkpoints, RNG, rows, diagnostics; LR milestones,
    partial batch, probe pass, resume). The production-shape check below matched too.
  - Nondeterministic mode (`deterministic: false`, with or without `cudnn_benchmark`): every RNG stream,
    the global step, scheduler and sampler state are exactly equal over whole runs, and one step from one
    exact state is within eager-vs-eager noise (see Verification). Not bitwise: two eager nondeterministic
    runs are not bitwise equal either.
  `training.deterministic` is already in `training_protocol_identity`, and `cudnn_benchmark` is there when
  true, so this exclusion never pools across determinism classes: a deterministic run pools only with
  deterministic runs, and a nondeterministic run only with nondeterministic runs of the same benchmark
  setting, whether or not they use the graph. This is the existing rule for nondeterministic eager seeds,
  which also differ run to run and pool. Widening the scope, or upgrading torch or the driver, requires
  rerunning the parity and equivalence tests first.

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
  architecture, the architectures also with cuDNN benchmark. Bitwise parity is impossible here, so:
  (1) whole runs (the 3-epoch run above, eager and graph each in their own process): Python/NumPy/torch
  CPU/CUDA RNG states in `last.pt`, the final CUDA RNG state (the fixture's dropout advances it inside the
  graph), the seed and final philox state of every per-step attack generator, global step, scheduler and
  sampler state are exactly equal, and the audit counts show the graph ran. This confirms that
  nondeterministic kernels do not touch any RNG stream. Weights are not compared over whole runs: PGD's
  sign step makes training chaotic. A 1e-9 relative state difference flips the sign of up to a quarter of
  the attack's input gradients one step later, so two eager nondeterministic runs drift apart to the size of
  a different seed within a few steps (measured).
  (2) One step from one exact state: a reference eager run, three eager replicas, two graph runs and a
  graph negative control (attack seed + 1) each take the capture step (batch 1) and a later replay
  (batch 3) from the reference's exact state (parameters, BatchNorm buffers, momentum buffers and CUDA RNG
  state copied in place, so the graph stays valid). Distance = relative L2 change of the model state,
  divided by the reference step's update size. Rule: graph <= 4 x max(eager replicas, one FP32 rounding of
  the new state), and negative control >= 100 x that bound. Measured: real architectures 1e-23 to 1e-13
  for both eager replicas and graph (FP32 rounding level 1.2e-7 to 1.8e-7), negative control 4e-3 to 1.3e-2;
  fixture: eager replicas 0 to 2.5e-8, graph 0 to 5.7e-8 (FP32 rounding 2e-6), negative control 1.2e-2.
  In the fixture the graph step can differ from eager by a fixed sub-rounding amount while eager replicas
  agree bitwise, so in nondeterministic mode the graph is not guaranteed to pick exactly the eager kernels;
  the difference is a summation-order effect below one FP32 rounding.
- `tests/unit/test_cuda_graph_config.py` (CPU): default off and unserialized; every config that existed at
  4fd5ceb (read from git) resolves and hashes byte-identically; every fail-closed combination (including an
  off-list architecture and a train-mode attack); `deterministic: false` with and without
  `cudnn_benchmark` admitted while the other restrictions hold; the Trainer's determinism scope (strict on,
  off, off with benchmark admitted; `warn_only` and benchmark with deterministic algorithms refused); `reject_throughput_options`; Trainer scope (architecture,
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

  Under this contention, dropping deterministic mode adds about 15-20% to the graph step at 112 px and
  about 20-25% at 224 px; cuDNN benchmark adds a little more (about 23% and 40-50% over the deterministic
  graph). A clean measurement needs a free GPU.

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
