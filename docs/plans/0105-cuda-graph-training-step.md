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
It must give bit-identical checkpoints, RNG streams, epoch rows and diagnostics to the eager trainer.
It must refuse every configuration where that is not proven.

## Design

- **Attack core.** `LinfPGD.generate` is split into a sync-free core, `LinfPGD.perturb` (inputs, labels,
  epsilon/step tensors, raw random-start noise -> adversarial batch), and a validating wrapper
  (`generate`: pixel check, budget checks, random-start draw, `max_abs_delta`). The eager path and the
  graph path call the same core. `generate` is bit-identical to the pre-change version (tested against
  golden output digests produced by the pre-change file, `tests/unit/fixtures/pgd_generate_golden_4fd5ceb.json`).
- **Config.** `training.cuda_graph: bool = False`, serialized only when true, so every existing config
  keeps a byte-identical resolved config and hash (tested against the pre-change schema). It is refused
  unless: `device: cuda`, deterministic, no AMP, no compile, no cuDNN benchmark,
  `step_diagnostics: false`, world size 1 (`global_batch_size == per_rank_batch_size`), no epsilon
  warmup, no weight EMA, method `pgd_at`, SGD, no teacher / ADR / target policy / intervention /
  prescriptive route / observation profile, and a CE, eval-mode, batch-keyed, untraced attack with a
  fixed budget (0 < epsilon, step <= epsilon). The student architecture must be on the parity-tested
  allowlist `CUDA_GRAPH_ARCHITECTURES` (MobileNetV4-Conv-Small, MobileNetV4-Conv-Medium, EfficientNet-B0).
  The Trainer repeats the same checks for direct construction (it also refuses a `LinfPGD` subclass,
  deterministic `warn_only` mode and cuDNN benchmark).
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
  the flag to what the parity test proves bitwise equal end to end (checkpoints, RNG, rows, diagnostics;
  LR milestones, partial batch, probe pass, resume): the allowlisted architectures, an eval-mode training
  attack and deterministic mode, measured on torch 2.11 with an RTX 4090. The production-shape check
  below matched too. Within that scope, seeds of one arm that differ only in this flag may pool. Widening
  the scope, or upgrading torch or the driver, requires rerunning the parity test first.

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
- `tests/unit/test_cuda_graph_config.py` (CPU): default off and unserialized; every config that existed at
  4fd5ceb (read from git) resolves and hashes byte-identically; every fail-closed combination (including an
  off-list architecture and a train-mode attack); `reject_throughput_options`; Trainer scope (architecture,
  train-mode attack, `LinfPGD` subclass, non-strict determinism); static-buffer admission (contiguity,
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
