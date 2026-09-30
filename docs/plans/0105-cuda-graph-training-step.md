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
  the pre-change file from git).
- **Config.** `training.cuda_graph: bool = False`, serialized only when true, so every existing config
  keeps a byte-identical resolved config and hash (tested against the pre-change schema). It is refused
  unless: `device: cuda`, deterministic, no AMP, no compile, no cuDNN benchmark,
  `step_diagnostics: false`, world size 1 (`global_batch_size == per_rank_batch_size`), no epsilon
  warmup, no weight EMA, method `pgd_at`, SGD, no teacher / ADR / target policy / intervention /
  prescriptive route / observation profile, and a CE, batch-keyed, untraced attack with a fixed budget
  (0 < epsilon, step <= epsilon). The Trainer repeats the same checks for direct construction.
  Only `ard.cli.train` applies the option; `reject_throughput_options` refuses it everywhere else.
- **Capture lifecycle.** Static device buffers for images, labels, the valid mask and the random-start
  noise. The loader batch is copied into them (non-blocking from pinned memory). The first full batch
  of every epoch runs eagerly (it creates SGD momentum buffers on a fresh run and warms up lazily
  initialized CUDA libraries); the next batch captures the graph and every later full batch replays it.
  The last partial batch runs eagerly. Eager and replayed steps add into one static epoch accumulator,
  in step order, read once at epoch end.
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
  `torch._assert_async` (no host sync). The in-graph guards (the model adapter's pixel check and the
  finite-loss check) are device asserts that do fire from inside a replay (tested). A fired assert kills
  the CUDA context, so the poisoned update never reaches a checkpoint (tested). The host-side eager
  guards are unchanged.
- **Diagnostics.** Production configs require panel diagnostics, so they are supported: the eval-mode
  clean forward runs inside the graph at the same point as in the eager step, and its predictions are
  read back in one transfer per epoch (and before any eager step, to keep row order).
- **Pooling identity.** `training.cuda_graph` is in the config hash but not in
  `training_protocol_identity`, like `step_diagnostics`. Reason: the parity test proves bitwise equality
  end to end (checkpoints, RNG, rows, diagnostics; LR milestone, partial batch, resume), and the
  production-shape check below matched too. Seeds of one arm that differ only in this flag may pool.

## Verification

- `tests/integration/test_cuda_graph_training_step.py` (GPU, subprocess, deterministic): real `Trainer.fit`,
  3 epochs, LR milestones after epochs 1 and 2, partial last batch, BatchNorm + dropout student and
  MobileNetV4-Conv-Small at 32 px. Eager vs graph: identical epoch rows, `last.pt`/`best.pt` state (model,
  optimizer, scheduler, all RNG streams, sampler, selection), diagnostics rows and panel media, CUDA RNG
  state. A graph run resumed from the epoch-1 checkpoint matches the uninterrupted eager run. Guard tests:
  a mid-epoch change of `lr`, `momentum`, `weight_decay`, optimizer state tensors or model mode is refused;
  in-graph pixel and finite-loss asserts fire from a replay; an out-of-range pixel on the graph path aborts
  before any checkpoint.
- `tests/unit/test_cuda_graph_config.py`: default off and unserialized, byte-identical configs, every
  fail-closed combination, `reject_throughput_options`, Trainer scope, fingerprint sensitivity,
  `generate` bit-identical to the pre-change attack, and `perturb` makes no host sync
  (`torch.cuda.set_sync_debug_mode("error")`).
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
