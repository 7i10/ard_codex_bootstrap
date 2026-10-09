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
  Since 2026-10-08 (human decision: speedups that save >= 1 GPU-hour per run) the scope also covers, under the
  same bitwise / one-step tests and nothing else relaxed:
  (1) ImageNet `rslad` and `rslad_advt` with a `distillation` block (KL-to-teacher-clean training attack, RSLAD
  baseline policy, distillation hooks). Soft-label bank target: the host looks up the batch's stored rows
  (`SoftLabelBankTeacher.clean_rows`: crop-key and pixel-sentinel checks, no device sync) and copies them into
  three static buffers like images and labels; the captured step reconstructs the target with the bank's own
  kernels (`reconstruct_probabilities`, `pseudo_logits`). The advT teacher forward on x' and an online
  (`target_source: online_teacher`) clean teacher forward run INSIDE the captured step; that teacher must be
  frozen (no trainable parameter), in eval mode (every submodule) and on `CUDA_GRAPH_TEACHER_ARCHITECTURES`
  (the five Phase 2 ImageNet teachers, each parity-tested); plain bank RSLAD never runs the teacher.
  (2) `method.mixed_batch` without split BN: `k` is fixed by the full batch size, the noise buffer holds the
  `k` attacked rows, the mixed-batch accumulator is a static buffer; the last partial batch (own `k`) stays
  eager. Split BN stays refused.
  (3) `method.awp`: proxy `load_state_dict`, proxy forward/backward and SGD step, rescaled difference, perturb
  and (after the SGD step) restore are all captured; the AWP activity is per epoch (re-captured every epoch).
  The host checks the shared code makes on values (bank reconstruction, advT target, KL-target validation,
  policy weights, teacher entropy) go through `ard.device_checks.require`: the unchanged host check outside a
  capture, a `torch._assert_async` inside one (fires from every replay; fatal before any checkpoint).
  The stale-graph fingerprint also covers: every static buffer's address (images, labels, mask, noise, bank
  rows, accumulators), the advT metric accumulator, the in-step teacher (parameter / buffer addresses, every
  submodule's mode), the AWP proxy (addresses, modes) and its SGD (hyperparameters, parameters), and the baked-in
  Python scalars (mixed `k`, fraction, lambda; AWP gamma and activity).
  Only `ard.cli.train` applies the option; `reject_throughput_options` refuses it everywhere else.
- **AdamW and the ConvNeXt-Atto / DeiT-Tiny family (2026-10-09, human-approved).** `optimizer.id: adamw` is
  admitted (every method above). torch's AdamW is captured through its own capturable foreach implementation
  (device step counter, bias corrections computed on the device); `ard.cli.train` builds AdamW with
  `capturable=True` exactly when `training.cuda_graph` is on, so the run's eager steps (first full batch, partial
  batch) and its replays use the same arithmetic. Without the flag the AdamW call is unchanged. Capturable and
  default (`capturable=False`: host float64 bias corrections) AdamW are NOT bitwise equal (tested), so for AdamW
  `cuda_graph: true` is recorded in `training_protocol_identity` in both determinism classes (SGD keeps its rule);
  no AdamW graph run existed before, so no recorded identity changes. The Trainer admits only
  `torch.optim.AdamW` with `capturable=True`, foreach (not fused), amsgrad / maximize / differentiable off; capture
  waits until every parameter has its AdamW state (`optimizer_state_ready`). The stale-graph fingerprint covers
  every per-parameter optimizer state tensor by address (SGD momentum; AdamW `exp_avg`, `exp_avg_sq`, `step`) and
  every group hyperparameter (lr, betas, eps, weight decay, flags; a tensor beta is refused like a tensor lr). The
  allowlist adds `convnext_atto_imagenet`, `convnext_atto_deep_narrow_imagenet`, `convnext_atto_ols_imagenet`,
  `convnext_atto_convstem_imagenet`, `deit_tiny_imagenet`, `deit_tiny_convstem_imagenet` (LayerNorm, GELU,
  depthwise 7x7, SDPA attention), after their parity tests passed with SGD and AdamW. `cudnn_benchmark` stays
  refused with the graph, so the Phase 1/2 ConvNeXt / DeiT configs (nondeterministic + benchmark today) need
  `cudnn_benchmark: false` to use it; no config is edited here.
  Consequences (review of 1f17fa2): an AdamW graph run cannot be reproduced bitwise by rerunning it eagerly with
  `cuda_graph: false` (that builds the default AdamW); capturable and default AdamW are the same algorithm up to
  rounding (`test_capturable_adamw_step_equals_the_default_step_up_to_rounding`: from one exact state and
  gradient, exp_avg / exp_avg_sq / step bitwise equal, every parameter within 4 x (8 + kappa1 + kappa2 / 2) eps32
  of the update plus one ulp of the new value, kappa = beta^t / (1 - beta^t); measured at most 0.08 (t = 2) and
  0.44 (t = 21) of the update term). The run's eager steps emit torch's one-time UserWarning ("constructed with
  capturable=True ... step() is running without cuda graph capture"); expected. Until the opt-in production-shape
  bitwise SGD check (`test_production_shape_layernorm_graph_run_is_bit_identical[...-sgd]`) has passed for them,
  the six new students are in `CUDA_GRAPH_IDENTITY_RECORDED_ARCHITECTURES`: their graph runs record
  `cuda_graph: true` in `training_protocol_identity` with deterministic SGD too (their bitwise parity is shown at
  test shapes only). Remove an architecture from that set once its check passes. `optimizer_state_ready` raises
  for an optimizer other than SGD / AdamW.
- **Memory (2026-10-08).** The graph's private pool used to stay allocated until the next epoch's re-capture,
  so the eager last partial batch, the validation pass and the train probe allocated their activations next
  to it (EfficientNet-B0, 224 px, batch 128: OOM at ~22.6 GB on a 24 GB 4090). The Trainer now releases the
  graph (and the parameter gradients the last replay left in its pool) and empties the CUDA cache before any
  eager step that follows a capture and at the end of every training epoch (`Trainer._release_cuda_graph`).
  `torch.cuda.graph` already empties the cache before a capture. Nothing that reaches a checkpoint, a row or
  an RNG stream changes (the parity tests are unchanged and pass).
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
- 2026-10-08 scope widening, same file, Hamster GPU1 next to the production job (every test process <= 0.5 GB
  peak reserved), all passing (deterministic unless stated):
  bitwise eager = graph (rows, `last.pt`/`best.pt` state, diagnostics rows and panel media incl. kd weights and
  advT teacher predictions/entropies, CUDA RNG) for RSLAD from a top-K bank (K < C, so the residual-mass
  reconstruction acts), RSLAD from an online teacher, RSLAD-advT from a bank (teacher forward and top-K truncation
  of T(x') inside the step), mixed batch (k = 2 of 4; partial batch k = 1 eager) and AWP (inactive in epoch 0,
  active after; dropout in the student and the AWP proxy draws the CUDA RNG inside the graph) on the fixture and
  every allowlisted student; each method with the Phase 2 weight EMA (fixture, MobileNetV4-S); the in-step teacher
  forward of every `CUDA_GRAPH_TEACHER_ARCHITECTURES` entry (random weights, 1000 classes; RSLAD online and advT;
  ViT-S at 224 px); graph run resumed at epoch 1 = eager run (fixture, MobileNetV4-S, every method); negative
  controls confined to the graph path (bank rows of the neighbouring sample, a 1% shifted online-teacher input,
  k + 1 attacked positions, AWP off inside the graph) all change the run; stale-graph refusals for a reallocated
  bank buffer, a non-in-place teacher weight, a train-mode teacher, a reallocated AWP proxy buffer, a changed AWP
  gamma and a changed mixed-batch lambda; a NaN bank row fires the in-graph reconstruction assert from a replay.
  Nondeterministic: whole-run RNG exactness for every new method (fixture, MobileNetV4-S) and the one-step rule on
  MobileNetV4-S for RSLAD bank / online / advT (lr 0.006: at lr 0.002 the KL objective's smaller update left the
  lr x 1.001 control only 8-9x the bound) and mixed batch: graph arms at 0.03-0.04 of the bound, controls >= 14x.
  AWP failed the nondeterministic rule (not the graph: the EAGER outcomes of one AWP step were 0.012-0.014 of a
  step apart at one sync point in one of two runs at gamma 0.01, and 0.04 at gamma 0.05; above 100x the FP32 floor,
  so the check cannot resolve a defect), so AWP + graph is admitted only with `deterministic: true` (schema and
  Trainer; tested). Earlier tests of this file all still pass with the pool release.
- Memory (2026-10-08, `scripts/cuda_graph_benchmark.py`, process-wide peak over 2 epochs incl. the last partial
  batch and validation, EfficientNet-B0, 224 px, PGD-3, deterministic, GPU1 next to production):

  | batch | eager | graph, pool kept (old) | graph, pool released (new) |
  |---|---|---|---|
  | 8 | 0.89 GiB | 1.84 | 1.04 |
  | 16 | 1.76 | 3.71 | 2.08 |
  | 24 | 2.63 | 5.53 | 3.09 |

  Linear in the batch: the old behaviour costs 0.231 GiB per example (batch 128 ~ 29.5 GiB: the OOM), the new
  one 0.128 GiB (batch 128 ~ 16.4 GiB, ~7 GiB headroom on a 24 GB 4090; eager ~ 13.9 GiB). Allocated (live) memory
  is the same in both graph modes: the old peak was the pool's freed-but-reserved blocks next to the eager
  partial batch and validation. Batch 128 must be measured on a FREE GPU:

      CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=src python scripts/cuda_graph_benchmark.py \
        --architecture efficientnet_b0_imagenet --image-size 224 --batch-size 128 --batches 4 --partial 127 \
        --epochs 2 --mode graph            # then --mode eager, and --mode graph --keep-graph-pool (old)

- Throughput (2026-10-08, same script, MobileNetV4-S, 112 px, batch 64, PGD-3, deterministic, 40 batches, second
  epoch's img/s; GPU1 SHARED with the production job, so only ratios are indicative): PGD-AT 774 -> 1558 (2.0x),
  RSLAD bank 530 -> 1145 (2.2x), RSLAD-advT bank with a ResNet-50 teacher 409 -> 868 (2.1x), mixed batch 772 ->
  1766 (2.3x), AWP 547 -> 1110 (2.0x).
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

- 2026-10-09 AdamW / ConvNeXt-Atto / DeiT-Tiny, same file, Hamster GPU1 next to the production job (every test
  process <= 0.65 GB on nvidia-smi, peak reserved <= 0.53 GiB), all passing:
  deterministic bitwise eager (capturable AdamW) = graph -- rows, `last.pt`/`best.pt` (model, optimizer incl.
  exp_avg / exp_avg_sq / step, scheduler, EMA, RNG), diagnostics, CUDA RNG -- for AdamW on the fixture, AdamW + EMA +
  label smoothing on every allowlisted student (MobileNetV4-S/M, EfficientNet-B0 and the six new ones; ConvNeXt at
  64 px, DeiT at 224 px), AdamW + EMA with RSLAD bank / online / advT, mixed batch and AWP on the fixture,
  ConvNeXt-Atto and DeiT-Tiny; graph run resumed at epoch 1 = eager run (fixture, ConvNeXt-Atto, DeiT-Tiny); a
  graph-only defect (every AdamW step counter advanced once more inside the step) changes model, optimizer and
  rows; stale-graph refusals for lr (group 1), beta1, beta2, eps, the norm/bias group's weight decay, a reallocated
  exp_avg / exp_avg_sq / step tensor and an optimizer `load_state_dict`; eager capturable=False vs capturable=True
  differ (models, optimizer, rows). The SGD cases of the file (parity, Phase 2 options, all five methods, the
  in-step teachers) also ran for the six new students and passed. Nondeterministic: whole-run RNG exactness with
  AdamW + EMA + label smoothing (fixture and the six new students) and with AdamW + EMA for RSLAD bank / online /
  advT and mixed batch (ConvNeXt-Atto, DeiT-Tiny), SGD (the six new students); the one-step rule with AdamW
  (exp_avg and exp_avg_sq are tensor groups, step counters must be exact, a group the step leaves unchanged --
  e.g. pixel-normalization constants of a model without BatchNorm -- must be bitwise equal; fixture and the six
  new students, lr 6e-4 (fixture 3e-3); RSLAD bank / online / advT and mixed batch on ConvNeXt-Atto) and with SGD
  (the six new students, lr 0.02, convstem 0.04: at lr 0.002 the lr x 1.001 control sat only 1.1-2.0x the bound).
  Measured (AdamW one-step, 64 px ConvNeXt-Atto / 224 px DeiT-Tiny, batch 32 / 8): update 1.0-1.3% of the weights;
  graph arms at 0.05-0.09 of the bound (ConvNeXt) and bitwise equal to the nearest eager outcome (DeiT); controls
  lr x 1.001 16-26x, weight decay 0 35-37x, attack seed + 1 >= 78x, lr = 0 >= 15,000x the bound.
  Opt-in production shape (not run: needs a FREE GPU, about 6-8 GB per process), commands in the test file:
  `test_production_shape_layernorm_graph_run_is_bit_identical` (each of the six students x AdamW (lr 1e-3) / SGD
  (lr 0.025, the Phase 1 recipe), 224 px, batch 128, 1000 classes, PGD-3, EMA 0.9999, 4 full batches + a partial
  batch, deterministic, bitwise) and `test_production_shape_layernorm_graph_step_matches_eager_within_one_step_noise`
  (nondeterministic one-step rule at 224 px / batch 128 / 1000 classes / PGD-3, random init: AdamW lr 6e-4 + EMA
  for the six, SGD lr 0.02 for ConvNeXt-Atto and DeiT-Tiny). The free-GPU window runs all of it with the speed
  benchmark in one command:

      cd <checkout at this commit> && GPU=<free gpu> LOG=cuda_graph_free_gpu_$(date +%Y%m%dT%H%M).log && {
        for arch in convnext_atto_imagenet deit_tiny_imagenet; do
          for arm in "eager --nondeterministic --cudnn-benchmark" "eager --nondeterministic" \
                     "graph --nondeterministic" "eager" "graph"; do
            set -- $arm; mode=$1; shift
            CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python scripts/cuda_graph_benchmark.py --architecture $arch \
              --optimizer adamw --image-size 224 --batch-size 128 --batches 40 --epochs 2 --validation-batches 1 \
              --mode $mode "$@"
          done
        done
        ARD_CUDA_GRAPH_PRODUCTION_SHAPE_CHECK=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python -m pytest -s -rA \
          tests/integration/test_cuda_graph_training_step.py -k "production_shape_layernorm"
      } 2>&1 | tee "$LOG"
- Throughput potential, AdamW ConvNeXt-Atto / DeiT-Tiny (2026-10-09, `scripts/cuda_graph_benchmark.py --optimizer
  adamw`, real `Trainer.fit`, 224 px, PGD-3, panel diagnostics, 40 batches, second epoch). NO GPU WAS FREE (Hamster
  GPU0/1 and all three Ferret GPUs busy), so it ran on Hamster GPU1 SHARED with a production job at batch 2 / 4 / 6
  only (<= 0.5 GiB reserved; batch 8 already needed 0.53 and cuDNN benchmark's trial workspaces 1.08 GiB at batch
  4, so the benchmark baseline could not be measured). ms per step, mean of 2 interleaved repeats:

  | student, mode | b2 eager / graph | b4 eager / graph | b6 eager / graph |
  |---|---|---|---|
  | ConvNeXt-Atto, nondeterministic | 70.5 / 17.5 | 68.7 / 22.3 | 70.0 / 25.9 |
  | ConvNeXt-Atto, deterministic | 74.3 / 20.6 | 73.5 / 25.3 | 72.8 / 29.1 |
  | DeiT-Tiny, nondeterministic | 84.7 / 34.7 | 85.2 / 39.4 | 85.3 / 43.3 |
  | DeiT-Tiny, deterministic | 92.6 / 55.6 | 93.3 / 61.0 | 93.9 / 64.7 |
  | MobileNetV4-S (SGD, anchor), nondeterministic | 76.5 / 16.4 | 77.0 / 17.3 | 74.2 / 18.0 |
  | MobileNetV4-S (SGD, anchor), deterministic | 87.7 / 18.7 | 87.5 / 21.1 | 84.3 / 22.5 |

  The eager step is flat in the batch size: its host time (launching about 70 ms of kernels for ConvNeXt-Atto,
  85-93 ms for DeiT-Tiny, 75-87 ms for MobileNetV4-S) bounds it at these sizes, so these ratios are NOT the
  production speedup. At batch 128 the eager steps take about 147 ms (ConvNeXt-Atto) and 203 ms (DeiT-Tiny) on an
  idle 4090 (plan 0103, 2026-09-27 probe, nondeterministic + cuDNN benchmark), i.e. longer than their host time, so
  only part of the host time is exposed there. Anchor: MobileNetV4-S at 224 px / batch 128 / PGD-3 has a similar
  host time and its graph saved about 30 ms per step in production (eager 919-943 img/s, graph 1,186-1,207;
  different sources, not controlled). Scaling that saving by host time gives an estimate of roughly 25-30 ms per
  step for ConvNeXt-Atto (about 15-20% faster) and DeiT-Tiny (about 15%), but DeiT's uniform, large kernels may
  hide more of the host time, deterministic DeiT runs the slow attention path (graph 55-65 ms vs 35-43 ms
  nondeterministic here), and losing cuDNN benchmark costs an unmeasured amount. Extrapolated, not measured: the
  decision needs the free-GPU command in `scripts/cuda_graph_benchmark.py` (all five arms, batch 128).

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
- 2026-10-08 (human decision: speedups worth >= 1 GPU-hour per run): scope widened to ImageNet RSLAD /
  RSLAD-advT (bank or online target; in-step teachers allowlisted and parity-tested), mixed batch without split
  BN, and AWP (deterministic mode only, see Verification); value checks of the shared code become device asserts
  under capture (`ard.device_checks.require`), fingerprint extended (static buffers, teacher, AWP proxy and its
  SGD, baked-in scalars); the graph pool is released before eager steps and at every epoch end (EfficientNet-B0
  224 px / batch 128 estimated ~16.4 GiB instead of ~29.5 GiB). Existing configs unchanged (the flag stays off
  everywhere; no config edited). Pooling identity unchanged. Not merged, not launched; enabling the flag in the
  Phase 2 configs (and the batch-128 memory measurement on a free GPU) is the human's decision.
- 2026-10-08, scientific review of d2e82b2 (no P0/P1), one batch: P2-1 the admitted scope is narrowed to
  the proven one: `rslad_advt` only from a soft-label bank (online advT refused) and RSLAD / advT only at
  temperature 1 (method, attack and hooks; schema and Trainer, refusal tests). P3-1 opt-in production-shape
  BITWISE check for distillation (`test_production_shape_distillation_graph_run_is_bit_identical`: MNv4-S,
  224 px, batch 128, 1000 classes, PGD-3; online RSLAD with ConvNeXt-B-cvst and ViT-S-cvst, bank RSLAD, bank
  advT with ResNet-50; real teacher checkpoints from `ARD_EXTERNAL_CHECKPOINT_ROOT` when present, registry
  normalization); not run (needs a free GPU, command in the test file); its harness was smoke-checked bitwise
  at batch 8 on GPU1. P3-2 end-to-end: a NaN bank row in a replayed batch, with the pool release active,
  kills the process before `last.pt` (tested). P3-3 `tests/unit/test_device_checks_equivalence.py`: 23 good /
  bad inputs through the converted checks give the pre-change (git d5bae49) accepted values, exception types
  and messages.
- 2026-10-09 (human-approved: AdamW cuda_graph for the ConvNeXt-Atto / DeiT-Tiny runs if it saves >= ~1 h per
  ~30 h run): measured first (no free GPU: shared GPU1 at batch 2-6, extrapolated, see Verification; the
  production-shape number is still owed), then implemented AdamW (capturable, built only with the flag), the
  six-architecture allowlist extension after the tests passed, the AdamW fingerprint / readiness, the AdamW
  identity rule, and the tests above; `scripts/cuda_graph_benchmark.py` gained `--optimizer adamw` and
  `--cudnn-benchmark`. Existing configs resolve and hash byte-identically (the git-pinned test) and their
  optimizer is built exactly as before. No config edited; not merged, not launched; switching the Phase 2
  ConvNeXt / DeiT configs to `cudnn_benchmark: false` + `cuda_graph: true` is the human's decision.
