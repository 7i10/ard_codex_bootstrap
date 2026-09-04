---
paths: ["src/ard/**", "configs/**"]
---
# Scientific core invariants
Short checklist; `docs/SCIENTIFIC_INVARIANTS.md` wins on conflict.
## Attack identity
- Identity = all 14 `AttackConfig` fields: `norm`, `input_domain`, `epsilon`, `epsilon_value`, `step_size`,
  `step_size_value`, `steps`, `random_start`, `loss`, `kl_target`, `temperature`, `temperature_squared`,
  `student_mode`, `teacher_mode`. Exact equality of the complete mapping; threat hash = SHA-256 of its
  canonical JSON. Never compare budget only, never omit a field. `trace_step_losses` is debug-only (default
  false) and not one of the 14.
- Images reach attacks as float **pixel space [0,1]**; CIFAR normalization belongs to the model adapter and is
  never applied twice. Rationals stay strings (`8/255`, `2/255`) in the resolved config, checked against their
  value. Canonical budget: linf, eps 8/255, step 2/255, 10 steps, random start. PGD projects onto the Linf ball
  **in pixel space, then clamps to [0,1]** — projection before clamp.

## Gradients, modes, objective
- Requests state student/teacher mode explicitly and restore it on exit; selection/evaluation attacks keep both
  in eval, and the frozen teacher stays eval (nested BN included) even if a caller asks for `train()`.
- Teacher params `requires_grad=False`, no leftover `.grad`; parameter freeze and teacher *input* gradients are
  different properties, tested separately.
- KL flows teacher/target → student; temperature applies to both logit sets, `temperature_squared` multiplies by
  `T^2`. Policy weights multiply per-sample terms **before** reduction; DDP padding is masked out of loss,
  signals and state.
- Observation tensors are pre-update detached FP32 and never reach attack, loss, policy, gradient, optimizer,
  scheduler or RNG. Non-finite loss/weight fails; never clamp or widen tolerance to hide it.

## Sample state, checkpoints, resume
- Sample ID = original dataset index; never rebuilt from subset, augmentation, shuffle or rank. Checkpoints
  carry all 18 required keys: model, optimizer, scheduler, scaler, Python/NumPy/PyTorch/CUDA RNG,
  sampler epoch, sample state, global step, config hash, tracking run ID, best-selection state; `best.pt` and
  `last.pt` are separate files/artifacts. Resume is epoch-boundary only and refuses drift in config hash, run
  ID, output dir or world size. Per-rank batch 128 (1 GPU) and 64 (2 GPU) are distinct execution identities.

## Evaluation integrity
- Selection is hard-label CE. The evaluation attack defaults to the selection attack resolved in the saved
  training config and requires exact equality of every identity field, `loss` included.
- Clean / PGD / AutoAttack reported separately; **best AND last** evaluated under the same threat hash. Full
  AutoAttack only from a saved checkpoint, in a separate process, `autoattack=true` plus `--allow-autoattack`.
  Training-time validation PGD is never an official test result.

## Changing any of this
Epsilon, steps, step size, random start, normalization, temperature, schedule, checkpoint selection, evaluation
attack or a tolerance change only with an explicit config flag / new method identity plus a regression test that
fails before and passes after. Never weaken an attack, tolerance or guard to make something pass.
