# Debug report: CUDA RNG restore ignored the visible-device count

- Date: 2026-09-05
- Git SHA: ba767a7 (fix committed on top)
- Config / seed: n/a (unit fixture `tests/unit/test_schedule_control_fork.py`)
- Hardware: Hamster, 2x RTX 4090, torch 2.11.0+cu128

## Failure signature

`tests/unit/test_schedule_control_fork.py::test_epoch79_schedule_control_preserves_state_replaces_only_future_milestones_and_resumes`
raised `RuntimeError: RNG state is wrong size` from `restore_rng_state` when CUDA devices were visible; it passed with
`CUDA_VISIBLE_DEVICES=''`.

## Root cause

1. The fixture injected a 1-byte placeholder as the CUDA RNG state (a real state is 16 bytes) because the fork validators
   reject `torch_cuda is None` on CPU-only hosts. On a CUDA host the checkpoint is actually resumed, so torch rejected it.
2. Independently, `restore_rng_state` called `torch.cuda.set_rng_state_all` without comparing the number of saved states to
   `torch.cuda.device_count()`. `validate_resume_checkpoint` checks `world_size`, not the device count, so a checkpoint
   written unpinned (2 states) and resumed under `CUDA_VISIBLE_DEVICES=<one gpu>` failed with a bare `IndexError`.

## Fix

- `src/ard/engine/checkpoint.py`: restore saved CUDA states per ordinal; fail closed with an explicit message when more
  states are saved than devices are visible. Equal counts behave exactly as before.
- Fixture uses real `torch.cuda.get_rng_state_all()` on CUDA hosts (same pattern as `tests/integration/test_checkpoint_resume.py`).

## Regression test

`tests/unit/test_checkpoint_rng_restore.py` (partial restore, fail-closed, matching count, real two-GPU check).

## Scientific impact

No completed run resumed across a changed visible-device set with a wrong stream: the mismatched case raised instead of
resuming. No result is invalidated.
