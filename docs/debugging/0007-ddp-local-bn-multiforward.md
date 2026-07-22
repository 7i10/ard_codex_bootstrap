# DDP local-BatchNorm multi-forward failure

## Symptom

The first two-GPU Chen RSLAD pilot reached training, then failed on its first
backward pass with an autograd version error for a 512-element CUDA tensor.
No checkpoint was produced; the failed run bundle is retained as evidence.

## Root cause

RSLAD intentionally evaluates the student on adversarial and clean inputs
before one backward pass.  PyTorch DDP defaults to `broadcast_buffers=True`,
so its pre-forward hook modifies BatchNorm running buffers before the second
forward.  Autograd had saved the same buffer during the first forward and
correctly rejected the in-place version change.  A single-process reproduction
did not fail, which localized the issue to DDP's buffer broadcast rather than
the RSLAD objective or attack.

The declared execution protocol is `batchnorm_mode: local_per_rank`: training
uses each rank's mini-batch statistics rather than SyncBatchNorm. DDP's normal
buffer synchronization at the start of the first forward remains part of the
existing execution behavior.

## Fix and regression contract

DDP buffer broadcast is now suppressed only for additional student forwards
performed while the adversarial graph is retained, and restored before
backward. The first forward and the next iteration keep PyTorch's default
buffer synchronization. This does not introduce SyncBatchNorm or change the
global/per-rank batch sizes, attack, objective, or validation behavior.

Scientific review found that a no-grad diagnostic DDP forward also clears
DDP's `require_forward_param_sync` flag. The context therefore restores both
that flag and `broadcast_buffers`; otherwise the post-backward synchronization
and subsequent validation could depend on whether diagnostics were enabled.

The two-rank Gloo regression performs two BatchNorm-bearing DDP forwards before
one backward, checks that the broadcast flag is restored, checks finite
parameter updates, and checks that the normal post-backward buffer sync resumes.
The real Ferret pilot must still verify the CUDA/NCCL path before this incident
is closed.
