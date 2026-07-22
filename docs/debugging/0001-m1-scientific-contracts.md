# M1 scientific-contract corrections

## Scope

M1 initially used training-batch robust accuracy for checkpoint selection, wrote
the resolved config before checking collisions or resume eligibility, allowed a
frozen teacher adapter to enter train mode, and treated DDP padding as ordinary
samples.  These behaviors could respectively bias selection, overwrite run
lineage, mutate teacher BatchNorm statistics, and double count/update samples.

## Corrections

- The training CLI now accepts only the official train split and creates a
  deterministic, seed-fixed stratified validation subset from it.  The official
  test split remains evaluation-only.  `best.pt` is selected by post-update
  validation PGD accuracy; equal metrics retain the earliest epoch.  The
  selection attack and rule are checkpoint metadata.
- Repro/production are unconditionally rejected while M4 tracking lineage is
  unavailable.  There is no user-overridable M1 availability flag.  Collision
  and checkpoint validation occur before any output write; a non-resume
  invocation cannot reuse a populated output directory.
- `TeacherAdapter` is permanently frozen/eval, including nested BatchNorm.  It
  still permits gradients with respect to input pixels.
- DDP padding now repeats enough even when dataset size is smaller than world
  size, retains original sample IDs, has a false state-update mask for every
  padded position, and records multiplicity in linear time.  M1 loss and
  accuracy reductions exclude padding.
- The CLI supports torchrun CPU/Gloo or CUDA/NCCL, unwrapped checkpoint state
  dicts, and coordinated output phases.  Only rank zero checks output/resume
  state or writes; each phase broadcasts success or failure before peers can
  enter a barrier, avoiding fresh-output races and error-path hangs.
- Checkpoint writes use the same rank-zero outcome broadcast after every rank
  gathers RNG and sampler state.  Rank-zero serialization or replacement
  errors are therefore reported by every rank instead of leaving peers at a
  barrier.
- Checkpoint selection owns a distinct, same-strength CE PGD configuration
  whose student and teacher modes are fixed to eval.  One deterministic,
  rank-separated generator is advanced across the complete validation pass,
  preventing BatchNorm mutation and repeated random-start templates.
- Normalization uses named finite profiles with provenance.  Every real dataset
  requires its documented profile in dev/smoke as well, and M1 requires teacher
  and student profiles to match.  Synthetic fixtures use identity normalization.
  The CIFAR-100 profile is repository policy and is explicitly not claimed
  upstream-exact.

## Deferred

M4 owns tracker implementation and saved-checkpoint evaluation (including
AutoAttack).  M1 therefore fails closed for repro/production instead of
creating incomplete lineage.

## Verification environment note

The real delayed-rank two-process CPU/Gloo regression is retained.  Inside the
managed filesystem sandbox, however, even a launcher-only `/bin/true` torchrun
reproducer fails while creating the localhost TCPStore with `EPERM`, before any
ARD module or collective runs.  The same launcher exits successfully outside
that sandbox.  An abrupt in-sandbox pytest termination on this node is therefore
an environment restriction, not evidence of an ARD distributed failure.  Run
the exact Gloo test with narrowly granted localhost-socket permission; do not
weaken its assertions or extend its timeout.
