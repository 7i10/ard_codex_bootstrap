# Cross-host phase reassignment

Status: in progress

## Goal

Use idle GPUs without changing scientific settings or allowing duplicate phases:

- Hamster GPU 0: `prod-f0-chen-student-s0` train and PGD.
- Ferret GPU 2: `prod-h1-bart-entropy-s0` AutoAttack only.

The original campaign Git SHA, method config, W&B identity, checkpoint bytes, and evaluation threat model remain fixed.

## Checklist

- [x] Confirm both target phases are not live and both destination GPUs are idle.
- [ ] Stop automatic launch of the two source assignments without stopping unrelated live phases.
- [ ] Transfer only the Bartoldson entropy evaluation inputs and verify their SHA-256 on Ferret.
- [ ] Launch both phases with durable process/exit identity and GPU UUID records.
- [ ] Import terminal evidence back to the canonical output/state location without overwriting prior evidence.
- [ ] Add focused duplicate-prevention and lineage tests, then run `scripts/verify.py --changed`.
- [ ] Update the experiment dashboard with actual launch/result evidence.

## Safety and rollback

- Never edit the immutable scientific worktree or its resolved config.
- Never start a destination phase while its source controller may still launch the same phase.
- Never copy the 4.7 GiB training bundle when resolved config, best/last checkpoints, and training manifest suffice.
- A failed transfer or launch leaves the original completed training/PGD artifacts untouched and restores the source queue.
