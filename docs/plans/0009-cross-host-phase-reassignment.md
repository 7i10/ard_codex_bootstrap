# Cross-host phase reassignment

Status: in progress

## Goal

Use idle GPUs without changing scientific settings or allowing duplicate phases:

- Hamster GPU 0: `prod-f0-chen-student-s0` train and PGD.
- Ferret GPU 2: `prod-h1-bart-entropy-s0` AutoAttack only.

The original campaign Git SHA, method config, W&B identity, checkpoint bytes, and evaluation threat model remain fixed.

## Checklist

- [x] Confirm both target phases are not live and both destination GPUs are idle.
- [x] Stop automatic launch of the two source assignments without stopping unrelated live phases. The user approved
  the pause; both watchdogs/controllers stopped and detached scientific children remained live.
- [x] Transfer only the Bartoldson entropy evaluation inputs and verify their SHA-256 on Ferret.
- [x] Launch both phases with durable process/exit identity and GPU UUID records.
- [ ] Import terminal evidence back to the canonical output/state location without overwriting prior evidence.
- [x] Add focused portability/lineage tests, then run the impact-selected gate.
- [x] Update the experiment dashboard with actual launch/result evidence.

## Safety and rollback

- Never edit the immutable scientific worktree or its resolved config.
- Never start a destination phase while its source controller may still launch the same phase.
- Never copy the 4.7 GiB training bundle when resolved config, best/last checkpoints, and training manifest suffice.
- A failed transfer or launch leaves the original completed training/PGD artifacts untouched and restores the source queue.

## Execution evidence

- Hamster GPU 0 launched Chen/Student train+PGD at `2026-07-29T05:48:11Z`; W&B online initialization and CUDA
  utilization were observed.
- Ferret GPU 2 launched Bartoldson/Entropy AutoAttack at `2026-07-29T06:01:29Z`; W&B evaluation initialization and
  GPU utilization were observed.
- Chen/Student train+PGD completed with exit code 0 at `2026-07-29T10:12:55Z`. Official test best/last clean was
  83.35/83.59%, and PGD-20 was 55.44/55.21%. AutoAttack was not a registered phase for this ablation.
- Bartoldson/Entropy AutoAttack completed with exit code 0 at `2026-07-29T07:42:09Z`. Official test best/last
  AutoAttack was 47.37/46.16%; the corresponding PGD-20 values remained 50.09/48.72%.
- At `2026-07-29T23:24+09:00`, Hamster GPUs 0/1 and Ferret GPU 2 were idle. The canonical state import remains
  intentionally pending, so this idle capacity is not treated as permission to resume the paused schedulers.
- The user approved pausing both source schedulers. Hamster controller PGID `47653` and Ferret controller PGID
  `13019` stopped after their watchdog screen sessions; GPU utilization and detached scientific wrappers remained
  live. Each host has an atomic `control/reassignment-controller-pause.json` record.
- Cross-host evaluation initially failed closed because absolute teacher checkpoint paths were treated as scientific
  identity in three tracking/preflight layers. The runtime now permits only a path relocation: registered SHA-256,
  teacher metadata, normalization, threat model, training config hash, and checkpoint lineage remain exact.
- Failed preflight outputs contain no metrics and are retained under explicit `evaluation-autoattack-failed-*` names.
