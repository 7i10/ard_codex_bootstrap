# ERT / RSLAD Student History Predictive Validity

## Status

- Owner: Codex root
- Status: complete (read-only analysis; no training)
- Source HEAD at reconciliation: `7c3320361965406f4acd5f5e9a756c9abb2751e8`
- Methods: `BASE` reconvergence diagnostic and frozen `I100` history analysis
- Seeds: `dev-1`, `dev-2`, `confirm-a`, `confirm-b`, `confirm-c`

## Scope and non-goals

Use existing metrics, endpoint rows, and checkpoint `sample_state` only. No
training, endpoint attack change, new seed, coefficient/threshold tuning,
History intervention, Ordering intervention, official test, or AutoAttack.

## Frozen contracts

- BASE reconvergence uses the existing five-seed dense `val_pgd_accuracy`
  trajectories and existing CE-PGD20 endpoint rows.
- History cutoffs are scientific epochs `49`, `99`, and `149`; the future
  primary target is robust-correct rate over `150:199`, computed from
  cumulative counters at the two epoch-boundary checkpoints.
- Feature families are fixed: P0 current correctness; P1 current margin; P2
  current correctness plus current margin; P3 history-only counters/EMA/streak;
  P4 P2 plus P3. Ridge uses alpha `1.0` and dev-only standardization.
- Confirmation seeds are evaluation-only; fitting uses `dev-1` and `dev-2`.

## Execution record

1. Existing five-seed artifacts and the prior global/sample report were
   reconciled. BASE dense trajectories and all 60 endpoint row artifacts are
   present.
2. I100 checkpoints were inspected and contain format-v3, complete 45,000-ID
   sample state. Confirm-c checkpoints were recovered from Ferret using the
   exact paths and SHA-256 values recorded in the historical endpoint manifest;
   no training was started.
3. The analysis audited checkpoint semantics, computed the BASE RO diagnostic,
   fit/evaluated the preregistered feature families, and wrote non-overwriting
   machine reports plus a human report.
4. Confirm-c dense prefix metrics and checkpoints were materialized from Ferret
   with SHA-256 verification; no new training or endpoint attack was run.
5. The analysis produced 600 dense global rows, 20 complete checkpoint cells,
   and 93 predictive-validity rows. Artifact consistency, compilation, the
   dense-metric validator, and six focused unit tests passed. The broad changed
   test gate also exposed one unrelated pre-existing CUDA RNG-state failure in
   `test_schedule_control_fork`; it was not modified here.

## Known lineage caveats

- Checkpoint filenames use a payload epoch one lower than the scientific
  endpoint label; payload epoch, `seen`, and boundary fields are recorded rather
  than inferred.
- The I100 `epoch-049`/`epoch-099` feature checkpoints are the shared
  CROPSHIFT prefix. The recovered confirm-c files are local cache copies whose
  SHA is rechecked against Ferret and the endpoint manifest.

## Acceptance gates

- Each checkpoint contains exactly 45,000 unique stable IDs; labels are
  consistent across cutoffs. IDs are the repository's original CIFAR stable
  IDs and are not assumed to be the contiguous range `0..44,999`.
- `sample_state` is format-v3, epoch-boundary, pending-empty, complete, and
  uses the actual `seen` denominator.
- BASE dense rows are contiguous, unique, finite, and have the expected five
  seeds. Missing prefix metrics are not imputed.
- Predictive results distinguish dev fitting from confirmation evaluation and
  report unavailable cells explicitly.

## Decision

Reports and validators passed under the frozen read-only contract. Commit/push
only the analysis code, plan, reports, and small artifacts. Stop without
launching Ordering or any training.
