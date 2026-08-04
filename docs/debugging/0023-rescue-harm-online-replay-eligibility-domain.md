# Rescue/harm eligibility-domain mismatch

## Failure signature

The completed-v2 `rescue_harm report` command rejected the real L3 selector
bundle with:

```text
selector mask does not match epoch39 feature-route eligibility
```

The failure occurred after all fixed-checkpoint CE-PGD20 replays had completed;
it did not affect training or the replayed checkpoint predictions.

## Evidence

- The selector bundle declares `source: online_history_epoch39_v2` and binds the
  epoch-39 parent checkpoint and `SampleStateStore` SHA-256.
- The report reconstructed PF/NR eligibility from
  `feature-observations.parquet:robust_correct` at replay epoch 39.
- Those replay predictions use a common evaluation PGD draw and are not the
  pre-update online training observations stored in the parent checkpoint.
- On L3, 1,617/2,149 PF-history IDs and 25/2,350 NR-history IDs disagreed with
  replay-derived eligibility. The mismatch is therefore structural, not a
  floating-point tolerance issue.
- Attack identity, stable IDs, checkpoint hashes, and class joins remained
  valid; no evidence points to attack, normalization, or checkpoint corruption.

## Ranked hypotheses

1. **Confirmed:** online training-state eligibility was incorrectly
   reconstructed from checkpoint replay correctness in a different observation
   domain.
2. Rejected: a stale selector bundle. Its parent checkpoint, sample-state, mask,
   and class-count hashes are internally consistent.
3. Rejected: sparse-ID join corruption. IDs are unique original CIFAR train IDs,
   and class joins pass.
4. Rejected: numerical tolerance. The failed field is boolean correctness and
   the disagreement is large.

## Root cause

The analysis mixed two scientifically distinct domains:

- **selector eligibility:** exact epoch-39 `SampleStateStore` online
  `previous_robust_correct` used by the completed training arms;
- **outcome replay:** common CE-PGD20 checkpoint inference used to compare
  Control and interventions.

The latter is valid for Rescue/Harm outcomes but cannot redefine which samples
the already-completed intervention selected as PF or NR.

## Bounded fix

Require the exact parent checkpoint as a report input. Verify its SHA against
the selector bundle, verify the serialized sample-state SHA and all sparse
ID/class records, and derive PF/NR eligibility only from the checkpoint's
online correctness state. Continue using the common replay panels exclusively
for Control-to-arm Rescue/Harm outcomes. Record both domain identities in the
report lineage.

## Regression

Add a fixture where online correctness deliberately differs from replay
correctness. The report must accept masks consistent with the online checkpoint
and must reject a wrong checkpoint/sample-state hash. Existing attack and
training tests are not rerun because neither the attack nor training path
changes; run the focused rescue/harm tests and the final impact-selected
non-scientific gate.
