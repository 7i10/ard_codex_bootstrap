# Verified teacher cache restore

## Failure signature

A fresh clone contains verified checkpoint identities in `teachers.lock.yaml`
but intentionally lacks the Git-ignored `teacher_cache/`. The documented
`bootstrap_teacher.py --update-lock` command then failed because a verified SHA
cannot be advanced, leaving no supported way to materialize the locked bytes.

## Evidence and root cause

The workflow conflated two separate operations: a maintainer registering the
first project-owned checkpoint identity, and an operator restoring an ignored
cache from an already committed identity. `--update-lock` correctly handled the
first operation only. Separately, reacquisition validated model shape and
parameter count but needed to bind staged bytes to an existing verified SHA
before external publication.

## Bounded fix

`bootstrap_teacher.py --install-locked` now requires a consistent verified raw
lock and registry SHA, checks the supplied file before no-clobber publication,
and leaves `teachers.lock.yaml` byte-for-byte unchanged. `--update-lock` remains
available only for a `missing` entry's initial identity. The acquisition CLI
likewise checks staged bytes against a verified lock SHA before publishing them;
a mismatch leaves neither a final external file nor staging residue.

## Regression coverage

- `test_teacher_bootstrap_install_locked_restores_verified_bytes_without_changing_lock`
- `test_teacher_bootstrap_install_locked_rejects_mismatch_without_destination_or_lock_change`
- `test_verified_acquisition_rejects_mismatched_staged_bytes_without_publish_or_stage`
- `test_verified_acquisition_publishes_only_matching_staged_bytes`

The affected teacher/bootstrap tests and impact-selected gate must run after the
code fix. This documentation-only synchronization does not repeat GPU audits,
CIFAR training, W&B, AutoAttack, or unchanged scientific regressions.

## Scientific impact

No training or evaluation result was contaminated. Before this fix, runtime
teacher loading and the local-only audit already rejected checkpoint bytes that
did not match the registered SHA. The defect blocked reproducible cache restore;
it did not weaken the threat model, alter normalization, or permit mismatched
teacher logits into a completed experiment.
