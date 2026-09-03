# I100 Dynamic BDD recovery audit

## Decision before recovery launch

The original eight-job campaign must not be rerun wholesale.  File-level
lineage, checkpoint, and endpoint inspection leaves three reusable complete
jobs and five recovery jobs.  No training-complete job has a missing endpoint.

| Job | Seed | Arm | Classification | Disposition |
| --- | --- | --- | --- | --- |
| dev1-control | dev-1 | Control | `VALID_COMPLETE` | Reuse |
| dev1-dpm | dev-1 | DPM | `VALID_COMPLETE` | Reuse |
| dev1-dbdd | dev-1 | D-BDD | `NOT_STARTED` | Launch after remote parent materialization |
| dev1-sbdd | dev-1 | S-BDD | `NOT_STARTED` | Launch with corrected formula and v2 calibration |
| dev2-control | dev-2 | Control | `NOT_STARTED` | Launch |
| dev2-dpm | dev-2 | DPM | `NOT_STARTED` | Launch after remote parent materialization |
| dev2-dbdd | dev-2 | D-BDD | `VALID_COMPLETE` | Reuse |
| dev2-sbdd | dev-2 | S-BDD | `SCIENTIFICALLY_INVALID` | Rerun from e99 with corrected formula and v2 calibration |

The machine-readable inventory is
[`ert_rslad_i100_s2_dynamic_bdd_recovery_audit_v1.json`](experiments/ert_rslad_i100_s2_dynamic_bdd_recovery_audit_v1.json).

## Evidence

Each reusable job has an exact e99 causal parent, the registered
sample-keyed KL-PGD10 training identity, frozen Teacher provenance, an e114
checkpoint, a completion marker, and CE-PGD20 validation endpoints at e104,
e109, and e114 plus the e114 train endpoint.  The validation endpoint row
count is 5,000 and the train row count is 45,000.  The endpoint input
checkpoint SHA-256 values were checked against their respective horizon
checkpoint.

The reusable outputs span source commits `1504e7b` and `865567d`.  Their
intervening diff changes only Ferret worktree preparation serialization and
its regression test; it does not alter the Trainer, objective, attack, mask,
or endpoint runtime.  Reuse remains conditional on the recovery source-delta
one-batch parity record for Control, DPM, and D-BDD.

The `dev1-sbdd` job failed in Ferret worktree preparation before scientific
compute.  `dev1-dbdd` and `dev2-dpm` failed before scientific compute because
Ferret lacked the required exact parent config.  `dev2-sbdd` must not be
resumed: it used the detached Student secant denominator and became non-finite
at epoch 107, so it is a different intervention rather than a technical
interruption.

## Recovery constraints

The recovery is limited to five e99-to-e114 continuations:

- `dev1-dbdd`
- `dev1-sbdd`
- `dev2-control`
- `dev2-dpm`
- `dev2-sbdd`

The valid Control/DPM/D-BDD outputs retain their original v1 calibration.
Only corrected S-BDD receives a new, pooled no-update v2 calibration artifact.
No new arm, threshold, seed, e199 extension, official test, or AutoAttack is
permitted.
