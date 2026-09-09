# ERT / RSLAD Ordering Mechanism Discovery — Existing Runs Audit

## Conclusion

Phase A is **blocked** and the mechanism gate is not identified. Existing ordering
runs preserve epoch-level permutation digests, but not the batch stable IDs or
per-sample risk values needed to compute D1–D6. No GPU probe, pure-order probe,
or second intervention was launched.

The exact dev I100 e99 parents are present and hash-matching, but that does not
remedy the missing historical batch telemetry. The completed history run also
predates the corrected HIGH/LOW direction and is retained only for lineage and
failure-context reporting; it is not corrected-direction mechanism evidence.

## Existing run inventory

| seed | arm | epoch metrics | ordering rows | ordering fields | batch telemetry |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | NEW_CONTROL | 100 | 0 | none | no |
| 1 | NEW_HISTORY | 100 | 100 | epoch, global_step, permutation_sha256, policy, risk_definition, seed, strata_counts, strata_pattern | no |
| 2 | NEW_CONTROL | 100 | 0 | none | no |
| 2 | NEW_HISTORY | 100 | 100 | epoch, global_step, permutation_sha256, policy, risk_definition, seed, strata_counts, strata_pattern | no |

`NEW_HISTORY` contains 100 rows (epochs 100–199) with `permutation_sha256`,
stratum counts, and pattern only. `NEW_CONTROL` has no ordering-metrics file.
A digest cannot be inverted into the batch risk sequence.

## e99 parent inventory

| seed | path | payload boundary | SHA-256 | match |
| ---: | --- | --- | --- | --- |
| 1 | `/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed1/s100/epoch-100.pt` | epoch=99, epoch_boundary=end | `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835` | yes |
| 2 | `/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1/seed2/s100/epoch-100.pt` | epoch=99, epoch_boundary=end | `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7` | yes |

## D1–D6 availability

| descriptor | available | reason |
| --- | --- | --- |
| D1_batch_mean_risk_sd | no | Existing artifacts contain permutation digests and final sample state, but no per-batch stable-ID/risk observations. |
| D2_within_batch_risk_sd | no | Existing artifacts contain permutation digests and final sample state, but no per-batch stable-ID/risk observations. |
| D3_high_risk_fraction_sd | no | Existing artifacts contain permutation digests and final sample state, but no per-batch stable-ID/risk observations. |
| D4_batch_mean_risk_lag1_acf | no | Existing artifacts contain permutation digests and final sample state, but no per-batch stable-ID/risk observations. |
| D5_hard_batch_clustering | no | Existing artifacts contain permutation digests and final sample state, but no per-batch stable-ID/risk observations. |
| D6_batch_position_risk_spearman | no | Existing artifacts contain permutation digests and final sample state, but no per-batch stable-ID/risk observations. |

## Contract correction

The current sampler now places low-margin samples in HIGH (high-risk) and
high-margin samples in LOW. A focused regression test covers this direction.
The prior history run was not rerun, as required.

## Gate decision

Because no descriptor can be computed from existing artifacts, mechanism selection
is not scientifically identified. Phase B/C/D and holdout training remain not run.
Recovering a batch-level telemetry artifact or running a separately registered
telemetry-producing diagnostic would be required before a second intervention.

Machine artifact: `/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap/docs/experiments/ert_rslad_ordering_mechanism_existing_runs_v1.json` (the pre-self-reference content hash is recorded in the JSON).
