# Clean-Wrong margin screen calibration

Status: complete, no-update calibration.

The calibration was run after exact L4 parent recovery from clean source
`95346cf14ffd5e4a35f7f31742eafa50fde8681d`.  No optimizer, scheduler,
checkpoint, sample-state, or endpoint metric was used or mutated.  The
calibration used the hash-bound KL-PGD10 pre-treatment feature rows and
deterministic class-stratified Clean-Wrong samples from both Chen seeds.

## Frozen values

| quantity | frozen value |
|---|---:|
| target gradient ratio | 0.25 |
| shared AdvCE coefficient | 0.07726448029279709 |
| shared probability-margin coefficient | 0.2388051152229309 |
| target-policy temperature contract | 2.0 (unchanged; no target softening arm in 0050) |

Positive Teacher adversarial-margin quantiles used for the margin target
family were computed separately from each pre-treatment KL-PGD10 feature
distribution:

| seed | positive rows | Q25 (floor) | Q50 (fixed) | Q75 (cap) |
|---|---:|---:|---:|---:|
| L2 | 3,452 | 0.0310035981 | 0.0715777874 | 0.1374905221 |
| L4 | 3,729 | 0.0335526764 | 0.0752260089 | 0.1408998668 |

The pooled positive-margin quantiles frozen for the shared target family are
Q25 `0.0322171003`, Q50 `0.0732170194`, and Q75 `0.1395255029`.

The A4/A5 fixed-margin arms use the preregistered pooled fixed target (the
calibration artifact records per-seed Q50 values); Teacher-target arms use
the frozen cap/floor contract and do not retune thresholds from outcomes.

## Provenance

Machine-readable artifact:

```text
docs/experiments/ert_cw_margin_calibration_v1.json
SHA256: 058d3c440511308df2b004c3015f3fdec8aca027176c01620bba1913c0dc1582
```

Both parent checkpoint hashes, both fixed Clean-Wrong mask hashes, feature
metadata/row hashes, calibration sample-ID hashes, attack contract, and the
clean Git SHA are embedded in the JSON.  The corresponding sidecar contains
the same artifact hash.

## Gate result

The no-update calibration passed with finite, non-zero AdvCE and margin
gradient denominators.  Production canary is now the next gate; no 0050
production trajectory has been started yet.
