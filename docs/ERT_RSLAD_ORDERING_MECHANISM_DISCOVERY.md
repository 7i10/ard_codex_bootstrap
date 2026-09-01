# ERT / RSLAD Ordering Mechanism Discovery

## Status

The corrected pure-order probe campaign is complete. All 16 short forks (8
pre-registered order schedules × development seeds 1 and 2) reached epoch 114,
and batch telemetry/descriptors are present for all 16 runs. The mechanism gate
did not pass, so the second ordering intervention was not launched.

The first detached aggregation attempt failed for a technical reason in the
AUC helper: adjacent point lists were passed to `zip(..., strict=True)` even
though they necessarily differ in length by one. The helper was corrected,
covered by a regression test, and the result was recomputed from the completed
run artifacts without retraining.

## Lineage and integrity

- Campaign: `ert-rslad-ordering-mechanism-probe-retry4-v1`
- Source used by the retry manifest: `0273abda01c6c80ebaa8a182d165884edce7237a`
- Registry: `docs/experiments/ert_rslad_pure_order_probe_registry_v2.json`
- Registry SHA-256: `eff64c9cfb790d48b96344c8672b175cdf22e6cecaf0b91e64fbc512db592721`
- Run root: `/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-ordering-probes-v2`
- Probe epochs: 100–114 (15 epochs per fork)
- Telemetry files: 16/16
- Descriptor files: 16/16
- Training runs: 16/16 completed
- Second intervention: not allowed by the gate

The machine-readable result is
`docs/experiments/ert_rslad_pure_order_probe_results_v5.json`.

## Probe AUC and endpoint metrics

Values below are percentages. `Probe AUC` is the mean trapezoidal training
robust accuracy over epochs 100–114; `RA@114` and `Clean@114` are the final
probe-epoch metrics.

### Development seed 1

| schedule | Probe AUC | RA@114 | Clean@114 |
| --- | ---: | ---: | ---: |
| SHUFFLE_PLUS_0 | 50.9745 | 52.6733 | 66.0889 |
| SHUFFLE_PLUS_1 | 50.9848 | 52.6089 | 66.1644 |
| SHUFFLE_PLUS_2 | 50.9621 | 52.5956 | 66.1711 |
| SHUFFLE_PLUS_3 | 50.9431 | 52.5489 | 66.1200 |
| SHUFFLE_PLUS_4 | 50.9597 | 52.6467 | 66.1067 |
| SHUFFLE_PLUS_5 | 50.9324 | 52.7356 | 66.2733 |
| SHUFFLE_PLUS_6 | 50.9887 | 52.6200 | 66.1133 |
| SHUFFLE_PLUS_7 | 50.9752 | 52.4289 | 66.2178 |

### Development seed 2

| schedule | Probe AUC | RA@114 | Clean@114 |
| --- | ---: | ---: | ---: |
| SHUFFLE_PLUS_0 | 50.9382 | 52.2111 | 65.8844 |
| SHUFFLE_PLUS_1 | 50.8990 | 52.1089 | 65.8467 |
| SHUFFLE_PLUS_2 | 50.9335 | 51.9667 | 65.8156 |
| SHUFFLE_PLUS_3 | 50.9783 | 52.1156 | 65.8844 |
| SHUFFLE_PLUS_4 | 50.9168 | 52.0644 | 65.8867 |
| SHUFFLE_PLUS_5 | 50.9657 | 52.1644 | 65.6911 |
| SHUFFLE_PLUS_6 | 50.9131 | 52.1911 | 65.8178 |
| SHUFFLE_PLUS_7 | 50.9495 | 52.0822 | 65.8356 |

The best schedule differs by seed for both the final robust metric and the
probe AUC. No schedule shows a consistent seed-paired advantage.

## Descriptor × probe-AUC associations

Spearman associations are computed across the eight frozen schedules within
each seed. The preregistered gate requires equal sign, minimum absolute rho
`>= 0.40` in both seeds, and mean absolute rho `>= 0.50`.

| descriptor | rho seed 1 | rho seed 2 | same sign | gate |
| --- | ---: | ---: | :---: | :---: |
| D1 batch-mean risk SD | 0.119 | -0.310 | no | fail |
| D2 within-batch risk SD mean | 0.119 | 0.190 | yes | fail |
| D3 high-risk-fraction SD | 0.190 | 0.238 | yes | fail |
| D4 lag-1 batch-risk ACF | -0.143 | -0.214 | yes | fail |
| D5 longest hard-batch run | -0.406 | -0.170 | yes | fail |
| D6 position vs batch-risk Spearman | -0.571 | -0.238 | yes | fail |

D6 has the largest mean absolute association (`0.405`) but does not satisfy
the per-seed minimum. D5 is also below the gate in seed 2. The remaining
descriptors are weak and/or inconsistent.

## Mechanism decision

**MECHANISM NOT IDENTIFIED.** The observed batch-order descriptors do not give
a reproducible association with short-horizon probe AUC across the two seeds.
The measured descriptor ranges are also small relative to the schedule changes;
this probe therefore provides no defensible basis for selecting a batch-order
policy.

Because the mechanism gate failed:

- no second history-conditioned ordering intervention was started;
- no order schedule or coefficient was promoted;
- no new seed, endpoint, or training continuation was started.

This result does not prove that ordering can never matter. It says that this
pre-registered eight-schedule pure-order probe, with the corrected HIGH/LOW
telemetry contract, did not identify a stable mechanism. Further ordering
design should remain blocked pending human review and a new, explicitly
pre-registered question.

## Verification

- Focused ordering/AUC tests: passed (4 tests).
- `scripts/verify.py --changed`: the changed-test run reached the existing
  schedule-control suite, where one unrelated pre-existing test failed because
  a synthetic CUDA RNG state had the wrong size (`test_schedule_control_fork.py`).
  This is not a probe-result failure and did not alter the completed run
  artifacts. The new AUC regression itself passed.
