# ERT S3 history routing offline replay

## Status

- Status: complete, offline-only
- Base experiment: Dynamic S3 recovery screen (`d422948` result record)
- Analysis implementation: `fadb274`
- No GPU or training process was launched.

## Frozen scope

Replay the saved L2/L4 S3DYN075 KL-PGD10 trajectories for Instant,
Consecutive-2/3, Majority-3/5, Loose-5, and Majority-5 with two- and
three-consecutive-correct exits. Student adversarial-wrong history is the only
historical signal. Teacher adversarial correctness is a current gate. The
near-future reference is descriptive only: at least two wrong visits in the
next three visits.

## Progress

- [x] Verify HEAD, result docs, trajectory paths, row counts, and stable ID/class joins.
- [x] Add read-only replay CLI and frozen config.
- [x] Add causal-window and asymmetric state-machine tests.
- [x] Run the replay on both 675,000-row trajectories.
- [x] Save human report and machine-readable summary.
- [x] Stop without training, endpoint evaluation, coefficient tuning, or new seed.

## Evidence

- L2/L4 each contain 45,000 IDs × 15 epochs (80--94), 675,000 rows.
- Both stable ID/class hashes equal
  `615e67980788b40c78793c6bbdfcdccd292f805869eb9085f445b202fe9a49c8`.
- Report: `.cache/analysis/ert-s3-history-replay-v1/report-v2.json`.
- Report SHA-256:
  `7c6b41f482ebda34d0f3511cad734e00643f333fd84dcc6e3871d37153701932`.
- Focused test: `3 passed`; Ruff passed.

## Decision

Majority-3 is the primary offline candidate: it reduces action switching by
about 43% relative to Instant, retains about 69% of the descriptive
persistent-failure capture, and has a two-visit median entry delay in both
seeds. Majority-5 with a three-correct exit is retained only as a stability
comparator. Neither rule is approved for training until a separate
preregistered intervention plan is reviewed.
