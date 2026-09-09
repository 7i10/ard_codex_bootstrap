# ERT S3 history routing: offline replay

Date: 2026-08-15
Status: complete; no training, endpoint evaluation, coefficient change, or
new seed was run.

## Scope and inputs

This analysis uses only the saved S3DYN075 KL-PGD10 state trajectories from the
completed Dynamic S3 screen. It does not load a checkpoint or validation/test
metric. Both inputs contain exactly 45,000 stable IDs × 15 visits (epochs
80--94), with no duplicate `(sample_id, epoch)` pair and a stable ID/class
hash of `615e67980788b40c78793c6bbdfcdccd292f805869eb9085f445b202fe9a49c8`.

| run | trajectory SHA-256 |
|---|---|
| L2 | `cf6827c7b3b8b605720152ea6ebaf7532f434f43d97787396679101661d1cfff` |
| L4 | `e3f40b10e4926c09aefc7a381f2f684b028aeb82f59444675d14be17908a0d5a` |

The rules use only Student adversarial-wrong history. The action is

```text
history state active AND current Teacher adversarial correct
```

Teacher correctness is therefore a current gate, not a historical feature.
Majority/Loose windows remain inactive until the full 3- or 5-visit window is
available. This prevents partial-history activation at epoch 80.

The descriptive near-future reference is fixed as: “the next three visits
contain at least two adversarial-wrong visits.” It is used only to report
capture/transient trade-offs; it is not a training target or selection metric.

## Aggregate comparison (L2/L4 mean)

`switches` counts final action changes after the current Teacher gate. Capture
is the Student-state capture rate for the descriptive near-future reference;
the report also stores the lower Teacher-gated action capture rate. Delay is
the median visit delay from the first observed wrong visit to first activation.

| rule | action fraction | action switches | reduction vs Instant | persistent capture | transient activation | median entry delay |
|---|---:|---:|---:|---:|---:|---:|
| Instant | 0.303 | 150,346 | reference | 0.790 | 1.000 | 0 |
| Consecutive-2 | 0.193 | 86,975 | 42.1% | 0.611 | 0.392 | 1 |
| Consecutive-3 | 0.139 | 57,224 | 61.9% | 0.487 | 0.194 | 2 |
| Majority-3 | 0.264 | 85,628 | 43.0% | 0.694 | 0.520 | 2 |
| Majority-5 | 0.224 | 62,257 | 58.6% | 0.566 | 0.361 | 4 |
| Loose-5 | 0.293 | 67,267 | 55.3% | 0.623 | 0.529 | 4 |
| Majority-5, exit 2 correct | 0.230 | 61,886 | 58.8% | 0.571 | 0.368 | 4 |
| Majority-5, exit 3 correct | 0.250 | 53,138 | 64.6% | 0.593 | 0.382 | 4 |

The L2 and L4 values agree closely. For example, Majority-3 action switches
are 85,690 / 85,566 and state capture is 0.694 / 0.693; the
Majority-5/exit-3 action switches are 53,291 / 52,986 and state capture is
0.593 / 0.592.

## Student versus Teacher switch source

The decomposition classifies an action switch as Student-only, Teacher-only,
or both when the corresponding state changes at that visit. Aggregate counts
are:

| rule | Student-only | Teacher-only | both | Teacher-only share of action switches |
|---|---:|---:|---:|---:|
| Instant | 120,641 | 29,442 | 263 | 19.6% |
| Majority-3 | 51,850 | 32,046 | 1,582 | 37.4% |
| Majority-5 | 32,872 | 27,762 | 1,308 | 44.6% |
| Majority-5, exit 3 correct | 22,608 | 28,982 | 1,242 | 54.6% |

History smoothing removes many Student-only switches, but it cannot remove
Teacher-gate switches. Under the strongest smoothing candidate, more than half
of the remaining action switches are Teacher-only. A future Teacher hysteresis
design would therefore be a separate mechanism, not a reason to alter the
Student history rule in this replay.

## Entry/exit interpretation

- `Majority-3` is the simplest balanced rule: it cuts switches by about 43%,
  retains about 69% of the near-future persistent-failure positions, and has a
  two-visit median entry delay.
- `Majority-5` and `Loose-5` reduce switches further but delay entry by four
  visits. Plain Majority-5 loses more persistent-failure capture than
  Majority-3; Loose-5 activates more transient failures and nearly restores
  the Instant active fraction.
- `Majority-5 + exit-2` adds state-machine complexity without materially
  improving capture or switching over plain Majority-5.
- `Majority-5 + exit-3` has the fewest switches among the majority rules and
  eliminates one-visit re-entry in both runs, but it still waits four visits
  to enter and captures only about 59% of the reference. It is a useful
  secondary stability candidate, not the default.
- `Consecutive-3` is stable but misses about half of the descriptive
  persistent-failure positions, so it is too conservative for the stated goal.

## Recommendation before any new training

Use **Majority-3 Student history + current Teacher adversarial-correct gate**
as the primary next-rule candidate if a new training is later approved. It is
the simplest rule satisfying the cross-seed stability requirement while
preserving substantially more early persistent-failure capture than the
5-visit rules. Keep Majority-5/exit-3 as a preregistered stability comparator
if the primary objective changes to minimizing action switching.

This is an offline mechanism recommendation only. It is not evidence that the
intervention improves robust accuracy. Before training, remaining concerns are
Teacher-gate flapping, the four-visit censoring of late trajectory behavior,
and whether the descriptive next-three-visit reference transfers to a real
future horizon. No threshold or coefficient was tuned from endpoint results.

## Reproduction

```bash
PYTHONPATH=src python -m ard.cli.ert_s3_history_replay \
  --config configs/analysis/ert_s3_history_replay_v1.yaml \
  --output .cache/analysis/ert-s3-history-replay-v1/report-v2.json
```

The generated report SHA-256 is
`7c6b41f482ebda34d0f3511cad734e00643f333fd84dcc6e3871d37153701932` and the
clean-source implementation commit is `fadb274b8aa94d3d9dd856978d903d359f0da2d8`.
