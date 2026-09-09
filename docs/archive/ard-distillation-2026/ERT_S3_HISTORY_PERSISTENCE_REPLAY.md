# Majority-3 state smoothing versus action persistence

Date: 2026-08-15
Status: complete, offline-only; no training or endpoint evaluation was run.

## Purpose and contract

The previous replay showed that history smoothing and action persistence had
not been separated on the same Majority-3 entry rule. This extension uses the
same L2/L4 S3DYN075 trajectories and compares:

```text
Majority-3 plain
Majority-3 + 2 consecutive correct visits to exit
Majority-3 + 3 consecutive correct visits to exit
Majority-3 + minimum dwell 2 visits
Majority-3 + minimum dwell 3 visits
```

The action remains `Student history state active AND current Teacher
adversarial correct`. The descriptive reference remains “the next three
visits contain at least two Student adversarial-wrong visits.” No validation,
endpoint, or test accuracy is used for selection.

Inputs are unchanged: 45,000 IDs × 15 visits for each seed, with the same
stable ID/class universe. The corrected full-window semantics are retained:
Majority-3 is inactive until three visits are available.

## Results (L2/L4 mean)

Switch reduction is relative to Instant from the previous replay. Persistent
capture and transient activation use the descriptive next-three-visit
reference. `median duration` is the active-state run length.

| rule | action fraction | switches | reduction vs Instant | re-entries | 1-visit re-entry | persistent capture | transient activation | median entry delay | median duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority-3 plain | 0.264 | 85,628 | 43.0% | 13,567 | 12.6% | 69.4% | 52.0% | 2 | 4 |
| Majority-3 + exit2 | 0.282 | 76,738 | 49.0% | 8,011 | 0% | 71.5% | 52.4% | 2 | 7 |
| Majority-3 + exit3 | 0.320 | 66,760 | 55.6% | 3,197 | 0% | 75.4% | 56.7% | 2 | 13 |
| Majority-3 + min-dwell2 | 0.273 | 82,126 | 45.4% | 11,559 | 8.8% | 70.2% | 52.0% | 2 | 5 |
| Majority-3 + min-dwell3 | 0.283 | 80,994 | 46.1% | 11,064 | 10.4% | 71.2% | 52.4% | 2 | 5 |

The L2/L4 results are closely aligned. For example, exit2 has 76,780/76,696
switches and 71.5%/71.4% capture, while exit3 has 66,827/66,693 switches and
75.5%/75.3% capture.

## Interpretation

### State smoothing

Majority-3 plain is the clean state-smoothing comparison. It already removes
43% of Instant action switches and retains about 69% of the reference failure
capture with a two-visit median entry delay.

### Action persistence

The consecutive-correct exit rules add a distinct persistence mechanism:

- **Exit2** is the best small persistence increment. It removes another 10.4%
  of Majority-3 switches, eliminates one-visit re-entry, slightly increases
  capture (+2.1 pp), and does not increase entry delay. Its active fraction and
  transient rate change only modestly.
- **Exit3** removes another 22.0% of Majority-3 switches and raises capture by
  about 6 pp, but it produces median active runs of 13 visits and raises the
  active fraction to 32.0%. This is likely over-persistent for a 15-visit
  observation window and leaves less opportunity to recover from a mistaken
  entry.
- **Minimum dwell2/3** is not equivalent to a correct-streak exit. It only
  guarantees a short minimum duration, so one-visit re-entry remains (8.8% and
  10.4%) and switch reduction is small. It does not provide a clear advantage
  over exit2 in this trajectory.

## Recommendation

For a future preregistered training screen, use **Majority-3 + exit2-correct**
as the primary action-persistence candidate, with Majority-3 plain as the
state-smoothing control. Keep exit3 only as a secondary “strong persistence”
ablation. Do not use minimum-dwell2/3 as the primary candidate based on this
replay.

This recommendation separates mechanism, not performance: it says exit2 is a
compact way to test action persistence while preserving the Majority-3 entry
behavior. It does not establish an improvement in robust accuracy. Teacher
correctness flips remain a separate source of action switches and are not
smoothed by this Student-only replay.

## Reproduction and provenance

```bash
PYTHONPATH=src python -m ard.cli.ert_s3_history_replay \
  --config configs/analysis/ert_s3_history_replay_v1.yaml \
  --output .cache/analysis/ert-s3-history-replay-v1/report-v3.json
```

Implementation commit: `5836f18`
Report SHA-256: `6546e32adf783d3d079e656596dcad44f4c419e564c3d1807add070f519e57f7`

No production run, coefficient change, new seed, official test, or AutoAttack
was started.
