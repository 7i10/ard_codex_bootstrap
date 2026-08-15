# ERT Majority-3 action persistence replay

## Status

- Status: complete, offline-only
- Base: `9f2da3a` (initial history replay)
- Implementation extension: `5836f18`
- No GPU/training/endpoint process was started.

## Frozen comparison

Use the same L2/L4 S3DYN075 trajectory and current Teacher adversarial-correct
gate for:

- Majority-3 plain (state smoothing control)
- Majority-3 with 2 or 3 consecutive-correct exit
- Majority-3 with minimum dwell 2 or 3 visits

Compare only switch count/reduction, re-entry, persistent capture, entry delay,
active duration, transient activation, and Student/Teacher switch attribution.

## Progress

- [x] Add exit-rule and minimum-dwell replay functions.
- [x] Add unit coverage separating exit streak from minimum dwell.
- [x] Replay both 675,000-row trajectories.
- [x] Save human and machine-readable results.
- [x] Stop before any production experiment.

## Result and decision

Majority-3 + exit2-correct is the primary future candidate. It reduces action
switches from 85,628 to 76,738 on the L2/L4 mean, eliminates one-visit
re-entry, and increases descriptive persistent capture from 69.4% to 71.5%.
Exit3 is more stable but has median active duration 13 visits and is therefore
too sticky as the default. Minimum-dwell2/3 provides little additional switch
reduction and does not eliminate one-visit re-entry.

The recommendation is mechanism-only; no robust-accuracy claim or automatic
training launch follows from it.
