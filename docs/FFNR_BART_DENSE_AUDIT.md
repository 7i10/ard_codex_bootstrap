# Bartoldson dense-checkpoint audit

This audit uses existing saved artifacts only. It does not retrain Bartoldson
and does not claim a resume-parity result.

## Available state

`outputs/scientific/bart-rslad-logging-only-s1-confirm-v1` contains periodic
`last.pt` artifacts every five epochs, including epochs 99, 104, 109 and 114.
The inspected checkpoints contain model, optimizer, scheduler, scaler/RNG,
sampler epoch, global step, and 45,000-sample state. The scheduler still has
milestones 100 and 150; the observed learning rates and `last_epoch` values
are consistent with the ordinary schedule (0.01 after the epoch-100 step).

This means the existing IRT seed-1 trajectory already provides a checkpoint
near the plateau. A dense continuation is not needed merely to obtain a
checkpoint every five epochs, and changing the scheduler or save cadence in a
resume would change the experiment identity.

## Missing state and decision

No local Bartoldson L3/seed-2 run bundle or model checkpoint was found. The
current evidence therefore supports:

1. use the existing seed-1 periodic artifacts for offline diagnostics;
2. do not launch a 200-epoch retrain in this phase;
3. only attempt epoch-99→115 continuation after an explicit, immutable
   resume-parity harness is available; and
4. treat the IRT CE-PGD20 factorial cell as blocked until a complete L3
   checkpoint inventory is recovered or a new dense-save run is approved.

The CE/KL factorial currently executed in this phase is Chen L2/L4 only, as
specified. Bartoldson remains a follow-up audit rather than an unplanned new
training campaign.
