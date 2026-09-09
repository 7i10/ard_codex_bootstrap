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

The seed-2 control run was subsequently recovered from the single identified
W&B artifact `run-bundle-bart-h3-c-s2-20260802-v1:v0`; no retraining was
needed. The recovered bundle contains complete-state checkpoints at epochs
104, 109, and 114. CE-PGD20 replay was run for both seed 1 and seed 2 with
the same attack identity and stable-ID contract. The replay outputs are kept
under the ignored `.cache/analysis/ffnr-strong-replay/` tree and are bound by
their `lineage.json` files.

The current evidence therefore supports:

1. use the existing seed-1 periodic artifacts for offline diagnostics;
2. do not launch a 200-epoch retrain in this phase;
3. only attempt epoch-99→115 continuation after an explicit, immutable
   resume-parity harness is available; and
4. treat the IRT CE-PGD20 replay as an offline diagnostic, not as a new
   training result; no dense-save retraining was launched.

The CE/KL factorial currently executed in this phase is Chen L2/L4 only, as
specified. Bartoldson remains a follow-up audit rather than an unplanned new
training campaign. The replay comparison is documented in
`docs/FFNR_BART_CE_PGD20_REPLAY.md`.
