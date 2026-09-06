# Is a "campaign" a source of variation at all?

## Status

- Owner: human (direction), Claude Code (execution)
- Launched 2026-09-06 23:25 JST on both hosts from pinned worktree `ed3b77d`
- Complete 2026-09-07 00:05 JST

## Goal

`docs/MEASUREMENT_STANDARD.md` forbids comparing an arm in one campaign with a
control in another.  The evidence for that rule is one observation: two
nominally identical controls, `C79` and `C79CONF`, differ by 0.94 and 1.78 pp.

That rule costs real flexibility — it makes every past result unpoolable — so it
is worth knowing whether a campaign is actually a source of variation, or whether
that observation has an ordinary explanation.

## What the forensic pass established, before any GPU time

**The 0.94 / 1.78 pp observation is pre-decay.**  Both controls are measured at
epoch 84, sixteen epochs before the learning-rate decay.  The pre-decay floor is
independently measured at **1.14 to 1.25 pp** by five campaign families
(`docs/MEASUREMENT_DESIGN.md` section 2.4).  The observation sits inside it.

Three candidate mechanical explanations were checked and all three are dead:

| candidate | verdict |
| --- | --- |
| The two campaigns ran different training code | **No.** `C79` ran at `54b21dd2`, `C79CONF` at `5b707dbe`. The diff to `ert_stage_a_runtime.py` is 60 insertions and 6 deletions, and every line of it is bookkeeping: horizon checkpoint saving, run naming, validation guards, artifact logging. Nothing touches the loss, the attack, the optimizer, the data path or the RNG. |
| Control and treatment within the Confirmatory campaign ran different code | **True but harmless.** `C79CONF` ran at `5b707dbe` and `T1WCONF` at `e0714671`, six minutes apart. The diff is three new files (a report generator, its CLI, an impact-map entry) and no change to any existing module. It still breaks the rule against committing during a campaign. |
| Running to e94 instead of e84 changed the learning rate at e84 | **No.** The scheduler is `multistep` with milestones at absolute epochs 100 and 150, so the rate at every epoch below 100 is independent of the endpoint. |

What is left is that the two controls differ in their random stream: they are RNG
replicates rather than repetitions, and their gap is the pre-decay floor doing
exactly what it is measured to do.

## Design

One untreated control from the dev-1 epoch-99 parent, epoch-100 prefix then
101 to 114, endpoints at e104/e109/e114, `continuation_seed = 1` — identical in
every recorded respect to plan 0092's `dev-1 rep1`, and run as a **separate
campaign** in a separate directory.

Run twice: once on Hamster under `ard-v2`, once on Ferret under `ard-v2`.
Against plan 0092's `dev-1 rep1`, which ran on Hamster under `adv`, that gives
three comparisons from two runs:

| comparison | isolates |
| --- | --- |
| Hamster/ard-v2 vs Hamster/adv | environment generation, and campaign |
| Hamster/ard-v2 vs Ferret/ard-v2 | host |
| either vs plan 0092 | campaign |

Training is deterministic on fixed inputs, so the readout is **bit-identity of
the checkpoint component hashes**, not a statistic.  No statistics are needed for
a yes-or-no question, and a difference of any size would be the finding.

## Result — the epoch-100 prefix

All nine component hashes are identical across all three: `model`, `optimizer`,
`rng`, `sample_state`, `sampler_epoch`, `sampler_state`, `scaler`, `scheduler`,
`global_step`.  The epoch-100 online-state parquet is `0bb0701f…` in all three.

The checkpoint **file** hashes differ (`e3a3975f`, `6c8486cc`, `66cfd8dd`).  That
is serialization, not science, and it is the second independent confirmation of
`docs/ERT_RSLAD_REAL_DATA_TRAINING_DETERMINISM.md`: a checkpoint file hash must
never be used as a reproducibility test.

## Milestones

- [x] M0: forensic pass on `C79` / `C79CONF`; three mechanical explanations ruled out
- [x] M1: one control replicate per host under `ard-v2`, epochs 101-114
- [x] M2: compare component hashes at e114 and the endpoint accuracies
- [x] M3: invariance holds; the proposal is decision packet 0003

## What a positive result would license

If the e114 states are bit-identical across host, environment and campaign, then
a campaign is not a source of variation when the recorded scientific identity is
held fixed.  The rule against cross-campaign comparison would then rest on
whether the identities match, which is checkable from the records, rather than on
the directory a run happened to live in.

That is a decision for the human, not a conclusion this plan may draw on its own.

## What it would not license

Nothing about pooling across **environment generations** for results already
committed under generation v1: this tests one 15-epoch continuation, not the
200-epoch runs those results came from.  Nor anything about a different horizon,
endpoint or split.

## Completion report

**A campaign is not a source of variation.  Neither is the host, nor the
environment generation.**

### Training

After fourteen epochs past the learning-rate decay, every checkpoint component
is bit-identical across all three runs:

| component | Hamster/adv vs Hamster/ard-v2 | Hamster vs Ferret |
| --- | --- | --- |
| `model` | identical | identical |
| `optimizer` | identical | identical |
| `rng`, `sample_state`, `sampler_state` | identical | identical |
| `scaler`, `scheduler` | identical | identical |

The checkpoint **file** hashes differ in all three.  That is serialization, and
it is now confirmed twice independently that a file hash is not a
reproducibility test.

### Evaluation

Held-out CE-PGD20 robust accuracy is identical to the third decimal at every
horizon in all three runs: 56.060, 56.940, 57.320 per cent.

The per-sample rows are very nearly identical and not exactly so:

| comparison | samples differing | correctness flips | largest logit-margin difference |
| --- | ---: | ---: | ---: |
| e104, Hamster/adv vs Hamster/ard-v2 | 1 / 5000 | 0 | 1.08e-03 |
| e104, Hamster/adv vs Ferret/ard-v2 | 0 / 5000 | 0 | 0 |
| e109, Hamster/adv vs Hamster/ard-v2 | 0 / 5000 | 0 | 0 |
| e109, Hamster/adv vs Ferret/ard-v2 | 2 / 5000 | 0 | 7.62e-04 |
| e114, both comparisons | 0 / 5000 | 0 | 0 |

Three samples out of thirty thousand measurements, no prediction changed, and
**the pattern does not follow the host or the environment**: at e104 the two
Hamster runs disagree while Hamster and Ferret agree, and at e109 the reverse.
This is run-to-run non-determinism inside the PGD attack — a non-deterministic
reduction on one example — and not a property of either machine.

It is worth stating rather than rounding to "identical", because it sets a hard
resolution limit: the adversarial evaluation is reproducible to about 1e-3 in
logit margin per sample, so any future claim resting on margin differences
smaller than that is resting on noise.  No accuracy claim is affected.

### What this means for the C79 / C79CONF observation

Training is deterministic given the same parent, config, source and seeds.  The
two historical controls therefore cannot have been repetitions of one another;
they must differ in their random stream, which makes them RNG replicates.  Their
gap of 0.94 and 1.78 pp is a pre-decay measurement, and the pre-decay floor for
RNG replicates is independently measured at 1.14 to 1.25 pp.  The observation is
the floor.  Nothing else needs explaining.

### Cost

Two runs of about 35 minutes each on one GPU per host, run concurrently: 1.2
GPU-hours, 40 minutes of wall clock.  The forensic pass that preceded it cost
none.

## Extension, 2026-09-07: it holds for a hundred epochs from scratch too

This plan tested a fifteen-epoch continuation.  The parent regeneration for plan
0093 supplied the stronger test for free.

Seed 1's CropShift parent was retrained **from initialisation for a hundred
epochs** under environment generation v2, four months after the original was
produced under generation v1, and materialised to the same epoch-99 fork point.
Every checkpoint component is identical: `model`, `optimizer`, `rng`,
`sample_state`, `sampler_state`, `scaler`, `scheduler`, at the same
`epoch=99, global_step=35200`.  The one-epoch continuation inside the
materialisation reproduced the historical run's metrics to the last digit,
including `train_loss = 0.10944159670935737`.

Only the checkpoint file hashes differ, for the third independent time.

**This refutes a premise of plan 0093.**  Its section 4 says the six parents must
be regenerated because "the existing dev-1 / dev-2 are of the old environment
generation and are therefore not used".  The two generations produce the same
parent, bit for bit, so that reason does not hold.

The regeneration was not wasted: the same runs continue to epoch 199 and give a
six-seed CropShift baseline, which is the comparator the numeric audit found the
published I100 headline had muddled.  But a future plan should not spend GPU time
on the environment-generation argument again.
