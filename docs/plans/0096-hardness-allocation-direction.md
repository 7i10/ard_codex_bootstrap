# Which samples should get the richer augmentation?

## Status

- Owner: human (approved 2026-09-07 to run), Claude Code (execution)
- Launched 2026-09-07 13:17 JST at source `815dabd`; M2 stalled after p1
- **Parent p1 is complete, all four arms**: `alloc-v1-p1-safe-fork` at
  2026-09-07 06:34 UTC, `alloc-v1-p1-fragile-fork` at 08:51 UTC,
  `alloc-v1-p1-random-fork` — the comparator both treatments are judged against —
  at 11:07 UTC, and `alloc-v1-p1-all-fork`, the `I100` reference, at 13:24 UTC,
  all four at epoch 199. **Nothing of this plan is running now**: the campaign root
  holds arm bundles for p1 only, and every arm of p2-p6 (twenty runs) has not been
  created. See the Progress log.
- **No contrast is evaluated yet.** The endpoint is held-out CE-PGD20 and it has
  not been run for any arm; the preregistered rule is a parent-level paired
  difference over six parents, and one parent exists.
- **The frozen config saves no epoch-114 checkpoint**, so the e114 horizon this
  plan asks for cannot be evaluated on the arms that have already run. See the
  Progress log and decision packet 0006.
- Blocked on: materialising the six environment-v2 fork parents.
  **M0's input is now complete**: all six `epoch-099.pt` checkpoints are on
  Hamster (seeds 1/2/6 trained there, seeds 3/4/5 fetched from Ferret), and
  `materialize_chain.sh` passed its wait at 2026-09-07 07:24 JST. It then called
  the right tool from the wrong tree. The chain runs in the worktree pinned at
  `6ab179d`, but the commit that extended
  `scripts/analysis/materialize_stagewise_parents.py` beyond seeds 1–2 is
  `d7c05d7`, a descendant of it, so the worktree holds a version that rejects
  `--source-root` and `--seed 6`. Seeds 2–6 all exited rc=2 and **only 1 of 6
  parents exists**. The chain has since exited; nothing is waiting. Fix: pin a
  worktree at `d7c05d7` and re-run the chain — `6ab179d..d7c05d7` changes the
  scientific core only by additions whose defaults preserve the old behaviour,
  which is verified file by file in plan 0093's Progress log (2026-09-07, seed6
  entry) together with the exact command. Separately, `parent.sh:27` still
  carries the old path in its completion-skip guard and would now retrain three
  finished seeds over themselves.

## Goal

I100 — weak augmentation to epoch 99, then a richer one from epoch 100 — is this
project's only reproduced result.  It gives every image the same policy.

**This asks which images should get the richer one.**  Two published accounts
answer differently, neither tests it, and the answer is cheap.

## The two accounts

**AROID says hard samples.**  Its policy-learning objective is "Vulnerability,
Affinity and Diversity", where Vulnerability targets the instances most
susceptible to adversarial attack (VERIFIED, arXiv 2306.07197).  The one
published instance-wise augmentation method for adversarial training therefore
sends more augmentation towards the samples the model handles worst.

**IDBH's own account says the opposite.**  Hardness helps only by reducing robust
overfitting and costs clean accuracy in a way that depends on capacity, while
diversity helps unconditionally.  On that reading the budget belongs where there
is margin to spend.  Three instance-wise methods in ordinary training allocate
the same way.

Neither paper tests the direction, and AROID has no random-allocation control.

## Why this is not AROID

AROID learns a policy from a per-instance hardness reward.  This plan does not
learn a policy and does not claim to beat AROID.  It asks a question AROID never
poses: **does the direction of allocation matter, and is a hand-specified
allocation better or worse than giving everyone the same thing?**  A reviewer
will accept that as distinct only if the random-allocation arm is present, so it
is not optional.

## Design

All arms fork from the same epoch-99 parent.  Before epoch 100 nothing differs.
From epoch 100 the late policy is IDBH_WEAK for the images in that arm's mask and
CropShift for the rest.

**Every mask has the same size and the same class composition.**  That is the
whole design.  If the arms differed in how many images they treated, or in which
classes, the comparison would be about dose or about class balance, and neither
is the question.

| arm | receives IDBH_WEAK from e100 | role |
| --- | --- | --- |
| `ALLOC_SAFE` | S1: adversarially correct with margin above the frozen 10th percentile | primary treatment |
| `ALLOC_FRAGILE` | per class, the **lowest-margin** images outside S1 | secondary treatment |
| `ALLOC_RANDOM` | per class, a random draw of the same size | **the comparator for both** |
| `I100` | every image | reference only; see below |

Six parents, treated and control branches paired on a shared `continuation_seed`,
two replicates per parent, to **epoch 199**.

**Judged at e199, recorded at e149 and e114.**  I100's own effect is absent at
e114, gains +0.24 pp by e149 and +0.69 pp by e199
(`docs/ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md:51,55`), so a screen that stops at
e114 cannot see an effect of this shape however precise it is.

Endpoint: held-out CE-PGD20 at all three horizons; official test with AutoAttack
on the primary and its comparator only, seen once.

### `I100` is a reference, not a comparator — corrected 2026-09-07

The first version of this plan judged every allocation arm against `I100`.  That
was wrong, and the commit that introduced it said why in its own message: `I100`
treats 45,000 images and each allocation arm treats 19,459, so "`ALLOC_SAFE` is
0.3 pp below `I100`" is fully explained by treating 57 % fewer images and refutes
nothing.  **The dose-controlled contrasts are the ones between allocation arms**,
and those are what the rule below uses.  `I100` is still run and reported,
because knowing what full-dose costs or buys is worth having, but no directional
claim rests on it.

### `ALLOC_FRAGILE` takes the lowest margins — corrected 2026-09-07

The first version took the complement's **highest**-margin end.  On dev-1 that
excluded the 6,082 most negative-margin images, median margin −1.506: the arm
meant to stand for AROID's vulnerability direction systematically omitted the
most vulnerable quarter of the wrong set, so a null result could not have closed
that direction.  It now takes the lowest-margin images, per class, so that it is
the genuine opposite end of `ALLOC_SAFE`.

## Why the arms are student-only, with the arithmetic

A teacher term was considered and dropped.  At epoch 100 on dev-1, **94.7 % of S1
images are also T1**: conditioning on the teacher inside S1 changes almost
nothing, and a reviewer can compute that from the released state file in a
minute.  The teacher earns its place only on the disagreement set — 23.9 % of S3
is T1 — and that is a separate plan, deliberately gated on this one.

VERIFIED from `prefix/dev-1/training/online-state/epoch-100.parquet` and the
frozen thresholds; the table is in `docs/METHOD_DIRECTIONS_V2.md`.

## Preregistered rule

Effects are parent-level paired differences at e199 on the held-out endpoint.
**Every comparison is between arms that treat the same number of images in the
same class proportions**, so a difference cannot be a dose or a class effect.

The floor for a paired fork at e199 is not yet measured; plan 0093's twelve
controls produce it (that plan's section 8), and the threshold is set from that
number **before** any result here is read.  If plan 0093 has not produced it, the
threshold is 0.25 pp, the widest end of the pre-registered post-decay bracket,
and that choice is recorded here now.

- **Primary.** `ALLOC_SAFE − ALLOC_RANDOM`.  Positive beyond the threshold means
  giving the richer augmentation to samples with margin to spare beats giving it
  to an arbitrary group of the same size and shape, which is what the IDBH
  capacity account predicts.
- **Secondary, same threshold and same analysis.** `ALLOC_FRAGILE − ALLOC_RANDOM`.
  Positive means the vulnerability direction wins instead, which is what AROID's
  objective predicts.
- **The direction itself.** `ALLOC_SAFE − ALLOC_FRAGILE`.  This is the largest
  contrast of the three and the one the two accounts disagree about most
  directly.
- **All three within the threshold of one another** closes per-sample allocation
  in both directions at once.  That is a result and is reported as one.

Neither treatment arm gets a lower bar or a higher one.  The safe-side arm is
called primary because the published evidence points there, not because it is
preferred.

**Attenuation is stated in advance.**  The random draw overlaps `ALLOC_SAFE` by
about 47 % on dev-1, because S1 is 43 % of the training set.  So
`ALLOC_SAFE − ALLOC_RANDOM` carries roughly half the allocation contrast of
`ALLOC_SAFE − ALLOC_FRAGILE`, and a null on the primary with a positive on the
direction contrast is a coherent outcome rather than a contradiction.

**Clean accuracy is reported alongside**, and a drop beyond 0.5 pp disqualifies
an arm regardless of its robust accuracy.

## What has to be built

Every augmentation transform is keyed on the sample id, so restricting a late
policy to a mask leaves the untreated images' views **bit-identical across
arms**: the comparison is exactly paired everywhere it is not treated.  The
fixed-mask machinery and the class-matched random selector already exist.

What is new: a stagewise late policy that consults a frozen mask.  This touches
the scientific core and goes through review before a source SHA is frozen.

## Cost

123 GPU-hours as scoped; 92 if plan 0093's controls are shared rather than
duplicated, which is the intention.

## Milestones

- [x] M0: eighteen masks frozen across six parents, 15,085 to 15,343 images each,
  all three arms of a parent sharing one size and one class composition
- [x] M1: mask-gated late policy implemented, reviewed, four defects and three
  design objections fixed, source frozen at `815dabd`
- [~] M2: four arms of twenty-four finished — `p1` / `ALLOC_SAFE`,
  `p1` / `ALLOC_FRAGILE`, `p1` / `ALLOC_RANDOM` and `p1` / `I100`, which is the
  whole of parent p1; the remaining twenty have not started and nothing is running
- [ ] M3: endpoints at e114, e149, e199; aggregate; close.
  **e114 is not reachable for the finished arms** — the frozen config saves
  checkpoints at 99, 149 and 199 only.

## What this cannot settle

It tests one richer policy against one weaker one at one switch epoch.  A null
does not show that augmentation allocation never matters, only that it does not
matter for this pair at this point in this schedule.

And it says nothing about distillation.  **A result here would be a result about
adversarial training**, which is worth having and is not by itself an argument
for a teacher.  That argument, if it exists, is the disagreement-set plan.

## Progress log

### 2026-09-07 — `alloc-v1-p1-all-fork` finished; parent p1 is complete and nothing is running

Terminal status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p1/all/run-bundle/manifest.json`), not from the
watcher's hint. The hand-run completion triple holds: `run-bundle/completion.json`
declares `completed`, the manifest declares `completed`, and the error marker reads
`no application error recorded`. The fork was created 2026-09-07T11:08:01Z —
fifteen seconds after `ALLOC_RANDOM` released the GPU — and finished 13:24:22Z at
epoch 199, global step 70400, in 2 h 16 min, with one hundred epoch rows covering
100 through 199, the whole post-fork range.

Lineage, read from the manifest and `fork-lineage.json`: source SHA `815dabd`,
worktree `p0096-815dabd20c7b`, parent `parents-v2-cropshift-s1` payload epoch 99
(checkpoint SHA `03feadbb…`, parent config SHA `d4715a2e…`), child config SHA
`b6d87b18…`, fork kind `stagewise_augmentation_fork_v1`, switch epoch 100, prefix
policy `cropshift`, late policy `idbh_weak`, **no mask**. Fixed identity: CIFAR-10 /
`saad_resnet18_cifar_v1` / RSLAD / teacher `chen2021_ltd_wrn34_10` (SHA
`fc398a48…`) / training seed 1 / evaluation-attack seed 0 / linf eps 8-255 step
2-255 / world size 1 / effective global batch 128.
W&B: `single-teacher-ard/runs/alloc-v1-p1-all-fork`.

**This arm is `I100` by construction, and the config proves it.** Its
`resolved_config.yaml` was compared line by line against `ALLOC_SAFE`'s and they
disagree on exactly four lines: `stagewise_late_mask_selected_ids_sha256`
(`null` here, `31912f40…` there), `stagewise_late_mask_selected_count` (`null`
against `15317`), the `tracking.run_id` and the `output_dir`. A null mask means the
late policy applies to every training image, so this run gives all 45,000 images
`IDBH_WEAK` from epoch 100 — the full dose — while everything else, every attack,
schedule, optimizer, seed and normalization field, is identical to the three
allocation arms.

**No result is recorded, because none exists yet.** The numbers below are the
validation-split accuracies the run used to select its own checkpoint. They are
internal validation, not the plan's endpoint.

| checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- |
| best | 197 | 0.8616 | 0.6120 |
| last | 199 | 0.8574 | 0.6092 |

Robust overfit gap 0.28 pp. Each number was read back from the row for its epoch in
`epoch-metrics.jsonl` (rows 98 and 100) and matches the manifest summary.
`train_valid_examples` is 45,000 at epochs 100 and 199, so the run used the full
training split and `resolved_config.yaml:18` `num_samples: 16` is again the inert
display artifact recorded in decision 0003. As on the other three arms, the
manifest's `epoch_metrics_complete: false` (100 recorded against 200 expected) is
correct for a fork that resumes at epoch 100 and must not be read as truncation.
`best_epoch` 197 is inside the post-fork range, as `post_fork_best_scope: true`
requires.

Checkpoints on disk: `epoch-149.pt`, `epoch-199.pt`, `best.pt`, `last.pt`. There is
**no `epoch-114.pt`**, exactly as on the other three, so decision packet 0006 now
describes all four arms of p1 rather than three.

**This arm's validation numbers are higher than the three allocation arms', and
that is not a result.** `I100` is a reference and not a comparator — it treats
45,000 images against their 15,317, so any gap is fully explained by dose, which is
the correction already written into this plan. Beyond that, these are
checkpoint-selection accuracies rather than the held-out CE-PGD20 endpoint, which
has not been run for any arm; the unit of the preregistered rule is a parent-level
paired difference over six parents and one parent exists; and the floor is still
unmeasured, so the fallback threshold of 0.25 pp stands. No contrast is computed and
no direction is claimed.

**Nothing of this plan is running.** Scanning the campaign root finds arm bundles
for `p1/safe`, `p1/fragile`, `p1/random` and `p1/all` and no others, and the chain
did not start a fifth run. The process holding Hamster GPU 1 belongs to
`trades-fix-v1/seed0/eval-aa`, an unrelated AutoAttack evaluation. Materialising
the p2-p6 parents remains the blocker described in the Status block.

### 2026-09-07 — `alloc-v1-p1-random-fork` finished; p1's triple is complete

Terminal status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p1/random/run-bundle/manifest.json`), not from the
watcher's hint. The hand-run completion triple holds: `run-bundle/completion.json`
declares `completed`, the manifest declares `completed`, and the error marker reads
`no application error recorded`. The fork was created 2026-09-07T08:51:32Z —
fifteen seconds after `ALLOC_FRAGILE` released the GPU — and finished 11:07:46Z at
epoch 199, global step 70400, with one hundred epoch rows covering 100 through 199,
the whole post-fork range.

Lineage, read from the manifest and `fork-lineage.json`: source SHA `815dabd`,
worktree `p0096-815dabd20c7b`, parent `parents-v2-cropshift-s1` payload epoch 99
(checkpoint SHA `03feadbb…`, parent config SHA `d4715a2e…`), child config SHA
`01410c4a…`, fork kind `stagewise_augmentation_fork_v1`, switch epoch 100, prefix
policy `cropshift`, late policy `idbh_weak`, mask `masks/p1/random.json`. Fixed
identity: CIFAR-10 / `saad_resnet18_cifar_v1` / RSLAD / teacher
`chen2021_ltd_wrn34_10` (SHA `fc398a48…`) / training seed 1 / evaluation-attack
seed 0 / linf eps 8-255 step 2-255 / world size 1 / effective global batch 128.

**All three arms share one configuration apart from the mask.** The `random` and
`safe` `resolved_config.yaml` files were read in full and disagree on exactly three
lines: the mask digest (`f5af3a8e…` against `31912f40…`), the `tracking.run_id` and
the `output_dir`. The fragile-versus-safe comparison recorded below found the same
three lines, so the property now holds across the whole triple. Every attack,
schedule, optimizer, seed and normalization field is identical.

**The dose-and-composition invariant is now verified for all three arms, not two.**
`masks/p1/manifest.json` gives each of `safe`, `fragile` and `random` the same
`selected_count` of 15,317 and the *same per-class counts* — 1687, 935, 1509, 1421,
1637, 1770, 2152, 1582, 1188, 1436 — all three derived from one epoch-100 state
(`77bb3dd8…`). This is the whole design: the arms cannot differ in how many images
they treat or in which classes, so a difference between them cannot be a dose
effect or a class-balance effect. The `random` mask's `selected_ids_sha256`
`f5af3a8e…` is the digest the run's config declares, so the mask the run used is the
mask the manifest describes.

**No result is recorded, because none exists yet.** The numbers below are the
validation-split accuracies the run used to select its own checkpoint. They are
internal validation, not the plan's endpoint.

| checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- |
| best | 191 | 0.8596 | 0.6006 |
| last | 199 | 0.8610 | 0.5970 |

Robust overfit gap 0.36 pp. Each number was read back from the row for its epoch in
`epoch-metrics.jsonl` (rows 92 and 100) and matches the manifest summary.
`train_valid_examples` is 45,000 at epoch 199, so the run used the full training
split and `resolved_config.yaml:18` `num_samples: 16` is again the inert display
artifact recorded in decision 0003. As on the other two arms, the manifest's
`epoch_metrics_complete: false` (100 recorded against 200 expected) is correct for a
fork that resumes at epoch 100 and must not be read as truncation.

Checkpoints on disk: `epoch-149.pt`, `epoch-199.pt`, `best.pt`, `last.pt`. There is
**no `epoch-114.pt`**, exactly as on `safe` and `fragile`, so decision packet 0006
now describes all three arms of p1 rather than two.

**What this changes, and what it does not.** The comparator exists, so for the first
time all three preregistered contrasts are computable in principle. None is computed
here, for two independent reasons. First, the endpoint is held-out CE-PGD20 and *no
endpoint evaluation has been run for any arm* — the table above is validation, and
substituting it would change the measurement. Second, the rule is a parent-level
paired difference over six parents, and one parent exists. The floor is also still
unmeasured, so the fallback threshold of 0.25 pp stands. No contrast is reported and
no direction is claimed.

A fourth run has started on this parent: `alloc-v1-p1-all-fork`, the `I100`
reference that treats all 45,000 images, at epoch 100 as of 11:09:31Z. It is a
reference, not a comparator, and no directional claim will rest on it.

### 2026-09-07 — `alloc-v1-p1-fragile-fork` finished at epoch 199

Terminal status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p1/fragile/run-bundle/manifest.json`), not from
the watcher's hint. The hand-run completion triple holds: `run-bundle/completion.json`
declares `completed`, the manifest declares `completed`, and `error-marker.txt`
reads `no application error recorded`. Started 2026-09-07T06:35:05Z — fifteen
seconds after `ALLOC_SAFE` released the GPU — and finished 08:51:17Z at epoch 199,
global step 70400. One hundred epoch rows, 100 through 199, which is the whole
post-fork range.

Lineage, all read from the manifest and `fork-lineage.json`: source SHA `815dabd`,
worktree `p0096-815dabd20c7b`, parent `parents-v2-cropshift-s1` payload epoch 99
(checkpoint SHA `03feadbb…`, parent config SHA `d4715a2e…`), child config SHA
`cfc5147a…`, fork kind `stagewise_augmentation_fork_v1`, switch epoch 100,
prefix policy `cropshift`, late policy `idbh_weak`. Fixed identity: CIFAR-10 /
`saad_resnet18_cifar_v1` / RSLAD / teacher `chen2021_ltd_wrn34_10`
(SHA `fc398a48…`) / training seed 1 / evaluation-attack seed 0 / linf eps 8-255
step 2-255 / world size 1 / effective global batch 128.

**The two finished arms differ in the mask and in nothing else.** Their
`resolved_config.yaml` files were compared line by line and disagree on exactly
three lines: the mask digest
(`31912f40…` for `ALLOC_SAFE`, `8cc7f286…` for `ALLOC_FRAGILE`), the
`tracking.run_id`, and the `output_dir`. `stagewise_late_mask_selected_count`
is 15,317 on both, which is p1's mask size, so the dose is equal as the design
requires. Every attack, schedule, optimizer, seed and normalization field is
identical.

**No result is recorded, because none exists yet.** The numbers below are the
validation-split accuracies the run used to select its own checkpoint. They are
internal validation, not the plan's endpoint: the endpoint is held-out CE-PGD20,
and no endpoint evaluation has been run for any arm.

| checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- |
| best | 188 | 0.8592 | 0.6006 |
| last | 199 | 0.8602 | 0.5980 |

Robust overfit gap 0.26 pp. Each of these four numbers was read back from the row
for its epoch in `epoch-metrics.jsonl` and matches the manifest summary.
`train_valid_examples` is 45,000 at both epoch 100 and epoch 199, so the run used
the full training split and the `num_samples: 16` line in `resolved_config.yaml:18`
is the same inert display artifact recorded in decision 0003.

**This still closes nothing.** Every preregistered contrast is a difference
between arms on the same parent, and `ALLOC_RANDOM` — the comparator for both
the primary and the secondary — is still running. Comparing the two finished
arms to each other would be the `ALLOC_SAFE − ALLOC_FRAGILE` direction contrast,
but on validation accuracy rather than the held-out endpoint, on one parent
rather than six, with no measured floor, so it is not evaluated here and no
such number is reported.

#### The frozen config cannot produce the e114 endpoint

`resolved_config.yaml:130-133` sets `checkpoint_epochs: [99, 149, 199]`. The
design section of this plan asks for the held-out CE-PGD20 endpoint at e114,
e149 and e199. Both finished arms hold `epoch-149.pt`, `epoch-199.pt`, `best.pt`
and `last.pt` and **no `epoch-114.pt`**; the weights that existed at e114 are
gone, and recovering them would mean retraining the arm.

This does not touch the primary judgment, which is at e199, and e149 survives.
What is lost is the earliest of the three horizons — the one the plan included
to show the shape of the effect over time, by analogy with I100 being absent at
e114 and present by e149. The choice is whether the twenty-two arms that have
not started should be frozen with `114` added to `checkpoint_epochs`, which
would leave p1's pair on a different checkpoint schedule from every other
parent. That is a scientific decision and is written up as decision packet 0006.

### 2026-09-07 — `alloc-v1-p1-safe-fork` finished at epoch 199

Terminal status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p1/safe/run-bundle/manifest.json`), not from the
watcher's hint. The hand-run completion triple holds: `run-bundle/completion.json`
is present, the manifest declares `completed`, and `error-marker.txt` reads
`no application error recorded`. Finished 2026-09-07T06:34:50Z at epoch 199,
global step 70400.

Lineage, all read from the manifest: source SHA `815dabd`, worktree
`p0096-815dabd20c7b`, parent `parents-v2-cropshift-s1` payload epoch 99
(checkpoint SHA `03feadbb…`, parent config SHA `d4715a2e…`), child config SHA
`1c6d9696…`, fork kind `stagewise_augmentation_fork_v1`, switch epoch 100,
prefix policy `cropshift`, late policy `idbh_weak`, mask `masks/p1/safe.json`.
Fixed identity: CIFAR-10 / `saad_resnet18_cifar_v1` / RSLAD / teacher
`chen2021_ltd_wrn34_10` (SHA `fc398a48…`) / training seed 1 / evaluation-attack
seed 0 / linf eps 8-255 / world size 1 / effective global batch 128.

**No result is recorded, because none exists yet.** The numbers below are the
validation-split accuracies the run used to select its own checkpoint. They are
internal validation, not the plan's endpoint: the endpoint is held-out CE-PGD20
at e114, e149 and e199, and no endpoint evaluation has been run for any arm.

| checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- |
| best | 178 | 0.8618 | 0.6004 |
| last | 199 | 0.8640 | 0.5968 |

Robust overfit gap 0.36 pp. A single arm has no comparator, so this closes
nothing: every preregistered contrast is a difference between arms on the same
parent, and the comparator arm `ALLOC_RANDOM` does not exist yet.

Two things to carry forward rather than treat as defects:

1. The manifest summary says `epoch_metrics_complete: false`, with 100 recorded
   epochs against 200 expected. That is correct for a fork that resumes at
   epoch 100 and holds rows for epochs 100-199 only. The aggregator must not
   read it as a truncated run.
2. The six `I100_ONLINE_PREFIX` runs under `prefix/p1`-`p6` all sit at epoch 100
   with `completion.json` absent, so the watcher counts them non-terminal. They
   are the finished fork parents; their bundles were never closed out.

### 2026-09-07 — what actually exists versus what the Status block claimed

Scanning the whole campaign root found arm bundles for `p1/safe` and
`p1/fragile` only. The earlier Status line ("four arms on parents p1-p4 running
on four GPUs; p5 and p6 queued") did not match disk, and has been corrected.

`alloc-v1-p1-fragile-fork` is alive, resuming from its own `last.pt`, and had
written no progress row at the time of this check.

### 2026-09-07 — relation to decision packet 0004

Decision 0004 describes a run-ID collision on this same arm and asks whether to
resume the first attempt or retake it under a new run ID; its `chosen` is still
`null`. The successful run recorded above uses run ID `alloc-v1-p1-safe-fork`,
not the colliding `alloc-p1-safe-fork`, and its manifest was created at
04:17:58Z — about four minutes after the failed attempt and roughly two hours
before the packet was written. The retake had therefore already happened when
0004 was authored, so the packet's question is settled in fact but not on
record, and the orphan W&B run holding only epochs 100-101 that 0004 listed as
the cost of that route now exists. The preflight gate 0004 recommends is
untouched and remains open. Resolving 0004 is the human's call.

## Execution notes

The six parents are registered as `p1` through `p6` in the online-state runner
alongside `dev-1` and `dev-2`, which other analyses cite by name.  Each arm is
built in two steps: the fork builder rewrites the parent checkpoint's config hash
to a stage-wise config carrying that arm's mask identity, so the four arms are
four distinct recorded objects rather than relabelled copies of one; training
then resumes from that fork with the mask file, which is checked against the
declared digest, count and training partition before the first image is
transformed.

Mask sizes per parent: p1 15,317; p2 15,145; p3 15,253; p4 15,324; p5 15,085;
p6 15,343.  They differ because each parent's own epoch-99 state decides its
split, and within a parent all three arms are identical in size and class
composition, which is what the comparison requires.
