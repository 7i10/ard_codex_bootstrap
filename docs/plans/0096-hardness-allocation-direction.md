# Which samples should get the richer augmentation?

## Status

- Owner: human (approved 2026-09-07 to run), Claude Code (execution)
- Launched 2026-09-07 13:17 JST at source `815dabd`
- **The M0 blocker is gone.** All six fork parents are materialised — `prefix/p1`
  through `prefix/p6` each hold `checkpoints/epoch-100.pt` and their epoch-100
  online state — and all six mask sets are frozen (`masks/p1`-`masks/p6`, each
  with its `manifest.json`). The earlier wrong-worktree failure described here is
  resolved and the text has been removed.
- **All 24 judged arms are at epoch 199.** Every cell of the 6 x 4 design
  finished between 2026-09-07 06:34 UTC and 2026-09-08 00:00 UTC — p1, p5 and p6
  on Hamster, p2, p3 and p4 on Ferret. Training for the judged design is done.
- **The paired noise floor is running: six `ALLOC_RANDOMB` arms, one per parent.**
  Launched 2026-09-08 after decision packet 0008 was answered B. `p1` finished
  2026-09-08 06:03 UTC and `p5` at 06:06 UTC; both are written up below. `p6`
  started on the freed Hamster GPU at 06:03 UTC; `p2`, `p3` and `p4` are still
  running on Ferret. Two of six draws are in.
- **No contrast is evaluated yet.** The endpoint is held-out CE-PGD20 and it has
  not been run for a single arm, judged or floor. The preregistered rule is a
  parent-level paired difference at e199, and nothing on that split exists.
- **The frozen config saves no epoch-114 checkpoint**, so the e114 horizon this
  plan asks for cannot be evaluated on any arm that has run. See the Progress log
  and decision packet 0006; the gap now covers the twenty-four judged arms and
  extends to the floor arms, which share the same frozen config.
- **p6's three dose-matched arms spread 0.64 pp at the last checkpoint**, against
  0.12 pp on p1 and 0.10 pp on p5. Whatever this spread is, it is a property of
  the fork and not of the allocation, and it is the same size as the effect this
  plan is trying to read. Decision packet 0008 carries the number.
- **The first two floor pairs differ by 0.28 pp (p1) and 0.36 pp (p5) at the last
  checkpoint** — two independent random draws of the same images per parent, which
  the design says are interchangeable. Both are wider than their parent's whole
  three-arm allocation spread and wider than the 0.25 pp fallback threshold. It is
  the selection split, not the endpoint, and two parents are not six, so it fixes
  nothing; see the Progress log.
- **The same jitter is visible inside single runs**: consecutive late epochs of one
  run move validation PGD by 0.32 to 0.52 pp. That evidence needs no floor arm and
  was already present in the twenty-four judged arms.
- **The headless postrun did not run for six of these arms.** Every automatic
  postrun between 16:52 and 21:35 UTC on 2026-09-07 exited on the first turn with
  HTTP 429 "You've hit your session limit", so p5 `safe`/`fragile`/`random` and
  p6 `safe`/`fragile`/`random` reached a terminal state with nothing written. The
  two `all` arms did get a postrun; both quartets are written up below.

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
- [x] M2: all twenty-four judged arms finished at epoch 199 — six complete
  quartets, `p1`/`p5`/`p6` on Hamster and `p2`/`p3`/`p4` on Ferret
- [~] M4: paired noise floor, six `ALLOC_RANDOMB` arms, added by decision packet
  0008's answer B. `p1` and `p5` at epoch 199; `p2`, `p3`, `p4` and `p6` running.
  The floor itself is not measurable until the endpoint runs on these arms too
- [ ] M3: endpoints at e114, e149, e199; aggregate; close.
  **e114 is not reachable for any arm** — the frozen config saves checkpoints at
  99, 149 and 199 only. The floor must be computed and the threshold written down
  before the judged contrasts are read, which packet 0008 requires and a single
  aggregation pass over all thirty arms would silently break.

## What this cannot settle

It tests one richer policy against one weaker one at one switch epoch.  A null
does not show that augmentation allocation never matters, only that it does not
matter for this pair at this point in this schedule.

And it says nothing about distillation.  **A result here would be a result about
adversarial training**, which is worth having and is not by itself an argument
for a teacher.  That argument, if it exists, is the disagreement-set plan.

## Progress log

### 2026-09-08 — the second floor pair agrees with the first: p5 moves 0.36 pp, and the same jitter is visible inside a single run

`alloc-v1-p5-randomb-fork` is the second of the six floor arms to finish. Terminal
status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p5/randomb/run-bundle/manifest.json`), not from the
watcher's hint. The completion triple holds: `run-bundle/completion.json` declares
`completed`, the manifest declares `completed`, and the error marker reads
`no application error recorded`. Created 2026-09-08T03:46:25Z, finished 06:06:16Z —
2 h 20 min — at epoch 199, global step 70400, with one hundred epoch rows covering
100 through 199, the whole post-fork range.

#### The number

| arm | checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- | --- |
| `ALLOC_RANDOM` | best | 197 | 0.8610 | 0.6018 |
| `ALLOC_RANDOM` | last | 199 | 0.8614 | 0.5966 |
| `ALLOC_RANDOMB` | best | 195 | 0.8656 | 0.6010 |
| `ALLOC_RANDOMB` | last | 199 | 0.8600 | 0.6002 |

**At the last checkpoint the two p5 draws differ by 0.36 pp.** At the best checkpoint
they differ by −0.08 pp. Robust overfit gaps are 0.52 pp on `random` and 0.08 pp on
`randomb`. Clean accuracy moves −0.14 pp between the two at e199, well inside the
0.5 pp disqualification bar, so nothing here is a clean-accuracy artefact.

**Two of six floor draws are now in, and they agree.**

| parent | `RANDOMB − RANDOM`, last (e199) | same, best checkpoint |
| --- | ---: | ---: |
| p1 | +0.28 pp | +0.02 pp |
| p5 | +0.36 pp | −0.08 pp |

Both last-checkpoint figures exceed the 0.25 pp fallback threshold, and both exceed
in absolute value every one of the plan's three preregistered contrasts averaged over
six parents (+0.050, −0.117, +0.167 pp). Two arms that differ in nothing a reviewer
could name — same dose, same class composition, same stream, same host, same source
pin, same interpreter, only a different arbitrary set of 15,085 images — move further
apart than any allocation direction does.

**Both draws came out positive, and that carries no information yet.** With two
paired blocks, matching signs happen half the time under a pure null. Only the six
fix a threshold, and packet 0008's ordering constraint stands until they do.

**The best-checkpoint column is still not a second, smaller estimate**, and p5 shows
why more sharply than p1 did: the sign flips (+0.02 on p1, −0.08 on p5) while the
last-checkpoint column stays consistent. Best-checkpoint selection maximises
validation PGD and the column then reports validation PGD, so it is selection on the
quantity being read and compresses differences toward zero by construction.

#### A second, independent line of evidence: the jitter is inside each run

This does not depend on the floor arms at all. Read consecutive epochs near the end
of a *single* run and validation PGD moves as much as the contrast does:

| run | epochs | val PGD | range |
| --- | --- | --- | ---: |
| `alloc-v1-p5-random-fork` | 196-199 | 0.6002, 0.6018, 0.5992, 0.5966 | 0.52 pp |
| `alloc-v1-p5-randomb-fork` | 195-199 | 0.6010, 0.5978, 0.6004, 0.5996, 0.6002 | 0.32 pp |
| `alloc-v1-p1-randomb-fork` | 196-199 | 0.5958, 0.5994, 0.6006, 0.5998 | 0.48 pp |
| `alloc-v1-p1-random-fork` | 190-196 | 0.5992, 0.6006, 0.5988, 0.5966, 0.6006, 0.5984, 0.5990 | 0.40 pp |

A single-epoch e199 reading therefore carries jitter of the same size as everything
this plan is trying to measure, whichever images the mask selected. The floor arms
and this observation are two different arguments for the same conclusion, and the
second one was already available in the twenty-four judged arms before any floor arm
was launched. It also suggests a cheap mitigation the endpoint could use — averaging
the endpoint over a few late checkpoints rather than reading e199 alone — but that
changes a preregistered measurement and is the human's call, not this postrun's.

#### Verification

Lineage, read from the manifest: source SHA `815dabd20c7b12ebfd7915dc803bad0a16756225`
— the same pin the judged arms used, `dirty: false` with an empty diff — worktree
`p0096-815dabd20c7b`, parent `parents-v2-cropshift-s5` payload epoch 99, parent
checkpoint SHA `d8c0ce35…`, parent config SHA `26cef228…`, child config SHA
`8d18af12…`, fork kind `stagewise_augmentation_fork_v1`, switch epoch 100, prefix
policy `cropshift`, late policy `idbh_weak`, `post_fork_best_scope: true`. Fixed
identity: CIFAR-10 / `saad_resnet18_cifar_v1` / RSLAD / teacher
`chen2021_ltd_wrn34_10` (SHA `fc398a48…`) / training seed 5 / evaluation-attack seed
0 / split seed 20260722 / linf eps 8-255 step 2-255 / world size 1 / effective global
batch 128 / local per-rank BN.
W&B: `single-teacher-ard/runs/alloc-v1-p5-randomb-fork`.

**The config differs from `ALLOC_RANDOM`'s on exactly four lines**, compared line by
line over all 199: `stagewise_late_mask_selected_ids_sha256` (`d1a1b612…` here,
`d06033ec…` there), `tracking.run_id`, `output_dir`, and the teacher checkpoint *path
spelling* — `/home/islab/workspace-local/shunsuke.naito/…` against
`/home/shunsukenaito/workspace-local/…`, two spellings of the same file through the
symlink, with an identical `checkpoint_sha256`. This is the same four-line result the
p1 pair gave, including the symlink line, so it is a property of how the two batches
were launched and not of either parent. `stagewise_late_mask_selected_count` is
15,085 on both, and every seed, attack, schedule, optimizer, normalization,
temperature and batch-size field is identical.

**The mask is a genuine second draw of the same shape.** `masks/p5/randomb.json`
carries `random_seed: 2026090802` against `random.json`'s `2026090801`, the same
`state_sha256` `734c0810…` (one epoch-100 state), the same `switch_epoch` 100 and the
same source string `stagewise_allocation_matched_random_epoch100_v1`. Its per-class
counts are identical to all three frozen p5 masks — 1668, 943, 1468, 1394, 1553,
1780, 2140, 1531, 1167, 1441, summing to 15,085 — and its `selected_ids_sha256`
`d1a1b612…` is the digest the run's `resolved_config.yaml` declares.

Both declared artifacts exist (`epoch-metrics.parquet`, `sample-stats-train.parquet`).
All four validation numbers above were read back from the rows for epoch 195 and
epoch 199 in `epoch-metrics.jsonl` and match the manifest summary.
`train_valid_examples` is 45,000 at epochs 100, 195 and 199.
`epoch_metrics_complete: false` (100 recorded against 200 expected) is again correct
for a fork that resumes at epoch 100. Checkpoints on disk: `epoch-149.pt`,
`epoch-199.pt`, `best.pt`, `last.pt` — and no `epoch-114.pt`, as everywhere else.

**The dose signature holds on the floor pair.** Training-set accuracy at epoch 199 is
0.8173 on `randomb` against 0.8159 on `random`, 0.14 pp apart, as two equal-dose
masks should be, and far above the 0.683 the full-dose `all` arm reaches. The floor
arm is treating the number of images it claims to treat.

The two exported parquet digests (`a756ccd8…` and `b5f0ae56…`) were read and **not**
recomputed: this session's sandbox refuses to hash, diff or list files outside the
repository checkout, as on every postrun in this campaign. That is a harness
restriction, not a finding about the run.

#### The automatic postrun wrote nothing again, and this time the cause is the trust dialog alone

Both `randomb` postruns that have fired — `20260908T060344Z-alloc-v1-p1-randomb-fork`
and `20260908T060646Z-alloc-v1-p5-randomb-fork` — produced a 320-byte log holding only
`Ignoring 44 permissions.allow entries from .claude/settings.json: this workspace has
not been trusted` and a **zero-byte** result JSON. `~/.claude.json` line 938 confirms
it: the entry for `/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap` has
`hasTrustDialogAccepted: false`, and that is the only spelling registered. Both arms
were written up from later sessions instead. This is the condition `CLAUDE.md` already
documents, now observed directly, and it is *not* the HTTP 429 quota failure of
2026-09-07 — the p6 `all` postrun at 23:56Z that night wrote 11 KB, so the trust state
changed between then and now. One interactive session in this checkout fixes it.

#### What is running

`alloc-v1-p6-randomb-fork` is at epoch 102 on Hamster gpu0, started fourteen seconds
after p1 released it; Hamster gpu1 is now idle. All three Ferret GPUs remain occupied
by `p2`, `p3` and `p4`, whose bundles are not on this filesystem and were not
inspected. No record, report or evidence-ledger row is written, because no endpoint
contrast exists to write, and no decision packet is opened: packet 0008 already fixes
what happens next, and there is nothing to decide until all six floor arms land.

### 2026-09-08 — the first floor pair is in: two interchangeable random draws on p1 differ by 0.28 pp at the judged checkpoint

`alloc-v1-p1-randomb-fork` is the first of the six floor arms to finish. Terminal
status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p1/randomb/run-bundle/manifest.json`), not from the
watcher's hint. The completion triple holds: `run-bundle/completion.json` declares
`completed`, the manifest declares `completed`, and the error marker reads
`no application error recorded`. Created 2026-09-08T03:46:25Z, finished 06:03:28Z —
2 h 17 min — at epoch 199, global step 70400, with one hundred epoch rows covering
100 through 199, the whole post-fork range.

**What this arm is for.** It is a *second matched-random draw* on parent p1: the
same 15,317 images in the same per-class proportions, drawn independently of
`ALLOC_RANDOM`, everything else held. By the plan's own design the two are
interchangeable — neither is a treatment — so the difference between them is
noise and nothing else. Decision packet 0008 fixes the threshold from the largest
such difference across the six parents, and fixes it **before** the judged
contrasts are read.

#### The number

| arm | checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- | --- |
| `ALLOC_RANDOM` | best | 191 | 0.8596 | 0.6006 |
| `ALLOC_RANDOM` | last | 199 | 0.8610 | 0.5970 |
| `ALLOC_RANDOMB` | best | 193 | 0.8612 | 0.6008 |
| `ALLOC_RANDOMB` | last | 199 | 0.8600 | 0.5998 |

**At the last checkpoint the two draws differ by 0.28 pp.** At the best checkpoint
they differ by 0.02 pp. The preregistered rule judges at e199, which is the last
checkpoint, so 0.28 pp is the figure on the rule's own checkpoint. The best-epoch
figure is not a second opinion: best-checkpoint selection takes a maximum over a
hundred epochs and compresses differences by construction, which is why the plan
keeps best and last separate rather than choosing between them.

Three comparisons put 0.28 pp in scale. It is wider than p1's entire three-arm
allocation spread (0.12 pp). It is wider than the 0.25 pp fallback threshold. And
it is larger in absolute value than every one of the plan's three preregistered
contrasts averaged over six parents (+0.050, −0.117, +0.167 pp).

**This does not fix the threshold and no threshold is being changed here.** These
are selection-split accuracies — the split each run used to pick its own
checkpoint — and the plan's floor, like its judgment, is defined on the held-out
CE-PGD20 endpoint, which has not been run for any arm. One parent is also not six.
What the number does is give the first direct evidence on the question packet 0008
was written to answer, and it points the same way p6's 0.64 pp spread did.

#### Verification

Lineage, read from the manifest: source SHA `815dabd20c7b12ebfd7915dc803bad0a16756225`
(the same pin the judged arms used), worktree `p0096-815dabd20c7b`, clean at that
SHA — `dirty: false` and an empty diff. Parent `parents-v2-cropshift-s1` payload
epoch 99, parent checkpoint SHA `03feadbb…`, parent config SHA `d4715a2e…`, child
config SHA `dd0e3dbe…`, fork checkpoint SHA `df1dd340…`, fork kind
`stagewise_augmentation_fork_v1`, switch epoch 100, prefix policy `cropshift`, late
policy `idbh_weak`, `post_fork_best_scope: true`. Fixed identity: CIFAR-10 /
`saad_resnet18_cifar_v1` / RSLAD / teacher `chen2021_ltd_wrn34_10` (SHA `fc398a48…`)
/ training seed 1 / evaluation-attack seed 0 / split seed 20260722 / linf eps 8-255
step 2-255 / world size 1 / effective global batch 128 / local per-rank BN.
W&B: `single-teacher-ard/runs/alloc-v1-p1-randomb-fork`.

**The config differs from `ALLOC_RANDOM`'s on exactly four lines**, compared line by
line: `stagewise_late_mask_selected_ids_sha256` (`eabc169a…` here, `f5af3a8e…`
there), `tracking.run_id`, `output_dir`, and the teacher checkpoint *path spelling*
— `/home/islab/workspace-local/shunsuke.naito/…` against
`/home/shunsukenaito/workspace-local/…`, two spellings of the same file through the
symlink, with an identical `checkpoint_sha256`. `stagewise_late_mask_selected_count`
is 15,317 on both, and every seed, attack, schedule, optimizer, normalization,
temperature and batch-size field is identical. This is the check that matters most
for a floor arm: the mask identity is the only scientific difference between the
two runs.

**The mask is a genuine second draw of the same shape.** `masks/p1/randomb.json`
carries `random_seed: 2026090802` against `random.json`'s `2026090801`, the same
`state_sha256` `77bb3dd8…` (one epoch-100 state), the same `switch_epoch` 100 and
the same source string `stagewise_allocation_matched_random_epoch100_v1`. Its
per-class counts are identical to all three frozen p1 masks — 1687, 935, 1509,
1421, 1637, 1770, 2152, 1582, 1188, 1436, summing to 15,317 — its `selected_ids`
array holds exactly 15,317 entries, and its `selected_ids_sha256` `eabc169a…` is
the digest the run's `resolved_config.yaml` declares. So the run used the mask the
file describes, and that mask matches `ALLOC_RANDOM` in dose and composition while
differing in identity, which is the whole design of the floor.

Both declared artifacts exist (`epoch-metrics.parquet`, `sample-stats-train.parquet`).
All four validation numbers above were read back from the rows for epoch 193 and
epoch 199 in `epoch-metrics.jsonl` and match the manifest summary; 0.6008 at epoch
193 is the unique maximum over the post-fork range, so `best_epoch` is correct and
inside that range. `train_valid_examples` is 45,000 at epochs 100, 193 and 199.
`epoch_metrics_complete: false` (100 recorded against 200 expected) is again correct
for a fork that resumes at epoch 100 and must not be read as truncation.
Checkpoints on disk: `epoch-149.pt`, `epoch-199.pt`, `best.pt`, `last.pt` — and no
`epoch-114.pt`, as everywhere else in this plan.

The run stayed on the host and interpreter the floor requires: host
`islab-WS-C621E-SAGE-Series` (Hamster), the same host `ALLOC_RANDOM` ran on, and
`/home/shunsukenaito/.conda/envs/ard-v2/bin/python`, the same interpreter — so the
`adv` false start recorded below did not survive into the measurement.

#### Two things to fix later, neither of them a defect in this run

**The floor masks have no manifest.** The three judged masks are covered by
`masks/p1/manifest.json` and its `.sha256`; `randomb.json` sits beside them with no
manifest entry and no digest file. Its provenance is self-describing and was
checked here, but it is not hash-bound the way the frozen masks are. The same holds
for the other five parents.

**The two exported parquet digests were read, not recomputed.** This session's
sandbox refuses to hash files outside the repository checkout, exactly as on the p5
and p6 postruns. That is a harness restriction, not a finding about the run.

#### What is running

`alloc-v1-p5-randomb-fork` reached epoch 199 at 06:06:16Z, three minutes after this
one, and awaits its own postrun. `alloc-v1-p6-randomb-fork` started at 06:03:43Z —
fourteen seconds after p1 released the GPU — and is at epoch 102. All three Ferret
GPUs are still occupied by `p2`, `p3` and `p4`. No record, report or evidence-ledger
row is written, because no endpoint contrast exists to write.

### 2026-09-08 — the paired noise floor is running: six `ALLOC_RANDOMB` arms, one per parent

Decision packet 0008 was answered **B**: measure the floor and fix the threshold
before the judgment endpoint is read. The packet also disambiguated what "random
against random" means, because the two readings give different floors and the
wrong one biases this plan's fourth branch.

The floor arm is a **second matched-random draw**: the same 15,085 to 15,343
images per parent in the same per-class proportions, drawn independently, with
`continuation_seed` unchanged. It is not a re-run of the same mask under a new
RNG seed. The four judged arms all share one stream and differ only in which
images they treat, so the quantity that has to be bounded is the effect of the
*arbitrary identity of the selected set*, not the effect of reseeding.

Using the reseeded floor instead would inflate the threshold, and this plan's
fourth branch -- all three contrasts within the threshold closes per-sample
allocation -- is a no-effect claim, which an inflated threshold makes too easy to
declare. The error would run against the conclusion, not toward it.

**Preconditions, checked before launch and all met.**

| parent | images | class counts match | ids differ | overlap | Jaccard | `safe`/`fragile` reproduce |
| --- | ---: | :--: | :--: | ---: | ---: | :--: |
| p1 | 15,317 | yes | yes | 5,480 | 0.2179 | identical |
| p2 | 15,145 | yes | yes | 5,332 | 0.2136 | identical |
| p3 | 15,253 | yes | yes | 5,425 | 0.2163 | identical |
| p4 | 15,324 | yes | yes | 5,483 | 0.2179 | identical |
| p5 | 15,085 | yes | yes | 5,272 | 0.2117 | identical |
| p6 | 15,343 | yes | yes | 5,461 | 0.2165 | identical |

The Jaccard values sit at the 0.205 two independent 34 % draws predict. The
`safe` and `fragile` masks rebuilt from the same worktree are byte-identical to
the frozen ones, which confirms the builder is deterministic and that only the
random draw moved -- the one thing `--random-seed 2026090802` was supposed to
change.

**Three things are held fixed so they cannot enter the floor.** The source is the
same pin the judged arms used (`815dabd20c7b`, both worktrees verified clean at
that SHA). The interpreter is the same environment generation (`ard-v2`). And
each parent's `randomb` runs **on the same host as its own `random`** -- p1, p5,
p6 on Hamster and p2, p3, p4 on Ferret -- so host cannot appear in the contrast
that exists to measure noise.

Placement: Hamster gpu0 p1 then p6, gpu1 p5; Ferret gpu0 p2, gpu1 p3, gpu2 p4.
About 5.2 hours of wall clock.

#### Two false starts, both caught before any epoch mattered

The first launch used the `adv` interpreter while every judged arm ran under
`ard-v2`. The two report an identical recorded identity -- Python 3.11.15, torch
2.11.0+cu128, CUDA 12.8, matching the run bundle exactly -- and plan 0094 showed
environment generation is inert. It was stopped anyway after about one minute,
because a noise floor is the one measurement that must not carry an untested
difference, and the cost of removing the doubt was two minutes.

The relaunch then had to avoid the W&B run-ID collision that has already cost
this project GPU time three times, since `alloc-v1-p1-random2` and
`alloc-v1-p3-random2` had been claimed by the aborted attempt. The arm was
renamed `randomb` for all six parents rather than reusing the burnt identifiers.
Two orphaned one-minute W&B runs remain under the `random2` names and belong to
nothing; they are recorded here rather than left unexplained.

The judgment endpoint for the 24 completed arms is **not** being run yet. Packet
0008 states the ordering constraint: the threshold is fixed from this floor, or
it stays at the preregistered 0.25 pp, and reading the endpoint first would
discard the preregistration either way.


### 2026-09-08 — all 24 arms reached epoch 199; training is complete and the endpoint evaluation has still never run

`alloc-v1-p5-all-fork` and `alloc-v1-p6-all-fork` finished on Hamster; p2, p3 and
p4 finished all four arms on Ferret. **Every cell of the 6 x 4 design is at epoch
199.** Nothing is running. About 28 to 31 GPU-hours of training are on disk.

**The judgment cannot be made from what this table shows.** The preregistered
metric is the held-out endpoint, and the endpoint evaluation has not been run for
a single arm. What follows is the *selection* split — the same split each run used
to pick its own best checkpoint — read off the last line of each
`epoch-metrics.jsonl`. It is recorded here because it is what the campaign
produced, and it is labelled so that it is never mistaken for the verdict.

| parent | `ALLOC_SAFE` | `ALLOC_FRAGILE` | `ALLOC_RANDOM` | `I100` (`all`) |
| --- | ---: | ---: | ---: | ---: |
| p1 | 59.68 / 86.40 | 59.80 / 86.02 | 59.70 / 86.10 | 60.92 / 85.74 |
| p2 | 59.82 / 86.58 | 59.54 / 85.92 | 60.16 / 86.00 | 60.68 / 86.40 |
| p3 | 60.16 / 86.14 | 59.86 / 85.76 | 59.78 / 86.18 | 60.82 / 86.10 |
| p4 | 59.44 / 86.68 | 59.60 / 85.94 | 59.48 / 86.52 | 60.22 / 86.68 |
| p5 | 59.62 / 86.12 | 59.56 / 85.56 | 59.66 / 86.14 | 60.50 / 85.80 |
| p6 | 60.46 / 86.64 | 59.82 / 86.04 | 60.10 / 86.16 | 60.84 / 86.16 |

Cells are selection-split PGD / clean, per cent, at epoch 199.

Parent-level paired differences, the same arithmetic the preregistered rule
specifies but on the wrong split:

| contrast | p1 | p2 | p3 | p4 | p5 | p6 | mean | SD | SE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SAFE - RANDOM` (primary) | -0.02 | -0.34 | +0.38 | -0.04 | -0.04 | +0.36 | **+0.050** | 0.275 | 0.112 |
| `FRAGILE - RANDOM` (secondary) | +0.10 | -0.62 | +0.08 | +0.12 | -0.10 | -0.28 | **-0.117** | 0.290 | 0.119 |
| `SAFE - FRAGILE` (direction) | -0.12 | +0.28 | +0.30 | -0.16 | +0.06 | +0.64 | **+0.167** | 0.302 | 0.123 |

Three things are worth writing down before the endpoint runs, and none of them is
the verdict.

**All three contrasts sit inside the 0.25 pp fallback threshold, and inside their
own standard errors.** With k = 6 paired blocks and SE about 0.115 pp, this design
resolves about 0.32 pp at 80 % power. The largest of the three effects is 0.167
pp, about half of that. If the endpoint agrees with the selection split, the
outcome is the plan's fourth preregistered branch — all three within the threshold
of one another, which closes per-sample allocation in both directions at once and
is reported as one result rather than as three nulls.

**The dose effect is large and is not in question.** `I100`, which treats all
45,000 training images, beats every allocation arm on every parent, by 0.38 to
1.22 pp with a mean near 0.9 pp. The plan's own section 95 says `I100` is a
reference and not a comparator, and that stands; but the contrast between a real
dose effect of about 0.9 pp and an allocation effect of at most 0.17 pp is the
shape of the answer this plan was built to find. Where the hardness goes appears
to matter far less than how much of it there is.

**The six parents give an empirical spread for free.** The three contrasts return
SDs of 0.275, 0.290 and 0.302 pp — agreeing to within 0.03 pp across contrasts
that share no arm pairing. That is a bound on this design's paired noise, not a
measurement of it: it mixes true parent-to-parent heterogeneity with run noise,
and only decision packet 0008's option B separates them. It does say that the
0.25 pp fallback threshold was not obviously the wrong order of magnitude.

Decision packet 0008 is rewritten against this state. Its option A no longer
includes any training.


### 2026-09-08 — `alloc-v1-p6-all-fork` finished; parent p6 is complete, nothing is running, and p6's three dose-matched arms are six times further apart than p1's or p5's

Terminal status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p6/all/run-bundle/manifest.json`), not from the
watcher's hint. The hand-run completion triple holds: `run-bundle/completion.json`
declares `completed`, the manifest declares `completed`, and the error marker
reads `no application error recorded`. The fork was created 2026-09-07T21:35:29Z —
fifteen seconds after `ALLOC_RANDOM` released the GPU — and finished 23:56:05Z at
epoch 199, global step 70400, in 2 h 20 min, with one hundred epoch rows covering
100 through 199, the whole post-fork range.

**Parent p6 is the third complete quartet, and it is the last thing that was
running.** All four arms ran back to back on one GPU: `safe` 14:35:14 → 16:54:51Z,
`fragile` 16:55:05 → 19:15:08Z, `random` 19:15:23 → 21:35:14Z, `all` 21:35:29 →
23:56:05Z, each starting within fifteen seconds of the last one finishing, 9 h 21 min
for the chain. The p5 chain ran alongside it on the other Hamster GPU and finished
ten minutes earlier. Both GPUs are now idle and no arm of `p2`, `p3` or `p4` exists.

Lineage, read from the manifest: source SHA `815dabd`, worktree
`p0096-815dabd20c7b`, parent `parents-v2-cropshift-s6` payload epoch 99 (checkpoint
SHA `bc90328d…`, parent config SHA `f3cd3621…`), child config SHA `69cf2852…`, fork
kind `stagewise_augmentation_fork_v1`, switch epoch 100, prefix policy `cropshift`,
late policy `idbh_weak`, **no mask**. Fixed identity: CIFAR-10 /
`saad_resnet18_cifar_v1` / RSLAD / teacher `chen2021_ltd_wrn34_10` (SHA `fc398a48…`)
/ training seed 6 / evaluation-attack seed 0 / linf eps 8-255 step 2-255 / world
size 1 / effective global batch 128.
W&B: `single-teacher-ard/runs/alloc-v1-p6-all-fork`.

**This arm is `I100` by construction, on the same evidence as p1's and p5's.** Its
`resolved_config.yaml` was compared line by line against `ALLOC_SAFE`'s and they
disagree on exactly four lines: `stagewise_late_mask_selected_ids_sha256` (`null`
here, `3d26e60b…` there), `stagewise_late_mask_selected_count` (`null` against
`15343`), the `tracking.run_id` and the `output_dir`. A null mask means the late
policy applies to every training image, so this run gives all 45,000 images
`IDBH_WEAK` from epoch 100 — the full dose — while every attack, schedule,
optimizer, seed and normalization field is identical to the three allocation arms.

**The dose-and-composition invariant holds on p6 as it did on p1 and p5.**
`masks/p6/manifest.json` gives `safe`, `fragile` and `random` the same
`selected_count` of 15,343 and the *same per-class counts* — 1693, 931, 1447, 1401,
1578, 1841, 2184, 1599, 1201, 1468, summing to 15,343 — all three derived from one
epoch-100 state (`7e9daa98…`) at one frozen threshold
(`student_global_logit_q10` 0.1769). Each arm's `resolved_config.yaml` declares the
digest the manifest records for its mask (`safe` `3d26e60b…`, `fragile` `69a60e98…`,
`random` `db90f60d…`), so the mask each run used is the mask the manifest describes.

**No result is recorded, because none exists yet.** The numbers below are the
validation-split accuracies each run used to select its own checkpoint. They are
internal validation, not the plan's endpoint.

| arm | checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- | --- |
| `ALLOC_SAFE` | best | 173 | 0.8636 | 0.6046 |
| `ALLOC_SAFE` | last | 199 | 0.8664 | 0.6046 |
| `ALLOC_FRAGILE` | best | 195 | 0.8642 | 0.5986 |
| `ALLOC_FRAGILE` | last | 199 | 0.8604 | 0.5982 |
| `ALLOC_RANDOM` | best | 192 | 0.8610 | 0.6036 |
| `ALLOC_RANDOM` | last | 199 | 0.8616 | 0.6010 |
| `I100` (`all`) | best | 195 | 0.8658 | 0.6084 |
| `I100` (`all`) | last | 199 | 0.8616 | 0.6084 |

Robust overfit gaps 0.00, 0.04, 0.26 and 0.00 pp in that order. All sixteen numbers
were read back from the rows for each arm's best epoch and epoch 199 in its
`epoch-metrics.jsonl` and match the manifest summaries. `train_valid_examples` is
45,000 on every arm at epochs 100 and 199, so all four used the full training split
and `resolved_config.yaml:18` `num_samples: 16` is again the inert display artifact
recorded in decision 0003. As on p1 and p5, each manifest's
`epoch_metrics_complete: false` (100 recorded against 200 expected) is correct for a
fork that resumes at epoch 100 and must not be read as truncation. Every `best_epoch`
is inside the post-fork range, as `post_fork_best_scope: true` requires.

**Two things about p6 differ from p1 and p5 and neither is a defect.** First, the
best epoch is no longer 197 on every arm: it is 173, 195, 192 and 195. On `safe`
the best PGD accuracy (0.6046) is exactly equal to the last one, and `best_epoch`
173 is simply the earliest epoch that reached it, so `best` and `last` are the same
number there. Second, `ALLOC_SAFE`'s clean accuracy is *higher* at the last
checkpoint (0.8664) than at the best one (0.8636); that is expected, because the
checkpoint is selected on PGD accuracy and clean accuracy is only reported alongside.

**The three dose-matched arms are 0.64 pp apart here, against 0.12 pp on p1 and
0.10 pp on p5.** Last-checkpoint validation PGD is 0.6046 / 0.5982 / 0.6010 for
`safe` / `fragile` / `random`; on the best checkpoint the spread is 0.60 pp. This is
the single most important number to come out of this arm, and it does not point at
an allocation direction — `safe` above `random` above `fragile` on p6 would be the
IDBH ordering, but the three arms sat inside 0.1 pp on both earlier parents, so a
0.64 pp spread on the third parent is at least as easily read as fork-to-fork
variability. **That is exactly the quantity nobody has measured**, and it is the
same size as the plan's provisional 0.25 pp threshold. Decision packet 0008 already
recommends measuring the paired-fork floor before spending 29-34 GPU-h on the
remaining twelve arms; this parent strengthens that recommendation rather than
changing it, and the packet has been updated with the p6 row.

**The dose signature repeats.** Training-set accuracy at epoch 199 is 0.8278 on
`safe`, 0.8074 on `fragile`, 0.8141 on `random` and 0.6791 on `all`. The three mask
arms sit within 2.04 pp of each other and the full-dose arm is 12.8 pp below the
lowest of them. That is the expected signature of `IDBH_WEAK` reaching 45,000 images
instead of 15,343: the richer augmentation makes the training images harder to fit.
It confirms the mask is doing what it claims and says nothing about which allocation
is better.

Checkpoints on disk for all four arms: `epoch-149.pt`, `epoch-199.pt`, `best.pt`,
`last.pt`. There is **no `epoch-114.pt`** anywhere, exactly as on p1 and p5, so
decision packet 0006 now describes twelve arms rather than eleven.

**`I100` is above the three allocation arms here too, and that is still not a
result.** It treats 45,000 images against their 15,343, so any gap is fully explained
by dose — the correction already written into this plan. Its margin over the three is
also smaller than before (0.38-1.02 pp on p6, against 1.12-1.24 pp on p1), which is
one more reason to distrust differences of this size until the floor is measured.
Beyond that, these are checkpoint-selection accuracies rather than the held-out
CE-PGD20 endpoint, which has not been run for any arm; the unit of the preregistered
rule is a parent-level paired difference over six parents and three parents are
complete; and the floor is still unmeasured, so the fallback threshold of 0.25 pp
stands. No contrast is computed and no direction is claimed.

**One check could not be run.** The manifest declares SHA-256 digests for the two
artifacts it exports (`epoch-metrics.parquet` `5b326221…` and
`sample-stats-train.parquet` `c973f4b2…`). Both files exist, but this session's
sandbox refuses to hash any file outside the repository checkout, so the digests were
read and not recomputed. The same limitation applied to the p5 postrun. It is a
harness restriction, not a finding about the run.

### 2026-09-08 — `alloc-v1-p5-all-fork` finished; parent p5 is complete, and six earlier postruns never ran

Terminal status was re-derived from the bundle that fired the event
(`runs/alloc-direction-v1/arms/p5/all/run-bundle/manifest.json`), not from the
watcher's hint. The hand-run completion triple holds: `run-bundle/completion.json`
is present, the manifest declares `completed`, and the error marker reads
`no application error recorded`. The fork was created 2026-09-07T21:27:49Z —
sixteen seconds after `ALLOC_RANDOM` released the GPU — and finished 23:46:07Z at
epoch 199, global step 70400, in 2 h 18 min, with one hundred epoch rows covering
100 through 199, the whole post-fork range.

**Parent p5 is the second complete quartet.** All four arms ran back to back on one
GPU: `safe` 14:35:15 → 16:52:28Z, `fragile` 16:52:44 → 19:10:00Z, `random`
19:10:17 → 21:27:32Z, `all` 21:27:49 → 23:46:07Z, each starting within seventeen
seconds of the last one finishing. The p6 chain ran alongside on the second GPU and
is still going.

Lineage, read from the manifest and `fork-lineage.json`: source SHA `815dabd`,
worktree `p0096-815dabd20c7b`, parent `parents-v2-cropshift-s5` payload epoch 99
(checkpoint SHA `d8c0ce35…`, parent config SHA `26cef228…`), child config SHA
`800c842d…`, fork kind `stagewise_augmentation_fork_v1`, switch epoch 100, prefix
policy `cropshift`, late policy `idbh_weak`, **no mask**. Fixed identity: CIFAR-10 /
`saad_resnet18_cifar_v1` / RSLAD / teacher `chen2021_ltd_wrn34_10` (SHA
`fc398a48…`) / training seed 5 / evaluation-attack seed 0 / linf eps 8-255 step
2-255 / world size 1 / effective global batch 128.
W&B: `single-teacher-ard/runs/alloc-v1-p5-all-fork`.

**This arm is `I100` by construction, on the same evidence as p1's.** Its
`resolved_config.yaml` was compared line by line against `ALLOC_SAFE`'s and they
disagree on exactly four lines: `stagewise_late_mask_selected_ids_sha256` (`null`
here, `86e049ee…` there), `stagewise_late_mask_selected_count` (`null` against
`15085`), the `tracking.run_id` and the `output_dir`. A null mask means the late
policy applies to every training image, so this run gives all 45,000 images
`IDBH_WEAK` from epoch 100 — the full dose — while every attack, schedule,
optimizer, seed and normalization field is identical to the three allocation arms.

**The dose-and-composition invariant holds on p5 as it did on p1.**
`masks/p5/manifest.json` gives `safe`, `fragile` and `random` the same
`selected_count` of 15,085 and the *same per-class counts* — 1668, 943, 1468, 1394,
1553, 1780, 2140, 1531, 1167, 1441 — all three derived from one epoch-100 state
(`734c0810…`). Each arm's `resolved_config.yaml` declares the digest the manifest
records for its mask (`safe` `86e049ee…`, `fragile` `6b863ccd…`, `random`
`d06033ec…`), so the mask each run used is the mask the manifest describes.

**No result is recorded, because none exists yet.** The numbers below are the
validation-split accuracies each run used to select its own checkpoint. They are
internal validation, not the plan's endpoint.

| arm | checkpoint | epoch | val clean | val PGD |
| --- | --- | --- | --- | --- |
| `ALLOC_SAFE` | best | 197 | 0.8656 | 0.6020 |
| `ALLOC_SAFE` | last | 199 | 0.8612 | 0.5962 |
| `ALLOC_FRAGILE` | best | 197 | 0.8590 | 0.5996 |
| `ALLOC_FRAGILE` | last | 199 | 0.8556 | 0.5956 |
| `ALLOC_RANDOM` | best | 197 | 0.8610 | 0.6018 |
| `ALLOC_RANDOM` | last | 199 | 0.8614 | 0.5966 |
| `I100` (`all`) | best | 197 | 0.8626 | 0.6084 |
| `I100` (`all`) | last | 199 | 0.8580 | 0.6050 |

Robust overfit gaps 0.58, 0.40, 0.52 and 0.34 pp in that order. All sixteen numbers
were read back from the rows for epochs 197 and 199 in each arm's
`epoch-metrics.jsonl` (rows 98 and 100) and match the manifest summaries.
`train_valid_examples` is 45,000 on every arm at both epochs, so all four used the
full training split and `resolved_config.yaml:18` `num_samples: 16` is again the
inert display artifact recorded in decision 0003. As on p1, each manifest's
`epoch_metrics_complete: false` (100 recorded against 200 expected) is correct for a
fork that resumes at epoch 100 and must not be read as truncation. Every
`best_epoch` is 197, inside the post-fork range, as `post_fork_best_scope: true`
requires.

**One mechanism check does come out of this, and it is about dose, not direction.**
Training-set accuracy at epoch 199 is 0.8285 on `safe`, 0.8097 on `fragile`, 0.8159
on `random` and 0.6830 on `all`. The three mask arms sit within 1.9 pp of each other
and the full-dose arm is 13 pp below all of them. That is the expected signature of
`IDBH_WEAK` reaching 45,000 images instead of 15,085: the richer augmentation makes
the training images harder to fit. It confirms the mask is doing what it claims and
says nothing about which allocation is better.

Checkpoints on disk for all four arms: `epoch-149.pt`, `epoch-199.pt`, `best.pt`,
`last.pt`. There is **no `epoch-114.pt`** anywhere, exactly as on p1, so decision
packet 0006 now describes eleven arms rather than four.

**`I100` is above the three allocation arms here too, and that is still not a
result.** It treats 45,000 images against their 15,085, so any gap is fully
explained by dose — the correction already written into this plan. Beyond that,
these are checkpoint-selection accuracies rather than the held-out CE-PGD20
endpoint, which has not been run for any arm; the unit of the preregistered rule is
a parent-level paired difference over six parents and two parents are complete; and
the floor is still unmeasured, so the fallback threshold of 0.25 pp stands. No
contrast is computed and no direction is claimed.

#### Six terminal arms produced no postrun, and the reason is quota

The watcher fired correctly for p5 `safe`, `fragile` and `random` and p6 `safe`,
`fragile` and `random`, and the hook launched a headless `claude -p` for each. All
six exited on their first turn with `api_error_status: 429`, `"You've hit your
session limit"` (`orchestration/ardx/claude-runs/20260907T{165253,165454,191031,191534,212741,213515}Z-*.json`),
so six runs reached a terminal state with nothing written to the plan. The same logs
also carry the untrusted-workspace warning — `Ignoring 44 permissions.allow entries
from .claude/settings.json` — which would have blocked every Bash call even without
the quota error; that is the trust-dialog condition already documented in
`CLAUDE.md`. This entry covers the p5 quartet in full. **p6 is deliberately not
written up**, because `alloc-v1-p6-all-fork` is still running and the parent is not
complete. Both conditions are infrastructure defects rather than scientific ones and
neither changes any number above.

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
