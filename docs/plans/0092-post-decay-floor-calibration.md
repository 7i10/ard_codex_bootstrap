# Post-decay floor calibration

## Status

- Owner: human (approval to spend GPU), Claude Code (execution)
- Branch / base SHA: master, see progress log
- Current milestone: M3 complete; M4 (folding the number into
  `docs/MEASUREMENT_DESIGN.md`) is the remaining step
- Last updated: 2026-09-06

## Goal

Measure how far apart two *identical* continuations land.

This is instrument calibration, not a scientific experiment.  It tests no
hypothesis about adversarial robustness distillation.  It measures one number:
the standard deviation of the difference between two untreated forks of the same
epoch-99 parent, run for fourteen epochs past the learning-rate decay, evaluated
at the registered CE-PGD20 endpoint.

That number decides what every future screen can conclude, and it has never been
measured.  `docs/MEASUREMENT_DESIGN.md` brackets it at 0.25 to 0.50 pp from three
indirect estimates that disagree by a factor of two, and notes that the exact
quantity — two untreated forks from a common epoch-99 parent over fourteen epochs
— is the shortest fork of the three, so it could plausibly sit below the whole
bracket.  The bracket is not academic: it moves the number of paired blocks
needed to resolve a 0.17 pp effect from seventeen to sixty-eight.

## Non-goals

- No treatment.  Every arm is an untreated control.  Nothing is being screened.
- No new parent, seed, teacher, attack or schedule.  The design copies plan 0087
  exactly except that all arms are controls.
- No conclusion about PMP, ST1W, TPFM or any other intervention.  This plan
  supplies the denominator those questions need; it does not answer them.

## Existing state

`docs/MEASUREMENT_DESIGN.md` establishes the surrounding picture and is the
reason this plan exists:

- Before the learning-rate decay the floor is 1.14 to 1.25 pp and flat in
  horizon.  Training longer does not help; the decay does.
- After the decay the per-run spread collapses about fourfold within four epochs.
- A shared 79-epoch prefix removes only about six per cent of the variance
  before the decay, so pairing is much weaker protection than it looks.
- Two nominally identical controls in different campaigns differ by 0.94 and
  1.78 pp (`docs/COEFFICIENT_AUDIT.md`).  That is the failure this plan prevents.

`docs/EVIDENCE_RECLASSIFICATION.md` ranks this measurement first among all open
work, as the prerequisite for the other four.

## Design

Fork each of the two I100 epoch-99 parents into **three untreated control
replicates**: identical config, identical schedule, differing only in the
post-fork random stream.  Run epochs 101 to 114 exactly as plan 0087 did, sharing
the same epoch-100 no-action prefix per seed.  Evaluate the registered CE-PGD20
endpoint at epochs 104, 109 and 114.

That yields `2 seeds x C(3,2) = 6` control-versus-control comparisons at each of
three post-decay horizons: the missing cell in the floor table, measured on
exactly the design this project keeps using.

Parents are the plan-0087 lineage: dev-1 `360910a8...7630835`, dev-2
`bb0c7c1a...f7aaf7`, teacher `fc398a48...c383983`, training attack
`97a41870...9623d4d`, endpoint `70811016...dcc4f2`.

## What a replicate varies — settled

"Differing only in the post-fork random stream" already has a definition in this
runtime, and a precedent.  Plan 0054 used it to build the `L2-R1`/`L2-R2`/`L4-R1`/
`L4-R2` blocks whose control-versus-control gaps are the pre-decay floor that
`docs/MEASUREMENT_DESIGN.md` reports.

The mechanism is the `continuation_seed` argument of `run_stage_a_arm`
(`src/ard/analysis/ert_stage_a_runtime.py:876-882`).  Setting it produces:

| random source | replicate behaviour |
| --- | --- |
| data order | unchanged — keeps the parent's `seeds.data_order` |
| augmentation view | unchanged — same seed, so the same view per epoch and sample |
| attack random start | re-seeded |
| Python / NumPy / global Torch | re-seeded |

The full epoch-100 model, optimizer, scheduler, scaler, sampler and sample state
are restored from the parent checkpoint first; only the post-resume streams above
are re-seeded.  The value enters the child identity hash
(`_arm_hash(..., continuation_seed=...)`), so each replicate is a distinct,
recorded object rather than a silent re-run, and two arms differing only in this
value are the same experiment run twice.

This also settles a question the audit left open: because R1/R2 were built this
way, the 0.16 to 1.88 pp pre-decay gaps really are same-parent, same-data-order
control-versus-control differences, not a comparison between different parents.
`docs/MEASUREMENT_DESIGN.md`'s classification of them as type (a) is correct.

The online-state runner did not expose the flag; a pass-through was added to its
`arm` subcommand.  Nothing in the underlying contract changed.

## Scientific contracts affected

None are changed.  The arms are untreated, so no threat model, normalization,
objective, checkpoint or evaluation contract moves.  The only new recorded
quantity is the replicate index.

## Preregistered analysis

Report, per horizon and per seed, the six pairwise absolute differences in
held-out CE-PGD20 accuracy and their standard deviation.  Report the pooled
estimate across seeds.  State the resulting minimum detectable effect for a
two-arm, k-block design at k = 2, 5 and 10, so a future screen can size itself.

There is no pass or fail.  A large floor and a small floor are equally
informative; only the number matters.  It will be reported with its own
uncertainty, from six comparisons, and not presented as precise.

## Milestones

- [x] M0: define and freeze what a replicate varies; confirm the runtime can
  express it; confirm the epoch-100 prefixes and frozen thresholds from plan
  0087 are reusable, or rebuild them.
- [x] M1: launch six continuations from a pinned worktree, epochs 101 to 114.
- [x] M2: evaluate eighteen endpoints at e104, e109, e114.
- [x] M3: compute the floor, write the record and report, commit.
- [ ] M4: fold the number into `docs/MEASUREMENT_DESIGN.md` and restate the
  minimum detectable effects there.

## Test plan

`PYTHONPATH=src python scripts/verify.py --changed` for any config or script
addition.  A one-epoch canary on one replicate before the fan-out, as plan 0087
did, to prove the lineage checks pass.

## Risks and mitigations

- **The replicate definition is contestable.** Mitigated by settling it in M0,
  in writing, before any GPU time, and by recording it in the contract.
- **Six comparisons give a noisy estimate of a standard deviation.** True and
  unavoidable at this cost.  The estimate will carry an interval and the report
  will not round it into false precision.
- **The result may widen rather than narrow the bracket.** That is a legitimate
  outcome and does not make the measurement wasted; it would mean short screens
  are even less usable than currently thought.

## Cost

Fourteen epochs at 92.2 s (dev-1) and 95.2 s (dev-2) per epoch is about 22
minutes per run.  Six runs is **2.2 GPU-hours**, plus eighteen endpoint
evaluations.  On two idle GPUs that is under two hours of wall clock.  Less than
the three hours budgeted for option A in decision packet 0001, and a prerequisite
for interpreting option A rather than an alternative to it.

## Progress log

- 2026-09-05: plan authored from the prescription in `docs/MEASUREMENT_DESIGN.md`
  section 3.5 and the ranking in `docs/EVIDENCE_RECLASSIFICATION.md` Part 3(c).
- 2026-09-06: recorded the interrupted state after the host outage; see the
  completion report.  The prefixes and frozen thresholds survive and are reusable.
- 2026-09-05 M0: replicate definition settled from the plan-0054 precedent and the
  `continuation_seed` implementation; see the section above.  Verified that three
  values of the seed give three distinct arm identity hashes.  Added the
  `--continuation-seed` pass-through to the online-state runner's `arm`
  subcommand; 54 focused tests pass.  Approved to spend GPU.
- 2026-09-06 M1 attempt 1: launched, four of six replicates lost to a W&B run-id
  collision.  Diagnosis below.  The attempt tree was moved aside to
  `runs/post-decay-floor-v1/arms-attempt1-wandb-id-collision/` and M1 was
  relaunched at 21:34 JST with a per-replicate run namespace.
- 2026-09-06 M1 attempt 2: all six relaunched together under
  `post-decay-floor-v1-rep<N>` namespaces.  `run_namespace` was confirmed absent
  from `_arm_hash`, so it is a tracking label only and changing it leaves the
  scientific identity untouched.  The two replicates that had already succeeded
  under the old namespace were discarded rather than reused: a floor measurement
  compares replicates to each other, so all six must be produced identically.
  Each replicate takes about 26 minutes on one 4090.
- 2026-09-06: the frozen thresholds were re-verified after the relaunch, because
  the launcher bug below could in principle have hidden a failed freeze step.  It
  had not: `thresholds/dev-1` and `thresholds/dev-2` match their recorded
  SHA-256, carry the q10 cut points (student 0.140155 / 0.138205, teacher
  0.174019 / 0.177792) and name the registered parents.
- 2026-09-06 M2/M3 prepared while the GPUs were busy: `endpoints.sh` for the
  eighteen endpoint sweeps and `scripts/aggregate_post_decay_floor.py` for the
  floor itself.  The aggregator's statistics were checked against hand-computed
  values on synthetic input before any real data existed.

### A second launcher defect, found alongside the first

The fan-out script reported `rc=0` for every replicate, including the four that
had crashed after fourteen seconds.  The line was

    echo "$(date -Is) done $name rc=$?"

and the command substitution runs *before* the expansion of `$?`, so `$?` reports
the status of `date`, which always succeeds.  Every launcher in this campaign and
in plan 0091 carried the same line.  It is why the failures were noticed only
when the log stream showed a traceback, rather than at the moment they happened.

The corrected launchers capture `rc=$?` on its own line immediately after the
command, and stop the remaining replicates on the first failure instead of
continuing into a partially populated campaign.

### The aggregator reports two spread numbers, deliberately

The plan preregistered "the six pairwise absolute differences and their standard
deviation".  That is reported unchanged.  But three replicates give two degrees
of freedom per parent, and the six pairwise differences drawn from them are not
independent, so the standard deviation of those six numbers is not the sigma_d a
future screen needs.  The aggregator therefore also pools the two within-parent
variances into four degrees of freedom and reports sigma_d = sigma_within * sqrt(2)
with a 95% interval.  Both appear in the record, so neither the preregistration
nor the better-founded estimator is hidden.

### M1 attempt 1: the run-id collision

`_tracking_run_id` (`src/ard/analysis/ert_stage_a_runtime.py:165-187`) builds the
W&B run id from `(run_namespace, model_init seed, arm, source_sha[:7])`.  It does
**not** include `continuation_seed`.  Plan 0092's replicates differ *only* in
`continuation_seed`, so all three replicates of a seed asked W&B for one id:

    ert-post-decay-floor-v1-<seed>-I100_CONTROL-ed3b77d

The docstring's collision guard is the `-orch-`/`-retry-` suffix, and both are
driven by `ARD_ORCH_*` environment variables that only the orchestrator sets.
This was a hand-run, so no suffix was added.  `r1` reserved the id; `r2` and `r3`
called `wandb.init(id=<taken>, resume="never")`, W&B raised `UsageError`, the
adapter turned it into `TrackingError`, and the run-bundle recorded `failed`
about 1.5 s after start.  Deterministic, not flaky: a verbatim retry fails
identically.

Outcome per replicate, all from pinned worktree `source-ed3b77daa1de`:

| seed | replicate | result |
| --- | --- | --- |
| dev-1 | r1 | reached epoch 114, all three horizon checkpoints written |
| dev-1 | r2, r3 | failed at start, W&B id collision |
| dev-2 | r1 | reached epoch 114, all three horizon checkpoints written |
| dev-2 | r2, r3 | failed at start, W&B id collision |

Two surviving replicates give `C(1,2) = 0` control-versus-control pairs per seed,
so attempt 1 yields **no** floor estimate.  `sigma_d` remains the provisional
0.35 pp and everything downstream of it stays provisional.

Two driver defects surfaced with it, both already fixed in the relaunched
`arms.sh`: `rc=$?` was read after a `$(date -Is)` substitution had reset it, so
every run logged `rc=0` including the failures; and there was no abort, so the
driver marched on to the next doomed replicate.

Two things the relaunch leaves open, neither blocking it:

1. The runtime defect is worked around, not fixed.  `--run-namespace
   post-decay-floor-v1-rep<N>` makes the ids distinct without touching the arm
   identity hash or the pinned source SHA, but any future stage-A hand-run that
   forks replicates under one namespace hits the collision again.  A fix means
   adding `continuation_seed` to `_tracking_run_id` plus a regression test.
2. Because the workaround varies the namespace, `run_namespace` now differs per
   replicate, and it *is* recorded (`ert_stage_a_runtime.py:1292` and `:1410`).
   The M3 aggregator must group replicates by seed and `continuation_seed`, never
   by namespace, or six replicates of one campaign will read as six campaigns.

### The watcher cannot see a successful stage-A replicate

Independent of the collision, and found while re-deriving attempt 1's status:
stage-A runs never write `run-bundle/completion.json`.  Only
`src/ard/cli/train.py:1173` writes it, and `run_stage_a_arm` does not go through
that path.  The hand-run completion contract
(`scripts/ardx/ardx_common.py:322-344`) requires the completion marker, so a
successful stage-A replicate is reported `incomplete`, stays non-terminal, and
**never fires the postrun hook**.

Confirmed on this campaign: both epoch-114 replicates of attempt 1 read
`incomplete`, and the only terminal events the watcher emitted were the four
failures — the automation could see the campaign break but could not have seen it
succeed.  M2 and M3 cannot be driven by the watcher until this is closed; until
then the arms have to be collected by hand.

## Completion report

**The floor is measured.  It is far below the bracket that stood in for it.**

Record: `docs/experiments/ard_post_decay_floor_v1.json`
(sha256 `a9ea3baf923d22ee...`).
Report: `docs/POST_DECAY_FLOOR.md`.  Runs produced at `ed3b77daa1de`.

| horizon | sigma_d | 95% interval | mean pairwise gap | largest pairwise gap |
| --- | ---: | ---: | ---: | ---: |
| e104 | 0.159 pp | 0.095 to 0.457 pp | 0.133 pp | 0.260 pp |
| e109 | 0.124 pp | 0.074 to 0.356 pp | 0.113 pp | 0.200 pp |
| e114 | 0.092 pp | 0.055 to 0.265 pp | 0.087 pp | 0.140 pp |

`docs/MEASUREMENT_DESIGN.md` section 3.5 bracketed this quantity at **0.25 to
0.50 pp** and adopted 0.40 pp as a working value; this plan's own prerequisites
table carried 0.35 pp.  The measured value at e114 is **0.092 pp** -- a quarter
of the working value, and below the whole bracket.  Even the upper end of its
95% interval, 0.265 pp, sits at the bracket's floor.

That section had predicted exactly this.  It noted that the quantity being
bracketed -- two untreated forks of a common epoch-99 parent over fourteen
epochs -- is the shortest fork of the three it reasoned from, and "could
plausibly sit below the whole bracket".  It does.

### The floor keeps shrinking after the decay

0.159, 0.124, 0.092 pp at e104, e109, e114.  The collapse the design document
observed within four epochs of the decay does not stop there; it continues
through e114.  Judging later is not merely safer, it is measurably cheaper in
replication.  Whether it keeps falling past e114 is untested.

### What a screen can now resolve

| paired blocks | e104 | e109 | e114 |
| ---: | ---: | ---: | ---: |
| 2 | 0.418 pp | 0.325 pp | 0.243 pp |
| 5 | 0.265 pp | 0.206 pp | 0.154 pp |
| 10 | 0.187 pp | 0.146 pp | 0.109 pp |

The standard two-seed screen was believed to resolve nothing below about
0.69 pp.  At e114 it resolves **0.242 pp**.  The 0.3 to 0.6 pp band that
`docs/EVIDENCE_RECLASSIFICATION.md` called invisible is inside reach of the
design this project already runs.

These use the calibration's four degrees of freedom rather than the screen's
k-1, because the point of measuring the floor is that a screen need not
rediscover it.  They are 1.33x the `sqrt(7.85/k)` convention in
`docs/MEASUREMENT_DESIGN.md`, which treats sigma as exactly known; six runs do
not make it so, and that factor is the honest price of the sample size.

### What this does not license

The replicates share a parent, an epoch-100 prefix, a data order and a campaign,
differing only in the post-fork random stream.  That is precisely a
within-campaign treatment-versus-control screen, so the number applies there.

It does not license comparisons across campaigns.  But the reason is not the one
first written here.

**Amendment, same day.**  The completion report as first committed (11c960c)
claimed that this floor sits an order of magnitude below the 0.94 and 1.78 pp
shift between two campaigns' controls, and called that gap "the largest
unexplained quantity in the project".  That was a mistake of regime.  Those
controls are measured at **epoch 84**, sixteen epochs *before* the learning-rate
decay; this floor is measured at 104 to 114, *after* it.  The 0.94 and 1.78 pp
figures sit inside the pre-decay floor of **1.14 to 1.25 pp** that five campaign
families independently agree on (`docs/MEASUREMENT_DESIGN.md` section 2.4).
**They are that floor.  There is no anomaly, and nothing here is unexplained.**

`docs/EVIDENCE_RECLASSIFICATION.md` had it right all along: its C3 row records
the horizon as `e84 / e94`, the regime as `pre`, and the floor it applied as
`1.14 (e84)`.  The reclassification is horizon-aware.  This measurement does not
overturn it.

What *is* open, and had been obscured by the false claim: **whether a
cross-campaign penalty exists after the decay is untested.**  No two campaigns
have ever run untreated controls from the same parent past epoch 100.  Until one
does, a post-decay comparison across campaigns has no measured floor at all, so
the practical rule is unchanged even though the reasoning behind it was wrong.

It is also specific to this design: two parents, fourteen epochs past the decay,
the registered CE-PGD20 endpoint on the validation split.  Nothing transfers to
another horizon, another endpoint, or the official test split.

### How much past evidence this actually reopens

Also corrected.  `docs/EVIDENCE_RECLASSIFICATION.md` has 38 rows, not "about
forty screens" all of which this touches:

| regime | rows | floor they were judged against | does this measurement change it? |
| --- | ---: | --- | --- |
| post-decay | 17 | 0.25-0.50 pp bracket, 0.40 pp working | **yes** -- the measured value is 0.092 to 0.159 pp, so 2.5 to 4x too large |
| pre-decay | 21 | 1.14-1.25 pp, measured separately | no |

Two of the seventeen (B2, B3) were judged against floors derived from their own
campaign's placebo or random arms (0.601 and 0.426 pp).  Those are empirical and
design-specific and should not be replaced by this number without checking that
the designs match.

### Cost

Six replicates at about 26 minutes each on one 4090, run two at a time: 2.6
GPU-hours.  Eighteen endpoint sweeps at about 31 seconds each: 0.2 GPU-hours.
A further 0.9 GPU-hours were spent and discarded on attempt 1.

### What ran, and when

| stage | result |
| --- | --- |
| e100 prefix, both seeds | 2026-09-05 18:03-18:04, rc 0 |
| threshold freeze, both seeds | 2026-09-05 18:04, rc 0 |
| six control replicates, attempt 1 | 2026-09-06 21:02-21:31, four of six lost to a W&B run-id collision, discarded |
| six control replicates, attempt 2 | 2026-09-06 21:34-22:53, all six rc 0 |
| eighteen endpoints | 2026-09-06 22:54-22:56, all rc 0 |

