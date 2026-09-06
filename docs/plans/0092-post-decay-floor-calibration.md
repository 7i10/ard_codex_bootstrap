# Post-decay floor calibration

## Status

- Owner: human (approval to spend GPU), Claude Code (execution)
- Branch / base SHA: master, see progress log
- Current milestone: M1, relaunched 2026-09-06 21:34 after the first fan-out lost
  four of six replicates to a W&B run-id collision
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
- [~] M1: launch six continuations from a pinned worktree, epochs 101 to 114.
- [ ] M2: evaluate eighteen endpoints at e104, e109, e114.
- [ ] M3: compute the floor, write the record and report, commit.
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

Pending.  The campaign is **not** complete, contrary to the assumption recorded in
`docs/HANDOFF_2026-09-06_FERRET_TO_HAMSTER.md` chapter 0 step (4).

What ran on 2026-09-05, from pinned worktree `source-ed3b77daa1de`:

| stage | seed | result |
| --- | --- | --- |
| e100 prefix | dev-1 | completed 18:03:59, rc 0 |
| e100 prefix | dev-2 | completed 18:04:03, rc 0 |
| threshold freeze | dev-1 | completed 18:04:04, rc 0 |
| threshold freeze | dev-2 | completed 18:04:08, rc 0 |
| six control replicates | — | **never launched** |
| eighteen endpoints | — | **never launched** |

The session ended between the freeze and the arm launch, and the host became
unreachable shortly afterwards.  `runs/post-decay-floor-v1/arms/` did not exist.
No GPU work was lost, because none had started: the four stages above total about
four minutes.

M1 attempt 1 ran on 2026-09-06 from the same pinned worktree and produced no
usable measurement: two of six replicates reached epoch 114, the other four never
started, and one replicate per seed gives zero control-versus-control pairs —
the entire quantity this plan exists to measure.  About 0.9 GPU-hours were spent
and discarded.  See "M1 attempt 1: the run-id collision" in the progress log.
The e100 prefixes and frozen thresholds were untouched by all of this and are
still the plan-0087 lineage.

**Therefore the post-decay floor is still unmeasured and `sigma_d` remains the
provisional 0.35 pp.**  Everything that depends on it is still provisional:
`docs/MEASUREMENT_STANDARD.md` section 5, plan 0093's detectable effect, and the
interpretation of the plan-0091 official-test result (`+0.31 pp` mean, which sits
inside the unmeasured 0.25 to 0.50 pp bracket).

### A verification the interrupted stage did deliver

Regenerating the e100 prefix at source `ed3b77d` reproduced the plan-0087 prefix
computation exactly.  The online-state Parquet is bit-identical across the two
source revisions:

| artifact | plan 0087 (`bcb09a7`) | plan 0092 (`ed3b77d`) | identical |
| --- | --- | --- | --- |
| dev-1 e100 online-state | `0bb0701f...70441a` | `0bb0701f...70441a` | yes |
| dev-1 e100 checkpoint file | `89a43b3c...687158` | `e3a3975f...23e17d` | no |

The computed state matches; the checkpoint *file* hash does not, because the
checkpoint embeds the resolved config and source metadata.  This independently
confirms, from a different angle, what the Ferret session found by re-running the
same source twice: a checkpoint file hash is not a test of reproducibility.  It
also confirms that exposing `--continuation-seed` did not perturb the
computation.
