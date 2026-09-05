# I100 official CIFAR-10 test and AutoAttack confirmation

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Branch / base SHA: master, see progress log
- Current milestone: M0
- Last updated: 2026-09-05

## Goal

Decide whether the I100 advantage survives on the official CIFAR-10 test set.

I100 is the frozen augmentation schedule: CropShift for epochs 0-99, then
IDBH_WEAK from epoch 100.  It is the only reproducible improvement the project
has.  Its evidence today is an epoch-199 CE-PGD20 gain over a matched
CROP_SUFFIX control of `+0.78 / +0.68 / +0.62 pp` on three previously unused
confirmation seeds.  Every one of those numbers comes from the 5,000-sample
internal validation split.  No I100 number has ever been measured on the
official 10,000-example test set, and none has ever faced AutoAttack.

This plan closes that gap and nothing else.

## Non-goals

- No training of any kind.
- No development seeds.  dev-1 and dev-2 reach I100 through a different
  campaign structure and are not a matched shared-prefix pair, so adding them
  would mix two comparison designs in one table.
- No sample-level intervention (PMP, ST1W, CW, S2, S3).  Those wait for the
  measurement-design audit.
- No promotion decision.  A positive result makes I100 reportable, not adopted.

## Existing state

Each confirmation bundle was built as a matched pair: one shared CROP_PREFIX
through epoch 99, then two continuations to epoch 199 that differ only in the
augmentation policy.  This is why the comparison is paired and why its spread
across seeds is small.

| seed | I100 arm | matched control | location |
|---|---|---|---|
| confirm-a | `unseen-confirm-a-i100-suffix` | `unseen-confirm-a-crop-suffix-r2` | Hamster / Ferret |
| confirm-b | `unseen-confirm-b-i100-suffix` | `unseen-confirm-b-crop-suffix` | Hamster |
| confirm-c | `unseen-confirm-c-i100-suffix-r3` | `unseen-confirm-c-crop-suffix-r2` | Ferret / Ferret |

Three runs still hold their checkpoints only on Ferret, about 445 MB each.
`ferret-collect` excludes model binaries by default, which is why they were
never mirrored.  They must be copied before evaluation.

`configs/evaluation/pgd_saved_checkpoint.yaml` sets `autoattack: false`.  A
second evaluation config is required; the existing one is not edited.

## Scientific contracts affected

- Evaluation integrity.  `ard.cli.evaluate` reads only saved weights, loads no
  optimizer, teacher or sample state, and re-checks the full 14-field attack
  identity against the training selection attack.  AutoAttack additionally
  requires `evaluation.autoattack: true` in the config *and* `--allow-autoattack`
  on the command line, and verifies the installed package against the pinned
  upstream commit `a3922004...` and its source digest.
- Reporting.  Clean, CE-PGD20 and AutoAttack accuracy are reported separately.
  Both the best and the last checkpoint are evaluated and reported separately.
- Provenance.  Historical AutoAttack manifests recorded library version
  `unknown` (`docs/experiments/0002-autoattack-provenance-amendment.json`).
  Every result from this plan must carry the pinned commit and source digest.

## Decisions

- **Scope is the three confirmation seeds.**  They are the seeds the +0.62 to
  +0.78 pp claim was made on, and they are matched pairs.  Testing the claim
  where it was made is the point.
- **Both checkpoints.**  Best and last, per the scientific invariants.  Robust
  overfitting is part of what distinguishes these arms.
- **AutoAttack standard, not rand.**  Standard is what the eight-cell seed-0
  table used, so the numbers stay comparable to the existing official results.
- **Run from a pinned worktree, detached, one job per GPU.**  Six independent
  evaluation jobs with no dependencies do not need the DAG orchestrator.  The
  simplest sanctioned path is a hand run under the run-bundle contract, which
  `ardx-watch.service` already detects.

## Preregistered decision rule

Primary quantity: AutoAttack accuracy of the last checkpoint at epoch 199,
I100 minus matched CROP_SUFFIX, per seed, on the official 10,000-example test
set.

- **Confirmed** if the difference is positive in all three seeds.
- **Not confirmed** if it is negative in two or three seeds.
- **Mixed** otherwise.

Secondary, reported but not decisive: the same difference on the best
checkpoint, on CE-PGD20, and on clean accuracy; and the best-minus-last gap per
arm, which measures robust overfitting.

Resolution and expectation, stated before the runs so they cannot be read into
the result afterwards.  The official test split has 10,000 examples, so one
example is 0.01 pp.  The internal-validation CE-PGD20 differences were +0.78,
+0.68 and +0.62 pp, a spread of 0.16 pp.  A paired shared-prefix comparison at
epoch 199 is the low-variance regime for this project: the five-seed study puts
the epoch-199 between-seed standard deviation at 0.118 pp for I100.  An effect
of roughly 0.7 pp against that background is resolvable, which is precisely why
this comparison is worth running while the sub-noise sample-level screens are
not.  AutoAttack accuracy is expected to be several points below CE-PGD20 in
absolute terms; only the difference between arms is being tested.

## Milestones

- [ ] M0: mirror the three Ferret runs, verify every checkpoint by SHA-256
  against its run-bundle manifest, and confirm each run has a sibling
  `resolved_config.yaml`.
- [ ] M1: add `configs/evaluation/autoattack_saved_checkpoint.yaml`
  (`autoattack: true`, `checkpoints: both`), pin a source worktree, and run one
  bounded smoke: CE-PGD20 only, one arm, to prove the lineage and attack-identity
  checks pass before spending GPU hours.
- [ ] M2: run the six AutoAttack evaluations detached, one per GPU.
- [ ] M3: aggregate into one record and report, apply the decision rule, commit.
- [ ] M4: write the decision packet for what the result implies.

## Test plan

- `PYTHONPATH=src python scripts/verify.py --changed` for the config addition.
- The M1 smoke is the real gate: it exercises the same lineage and attack
  identity path as the full runs at negligible cost.
- No new unit test is required; no production code changes.

## Risks and mitigations

- **A resolved config is missing or mismatched.**  Evaluation fails closed
  rather than producing a number.  M0 checks this before any GPU time.
- **AutoAttack runtime.**  Measured from the seed-0 evaluations of 2026-08-07:
  about 60 minutes per job, against about 1 minute for CE-PGD20 alone.  Six
  jobs on five idle GPUs is roughly two hours of wall clock.
- **A negative result.**  This is a real possibility and the plan is written so
  that it is publishable either way.  I100 currently rests on internal
  validation; a negative official-test result would be an important correction,
  not a failure of the plan.
- **Reading the result as a promotion.**  It is not.  Promotion needs a separate
  decision.

## Progress log

- 2026-09-05: plan authored.  Checkpoint inventory verified: all six arms exist
  with best and last; three are Ferret-only.  All five GPUs idle.

## Completion report

Pending.
