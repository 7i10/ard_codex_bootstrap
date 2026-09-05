# I100 official CIFAR-10 test and AutoAttack confirmation

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Branch / base SHA: master, see progress log
- Current milestone: M2 (running)
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

- [x] M0: mirror the three Ferret runs, verify every checkpoint by SHA-256
  against its run-bundle manifest, and confirm each run has a sibling
  `resolved_config.yaml`.
- [x] M1: add `configs/evaluation/autoattack_saved_checkpoint.yaml`
  (`autoattack: true`, `checkpoints: both`), pin a source worktree, and run one
  bounded smoke: CE-PGD20 only, one arm, to prove the lineage and attack-identity
  checks pass before spending GPU hours.
- [~] M2: run the six AutoAttack evaluations detached, one per GPU.
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
- 2026-09-05 M0: mirrored `unseen-confirm-{a-crop-suffix-r2,c-crop-suffix-r2,c-i100-suffix-r3}`
  from Ferret into `<runtime>/runs/i100-official-test-v1/arms/`, 208.7 MB of
  checkpoints each, best and last plus `resolved_config.yaml` for all three.
- 2026-09-05 M1: added `configs/evaluation/autoattack_saved_checkpoint.yaml`
  (commit `9ffc1ae`) and pinned worktree `source-9ffc1aedf3b1`.  Bounded smoke on
  `confirm-b-i100` with the CE-PGD20-only config passed: `count = 10000`
  (official test split), threat hash `70811016...dcc4f2` matched the registered
  selection attack, clean 82.86% and CE-PGD20 57.07% for both checkpoints
  (confirm-b's best epoch is 199, so best and last hold the same weights).
- 2026-09-05 M2: six evaluations launched detached from the pinned worktree,
  three per GPU on Hamster, `13:29 JST`.  Round 1 confirmed running at 91% GPU
  utilisation on both devices.  Budget about 60-90 minutes per job from the
  2026-08-07 seed-0 AutoAttack evaluations, so roughly three to four hours.
- 2026-09-05 M2 round 1 done (`14:41 JST`): `confirm-a-i100` and `confirm-b-crop`
  are terminal and successful.  Both write `evaluation-results.json` with two
  checkpoint rows each, `count = 10000` on the official test split, threat hash
  `70811016...dcc4f2`, and AutoAttack actually run under the pinned upstream
  commit `a3922004...`.  Round 2 (`confirm-a-crop`, `confirm-c-i100`) started at
  `14:41 JST`; round 3 (`confirm-b-i100`, `confirm-c-crop`) has not started.  No
  record can be imported before all six exist: `scripts/aggregate_i100_official_test.py`
  reads all three seeds and both arms and fails closed on a missing one.  Numbers
  are deliberately not copied into this plan; the aggregator is the only place
  they may appear.
- 2026-09-05 M2 round 2, first half done (`15:52 JST`): `confirm-a-crop` is terminal
  and successful.  Its `evaluation-results.json` holds two checkpoint rows,
  `count = 10000` with `dataset_identity.split = test`, threat hash
  `70811016...dcc4f2`, AutoAttack run under the pinned upstream commit
  `a3922004...` with source digest `e74d6dab...`, and lineage to
  `unseen-confirm-a-crop-suffix-r2-training`.  Numbers stay out of this plan; the
  aggregator is the only place they may appear.  Three of six are now terminal
  (`confirm-a-i100`, `confirm-b-crop`, `confirm-a-crop`); `confirm-b-i100` and
  `confirm-c-i100` are running and `confirm-c-crop` is queued behind
  `confirm-c-i100` on GPU 1.  Both `launch.sh` drivers are alive and each walks its
  own three-arm queue, so no arm is stranded and no relaunch is needed.
  `scripts/aggregate_i100_official_test.py` iterates all three seeds and both arms
  and raises on a missing one, so M3 stays closed until the sixth run is terminal.
- 2026-09-05 M2 round 2 done, round 3 half done (`17:03 JST`): `confirm-c-i100` and
  `confirm-b-i100` are terminal and successful.  Checked `confirm-b-i100` in full: two
  checkpoint rows (`best.pt`, `last.pt`, distinct checkpoint SHA-256), `count = 10000`
  with `dataset_identity.split = test`, threat hash `70811016...dcc4f2`, AutoAttack
  actually run (`attack_version = standard`) under the pinned upstream commit
  `a3922004...` with source digest `e74d6dab...`, and lineage to
  `unseen-confirm-b-i100-suffix-training`.  Source SHA `9ffc1aedf3b1`, clean worktree,
  empty diff.  All seven manifest artifacts exist and each is stored in
  `run-bundle/artifacts/<name>/<sha256>/` under a directory named by its manifest hash,
  so the bundle is internally consistent; an independent re-hash was not possible
  because the runtime tree is outside this session's sandbox.  Numbers stay out of this
  plan; the aggregator is the only place they may appear.
  Five of six are now terminal (`confirm-a-i100`, `confirm-b-crop`, `confirm-a-crop`,
  `confirm-c-i100`, `confirm-b-i100`); only `confirm-c-crop` is still running, on the
  GPU-1 driver's queue.  M3 stays closed until it is terminal, because
  `scripts/aggregate_i100_official_test.py` iterates all three seeds and both arms and
  raises on a missing one.  No relaunch is needed and none was attempted.
- 2026-09-05 defect update: this postrun ran headless from `ardx-watch.service` and
  did its work, unlike the `confirm-a-i100` no-op below.  The headless invocation
  passes its tool allowlist explicitly on the command line, so it does not depend
  on the workspace-trust state that broke the earlier one.  Treat the entry below
  as describing that one event, not a standing blocker.
- 2026-09-05 defect (execution plane, not science): the headless postrun fired by
  `ardx-watch.service` for `confirm-a-i100` did no work.  Its whole log is
  `Ignoring 44 permissions.allow entries from .claude/settings.json: this workspace
  has not been trusted`, for the repo-root spelling
  `/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap`.  Every headless
  postrun on this checkout will be a no-op until that spelling is trusted, so the
  remaining four terminal events need a hand-run `/experiment-postrun`.  This does
  not affect any running evaluation.

## Completion report

Pending.
