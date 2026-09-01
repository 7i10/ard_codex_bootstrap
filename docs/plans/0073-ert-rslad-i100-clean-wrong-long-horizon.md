# 0073 — ERT/RSLAD I100 Clean-Wrong long-horizon confirmation

## Status

- Owner: Codex
- Branch / base SHA: `master` / `53c85b88e306e9b5e559821789e26776b6c2bc93`
- Current milestone: e114 lineage audit and continuation manifest preparation
- Last updated: 2026-09-02

## Goal

Continue the six completed I100 e114 trajectories (control, Clean-Wrong plain
AdvCE, and Clean-Wrong Teacher Positive-Floor Margin; dev-1/dev-2) through
epoch 199 without regenerating epochs 100–114, then evaluate the preregistered
sparse CE-PGD20 endpoints, e199 train direct/spillover effects, runtime, and
the frozen long-horizon decision.

## Non-goals

No new action, coefficient, threshold, seed, dynamic routing, combination,
official test, AutoAttack, or e99 restart. No Clean-Wrong mask or calibration
update.

## Existing state

The short transfer screen is complete and recorded in
`docs/ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md` and its machine artifact. The
six exact e114 checkpoints are stored under the completed Ferret run root
`ert-i100-action-transfer-v6`; their parent e99, fixed-mask, calibration,
Teacher, and CE-PGD20 identities are inherited unchanged.

Ferret has the required dataset, Teacher, disk, and two validated fast GPUs
(GPU0/GPU1). The local Hamster GPUs are idle but the required dataset is not
present and the root filesystem has only about 2.6 GB free, so Hamster is
ineligible for this campaign until a safe data/storage profile exists.

## Scientific contracts affected

Only the continuation boundary and endpoint horizon are new. The I100
RSLAD/IDBH_WEAK objective, KL-PGD10 training attack, source-keyed RNG,
fixed epoch-99 masks, frozen AdvCE/TPFM coefficients, Teacher, and evaluation
CE-PGD20 identity remain unchanged.

## Decisions

- Resume each arm from its own exact epoch-114 checkpoint; never resume from
  the epoch-99 parent or a different arm.
- Use `epochs=200` because the trainer's upper bound is exclusive, thereby
  producing epochs 115–199 inclusive.
- Use a detached completion-marker DAG with six training jobs followed by
  sparse validation and e199 train endpoints, aggregation, and report.
- Use Ferret GPU0/1 only for this campaign. Hamster is not launched because
  preflight currently fails on data locality and disk capacity.

## Milestones

- [x] Reconcile current source and read the long-horizon contract.
- [x] Audit six e114 checkpoint payloads, manifests, masks, and calibration.
- [x] Implement/validate the continuation and endpoint manifest.
- [x] Run bounded e114-resume canary and permutation/lineage checks.
- [x] Launch six detached continuations through epoch 199.
- [ ] Chain e129/e149/e169/e189/e199 validation and e199 train endpoints.
- [ ] Aggregate sparse trajectory, direct/spillover, and runtime results.
- [ ] Write report, record CW1–CW6 decision, commit, push, and stop.

## Agent and review budget

One owning implementation pass is sufficient; no subagent is needed. One
consolidated scientific review is required after endpoint artifacts are stable.

## Test plan

Run focused resume/lineage/permutation tests and `scripts/verify.py --changed`
before GPU launch. The e114 canary must prove the exact checkpoint boundary,
frozen mask/calibration, unchanged non-selected loss, Teacher freeze, and
inclusive epoch-199 endpoint. Production training and endpoint evaluation are
outside the automated test suite.

## Risks and mitigations

- Resume drift: verify byte SHA, payload epoch, config hash, parent lineage,
  RNG/sampler/sample state, and next epoch before launch.
- Off-by-one: assert `epoch-199.pt` exists and `epoch-200.pt` does not.
- Pairing drift: record source-ID permutation hashes for all three arms per
  seed and fail closed on mismatch.
- Host drift: keep Ferret GPU0/1 and exact source/config identities fixed;
  do not migrate active jobs to Hamster.
- Endpoint cost: split endpoint jobs by dependency where supported, but do not
  weaken the common CE-PGD20 contract.
- W&B/storage: metrics-only online tracking; large model uploads remain local.

## Progress log

- 2026-09-02: short-screen e114 artifacts and all 32 endpoint cells verified
  present; campaign source `2522bc9` and manifest lineage retained.
- 2026-09-02: Hamster preflight failed for this campaign (dataset missing,
  root filesystem 100%); Ferret GPU0/1 preflight passed.
- 2026-09-02: Source `c6032f9` was pushed, Ferret worktree prepared, all six
  e114 hashes matched, and the e115 resume canary passed. The detached
  manifest controller launched the first two training jobs on Ferret GPU0/1;
  no endpoint or long-running job was manually polled after the bounded check.

## Completion report

To be filled after the six continuations and all preregistered endpoints are
complete. The report must distinguish long-horizon point estimates from
training-seed uncertainty and must not start an automatic follow-up.
