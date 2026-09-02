# 0073 — ERT/RSLAD I100 Clean-Wrong long-horizon confirmation

## Status

- Owner: Codex
- Branch / base SHA: `master` / `53c85b88e306e9b5e559821789e26776b6c2bc93`
- Current milestone: completed continuation, endpoint aggregation, and decision record
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
(GPU0/GPU1). The original campaign was pinned to Ferret before launch. A later
host re-audit found that Hamster also has the dataset at
`/home/shunsuke.naito/workspace-local/datasets/ard/torchvision` and about
2.3 TB free. The running campaign was not migrated; only the failed dev-2
TPFM child was technically recovered on Hamster after preserving its
immutable scientific identity.

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
- Use Ferret GPU0/1 for the original campaign. If a technical recovery is
  required, Hamster GPU0 may be used only after exact parent/config/hash
  verification and without changing the scientific identity.

## Milestones

- [x] Reconcile current source and read the long-horizon contract.
- [x] Audit six e114 checkpoint payloads, manifests, masks, and calibration.
- [x] Implement/validate the continuation and endpoint manifest.
- [x] Run bounded e114-resume canary and permutation/lineage checks.
- [x] Launch six detached continuations through epoch 199.
- [x] Chain e129/e149/e169/e189/e199 validation and e199 train endpoints.
- [x] Aggregate sparse trajectory, direct/spillover, and runtime results.
- [x] Write report, record CW1–CW6 decision, commit, push, and stop.

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
- 2026-09-02: The initial Hamster check used an incorrect dataset alias and
  incorrectly reported the dataset missing. Re-audit confirmed the canonical
  dataset path and free disk on Hamster; no active campaign was changed.
- 2026-09-02: Source `c6032f9` was pushed, Ferret worktree prepared, all six
  e114 hashes matched, and the e115 resume canary passed. The detached
  manifest controller launched the first two training jobs on Ferret GPU0/1;
  no endpoint or long-running job was manually polled after the bounded check.
- 2026-09-02: Ferret completed five arms. The dev-2 TPFM continuation was
  recovered on Hamster GPU0 after repeated technical parent-SHA/path
  validation failures; the exact e114 parent bytes and scientific identity
  were retained. Epoch-199 training and all six sparse endpoint cells are
  complete.
- 2026-09-02: Aggregation produced the contract, results, direct/spillover,
  and runtime artifacts. TPFM is directionally positive on held-out e199 in
  both seeds (+0.04/+0.20 pp robust), while plain AdvCE is not (-0.30/0.00
  pp). Direct gains do not transfer at the same magnitude. No automatic
  follow-up was started.

## Completion report

To be filled after the six continuations and all preregistered endpoints are
complete. The report must distinguish long-horizon point estimates from
training-seed uncertainty and must not start an automatic follow-up.

Completion report: see `docs/ERT_RSLAD_I100_CLEAN_WRONG_LONG_HORIZON.md` and
the four `docs/experiments/ert_rslad_i100_cw_long_horizon_*_v1.json` artifacts.
