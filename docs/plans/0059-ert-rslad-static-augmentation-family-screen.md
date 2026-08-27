# 0059 — ERT/RSLAD static augmentation family screen

## Status

- Owner: root
- Branch / base SHA: `master` / `940762075d7eeaa65522fa7872a76ca33f179d0d`
- Current milestone: implementation and pre-production audit
- Last updated: 2026-08-28

## Goal

Compare the predefined nested CIFAR-10 family `CROPSHIFT`, `CROP_RE`, and
`IDBH_WEAK` under the frozen Chen ERT RSLAD contract, then freeze one static
policy before the three-unseen-seed stochasticity campaign.

## Non-goals

No BASE/CROPSHIFT rerun, augmentation sweep, new seed, Tiny-ImageNet,
official test, AutoAttack, Student-History intervention, ordering,
CutMix/MixUp, objective change, or post-hoc tuning.

## Existing state

BASE and CROPSHIFT are complete for seeds 1/2 in the Hamster-only static
trajectory screen. CROPSHIFT improved every registered validation CE-PGD20
endpoint in both seeds (mean final +1.24 pp; trajectory AUC +0.897 pp). The
pinned upstream reference is TreeLLi/DA-Alone-Improves-AT
`38b740aeffe5933c16869a126c6972ef443a8352` (MIT).

## Scientific contracts affected

- Dataset train-view augmentation policy and protocol identity.
- Source/epoch/sample-ID deterministic RNG ownership, with named `colorshape`
  and `erase` substreams.
- No change to pixel-domain attack, RSLAD objective, teacher target, optimizer,
  scheduler, split, checkpoint, or endpoint evaluation contracts.

## Decisions

- `CROP_RE` is exactly CROPSHIFT plus torchvision 0.26.0 RandomErasing defaults
  (`p=.5`, `scale=(.02,.33)`, `ratio=(.3,3.3)`, `value=0`, `inplace=False`).
- `IDBH_WEAK` adds only upstream `ColorShape('color')` before ToTensor and
  uses the same independent erase substream; ColorShape is not used in
  `CROP_RE`.
- Existing CROPSHIFT spatial draws retain the arithmetic source-keyed seed so
  its prior canary hashes remain unchanged. Named substreams are not claimed
  to be upstream bitstream reproduction; only operation distributions and
  ordering are matched.
- The nested promotion rule and clean/throughput guardrails in the user
  protocol are frozen before outcome collection.

## Milestones

- [x] M0 reconcile repository, upstream source, and current CROPSHIFT result.
- [x] M1 implement CROP_RE/IDBH_WEAK policies, protocol IDs, and configs.
- [x] M2 run focused data/config tests and the existing CROPSHIFT canary.
- [x] M3 run bounded distribution/RNG canary and freeze audit artifacts.
- [~] M4 launch four fresh full Hamster trajectories (CROP_RE/IDBH_WEAK ×
  seeds 1/2), with metrics-only W&B and checkpoints 49/99/149/199.
- [ ] M5 run 16 independent validation CE-PGD20 endpoints and aggregate
  trajectory/AUC, incremental, clean, RO, sample, and throughput metrics.
- [ ] M6 apply the frozen promotion tree, write the report/artifacts, review,
  commit, and stop.

## Agent and review budget

No subagent is needed: one owner handles the small transform extension and
Hamster launch. One consolidated scientific review is planned after endpoint
aggregation; no per-run reviewers.

## Test plan

- Cached: existing source-keyed data and protocol tests.
- New: deterministic bounded output tests for both nested transforms and config
  acceptance.
- Required before GPU: `scripts/analysis/ert_rslad_cropshift_canary.py`,
  focused unit tests, and `scripts/verify.py --changed`.
- Production training and endpoint attacks remain outside automated tests.

## Risks and mitigations

- Spatial-prefix drift: preserve the existing CropShift generator and canary
  hash; fail closed if it changes.
- Upstream parameter drift: record installed torchvision version/defaults and
  pinned IDBH source hash before launch.
- RNG leakage: all added draws use deterministic named generators, independent
  of worker/sampler order.
- Cost: no extra model forward/backward; use Hamster GPUs and metrics-only
  W&B artifacts.
- Lineage: freeze one source SHA/config/protocol set for all four runs and
  verify checkpoint/endpoint hashes before interpretation.

## Progress log

- 2026-08-28: reconciled `master` at `9407620`; confirmed BASE/CROPSHIFT
  incumbent results and upstream IDBH source.
- 2026-08-28: added nested deterministic transforms, protocol IDs, configs,
  and focused tests; existing CROPSHIFT canary hash remained unchanged.
- 2026-08-28: bounded 4096-source RNG canary passed on Hamster's torchvision
  `0.26.0+cu128`; RandomErasing rate was `0.4963`, each ColorShape operation
  was within `0.03` of its expected `0.125`, global RNG and source/order
  independence checks passed, and the frozen CROPSHIFT canary hashes matched.
- 2026-08-28: after the production source was frozen at `63bfe7b`, launched
  CROP_RE seed 1 on Hamster GPU0 and IDBH_WEAK seed 1 on GPU1 as persistent
  metrics-only W&B services; seed 2 will follow on each GPU after its seed-1
  service completes.

## Completion report

M3 is complete and the audit JSON is frozen before production. M4 is in
progress; M5–M6 remain pending. Production services are pinned to
`63bfe7b` and use local checkpoints with no model/run-bundle W&B uploads.
