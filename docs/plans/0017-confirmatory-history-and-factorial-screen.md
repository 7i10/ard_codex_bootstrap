# Confirmatory history block and factorial intervention screen

## Status

- Owner: main thread; shell processes own GPU execution
- Branch / base SHA: `cleanup/observability-first` / `ad6d26e1c15d90eeb8c1d8b10cfa081a433364b9`
- Current milestone: H4 five-arm continuation active from the exact epoch-99 L3 parent
- Last updated: 2026-08-02

## Goal

Replace the unsafe single-run decision with a frozen two-teacher by two-seed
confirmatory block. Each loss-identical observed RSLAD trajectory records all
student-history and teacher-response primitives once. If the Bartoldson history
selector replicates under both confirmatory seeds, test selector utility and
intervention utility separately with one common-state five-arm continuation.

## Non-goals

- Do not tune predictors, anchor/outcome definitions, splits, thresholds, or
  feature transforms after inspecting seed-1/2 outcomes.
- Do not launch five separate feature-specific training runs; candidate
  features are offline views of the same four trajectories.
- Do not start the five-arm intervention before the frozen selector decision.
- Do not run full AutoAttack for the confirmatory block automatically.

## Existing state

- L1 Bartoldson seed 1 is active from clean SHA `0fdfaeb...` on Hamster GPU 1.
  It uses the historical `rslad_logging_only` identity and exact detached state.
- The public cleanup at `ad6d26e...` represents the same optimization as
  `method=rslad, observation.profile=teacher_response`; the one-off launch gate
  and campaign watchdog code are no longer in the public runtime.
- Seed-0 common-trajectory replay met History Go for Chen (`+0.0622` AUROC)
  and Bartoldson (`+0.0525` AUROC). The frozen H2 block also met History Go
  for L1/L2/L3/L4 with deltas `+0.0499/+0.0686/+0.0513/+0.0663` AUROC.
  These are conditional trajectory analyses, not between-seed uncertainty.
- Hamster exposes two RTX 4090 GPUs in a non-isolated shell; Ferret exposes
  three. L1 occupies Hamster GPU 1 and the other four GPUs were idle at the
  launch preflight.

## Scientific contracts affected

- RSLAD KL PGD-10 remains pixel-space Linf `8/255`, step `2/255`, random start,
  teacher-clean target, temperature 1 and `T^2` scaling.
- Observation is detached FP32 and cannot affect loss, attack, gradients,
  optimizer/scheduler, RNG, BatchNorm, or teacher parameter gradients.
- All runs use the same CIFAR-10 train/validation split, student, 200-epoch
  schedule, single-GPU batch 128, anchor epoch 99 and prospective epochs
  100--199. Official-test and AutoAttack outcomes are not predictor inputs.
- The later five arms must restore the same epoch-99 model, optimizer,
  scheduler, scaler, sample state, sampler/data/augmentation/attack RNG and
  differ only in the registered selector/intervention arm.

## Decisions

- L1 is retained rather than duplicated only after source/runtime identity is
  explicitly migrated and a direct cross-version CUDA comparison passes.
  The per-version parity checks are necessary preflight but not sufficient
  bridge evidence; a bridge failure triggers one new-SHA Bartoldson seed-1
  replacement before pooling and does not invalidate L2--L4.
  Its distinct Git SHA remains part of lineage and is not concealed.
- L2 runs on Hamster GPU 0; L3/L4 run on Ferret GPUs 0/1. This avoids three
  simultaneous Ferret training jobs and leaves Ferret GPU 2 available without
  weakening the single-GPU protocol.
- For independent future runs, compare measured host throughput with artifact
  transfer size instead of assuming the current artifact location is fixed.
  Hash-verified rsync is preferred when transfer is cheaper than slower compute.
- Do not run a short cross-host C continuation: it cannot establish 100-epoch
  trajectory equivalence. Use static environment/identity preflight, place
  `HS/RS` on Hamster and `C/HD/RD` on Ferret, and keep the primary selector
  contrasts within host. Cross-treatment contrasts remain exploratory until a
  promising effect is replicated without host confounding.
- No result-dependent runtime gate belongs in `ard.cli.train`. The frozen YAML,
  clean Git SHA, resolved config, W&B identity and run bundle are the evidence.
- Two confirmatory seeds estimate direction replication, not a precise
  between-seed confidence interval. Publication performance claims require a
  later seed policy separate from this signal screen.
- The five-arm screen is `C + (history/random) x (softening/downweight)`.
  Random masks are class matched and use the same intervention budget.
- The first class-matched random mask is a pre-specified screening draw, not an
  estimate of the random-mask distribution.  For each treatment, if the saved-
  checkpoint PGD contrast `H-R` is positive or within `0.5` percentage points
  of zero, generate two additional independently seeded class-matched masks
  from the same parent and budget before making a selector claim.  A positive
  selector claim requires History to exceed all three random controls and the
  random-control mean by at least `0.5` percentage points without a clean-
  accuracy loss above `0.5` points.  If `H-R < -0.5` points, stop that treatment
  as a negative screen; do not generalize the result to every possible random
  draw.  This sequential rule is fixed before H4 outcomes are observed.
- The intervention budget is fixed at `K=3566`, the earlier seed-0
  Bartoldson final-error oracle budget. This reuses a pre-existing development
  budget for comparability; it is not selected from L3 outcomes. History scores
  are ordered descending with stable sample ID ascending as the tie-break.
  The class-matched control ranks each candidate by
  `SHA256(2026080201:class_id:sample_id)` and takes the same per-class counts.
- If both Bartoldson history runs are No-Go, the screen may use only the same
  frozen current-state baseline if it wins consistently in both runs. Mixed or
  inconclusive selector evidence stops the screen instead of choosing post hoc.

## Milestones

- [x] H0 — freeze block and bounded parity preflight.
  - Files: this plan and `configs/analysis/history_confirmatory_block_v2.yaml`.
  - Test: one CUDA random-start RSLAD observation parity case.
  - Acceptance: design frozen before L2--L4 outcomes; L1 bridge is explicit.
  - Commit: `research: freeze confirmatory history block`.
- [x] H1 — launch L2--L4 as clean fixed-SHA long runs.
  - Files: no runtime mutation; use canonical observed scientific configs.
  - Acceptance: unique outputs/W&B IDs, exact Git/config/teacher hashes, correct
    GPU, live process, first finite batch/epoch, sample state in checkpoint.
  - Rollback: cancel only a failing named run; do not restart successful runs.
- [x] H2 — analyze the complete four-trajectory block once.
  - Output: frozen full table for Teacher-only, Student-only, Main effects and
    Main+product; per-run AUROC/AUPRC/log-loss/prevalence and conditional CIs.
  - Acceptance: no feature/spec changes; teacher/seed dependence reported.
  - Result: L1 (Bartoldson seed 1) `+0.0499`, L2 (Chen seed 1) `+0.0686`,
    L3 (Bartoldson seed 2) `+0.0513`, and L4 (Chen seed 2) `+0.0663` AUROC;
    all four are History Go. The direction replicates across teachers and
    seeds, with Chen's deltas larger in this block. Teacher-response and
    student-history product terms remain descriptive predictors, not causal
    evidence for an intervention.
- [x] H2a — close the old/new L1 bridge before pooling.
  - Acceptance: direct bounded CUDA equality of model, optimizer, scheduler,
    scaler, RNG, sampler, global step and every format-v3 primitive/count.
    Config identity and added forward-count telemetry are the only allowed
    differences. Failure requires a new-SHA L1 replacement.
- [x] H3 — implement a post-H2 prospective intervention screen v2, but do not launch early.
  - Files: analysis/fork CLI, arm configs, focused checkpoint/RNG/lineage tests.
  - Acceptance: the five arms share exact parent state and class budget; only
    selector/intervention changes; test/AA leakage fails closed.
  - Test: actual factory-returned C continuation parity with random-start PGD,
    strict fork/resume, transactional screen creation, selector and artifact
    provenance, fixed-mask formula/gradient and lineage contracts.
  - This is not a retroactive preregistration or an independent confirmation
    of H2. Launch remains blocked until the feature-only L3 replay, selector
    mask generation, and real epoch-99 parent/W&B inputs are attested.
- [ ] H4 — conditionally launch the five arms and evaluate saved checkpoints.
  - Acceptance: control and factorial contrasts are computed from the same
    branch state; best/last clean/PGD are mandatory; AA follows only for a
    scientifically justified reduced set.

## Agent and review budget

One bounded research-planner pass checks the frozen design while the main
thread performs non-overlapping preflight. GPU jobs are detached shell
processes. One Terra writer will own the common-state fork after the API is
frozen. One consolidated scientific review occurs before H4, not per run or
during waiting. No watchdog/recovery agent is used.

## Test plan

- Actual bounded GPU parity: `CUDA_VISIBLE_DEVICES=0 ... pytest -q
  tests/integration/test_checkpoint_resume.py::test_teacher_response_cuda_parity_with_random_start_pgd`.
- Launch preflight: config resolution, clean/pushed full SHA, teacher SHA,
  dataset, unique output/W&B IDs and idle named GPU.
- First progress: one bounded status/log read after launch; finite metrics and
  complete sample-state checkpoint check at the first checkpoint boundary.
- H3: focused exact parent-state, RNG equality, mask class/count equality,
  arm-only config-delta and resume-lineage regression tests.
- T4/T5 and full AutoAttack are not automated gates.

## Risks and mitigations

- Cross-SHA L1 bridge can hide drift. Preserve both identities and require
  explicit migration plus exact bounded CUDA parity; otherwise rerun L1.
- Three runs can contend for Ferret CPU/data I/O. Allocate only two there and
  put L2 on Hamster's free GPU.
- W&B artifact intervals may be mistaken for complete online history. Exact
  history lives in `SampleStateStore`; periodic artifacts are checkpoint
  snapshots, not per-epoch raw trajectories.
- Factorial continuation can accidentally become an inexact resume. A separate
  registered fork operation must preserve parent state and record parent SHA;
  ordinary resume remains exact-config only.
- A single failed arm is ambiguous. Launch all five from one validated parent
  only after the selector gate, and distinguish selector, treatment and random
  regularization contrasts before seeing results.

## Progress log

- 2026-08-01: final non-scientific cleanup gate exited 0; cleanup commit
  `ad6d26e...` was pushed to `origin/cleanup/observability-first` without
  touching the active old-SHA L1 worktree.
- 2026-08-01: Hamster read-only inventory found L1 healthy at epoch 22,
  approximately 201 images/s with finite train/validation metrics. Hamster GPU
  0 and all three Ferret GPUs were idle. Ferret fixed-SHA preflight returned
  `ready=true` with verified external checkouts and teacher cache.
- 2026-08-01: cleanup-SHA CUDA optimization parity passed (`1 passed in
  2.80s`) on Hamster GPU 0. Independent planning identified that this does not
  by itself prove the old/new L1 bridge; direct cross-version parity is now H2a.
  No long run or evaluation was treated as a test.
- 2026-08-01: frozen design SHA-256 is `a0a7fe0e...`; launch Git SHA is
  `8254a8899ae7373c2f541d108593e5c8185b26f5`. L2 (Chen seed 1) started on
  Hamster GPU 0, L3 (Bartoldson seed 2) on Ferret GPU 0 and L4 (Chen seed 2)
  on Ferret GPU 1. All initialized unique online W&B runs and saved a finite
  epoch-0 checkpoint with 45,000 format-v3 sample records, `seen=1`, and no
  missing required teacher-response primitive. Epoch-0 throughput was
  636.2/163.2/367.0 images/s for L2/L3/L4 respectively. L1 remains active on
  Hamster GPU 1; Ferret GPU 2 remains free for bounded bridge verification.
- 2026-08-01: direct old/new CUDA bridge passed on the same Ferret RTX 4090
  (`0fdfaeb...` `observe_teacher_signals` versus `8254a88...`
  `observation_profile`). The two-epoch deterministic fixture produced equal
  checkpoint state after removing identity-only fields (`b469c5d3...`) and
  equal format-v3 sample state (`907ea4c2...`, five records). The raw
  attestation SHA-256 is `4f98fda4...`; the compact provenance record is
  `tools/internal/history_replication/provenance/observation_bridge_2026-08-01.yaml`.
  Therefore L1 is admissible for the frozen observation analysis with its old
  Git lineage preserved. This bridge does not claim 200-epoch performance
  equivalence. An initial device-activation failure was fixed by selecting the
  CUDA device before resetting peak-memory statistics, with a focused unit
  regression; no long run was affected.
- 2026-08-01: L4 (Chen seed 2) completed with exit code 0 at 11:01 JST. A
  bounded Ferret GPU-2 worker profile found 4 workers faster than 8 for the
  exact teacher-response observation workload (387.4 versus 338.4 images/s,
  +14.48%) with identical loss and accuracy metrics. Active runs remain
  unchanged; future Ferret launches use 4 workers. Evidence is recorded in
  `tools/internal/performance/provenance/ferret_workers_2026-08-01.yaml`.
- 2026-08-01: post-H2 intervention screen v2 fail-closed fork implementation completed without launching
  an arm. Focused final evidence: intervention unit tests `4 passed`; actual
  factory-C parity plus one-epoch strict resume `2 passed`; changed impact gate
  `23 passed, 1 skipped`; Ruff and changed-source mypy passed. One consolidated
  scientific review found seven initial P1 and four closure-delta P1 issues,
  including broken post-fork resume, unmatched random budgets, partial screen
  publication, self-declared selector provenance, incomplete parent state,
  missing parent-best/W&B lineage, dirty launch, and a synthetic rather than
  actual-C parity test. All were fixed; the final delta review reported no
  remaining P0/P1. This experimental runtime stays on the research branch and
  is not a license to launch before H2.
- 2026-08-01: L1 (Bartoldson seed 1) completed at epoch 199 and its separate
  saved-checkpoint PGD-20 evaluation completed online (`best` clean/PGD
  `83.87/51.00`, `last` `84.70/45.35`). L2 and L4 Chen evaluations also
  completed (`seed 1` best/last PGD `55.99/55.76`; `seed 2` `55.79/55.67`).
  These official-test results are recorded but are not predictor inputs and do
  not change the frozen H2 specification. L3 remained healthy on Ferret at
  epoch 139; H2 outcome analysis remains blocked until it is terminal.
- 2026-08-01: evaluation preflight exposed a cross-version operations rule:
  schema-evolving runs must be evaluated with the exact training Git SHA, not
  the active canonical worktree. An isolated evaluator must receive the pinned
  teacher cache and external checkout before its first launch. The initial L2
  attempts stopped before output/W&B initialization; the exact-SHA run then
  completed with one evaluation identity. Future launchers must check runtime
  SHA, config parse, teacher bytes and external checkout in one preflight.
- 2026-08-02: H2 completed with History Go for all L1--L4 (`+0.0499`,
  `+0.0686`, `+0.0513`, `+0.0663` AUROC, respectively). The next state is
  feature-only L3 replay, then frozen selector masks, then the common-state
  five-arm screen. No live replay or H3 launch is claimed here.
- 2026-08-02: L3 feature-only replay completed on Ferret GPU 2 in about 76
  minutes from fixed SHA `6d77338...`; all 20 epoch-4--99 checkpoints and
  45,000 stable IDs passed lineage/attack checks. The deterministic selector
  fit produced `K=3566` with threshold `0.7550193467`; independent output
  directories reproduced the same coefficients, selected-ID hashes, and class
  counts. The five-arm launch drops the previously proposed short C parity in
  favor of static identity checks and within-host primary contrasts.
- 2026-08-02: the random-control sequential replication rule was fixed and
  pushed as `561025c` before H4 outcomes. The five checkpoints were created
  transactionally at clean fork SHA `6d77338...` from epoch-99 parent SHA
  `44ac2edb...`; copied HS/RS bytes matched Ferret exactly. HS/RS launched on
  Hamster GPUs 0/1 and C/HD/RD on Ferret GPUs 0/1/2 with online W&B IDs
  `bart-h3-{hs,rs,c,hd,rd}-s2-20260802-v1`. All five completed finite epoch
  100: Hamster throughput was `206.5/202.2` images/s and Ferret throughput was
  `166.7/173.6/165.6` images/s. Epoch-100 validation PGD was
  `50.62/50.40/50.50/50.56/50.44%` for HS/RS/C/HD/RD. These early values are
  launch evidence only and are not used for an intervention conclusion.

## Completion report

Pending.
