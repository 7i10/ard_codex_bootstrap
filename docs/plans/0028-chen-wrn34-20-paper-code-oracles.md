# Chen WRN34-20 upstream and paper-aligned SAAD oracles

## Status

- Owner: main thread; one core writer after contracts are frozen
- Branch / base SHA: `master`; implementation begins from the then-current
  clean pushed SHA
- Host: Hamster only; Ferret is forbidden
- Current milestone: M0--M2 complete; M3 retry active with `Linger=yes`
- Last updated: 2026-08-07

## Goal

Acquire and hash-lock the RobustBench teacher `Chen2021LTD_WRN34_20`, then use
the same teacher, student, dataset and seed to separate two full-SAAD oracles:

- **U (unmodified upstream):** exact pinned source behavior, including the
  effective hard-coded SGD weight decay `2e-4`;
- **P (paper-hyperparameter-aligned):** the same pinned source plus one isolated
  patch that makes the existing `--wd 0.0005` option effective.

The primary question is whether the teacher mismatch (WRN34-10 versus the
paper's WRN34-20) or the code/paper weight-decay mismatch explains the current
gap. Seed 0 is diagnostic and is not a reproduction claim.

## Non-goals

- Do not call P an exact paper reproduction. Runtime, hardware, stochastic
  execution, upstream checkpoint/evaluation lifecycle and other implementation
  details remain distinct.
- Do not silently reduce batch size, use DDP, accumulate gradients, enable AMP,
  alter entropy weighting, weaken attacks or change SWA to make the run fit.
- Do not copy or vendor license-absent upstream source into `src/ard/`.
- Do not expand beyond the frozen paired U/P seeds `0,1,2` cohort or alter its
  arms/endpoints after observing seed-0 official-test values.

## Existing state

- SAAD is pinned at `295121c5d2eed827b5b2d6aa42307de809bdfada` and
  RobustBench at `78fcc9e48a07a861268f295a777b975f25155964`.
- The RobustBench registry and local verified cache contain
  `Chen2021LTD_WRN34_20`. The architecture has `184,531,674` parameters
  versus `46,160,474` for WRN34-10.
- The completed WRN34-10 U oracle reported final SWA clean/PGD-20/AA
  `83.85/56.40/51.90%`. It used the wrong teacher for the paper's ERT table and
  upstream's hard-coded weight decay `2e-4`.
- The paper reports Chen WRN34-20 + ResNet-18 SAAD clean/PGD/AA
  `85.78/57.25/52.69%`, and Appendix B specifies weight decay `5e-4`.
- Bartoldson batch 128 failed on one RTX 4090 during the teacher input-gradient
  path. Ordinary data parallelism does not remove this bottleneck because the
  upstream teacher remains on GPU 0.

## Scientific contracts affected

- Teacher architecture, checkpoint SHA-256, internal preprocessing, threat
  model and exact RobustBench source identity.
- Upstream source/patch identity and effective optimizer weight decay.
- Pixel-space PGD-10 (`8/255`, `2/255`), full-SAAD teacher-input gradient,
  batch-min entropy weighting, IGDM, SWA and final evaluation.
- One-GPU batch 128 remains the primary execution identity. Any multi-device
  memory workaround is a separately labeled patched execution variant.

## Decisions

1. **Run U and P, not only P.** Their only intended scientific difference is
   effective weight decay, so the comparison isolates the documented
   code/paper mismatch.
2. **Label P accurately.** It is a paper-hyperparameter-aligned code variant,
   not an exact paper reproduction.
3. **Parallelize independent runs only after memory smoke.** If each variant
   fits a single GPU at batch 128 with peak at most `22,500 MiB`, launch U on
   one Hamster GPU and P on the other. This reduces wall time but does not reduce
   per-run VRAM.
4. **Do not use ordinary DataParallel/DDP as a memory fix.** Upstream
   `DataParallel` scatters the student while the full teacher and teacher-input
   gradient stay on GPU 0. DDP instead duplicates the teacher and changes
   per-rank BatchNorm, batch-min entropy weights, RNG/data order and execution
   identity.
5. **If one-GPU batch 128 fails, test a device-split patch, not a silent batch
   change.** Place the student on GPU 0 and frozen teacher on GPU 1, retain one
   global batch 128, transfer only inputs/logits/input gradients, and require a
   fixed-batch FP32 parity test against U. This uses two GPUs per run and is a
   patched execution oracle. If teacher-only batch-128 input-gradient itself
   does not fit one GPU, stop: simple parallelism cannot preserve the contract.

## Milestones

- [x] M0 -- extend acquisition/teacher/launcher registries for WRN34-20,
  acquire once through pinned RobustBench, calculate SHA-256, strict-load and
  bounded-forward audit, and atomically update `teachers.lock.yaml`.
  - Files: teacher lock, acquisition script, teacher registry, upstream launcher
    profiles, focused tests.
  - Acceptance: exact ID/architecture/parameter count/checkpoint hash and
    logits are recorded; runtime never auto-downloads.
  - Commit: `feat: lock Chen WRN34-20 teacher`.
- [x] M1 -- create U and P immutable source identities. Store the minimal P
  patch as a hash-bound external patch and apply it only to an ephemeral
  external worktree/staging area.
  - Acceptance: U remains byte-identical and clean; P changes only optimizer
    weight-decay argument plumbing; exact commands and patch hash are recorded.
  - Tests: command/source drift, non-overwrite, unknown teacher and effective
    optimizer contract.
- [x] M2 -- run one real batch-16 end-to-end smoke and one batch-128 memory
  smoke for each source identity. Record finite loss, peak allocated/reserved
  VRAM, runtime/import/teacher/data hashes and first-epoch time projection.
  - Acceptance: source-specific evidence passes and peak is at most
    `22,500 MiB`; no result-based contract changes.
- [ ] M3 -- if both one-GPU gates pass, launch U/P concurrently on Hamster GPU
  0/1. If either fails, execute the predeclared teacher-only probe and evaluate
  the device-split parity path before launching any long run.
  - Acceptance: immutable launch manifest, process/GPU/W&B identity, first
    finite update/epoch, telemetry and terminal evidence.
  - First attempt: launch identity and finite training passed, but both jobs
    were terminated by the user manager exiting on logout. No terminal result
    or resumable upstream checkpoint exists, so M3 remains open.
- [ ] M4 -- record final SWA clean, PGD-20, C&W, FGSM and AutoAttack for U/P;
  compare only within this pair and against the paper as descriptive seed-0
  evidence.
- [ ] M5 -- execute the already frozen paired U/P seeds 1/2 waves without
  result-dependent arm selection, then report all paired deltas and
  three-seed mean/std/worst/best.

## Agent and review budget

Use one bounded core writer for M0--M2. Run one consolidated scientific review
only after acquisition, source identities and smoke evidence are stable because
these changes affect teacher lineage, gradients and scientific values. A second
review is allowed only for an actual P0/P1. GPU execution and monitoring use
shell services, not reasoning agents. No Luna pass is needed unless repeated
configs/docs remain after the API is fixed.

## Test plan

- Focused unit: WRN34-20 allowlist, exact parameter count/hash/staged path,
  strict load, no runtime download, U/P command and patch identity.
- Fixed-batch regression: U versus P pre-optimizer logits/attack/loss/gradient
  equality; optimizer delta differs only as implied by weight decay.
- GPU smoke: batch 16 finite path and batch 128 VRAM for each immutable source
  identity. Do not repeat identical successful evidence.
- Full training and AutoAttack are scientific jobs outside the automated suite.
- `scripts/verify.py --changed` selects the smallest affected gate; docs-only
  updates do not trigger GPU tests.

## Risks and mitigations

- **Checkpoint size/network:** download once, reject partial bytes, compute
  SHA-256 and run offline thereafter.
- **License:** upstream SAAD license remains absent; retain source externally
  and distribute only our patch metadata, not copied upstream code.
- **Memory:** parallel independent runs reduce wall time, not memory. Use the
  frozen decision tree above; never call changed BN/batch entropy semantics an
  exact oracle.
- **One-seed uncertainty:** treat seed 0 as diagnosis. Do not infer superiority
  from sub-percentage differences without replication.
- **Evaluation mismatch:** upstream final live-SWA/in-process evaluation is not
  the local best/last separate-process protocol; report it separately.
- **User-session lifetime:** a transient `systemd --user` service is killed at
  logout when `Linger=no`. Require `Linger=yes` before every long user service;
  a launch manifest is not evidence that the process will outlive the session.
- **No upstream resume:** the pinned `saad.py` saves only the final SWA
  `state_dict`, after all 200 epochs and final PGD/CW/FGSM evaluation. An
  interrupted exact upstream run must restart at epoch 0. Do not label a
  checkpoint-instrumented patch as unmodified upstream.

## Frozen replication cohort

The upstream program evaluates CIFAR-10 official test data during training, so
the seed-0 endpoint cannot validly choose which arm receives confirmatory
replication. Before restarting seed 0, freeze the complete U/P seeds `0,1,2`
cohort. Execute it as three paired waves on the same two physical GPUs; do not
drop an arm or stop a wave using interim or seed-0 accuracy. If compute is
stopped after seed 0, report only exploratory seed-0 evidence and make no
multi-seed or superiority claim.

One paired wave is expected to take about 18 hours and reserves 20 hours. The
complete three-seed cohort therefore budgets about 54 hours wall time and 108
GPU-hours. Seed 0 is restarted first; seeds 1/2 may be staged later, but their
U/P membership and endpoints are fixed now and do not depend on seed-0 values.

## Restart and interpretation gate

1. Enable and verify user lingering before launch. This is a host persistence
   prerequisite, not a scientific change.
2. Reuse the passed source-specific batch-16 and batch-128 smoke evidence; do
   not rerun unchanged GPU smokes.
3. Restart U and P from epoch 0 in fresh output directories, from the same
   immutable `52affda329562d1493cbea1e77154be81b24ac3c` worktree. Launch them
   concurrently on physical GPUs 0 and 1.
4. Treat the interrupted epoch-wise test PGD curves as operational evidence
   only. Upstream evaluates the official test set each epoch, so those curves
   must not select a variant, epoch, or follow-up hyperparameter.
5. After both final SWA evaluations complete, report U/P clean, PGD-20, C&W,
   FGSM and AutoAttack together. Compare with WRN34-10 U and the paper only as
   descriptive seed-0 evidence.

At the observed `13.1--13.4 epochs/hour`, 200 epochs require approximately
`15.0--15.3 hours`; final evaluation and AutoAttack add further time. Expect
about `18 hours` and reserve `20 hours` wall time with both jobs running in
parallel. Start seeds 1/2 only after the preceding paired wave reaches two
valid terminal manifests; this is an operational dependency, not a
result-dependent gate.

The primary endpoint for the U/P contrast is final-SWA AutoAttack; PGD-20 is a
corroborating endpoint and clean accuracy is reported separately. Apply the
following practical materiality summary only to the completed three-seed
paired cohort; it is not a significance claim:

- if mean paired `P - U >= +0.5 pp` AA and mean PGD delta is non-negative,
  treat the weight-decay correction as materially positive;
- if both absolute mean paired AA and PGD deltas are below `0.5 pp`, treat the
  weight-decay explanation as practically small in this cohort;
- if mean paired `P - U <= -0.5 pp` AA and mean PGD is non-positive, treat P as
  materially worse;
- mixed signs are inconclusive and do not justify selecting one arm.

Report all three paired deltas, mean, standard deviation and worst/best seed;
do not hide sign disagreement behind the mean. WRN34-20 U versus completed
WRN34-10 U and proximity to the paper remain descriptive because they are not
paired multi-seed comparisons. After the frozen cohort, statically audit any
remaining paper/code/runtime gap before another oracle method or teacher.

## Progress log

- 2026-08-07: plan frozen after PGD-AT closure and while TRADES AutoAttack was
  active. Hamster GPU 1 was idle. No WRN34-20 checkpoint had been downloaded and
  no U/P long run had been launched.
- 2026-08-07: M0 acquired `Chen2021LTD_WRN34_20` once through the pinned
  RobustBench downloader. The complete checkpoint is `738,377,702` bytes with
  SHA-256 `dbfc7cfe402d9ddf6cbe47c4809eab97fcccce7b6a254030cdca2640639cfa28`;
  strict construction found exactly `184,531,674` parameters and finite
  `[1,10]` logits. It was atomically installed into `teacher_cache` and the lock
  was advanced from `missing` to `verified`. No runtime download path was added.
- 2026-08-07: the cross-runtime four-input teacher probe passed between PyTorch
  `2.11.0+cu128` and `2.4.1+cu121` with identical argmax and zero observed logit
  difference. M1 added explicit U/P configs and a valid one-line external patch;
  dry-run commands bind the same teacher and `--wd 0.0002` versus `0.0005`.
  The executed `saad.py` hash, variant, patch hash, changed-line count and
  physical GPU are part of smoke/heavy lineage. Focused verification reported
  `161 passed`, Ruff passed, targeted mypy passed, `git diff --check` passed and
  the patch applies cleanly to the pinned upstream source. Consolidated review
  initially found two P1 and one P2 lineage/regression gaps; the focused fixes
  added three-way GPU identity validation, exact staged-entrypoint evidence and
  an optimizer-delta regression. Delta re-review reported no remaining P0/P1.
- 2026-08-07: M2 passed for both immutable source variants. Paper-aligned P
  used GPU 1 and reported batch-16 `2/2` finite loss events with peak
  `2,306 MiB`, then batch-128 `10/10` with peak `7,410 MiB` and `72 C`.
  Unmodified U used GPU 0 and passed the same `2/2` and `10/10` quotas; its
  peak memory was `2,366/7,470 MiB`, with batch-128 peak temperature `56 C`.
  Its manifests are bound to the same clean SHA and teacher/logit evidence.
  The 24-GB memory workaround branch is therefore unnecessary for WRN34-20.
- 2026-08-07: U/P started concurrently at 06:43:05 from clean SHA `52affda`.
  At 17:23:41 the complete per-user systemd manager entered `exit.target` and
  sent SIGTERM to both healthy jobs simultaneously. Kernel/telemetry evidence
  shows no OOM, CUDA error or temperature violation: U/P peak memory was
  `7,566/7,506 MiB`, peak temperature `63/76 C`, and the last samples were
  actively computing. U had completed SWA evaluation through epoch 136 and was
  training epoch 137; P had completed through epoch 133 and was training epoch
  134. `loginctl` subsequently confirmed `Linger=no`.
- The partial common-epoch SWA test PGD trace is not a final result and is not
  compared between arms because it uses official test data. Neither run
  produced the final model, PGD/CW/FGSM result or AutoAttack result.
- 2026-08-07 21:54 JST: `loginctl` reported `Linger=yes`; both GPUs were idle,
  the detached execution worktree was clean at
  `52affda329562d1493cbea1e77154be81b24ac3c`, existing smoke bundles matched,
  and fresh retry outputs were absent. U/P seed 0 restarted from epoch 0 as
  `ard-saad-chen3420-u-s0-r1-52affda.service` on GPU 0 and
  `ard-saad-chen3420-p-s0-r1-52affda.service` on GPU 1. Invocation IDs are
  `65959f08729e40408fe4af588d895814` and
  `a7db1a9f43a14af3aa0cbbca912df1ad`. Both reported finite epoch-1 losses,
  correct source/GPU lineage, about `7.5 GiB` memory, 98% GPU utilization and
  empty telemetry error lists. No smoke or numerical test was repeated.

## Completion report

Pending.
