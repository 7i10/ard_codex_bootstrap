# Chen WRN34-20 upstream and paper-aligned SAAD oracles

## Status

- Owner: main thread; one core writer after contracts are frozen
- Branch / base SHA: `master`; implementation begins from the then-current
  clean pushed SHA
- Host: Hamster only; Ferret is forbidden
- Current milestone: M0/M1 complete; M2 GPU smoke pending
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
- Do not start multiple seeds until the seed-0 diagnostic has been interpreted.

## Existing state

- SAAD is pinned at `295121c5d2eed827b5b2d6aa42307de809bdfada` and
  RobustBench at `78fcc9e48a07a861268f295a777b975f25155964`.
- The RobustBench registry contains `Chen2021LTD_WRN34_20`; the checkpoint is
  not yet present locally. The architecture has `184,531,674` parameters
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
- [ ] M2 -- run one real batch-16 end-to-end smoke and one batch-128 memory
  smoke for each source identity. Record finite loss, peak allocated/reserved
  VRAM, runtime/import/teacher/data hashes and first-epoch time projection.
  - Acceptance: source-specific evidence passes and peak is at most
    `22,500 MiB`; no result-based contract changes.
- [ ] M3 -- if both one-GPU gates pass, launch U/P concurrently on Hamster GPU
  0/1. If either fails, execute the predeclared teacher-only probe and evaluate
  the device-split parity path before launching any long run.
  - Acceptance: immutable launch manifest, process/GPU/W&B identity, first
    finite update/epoch, telemetry and terminal evidence.
- [ ] M4 -- record final SWA clean, PGD-20, C&W, FGSM and AutoAttack for U/P;
  compare only within this pair and against the paper as descriptive seed-0
  evidence. Decide whether a frozen multi-seed replication is justified.

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

## Decision table after M4

| Observation | Next conclusion/action |
|---|---|
| P approaches the paper while U is lower | weight-decay code/paper mismatch materially matters; freeze seeds 1/2 for U/P |
| U and P both approach the paper | WRN34-20 teacher mismatch was the dominant prior issue; replicate the cheaper scientifically relevant arm |
| U and P remain similarly below the paper | investigate remaining runtime/stochastic/protocol differences before more seeds |
| One-GPU smoke fails but device-split parity passes | run separately labeled two-device patched oracles, one at a time |
| Teacher-only batch-128 input-gradient fails | stop; simple GPU parallelism cannot preserve the frozen batch semantics |

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

## Completion report

Pending.
