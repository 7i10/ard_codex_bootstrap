# Prescriptive v3: retention and learnability screen

## Status

- Owner: main thread; one Terra implementation owner; one consolidated
  scientific review after formula/resume evidence is stable
- Development teachers/seeds: Bartoldson L1 seed 1 and L3 seed 2 only
- Current milestone: M2 real-parent fork and CUDA smoke
- Last updated: 2026-08-05

## Why this screen exists

Exact online Student history predicts later PF/NR outcomes, but completed-v2
true-label mixing did not improve Best validation PGD. The completed-arm audit
shows that selected samples can be rescued while model-wide spillover cancels
or reverses the gain. It also confirms that
`0.5*p_teacher + 0.5*one_hot(y)` materially hardens the target: the selected
mean L1 target changes are about `.19` for PF and `.15` for NR, while the
Bartoldson teacher is correct on nearly all relevant samples. Prediction is
therefore retained as prognosis; treatment utility is tested separately.

This screen changes two orthogonal mechanisms only:

1. PF retention: preserve a fraction of the last pre-treatment Student soft
   target without changing the inner attack.
2. NR learnability: use an earlier prefix of the same PGD trajectory without
   changing teacher targets or outer-loss weights.

The screen is development-only. Official CIFAR-10 test, AutoAttack, Chen,
MobileNetV2, and unused confirmation seeds remain sealed.

## Frozen parents and schedule

Both routes resume exact epoch-79 end-boundary checkpoints and keep delayed
milestones `[120, 170]`.

| Run | Training seed | Parent checkpoint SHA-256 |
|---|---:|---|
| L1 | 1 | `2708da289c7ccb576b7f61c19b8abe7a1b6f8e72c54bc08b7e77c82ac3cf93d6` |
| L3 | 2 | `d5686355a20c9afdd1305ae5901ee6c85fd9406577c1bcdc4b1b85952bd369bd` |

Optimizer, scaler, RNG, sampler, SampleStateStore, W&B lineage, attack budget,
temperature, objective coefficients, and best-selection attack are restored
from the parent. Existing delayed controls `C` are reused after inactive-path
resume parity; they are not retrained merely to obtain a newer Git SHA.

## Frozen route construction

The epoch-34 exact-online score remains:

```text
0.5 * midrank_all(1 - inclusive robust-correct frequency)
+ 0.5 * midrank_all(-margin EMA)
```

The top decile is formed separately inside epoch-34 online PF and NR states.
Before treatment, route eligibility is refreshed exactly once using the
epoch-79 parent SampleStateStore:

- PF-H: epoch-34 PF top-decile intersect epoch-79 robust-correct.
- NR-H: epoch-34 NR top-decile intersect epoch-79 robust-wrong.

No score is recomputed after treatment begins. For each route, R is sampled
without replacement and matches H exactly by true class, epoch-79 route state,
epoch-34 teacher-adversarial correctness, and total count. Ties/random order
use SHA-256 over:

```text
["prescriptive-v3", parent_checkpoint_sha256, route,
 "random-control", sample_id]
```

Both H and R are preregistered and launched as one blocked matrix; random arms
are not chosen after viewing H results.

## PF treatment: anchored outer-target consistency

Freeze an eval-mode, parameter-gradient-disabled copy `A=S_79`. For selected
PF samples only during epochs 80--129:

```text
p_T = softmax(z_T(x_clean) / T)
p_A = softmax(z_A(x_clean) / T)
q_PF = 0.75 * p_T + 0.25 * p_A
```

The ordinary RSLAD sample objective becomes:

```text
(5/6) * T^2 * KL(q_PF || p_S(x_adv; T))
+ (1/6) * T^2 * KL(p_T || p_S(x_clean; T))
```

At epochs 130--199 and for every unselected sample, `q_PF=p_T` exactly. The
PGD-10 inner attack continues to maximize KL against the original detached
`p_T`; the clean branch is unchanged. The anchor is never trained, never
updates BatchNorm, and is restored from the hash-bound epoch-79 parent on
resume rather than serialized as a second mutable model.

## NR treatment: exact PGD-prefix curriculum

For selected NR samples only during epochs 80--99, generate the existing
PGD-10 trajectory once from the ordinary random start and retain its exact
step-5 prefix:

```text
x_adv_i = x_i^(5)   if selected
x_adv_i = x_i^(10)  otherwise
```

From epoch 100 onward all samples use step 10. Epsilon remains `8/255`, step
size `2/255`, projection/clamping are unchanged at every step, and the outer
RSLAD target, temperature, branch coefficients, and weights are unchanged.
The prefix must be captured from the same trajectory; a second random start or
a separately generated PGD-5 attack is forbidden.

The treatment mask and applied prefix/full-step identity are retained in run
artifacts so later observations cannot silently conflate the two training
inputs. Validation remains the saved full CE-PGD20 selection attack.

## Experiment matrix and launch policy

The matrix adds eight children and reuses two exact controls:

| Route | Selector | L1 | L3 |
|---|---|---|---|
| PF retention | history | PF-H | PF-H |
| PF retention | matched random | PF-R | PF-R |
| NR prefix curriculum | history | NR-H | NR-H |
| NR prefix curriculum | matched random | NR-R | NR-R |

Independent children are queued across the five available GPUs. One-GPU,
per-rank batch 128 remains the execution identity. Jobs are assigned by
measured host throughput and artifact locality; no 2-GPU DDP conversion is
made to accelerate this development block.

## Gates

Prediction authorization, applied before training:

- for PF and NR on both L1/L3, exact-online Student AUROC must exceed both
  instantaneous margin and teacher entropy by at least `.02`;
- paired class-stratified 2,000-replicate lower bounds must be positive;
- Student precision@10% may be at most `.01` below either comparator.

Authorization completed on 2026-08-05. All eight paired 95% intervals for
Student-history AUROC minus the comparator were strictly positive:

| Run | Route | vs instantaneous | vs teacher entropy |
|---|---|---:|---:|
| L1 | PF | `[.0348, .0482]` | `[.2244, .2565]` |
| L3 | PF | `[.0208, .0335]` | `[.1970, .2309]` |
| L1 | NR | `[.0625, .0754]` | `[.3164, .3381]` |
| L3 | NR | `[.0599, .0720]` | `[.2951, .3167]` |

Each interval used the frozen 2,000-replicate, class-stratified, paired
bootstrap with seed `2026080501`; resumable progress and final JSON are kept
outside Git under `.cache/analysis/pre39-68b7177/`.

Treatment Go, computed from validation only:

- two-seed mean Best CE-PGD20 improvement H-C at least `+0.50 pp`;
- H-C non-negative in each seed;
- mean Best clean degradation at most `0.50 pp`;
- mean robust-overfitting gap does not worsen;
- mean H-R Best CE-PGD20 is positive and H-R is non-negative in each seed.

PF and NR are judged separately and are not combined on development seeds.
A No-Go is still useful mechanism evidence; strengths, duration, q, masks, and
schedule are not tuned on L1/L3 outcomes.

## Milestones

- [x] M0 -- close exact-online point/bootstrap authorization and freeze route
  counts/hashes for L1/L3.
- [x] M1 -- implement PF target and NR attack-prefix boundaries plus config,
  checkpoint/resume, artifact identity, and focused tests.
- [ ] M2 -- create hash-bound H/R forks from both epoch-79 parents and run one
  real-parent one-epoch smoke per mechanism.
- [x] M3 -- run one consolidated scientific review; fix only P0/P1 findings.
- [ ] M4 -- launch the eight-arm development queue, monitor automatically, and
  evaluate Best/Last on validation only.
- [ ] M5 -- update result docs, commit, push, and merge the coherent milestone.

M1 verification evidence (2026-08-05):

- initial implementation gate: 56 focused tests passed; the impact-selected
  non-scientific gate reported 23 passed and 1 skipped;
- review-delta artifact/provenance checks: 16 focused tests passed;
- exact initial-fork mutation and route-boundary nodes: 2 passed;
- actual-Trainer inactive PF/NR loss/gradient/optimizer/RNG parity and active
  PF frozen-anchor contracts: 3 passed;
- uninterrupted versus resumed NR epoch-99-to-100 boundary, including model,
  optimizer, scheduler, SampleStateStore, RNG, selection state, and metrics:
  1 passed;
- targeted Ruff and `git diff --check` passed. Exact commands are retained in
  the milestone handoff; unchanged successful commands were not rerun.

The consolidated review found no P0 and initially found two P1s: the initial
epoch-79 child bytes were not checked against the completed-screen SHA, and
the central numerical/resume contracts lacked direct regressions. Both were
fixed. The fix-delta review found no remaining P0/P1 and authorized long-run
launch after the preregistered real-parent CUDA smoke. The review also kept
`num_workers=8`; the measured Ferret speedup at 4 workers is not mixed into
this paired screen without a separate execution-identity delta and parity
evidence.

## Test selection

- PF formula: target normalization/finite values, `0.75/0.25` equation,
  selected-only behavior, epoch 80/129/130 boundaries, unchanged clean term,
  unchanged inner attack, frozen anchor eval/no-grad/BN behavior.
- NR formula: step 5 is byte/tensor-identical to the prefix of the same step-10
  trajectory, selected-only behavior, epoch 80/99/100 boundaries, pixel clamp
  and `8/255+1e-7` bound, unchanged outer objective.
- Shared: stable sparse IDs, route intersection, matched-random strata/counts,
  deterministic masks, inactive-path fixed-batch parity, complete resume,
  DDP rank consistency, unique W&B/output lineage.
- Run focused T1/T2 first, then `scripts/verify.py --changed
  --non-scientific` once. Run one bounded real-parent GPU smoke per mechanism;
  do not use production training as a test.

## Risks

- Rescue/Harm is model-level moderation under interference, not individual
  causal effect; it cannot prove selected-sample treatment benefit.
- PF adds one frozen Student forward for selected samples and must not update
  its BatchNorm or parameters.
- NR state observations during epochs 80--99 are responses to a five-step
  training input; route assignment is fixed beforehand, and the applied step
  must be explicit in artifacts.
- The exact-online evidence is Bartoldson development evidence. Any successful
  route still requires unused Bartoldson seeds and Chen no-harm before official
  evaluation.
