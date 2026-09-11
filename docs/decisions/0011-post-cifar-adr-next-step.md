---
id: 0011
status: pending
created: 2026-09-11
campaign: adr-cifar10-campaign-v1
question: plan 0097's preregistered CIFAR-10 comparison returned SIGN_CONFIRMED
  (decision packet 0009, option B) — proceed straight to ImageNet Stage 0/1,
  first validate a new self-distillation refinement at CIFAR scale, or pause
  on new scientific work until the open infrastructure items close?
options:
  A: Proceed to the ImageNet Stage 0/1 campaign already designed in
    `docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md` (capacity-aware recipe,
    plain-AT baseline then ADR, RSLAD comparison arm for the
    mechanism-mapping framing). GPU-hours: not reliably known yet — the
    engine still needs ImageNet dataset/augmentation support and a real
    throughput benchmark (deferred twice already this session); do not
    treat any prior throughput figure in that document as costed.
  B: Before any ImageNet work, validate the "reliability-adaptive
    self-distillation" mechanism discussed this session (gate ADR's λ/γ by
    a measured student-EMA agreement signal instead of a fixed cosine
    schedule) at CIFAR-10 scale, reusing this project's own engine and the
    architectures already run here. ~85-100 GPU-hours (est.), ~1-2 days
    wall-clock across 5 GPUs, plus implementation and scientific review.
  C: Do nothing further on new scientific work right now. Close the open
    infrastructure items first — decision packet 0010 (`chosen: null`,
    AutoAttack OOM), the `adv`→`ard-v2` environment cutover
    (`docs/BACKLOG.md` A8, now understood to need pairing with packet
    0010's Option A), and reconcile the still-uncommitted forensic detail
    the concurrent automated postrun session accumulated in this plan. 0
    GPU-hours.
recommendation: B
chosen: null
---

## Evidence

Record: `docs/experiments/adr_cifar10_replication_v1.json` (`.sha256` alongside).
Report: `docs/ADR_CIFAR10_REPLICATION_RESULTS.md`. Plan: `docs/plans/0097-adr-cifar10-replication.md`
(Completion report). Ledger row: `docs/RESEARCH_STATUS_SUMMARY.md`.

**Primary preregistered comparison (best checkpoint, model weights, official-test
AutoAttack, n=3 seeds each arm):**

| | baseline mean | ADR mean | gain | sd of gain |
|---|---:|---:|---:|---:|
| ResNet-18 (Nesterov-matched) | 47.32 | 48.54 | **+1.22 pp** | 0.36 |
| MobileNetV2 | 39.92 | 42.72 | **+2.79 pp** | 0.36 |

**SIGN_CONFIRMED**: MobileNetV2's gain is positive and exceeds ResNet-18's. Same
ordering on EMA weights (+2.08 pp vs +3.18 pp). The two gains' difference
(1.57 pp) is roughly 3x the combined per-arm sd (≈0.51 pp) — a real signal at
n=3, not proof at a population level; this project's claims discipline treats
three seeds as directional, and this packet does the same.

**Secondary, informative, not decisive:**
- `trades_adr`'s EMA mean (49.50%) meets or exceeds the literature comparators
  (AdaAD 49.21%, RSLAD 49.23%, RSLAD-300 49.50%) that plain `trades` (47.55%)
  could not reach — ADR's addition appears to close a gap this project had
  disclosed but not resolved.
- `trades_49k_validation` pilot (n=1, diagnostic only): narrowing
  `validation_fraction` moved the number slightly *away* from the literature
  range (47.64% vs the standard split's 47.87%) — the held-out-split-size
  hypothesis for the TRADES gap does not hold. This does not bear on the
  primary comparison or on either option below.

**Standing constraint this packet does not touch**: decision packet 0010
(AutoAttack's unbatched-forward OOM) remains `status: pending`, `chosen: null`.
Per the project's own rule, no new scientific GPU job should start until a
human resolves it — this applies to both options A and B below equally, since
both need AutoAttack evaluation runs.

## Option A — ImageNet Stage 0/1

**What it would establish**: whether the CIFAR-scale capacity-inverse
self-distillation benefit replicates at ImageNet-1k scale on genuinely mobile
architectures (<5M params) — the two open literature gaps this project's own
2024+ survey confirmed (mobile-scale ImageNet AT with strong evaluation;
ImageNet-scale self-distillation) — using the recipe and pilot design already
worked out in `docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md` §7-9.

**GPU-hours**: not reliably costed. The engine has no ImageNet-1k data loader
or augmentation pipeline yet (confirmed this session — `src/ard/data/` only
wires up CIFAR-10/100/TinyImageNet), AMP is implemented but never enabled
(`scaler=None` hardcoded in `src/ard/cli/train.py`), and no real
attack-inclusive throughput benchmark on this hardware exists — an earlier
autonomous session's throughput claim was independently proven physically
impossible by FLOP arithmetic this session. Any number quoted before that
benchmark runs would be fabricated precision.

**Preregistered decision rule**: none can be written yet — Stage 0's own
plain-AT-vs-modern-recipe-ceiling comparison needs to run and be measured
before Stage 1's self-distillation arm has a baseline to be compared against.

**Risks**: this session's own red-team review (recorded in
`docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md` §9) found the effect size at stake
(ADR's own capacity-inverse delta, ~0.3-1.0 pp) sits close to this project's
measured noise floor, and Lee & Chung's mechanism paper's own Fig. 1b permits
the trend reversing rather than strengthening below the tested capacity range
— i.e., a real chance Stage 1 returns "consistent with zero" regardless of
which way the true effect points, after a large, expensive, hard-to-iterate
campaign.

## Option B — validate the reliability-adaptive mechanism on CIFAR first

**What it would establish**: whether gating ADR's λ (trust in the teacher)
and γ (EMA decay) by a measured student-EMA agreement/reliability signal,
instead of ADR's fixed cosine schedule (tuned once, only ever validated at
11-46M params), measurably narrows or widens the capacity-inverse gap
observed in this campaign — cheaply, at CIFAR scale, before committing
ImageNet budget to either the base mechanism or this refinement.

**GPU-hours (est.)**: reuses this project's own just-measured per-run costs
(plain-AT ≈2.5-3h, ADR-family ≈3h per 200-epoch run on one RTX 4090;
AutoAttack evaluation ≈1-3h per checkpoint, this campaign's own measured
range). A minimal design — vanilla ADR vs. the reliability-adaptive variant,
both architectures already validated here (ResNet-18, MobileNetV2), 3 seeds
each — is 12 new training runs (~36 GPU-hours) plus their evaluations
(~48 GPU-hours), **≈85-100 GPU-hours total**, roughly 1-2 days wall-clock
across the 5 available GPUs at a sane (post-packet-0010) concurrency level.
A cheaper first pass (a post-hoc analysis of this campaign's own already-
logged per-epoch EMA/student agreement metrics, zero new GPU-hours) can check
the mechanism's core premise before spending any of the above.

**Preregistered decision rule**: the reliability-adaptive variant's mean
AutoAttack gain over its own arm's baseline must exceed vanilla ADR's gain
(from this campaign's own record, +1.22 pp ResNet-18 / +2.79 pp MobileNetV2)
by more than the combined per-arm sd (~0.5 pp) to count as an improvement,
matching this packet's own sign-confirmation logic above.

**Risks**: the mechanism is new and unreviewed — it needs scientific review
before any source SHA freeze (touches `src/ard/engine/trainer.py` and the ADR
objective), and there is no guarantee the reliability signal is easy to
define or measure well; a null result here is itself useful (rules out one
refinement direction cheaply) but does not on its own answer the ImageNet
question either.

## Option C — pause on new scientific work

**What it would establish**: nothing new; it is the option of closing
already-open items (packet 0010, the environment cutover, plan 0097's
remaining forensic housekeeping) before starting anything else. 0 GPU-hours,
real but bounded engineering/process time.

## Recommendation

**B.** The evaluation-phase incident just closed out (`docs/debugging/0029`,
packet 0010) demonstrated concretely how expensive and hard to debug a
multi-day CIFAR-scale campaign already is — ImageNet-scale iteration on an
unreviewed, untested mechanism would multiply that cost and latency
substantially, per Option A's own unresolved throughput-costing problem.
Option B answers a real open question (does this project's own new mechanism
idea improve on the base method this campaign just validated) using
infrastructure and cost data already in hand, at a small fraction of Option
A's cost and risk, and its cheapest first step (the post-hoc log analysis)
costs nothing at all. This recommendation would change if the post-hoc
analysis found the core premise (student-EMA divergence tracking onset of
memorization) does not hold in this campaign's own data — in that case,
Option A or C would be better than spending GPU-hours on a mechanism whose
premise already failed a free check.

Packet 0010 should also be resolved before either A or B launches any new
AutoAttack evaluation, since both need them.
