# Mobile-scale ImageNet robustness: what closes the small-model gap, and a proposed method

Date: 2026-09-09. Written autonomously overnight from four independent literature
investigations (each run separately, cross-checking the others' claims where they
overlapped) plus a direct read of this repository's engine code. No GPU job was
run. This document proposes a method and a staged test of it; it is not a launch
authorisation. Every external claim is marked **VERIFIED** (an agent read the
primary source directly — PDF, proceedings page, or a pinned local artifact),
**RELAYED** (from a search summary, not independently read), or **DISPUTED**
(sources disagree). Two of the four investigations found fabricated author lists
and fabricated table values in raw search results this session; every citation
below survived a second check before being included here.

---

## 0. The three questions asked, answered first

**Should the route avoid generated/synthetic data?** Yes, and this is not only a
compute-realism preference — there is direct evidence it is also the right
scientific call at mobile scale. RobustART (Tang et al., arXiv:2109.05211)
states, verbatim, that for light-weight architectures (EfficientNet, MobileNetV2,
MobileNetV3) **"increasing model sizes or using extra training data reduces
robustness"** — the opposite of what it does at large scale. **VERIFIED**, read
from the authors' own repository README. The generated-data lever that produces
the CIFAR-10 and large-ImageNet frontier is evidenced to be the *wrong* lever
specifically in the mobile regime.

**Is a teacher (distillation) good for small models?** The evidence says no, and
it is stronger evidence than existed for the CIFAR-10 case this project already
investigated. Three independent lines converge:

- Every one of eight distillation methods loses ground against **teacher-free**
  TRADES when the student shrinks from ResNet-18 (11.2M) to MobileNetV2 (2.3M) on
  CIFAR-10, in a three-seed, AutoAttack-evaluated table (SAAD, Lee & Chung, TMLR
  2026, Table 4). **VERIFIED**, table read directly from the PDF.
- The one ImageNet-scale table cited as pro-teacher evidence is not: read against
  its own second table, a fully adversarially trained teacher-free ResNet-50
  **beats** the distilled ResNet-50 in the same paper by 7.2 points AutoAttack
  (Kuang et al., "Improving Adversarial Robustness via Information Bottleneck
  Distillation," NeurIPS 2023, Tables 2 and 5). **VERIFIED**, both tables read
  from the NeurIPS proceedings PDF.
- There is now a mechanism, not just a correlation. Lee & Chung, "Toward
  Understanding Adversarial Distillation: Why Robust Teachers Fail," ICML 2026
  (arXiv:2605.21999), measures a "robustly unlearnable set" — training samples a
  capacity-limited student cannot represent robustly — at **8,979 of 50,000**
  images for MobileNetV2 versus **1,559** for WRN-34-10, a 5.8x difference, and
  proves that a confident teacher's supervision on exactly those samples drives
  the student to memorise noise rather than the robust signal. **VERIFIED**, read
  from the PDF. The mechanism predicts the harm gets *worse*, not better, as the
  capacity gap widens — which is exactly what moving from CIFAR-10 to
  ImageNet-1k mobile scale would do.

**This closes the door your earlier question opened.** Distillation is not the
route. The rest of this document does not use a teacher.

**Is there a small-model-specific approach?** Yes — one candidate has genuine
positive evidence, not merely an absence of prior work. See section 2.

---

## 1. The problem, quantified and re-framed

VERIFIED from the locally pinned RobustBench checkout (`.external/robustbench`,
commit `78fcc9e4`, read directly — no web dependency): the ImageNet L-infinity
leaderboard's smallest entry is a ResNet-18 at 52.92% clean / **25.32%**
AutoAttack (eps=4/255, Salman et al. 2020). **No MobileNet, ShuffleNet,
EfficientNet, or any sub-11M-parameter architecture appears on any RobustBench
L-infinity board, at any dataset.** This gap is total, not partial.

**One correction to the framing this project has been using.** That 25.32%
number is not evidence of a capacity ceiling. It is what a 2020 training recipe
gets (90 epochs, 3-step PGD, no clean pretraining — VERIFIED from Salman et al.'s
own appendix). Under a 2023 recipe, ConvNeXt-T (28.6M, barely bigger than
ResNet-50) reaches 49.46% AutoAttack — 14.5 points above ResNet-50 at nearly the
same parameter count (Singh, Croce & Hein, "Revisiting Adversarial Training for
ImageNet," NeurIPS 2023, arXiv:2303.01870). **VERIFIED**, cross-checked against
the pinned checkout. The parameter axis explains far less of the leaderboard's
ordering than the leaderboard's chronology suggests.

**The scaling-law evidence says the true capacity ceiling is real but shallow and
far away.** Debenedetti et al. ("Scaling Compute Is Not All You Need for
Adversarial Robustness," arXiv:2312.13131) fit robust accuracy ~ FLOPs^0.01 across
published models — roughly one point of AutoAttack per order of magnitude of
compute. Huang et al. (NeurIPS 2021, arXiv:2110.03825) found a 64x parameter
increase buys +4.45 points (PGD-20, not AutoAttack), with the curve flat above
roughly 17M parameters, and — the sharpest finding — that *reducing* capacity at
the network's last stage can *improve* robustness. **VERIFIED**, both read from
the primary text.

**Recipe, not architecture, is the larger term wherever it has been measured
directly.** Debenedetti, Sehwag & Mittal ("A Light Recipe to Train Robust Vision
Transformers," SaTML 2023, arXiv:2209.07399) hold a 26M-parameter architecture
fixed and change only epsilon warm-up, *lighter* augmentation (dropping
MixUp/CutMix/RandAugment), and a 10x higher weight decay: **+13.08 points
AutoAttack, from recipe alone.** The endpoint (41.78% AA) matches the pinned
RobustBench checkout exactly. **VERIFIED**, both the primary text and the
cross-check. Pang et al. ("Bag of Tricks for Adversarial Training," ICLR 2021)
independently show weight decay alone moving robust accuracy by roughly 7 points.
**VERIFIED.**

**Honest synthesis** (an agent's own words, and I concur after reviewing the
sources it named): roughly two-thirds of the small-model deficit is recipe and
architecture-allocation, one-third is a real but shallow capacity term. **The
largest gap in the literature is not a missing method — it is a missing
measurement.** Nobody has ever adversarially trained a genuinely mobile-scale
model (under ~5M parameters, ImageNet-1k, 224px) with a modern recipe and
evaluated it with AutoAttack. Every existing small-model number uses either a
stale recipe, a crippled baseline, or a proxy dataset (CIFAR-10/100,
Tiny-ImageNet at 64px).

**Correction, 2026-09-10: the throughput claim below is unverified and almost
certainly wrong; do not cost anything from it.** This document's own header
states "No GPU job was run" for the whole investigation, yet the paragraph
below claims a direct benchmark "this session" -- a plain self-contradiction,
of exactly the kind this document's own opening paragraph warns about
(fabricated values surviving into a claim marked confidently). No benchmark
script, log, or artifact backing this number exists anywhere in this
repository. A FLOP-based sanity check makes the number physically impossible
as an adversarial-training figure: ResNet-18 forward is ~3.64 GFLOPs at
224px; a 3-step PGD training step needs 4 forward+backward passes per image
(3 for the attack, 1 for the update), each ~3x a forward pass, so ~43.6
GFLOPs/image. At the claimed 5,208 img/s that is ~227 TFLOPS sustained --
about 2.75x an RTX 4090's ~82.6 TFLOPS BF16 peak, which no software could
reach. The number is plausible only as a *clean*-training (no attack)
figure: at ~10.9 GFLOPs/image (forward+backward, no attack), 5,208 img/s is
~56.8 TFLOPS, a believable ~69% of peak. If so, real 3-step adversarial
throughput is roughly 4x slower -- order **1,300 img/s**, making a 100-epoch
ImageNet-1k run (~1.28M images/epoch) closer to **~27 hours**, not 4, per
GPU for ResNet-18 (and correspondingly longer for MobileNetV2). **A real
benchmark on this hardware, with the actual intended attack step count and
224px ImageNet-shaped batches, is required before Stage 0 is costed or
launched; nothing below this line should be treated as measured.**

One investigation claimed to have benchmarked this repository's engine
directly, this session: ResNet-18 at 5,208 img/s and MobileNetV2 at 3,040
img/s per 4090 (batch size 256, bf16). At a 2-3 step attack (the step count
the cited modern recipes actually use), this would put a 100-epoch ImageNet
run at roughly 4 hours (ResNet-18) to 7 hours (MobileNetV2) per GPU -- a full
multi-seed study across five GPUs fitting in about a week. Per the
correction above, treat none of this as measured; it is preserved here only
so the original (unverified) claim is visible next to its correction, not
silently deleted.

---

## 2. The proposed method: teacher-free self-distillation, tested for a capacity-inverse effect

### 2.1 The evidence this rests on

Wu, Chen & Wang, "Alleviating Robust Overfitting in Adversarial Training" (ADR,
ICLR 2024, arXiv:2305.12118) proposes: maintain an exponential moving average
(EMA) of the student's *own* weights during training, anneal a temperature from
high to low over the schedule, and use the EMA copy's soft predictions as an
additional distillation target blended with the hard label. **No external
teacher. No extra data.** Overhead measured at **+1.83% wall-clock per epoch on
ResNet-18**, against +12.6% for AWP. **VERIFIED**, read from the PDF.

Their Table 2 reports the method across two capacity points (ResNet-18 ~11.2M,
WRN-34-10 ~46.2M), three datasets, held to identical recipe and seed count. Two
things are visible in the same table, and neither paper that reports either
number comments on the pattern:

| Gain over plain AT | ResNet-18 (~11.2M) | WRN-34-10 (~46.2M) |
|---|---:|---:|
| **Weight averaging** (a different mitigation, same table) | +1.12 to +1.32 | **+1.77 to +2.01** — helps the *larger* model more |
| **ADR self-distillation** | **+1.40 to +1.92** — helps the *smaller* model more | +0.90 to +1.13 |

**These two mitigations dissociate in opposite directions, inside one controlled
table.** Weight averaging's benefit grows with capacity; self-distillation's
benefit shrinks with it. **VERIFIED**, both halves of the table read directly
from the PDF (this is the single most decision-relevant number found across all
four investigations).

**This is the one candidate with positive directional evidence, not merely an
absence of prior work.** Every other angle investigated (SAM/AWP, robust NAS
under a mobile budget, structural re-parameterization) is either already
well-populated by prior work, structurally incapable of doing what it claims to
do, or simply unstudied with no evidence pointing either way. Self-distillation
is the only one where a controlled measurement already points in the direction
this document's question needs.

### 2.2 What has never been tested, and is the actual proposal

The ADR result spans exactly two capacity points, both on CIFAR-10/100/
Tiny-ImageNet (32-64px), a roughly **5x** parameter ratio (11.2M -> 2.3M would be
the comparable small point in their own line of work, per SAAD's use of the same
architectures). **Nobody has tested whether this capacity-inverse trend holds, or
strengthens, at the capacity gap ImageNet-1k mobile scale actually presents** —
roughly 11M (ResNet-18) down to 2-5M (MobileNetV3-Small, EfficientNet-Lite0), at
224px, on the real 1000-class task where the earlier finding (§0) shows the
teacher-based version of this problem gets *worse* with a wider capacity gap, not
better. Whether self-distillation's opposite-signed trend also strengthens, holds
flat, or reverses at this larger gap is a genuinely open, falsifiable question.

**The proposed method, stated as a testable claim:**

> Teacher-free EMA self-distillation's robustness margin over a matched,
> modern-recipe adversarially trained baseline grows monotonically as model
> capacity shrinks, across a width sweep from ResNet-18 (~11M) to a genuinely
> mobile architecture (~2-5M) on ImageNet-1k under AutoAttack — a capacity range
> roughly 3-5x wider than has ever been tested for this mechanism, in the one
> regime (ImageNet-1k, 224px, 1000 classes) where the analogous teacher-based
> mechanism is now known to move the opposite way.

Both outcomes are worth reporting. If the trend holds or strengthens: a genuinely
novel, small-model-specific, teacher-free, generated-data-free technique with a
plausible mechanism (this project's own state-conditional-intervention work
already established that per-sample state signals correlate with capacity-driven
failure, which is a compatible, though not identical, story). If it flattens or
reverses: that is itself informative, since it would mean the ADR CIFAR-10 result
does not generalise past a certain absolute or relative capacity, which nobody
has shown either.

### 2.3 Why this and not the two other candidates investigated

**Structural re-parameterization (train a multi-branch block, fold it into a
single cheap convolution before deployment) has zero prior art combined with
adversarial training** — confirmed independently, twice, across roughly 150
searched abstracts. But the mechanism as originally framed here overnight
("gives the small model free extra capacity") is wrong: the folding operation is
an exact algebraic identity for linear branches, so **the folded model computes
literally the same function it would have without re-parameterization — there is
no representational capacity gain, at any point.** **VERIFIED**, confirmed by
reading RepVGG's own derivation directly. A defensible version of this idea
survives only as a much narrower, more speculative claim about optimisation
dynamics (per-branch batch-norm statistics interacting with the adversarial
min-max objective differently under attack-mode versus eval-mode folding) — untested,
with a real risk of an effect size (RepVGG's own reported gains are order
2-3 points on clean accuracy) too close to this project's own measured
control-vs-control noise floor (0.16-1.88 pp) to resolve cheaply. This is kept as
an optional, lower-priority exploratory arm (§5), not the primary proposal.

**Robust neural architecture search under a genuine mobile FLOPs budget is a real
gap** (the field's smallest published budget is roughly 5 GFLOPs on 32x32 inputs
— about 10x a MobileNetV3 at 224px) **but is expensive, needs either an external
teacher (RNAS-CL, the one method spanning both worlds) or a full search loop, and
a null result is hard to publish.** Deprioritised for cost, not for lack of
novelty.

---

## 3. What has to be built

VERIFIED by reading the engine directly, this session:

- **Dataset**: `DatasetConfig.name` is a closed `Literal["synthetic_cifar",
  "cifar10", "cifar100", "tiny_imagenet"]` (`src/ard/config/schema.py:259`).
  ImageNet is not wired in. This is the load-bearing engineering item;
  everything else is smaller.
- **Architecture**: `build_architecture` (`src/ard/models/registry.py:116`)
  already has a `mobilenet_v2_cifar` entry (torchvision's `mobilenet_v2`, no
  pretrained weights, CIFAR-sized) as a working precedent. `timm==1.0.9` and
  `torchvision==0.26.0` are already pinned dependencies
  (`requirements/environment.lock`), so MobileNetV3-Small/Large and
  EfficientNet-Lite architectures are a registry-entry addition, not a new
  dependency.
- **Objective**: the existing `RSLADObjective`
  (`src/ard/objectives/rslad.py`) already factors the KD term through
  `target_to_student_kl`, the same primitive the corrected TRADES objective
  uses. A self-distillation objective needs (a) an EMA copy of the student's
  weights maintained alongside the optimiser step — there is no existing EMA
  machinery in the trainer (`student_ema_decay` exists as a config field but is
  currently unused for this purpose; the only EMA-like machinery in
  `trainer.py` is a per-sample *margin* EMA for the state-conditional work,
  unrelated), and (b) a temperature-annealing schedule over training epochs,
  which the engine's schedule primitives (`identity`, `multistep`) do not
  currently express and would need extending, matching what `CLEAN_SLATE_DIRECTIONS.md`
  already flagged as missing for a different reason.
- **This touches the scientific core** (a new objective, new EMA state that
  must be checkpointed and resumed correctly, a new schedule primitive) and
  should go through the scientific reviewer before a source SHA is frozen for
  any campaign, per this project's standing rule.

None of this needs generated data, a teacher checkpoint, or anything beyond what
is already on local disk (ImageNet, 146GB, VERIFIED present and fast to load —
2,251-3,235 img/s at 8-24 workers, measured in an earlier session).

---

## 4. A staged plan, cost from measured throughput

**Stage 0 — the floor measurement (this is a real, standalone contribution even
alone).** Adversarially train ResNet-18 and one mobile architecture
(MobileNetV3-Small or similar, ~2.5-4M params) on full ImageNet-1k with a
modern recipe (clean-pretrained initialisation, epsilon warm-up, lighter
augmentation than the classical AT recipe, tuned weight decay — following
Debenedetti/Singh's findings in §1, adapted to a 2-3 step attack for cost), no
teacher, no generated data. Evaluate with AutoAttack from a saved checkpoint on
the standard 5,000-image RobustBench ImageNet subset, then the full 50,000 for
the surviving arm. **This alone answers whether the field's small-model deficit
is a recipe artifact — nobody has this number.**

- Cost: **the "4 h / 7 h" figures below are unverified and likely wrong by
  roughly 4-7x** -- see the correction after §1's throughput paragraph. Do
  not schedule or promise anything from them; re-benchmark this engine on
  real 224px ImageNet-shaped batches under the actual intended attack step
  count before costing Stage 0. (Original, uncorrected claim, kept only for
  visibility: roughly 4 h (ResNet-18) to 7 h (mobile arch) per 100-epoch run
  per GPU; three seeds each across five GPUs fits in under two days of wall
  clock, plus AutoAttack (~1 GPU-h per checkpoint on the 5k subset, per this
  project's existing anchor).)
- Preregistered rule (to be confirmed by the reviewer, not asserted here):
  compare against Salman2020Do_R18's 25.32% as the stale-recipe floor and
  Singh2023's ConvNeXt-T 49.46% as the modern-recipe large-model ceiling; a
  result materially above 25.32% at mobile scale would already be worth
  reporting on its own.

**Stage 1 — the proposed method.** Add a teacher-free EMA self-distillation arm
identical in every other respect to Stage 0's recipe, across the same width
points (ResNet-18, one intermediate point if budget allows, and the mobile
architecture). Test whether the margin over Stage 0's baseline grows as capacity
shrinks, matching or refuting the CIFAR-10-scale ADR trend.

- Cost: **the "+1.83% wall-clock" EMA overhead figure below is also wrong**,
  now that plan 0097's real CIFAR-10 canary has measured it directly: EMA's
  per-step weight update is cheap, but ADR also runs a second, independent
  validation pass (student *and* EMA, each a full clean+PGD-20 forward)
  every epoch, and that pass dominates -- measured overhead is **~37% per
  epoch** (54s/epoch full vs ~39.4s training-loop-only, CIFAR-10 ResNet-18),
  not 1.83%. Stage 1 does not cost "essentially the same as Stage 0"; budget
  it at roughly 1.35-1.4x Stage 0's (corrected, still-unverified-for-ImageNet)
  training cost, on top of whatever the real Stage-0-scale ImageNet
  throughput turns out to be. (Original, uncorrected claim, kept only for
  visibility: EMA overhead is measured at +1.83% wall-clock (ADR,
  ResNet-18) — Stage 1 costs essentially the same as Stage 0, roughly
  doubled for the extra arm. Total across both stages, three seeds, two
  architectures: order 150-250 GPU-hours, well inside a few days of this
  lab's measured throughput.) Both the base ImageNet throughput and this
  EMA multiplier must be re-derived from real measurements before Stage 0
  or Stage 1 is costed.
- This is the honest headline experiment. It is small, it is cheap relative to
  every other direction this project has considered this month, and both of its
  possible outcomes (trend holds/strengthens, or trend flattens/reverses) are
  reportable.

**Stage 2 — optional, lower priority.** The re-parameterization arm from §2.3,
run only after Stage 0 has established this project's own noise floor at mobile
ImageNet scale (needed regardless, since the effect sizes at stake are close to
this project's previously measured 0.16-1.88 pp control-vs-control drift), and
only with a RepOpt-style optimiser-reparameterization control arm to rule out the
confound one investigation flagged.

---

## 5. What would have to be true for this to fail, and how it would be visible early

If Stage 0 already reaches near the modern-recipe large-model ceiling (unlikely
given the scaling-law evidence in §1, but possible), there is no capacity gap
left for Stage 1 to close, and the honest report becomes "recipe alone closes
the mobile-scale gap; no method contribution is needed" — itself a real, citable
finding, and cheaper than any other outcome considered for this project this
month.

If Stage 1's self-distillation margin does not grow with shrinking capacity at
ImageNet scale, that refutes the CIFAR-10-scale ADR trend's generality and is
worth reporting as a negative result with a plausible explanation already in
hand from §0: the ImageNet-1k, 224px, 1000-class setting may simply present a
capacity gap large enough that the "robustly unlearnable set" mechanism (§0,
ICML 2026) dominates regardless of which model provides the soft target, teacher
or self.

## 6. What was not resolved and needs a human check

Two items, both cheap and both flagged by more than one investigation:

1. **RC-QAT** ("Efficient Robust Quantization via Attention-Guided Adversarial
   Distillation," Chen, Wang & Zhang, CSCWD 2026, DOI
   `10.1109/cscwd68734.2026.11582282`) is confirmed real (via Crossref,
   Semantic Scholar and Unpaywall metadata, not web search — it has no arXiv
   version or open-access copy) but is teacher-based, so it does not occupy the
   ground this document proposes. If this direction ever moves toward
   quantised deployment, this is the paper to read behind the paywall first.
2. **GRAPE** (arXiv:2606.14865, June 2026) reports a progressive-width-growth
   technique for compact adversarial robustness with a similar spirit to §2.3's
   re-parameterization idea, but reports PGD-20 only (no AutoAttack) and its
   venue is unconfirmed. Worth a five-minute check before citing it as
   supporting or competing evidence for any future re-parameterization work.

---

## 7. Fact-check pass, 2026-09-10: which claims in this document survive independent re-verification

This document's own header warns that two of its four source investigations found
fabricated values, and this document has *already* needed two corrections above
(the throughput number and the EMA-overhead number, both proven wrong by direct
measurement or FLOP arithmetic this session). Given that track record, every
specific quoted number bearing on the core proposal (§2) was re-checked against
the primary source directly, independently of this document, before being relied
on further. Results:

**§2.1's central table (ADR paper's own Table 2, the capacity-inverse
dissociation between weight averaging and self-distillation) — VERIFIED,
accurate, and the strongest surviving piece of evidence in this document.**
Read directly from the arXiv HTML text a second time, independently. The real
numbers (AutoAttack gain over plain AT):

| Dataset | WA: R18 gain | WA: WRN gain | ADR: R18 gain | ADR: WRN gain |
|---|---:|---:|---:|---:|
| CIFAR-10 | +1.12 | +1.85 | +1.57 | +1.13 |
| CIFAR-100 | +1.32 | +1.77 | +1.92 | +0.90 |
| TinyImageNet-200 | +1.24 | +2.01 | +1.40 | +1.09 |

WA's gain exceeds ADR's gain-direction-with-capacity in every dataset one way,
and ADR's gain does the reverse in every dataset, with no exceptions — this is a
robust, three-for-three pattern, not a cherry-picked single number. (One
correction with no scientific consequence: the paper's actual title is
"Annealing Self-Distillation Rectification Improves Adversarial Training," not
"Alleviating Robust Overfitting in Adversarial Training" as an earlier draft of
this document had it — same arXiv id, same paper, same authors, title only.)

**§0's "distillation is not the route, the door is closed" conclusion —
PARTIALLY WRONG, overclaimed from a misread table. Correct before relying on
it further.**

- The Kuang et al. (NeurIPS 2023) "teacher-free ResNet-50 beats this paper's own
  distilled ResNet-50 by 7.2 points" claim: the 7.2-point subtraction is
  arithmetically real, but it compares two things the paper itself never
  compares — Table 5's "fast adversarial training" framework distillation
  ablation against Table 2's separately-sourced, full-AT, undisclosed-epsilon
  teacher checkpoint from unrelated prior work. **This is not evidence that
  teacher-free beats distilled under a matched protocol; withdraw it as support
  for anything.**
- The SAAD (Lee & Chung, TMLR 2026, arXiv:2512.10275) "every one of eight
  distillation methods loses to teacher-free TRADES going from ResNet-18 to
  MobileNetV2" claim: **false as stated.** Re-reading Table 4 directly: six of
  the eight lose (ARD, IAD, RSLAD, AKD, AdaAD, IGDM), but the paper's own two
  proposed methods (SAAD-C, SAAD) **beat** teacher-free TRADES at *both*
  capacities, including MobileNetV2 (SAAD 49.88% vs. TRADES 46.50% AutoAttack).
  The paper's entire point is that its adaptive-weighting scheme is the
  exception to the pattern the six baselines show — eliding that exception to
  claim "the door is closed" on all distillation is a real distortion of the
  source, not a defensible summary.
- What does survive: the *mechanism* paper (Lee & Chung, ICML 2026,
  arXiv:2605.21999, "robustly unlearnable set") is independently verified
  accurate down to the exact counts (8,979/50,000 MobileNetV2 vs. 1,559/50,000
  WRN-34-10) and the causal story (naive, uniform-weight teacher supervision on
  those samples drives noise memorisation). **This explains why naive,
  fixed-weight external-teacher distillation specifically struggles as the
  capacity gap widens — it does not show that all distillation is doomed.**
  SAAD's own result is the demonstration that a capacity-aware, sample-adaptive
  weighting scheme can route around exactly this failure mode.

**Corrected framing:** self-distillation (ADR) remains the primary, best-evidenced
lever for this project specifically because (a) its own capacity-inverse trend is
now confirmed accurate and robust across three datasets, not one, and (b) it is
already implemented, checkpoint-tested and CIFAR-10-validated in this repository
— not because "distillation in general has been ruled out." An adaptive-weighted
external-teacher method in the SAAD family is not eliminated as a direction; it
is a plausible, separately-evidenced *contingency lever* if plain self-distillation
does not close enough of the mobile-scale gap at ImageNet scale (see §8).

## 8. Refined proposal, 2026-09-10: recipe, pilot design, and a measure-then-improve structure

Folded in from this session's own recipe-extraction and literature work
(complementing, not replacing, §§2-4 above):

**Recipe base, not a Debenedetti-vs-Singh binary choice.** Debenedetti et al.
(SaTML 2023) and Singh et al. (NeurIPS 2023) are internally-coherent but mutually
exclusive bundles (scratch-init/weak-aug/high-WD/epsilon-warmup vs.
pretrained-init/heavy-aug/standard-WD/no-warmup) — confirmed by direct reading,
Singh's own ablations report heavy-aug-plus-random-init failing outright. Rather
than adopt one bundle whole, the components are chosen individually against
mechanistic literature:

- **Pretrained ImageNet-1k initialisation (Singh's choice)**: kept. Mobile
  architectures have off-the-shelf torchvision pretrained weights; pretraining's
  robustness-transfer benefit is supported broadly across AT literature, not just
  Singh's own paper, and this project uses no additional/generated data, which
  makes a free pretraining signal more valuable, not less.
- **Epsilon warm-up (Debenedetti's choice, absent from Singh's)**: added back in,
  on mechanistic grounds independent of either paper's own framing. Wu, Xia &
  Wang (NeurIPS 2020, arXiv:2006.08403) show AT specifically (not clean training)
  fails to converge below a capacity threshold via a "dead layers" /
  blocked-gradient mechanism under a constant-epsilon schedule, and demonstrate
  epsilon warm-up (their Periodic Adversarial Scheduling) as a working mitigation
  — directly relevant since this project's target models sit exactly in the
  capacity-constrained regime this paper studies.
- **Augmentation intensity**: reduced from Singh's full heavy stack (drop
  CutMix/MixUp; RandAugment light or off at first), on the same capacity-budget
  logic — heavy augmentation was validated only at ViT-S/ConvNeXt-T-and-larger
  capacity, and stacking it with an already-hard adversarial objective is the
  most likely single ingredient to cause outright underfitting at mobile scale.
- Both the epsilon warm-up and the reduced augmentation apply identically to the
  plain-AT baseline arm and the self-distillation arm, so neither can be mistaken
  for part of self-distillation's own effect — the preregistered comparison must
  isolate self-distillation specifically, not the capacity-rescue tricks that
  make training feasible at all.
- **Architecture**: do not treat off-the-shelf MobileNetV2/V3 width/depth as
  fixed. Huang, Le & Phung (NeurIPS 2021, arXiv:2110.03825, independently
  reconfirmed this session) found *reducing* capacity specifically at a network's
  last stage can *improve* robustness — a candidate secondary ablation axis, not
  the primary claim, since redesigning the architecture risks scope creep beyond
  what a first ImageNet-scale test needs.

**Small-scale pilot: not ImageNet-100, and not judged by early-training curves.**
This session's own literature check found ImageNet-100-style class-reduced
subsets are near-absent in AT-specific literature (one isolated instance, Kireev
et al. 2022, for an unrelated corruption-robustness ablation) and **no paper
tests whether a subset-tuned AT recipe transfers to the full 1000-class problem**
— adopting it here would import an unvalidated methodological risk this project
would be the first to lean on. Two further problems with the originally-discussed
pilot design were caught before being adopted:

1. Truncating epoch count breaks any recipe with a multi-epoch warm-up (both the
   epsilon warm-up above and Debenedetti's own LR warm-up run 10 epochs) —
   comparing configurations before their own warm-up completes measures "which
   one got through warm-up," not the recipe.
2. Comparing recipes by their early-training accuracy curve is confounded by
   initialisation strategy alone (pretrained-init starts far ahead of
   scratch-init regardless of the rest of the recipe) — since this project
   commits to pretrained init for all candidates up front (settled by literature,
   not by pilot), this confound is mostly moot, but the general principle (never
   rank configurations before their own schedule has completed) still applies to
   any remaining ablation axis (augmentation intensity, primarily).

**Chosen pilot design**: hold the full 1000-class label space and the full,
un-truncated epoch/warm-up schedule; reduce cost by subsampling *images per
class* (e.g. roughly 10-15% of the ~1,300/class average) rather than dropping
classes. This preserves classifier-head shape, class-count-dependent difficulty,
and the schedule's own temporal dynamics, while cutting wall-clock cost roughly
in proportion to the image reduction — and it aligns with machinery this project
already has (the existing `stratified_train_validation_split` label-aware
deterministic-subsetting pattern in `src/ard/data/datasets.py`), rather than
requiring a wholly new class-subset-filtering code path with no precedent in this
codebase or (per the literature check) validated precedent anywhere else in AT
research. The only open empirical question the pilot needs to resolve is
augmentation intensity (a small grid: none / light RandAugment / Singh's full
stack) — pretrained init and epsilon warm-up are pre-committed by mechanistic
literature, not treated as pilot variables — and configurations are compared
only at each one's own schedule-completion point, never on an early curve.

**Measure-then-improve structure, not measure-and-stop.** Naive transplant of
even a capacity-aware recipe onto a genuinely mobile-scale model under AT is
expected to underperform the modern-recipe large-model ceiling by a real, likely
large margin — that expectation is itself the reason this project's Stage 0/
Stage 1 split (§4) exists, but §4 as written stops at "measure the gap." Extend
it:

1. Stage 0 (plain-AT, capacity-aware recipe, mobile architecture): measure the
   gap against Stage 0's own ResNet-18 point, and against the modern-recipe
   large-model ceiling (Singh's ConvNeXt-T, §1).
2. Stage 1 (add self-distillation, identical recipe otherwise): measure how much
   of Stage 0's gap it recovers — this is the paper's headline, already-planned
   comparison.
3. **If Stage 1 does not fully close the gap** (the likely outcome, per the
   above), do not stop at reporting a partial recovery. Contingency levers, in
   the order this project can test them cheaply given what already exists or is
   evidenced above, each applied as an *additional* arm rather than a silent
   substitution:
   - Re-tune the EMA decay specifically for a smaller, noisier-gradient student
     (the 0.995 decay in the ADR paper was tuned for ResNet-18/WRN, not
     validated at mobile scale) — cheapest, reuses all existing engine code, no
     new method.
   - An adaptive-weighted external-teacher arm in the SAAD family (§7) — a real,
     separately-evidenced mechanism for exactly the capacity-gap failure mode,
     not yet implemented in this engine, meaningfully larger scope than the
     EMA-retune option.
   - The AdvXL-style coarse-to-fine (low-resolution/weak-attack pre-training
     stage, then high-resolution/full-attack fine-tuning) schedule, applied
     identically to every arm as an orthogonal efficiency/optimisation-dynamics
     layer, not a substitute for either mitigation above.

## 9. Red-team review, 2026-09-10: three independent adversarial passes on §§7-8

Per the user's explicit request, the refined proposal in §8 was reviewed
adversarially from three angles before any implementation work. All three
found real problems; one found that this whole document's *current* activity
is premature on process grounds, independent of its scientific content.

### 9.1 Effectiveness: the target effect may be too small to resolve, and the mechanism may not extrapolate monotonically

Re-reading §7's own table more carefully: the *direction* of the
weight-averaging-vs-self-distillation dissociation is robust (3/3 datasets),
but the **magnitude of self-distillation's own capacity-inverse delta**
(R18 gain minus WRN gain) is only **+0.44, +1.02, +0.31 pp** across the three
CIFAR datasets — mean ~0.6pp, mostly inside this project's own measured
0.16-1.88pp control-vs-control noise floor. A 2-3 seed ImageNet campaign, where
each seed is far more expensive than a CIFAR seed (so fewer seeds are
affordable, not more), is not obviously powered to resolve an effect this
small even before adding cross-dataset transfer uncertainty.

More seriously: Lee & Chung's own Figure 1b (read directly, not just the
counts cited in §7) shows self-distillation is **not immune** to the
mechanism that hurts external-teacher distillation — an EMA/self-teacher drawn
from an early, well-generalised checkpoint suppresses overfitting, but one
drawn from a later, already-overfit checkpoint **amplifies** it, exactly like
a bad external teacher. Robust overfitting is known to set in earlier at lower
capacity. If a mobile-scale EMA copy spends a larger fraction of the schedule
in its own "overfit, confidently-wrong" phase, ADR's cosine λ/τ schedule
(tuned once, at 11-46M, never retuned or re-validated below that) could push
the mechanism toward *amplifying* noise memorisation over more of training,
not less — a mechanistically grounded case for the trend **reversing**, not
merely flattening, below the tested capacity range. Separately, the
"unlearnable set" is still a *minority* of the pool at every capacity ADR's
own table covers (~9-32% by Lee & Chung's own numbers); ImageNet-1k's added
task complexity (1000 classes vs. 10, 224px vs. 32px) could push a mobile
architecture's unlearnable fraction past 50% — a majority regime neither paper
has tested, where the mechanism might break qualitatively rather than just
weaken.

**Implication**: treat the core hypothesis as higher-risk of returning
"consistent with zero" than §2/§8 currently frame it, regardless of which way
the true effect points. This does not kill the proposal, but it changes what
"success" should be defined as before launching anything — see §9.4.

### 9.2 Novelty: currently borderline, with a concrete fix

An adversarial reviewer-role pass judged the current framing as
**borderline-reject as scoped**: "does a known dissociation hold at a new
scale" reads as a measurement, not yet a contribution, unless paired with new
transferable design knowledge (the accepted precedents that clear this bar —
Singh/Croce/Hein 2023, AdvXL — earn it via new recipe rules that change
practice, not via a new number on a new architecture alone). Two concrete
fixes, both cheap relative to the planned budget:

1. **Add a naive fixed-weight external-teacher arm (reusing this project's
   existing `RSLADObjective`, no new code) at the same capacity points**,
   run alongside Stage 0/1. This directly operationalises §7's corrected
   finding (SAAD's adaptive weighting beats teacher-free TRADES specifically
   at MobileNetV2 scale by being sample-adaptive, while naive/fixed-weight
   distillation does not) and reframes the paper's claim from "self-distillation
   generalises" to **"which distillation-family mechanism wins as the capacity
   gap widens, and why"** — a mechanism-mapping contribution, not a replication.
2. **Reframe around the field's own AdvXL-vs-RobustART tension already
   documented in §0/§1**: AdvXL buys robustness with data/compute scale;
   RobustART's own finding is that *more data hurts small models*. The
   contrarian framing this project is positioned to make is not "we lack
   AdvXL's budget" but **"the scale-and-external-teacher lever is
   documented to be the wrong one precisely where deployment constraints
   (mobile, on-device) matter most — this maps what the right lever is in
   the regime a compute-rich lab has no incentive to study, because
   its own recipes don't transfer down."**

The §8 "measure-then-improve" contingency list (EMA retune, SAAD-style
adaptive weighting, AdvXL coarse-to-fine) should **not** be executed as a
sequential grab-bag tried one at a time after a failure — that reads as
incremental engineering, not a method. If it survives at all, it should be
unified as one hypothesis (capacity-aware, sample-adaptive distillation, with
EMA-self and SAAD-external as two operationalisations of the same idea) and
evaluated together, not as fallbacks.

### 9.3 Pilot design: three fixable problems, none fatal on their own

1. **Per-class image subsampling confounds augmentation intensity with data
   volume.** At ~130-195 images/class (10-15% of ~1,300), heavy augmentation
   is documented to underperform for reasons unrelated to capacity (RandAugment
   itself is tuned jointly with dataset scale; CutMix/MixUp are reported to hurt
   under data-scarce conditions elsewhere in the literature) — the pilot cannot
   distinguish "light augmentation wins because of mobile capacity" from "light
   augmentation wins only because the pilot itself is data-starved," since both
   push the same direction. Fix: re-run only the top-1/top-2 candidates at a
   second subsample fraction (e.g. 10% and 25%) and require the ranking to be
   stable before trusting it — multiplies only the contested arms, not the
   whole grid.
2. **A single run per candidate cannot support a hard ranking.** Fix: the pilot
   should produce a "not clearly broken" gate (reject only a candidate whose
   accuracy sits far outside the others, using this project's own noise-floor
   magnitude as the threshold for "far"), with literature/capacity-budget
   reasoning (already in §8) as the tie-breaker — a decision-rule change, not
   an added cost.
3. **The pilot only ever tests the plain-AT arm; its answer is silently
   inherited by the self-distillation arm without being checked.** Consistency/
   self-distillation losses are documented elsewhere to interact with
   augmentation strength differently than plain cross-entropy does. Since the
   ImageNet self-distillation objective does not exist yet (§3), the pilot
   genuinely cannot test this now — but should say so explicitly. Fix, at zero
   added cost until Stage 1 is already committed: when Stage 1 runs, add one
   extra arm at an adjacent augmentation setting for a fraction of a schedule,
   purely to check the Stage-0 ranking doesn't flip once rectified targets are
   in the loop.

### 9.4 The most important finding: this document's own current activity is ahead of this project's preregistered gate

Decision packet `docs/decisions/0009-mobile-robustness-direction-and-first-step.md`
is `status: decided`, `chosen: B` — **not** A, the full ImageNet campaign this
document (§§2-8) prepares for. Plan `docs/plans/0097-adr-cifar10-replication.md`,
which executes option B, has **M1c (launch the 20 CIFAR-10 jobs), M2 (verify
completion) and M3 (aggregate and apply the preregistered sign-match rule)
still open** as of this section's writing (2026-09-10) — the CIFAR-10 campaign
this project committed to running *before* any ImageNet work is still mid-flight.
Packet 0009's own preregistered rule is explicit: if the CIFAR-10 sign-match
does not hold, doubt the engine implementation before proceeding to ImageNet
scale at all.

**This means the ImageNet-track design work in §§7-8 above (recipe choice,
pilot design, the measure-then-improve structure) is legitimate preparatory
writing — it costs no GPU-hours and commits nothing — but implementation
(registering ImageNet in `DatasetConfig`, building the augmentation pipeline,
enabling AMP, running the augmentation-intensity pilot itself) should wait
until plan 0097 reaches M3 and packet 0009's sign-match rule actually
resolves.** Doing so now would mean building toward option A while the
human-decided gate for whether to do option A at all is still open — exactly
the sequencing the packet's preregistered rule exists to prevent. Nothing
below M3 changes by waiting; nothing is lost by doing so.

## Sources (all URLs from the four background investigations; re-verify before quoting a specific number)

- [RobustBench](https://robustbench.github.io/) / `.external/robustbench` (local pin, commit `78fcc9e4`)
- [Salman et al. 2020, "Do Adversarially Robust ImageNet Models Transfer Better?"](https://arxiv.org/abs/2007.08489)
- [Singh, Croce & Hein 2023, "Revisiting Adversarial Training for ImageNet"](https://arxiv.org/abs/2303.01870)
- [Debenedetti, Sehwag & Mittal 2023, "A Light Recipe to Train Robust Vision Transformers"](https://arxiv.org/abs/2209.07399)
- [Debenedetti et al. 2023, "Scaling Compute Is Not All You Need for Adversarial Robustness"](https://arxiv.org/abs/2312.13131)
- [Bartoldson et al. 2024, "Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies"](https://arxiv.org/abs/2404.09349)
- [Huang et al. 2021, "Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks"](https://arxiv.org/abs/2110.03825)
- [Pang et al. 2021, "Bag of Tricks for Adversarial Training"](https://arxiv.org/abs/2010.00467)
- [Tang et al., "RobustART"](https://arxiv.org/abs/2109.05211)
- [Kuang et al. 2023, "Improving Adversarial Robustness via Information Bottleneck Distillation"](https://proceedings.neurips.cc/paper_files/paper/2023/hash/233278d812e74a4f9848410881db86b1-Abstract-Conference.html)
- [Lee & Chung 2026, "Toward Understanding Adversarial Distillation: Why Robust Teachers Fail"](https://arxiv.org/abs/2605.21999)
- [Lee & Chung 2026, "Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation" (SAAD)](https://arxiv.org/pdf/2512.10275)
- [Wu, Chen & Wang 2024, "Alleviating Robust Overfitting in Adversarial Training" (ADR)](https://arxiv.org/abs/2305.12118)
- [RepVGG (Ding et al., CVPR 2021)](https://arxiv.org/abs/2101.03697)
- [Twin-Rep (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26027)
- [GRAPE (2026, venue unconfirmed)](https://arxiv.org/html/2606.14865)
