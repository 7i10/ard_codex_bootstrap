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
