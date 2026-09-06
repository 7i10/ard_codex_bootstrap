# Prior art: state-conditioned augmentation in adversarial robustness distillation

Written 2026-09-07. Read-only literature and code check for the proposal to give the richer
augmentation (`IDBH_WEAK`) only to training images in a chosen per-sample state and keep the weaker
one (`CropShift`) for the rest.

Evidence labels used throughout:

- **VERIFIED** — I read the paper text or the code and cite the table, equation, section or line.
- **REPORTED** — I relay a claim from an abstract, a search snippet or another paper's related-work section without checking it.
- **INFERRED** — my own reasoning from verified facts.

Terms used once and then assumed: *adversarial training* (AT) trains on inputs perturbed by an
attack; *adversarial robustness distillation* (ARD) is AT where a large robust teacher's outputs
replace or supplement the labels; *robust overfitting* is the fall of test robustness late in AT while
training robustness keeps rising; *per-sample state* is a discrete label computed for one training
image from the current student (and optionally the teacher), such as "correct under attack",
"correct but with small margin", "wrong under attack".

---

## Verdict

**Variant 1 — augmentation policy chosen by the student's own state — is not new as an idea.**
The nearest work is AROID (Li, Qiu and Spratling, *International Journal of Computer Vision*, 2024;
arXiv 2306.07197), written by the same group as IDBH. AROID learns an online, instance-wise
augmentation policy inside adversarial training, and it says so in its related-work section: "Our
proposed approach is also instance-wise, but contrary to existing methods tackles robust overfitting via
DA instead of robust regularization" (VERIFIED, Sec. 2, related work on AT). Its per-sample signal is a
continuous loss gap ("Vulnerability", Eq. 2) on the *augmented* image, not a discrete correctness state,
and it uses no teacher. A hand-built two-policy gate keyed on adversarial correctness or margin is
therefore a special, cheaper case of a family that already exists. Presenting it as a new method would
not survive a competent reviewer; presenting it as an ablation question ("does AROID's instance-wise
gain reduce to a binary gate on the student's adversarial state?") would.

**Variant 2 — augmentation policy chosen by the student–teacher relation — is new in the narrow
sense.** I found no paper that selects, weights or schedules *augmentation* from the teacher's and
student's joint behaviour on a sample in adversarial training or adversarial distillation. The idea
has, however, two clear ancestors that the thesis must cite and distinguish:

- *TST — "Teaching What You Should Teach: A Data-Based Distillation Method"* (Shao et al., 2023,
  arXiv 2212.05422; venue REPORTED as IJCAI 2023). Standard (non-adversarial) distillation. It
  trains an augmentation module to produce "augmented samples that the teacher is good at but the
  student is not, by minimizing the cross-entropy of the teacher and maximizing the cross-entropy of
  the student" (VERIFIED, Sec. 3.1). That is exactly the "teacher confident, student fragile" relation
  of Variant 2, realised as a learned global augmentation search instead of a discrete per-sample
  gate, and without any adversary.
- *TeachAugment* (Suzuki, CVPR 2022, arXiv 2202.12513). Standard training. Augmentation is
  optimised to be "adversarial for the target model and recognizable for the teacher model"
  (VERIFIED, abstract; objective Eq. 1; teacher is a pre-trained model or an exponential moving
  average of the target model, Sec. 3). Again teacher-bounded hardness, again no Lp adversary, again
  no discrete gate.

In adversarial distillation itself, the teacher already gates *losses and attacks* per sample in IAD
(ICLR 2022), DGAD (ECCV 2024) and SAAD (TMLR 2026), all VERIFIED below; none of them touches the
augmentation. So the exact combination "per-sample state × augmentation policy × robust teacher" is
unoccupied. How close is the nearest neighbour? TST is the same teaching principle in a different
regime; DGAD is the same gating mechanism on a different lever. Variant 2 is a recombination, not a
new principle. It is defensible as a master's thesis contribution only if the design can attribute any
gain to the teacher term, which means the student-only gate (Variant 1) and a random gate with the
same treated fraction must both be run as controls.

**The larger risk is not novelty; it is power.** Every per-sample ARD paper below claims 0.2–1.6 pp
from a single run, while three published re-implementations of the same RSLAD baseline with the same
teacher disagree by 3.04 pp (Section 3). This project's own measured floor is a per-run standard
deviation of 0.092 pp after the learning-rate decay and 1.14–1.25 pp before it
(`docs/POST_DECAY_FLOOR.md`). A per-sample augmentation effect will be in the same 0.1–0.5 pp
range as every other per-sample intervention this project has tried, so the experiment is feasible only
in the post-decay regime with at least five paired blocks (0.154 pp detectable at epoch 114 with five
blocks, `docs/POST_DECAY_FLOOR.md`, epoch-114 table).

---

## 1. Instance-wise or adaptive augmentation

The question is whether anyone has made augmentation strength or policy depend on the individual
example, and on what signal. The table separates the *lever* (what is changed per sample) from the
*signal* (what decides it), because a method keyed on loss magnitude is not the same as one keyed on
adversarial correctness.

| Work | Setting | Lever changed per sample | Signal | Reporting | Label |
|---|---|---|---|---|---|
| AROID (Li, Qiu, Spratling, IJCV 2024) | AT, CIFAR-10/100, Imagenette, ImageNet | Full augmentation policy (flip, crop, colour/shape, dropout heads) sampled from a policy network conditioned on the image | Three objectives on the augmented image: Vulnerability = loss under attack minus clean loss on the target model (Eq. 2); Affinity = loss shift on a separately pre-trained standard model (Eq. 5); Diversity regulariser. Continuous; no correctness gate; no teacher | Mean of 3 runs, no std in Table 1 | VERIFIED |
| IDBH (Li and Spratling, ICLR 2023) | AT | None: one fixed pipeline for every image | None | 3 runs, std in Appendix D.5 | VERIFIED (code: `IDBH.forward` applies the same `T.Compose` to every image, no epoch or state input, `.external/DA-Alone-Improves-AT/src/data/idbh.py`) |
| DAJAT (Addepalli et al., NeurIPS 2022) | AT | None per sample: every image gets both the simple and the complex augmentation, through separate batch-norm statistics, tied by a JS-divergence term (Eq. 5 sums over all i) | None | not checked | VERIFIED for the mechanism |
| FAT (Zhang et al., ICML 2020, arXiv 2002.11242) | AT | Attack strength: PGD stops early per sample | Whether the sample is already "confidently misclassified" under attack (abstract) | Median ± std over 5 trials (Sec. 5, Table 2) | VERIFIED |
| MART, GAIRAT, IAAT, CAT, MMA, LAS-AT | AT | Loss weight, per-sample epsilon or attack strategy | Misclassification, PGD steps to the boundary, margin, learned strategy | mostly single runs | REPORTED (from AROID related work and search summaries) |
| ISEAT (Li and Spratling, 2023, arXiv 2303.14077) | AT | Instance-wise loss smoothing | Adversarial vulnerability | not checked | REPORTED |
| CUDA (Ahn et al., ICLR 2023, arXiv 2302.05499) | Standard long-tailed training | Augmentation strength, **per class** | Whether the model currently predicts the class correctly: "CUDA increases and decreases the augmentation strength of the class that was successfully and wrongly predicted by the trained model" (Sec. 1) | not checked | VERIFIED |
| EntAugment (ECCV 2024, arXiv 2409.06290) | Standard training | Augmentation magnitude per sample | Softmax entropy of the model's own prediction; low entropy (easy, confident) samples get **stronger** augmentation, hard ones milder (Sec. 1) | not checked | VERIFIED |
| AdaAugment (arXiv 2405.11467, 2024–25) | Standard training | Magnitude per sample via an RL policy network | Real-time loss feedback from the target network | not checked | VERIFIED abstract |
| MADAug (Hou, Zhang, Zhou, ICCV 2023) | Standard training | Operator selection per image by a model-adaptive policy | Bilevel validation-loss objective; forms an easy-to-hard curriculum | not checked | VERIFIED abstract |
| AdaAug (ICLR 2022), InstaAug (NeurIPS 2022) | Standard training | Instance-wise policies | Learned | — | REPORTED |
| On-the-fly influence augmentation (Yang et al., 2025, arXiv 2510.00434) | Standard training | Strength per sample | Temporal variance of a gradient-projection influence estimate; stable samples get stronger augmentation | — | REPORTED |
| Implicit adversarial data augmentation (Zhou et al., IJCAI 2024, arXiv 2404.16307) | Standard / imbalanced training | Feature-space perturbation direction and count per sample | Class proportion sets the count; hard samples in major classes get adversarial rather than anti-adversarial directions | — | VERIFIED in part |
| RSDA (Rasheed et al., IJCIS 2023) | AT, MNIST/CIFAR-10 | Extra augmented copies of latent-space neighbours of each adversarial example | Latent neighbourhood, not the model's correctness on the sample | — | REPORTED (abstract only; full text behind a login) |
| DYNACL (ICLR 2023) | Adversarial contrastive pre-training | Global schedule from strong to weak augmentation | Epoch | — | REPORTED |

Three facts follow.

1. **No published method assigns a different augmentation policy to an image according to whether the
   model currently classifies it correctly under attack**, in AT or in ARD. The closest signal match is
   FAT (adversarial misclassification decides attack strength, not augmentation) and CUDA (correctness
   decides augmentation strength, but per class and without an adversary). The closest mechanism match is
   AROID (per-instance augmentation inside AT, driven by a continuous loss gap).
2. **AROID occupies the "instance-wise augmentation for adversarial robustness" slot** and reports gains
   over IDBH of +0.62 pp AutoAttack on CIFAR-10 WRN-34-10 (55.91 vs 55.29), +0.72 pp on CIFAR-100
   WRN-34-10, +0.32 pp on CIFAR-100 PRN18 and +1.62 pp on CIFAR-10 ViT-B/4 (VERIFIED, Table 1). Any
   student-state gate has to be positioned relative to it.
3. **The direction of the gate is contested by the standard-training literature.** EntAugment, MADAug and
   the 2025 influence paper all end up giving *stronger* augmentation to easy or stable samples and
   milder augmentation to hard ones. The proposal as stated gives the richer augmentation to fragile
   samples. IDBH's own hardness result (Section 4) says hardness beyond what the model can fit costs
   accuracy and eventually robustness. INFERRED: the opposite assignment — richer augmentation to
   safe-correct samples, weaker to fragile ones — is at least as well motivated, and both directions
   should be preregistered as arms rather than one chosen by intuition.

## 2. Teacher-guided augmentation

Has a teacher network been used to decide how to augment?

**In ordinary distillation, yes, three times, none per-sample-gated and none adversarial.**

- TST (Shao et al., 2023, arXiv 2212.05422). A "neural network-based data augmentation module with
  priori bias" learns "magnitudes and probabilities" (abstract) so that augmented samples are ones "that
  can be recognized by the teacher but not by the student" (Sec. 1). Stage II updates the learnable
  magnitude θm and probability θp "by minimizing the cross-entropy of the teacher and maximizing the
  cross-entropy of the student"; Stage III distils on the result; the two alternate (Sec. 3.1). The
  learned quantities are global lists ("Magnitude List", "Probability List", Fig. 1), applied through a
  per-image encoder. Datasets: CIFAR-100, ImageNet-1k, MS-COCO, Cityscapes. No adversarial
  robustness. I found no seed or standard-deviation statement in the text (VERIFIED absence by search
  of the extracted text for "seed", "±", "deviation").
- TeachAugment (Suzuki, CVPR 2022). Augmentation network aφ is updated to maximise target-model
  loss minus teacher loss on the same augmented image (Eq. 1); the teacher is an EMA copy or a
  pre-trained model (Sec. 3). Per-image generation, but no discrete gate, and the teacher's role is a
  bound on hardness, not a selector. Standard training only (VERIFIED).
- HARD (2023, arXiv 2305.14890). Generates augmented inputs on which teacher and student disagree,
  from spatial transforms up to VAE manipulations (REPORTED, abstract). Not Lp robustness.
- "What Makes a Good Data Augmentation in Knowledge Distillation" (Li et al., NeurIPS 2022) scores
  augmentation *types* by the standard deviation of the teacher's mean probability — a global choice,
  not per sample (REPORTED).

**In adversarial distillation, no.** What exists is teacher-gated *losses and attacks*:

- IAD (Zhu et al., ICLR 2022, arXiv 2106.04928) "distinguishes between three cases given a query of a
  natural data (ND) and the corresponding adversarial data (AD): (a) if a teacher is good at AD, its SL
  is fully trusted; (b) if a teacher is good at ND but not AD, its SL is partially trusted and the student
  also takes its own SL into account; (c) otherwise, the student only relies on its own SL" (VERIFIED,
  abstract). Lever: the soft-label weight.
- DGAD (Park et al., ECCV 2024, paper 09110) splits every batch by the teacher's clean correctness:
  x_ST = {x | argmax T(x) ≠ y} goes to a clean distillation loss, x_AT = {x | argmax T(x) = y} goes to
  adversarial distillation (VERIFIED, Sec. 4.2, Eq. 5). Lever: which loss and whether an attack is
  generated. Augmentation is "standard data augmentation" for all (Sec. 5.1).
- SAAD (Lee and Chung, TMLR 2026, arXiv 2512.10275) weights each sample by "the fraction of
  student-crafted adversarial examples that remain effective against the teacher" (VERIFIED,
  abstract). Lever: loss weight. Augmentation: "standard data augmentations (random crop and horizontal
  flip)" (Sec. 5.1).
- Curriculum-guided reliable distillation (Computers & Security, 2023) weights teacher versus student
  knowledge by teacher reliability on clean and adversarial inputs (REPORTED, abstract; full text behind
  a login).
- AdaAD (Huang et al., CVPR 2023), PeerAiD (Jung et al., CVPR 2024), IGDM (ICLR 2025) change the
  inner attack, the teacher, or a gradient-matching term; all use random crop and flip for every image
  (VERIFIED for AdaAD Sec. 4.1 and IGDM Sec. 5; PeerAiD REPORTED).

So the teacher–student relation has been used to choose *what to teach* (TST) and *which loss to
apply* (IAD, DGAD, SAAD), but never *which augmentation to apply* under an adversary. That gap is
real. It is also narrow: a reviewer will read Variant 2 as "DGAD's partition applied to IDBH's
augmentation", and the thesis must show that the augmentation lever behaves differently from the
loss lever on the same partition. This project already has the loss-lever data for that comparison
(`docs/ERT_RESEARCH_STATUS_SUMMARY.md`, Stage A and Clean-Wrong rows).

## 3. How the field measures per-sample ARD effects

For each paper: how many seeds, whether run-to-run variation is reported, and the size of the claimed
gain, on CIFAR-10 ResNet-18 under AutoAttack (AA) unless stated.

| Paper | Seeds | Variation reported | Claimed gain (AA, pp) | Label |
|---|---|---|---|---|
| RSLAD (Zi et al., ICCV 2021) | none stated | none | RSLAD 51.49 vs ARD 49.19 vs IAD 49.10 (best ckpt, WRN-34-10 TRADES teacher; Table 3) | VERIFIED |
| IAD (ICLR 2022) | none stated | none found | +0.20 over ARD in AdaAD's re-run (48.82 vs 48.62, Table 2); **−0.09** in RSLAD's Table 3 | VERIFIED |
| MTARD (ECCV 2022) | none stated | none | no AA column in Table 2 | VERIFIED |
| AdaAD (CVPR 2023) | none stated | none | +1.61 over RSLAD (50.06 vs 48.45, Table 2, WRN-34-10 teacher) | VERIFIED |
| PeerAiD (CVPR 2024) | none stated | none | PeerAiD 52.57 vs RSLAD 51.03 vs AdaAD 50.08 (Table 1, first student block) | VERIFIED numbers; student identity INFERRED from layout |
| DGAD (ECCV 2024) | none stated | none | +0.53 over AdaAD (50.59 vs 50.06, Table 4); the MAP partition alone "improved AA performance by 0.77%" (Sec. 5.2) | VERIFIED |
| IGDM (ICLR 2025) | none stated | none found | +0.97 on RSLAD (53.10 vs 52.13), +1.14 on AdaIAD (54.02 vs 52.88), LTD teacher (Table 3) | VERIFIED |
| SAAD (TMLR 2026) | "averaged over three random seeds" (Tables 4, 6) | no per-cell std | +5.40 over IGDM (50.34 vs 44.94, Bartoldson2024 teacher, Table 4) | VERIFIED |
| Why Robust Teachers Fail (Lee and Chung, ICML 2026) | 10 seeds | mean ± std in the analysis figures | analysis paper, no method gain | VERIFIED (lines "10 random seeds", "mean ± std over 10 seeds") |
| IDBH (ICLR 2023) | 3 runs | std in Appendix D.5, "no greater than 0.7" | IDBH[strong] vs IDBH[weak] end AA on PRN18: 49.99 vs 48.94 (+1.05, Table 1) | VERIFIED |
| AROID (IJCV 2024) | 3 runs | none in Table 1 | +0.32 to +0.72 over IDBH on CNNs (Table 1) | VERIFIED |
| FAT (ICML 2020) | 5 trials | median ± std, e.g. 46.13 ± 0.409 PGD-20 | — | VERIFIED |

Two comparisons put these numbers against noise.

**Cross-implementation spread of one baseline.** RSLAD with a ResNet-18 student and the same
WRN-34-10 TRADES teacher is reported at 51.49 pp AA by its authors, 48.45 pp by AdaAD and DGAD, and
51.03 pp by PeerAiD. That is a 3.04 pp spread for one method on one benchmark, larger than every
claimed gain in the table except SAAD's. IAD's sign relative to ARD flips between RSLAD's table
(−0.09) and AdaAD's (+0.20). None of these papers reports the variation that would let a reader tell
a re-implementation difference from a method difference.

**This project's floor.** Two identical runs differ by a standard deviation of 0.092 pp at epoch 114
after the learning-rate decay (pooled, 4 degrees of freedom, 95 % interval 0.055–0.265 pp) and by
1.14–1.25 pp before it (`docs/POST_DECAY_FLOOR.md`; `docs/ERT_RESEARCH_STATUS_SUMMARY.md`,
"Measurement limits"). Against the pre-decay floor, single-run claims of +0.20 (IAD), +0.53 (DGAD),
+0.77 (MAP alone), +0.97 (IGDM) and +0.32–0.72 (AROID) are indistinguishable from nothing. Against
the post-decay floor they would be resolvable, but only if the two arms share a parent, a data order
and a campaign, which no published comparison does. IDBH's own weak-versus-strong difference of
1.05 pp comes with a stated std of up to 0.7 pp per arm, so it is about 1.5 standard deviations of a
single arm — plausible, not established.

INFERRED consequence for the thesis: a per-sample augmentation gate will be judged by reviewers who
accept single-run 0.5 pp claims, but this project's own discipline forbids that reading. Preregister the
post-decay paired design, five blocks minimum, the CE-PGD20 validation endpoint and the AutoAttack
official-test confirmation exactly as I100 was run.

## 4. Why weak-then-strong works: what IDBH actually says

The augmentation this project switches to at epoch 100 is the upstream IDBH CIFAR-10 weak pipeline:
horizontal flip, `CropShift(0, 11)`, `ColorShape('color')`, `ToTensor`, `RandomErasing(p=0.5)`; the
strong variant differs only by `RandomErasing(p=1)` (VERIFIED,
`.external/DA-Alone-Improves-AT/src/data/idbh.py`, class `IDBH`, commit 38b740a). The project's
`CropShift` arm is flip plus `CropShift(0, 11)` alone (VERIFIED, `src/ard/protocols/__init__.py:152`),
and `EpochIdbhWeakTransform` reproduces the upstream weak pipeline with isolated RNG streams
(`src/ard/data/datasets.py:240`).

### What the paper claims about *why*

All quotations are from Li and Spratling, "Data Augmentation Alone Can Improve Adversarial Training",
ICLR 2023 (arXiv 2301.09879), VERIFIED against the PDF text.

Abstract: "the hardness and the diversity of data augmentation are important factors in combating
robust overfitting. In general, diversity can improve both accuracy and robustness, while hardness can
boost robustness at the cost of accuracy within a certain limit and degrade them both over that limit."

Definition of hardness (Sec. 3.1, Eq. 1): the ratio of a fixed adversarially trained model's PGD-50
robust accuracy on the original test set to its robust accuracy on the augmented test set. "It
increases as the augmentation causes the data to become easier to attack." Robust overfitting is
measured as "the best robustness minus the end robustness" (Sec. 3).

Hardness result (Sec. 3.1): "moderate levels hardness can alleviate robust overfitting and improve the
robustness but at the price of accuracy. Further increasing hardness causes both accuracy and
robustness to decline, even though robust overfitting is alleviated further. The value of hardness
where this occurs is very sensitive to the capacity of the model. Therefore, to maximize robustness,
hardness should be carefully balanced, for each model, between alleviating robust overfitting and
impairing accuracy."

The mechanism they offer (Sec. 3.1, discussion of Fig. 2): "Adversarial training can be considered as
standard training plus gradient regularization (Li & Spratling, 2022). Roughly speaking, accuracy drops
in this stage because the model's capacity is insufficient to fit the increasingly hard examples for the
optimization of the clean loss (standard training) under the constraint of gradient regularization.
Nevertheless, robustness could still increase due to the benefit of increasing robust generalization,
i.e., smaller adversarial vulnerability." And in the third phase: "the harm of decreasing accuracy now
outweighs the benefit of reduced robust overfitting, which results in the degeneration of robustness."

Diversity result (Sec. 3.2): three kinds are tested — type, spatial and strength diversity — and
"diverse augmentation generally can alleviate robust overfitting and boost both accuracy and
robustness". Cropshift is introduced because it has "enhanced diversity and disentangled hardness"
relative to pad-and-crop (Sec. 4). The design rule they draw: "adversarial training should have as
much diversity as possible and well-balanced hardness" (Sec. 1).

Evidence on the overfitting gap (Table 1, PRN18, AutoAttack, no SWA): baseline best 48.21 → end 42.46
(gap 5.74); IDBH[weak] 50.34 → 48.94 (gap 1.40); IDBH[strong] 50.74 → 49.99 (gap 0.75). Stronger
augmentation buys its end-of-training advantage mainly by shrinking the best-to-end fall.

### What the paper says about *when*

Nothing. The training setup is fixed for the whole run: "Models were trained by stochastic gradient
descent for 200 epochs with initial learning rate 0.1, divided by 10 at the epochs 100 and 150"
(Appendix C), with one augmentation from epoch 0 to 200. The code has no epoch or schedule argument
(VERIFIED, `idbh.py`; `train.py` exposes `--idbh` as a single choice). The word "schedule" in the paper
("the final augmentation schedule we have described may be suboptimal", Sec. 6, and Appendix E) means
the searched hyperparameter configuration, not a time schedule; do not cite it as support for
switching. The only temporal statement is the definition of the problem: "while performance on
classifying training adversarial examples improves during the later stages of training, test adversarial
robustness degenerates" (Sec. 1).

The paper also says nothing about per-sample or adaptive application; it applies one policy to every
image and lists more expensive automatic search as the only future direction (Sec. 6).

### What this gives I100 as an explanation (INFERRED)

Put the verified claims together with two facts from outside the paper: robust overfitting begins
right after the first learning-rate decay (Rice, Wong and Kolter, ICML 2020 — REPORTED), and this
project's own trajectories show that IDBH from epoch 0 (I0) ends below I100 and loses full-trajectory
AUC, while switching at epoch 50 produces a −5.34 pp shock that takes until epoch 52–62 to recover
(`docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md`, VERIFIED). The hypothesis consistent with all of it:

1. By IDBH's account, extra hardness pays an accuracy and fitting cost at all times and delivers a
   robustness benefit only through reduced robust overfitting.
2. Before the decay there is no robust overfitting to reduce, so early hardness is cost without
   benefit; that is why I0 and the early switches lose AUC and why the early switches shock.
3. From the decay onward the benefit exists, so hardness introduced there is nearly pure gain; the
   +1.0–1.2 pp final advantage of I100 over `CropShift` is the same best-to-end gap closure that
   IDBH's Table 1 shows, delivered only in the phase where the gap opens.
4. Diversity, which IDBH says helps at all times, is already supplied by `CropShift` throughout.

This is the starting explanation. It is not stated by IDBH, and it predicts something testable that the
timing screen already half-confirms: the gain should track the start of robust overfitting rather than
the calendar epoch, so it should move if the learning-rate schedule moves. It also bears on the
per-sample proposal. IDBH says the right hardness is set by capacity ("very sensitive to the capacity of
the model"); a per-sample margin is a per-sample proxy for spare capacity; the natural gate is therefore
"richer augmentation where the student has margin to spend", which is the S1 (safe-correct) population,
not the fragile one. That is the opposite of the proposal's stated direction and the same direction as
EntAugment and MADAug. Both directions belong in the preregistration.

Supporting scheduling literature, all REPORTED: DYNACL (ICLR 2023) anneals from strong to weak in
adversarial contrastive pre-training, the opposite direction, for a different objective; "Augmentation
Curriculum Learning" (OpenReview, RL) uses a weak phase then a strong phase; MADAug finds an
easy-to-hard curriculum emerges from bilevel optimisation in standard training; DAJAT (NeurIPS 2022)
uses an ascending epsilon schedule but a fixed augmentation pair.

---

## What the thesis must do to be distinct

1. Cite AROID as the existing instance-wise augmentation method for AT and state precisely what the
   proposed gate adds: a discrete state, no policy network, no affinity model, one switch of policy.
   Include `IDBH_WEAK` throughout and I100 as arms so the gate is measured against the global incumbent,
   not against `CropShift`.
2. Cite TST and TeachAugment as the teacher-guided augmentation ancestors and DGAD, IAD and SAAD as
   the teacher-gated loss analogues in ARD. State that the contribution is moving the gate from the loss
   to the augmentation under an adversary.
3. Run the student-only gate, the teacher-relation gate, and a random gate with the same treated
   fraction, from shared parents, after the decay, with five or more paired blocks. Without the random
   gate the effect of "who is treated" cannot be separated from "how many are treated".
4. Preregister the direction of the gate, or run both.
5. Report clean and robust accuracy separately, best and last checkpoints, and finish with AutoAttack
   on the official test set from a saved checkpoint, as I100 did.

## Sources

Local, VERIFIED: `.external/DA-Alone-Improves-AT/src/data/idbh.py`, `src/config/train.py`, `README.md`
(commit 38b740a); `src/ard/data/datasets.py`; `src/ard/protocols/__init__.py`;
`docs/POST_DECAY_FLOOR.md`; `docs/ERT_RESEARCH_STATUS_SUMMARY.md`;
`docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md`; `docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md`;
`docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md`.

Papers read in full text (PDF extracted): Li and Spratling, ICLR 2023, arXiv 2301.09879; Li, Qiu and
Spratling, IJCV 2024, arXiv 2306.07197; Park et al., ECCV 2024 (ecva.net paper 09110); Shao et al.,
arXiv 2212.05422; Suzuki, CVPR 2022, arXiv 2202.12513; Zhu et al., ICLR 2022, arXiv 2106.04928; Zi et
al., ICCV 2021, arXiv 2108.07969; Huang et al., CVPR 2023 (CVF open access); Jung et al., CVPR 2024,
arXiv 2403.06668; IGDM, ICLR 2025, arXiv 2312.03286; Lee and Chung, TMLR 2026, arXiv 2512.10275; Lee
and Chung, ICML 2026, arXiv 2605.21999; Zhao et al., ECCV 2022 (ecva.net 136640577); Zhang et al.,
ICML 2020, arXiv 2002.11242; Addepalli et al., NeurIPS 2022, arXiv 2210.15318; Ahn et al., ICLR 2023,
arXiv 2302.05499; EntAugment, ECCV 2024 (ecva.net 08301); AdaAugment, arXiv 2405.11467; Hou et al.,
ICCV 2023, arXiv 2309.04747; Zhou et al., IJCAI 2024, arXiv 2404.16307.

Abstract or snippet only (REPORTED): HARD, arXiv 2305.14890; RSDA, Int. J. Comput. Intell. Syst. 2023,
doi 10.1007/s44196-023-00266-x; curriculum-guided reliable distillation, Computers & Security 2023,
doi 10.1016/j.cose.2023.103433 (ScienceDirect S0167404823003218); Yang et al., arXiv 2510.00434; Li
et al., NeurIPS 2022, arXiv 2012.02909; DYNACL, arXiv 2303.01289; ISEAT, arXiv 2303.14077; MART,
GAIRAT, IAAT, CAT, MMA, LAS-AT, AdaAug, InstaAug, Rice et al. 2020.

---

## Independent check of the finding that kills variant 1

The claim that decides whether the student-state-gated variant is worth building
was checked directly against the paper rather than relayed.

**AROID, arXiv 2306.07197.** Its abstract states a policy-learning objective of
"Vulnerability, Affinity and Diversity" for "automatic DA generation during AT",
where Vulnerability targets the individual instances most susceptible to
adversarial attack.  That is per-example augmentation selection inside
adversarial training, driven by a per-example adversarial signal.  **VERIFIED by
reading the paper.**

So gating augmentation on the student's own adversarial state is not a new idea.
A hand-built two-policy gate is a coarse special case of a method that already
learns the policy online, published by the same group that produced the IDBH
augmentation this project switches to.

What survives is narrower and should be stated as such: **whether the
student-teacher relation carries information for this decision that the
student's own state does not.**  That question needs the student-only gate as a
control, which is now the AROID special case rather than a candidate method, and
a random gate treating the same fraction of images.
