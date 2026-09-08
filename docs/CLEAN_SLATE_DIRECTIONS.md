# Clean-slate direction selection, 2026-09-09

Status: hand-assembled from a multi-agent pipeline whose automatic synthesis step never
completed (five session-limit interruptions in one day). What follows is the raw pipeline
output, cross-checked by hand, not a machine-generated summary. Two citations were
independently re-verified against the live web on 2026-09-09; everything else marked
VERIFIED was verified by an agent, not by a human, and should be treated accordingly.

The brief given to the pipeline: propose adversarial-defense research programmes for a
lab with five RTX 4090s, ImageNet on local disk, a bit-deterministic training engine and a
measured noise floor — blind to any of this project's own past experimental results, and
without a stated deadline (person-weeks were estimated honestly, not fitted to a calendar).

---

## 0. What actually ran

| Stage | Slots | Completed | Notes |
|---|---|---|---|
| Survey (7 areas) | 7 | 7 | see `docs/RESEARCH_RESTART.md` §survey and the prior session's report for content |
| Programme proposals | 8 | 9 (one slot re-ran) | 8 distinct programmes carried forward |
| Prior-art examination | 16 (8×2) | 16 | every one returned the boolean `preempted: false` — **the boolean is not trustworthy; read the free text** |
| Judging (4 lenses × 8 programmes) | 32 | 34 (two lenses re-ran) | scores averaged per programme below |
| Red team (3 finalists × 2 assassins) | 6 | 5 recovered (1 lost) | 4 of 5 verdicts were `killed: true` |
| Synthesis | 1 | **0** | never completed; this document is the substitute |

---

## 1. The eight programmes

| # | Title | One sentence |
|---|---|---|
| P1 | Cost-to-Break: Attacker-Budget Curves for Artist Protections | Per-image attacker spend to strip Glaze-class protections, against a measured noise floor |
| P2 | Cloak Budget Curves | Same territory as P1, independently generated |
| P3 | Teacher or Data? (noise-floor accounting, ResNet-18) | Does a robust teacher or the same compute spent on generated data buy more CIFAR-10 robustness |
| P4 | Teacher or Data? (small models) | Same question, independently generated, different phrasing |
| P5 | Does the Teacher Buy Anything? | Same question again, ResNet-18 + MobileNetV2 |
| P6 | Does the Teacher Pay for Itself? | Same question extended from CIFAR-10 ResNet-18 to int8 ImageNet MobileNets |
| P7 | Teacher, Init and int8: Robust ImageNet at Mobile Scale | ImageNet-scale version, teacher vs plain adversarial fine-tuning, int8 survival |
| P8 | Keyed Deployment | Does Gmail's secret-key input-scrambling trick (the shipped Magika fix) generalise to image classifiers, and at what query budget does it break |

**The "blind to past work" generation converged hard**: two near-duplicate artist-protection
programmes (P1/P2) and four near-duplicate teacher-vs-data programmes (P3/P4/P5/P6). Only
P7 and P8 are structurally distinct from something else on the list.

---

## 2. Judge scores (0–10 per axis, 4 lenses averaged; ranked by total)

Lenses: **deployment** (a sceptical engineer who ships models under attack — applies the
student's own "does anything change" test), **novelty-lit** (a hostile examiner who knows
the literature), **tract-ops** (the person who has to actually run this alone, on five
4090s), **pessimist** (assumes the main experiment returns null).

| Rank | Programme | deploy | novelty | tract | robust | legib | **Total /50** |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | P3 Teacher or Data? (noise-floor) | 3.75 | 4.75 | 7.50 | 7.75 | 8.25 | **32.00** |
| 2 | P6 Does the Teacher Pay for Itself? | 4.50 | 5.00 | 5.75 | 8.00 | 8.00 | **31.25** |
| 3 | P5 Does the Teacher Buy Anything? | 3.00 | 4.00 | 7.75 | 8.00 | 8.00 | **30.75** |
| 4 | P8 Keyed Deployment | 4.25 | 4.00 | 6.75 | 7.00 | 8.00 | **30.00** |
| 5 | P4 Teacher or Data? (small models) | 2.88 | 4.00 | 7.00 | 7.25 | 8.00 | **29.13** |
| 6 | P1 Cost-to-Break | 4.00 | 5.00 | 4.50 | 7.00 | 8.25 | **28.75** |
| 7 | P7 Teacher, Init and int8 | 3.75 | 3.50 | 5.75 | 7.25 | 8.00 | **28.25** |
| 8 | P2 Cloak Budget Curves | 2.75 | 3.75 | 3.75 | 7.00 | 8.25 | **25.50** |

**Read this table with two caveats before drawing any conclusion from it.**

First, **legibility (~8) and robustness (~7–8) are constant across all eight programmes** —
they carry no discriminating information; every proposal is well-written and every proposal
survives a null result on paper. The entire ranking is decided by three axes:
deployment (2.75–4.50), novelty (3.75–5.00), tractability (3.75–7.75). **No programme
scored above 5 on deployment or above 6 on novelty, from any judge, on any programme.**

Second, **the top four are a statistical tie.** The gaps are 0.75, 0.50, 0.75 points of 50,
and a single re-run of one judging lens on P4 moved its total by 0.28 points. Score alone
does not decide anything here; prior art and the red team do.

---

## 3. Prior art, read past the boolean

All 16 examiner verdicts returned `preempted: false`. Read as free text, the picture is
completely different: almost every programme has a "the headline dies, a narrower core
survives" verdict.

### P1 / P2 — artist protections

Both cite **Hönig, Rando, Carlini & Tramèr, "Adversarial Perturbations Cannot Reliably
Protect Artists From Generative AI", ICLR 2025 (arXiv:2406.12027)** as the closest work
(no seeds, no error bars, no attacker-cost axis — confirmed by both examiners).

The more damaging find: **an examiner named "IPV-Bench" (arXiv:2603.26154), "Benchmarking
Image Protection Methods under Diverse Image-to-Video Generation Scenarios."** Re-verified
directly on 2026-09-09: **the paper is real.** Its actual authors are **Xiaofeng Li, Leyi
Sheng, Yifan Zhao, Zhen Sun, Zongmin Zhang, Jiaheng Wei, Xinlei He**, submitted 27 March
2026. It already measures a per-model noise floor (95th percentile of FVD between two
bootstrap resamples of the unprotected set) and credits a protection only above that floor
— "the exact rhetorical move" both P1/P2 proposals were built around.

**However, one examiner's citation of this same real paper is wrong**: he wrote "Fang et
al. (HKUST-GZ / Wuhan Univ.)... 19 Aug 2026" — both the author name and the date are
incorrect for the paper he was pointing at. This is not "the paper doesn't exist"; it is
"an agent attached fabricated metadata to a real paper's real arXiv ID." Treat every
examiner-supplied author/date pair as unverified until independently checked, even when
the arXiv ID resolves.

A second examiner named **Pleimling et al., "Off-The-Shelf Image-to-Image Models Are All
You Need To Defeat Image Protection Schemes", SaTML 2026 (arXiv:2602.22197)** — corroborated
by five independent agents with consistent details (six protection schemes named) — showing
**the cost-free attacker already wins across six schemes**, which removes the "interesting
region" from any attacker-budget curve before a single run.

**What survives, per both examiners' own words:** the seed-replicated noise floor of the
mimicry-success metric itself ("no paper in this subfield has ever re-run an identical
mimicry pipeline under a different seed"), rescoped from ~2,300–3,100 GPU-hours down to
~250–600 GPU-hours and 10–15 person-weeks.

### P3 / P4 / P5 / P6 — the ARD-vs-data cluster

**The load-bearing disagreement.** Four examiners (across P3, P4, P5) independently
searched and found **no paper training an adversarial-robustness-distillation student on
diffusion-generated images** — the exact experimental cell three of these four programmes
are built on. Two other examiners (on P4 and P5) each named a *different* paper claiming
that cell is occupied:

- **Dong, Koniusz, Chen & Ong, "Adversarially Robust Distillation by Reducing the
  Student-Teacher Variance Gap", ECCV 2024** (the "STARSHIP" paper) — cited with a full
  table: *"CIFAR-10 ResNet-18 AA with 1M synthetic: ARD 51.04 (+1.64), IAD 50.30 (+1.21),
  RSLAD 52.42 (+1.06), AKD 51.34 (+1.23), AdaAD 53.08 (+0.72), STARSHIP 54.49 (+0.71),
  Ada-STARSHIP 55.75 (+0.83)"* — Table 5.
- **Dong, Koniusz, Chen, Z. Jane Wang & Ong, "Robust Distillation via Untargeted and
  Targeted Intermediate Adversarial Samples" (DARWIN), CVPR 2024** — cited with a different
  table: *"teacher TRADES (no DDPM) 82.45/48.90; students WITH 1M DDPM images: ARD
  82.89/53.41, RSLAD 82.05/52.60, ... DARWIN 84.13/55.92"* — Table 4.

**Re-verified directly on 2026-09-09: both papers are real.** ECCV 2024 confirmed via
Springer, ECVA and ACM DL. CVPR 2024 confirmed via the official CVPR virtual program and
CVF open access. The two papers share three authors (Dong, Koniusz, Chen, Ong) with Z. Jane
Wang added on the CVPR one — a plausible pair from one research group, not two names for
one hallucinated paper.

**What remains unresolved, and requires a human to open a PDF:** whether the *specific
table content* quoted (Table 5 of the ECCV paper, Table 4 of the CVPR paper) actually
contains those exact numbers, or whether an agent confabulated plausible-looking numbers
attached to a real paper and a real table caption. Both PDFs 403'd or otherwise resisted
every agent's fetch attempt; two other agents (an examiner and a judge) claimed to have
read one or the other directly, with conflicting confidence. **This is the single most
consequential open fact in this document** — if either table is real as quoted, the
"no ARD-on-generated-data study exists" premise underlying P3/P4/P5's novelty claims is
false, and the whole cluster needs to be rescoped down to the narrower, still-open cells
each examiner separately identified (a teacher-free control on the same generated data;
matched-compute / iso-GPU-hour budgeting; seeds and a noise floor; checkpoint-selection
discipline — none of which any cited table provides).

**Independent of that resolution, two other things were found and confirmed real:**

- **SAAD (Lee & Chung, KAIST, TMLR 2026, arXiv:2512.10275)** — quoted identically by six
  independent agents (a strong cross-agent consistency signal), Table 12: TRADES
  46.46±0.57, RSLAD 44.42±0.34, SAAD 50.34±0.08 on ResNet-18 (MobileNetV2 TRADES
  46.50±0.14). **This already falsifies the shared motivating premise** of all four
  ARD-vs-data programmes — "no published ARD method beats teacher-free AT on the same
  architecture" — with seeds and error bars.
- **Gowal2021Improving_R18_ddpm_100m** on the locally pinned RobustBench checkout
  (`.external/robustbench`, commit `78fcc9e4`, independently checkable offline): a
  **PreActResNet-18 at 58.63% AutoAttack with no teacher at all**, against the best
  published ARD ResNet-18 at roughly 53.45%. **The student's own question — can a
  lightweight model reach good robustness — is already answered, for free, and the
  answer is that the route that gets there is not ARD.**

### P7 — ImageNet mobile-scale

One examiner found the primary contrast (ARD vs teacher-free AT, ImageNet, sub-12M student,
4/255, AutoAttack) unoccupied. The other examiner killed it with a table the first missed:
**Kuang, Liu, Wu, Satoh & Ji, "Improving Adversarial Robustness via Information Bottleneck
Distillation", NeurIPS 2023, Table 5** — reporting the same contrast at ImageNet-1k scale,
answer: **yes, +1.71 pp**, the opposite sign the proposal predicted as most likely. Also
falsified: "no mobile-scale architecture has ever appeared" in this literature — RobustART
(Tang et al., TPAMI) already runs MobileNetV3, ShuffleNetV2 and more; and "first int8
AutoAttack number" — Thorsteinsson et al. (arXiv:2403.09441) already report a CIFAR-10
ResNet-18 at 85.64/58.08 after int8 post-training quantization with no fine-tuning.

**What survives**, both examiners agree: no AutoAttack number exists for any adversarially
trained **ImageNet** classifier after int8 post-training quantization. This became P6's
surviving headline (below).

### P8 — Keyed deployment

Two clean kills: **Rusu, Calian, Gowal & Hadsell, "Hindering Adversarial Attacks with
Implicit Neural Representations", ICML 2022 (arXiv:2210.13982)**, whose Figure 3 is
literally the flagship attack-success-vs-attacker-keys curve the proposal wanted to build;
and **Ali, Mohammed & Ahmad, "Evaluating Adversarial Robustness of Secret Key-Based
Defenses", IEEE Access 2022**, which already broke pixel-shuffling (7.45%), bit-flipping
(4.20%) and Feistel encryption (9.45%) adaptively. Also: **Tanaka, Echizen & Kiya, "On the
Transferability of Adversarial Examples between Encrypted Models", ISPACS 2022
(arXiv:2209.02997)**, Table VI/VIII, gives exact transfer numbers under a genuine
secret-weights threat model the proposal's own framing had overstated as unstudied.

**What survives, both examiners agree:** the secret-*weights* threat model specifically
(most keyed-defense work assumes weights are known and only the key is secret), the
extraction query-budget Q*, and a one-week, 10–15 GPU-hour replication of Tanaka's Table VI
numbers as a cheap kill point before committing further budget.

---

## 4. Red team verdicts (P3, P5, P6 — the three that reached this stage)

### P3 (Teacher or Data?, noise-floor accounting) — killed by both assassins

**Assassin A**: the primary contrast is a confound wearing a one-variable name. RSLAD's own
objective (Zi, Zhao, Ma & Jiang, ICCV 2021, arXiv:2108.07969, Eq. 3 — note the author list
is commonly miscited; the paper is real) evaluates the teacher **only on the clean image**
and has **no hard-label cross-entropy term**, so on generated data with no ground truth,
"does the teacher help" collapses into "are a 57%-AA model's soft predictions better
supervision than the generator's own labels" — not a distillation question. Separately, an
arithmetic error was found in the proposal's compute-matching (RSLAD needs one teacher
forward pass, not two — this changes the matched-epoch count by ~25%).

**Assassin B**: a power argument. The best available prior point estimate for exactly this
contrast (SAAD Table 10, Gowal2021 as teacher: RSLAD 46.61 vs teacher-free TRADES 46.46) is
**+0.15 pp** — against this lab's own measured five-seed endpoint SD (0.272–0.532 pp),
resolving 0.15 pp at 80% power needs **52–112 runs per arm**, i.e. 1,750–3,800 GPU-hours for
one contrast, more than the entire proposed budget. He also identified a **free** pre-screen
(one forward pass of the already-pinned Chen2021LTD_WRN34_10 teacher over the 1M generated
set, minutes of GPU time) that predicts the sign and most of the magnitude of the whole
520-GPU-hour phase this design is built around.

### P5 (Does the Teacher Buy Anything?) — one assassin killed, one did not

**Assassin A — killed.** The primary endpoint declares a 1.0 pp difference decisive, but the
control arm (in-house TRADES) carries a **known, unexplained, 1.2–1.5 pp bias in exactly this
harness**, in the direction that flatters the teacher (`docs/debugging/0028-trades-clean-target-detached.md`:
best-checkpoint AutoAttack moved 45.14 → 47.87 after one line was fixed; the remaining
1.2–1.5 pp gap to the 49.0–49.4 literature range is unexplained, and PGD-AT/RSLAD in the
same harness are exact). The design's error model budgets for variance and is structurally
blind to bias — the error class that actually dominates here.

**Assassin B — did not kill**, but narrowed the claim severely: the pitch is "measured
carefully enough that a one-point difference can actually be believed," and the lab
currently cannot believe a one-point difference in the arm every decision rule depends on.
Also found concrete engine blockers: no CutMix/AutoAugment/Mixup anywhere in the codebase;
MobileNetV2 exists as three unused lines in the model registry and has never been trained
once; per-class AutoAttack reporting is not implemented.

### P6 (Does the Teacher Pay for Itself?) — killed, with measurements taken on the lab's own hardware

This is the highest-value single result in the run, because it is the only verdict grounded
in **actual measurements taken on Hamster/Ferret**, not literature argument.

**Killed on two grounds.** (1) The design puts the distilled arm at exactly one compute
budget per stage and cannot locate a crossover — Busbridge et al., "Distillation Scaling
Laws" (ICML 2025, arXiv:2502.08606), the paper this design's own framing depends on,
establishes that distillation beats supervised training only below a compute threshold that
scales with student size; one point cannot find that threshold, so the design's most likely
outcome is a contradiction between its two stages rather than a verdict. (2) Three of
AutoAttack's four components need gradients and cannot run through a real quantized graph;
the design's proposed workaround (fp32 fake-quant simulation for the gradient attacks, and
only the weakest black-box component — Square — on the real int8 artifact) was measured to
fail outright: **the converted int8 model segfaults on `.to('cuda')` on this lab's hardware
(SIGSEGV, exit 139, reproduced twice)**, forcing CPU-only inference at 567–607 img/s, which
turns the "3 GPU-hours" budgeted for int8 AutoAttack into 12.2 **CPU**-hours per checkpoint
contending with the ImageNet dataloader on the same machine.

**Two objections this assassin tried to raise and explicitly retracted, after measuring:**
bit-determinism was tested directly (two independent runs of MobileNetV3-Large and
EfficientNet-B0 under a full PGD-3 training loop produced bit-identical state-dict hashes in
45 minutes — "it works"), and the local ImageNet data path was tested directly (1,281,167
images indexed in 2.1s on local NVMe, loader throughput up to 2,251 img/s at 8 workers).
Compute was found "wrong by ~1.4×, not 5×" — a correction, not a kill.

**The null that would empty the whole document**, per this assassin: the design never
calibrates its own teacher-free training recipe against a known-good reference (EasyRobust's
EfficientNet-B0, 61.83% clean / 35.06% AA at 90 epochs from scratch). Without that ~94
GPU-hour calibration arm, a disappointing result is indistinguishable from an under-trained
model.

**What survives, in the assassin's own words:** *"the seed-replicated, noise-floored cost
sheet for a robust mobile-scale ImageNet classifier — MobileNetV3-Large at 4/255, clean,
AutoAttack on 5k and full 50k, worst-class, ImageNet-C/R/A/V2, against the untouched
pretrained model. Nothing in the literature occupies it... it needs no teacher, so it is
immune to the SAAD strawman and the scaling-law objection." Recommendation: "cut the
teacher entirely."* On int8: report Square-only robustness of the real int8 artifact as an
*upper bound*, and white-box robustness of the fp32 fake-quant simulation as a *separate,
labelled* quantity — never conflate the two. Honest ceiling, same assassin: *"a competent,
correct, unsurprising engineering table... worth roughly one solid workshop paper and five
months."*

(One of the two P6 assassins never produced a verdict — session limit mid-run. Its slot is
unfilled; treat P6 as having received one, not two, adversarial reviews.)

---

## 5. Citations flagged for independent human verification

Ranked by how load-bearing they are, not by how suspicious they look.

**Resolved on 2026-09-09 (re-verified against the live web):**

1. **Dong, Koniusz, Chen & Ong, ECCV 2024** (the "STARSHIP" variance-gap paper) — **paper is
   real.** Confirmed via Springer, ECVA, ACM DL. Table-content (the exact numbers quoted)
   still unverified — a human needs to open `ecva.net/papers/eccv_2024/papers_ECCV/papers/00499.pdf`.
2. **Dong, Koniusz, Chen, Z. Jane Wang & Ong, CVPR 2024 (DARWIN)** — **paper is real.**
   Confirmed via the CVPR virtual program and CVF open access. Table-content likewise
   unverified — open `openaccess.thecvf.com/content/CVPR2024/papers/Dong_Robust_Distillation_via_Untargeted_and_Targeted_Intermediate_Adversarial_Samples_CVPR_2024_paper.pdf`.
3. **IPV-Bench, arXiv:2603.26154** — **paper is real**, title *"IPV-Bench: Benchmarking
   Image Protection Methods under Diverse Image-to-Video Generation Scenarios,"* authors
   **Xiaofeng Li, Leyi Sheng, Yifan Zhao, Zhen Sun, Zongmin Zhang, Jiaheng Wei, Xinlei He**,
   submitted 27 March 2026. **One examiner's citation of this real paper carried fabricated
   metadata** ("Fang et al. (HKUST-GZ/Wuhan Univ.)... 19 Aug 2026" — both the author and the
   date are wrong). Lesson: a correct arXiv ID does not guarantee correct surrounding
   metadata from the agent that supplied it.
4. **Frochte, arXiv:2605.09030** — **paper is real**, title *"When Style Similarity Scores
   Fail: Diagnosing Raw CSD Cosine in Artist-Style Evaluation."* Single-author attribution,
   which had looked unusual, turns out to be correct (Jörg Frochte).

**Still unresolved — a human should check before relying on the verdict that used them:**

5. **"Wu, Huang, Chen, Pang, Wang, 'Scaling and Taming Adversarial Training with Synthetic
   Data', ICCV 2025"** — the only live prior-art risk named against P3's Phase 2. Described
   with five independent identifiers (DOI, page range, institution, OpenAlex ID, DBLP key)
   by an agent who admitted every fetch attempt 403'd. That combination of precision and
   unreadability is the classic shape of a confabulation riding on a real-sounding DOI
   pattern; not yet checked directly.
6. **"RC-QAT: Efficient Robust Quantization via Attention-Guided Adversarial Distillation",
   CSCWD 2026** — named by two independent examiners (P6, P7) as sitting exactly on the
   int8-distillation intersection that is P6's surviving headline. Neither could retrieve
   it. Check before treating P6's "nothing occupies this ground" claim as final.
7. **"Di Mi et al., 'Rethinking Data Augmentation for Adversarial Distillation: An Excess
   Risk Perspective,' ICLR 2026 (rejected, OpenReview forum ZZGn6GXJDH, ratings 2/4/3)"** —
   this single citation is what killed P6's sub-clause about generated-data augmentation
   selection; the examiner admits he could read only the abstract, not the tables. An
   11-character OpenReview forum ID is trivial to check directly.

**Reassuring — cross-agent corroborated, low fabrication risk:**

8. SAAD (Lee & Chung, TMLR 2026, arXiv:2512.10275) — identical numbers quoted by six
   independent agents examining different programmes.
9. RobustBench anchors (Gowal2021Improving_R18_ddpm_100m, Rade2021Helper, etc.) — read
   directly from the locally pinned checkout, independently checkable offline.
10. Pleimling et al., SaTML 2026 (arXiv:2602.22197) — consistent scheme list (six named
    protections) across five independent agents.

---

## 6. Recommendation

**No programme clears a high bar. Every judge, on every programme, capped deployment_value
at 5 and novelty at 6.** The pipeline correctly refused to manufacture a strong winner where
none exists; do not read the ranking below as more confident than that.

**Top pick: P6, cut down to its surviving core only** — a seed-replicated, noise-floored
cost-and-robustness sheet for a mobile-scale ImageNet classifier (MobileNetV3-Large / int8),
with **the teacher cut entirely**. Reasons: highest deployment_value of any programme
(4.50); the only surviving clause that both its prior-art examiners *and* its assassin
independently converged on as the real, unoccupied ground; the only programme whose
feasibility was measured on this lab's actual hardware rather than argued (bit-determinism
confirmed in 45 minutes, ImageNet data path confirmed, compute overrun found to be 1.4× not
5×); and it depends on none of the disputed citations above. Mandatory changes before
launch: add the ~94 GPU-hour EasyRobust-calibration arm first; never report a single
"AutoAttack accuracy of an int8 model" number — report Square-only-on-real-int8 (upper
bound) and white-box-on-fake-quant (separate quantity) side by side; budget int8 evaluation
in CPU-hours, not GPU-hours. Honest ceiling, in the assassin's words: one solid workshop
paper, about five months, in a threat model with no recorded deployed victim.

**Close second, and the only structurally distinct idea in the set: P8**, reduced to its
~300–350 GPU-hour surviving shape (secret-weights threat model, extraction budget Q*,
invariant-subspace design rule). Best risk profile in the whole run: a one-week,
10–15 GPU-hour replication of a single published table (Tanaka et al. Table VI) either
confirms the mechanism or kills the whole programme for 1% of its budget. Never went to the
red team, so this is unverified against an adversarial reader. Its ceiling is narrow — the
beneficiary is a team that already open-sourced its own model.

**Do not run P3, P4, P5 as designed.** All three chase a contrast whose best available point
estimate (+0.15 pp) is smaller than this lab's own measured noise floor by roughly an order
of magnitude, and whose motivating premise is already falsified by a downloadable,
zero-GPU-hour fact: a PreActResNet-18 with no teacher scores 58.63% AutoAttack
(Gowal2021Improving_R18_ddpm_100m), while the best published ARD ResNet-18 anywhere is
roughly 53.45%. The student's own question is already answered, for free.

**Do not run P1/P2.** The interesting region of any attacker-cost curve for image-cloak
protections was already consumed by a March-2026 paper before a single run here, and the
critical path (IRB, artist consent, a closed Windows GUI applied by hand to hundreds of
images) runs entirely outside this lab's competence while its actual instrument — the
measured noise floor — sits off that critical path.

**Run these three things regardless of which direction is chosen — they are free or nearly
free, and every distillation-adjacent programme in this document depends on at least one:**

1. **Close the 1.2–1.5 pp TRADES gap** (`docs/debugging/0028-trades-clean-target-detached.md`).
   Three candidate causes are already named and ordered there. Until this closes, no ARD
   comparison from this harness with a threshold under ~2 pp means anything.
2. **The checkpoint-selection table** (best-on-held-out-validation vs best-on-test vs last).
   A within-run, within-arm contrast, immune by construction to the arm-specific bias that
   killed P5. Nearly free once any training grid exists.
3. **The 10-GPU-minute teacher pre-screen**: one forward pass of the already-pinned
   `Chen2021LTD_WRN34_10` teacher over the generated-image set, reading off label agreement
   and predictive entropy on synthetic vs real data. Predicts the sign and most of the
   magnitude of the entire 520-GPU-hour phase P3/P4/P5 are built around, for the cost of a
   coffee break.

**Before treating any of the above as settled**, a human should do two cheap things no
agent could: (1) open the ECCV 2024 and CVPR 2024 PDFs named in §3/§5 and read Table 5 /
Table 4 directly — the entire ARD-vs-data cluster's novelty claim hangs on whether those
specific numbers are really printed there; (2) spend ten minutes checking citations 5–7 in
§5, each of which was used to kill or narrow a specific clause while the agent using it
admitted it could not read the source.
