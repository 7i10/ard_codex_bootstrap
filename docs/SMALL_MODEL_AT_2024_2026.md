# Small-model adversarial training, 2024 to 2026: does the 52.5 record still stand?

Date: 2026-09-07. Read-only survey that closes the RobustBench gap left open in `docs/ARD_VERSUS_AT_ASSESSMENT.md`. This is the only file written. No GPU job was run.

Setting throughout: CIFAR-10 test set, ResNet-18 (RN-18) or PreActResNet-18 (PRN-18), L-infinity, epsilon 8/255, AutoAttack (AA), no extra real data and no generated data unless a row says otherwise. Marks: **VERIFIED** = I read the paper text or the repository record and can point to the table; **REPORTED** = I relay a statement I could not check against the table myself; **INFERRED** = my reasoning from the other two.

---

## 0. The number

**The strongest no-extra-data RN-18 AutoAttack result I can find is 52.76 +/- 0.14, from DAT + AWP + SWA (Li, Li, Wu, Tian, Zhou, "DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain", NeurIPS 2024, Table 3). It beats 52.48 by 0.28 pp.** VERIFIED from the arXiv PDF text (2410.12307), Table 3 rows, caption "average experimental results ... in 7 runs".

The same Table 3 re-measures the two previous record holders under the authors' own protocol, RN-18, CIFAR-10, seven runs:

| method (as labelled in DAT Table 3) | PGD-20 | AA | mark |
| --- | ---: | ---: | --- |
| TRADES + AWP | 54.56 +/- 0.06 | 50.31 +/- 0.10 | VERIFIED |
| TRADES + AWP + SWA | 55.21 +/- 0.24 | 51.14 +/- 0.13 | VERIFIED |
| DAJAT (AWP + SWA + variable eps and alpha), one augmentation | 56.52 +/- 0.47 | 51.85 +/- 0.26 | VERIFIED |
| IDBH (AWP + SWA + variable eps) | 57.48 +/- 0.34 | 52.31 +/- 0.26 | VERIFIED |
| DAT + AWP | 58.57 +/- 0.14 | 52.54 +/- 0.12 | VERIFIED |
| **DAT + AWP + SWA** | 58.84 +/- 0.16 | **52.76 +/- 0.14** | VERIFIED |

What the number is and is not.

- It is a seven-run mean with a standard deviation, which is better evidence than the single-run RobustBench entry for DAJAT (52.48) and equal to the three-run IDBH result (52.46, SD at most 0.25). VERIFIED (DAT Table 3 caption; IDBH Tables 2 and 8 as recorded in the earlier assessment).
- The gap to IDBH is 0.30 pp against DAT's own IDBH rerun (52.31) and 0.30 pp against IDBH's published 52.46. With SDs of 0.14 and 0.26, that gap is about one standard deviation of the weaker arm. The correct statement is "DAT + AWP + SWA sits at the top of a 52.3 to 52.8 band that also contains IDBH + AWP + SWA and DAJAT", not "DAT is better". INFERRED.
- DAT uses no extra or generated images. The recombined images mix the amplitude spectrum of a training image with that of another training image (a "distractor drawn i.i.d. from the training set") and an amplitude generator trained jointly on the same data. VERIFIED (Section 3, "x0 is the distractor i.i.d. drawn from Dt"). Its generated-data results are WRN-28-10 only (Table 6, see section 3 below).
- Protocol: SGD, weight decay 5e-4, 5-step PGD with step 2/255 for training, random crop and flip only. Base runs are 150 epochs with decay at 100 and 110; the AWP and AWP + SWA runs use decay epochs 100 and 150, so they run at least 150 epochs and the total is not stated for them. Whether best or last checkpoint is reported is not stated in the text I searched. VERIFIED absence (Appendix E.2; grep for "best checkpoint" and "last checkpoint" finds nothing).
- Code: `github.com/Feng-peng-Li/DAT`, MIT licence, contains `train_cifar.py`, `Generator.py`, `utils_awp.py`. VERIFIED via the GitHub API. Not pinned in this repository.
- Cross-check against this repository: DAT Table 1 reports PGD-AT on RN-18 at 82.78 / 47.63 +/- 0.08 AA. This repository's PGD-AT best checkpoint is 47.63 AA (`docs/EXPERIMENT_DASHBOARD.md`). The two protocols agree to the second decimal, which supports treating DAT's table as a same-protocol reference. VERIFIED both numbers; INFERRED conclusion.

**Three claims above this band that do not meet the evaluation standard and must not be adopted as a baseline.**

| claim | reported AA on RN-18 | why it is excluded | mark |
| --- | ---: | --- | --- |
| DUCAT, dummy-class AT (Wang et al., arXiv 2410.12671 v2, May 2025, no venue found) | 58.61 (Table 1, PGD-AT + DUCAT) | Predictions of dummy classes are mapped back to the true class by a projection that sits outside the attacked graph, so standard AutoAttack counts a push into the dummy class as a success for the defence. DAWA ("Dummy-Aware Weighted Attack", arXiv 2603.29182, March 2026) reduces PGD-AT + DUCAT from 58.61 to 32.01 (100 steps) and 29.52 (1,000 steps), below plain PGD-AT. | REPORTED for both papers (tool summaries of the HTML and the DAWA abstract) |
| DCS, deformable convolution with stochasticity (Ma et al., ICML 2025, PMLR v267) | 73.41 +/- 2.04 (paper Table 1); 67.56 in the RobustBench submission issue #213 | A randomised defence evaluated with standard AutoAttack; the AA standard deviation over ten repeats is 2 pp; the authors write of their BPDA + EOT results "We attribute this to randomness in conjunction with gradient masking" (Section 5.3.6). RobustBench requires the randomised AutoAttack variant for such defences and has not processed the submission. | VERIFIED from the PMLR PDF text and the GitHub issue |
| LCAT, "logits constant amplitude training" (RobustBench issue #209, March 2025, closed) | 80.5 | No paper URL, no author list, no comment from maintainers; closed the same month. | VERIFIED issue text |

INFERRED: the reviewer who cites one of these has cited an evaluation failure, and the thesis should say so with the DAWA and Section 5.3.6 references above.

---

## 1. Where I looked

1. **RobustBench library repository** (`github.com/RobustBench/robustbench`), via the GitHub API. Last commit 2025-03-31, SHA `78fcc9e4`, identical to the checkout pinned in this repository. The only pull requests since are three maintenance PRs (2026-04 to 2026-05: `pkg_resources`, `gdown` bump, a README fix), none touching `model_info/`. The last merged model PR is #202 (2024-12-20), adding MeanSparse and ImageNet Swin models. Open model submissions: issue #213 (2025-11-04, DCS, no reply). Closed without merge: #209 (LCAT). VERIFIED.
2. **RobustBench website repository** (`robustbench.github.io`). Last model addition 2025-02-05; one cosmetic commit 2026-06-15. So the live page shows the same CIFAR-10 L-infinity list as the pinned checkout. VERIFIED. This confirms the user's statement that the leaderboard stopped after early 2025 and makes the pinned 99-entry list the complete leaderboard evidence.
3. **Pinned leaderboard content**, `.external/robustbench/model_info/cifar10/Linf/`. Every entry dated 2023 or 2024 is WideResNet-28-10 or larger (Wang 2023, Cui 2023, Peng 2023, Xu 2023, Bai 2023/2024, Amini 2024, Bartoldson 2024, Chen 2024). No RN-18 entry was added after DAJAT (2022). VERIFIED.
4. **Citation graphs** of DAJAT (77 citing papers) and IDBH (72) through the Semantic Scholar API, filtered to 2024 to 2026, every title read. The only citing paper that reports a higher RN-18 number is DAT. Others that touch RN-18 AT are fairness (DAFA, TRIX), long-tail (AT-BSL), fast AT, self-supervised AT, surveys, and application papers. VERIFIED titles; individual numbers checked where the title suggested a record (section 2).
5. **Citation graphs** of IGDM (6), PeerAiD (13), DGAD (0) and SAAD (0) for 2025 to 2026 distillation papers. VERIFIED (section 4).
6. **arXiv API** listings: all papers with "adversarial distillation" or "robust distillation" in the abstract, 2025 to 2026 (about 55 titles read); all papers with "adversarial training", "AutoAttack" and "CIFAR" in the abstract, 2025 to 2026 (6 hits, none an RN-18 record). VERIFIED.
7. **Semantic Scholar keyword sweep**, 2025 to 2026, 100 abstracts scanned for RN-18 percentages between 50 and 80. No credible RN-18 AutoAttack record; the numbers found were PGD-only (GRAPE, 56.94 PGD-20) or unrelated. VERIFIED.
8. **Papers read individually** for RN-18 or PRN-18 AutoAttack rows: DAT, CAT, SimpleAT, AROID, FOMO, CURE, ADR, ReBAT, CLAT, MEAT, TRIX, RAAT (CVPR 2026 Findings), GKL, DUCAT, DAWA, DCS, "Expanding the Role of Diffusion Models" (2026), dataset pruning (2024), Wang 2023, SCORE 2022, DKL 2023, the 2024 survey "Adversarial Training: A Survey", VanillaBench (2026). Results in the sections below.

Not reached: the full text of four journal papers behind IEEE paywalls (Mix2Aug, TDSC 2026; "Allies Teach Better Than Enemies", TPAMI 2026; InfoARD, TIP 2025; "Dual Adversarial Distillation With Hybrid Supervision", TMM 2026). Their abstracts (VERIFIED via Semantic Scholar) state no number. Mix2Aug is the only one of the four that could in principle hold an RN-18 AT record; its abstract claims "state-of-the-art accuracy and robustness" without a figure. REPORTED, unresolved.

---

## 2. Follow-up 1: did small-model AT stall, or did the field leave?

**Answer: the field left. The RN-18 frontier moved from 52.5 (2022 to 2023) to 52.8 (2024) and has not moved since. The papers that still train RN-18 in 2024 to 2026 mostly optimise something other than peak AutoAttack accuracy.** INFERRED from the two tables below.

RN-18 and PRN-18 results published 2023 to 2026 without extra data, best checkpoint unless stated:

| paper | venue, year | arch | AA | seeds | what the paper is about | mark |
| --- | --- | --- | ---: | ---: | --- | --- |
| DAT + AWP + SWA | NeurIPS 2024, Table 3 | RN-18 | 52.76 +/- 0.14 | 7 | frequency-domain augmentation for robustness | VERIFIED |
| SimpleAT (IDBH + weight averaging) | arXiv 2306.07613, 2023, Table 2 | RN-18 | 52.30 | 1 | square loss, one-cycle schedule, erasing | REPORTED |
| ReBAT++ | NeurIPS 2023 | PRN-18 | 51.49 best / 51.39 last | not stated | robust overfitting as a minimax imbalance | REPORTED |
| FOMO | ICLR 2024, Table 3 | PRN-18 | 51.37 best / 51.28 last | 3 | robust overfitting via periodic forgetting | REPORTED |
| ADR + AWP + WA | ICLR 2024, Table 2 | RN-18 | 51.18 | 5 | self-distilled soft labels for AT | REPORTED |
| CAT (two RN-18 trained jointly) | arXiv 2303.14922, 2023, Table 2 | RN-18 | 51.02 best / 49.64 last | 1 | collaborative training; also compares to RSLAD + AWP | VERIFIED |
| AROID | arXiv 2306.07197 v2, 2024, Table 4 | PRN-18 | 50.57 | 3 | learned online augmentation policy | REPORTED |
| CURE | ICLR 2024 | PRN-18 | 50.23 | not stated | accuracy-robustness trade-off, weight conservation | REPORTED |
| TRIX | arXiv 2507.07768, 2025, Table 1a | RN-18 | 49.09 +/- 0.24 (last-5 mean) | 5 | class-wise robust fairness | REPORTED |
| RAAT | CVPR 2026 Findings, Table 1 | RN-18 | 47.94 +/- 0.25 | 3 | clean-accuracy alignment; TRADES baseline 47.62 | REPORTED |
| CLAT | arXiv 2408.10204, 2024 | PRN-18 | 51.39 on top of a fast-AT baseline | 10 | parameter-efficient fine-tuning of critical layers | REPORTED |

Only one row (DAT) is above the 2022 to 2023 band. Five of the eleven are 2024 or later, and of those only DAT aims at peak robustness on a full-budget RN-18. INFERRED.

Where the effort went instead, with the evidence:

- **Generated data at WideResNet scale.** Every CIFAR-10 L-infinity entry added to RobustBench in 2023 and 2024 is WRN-28-10 or larger, and the top of the list is Wang 2023 (WRN-70-16, 70.69, 50 M EDM images), Peng 2023 (RaWRN-70-16, 71.07), Bartoldson 2024 (WRN-94-16, 73.71, a scaling-law study), Amini 2024 MeanSparse (75.28, a post-hoc activation sparsifier on Bartoldson's model). VERIFIED from the pinned `model_info`. DAT itself reports its generated-data results only on WRN-28-10 (Table 6). VERIFIED. The February 2026 diffusion-representation paper (arXiv 2602.19931) reports WRN-28-10 and ViT-B/2 only. REPORTED.
- **Post-hoc and ensemble tricks on large models**: MixedNUTS (TMLR 2024, ResNet-152 + WRN-70-16 mixing network, 70.08), MeanSparse. VERIFIED from `model_info`.
- **Randomised and structural defences with contested evaluations**: DCS (ICML 2025) and DUCAT (2024 to 2025), section 0. VERIFIED / REPORTED.
- **Other objectives on the small model**: robust overfitting (ReBAT, FOMO, MEAT), the clean-robust trade-off (CURE, RAAT, DUCAT), fairness (DAFA, TRIX), efficiency (CLAT, fast AT, the 2026 FastAT benchmark). REPORTED from the abstracts and tables listed above.
- **Larger inputs and other model classes**: the 2024 survey (arXiv 2410.15042) frames the recent directions as generated data, ConvNeXt and ViT robustness on ImageNet, and adversarial training of language models; its only small-model table (Table III, PRN-18) is about fast single-step AT and tops out at 48.04 AA. REPORTED. VanillaBench (arXiv 2607.12545, July 2026) benchmarks only the top-10 RobustBench models per track, all large. REPORTED.

Consequence for the thesis. A field that had overtaken the RN-18 numbers would have left a trail of 53s and 54s in the citation graphs of DAJAT and IDBH. There is one 52.76 with seven seeds, and nothing after it. The RN-18 no-extra-data regime is under-served, and the strongest recent result was obtained with the same ingredients as the 2022 to 2023 records (augmentation plus AWP plus weight averaging). INFERRED.

---

## 3. Follow-up 2: a 2024 to 2026 result above 52.5 on RN-18 with extra or generated data?

**Answer: none found. The number to state as the scope condition is still Gowal et al., NeurIPS 2021: PRN-18, 100 M DDPM-generated images, 87.35 clean / 58.63 AA (RobustBench entry `Gowal2021Improving_R18_ddpm_100m`).** VERIFIED from the pinned `model_info`. Next: Rade 2021 with extra real data, 57.67; Rade 2021 DDPM, 57.09; Rebuffi 2021 DDPM, 56.66; Sehwag 2022 proxy distribution, 55.54. VERIFIED, same source.

What I checked for a newer RN-18 generated-data number and what each contains:

| paper | RN-18 / PRN-18 with generated data? | what it does report | mark |
| --- | --- | --- | --- |
| DAT, NeurIPS 2024, Table 6 | no | WRN-28-10: 1 M images 91.37 / 64.25; 20 M images 92.86 / 68.18 | VERIFIED |
| Wang et al., ICML 2023 | no (searched full text incl. appendix) | WRN-28-10: 1 M 63.35, 20 M 67.31, 50 M 67.17 | REPORTED |
| GKL / IKL-AT, TPAMI 2025 | no | WRN-28-10 20 M: 67.75; WRN-34-10 no generated data: 57.09 | REPORTED |
| SCORE, ICML 2022 | no PRN-18 with DDPM data | RN-18 without generated data: 49.63 +/- 0.17 | REPORTED |
| DKL, 2023 | no | WRN only | REPORTED |
| dataset pruning for AT, arXiv 2406.13283, 2024 | no | WRN-28-10 on 1 M to 22 M images | REPORTED |
| "Expanding the Role of Diffusion Models", arXiv 2602.19931, 2026 | no | WRN-28-10 and ViT-B/2 | REPORTED |
| Bartoldson et al., ICML 2024 | no (earlier assessment, text search) | WRN-82-8 and WRN-94-16 | VERIFIED |

These rows are kept apart from section 0 and must stay apart in the thesis. The honest scope sentence is: "With generated data the small-model frontier is 58.6 AA (Gowal 2021, PRN-18, 100 M images); this thesis does not enter that regime, and no result after 2021 has been published for RN-18 in it, because the field moved that work to WideResNets." INFERRED wording from the VERIFIED and REPORTED rows above.

---

## 4. Follow-up 3: adversarial robustness distillation in 2025 and 2026

Every distillation paper found that post-dates the earlier survey's list (which ended at SAAD, TMLR 2026, and "Why Robust Teachers Fail", ICML 2026), with the teacher-free baselines it uses:

| paper | venue, date | student result on CIFAR-10 RN-18, AA | teacher-free baselines | post-2019 teacher-free baseline? | mark |
| --- | --- | ---: | --- | --- | --- |
| MMARD (arXiv 2503.06559) | arXiv, Mar 2025 | 84.63 / 52.50 with WRN-34-10 teacher, best checkpoint (Table 3); RSLAD row 51.49, TRADES row 49.23 copied digit-for-digit from RSLAD Table 3 | SAT, TRADES | no | VERIFIED (PDF text); copying INFERRED from the digit match |
| ProARD (arXiv 2506.07666) | IJCNN 2025 | one-shot training of many student sizes; single runs | SAT, TRADES (from the earlier assessment) | no | REPORTED |
| Anti-bias soft label distillation (arXiv 2506.08611) | Jun 2025 | fairness objective | not the point of the paper | no | REPORTED |
| AdaGAT (arXiv 2508.17265) | Aug 2025 | guide model is the RN-18; the target is WRN-34-10 (53.87 AA) | PGD-AT, TRADES, MART, FAT, GAIRAT, LBGAT | LBGAT (2021) is itself guided training, not teacher-free | REPORTED |
| DARD (arXiv 2509.11525) | Sep 2025 | 47.75 with a ResNet-56 teacher (Table 1) | SAT | no | REPORTED |
| CIARD (arXiv 2509.12633) | Sep 2025 | no AutoAttack reported | SAT, TRADES | no | REPORTED |
| FERD (arXiv 2509.20793) | Sep 2025 | data-free setting | out of scope | no | REPORTED |
| MMT-ARD (arXiv 2511.17448) | Nov 2025 | vision-language models | out of scope | no | REPORTED |
| InfoARD | IEEE TIP, Dec 2025 | abstract gives no number | abstract names none | unknown | REPORTED (abstract) |
| SAAD | TMLR 2026 | 50.34 +/- 0.08 (Bartoldson teacher), 50.35 +/- 0.22 (Gowal teacher), last epoch | PGD-AT, TRADES | no | VERIFIED (earlier assessment) |
| "Allies Teach Better Than Enemies" | IEEE TPAMI, Feb 2026 | abstract gives ImageNet gains only | abstract names none | unknown | REPORTED (abstract) |
| "Dual Adversarial Distillation With Hybrid Supervision" | IEEE TMM, 2026 | abstract gives no number; the paper cites DAJAT | unknown | unknown; the DAJAT citation is the only hint and cannot be checked without the full text | REPORTED (abstract; citation from Semantic Scholar) |
| "Why Robust Teachers Fail" | ICML 2026 | analysis paper, Table 2 single values | PGD-AT as a reference | no | VERIFIED (earlier assessment) |
| Improving Certified Robustness via AD (arXiv 2606.31653) | Jun 2026 | certified setting | out of scope | no | REPORTED |
| Information-bottleneck distillation, dual teachers (arXiv 2607.27737) | Jul 2026 | PRN-18, 51.89 (joint distillation) versus IBD 51.78 | AT, TRADES | no | REPORTED |

**Verdict: no 2025 or 2026 distillation paper compares against a teacher-free method newer than TRADES (2019).** The earlier assessment's claim that the fair comparison has never been made stands, with one qualification that should be written into the thesis: the comparison has been approached once, from the AT side and before the current records. CAT (arXiv 2303.14922, 2023, Table 5) trains a WRN-34-10 and an RN-18 jointly and sets the RN-18 against distillation students of the same TRADES WRN-34-10 teacher: ARD 49.19, IAD 49.10, RSLAD 51.49, RSLAD + AWP 51.62, CAT 51.72, single runs, baseline rows taken from the RSLAD paper. VERIFIED from the CAT PDF text. That is a distillation-versus-collaborative comparison, not distillation versus DAJAT, IDBH or DAT, and it has no seeds, so it does not close the gap. INFERRED.

Two further facts from this pass that bear on the ARD side of the comparison:

- MMARD's 52.50 (single run, best checkpoint, WRN-34-10 teacher) is the second published ARD number on RN-18 that lands inside the 52.3 to 52.8 teacher-free band, next to PeerAiD's 52.57. Together with IGDM's 54.02 (one run) and this repository's 53.0 to 53.2 (three seeds, official test), the ARD side is 0 to 1.3 pp above the teacher-free band, and only the repository's own numbers carry a measured spread. INFERRED from VERIFIED numbers.
- The teacher-free comparator that ARD papers do use keeps degrading: RAAT (CVPR 2026 Findings) reports TRADES on RN-18 at 47.62 +/- 0.17, DAT reports it at 49.37 +/- 0.08, RSLAD at 49.23. A thesis that reruns TRADES must land near 49.2 to 49.4 or explain why; the in-house 45.14 noted in the earlier assessment is still the first thing to fix. VERIFIED numbers; INFERRED requirement.

---

## 5. Recommendation

Name DAT + AWP + SWA (NeurIPS 2024, 52.76 +/- 0.14 over seven runs) as the strongest published teacher-free RN-18 result without extra data as of September 2026, and state in the same sentence that IDBH + AWP + SWA (ICLR 2023, 52.46, three runs) and DAJAT (NeurIPS 2022, 52.48) lie within about one standard deviation of it, so the teacher-free frontier is a band at 52.3 to 52.8 and not a single method. For the arm the thesis actually trains, adopt IDBH + AWP + SWA: its code is already pinned and licensed in this repository, its transforms are already re-implemented in `src/ard/data/datasets.py`, its 200-epoch schedule matches the engine, it reports three-seed spreads for every row, and DAT's own seven-run rerun (52.31 +/- 0.26) confirms that it reproduces across groups. Cite DAT for the frontier number; add DAT as a further arm only if the AWP implementation lands early, since its code is MIT-licensed and small (`Generator.py`, `utils_awp.py`) but its schedule (150 epochs, decay 100/110, or decay 100/150 with AWP and SWA) differs from the engine's and would have to be matched or declared. Do not cite DUCAT, DCS or LCAT as records; cite DAWA and DCS Section 5.3.6 if a reviewer raises them. State the generated-data regime as a scope condition using Gowal 2021's 58.63 on PRN-18 and say that no RN-18 result has been published in that regime since 2021. None of this changes the earlier assessment's ranking: the 54-pp overturn condition it set was not met (the best new number is 52.76), and the fair comparison between a distilled RN-18 and this teacher-free band remains unmade.

---

## 6. Sources

Papers (text read unless marked REPORTED above).

- Li, Li, Wu, Tian, Zhou. *DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain.* NeurIPS 2024. arXiv 2410.12307. Tables 1, 3, 6; Section 3; Appendix E.2. Code `github.com/Feng-peng-Li/DAT` (MIT).
- Liu, Kuang, Lin, Wu, Ji. *CAT: Collaborative Adversarial Training.* arXiv 2303.14922, 2023. Tables 1, 2, 4, 5.
- Ma, Huang, Dong, You, Xu. *Adversarial Robustness via Deformable Convolution with Stochasticity.* ICML 2025, PMLR v267. Table 1, Table 6, Section 5.1 and 5.3.6. RobustBench issue #213.
- Wang et al. *New Paradigm of Adversarial Training: Releasing Accuracy-Robustness Trade-Off via Dummy Class (DUCAT).* arXiv 2410.12671 v2, May 2025. Table 1, Equation 6 (REPORTED).
- *Dummy-Aware Weighted Attack (DAWA): Breaking the Safe Sink in Dummy Class Defenses.* arXiv 2603.29182, March 2026. Abstract and main table (REPORTED).
- Liu. *Rethinking Adversarial Training with A Simple Baseline (SimpleAT).* arXiv 2306.07613, 2023. Table 2 (REPORTED).
- Wang et al. *Balance, Imbalance, and Rebalance (ReBAT).* NeurIPS 2023. arXiv 2310.19360 (REPORTED).
- *The Effectiveness of Random Forgetting for Robust Generalization (FOMO).* ICLR 2024. arXiv 2402.11733. Table 3 (REPORTED).
- Wu et al. *Annealing Self-Distillation Rectification (ADR).* ICLR 2024. arXiv 2305.12118. Tables 1, 2 (REPORTED).
- Li, Qiu, Spratling. *AROID.* arXiv 2306.07197 v2, 2024. Tables 1, 4, 5 (REPORTED).
- *Conserve-Update-Revise (CURE).* ICLR 2024. arXiv 2401.14948 (REPORTED).
- *TRIX: Trading Adversarial Fairness via Mixed Adversarial Training.* arXiv 2507.07768, 2025. Table 1a (REPORTED).
- *Robust Alignment Adversarial Training (RAAT).* CVPR 2026 Findings. arXiv 2604.26496. Table 1 (REPORTED).
- *Criticality Leveraged Adversarial Training (CLAT).* arXiv 2408.10204 (REPORTED).
- *MEAT: Median-Ensemble Adversarial Training.* ICASSP 2024. arXiv 2406.14259 (REPORTED).
- Cui et al. *Generalized Kullback-Leibler Divergence Loss.* TPAMI 2025. arXiv 2503.08038 (REPORTED).
- Wang et al. *Better Diffusion Models Further Improve Adversarial Training.* ICML 2023. arXiv 2302.04638. Table 2 (REPORTED).
- Pang et al. *Robustness and Accuracy Could Be Reconcilable by (Proper) Definition (SCORE).* ICML 2022. arXiv 2202.10103. Table 2 (REPORTED).
- Cui et al. *Decoupled Kullback-Leibler Divergence Loss.* arXiv 2305.13948 (REPORTED).
- *Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation.* arXiv 2406.13283 (REPORTED).
- Huang, Chen, Lin. *Expanding the Role of Diffusion Models for Robust Classifier Training.* arXiv 2602.19931, Feb 2026. Table 2 (REPORTED).
- Zhao et al. *Adversarial Training: A Survey.* arXiv 2410.15042, 2024. Table III (REPORTED).
- *VanillaBench: The Hidden Accuracy Cost of Adversarial Robustness.* arXiv 2607.12545, July 2026 (REPORTED).
- *MMARD: Improving the Min-Max Optimization Process in Adversarial Robustness Distillation.* arXiv 2503.06559, March 2025. Tables 2, 3, 5.
- *ProARD.* arXiv 2506.07666, IJCNN 2025 (REPORTED). *Towards Class-wise Fair Adversarial Training via Anti-Bias Soft Label Distillation.* arXiv 2506.08611 (REPORTED). *AdaGAT.* arXiv 2508.17265 (REPORTED). *DARD.* arXiv 2509.11525, Table 1 (REPORTED). *CIARD.* arXiv 2509.12633 (REPORTED). *FERD.* arXiv 2509.20793 (REPORTED). *MMT-ARD.* arXiv 2511.17448 (REPORTED). *Improving Certified Robustness via Adversarial Distillation.* arXiv 2606.31653 (REPORTED). *Information Bottleneck Distillation Through Dual Teachers.* arXiv 2607.27737 (REPORTED).
- *InfoARD.* IEEE TIP, Dec 2025. *Allies Teach Better Than Enemies: Inverse Adversaries for Robust Knowledge Distillation.* IEEE TPAMI, Feb 2026. *Efficient Robustness for Small Models via Dual Adversarial Distillation With Hybrid Supervision.* IEEE TMM, 2026. *Mix2Aug.* IEEE TDSC, March 2026. Abstracts only, via the Semantic Scholar API (REPORTED).
- Earlier-verified numbers reused from `docs/ARD_VERSUS_AT_ASSESSMENT.md`: DAJAT, IDBH, RSLAD, AdaAD, IGDM, PeerAiD, SAAD, "Why Robust Teachers Fail", Bartoldson 2024, Gowal 2021, Rade 2021, Rebuffi 2021, Sehwag 2022.

Repositories and APIs.

- `github.com/RobustBench/robustbench`: commits since 2025-03-31, pull requests #182 to #216, issues #167 to #213 (GitHub REST API, 2026-09-07).
- `github.com/RobustBench/robustbench.github.io`: last 15 commits (GitHub REST API, 2026-09-07).
- `.external/robustbench/model_info/cifar10/Linf/*.json` at `78fcc9e4` (99 entries).
- Semantic Scholar Graph API: citations of arXiv 2210.15318, 2301.09879, 2312.03286, 2403.06668, 2409.01627, 2512.10275; keyword and title searches as described in section 1.
- arXiv API: abstract searches as described in section 1.
- `github.com/Feng-peng-Li/DAT` (repository metadata and file list), `github.com/theSleepyPig/Deformable_Convolution_with_Stochasticity` (README and `model_info` JSON).

---

## Independent check of the finding that matters most

The survey's most consequential observation was checked against the paper by the
session that commissioned it.

**DAT, arXiv 2410.12307, NeurIPS 2024, Table 1**, captioned "Average natural and
robust accuracy (%) of ResNet-18 against l-infinity threat with eps=8/255 in
**7 runs**":

| method | AutoAttack | SD over 7 runs |
| --- | ---: | ---: |
| PGD-AT | **47.63** | ±0.08 |
| DAT | 51.36 | ±0.14 |
| DAT + AWP | 52.54 | ±0.12 |

**This repository's PGD-AT scores 47.63 on the official test set under
AutoAttack** (`docs/EXPERIMENT_DASHBOARD.md:165`, seed 0, best checkpoint).
VERIFIED both sides.

That is an exact match to two decimal places against a seven-run mean, and it
lands inside the interval those seven runs define.  **The engine and the
official-test AutoAttack evaluation in this repository reproduce an external,
variance-reporting reference.**  Nothing else in this project has that kind of
external validation, and it was obtained without spending a GPU-hour.

Three consequences.

**The evaluation stack is trustworthy enough to build a comparison on.**  The
DAJAT checkpoint reproduction that is queued remains worth running, because it
tests the evaluation on weights this project did not train, but the training side
is now independently anchored.

**The TRADES defect is confirmed from a second direction.**  Same engine, same
split, same schedule, same evaluation: PGD-AT is exact and TRADES is four points
low.  A difference of protocol would move both.  See
`docs/debugging/0028-trades-clean-target-detached.md`.

**The claim that "the field does not measure" must be narrowed.**  DAT reports
seven runs with standard deviations, and it is not alone among adversarial
training papers.  The defensible claim is about **distillation** papers
specifically: fifteen from 2025-2026 were checked and none reports variance or
compares against a teacher-free method newer than 2019.  Stated broadly the
criticism invites a one-paper rebuttal; stated narrowly it holds.
