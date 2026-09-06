# ARD versus AT: should the thesis stay with distillation?

Date: 2026-09-07. Read-only assessment. This is the only file written. No GPU job was run.

## How to read this document

Every factual claim carries one of three marks.

- **VERIFIED**: I read the paper text, the code, or a hash-bound record in this repository and can point to the table or line.
- **REPORTED**: a paper or a tool summary states it and I am relaying the statement without having checked the underlying table myself.
- **INFERRED**: my reasoning from VERIFIED or REPORTED items. The reasoning is shown.

Terms used throughout, defined once:

- **AT** (adversarial training): training a network on inputs that an attacker has perturbed, so that it learns to classify them correctly. PGD-AT (Madry et al., 2018) and TRADES (Zhang et al., 2019) are the two standard forms.
- **ARD** (adversarial robustness distillation): AT for a small "student" network where the training target is the output of a large, already robust "teacher" network instead of, or in addition to, the true label. ARD, IAD, RSLAD, AdaAD, IGDM, PeerAiD, DGAD and SAAD are all of this kind.
- **AutoAttack (AA)**: a fixed ensemble of four attacks (Croce & Hein, 2020) used as the standard robustness measurement. "AA accuracy" is the fraction of official test images still classified correctly under it. All numbers below use the CIFAR-10 test set, the L-infinity threat model, and perturbation size epsilon = 8/255, unless stated.
- **Clean accuracy**: accuracy on unperturbed images.
- **pp**: percentage point.
- **Best / last checkpoint**: the saved weights at the epoch with the highest validation robustness, versus the weights at the final epoch. Under AT the two can differ by 5 to 7 pp because of **robust overfitting** (robust test accuracy falls late in training while training accuracy keeps rising; Rice et al., 2020).
- **Seed**: the random-number starting value of a run. Two runs with different seeds but identical settings differ by an amount called the **noise floor** here.
- **Extra data / generated data**: real images beyond the 50,000 CIFAR-10 training images, or images produced by a generative model. RobustBench lists them separately from the standard setting.
- **AWP** (adversarial weight perturbation, Wu et al., 2020) and **SWA** (stochastic weight averaging): two add-ons to AT that reduce robust overfitting. Most of the strongest small-model AT results use one or both.
- **RN-18 / PRN-18**: ResNet-18 and the pre-activation variant PreActResNet-18. They are close in size (about 11 M parameters) but are not the same network, and papers rarely say whether the difference matters.

Sources are listed in section 5 with enough detail to find each table again. Extracted paper texts used for this document sit under the session scratchpad and are not part of the repository.

---

## 1. Has the fair comparison been made?

**Short answer: no single paper has made it.** Every ARD paper compares against PGD-AT and TRADES as they stood in 2018 and 2019. No ARD paper compares against AWP (2020), against the augmentation-based AT recipes that hold the current small-model records without extra data (DAJAT, NeurIPS 2022; IDBH, ICLR 2023), or against AT with generated data (2021 onward). The strong recent AT papers, in turn, do not report ResNet-18 at all. What exists is a comparison that has to be assembled from separate papers, and it does not match training budget.

### 1.1 What the ARD papers compare against

| ARD paper | venue, year | AT baselines used | year of those baselines | protocol detail that matters | mark |
| --- | --- | --- | --- | --- | --- |
| ARD (Goldblum et al.) | AAAI 2020 (arXiv May 2019) | Madry AT, TRADES | 2018, 2019 | evaluation is PGD-20 only; AutoAttack did not exist yet. Table 6: TRADES-WRN teacher -> MobileNetV2 student 82.63 / 50.42 (PGD-20) versus TRADES MNV2 83.59 / 44.79 and AT MNV2 80.50 / 46.90 | VERIFIED |
| IAD (Zhu et al.) | ICLR 2022 (arXiv Jun 2021) | AT, TRADES | 2018, 2019 | teacher is a ResNet-18 of the same size; 200 epochs, decay at 100/150, weight decay 2e-4. Table 2, RN-18, AT teacher: AT 83.06 / **46.70** AA; ARD 48.05; AKD2 48.08; IAD-I 48.66; IAD-II 48.58 | VERIFIED |
| RSLAD (Zi et al.) | ICCV 2021 | SAT (= Madry AT), TRADES | 2018, 2019 | RSLAD trained 300 epochs (decay 215/260/285); baselines "strictly follow their original settings", i.e. fewer epochs. Table 3, RN-18, best checkpoint: SAT 83.38 / **45.83** AA; TRADES 81.93 / **49.23**; ARD 49.19; IAD 49.10; RSLAD 83.38 / **51.49**. Appendix Table 11 reruns baselines at 300 epochs: SAT-300 44.73, TRADES-300 49.50. Teacher: WRN-34-10 trained by TRADES, AA 53.08 (Table 2) | VERIFIED |
| AdaAD (Huang et al.) | CVPR 2023 | PGD-AT, TRADES | 2018, 2019 | PGD-AT: 110 epochs with early stopping; TRADES and all distillation methods: 200 epochs, decay 100/150; best-PGD-10 checkpoint. Table 2, RN-18, WRN-34-10 teacher: PGD-AT 82.95 / **47.69**; TRADES 83.00 / **49.21**; ARD 48.62; IAD 48.82; RSLAD 48.45; AdaAD 86.75 / **50.06**; AdaIAD 50.74. With the LTD WRN-34-20 teacher: AdaAD 51.37, AdaIAD 52.96, RSLAD 48.66 | VERIFIED |
| PeerAiD (Jung et al.) | CVPR 2024 | PGD-AT, TRADES | 2018, 2019 | 300 epochs. Table 1, RN-18: PGD-AT 84.21 / 46.79; TRADES 81.47 / 49.35; RSLAD 51.03; AdaAD 50.08; PeerAiD 85.01 / **52.57**. The "teacher" is a peer RN-18 trained jointly | REPORTED (tool summary of the arXiv HTML); absence of any seed statement in the PDF text VERIFIED by search |
| DGAD (Park et al.) | ECCV 2024 | PGD-AT, TRADES | 2018, 2019 | Table 4, RN-18: PGD-AT 82.95 / 47.69; TRADES 83.00 / 49.21; AdaAD 86.75 / 50.06; DGAD 87.58 / **50.59** | REPORTED; the baseline rows are digit-for-digit the AdaAD Table 2 rows, so they were copied rather than rerun (INFERRED) |
| IGDM (Lee, Cho, Kim) | ICLR 2025 | PGD-AT, TRADES | 2018, 2019 | 200 epochs (RSLAD 300). Table 3, RN-18, LTD teacher: PGD-AT 84.52 / **41.12**; TRADES 82.46 / 47.09; RSLAD 52.13 -> 53.10 with IGDM; AdaIAD 52.88 -> **54.02** with IGDM | VERIFIED. The PGD-AT value of 41.12 is a last-checkpoint value under robust overfitting: this repository's own PGD-AT last checkpoint is 40.36 AA and its best checkpoint is 47.63 (INFERRED from the match) |
| SAAD (Lee & Chung) | TMLR 2026 | PGD-AT, TRADES | 2018, 2019 | 200 epochs, weight decay 5e-4; SAAD applies SWA from epoch 95 (Algorithm, lines 14-15; `args.py` default `swa_epoch=95`). Table 4 / Table 12, RN-18, three seeds: PGD-AT 84.27 / **40.85 +/- 0.14**; TRADES 82.70 / **46.46 +/- 0.57**; RSLAD 44.42 +/- 0.34 (Bartoldson teacher) and 40.57 +/- 0.38 (Gowal teacher); AdaAD 44.55 / 43.27; IGDM 44.94 / 44.76; SAAD 50.34 +/- 0.08 / 50.35 +/- 0.22 | VERIFIED. The PGD-AT value is again a last-epoch value under robust overfitting (INFERRED, same reasoning). Whether the baselines also received SWA is not stated in the text I read |
| Why Robust Teachers Fail (Lee & Chung) | ICML 2026 | PGD-AT (as a reference paradigm) | 2018 | 10 seeds were trained per paradigm to define learnable/unlearnable samples, but Table 2 reports single final AA values without spread: from the Chen LTD WRN-34-20 teacher, ARD 49.62, IAD 48.66, RSLAD 49.86, AdaAD 52.08, IGDM 52.73; from the Gowal WRN-70-16 teacher, 38.40 to 43.91 | VERIFIED |

Three facts follow from the table.

1. The AT baselines in ARD papers are 2 to 8 years older than the paper that cites them, and they get older every year. VERIFIED from the table.
2. None of the eight ARD papers compares against AWP, DAJAT, IDBH, or any AT recipe that uses generated data, although several cite AWP or Rebuffi et al. in passing. VERIFIED by text search of each paper for "AWP", "weight perturbation", "DAJAT", "IDBH", "Addepalli"; RSLAD uses AWP only to build a stronger teacher (Appendix B).
3. The AT baseline numbers drift downward over time inside the ARD literature. TRADES on RN-18 is 49.23 in RSLAD (2021, best checkpoint), 49.21 in AdaAD (2023, best), 47.09 in IGDM (2025) and 46.46 in SAAD (2026); PGD-AT goes from 45.83 (RSLAD, best) to 41.12 (IGDM) and 40.85 (SAAD). The two most recent papers report last-epoch values, where robust overfitting has already cost 6 to 7 pp. VERIFIED numbers; INFERRED cause.

### 1.2 What the strong AT papers report for ResNet-18

The RobustBench CIFAR-10 L-infinity list at the commit pinned in this repository (`78fcc9e4`, 2025-03-31, 99 entries) contains the following ResNet-18-class entries. Read from `model_info/cifar10/Linf/*.json`. VERIFIED.

| entry | architecture | clean | AA | data | venue |
| --- | --- | ---: | ---: | --- | --- |
| Gowal2021Improving_R18_ddpm_100m | PRN-18 | 87.35 | **58.63** | 100 M DDPM-generated images | NeurIPS 2021 |
| Rade2021Helper_R18_extra | PRN-18 | 89.02 | 57.67 | extra real data | OpenReview 2021 |
| Rade2021Helper_R18_ddpm | PRN-18 | 86.86 | 57.09 | generated | OpenReview 2021 |
| Rebuffi2021Fixing_R18_ddpm | PRN-18 | 83.53 | 56.66 | generated | arXiv 2021 |
| Sehwag2021Proxy_R18 | RN-18 | 84.59 | 55.54 | generated (proxy distribution) | ICLR 2022 |
| Addepalli2022Efficient_RN18 (DAJAT) | RN-18 | 85.71 | **52.48** | none | NeurIPS 2022 |
| Addepalli2021Towards_RN18 | RN-18 | 80.24 | 51.06 | none | ECCV 2022 |
| Andriushchenko2020Understanding | PRN-18 | 79.84 | 43.93 | none | NeurIPS 2020 |
| Wong2020Fast | PRN-18 | 83.34 | 43.21 | none | ICLR 2020 |

Two further things about this list. No distillation-trained small model appears on it at all; the only entries whose paper title contains "distillation" are the two Chen2021LTD WideResNets, which are teacher-sized. VERIFIED by search of the 99 JSON files. And the top of the list is WideResNet only: Bartoldson 2024 (WRN-94-16, 73.71), Peng 2023 (RaWRN-70-16), Wang 2023 (WRN-70-16 and WRN-28-10), Cui 2024 DKL (WRN-28-10 and WRN-34-10). VERIFIED from the list. Wang et al. 2023 and Bartoldson et al. 2024 contain no ResNet-18 result (zero matches for "ResNet-18" or "PreAct" in either paper's text). VERIFIED.

The live leaderboard may contain entries newer than March 2025. I could not render it (the page loads its table by script), so the pinned checkout is the reference here. Any ResNet-18 entry added since would change section 1.3 and should be checked once before the decision.

Outside RobustBench, the best no-extra-data numbers for a PRN-18 I found are in the IDBH paper (Li & Spratling, ICLR 2023), Table 2: IDBH[weak] + AWP 52.27 best / 52.21 last; IDBH[weak] + AWP + SWA **52.46** best / 52.52 last; plain AT 48.21 best / 42.46 last; TRADES 49.03; AWP 50.57. Mean of three runs; standard deviations in their Table 8 are at most 0.25 pp for these rows. VERIFIED. DAJAT's own Table 2 (110 epochs, 1,000 images held out) gives RN-18: PGD-AT 48.75, TRADES-AWP 49.87, TRADES-AWP with weight averaging at 200 epochs 51.45, DAJAT 51.48 to 51.56; Table 4 (full training set) gives 85.71 / 52.50, which is the RobustBench entry. VERIFIED.

So the ResNet-18 AT frontier without generated data has stood at about 52.5 AA since 2022-2023, and with generated data at 56.7 to 58.6 since 2021. INFERRED from the two paragraphs above.

### 1.3 The assembled comparison

Same architecture class, official test set, AutoAttack, epsilon 8/255. Budgets are not matched and are listed.

**Rule A: no extra or generated data anywhere in the pipeline (teacher included).**

| side | method | student | AA | seeds | training budget | source |
| --- | --- | --- | ---: | ---: | --- | --- |
| AT | DAJAT | RN-18 | 52.48 | 1 (RobustBench entry); 3 reruns in Table 18 with SD 0.07 on a different attack | 110 epochs, 2+4 attack steps, AWP, weight averaging | VERIFIED |
| AT | IDBH[weak]+AWP+SWA | PRN-18 | 52.46 | 3, SD <= 0.25 | 200 epochs, PGD-10 | VERIFIED |
| ARD | IGDM on AdaIAD, LTD teacher | RN-18 | 54.02 | 1 | 200 epochs, teacher forward and backward passes per step | VERIFIED |
| ARD | AdaIAD, LTD WRN-34-20 teacher | RN-18 | 52.96 | 1 | 200 epochs | VERIFIED |
| ARD | PeerAiD | RN-18 | 52.57 | 1 | 300 epochs, two networks trained | REPORTED |
| ARD | RSLAD, TRADES WRN-34-10 teacher | RN-18 | 51.49 | 1 | 300 epochs | VERIFIED |
| ARD | this repository: RSLAD + I100 schedule, Chen LTD WRN-34-10 teacher (AA 56.94, no extra data) | RN-18 | **53.08 / 53.02 / 53.19** last, 53.08 / 53.02 / 52.96 best | 3 unused confirmation seeds | 200 epochs, PGD-10 | VERIFIED, `docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md` |
| ARD | this repository: RSLAD, CropShift then RandomErasing control | RN-18 | 52.67 / 52.96 / 52.73 last | 3 | 200 epochs | VERIFIED, same file |
| ARD | this repository: RSLAD canonical, seed 0 | RN-18 | 51.90 best / 51.78 last | 1 | 200 epochs | VERIFIED, `docs/EXPERIMENT_DASHBOARD.md` section 4 |

Reading: under Rule A the best ARD numbers sit 0.1 to 1.5 pp above the best AT numbers. The three-seed value in this repository sits 0.5 to 0.7 pp above. The single-seed published ARD values carry no measured spread, and the independent-seed spread for this kind of number is 0.1 to 0.6 pp (section 3), so only the 54.02 value is clearly above the AT frontier, and it is one run. INFERRED.

**Rule B: generated data allowed.**

| side | method | student | AA | source |
| --- | --- | --- | ---: | --- |
| AT | Gowal 2021, 100 M DDPM images, trained directly | PRN-18 | 58.63 | VERIFIED (RobustBench) |
| AT | Rebuffi 2021, DDPM images | PRN-18 | 56.66 | VERIFIED (RobustBench) |
| ARD | SAAD, teacher = Gowal 2021 WRN-28-10 trained on the same 100 M DDPM images (AA 63.44) | RN-18 | 50.35 +/- 0.22 | VERIFIED |
| ARD | SAAD, teacher = Bartoldson 2024 WRN-94-16 (AA 73.71) | RN-18 | 50.34 +/- 0.08 | VERIFIED |
| ARD | RSLAD, same two teachers | RN-18 | 40.57 / 44.42 | VERIFIED |

Reading: when the data that made the teacher strong is given directly to the small network, the small network reaches 56.7 to 58.6 AA. When the same data reaches it only through a teacher, the best published student reaches 50.3 AA. The gap is 6 to 8 pp in favour of direct training. The SAAD student numbers are last-epoch values and would be somewhat higher at the best checkpoint, but not by 6 pp (robust overfitting under distillation is small; SAAD's Table 5 reports a peak-to-final drop of 0.93 pp for SAAD and 5.44 for its baseline). INFERRED.

### 1.4 Verdict on question (1)

- A fair, single-source comparison of the best AT and the best ARD on ResNet-18 does not exist. VERIFIED absence across the eight ARD papers and the AT papers examined.
- ARD papers compare against AT baselines that are 2 to 8 years old, and the two most recent report those baselines at the last checkpoint, where robust overfitting has removed 6 to 7 pp. VERIFIED.
- The strong recent AT papers (Wang 2023, Bartoldson 2024, Peng 2023, Cui 2024) do not report ResNet-18. The augmentation-based AT papers of 2022-2023 do. VERIFIED.
- Assembled under a no-generated-data rule, ARD leads AT on ResNet-18 by roughly 0.5 to 1.5 pp, with unmeasured noise on the ARD side and unmatched budgets. Assembled under a data-allowed rule, direct AT on the small model leads every published distilled student by 6 to 8 pp. INFERRED.
- The student's second doubt is therefore correct in its premise (the comparison was never made fairly) and only partly correct in its implied conclusion (ARD does appear to buy about 1 pp on ResNet-18 when data is held fixed; it buys nothing when the teacher's data can be used directly).

The absence is the consequential finding. A thesis that makes this comparison properly would be the first to do so.

---

## 2. What each path costs from here

### 2.1 The AT path

**Current best-supported method for CIFAR-10 ResNet-18 without extra data.** Two candidates, effectively tied: DAJAT (NeurIPS 2022) at 52.48 AA on RobustBench, and IDBH[weak] + AWP + SWA (ICLR 2023) at 52.46 AA (mean of three, PRN-18). VERIFIED (section 1.2). Both are augmentation recipes on top of TRADES-AWP or PGD-AT with weight averaging. Nothing published since has moved the ResNet-18 number under this rule, as far as the pinned RobustBench list and my search show. INFERRED, with the live-leaderboard caveat from section 1.2.

**Reference implementations.** IDBH: the official code is already pinned in this repository at `.external/DA-Alone-Improves-AT` (commit `38b740a`, 2023-03-24, MIT licence file present) and its README points to pre-trained checkpoints on OneDrive. VERIFIED. DAJAT: RobustBench hosts the RN-18 checkpoint (`Addepalli2022Efficient_RN18`, Google Drive id in `robustbench/model_zoo/cifar10.py`). VERIFIED. The DAJAT paper states that code is released; I did not open the repository. REPORTED.

**Reproducibility judged from what is released.** IDBH reports three-run means and standard deviations for every table row (Tables 8 and 9), uses a plain 200-epoch schedule identical to this repository's, and its CropShift and IDBH-weak transforms are already re-implemented here (`src/ard/data/datasets.py`, `EpochCropshiftTransform`, `_idbh_color_with_generator`). VERIFIED. DAJAT reports three reruns for its own methods only (Table 18), uses a 110-epoch cosine schedule with a 1,000-image validation split, an ascending-epsilon curriculum, AWP and weight averaging. VERIFIED. IDBH is therefore the cheaper and better-characterised target.

**What is missing in this repository for the AT path.** PGD-AT and TRADES exist as methods on the same engine with production configs (`configs/scientific/cifar10_r18_pgd_at.yaml`, `cifar10_r18_trades.yaml`) and have seed-0 official-test AutoAttack results: PGD-AT best 82.01 / 51.12 PGD-20 / **47.63** AA, last 84.46 / 41.89 / 40.36; TRADES best 81.35 / 47.83 / **45.14**, last 82.20 / 45.46 / 43.25. VERIFIED, `docs/EXPERIMENT_DASHBOARD.md` lines 165-167. AWP and SWA are not implemented; a search of `src/ard` for AWP, weight perturbation, SWA and weight averaging finds only a protocol-audit note. VERIFIED. The in-house TRADES value of 45.14 is about 4 pp below every literature value for TRADES on RN-18 (49.0 to 49.4 in RSLAD, AdaAD and IDBH); the in-house PGD-AT value of 47.63 is in line with the literature (47.7 to 48.8). Before TRADES is used as a comparator, that gap has to be explained (candidates: 45,000-image training split, weight decay 2e-4 versus 5e-4, beta, the detached clean-KL target noted in `docs/UPSTREAM_BASELINES.md` section 8). INFERRED.

**Cost.** Implementing AWP is roughly one to two hundred lines plus tests, and it changes the training step, so it must go through the scientific reviewer before a source SHA is pinned; call it one week of attention. SWA is smaller. One 200-epoch RN-18 run without a teacher costs less than the 4 GPU-hours quoted for the teacher-bearing runs. Five seeds of three AT arms is 15 runs, well under one day of wall-clock on five GPUs. GPU time is not the constraint. INFERRED from the user's figures and the repository runtime table.

**Reported number to expect.** If IDBH + AWP + SWA is reproduced faithfully on PRN-18: 52.5 +/- 0.25 AA. On RN-18 with the engine's existing CropShift-then-IDBH-weak schedule but without AWP: the closest published analogue is IDBH[weak] + SWA at 51.73 or IDBH[strong] at 50.74 (best checkpoint). VERIFIED numbers; INFERRED mapping.

### 2.2 The ARD path

**Current best-supported method.** By single-run published AA on RN-18 without extra data: IGDM on AdaIAD with an LTD teacher, 54.02 (ICLR 2025, Table 3). By three-seed evidence: SAAD, but its published numbers are with generated-data teachers and last-epoch reporting (50.3 AA), and its own Table 12 shows RSLAD, AdaAD and IGDM at 40.6 to 44.9 under the same protocol, so the SAAD protocol is not comparable to the rest of the field. VERIFIED. By this repository's own three-seed official-test evidence: RSLAD + I100 schedule with the Chen teacher, 53.0 to 53.2 AA last checkpoint. VERIFIED.

**Reference implementations.** The SAAD repository (pinned at `.external/saad`, commit `295121c`, 2026-05-22) contains `ard.py`, `rslad.py`, `adaad.py`, `igdm.py`, `saad.py`. VERIFIED. It has no licence file, so its code cannot be copied into this repository (`docs/UPSTREAM_BASELINES.md` section 7). VERIFIED. Its full-SAAD path with the Bartoldson teacher does not fit on a 24 GB GPU at batch 128 (peak 23.7 GB against a 22.5 GB safety ceiling). VERIFIED, same file. The default teacher in its `main.sh` is `Gowal2021Improving_28_10_ddpm_100m`, a generated-data teacher. VERIFIED. AdaAD code is at `github.com/boyellow/AdaAD` per the paper; PeerAiD code is on GitHub per the search result. REPORTED. IGDM is included in the SAAD repository (`igdm.py`). VERIFIED.

**Reproducibility judged from what is released.** RSLAD is reproduced in-house to within 0.4 pp of the paper's AA under a different teacher and schedule (51.90 versus 51.49). VERIFIED. AdaAD's own RSLAD reproduction is 3 pp below the RSLAD paper (48.45 to 48.66 versus 51.49), and DGAD copied AdaAD's rows rather than rerunning them. VERIFIED / INFERRED (section 1.1). The 54.02 IGDM number is one run with no spread, from a paper whose PGD-AT baseline is a last-epoch 41.12. VERIFIED. The ARD literature's numbers therefore reproduce to within about 1 pp only when the same group runs them, and that is the size of the claimed improvements. INFERRED.

**Cost.** Nothing new to build for RSLAD arms; the I100 arms exist with three confirmation seeds and official AutoAttack. Adding an AdaAD or IGDM arm means re-implementing the inner maximisation with teacher gradients (clean-room, because of the licence), which is a scientific-surface change of about the same size as AWP, and it raises per-step cost because the teacher needs a backward pass. INFERRED from the SAAD repository contents and the licence note.

### 2.3 What already transfers to either path

Method-agnostic and finished: the measurement standard, the bit-deterministic pipeline (`REF2 - REF1 = 0.0000`), the five-seed independent-seed spread, the post-decay paired-fork floor (0.092 pp at e114), the launch gate and watcher, the official-test AutoAttack procedure, and the augmentation-schedule result (CropShift for epochs 0-99, then IDBH-weak; +0.62 to +1.20 pp across five seeds against two comparators). VERIFIED from `docs/EVIDENCE_RECLASSIFICATION.md` groups A and G and `docs/POST_DECAY_FLOOR.md`. Teacher-dependent and therefore ARD-only: the state-conditional intervention lineage (rows B to F of the reclassification; 23 of 38 rows UNDERPOWERED), the teacher-response signals, and the sample-weighting objectives. VERIFIED from the same file.

---

## 3. Where is the room?

### 3.1 Do ARD papers report seeds or a noise floor?

| paper | seeds | spread reported | mark |
| --- | ---: | --- | --- |
| ARD 2020 | 1 | none | VERIFIED (text search) |
| IAD 2022 | 1 | none | VERIFIED |
| RSLAD 2021 | 1 | none | VERIFIED |
| AdaAD 2023 | 1 | none | VERIFIED |
| PeerAiD 2024 | 1 | none | VERIFIED (PDF text search) |
| DGAD 2024 | 1 | none | VERIFIED (PDF text search) |
| IGDM 2025 | 1 | none | VERIFIED |
| SAAD 2026 | 3 | mean +/- SD in appendix Table 12; RN-18 AA SDs 0.08 to 0.57 pp | VERIFIED |
| Why Robust Teachers Fail 2026 | 10 trained, used only to define sample sets | none in the result tables | VERIFIED |
| DARD 2025, ProARD 2025 | 1 | none | REPORTED / VERIFIED (text search) |

One paper in ten reports a spread. No ARD paper reports a paired or common-parent design. No ARD paper states a minimum effect it set out to detect.

### 3.2 Do AT papers report seeds or a noise floor?

| paper | seeds | what is reported | mark |
| --- | ---: | --- | --- |
| Rice et al., ICML 2020 | several | final robust error with +/- (CIFAR-10 L-inf: 51.4 +/- 0.41 error) | VERIFIED, Table 1 |
| Gowal et al., 2020 | 10 | WRN-28-10 AT: 50.80 +/- 0.23 AA (CIFAR-10 only), 58.41 +/- 0.25 with extra data; clean 84.85 +/- 1.20 | VERIFIED, section 3.2 |
| Pang et al., ICLR 2021 (Bag of Tricks) | 5 | WRN-34-10 TRADES: 53.94 +/- 0.10 AA; RN-18 tables single-run | VERIFIED, Table 16 |
| Rebuffi et al., 2021 | 10 | WRN-28-10: 54.44 +/- 0.39 (Pad&Crop), 57.50 +/- 0.24 (CutMix); no intervals elsewhere "as doing adversarial training ... is roughly ten times more computationally expensive" | VERIFIED |
| Tack et al., AAAI 2022 (Consistency) | 5 | RN-18 AT: 40.71 +/- 0.28 AA (last epoch); +Consistency 48.87 +/- 0.14 | VERIFIED, Table 12 |
| DAJAT, NeurIPS 2022 | 3 (own methods only) | RN-18 robust SD 0.06 to 0.07 (GAMA PGD-100); baselines single-run | VERIFIED, Table 18 |
| IDBH, ICLR 2023 | 3 (all rows) | PRN-18 AA SDs 0.06 to 0.57 best, up to 1.10 end; "no greater than 0.7" | VERIFIED, Tables 8-9 |
| Wang et al., ICML 2023 | 1 | "we cannot afford to report standard deviation"; one config: 63.35 +/- 0.12 | VERIFIED |
| VAIR (Zhang et al., 2023) | 4 | RN-18 AT 47.92 +/- 0.35 AA; TRADES 48.32 +/- 0.19; VIR-TRADES 51.03 +/- 0.16 | VERIFIED, Table 1 |
| ADR, ICLR 2024 | 5 | RN-18 PGD-100 SD 0.182 (AT), 0.128 (AT+ADR) | VERIFIED, Table 7 |
| AR-AT (2024) | 5 | RN-18 AA: LBGAT 48.85 +/- 0.46; AR-AT 49.02 +/- 0.47; AR-AT+SWA 50.28 +/- 0.14 | VERIFIED |
| TRADES instability (2024) | 10 | some seeds show robustness overestimation; shows that seed variance is not always Gaussian noise | VERIFIED, Tables 1-2 |
| Bartoldson et al., ICML 2024 | not stated in the text searched | none found | VERIFIED absence |

About half of the AT papers examined report a spread, typically three to five seeds, occasionally ten. The reported independent-seed standard deviation of AA on ResNet-18-class models is 0.1 to 0.6 pp; on WideResNets 0.1 to 0.4 pp. VERIFIED range. This repository's own five-seed final-epoch spread (held-out CE-PGD20, SD 0.12 to 0.53 pp across arms; `docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md`) sits inside that range. VERIFIED.

### 3.3 Claimed improvements smaller than their own noise

With an independent-seed SD of about 0.3 pp, the standard error of a difference between two three-seed means is about 0.25 pp, and a difference below about 0.5 pp cannot be told from zero. INFERRED arithmetic. Against that yardstick:

ARD side (single-run differences unless stated).

- IAD-I over ARD, RN-18, AT teacher: 48.66 - 48.05 = **0.61 pp**. VERIFIED numbers. Below two standard errors.
- AdaAD over IAD, WRN-34-10 teacher: 50.06 - 48.82 = 1.24 pp. Probably real. AdaAD over RSLAD in SAAD's three-seed protocol: 44.55 +/- 0.17 - 44.42 +/- 0.34 = **0.13 pp**. Inside one SD. VERIFIED.
- DGAD over AdaAD: 50.59 - 50.06 = **0.53 pp**, against copied baselines. REPORTED / INFERRED.
- IGDM on RSLAD: +0.97 pp; on AdaIAD: +1.14 pp; single runs. Borderline. VERIFIED.
- PeerAiD over RSLAD: 52.57 - 51.03 = 1.54 pp, but RSLAD is 0.46 pp below its own paper here. REPORTED.
- Method-to-method spread in "Why Robust Teachers Fail", Table 2, Chen teacher: 48.66 to 52.73, no spread reported although ten seeds were trained. VERIFIED.
- The teacher effect is larger than any method effect: the same RSLAD moves from 51.9 (Chen teacher, in-house) to 47.1 best / 43.1 last (Bartoldson teacher, in-house) and from 44.42 to 40.57 (SAAD, two teachers). VERIFIED. Both SAAD and its follow-up build their contribution on this, correctly.

AT side.

- DAJAT over TRADES-AWP with weight averaging at 200 epochs, RN-18: 51.48 - 51.45 = **0.03 pp** in DAJAT's own Table 2. The headline gain (1.8 pp) is against the 110-epoch baseline. VERIFIED.
- AR-AT over LBGAT, RN-18, five runs: 49.02 +/- 0.47 - 48.85 +/- 0.46 = **0.17 pp**. Inside one SD; still claimed. VERIFIED.
- IDBH[strong] over the best prior augmentation (AuA), PRN-18: 50.74 - 49.15 = 1.59 pp with SDs 0.06 and 0.38. Real. IDBH[weak]+SWA over IDBH[strong]: 0.99 pp with SDs 0.16 and 0.06. Real. VERIFIED.
- Label smoothing in Bag of Tricks, RN-18: +0.69 pp AA, single run. VERIFIED. Their WRN five-seed SD is 0.10 to 0.24.
- VAIR over TRADES: 2.7 pp with SDs under 0.2. Real. ADR over AT: 2.3 pp (PGD-100), five runs. Real. VERIFIED.

Summary: the practice is bad in both fields and worse in ARD. In ARD, nine of ten papers report one run, and about half of the year-over-year increments (0.1 to 0.6 pp) are smaller than the spread the one paper with seeds measured. In AT, the augmentation and regularisation papers' headline gains (1.5 to 2.7 pp) are above noise and are measured with seeds, but their final-step comparisons (0.03 pp, 0.17 pp) fall inside it too. INFERRED from the lists above.

### 3.4 What this project's instrument adds

Three things that no paper in either list has.

1. **A paired-fork floor.** Every published spread is an independent-seed spread. This project measured what two continuations of the same epoch-99 parent do when only the post-fork randomness differs: SD 0.092 pp at epoch 114 (three replicates per parent, two parents), 0.159 at epoch 104, and 1.14 to 1.25 pp before the learning-rate decay. VERIFIED, `docs/POST_DECAY_FLOOR.md`. With that floor, two paired blocks resolve 0.24 pp; the field's three-seed independent design resolves about 0.7 pp. INFERRED arithmetic from the same file.
2. **A demonstrated bit-deterministic pipeline** (`REF2 - REF1 = 0.0000`), so that any difference between two runs is attributable to a named random stream. VERIFIED, `docs/EVIDENCE_RECLASSIFICATION.md` row G1. No paper examined claims this.
3. **Best and last checkpoints reported side by side under the same attack**, plus the rule that AutoAttack runs only from saved weights in a separate process. VERIFIED, `CLAUDE.md` rule 6 and the I100 record. Section 1.1 shows why this matters: the two most recent ARD papers report their AT baselines at the last epoch and lose 6 to 7 pp doing so.

The obvious application of that instrument is the comparison section 1 says is missing.

---

## 4. Recommendation

Ranked. Costs use the user's figures: about 14 weeks, five RTX 4090s, about 4 GPU-hours per teacher-bearing 200-epoch run, attention and calendar as the scarce resources.

### Rank 1: reframe. Make the fair comparison the thesis question.

**The question.** For a ResNet-18 on CIFAR-10 under a fixed data budget, how much robustness does a robust teacher add beyond what strong augmentation gives teacher-free adversarial training, measured against a known noise floor? Section 1 shows nobody has answered it; section 3 shows the field could not have, because it lacks the floor.

**The design.** One engine, one 200-epoch schedule, one augmentation schedule (the I100 schedule, since it is the project's replicated method-agnostic gain), one evaluation (official test, best and last checkpoint, CE-PGD20 and AutoAttack from saved weights), five seeds per arm with common data order and augmentation view across arms. Arms: PGD-AT + I100; TRADES + I100; TRADES-AWP + I100 (the strongest published teacher-free recipe on this architecture); RSLAD(Chen) + I100 (exists: two dev seeds and three confirmation seeds with official AutoAttack); RSLAD(Chen) + CropShift-then-RandomErasing (exists). Optional: RSLAD(Bartoldson) + I100, which exists at seed 0 and shows what a strong teacher trained at scale does to a student.

**Cost.** AWP implementation and review: about one week. Diagnosis of the in-house TRADES gap (45.14 versus 49.2): two to three runs and a few days; this is required regardless of the path, because a 4 pp low TRADES is not a usable comparator. Runs: three new AT arms times five seeds is 15 runs, under 60 GPU-hours and under one day of wall-clock on five GPUs; AutoAttack evaluations on 30 checkpoints, a further day. Records and decision packet: one week. Total about four weeks of calendar, leaving eight to nine for one follow-up and the write-up. INFERRED.

**Power.** With independent seeds and SD about 0.3 pp, five seeds per arm give a minimum detectable difference of about 0.5 pp; with common random numbers across arms it is somewhat smaller but not measured for cross-method contrasts. A teacher effect of 1 pp is resolvable; one of 0.3 pp is not, and a null at that size is still an answer ("the teacher adds less than half a point"). INFERRED from section 3.

**What it buys against the three doubts.** Doubt 1 (deployment): the teacher is a training-time cost only, and the thesis will state in numbers what it buys. Doubt 2 (fairness): the thesis is the fair comparison. Doubt 3 (committee): the defensible core becomes the measurement, which is method-agnostic and already built, rather than a 0.5 pp method claim. Either sign of the result is publishable inside the thesis: "a teacher adds about 1 pp on ResNet-18 under a fixed data budget" or "a teacher adds nothing that augmentation and AWP do not".

**Risk.** The result may land at 0.3 to 0.5 pp, at the edge of resolution. The mitigation is preregistering the minimum effect and reporting the interval, which the measurement standard already requires. The second risk is that AWP, as a scientific-surface change, takes longer than a week; if it does, run the comparison without it first and add it as the last arm.

### Rank 2: stay with ARD, narrowed to what is measurable

Keep the RSLAD-with-Chen lineage, run Direction 1 Stage 1 (the placebo arm, about 3 GPU-hours, `docs/METHOD_DIRECTIONS.md` section 3) and Direction 2 Stage 0 (no GPU), and write the thesis around the I100 result, the measurement standard and the reclassified negative results. Cost: lowest; nothing to build. Risk: the committee's first question, "why a teacher?", has no measured answer, and the project's only replicated positive result is an augmentation schedule that is not about distillation. The 23 UNDERPOWERED rows remain underpowered. This is the fallback if the human decides that an AT arm is out of scope.

### Rank 3: switch to AT outright

Drop the teacher arms, reproduce IDBH + AWP + SWA or DAJAT on RN-18, and study the I100 schedule inside AT. Cost: the same AWP work as rank 1, plus the loss of four months of teacher-lineage results and of the one comparison nobody has made. Novelty: Li & Spratling already showed IDBH on PRN-18 AT with three seeds; the stagewise switch at epoch 100 is new but small (about 1 pp), and a thesis built on it is a replication with one delta. It answers doubt 3 by leaving the room rather than by answering the question. Not recommended.

### The reason that would change my mind, and the evidence that would overturn the ranking

- **If a seed-reported, no-generated-data AT result at or above 54 pp AA on RN-18 or PRN-18 has been published since March 2025**, the assembled ARD lead in section 1.3 disappears, and rank 3 becomes rank 1: the teacher is then demonstrably unnecessary and the thesis should study AT. Check the live RobustBench leaderboard once before deciding; I could not render it.
- **If the matched comparison of rank 1 shows RSLAD + I100 at or below TRADES-AWP + I100 within the floor**, the conclusion flips to "AT suffices for ResNet-18", but the reframed thesis stands, because the question was the comparison.
- **If the in-house TRADES gap (45.14 versus 49.2) turns out to be a defect in the engine's TRADES or in the 45,000-image split**, every in-house AT number is suspect until it is fixed, and the comparison must wait for the fix. This is the first thing to run.
- **If the committee requires a method contribution rather than a measurement contribution**, rank 2's Direction 1 is the only cheap path with a method-shaped payoff, and it is inside ARD; rank 1 should then include Direction 1 Stage 1 as its follow-up rather than a further AT arm.
- **If the human has access to generated CIFAR-10 data** (the Gowal or Wang diffusion samples), one AT-with-generated-data arm on RN-18 would test section 1.3's Rule B claim directly and would be the single most informative extra arm; it is outside the current data contract and is a human decision.

---

## 5. Sources

Papers (text extracted from the arXiv or open-access PDF; table numbers are the papers' own).

- Goldblum, Fowl, Feizi, Goldstein. *Adversarially Robust Distillation.* AAAI 2020. arXiv 1905.09747. Tables 6 and 8.
- Zhu et al. *Reliable Adversarial Distillation with Unreliable Teachers.* ICLR 2022. arXiv 2106.04928. Table 2; section 5.1 experiment setup.
- Zi, Zhao, Ma, Jiang. *Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better.* ICCV 2021. arXiv 2108.07969. Tables 2, 3, 11; section 4.1 training setting; Appendix B and D.
- Huang et al. *Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation.* CVPR 2023. CVF open-access PDF. Table 2; section 4 setup.
- Jung et al. *PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor.* CVPR 2024. arXiv 2403.06668. Table 1 (via tool summary of the HTML version).
- Park et al. *Dynamic Guidance Adversarial Distillation with Enhanced Teacher Knowledge.* ECCV 2024. arXiv 2409.01627. Table 4 (via tool summary of the HTML version).
- Lee, Cho, Kim. *Indirect Gradient Matching for Adversarial Robust Distillation.* ICLR 2025. arXiv 2312.03286. Table 3; Appendix training details.
- Lee, Chung. *Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation.* TMLR 2026. arXiv 2512.10275. Tables 4, 5, 12; Algorithm 1; Appendix C.1. Code: `.external/saad` at `295121c` (`main.sh`, `args.py`).
- Lee, Chung. *Toward Understanding Adversarial Distillation: Why Robust Teachers Fail.* ICML 2026. arXiv 2605.21999. Tables 1, 2, 3.
- Wang, Xu et al. *DARD: Dice Adversarial Robustness Distillation.* arXiv 2509.11525 (via tool summary). Table 1.
- *ProARD.* arXiv 2506.07666. Baseline list only.
- Addepalli, Jain, Babu. *Efficient and Effective Augmentation Strategy for Adversarial Training.* NeurIPS 2022. arXiv 2210.15318. Tables 2, 4, 18; Appendix F.2.
- Li, Spratling. *Data Augmentation Alone Can Improve Adversarial Training.* ICLR 2023. arXiv 2301.09879. Tables 1, 2, 8, 9; Appendix D.5. Code: `.external/DA-Alone-Improves-AT` at `38b740a`.
- Li, Qiu, Spratling. *AROID: Improving Adversarial Robustness Through Online Instance-Wise Data Augmentation.* arXiv 2306.07197. PRN-18 rows only, three runs.
- Rice, Wong, Kolter. *Overfitting in Adversarially Robust Deep Learning.* ICML 2020. arXiv 2002.11569. Table 1.
- Gowal et al. *Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples.* arXiv 2010.03593. Section 3.2.
- Pang et al. *Bag of Tricks for Adversarial Training.* ICLR 2021. arXiv 2010.00467. Tables 4 and 16.
- Rebuffi et al. *Fixing Data Augmentation to Improve Adversarial Robustness.* arXiv 2103.01946. Section 4 variance paragraph.
- Gowal et al. *Improving Robustness using Generated Data.* NeurIPS 2021. arXiv 2110.09468. ResNet-18 rows.
- Tack et al. *Consistency Regularization for Adversarial Robustness.* AAAI 2022. arXiv 2103.04623. Table 12.
- Wang et al. *Better Diffusion Models Further Improve Adversarial Training.* ICML 2023. arXiv 2302.04638. Section 4 setup.
- Wu et al. (ADR). *Annealing Self-Distillation Rectification Improves Adversarial Training.* ICLR 2024. arXiv 2305.12118. Table 7.
- Zhang et al. (VAIR). *Vulnerability-Aware Instance Reweighting for Adversarial Training.* arXiv 2307.07167. Table 1.
- *Rethinking Invariance Regularization in Adversarial Training* (AR-AT). arXiv 2402.14648. Main CIFAR table and its footnote 2.
- *Adversarial Robustness Overestimation and Instability in TRADES.* arXiv 2410.07675. Tables 1, 2.
- Bartoldson et al. *Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies.* ICML 2024. arXiv 2404.09349. Searched for ResNet-18 and seed statements; none found.
- Chen, Lee. *LTD: Low Temperature Distillation for Robust Adversarial Training.* arXiv 2111.02331. Teacher description only.

RobustBench: pinned checkout `.external/robustbench` at `78fcc9e48a07a861268f295a777b975f25155964` (2025-03-31); `model_info/cifar10/Linf/*.json` (99 entries) and `robustbench/model_zoo/cifar10.py`.

Repository records: `docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md` (record `docs/experiments/ard_i100_official_test_autoattack_v1.json`); `docs/EXPERIMENT_DASHBOARD.md` section 4 and lines 165-167; `docs/POST_DECAY_FLOOR.md` (record `ard_post_decay_floor_v1.json`); `docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md`; `docs/EVIDENCE_RECLASSIFICATION.md`; `docs/METHOD_DIRECTIONS.md`; `docs/UPSTREAM_BASELINES.md` sections 7, 8, 10; `docs/REPRODUCTION_STATUS.md`; `src/ard/data/datasets.py`; `configs/scientific/`.

---

## What the commissioning session checked itself

**The in-house TRADES gap is real in our own records.**  `docs/EXPERIMENT_DASHBOARD.md:167`
and `docs/plans/0027-controlled-teacherless-baselines.md:140` both give the seed-0
official-test result as clean 81.35 / PGD-20 47.83 / **AA 45.14**, against a
literature range of 49.0 to 49.4 for TRADES on ResNet-18.  The in-house PGD-AT
figure of 47.63 sits inside its own literature range.  **Only TRADES is low**,
which is what makes it look like a defect rather than a systematic difference in
the engine or the 45,000-image split — a split effect would move both.
Diagnosing it is the first action on every path, because a comparator four points
below the literature would make any comparison built on it worthless.

**The RobustBench check could not be completed and remains open.**  The
leaderboard is rendered client-side and neither the page nor the guessed raw
paths returned the table.  The stated overturn condition — a seed-reported,
no-generated-data ResNet-18 result at or above 54 pp AutoAttack published since
March 2025 — is therefore **unverified**, not absent.  It takes two minutes in a
browser and should be done before the recommendation is acted on.
