# Clean-slate directions in adversarial defense

Date: 2026-09-09.
Status: decision document. No GPU job was run to produce it. Body written by the
workflow's own synthesis agent after five prior session-limit interruptions;
items 6.3.7-6.3.9, the int8 CPU-throughput correction in 6.1, and the three
near-free measurements before section 8 were added afterward from a hand
cross-check of the same raw run, including two live web verifications the
synthesis agent could not perform (it had no network access — see section 0).
Audience: the human. Nothing here is a launch authorisation.

## 0. How to read this, and what I actually checked

Seven surveys of the field were produced, then eight research programmes were
generated without reference to any past work in this repository. Each programme
was examined for prior art twice, independently. The survivors were scored by
four judges on five axes. The top three were attacked by two adversarial
reviewers each, whose job was to kill them.

I did three kinds of checking, and they are not equally strong.

- **Verified by me today, in this repository or on this machine.** Every claim
  marked VERIFIED below was confirmed with a shell command in this session.
- **Relayed.** Every citation to an external paper comes from the survey or
  examiner reports. I had no network access in this session, so I did not open a
  single one of those papers myself. Treat every paper number below as a claim
  someone else checked, not as a claim I checked.
- **Reported measurements from other sessions.** Two of the adversarial reviewers
  ran real code on Hamster and reported numbers. Those are plausible and specific,
  but I did not reproduce them.

Where a number matters to a decision, I say which of the three it is.

---

## 1. What the surveys found

Four things are wrong with adversarial defense right now. They are separate
problems and they compound.

### 1.1 The founding premise was searched for, twice, and essentially not found

The field exists because L_p-bounded adversarial examples were expected to be a
threat to deployed systems. Two deliberate searches by people with production
access came back nearly empty.

Apruzzese, Anderson, Dambra, Freeman, Pierazzi and Roundy (IEEE SaTML 2023,
arXiv:2212.14315) studied a commercial anti-phishing image classifier over a
month of live traffic. They hand-reviewed 4,600 flagged samples to assemble 100
plausible evasions. What the attackers actually did was crop images, blur logos,
remove company names, stretch logos and add background patterns. Their words:
"the strategies employed by real attackers have little in common with
gradient-based adversarial examples," and "evidence of adversarial examples in
the wild is scarce."

Grosse, Bieringer, Besold, Krombholz and Biggio (IEEE TIFS 2023,
arXiv:2207.05164) surveyed 139 industrial practitioners. Of those who reported
any circumvention of an AI workflow, coding the free text left 3 evasion cases
out of 139, and two of those three were autonomous-vehicle recognition errors
the participants themselves doubted were intentional.

The MITRE ATLAS corpus, at version 2026.08, contains 72 case studies. Exactly one
typed Incident is evasion of a perception model by input manipulation, and it is
the Apruzzese phishing study. Every other perturbation-style entry is a vendor or
red-team exercise dated 2019 to 2021. Nothing since.

All of this is relayed. I could not open ATLAS or either paper.

This does not make the field worthless. It does mean that a robust-accuracy
number is not, on current evidence, a security quantity, and that anyone
presenting it as one is making a claim the evidence does not support.

### 1.2 Where evasion is real, it is not norm-bounded, and the one defense that shipped used none of the field's tools

Real bypasses of deployed ML security products exist and they are crude. Skylight
Cyber's 2019 bypass of CylancePROTECT (ATLAS AML.CS0003) appended a string list
harvested from a video game to malware files and got 100% bypass on the top-10
malware of that month. No gradients.

The important counterexample is real and should not be waved away. Nasr,
Fratantonio, Invernizzi, Albertini, Farah, Petit-Bianco, Terzis, Thomas,
Bursztein and Carlini (ACM CCS 2025, arXiv:2510.01676) attacked Magika, the
machine-learning file-type router inside Gmail's malware pipeline. Changing 13
bytes evaded it in 90% of cases, and they delivered malicious PDFs end to end.
Gradient attacks on production systems are feasible.

Look at what shipped as the fix. Not adversarial training. Not distillation. Not
certification. One round of AES applied in preprocessing to destroy
transferability, at sub-microsecond cost, now running in production. The single
documented production deployment of an adversarial-robustness defense used none
of this field's flagship techniques. Relayed.

### 1.3 Below the frontier, the literature is not falsifiable

This is the finding that most directly concerns anyone choosing what to work on.

Defenses overestimate their own robustness at a measured rate. Of 236 RobustBench
model records, 185 carry both the paper's self-reported robust accuracy and an
independent AutoAttack number; in 50 of those 185 (27%) AutoAttack found the model
less robust than the paper claimed, with a mean overestimate of 9.2 points and a
maximum of 74. That is the same pattern Athalye, Carlini and Wagner found at ICML
2018 (6 of 9 ICLR 2018 defenses fully circumvented), and Tramèr, Carlini, Brendel
and Madry found at NeurIPS 2020 (13 of 13 defenses circumvented, with attacks
simpler than the ones the original papers ran).

Underneath that sits a measurement problem nobody addresses. Rice, Wong and Kolter
(ICML 2020) showed early stopping alone matches the gains of an entire generation
of algorithmic improvements. Pang, Yang, Dong, Su and Zhu (ICLR 2021) showed a
slightly different weight decay moves robust accuracy by more than 7 points —
larger than nearly every claimed contribution in the field.

Against that, the median gap between adjacent ranks on the CIFAR-10 leaderboard is
0.26 points, and 29% of adjacent pairs differ by less than 0.1 points. No entry on
any leaderboard carries an error bar; the record format has no field in which one
could be stored. So most published rankings are not distinguishable from noise,
and nobody can tell because nobody publishes the noise floor. All relayed.

### 1.4 The frontier is a compute result, and the small-model case, which is the only deployable one, is far behind

RobustBench CIFAR-10 running-best by year: 44.04 (2018), 59.53 (2019), 65.88
(2020), 71.28 (2021), 71.28 (2022), 71.28 (2023), 75.28 (2024). Three flat years,
then four points bought with a WideResNet-94-16 and 300 million generated images.
That is a scaling result, not a method.

The small-model case is the one that matters for the student's question, and it is
worse. VERIFIED in this repository: the pinned RobustBench checkout
(`.external/robustbench`, commit 78fcc9e4, 2025-03-31, which is the project head)
holds 30 ImageNet L-inf entries, and the smallest is a ResNet-18 at 52.92% clean
and 25.32% AutoAttack at eps=4/255. There is no mobile-scale architecture on that
list at all — the entries are ResNet-18/50, WRN-50-2, XCiT, ViT, Swin and ConvNeXt.
An adversarially trained ImageNet ResNet-18 surrenders roughly 17 points of clean
accuracy to buy 25% robustness against a threat model with no recorded production
incident. That is not a product.

Also VERIFIED: the leaderboard has stopped. The RobustBench repository's most
recent commit of any kind is 2025-03-31, and this repository's own survey
(`docs/SMALL_MODEL_AT_2024_2026.md`, 2026-09-08) already recorded the consequence:
"not on RobustBench" no longer means "does not exist."

### 1.5 Where the money actually went

For completeness, because it bears on what "useful in the real world" means now.
The 2024–2026 ATLAS entries are prompt injection, agent hijacking, supply-chain
compromise and deepfake identity fraud. Documented losses include roughly $77M in
a camera-hijack tax fraud in Shanghai and $25.6M at Arup from a deepfake video
call. None of that is defended by a smaller epsilon ball, and an adversarially
trained face model is exactly as vulnerable to a virtual camera as a standard one.
Relayed. This is context, not a research direction — a small lab is not positioned
to lead on any of it.

---

## 2. The student's own question already has an answer, and it is free

The question was: adversarial robustness distillation is meaningful if lightweight
models can actually reach good performance through it, and meaningless if they
cannot.

That is answerable today at zero GPU-hours, and this repository already wrote the
answer down. VERIFIED, `docs/ARD_VERSUS_AT_ASSESSMENT.md` section 1.3:

> when the data that made the teacher strong is given directly to the small
> network, the small network reaches 56.7 to 58.6 AA. When the same data reaches
> it only through a teacher, the best published student reaches 50.3 AA. The gap
> is 6 to 8 pp in favour of direct training.

The endpoints are downloadable. Gowal 2021's PreActResNet-18, trained on 100
million generated images with no teacher, sits at 87.35% clean and 58.63%
AutoAttack (VERIFIED from the pinned RobustBench record). The best three-seed
distilled ResNet-18 in the literature sits around 50.3, and the best single-run
distilled number is around 53 to 54.

So: yes, a lightweight model can be made meaningfully robust on CIFAR-10. No, a
teacher is not the route — giving the small model the data directly is 6 to 8
points better at the same inference cost. That answer cost nothing and it is
already in the repository.

Three of the eight programmes, including the two highest-scoring ones, are
variations on re-asking this question more carefully. That is why they scored 3 to
4 out of 10 on deployment value. A more precise answer to a question that already
has one does not become useful by having error bars.

---

## 3. Does anything clear the deployment test? No.

The test, in the student's own form: if this quantity improved by the amount the
research could plausibly deliver, whose situation actually gets better, and how
would they notice?

Applied honestly to all eight programmes, none passes.

- **The four distillation programmes** (numbers 1, 2, 3, 5 in the table below)
  improve the precision of an answer about CIFAR-10 robust accuracy at eps=8/255.
  Nobody deploys that classifier. The beneficiary is the research community's
  bookkeeping.
- **The two ImageNet-mobile programmes** (2's second stage, and 7) produce a cost
  sheet for a 5-million-parameter robust classifier. That artifact does not exist
  and someone considering shipping one would consult it. But it would still cost
  15 or more points of clean accuracy for robustness against an attacker with no
  recorded production incident, so nobody would ship it either.
- **The keyed-transform programme** (4) is the only one whose defense class has
  actually shipped to production, at Gmail. But its flagship transform families
  appear to be already dead at zero attacker effort (Tanaka, Echizen and Kiya,
  ISPACS 2022, arXiv:2209.02997, Table VI — relayed, and it is the single most
  important unchecked citation in this document), and its central novelty claim
  appears false (MaungMaung, Echizen and Kiya, IEEE OJSP 2024 — relayed).
- **The two artist-protection programmes** (6, 8) are the ones the survey itself
  liked best, because real users made real decisions on a false assurance. But the
  answer is already published three times over (Hönig, Rando, Carlini and Tramèr,
  ICLR 2025; Foerster et al., LightShed, USENIX Security 2025; Pleimling et al.,
  IEEE SaTML 2026 — all relayed), and a tighter error bar on "Glaze does not
  protect you" changes no artist's decision. Their tractability scores are the
  worst in the set (4.5 and 3.75) because their critical path is ethics approval,
  artist consent, human raters, a closed Windows GUI binary and thirteen research
  repositories — none of which five 4090s help with, and all of which consume the
  one resource declared scarce.

I am not going to manufacture a winner. The correct reading of the survey is that
this field's centre of gravity optimises a quantity with no recorded deployed
victim, and that no programme sized for this lab escapes that. What follows is the
least bad option, and I will say plainly why it is still worth doing.

---

## 4. Recommendation: the mobile robustness cost sheet

**Recommended: programme 2, stripped to its second stage and re-aimed.**

Concretely, what I recommend running is not the programme as written. It is what
survived after both adversarial reviewers were done with it, which is what both of
them independently identified as the part they could not kill:

> The first seed-replicated, noise-floored cost sheet for a deployable-size robust
> ImageNet classifier — clean accuracy, AutoAttack at eps=4/255, worst-class
> accuracy, natural-shift panel, and robustness retention after quantisation —
> measured on the number format the model would actually run in.

Four changes from the written proposal, each forced by the red team:

1. **Delete the CIFAR-10 stage.** Its question is answered (section 2), the
   repository already has a better-designed five-seed plan for it, and the
   distillation champion it names is two years stale.
2. **Demote the teacher contrast to an optional secondary arm.** It cannot be
   answered at one compute point, its compute ledger is undefined, and at n=3 it
   is not powered to resolve the 1–2 point effect at issue.
3. **Make the headline the cost sheet, not the teacher.** The teacher-free arm
   plus the untouched pretrained control plus the quantised evaluation is the
   whole deliverable, and it needs no distillation at all.
4. **Replace "int8" with a bit-width sweep including int4.** One adversarial
   reviewer measured int8 retention at 0.995 on this lab's own robust checkpoint,
   which would make the proposal's preregistered 0.9 threshold unreachable from
   below. Reported measurement, not verified by me — and it is the reason for the
   gate in section 6.

### The margin, honestly

Programme 2 scored 31.25 out of 50; programme 1 scored 32. I am not recommending
the highest-scored programme, and the 0.75-point difference between them is
meaningless — it is four judges' arithmetic on a subjective scale.

The real reason is the kill record, and it is not close. Programme 1 was attacked
twice and killed twice; both reviewers independently found that its primary
contrast is confounded four ways and that its sign is predictable before any GPU
runs. Programme 3 was killed once on a calibration defect this repository has
already documented. Programme 2 was attacked twice, killed once, and the reviewer
who failed to kill it did so after running four empirical attacks on this lab's
own hardware and losing all four.

That is the margin: not a score, a survival record under adversarial testing where
the attacks were actually executed rather than argued.

### Why it is still worth doing even though it fails the deployment test

Three reasons, in decreasing strength.

1. **It tests whether a number the field reports is the number the artifact has.**
   Every "efficient robustness" paper reports float32 robust accuracy of a small
   model. If that number does not survive quantisation, the subfield has been
   reporting a quantity that cannot leave a GPU, and every future paper in the
   line owes a retention row. That is a falsifiable claim about a literature, and
   it is cheap to test.
2. **The artifact does not exist and is genuinely unoccupied.** VERIFIED: there is
   no sub-ResNet-18 entry on the pinned ImageNet leaderboard, and no mobile
   architecture at all. Relayed: one from-scratch EfficientNet-B0 exists outside
   the leaderboard (EasyRobust, arXiv:2503.16975, 61.83 clean / 35.06 AutoAttack)
   as a single number with no seed, no panel and no quantised evaluation.
3. **It is the version of the student's own question at deployment scale.** The
   CIFAR answer in section 2 is free. Whether it holds at ImageNet, at mobile
   size, in int8, is not free, and it is the only version anyone shipping anything
   would care about.

---

## 5. Every programme

Scores are out of 10 on each axis: deployment value (D), novelty (N),
tractability (T), robustness of the design (R), legibility (L). Total out of 50.
None was preempted outright at the gate; the preemption column records partial
preemption found by the examiners. **Every paper in the preemption column is
relayed and unverified by me.**

| # | Title | In one sentence | D | N | T | R | L | Tot | Preempted? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Teacher or Data? A noise-floor accounting of ARD | Does a robust teacher add anything to a ResNet-18 that the same GPU budget spent on generated data does not? | 3.75 | 4.75 | 7.5 | 7.75 | 8.25 | 32 | Partly. The fixed-data half is owned by Lee & Chung, SAAD, TMLR 2026 (arXiv:2512.10275), 15 teachers x 6 methods x 3 seeds. The compute-matched framing is Busbridge et al., Distillation Scaling Laws, ICML 2025 (arXiv:2502.08606), in another modality. |
| 2 | **Does the Teacher Pay for Itself? CIFAR-10 to int8 ImageNet MobileNets** | **Does distilling from a robust teacher beat plain adversarial fine-tuning for a phone-sized ImageNet model, and does the robustness survive int8?** | **4.5** | **5** | **5.75** | **8** | **8** | **31.25** | **Partly. Its "ARD + diffusion data" clause is occupied by ASDA (ICLR 2026 submission, OpenReview ZZGn6GXJDH), which reports the direction negative. Its motivating premise is stale: SAAD beats teacher-free TRADES on both students with error bars. The ImageNet-mobile and quantisation clauses survived.** |
| 3 | Does the Teacher Buy Anything? Matched-compute ARD vs generated data | Same question as 1, with AdvFunMatch as the steelman and a replication verdict on its unrefereed 58.30 claim. | 3 | 4 | 7.75 | 8 | 8 | 30.75 | Partly, and its own stated falsifier fired. Dong, Koniusz, Chen, Wang & Ong, CVPR 2024, Table 4, reports ARD/RSLAD/IAD ResNet-18 students trained on 1M DDPM CIFAR-10 under AutoAttack. A second examiner attributes an equivalent table to Dong, Koniusz, Chen & Ong, ECCV 2024 (STARSHIP), Table 5. Both relayed; the two attributions conflict and must be resolved before anything is preregistered. |
| 4 | Keyed Deployment: secret input transforms under an attacker who trains his own keys | Gmail scrambles inputs with a secret key to break transfer attacks; does that work for image classifiers, and at what query budget does it stop? | 4.25 | 4 | 6.75 | 7 | 8 | 30 | Probably, and badly. Tanaka, Echizen & Kiya, ISPACS 2022 (arXiv:2209.02997), Table VI reportedly drives two of the three flagship transform families to about 3% robust accuracy with a plain public surrogate and zero attacker keys. The secret-weights threat model is reportedly already published by MaungMaung, Echizen & Kiya, IEEE OJSP 2024. Both relayed. |
| 5 | Teacher or Data? Robust distillation at a measured noise floor | A third variant of 1 and 3, with a recipe-nuisance floor added. | 2.75 | 4 | 7 | 7.25 | 8 | 29 | Materially. Its {robust teacher} x {real vs +generated data} factorial is reportedly STARSHIP (ECCV 2024) Table 5, and its motivating claim that no ARD method beats teacher-free AT is false inside that same paper's Table 2. Relayed. |
| 6 | Cost-to-Break: attacker-budget curves for artist protections | How much must someone spend per image to strip Glaze or Nightshade, and how big is the error bar on that number? | 4 | 5 | 4.5 | 7 | 8.25 | 28.75 | Qualitatively. Hönig, Rando, Carlini & Tramèr (ICLR 2025) and Foerster et al. (LightShed, USENIX Security 2025) already published the negative; Pleimling et al. (IEEE SaTML 2026) reportedly strip six families with one off-the-shelf model and one prompt, i.e. at the zero point of the proposed cost axis. Relayed. |
| 7 | Teacher, Init and int8: robust ImageNet at mobile scale | Near-duplicate of programme 2's second stage, without the CIFAR stage. | 4 | 3.5 | 5.75 | 7.25 | 8 | 28.5 | Partly. Its novelty sentence is reportedly false in both halves: EasyRobust's EfficientNet-B0 exists, and Thorsteinsson et al. (arXiv:2403.09441) reportedly already show int8 does not erase robustness at CIFAR/Tiny-ImageNet scale. Relayed. |
| 8 | Cloak Budget Curves: what anti-mimicry protections buy per attacker-second | Variant of 6 with a wider protection set and a human-rater study. | 2.75 | 3.75 | 3.75 | 7 | 8.25 | 25.5 | Same as 6, plus IPV-Bench (arXiv:2603.26154) reportedly claimed the unified-benchmark slot in a sibling application. Relayed. |

No programme was preempted outright at the gate.

---

## 6. What the red team could and could not kill

This section is the point of the document. The three categories are not the same
and should not be blurred.

### 6.1 What it could not kill — attacks that were executed and failed

One adversarial reviewer ran four attacks on Hamster against programme 2's second
stage and lost all four. These are reported measurements from another session, not
verified by me, but they are specific enough to be worth acting on.

- **Bit-determinism of the mobile architectures under mixed precision.** This was
  the proposal's own self-declared top schedule risk, budgeted at up to two weeks.
  Running a real PGD-3 adversarial-training loop at 224 pixels in separate
  processes gave bit-identical parameter hashes for MobileNetV3-Large,
  EfficientNet-B0 and a ResNet-18 control, with and without autocast. Depthwise
  convolution, squeeze-excite pooling and hard-swish were all deterministic. The
  second reviewer independently reproduced this with a different configuration.
- **The data path.** VERIFIED by me today: ImageNet is 146 GB at
  `/home/shunsukenaito/workspace-local/datasets/imagenet` on local NVMe with train
  and val both present, on a filesystem with 2.2 TB free. The reviewers measured
  the folder scan at about 2 seconds for 1,281,167 images and the 224-pixel loader
  at 2,251 to 3,235 images per second, which is above what training consumes. An
  earlier objection that ImageNet sat on a 95%-full NFS share was simply wrong.
- **The compute model.** Measured per-epoch costs came in within roughly 11 to 37%
  of the proposal's estimates for the mobile student, and the fleet has the
  throughput. GPU-hours are not the binding constraint.
- **The quantisation toolchain.** It is degraded, not merely slower: the
  converted int8 model's `.to('cuda')` succeeds silently and the forward pass
  then **segfaults (SIGSEGV, process exit 139, no Python traceback, core
  dumped, reproduced twice)**. It must run on CPU, measured at **567 to 607
  images per second on 26 threads** — the document this replaces an earlier
  draft's rounding of that figure up to "600 to 900," which is not what was
  measured. At that rate, Square's 5,000 queries against the 5,000-image
  RobustBench ImageNet subset is 25 million forward passes, about **12.2
  CPU-hours per checkpoint**. Across 22 planned final runs x 2 checkpoints that
  is roughly **260 to 540 CPU-hours contending with the ImageNet dataloader on
  the same machine**, not the "about 3 hours" the original proposal budgeted
  against GPU-hours. It is affordable and exactly reproducible, as stated, but
  it is a CPU-bound line item that section 7's cost table should book as such
  rather than folding into a small parenthetical.

### 6.2 What it could kill — and did

- **The teacher contrast, three independent ways.** First, the motivating premise
  is false: SAAD (Lee & Chung, TMLR 2026) reportedly beats teacher-free TRADES by
  3.4 to 3.9 points on both ResNet-18 and MobileNetV2 with three seeds, so a
  contest whose champion is RSLAD is a strawman. Second, Busbridge et al. (ICML
  2025) make "does the teacher pay for itself" a crossover question whose answer
  is a function of budget, and a single compute point cannot locate a crossover.
  Third, the compute ledger is undefined — the teacher's own training cost is
  excluded, and the sign of the answer depends on that unstated convention.
- **The int8 headline.** A reviewer measured, on this lab's own robust CIFAR-10
  teacher checkpoint, int8 retention of 0.995 and int4 retention of 0.918. If that
  transfers, the preregistered "retention above 0.9" bar is unreachable from
  below at int8, and the experiment as specified was designed to observe nothing.
- **The CIFAR stage of every distillation programme.** Section 2 above. The
  repository already contains the answer and a better-designed plan for the
  residual.
- **The "first noise floor" claim.** Reportedly, SAAD Table 12, ADR Table 7 and
  DAT Table 3 already publish multi-seed AutoAttack spreads. What is unpublished
  is the *decomposition* into initialisation, data order and attack seed, which is
  a smaller claim.
- **"No robust small ImageNet model exists."** False; EasyRobust's EfficientNet-B0
  is reported at 61.83 clean / 35.06 AutoAttack.

### 6.3 What survived only because nobody checked

This is the list that should worry us most, because each item is currently load
bearing and untested.

1. **Whether a mobile student adversarially fine-tuned from standard pretrained
   weights reaches anything near a from-scratch anchor.** No arm in any version of
   the programme reproduces a known-good *training* recipe; every calibration in
   the plan calibrates the *evaluator* instead. Both reviewers flagged this and
   neither ran it. If a 30-epoch fine-tune lands far below EasyRobust's 35.06,
   then the cost sheet is a cost sheet for an undertrained model and the retention
   ratio is a ratio on an undertrained numerator. This is the real load-bearing
   unknown and it is why calibration is gate 2 in section 8.
2. **Whether quantisation retention on a depthwise, hard-swish mobile network at
   ImageNet scale resembles the measurement on a WideResNet at CIFAR scale.** The
   reviewer who took the measurement said so himself. Depthwise convolutions and
   squeeze-excite blocks are known to quantise worse than plain convolutions. The
   0.995 number may not transfer at all, in either direction.
3. **Whether the three-way attack guard produces a number that means anything.**
   Three of AutoAttack's four components need gradients and cannot run through a
   real quantised graph, so the shipped artifact is only ever attacked black-box
   with Square, while the strong attacks run on a float simulation. Nobody tested
   how big the gap between those two is. If it is large, the honest deliverable
   shrinks to an upper bound.
4. **Cross-host bit-identity.** VERIFIED by me today, and it corrects the brief:
   `docs/ERT_RSLAD_REAL_DATA_TRAINING_DETERMINISM.md` establishes bit-identical
   weights and optimiser state for two runs on the *same GPU and same host*, and
   its own limitations section lists cross-host agreement as not yet performed.
   The lab's distinguishing capability is real within a host and unproven across
   hosts. Any claim that results are bit-reproducible across machines is currently
   unsupported by our own records.
5. **Whether the hash-bound lineage contract scales to 1.28 million files.** The
   content-identity machinery was built for CIFAR and Tiny-ImageNet. One reviewer
   measured the digest cost at about 1,694 files per second, which is fine, but
   nobody tested the surrounding stable-identifier machinery at that scale.
6. **The claim that RobustART already contains adversarially trained ImageNet
   MobileNetV3 and ShuffleNetV2 models evaluated with AutoAttack.** This came from
   a judge, unverified. If it is true it materially dents the novelty of the cost
   sheet. It is a twenty-minute check and it should be the first thing done.
7. **The Dong et al. attribution, live-checked and worse than "may be the same
   table."** Independently verified against the web on 2026-09-09, not merely
   relayed: **both papers are real, and they are two different papers, not one
   paper described twice.** Dong, Koniusz, Chen & Ong, "Adversarially Robust
   Distillation by Reducing the Student-Teacher Variance Gap," ECCV 2024
   (Springer LNCS, doi 10.1007/978-3-031-73235-5_6) — confirmed via Springer,
   ECVA and ACM DL. Dong, Koniusz, Chen, Z. Jane Wang & Ong, "Robust Distillation
   via Untargeted and Targeted Intermediate Adversarial Samples" (DARWIN), CVPR
   2024 — confirmed via the CVPR virtual program and CVF open access. They share
   three authors with Z. Jane Wang added on the CVPR paper: a plausible pair from
   one research group. **What is not verified is whether the specific table
   content quoted (ECCV Table 5; CVPR Table 4) actually contains the numbers
   attributed to it** — every fetch attempt against both PDFs was refused, and
   two different examiners each claim to have read one of the two tables
   directly, with numbers that disagree with each other (RSLAD reported as 52.60
   in one, 52.42 in the other). **Four other examiners, working independently
   across three different programmes, searched for the same experimental cell
   and reported finding nothing** — that 4-against-2 asymmetry does not appear
   above. This is the single highest-value fact for a human to settle before
   preregistering anything in the ARD-vs-data cluster, since the one cell all
   three of those programmes' novelty claims depend on stands or falls on it:
   open `ecva.net/papers/eccv_2024/papers_ECCV/papers/00499.pdf` and the CVPR
   2024 open-access page, and read Table 5 and Table 4 directly.
8. **A citation with fabricated metadata attached to a real paper, found by
   live-checking rather than by symptom.** One examiner (reviewing programme 8)
   cited "IPV-Bench (arXiv:2603.26154), Fang et al. (HKUST-GZ / Wuhan Univ.),
   19 Aug 2026." Verified against the web on 2026-09-09: the arXiv ID is real,
   but the paper is titled "IPV-Bench: Benchmarking Image Protection Methods
   under Diverse Image-to-Video Generation Scenarios," its actual authors are
   **Xiaofeng Li, Leyi Sheng, Yifan Zhao, Zhen Sun, Zongmin Zhang, Jiaheng Wei,
   Xinlei He**, and it was submitted **27 March 2026**, not 19 August. Both the
   author and the date attached to a correct arXiv ID were invented. This is
   concrete evidence that at least one examiner in this run fabricated
   supporting detail around an otherwise-real citation, and it is the reason
   every citation below marked "relayed" should be read as "an arXiv ID that
   probably resolves, with surrounding claims of unknown reliability" rather
   than as a checked fact.
9. **Citations named above that no agent could open, load-bearing enough to
   check before relying on the verdict that used them.** "RC-QAT: Efficient
   Robust Quantization via Attention-Guided Adversarial Distillation" (CSCWD
   2026, DOI 10.1109/cscwd68734.2026.11582282) does not appear elsewhere in this
   document but was named by two independent examiners as sitting exactly on
   the int8-x-distillation intersection that is this document's own recommended
   headline; neither could retrieve it. The ASDA finding cited above as
   settling programme 2's generated-data clause negative rests on an ICLR 2026
   submission (OpenReview forum `ZZGn6GXJDH`, ratings 2/4/3) whose examiner
   states plainly that the PDF's tables were unreadable and only the abstract
   was seen — the negative direction is the abstract's own framing, not a
   verified table. "Wu, Huang, Chen, Pang, Wang, 'Scaling and Taming
   Adversarial Training with Synthetic Data,' ICCV 2025" was described with a
   DOI, an exact page range, an institution, an OpenAlex ID and a DBLP key by
   an examiner who never got past a 403; that combination of precision and
   total unreadability is worth ten minutes of direct checking before treating
   it as a settled risk to any programme. "Frochte, arXiv:2605.09030" — the
   sole-author style-metric paper cited to justify the artist-protection
   cluster's metric-calibration rule — was independently confirmed real on
   2026-09-09 ("When Style Similarity Scores Fail: Diagnosing Raw CSD Cosine in
   Artist-Style Evaluation," Jörg Frochte), so this one specific worry is
   retired.

Also VERIFIED, and relevant to everything: the engine does not currently support
this work. `src/ard/config/schema.py:259` restricts datasets to synthetic CIFAR,
CIFAR-10, CIFAR-100 and Tiny-ImageNet; line 88 restricts schedulers to identity
and multistep; there are zero references to `timm` anywhere in `src/ard`; there is
no quantisation code in `src/ard` or `scripts`; and there is no weight averaging.
A 224-pixel ImageNet path, a model registry entry for mobile architectures, and a
quantisation-plus-attack path all have to be built before any scientific run.

---

## 7. Cost and honest duration

No deadline was given, so this is an unforced estimate.

**GPU-hours, core programme: about 950.**

| Block | GPU-hours |
|---|---|
| Gate: quantisation retention on existing checkpoints, no training | 60 (plus ~60 CPU-hours) |
| Calibration: reproduce a published from-scratch mobile anchor | 100 |
| Cost sheet: 2 students x teacher-free arm x 3 seeds | 370 |
| Evaluation: AutoAttack on the 5k subset, full-50k on finalists, shift panel, quantised retention on new checkpoints | 230 |
| Contingency at 25% | 190 |

At 100 to 120 GPU-hours per day of fully loaded throughput that is about 8 to 10
days of pure compute, which is not the constraint. The optional teacher arm adds
roughly 400 more and should not be launched until the cost sheet exists.

**Person-weeks: 18 to 22. Calendar: 5 to 7 months for one person.**

| Block | Person-weeks |
|---|---|
| Quantisation and three-route attack path, evaluation-only | 3–4 |
| 224-pixel ImageNet loader inside the identity and determinism contract, mobile model registry, validation split | 4–5 |
| Calibration and its debugging | 2–3 |
| Campaign supervision, aggregation, analysis | 4 |
| Write-up, reproducibility appendix, released checkpoints | 5 |

The calendar figure is longer than the person-week figure because the engineering
is a strict serial prefix: nothing scientific can run until the ImageNet path
exists, and one person cannot parallelise that against themselves. The campaigns
themselves run unattended and cost almost no attention.

**Abandon it at any of these three points.**

1. **After the gate, at about week 4 and 60 GPU-hours.** If robustness retention
   is at or above 0.95 at every bit-width anyone would ship, across all three
   attack routes and all four checkpoints, the quantisation clause has no content.
   Stop. Publish the retention table as a three-page note — it is the first such
   measurement either way — and pick a different direction.
2. **After calibration, at about week 9 and 160 GPU-hours cumulative.** If a
   teacher-free adversarially fine-tuned mobile student cannot come within about 2
   points of a published from-scratch anchor at any budget the lab can afford,
   stop. Report the number as a floor on the recipe, not as a ceiling on the
   architecture — the difference matters and the proposal originally got it wrong.
3. **After the first seeded cell, at about week 14.** If the seed standard
   deviation at ImageNet scale exceeds about 1.0 point, no contrast at the 1 to 2
   point scale is resolvable at n=3. Drop the optional teacher arm permanently and
   ship only the cost sheet, which does not require a contrast to be valuable. For
   reference, VERIFIED in this repository: our measured five-seed endpoint standard
   deviations at CIFAR scale are 0.118, 0.272 and 0.532 points depending on the
   arm (`docs/ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md`), so the floor is
   arm-dependent by a factor of four and must be measured per arm, not assumed.

This repository's teacher-free TRADES baseline is still 1.2 to 1.5 points below
every literature value after a real bug was fixed (VERIFIED,
`docs/debugging/0028-trades-clean-target-detached.md`: 45.14 to 47.87 AutoAttack
after removing a detached gradient target, against a literature range of 49.0 to
49.4). A baseline that is low by more than the effect size, in the direction that
flatters the treatment, has already produced one wrong number here. This is not a
footnote to file under "one more precondition" — it is the single cheapest,
highest-value item identified anywhere in this run, because it blocks every
future contrast this harness could run against a teacher-free control, in any
direction chosen. If any contrast is run against a teacher-free control, that gap
must be closed or measured first.

### Three more measurements worth taking regardless of which direction is chosen

None of these costs more than a coffee break, and each one either predicts or
retires a much larger piece of work above.

1. **A ten-GPU-minute pre-screen of the teacher on the generated data.** One
   forward pass of the already-pinned `Chen2021LTD_WRN34_10` teacher
   (`teachers.lock.yaml`, sha256 `fc398a48…`) over the 1M-image generated set,
   reading off the teacher's label agreement with the generator's own labels and
   its predictive entropy on synthetic versus real inputs. Minutes of GPU time,
   no training. This single number predicts the sign and most of the magnitude
   of the entire distillation-versus-generated-data question that the killed
   programmes 1, 3 and 5 were each built to answer at 500 to 1,200 GPU-hours. Run
   it before spending any of that budget on a follow-up to this cluster, even a
   narrowed one.
2. **A checkpoint-selection table**: best-on-held-out-validation versus
   best-on-official-test versus last-epoch, for whatever grid of runs any chosen
   direction produces. This is a within-run, within-arm contrast, and is
   therefore immune by construction to the arm-specific TRADES bias documented
   above — it cannot be confounded by which control a comparison happens to use.
   No ARD paper in the literature defines a held-out selection split at all.
   Nearly free once any training grid exists for any other reason.
3. **A one-week, 10-to-15-GPU-hour replication of a single published table**, for
   anyone who reopens the keyed-deployment direction (programme 4 above): Tanaka,
   Echizen & Kiya's Table VI (block-shuffle-encrypted ResNet-18 under a plain
   public surrogate, no attacker keys at all). If it replicates, the secret-key
   mechanism has already been shown fragile and the programme is over for 1% of
   its proposed budget; if it does not, the programme's central premise survives
   its first real test. Best risk profile of anything considered in this run: a
   defensible answer within a week either way.

---

## 8. The one thing that must be true, and the cheapest test of it

**The one thing:** the robustness reported on a float32 checkpoint must not be the
robustness the deployed artifact has.

If quantisation is lossless for robustness, the programme collapses into "an
ImageNet cost sheet for a model nobody will ship," which fails the student's test
with nothing left over. If quantisation costs real robustness, then the entire
small-model robustness literature has been reporting a number that does not
survive the format it would run in, and every future paper in that line owes a
retention row. That second world is worth a thesis chapter. The first is worth a
note.

**The cheapest experiment: run the retention measurement on checkpoints that
already exist, before training anything.**

- Models: Salman2020Do_R18, Salman2020Do_R50 and Singh2023 ConvNeXt-T+ConvStem
  from the pinned RobustBench zoo (VERIFIED present), Standard_R50 as a
  non-robust control, and EasyRobust's EfficientNet-B0 if it downloads.
- Formats: float32, int8 post-training quantisation, simulated 4-bit.
- Attacks, and report the minimum of the three: white-box through the fake-quant
  graph with straight-through gradients; transfer of the float32 adversarial
  examples to the quantised model; and Square on the real quantised graph.
- Data: the 5,000-image RobustBench ImageNet subset.
- **Cost: about 60 GPU-hours, about 60 CPU-hours, and 3 to 4 person-weeks**, most
  of which is building the quantisation and attack path rather than waiting.
- No training. No determinism contract. No campaign infrastructure.

Preregister the rule before looking: if the minimum-of-three retention is at or
above 0.95 at both int8 and 4-bit on every checkpoint, stop the programme. If it
falls below 0.9 anywhere, or if the three attack routes disagree by more than
about 5 points — which would mean the field's fake-quant evaluations are masking
gradients — the programme is justified and the training campaign should proceed.

Two properties make this the right first move. It is the gate and a deliverable at
the same time: whichever way it lands, nobody has published these numbers, so four
person-weeks buys a result rather than only a decision. And it fails fast in the
direction the evidence currently favours — the one reported measurement we have
says int8 retention is 0.995, which means the most likely outcome is that this
test stops the programme for 60 GPU-hours instead of 950.

---

## 9. The objection I cannot answer

The programme prices an insurance policy against a peril with no recorded claims.

Apruzzese et al. hand-reviewed 4,600 flagged samples from a live production
classifier and found zero gradient-based evasions. ATLAS v2026.08 contains one
typed Incident of perception-model evasion by input manipulation, and it is that
same study; the last perturbation-style case of any kind is dated 2021. So a cost
sheet for a robust mobile classifier prices a defense that no attacker has been
observed to require.

It is worse than that, and this is the part I genuinely cannot answer. The threat
model at mobile scale is close to incoherent. A white-box L-infinity attacker
against an on-device classifier must be able to compute gradients through the
quantised graph on the handset — which means they already hold the weights and
control the camera, at which point perturbing an input is the least interesting
thing they could do. The threat model that makes the measurement meaningful is
one in which the attacker's position already defeats the classifier by other means.

My reply is that the field will keep publishing float32 small-model robustness
numbers whether or not anyone checks whether they survive deployment, and that
checking is cheap. That is a reply about the hygiene of a literature. It is not a
reply about anyone's safety, and it is precisely the substitution the student said
they wanted to avoid: improving a real quantity, rigorously, for an audience that
is the research community rather than a deployer.

The honest position is that the correct budget for this may be zero, and that the
reason nobody has published the table is not an oversight but a correct allocation
of attention. I recommend it as the least bad of eight options that all fail the
same test, at a cost of 60 GPU-hours to find out whether it has content at all —
not because I can defend it against this objection.

If the student would rather not spend a thesis inside a threat model with no
recorded victim, the survey's own answer to "where would the same skills matter"
is the Magika direction in section 1.2: an open-source production model, a
published gradient attack that worked, a shipped defense whose effect size nobody
has independently measured, and a threat model with an actual incident behind it.
No programme in this set covers it well — programme 4 is the closest and appears
pre-refuted — so it would need a proposal written from scratch. That is a real
option and it should not be foreclosed by this document.
