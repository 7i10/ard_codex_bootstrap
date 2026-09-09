# Research restart: which direction, and why

- Written 2026-09-08, after eight candidate programmes were generated, judged, and three of the top four were attacked by red teams.
- Audience: the student. Written so that someone who ran none of this can follow it.
- Status: a recommendation. Every scientific decision in it is still the human's (hard rule 7).

---

## 1. The recommendation, and the margin

**Recommended: a floored audit of where the variance in adversarial distillation actually
lives — teacher choice, robust overfitting and checkpoint policy, protocol, and last of all
method — on CIFAR-10 ResNet-18, using only what is already built.**

Working title: **"The teacher sets the overfitting, not the ceiling."**

The margin is **large against all four red-teamed programmes as they were written, and
narrow against one alternative assembly.**

That sentence needs unpacking, because the ranking inverted after the red team ran.

The five judges scored the programmes before the assassins. Their order was Robustness
That Ships (33.2), Fail-safe at fixed inference cost (32.6), Borrowed Robustness (32.0),
Global Not Per-Sample (32.0). After the assassins, that order is not usable:

| Programme | Judge score | What the red team did to its headline |
|---|---|---|
| Robustness That Ships (quantization) | 33.2 | Headline destroyed. The INT8 delta is already published at −0.09 pp, and an assassin measured on this machine that 4-bit post-training quantization simply breaks the models. |
| Fail-safe (selective classification, cascades) | 32.6 | Killed. The primary experiment was published in 2022 with the same architecture, epsilon, and operating point. |
| Borrowed Robustness (ImageNet distillation) | 32.0 | Killed. The primary contrast is published at ImageNet scale with a smaller effect than the proposed detection threshold. |
| Global, Not Per-Sample | 32.0 | Primary contrast is a replication of a 2024 paper's ablation table. One block (student trained on its teacher's generated data) survived. |

So none of the four survives as its own headline. What I am recommending is assembled from
the pieces that did survive, and the margin over each programme *as written* is therefore
not a matter of taste — three of them lost their main experiment to prior art or to
measurement.

**Against one alternative assembly the margin is narrow.** That alternative is the
masking-aware evaluation protocol from the quantization programme, run as a standalone
instruments thesis. All five judges of that programme independently named it as the single
best idea in the document. I did not pick it, for three reasons given in section 5, and I
would put the gap at roughly four or five points on the judges' fifty-point scale — real,
but not overwhelming. If you dislike my recommendation, that is the one to look at next.

---

## 2. The recommendation in one page

### The question

When you train a small robust classifier by distilling from a large robust teacher, several
things vary between one paper and another: which teacher, how many epochs, which checkpoint
is reported, which augmentation, which training-set split, and which distillation method.
The field writes papers about the last of these. Nobody has measured the others against a
noise floor on one engine.

**This repository already contains evidence that the ranking is upside down.**
Verified in `docs/archive/ard-distillation-2026/EXPERIMENT_DASHBOARD.md`, all CIFAR-10 official test, 10,000 images,
standard AutoAttack, seed 0, identical student and schedule, only the teacher changed:

| Teacher / method | Best AA | Last AA | Best−Last |
|---|---:|---:|---:|
| Chen LTD / RSLAD | 51.90 | 51.78 | 0.12 |
| Chen LTD / Entropy | 51.06 | 51.00 | 0.06 |
| Chen LTD / Student | 51.46 | 51.41 | 0.05 |
| Chen LTD / Joint | 51.65 | 51.54 | 0.11 |
| Bartoldson / RSLAD | 47.11 | 43.12 | **3.99** |
| Bartoldson / Entropy | 47.37 | 46.16 | **1.21** |
| Bartoldson / Student | 46.89 | 43.07 | **3.82** |
| Bartoldson / Joint | 47.31 | 42.89 | **4.42** |

Two things jump out, and the second is not in any of the eight programmes.

1. The teacher moves the result by 4.79 pp at the best checkpoint and 8.66 pp at the last
   one. Method choice, across those same four methods, moves it by 0.5–0.8 pp. The
   repository's own assessment already states this (`docs/archive/ard-distillation-2026/ARD_VERSUS_AT_ASSESSMENT.md`
   line 205, marked VERIFIED) and notes the published SAAD numbers show the same thing
   (44.42 versus 40.57 across two teachers).
2. **The teacher effect is mostly an induced-robust-overfitting effect.** All four Chen arms
   have a best-minus-last gap under 0.12 pp. All four Bartoldson arms have a gap of
   1.2 to 4.4 pp. The stronger teacher also gives *higher clean accuracy* (83.4–85.2 versus
   82.9–83.4). This is a 4-against-4 pattern, not a single point, and it is already on disk.

That second observation is a mechanism claim that is different from the published account.
The 2026 literature explains weak students from strong teachers by which *samples* the
teacher cannot teach. This table says the student learns fine and then loses it — the
teacher changes the shape of the training trajectory, not the reachable ceiling.

### The claim, if it lands

On CIFAR-10 ResNet-18, official-test AutoAttack moves by roughly:

- ~5 pp for teacher choice (and most of that is overfitting the teacher induces),
- ~3–4 pp for protocol choice (epochs, weight decay, checkpoint policy, training split),
- ~0.1–1.5 pp for method choice,
- ~0.1 pp for the random seed.

And the ARD literature writes its papers about the third term while its baselines are set
by the second. That is a legible, defensible sentence, and every term in it is measurable
here with no new scientific surface.

### The chapters

1. **Teacher.** A dose–response curve: student official-test AA against teacher AA, five or
   six CIFAR-10 teachers spanning roughly 30 AA points, fixed student, fixed schedule, fixed
   objective, three seeds. Plus the overfitting decomposition — where in training the
   trajectories separate — from intermediate checkpoints.
2. **Protocol.** Why the same method with the same teacher family is published at 48.66,
   49.86, 51.03 and 52.13, and measured here at 51.90 (all repo-VERIFIED rows in
   `docs/archive/ard-distillation-2026/ARD_VERSUS_AT_ASSESSMENT.md`). Fix teacher, architecture, data and evaluation;
   vary one protocol axis at a time. Roughly ten to twelve runs, no new code.
3. **Selection.** Plan 0095, already written and unrun: the dose-matched, class-matched
   random-cohort placebo that closes the project's four-month negative result. 2.9 GPU-hours.
4. **Instruments.** The floors, the calibration, and the TRADES implementation-identity
   finding, written up as a chapter rather than as an apology.

---

## 3. The deployment test, applied to my own recommendation

Your test: *if this quantity improved by the amount the research could plausibly deliver,
whose situation actually gets better, and how would they notice?*

**The honest answer is: this recommendation's deployment value is weak.** I am not going to
dress it up.

Nobody's deployed model becomes more robust because of this thesis. The quantity being
moved is official-test AutoAttack at 8/255 on 32×32 CIFAR-10, which is exactly the kind of
number you correctly identified as a benchmark artefact. The leaderboard it belongs to has
been frozen since 2025-03-31.

There is one real user, and they are narrow. It is someone who must build a robust small
classifier on their own data, in their own label space, where no public robust checkpoint
exists — so they cannot download one, and distillation is genuinely on the table. Chapter 1
tells them that picking the teacher matters five to ten times more than picking the method,
and it gives them a rule or tells them the published rule does not transfer. They notice by
choosing a different teacher and gaining a few points. That user exists but there are not
many of them.

Everyone else is better served by an observation the red team surfaced and this repository
already records: the strongest small robust CIFAR-10 model is downloadable, not distilled.
By the repo's own VERIFIED table, the best teacher-free ResNet-18-class recipes sit at
52.46–52.48 without extra data, and generated-data recipes go substantially higher, with no
teacher at all. **Your own criterion — "ARD is meaningful if lightweight models can actually
reach good performance through it" — is close to already answered, and the answer is that
they reach it without a teacher.** That is worth knowing before you spend fourteen weeks.

So why do it anyway?

- Because the three programmes that tried harder to pass your deployment test were killed
  or pre-empted. Applied strictly in this environment, on this budget, the deployment test
  may have no affordable survivor. I would rather say that than pretend otherwise.
- Because this is the only direction where the effect being measured is 30 to 50 times the
  noise floor. Every one of the last four months' failed screens had a signal-to-floor ratio
  near one. That is the actual lesson of the audit, and this direction is the one that
  applies it.
- Because "which axis carries the variance" has to be answered before any deployment claim
  about distillation can be made at all. If the teacher is worth 5 pp and the method 0.5 pp,
  then every paper comparing methods at n=1 is reporting something else.
- Because it can be finished. Nothing in it needs a new dataset, a new architecture, a new
  objective, or a new attack.

Call it what it is when you write it: **a measurement thesis with a research audience.**
Do not put a deployment story on the cover.

---

## 4. What the red team killed, and what it could not

This section separates "survived an attack" from "survived because nobody looked."

### Killed, with the evidence named

- **Fail-safe / selective classification (1 kill vote).** The primary experiment — confident
  error at 95 % clean coverage, ResNet-18, CIFAR-10, 8/255, under an attacker that jointly
  optimises acceptance and misclassification — was published in 2022 with the same
  architecture, epsilon, threshold convention and baseline set, and again in 2023 with the
  two-objective BPDA attack and a rejection-loss curve. The cascade half was published in
  2020 and 2023. *(Red-team claim, from PDFs the assassin extracted; I did not
  independently re-verify these papers today.)*
- **Borrowed Robustness / ImageNet distillation (1 kill vote).** The primary contrast —
  RSLAD versus a same-budget teacher-free control on ImageNet-1k at 4/255 under AutoAttack —
  is published, at +1.71 pp, which is *below* the programme's own 2.0 pp detection threshold.
  Separately, the headroom on the primary student is capped by its own source paper at
  +2.0 pp for 250 extra epochs. *(Red-team claim, not independently re-verified here.)*
- **Robustness That Ships / quantization (0 kill votes, headline dead anyway).** One assassin
  found the INT8 robust-accuracy delta already published at −0.09 pp, which is inside this
  project's own noise floor. A second assassin implemented the proposed quantization recipe
  on this machine and measured, in clean accuracy alone: ResNet-50 79.7 → 79.5 at W8A8 and
  → 0.6 at W4A8; ConvNeXt-T 84.2 → 83.8 → 56.8. So the only deployed precision is a
  confirmed null, and the precision with a signal breaks the model. **That measurement was
  made in under an hour on this hardware, which is the most useful single fact the red team
  produced.**
- **Global, Not Per-Sample, Block A.** The teacher-versus-flatness factorial was run and
  published in 2024, including the saturation conclusion, in that paper's own limitations
  section. *(Red-team claim, not independently re-verified here.)*

### Survived a real check

- **Plan 0095, the placebo.** An assassin searched specifically for prior art and found the
  nearest neighbours: a 2026 paper runs a *size*-matched random control, and a 2025 paper
  runs a placebo teacher for standard knowledge distillation. Neither is a dose-matched,
  class-matched random-cohort control for the adversarial per-sample reweighting family.
  The narrow claim survived a search that was trying to kill it.
- **Global, Block B** (a distilled student trained on the generated data its teacher was
  trained on). The assassin stated explicitly that six searches and two arXiv API queries
  found nothing, and reported survival rather than manufacture a kill.
- **The DAJAT calibration.** Two assassins tried to discredit it and both failed. It is real:
  full 10,000 images, clean 85.71 against a published 85.71, AutoAttack 52.45 against a
  published 52.48.
- **The pinned AutoAttack package is extensible without breaking its hash.** An assassin
  verified that `autopgd_base.py` calls `self.model(x_adv)` on an arbitrary callable and that
  the package supports a custom attack list. So a custom-objective attack can be a *model
  wrapper*, leaving the pinned source byte-identical and the provenance guard intact. That is
  a genuinely useful engineering discovery and it is worth writing down.

### Survived because nobody checked — flag these

- **The protocol-attribution chapter (my chapter 2) was proposed by an assassin as a pivot
  and was never itself red-teamed.** The nearest prior art I am aware of is a 2023 paper that
  reimplemented a published adversarial-training recipe from its released code and found a
  3.66 pp gap, making irreproducibility a headline finding. *(Judge's claim about a programme
  that did not reach the red team; not verified here.)* **Before week 3, do a proper prior-art
  search on reproducibility studies of adversarial distillation specifically.** If that 2023
  paper covers the ARD case too, chapter 2 shrinks to a replication.
- **The teacher dose–response chapter (my chapter 1) was proposed by an assassin and never
  red-teamed, and it is partly occupied.** A 2026 ICML paper reportedly correlates a
  teacher-side quantity against student robustness across roughly nineteen public teachers,
  with ResNet-18 and MobileNetV2 students. *(Judge's claim; the repo's own VERIFIED row for
  that paper records only its Table 2 numbers, not the correlation analysis. Confirm this
  before writing any novelty sentence.)* If it is accurate, chapter 1 is a
  replication-with-instruments plus the overfitting mechanism, not a discovery. **The
  overfitting angle is the part I believe is genuinely unclaimed, and it is the part to lead
  with.**

---

## 5. Grafts: what I took, from where, and what I rejected

### Taken

| Element | From | Where it goes |
|---|---|---|
| The dose-matched, class-matched random-cohort placebo (plan 0095) | Global, Not Per-Sample (Block C) — named top steal by all five of its judges | Chapter 3, run in week 1. It also produces the missing floor for a treated-versus-control contrast that shares a random stream, which the plan document itself says several past verdicts needed. |
| The teacher dose–response curve | An assassin's rescue of Borrowed Robustness | Chapter 1. Its argument: this project has spent four months trying to resolve 0.3 pp effects against a 0.3 pp floor, and the teacher axis has an in-house 4.8–8.7 pp signal. |
| Protocol attribution of the cross-paper spread | An assassin's rescue of Global, Not Per-Sample | Chapter 2. |
| "Prefer directions where the manipulation is post-training and the comparison is within-checkpoint" | An assassin's steal from the quantization programme | A selection *rule* for the rest of the thesis, not an experiment. Chapter 1's overfitting decomposition obeys it: evaluating existing epoch-99/149/199 checkpoints has no training-seed variance at all. |
| The pseudo-logit wrapper trick (custom attack objectives without touching the pinned AutoAttack source) | An assassin's repo reading during the Fail-safe kill | Chapter 4, if any non-standard attack is ever needed. It preserves the one external calibration anchor. |
| Teacher accuracy measured on the *training-view* distribution, not the eval transform | An assassin's rescue of Borrowed Robustness | Chapter 1, week 1. Distillation queries the teacher on augmented views; a teacher degraded off-distribution supplies a weakened signal and manufactures a fake teacher effect. Nobody else raised this and it is cheap. |
| Report every arm's compute in GPU-hours in every table | Global, Not Per-Sample | House style. |

### Rejected, and why

- **The masking-aware per-sample-minimum evaluation protocol** (all five quantization judges'
  top steal, and genuinely the best single idea in the eight documents). Rejected as a spine
  for three reasons. It needs a quantization or pruning module across many heterogeneous
  foreign architectures, and an assassin's module-swap prototype broke on the first
  transformer it touched. The regime where it would fire has been shown, on this machine, to
  be a regime where the models are simply broken, so the finding would be confounded with
  clean-accuracy collapse. And its falsification target sits at precisions nobody runs CNN
  inference at. **It is still the best runner-up, and if chapter 1 dies at the week-2 gate it
  is the thing I would look at before starting anything else.**
- **Anything requiring ImageNet.** `DatasetConfig.name` is a closed literal of four
  CIFAR-scale sets, `image_size` defaults to 32, `build_architecture` offers a handful of
  CIFAR architectures with no timm, and the teacher registry refuses any epsilon but 8/255.
  Full ImageNet is on disk and it is a real asset, but reaching it is three to six weeks of
  fail-closed contract work, not a config line. It is the right first experiment for whoever
  comes after you.
- **Anything requiring AWP or SWA on the critical path.** Neither exists in `src/ard` (one
  grep hit, a protocol descriptor). Three assassins independently identified the same trap:
  the only way to validate a new AWP implementation is to reproduce a published number, and
  this project's record on that is one success (PGD-AT at 47.63), one four-point failure
  (TRADES), and one still-unexplained 1.2–1.5 pp residual. Build it if there is time; do not
  make any headline depend on it.
- **Anything requiring 50,000-image training.** Every config here sets
  `validation_fraction: 0.1`, and best-checkpoint selection depends on that split. Every
  published comparator trains on 50,000. This is a permanent asterisk on cross-paper
  comparisons and it must be labelled, not hidden — the red team flagged the repo for putting
  45k rows beside 50k rows once already.
- **Generated data.** `docs/archive/ard-distillation-2026/METHOD_DIRECTIONS_V2.md` currently parks it as outside the data
  contract and a human decision. Block B is the one genuinely unoccupied cell in the eight
  programmes, but reopening that contract, building a mixed loader that survives the
  determinism guarantee, and running 400-epoch arms is a second programme. Keep it as the
  named follow-up.

---

## 6. What carries forward, concretely — and what is lost

### Carries forward and is load-bearing

- **The bit-deterministic engine** and fork-from-parent checkpointing. This is what makes
  paired arms and the 0.092 pp paired floor possible at all. Verified across five independent
  lines, two hosts, two git SHAs and a four-month gap.
- **`src/ard/evaluation/autoattack.py`** with its `EXPECTED_AUTOATTACK_SOURCE_SHA256`
  provenance guard, pinned at `version="standard"`.
- **`scripts/evaluate_external_checkpoint.py`** — 131 lines, CIFAR-10 only, a handful of
  architectures. Enough for CIFAR teachers and foreign CIFAR checkpoints. **Not** enough for
  ImageNet or arbitrary foreign architectures; the "carries forward" claim in two of the eight
  programmes overstated this and an assassin caught it.
- **The DAJAT calibration**: clean 85.71 vs 85.71, AutoAttack 52.45 vs a published 52.48,
  −0.03 pp, on all 10,000 images. This is the single most valuable artefact of the four
  months, and it survived two attempts to discredit it.
- **The measured floors**: 1.14–1.25 pp before the learning-rate decay; 0.092 pp paired
  post-decay at epoch 114 with a 95 % interval of [0.055, 0.265] on four degrees of freedom.
  Plus, verified today, the three-seed official-test AutoAttack spread of the I100 arm:
  53.08 / 53.02 / 53.19 best-checkpoint, and 53.08 / 53.02 / 52.96 last — a standard
  deviation near 0.09 pp. **Use that last number, not the 0.092 pp paired figure and not the
  validation CE-PGD20 figure, when sizing anything against official-test AutoAttack.** Two
  assassins caught the previous designs importing the wrong floor.
- **The eight teacher arms** in `docs/archive/ard-distillation-2026/EXPERIMENT_DASHBOARD.md` section 4 — the table in
  section 2 above. These are chapter 1's pilot data and they already exist.
- **The pinned RobustBench checkout** (`.external/robustbench`, commit 78fcc9e, 2025-03-31)
  and `teachers.lock.yaml` with seven pinned teachers, per-checkpoint SHAs and a factory
  allowlist.
- **`docs/archive/ard-distillation-2026/ARD_VERSUS_AT_ASSESSMENT.md`** — the eight-paper table with VERIFIED / REPORTED /
  INFERRED marks on every row. This is chapter 2's literature side and it is already done. It
  even contains a design sketch for the matched comparison at line 237.
- **Plan 0095**, written, costed at 2.9 GPU-hours, and runnable on the existing state router
  with no new code.
- **The campaign machinery**: `launch_gate.py`, `orchestrate.py`, `scripts/ardx/`,
  `ardx-watch.service`, the headless postrun, hash-bound records, lineage-refusing
  aggregators, preregistration and decision packets.
- **The I100 augmentation schedule** and its three-seed official-test result — the project's
  one reproduced improvement, and it needs no teacher.
- **Two results that are results, not machinery**: the per-sample negative (cohorts move
  +1.68 to +4.41 pp, held-out gain unresolvable across four campaigns and three intervention
  families), and the TRADES implementation-identity finding (a one-line detached target cost
  2.73 pp of official-test AutoAttack, was documented as intentional, and was caught by no
  test).
- **696 checkpoint files on the host**, including some intermediate epoch-99 and epoch-149
  checkpoints. Verify which arms still have them in week 1 (see section 8).

### Genuinely lost

Say this plainly, because it is a lot.

- **The entire per-sample line as machinery.** The state routers, the S1/S2/S3 predicates,
  the teacher-margin gates, the boundary-distance signals, the sample-weighting objectives,
  and roughly fifty `ert_*` CLI entry points. Four months of engineering whose surviving
  product is one negative result, one placebo plan, and the instruments. That is a real loss
  and it is the correct thing to accept rather than defend.
- **The unexplained TRADES residual.** After the detach fix, in-house TRADES sits at 45.14 AA
  best-checkpoint against a literature range of roughly 49.0–49.4. The evaluation stack and
  the attack initialisation have both been ruled out by measurement. Anything that needs a
  TRADES comparator inherits this. Chapter 2 can *study* it; nothing should *depend* on it.
- **AWP and SWA.** Never built. Any comparison against the real teacher-free frontier
  (52.46–52.48 on this architecture without extra data) needs them, or needs a foreign
  checkpoint under a foreign protocol, which is the mismatch the whole thesis criticises.
- **50,000-image training.** Structurally unreachable without changing the split function and
  losing the validation split that best-checkpoint selection uses.
- **ImageNet capability.** The 146 GB is on disk and unusable by this engine today.
- **Automatic mixed precision.** Disabled everywhere for determinism, so anything large is
  fp32.
- **Calendar.** The commit log shows something like two to three weeks per fortnight going to
  infrastructure. Budget for it; do not assume it stopped.

---

## 7. Fourteen weeks

Experiments stop mid-December. Weeks 13–14 are already writing.

### Week 1, in detail

**Day 1 — zero GPU-hours.**
1. Write down the overfitting split from the section-2 table. Four Chen arms under 0.12 pp
   best-minus-last; four Bartoldson arms at 1.21–4.42 pp. This is the thesis hypothesis and
   it costs nothing.
2. Locate and *tag* the checkpoints for the eight teacher arms. I found the Chen arms and two
   Bartoldson arms under
   `/home/shunsukenaito/workspace-local/ard-campaign-runs/ard_codex_bootstrap/c10-r18-ws1-b128-core-s0-v1-2d54b82/outputs/production/`,
   but I could not find `bartoldson-rslad-s0` there. Find it, or establish it is gone. The
   retention policy classes these as cheap-to-regenerate with no backup, so this is a real
   risk and it is an hour's work to remove.
3. Tabulate, from rows already on disk, the untreated cohort's change beside the treated
   cohort's gain for every treated cohort in the Stage-A and T123 records. Free. If the
   training set conserves, the capacity-limited reading of the four-month negative is
   supported before anything is spent.

**Day 1–2 — 2.9 GPU-hours.** Launch plan 0095, the placebo. No new code. Preregistered rule
already fixed in the plan.

**Day 2–3 — ~8 GPU-hours.** The first experiment, described in section 8.

**Day 3–5 — engineering and preregistration.**
- Choose five or six CIFAR-10 teachers from the pinned zoo spanning roughly 30 AA points, in
  at least two architecture families. Add adapters. **Each teacher must reproduce its own
  published clean and AutoAttack accuracy through this stack before it is admitted as a
  rung** — that is the DAJAT move, applied five more times, and it is what makes chapter 1
  believable.
- Measure each candidate teacher's accuracy on the *training-view* distribution as well as
  the eval transform.
- Write the preregistration for chapters 1 and 2. Derive the minimum detectable effect from
  the three-seed official-test AutoAttack spread (≈0.09 pp, two degrees of freedom, so state
  the interval honestly and do not quote a point estimate as if it were known). Fix the
  estimand: teacher effect at the **best** checkpoint and at the **last** checkpoint,
  reported separately, never averaged.

### Week 2, in detail

- Read the placebo. Write the decision packet.
- **The go/no-go.** Two extra seeds each of Chen/RSLAD and Bartoldson/RSLAD, evaluated on
  the official test set. Four training runs at about 5.1 GPU-hours plus four AutoAttack
  evaluations: ~24 GPU-hours, well under one cluster-day. Rule, written before the numbers
  are read: *the teacher axis is real if the three-seed best-checkpoint difference has a
  95 % interval excluding 2.0 pp.*
- Launch chapter 2's protocol-attribution campaign off the canonical RSLAD(Chen) arm, one
  axis at a time: 200 versus 300 epochs, weight decay 2e-4 versus 5e-4, the decay schedule,
  the augmentation, and best versus last (free). Ten to twelve runs, roughly 60–70
  GPU-hours. The 45k-versus-50k axis needs a schema change and a decision packet — raise it
  now, run it later or not at all.
- Finish the teacher-rung reproductions and freeze the ladder.

### Weeks 3–14, in outline

- **Weeks 3–5.** Chapter 1's main campaign: five or six teachers × three seeds, roughly
  90–110 GPU-hours of training plus 30–36 of AutoAttack. Evaluate intermediate checkpoints,
  not just best and last — that is where the overfitting story lives and it is cheap.
- **Weeks 6–7.** Read chapter 1. Decision packet. Test whether any teacher-side quantity
  measured *before* student training predicts the student's AA out of sample, using
  leave-one-out rather than in-sample correlation. If the teacher effect is real, add the
  second student already in the registry (MobileNetV2-CIFAR) to get a capacity × teacher cell.
- **Weeks 8–9.** Extend chapter 3: run the placebo against one published per-sample method
  with released code, in addition to the project's own three families.
- **Weeks 10–11.** Confirmations and reruns. Plus one thing nobody else will do: no
  distillation-trained small model appears on RobustBench, so no distilled arm here has an
  external attack anchor. Spend ~10 GPU-hours re-evaluating one checkpoint per family under
  a second, stronger attack setting (more targeted APGD restarts, plus a cross-arm transfer
  matrix). If AutoAttack's slack is not uniform across training objectives, you need to know
  before you compare arms at sub-point precision.
- **Week 12.** Hard stop on new launches. Aggregate, import records, close plans, write
  decision packets.
- **Weeks 13–14.** Buffer and writing.

### The point at which you abandon this too

**End of week 2.** Two conditions, both preregistered:

1. The three-seed best-checkpoint teacher difference does **not** clear 2.0 pp, **and**
2. The placebo returns a result in the ambiguous band (the plan's own rule: |P| ≥ 0.15 and
   P ≤ 0.25).

If both hold, the two chapters with large effects are gone. **Do not start a fourth
direction in October.** Write the measurement thesis you already have — the four-month
negative result, the reproduced I100 improvement with three official-test seeds, the TRADES
identity finding, the DAJAT calibration, the floors — and spend the remaining ten weeks
making it airtight: floors on the endpoint the field actually reads, the second-attack
check, and the DAJAT-style calibration extended to two or three more foreign checkpoints.
That is a thinner document than this one promises, and it is defensible.

**A second, softer gate at end of week 7.** If the dose–response curve is flat within the
floor across 30 AA points of teacher strength, chapter 1 becomes a null — which contradicts
both the in-house table and the published SAAD spread, and is therefore itself interesting.
In that case weeks 8–14 go entirely to chapters 2 and 3.

---

## 8. The one thing that must be true, and the cheapest test

**The single thing that must be true:** the teacher axis moves official-test AutoAttack by
several points, and the Chen-versus-Bartoldson difference — especially the difference in
robust overfitting — is not a seed-0 accident.

Everything else in the programme is downstream of that. The whole design rests on effect
sizes measured at n = 1 per teacher, and this project's own four-month history is that
n = 1 effects did not survive replication. That is the risk, stated plainly.

### The first experiment costs about 8 GPU-hours. Run it before anything else.

**Run AutoAttack on the intermediate checkpoints that already exist.** Take
Chen/RSLAD and Bartoldson/RSLAD (and, if their weights survive, one more method pair), and
evaluate the epoch-99, epoch-149 and epoch-199 checkpoints on the official 10,000-image test
set. That is roughly eight AutoAttack runs at about 1 GPU-hour each. **No training at all.**

Why this is the right first test:

- It turns a two-point best/last comparison into a trajectory, and shows *where* the two
  teachers separate. If the Bartoldson student tracks the Chen student until the
  learning-rate decay and then falls, the overfitting mechanism is confirmed and it is the
  thesis. If it is behind from epoch 99, the teacher limits the ceiling and the story is the
  published one.
- It has **no training-seed variance at all**, because the weights are fixed. Its only noise
  is attack randomness, which this project has already characterised. That is a
  signal-to-floor ratio of roughly 40:1, against the ~1:1 the last four months lived with.
- It reuses the calibrated evaluation stack and needs no new code.

**Two caveats, both checkable in an hour.** First, confirm the intermediate checkpoints
exist for those arms — the host holds 696 checkpoint files including some epoch-099 and
epoch-149, but the retention policy does not guarantee them, and I could not locate a
`bartoldson-rslad` output directory. Second, if the intermediate checkpoints are gone,
retraining the two arms with three seeds is the fallback and the first experiment becomes
the ~24 GPU-hour week-2 gate instead.

**Also under 10 GPU-hours, and also for week 1:** plan 0095, the placebo, at 2.9 GPU-hours
with zero new code. And the cohort-conservation tabulation, at zero GPU-hours, from records
already on disk. There is no version of the next fortnight in which those two are not worth
doing.

---

## 9. The strongest objection I cannot answer

You said the topic has to be useful in the real world. I am recommending a thesis whose
reader is another researcher.

The defence I would like to make is that measuring which axis carries the variance is a
precondition for any deployment claim about distillation. That is true, and it is not
enough. The honest position is this: the three programmes that took your deployment test
most seriously were all killed — one by a paper from 2022, one by a paper from 2023, one by
a measurement an assassin made on your own GPU in under an hour — and what is left that can
be finished in fourteen weeks and defended in January is a careful audit of a benchmark
number on a 32×32 dataset, sitting behind a leaderboard that stopped accepting entries
eighteen months ago.

I cannot show you that this is more useful than that. What I can say is that if you apply
your test with full strictness to every one of the eight programmes, including this
recommendation, **none of them passes**, and the difference between them is only how long it
takes to notice. Choosing this one is choosing the direction where the measurement is
honest, the effect is large enough to see, and the failure mode is a small true result rather
than a large false one.

A hostile examiner will ask: *"so you spent a year discovering that a benchmark number is
sensitive to things other than the method."* The reply is that this is true, that the field
did not know it with error bars, and that nobody else had a bit-deterministic engine, a
calibrated attack stack and a measured floor with which to establish it. That reply is
adequate. It is not a triumph.

