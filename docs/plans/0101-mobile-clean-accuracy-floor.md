# 0101 — Mobile-scale clean-accuracy floor, then robustness on top

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Depends on: plan 0100 (ImageNet Stage 0/1, still running stage 1 as of
  this plan's creation) — this plan does **not** modify plan 0100's frozen
  contract (decided in chat: a new plan, not an amendment). Plan 0100's own
  stage-1 outcome and this plan's outcome are compared side by side, not
  merged.
- Current milestone: Stage A (zero-training-cost pretrained-checkpoint
  clean-accuracy sanity check) passed. Engine changes (architecture
  registration, ε-warmup schedule, trainer wiring) implemented and tested
  (`scripts/verify.py --changed` green). Scientific review requested before
  Stage B (a real, bounded GPU canary) — **Stage B itself has not launched**;
  it needs the human to confirm the "Option 1 vs. Option 2" ε-warmup design
  fork below before any GPU time is spent on it.

## Context

Plan 0100 (`docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md`) is
testing whether ADR's CIFAR-10-confirmed capacity-inverse self-distillation
effect (plan 0097, decision packet 0009) holds at ImageNet mobile scale.
Stage 1's seed-0 data came back with a real, unexplained-by-noise problem:
on MobileNetV3-Small (2019 torchvision checkpoint, ~67.4% published clean
top-1), this project's own PGD-AT recipe (pretrained non-robust init, 50
epochs, 3-step training PGD, ε=4/255) only reaches ~46% clean / ~25%
training-time-PGD accuracy, and `adr` does markedly worse on both axes
(~32% clean / ~21%). Root-cause analysis (plan 0100's Progress log,
2026-09-13 entries) traced the `adr`-specific failure to a real, plausible
mechanism (the EMA self-distillation target's true-class mass keeps eroding
through the LR-decay recovery window), but a literature survey run
alongside it (this session) surfaced a bigger, prior question: is 46% clean
even a reasonable ceiling for this architecture under this recipe, or is
this project leaving real accuracy on the table before robustness is even
considered?

The survey's answer: likely the latter. MobileNetV4-Conv-Small (Google,
arXiv:2404.10518, 2024) reports 73.8-75.5% clean top-1 at an almost
identical parameter budget (3.8M vs MobileNetV3-Small's 2.5M) using a
modern (2024) training recipe — a +6-9pp jump over MobileNetV3-Small's
2019-era number, for essentially the same "mobile scale" claim, before any
adversarial training is applied. The user's own framing, and the working
premise for this plan: **a model that cannot reach a reasonable clean
accuracy has no chance at a reasonable robust accuracy either** (robust
accuracy is bounded above by clean accuracy in practice), so the correct
sequencing is to first re-establish a strong clean-accuracy floor on a
genuinely mobile-scale architecture, and only then layer adversarial
training (and, later, self-distillation) on top — not the reverse.

Three further findings shape this plan directly:
- **ε-warmup** (Debenedetti, Sehwag, Mittal, "A Light Recipe to Train
  Robust Vision Transformers", arXiv:2209.07399, SaTML 2023): ramping the
  training attack's ε from near-zero up to the target budget over the
  first several epochs, instead of training at full ε from epoch 0,
  measurably improved *both* clean and AutoAttack robust accuracy for a
  small (3M param) architecture, and rescued a different architecture from
  outright training collapse. This directly targets the same failure shape
  this project just observed (MobileNetV3-Small's clean accuracy collapsing
  early and never recovering). The exact mechanism was fetched from the
  paper + the authors' own released code this session (not paraphrased)
  and is more specific — and less directly compatible with this project's
  existing attack code — than first assumed; see "ε-warmup" below.
- **A directly on-point 2025 paper formalizes exactly this project's
  pathology**: "Robust Fine-Tuning from Non-Robust Pretrained Models:
  Mitigating Suboptimal Transfer With Epsilon-Scheduling" (arXiv:2509.23325,
  Sept 2025) diagnoses that adversarially fine-tuning from a *non-robust*
  backbone (this project's own approach, per Singh/Croce/Hein 2023) causes
  the robust objective to "impede task adaptation" early in training,
  producing an avoidable clean-accuracy drop, and proposes epsilon-
  scheduling as the fix — independent confirmation, from a source specific
  to this exact transfer setting, that this plan's direction is the
  literature-supported fix rather than a guess. Full mechanism not yet
  read in detail (only abstract-level content was fetchable this session);
  read before finalizing the ε-warmup design if its schedule differs from
  Debenedetti's.
- **This project's exact niche remains an open gap, on both axes.** No
  paper was found adversarially training a genuinely small (2-10M param)
  **CNN** on full ImageNet-1k with both clean and AutoAttack/AutoPGD
  numbers reported (the closest transferable prior art is Debenedetti's
  *methodology*, validated on ViTs, not CNNs). Separately, no paper was
  found applying ADR-style EMA self-distillation to a sub-10M-param
  ImageNet-1k model (the two candidates found, ProARD arXiv:2506.07666 and
  CIARD arXiv:2509.12633, both 2025, use CIFAR/Tiny-ImageNet and different
  mechanisms — dynamic-supernet and multi-teacher-contrastive respectively,
  not EMA-of-student). Plan 0100's ADR-on-mobile-ImageNet question is not
  obsoleted by this session's literature pass — it strengthens the case
  that its outcome will be genuinely new information once this plan's own
  recipe question is resolved.
- **ConvStem is not applicable here.** Singh/Croce/Hein 2023's "ConvStem"
  replaces an aggressive *patchify* stem (ConvNeXt/ViT-style, one stride-4
  layer) with a stack of smaller-stride convs. MobileNetV3-Small and
  MobileNetV4-Conv-Small both already use a conventional small-stride conv
  stem (confirmed by inspecting the actual `timm` module tree —
  `conv_stem: Conv2d(3, 32, kernel_size=3, stride=2)`); there is no
  patchify stem here to fix. Decided in chat: adopt MobileNetV4 and skip a
  separate ConvStem change rather than force an inapplicable technique.

A follow-up sweep (this session) confirmed the architecture choice against
2023-2026 alternatives: RepViT-M0.9/M1.0, StarNet-S3/S4, EfficientViT-B1
and MobileCLIP-MCi0 all report higher top-1 than MobileNetV4-Conv-Small,
but only by spending materially more capacity (+34% to +210% params) — not
a fair same-budget comparison. One genuine, weight-matched candidate
exists, **FastViT-T8** (Apple, arXiv:2303.14189, ICCV 2023, 4.0M params
timm / 3.6M paper, 76.7% top-1 vs. MobileNetV4's 75.5% at 3.8M), but its
published number uses 256×256 test resolution vs. MobileNetV4's 224×224 —
not a clean comparison without re-validation. **Decision: keep
MobileNetV4-Conv-Small as the primary choice; FastViT-T8 is a documented,
not-yet-pursued secondary candidate.**

## Goal

Establish, cheaply and in stages, whether a modern mobile-scale checkpoint
(MobileNetV4-Conv-Small) plus an ε-warmup training schedule can raise this
project's own adversarially-fine-tuned clean-accuracy ceiling for a
genuinely mobile-scale (<5M param) ImageNet-1k architecture into a
plausible range (the user's stated target: ~70% clean, i.e. close to the
modern *standard*-training number, not the ~46% this project's current
recipe reached on the old architecture) — before spending any GPU time on
self-distillation or other robustness mechanisms on top of it. This plan
answers *only* the clean-accuracy-floor question; it does not re-attempt
ADR. What (if anything) to layer on top once a strong floor exists is a
follow-up decision, not part of this plan's scope. Every technique here
(MobileNetV4, ε-warmup) is adopted from existing literature, not invented
— the user's longer-term interest is a genuinely novel proposed method,
which requires this floor to exist first and is explicitly deferred past
this plan.

## Concrete changes, implemented this session

### 1. `mobilenetv4_conv_small_imagenet` (`src/ard/models/registry.py`)

Uses timm's `create_model("mobilenetv4_conv_small.e1200_r224_in1k",
pretrained=..., num_classes=...)` — unlike `resnet18_imagenet`/
`mobilenet_v3_small_imagenet` (torchvision, need `_with_replaced_head`),
timm's own `pretrained=True` already replaces the classifier head
correctly for `num_classes != 1000`, matching the existing
`convnext_tiny_imagenet` lazy-`timm`-import precedent. Added to
`ModelConfig.architecture`'s `Literal` and the `pretrained` allow-list in
both `registry.py` and `schema.py`'s `validate_pretrained`. New protocol
id `controlled_imagenet_stage01_mobilenetv4_pgd_at_v1` registered in
`src/ard/protocols/__init__.py` (deliberately not added to
`_validate_protocol_contract`'s strict CIFAR-protocol field-matching
allowlist, matching plan 0100's own precedent and rationale for its single-
campaign identity). Verified: 3,774,024 params (matches the paper's 3.8M).

### 2. ε-warmup — the exact verified mechanism, and a real conflict with this project's own attack guard

`AttackRequest` already carries `epsilon_override`/`step_size_override`
(per-example budget tensors, `src/ard/attacks/base.py`), consumed by
`LinfPGD.generate()`'s `_budget()` helper — built for an existing "mixed
selected attack budget" mechanism (the hardness-allocation plans, e.g.
plan 0096), already scientific-reviewer-approved and tested. A uniform-
across-batch override for a global, epoch-indexed warmup is an in-kind
reuse of this field — *if* the mechanism fits through it cleanly, which it
does not, per the exact mechanism below.

**Exact mechanism**, fetched from the paper (arXiv:2209.07399) and the
authors' released code (`github.com/dedeswim/vits-robustness-torch`,
`src/attacks.py`, `src/setup_task.py`):
- Linear ramp, epoch-indexed: `eps(t) = (t/period) * final_eps` for
  `t < period`, else `final_eps`. Full-ImageNet-1k recipe: `period = 10`
  epochs.
- **Step size is fixed at the target epsilon's own ratio for the entire
  run** (`step_size = 1.5 * final_eps / attack_steps`, computed once from
  `final_eps`, not the scheduled `eps(t)`) — during warmup, step size is
  *larger* than the current, still-small allowed radius for most of the
  ramp; only the random-start range and projection radius shrink.
- Attack is 1-step (FGSM with random start) for the entire run — the step
  count never changes, unlike this project's own established 3-step
  training PGD (plan 0100, sourced from Salman et al. 2020's convention).
- Other exact deltas (weight decay 0.5 with AdamW vs. canonical 0.05;
  MixUp/CutMix/RandAugment/RandomErasing all dropped; batch 512; peak LR
  5e-4 cosine) are a whole different optimizer/augmentation recipe, not
  just an epsilon schedule.

**The concrete conflict**: `pgd.py`'s existing guard,
`if bool((step_tensor > epsilon_tensor).any()): raise ValueError(...)`,
exists to catch a nonsensical per-sample budget. Debenedetti's verified
mechanism deliberately violates exactly this invariant during warmup
(fixed step size, shrinking epsilon) — replicating it faithfully would hit
this project's own guard and crash. Reconciled as:

- **Option 1 (implemented, this session's default)**: couple step size to
  the *current* ramped epsilon (`step_size(t) = ratio * eps(t)`, using this
  config's own existing `step_size_value / epsilon_value` ratio rather than
  a hardcoded 2/3) instead of Debenedetti's fixed-at-target value. Keeps
  the existing guard meaningful and untouched, provably satisfies it by
  construction (`ratio <= 1` whenever the baseline config itself is valid),
  requires no attack-layer change. **Not the exact mechanism that produced
  Debenedetti's measured effect** — this must be stated plainly in any
  report, not blurred into "we did ε-warmup per Debenedetti et al." Also
  keeps this project's existing 3-step training PGD rather than
  Debenedetti's 1-step FGSM.
- **Option 2 (not implemented, documented alternative)**: replicate the
  mechanism exactly (fixed step size, 1-step FGSM) and make the guard's
  scope explicit for this case, with its own regression test. More
  faithful to the literature, bigger and riskier (also changes training
  attack steps 3→1), needs explicit human sign-off (CLAUDE.md rule 6
  protects "steps").

**This is the fork the human must confirm before Stage B's GPU time is
spent** — implemented as Option 1 in this session's code, but the choice
itself was not re-confirmed with the human before implementation (the
approving turn happened while the human had stepped away and asked for
autonomous progress); flag this explicitly at the next check-in.

Implementation: `src/ard/schedules/epsilon_warmup.py` (`epsilon_warmup_value`,
linear, epoch-indexed, mirrors `warmup_multistep_multiplier`'s convention),
`TrainingConfig.epsilon_warmup_epochs: int | None = None`
(`src/ard/config/schema.py`, default preserves today's exact behavior),
wired into `Trainer.train_epoch`'s training-attack call only
(`src/ard/engine/trainer.py`) — mutually exclusive with the existing mixed-
selected-attack-budget mechanism (raises in `Trainer.__init__` if both are
configured; not a combination this plan uses). Selection and evaluation
attacks are structurally untouched (their `AttackRequest` construction in
`validate_epoch` never sets these fields at all).

### 3. New configs (two, after the P1-1 fix -- see Staged verification)

`configs/scientific/imagenet_mobilenetv4_pgd_at.yaml` (Stage B arm B) and
`imagenet_mobilenetv4_pgd_at_no_warmup.yaml` (arm A) — same dataset,
threat model, optimizer, and LR schedule as plan 0100's
`imagenet_mobilenetv3_pgd_at.yaml`; deltas: `student.architecture` (both)
and `training.epsilon_warmup_epochs` (arm B only, `: 10`; arm A leaves it
unset). Both load and resolve end to end (verified via
`ard.config.loader.load_config`) and are proven field-identical to each
other and to plan 0100's config except the intended deltas
(`tests/unit/test_config.py`).

## Tests

- `tests/unit/test_epsilon_warmup_schedule.py` — hand-computed values at
  the first ramp fraction/mid/last-warmup-epoch/post-warmup, the
  `warmup_epochs=0` no-op case, negative-input rejection.
- `tests/integration/test_epsilon_warmup_trainer.py` — fixture-scale
  (`fixture_cnn`/`SyntheticCIFAR`, CPU), proves: the trainer threads exact
  ramped values into the training attack only (never selection) at epoch
  0 and mid-ramp; the realized perturbation (`AttackResult.max_abs_delta`)
  stays inside the ramped ball, not the full target; an epoch at/past
  `warmup_epochs` is indistinguishable from an unwarmed config; the
  realized budget is reported in the epoch metrics, always, even when
  warmup isn't configured; and the mutual-exclusivity guard against the
  mixed-selected-attack-budget mechanism actually fires.
- `tests/unit/test_registry.py` — new architecture builds via timm (not
  torchvision), round-trips a forward pass, `pretrained=True` requests the
  right timm checkpoint tag (mocked, no download, matching every sibling
  test's discipline), replaces the head for a non-1000-class config
  (mocked), and `pretrained=False` default still passes `pretrained=False`
  to timm.
- `tests/unit/test_config.py` — the two mobilenetv4 configs are field-
  identical to each other and to plan 0100's mobilenetv3 config except the
  intended deltas.
- `scripts/verify.py --changed` green across T0-T3 after every file group,
  including a real regression it caught (`test_checkpoint_resume.py`'s
  exact-key-set assertion, updated for the two new always-on metrics).

## Staged verification

**Stage A — zero GPU-training cost. PASSED this session.** A throwaway
script (`scripts/verify.py`-independent, not itself a scientific record)
built the pretrained checkpoint through this project's own
`build_student`/`ImageNetEvalTransform` pipeline and ran it against the
full, real, official 50,000-image ImageNet validation split (shared
briefly with plan 0100's own running GPU jobs, forward-pass only, ~40
seconds): **73.41% top-1**, matching the paper's published 73.8% closely
(the small residual gap is plausibly interpolation/resize-method detail,
not a pipeline bug). Confirms the registry wiring and this project's own
data/normalization pipeline are both correct for this checkpoint, before
any training epoch is spent.

**Stage B — cheap short canary, real GPU but bounded. Not yet launched.
Redesigned this session per scientific review finding P1-1 (blocking).**
The original single-arm design (mobilenetv4 + ε-warmup, readout against
plan 0100's recorded epoch 0-3 numbers) was confounded: at epoch 0 of a
10-epoch warmup, the ramped ε is a small fraction of the target (this
project's own convention, `(epoch+1)/warmup_epochs`; see the schedule
module's docstring), so a higher early clean accuracy there is *guaranteed
by the weak attack itself*, independent of the architecture -- the canary
could not tell "MobileNetV4 lifts the floor" apart from "we barely
attacked it," and two variables (architecture, ε schedule) moved at once.

**Corrected design: two arms**, both short (3-5 epoch) PGD-AT-only (no
`adr`) canaries:
- **Arm A** (`configs/scientific/imagenet_mobilenetv4_pgd_at_no_warmup.yaml`):
  MobileNetV4-Conv-Small, no ε-warmup -- directly comparable to plan
  0100's recorded MobileNetV3-Small epoch 0-3 numbers (baseline ~42-46%,
  `adr` ~41-43%), isolating the architecture's own effect.
- **Arm B** (`configs/scientific/imagenet_mobilenetv4_pgd_at.yaml`):
  MobileNetV4-Conv-Small with ε-warmup -- compared against Arm A (not
  against plan 0100) to isolate the warmup's own incremental effect on top
  of the architecture change.

Both configs verified field-identical to each other (except
`training.epsilon_warmup_epochs` and `tracking.group`) and to plan 0100's
`imagenet_mobilenetv3_pgd_at.yaml` (except `student.architecture`,
`protocol.id`, and those same two fields) --
`tests/unit/test_config.py::test_mobilenetv4_pgd_at_configs_are_field_identical_except_the_intended_deltas`.
**Blocked on**: (a) the human confirming Option 1 vs. Option 2 above (Arm
B only), (b) scientific review of the ε-warmup/registry/schema/trainer
changes (requested and returned this session with one blocking finding,
now fixed -- see Progress log).

**Stage C — decision point.** Only after Stage B: decide, with the human,
whether to commit a full 50-epoch PGD-AT run (and how many seeds) on the
new architecture, based on Stage B's signal and plan 0100 stage 1's
by-then-complete result (does R18 recover better than MobileNetV3-Small
after its own epoch-25 decay). Whether/when to re-attempt `adr` on top of
a strong clean floor is also decided here, not before.

## Scientific review

This touches `src/ard/attacks/` request wiring (reusing, not adding, the
override fields), `src/ard/config/schema.py`, `src/ard/engine/trainer.py`,
`src/ard/models/registry.py`, and `src/ard/protocols/__init__.py`. Per
CLAUDE.md standing rule, routed through scientific-reviewer before
freezing a source SHA for Stage B's GPU canary. Ran this session; one
blocking finding (P1-1, the Stage B design confound above) and ten
non-blocking findings, all recorded and addressed or explicitly deferred
below (Progress log has the full account).

## Known, deliberately-not-fixed gaps (documented, not silent)

- **Pretrained checkpoint is an unhashed campaign input** (review P2-5).
  `mobilenetv4_conv_small.e1200_r224_in1k` resolves from the HF hub at run
  time with no revision pin, and nothing records its resolved SHA-256 in
  the run bundle -- the results-records rule wants hash-bound lineage for
  every consumed input, and here the pretrained weights *are* the claim
  (Stage A's 73.41%). Pre-exists for `resnet18_imagenet`/
  `mobilenet_v3_small_imagenet` too; not fixed here (real engineering, out
  of this plan's scope), but flagged before trusting Stage B's numbers on
  a host with a cold cache (e.g. Ferret) -- record the cache file's
  SHA-256 and pinned `timm` version alongside Stage B's result.
- **Preprocessing interpolation is not this checkpoint's own default**
  (review P2-6). timm declares `interpolation='bicubic'` for this exact
  checkpoint tag; `ImageNetEvalTransform` resizes with torchvision's
  default bilinear. Very likely the source of Stage A's 73.41% vs. 73.8%
  residual, and it applies to training augmentation too. Not a bug, but an
  explicit, recorded choice for a plan whose entire subject is the clean-
  accuracy floor -- changing it later would also move plan 0100's own
  numbers, so any future change here must not be silent.
- **Config-digest churn** (review P2-9): `epsilon_warmup_epochs` entering
  every config's hash means checkpoints written before commit `d81064d`
  cannot resume at a SHA including it (official evaluation of old
  checkpoints is unaffected -- confirmed by the reviewer, it hashes the
  saved YAML directly). Same consequence plan 0100's own `pretrained`
  field had. Operational note: resume plan 0100's own stage-1/stage-2 jobs
  only from their own already-pinned SHA, not a SHA that includes this
  plan's changes.
- **The Debenedetti mechanism is reported as "inspired by," not
  replicated** (review P2-11): `dedeswim/vits-robustness-torch` is not
  pinned in `.external/` the way every other adopted baseline is, so this
  plan's epoch-indexed reading of the paper's schedule could not be
  independently re-verified against the actual upstream code in-repo.
  Combined with the already-documented step-size and PGD-step deviations,
  any report from this plan must say "an epsilon warmup inspired by
  Debenedetti et al.," never "we replicated arXiv:2209.07399."
- **Not guarded, no current impact**: combining `epsilon_warmup_epochs`
  with `adr`/`rslad_student|joint` (per-sample EMA state and the
  rectified target both assume a fixed threat) is not prevented by any
  validator -- this plan doesn't use that combination, but a future plan
  that does should add the guard first.

## Also fixed this session, unrelated to this plan's own content

The prior session's `src/ard/tracking/adapter.py` W&B-collision preflight
fix (commit `6b560e7`, for plan 0100 stage 2) came back from
scientific-reviewer with two blocking findings (P1-1: false-positive risk
at the `ard.cli.evaluate` call site for externally-staged checkpoints;
P1-2: the resume-exemption regression test could not actually fail). Both
fixed, mechanically, in commit `1092432` — `evaluate.py`'s preflight call
now passes `check_remote_run_collision=False` and checks the evaluation's
own real run ID separately once its output dir exists; the vacuous test
replaced with an access-recording fake. This unblocks *plan 0100 stage 2's*
eventual source freeze; it is not part of this plan's own scope.

## Progress log

- 2026-09-13 (chat, autonomous session while the human was away): plan
  approved; implemented architecture registration, ε-warmup schedule +
  trainer wiring (Option 1), new config, and all tests above;
  `scripts/verify.py --changed` green; Stage A run and passed (73.41% top-1
  on the real val split); scientific review requested for the engine
  changes. Stage B intentionally not launched — needs the human's
  confirmation of the Option 1/Option 2 fork first, and the scientific
  review to land.
- 2026-09-13 (chat, same autonomous session): scientific review returned.
  **P1-1 (blocking)**: Stage B's single-arm design was confounded (epoch 0
  of a 10-epoch warmup attacks at a small fraction of target ε, so a
  higher clean accuracy there is guaranteed by attack weakness, not
  architecture). Fixed by splitting Stage B into two arms (no-warmup vs.
  warmup, new config `imagenet_mobilenetv4_pgd_at_no_warmup.yaml`) and
  adding a 3-way config drift-guard test
  (`tests/unit/test_config.py::test_mobilenetv4_pgd_at_configs_are_field_identical_except_the_intended_deltas`).
  Ten non-blocking findings, all addressed:
  - **P2-8** (schedule docstring wrong -- claimed to mirror
    `warmup_multistep_multiplier` but didn't, off by one at both ends):
    fixed the ramp formula itself to `(epoch+1)/warmup_epochs`, matching
    that convention exactly (never exactly 0 at epoch 0, reaches target at
    the last warmup epoch) -- this doubles as part of the P1-1 fix, since
    it removes the fully-non-adversarial epoch-0 case entirely.
  - **P2-1**: added `train_attack_epsilon`/`train_attack_step_size` to the
    per-epoch metrics (`Trainer.train_epoch`/`fit`), computed once per
    epoch, always populated. Updated the one existing test with an exact-
    key-set assertion (`tests/integration/test_checkpoint_resume.py`).
  - **P2-2**: added an exact-value test at epoch 0, an exact-value test
    mid-ramp, and (the highest-priority missing case) a test proving an
    epoch at/past `warmup_epochs` is indistinguishable from an unwarmed
    config, via the realized `AttackResult.max_abs_delta`, not just the
    request tensors (which correctly become `None` post-ramp, falling
    through to the config's own resolved budget).
  - **P2-3**: added a regression test constructing a `Trainer` with both
    `epsilon_warmup_epochs` and `selected_attack_epsilon` set, confirming
    the existing (previously untested) guard fires.
  - **P2-4**: the one real-network-download unit test
    (`test_registry.py`) now mocks `timm.create_model` like every sibling
    test in that file; added the missing `pretrained=False` default-
    preserving test for this architecture too.
  - **P2-5, P2-6, P2-9, P2-11**: no code change (real engineering or
    genuinely out of scope); recorded explicitly under "Known,
    deliberately-not-fixed gaps" above rather than left silent.
  - **P2-7**: superseded by the P1-1 fix's 3-way drift-guard test.
  - **P2-10(a)**: recorded as a caveat above (no guard needed for this
    plan's own scope). **P2-10(b)** (an `epsilon: "0"` config would hit
    `ZeroDivisionError` instead of a clear guard message): fixed with an
    explicit `baseline_epsilon <= 0` check raising `ValueError`, matching
    this method's existing exception style.
  `scripts/verify.py --changed` green again after all fixes (T0-T3,
  including the newly-caught `test_checkpoint_resume.py` key-set
  regression). Stage B still not launched -- still needs the human's
  Option 1/Option 2 confirmation; the engine-change review itself is now
  resolved.
