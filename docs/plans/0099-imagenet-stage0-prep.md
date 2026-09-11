# 0099 — ImageNet Stage 0 prep: data, AMP, one ImageNet-scale architecture

## Status

- Owner: human (scientific decisions), Claude Code (execution)
- Depends on: plan 0097 reaching M3 (done — `836b3ae`); nothing in this plan
  depends on plan 0098's result. Runs in parallel with it, per the user's
  explicit request in this session to stop treating unrelated ImageNet
  engineering prep as gated behind the CIFAR-scale method validation.
- This is infrastructure (BACKLOG.md A-tier): no new arm, seed, epoch
  horizon, official test, or promotion is decided here. It does not by
  itself authorize an ImageNet training campaign — that is a separate
  decision packet once this plan's engineering exists to cost against.

## Goal

Close the three gaps `docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md` §3 and this
session's own survey confirmed still block any ImageNet-scale run:

1. No ImageNet dataset support anywhere under `src/ard/` (`DatasetConfig.name`
   is a closed `Literal["synthetic_cifar", "cifar10", "cifar100",
   "tiny_imagenet"]`).
2. AMP is implemented but dead: `torch.autocast`/`GradScaler` plumbing exists
   in `Trainer`, but `src/ard/cli/train.py:919` hardcodes `scaler=None`
   (already recorded in `docs/decisions/0011-post-cifar-adr-next-step.md`).
3. No ImageNet-scale architecture is registered (`src/ard/models/registry.py`
   has only 32px-patched CIFAR variants).

Plus one thing this plan does **not** get to skip just because it's
"only prep": a real throughput benchmark. The proposal doc's own existing
ImageNet cost figures are flagged unverified and "likely wrong by 4-7x" —
this plan's job is to replace that guess with a measured number, not to
carry it forward into a future campaign's costing.

## What this plan explicitly does NOT decide

Per CLAUDE.md rule 7 and rule 6, the following stay open, human decisions for
a later plan/decision packet, and this plan's own tests and code must not
bake in an unstated choice for any of them:

- **Epsilon / threat model for an actual ImageNet training run.** The only
  figure in this repo's docs today is the RobustBench/Salman et al. 2020
  leaderboard convention (linf, eps=4/255), cited as literature context, not
  asserted as this project's choice. This plan's dataset/attack plumbing must
  stay parametric (reuse the existing `AttackConfig`, which already takes
  epsilon as a string quantity) rather than hardcoding a value anywhere.
- **Which architecture(s) get an actual training campaign.** This plan
  registers ImageNet-shaped architectures as *available* (so a future config
  can select one); it does not choose which one plan/decision Stage 1 trains.
- **Epoch horizon, augmentation policy, pretrained-init choice for Stage 1.**

## Design question this plan must resolve before writing code: dataset content identity at ImageNet scale

`TinyImageNetDataset` (`src/ard/data/datasets.py:388-450`) hashes every image
file's full bytes into a single SHA-256 on **every dataset construction**
(i.e. every job launch), and `DatasetConfig.content_sha256` is required at
repro/pilot/production tiers for it. This is already a real cost at
Tiny-ImageNet's ~100k 64px images; at full ImageNet-1k's ~1.28M images across
146GB, doing this on every process start is not viable (survey put it at
minutes-to-tens-of-minutes per launch, unmeasured but clearly disproportionate
next to a ~28min CIFAR arm this project already anchors its cost estimates
on).

**Recommendation, to be confirmed during implementation, not silently
assumed:** compute the full byte-level SHA-256 once, offline, and pin it the
same way a teacher checkpoint's `checkpoint_sha256` is pinned (a recorded,
reviewed constant) rather than an adapter that reproduces it from scratch on
every launch. At construction time, the adapter instead validates a cheap
**manifest** (per-class directory listing: relative path + file size; a few
seconds, not minutes, over 1.28M entries) against a manifest digest recorded
alongside the pinned full-content digest, and only recomputes the expensive
full byte hash on demand (a documented one-off maintenance command,
analogous to a checkpoint's own provenance check) — never implicitly on the
hot launch path. This is a provenance/reproducibility bookkeeping mechanism,
not one of CLAUDE.md rule 6's protected scientific knobs, so this is an
engineering call, but it changes a real project convention
(`content_sha256` today always means "recomputed and matched live") and must
be written down here rather than decided silently inside a diff.

## Implementation checklist

1. **`src/ard/data/datasets.py`**: `ImageNetDataset`, following
   `TinyImageNetDataset`'s shape (standard `train/<wnid>/*.JPEG` and
   `val/<wnid>/*.JPEG` layout — already confirmed present at
   `/home/shunsukenaito/workspace-local/datasets/imagenet/` with a
   `class_names.json`) but using the manifest-based content identity above
   instead of a full re-hash.
2. **`src/ard/config/schema.py`**: add `"imagenet"` to `DatasetConfig.name`'s
   `Literal`; add `image_size` handling for 224 (the field already exists,
   default 32 — no schema shape change, just a new valid value range) and a
   `NormalizationConfig` profile (`imagenet_standard`, standard
   ILSVRC mean/std) alongside the existing named profiles. Follow the
   existing `tiny_imagenet` pattern for requiring `root` and (per the
   resolved design question above) `content_sha256` at repro/pilot/production
   tiers.
3. **`src/ard/models/registry.py`**: register at least one ImageNet-scale
   architecture (a plain `torchvision.models.resnet50` at native resolution,
   unpatched — the existing CIFAR entries patch conv1/maxpool specifically
   *because* CIFAR is 32px; ImageNet needs the untouched torchvision
   definition) and one mobile-scale candidate (full-resolution
   `mobilenet_v3_small` or `v2`, matching this project's mobile-robustness
   framing) — exact choice of mobile architecture left to whoever picks
   Stage 1's arms, so register more than one if cheap to do.
4. **`src/ard/cli/train.py`**: make AMP actually reachable — replace the
   hardcoded `scaler=None` with a config-driven `GradScaler` construction
   (new `TrainingConfig` field, e.g. `amp: bool = False`, default false so
   every existing CIFAR config's behavior is byte-identical unless it opts
   in). This is the one change here that touches an existing, running
   protocol's config surface — needs the same schedule-only regression
   discipline plan 0098 used (a test asserting `amp: false`/absent produces
   identical trainer construction to today).
5. **Throughput benchmark**: once 1-4 exist, one real measurement — a short
   plain-AT ResNet-50 run on Hamster/Ferret at the real ImageNet dataloader
   and AMP on, images/sec logged the same way `train_images_per_second`
   already is for CIFAR. Replace the proposal doc's flagged-unverified
   figures with this number and correct any cost estimate downstream of it.

## Tests

- Schema: `imagenet` dataset name validates, requires `root`, and (matching
  the design question's resolution) whatever content-identity field it
  actually ends up requiring at repro/pilot/production tiers.
- `ImageNetDataset`: a small synthetic on-disk fixture (a handful of classes,
  a handful of images each — not the real 146GB set) exercising both splits,
  matching `TinyImageNetDataset`'s own test style.
- AMP: `amp: false`/absent is bit-identical to today (schedule-only-style
  regression, same discipline as plan 0098's cosine-default test); `amp:
  true` actually constructs a `GradScaler` and is exercised by a CPU-fallback
  or fixture-scale test for the wiring (real mixed-precision numerics need a
  GPU and are out of scope for the unit suite, same convention as this
  project's other CUDA-only paths).
- Registry: new architectures build at their native (224px) input size and
  round-trip a forward pass without shape errors.

## Verification, end to end

1. `scripts/verify.py --changed` after each of the four implementation
   groups (data, schema, registry, AMP/train.py), not one big change.
2. Fixture-scale CPU test for the new dataset adapter and the AMP config
   plumbing, before anything touches the real 146GB dataset or a GPU.
3. One real, short GPU run for the throughput benchmark (item 5) — hand-run
   from a pinned worktree, run-bundle contract, per CLAUDE.md rule 3's
   sanctioned path, since this is prep/measurement, not itself a scientific
   campaign.
4. This plan's own completion does not launch a Stage 1 campaign; that needs
   its own plan and a decision packet costing it against the throughput this
   plan measures.
