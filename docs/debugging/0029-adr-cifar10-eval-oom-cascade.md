# 0029 — ADR CIFAR-10 campaign: an evaluation OOM cascade, allocator fragmentation, not a leak

## What happened

While clearing the evaluation backlog for plan 0097 (ADR CIFAR-10 replication),
9 concurrent `ard.cli.evaluate` processes across Hamster's 2 GPUs (5 on one,
4 on the other) began crashing with `torch.OutOfMemoryError`, each after
fully completing AutoAttack for `best.pt` (a real, correct robust-accuracy
number computed and printed) but before writing `evaluation-results.json`
(written once, at the very end) — so every crash lost a real number, not a
placeholder. A lane-runner script's failure-handling bug then made this much
worse: on any non-zero exit it silently launched the *next* queued job on the
same still-crowded GPU, so one crash could fire a fresh process into a GPU
that was already at the edge, producing a second crash within minutes, in a
loop. Roughly 20 job attempts crashed or had to be killed before any new
canonical result was saved.

This document went through one incomplete revision before this one. The
first version concluded the outlier was purely architectural
(`mobilenetv2_*`'s ~13GB baseline) and that plain concurrency reduction was
an overcorrection. Both halves needed revision once more data came in: a
dedicated code investigation found the actual mechanism, and it explains
observations the first version couldn't (two ResNet18-family jobs — one
plain model weights, one EMA — independently grew to ~14.7-14.8GB after
1.5-2h of runtime, nothing to do with `mobilenetv2` at all).

## The real mechanism: PyTorch allocator fragmentation, confirmed by code, not a leak

Read end to end: `src/ard/evaluation/autoattack.py`, the installed
`autoattack` package's `run_standard_evaluation`, `src/ard/cli/evaluate.py`'s
checkpoint loop, and `src/ard/evaluation/saved_checkpoint.py`.

- **No unbounded Python-level accumulation exists anywhere in this path.**
  `saved_checkpoint.py:115`'s `rows: list[dict]` grows to one small dict per
  sample (10,000 total) for the PGD/clean pass — bounded, CPU-side, not a GPU
  growth mechanism. AutoAttack's own `run_standard_evaluation` pre-allocates
  fixed-size tensors once (`x_adv`, `y_adv`, `robust_flags`) and only writes
  into them per stage; later stages correctly iterate over the shrinking set
  of still-unbroken examples (`num_robust = torch.sum(robust_flags).item()`),
  matching the campaign logs (e.g. "square - 38/38 - 0 out of 26 successfully
  perturbed"). The library is not the source of growth.
- **What's actually missing: any `torch.cuda.empty_cache()` call.**
  `grep -rn "empty_cache" src/ard/evaluation/ src/ard/cli/evaluate.py` returns
  nothing. PyTorch's caching allocator never returns freed blocks to the
  driver on its own. AutoAttack's four sub-attacks (APGD-CE, APGD-T over 9
  target classes, FAB-T, Square) each churn through many differently-shaped
  tensors as the batch shrinks stage to stage — without an `empty_cache()`
  call anywhere, the allocator accumulates a growing set of cached-but-
  differently-shaped blocks it can't reuse for the next allocation.
  `nvidia-smi`'s reported "memory.used" climbs from this even though nothing
  is truly leaked — this is exactly what the OOM traceback's own suggestion
  (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) targets.
- **A compounding factor: `evaluate.py` runs both checkpoints (`best.pt` and
  `last.pt`) in one process.** Fragmentation from evaluating the first
  checkpoint's AutoAttack pass is never cleared before the second one starts
  in the same process — only a full process exit (returning the CUDA context
  to the OS) resets it. This doubles exposure per process and explains why
  the growth pattern showed up regardless of architecture or weights type
  (plain model, EMA) — it is a property of the evaluation harness's process
  structure, not of any one arm.
- **`mobilenetv2_*`'s separate, real property**: an unusually high *baseline*
  (~13GB from very early in the run, not from accumulation), plausibly from
  this project forcing the MobileNetV2 stem's first conv stride to `(1,1)`
  for CIFAR's 32px input (`mobilenet_v2_cifar`, `src/ard/models/registry.py`)
  — this keeps feature maps far larger through more of the network than
  MobileNetV2 has at 224px. This is a real, separate effect from the
  fragmentation growth above, and both apply simultaneously to a
  `mobilenetv2_*` job (high baseline *and* growth on top of it).

## What was wrong about the first fix, and the second

**First response (during the incident): cut concurrency to 2-3 per GPU.**
Wrong framing — treated "too many processes were present" as the cause.

**Second response: raise it back to 4-5, on the theory that only
`mobilenetv2_*` was the outlier and ResNet18-scale jobs stay flat at
~1.7-2GB.** Also wrong, and caught before being applied at scale: two
independent ResNet18-family jobs were measured at ~14.7-14.8GB after 1.5-2h,
matching `mobilenetv2`'s footprint almost exactly. **Any** sufficiently
long-running AutoAttack evaluation can reach this plateau, regardless of
architecture — the earlier "ResNet18 jobs are ~2GB" reading was true only
early in a job's life, before fragmentation accumulated.

## The corrected policy, going forward

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on every
  `ard.cli.evaluate` invocation — mitigates the fragmentation, does not
  eliminate the underlying growth.
- Treat **every** AutoAttack evaluation job as one that will grow toward
  roughly 13-15GB given 1.5-2 hours of runtime, not just `mobilenetv2_*`
  ones. Never assume a job stays at its launch-time footprint.
- A lane/queue runner **stops and reports on any non-zero exit** — never
  silently launches the next job onto a GPU whose post-failure state is
  unknown.
- Concurrency should be set by *live, current* `nvidia-smi` memory numbers
  and by how long jobs already on a GPU have been running (a job at 10
  minutes is not the same risk as one at 90 minutes), not by a flat
  process-count rule, and not by an architecture-based assumption about
  which jobs are "light."
- **Real fix candidate for later, not applied here**: one or more
  `torch.cuda.empty_cache()` calls in `src/ard/cli/evaluate.py` (between
  the two checkpoints' AutoAttack passes at minimum, possibly between
  AutoAttack's own internal stages if the library exposes a hook) would
  directly address the fragmentation this document identifies. This has
  zero effect on any computed number — cache eviction does not change
  tensor values, RNG state, or attack behavior, only which memory blocks
  are held in reserve — so it is a safe candidate, but it is still a change
  to `src/ard/`, made during an active evaluation phase, and belongs in a
  reviewed batch of fixes per this project's launch discipline, not
  patched in unilaterally mid-campaign.

## Lesson

Two corrections in a row on the same incident is itself the lesson: a
plausible-looking mechanical story ("one arm has a fixed high memory
footprint") can survive a first check and still be incomplete. The fix was
to go read the actual code and get real time-series memory measurements
rather than stopping at the first explanation that fit the initially
available data.
