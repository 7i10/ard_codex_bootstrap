# ERT / RSLAD Runtime Performance Audit v2

Status: complete for the safe Hamster benchmark scope. Ferret's controlled
benchmark is intentionally deferred because GPU2 was running an unrelated
production continuation. No production process, environment, checkpoint, or
W&B run was changed.

## Executive decision

- The measured Hamster eager baseline is **679.1 images/s** under the current
  resolved config (`num_workers=8`, 32 measured batches), using the real
  `last.pt` checkpoint; a four-worker reference was 681.5 images/s.
- `num_workers=0` drops to **539.8 images/s** (-20.5% versus the current
  eight-worker config); input supply is a real bottleneck when workers are
  absent.
- `pin_memory` + `non_blocking` + persistent workers and eight workers do not
  improve the four-worker baseline (−0.51% and −0.15%). They are not adopted.
- All tested `torch.compile` modes are 21.2–22.8% slower in steady state and
  fail the predeclared one-step numerical parity gate. They are rejected for
  this trajectory-sensitive project.
- No runtime production code was changed. The safe speedup adopted today is
  **0%**; the correct operational action is host-aware scheduling.
- Ferret's slowdown remains **H9 mixed/unresolved** until a clean, controlled
  Ferret benchmark can be run after the protected GPU2 job exits.

## Reconciliation and protected production

Audit source at start: `5c87e705925ef1bb2e6600ee01e33034d0cd63da`.
The active timing campaign used a separate pinned checkout at
`8083f9c5df9b46a3a02399fbf293ceee6db85083`; it was not modified.

The initial one-time snapshot found Hamster idle and Ferret GPU2 occupied by
the I75 seed-1 continuation. Ferret GPU0/GPU1 were not used for host-wide
benchmarks because CPU, RAM, filesystem, PCIe, and power resources are shared.
No repeated completion polling was performed.

## Host and software audit

| Host | GPU layout | CPU / NUMA | stack | filesystem at snapshot |
|---|---|---|---|---|
| Hamster | 2 × RTX 4090; both PCIe Gen3 x16, `NODE` | Xeon Gold 6230R, 52 logical CPUs, 1 socket / 1 NUMA | Python `/home/shunsukenaito/.conda/envs/adv/bin/python`, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, cuDNN 91900, driver 595.84 | root NVMe ext4, about 23 GiB free, reported 100% |
| Ferret | 3 × RTX 4090; GPU0/1 `NODE`, GPU2 `SYS`, GPU2 Gen3 x16 | same CPU model, 104 logical CPUs, 2 sockets / 2 NUMA | same observed stack | about 878 GiB free |

Both hosts therefore have a matched software stack and the same GPU model.
The topology differs: Ferret GPU2 is remote (`SYS`) from its CPU locality,
which is a plausible contributor, not a proved sole cause. Power, clocks,
thermal state, and PCIe width were not abnormal in the initial snapshot. The
Ferret controlled comparison is intentionally missing, so a compute-vs-CPU
causal attribution is not claimed.

## Historical throughput and scheduling

Historical production rows (some manifests omit host, so these are not a clean
cross-host causal comparison) were approximately:

| workload | median images/s |
|---|---:|
| Hamster BASE seed 1 / seed 2 | 635.24 / 636.76 |
| Hamster CROPSHIFT seed 1 / seed 2 | 623.25 / 622.45 |
| Fresh I50 rows (host known from campaign context) | 632.67 / 617.82 |
| Ferret I75/I125 rows (host known from campaign context) | 364.64 / 364.17 / 373.73 |

Using the conservative historical rates and 45,000 training samples per epoch,
200 epochs are about 4.0 h on a 620 img/s Hamster GPU and 6.8–6.9 h on a
360–370 img/s Ferret GPU. The bounded harness rate (681.5 img/s) corresponds to
about 3.7 h and should be treated as an upper-bound microbenchmark, not a
production ETA.

Recommended policy:

1. Assign the longest sequential jobs to Hamster first.
2. Use Ferret for independent shorter jobs, endpoint evaluations, and
   materialization after GPU2 is clean.
3. Greedily schedule by measured host-specific duration (longest-processing-
   time first); do not change batch size, precision, attack, objective, RNG,
   or sample order to equalize hosts.

At the current-config measured rate, two Hamster GPUs provide about 1,358 img/s aggregate
(the four-worker comparison is 1,363 img/s),
while three historical Ferret GPUs provide about 1,080–1,110 img/s. Ferret
still matters for parallel makespan; it should not simply be left unused.

## Bounded workload profile

The harness uses a real resolved timing config, CIFAR-10 train view, Chen
teacher, current RSLAD objective, and KL-PGD10. It measures the core model /
attack path; it intentionally omits the production diagnostics panel and
per-sample history bookkeeping. Therefore the numbers are a clean lower-level
profile, while the historical production rows remain the better estimate of a
full run. It writes only small local JSON results and never writes
checkpoints, run bundles, or W&B artifacts.

Hamster GPU1, eager, four workers, no pinning (the segment reference; the
current resolved config uses eight workers):

| segment | median seconds | share of measured total |
|---|---:|---:|
| data wait | 0.001868 | 0.99% |
| H2D | 0.000570 | 0.30% |
| Teacher clean | 0.037117 | 19.76% |
| PGD10 | 0.082546 | 43.95% |
| Teacher adversarial | 0.037133 | 19.77% |
| outer forward/objective/backward/step | 0.028545 | 15.23% |
| total batch | 0.187814 | 100% |

PGD10 is the largest measured segment. With workers disabled, data wait rises
to 0.049965 s and total batch time to 0.237113 s; the other segments remain
nearly unchanged. This explains the roughly 20.5% end-to-end loss without attributing
the Ferret gap to DataLoader alone.

## Low-risk screen

| candidate | images/s | delta vs eager | decision |
|---|---:|---:|---|
| eager, workers=8 (current config) | 679.07 | 0.00% | reference |
| eager, workers=4 (comparison) | 681.52 | +0.36% | no material change |
| pin + non-blocking + persistent, workers=4 | 682.93 | +0.57% | reject; below retention threshold |
| pin + non-blocking + persistent, workers=8 | 680.31 | +0.18% | no material change |
| eager, workers=0 | 540.83 | −20.36% | reject |

The production source currently does not expose pinning/non-blocking/
persistent/prefetch settings. Because the measured settings do not improve
the real workload, no production option was added. The four-worker setting is
the useful observed operating point; its exact production adoption remains a
configuration/host decision, not a scientific-method change.

Per-batch CUDA scalar extraction is present in
`src/ard/engine/trainer.py:1063-1074`, and PGD computes
`max_abs_delta` through a CPU scalar at `src/ard/attacks/pgd.py:144-150`.
These are synchronization candidates. They were not removed speculatively:
their metric/cadence and checkpoint/resume implications need a separate
semantics-preserving candidate and parity test. Thus “sync removal speedup” is
**not measured / not claimed**, rather than silently estimated.

## Forward and logging audit

Teacher adversarial logits are cached and reused for observation, policy, and
diagnostic consumers. The clean Student forwards are not trivially redundant:
the pre-update train-mode logits, diagnostic eval-mode logits, and post-update
eval-mode metrics have different state/semantics. Removing any one would
change observations or metrics. No duplicate forward was removed.

The current trainer also performs per-batch scalar extraction for epoch totals.
This is a likely launch-synchronization cost, but no production change was
made because a batch/epoch accumulation rewrite would need explicit metric,
resume, and DDP parity evidence.

## `torch.compile` screen

Student-only compile was tested on Hamster. Steady-state results:

| mode | compile seconds | images/s | delta vs eager | result |
|---|---:|---:|---:|---|
| default | 0.734 | 527.30 | −22.35% | reject |
| reduce-overhead | 0.734 | 532.14 | −21.64% | reject |
| max-autotune-no-cudagraphs | 0.740 | 524.02 | −22.83% | reject |
| max-autotune | 0.750 | 530.67 | −21.85% | reject |

The four-batch compile-log run recorded six graph-break lines and five
recompile lines. Breaks were caused by data-dependent pixel validation in
`ard.models.registry` and by changing `requires_grad` guards between PGD and
outer forwards. The log is retained by hash in the compile artifact.

The one-batch eager/compiled parity harness used strict predeclared tolerances
and the same real checkpoint. Default mode failed with max absolute differences
of approximately 0.0627 in the adversarial tensor, 0.0767 in adversarial
logits, 0.000575 in clean logits, 0.000117 in loss, 0.00119 in parameters,
and 0.0119 in gradients. The
reduce-overhead check showed the same class of divergence. These differences
are material for a trajectory-sensitive study; compile is not adopted and no
break-even calculation is applicable because the steady path is slower.

Teacher compile, PGD-region compile, manual CUDA graphs, channels-last,
cuDNN benchmark mode, and a sync-removal implementation were not run. They
would either be conditional follow-ups requiring a clean host and a new
parity harness, or require changing a hot-path contract that has not yet been
shown unnecessary. The active Ferret job and the absence of a measured
Hamster bottleneck made additional speculative screens lower-value.

## Decisions and limitations

This audit classifies the Hamster/Ferret gap as **H9 mixed/unresolved**. The
software stack is matched; the observed Ferret topology/NUMA arrangement and
historical host-local throughput make hardware/CPU-launch interaction plausible,
but only an idle-host Ferret benchmark can distinguish H1/H3/H5/H8. The
protected production run is not a valid benchmark baseline.

No official test, AutoAttack, training continuation, W&B upload, model upload,
or run-bundle upload was performed. The benchmark scripts are analysis-only;
they do not enter the production trainer. Existing scientific runs remain
comparable because no scientific source or runtime path was altered.

Machine records:

- [`host audit`](experiments/ert_rslad_runtime_host_audit_v2.json)
- [`baseline profile`](experiments/ert_rslad_runtime_baseline_profile_v2.json)
- [`low-risk screen`](experiments/ert_rslad_runtime_lowrisk_screen_v2.json)
- [`compile screen`](experiments/ert_rslad_runtime_compile_screen_v2.json)
- [`parity`](experiments/ert_rslad_runtime_parity_v2.json)
- [`host scheduling`](experiments/ert_rslad_runtime_host_scheduling_v2.json)
- [`final bundle decision`](experiments/ert_rslad_runtime_final_bundle_v2.json)

The benchmark JSON hashes and raw `/tmp` output hashes are recorded in those
machine records. No candidate is promoted automatically. A future Ferret
audit should reuse the same harness and contract once GPU2 has naturally
finished; until then no active waiting is required.
