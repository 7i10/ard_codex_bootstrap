# Plan 0106: resume across host kernel and NVIDIA driver updates

## Context

On 2026-10-12 Ferret was rebooted to clear an NVIDIA driver/library version mismatch (kernel module
595.84, userspace 595.99 after an unattended upgrade). The reboot came up on the kernel that the same
upgrade had installed (7.0.0-28 → 7.0.0-38). All three interrupted plan 0103 runs then refused
`--resume` with `resume manifest lineage drift: environment`. The only difference was the
`platform` string in `run-bundle/environment.json`. Python, PyTorch, CUDA, cuDNN, the GPU list and
the source SHA were unchanged. A one-off override flag (f9b37d8) was reverted (1e46a67): the
ordinary resume path also checks the full git record, so it could not help runs already in flight.
The runs were resumed by booting the old kernel once. The human approved this plan as the fix for
runs started from now on (discussion register, 2026-10-12).

While the host has an updated kernel installed, any reboot (planned, power loss, another user)
would orphan every interrupted run. One incident cost about 50 GPU-hours of restarts or a second
reboot.

## Design

- `collect_environment()` also records `nvidia_driver`: the loaded kernel-module version read from
  `/proc/driver/nvidia/version`, or `null` when no NVIDIA driver is loaded. The driver was not
  recorded before.
- `platform` and `nvidia_driver` are **informational**
  (`RESUME_INFORMATIONAL_ENVIRONMENT_FIELDS`). On resume, a change in them no longer refuses. The
  resume event in `manifest.json` records the change as
  `environment_drift: {field: {previous, current}}`. `environment.json` is rewritten with the
  current values, as before.
- Every other field stays a hard check, compared exactly with the field sets included: `python`,
  `machine` (CPU architecture), `libc` (C library and version), `torch`, `cuda`, `cudnn`, `gpu`.
  `machine` and `libc` are new hard fields: the `platform` string also carries them, so they are split
  out before `platform` becomes informational. An extra or missing hard field refuses. The old dict
  `!=` did the same.
- The git, run-name, config-hash, checkpoint and W&B checks are unchanged. A run still resumes only
  from its own source SHA.
- The terminal-resume path uses the same comparison. A terminal resume writes nothing, so it records
  no drift.
- **Driver drift and CUDA Graphs (rule, not code).** Deterministic CUDA-graph runs leave
  `cuda_graph` out of their evaluation identity because the graph-vs-eager bitwise parity test
  passed. The repo says to rerun that test after a driver upgrade (`src/ard/config/schema.py`,
  `tests/integration/test_cuda_graph_training_step.py`). So when a run's resume event records
  `environment_drift.nvidia_driver` and the run used `training.cuda_graph`, the parity test is rerun
  on the new driver before the run is pooled with eager runs. If it fails, the run is flagged in its
  record. This applies now to `plan0103-phase2-mnv4s-awp-s0-v1`: deterministic, CUDA Graphs,
  driver 595.84 before its 2026-10-12 resume and 595.99 after.

Runs started before this change carry the old code. They still need the original kernel to resume.

## Verification

- Unit tests (`tests/unit/test_tracking.py`):
  - A kernel+driver change on an interrupted run is recorded and not refused. This goes through
    `coordinated_create_tracker`, the constructor training uses.
  - A resume with no change records no drift.
  - Changes to torch, cuda, the GPU list, cudnn (together with a kernel change), and an extra field
    still refuse.
  - A missing hard field refuses. An informational field absent from an older snapshot is recorded.
  - The terminal path accepts a kernel change but refuses a torch change.
  - A glibc or architecture change still refuses, even together with a kernel change.
  - The driver-version parser reads both the open-module format (Hamster) and the proprietary
    format (Ferret). A missing file gives `null`.
- End to end on Hamster (CPU, synthetic smoke config, 3 epochs):
  - Procedure: SIGKILL the training right after its first epoch-boundary `last.pt`. Rewrite
    `platform`/`nvidia_driver` in `environment.json` to simulate the reboot. Run
    `ard.cli.train --resume`.
  - With this change, the run completes and the resume event records both fields.
  - On the parent commit, the same procedure fails with `lineage drift: environment`.

## Rollout

- Ferret's queue items are re-pinned to the commit that carries this change. Their configs are
  byte-identical at that commit.
  - The DeiT-Tiny, MobileNetV4-M and EfficientNet-B0 items move from d97bf6e, and the
    ConvNeXt-Atto baseline+EMA from 3e52df6. Their `src/` is identical apart from this change.
  - ConvNeXt-Atto ls01/ols move from 5dd329e. Every non-tracking change between 5dd329e and this
    commit is plan 0105 code that acts only when `training.cuda_graph` is on: capturable AdamW,
    graph eligibility, evaluation identity, allowlists. No ConvNeXt Phase 2 config sets it. The move
    therefore changes neither training behaviour nor evaluation identity.
- Hamster's queue is left unchanged. Its MobileNetV4-S comparison group stays on 5dd329e by the
  2026-10-11 decision.
  - Hamster has the same exposure: it runs 7.0.0-31 and has 7.0.0-34 and a libc6 update pending
    (`/var/run/reboot-required`).
  - Until the runs on pre-0106 code finish, reboot Hamster only into 7.0.0-31:
    `sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 7.0.0-31-generic"`, then reboot.
  - The pending libc6 update changes the hard `libc` field only if its version string changes. Ubuntu
    security updates keep it at 2.39.
- Ferret goes back to the current kernel with an ordinary reboot after the runs started under the
  old code finish (expected 2026-10-14).

## Progress

- 2026-10-12: implemented; unit tests and the end-to-end check pass. Review: no P1. Its P2s were
  folded in: `machine`/`libc` hard fields, a parser test, the CUDA-graph driver rule, Hamster's
  recovery path, and the accurate ConvNeXt rollout statement.
