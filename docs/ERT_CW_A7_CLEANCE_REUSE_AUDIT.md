# ERT A7 CleanCE reuse audit

Status: PASS for historical F0/F1/F2 reuse; F3 requires fresh training.

The historical A7 resolved treatment is `teacher_floor` margin-only with `extra_clean_ce: null`.
It is therefore F2, not the full F3 (CleanCE + margin) arm described in the proposal.

| Factorial arm | Historical source | Decision |
|---|---|---|
| F0 | A0 (baseline) | reuse |
| F1 | A1 (CleanCE 0.15) | reuse |
| F2 | A7 (teacher-floor margin only) | reuse |
| F3 | no historical equivalent | fresh L2/L4 required |

Calibration SHA256: `a625b43ec12277bbf698270193f27e0e1f62e0a2a9f9a6a49e7fc0702593b2b5`.
Relevant training/runtime source components match historical `bb59b512185af7bb70633c3266efd95bb24a563f`.
All six historical manifests bind the exact epoch-79 parent, fixed endpoint horizons 84/89/94, and the frozen calibration.

No production launch is authorized by this audit alone; F3 must pass a clean-tree canary before GPU launch.
