# RobustBench Teacher Acquisition and Bounded Audit

## Status

- Owner: primary agent; one Terra owns tooling/tests; Luna updates docs only after API freeze; Sol performs final scientific review
- Base: `28774a7`
- Current milestone: complete
- Last updated: 2026-07-22

## Goal and scope

Acquire exactly two official CIFAR-10 `Linf` `8/255` RobustBench checkpoints once from the pinned checkout, fix their observed bytes by SHA-256, publish them into the existing no-download ARD teacher cache, and audit loader, normalization, logits, input gradients, and a one-step PGD contract. Do not run CIFAR training, AutoAttack, W&B, or claim the reported `56.94%` / `73.71%` AutoAttack values as locally reproduced.

Selected IDs:

- `chen2021_ltd_wrn34_10` / `Chen2021LTD_WRN34_10` / 46,160,474 parameters / identity preprocessing
- `bartoldson2024_adversarial_wrn94_16` / `Bartoldson2024Adversarial_WRN-94-16` / 365,915,610 parameters / model-embedded preprocessing

The optional Chen WRN-34-20 and Gowal WRN-28-10 teachers are deferred.

## Decisions and invariants

- Acquisition is an explicit operator CLI and the only path allowed to call pinned RobustBench `load_model`. ARD production runtime continues to reject missing checkpoints without downloading.
- The exact verified checkout is `.external/robustbench` at `78fcc9e48a07a861268f295a777b975f25155964`. Fresh-process module provenance must resolve under that checkout.
- The external `model_dir` is `/home/shunsukenaito/workspace-local/datasets/ard/teachers/robustbench`; pinned RobustBench therefore writes the exact lowercase path `cifar10/Linf/<model-id>.pt`.
- Download into a same-filesystem temporary directory. Load, parameter-count, eval, finite `(1,10)` logits, and SHA checks must pass before a no-clobber atomic publication. A failed or partial download never becomes a final file.
- Acquire and validate both external source files before publishing either project lock entry. Register one teacher at a time through the existing atomic `bootstrap_teacher.py --update-lock`; preserve a successful first registration if the second fails.
- Checkpoints stay ignored and are never committed, uploaded to W&B, or redistributed. `teachers.lock.yaml` records observed hashes only.
- Each real-model audit runs sequentially in a fresh process, batch size one, on one GPU. Chen normalization remains adapter identity; Bartoldson normalization remains model-embedded.

## Milestones

- [x] A0 — Acquisition/audit tooling
  - Add explicit allowlisted, pinned-provenance, network-opt-in acquisition CLI with staging and no-clobber publication.
  - Add a local-only audit CLI for strict ARD loading, pinned-loader logit parity, freeze/input-gradient, and one-step PGD bounds.
  - Add offline failure/impact tests; no network or large models in unit tests.
- [x] A1 — One-time external acquisition
  - Acquire Chen then Bartoldson sequentially.
  - Record final paths, byte sizes, observed SHA-256, parameter counts, and finite output checks.
- [x] A2 — Registry publication
  - Register both source files through `bootstrap_teacher.py --update-lock`.
  - Verify external source, ignored runtime cache, and `teachers.lock.yaml` hashes agree.
- [x] A3 — Bounded scientific audit
  - Fresh-process, local-only strict load for each teacher.
  - Fixed-input pinned-loader/ARD-logit parity, normalization owner, eval/freeze, finite input gradient.
  - One-step CE PGD with pixel clamp and `Linf <= 8/255 + 1e-7`; teacher parameter gradients stay `None`.
- [x] A4 — Review, gate, docs, commit
  - Sol scientific review with no open P0/P1/P2.
  - Run impact-selected non-scientific gate and demonstrate final cache hits.
  - Record observed evidence and deferred heavy work, then create one local commit without weights.

## Risks and rollback

- Google Drive can fail or leave partial bytes: only staging is writable until the model has loaded successfully; final paths are no-clobber.
- Bartoldson requires about 1.5 GB for parameters and duplicate external/runtime files. Current preflight shows 394 GB disk and about 220 GB available RAM; process teachers sequentially.
- Upstream checkpoint bytes have no published checksum in the pinned registry. The first verified download becomes the project-owned identity; a later different download must never silently advance the lock.
- Upstream loader uses model-internal preprocessing for Bartoldson. Any adapter-side second normalization is a scientific failure, not a tolerance issue.
- Registration is not a two-file transaction. Each `bootstrap_teacher` invocation is atomic; retry only the missing teacher after a failure.

## Acceptance and execution ledger

Completion requires two external files and two ignored runtime-cache files whose SHA-256 values exactly match verified lock entries, clean pinned external state, strict fresh-process loads, finite `(1,10)` outputs, logit parity, input-gradient and one-step PGD contracts, green selected tests, and no checkpoint in Git. Commands/results and review findings are appended here as work proceeds.

### 2026-07-22 — A0 evidence

- Focused offline suite: `PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest -q tests/unit/test_teacher_acquisition.py tests/unit/test_models_teacher.py tests/unit/test_external_management.py tests/unit/test_verify_gate.py` — `74 passed`.
- Final PGD-contract delta: `PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest -q tests/unit/test_teacher_acquisition.py` — `8 passed`.
- Ruff format/check, focused mypy, and `git diff --check` passed. The exact 74-test command was not repeated after its successful unchanged run; only the affected acquisition test file was rerun.
- Scientific review closed the staging rollback, production-PGD execution, lock-transition fixture, local-only download blocking, and deterministic GPU parity findings. A final P2 was fixed by pinning the no-random-start one-step perturbation to `0 < Linf <= 2/255 + 1e-7` and checking `AttackResult.max_abs_delta`; the delta review reported no remaining P0–P2.
- GPU preflight from the privileged execution shell found two NVIDIA GeForce RTX 4090 devices. W&B is intentionally unused for acquisition and local audit.

### 2026-07-22 — A1/A2 evidence

- Chen source: `/home/shunsukenaito/workspace-local/datasets/ard/teachers/robustbench/cifar10/Linf/Chen2021LTD_WRN34_10.pt`; 184,803,174 bytes; SHA-256 `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`; 46,160,474 parameters; finite `(1, 10)` logits on `cuda:0`.
- Bartoldson source: `/home/shunsukenaito/workspace-local/datasets/ard/teachers/robustbench/cifar10/Linf/Bartoldson2024Adversarial_WRN-94-16.pt`; 1,464,289,203 bytes; SHA-256 `56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b`; 365,915,610 parameters; finite `(1, 10)` logits on `cuda:0`.
- Both downloads used the verified pinned RobustBench loader and a same-filesystem staging directory. Each completed load validation before no-clobber publication; neither lock entry was updated until both source files existed.
- `bootstrap_teacher.py --update-lock` registered each source through the existing strict loader. `verify_teacher.py` passed for both. Independent `sha256sum` checks showed exact agreement among external source, ignored runtime cache, and `teachers.lock.yaml`.
- `git check-ignore -v` confirmed both runtime checkpoint files are excluded by `.gitignore`; no checkpoint appears in `git status`.

### 2026-07-22 — A3 evidence

- Chen fresh-process local-only audit on physical GPU 0: exact pinned-loader/ARD FP32 logit parity (`max_abs_diff=0.0`, `atol=1e-7`), identity adapter preprocessing, finite nonzero input-gradient L1 `14.360624313354492`, and production one-step PGD `Linf=0.007843166589736938`.
- Bartoldson fresh-process local-only audit on physical GPU 1: exact pinned-loader/ARD FP32 logit parity (`max_abs_diff=0.0`, `atol=1e-7`), model-embedded preprocessing, finite nonzero input-gradient L1 `16.796039581298828`, and production one-step PGD `Linf=0.007843166589736938`.
- Both audits enforced deterministic algorithms, deterministic cuDNN, disabled cuDNN benchmark and TF32, left teacher parameters frozen with no parameter gradients, restored model mode, and held inputs in pixel `[0,1]`.
- The reported AutoAttack accuracies `56.94%` and `73.71%` were not reproduced here. No AutoAttack, CIFAR training, W&B run, or accuracy measurement was performed.

### 2026-07-22 — A4 evidence

- Final review found a P1 fresh-clone restore gap: a committed verified lock could not materialize the Git-ignored cache. It also found a P2 acquisition gap: re-downloaded bytes were not compared with an existing locked SHA before external publication.
- `$ard-bug-hunt` identified the shared cause as conflating first identity registration with later local materialization. The fix split explicit `--update-lock` (missing-to-verified registration) from `--install-locked` (verified exact-SHA cache restore with byte-for-byte lock preservation), and made verified acquisition reject SHA drift before publication. Focused regressions passed: bootstrap `6 passed, 13 deselected`; acquisition `10 passed`. Details are in `docs/debugging/0005-verified-teacher-cache-restore.md`.
- Final impact-selected gate: `PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python scripts/verify.py --changed --non-scientific` selected T0/T1/T2 and passed `19 + 20 + 10 + 44 + 29 = 122` tests. W&B tests used offline/mock paths and emitted only known NVML/SDK warnings.
- The unchanged repeat of that exact gate reported five `cached pass` commands and executed no pytest command. `make lint` passed Ruff format/check for 102 files, mypy for 57 source files, import tests (`2 passed`), and both CLI help commands.
- Final scientific delta review closed both findings and reported no remaining P0–P2. Checkpoints remain outside Git. Heavy teacher accuracy evaluation, AutoAttack, CIFAR training, W&B online logging, and optional teachers remain deferred.
