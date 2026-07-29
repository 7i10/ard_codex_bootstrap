# GPU outage and campaign recovery

Date: 2026-07-29

## Observed failures

After both hosts recovered from an NVIDIA driver outage, three independent control-plane defects became visible:

1. Ferret jobs that had never launched were terminal `blocked` with the exact admission evidence
   `GPUInspectionError('nvidia-smi inventory failed; refusing GPU admission')`.
2. A controller reconciling completed PGD phases exited with
   `invalid job transition pgd_completed -> waiting_gpu` when a selected AutoAttack shared a busy GPU. Adding the
   missing transition alone would have been unsafe: the generic wait state previously resolved to `train`, which
   could have relaunched training instead of the pending AutoAttack.
3. The protected Ferret finalizer could not find the same-environment `wandb` console script. After that was
   corrected, evaluation rejected a teacher-identity mismatch because the finalizer loaded a separate config whose
   checkpoint path used a different absolute-path alias than the training run.
4. The first control-overlay launch omitted `ARD_CIFAR10_ROOT` from the controller environment. Three Ferret train
   wrappers exited during config expansion, before output creation, W&B initialization, model construction, or a GPU
   step. Their exact phase records are archived before the same job IDs are requeued.

No completed scientific phase was rerun, and no attack, checkpoint, batch size, or evaluation threshold was changed.

## Corrective contracts

- Recovery is dry-run by default and can requeue only explicitly named jobs with the exact GPU-inventory failure,
  no launch/phase evidence, no scientific output, healthy current inventory, and an exact protected-run release
  marker. The original failure is retained in `recovery_history`.
- Waiting states persist `pending_successor_phase`. `training_completed` resumes only PGD and `pgd_completed` resumes
  only AutoAttack. Unknown or incompatible markers block fail-closed rather than falling back to training.
- The watchdog is a host-local singleton. It runs a committed control-plane revision while keeping the scientific
  repository/config/phase commands fixed at
  `2d54b8230b8d14d13c1ea7472ccba53491b4d38d`; both revisions are recorded in `controller.json`.
- The watchdog supplies the dataset root, worker count, and exact teacher checkpoint paths/hashes itself; successful
  launch does not depend on the invoking shell exporting those runtime inputs.
- The protected finalizer prepends the active Python environment to `PATH` and evaluates with the training run's
  persisted `resolved_config.yaml`. Exact teacher identity validation remains enabled.

## Regression evidence

- Successor wait/resume and invalid-marker tests:
  `pytest -q tests/unit/test_campaign.py` — 24 passed.
- Transient recovery/watchdog tests:
  `pytest -q tests/unit/test_campaign_recovery.py` — 4 passed.
- Protected finalizer tests:
  `pytest -q tests/unit/test_campaign_finalizer.py` — 2 passed.

Real GPU training and AutoAttack were not repeated as tests. Runtime state and phase exit records remain the source of
truth for the resumed campaign.
