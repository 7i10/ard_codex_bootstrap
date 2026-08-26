# ERT / RSLAD shuffle-vs-augmentation RNG decomposition

## Status

- Owner: repository owner / one campaign owner
- Branch: `master`
- Current milestone: implementation and CPU isolation canary
- Production implementation: `3d26183b4861fa423c4cb6beba7105677284a85f`; the
  current clean metadata tip is `c4bf46630601e6aed70833b80ef1bc3f1dcfa30a`.
  The actual launch SHA must be captured in every run manifest; production has
  not started.

## Goal

Separate the data-side stochasticity observed in the completed RNG-source
decomposition into sample ordering (`S`, sampler/shuffle) and source-keyed
augmentation (`U`). The experiment keeps the RSLAD objective, training attack,
attack RNG, and other RNG fixed.

The key scientific question is whether `S` or `U` is the larger candidate source
of validation robust-accuracy trajectory variation. The two-factor interaction is
reported descriptively. No best-seed shopping and no state-aware ordering are
allowed in this campaign.

## Frozen contract

- Exact epoch-79 L2/L4 parents from the prior decomposition.
- BASE RSLAD, Teacher-clean KL-PGD10, $8/255$, $2/255$, 10 steps, random start.
- Fixed attack stream `A0` and fixed other stream `O0` across all arms.
- Source-keyed augmentation remains `epoch × source ID` keyed; changing order
  must not change a sample's augmentation view.
- 8 arms per teacher: REF1, REF2, SHUF1/2, AUG1/2, BOTH1/2.
- Continuation epochs 80–94, with checkpoints 84/89/94.
- Independent CE-PGD20 endpoint (epsilon `8/255`, step `2/255`, 20 steps,
  random start, eval mode, evaluation seed 0) on train and fixed validation.
- Hamster-only execution; Ferret excluded so matched arms share one host family.

The complete machine contract and deterministic seed registry are in
[`ert_rslad_shuffle_augmentation_decomposition_v1.json`](../experiments/ert_rslad_shuffle_augmentation_decomposition_v1.json).

## Stream audit and implementation

The existing `EpochSourceTransform` is already source-ID keyed. The code change
adds a separate `ShuffleAugmentationSeeds` contract and rebinds only the sampler
and augmentation seed after exact parent restore. The DataLoader worker generator
stays fixed, and attack/other streams stay fixed by the explicit four-way
contract. The split-stream canary must pass before GPU launch; its assertions are
recorded in [`ert_rslad_shuffle_augmentation_rng_audit_v1.json`](../experiments/ert_rslad_shuffle_augmentation_rng_audit_v1.json).

## Acceptance criteria

1. REF1/REF2 repeat exactly in the canary and have a residual reported rather
   than hidden behind an arbitrary threshold.
2. SHUF-only changes order while preserving each source ID's augmentation and
   attack random stream.
3. AUG-only preserves order while changing augmentation for at least one source
   ID and preserves the attack stream.
4. All 16 trajectories reach epoch 94 from the registered parents and retain
   complete source, parent, seed, host, and checkpoint lineage.
5. All 96 endpoint records (16 × 3 horizons × train/validation) use one fixed
   CE-PGD20 identity.
6. Report absolute metrics, REF residual, $Δ_S$, $Δ_U$, $Δ_{SU}$,
   interaction, mean absolute sensitivities, trajectory divergence, and the
   robust-overfitting gap from the full epoch log.

## Execution gates

- No production launch while the source tree is dirty or the source SHA is not
  pinned in the machine contract.
- Run the split-stream canary and a real parent/checkpoint one-batch smoke before
  the full matrix.
- Use Hamster GPUs with persistent supervision (`loginctl show-user "$USER"
  -p Linger --value` must be `yes`). Do not kill unrelated jobs.
- Feature/outcome endpoint work is independent and should be scheduled in
  parallel only after the training matrix is complete.
- Do not run official test, AutoAttack, new seeds, state-aware ordering,
  curriculum, weighted sampling, or any adaptive treatment.

## Analysis and stop rule

For each teacher/split/epoch, use REF1 as the reference:

$$
\Delta_S^{(r)} = Y(S_r,U_0)-Y(S_0,U_0),\quad
\Delta_U^{(r)} = Y(S_0,U_r)-Y(S_0,U_0)
$$

$$
\Delta_{SU}^{(r)} = Y(S_r,U_r)-Y(S_0,U_0),\quad
I^{(r)}=\Delta_{SU}^{(r)}-\Delta_S^{(r)}-\Delta_U^{(r)}.
$$

The primary endpoint is validation CE-PGD20 robust accuracy at epoch 94.
Source ranking is descriptive over the two registered perturbations and is not
used to choose a seed. After the 16 trajectories, endpoint collection, point
decomposition, and report are complete, stop and request human review.

## Milestones

- [x] M0: audit existing source-keyed augmentation, sampler, attack, and parent
  contracts.
- [x] M1: implement split shuffle/augmentation seed contract and CPU canary.
- [x] M2: preregister seed registry, arms, endpoint, and analysis rules.
- [ ] M3: commit/pin immutable source SHA and run Hamster preflight + real-parent
  smoke.
- [ ] M4: run 16 trajectories through epoch 94 with checkpoints 84/89/94.
- [ ] M5: run 96 independent endpoints and deterministic aggregation.
- [ ] M6: write report, update result status, run focused verification, and stop.

## Progress log

- 2026-08-24: Prior RNG decomposition was complete at commit `594f31a` with
  16 trajectories and 96 endpoint records; it found data-side effects larger
  than attack-side effects for the particular matched perturbations.
- 2026-08-24: Source audit confirmed CIFAR augmentation is already keyed by
  `augmentation_seed`, epoch, and stable source ID. A new independent
  shuffle/augmentation seed contract and split-stream canary were implemented.
- 2026-08-24: Focused RNG/Stage-A tests passed `13/13`; split-stream CPU canary
  passed all four assertions. GPU production remains gated on immutable commit
  and real-parent smoke.
- 2026-08-24: Split-stream implementation and preregistration were committed
  as `3d26183b4861fa423c4cb6beba7105677284a85f`; this is the implementation
  source SHA. A metadata-only pin followed at `c4bf466`; the launch SHA is
  recorded separately in each immutable run bundle.

## Completion report

Pending M3–M6. No scientific conclusion about shuffle versus augmentation is
made until the registered 16-arm campaign and endpoint matrix are complete.
