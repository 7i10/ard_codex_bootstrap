# Documentation index

`docs/` has grown to over a hundred files. This index gets a reader who has never seen this
project to the right document in one hop. It is grouped by the question a document answers, in
the order a newcomer needs the answers, and it does not invent new names for anything — headings
below describe groups that already exist in the file names and in `docs/CLAUDE_CODE_WORKFLOW.md`.

If two documents disagree about a number, both are listed as they are. This project's rule is:
disagreements are a finding for the human, not something a later document quietly overwrites.
`docs/NUMERIC_CONSISTENCY_AUDIT.md` (section 0) lists the fourteen numeric defects already known
about the documents in this index; read it before treating any single quoted number as final.

`docs/CONSOLIDATION_LOG.md` records what changed when this index was rebuilt on 2026-09-07 and
where every touched document's content now lives.

## 0. Start here

- [`../CLAUDE.md`](../CLAUDE.md) — the operating contract for this repository: the three planes, daily commands, hard rules, model policy.
- [Claude Code workflow](CLAUDE_CODE_WORKFLOW.md): why the repository is split into a scientific/execution/agent plane, and how a session runs day to day.
- [Scientific invariants](SCIENTIFIC_INVARIANTS.md): the attack, gradient, checkpoint, and evaluation rules that never change regardless of method.
- [Numeric consistency audit](NUMERIC_CONSISTENCY_AUDIT.md): fourteen verified numeric defects across the measurement and evidence documents below, with what each one could misdirect.

## 1. Measurement method — how big an effect has to be before it is real

Read this group before trusting any effect size quoted anywhere else in `docs/`.

- [Measurement standard v1](MEASUREMENT_STANDARD.md): the pre-registration rules now in force — arm naming, parent/replicate counts, horizon, power, coefficient calibration. Answers "what must a new plan do before it runs?"
- [Measurement design](MEASUREMENT_DESIGN.md): the read-only audit the standard is built on — how large the noise floor is before and after the epoch-100 learning-rate decay, and where a verdict actually stabilises. Answers "what could this project's screens ever have detected?"
- [Coefficient audit](COEFFICIENT_AUDIT.md): how each of the project's calibrated scalars was set, rounded, and whether any neighbouring value was ever tested.
- [Arm registry](ARM_REGISTRY.md): the canonical name, coefficient, and every alias for all 118 arms this project has run. Check here before naming or re-running an arm.
- [Post-decay noise floor](POST_DECAY_FLOOR.md): generated report — the measured (not bracketed) post-decay floor from plan 0092, six replicates at epochs 104/109/114.
- [Post-decay floor reclassification](POST_DECAY_FLOOR_RECLASSIFICATION.md): generated report — re-judges the epoch-114 online-state screen against the measured floor above instead of the old bracket.
- [Artifact location map](ARTIFACT_LOCATION_MAP.md): inventory of what exists on Hamster and Ferret, with a VERIFIED/DERIVED/etc. marker on every figure.
- [Artifact retention policy](ARTIFACT_RETENTION_POLICY.md): what gets kept vs. regenerated, classified by regeneration cost now that determinism is confirmed.

## 2. Evidence and verdicts — what the experiments so far actually show

- [Evidence reclassification](EVIDENCE_RECLASSIFICATION.md): every preregistered verdict in the project (38 rows), reclassified as STANDS / REFUTED / UNDERPOWERED / NOT-AN-EFFECT-CLAIM / UNRESOLVED against the measured floor. The single most load-bearing document in this group.
- [ERT/RSLAD research status summary](ERT_RESEARCH_STATUS_SUMMARY.md): the top-level navigation readout — one row per research question, linking to its source report. Not fully current with the 2026-09-06 floor measurement; see the numeric audit's finding #9.
- [Experiment dashboard](EXPERIMENT_DASHBOARD.md): the human-facing ledger of research question, conditions, confirmed progress, and W&B run roles.
- [Handoff 2026-09-06, Ferret to Hamster](HANDOFF_2026-09-06_FERRET_TO_HAMSTER.md): the point-in-time state transfer between hosts, including the post-decay floor discussion the numeric audit's finding #6 examines.
- [Best-oriented history-routing v2 results](HISTORY_ROUTING_V2_RESULTS.md): the project's only post-decay paired-fork placebo measurement; a Development No-Go verdict for that routing method.

## 3. Experiment records, by family

One report per completed screen or analysis. Filed under the family already named in its own
file name. A report's own verdict is not restated here — see group 2 for that.

### Stage A and the Clean-Wrong family (epoch-79 parent, pre-decay)

- [Stage A calibration](ERT_STAGE_A_CALIBRATION.md): coefficient calibration for the twelve Stage A treatment arms.
- [Stage A treatment results](ERT_STAGE_A_TREATMENT_RESULTS.md): the twelve-arm epoch-84 screen itself.
- [Stage A effect decomposition](ERT_STAGE_A_EFFECT_DECOMPOSITION.md): direct (selected-cohort), spillover, and held-out effects for every Stage A arm.
- [Stage A formula audit](ERT_STAGE_A_FORMULA_AUDIT.md): checks the implemented RSLAD loss against the intended formula.
- [Confirmatory T1/T2/T3 results](ERT_CONFIRMATORY_T123_RESULTS.md): re-run of three Stage A arms under new names, at three horizons.
- [Clean-Wrong broad screen results](ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md): fifteen Clean-Wrong mechanism variants against one control.
- [Clean-Wrong reliability proxy safety](ERT_CLEAN_WRONG_RELIABILITY_PROXY_SAFETY.md): whether the cheap KL-PGD10 online proxy is a safe stand-in for the CE-PGD20 endpoint.
- [Clean-Wrong rescue subtypes](ERT_CLEAN_WRONG_RESCUE_SUBTYPES.md): C0/C10/C12/C13 epoch-84 endpoint transitions broken into rescue/harm subtypes.
- [Clean-Wrong reliability stratified](ERT_CLEAN_WRONG_RELIABILITY_STRATIFIED.md): the same C0/C10/C12/C13 transition analysis, additionally stratified by pre-treatment Teacher reliability. Its per-subtype numbers differ from the document above at the third decimal (a different feature replay); both are kept as recorded — this was checked and is a genuine unresolved discrepancy, not a copy that can be merged.
- [Clean-Wrong reliability-gated CleanCE results](ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md): teacher-reliability-gated CleanCE variants against BASE and against the ungated arm.
- [Clean-Wrong margin calibration](ERT_CW_MARGIN_CALIBRATION.md): coefficient calibration for the teacher-probability-floor margin (TPFM) treatment.
- [Clean-Wrong margin generalization screen](ERT_CW_MARGIN_GENERALIZATION_SCREEN.md): held-out screen for the calibrated margin treatment.
- [Clean-Wrong margin lambda sensitivity](ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md): sweep of the margin coefficient lambda around its calibrated value.
- [Clean-Wrong margin local lambda stability](ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md): four matched RNG blocks used to check whether the lambda sweep's sign is stable.
- [Clean-Wrong margin RNG stability diagnostic](ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md): the project's primary control-vs-control noise measurement at epochs 84/89/94 — source of most of the pre-decay floor figure.
- [Clean-Wrong margin action map](ERT_CW_MARGIN_ACTION_MAP.md): whether action utility is heterogeneous by teacher margin bin, on the fixed training cohort.
- [Clean-Wrong A7 CleanCE ablation](ERT_CW_A7_CLEANCE_ABLATION.md): margin-only vs. margin-plus-CleanCE ablation for arm A7.
- [Clean-Wrong A7 CleanCE reuse audit](ERT_CW_A7_CLEANCE_REUSE_AUDIT.md): checks whether A7 was reused or relabelled across campaigns.
- [Clean-Wrong A7 mechanism diagnostic](ERT_CW_A7_MECHANISM_DIAGNOSTIC.md): whether A7's intended mechanism (margin pressure) actually moved.
- [Clean-Wrong generalization diagnostic](ERT_CW_GENERALIZATION_DIAGNOSTIC.md): generalization check for the Clean-Wrong treatment family. Generated report.
- [Clean-Wrong L4 parent recovery audit](ERT_CW_L4_PARENT_RECOVERY_AUDIT.md): recovers and audits the L4 parent lineage used by the Clean-Wrong screens.

### S3 history routing (epoch-79 parent, pre-decay)

- [S3 history routing: offline replay](ERT_S3_HISTORY_REPLAY.md): offline replay establishing the S3 routing mechanism before any new training.
- [S3 corrected history audit](ERT_S3_HISTORY_CORRECTED_AUDIT.md): mechanism audit after a correction to the history definition; passed the pre-GPU gate.
- [Majority-3 state smoothing versus action persistence](ERT_S3_HISTORY_PERSISTENCE_REPLAY.md): offline check of whether majority-3 smoothing changes action persistence.
- [History-smoothed S3 production screen](ERT_S3_HISTORY_PRODUCTION_RESULTS.md): the production screen for majority-3-smoothed online S3 routing.
- [Dynamic S3 recovery screen](ERT_DYNAMIC_S3_RECOVERY_RESULTS.md): freezing the S3 cohort vs. recomputing it every step.
- [Ordering mechanism discovery](ERT_RSLAD_ORDERING_MECHANISM_DISCOVERY.md): sixteen short forks probing whether a batch-order descriptor explains the ordering effect. Generated report.
- [Ordering mechanism and second intervention](ERT_RSLAD_ORDERING_MECHANISM_AND_SECOND_INTERVENTION.md): audit of existing ordering runs; concludes Phase A is blocked and the mechanism gate is not identified. Generated report.
- [History-balanced ordering dev](ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV.md): the original CONTROL/HISTORY_BALANCED launch, blocked pre-launch by a fail-closed RNG audit (changing sample order changed attack random-start assignment).
- [History-balanced ordering dev v2](ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md): the completed re-run under the corrected sample-keyed attack RNG contract. Generated report; supersedes v1's blocked attempt as the actual result, but v1 is kept because it documents why the design changed.
- [Best-oriented history-routing v2 results](HISTORY_ROUTING_V2_RESULTS.md): see group 2 above.

### RSLAD global augmentation and the I100 lineage

- [Static trajectory stabilization](ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md): CropShift vs. canonical RSLAD augmentation over the full 200-epoch trajectory.
- [Static augmentation family screen](ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md): whether stronger static policies (CROP_RE, IDBH_WEAK) applied throughout training help.
- [Stage-wise augmentation results](ERT_RSLAD_STAGEWISE_AUGMENTATION.md): switching augmentation policy partway through training. Generated report.
- [Single-switch augmentation timing](ERT_RSLAD_SINGLE_SWITCH_TIMING.md): sweep of the epoch at which the augmentation switch happens. Generated report.
- [Unseen-seed confirmation results](ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md): three fresh confirmation seeds for I100 vs. CROP_SUFFIX.
- [Unseen confirmation — orchestration audit](ERT_RSLAD_UNSEEN_CONFIRMATION_ORCHESTRATION_AUDIT.md): audits how the unseen-confirmation campaign was orchestrated.
- [Five-seed global & sample-level stochasticity](ERT_RSLAD_FIVE_SEED_STOCHASTICITY.md): BASE/CROPSHIFT/I100 across five independently seeded runs — the project's between-seed floor and its one five-seed confirmed effect.
- [I100 official CIFAR-10 test and AutoAttack](ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md): the project's only official-test, AutoAttack confirmation. Generated report.
- [I100 Clean-Wrong long-horizon confirmation](ERT_RSLAD_I100_CLEAN_WRONG_LONG_HORIZON.md): whether the Clean-Wrong treated-cohort rescue survives to epoch 199 on the I100 lineage.
- [I100 Clean-Wrong held-out generalization gap](ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md): direct train-cohort rescue vs. held-out accuracy for I100-lineage Clean-Wrong arms.
- [I100 Clean-Wrong gap completion and S2 bridge](ERT_RSLAD_I100_CW_GAP_COMPLETION_AND_S2_BRIDGE.md): closes the remaining Clean-Wrong/S2 comparison gap on the I100 lineage. Generated report.
- [I100 historical-action transfer screen](ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md): whether Stage-A-derived actions transfer to the I100 lineage.
- [Student History Predictive Validity](ERT_RSLAD_STUDENT_HISTORY_PREDICTIVE_VALIDITY.md): does student margin-history predict future failure better than current state.

### I100 post-decay S2 online-state family (epoch-99 parent, epochs 100-199)

- [I100 Online-State S2xT1 preservation screen](ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md): OS-PMP / OS-DBDP two-seed screen at e104/e109/e114. Generated report.
- [I100 S2xT1 Dynamic Boundary-Distance recovery results](ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md): DPM/D-BDD recovery attempt after the original secant formulation went non-finite.
- [I100 Dynamic BDD recovery audit](ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_AUDIT.md): audits the recovery campaign above.
- [I100 canonical S2 robust-boundary preservation screen](ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md): SBF and TPFM@S2T1 screened on the canonical S2 definition.
- [I100 canonical S2xT1 boundary geometry audit](ERT_RSLAD_I100_CANONICAL_S2_BOUNDARY_GEOMETRY_AUDIT.md): audits the boundary-distance geometry the canonical S2 screen relies on.
- [I100 Secant Boundary-Distance forensic audit](ERT_RSLAD_I100_SECANT_BOUNDARY_DISTANCE_FORENSIC.md): forensic root-cause for why the original secant BDD formulation diverged.
- [I100 S2xT1 longitudinal state audit](ERT_RSLAD_I100_S2_LONGITUDINAL_STATE_AUDIT.md): tracks S2xT1 sample-state membership over the post-decay horizon.
- [I100 Dynamic Boundary-Distance execution postmortem](ERT_RSLAD_I100_DYNAMIC_BDD_EXECUTION_POSTMORTEM.md): postmortem on launch/recovery delays during the Dynamic-BDD screen; basis for the launch-discipline documents in group 6.
- [I100 Online-State S2 launch postmortem](ERT_I100_ONLINE_STATE_S2_LAUNCH_POSTMORTEM.md): postmortem specific to the online-state S2 campaign's launch.
- [I100 Online-State S2: request-to-launch retrospective](ERT_I100_ONLINE_STATE_S2_REQUEST_TO_LAUNCH_RETRO.md): retrospective timing analysis for the same campaign's request-to-controller latency.
- [Online routing proxy results](ERT_ONLINE_ROUTING_PROXY_RESULTS.md): pre-screen diagnostics on the CE-PGD20 oracle vs. the KL-PGD10 online proxy for S1xT3-style transitions.

### Noise, RNG, and runtime characterization

- [RNG-source decomposition results](ERT_RSLAD_RNG_SOURCE_DECOMPOSITION.md): decomposes run-to-run divergence into attack-RNG, data-order, and combined sources; one of the two campaigns the pre-decay floor is built from.
- [Shuffle-vs-augmentation RNG decomposition results](ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md): the second of the two RNG-decomposition campaigns the pre-decay floor is built from.
- [Attack random-start randomness characterization](ERT_RSLAD_ATTACK_RANDOMNESS_CHARACTERIZATION.md): whether attack random-start choice materially moves an endpoint. Generated report.
- [Real-data long-run training determinism](ERT_RSLAD_REAL_DATA_TRAINING_DETERMINISM.md): confirms bit-identical `state_dict` reproduction under identical seed/environment for a real 15-epoch continuation.
- [Runtime performance audit v2](ERT_RSLAD_RUNTIME_PERFORMANCE_AUDIT.md): Hamster/Ferret throughput characterization, GPU2 NUMA finding, `torch.compile` parity check.
- [Historical treatment-response analysis](ERT_RSLAD_HISTORICAL_TREATMENT_RESPONSE_ANALYSIS.md): re-analysis of historical treatment-response primitives across the existing runs. Generated report.

### FF/NR causal routing and future-failure forecasting

- [FF/current-wrong future-failure forecasting status](FFNR_FORECASTING_STATUS.md): whether training-time student state forecasts failure on a later plateau; L/T/S/D CPU ablation.
- [FF/NR causal-pilot preparation status](FFNR_CAUSAL_PILOT_STATUS.md): fixed-cohort selector construction and a one-off loss-scale dry-run, before any GPU pilot.
- [FF/NR causal pilot results](FFNR_CAUSAL_PILOT_RESULTS.md): the two-seed, five-arm rescue/harm pilot itself.
- [FF/NR causal pilot subgroups](FFNR_CAUSAL_PILOT_SUBGROUPS.md): teacher/student strata and class-stratified bootstrap on the pilot's selected-vs-random contrast.
- [FF/NR causal pilot: epoch-79 to horizon-94 extension](FFNR_CAUSAL_HORIZON_EXTENSION_RESULTS.md): extends the pilot's horizon from epoch 83 to epoch 94.
- [FF/NR causal horizon: common CE-PGD20 endpoint](FFNR_CAUSAL_HORIZON_CE20_RESULTS.md): fixed-mask L2/L4 horizons 84/89/94 under one common eval-mode endpoint; a placebo-arm source for the pre-decay floor.
- [FFNR CE/KL x PGD10/20 factorial](FFNR_ATTACK_FACTORIAL_RESULTS.md): whether attack loss (CE/KL) and step count (PGD10/20) change which samples are flagged as failures.
- [Bartoldson CE-PGD20 replay](FFNR_BART_CE_PGD20_REPLAY.md): replays the Bartoldson lineage (L1/L3) under CE-PGD20 for comparability with the Chen lineage.
- [Bartoldson dense-checkpoint audit](FFNR_BART_DENSE_AUDIT.md): checks existing-state completeness and a blocked L3 recovery decision for the Bartoldson lineage.
- [FF/NR state and Teacher mechanism](FFNR_STATE_AND_TEACHER_MECHANISM.md): 3-state candidates, margin risk surfaces, and Teacher clean/response decomposition.
- [FF/NR next-stage evidence results](FFNR_NEXT_EVIDENCE_STATUS.md): chance-adjusted ground truth, cross-seed Teacher information, Teacher-correct subset, IRT gate — all CPU-only replay, no new training.

### FFNR human image review

- [FFNR blind image review](FFNR_HUMAN_REVIEW.md): how to render and run the role-blind CIFAR review panel, and the judgement categories.
- [FFNR human review results](FFNR_HUMAN_REVIEW_RESULTS.md): the 200-image panel's judgements and input-hash provenance.
- [FFNR human review analysis](FFNR_HUMAN_REVIEW_ANALYSIS.md): class-level error rates, teacher confusion matrix, and panel-conditioned model error breakdown.

### Other diagnostics

- [Seed-0 signal audit](SIGNAL_AUDIT.md): exploratory association between Student/Joint signals and periodic checkpoints, four seed-0 runs.
- [ERT state-conditioned mechanism pilot](ERT_STATE_CONDITIONED_MECHANISM_RESULTS.md): the M0 overlay pilot; Stage A GPU training not yet started at time of writing.

## 4. Plans and decisions

- [`plans/`](plans/) (94 files): one implementation/experiment plan per file, numbered sequentially. `0001` is the bootstrap plan; the newest is the current campaign-host-environment plan.
- [`decisions/`](decisions/) (2 decision packets plus a README): human decisions made from imported campaign results — `chosen: null` means no decision has been made yet and no new scientific job should start on that question.
- [`reviews/`](reviews/): point-in-time contract reviews (currently the ERT routing-contract review).

## 5. Debugging and postmortems

- [`debugging/`](debugging/) (20 files): one record per resolved bug or operational incident, numbered sequentially — DDP races, checkpoint/resume failures, W&B duplication, host outages. Never deleted.
- Two I100-campaign postmortems live outside `debugging/` because they are ERT-numbered experiment records rather than bug records: [Dynamic BDD execution postmortem](ERT_RSLAD_I100_DYNAMIC_BDD_EXECUTION_POSTMORTEM.md) and [Online-State S2 launch postmortem](ERT_I100_ONLINE_STATE_S2_LAUNCH_POSTMORTEM.md); see also the [request-to-launch retrospective](ERT_I100_ONLINE_STATE_S2_REQUEST_TO_LAUNCH_RETRO.md).

## 6. Execution plane — running GPU jobs

Currently used, per `docs/CLAUDE_CODE_WORKFLOW.md`:

- [Production Launch Gate](PRODUCTION_LAUNCH_GATE.md): the validation/freeze layer in front of the orchestrator — preflight, dry-run, canary, launch.
- [Multi-GPU Experiment Orchestration Skill](MULTI_GPU_EXPERIMENT_ORCHESTRATION_SKILL.md): the generic, RSLAD-unaware scheduling and DAG-chaining layer the gate hands off to.
- [Ferret execution protocol](FERRET_EXECUTION_PROTOCOL.md): Hamster-as-planning-node / Ferret-as-execution-node split, and the SSH/rsync boundary between them.
- [ARD workspace contract](WORKSPACE_CONTRACT.md): the canonical future runtime paths on Hamster and Ferret, and the historical-root policy.
- [ARD operational foundation](ARD_OPERATIONAL_FOUNDATION.md): completed milestone that standardized runtime-state location and the generic launch path; its layer-ownership policy is still cited by the launch-gate test suite.

Superseded (Codex-era; the systems they describe are listed as frozen in `CLAUDE.md` and
`docs/CLAUDE_CODE_WORKFLOW.md` — kept for historical reference only, each with a header saying so):

- [Experiment Launch Discipline](EXPERIMENT_LAUNCH_DISCIPLINE.md) — the launch-ledger SLO procedure `launch_ledger.py` implemented.
- [Experiment Execution Fast Path](EXPERIMENT_FAST_PATH.md) — the validated-runtime-signature fast lane.
- [Experiment reconciler and single-owner postprocessing](EXPERIMENT_RECONCILER.md) — `reconcile_experiment.py`.
- [Experiment Automation Bridge Hardening](EXPERIMENT_AUTOMATION_BRIDGE_HARDENING.md) — the reconciler plus the PR #1 event bus.
- [Task-context protocol](TASK_CONTEXT_PROTOCOL.md) — `task_context.py`, replaced by reading `state.json` / `run-bundle` directly.

## 7. Foundations and reference

- [Implementation specification](IMPLEMENTATION_SPEC.md): repository structure and module responsibilities.
- [Test strategy](TEST_STRATEGY.md): test tiers, change-impact selection, pass caching, numeric tolerances, GPU exclusivity.
- [W&B protocol](WANDB_PROTOCOL.md): tier/state naming, group/job-type conventions, artifacts, the fixed sample table.
- [Experiment protocol](EXPERIMENT_PROTOCOL.md): run tiers (audit/pilot/canonical production), baselines, seeds, evaluation.
- [Upstream baselines](UPSTREAM_BASELINES.md): pinned SAAD/TRADES commits, license evidence, known differences from upstream.
- [Reproduction status](REPRODUCTION_STATUS.md): what is implemented, what has been run, what heavy experiments have not, and the actual CLI commands.
- [Research decisions](RESEARCH_DECISIONS.md): the early implementation-level decisions (D1-D8) fixed before this project's first training run — single-teacher scope, RSLAD as the development base, which student signal comes first.
- [ARD research issues and proposals](ARD_RESEARCH_ISSEUES_AND_PROPOSALS.md) (2026-07-31): early independent research-proposal survey, in Japanese; still cited by plans 0013/0014.
- [ARD future-forgetting prior work and research plan](ARD_FUTURE_FORGETTING_PRIOR_WORK_AND_RESEARCH_PLAN.md) (2026-07-31): literature review separating what prior work already shows from what this project's own experiments were expected to add.
- [Teacher config fragments](../configs/teachers/): strict RobustBench teacher configs; checkpoints are registered explicitly.

## CLI entry points

```bash
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml>
PYTHONPATH=src python -m ard.cli.train --config <experiment.yaml> --resume <output>/last.pt
PYTHONPATH=src python -m ard.cli.evaluate --config <experiment.yaml> --checkpoint-dir <output>
PYTHONPATH=src python -m ard.cli.status --root <output-root> --format markdown
python scripts/verify.py --changed
```

`evaluate` reads only saved checkpoints; with `--checkpoint-dir` it follows the config's
`evaluation.checkpoints` setting (default `both`) and evaluates `best.pt` and `last.pt`
separately. Full AutoAttack never starts from ordinary train/test; it must be enabled explicitly
in the evaluation config and run as a separate process with `--allow-autoattack`.

Running epoch, step, update time, and terminal state are derived by the status CLI from
`run-bundle/manifest.json`. Cross-host live viewing uses W&B; the Git-tracked dashboard body is
not hand-edited for process monitoring.

Run the Teacher audit (W&B-free, PGD screening) on one GPU first. After that, use the two-GPU
pilot and production commands in [Experiment protocol](EXPERIMENT_PROTOCOL.md); set
`WANDB_PROJECT=single-teacher-ard` (and teacher-specific group variables) for pilot/production.
Smoke runs may stay disabled and do not upload to W&B; production may not disable it.

Before starting CIFAR-10 production training, check the unexecuted items and production guard in
[Reproduction status](REPRODUCTION_STATUS.md).
