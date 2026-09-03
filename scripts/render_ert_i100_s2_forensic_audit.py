#!/usr/bin/env python3
"""Render hash-bound human reports for the completed I100 S2 forensic audit.

This is deliberately a CPU-only renderer.  It consumes only completed
read-only replay/forensic artifacts and the registered Dynamic-BDD recovery
result; it neither loads a model nor creates an attack.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def number(value: float) -> str:
    return f"{value:.3g}"


def count(mapping: dict[str, int], key: str) -> int:
    return int(mapping.get(key, 0))


def fixed_count(arm: dict[str, Any], epoch: int, joint: str) -> int:
    return count(arm["fixed_mask_current_branch_counts"][str(epoch)], joint)


def fd_summary(secant: dict[str, Any]) -> dict[str, Any]:
    scalar_errors = []
    scalar_relative_errors = []
    for candidate in secant["scalar_finite_difference"]:
        for step in candidate["steps"]:
            if step.get("kink_safe"):
                scalar_errors.extend((step["absolute_error_adv"], step["absolute_error_clean"]))
                scalar_relative_errors.extend(
                    (
                        step["absolute_error_adv"]
                        / max(abs(step["autograd_adv"]), abs(step["finite_difference_adv"]), 1e-12),
                        step["absolute_error_clean"]
                        / max(abs(step["autograd_clean"]), abs(step["finite_difference_clean"]), 1e-12),
                    )
                )
    parameter = secant["parameter_directional_finite_difference"]
    parameter_checks = [step for step in parameter["checks"] if step["kink_safe"]]
    if not parameter_checks:
        raise ValueError("no kink-safe train-mode parameter finite-difference points")
    parameter_errors = [step["absolute_error"] for step in parameter_checks]
    best_parameter = min(
        parameter_checks,
        key=lambda step: (
            step["absolute_error"] / max(abs(step["autograd_directional"]), abs(step["central_difference"]), 1e-12)
        ),
    )
    return {
        "scalar_max_absolute_error": max(scalar_errors),
        "scalar_max_relative_error": max(scalar_relative_errors),
        "parameter_max_absolute_error": max(parameter_errors),
        "parameter_best_absolute_error": best_parameter["absolute_error"],
        "parameter_best_relative_error": best_parameter["absolute_error"]
        / max(abs(best_parameter["autograd_directional"]), abs(best_parameter["central_difference"]), 1e-12),
        "parameter_best_step": best_parameter["step"],
        "parameter": parameter["parameter"],
        "state_restored": parameter.get("state_hash_before_after") == "identical",
        "parameter_mode": parameter["student_mode"],
    }


def longitudinal_markdown(longitudinal: dict[str, Any], *, artifact_path: Path, artifact_sha: str) -> str:
    lines = [
        "# I100 S2×T1 Longitudinal State Audit",
        "",
        "## Conclusion",
        "",
        "The fixed epoch-99 S2×T1 cohort is highly nonstationary at the four registered "
        "CE-PGD20 observations.  Therefore an e114 S2×T1 membership must not be described "
        "as continuous membership since e99.  These results are an offline, fixed-cohort "
        "description only; they do not validate or instantiate an online selector.",
        "",
        "The primary trajectory uses raw, unaugmented train images and sample-keyed CE-PGD20 "
        "under the e99 key at e99/e104/e109/e114.  Historical augmented/batch-keyed replays "
        "are intentionally excluded from this join.  The KL-PGD10 values below are separate "
        "checkpoint no-update runtime proxies, not historical action logs.",
        "",
        "## Contract and lineage",
        "",
        "- Student branches are mutually exclusive: Clean-Wrong; S3-non-Clean-Wrong; S2; and S1. "
        "Teacher T1/T2/T3 is recorded independently.",
        "- Primary fixed cohort is e99 S2×T1: dev-1 $n=2{,}212$, dev-2 $n=2{,}141$.",
        "- Observation attack is the registered sample-keyed CE-PGD20 contract.  Endpoint continuity is "
        "observed-only, not continuous-time.",
        "- The replay configs are host-path-rebased copies only: their sole semantic diff from the accepted "
        "parent configs is the absolute Teacher checkpoint path, while the frozen Teacher SHA-256 is unchanged.",
        f"- Machine artifact: `{artifact_path}` (SHA-256 `{artifact_sha}`).",
        "",
        "## Fixed-cohort current state",
        "",
        "Counts below retain the same e99 S2×T1 IDs; S2×T1 is the currently target-matching subset.",
        "",
        "| seed | arm | e99 S2×T1 | e104 S2×T1 | e109 S2×T1 | e114 S2×T1 | e114 S1 | e114 S3-non-CW | e114 Clean-Wrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, seed_data in longitudinal["seeds"].items():
        for arm_name, arm in seed_data["arms"].items():
            e114 = arm["fixed_mask_current_branch_counts"]["114"]
            lines.append(
                "| {seed} | {arm} | {e99} | {e104} | {e109} | {e114} | {s1} | {s3} | {cw} |".format(
                    seed=seed,
                    arm=arm_name.upper(),
                    e99=seed_data["fixed_mask"]["n"],
                    e104=fixed_count(arm, 104, "S2xT1"),
                    e109=fixed_count(arm, 109, "S2xT1"),
                    e114=fixed_count(arm, 114, "S2xT1"),
                    s1=count(e114, "S1xT1") + count(e114, "S1xT2") + count(e114, "S1xT3"),
                    s3=(count(e114, "S3-non-CWxT1") + count(e114, "S3-non-CWxT2") + count(e114, "S3-non-CWxT3")),
                    cw=count(e114, "Clean-WrongxT1") + count(e114, "Clean-WrongxT2") + count(e114, "Clean-WrongxT3"),
                )
            )
    lines += [
        "",
        "## Observed membership and re-entry",
        "",
        "P1–P5 are intentionally overlapping observed-endpoint indicators; `membership patterns` are the "
        "mutually exclusive partition.  P6 (multiple exit/re-entry) is not observable under four endpoints "
        "that begin target-active.  An explicit route only means that route at the registered endpoints.",
        "",
        "| seed | arm | P1 all observed S2×T1 | P2 terminal S1/outside S2 | P3 observed S3-non-CW | P4 observed Clean-Wrong | P5 observed leave→re-enter | P6 multiple exit/re-entry |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for seed, seed_data in longitudinal["seeds"].items():
        for arm_name, arm in seed_data["arms"].items():
            flags = arm["fixed_cohort"]["overlapping_observed_indicators"]
            lines.append(
                "| {seed} | {arm} | {p1} | {p2} | {p3} | {p4} | {p5} | {p6} |".format(
                    seed=seed,
                    arm=arm_name.upper(),
                    p1=count(flags, "P1_all_observed_S2xT1"),
                    p2=count(flags, "P2_S1_then_outside_S2_later"),
                    p3=count(flags, "P3_observed_S3_nonCW"),
                    p4=count(flags, "P4_observed_CleanWrong"),
                    p5=count(flags, "P5_leave_then_reenter_S2xT1"),
                    p6="not observable",
                )
            )
    lines += [
        "",
        "| seed | arm | P1 / fixed e99 mask | P1 / current e114 S2×T1 target | P5 / fixed e99 mask | S2→S1→S2 | S2→S3-non-CW→S2 | S2→Clean-Wrong→S2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, seed_data in longitudinal["seeds"].items():
        for arm_name, arm in seed_data["arms"].items():
            fixed = arm["fixed_cohort"]
            n = fixed["n"]
            e114_target_n = arm["current_target_n_by_epoch"]["114"]
            flags = fixed["overlapping_observed_indicators"]
            routes = fixed["explicit_observed_reentry_routes"]
            lines.append(
                "| {seed} | {arm} | {p1}/{n} ({p1_fraction:.1%}) | {p1}/{e114_target_n} ({p1_target_fraction:.1%}) | "
                "{p5}/{n} ({p5_fraction:.1%}) | {s1} | {s3} | {cw} |".format(
                    seed=seed,
                    arm=arm_name.upper(),
                    p1=count(flags, "P1_all_observed_S2xT1"),
                    p5=count(flags, "P5_leave_then_reenter_S2xT1"),
                    n=n,
                    e114_target_n=e114_target_n,
                    p1_fraction=count(flags, "P1_all_observed_S2xT1") / n,
                    p1_target_fraction=count(flags, "P1_all_observed_S2xT1") / e114_target_n,
                    p5_fraction=count(flags, "P5_leave_then_reenter_S2xT1") / n,
                    s1=count(routes, "S2_to_S1_to_S2"),
                    s3=count(routes, "S2_to_S3_nonCW_to_S2"),
                    cw=count(routes, "S2_to_CleanWrong_to_S2"),
                )
            )
    lines += [
        "",
        "## Fixed-mask divergence from current S2×T1",
        "",
        "| seed | arm | epoch | current S2×T1 | fraction of fixed mask | current S2×T2/T3 | current S1 | current S3-non-CW | current Clean-Wrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, seed_data in longitudinal["seeds"].items():
        for arm_name, arm in seed_data["arms"].items():
            n = arm["fixed_cohort"]["n"]
            for epoch in (104, 109, 114):
                values = arm["fixed_mask_current_branch_counts"][str(epoch)]
                target_n = count(values, "S2xT1")
                s2_other = count(values, "S2xT2") + count(values, "S2xT3")
                s1 = count(values, "S1xT1") + count(values, "S1xT2") + count(values, "S1xT3")
                s3 = count(values, "S3-non-CWxT1") + count(values, "S3-non-CWxT2") + count(values, "S3-non-CWxT3")
                cw = count(values, "Clean-WrongxT1") + count(values, "Clean-WrongxT2") + count(values, "Clean-WrongxT3")
                lines.append(
                    f"| {seed} | {arm_name.upper()} | {epoch} | {target_n} | {target_n / n:.1%} | "
                    f"{s2_other} | {s1} | {s3} | {cw} |"
                )
    lines += [
        "",
        "### Teacher transitions whenever Student is S2 at either adjacent endpoint",
        "",
        "| seed | arm | T1→T1 | T1→T2 | T1→T3 | T2→T1 | T2→T2 | T2→T3 | T3→T1 | T3→T2 | T3→T3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, seed_data in longitudinal["seeds"].items():
        for arm_name, arm in seed_data["arms"].items():
            transitions = arm["fixed_cohort"]["teacher_transitions_when_student_S2_at_either_endpoint"]
            lines.append(
                "| {seed} | {arm} | {t1t1} | {t1t2} | {t1t3} | {t2t1} | {t2t2} | {t2t3} | {t3t1} | {t3t2} | {t3t3} |".format(
                    seed=seed,
                    arm=arm_name.upper(),
                    t1t1=count(transitions, "T1_to_T1"),
                    t1t2=count(transitions, "T1_to_T2"),
                    t1t3=count(transitions, "T1_to_T3"),
                    t2t1=count(transitions, "T2_to_T1"),
                    t2t2=count(transitions, "T2_to_T2"),
                    t2t3=count(transitions, "T2_to_T3"),
                    t3t1=count(transitions, "T3_to_T1"),
                    t3t2=count(transitions, "T3_to_T2"),
                    t3t3=count(transitions, "T3_to_T3"),
                )
            )
    lines += [
        "",
        "### New observed S2×T1 entrants",
        "",
        "| seed | arm | entrants | e99 S2×T2/T3 | e99 S1 | e99 S3-non-CW | e99 Clean-Wrong | one endpoint | repeated one run | re-entry (≥2 runs) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, seed_data in longitudinal["seeds"].items():
        for arm_name, arm in seed_data["arms"].items():
            entrants = arm["new_entrants"]
            origins = entrants["e99_origin"]
            persistence = entrants["persistence"]
            lines.append(
                "| {seed} | {arm} | {n} | {s2_other} | {s1} | {s3} | {cw} | {one} | {repeated} | {reentry} |".format(
                    seed=seed,
                    arm=arm_name.upper(),
                    n=entrants["n"],
                    s2_other=count(origins, "e99_S2xT2T3"),
                    s1=count(origins, "e99_S1"),
                    s3=count(origins, "e99_S3-non-CW"),
                    cw=count(origins, "e99_Clean-Wrong"),
                    one=count(persistence, "one-endpoint-only"),
                    repeated=count(persistence, "repeated"),
                    reentry=count(persistence, "re-entry"),
                )
            )
    lines += [
        "",
        "## Checkpoint no-update runtime proxy",
        "",
        "The following table evaluates the exact fixed e99 mask at each saved checkpoint under a fresh KL-PGD10 "
        "training-view proxy.  It is not a reconstruction of historical minibatch activity.  `extra-loss positive` "
        "uses the KL10 Student-selected-rival Teacher-pair gate; it is not CE20 Teacher T1 and may be positive while "
        "the Teacher is globally CE20-wrong.  Control has no extra loss.",
        "",
        "| seed | arm | epoch | current branch | fixed IDs | proxy extra-loss positive | fraction |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    activity = longitudinal["checkpoint_no_update_runtime_activity_proxy"]
    for seed, seed_data in activity.items():
        for arm_name, arm_data in seed_data.items():
            for epoch, epoch_data in arm_data.items():
                for branch, values in epoch_data["by_current_branch"].items():
                    if values["fixed_mask_n"]:
                        lines.append(
                            "| {seed} | {arm} | {epoch} | {branch} | {n} | {active} | {fraction:.3f} |".format(
                                seed=seed,
                                arm=arm_name.upper(),
                                epoch=epoch,
                                branch=branch,
                                n=values["fixed_mask_n"],
                                active=values["extra_loss_active_n"],
                                fraction=values["extra_loss_active_fraction"],
                            )
                        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- The fixed-mask causal screen remains evaluable, but its treatment effect should not be narrated as an "
        "effect on a persistent S2×T1 population.",
        "- Current S2×T1 entrants are a distinct, untreated-by-fixed-mask descriptive population; their origin "
        "mix is recorded above rather than folded into the fixed cohort.",
        "- These data motivate a separate, preregistered online-state contract if an intervention is proposed. "
        "This audit does not launch or endorse one.",
        "",
        "## Discussion-only next-experiment frame",
        "",
        "If human review authorizes a successor, the minimal candidate comparison is I100 Control versus "
        "Online-State Pair-Margin Preservation versus Online-State Detached Boundary-Distance Preservation. "
        "Its Phase-1 action contract should keep current Clean-Wrong, S3-non-CW, S2×T2/T3, and S1 on baseline "
        "RSLAD, and act only on current S2×T1.  This is a proposal boundary, not a launch authorization.",
        "",
    ]
    return "\n".join(lines)


def secant_markdown(
    secant_by_seed: dict[str, dict[str, Any]],
    historical: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_sha: str,
) -> str:
    sbdd = historical["sbdd_numerics"]
    calibration = historical["sbdd_calibration"]
    lines = [
        "# I100 Secant Boundary-Distance Forensic Audit",
        "",
        "## Classification",
        "",
        "**SBPF2 — FORMULA_SINGULARITY_SUPPORTED.**  The corrected v2 Student parameter graph agrees in "
        "sign and scale with the audited scalar and directional finite differences, so this audit does not retain "
        "evidence for a v2 implementation/normalization bug.  At the same time, real selected samples show a strongly tailed "
        "gradient-ratio distribution tied descriptively to the reciprocal secant geometry.  The historical v2 "
        "training became non-finite on both development seeds.  The frozen median calibration did not control "
        "that tail.  This classification describes the audited formula/parameterization; it does not select an "
        "epsilon, a cap, or a replacement method.",
        "",
        "S-BDD: **NUMERICALLY_UNSUPPORTED** — corrected secant formulation became non-finite reproducibly in "
        "both dev seeds; excluded from causal utility comparison.",
        "",
        "## Source-to-equation audit",
        "",
        "| version | Student secant denominator | Student graph | Teacher terms | result |",
        "| --- | --- | --- | --- | --- |",
        "| v1 historical | $q_S=\lvert m_S^{adv}-m_S^{clean}\\rvert/(\\rho+\\epsilon)$ | $q_S$ detached | detached | superseded; not the registered recovery intervention |",
        "| v2 corrected | same $q_S$ | preserved; no detach | $m_T^{adv},m_T^{clean},q_T,d_T$ detached | audited here and used in both failed S-BDD recovery runs |",
        "",
        "For v2, $d_S=m_S^{adv}/(q_S+\\epsilon)$, $d_T=m_T^{adv}/(q_T+\\epsilon)$, and "
        "$L=\\tfrac12[\\max(0,d_T-d_S)]^2$.  The selected mask, positive-Teacher gate, and $\\rho$ are detached; "
        "the caller retains the full-batch mean and does not selected-count-normalize.",
        " The audited source and frozen v2 contract agree on the pair, clean/adversarial views, epsilon placement, "
        "detach ownership, gates, full-batch reduction, and a single coefficient application.",
        "",
        "## No-update derivative and restoration checks",
        "",
        "| seed | Teacher-pair-gated fixed-mask samples across 4 natural batches | scalar FD max abs/relative error | parameter FD best abs/relative error | best step | Student forward mode | state restored bitwise |",
        "| --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    for seed, result in secant_by_seed.items():
        summary = fd_summary(result)
        lines.append(
            f"| {seed} | {result['sample_count']} | {number(summary['scalar_max_absolute_error'])} / "
            f"{summary['scalar_max_relative_error']:.3g} | {number(summary['parameter_best_absolute_error'])} / "
            f"{summary['parameter_best_relative_error']:.3g} (`{summary['parameter']}`) | "
            f"{summary['parameter_best_step']:.0e} | `{summary['parameter_mode']}` | {str(summary['state_restored']).lower()} |"
        )
    lines += [
        "",
        "Scalar partials are reported only after their own $m_S^{adv}$ or $m_S^{clean}$ abs/ReLU regions remain "
        "unchanged at both perturbations.  Parameter checks likewise require preserved Student-margin abs sign, "
        "hinge, Teacher-pair gate, and $\rho$ gate, and restore the full Student parameter/buffer state before every "
        "train-mode forward.  The finite-difference audit is an implementation check, not an optimizer update or a training replay.",
        "",
        "## Real-checkpoint tail diagnostics at coefficient 1",
        "",
        "| seed | $q_S$ min / median / max | $1/(q_S+\\epsilon)$ p95 / max | ratio p50 / p95 / p99 / max | raw loss p50 / p99 / max | Spearman ratio vs $1/q_S$ |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for seed, result in secant_by_seed.items():
        sample = result["sample_summary"]
        q = sample["q_student"]
        inverse = sample["inverse_q_student"]
        ratio = sample["gradient_ratio_at_1"]
        raw_loss = sample["raw_loss"]
        corr = result["rank_correlations"]["all_teacher_pair_gated_selected"]["inverse_q_student"]
        lines.append(
            "| {seed} | {qmin:.3g} / {qmed:.3g} / {qmax:.3g} | {ip95:.3g} / {imax:.3g} | "
            "{rmed:.3g} / {rp95:.3g} / {rp99:.3g} / {rmax:.3g} | {lmed:.3g} / {lp99:.3g} / {lmax:.3g} | {corr:.3f} |".format(
                seed=seed,
                qmin=q["min"],
                qmed=q["median"],
                qmax=q["max"],
                ip95=inverse["p95"],
                imax=inverse["max"],
                rmed=ratio["median"],
                rp95=ratio["p95"],
                rp99=ratio["p99"],
                rmax=ratio["max"],
                lmed=raw_loss["median"],
                lp99=raw_loss["p99"],
                lmax=raw_loss["max"],
                corr=corr,
            )
        )
    lines += [
        "",
        "The observed $q_S$ values do not approach $\\epsilon$ in these four-batch probes.  The supported "
        "instability claim is therefore narrower: the reciprocal secant parameterization is highly sensitive to "
        "small clean–adversarial Student-margin differences, producing a heavy intervention-gradient tail even "
        "away from the literal epsilon limit.",
        "",
        "## Epsilon diagnostic only",
        "",
        "This scalar sensitivity holds the recorded real-batch values fixed.  It is not a coefficient/epsilon "
        "selection and is not evidence for a stabilized training variant.",
        "",
        "| seed | epsilon | raw loss median | raw loss max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed, result in secant_by_seed.items():
        for epsilon, values in result["epsilon_sensitivity_diagnostic_only"].items():
            raw_loss = values["raw_loss"]
            lines.append(f"| {seed} | {epsilon} | {raw_loss['median']:.6g} | {raw_loss['max']:.6g} |")
    lines += [
        "",
        "## Largest audited gradient-ratio samples",
        "",
        "| seed | stable ID | ratio at coefficient 1 | $q_S$ | $|m_S^{adv}-m_S^{clean}|$ | $d_S$ | $d_T$ | gap | raw loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, result in secant_by_seed.items():
        for sample in result["top_gradient_ratio_samples"][:5]:
            lines.append(
                "| {seed} | {sample_id} | {ratio:.3g} | {q:.3g} | {delta:.3g} | {ds:.3g} | {dt:.3g} | {gap:.3g} | {loss:.3g} |".format(
                    seed=seed,
                    sample_id=sample["sample_id"],
                    ratio=sample["gradient_ratio_at_1"],
                    q=sample["q_student"],
                    delta=sample["abs_student_margin_delta"],
                    ds=sample["d_student"],
                    dt=sample["d_teacher"],
                    gap=sample["hinge_gap"],
                    loss=sample["raw_loss"],
                )
            )
    ratio_summary = calibration["achieved_ratio_summary"]["secant_boundary_distance"]
    lines += [
        "",
        "## Historical v2 calibration and failure evidence",
        "",
        "The pooled v2 calibration used the frozen coefficient "
        f"`{calibration['coefficients']['secant_boundary_distance']:.12g}` for a median target of "
        f"`{calibration['target_gradient_ratio']:.2f}`.  Its achieved ratios were min "
        f"`{ratio_summary['min']:.5g}`, median `{ratio_summary['median']:.3g}`, max "
        f"`{ratio_summary['max']:.5g}`, IQR `{ratio_summary['iqr']:.5g}`.  Thus median matching did not "
        "bound the observed tail; this is a calibration limitation that coexists with, rather than replaces, "
        "the formula-sensitivity evidence above.",
        "",
        "| seed | source / host | v2 formula and coefficient | last retained finite evidence | first non-finite / terminal evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for seed, record in sbdd.items():
        last = f"e{record['last_retained_finite_epoch']}; loss {record['last_retained_train_loss']:.6g}"
        if record.get("last_checkpoint_model_max_abs") is not None:
            last += f"; $|w|_{{max}}$ {record['last_checkpoint_model_max_abs']:.6g}"
        lines.append(
            f"| {seed} | {record['host']} | `{record['formula_version']}`, "
            f"coefficient {record['coefficient']:.12g}, $\\epsilon={record['boundary_epsilon']:.0e}$ | "
            f"{last} | {record['first_nonfinite_detection']} |"
        )
    lines += [
        "",
        "Both failures used `student_parameter_graph_v2`, the same frozen v2 calibration artifact and epsilon, "
        "but occurred on Hamster GPU1 and Ferret GPU0.  This rules out a host/GPU-specific explanation at the "
        "available resolution.  Control, DPM, and D-BDD completed e114 with finite telemetry and registered "
        "endpoints, so the main causal comparison remains evaluable.",
        "",
        "## Causal utility remains evaluable without S-BDD",
        "",
        "| seed | e114 DPM − Control held-out robust | e114 D-BDD − Control | e114 D-BDD − DPM |",
        "| --- | ---: | ---: | ---: |",
    ]
    held_out = historical["held_out"]
    for seed, rows in held_out.items():
        control = rows["control"]["114"]["robust_accuracy"]
        dpm = rows["dpm"]["114"]["robust_accuracy"]
        dbdd = rows["dbdd"]["114"]["robust_accuracy"]
        lines.append(f"| {seed} | {pp(dpm - control)} | {pp(dbdd - control)} | {pp(dbdd - dpm)} |")
    lines += [
        "",
        "D-BDD versus DPM is mixed across the two development seeds, so this audit does not support a D-BDD "
        "promotion or e199 extension.  No floor/cap/smoothed reciprocal is tried here.  The current S-BDD "
        "contract is closed as numerically unsupported; any stabilized secant variant remains a discussion-only "
        "redesign candidate and would require a separate scientific contract, calibration, and experiment.",
        "",
        f"Machine artifact: `{artifact_path}` (SHA-256 `{artifact_sha}`).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longitudinal", type=Path, required=True)
    parser.add_argument("--secant-dev1", type=Path, required=True)
    parser.add_argument("--secant-dev2", type=Path, required=True)
    parser.add_argument("--historical-results", type=Path, required=True)
    parser.add_argument("--longitudinal-report", type=Path, required=True)
    parser.add_argument("--secant-report", type=Path, required=True)
    parser.add_argument("--secant-output", type=Path, required=True)
    parser.add_argument("--analysis-source-sha", required=True)
    parser.add_argument("--secant-replay-source-sha", required=True)
    args = parser.parse_args()

    longitudinal = read_json(args.longitudinal)
    historical = read_json(args.historical_results)
    secant = {"dev-1": read_json(args.secant_dev1), "dev-2": read_json(args.secant_dev2)}
    secant_output = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_secant_boundary_distance_forensic_v1",
        "render_source_git_sha": args.analysis_source_sha,
        "secant_replay_source_git_sha": args.secant_replay_source_sha,
        "classification": "SBPF2_FORMULA_SINGULARITY_SUPPORTED",
        "classification_scope": (
            "corrected v2 scalar/parameter gradients agree with finite differences; reciprocal secant "
            "geometry has a heavy real-checkpoint gradient tail and historical v2 training failed on both dev seeds"
        ),
        "historical_sbdd_status": historical["sbdd_status"],
        "v1_vs_v2": {
            "v1": "Student qS detached in historical v1 contract; superseded.",
            "v2": "student_parameter_graph_v2; qS preserved, Teacher terms detached, full-batch reduction.",
        },
        "source_to_equation_audit": {
            "class_pair": "Student-selected strongest non-true adversarial rival, detached and reused for Teacher.",
            "views": "clean and KL-PGD10 adversarial training views in the no-update forensic proxy.",
            "student_terms": "mS_clean, mS_adv, qS, dS; qS remains on the parameter graph in v2.",
            "teacher_terms": "mT_clean, mT_adv, qT, dT; all detached/frozen.",
            "epsilon": "1e-12 in q and distance denominators.",
            "gates": "fixed e99 mask, positive Teacher adversarial margin, nonzero rho; detached.",
            "reduction": "full rank-local batch mean; no selected-count normalization.",
            "coefficient": "applied once by the historical v2 runtime; forensic reports coefficient 1 and frozen coefficient separately.",
        },
        "execution_path_rebase": {
            "description": "Only absolute Teacher checkpoint paths were rebased for host-local read-only execution; Teacher SHA-256 was asserted.",
            "teacher_sha256": "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983",
            "dev-1": {
                "canonical_config_sha256": "e87905b549f741de04576bc36c00c51c0ac464f832dd89d7b5c855349926c5a6",
                "execution_copy_sha256": "ae621c5a62ef9b3b633586bd8b90d48a13757fa288ac42fee651e9298109d96b",
            },
            "dev-2": {
                "canonical_config_sha256": "1ce45685583518a66872b92f40a23c62129b85c6a07c01f50c134b36481dcd5c",
                "execution_copy_sha256": "49e83880c3b34b92c8bc8cd92203d3a7314f949429b91a776ddf82a11ff74265",
            },
        },
        "formula": secant["dev-1"]["formula"],
        "epsilon": secant["dev-1"]["epsilon"],
        "seeds": secant,
        "finite_difference_summary": {seed: fd_summary(payload) for seed, payload in secant.items()},
        "historical_v2_calibration": historical["sbdd_calibration"],
        "historical_v2_numerics": historical["sbdd_numerics"],
        "causal_utility_evaluable_without_secant": True,
        "e114_held_out": historical["held_out"],
        "source_artifacts": {
            "longitudinal": {"path": str(args.longitudinal), "sha256": sha256(args.longitudinal)},
            "historical_results": {"path": str(args.historical_results), "sha256": sha256(args.historical_results)},
            "secant_dev1": {"path": str(args.secant_dev1), "sha256": sha256(args.secant_dev1)},
            "secant_dev2": {"path": str(args.secant_dev2), "sha256": sha256(args.secant_dev2)},
        },
    }
    args.secant_output.parent.mkdir(parents=True, exist_ok=True)
    args.secant_output.write_text(json.dumps(secant_output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.secant_output.with_name(args.secant_output.name + ".sha256").write_text(
        sha256(args.secant_output) + "\n", encoding="utf-8"
    )
    secant_sha = sha256(args.secant_output)
    longitudinal_sha = sha256(args.longitudinal)
    args.longitudinal_report.write_text(
        longitudinal_markdown(longitudinal, artifact_path=args.longitudinal, artifact_sha=longitudinal_sha),
        encoding="utf-8",
    )
    args.secant_report.write_text(
        secant_markdown(secant, historical, artifact_path=args.secant_output, artifact_sha=secant_sha),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "longitudinal_report": str(args.longitudinal_report),
                "secant_report": str(args.secant_report),
                "secant_output": str(args.secant_output),
                "secant_sha256": secant_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
