"""Pre-treatment Teacher reliability proxy and Clean-Wrong safety analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.ert_clean_wrong_broad_screen import fixed_clean_wrong_mask
from ard.analysis.ert_clean_wrong_subtypes import (
    ARMS,
    CleanWrongSubtypeError,
    _effect,
    _probability_stats,
    _read_rows,
    _sha256,
)
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.tracking.adapter import collect_git_state


class ReliabilityProxyError(CleanWrongSubtypeError):
    """Raised when proxy replay or safety aggregation violates its contract."""


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denom = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denom if denom else None


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average
        cursor = end
    return ranks


def _pearson_spearman(x: list[float], y: list[float]) -> dict[str, float | None]:
    return {"pearson": _corr(x, y), "spearman": _corr(_rank(x), _rank(y))}


def _validate_kl_config(config: Any) -> Any:
    attack = config.method.attack
    if (
        attack.loss != "kl"
        or attack.kl_target != "teacher_clean"
        or attack.steps != 10
        or attack.epsilon != "8/255"
        or attack.step_size != "2/255"
        or not attack.random_start
    ):
        raise ReliabilityProxyError("KL replay requires the exact teacher-clean PGD10 training attack")
    return attack


def replay_kl_features(
    *,
    config_path: Path,
    checkpoint: Path,
    mask_path: Path,
    output_dir: Path,
    device: torch.device,
    expected_epoch: int = 79,
) -> dict[str, Any]:
    """Replay KL-PGD10 on the exact epoch-79 parent and retain CW IDs."""
    config = load_config(config_path)
    attack_config = _validate_kl_config(config)
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise ReliabilityProxyError("KL replay requires a clean source tree")
    mask = fixed_clean_wrong_mask(mask_path, run=config.seeds.model_init.__str__())
    selected = {int(item) for item in mask["selected_ids"]}
    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, student)
    if payload.get("epoch") != expected_epoch or payload.get("epoch_boundary") != "end":
        raise ReliabilityProxyError(f"KL replay requires the epoch-{expected_epoch} end checkpoint")
    if config.teacher is None:
        raise ReliabilityProxyError("KL replay requires a registered teacher")
    teacher = build_teacher(config.teacher, tier=config.tier)
    device = torch.device(device)
    student.to(device).eval()
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_rank_batch_size,
        sampler=EpochShuffleSampler(len(train_dataset), seed=0, rank=0, world_size=1, shuffle=False),
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
    )
    attack = LinfPGD(attack_config)
    checkpoint_global_step = payload.get("global_step")
    if not isinstance(checkpoint_global_step, int):
        raise ReliabilityProxyError("epoch-79 checkpoint lacks an integer global_step")
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(device)
        with torch.no_grad():
            student_clean_logits = student(batch.images.float())
            teacher_clean_logits = teacher(batch.images.float())
        generator = torch.Generator(device=device).manual_seed(
            config.seeds.model_init + 1_000_003 * (checkpoint_global_step + batch_index)
        )
        result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits.detach().float(),
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, result.max_abs_delta)
        with torch.no_grad():
            student_adv_logits = student(result.adversarial.float())
            teacher_adv_logits = teacher(result.adversarial.float())
        student_clean = _probability_stats(student_clean_logits, batch.labels)
        student_adv = _probability_stats(student_adv_logits, batch.labels)
        teacher_clean = _probability_stats(teacher_clean_logits, batch.labels)
        teacher_adv = _probability_stats(teacher_adv_logits, batch.labels)
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise ReliabilityProxyError("KL replay populated teacher parameter gradients")
        for idx, sample_id in enumerate(batch.sample_ids.tolist()):
            if int(sample_id) not in selected:
                continue
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "true_label": int(batch.labels[idx]),
                    "student_clean_correct": bool(student_clean["correct"][idx]),
                    "student_adv_correct": bool(student_adv["correct"][idx]),
                    "student_clean_margin": float(student_clean["margin"][idx]),
                    "student_adv_margin": float(student_adv["margin"][idx]),
                    "student_clean_true_probability": float(student_clean["true_probability"][idx]),
                    "student_adv_true_probability": float(student_adv["true_probability"][idx]),
                    "teacher_clean_correct": bool(teacher_clean["correct"][idx]),
                    "teacher_adv_correct": bool(teacher_adv["correct"][idx]),
                    "teacher_clean_margin": float(teacher_clean["margin"][idx]),
                    "teacher_adv_margin": float(teacher_adv["margin"][idx]),
                    "teacher_adv_true_probability": float(teacher_adv["true_probability"][idx]),
                    "teacher_clean_true_probability": float(teacher_clean["true_probability"][idx]),
                }
            )
    if len(rows) != len(selected) or {row["sample_id"] for row in rows} != selected:
        raise ReliabilityProxyError("KL replay did not recover the exact sparse Clean-Wrong ID set")
    if max_abs_delta > float(attack_config.epsilon_value) + 1e-7:
        raise ReliabilityProxyError("KL replay exceeded its pixel-space Linf bound")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "clean-wrong-kl10-feature-stats.parquet"
    write_sample_parquet(rows, rows_path)
    result = {
        "schema_version": 1,
        "contract": "ert_clean_wrong_c0_kl_pgd10_features_v1",
        "feature_epoch": expected_epoch,
        "run": mask["run"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": expected_epoch,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_global_step": checkpoint_global_step,
        "mask_path": mask["mask_path"],
        "mask_sha256": mask["mask_sha256"],
        "selected_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": _sha256(rows_path),
        "source_git_sha": source["sha"],
        "attack": attack_config.identity(),
        "attack_identity_sha256": attack_config.identity_sha256(),
        "attack_seed_protocol": "model_init_seed + 1000003*(checkpoint_global_step + batch_index)",
        "max_abs_delta": max_abs_delta,
        "full_train_order_replayed": True,
    }
    (output_dir / "clean-wrong-kl10-feature-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_meta(root: Path, filename: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    meta = json.loads((root / filename).read_text(encoding="utf-8"))
    rows_name = "clean-wrong-kl10-feature-stats.parquet" if "kl10" in filename else "clean-wrong-feature-stats.parquet"
    return meta, _read_rows(root / rows_name)


def _check_lineage(root: Path, meta: Mapping[str, Any], endpoint_root: Path, run: str) -> None:
    if meta.get("feature_epoch") != 79:
        raise ReliabilityProxyError(f"{run} reliability feature is not epoch 79")
    manifest = json.loads((endpoint_root / run / "C0" / "run-bundle" / "manifest.json").read_text())
    fork = manifest.get("fork_lineage", {})
    if meta.get("checkpoint_sha256") != fork.get("parent_checkpoint_sha256"):
        raise ReliabilityProxyError(f"{run} reliability feature checkpoint does not match endpoint fork parent")
    if not meta.get("full_train_order_replayed"):
        raise ReliabilityProxyError(f"{run} reliability replay did not use full train ordering")


def _validate_endpoint_rows(
    root: Path, run: str, feature_meta: Mapping[str, Any]
) -> dict[str, dict[int, dict[str, Any]]]:
    endpoints: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        path = root / run / arm / "endpoint" / "train" / "endpoint-sample-stats.parquet"
        meta_path = path.with_name("endpoint.json")
        meta = json.loads(meta_path.read_text())
        if meta.get("attack_identity_sha256") != feature_meta.get("attack_identity_sha256"):
            raise ReliabilityProxyError(f"{run} endpoint and CE feature attack identities differ")
        endpoints[arm] = _read_rows(path)
    return endpoints


def _quantile_bins(ids: list[int], values: Mapping[int, float], bins: int = 5) -> list[list[int]]:
    ordered = sorted(ids, key=lambda item: (float(values[item]), item))
    return [
        ordered[start : (len(ordered) * (index + 1)) // bins]
        for index, start in enumerate([(len(ordered) * index) // bins for index in range(bins)])
    ]


def _effect_record(
    base: Mapping[int, Mapping[str, Any]], treatment: Mapping[int, Mapping[str, Any]], ids: list[int]
) -> dict[str, Any]:
    selected_base = [base[item] for item in ids]
    selected_treatment = [treatment[item] for item in ids]
    clean = _effect(selected_base, selected_treatment, "clean_probability_margin")
    robust = _effect(selected_base, selected_treatment, "adversarial_probability_margin")
    return {"n": len(ids), "clean": clean, "robust": robust}


def build_proxy_report(
    *,
    endpoint_root: Path,
    ce_feature_roots: dict[str, Path],
    kl_feature_roots: dict[str, Path],
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_clean_wrong_reliability_proxy_safety_v1",
        "endpoint_epoch": 84,
        "pre_treatment_epoch": 79,
        "no_training": True,
        "reliability_rule": "CW-R iff teacher adversarial probability margin > 0; CW-U otherwise",
        "metric_semantics": {
            "accuracy_delta": "rescue_rate - harm_rate",
            "margin_delta": "mean(treatment_probability_margin - control_probability_margin)",
        },
        "runs": {},
    }
    for run in ("L2", "L4"):
        ce_meta, ce_rows = _load_meta(ce_feature_roots[run], "clean-wrong-feature-replay.json")
        kl_meta, kl_rows = _load_meta(kl_feature_roots[run], "clean-wrong-kl10-feature-replay.json")
        _check_lineage(ce_feature_roots[run], ce_meta, endpoint_root, run)
        _check_lineage(kl_feature_roots[run], kl_meta, endpoint_root, run)
        if set(ce_rows) != set(kl_rows):
            raise ReliabilityProxyError(f"{run} CE/KL feature stable-ID sets differ")
        endpoints = _validate_endpoint_rows(endpoint_root, run, ce_meta)
        selected = sorted(ce_rows)
        teacher_ce = [float(ce_rows[item]["teacher_adv_margin"]) for item in selected]
        teacher_kl = [float(kl_rows[item]["teacher_adv_margin"]) for item in selected]
        teacher_ce_correct = [bool(ce_rows[item]["teacher_adv_correct"]) for item in selected]
        teacher_kl_correct = [bool(kl_rows[item]["teacher_adv_correct"]) for item in selected]
        agreement = {
            **_pearson_spearman(teacher_kl, teacher_ce),
            "sign_agreement": sum((a > 0) == (b > 0) for a, b in zip(teacher_kl, teacher_ce, strict=True))
            / len(selected),
            "correctness_agreement": sum(a == b for a, b in zip(teacher_kl_correct, teacher_ce_correct, strict=True))
            / len(selected),
            "confusion": {
                "KL-R_to_CE-R": sum(
                    a and b for a, b in zip([x > 0 for x in teacher_kl], [x > 0 for x in teacher_ce], strict=True)
                ),
                "KL-R_to_CE-U": sum(
                    a and not b for a, b in zip([x > 0 for x in teacher_kl], [x > 0 for x in teacher_ce], strict=True)
                ),
                "KL-U_to_CE-R": sum(
                    (not a) and b for a, b in zip([x > 0 for x in teacher_kl], [x > 0 for x in teacher_ce], strict=True)
                ),
                "KL-U_to_CE-U": sum(
                    (not a) and (not b)
                    for a, b in zip([x > 0 for x in teacher_kl], [x > 0 for x in teacher_ce], strict=True)
                ),
            },
        }
        values = {item: float(ce_rows[item]["teacher_adv_margin"]) for item in selected}
        kl_values = {item: float(kl_rows[item]["teacher_adv_margin"]) for item in selected}
        ce_bins = _quantile_bins(selected, values)
        kl_bins = _quantile_bins(selected, kl_values)
        quantile_effects: dict[str, Any] = {}
        for label, bins in (("CE20", ce_bins), ("KL10", kl_bins)):
            quantile_effects[label] = {
                f"Q{index + 1}": {
                    "value_min": min(values[item] if label == "CE20" else kl_values[item] for item in ids),
                    "value_max": max(values[item] if label == "CE20" else kl_values[item] for item in ids),
                    "effects": {
                        arm: _effect_record(endpoints["C0"], endpoints[arm], ids) for arm in ("C10", "C12", "C13")
                    },
                }
                for index, ids in enumerate(bins)
            }
        med_student = sorted(float(ce_rows[item]["student_adv_margin"]) for item in selected)[len(selected) // 2]
        med_teacher = sorted(teacher_ce)[len(selected) // 2]
        two_by_two: dict[str, Any] = {}
        for t_name, t_pred in (
            ("teacher_high", lambda item: values[item] >= med_teacher),
            ("teacher_low", lambda item: values[item] < med_teacher),
        ):
            for s_name, s_pred in (
                ("student_high", lambda item: float(ce_rows[item]["student_adv_margin"]) >= med_student),
                ("student_low", lambda item: float(ce_rows[item]["student_adv_margin"]) < med_student),
            ):
                ids = [item for item in selected if t_pred(item) and s_pred(item)]
                two_by_two[f"{t_name}__{s_name}"] = {
                    "n": len(ids),
                    "C10": _effect_record(endpoints["C0"], endpoints["C10"], ids),
                }
        machine["runs"][run] = {
            "selected_count": len(selected),
            "ce_feature_meta": ce_meta,
            "kl_feature_meta": kl_meta,
            "endpoint_attack_identity_sha256": ce_meta.get("attack_identity_sha256"),
            "proxy_agreement": agreement,
            "quantile_effects": quantile_effects,
            "student_teacher_median_split": {
                "teacher_ce_margin_median": med_teacher,
                "student_adv_margin_median": med_student,
                "effects": two_by_two,
            },
        }
    machine["source_sha256"] = hashlib.sha256(json.dumps(machine, sort_keys=True).encode()).hexdigest()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Clean-Wrong Teacher Reliability — Online Proxy & Safety Analysis",
        "",
        (
            "Pre-treatment epoch-79 reliability replay versus C0/C10/C12/C13 "
            "epoch-84 endpoints. No training, tuning, or route selection."
        ),
        "",
        "## Metric semantics audit",
        "",
        (
            "`accuracy_delta = rescue_rate - harm_rate`; `margin_delta` is the "
            "mean paired probability-margin change. They are stored as separate "
            "fields in the JSON and tested independently."
        ),
        "",
        "## KL10 versus CE20 agreement",
        "",
        (
            "| run | Pearson | Spearman | sign agreement | correctness agreement | "
            "KL-R→CE-R | KL-R→CE-U | KL-U→CE-R | KL-U→CE-U |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run, report in machine["runs"].items():
        a = report["proxy_agreement"]
        c = a["confusion"]
        lines.append(
            f"| {run} | {a['pearson']:.4f} | {a['spearman']:.4f} | "
            f"{a['sign_agreement']:.3f} | {a['correctness_agreement']:.3f} | "
            f"{c['KL-R_to_CE-R']} | {c['KL-R_to_CE-U']} | "
            f"{c['KL-U_to_CE-R']} | {c['KL-U_to_CE-U']} |"
        )
    lines += [
        "",
        "## C10 CE20 quintile safety effects",
        "",
        (
            "| run | bin | n | mT range | robust accuracy Δ | robust rescue | "
            "robust harm | robust net rescue | clean accuracy Δ |"
        ),
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run, report in machine["runs"].items():
        for label, value in report["quantile_effects"]["CE20"].items():
            e = value["effects"]["C10"]
            lines.append(
                f"| {run} | {label} | {e['n']} | "
                f"[{value['value_min']:.4f}, {value['value_max']:.4f}] | "
                f"{e['robust']['accuracy_delta']:+.4f} | "
                f"{e['robust']['rescue_rate']:.4f} | "
                f"{e['robust']['harm_rate']:.4f} | "
                f"{e['robust']['net_rescue_rate']:+.4f} | "
                f"{e['clean']['accuracy_delta']:+.4f} |"
            )
    lines += [
        "",
        "## Secondary C12/C13 quintile effects",
        "",
        (
            "The JSON contains clean/robust accuracy, margin, rescue, harm, and "
            "net-rescue fields for C10, C12, and C13 under both CE20 and KL10 "
            "quantile bins. Bins are determined from pre-treatment feature "
            "distributions only."
        ),
        "",
        f"Machine report content hash: `{machine['source_sha256']}`.",
    ]
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n")
    return machine
