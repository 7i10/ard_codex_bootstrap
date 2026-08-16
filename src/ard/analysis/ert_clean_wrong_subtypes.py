"""Read-only sample-wise Clean-Wrong rescue subtype analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.ert_clean_wrong_broad_screen import fixed_clean_wrong_mask
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.tracking.adapter import collect_git_state


class CleanWrongSubtypeError(RuntimeError):
    """Raised when subtype replay or endpoint joins drift from the contract."""


ARMS = ("C0", "C10", "C12", "C13")
GROUPS = (
    "clean_and_robust_rescue",
    "clean_only_rescue",
    "robust_only_rescue",
    "neither_or_harm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=1)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong = probabilities.clone()
    wrong.scatter_(1, labels[:, None], float("-inf"))
    margin = true_probability - wrong.max(dim=1).values
    prediction = probabilities.argmax(dim=1)
    return {
        "true_probability": true_probability,
        "margin": margin,
        "prediction": prediction,
        "correct": prediction.eq(labels),
    }


def replay_features(
    *,
    config_path: Path,
    checkpoint: Path,
    mask_path: Path,
    output_dir: Path,
    device: torch.device,
    expected_epoch: int = 84,
) -> dict[str, Any]:
    """Replay the full train ordering and retain only the registered CW IDs."""
    config = load_config(config_path)
    attack_config = config.method.selection_attack
    if attack_config is None or attack_config.loss != "ce" or attack_config.steps != 20:
        raise CleanWrongSubtypeError("feature replay requires the CE-PGD20 endpoint attack")
    if attack_config.epsilon != "8/255" or attack_config.step_size != "2/255" or not attack_config.random_start:
        raise CleanWrongSubtypeError("feature replay requires the frozen 8/255, 2/255 random-start attack")
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise CleanWrongSubtypeError("feature replay requires a clean source tree")
    mask = fixed_clean_wrong_mask(mask_path, run=config.seeds.model_init.__str__())
    selected = set(int(item) for item in mask["selected_ids"])
    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, student)
    if payload.get("epoch") != expected_epoch or payload.get("epoch_boundary") != "end":
        raise CleanWrongSubtypeError("feature replay requires the epoch-84 end checkpoint")
    teacher_cfg = config.teacher
    if teacher_cfg is None:
        raise CleanWrongSubtypeError("feature replay requires a teacher")
    teacher = build_teacher(teacher_cfg, tier=config.tier)
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
    generator = torch.Generator(device=device).manual_seed(config.seeds.evaluation_attack)
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            student_clean_logits = student(batch.images)
            teacher_clean_logits = teacher(batch.images)
        adversarial = attack.generate(
            AttackRequest(inputs=batch.images, labels=batch.labels, student=student, teacher=None, generator=generator)
        ).adversarial
        with torch.no_grad():
            student_adv_logits = student(adversarial)
            teacher_adv_logits = teacher(adversarial)
        student_clean = _probability_stats(student_clean_logits, batch.labels)
        student_adv = _probability_stats(student_adv_logits, batch.labels)
        teacher_clean = _probability_stats(teacher_clean_logits, batch.labels)
        teacher_adv = _probability_stats(teacher_adv_logits, batch.labels)
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
                    "teacher_clean_true_probability": float(teacher_clean["true_probability"][idx]),
                    "teacher_adv_true_probability": float(teacher_adv["true_probability"][idx]),
                    "teacher_clean_prediction": int(teacher_clean["prediction"][idx]),
                    "teacher_adv_prediction": int(teacher_adv["prediction"][idx]),
                    "delta_teacher_margin": float(teacher_clean["margin"][idx] - teacher_adv["margin"][idx]),
                }
            )
    if len(rows) != len(selected) or {row["sample_id"] for row in rows} != selected:
        raise CleanWrongSubtypeError("feature replay did not recover the exact sparse Clean-Wrong ID set")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "clean-wrong-feature-stats.parquet"
    write_sample_parquet(rows, rows_path)
    result = {
        "schema_version": 1,
        "contract": "ert_clean_wrong_epoch84_c0_ce_pgd20_features_v1",
        "run": mask["run"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": expected_epoch,
        "checkpoint_sha256": _sha256(checkpoint),
        "mask_path": mask["mask_path"],
        "mask_sha256": mask["mask_sha256"],
        "selected_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": _sha256(rows_path),
        "source_git_sha": source["sha"],
        "attack": attack_config.identity(),
        "attack_identity_sha256": attack_config.identity_sha256(),
        "full_train_order_replayed": True,
    }
    (output_dir / "clean-wrong-feature-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _read_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise CleanWrongSubtypeError(f"duplicate sample IDs in {path}")
    return result


def _group(base: Mapping[str, Any], treatment: Mapping[str, Any]) -> str:
    clean = not bool(base["clean_correct"]) and bool(treatment["clean_correct"])
    robust = not bool(base["robust_correct"]) and bool(treatment["robust_correct"])
    if clean and robust:
        return "clean_and_robust_rescue"
    if clean:
        return "clean_only_rescue"
    if robust:
        return "robust_only_rescue"
    return "neither_or_harm"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    numeric = (
        "student_clean_margin",
        "student_adv_margin",
        "student_clean_true_probability",
        "student_adv_true_probability",
        "teacher_clean_margin",
        "teacher_adv_margin",
        "teacher_clean_true_probability",
        "teacher_adv_true_probability",
        "delta_teacher_margin",
    )
    result: dict[str, Any] = {"n": len(rows)}
    for name in numeric:
        values = sorted(float(row[name]) for row in rows)
        result[name] = {
            "mean": sum(values) / len(values),
            "median": values[len(values) // 2],
            "q25": values[int((len(values) - 1) * 0.25)],
            "q75": values[int((len(values) - 1) * 0.75)],
        }
    for name in (
        "teacher_clean_correct",
        "teacher_adv_correct",
        "student_clean_correct",
        "student_adv_correct",
    ):
        result[name + "_rate"] = sum(bool(row[name]) for row in rows) / len(rows)
    result["class_counts"] = {
        str(label): sum(int(row["true_label"]) == label for row in rows)
        for label in sorted({int(row["true_label"]) for row in rows})
    }
    return result


def build_report(
    *,
    root: Path,
    feature_roots: dict[str, Path],
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_clean_wrong_rescue_subtypes_v1",
        "endpoint_epoch": 84,
        "arms": list(ARMS),
        "runs": {},
        "no_training": True,
    }
    for run, feature_root in feature_roots.items():
        feature_meta = json.loads((feature_root / "clean-wrong-feature-replay.json").read_text(encoding="utf-8"))
        if feature_meta.get("contract") != "ert_clean_wrong_epoch84_c0_ce_pgd20_features_v1":
            raise CleanWrongSubtypeError("feature replay contract mismatch")
        feature_rows = _read_rows(feature_root / "clean-wrong-feature-stats.parquet")
        selected = set(feature_rows)
        endpoints: dict[str, dict[int, dict[str, Any]]] = {}
        endpoint_meta: dict[str, Any] = {}
        for arm in ARMS:
            path = root / run / arm / "endpoint" / "train" / "endpoint-sample-stats.parquet"
            meta_path = path.with_name("endpoint.json")
            if not path.is_file() or not meta_path.is_file():
                raise CleanWrongSubtypeError(f"missing train endpoint: {path}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("attack_identity_sha256") != feature_meta.get("attack_identity_sha256"):
                raise CleanWrongSubtypeError("endpoint and feature attack identities differ")
            rows = _read_rows(path)
            if not selected.issubset(rows):
                raise CleanWrongSubtypeError("endpoint is missing a registered Clean-Wrong ID")
            endpoints[arm] = {sample_id: rows[sample_id] for sample_id in selected}
            endpoint_meta[arm] = {
                "endpoint_path": str(meta_path.resolve()),
                "endpoint_sha256": _sha256(meta_path),
                "rows_path": str(path.resolve()),
                "rows_sha256": meta.get("rows_sha256"),
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
                "source_git_sha": meta.get("source_git_sha"),
                "attack_identity_sha256": meta.get("attack_identity_sha256"),
            }
        base = endpoints["C0"]
        treatment_reports: dict[str, Any] = {}
        for arm in ("C10", "C12", "C13"):
            group_ids = {group: [] for group in GROUPS}
            for sample_id in sorted(selected):
                group_ids[_group(base[sample_id], endpoints[arm][sample_id])].append(sample_id)
            groups = {}
            for group, ids in group_ids.items():
                groups[group] = _summary([feature_rows[item] for item in ids])
                groups[group]["sample_ids_sha256"] = hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest()
            treatment_reports[arm] = {"groups": groups, "group_ids": group_ids}
        overlaps: dict[str, Any] = {}
        for dimension, key in (("clean", "clean_only_rescue"), ("robust", "robust_only_rescue")):
            sets = {}
            for arm in ("C10", "C13"):
                ids = set(treatment_reports[arm]["group_ids"][key]) | set(
                    treatment_reports[arm]["group_ids"]["clean_and_robust_rescue"]
                )
                sets[arm] = ids
            intersection = sets["C10"] & sets["C13"]
            union = sets["C10"] | sets["C13"]
            overlaps[dimension] = {
                "C10_count": len(sets["C10"]),
                "C13_count": len(sets["C13"]),
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard": len(intersection) / len(union) if union else None,
            }
        machine["runs"][run] = {
            "feature_meta": feature_meta,
            "feature_rows_sha256": _sha256(feature_root / "clean-wrong-feature-stats.parquet"),
            "endpoint_meta": endpoint_meta,
            "selected_count": len(selected),
            "treatments": treatment_reports,
            "c10_c13_rescue_overlap": overlaps,
        }
    machine["source_sha256"] = hashlib.sha256(json.dumps(machine, sort_keys=True).encode()).hexdigest()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong Rescue Subtype Analysis",
        "",
        "Read-only C0/C10/C12/C13 epoch-84 endpoint transition analysis. No new training or route selection.",
        "",
    ]
    for run, report in machine["runs"].items():
        lines += [f"## {run}", "", f"Fixed Clean-Wrong cohort: {report['selected_count']} samples.", ""]
        lines += [
            "| arm | group | n | teacher clean correct | teacher adv correct | "
            "student clean p mean | teacher adv p mean | ΔT mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ("C10", "C12", "C13"):
            for group in GROUPS:
                row = report["treatments"][arm]["groups"][group]
                if not row["n"]:
                    lines.append(f"| {arm} | {group} | 0 | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| {arm} | {group} | {row['n']} | {row['teacher_clean_correct_rate']:.3f} | "
                    f"{row['teacher_adv_correct_rate']:.3f} | {row['student_clean_true_probability']['mean']:.4f} | "
                    f"{row['teacher_adv_true_probability']['mean']:.4f} | {row['delta_teacher_margin']['mean']:.4f} |"
                )
        lines += ["", "### C10/C13 rescue overlap", ""]
        for dimension, value in report["c10_c13_rescue_overlap"].items():
            lines.append(
                f"- {dimension}: C10={value['C10_count']}, C13={value['C13_count']}, "
                f"intersection={value['intersection_count']}, union={value['union_count']}, "
                f"Jaccard={value['jaccard'] if value['jaccard'] is not None else '—'}"
            )
        lines.append("")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    return machine
