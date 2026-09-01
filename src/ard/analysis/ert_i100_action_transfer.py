"""I100 epoch-99 fixed-action transfer replay and no-update calibration.

This module is intentionally read-only with respect to training state.  It
replays the exact e99 Student/Teacher pair, emits stable-ID observations for
the two registered attacks, builds fixed masks, and calibrates only the
intervention gradient scales used by the subsequent continuation screen.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_dataset, collate_indexed, stratified_train_validation_split
from ard.models import build_student, build_teacher
from ard.objectives import ObjectiveTerms, RSLADObjective


CONTRACT = "ert_rslad_i100_action_transfer_v1"
PARENT_EPOCH = 99
EXPECTED_PARENT_SHA = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}


class TransferReplayError(RuntimeError):
    """Raised when a fixed e99 replay contract cannot be proven."""


def _prepare_output_dir(output_dir: Path) -> None:
    """Allow the orchestrator's pre-created directory, but never overwrite results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise TransferReplayError(f"replay output path is not a directory: {output_dir}")
    allowed = {"orchestration"}
    unexpected = {item.name for item in output_dir.iterdir() if item.name not in allowed}
    if unexpected:
        raise TransferReplayError(f"replay output is not empty: {sorted(unexpected)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def git_sha(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _deterministic_backend() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_parent(config_path: Path, checkpoint_path: Path, expected_sha: str, device: torch.device):
    if sha256(checkpoint_path) != expected_sha:
        raise TransferReplayError("parent checkpoint SHA-256 mismatch")
    config = load_config(config_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != PARENT_EPOCH or payload.get("epoch_boundary") != "end":
        raise TransferReplayError("parent must be an epoch-99 end-boundary checkpoint")
    if not isinstance(payload.get("config_hash"), str):
        raise TransferReplayError("parent has no config hash")
    student = build_student(config.student, tier=config.tier)
    student.load_state_dict(payload["model"], strict=True)
    student = student.to(device).eval()
    if config.teacher is None:
        raise TransferReplayError("I100 transfer requires a Teacher")
    teacher = build_teacher(config.teacher, tier=config.tier).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return config, payload, student, teacher


def _loader(config: Any, batch_size: int, ids: set[int] | None = None) -> DataLoader:
    raw = build_dataset(config.dataset)
    train, _ = stratified_train_validation_split(
        raw, validation_fraction=config.training.validation_fraction, seed=config.seeds.split
    )
    if ids is None:
        dataset = train
    else:
        positions = [pos for pos, source_id in enumerate(train.indices) if int(source_id) in ids]
        if len(positions) != len(ids):
            raise TransferReplayError("calibration IDs are not all in the fixed training split")
        dataset = Subset(train, positions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_indexed)


def _primitives(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    logits = logits.detach().float()
    probabilities = F.softmax(logits, dim=1)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong = probabilities.scatter(1, labels[:, None], 0.0).amax(dim=1)
    margin = true_probability - wrong
    return {
        "correct": logits.argmax(dim=1).eq(labels),
        "probability": true_probability,
        "margin": margin,
    }


def _attack_config(config: Any, name: str):
    source = config.method.selection_attack if name == "ce20" else config.method.attack
    if source is None:
        raise TransferReplayError(f"missing {name} attack")
    if name == "ce20" and (source.loss != "ce" or source.steps != 20):
        raise TransferReplayError("strong pilot observation requires CE-PGD20")
    if name == "kl10" and (source.loss != "kl" or source.steps != 10 or source.kl_target != "teacher_clean"):
        raise TransferReplayError("training observation requires Teacher-clean KL-PGD10")
    return source.model_copy(update={"random_start_keying": "sample_keyed_v1"})


def _replay_rows(
    config: Any,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    ce_attack_config = _attack_config(config, "ce20")
    kl_attack_config = _attack_config(config, "kl10")
    ce_attack, kl_attack = LinfPGD(ce_attack_config), LinfPGD(kl_attack_config)
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            s_clean = student(batch.images.float()).detach().float()
            t_clean = teacher(batch.images.float()).detach().float()
        common = {
            "inputs": batch.images,
            "labels": batch.labels,
            "student": student,
            "teacher": teacher,
            "source_ids": batch.sample_ids,
            "epoch": PARENT_EPOCH,
            "generator": torch.Generator(device=device),
        }
        ce_result = ce_attack.generate(
            AttackRequest(**common, attack_seed=int(config.seeds.evaluation_attack), stream_tag="selection_pgd")
        )
        kl_result = kl_attack.generate(
            AttackRequest(
                **common,
                target_logits=t_clean,
                attack_seed=int(config.seeds.train_attack),
                stream_tag="train_pgd",
            )
        )
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            s_ce = student(ce_result.adversarial.float()).detach().float()
            s_kl = student(kl_result.adversarial.float()).detach().float()
            t_ce = teacher(ce_result.adversarial.float()).detach().float()
            t_kl = teacher(kl_result.adversarial.float()).detach().float()
        ps_clean, ps_ce, ps_kl = _primitives(s_clean, batch.labels), _primitives(s_ce, batch.labels), _primitives(s_kl, batch.labels)
        pt_clean, pt_ce, pt_kl = _primitives(t_clean, batch.labels), _primitives(t_ce, batch.labels), _primitives(t_kl, batch.labels)
        for i, (sid, label) in enumerate(zip(batch.sample_ids.tolist(), batch.labels.tolist(), strict=True)):
            rows.append(
                {
                    "sample_id": int(sid),
                    "class_id": int(label),
                    "epoch": PARENT_EPOCH,
                    "student_clean_correct": bool(ps_clean["correct"][i]),
                    "student_clean_probability": float(ps_clean["probability"][i]),
                    "student_clean_margin": float(ps_clean["margin"][i]),
                    "student_ce20_adv_correct": bool(ps_ce["correct"][i]),
                    "student_ce20_adv_probability": float(ps_ce["probability"][i]),
                    "student_ce20_adv_margin": float(ps_ce["margin"][i]),
                    "student_kl10_adv_correct": bool(ps_kl["correct"][i]),
                    "student_kl10_adv_probability": float(ps_kl["probability"][i]),
                    "student_kl10_adv_margin": float(ps_kl["margin"][i]),
                    "teacher_clean_correct": bool(pt_clean["correct"][i]),
                    "teacher_clean_probability": float(pt_clean["probability"][i]),
                    "teacher_clean_margin": float(pt_clean["margin"][i]),
                    "teacher_ce20_adv_correct": bool(pt_ce["correct"][i]),
                    "teacher_ce20_adv_probability": float(pt_ce["probability"][i]),
                    "teacher_ce20_adv_margin": float(pt_ce["margin"][i]),
                    "teacher_kl10_adv_correct": bool(pt_kl["correct"][i]),
                    "teacher_kl10_adv_probability": float(pt_kl["probability"][i]),
                    "teacher_kl10_adv_margin": float(pt_kl["margin"][i]),
                }
            )
    if len(rows) != len({int(row["sample_id"]) for row in rows}):
        raise TransferReplayError("replay produced duplicate stable IDs")
    return rows


def replay(
    *,
    config_path: Path,
    checkpoint_path: Path,
    expected_sha: str,
    output_dir: Path,
    device: str,
    batch_size: int = 128,
    max_batches: int | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    _deterministic_backend()
    device_obj = torch.device(device)
    config, payload, student, teacher = _load_parent(config_path, checkpoint_path, expected_sha, device_obj)
    rows = _replay_rows(config, student, teacher, _loader(config, batch_size), device_obj, max_batches=max_batches)
    if max_batches is None and len(rows) != 45000:
        raise TransferReplayError(f"expected 45000 train rows, got {len(rows)}")
    _prepare_output_dir(output_dir)
    table = pa.Table.from_pylist(rows)
    row_path = output_dir / "e99-observations.parquet"
    pq.write_table(table, row_path, compression="zstd")
    meta = {
        "schema_version": 1,
        "contract": CONTRACT,
        "role": "pre_treatment_replay",
        "source_git_sha": git_sha(root),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "payload_epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "teacher_checkpoint_sha256": config.teacher.checkpoint_sha256 if config.teacher else None,
        "row_count": len(rows),
        "complete_train_universe": max_batches is None,
        "rows_sha256": sha256(row_path),
        "batch_size": batch_size,
        "device": str(device_obj),
        "attack_identities": {
            "ce20": _attack_config(config, "ce20").identity(),
            "kl10": _attack_config(config, "kl10").identity(),
        },
    }
    (output_dir / "lineage.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "lineage.sha256").write_text(sha256(output_dir / "lineage.json") + "\n", encoding="utf-8")
    return meta


def _mask_digest(ids: list[int]) -> str:
    # Match the runtime's stable-ID digest exactly (no trailing newline).
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def _class_counts(rows: list[dict[str, Any]], ids: list[int]) -> dict[str, int]:
    by_id = {int(row["sample_id"]): row for row in rows}
    return {str(label): sum(int(by_id[sid]["class_id"]) == label for sid in ids) for label in range(10)}


def build_masks(*, replay_dirs: Mapping[str, Path], output_dir: Path) -> dict[str, Any]:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for seed, directory in replay_dirs.items():
        table = pq.read_table(directory / "e99-observations.parquet")
        rows = table.to_pylist()
        if len(rows) != 45000 or len({int(r["sample_id"]) for r in rows}) != 45000:
            raise TransferReplayError(f"invalid replay universe for {seed}")
        all_rows[seed] = rows
    positive: list[tuple[float, str, int]] = []
    for seed, rows in all_rows.items():
        positive.extend((float(row["teacher_ce20_adv_margin"]), seed, int(row["sample_id"])) for row in rows if bool(row["teacher_ce20_adv_correct"]))
    if not positive:
        raise TransferReplayError("Teacher-positive CE20 population is empty")
    positive.sort(key=lambda x: (x[0], x[1], x[2]))
    lower_count = math.ceil(0.10 * len(positive))
    lower = {(seed, sid) for _, seed, sid in positive[:lower_count]}
    q10_threshold = float(positive[lower_count - 1][0])
    outputs: dict[str, Any] = {"schema_version": 1, "contract": CONTRACT, "anchor_epoch": PARENT_EPOCH, "teacher_ce20_positive_q10_threshold": q10_threshold, "masks": {}}
    output_dir.mkdir(parents=True, exist_ok=False)
    for seed, rows in all_rows.items():
        def ids_for(predicate):
            return sorted(int(row["sample_id"]) for row in rows if predicate(row))
        cw = ids_for(lambda r: not bool(r["student_clean_correct"]))
        pilot = ids_for(lambda r: bool(r["student_clean_correct"]) and not bool(r["student_ce20_adv_correct"]))
        pilot_t1 = ids_for(lambda r: bool(r["student_clean_correct"]) and not bool(r["student_ce20_adv_correct"]) and bool(r["teacher_ce20_adv_correct"]) and (seed, int(r["sample_id"])) not in lower)
        masks = {}
        for name, ids in (("clean_wrong", cw), ("pilot_s3", pilot), ("pilot_s3_t1", pilot_t1)):
            masks[name] = {"selected_count": len(ids), "selected_ids": ids, "selected_ids_sha256": _mask_digest(ids), "selected_class_counts": _class_counts(rows, ids)}
        bundle = {
            "schema_version": 1,
            "contract": CONTRACT,
            "seed": seed,
            "anchor_epoch": PARENT_EPOCH,
            "replay_lineage_sha256": sha256(replay_dirs[seed] / "lineage.json"),
            "teacher_ce20_positive_q10_threshold": q10_threshold,
            "masks": masks,
            "definitions": {
                "pilot_s3": "student_clean_correct and student_ce20_adv_correct=false",
                "teacher_t1": "teacher_ce20_adv_correct and outside lowest positive CE20 margin q10",
                "pilot_s3_t1": "pilot_s3 and teacher_t1",
                "clean_wrong": "student_clean_correct=false; not canonical S2 relabeling",
            },
        }
        path = output_dir / f"{seed}.json"
        path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs["masks"][seed] = {"path": str(path.resolve()), "sha256": sha256(path), "counts": {name: value["selected_count"] for name, value in masks.items()}}
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["summary_path"] = str(summary_path.resolve())
    outputs["summary_sha256"] = sha256(summary_path)
    return outputs


def _norm(model: torch.nn.Module, value: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    value.mean().backward(retain_graph=True)
    total = torch.zeros((), device=value.device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt().item())


def _class_stratified_ids(rows: list[dict[str, Any]], ids: set[int], limit: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = {label: [] for label in range(10)}
    by_id = {int(row["sample_id"]): row for row in rows}
    for sid in sorted(ids):
        by_class[int(by_id[sid]["class_id"])].append(sid)
    rng = random.Random(seed)
    for values in by_class.values():
        rng.shuffle(values)
    selected: list[int] = []
    while len(selected) < min(limit, len(ids)):
        progressed = False
        for label in range(10):
            if by_class[label]:
                selected.append(by_class[label].pop())
                progressed = True
                if len(selected) >= min(limit, len(ids)):
                    break
        if not progressed:
            break
    return sorted(selected)


def calibrate(*, config_paths: Mapping[str, Path], checkpoint_paths: Mapping[str, Path], replay_dirs: Mapping[str, Path], mask_paths: Mapping[str, Path], output: Path, device: str, batch_size: int = 64, max_per_cohort: int = 256) -> dict[str, Any]:
    _deterministic_backend()
    device_obj = torch.device(device)
    all_rows = {seed: pq.read_table(directory / "e99-observations.parquet").to_pylist() for seed, directory in replay_dirs.items()}
    positive = [float(row["teacher_kl10_adv_margin"]) for rows in all_rows.values() for row in rows if bool(row["teacher_kl10_adv_correct"])]
    if not positive:
        raise TransferReplayError("no positive KL10 Teacher margins for A7 calibration")
    positive.sort()
    q25, q75 = float(np.quantile(np.asarray(positive, dtype=np.float64), 0.25)), float(np.quantile(np.asarray(positive, dtype=np.float64), 0.75))
    measurements: list[dict[str, Any]] = []
    for seed in ("dev-1", "dev-2"):
        config, payload, student, teacher = _load_parent(config_paths[seed], checkpoint_paths[seed], EXPECTED_PARENT_SHA[seed], device_obj)
        mask = json.loads(mask_paths[seed].read_text())
        mask_map = {name: set(int(x) for x in rec["selected_ids"]) for name, rec in mask["masks"].items()}
        pilot_ids = _class_stratified_ids(all_rows[seed], mask_map["pilot_s3_t1"], max_per_cohort, 9901 + len(seed))
        cw_ids = _class_stratified_ids(all_rows[seed], mask_map["clean_wrong"], max_per_cohort, 9907 + len(seed))
        kl_attack = LinfPGD(_attack_config(config, "kl10"))
        objective = RSLADObjective(temperature=config.method.temperature, temperature_squared=config.method.temperature_squared)
        by_id = {int(row["sample_id"]): float(row["teacher_kl10_adv_margin"]) for row in all_rows[seed]}
        # Separate cohort loaders ensure a mixed batch can never be mislabeled.
        for cohort, cohort_ids in (("pilot_s3_t1", pilot_ids), ("clean_wrong", cw_ids)):
            loader = _loader(config, batch_size, set(cohort_ids))
            for batch_index, batch in enumerate(loader):
                batch = batch.to(device_obj)
                with torch.no_grad(), torch.autocast(device_type=device_obj.type, enabled=False):
                    teacher_clean = teacher(batch.images.float()).detach().float()
                attack_result = kl_attack.generate(AttackRequest(inputs=batch.images, labels=batch.labels, student=student, teacher=teacher, target_logits=teacher_clean, source_ids=batch.sample_ids, epoch=PARENT_EPOCH, attack_seed=int(config.seeds.train_attack), stream_tag="train_pgd", generator=torch.Generator(device=device_obj)))
                adv_logits = student(attack_result.adversarial.float())
                clean_logits = student(batch.images.float())
                terms = objective(student_logits=adv_logits, labels=batch.labels, teacher_logits=teacher_clean, clean_student_logits=clean_logits)
                if terms.adversarial_kd is None:
                    raise TransferReplayError("RSLAD AdvKD branch unavailable")
                base = objective.ADVERSARIAL_COEFFICIENT * terms.adversarial_kd
                ce = F.cross_entropy(adv_logits, batch.labels, reduction="none")
                target = torch.tensor([min(max(by_id[int(sid)], q25), q75) for sid in batch.sample_ids.tolist()], device=device_obj)
                zero = torch.zeros_like(base)
                margin_terms = ObjectiveTerms(hard=zero, kd=zero, regularization=zero, adversarial_kd=zero, clean_kd=zero).add_adversarial_margin(adv_logits, batch.labels, target, torch.ones_like(base), coefficient=1.0)
                base_norm, ce_norm, margin_norm = _norm(student, base), _norm(student, ce), _norm(student, margin_terms.hard)
                measurements.append({"seed": seed, "batch": batch_index, "cohort": cohort, "base_advkd_norm": base_norm, "advce_norm": ce_norm, "margin_norm": margin_norm, "advce_ratio_at_beta_1": ce_norm / base_norm, "margin_ratio_at_coeff_1": margin_norm / base_norm})
                student.zero_grad(set_to_none=True)
    base = np.asarray([m["base_advkd_norm"] for m in measurements], dtype=np.float64)
    ce = np.asarray([m["advce_norm"] for m in measurements], dtype=np.float64)
    margin = np.asarray([m["margin_norm"] for m in measurements], dtype=np.float64)
    if not np.isfinite(np.stack([base, ce, margin])).all() or np.any(ce <= 0) or np.any(margin <= 0) or np.any(base <= 0):
        raise TransferReplayError("calibration has non-finite or zero gradient denominator")
    beta = float(0.25 * np.median(base / ce))
    margin_coeff = float(0.25 * np.median(base / margin))
    sample_counts: dict[str, dict[str, int]] = {}
    for seed in all_rows:
        mask_payload = json.loads(mask_paths[seed].read_text(encoding="utf-8"))
        pilot_mask = {int(x) for x in mask_payload["masks"]["pilot_s3_t1"]["selected_ids"]}
        cw_mask = {int(x) for x in mask_payload["masks"]["clean_wrong"]["selected_ids"]}
        sample_counts[seed] = {
            "pilot_s3_t1": len(_class_stratified_ids(all_rows[seed], pilot_mask, max_per_cohort, 9901 + len(seed))),
            "clean_wrong": len(_class_stratified_ids(all_rows[seed], cw_mask, max_per_cohort, 9907 + len(seed))),
        }
    result = {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "complete_no_update",
        "source_git_sha": git_sha(Path(__file__).resolve().parents[3]),
        "parent_epoch": PARENT_EPOCH,
        "calibration_rule": "shared coefficient; target gradient ratio 0.25; pooled dev-1/dev-2; no outcome used",
        "beta_advce": beta,
        "margin_coefficient": margin_coeff,
        "margin_target_mode": "teacher_floor",
        "margin_floor": q25,
        "margin_cap": q75,
        "target_ratio": 0.25,
        "calibration_sample_counts": sample_counts,
        "measurements": measurements,
        "achieved_ratios": {"advce": float(beta * np.median(ce / base)), "margin": float(margin_coeff * np.median(margin / base))},
        "lineage": {"replay": {seed: {"dir": str(replay_dirs[seed].resolve()), "lineage_sha256": sha256(replay_dirs[seed] / "lineage.json")} for seed in replay_dirs}, "masks": {seed: {"path": str(mask_paths[seed].resolve()), "sha256": sha256(mask_paths[seed])} for seed in mask_paths}, "parents": EXPECTED_PARENT_SHA},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result["artifact_sha256"] = sha256(output)
    output.with_name(output.name + ".sha256").write_text(result["artifact_sha256"] + "\n", encoding="utf-8")
    return result
