"""Read-only canonical longitudinal states for the I100 S2 forensic audit.

The historical Dynamic-BDD state replays were useful endpoint diagnostics, but
their augmented, batch-keyed observation contract is not interchangeable with
the epoch-99 fixed-mask observation.  This module provides the deliberately
separate, unaugmented and sample-keyed CE-PGD20 observation used for the
longitudinal audit.  It never constructs an optimiser or mutates a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from ard.analysis.ert_i100_action_transfer import _attack_config, _deterministic_backend, _primitives, sha256
from ard.analysis.ert_i100_s2_dynamic_bdd_state import canonical_state_summary
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import build_dataset, collate_indexed, stratified_train_validation_split
from ard.models import build_student, build_teacher

CONTRACT = "ert_rslad_i100_s2_longitudinal_ce20_state_v1"
OBSERVATION_EPOCH_KEY = 99


class LongitudinalStateError(RuntimeError):
    """A frozen longitudinal-observation invariant was violated."""


def _raw_train_loader(config: Any, *, batch_size: int) -> DataLoader:
    """Match the epoch-99 raw, unaugmented train observation exactly."""
    raw = build_dataset(config.dataset)
    train, _ = stratified_train_validation_split(
        raw, validation_fraction=config.training.validation_fraction, seed=config.seeds.split
    )
    return DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_indexed)


def _load_checkpoint(
    config_path: Path,
    checkpoint: Path,
    *,
    expected_sha256: str,
    expected_epoch: int,
    device: torch.device,
):
    if sha256(checkpoint) != expected_sha256:
        raise LongitudinalStateError("checkpoint SHA-256 differs from registered lineage")
    config = load_config(config_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("epoch") != expected_epoch
        or payload.get("epoch_boundary") != "end"
    ):
        raise LongitudinalStateError("checkpoint is not the requested epoch end-boundary state")
    student = build_student(config.student, tier=config.tier)
    student.load_state_dict(payload["model"], strict=True)
    student = student.to(device).eval()
    if config.teacher is None:
        raise LongitudinalStateError("canonical state replay requires a frozen Teacher")
    teacher = build_teacher(config.teacher, tier=config.tier).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return config, payload, student, teacher


def canonical_action_states(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify audit-only mutually-exclusive Student branches plus Teacher T state.

    The q10 membership is intentionally reconstructed with the historical
    canonical q10 universe (all adversarial-correct rows), then Clean-Wrong
    takes precedence in the action branch.  This preserves the registered
    epoch-99 S2xT1 mask while ensuring Clean-Wrong is never displayed as S3.
    """
    materialized = [dict(row) for row in rows]
    legacy = canonical_state_summary(materialized)
    by_id = {int(row["sample_id"]): row for row in materialized}
    if len(by_id) != len(materialized):
        raise LongitudinalStateError("state rows contain duplicate sample IDs")
    state_by_id: dict[int, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    for sample_id, row in by_id.items():
        prior = legacy["state_by_id"][sample_id]
        if not bool(row["student_clean_correct"]):
            branch = "Clean-Wrong"
        elif not bool(row["student_ce20_adv_correct"]):
            branch = "S3-non-CW"
        elif prior["student"] == "S2":
            branch = "S2"
        else:
            branch = "S1"
        teacher = str(prior["teacher"])
        state_by_id[sample_id] = {
            "branch": branch,
            "teacher": teacher,
            "joint": f"{branch}x{teacher}",
        }
        counts[f"{branch}x{teacher}"] += 1
    branches = {value["branch"] for value in state_by_id.values()}
    allowed = {"Clean-Wrong", "S3-non-CW", "S2", "S1"}
    if not branches <= allowed or sum(counts.values()) != len(by_id):
        raise LongitudinalStateError("action-branch partition is not conservative")
    return {
        "row_count": len(by_id),
        "student_q10_floor": legacy["student_q10_floor"],
        "teacher_q10_floor": legacy["teacher_q10_floor"],
        "joint_counts": dict(sorted(counts.items())),
        "state_by_id": state_by_id,
        "q10_universe": "all CE20 adversarial-correct rows; action branch gives Clean-Wrong precedence",
    }


def replay_canonical_state(
    *,
    config_path: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_epoch: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Replay one saved checkpoint under the frozen e99 CE20 observation contract."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LongitudinalStateError(f"refusing to overwrite replay output: {output_dir}")
    _deterministic_backend()
    config, payload, student, teacher = _load_checkpoint(
        config_path,
        checkpoint,
        expected_sha256=expected_checkpoint_sha256,
        expected_epoch=expected_epoch,
        device=device,
    )
    attack_config = _attack_config(config, "ce20")
    attack = LinfPGD(attack_config)
    rows: list[dict[str, Any]] = []
    for batch in _raw_train_loader(config, batch_size=batch_size):
        batch = batch.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            student_clean = student(batch.images.float()).detach().float()
            teacher_clean = teacher(batch.images.float()).detach().float()
        result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                source_ids=batch.sample_ids,
                epoch=OBSERVATION_EPOCH_KEY,
                attack_seed=int(config.seeds.evaluation_attack),
                stream_tag="selection_pgd",
                restart_index=0,
                generator=torch.Generator(device=device),
            )
        )
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            student_adv = student(result.adversarial.float()).detach().float()
            teacher_adv = teacher(result.adversarial.float()).detach().float()
        s_clean = _primitives(student_clean, batch.labels)
        s_adv = _primitives(student_adv, batch.labels)
        t_clean = _primitives(teacher_clean, batch.labels)
        t_adv = _primitives(teacher_adv, batch.labels)
        pairs = zip(batch.sample_ids.tolist(), batch.labels.tolist(), strict=True)
        for index, (sample_id, class_id) in enumerate(pairs):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "class_id": int(class_id),
                    "epoch": int(expected_epoch),
                    "student_clean_correct": bool(s_clean["correct"][index]),
                    "student_clean_probability": float(s_clean["probability"][index]),
                    "student_clean_margin": float(s_clean["margin"][index]),
                    "student_ce20_adv_correct": bool(s_adv["correct"][index]),
                    "student_ce20_adv_probability": float(s_adv["probability"][index]),
                    "student_ce20_adv_margin": float(s_adv["margin"][index]),
                    "teacher_clean_correct": bool(t_clean["correct"][index]),
                    "teacher_clean_probability": float(t_clean["probability"][index]),
                    "teacher_clean_margin": float(t_clean["margin"][index]),
                    "teacher_ce20_adv_correct": bool(t_adv["correct"][index]),
                    "teacher_ce20_adv_probability": float(t_adv["probability"][index]),
                    "teacher_ce20_adv_margin": float(t_adv["margin"][index]),
                }
            )
    if len(rows) != 45_000 or len({int(row["sample_id"]) for row in rows}) != 45_000:
        raise LongitudinalStateError("canonical replay did not cover exactly the fixed 45,000-row train split")
    states = canonical_action_states(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "state-rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), rows_path, compression="zstd")
    result = {
        "schema_version": 1,
        "contract": CONTRACT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_epoch": expected_epoch,
        "payload_epoch": int(payload["epoch"]),
        "teacher_checkpoint_sha256": config.teacher.checkpoint_sha256,
        "row_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256(rows_path),
        "observation": {
            "clean_view": "raw unaugmented train split",
            "attack": attack_config.identity(),
            "attack_identity_sha256": attack_config.identity_sha256(),
            "random_start_keying": "sample_keyed_v1",
            "observation_epoch_key": OBSERVATION_EPOCH_KEY,
            "attack_seed": int(config.seeds.evaluation_attack),
            "stream_tag": "selection_pgd",
            "student_mode": "eval",
            "teacher_mode": "eval",
            "batch_size": batch_size,
        },
        "state_summary": {key: value for key, value in states.items() if key != "state_by_id"},
    }
    metadata = output_dir / "state-replay.json"
    metadata.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    (output_dir / "state-replay.sha256").write_text(metadata_sha + "\n", encoding="utf-8")
    return result
