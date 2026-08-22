"""Read-only RNG/feedback stability diagnostics for the frozen 0054 campaign.

This module deliberately performs no optimizer, scheduler, sample-state, or
checkpoint update.  ``replay`` runs a deterministic fixed-probe KL-PGD10
observation on saved checkpoints; ``aggregate`` combines those observations
with the already completed endpoint and epoch-metric panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.ert_clean_wrong_subtypes import _probability_stats
from ard.analysis.ert_cw_a7_mechanism_replay import PARENTS, _regime, _runtime_config, _targets, _treatment
from ard.attacks import AttackRequest, LinfPGD
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student, build_teacher
from ard.tracking.adapter import collect_git_state


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / ".cache/analysis/ert-cw-margin-local-lambda-stability-v1-v2"
ENDPOINT_ROOT = ROOT / ".cache/analysis/ert-cw-margin-local-lambda-stability-v1-v2-endpoints"
MACHINE_0054 = ROOT / "docs/experiments/ert_cw_margin_local_lambda_stability_v1.json"
REPLAY_ROOT = ROOT / ".cache/analysis/ert-cw-margin-rng-stability-v1"
MASK_PATH = {
    "L2": ROOT / ".cache/analysis/ert-state-overlay-v1-review/anchor79-fixed-masks-L2.json",
    "L4": ROOT / ".cache/analysis/ert-state-overlay-v1-review/anchor79-fixed-masks-L4.json",
}
MASK_SHA = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}
BLOCKS = ("L2-R1", "L2-R2", "L4-R1", "L4-R2")
TEACHERS = ("L2", "L4")
REPLICATES = ("R1", "R2")
ARMS = ("N95", "A100", "N105")
EPOCHS = (84, 89, 94)
PROBE_COUNT = 256
FLOOR = 0.03221710026264191
CAP = 0.13952550292015076
ENDPOINT_SHA = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
SOURCE_0054 = "10cfc5c277866e97a3853e2ca1cf9ec700fee990"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_sha(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def mask_ids(teacher: str) -> list[int]:
    path = MASK_PATH[teacher]
    if sha256(path) != MASK_SHA[teacher]:
        raise RuntimeError(f"registered mask hash mismatch: {teacher}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = sorted(int(x) for x in payload["masks"]["student_clean_wrong"]["selected_ids"])
    if len(ids) < PROBE_COUNT:
        raise RuntimeError(f"mask too small for fixed probe: {teacher}")
    return ids


def probe_ids(teacher: str) -> list[int]:
    return mask_ids(teacher)[:PROBE_COUNT]


def config_path(block: str, arm: str) -> Path:
    path = CAMPAIGN / block / arm / "resolved_config.yaml"
    if not path.is_file():
        raise RuntimeError(f"missing resolved config: {path}")
    return path


def checkpoint_path(block: str, arm: str, epoch: int) -> Path:
    path = CAMPAIGN / block / arm / "checkpoints" / f"epoch-{epoch}.pt"
    if not path.is_file():
        raise RuntimeError(f"missing checkpoint: {path}")
    return path


def _assert_attack(config: Any) -> None:
    attack = config.method.attack
    if not (
        attack.loss == "kl"
        and attack.kl_target == "teacher_clean"
        and attack.steps == 10
        and attack.epsilon == "8/255"
        and attack.step_size == "2/255"
        and attack.random_start
        and attack.input_domain == "pixel_0_1"
    ):
        raise RuntimeError("fixed replay requires the frozen Teacher-clean KL-PGD10 contract")


def _probe_seed(teacher: str, epoch: int, batch_index: int) -> int:
    teacher_code = 2 if teacher == "L2" else 4
    return 910_000 + 100_000 * teacher_code + 101 * epoch + batch_index


def _load_probe_batches(config: Any, epoch: int, ids: set[int], device: torch.device):
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    # The input view is deterministic by (augmentation seed, epoch, source ID).
    train_dataset.set_epoch(epoch)
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_rank_batch_size,
        sampler=EpochShuffleSampler(len(train_dataset), seed=0, rank=0, world_size=1, shuffle=False),
        num_workers=0,
        collate_fn=collate_indexed,
    )
    for batch_index, raw_batch in enumerate(loader):
        positions = [i for i, item in enumerate(raw_batch.sample_ids.tolist()) if int(item) in ids]
        if not positions:
            continue
        batch = raw_batch.to(device)
        index = torch.tensor(positions, device=device, dtype=torch.long)
        yield batch_index, batch.images.index_select(0, index), batch.labels.index_select(0, index), batch.sample_ids.index_select(0, index)


def replay_one(*, block: str, arm: str, epoch: int, device: str, output_root: Path) -> dict[str, Any]:
    teacher_name, replicate = block.split("-", 1)
    if teacher_name not in TEACHERS or replicate not in REPLICATES or arm not in ARMS or epoch not in EPOCHS:
        raise RuntimeError(f"unsupported replay identity: {block}/{arm}/epoch-{epoch}")
    out_dir = output_root / block / arm / f"epoch-{epoch}"
    meta_path = out_dir / "replay.json"
    rows_path = out_dir / "probe-sample-stats.parquet"
    if meta_path.exists() or rows_path.exists():
        raise RuntimeError(f"refusing to overwrite existing replay output: {out_dir}")
    config = _runtime_config(config_path(block, arm))
    _assert_attack(config)
    treatment = _treatment(config_path(block, arm))
    checkpoint = checkpoint_path(block, arm, epoch)
    checkpoint_sha = sha256(checkpoint)
    mask = mask_ids(teacher_name)
    ids = probe_ids(teacher_name)
    probe_set = set(ids)
    source = collect_git_state(ROOT)
    if source.get("dirty") is not False:
        raise RuntimeError("fixed replay requires a clean diagnostic source commit")
    expected_parent = PARENTS[teacher_name]
    if config_path(block, arm).read_text(encoding="utf-8").find(expected_parent) < 0:
        raise RuntimeError(f"parent SHA is not bound in resolved config: {block}")
    student = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, student)
    if payload.get("epoch") != epoch or payload.get("epoch_boundary") != "end":
        raise RuntimeError(f"checkpoint epoch mismatch: {checkpoint}")
    if payload.get("checkpoint_sha256") not in (None, checkpoint_sha):
        raise RuntimeError(f"checkpoint self hash mismatch: {checkpoint}")
    if config.teacher is None:
        raise RuntimeError("registered teacher is required")
    teacher = build_teacher(config.teacher, tier=config.tier)
    torch_device = torch.device(device)
    student.to(torch_device).eval()
    teacher.to(torch_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    attack = LinfPGD(config.method.attack)
    floor = float(treatment.get("margin_floor") or FLOOR)
    cap = float(treatment.get("margin_cap") or CAP)
    rows: list[dict[str, Any]] = []
    initial_delta_hashes: list[str] = []
    max_abs_delta = 0.0
    for batch_index, images, labels, sample_ids in _load_probe_batches(config, epoch, probe_set, torch_device):
        with torch.no_grad():
            student_clean_logits = student(images.float())
            teacher_clean_logits = teacher(images.float())
        seed = _probe_seed(teacher_name, epoch, batch_index)
        generator = torch.Generator(device=torch_device).manual_seed(seed)
        attack_result = attack.generate(
            AttackRequest(
                inputs=images,
                labels=labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits.detach().float(),
                generator=generator,
            )
        )
        initial_delta_hashes.append(tensor_sha256(attack_result.initial_delta))
        max_abs_delta = max(max_abs_delta, attack_result.max_abs_delta)
        with torch.no_grad():
            student_adv_logits = student(attack_result.adversarial.float())
            teacher_adv_logits = teacher(attack_result.adversarial.float())
        student_clean = _probability_stats(student_clean_logits, labels)
        student_adv = _probability_stats(student_adv_logits, labels)
        teacher_clean = _probability_stats(teacher_clean_logits, labels)
        teacher_adv = _probability_stats(teacher_adv_logits, labels)
        student_margin = student_adv["margin"]
        teacher_margin = teacher_adv["margin"]
        target, active = _targets(teacher_margin, treatment)
        deficit = target - student_margin
        hinge = torch.relu(deficit)
        for index, sample_id in enumerate(sample_ids.tolist()):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "true_label": int(labels[index]),
                    "teacher": teacher_name,
                    "block": block,
                    "arm": arm,
                    "epoch": epoch,
                    "attack_seed": seed,
                    "student_clean_prediction": int(student_clean["prediction"][index]),
                    "student_adv_prediction": int(student_adv["prediction"][index]),
                    "teacher_clean_prediction": int(teacher_clean["prediction"][index]),
                    "teacher_adv_prediction": int(teacher_adv["prediction"][index]),
                    "student_clean_correct": bool(student_clean["correct"][index]),
                    "student_adv_correct": bool(student_adv["correct"][index]),
                    "teacher_clean_correct": bool(teacher_clean["correct"][index]),
                    "teacher_adv_correct": bool(teacher_adv["correct"][index]),
                    "student_clean_margin": float(student_clean["margin"][index]),
                    "student_adv_margin": float(student_margin[index]),
                    "teacher_clean_margin": float(teacher_clean["margin"][index]),
                    "teacher_adv_margin": float(teacher_margin[index]),
                    "student_clean_true_probability": float(student_clean["true_probability"][index]),
                    "student_adv_true_probability": float(student_adv["true_probability"][index]),
                    "teacher_clean_true_probability": float(teacher_clean["true_probability"][index]),
                    "teacher_adv_true_probability": float(teacher_adv["true_probability"][index]),
                    "target": float(target[index]),
                    "target_active": bool(active[index]),
                    "raw_deficit": float(deficit[index]),
                    "positive_deficit": float(hinge[index]),
                    "hinge_active": bool(hinge[index] > 0),
                    "regime": _regime(float(teacher_margin[index]), floor=floor, cap=cap),
                    "floor_distance": abs(float(teacher_margin[index]) - floor),
                    "cap_distance": abs(float(teacher_margin[index]) - cap),
                    "hinge_distance": abs(float(deficit[index])),
                }
            )
    if {int(row["sample_id"]) for row in rows} != probe_set or len(rows) != len(probe_set):
        raise RuntimeError(f"fixed probe did not recover exact IDs: {block}/{arm}/epoch-{epoch}")
    if max_abs_delta > float(config.method.attack.epsilon_value) + 1e-7:
        raise RuntimeError("fixed replay exceeded pixel-space Linf bound")
    out_dir.mkdir(parents=True, exist_ok=False)
    write_sample_parquet(rows, rows_path)
    result = {
        "schema_version": 1,
        "contract": "ert_cw_margin_rng_fixed_probe_v1",
        "no_update": True,
        "source_git_sha": source["sha"],
        "block": block,
        "teacher": teacher_name,
        "replicate": replicate,
        "arm": arm,
        "epoch": epoch,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": expected_parent,
        "mask_path": str(MASK_PATH[teacher_name].resolve()),
        "mask_sha256": MASK_SHA[teacher_name],
        "mask_count": len(mask),
        "probe_count": len(ids),
        "probe_ids_sha256": json_sha(ids),
        "probe_selection": "first 256 sorted IDs from fixed epoch-79 Clean-Wrong mask",
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256(rows_path),
        "attack": config.method.attack.identity(),
        "attack_identity_sha256": config.method.attack.identity_sha256(),
        "fixed_probe_seed_protocol": "910000 + teacher_code*100000 + 101*epoch + full-train-batch-index; shared by R1/R2",
        "initial_delta_sha256_by_batch": initial_delta_hashes,
        "max_abs_delta": max_abs_delta,
        "treatment": treatment,
        "regime_contract": {"R0": "mT<=0", "R1": "0<mT<floor", "R2": "floor<=mT<=cap", "R3": "mT>cap"},
        "boundary_windows": [0.005, 0.01, 0.02],
    }
    meta_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result: dict[int, dict[str, Any]] = {}
    for row in values:
        sample_id = int(row["sample_id"])
        if sample_id in result:
            raise RuntimeError(f"duplicate probe ID: {path} {sample_id}")
        result[sample_id] = row
    return result


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def load_endpoint_meta(block: str, arm: str, epoch: int, split: str = "validation") -> dict[str, Any]:
    path = ENDPOINT_ROOT / block / arm / f"epoch-{epoch}" / split / "endpoint.json"
    if not path.is_file():
        raise RuntimeError(f"missing endpoint: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("contract") != "ert_stage_a_common_ce_pgd20_endpoint_v1" or meta.get("attack_identity_sha256") != ENDPOINT_SHA:
        raise RuntimeError(f"endpoint contract mismatch: {path}")
    return meta


def endpoint_absolute_variance(machine: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for teacher in TEACHERS:
        result[teacher] = {}
        for epoch in EPOCHS:
            result[teacher][str(epoch)] = {}
            for metric in ("clean_accuracy", "robust_accuracy"):
                values: dict[str, dict[str, float | None]] = {}
                for arm in ("B0_BASE", *ARMS):
                    values[arm] = {}
                    for replicate in REPLICATES:
                        block = f"{teacher}-{replicate}"
                        split = "validation"
                        path = ENDPOINT_ROOT / block / arm / f"epoch-{epoch}" / split / "endpoint.json"
                        if path.is_file():
                            values[arm][replicate] = float(json.loads(path.read_text())[metric])
                        else:
                            values[arm][replicate] = None
                result[teacher][str(epoch)][metric] = {
                    "absolute": values,
                    "base_gap": abs(values["B0_BASE"]["R1"] - values["B0_BASE"]["R2"])
                    if values["B0_BASE"]["R1"] is not None and values["B0_BASE"]["R2"] is not None else None,
                    "treatment_effect_gap": {
                        arm: abs(
                            (values[arm]["R1"] - values["B0_BASE"]["R1"])
                            - (values[arm]["R2"] - values["B0_BASE"]["R2"])
                        )
                        if values[arm]["R1"] is not None and values[arm]["R2"] is not None else None
                        for arm in ARMS
                    },
                }
    return result


def training_curves() -> dict[str, Any]:
    metrics = (
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "val_clean_accuracy",
        "val_pgd_accuracy",
        "learning_rate",
        "next_learning_rate",
    )
    result: dict[str, Any] = {}
    for block in BLOCKS:
        result[block] = {}
        for arm in ("B0_BASE", *ARMS):
            path = CAMPAIGN / block / arm / "epoch-metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            result[block][arm] = {
                metric: {str(int(row["epoch"])): float(row[metric]) for row in rows if metric in row}
                for metric in metrics
            }
    gaps: dict[str, Any] = {}
    for teacher in TEACHERS:
        gaps[teacher] = {}
        for arm in ("B0_BASE", *ARMS):
            pair = (f"{teacher}-R1", f"{teacher}-R2")
            gaps[teacher][arm] = {}
            for metric in metrics:
                gaps[teacher][arm][metric] = {
                    str(epoch): abs(result[pair[0]][arm][metric][str(epoch)] - result[pair[1]][arm][metric][str(epoch)])
                    for epoch in range(80, 95)
                }
    return {"absolute": result, "r1_r2_absolute_gaps": gaps}


def fixed_probe_summary(replay_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for teacher in TEACHERS:
        summary[teacher] = {}
        for arm in ARMS:
            summary[teacher][arm] = {}
            for epoch in EPOCHS:
                pair_rows = []
                for replicate in REPLICATES:
                    block = f"{teacher}-{replicate}"
                    meta_path = replay_root / block / arm / f"epoch-{epoch}" / "replay.json"
                    rows_path = replay_root / block / arm / f"epoch-{epoch}" / "probe-sample-stats.parquet"
                    if not meta_path.is_file() or not rows_path.is_file():
                        raise RuntimeError(f"missing fixed replay artifact: {block}/{arm}/epoch-{epoch}")
                    meta = json.loads(meta_path.read_text())
                    if meta.get("rows_sha256") != sha256(rows_path) or meta.get("no_update") is not True:
                        raise RuntimeError(f"fixed replay hash/no-update mismatch: {meta_path}")
                    rows = load_rows(rows_path)
                    pair_rows.append((meta, rows))
                left_meta, left = pair_rows[0]
                right_meta, right = pair_rows[1]
                if set(left) != set(right):
                    raise RuntimeError(f"fixed replay ID mismatch: {teacher}/{arm}/{epoch}")
                for key in ("probe_ids_sha256", "attack_identity_sha256"):
                    if left_meta[key] != right_meta[key]:
                        raise RuntimeError(f"fixed replay identity mismatch: {teacher}/{arm}/{epoch}/{key}")
                if left_meta["initial_delta_sha256_by_batch"] != right_meta["initial_delta_sha256_by_batch"]:
                    raise RuntimeError(f"fixed attack initial delta mismatch: {teacher}/{arm}/{epoch}")
                fields = ("student_adv_margin", "teacher_adv_margin", "target", "raw_deficit", "positive_deficit")
                abs_diff = {field: _mean([abs(float(left[i][field]) - float(right[i][field])) for i in left]) for field in fields}
                regime_transition = Counter((left[i]["regime"], right[i]["regime"]) for i in left)
                hinge_disagreement = sum(bool(left[i]["hinge_active"]) != bool(right[i]["hinge_active"]) for i in left) / len(left)
                prediction_disagreement = {
                    field: sum(left[i][field] != right[i][field] for i in left) / len(left)
                    for field in ("student_clean_prediction", "student_adv_prediction", "teacher_adv_prediction")
                }
                boundary = {}
                for epsilon in (0.005, 0.01, 0.02):
                    boundary[str(epsilon)] = {
                        "floor": _mean([sum(abs(float(rows[i]["teacher_adv_margin"]) - FLOOR) < epsilon for i in rows) / len(rows) for rows in (left, right)]),
                        "cap": _mean([sum(abs(float(rows[i]["teacher_adv_margin"]) - CAP) < epsilon for i in rows) / len(rows) for rows in (left, right)]),
                        "hinge": _mean([sum(abs(float(rows[i]["raw_deficit"])) < epsilon for i in rows) / len(rows) for rows in (left, right)]),
                        "hinge_disagreement_near": _mean([
                            sum(
                                (bool(left[i]["hinge_active"]) != bool(right[i]["hinge_active"]))
                                and abs(float(left[i]["raw_deficit"])) < epsilon
                                for i in left
                            ) / max(1, sum(bool(left[i]["hinge_active"]) != bool(right[i]["hinge_active"]) for i in left)),
                            sum(
                                (bool(left[i]["hinge_active"]) != bool(right[i]["hinge_active"]))
                                and abs(float(right[i]["raw_deficit"])) < epsilon
                                for i in right
                            ) / max(1, sum(bool(left[i]["hinge_active"]) != bool(right[i]["hinge_active"]) for i in right)),
                        ]),
                    }
                contraction_max = max(
                    abs(float(left[i]["target"]) - float(right[i]["target"]))
                    - abs(float(left[i]["teacher_adv_margin"]) - float(right[i]["teacher_adv_margin"]))
                    for i in left
                )
                if contraction_max > 1e-6:
                    raise RuntimeError(f"clip contraction violated: {teacher}/{arm}/{epoch}: {contraction_max}")
                summary[teacher][arm][str(epoch)] = {
                    "probe_count": len(left),
                    "sample_abs_mean_difference": abs_diff,
                    "regime_transition": {f"{a}->{b}": count for (a, b), count in sorted(regime_transition.items())},
                    "regime_disagreement_rate": sum(a != b for a, b in regime_transition.elements()) / len(left),
                    "hinge_disagreement_rate": hinge_disagreement,
                    "prediction_disagreement_rate": prediction_disagreement,
                    "boundary_windows": boundary,
                    "target_contraction_max_violation": contraction_max,
                    "target_contraction_ratio": abs_diff["target"] / (abs_diff["teacher_adv_margin"] + 1e-12),
                    "teacher_margin_sign_agreement": sum((float(left[i]["teacher_adv_margin"]) > 0) == (float(right[i]["teacher_adv_margin"]) > 0) for i in left) / len(left),
                }
    return summary


def build_report(*, replay_root: Path, output_json: Path, output_md: Path) -> None:
    machine_0054 = json.loads(MACHINE_0054.read_text())
    source = collect_git_state(ROOT)
    if source.get("dirty") is not False:
        raise RuntimeError("aggregation requires a clean source tree")
    fixed = fixed_probe_summary(replay_root)
    curves = training_curves()
    endpoint = endpoint_absolute_variance(machine_0054)
    machine = {
        "schema_version": 1,
        "contract": "ert_cw_margin_rng_stability_diagnostic_v1",
        "status": "completed_read_only_point_estimates",
        "source_git_sha": source["sha"],
        "source_0054_git_sha": SOURCE_0054,
        "parent_checkpoint_sha256": PARENTS,
        "mask_sha256": MASK_SHA,
        "calibration_sha256": "a625b43ec12277bbf698270193f27e0e1f62e0a2a9f9a6a49e7fc0702593b2b5",
        "training_attack_identity": "Teacher-clean KL-PGD10 pixel [0,1] eps=8/255 step=2/255 random_start",
        "endpoint_attack_identity_sha256": ENDPOINT_SHA,
        "fixed_probe": {"count": PROBE_COUNT, "selection": "first 256 sorted IDs from fixed epoch-79 Clean-Wrong mask", "ids_sha256": {teacher: json_sha(probe_ids(teacher)) for teacher in TEACHERS}},
        "blocks": list(BLOCKS),
        "arms": list(ARMS),
        "epochs": list(EPOCHS),
        "gradient_probe": {"status": "pending_or_unavailable", "note": "Gradient vector comparison is run separately when the focused no-update probe is available."},
        "base_and_treatment_endpoint_variance": endpoint,
        "training_metric_curves": curves,
        "fixed_probe_replay": fixed,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT Clean-Wrong Teacher-Adaptive Margin RNG Stability Diagnostic",
        "",
        "Status: completed read-only point-estimate diagnostic from the valid 0054 campaign.",
        "",
        "The initial `--epochs 94` attempt from 0054 is excluded; only the valid `--epochs 95` v2 campaign and its epoch-84/89/94 endpoints are used.",
        "No optimizer, scheduler, sample-state, checkpoint, or new training update was performed.",
        "",
        "## Direct answers",
        "",
        "- BASE continuation variance is reported from absolute endpoint metrics; the BASE delta in the 0054 report must not be used for this question.",
        "- Treatment-effect continuation variance is reported after subtracting the matched same-replicate BASE.",
        "- Existing epoch logs contain loss/accuracy/LR only; margin target, hinge, and regime quantities were not logged during training. Fixed-probe replay therefore localizes state/target/hinge divergence at epochs 84/89/94, not the unobserved per-step causal onset.",
        "- Fixed replay uses the same deterministic probe IDs, epoch-aligned augmentation view, and KL-PGD10 initial-delta seed for R1/R2. Clip contraction is asserted.",
        "- Gradient-vector comparison is not inferred from endpoint metrics; it is reported only if the separate focused no-update probe completes.",
        "",
        "## Frozen contract",
        "",
        "- Blocks: L2-R1, L2-R2, L4-R1, L4-R2; arms: N95, A100, N105; epochs: 84, 89, 94.",
        "- Teacher-adaptive target: `clip(mT_adv, 0.03221710026264191, 0.13952550292015076)`; CleanCE is zero.",
        "- Probe: 256 first sorted IDs from each registered epoch-79 Clean-Wrong mask.",
        "- No population-level seed confidence interval is claimed; these are descriptive point estimates.",
        "",
        "## Endpoint variance summary",
        "",
        "The machine artifact contains absolute R1/R2 accuracy values, BASE absolute gaps, and paired treatment-effect gaps for each teacher, epoch, metric, and arm.",
        "",
        "| teacher | epoch | metric | BASE gap (pp) | N95 effect gap (pp) | A100 effect gap (pp) | N105 effect gap (pp) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for teacher in TEACHERS:
        for epoch in EPOCHS:
            item = endpoint[teacher][str(epoch)]["robust_accuracy"]
            values = [item["base_gap"], item["treatment_effect_gap"]["N95"], item["treatment_effect_gap"]["A100"], item["treatment_effect_gap"]["N105"]]
            lines.append(f"| {teacher} | {epoch} | validation robust | " + " | ".join(f"{100*v:.3f}" if v is not None else "n/a" for v in values) + " |")
    lines += [
        "",
        "## Training-log divergence",
        "",
        "The full epoch-80--94 absolute R1/R2 gap curves are stored in the machine artifact.  The available logs contain `train_loss`, clean/robust accuracy, validation metrics, and learning rates; margin-specific fields were absent and are not reconstructed from endpoint outcomes.",
        "",
        "## Fixed-probe replay summary",
        "",
        "For each teacher/arm/epoch, the machine artifact stores sample-wise absolute differences, R0--R3 transition counts, hinge disagreement, prediction disagreement, floor/cap/hinge boundary windows, and the target contraction check.",
        "",
        "| teacher | arm | epoch | |ΔmT| | |Δtarget| | |Δdeficit| | regime disagreement | hinge disagreement | target contraction ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for teacher in TEACHERS:
        for arm in ARMS:
            for epoch in EPOCHS:
                item = fixed[teacher][arm][str(epoch)]
                d = item["sample_abs_mean_difference"]
                lines.append(
                    f"| {teacher} | {arm} | {epoch} | {d['teacher_adv_margin']:.6f} | {d['target']:.6f} | {d['raw_deficit']:.6f} | {item['regime_disagreement_rate']:.3%} | {item['hinge_disagreement_rate']:.3%} | {item['target_contraction_ratio']:.3f} |"
                )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This diagnostic can distinguish large BASE variance from additional treatment-effect variance and can show whether already-diverged checkpoints differ in target, regime, or hinge state.  It cannot identify which training RNG stream caused the divergence, and it cannot establish that target smoothing, adaptive weighting, or a smooth hinge improves performance.  Those remain human-review candidates only.",
        "",
        "## Source",
        "",
        f"- Source Git SHA: `{source['sha']}`; 0054 source: `{SOURCE_0054}`.",
        f"- Machine artifact: `{output_json}`.",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "aggregate"), required=True)
    parser.add_argument("--block")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-root", type=Path, default=REPLAY_ROOT)
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs/experiments/ert_cw_margin_rng_stability_diagnostic_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/ERT_CW_MARGIN_RNG_STABILITY_DIAGNOSTIC.md")
    args = parser.parse_args()
    if args.mode == "replay":
        if args.block is None or args.arm is None or args.epoch is None:
            parser.error("replay requires --block, --arm, and --epoch")
        result = replay_one(block=args.block, arm=args.arm, epoch=args.epoch, device=args.device, output_root=args.replay_root)
        print(json.dumps({"contract": result["contract"], "rows_sha256": result["rows_sha256"]}, sort_keys=True))
        return 0
    build_report(replay_root=args.replay_root, output_json=args.output_json, output_md=args.output_md)
    print(json.dumps({"status": "completed", "output": str(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
