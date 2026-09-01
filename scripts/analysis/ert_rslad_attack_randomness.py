"""Read-only fixed-model and sample-level analysis for attack-seed probes.

The fixed-model phase uses the real epoch-99 parent, real training view at
epoch 100, and the public attack implementation.  It never restores or
mutates optimizer, scheduler, sampler, or sample state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ard.analysis import write_sample_parquet
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.models import build_student, build_teacher

PARENT_HASHES = {
    1: "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    2: "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
DATA_ROOT = "/home/islab/workspace-local/shunsuke.naito/datasets/ard/torchvision"
TEACHER_PATH = (
    "/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=1)
    true = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong = probabilities.clone()
    wrong.scatter_(1, labels[:, None], float("-inf"))
    return true - wrong.max(dim=1).values


def _stratified_ids(dataset: Any, *, count: int) -> list[int]:
    indices = [int(value) for value in dataset.indices]
    targets = getattr(dataset.dataset.dataset, "targets", None)
    if not isinstance(targets, (list, tuple)):
        raise ValueError("training view does not expose immutable labels")
    by_class: dict[int, list[int]] = defaultdict(list)
    for source_id in indices:
        by_class[int(targets[source_id])].append(source_id)
    if count > len(indices):
        count = len(indices)
    chosen: list[int] = []
    position = 0
    classes = sorted(by_class)
    while len(chosen) < count:
        progressed = False
        for label in classes:
            members = by_class[label]
            if position < len(members) and len(chosen) < count:
                chosen.append(members[position])
                progressed = True
        position += 1
        if not progressed:
            break
    return chosen


def _load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("status") != "frozen_before_training":
        raise ValueError("attack registry is not frozen")
    return registry


def _parent_payload(path: Path, seed: int) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != PARENT_HASHES[seed]:
        raise ValueError(f"seed {seed} parent bytes do not match registered epoch-99 SHA")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 99 or payload.get("epoch_boundary") != "end":
        raise ValueError("fixed-model replay requires the exact epoch-99 end boundary")
    if REQUIRED_KEYS.difference(payload):
        raise ValueError("parent checkpoint is incomplete")
    return payload


def _seed_rows(registry: Mapping[str, Any]) -> list[tuple[int, int]]:
    rows = registry.get("seeds")
    if not isinstance(rows, list):
        raise ValueError("registry seeds are missing")
    result = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("index"), int)
            or not isinstance(row.get("value"), int)
        ):
            raise ValueError("registry seed row is invalid")
        result.append((int(row["index"]), int(row["value"])))
    if [index for index, _ in result] != list(range(8)):
        raise ValueError("registry must contain attack indices 0--7 in order")
    return result


def fixed_model_replay(
    *, config_path: Path, parent_path: Path, registry_path: Path, output_dir: Path, seed: int, limit: int
) -> dict[str, Any]:
    parent = _parent_payload(parent_path, seed)
    registry = _load_registry(registry_path)
    attack_rows = _seed_rows(registry)
    config = load_config(
        config_path,
        [f"dataset.root={DATA_ROOT}", f"evaluation.dataset.root={DATA_ROOT}", f"teacher.checkpoint={TEACHER_PATH}"],
    )
    if config.seeds.model_init != seed or config.seeds.data_order != seed or config.seeds.augmentation != seed:
        raise ValueError("fixed replay config does not match parent streams")
    if config.method.attack.random_start_keying != "sample_keyed_v1":
        raise ValueError("fixed replay requires sample_keyed_v1")
    attack = LinfPGD(config.method.attack)
    teacher = build_teacher(config.teacher, tier=config.tier).eval() if config.teacher is not None else None
    if teacher is None:
        raise ValueError("RSLAD fixed replay requires the frozen Teacher")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = build_student(config.student, tier=config.tier)
    student.load_state_dict(parent["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device).eval()
    teacher.to(device).eval()
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    train_dataset.set_epoch(100)
    ids = _stratified_ids(train_dataset, count=limit)
    positions = [train_dataset.indices.index(source_id) for source_id in ids]
    subset = Subset(train_dataset, positions)
    loader = DataLoader(
        subset,
        batch_size=config.training.per_rank_batch_size,
        sampler=EpochShuffleSampler(len(subset), seed=0, shuffle=False),
        num_workers=0,
        collate_fn=collate_indexed,
    )
    base_margin_by_id = {
        int(source_id): float(record.get("margin_ema", 0.0))
        for source_id, record in parent["sample_state"]["records"].items()
        if isinstance(record, Mapping)
    }
    values: dict[int, dict[int, dict[str, Any]]] = {index: {} for index, _ in attack_rows}
    for attack_index, attack_seed_value in attack_rows:
        rows: list[dict[str, Any]] = []
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                clean_logits = student(batch.images.float())
                target_logits = teacher(batch.images.float()).detach().float()
            result = attack.generate(
                AttackRequest(
                    inputs=batch.images.float(),
                    labels=batch.labels,
                    student=student,
                    teacher=teacher,
                    target_logits=target_logits,
                    source_ids=batch.sample_ids,
                    epoch=100,
                    attack_seed=attack_seed_value,
                )
            )
            with torch.no_grad():
                adv_logits = student(result.adversarial.float())
            clean_margin = _margin(clean_logits, batch.labels)
            adv_margin = _margin(adv_logits, batch.labels)
            target_prob = target_logits.float().softmax(dim=1)
            student_log_prob = adv_logits.float().log_softmax(dim=1)
            per_sample_kl = F.kl_div(student_log_prob, target_prob, reduction="none").sum(dim=1)
            clean_pred = clean_logits.argmax(dim=1)
            adv_pred = adv_logits.argmax(dim=1)
            for position, source_tensor in enumerate(batch.sample_ids):
                source_id = int(source_tensor)
                rows.append(
                    {
                        "seed": seed,
                        "attack_index": attack_index,
                        "attack_seed": attack_seed_value,
                        "epoch": 100,
                        "sample_id": source_id,
                        "true_label": int(batch.labels[position]),
                        "clean_correct": bool(clean_pred[position] == batch.labels[position]),
                        "adv_correct": bool(adv_pred[position] == batch.labels[position]),
                        "clean_margin": float(clean_margin[position]),
                        "adv_margin": float(adv_margin[position]),
                        "attack_kl": float(per_sample_kl[position]),
                        "initial_delta_sha256": hashlib.sha256(
                            result.initial_delta[position].detach().cpu().contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "risk": -float(base_margin_by_id.get(source_id, 0.0)),
                    }
                )
        values[attack_index] = {int(row["sample_id"]): row for row in rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for attack_rows_ in values.values() for row in attack_rows_.values()]
    rows_path = output_dir / f"fixed-model-seed{seed}.parquet"
    write_sample_parquet(all_rows, rows_path)
    per_sample: list[dict[str, Any]] = []
    for source_id in ids:
        observations = [values[index][source_id] for index, _ in attack_rows]
        margins = [float(row["adv_margin"]) for row in observations]
        losses = [float(row["attack_kl"]) for row in observations]
        per_sample.append(
            {
                "seed": seed,
                "sample_id": source_id,
                "true_label": int(observations[0]["true_label"]),
                "risk": float(observations[0]["risk"]),
                "adv_margin_mean": statistics.fmean(margins),
                "adv_margin_sd": statistics.stdev(margins) if len(margins) > 1 else 0.0,
                "adv_margin_range": max(margins) - min(margins),
                "attack_kl_mean": statistics.fmean(losses),
                "attack_kl_sd": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "attack_kl_range": max(losses) - min(losses),
                "adv_correct_frequency": sum(bool(row["adv_correct"]) for row in observations),
                "initial_delta_sha256": [str(row["initial_delta_sha256"]) for row in observations],
            }
        )
    sample_path = output_dir / f"fixed-model-seed{seed}-summary.json"
    sample_path.write_text(json.dumps(per_sample, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    risk = [float(row["risk"]) for row in per_sample]
    margin_sd = [float(row["adv_margin_sd"]) for row in per_sample]
    loss_sd = [float(row["attack_kl_sd"]) for row in per_sample]
    return {
        "seed": seed,
        "parent_checkpoint_sha256": sha256(parent_path),
        "parent_payload_epoch": parent["epoch"],
        "epoch": 100,
        "sample_count": len(ids),
        "sample_id_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256(rows_path),
        "summary_path": str(sample_path.resolve()),
        "attack_count": len(attack_rows),
        "risk_margin_sd_spearman": _spearman(risk, margin_sd),
        "risk_attack_kl_sd_spearman": _spearman(risk, loss_sd),
        "attack_seed_rows": [{"index": index, "value": value} for index, value in attack_rows],
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        value = (start + end - 1) / 2.0
        for pos in order[start:end]:
            result[pos] = value
        start = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    x, y = _rank(left), _rank(right)
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    den = math.sqrt(sum((v - xm) ** 2 for v in x) * sum((v - ym) ** 2 for v in y))
    return None if den == 0.0 else sum((a - xm) * (b - ym) for a, b in zip(x, y)) / den


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixed = sub.add_parser("fixed-model")
    fixed.add_argument("--config", type=Path, required=True)
    fixed.add_argument("--parent", type=Path, required=True)
    fixed.add_argument("--registry", type=Path, required=True)
    fixed.add_argument("--output", type=Path, required=True)
    fixed.add_argument("--seed", type=int, choices=(1, 2), required=True)
    fixed.add_argument("--limit", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fixed-model":
        if args.limit < 8192:
            raise ValueError("fixed-model replay limit must be at least 8192")
        result = fixed_model_replay(
            config_path=args.config,
            parent_path=args.parent,
            registry_path=args.registry,
            output_dir=args.output,
            seed=args.seed,
            limit=args.limit,
        )
        (args.output / "fixed-model-result.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
