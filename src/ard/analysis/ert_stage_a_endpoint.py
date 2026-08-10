"""Independent common CE-PGD20 endpoint for ERT Stage A arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import EpochShuffleSampler, build_train_validation_views, collate_indexed
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student
from ard.tracking.adapter import collect_git_state


class StageAEndpointError(RuntimeError):
    """The independent Stage A endpoint contract is not satisfied."""


def _probability_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong = probabilities.clone()
    wrong.scatter_(1, labels[:, None], float("-inf"))
    return true_probability - wrong.max(dim=1).values


def evaluate_endpoint(
    *, config_path: Path, checkpoint: Path, output_dir: Path, device: torch.device, expected_epoch: int = 84
) -> dict[str, Any]:
    config = load_config(config_path)
    attack_config = config.method.selection_attack
    if attack_config is None or attack_config.loss != "ce" or attack_config.steps != 20:
        raise StageAEndpointError("Stage A endpoint must be the configured CE-PGD20 selection attack")
    if attack_config.epsilon != "8/255" or attack_config.step_size != "2/255" or not attack_config.random_start:
        raise StageAEndpointError("Stage A endpoint attack identity is not the frozen 8/255, 2/255 contract")
    model = build_student(config.student, tier=config.tier)
    payload = load_saved_student_checkpoint(checkpoint, model)
    if payload.get("epoch") != expected_epoch or payload.get("epoch_boundary") != "end":
        raise StageAEndpointError(f"endpoint requires epoch-{expected_epoch} end checkpoint")
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise StageAEndpointError("endpoint evaluation requires a clean source tree")
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
    model.to(device).eval()
    attack = LinfPGD(attack_config)
    generator = torch.Generator(device=device).manual_seed(config.seeds.evaluation_attack)
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            clean_logits = model(batch.images)
        adversarial = attack.generate(
            AttackRequest(inputs=batch.images, labels=batch.labels, student=model, teacher=None, generator=generator)
        ).adversarial
        with torch.no_grad():
            adv_logits = model(adversarial)
        clean_margin = _probability_margin(clean_logits, batch.labels)
        adv_margin = _probability_margin(adv_logits, batch.labels)
        clean_logit_margin = clean_logits.gather(1, batch.labels[:, None]).squeeze(1)
        wrong = clean_logits.clone()
        wrong.scatter_(1, batch.labels[:, None], float("-inf"))
        clean_logit_margin -= wrong.max(dim=1).values
        adv_logit_margin = adv_logits.gather(1, batch.labels[:, None]).squeeze(1)
        wrong = adv_logits.clone()
        wrong.scatter_(1, batch.labels[:, None], float("-inf"))
        adv_logit_margin -= wrong.max(dim=1).values
        clean_pred, adv_pred = clean_logits.argmax(1), adv_logits.argmax(1)
        for idx in range(batch.labels.shape[0]):
            rows.append(
                {
                    "sample_id": int(batch.sample_ids[idx]),
                    "true_label": int(batch.labels[idx]),
                    "clean_prediction": int(clean_pred[idx]),
                    "adversarial_prediction": int(adv_pred[idx]),
                    "clean_correct": bool(clean_pred[idx] == batch.labels[idx]),
                    "robust_correct": bool(adv_pred[idx] == batch.labels[idx]),
                    "clean_probability_margin": float(clean_margin[idx]),
                    "adversarial_probability_margin": float(adv_margin[idx]),
                    "probability_margin_delta": float(clean_margin[idx] - adv_margin[idx]),
                    "clean_logit_margin": float(clean_logit_margin[idx]),
                    "adversarial_logit_margin": float(adv_logit_margin[idx]),
                    "logit_margin_delta": float(clean_logit_margin[idx] - adv_logit_margin[idx]),
                }
            )
    if not rows:
        raise StageAEndpointError("endpoint dataset produced no rows")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "endpoint-sample-stats.parquet"
    write_sample_parquet(rows, rows_path)
    result = {
        "schema_version": 1,
        "contract": "ert_stage_a_common_ce_pgd20_endpoint_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": expected_epoch,
        "attack": attack_config.identity(),
        "attack_identity_sha256": attack_config.identity_sha256(),
        "source_git_sha": source["sha"],
        "row_count": len(rows),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "clean_accuracy": sum(row["clean_correct"] for row in rows) / len(rows),
        "robust_accuracy": sum(row["robust_correct"] for row in rows) / len(rows),
    }
    (output_dir / "endpoint.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
