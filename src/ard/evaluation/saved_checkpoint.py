"""Evaluation of immutable saved student checkpoints only."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ard.analysis import fixed_panel_ids, write_sample_parquet
from ard.attacks import AttackGenerator, AttackRequest
from ard.data import IndexedBatch
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.engine.distributed import reduce_sums, unwrap_model


@dataclass(frozen=True)
class EvaluationResult:
    checkpoint: str
    clean_accuracy: float
    pgd_accuracy: float
    count: int
    sample_stats: Path | None
    panel: Path


@contextmanager
def _eval_mode(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def load_saved_student_checkpoint(path: Path, model: nn.Module, *, weights_key: str = "model") -> Mapping[str, object]:
    """Load one named weight set from an already-written training checkpoint.

    Evaluation intentionally does not restore an optimizer, teacher, sample
    state, or any training-time policy.  It can therefore never use them as a
    test-time defence.

    ``weights_key`` selects which state dict to restore: ``"model"`` (the
    default) is the raw trained student, exactly what every non-ADR method
    has always evaluated.  ``"ema"`` restores the EMA/weight-averaged shadow
    model an ADR run also checkpoints (see ``ard.engine.trainer``) -- the
    official ADR code reports both, distinguished by an explicit ``--ema``
    flag ("ADR" rows use the raw student, "ADR + WA" rows use the EMA
    weights); this mirrors that choice rather than picking one silently.
    """
    if not path.is_file():
        raise FileNotFoundError(f"saved checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("evaluation requires a complete saved training checkpoint with model weights")
    weights = payload.get(weights_key)
    if not isinstance(weights, dict):
        available = [key for key in ("model", "ema") if isinstance(payload.get(key), dict)]
        raise ValueError(
            f"evaluation requested weights_key={weights_key!r} but the checkpoint carries no such state "
            f"(available weight sets: {available})"
        )
    unwrap_model(model).load_state_dict(weights, strict=True)
    return payload


def validate_checkpoint_lineage(path: Path, *, expected_config_hash: str) -> Mapping[str, object]:
    """Require the sibling resolved training config that created this model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("evaluation requires a complete saved training checkpoint with model weights")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError("evaluation requires a complete checkpoint; missing: " + ", ".join(sorted(missing)))
    if payload["epoch_boundary"] != "end":
        raise ValueError("evaluation requires an epoch-boundary checkpoint")
    world_size = payload["world_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("evaluation checkpoint world_size must be a positive integer")
    observed = payload.get("config_hash")
    if observed != expected_config_hash:
        raise ValueError("checkpoint config hash does not match its resolved training config")
    return payload


def evaluate_saved_checkpoint(
    *,
    checkpoint: Path,
    model: nn.Module,
    loader: DataLoader[IndexedBatch],
    attack: AttackGenerator,
    device: torch.device,
    seed: int,
    output_dir: Path,
    panel_size: int,
    write_sample_stats: bool = False,
    weights_key: str = "model",
) -> EvaluationResult:
    """Report clean and explicitly configured PGD accuracy from a saved model.

    ``weights_key`` picks which checkpointed state dict to evaluate -- see
    ``load_saved_student_checkpoint``. Output filenames are namespaced by it
    so evaluating both "model" and "ema" from the same checkpoint into the
    same output directory never collides.
    """
    load_saved_student_checkpoint(checkpoint, model, weights_key=weights_key)
    model.to(device)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    rows: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    with _eval_mode(model):
        for batch in loader:
            if not isinstance(batch, IndexedBatch):
                raise TypeError("evaluation requires IndexedBatch batches")
            batch = batch.to(device)
            valid = (
                batch.state_update_mask
                if batch.state_update_mask is not None
                else torch.ones_like(batch.labels, dtype=torch.bool)
            ).to(dtype=torch.bool)
            with torch.no_grad():
                clean_logits = model(batch.images)
            adversarial = attack.generate(
                AttackRequest(
                    inputs=batch.images,
                    labels=batch.labels,
                    student=model,
                    teacher=None,
                    generator=generator,
                )
            ).adversarial
            with torch.no_grad():
                adv_logits = model(adversarial)
            clean_correct = clean_logits.argmax(1).eq(batch.labels)
            pgd_correct = adv_logits.argmax(1).eq(batch.labels)
            totals += torch.tensor(
                [
                    float((clean_correct.to(torch.float64) * valid).sum()),
                    float((pgd_correct.to(torch.float64) * valid).sum()),
                    float(valid.sum()),
                ],
                dtype=torch.float64,
                device=device,
            )
            for sample_id, label, clean, robust, clean_prediction, adv_prediction in zip(
                batch.sample_ids[valid].tolist(),
                batch.labels[valid].tolist(),
                clean_correct[valid].tolist(),
                pgd_correct[valid].tolist(),
                clean_logits.argmax(1)[valid].tolist(),
                adv_logits.argmax(1)[valid].tolist(),
            ):
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "true_label": int(label),
                        "clean_correct": bool(clean),
                        "pgd_correct": bool(robust),
                        "student_clean_prediction": int(clean_prediction),
                        "student_adv_prediction": int(adv_prediction),
                    }
                )
    totals = reduce_sums(totals)
    count = int(totals[2].item())
    if count <= 0:
        raise ValueError("evaluation split contained no valid samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids: list[int] = []
    for row in rows:
        sample_id = row["sample_id"]
        if not isinstance(sample_id, int):
            raise TypeError("evaluation sample ID must be an integer")
        sample_ids.append(sample_id)
    selected = set(fixed_panel_ids(sample_ids, seed=seed, size=panel_size))
    # "model" keeps the pre-existing filename exactly (no evaluation ever
    # requested anything else before EMA support existed); any other weight
    # set gets an explicit suffix so it never collides with the student's.
    name_suffix = "" if weights_key == "model" else f"-{weights_key}"
    panel = output_dir / f"panel-{checkpoint.stem}{name_suffix}.jsonl"
    panel.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows if row["sample_id"] in selected),
        encoding="utf-8",
    )
    sample_stats = (
        write_sample_parquet(rows, output_dir / f"sample-stats-{checkpoint.stem}{name_suffix}.parquet")
        if write_sample_stats
        else None
    )
    return EvaluationResult(
        checkpoint=checkpoint.name,
        clean_accuracy=float(totals[0].item()) / count,
        pgd_accuracy=float(totals[1].item()) / count,
        count=count,
        sample_stats=sample_stats,
        panel=panel,
    )
