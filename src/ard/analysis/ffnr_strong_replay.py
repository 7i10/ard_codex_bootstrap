"""Hash-bound CE-PGD20 replay for the Chen FF/NR plateau analysis.

This is deliberately a new observation domain.  The existing schema-v2
``rslad_signal_replay`` records the RSLAD KL-PGD10 training attack and must
never be repurposed as a selection-attack result.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import (
    CheckpointInventory,
    SignalAuditError,
    canonical_json,
    inventory_run_bundle,
    sha256_file,
)
from ard.analysis.teacher_risk_replay import load_historical_student
from ard.attacks import AttackRequest, LinfPGD
from ard.config import ExperimentConfig
from ard.config.schema import AttackConfig


class StrongReplayError(SignalAuditError):
    """Raised when the CE-PGD20 FF/NR replay contract is not satisfied."""


STRONG_REPLAY_SCHEMA_VERSION = 1
CONTRACT_ID = "ffnr_strong_replay_ce_pgd20_v1"
SEED_FORMULA = "attack_seed=base_seed+1000003*batch_index"
EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256 = "9a1a7929e47196ca4cb49a7c2bea5029170ecdb1c18f9f38c05ea14d9913bf60"
OBSERVATION_COLUMNS = (
    "namespace",
    "sample_id",
    "class_id",
    "epoch",
    "observation_schema_version",
    "student_robust_correct",
    "student_adversarial_probability_margin",
    "student_adversarial_logit_margin",
    "student_adversarial_ce",
    "student_clean_probability_margin",
    "student_clean_logit_margin",
    "student_clean_correct",
    "student_clean_to_adversarial_prediction_flip",
    "student_clean_to_adversarial_true_probability_delta",
    "student_clean_to_adversarial_probability_margin_delta",
    "student_clean_to_adversarial_logit_margin_delta",
    "teacher_clean_probabilities",
    "teacher_adversarial_probabilities",
    "teacher_clean_adversarial_js",
)


@dataclass(frozen=True)
class StrongReplayResult:
    epoch: int
    checkpoint_sha256: str
    attack_seed_base: int
    max_abs_delta: float
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DeterministicBackendFlags:
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool


@contextmanager
def deterministic_replay_backend() -> Iterator[DeterministicBackendFlags]:
    """Enforce FP32-deterministic backend settings for the full replay scope."""
    previous = DeterministicBackendFlags(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    effective = DeterministicBackendFlags(True, False, True, False, False)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield effective
    finally:
        torch.use_deterministic_algorithms(previous.deterministic_algorithms, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous.cudnn_benchmark
        torch.backends.cudnn.deterministic = previous.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = previous.cuda_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = previous.cudnn_allow_tf32


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise StrongReplayError(f"{name} must be a lowercase SHA-256")
    return value


def expected_selection_attack() -> dict[str, object]:
    """Return the immutable pixel-space primary replay attack identity."""
    return {
        "norm": "linf",
        "input_domain": "pixel_0_1",
        "epsilon": "8/255",
        "epsilon_value": 8.0 / 255.0,
        "step_size": "2/255",
        "step_size_value": 2.0 / 255.0,
        "steps": 20,
        "random_start": True,
        "loss": "ce",
        "kl_target": None,
        "temperature": 1.0,
        "temperature_squared": True,
        "student_mode": "eval",
        "teacher_mode": "eval",
    }


def validate_selection_attack(attack: AttackConfig) -> None:
    """Reject any drift from the configured primary CE-PGD20 contract."""
    if attack.identity() != expected_selection_attack() or attack.trace_step_losses:
        raise StrongReplayError("strong replay requires exact pixel Linf CE-PGD20 selection attack")


def selection_attack_from_training(config: ExperimentConfig) -> AttackConfig:
    attack = config.method.selection_attack
    if attack is None:  # Defensive: schema normally resolves this field.
        raise StrongReplayError("training config has no resolved checkpoint-selection attack")
    validate_selection_attack(attack)
    return attack


def select_explicit_checkpoints(
    inventory: Sequence[CheckpointInventory], *, run_id: str, epochs: Sequence[int]
) -> tuple[CheckpointInventory, ...]:
    """Select exactly one hash-verified periodic-last checkpoint per configured epoch."""
    expected = tuple(epochs)
    if (
        not expected
        or expected != tuple(sorted(set(expected)))
        or any(isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 for epoch in expected)
    ):
        raise StrongReplayError("requested epochs must be a non-empty sorted unique sequence of non-negative integers")
    candidates = [item for item in inventory if item.run_id == run_id and item.periodic_last]
    selected: list[CheckpointInventory] = []
    for epoch in expected:
        matches = [item for item in candidates if item.epoch == epoch]
        if len(matches) != 1:
            raise StrongReplayError(
                f"full run-bundle inventory requires exactly one periodic last checkpoint for epoch {epoch}"
            )
        selected.append(matches[0])
    if len({item.sha256 for item in selected}) != len(selected):
        raise StrongReplayError("selected checkpoint hashes must be unique across requested epochs")
    config_hashes = {item.config_hash for item in selected}
    git_shas = {item.scientific_git_sha for item in selected}
    if len(config_hashes) != 1 or len(git_shas) != 1:
        raise StrongReplayError("selected checkpoints must share config and scientific Git identity")
    return tuple(selected)


def build_checkpoint_inventory_document(*, manifest_path: Path, run_id: str) -> dict[str, Any]:
    """Scan/hash a complete run bundle once and freeze every periodic-last checkpoint."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrongReplayError("full run-bundle manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or manifest.get("status") != "completed":
        raise StrongReplayError("full run-bundle manifest must be completed before strong replay")
    inventory = inventory_run_bundle(manifest_path)
    selected = [item for item in inventory if item.run_id == run_id and item.periodic_last]
    if not selected:
        raise StrongReplayError("full run-bundle inventory has no periodic last checkpoints for requested run")
    by_epoch: dict[int, list[CheckpointInventory]] = {}
    for item in selected:
        by_epoch.setdefault(item.epoch, []).append(item)
    if any(len(items) != 1 for items in by_epoch.values()):
        raise StrongReplayError("full run-bundle inventory has duplicate periodic last checkpoints for one epoch")
    selected = [by_epoch[epoch][0] for epoch in sorted(by_epoch)]
    config_hashes = {item.config_hash for item in selected}
    git_shas = {item.scientific_git_sha for item in selected}
    if len(config_hashes) != 1 or len(git_shas) != 1:
        raise StrongReplayError("full run-bundle inventory has inconsistent run/config/Git identity")
    return {
        "schema_version": 1,
        "contract": CONTRACT_ID,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "run_id": run_id,
        "config_hash": next(iter(config_hashes)),
        "scientific_git_sha": next(iter(git_shas)),
        "checkpoints": [asdict(item) for item in selected],
    }


def write_checkpoint_inventory(*, path: Path, document: Mapping[str, Any]) -> Path:
    """Atomically publish a full-bundle inventory; identical existing bytes are reusable."""
    rendered = json.dumps(dict(document), sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise StrongReplayError("checkpoint inventory already exists with different bytes")
        return path
    return _atomic_json(path, dict(document))


def load_checkpoint_inventory_document(
    *, path: Path, manifest_path: Path, run_id: str
) -> tuple[CheckpointInventory, ...]:
    """Reuse a full-bundle inventory without rescanning/deserializing its sibling checkpoints."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrongReplayError("checkpoint inventory or manifest is unreadable") from exc
    required = {
        "schema_version",
        "contract",
        "manifest",
        "manifest_sha256",
        "run_id",
        "config_hash",
        "scientific_git_sha",
        "checkpoints",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise StrongReplayError("checkpoint inventory schema is invalid")
    if document["schema_version"] != 1 or document["contract"] != CONTRACT_ID or document["run_id"] != run_id:
        raise StrongReplayError("checkpoint inventory contract/run ID mismatch")
    if document["manifest"] != str(manifest_path.resolve()):
        raise StrongReplayError("checkpoint inventory manifest path mismatch")
    if document["manifest_sha256"] != sha256_file(manifest_path):
        raise StrongReplayError("checkpoint inventory manifest hash drift")
    if not isinstance(manifest, Mapping) or manifest.get("run_id") != run_id or manifest.get("status") != "completed":
        raise StrongReplayError("checkpoint inventory manifest run ID mismatch")
    manifest_git = manifest.get("git")
    if (
        manifest.get("config_hash") != document["config_hash"]
        or not isinstance(manifest_git, Mapping)
        or manifest_git.get("sha") != document["scientific_git_sha"]
    ):
        raise StrongReplayError("checkpoint inventory manifest lineage drift")
    checkpoint_values = document["checkpoints"]
    if not isinstance(checkpoint_values, list) or not checkpoint_values:
        raise StrongReplayError("checkpoint inventory checkpoints are invalid")
    checkpoints: list[CheckpointInventory] = []
    for value in checkpoint_values:
        if not isinstance(value, Mapping):
            raise StrongReplayError("checkpoint inventory checkpoint entry is invalid")
        try:
            raw = dict(value)
            aliases = raw.get("aliases")
            if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
                raise TypeError("aliases")
            raw["aliases"] = tuple(aliases)
            item = CheckpointInventory(**raw)
        except TypeError as exc:
            raise StrongReplayError("checkpoint inventory checkpoint schema is invalid") from exc
        if (
            item.run_id != run_id
            or not item.periodic_last
            or item.config_hash != document["config_hash"]
            or item.scientific_git_sha != document["scientific_git_sha"]
        ):
            raise StrongReplayError("checkpoint inventory checkpoint lineage mismatch")
        checkpoints.append(item)
    return tuple(checkpoints)


def validate_selected_checkpoint_bytes(checkpoints: Sequence[CheckpointInventory]) -> None:
    """Re-hash only replayed checkpoint bytes after inventory reuse, never the full bundle."""
    for item in checkpoints:
        if sha256_file(Path(item.path)) != item.sha256:
            raise StrongReplayError("selected checkpoint bytes drifted from hash-bound inventory")


def stable_id_class_universe(rows: Sequence[Mapping[str, Any]], *, expected_count: int) -> dict[str, Any]:
    """Validate the exact sparse source-ID/class universe for one replay epoch."""
    if len(rows) != expected_count:
        raise StrongReplayError("strong replay stable universe row count mismatch")
    pairs: list[dict[str, int]] = []
    sample_ids: set[int] = set()
    for row in rows:
        sample_id, class_id = row.get("sample_id"), row.get("class_id")
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or row.get("namespace") != "train"
            or sample_id in sample_ids
        ):
            raise StrongReplayError("strong replay stable-ID/class universe is invalid")
        sample_ids.add(sample_id)
        pairs.append({"sample_id": sample_id, "class_id": class_id})
    pairs.sort(key=lambda pair: pair["sample_id"])
    digest = hashlib.sha256(canonical_json(pairs)).hexdigest()
    if digest != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256:
        raise StrongReplayError("strong replay stable-ID/class universe hash drift")
    return {"count": expected_count, "sha256": digest}


def validate_epoch_universes(results: Sequence[StrongReplayResult], *, expected_count: int) -> dict[str, Any]:
    """Require every selected epoch to expose precisely the same stable source universe."""
    reference: tuple[tuple[int, int], ...] | None = None
    identity: dict[str, Any] | None = None
    for result in results:
        current = stable_id_class_universe(result.rows, expected_count=expected_count)
        pairs = tuple(sorted((int(row["sample_id"]), int(row["class_id"])) for row in result.rows))
        if reference is not None and pairs != reference:
            raise StrongReplayError("strong replay stable-ID/class universe changed across epochs")
        reference = pairs
        identity = current
    if identity is None:
        raise StrongReplayError("strong replay requires at least one epoch result")
    return identity


def classification_primitives(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute FP32 per-sample probability and logit margins without reduction."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise StrongReplayError("classification logits/labels have incompatible shapes")
    logits = logits.detach().float()
    probabilities = F.softmax(logits, dim=1)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    masked_probabilities = probabilities.clone()
    masked_logits = logits.clone()
    masked_probabilities.scatter_(1, labels[:, None], float("-inf"))
    masked_logits.scatter_(1, labels[:, None], float("-inf"))
    max_wrong_probability = masked_probabilities.max(dim=1).values
    max_wrong_logit = masked_logits.max(dim=1).values
    probability_margin = true_probability - max_wrong_probability
    logit_margin = true_logit - max_wrong_logit
    correct = logits.argmax(dim=1).eq(labels)
    values = (probabilities, true_probability, probability_margin, logit_margin)
    if not all(torch.isfinite(value).all() for value in values):
        raise StrongReplayError("classification primitives must be finite")
    if bool((probability_margin < -1.0 - 1e-6).any()) or bool((probability_margin > 1.0 + 1e-6).any()):
        raise StrongReplayError("probability margin is outside [-1, 1]")
    return {
        "probabilities": probabilities,
        "true_probability": true_probability,
        "probability_margin": probability_margin,
        "logit_margin": logit_margin,
        "correct": correct,
        "prediction": logits.argmax(dim=1),
    }


def jensen_shannon(clean_probabilities: torch.Tensor, adversarial_probabilities: torch.Tensor) -> torch.Tensor:
    """Return per-sample JS(clean teacher || adversarial teacher), in nats."""
    clean = clean_probabilities.detach().float()
    adversarial = adversarial_probabilities.detach().float()
    if clean.shape != adversarial.shape or clean.ndim != 2:
        raise StrongReplayError("JS requires matching [batch, classes] probability tensors")
    if not torch.isfinite(clean).all() or not torch.isfinite(adversarial).all():
        raise StrongReplayError("JS probabilities must be finite")
    if bool((clean < 0).any()) or bool((adversarial < 0).any()):
        raise StrongReplayError("JS probabilities must be non-negative")
    if not torch.allclose(clean.sum(dim=1), torch.ones_like(clean[:, 0]), rtol=0.0, atol=1e-5) or not torch.allclose(
        adversarial.sum(dim=1), torch.ones_like(adversarial[:, 0]), rtol=0.0, atol=1e-5
    ):
        raise StrongReplayError("JS probabilities must sum to one")
    midpoint = 0.5 * (clean + adversarial)
    clean_log = clean.clamp_min(torch.finfo(clean.dtype).tiny).log()
    adversarial_log = adversarial.clamp_min(torch.finfo(adversarial.dtype).tiny).log()
    midpoint_log = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    result = 0.5 * (
        (clean * (clean_log - midpoint_log)).sum(dim=1) + (adversarial * (adversarial_log - midpoint_log)).sum(dim=1)
    )
    if not torch.isfinite(result).all() or bool((result < -1e-6).any()):
        raise StrongReplayError("JS must be finite and non-negative")
    return result.clamp_min(0.0)


def replay_checkpoint_rows(
    *,
    checkpoint: CheckpointInventory,
    training_config: ExperimentConfig,
    teacher: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    attack_seed_base: int,
) -> StrongReplayResult:
    """Replay one immutable checkpoint under CE-PGD20 and emit frozen primitives."""
    attack_config = selection_attack_from_training(training_config)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise StrongReplayError("strong replay requires frozen teacher parameters")
    student, _ = load_historical_student(
        checkpoint, config=training_config, device=device, expected_config_hash=checkpoint.config_hash
    )
    teacher.eval()
    student.eval()
    attack = LinfPGD(attack_config)
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(device)
        generator = torch.Generator(device=device).manual_seed(attack_seed_base + 1_000_003 * batch_index)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            student_clean_logits = student(batch.images.float()).detach().float()
            teacher_clean_logits = teacher(batch.images.float()).detach().float()
        result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, result.max_abs_delta)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            student_adv_logits = student(result.adversarial.float()).detach().float()
            teacher_adv_logits = teacher(result.adversarial.float()).detach().float()
            clean = classification_primitives(student_clean_logits, batch.labels)
            adversarial = classification_primitives(student_adv_logits, batch.labels)
            teacher_clean = F.softmax(teacher_clean_logits, dim=1)
            teacher_adversarial = F.softmax(teacher_adv_logits, dim=1)
            teacher_js = jensen_shannon(teacher_clean, teacher_adversarial)
            adversarial_ce = F.cross_entropy(student_adv_logits, batch.labels, reduction="none")
        if not torch.isfinite(adversarial_ce).all():
            raise StrongReplayError("per-sample adversarial CE must be finite")
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise StrongReplayError("strong replay populated a teacher parameter gradient")
        source_pairs = zip(batch.sample_ids.tolist(), batch.labels.tolist(), strict=True)
        for index, (sample_id, class_id) in enumerate(source_pairs):
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": int(sample_id),
                    "class_id": int(class_id),
                    "epoch": checkpoint.epoch,
                    "observation_schema_version": STRONG_REPLAY_SCHEMA_VERSION,
                    "student_robust_correct": bool(adversarial["correct"][index]),
                    "student_adversarial_probability_margin": float(adversarial["probability_margin"][index]),
                    "student_adversarial_logit_margin": float(adversarial["logit_margin"][index]),
                    "student_adversarial_ce": float(adversarial_ce[index]),
                    "student_clean_probability_margin": float(clean["probability_margin"][index]),
                    "student_clean_logit_margin": float(clean["logit_margin"][index]),
                    "student_clean_correct": bool(clean["correct"][index]),
                    "student_clean_to_adversarial_prediction_flip": bool(
                        clean["prediction"][index] != adversarial["prediction"][index]
                    ),
                    "student_clean_to_adversarial_true_probability_delta": float(
                        adversarial["true_probability"][index] - clean["true_probability"][index]
                    ),
                    "student_clean_to_adversarial_probability_margin_delta": float(
                        adversarial["probability_margin"][index] - clean["probability_margin"][index]
                    ),
                    "student_clean_to_adversarial_logit_margin_delta": float(
                        adversarial["logit_margin"][index] - clean["logit_margin"][index]
                    ),
                    "teacher_clean_probabilities": [float(value) for value in teacher_clean[index].tolist()],
                    "teacher_adversarial_probabilities": [
                        float(value) for value in teacher_adversarial[index].tolist()
                    ],
                    "teacher_clean_adversarial_js": float(teacher_js[index]),
                }
            )
        student.zero_grad(set_to_none=True)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise StrongReplayError("replayed checkpoint contains duplicate stable source sample IDs")
    epsilon = attack_config.epsilon_value
    assert epsilon is not None
    if max_abs_delta > epsilon + 1e-7:
        raise StrongReplayError("strong replay violated pixel-space Linf projection")
    return StrongReplayResult(
        epoch=checkpoint.epoch,
        checkpoint_sha256=checkpoint.sha256,
        attack_seed_base=attack_seed_base,
        max_abs_delta=max_abs_delta,
        rows=tuple(rows),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def source_provenance() -> dict[str, Any]:
    """Record tracked source bytes and Git identity; dirty execution is rejected."""
    root = Path(__file__).resolve().parents[3]
    paths = {
        "analysis_module": Path(__file__).resolve(),
        "cli_module": root / "src/ard/cli/ffnr_strong_replay.py",
        "pgd": root / "src/ard/attacks/pgd.py",
        "teacher_risk_replay": root / "src/ard/analysis/teacher_risk_replay.py",
    }
    if any(not path.is_file() for path in paths.values()):
        raise StrongReplayError("strong replay source tree is incomplete")
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative],
            check=True,
            capture_output=True,
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise StrongReplayError("strong replay requires tracked source and readable Git identity") from exc
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha) or dirty:
        raise StrongReplayError("strong replay requires a tracked-clean analysis revision")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": hashes, "source_sha256": _sha256_mapping(hashes)}


def checkpoint_cache_identity(
    *,
    checkpoint: CheckpointInventory,
    attack: AttackConfig,
    seed: int,
    replay_batch_size: int,
    expected_sample_count: int,
    teacher_metadata: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return content-addressed cache identity that cannot collide with schema-v2."""
    if replay_batch_size < 1 or seed < 0 or expected_sample_count < 1:
        raise StrongReplayError("cache identity requires positive batch/sample counts and non-negative seed")
    return {
        "contract": CONTRACT_ID,
        "schema_version": STRONG_REPLAY_SCHEMA_VERSION,
        "checkpoint": asdict(checkpoint),
        "attack_identity": attack.identity(),
        "attack_identity_sha256": attack.identity_sha256(),
        "attack_seed_base": seed,
        "seed_formula": SEED_FORMULA,
        "replay_batch_size": replay_batch_size,
        "expected_sample_count": expected_sample_count,
        "teacher": dict(teacher_metadata),
        "dataset": dict(dataset_identity),
        "runtime": dict(runtime),
        "analysis_provenance": dict(provenance),
    }


def load_cached_checkpoint(*, cache_dir: Path, identity: Mapping[str, Any]) -> StrongReplayResult | None:
    """Load only a byte-bound result whose identity and frozen schema match exactly."""
    digest = _sha256_mapping(dict(identity))
    payload_path = cache_dir / f"{digest}.pt"
    metadata_path = cache_dir / f"{digest}.json"
    if not payload_path.exists() and not metadata_path.exists():
        return None
    if not payload_path.is_file() or not metadata_path.is_file():
        raise StrongReplayError("strong replay cache is partially present")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrongReplayError("strong replay cache metadata is unreadable") from exc
    if not isinstance(metadata, Mapping) or canonical_json(metadata.get("identity")) != canonical_json(identity):
        raise StrongReplayError("strong replay cache identity mismatch")
    expected_payload_hash = _require_sha256(metadata.get("payload_sha256"), name="cached payload_sha256")
    if sha256_file(payload_path) != expected_payload_hash:
        raise StrongReplayError("strong replay cache payload hash mismatch")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch deserialization exposes several exception classes
        raise StrongReplayError("strong replay cache is unreadable") from exc
    if not isinstance(payload, Mapping) or canonical_json(payload.get("identity")) != canonical_json(identity):
        raise StrongReplayError("strong replay cache payload identity mismatch")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise StrongReplayError("strong replay cache payload lacks result")
    rows = result.get("rows")
    if not isinstance(rows, (list, tuple)) or any(not isinstance(row, Mapping) for row in rows):
        raise StrongReplayError("strong replay cache rows are invalid")
    if any(set(row) != set(OBSERVATION_COLUMNS) for row in rows):
        raise StrongReplayError("strong replay cache observation schema mismatch")
    expected_count = identity.get("expected_sample_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise StrongReplayError("strong replay cache identity lacks expected sample count")
    sample_ids = [row.get("sample_id") for row in rows]
    if (
        len(rows) != expected_count
        or any(isinstance(sample_id, bool) or not isinstance(sample_id, int) for sample_id in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or any(row.get("namespace") != "train" for row in rows)
    ):
        raise StrongReplayError("strong replay cache stable-ID row contract mismatch")
    epoch = result.get("epoch")
    checkpoint_sha256 = result.get("checkpoint_sha256")
    attack_seed_base = result.get("attack_seed_base")
    max_abs_delta = result.get("max_abs_delta")
    checkpoint_digest = _require_sha256(checkpoint_sha256, name="cached checkpoint_sha256")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or checkpoint_digest != identity["checkpoint"]["sha256"]
        or isinstance(attack_seed_base, bool)
        or not isinstance(attack_seed_base, int)
        or attack_seed_base != identity.get("attack_seed_base")
        or epoch != identity["checkpoint"]["epoch"]
        or not isinstance(max_abs_delta, float)
        or not math.isfinite(max_abs_delta)
    ):
        raise StrongReplayError("strong replay cached result metadata is invalid")
    attack = identity.get("attack_identity")
    epsilon = attack.get("epsilon_value") if isinstance(attack, Mapping) else None
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or max_abs_delta > float(epsilon) + 1e-7:
        raise StrongReplayError("strong replay cached result projection bound is invalid")
    return StrongReplayResult(
        epoch=epoch,
        checkpoint_sha256=checkpoint_digest,
        attack_seed_base=attack_seed_base,
        max_abs_delta=max_abs_delta,
        rows=tuple(dict(row) for row in rows),
    )


def write_checkpoint_cache(
    *, cache_dir: Path, identity: Mapping[str, Any], result: StrongReplayResult
) -> StrongReplayResult:
    """Atomically persist an immutable cache entry alongside its readable identity."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _sha256_mapping(dict(identity))
    payload_path = cache_dir / f"{digest}.pt"
    metadata_path = cache_dir / f"{digest}.json"
    if payload_path.exists() or metadata_path.exists():
        existing = load_cached_checkpoint(cache_dir=cache_dir, identity=identity)
        if existing is None:
            raise StrongReplayError("strong replay cache unexpectedly disappeared")
        return existing
    temporary_payload = payload_path.with_suffix(".pt.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    torch.save(
        {
            "identity": dict(identity),
            "result": {
                "epoch": result.epoch,
                "checkpoint_sha256": result.checkpoint_sha256,
                "attack_seed_base": result.attack_seed_base,
                "max_abs_delta": result.max_abs_delta,
                "rows": list(result.rows),
            },
        },
        temporary_payload,
    )
    temporary_metadata.write_text(
        json.dumps({"identity": dict(identity), "payload_sha256": sha256_file(temporary_payload)}, sort_keys=True),
        encoding="utf-8",
    )
    temporary_payload.replace(payload_path)
    temporary_metadata.replace(metadata_path)
    return result


def write_outputs(
    *, output_dir: Path, results: Sequence[StrongReplayResult], lineage: Mapping[str, Any]
) -> dict[str, Path]:
    """Write immutable multi-checkpoint observations and full provenance."""
    if output_dir.exists() and any(item.name != "checkpoint-cache" for item in output_dir.iterdir()):
        raise StrongReplayError("strong replay output directory already exists; refusing to overwrite")
    rows = tuple(row for result in results for row in result.rows)
    expected_columns = set(OBSERVATION_COLUMNS)
    if any(set(row) != expected_columns for row in rows):
        raise StrongReplayError("strong replay rows do not match the frozen observation schema")
    observations = write_sample_parquet(rows, output_dir / "strong-observations.parquet")
    envelope = {**dict(lineage), "observations_sha256": sha256_file(observations), "row_count": len(rows)}
    return {"observations": observations, "lineage": _atomic_json(output_dir / "lineage.json", envelope)}


def parse_replay_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small, intentionally non-composable replay launcher config."""
    required = {
        "schema_version",
        "contract",
        "run_id",
        "manifest",
        "checkpoint_inventory",
        "semantic_role",
        "epochs",
        "train_expected_count",
        "replay_batch_size",
        "attack_seed",
        "replay_device_type",
    }
    if set(mapping) != required:
        raise StrongReplayError(f"strong replay config keys must be exactly {sorted(required)}")
    if mapping["schema_version"] != STRONG_REPLAY_SCHEMA_VERSION or mapping["contract"] != CONTRACT_ID:
        raise StrongReplayError("strong replay config schema/contract mismatch")
    if not isinstance(mapping["run_id"], str) or not mapping["run_id"]:
        raise StrongReplayError("strong replay config run_id must be non-empty")
    if mapping["semantic_role"] not in {"feature", "outcome"}:
        raise StrongReplayError("strong replay semantic_role must be feature or outcome")
    epochs = mapping["epochs"]
    if (
        not isinstance(epochs, list)
        or not epochs
        or any(isinstance(item, bool) or not isinstance(item, int) for item in epochs)
    ):
        raise StrongReplayError("strong replay config epochs must be a non-empty integer list")
    for name in ("train_expected_count", "replay_batch_size", "attack_seed"):
        minimum = 1 if name != "attack_seed" else 0
        if isinstance(mapping[name], bool) or not isinstance(mapping[name], int) or mapping[name] < minimum:
            raise StrongReplayError(f"strong replay config {name} is invalid")
    if mapping["replay_device_type"] not in {"cpu", "cuda"}:
        raise StrongReplayError("strong replay config replay_device_type must be cpu or cuda")
    return dict(mapping)
