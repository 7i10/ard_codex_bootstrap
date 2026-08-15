"""Same-step dynamic S3 recovery routing for the registered Chen ERT screen.

The router is deliberately an observation-only object until it returns a
detached binary mask to :class:`ard.engine.Trainer`.  It owns the epoch-80
capture contract and the compact per-epoch state artifacts; it never creates
an attack, changes a model mode, or performs an optimizer update.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from ard.config import load_config
from ard.engine.distributed import gather_objects, get_rank


class DynamicS3RoutingError(RuntimeError):
    """The fixed/dynamic same-step routing contract cannot be proven."""


DynamicS3Arm = Literal["baseline", "fixed", "dynamic"]


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.detach().float(), dim=1)
    true = probabilities.gather(1, labels[:, None]).squeeze(1)
    wrong = probabilities.clone()
    wrong.scatter_(1, labels[:, None], float("-inf"))
    values = true - wrong.max(dim=1).values
    if not bool(torch.isfinite(values).all()):
        raise DynamicS3RoutingError("routing received non-finite probability margins")
    return values


def active_s3_recovery_mask(
    *,
    student_clean_logits: torch.Tensor,
    student_adversarial_logits: torch.Tensor,
    teacher_adversarial_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the pre-update, stop-gradient S3×Teacher-correct predicate."""
    if labels.ndim != 1 or valid_mask.dtype != torch.bool or valid_mask.shape != labels.shape:
        raise DynamicS3RoutingError("routing labels and valid mask must be aligned")
    expected = (labels.shape[0],)
    for name, logits in {
        "student clean": student_clean_logits,
        "student adversarial": student_adversarial_logits,
        "teacher adversarial": teacher_adversarial_logits,
    }.items():
        if logits.ndim != 2 or logits.shape[:1] != expected:
            raise DynamicS3RoutingError(f"{name} logits do not match the routing batch")
    clean_correct = student_clean_logits.detach().argmax(1).eq(labels)
    student_adv_wrong = student_adversarial_logits.detach().argmax(1).ne(labels)
    teacher_adv_correct = teacher_adversarial_logits.detach().argmax(1).eq(labels)
    return (clean_correct & student_adv_wrong & teacher_adv_correct & valid_mask).detach()


def _digest_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def _digest_labels(labels: dict[int, int]) -> str:
    pairs = sorted((int(sample_id), int(label)) for sample_id, label in labels.items())
    return hashlib.sha256(
        json.dumps(pairs, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DynamicS3Decision:
    """Detached same-step state and the actual action applied to this visit."""

    current_active: torch.Tensor
    action_active: torch.Tensor
    capture_action: torch.Tensor


class DynamicS3Router:
    """Capture-once fixed control and current-state dynamic recovery router."""

    def __init__(
        self,
        *,
        arm: DynamicS3Arm,
        train_labels: dict[int, int],
        output_dir: Path,
        capture_epoch: int = 80,
    ) -> None:
        if arm not in {"baseline", "fixed", "dynamic"}:
            raise DynamicS3RoutingError("dynamic S3 arm must be baseline, fixed, or dynamic")
        if capture_epoch < 0 or not train_labels:
            raise DynamicS3RoutingError("routing requires a non-empty exact train ID/label universe")
        self.arm = arm
        self.train_labels = {int(sample_id): int(label) for sample_id, label in train_labels.items()}
        self.output_dir = output_dir
        self.capture_epoch = capture_epoch
        self._pending: list[dict[str, Any]] = []
        self._capture_actions: dict[int, bool] = {}
        self._state_paths: dict[int, Path] = {}
        self.epoch_statistics: dict[int, dict[str, int | float]] = {}

    @property
    def capture_complete(self) -> bool:
        return set(self._capture_actions) == set(self.train_labels)

    @property
    def capture_ids(self) -> tuple[int, ...]:
        return tuple(sorted(sample_id for sample_id, active in self._capture_actions.items() if active))

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint the immutable capture immediately after epoch 80.

        Per-epoch Parquet rows are external immutable artifacts; the compact
        ID-to-action map is the state necessary to resume a fixed arm without
        silently recomputing its historical capture.
        """
        return {
            "schema_version": 1,
            "contract": "ert_dynamic_s3_recovery_v1",
            "arm": self.arm,
            "capture_epoch": self.capture_epoch,
            "train_ids_sha256": _digest_ids(list(self.train_labels)),
            "train_id_label_sha256": _digest_labels(self.train_labels),
            "capture_actions": {str(key): bool(value) for key, value in sorted(self._capture_actions.items())},
            "epoch_statistics": {str(key): value for key, value in sorted(self.epoch_statistics.items())},
            "state_paths": {
                str(epoch): {"path": str(path.resolve()), "sha256": _sha256(path)}
                for epoch, path in sorted(self._state_paths.items())
            },
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if (
            value.get("schema_version") != 1
            or value.get("contract") != "ert_dynamic_s3_recovery_v1"
            or value.get("arm") != self.arm
            or value.get("capture_epoch") != self.capture_epoch
            or value.get("train_ids_sha256") != _digest_ids(list(self.train_labels))
            or value.get("train_id_label_sha256") != _digest_labels(self.train_labels)
        ):
            raise DynamicS3RoutingError("dynamic routing checkpoint state has incompatible lineage")
        raw_actions = value.get("capture_actions")
        if not isinstance(raw_actions, dict):
            raise DynamicS3RoutingError("dynamic routing checkpoint lacks immutable capture actions")
        actions = {int(key): bool(item) for key, item in raw_actions.items()}
        if set(actions) != set(self.train_labels):
            raise DynamicS3RoutingError("dynamic routing checkpoint capture does not cover the exact train universe")
        self._capture_actions = actions
        raw_statistics = value.get("epoch_statistics", {})
        if not isinstance(raw_statistics, dict):
            raise DynamicS3RoutingError("dynamic routing checkpoint statistics are malformed")
        self.epoch_statistics = {int(key): dict(item) for key, item in raw_statistics.items() if isinstance(item, dict)}
        raw_paths = value.get("state_paths", {})
        if not isinstance(raw_paths, dict):
            raise DynamicS3RoutingError("dynamic routing checkpoint state paths are malformed")
        self._state_paths = {
            int(epoch): Path(item["path"])
            for epoch, item in raw_paths.items()
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str)
        }

    def _capture_payload(self) -> dict[str, Any]:
        if not self.capture_complete:
            raise DynamicS3RoutingError("cannot materialize a dynamic routing capture before full coverage")
        selected_ids = list(self.capture_ids)
        class_counts = Counter(self.train_labels[sample_id] for sample_id in selected_ids)
        return {
            "schema_version": 1,
            "contract": "ert_dynamic_s3_recovery_v1",
            "capture_epoch": self.capture_epoch,
            "arm": self.arm,
            "selected_ids": selected_ids,
            "selected_ids_sha256": _digest_ids(selected_ids),
            "selected_count": len(selected_ids),
            "selected_class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
            "train_universe_count": len(self.train_labels),
            "train_id_label_sha256": _digest_labels(self.train_labels),
            "capture_epoch_statistics": self.epoch_statistics.get(self.capture_epoch, {}),
        }

    def _write_capture(self) -> Path:
        capture_path = self.output_dir / "routing-capture-mask.json"
        serialized = json.dumps(self._capture_payload(), indent=2, sort_keys=True) + "\n"
        if capture_path.exists():
            if capture_path.read_text(encoding="utf-8") != serialized:
                raise DynamicS3RoutingError("existing routing capture artifact differs from immutable epoch-80 capture")
            return capture_path
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = capture_path.with_name(f".{capture_path.name}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, capture_path)
        return capture_path

    def observe(
        self,
        *,
        epoch: int,
        sample_ids: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        student_clean_logits: torch.Tensor,
        student_adversarial_logits: torch.Tensor,
        teacher_clean_logits: torch.Tensor,
        teacher_adversarial_logits: torch.Tensor,
    ) -> DynamicS3Decision:
        if epoch < self.capture_epoch:
            raise DynamicS3RoutingError("same-step routing cannot run before its capture epoch")
        if sample_ids.ndim != 1 or sample_ids.shape != labels.shape:
            raise DynamicS3RoutingError("routing stable IDs must align with labels")
        current = active_s3_recovery_mask(
            student_clean_logits=student_clean_logits,
            student_adversarial_logits=student_adversarial_logits,
            teacher_adversarial_logits=teacher_adversarial_logits,
            labels=labels,
            valid_mask=valid_mask,
        )
        ids = [int(value) for value in sample_ids.detach().cpu().tolist()]
        valid = [bool(value) for value in valid_mask.detach().cpu().tolist()]
        if epoch == self.capture_epoch:
            capture = current
        else:
            if not self.capture_complete:
                raise DynamicS3RoutingError("fixed/dynamic routing used before exact epoch-80 capture completed")
            capture = torch.tensor(
                [self._capture_actions.get(sample_id, False) for sample_id in ids],
                dtype=torch.bool,
                device=sample_ids.device,
            ) & valid_mask
        if self.arm == "baseline":
            action = torch.zeros_like(current)
        elif self.arm == "fixed" and epoch > self.capture_epoch:
            action = capture
        else:
            action = current
        student_clean_margin = _margin(student_clean_logits, labels)
        student_adv_margin = _margin(student_adversarial_logits, labels)
        teacher_clean_margin = _margin(teacher_clean_logits, labels)
        teacher_adv_margin = _margin(teacher_adversarial_logits, labels)
        values = {
            "student_clean_correct": student_clean_logits.detach().argmax(1).eq(labels),
            "student_adv_correct": student_adversarial_logits.detach().argmax(1).eq(labels),
            "teacher_clean_correct": teacher_clean_logits.detach().argmax(1).eq(labels),
            "teacher_adv_correct": teacher_adversarial_logits.detach().argmax(1).eq(labels),
            "mS_clean": student_clean_margin,
            "mS_adv": student_adv_margin,
            "mT_clean": teacher_clean_margin,
            "mT_adv": teacher_adv_margin,
            "current_active": current,
            "action_active": action,
            "capture_action_if_available": capture,
        }
        cpu_values = {name: value.detach().cpu().tolist() for name, value in values.items()}
        classes = [int(value) for value in labels.detach().cpu().tolist()]
        for position, sample_id in enumerate(ids):
            if not valid[position]:
                continue
            expected = self.train_labels.get(sample_id)
            if expected is None or expected != classes[position]:
                raise DynamicS3RoutingError("routing observed an ID/label outside the exact train universe")
            self._pending.append(
                {
                    "epoch": int(epoch),
                    "sample_id": sample_id,
                    "class_id": classes[position],
                    **{
                        key: (float(values_at[position]) if key.startswith("m") else bool(values_at[position]))
                        for key, values_at in cpu_values.items()
                    },
                    "DeltaS": float(cpu_values["mS_clean"][position] - cpu_values["mS_adv"][position]),
                    "DeltaT": float(cpu_values["mT_clean"][position] - cpu_values["mT_adv"][position]),
                    "rank": get_rank(),
                    "order": len(self._pending),
                }
            )
        return DynamicS3Decision(current_active=current, action_active=action, capture_action=capture)

    def flush_epoch(self, epoch: int) -> None:
        rows = [row for rank_rows in gather_objects(self._pending) for row in rank_rows]
        self._pending = []
        by_id: dict[int, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda item: (int(item["sample_id"]), int(item["rank"]), int(item["order"]))):
            sample_id = int(row["sample_id"])
            if sample_id in by_id:
                raise DynamicS3RoutingError(f"routing observed duplicate valid stable ID in epoch {epoch}: {sample_id}")
            by_id[sample_id] = row
        if set(by_id) != set(self.train_labels):
            missing = len(set(self.train_labels) - set(by_id))
            unexpected = len(set(by_id) - set(self.train_labels))
            raise DynamicS3RoutingError(
                "routing epoch "
                f"{epoch} does not cover the exact train universe (missing={missing}, unexpected={unexpected})"
            )
        if epoch == self.capture_epoch:
            self._capture_actions = {sample_id: bool(row["current_active"]) for sample_id, row in by_id.items()}
            if not self.capture_complete:
                raise DynamicS3RoutingError("epoch-80 capture did not cover each train stable ID exactly once")
        active = sum(bool(row["action_active"]) for row in by_id.values())
        current = sum(bool(row["current_active"]) for row in by_id.values())
        self.epoch_statistics[epoch] = {
            "active_count": active,
            "active_fraction": active / len(by_id),
            "current_active_count": current,
            "current_active_fraction": current / len(by_id),
            "unique_ids": len(by_id),
        }
        if epoch == self.capture_epoch and get_rank() == 0:
            self._write_capture()
        if get_rank() != 0:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise DynamicS3RoutingError("dynamic routing state artifacts require pyarrow") from exc
        state_dir = self.output_dir / "dynamic-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"epoch-{epoch}.parquet"
        if path.exists():
            raise DynamicS3RoutingError(f"refusing to overwrite dynamic state artifact: {path}")
        public_rows = [
            {key: value for key, value in row.items() if key not in {"rank", "order"}}
            for row in by_id.values()
        ]
        pq.write_table(pa.Table.from_pylist(public_rows), path, compression="zstd")
        self._state_paths[epoch] = path

    def finalize(self) -> dict[str, Any]:
        if not self.capture_complete:
            raise DynamicS3RoutingError("cannot finalize dynamic routing without an immutable full capture")
        capture = self._capture_payload()
        if get_rank() != 0:
            return capture
        capture_path = self._write_capture()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise DynamicS3RoutingError("dynamic routing state artifacts require pyarrow") from exc
        paths = [self._state_paths[epoch] for epoch in sorted(self._state_paths)]
        if not paths:
            raise DynamicS3RoutingError("dynamic routing wrote no state rows")
        combined_path = self.output_dir / "dynamic-state.parquet"
        if combined_path.exists():
            raise DynamicS3RoutingError(f"refusing to overwrite dynamic state artifact: {combined_path}")
        pq.write_table(pa.concat_tables([pq.read_table(path) for path in paths]), combined_path, compression="zstd")
        manifest = {
            "schema_version": 1,
            "contract": "ert_dynamic_s3_recovery_v1",
            "capture": {"path": str(capture_path.resolve()), "sha256": _sha256(capture_path)},
            "state": {
                "path": str(combined_path.resolve()),
                "sha256": _sha256(combined_path),
                "epochs": {
                    str(epoch): {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for epoch, path in sorted(self._state_paths.items())
                },
            },
        }
        manifest_path = self.output_dir / "dynamic-state-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**capture, "artifacts": manifest, "manifest_path": str(manifest_path.resolve())}


def run_dynamic_s3_arm(
    *,
    config_path: Path,
    run_key: str,
    arm: str,
    output_dir: Path,
    calibration_path: Path,
    device: torch.device,
    peer_epoch80_state: Path | None = None,
) -> dict[str, Any]:
    """Launch one immutable-parent dynamic-S3 arm through the shared trainer.

    The Stage-A composition helper is reused solely for complete epoch-boundary
    resume, tracking, and horizon checkpointing.  Dynamic selection is supplied
    through the engine hook above; no historical fixed mask is imported.
    """
    from ard.analysis.ert_stage_a_runtime import StageARuntimeError, StageATreatment, run_stage_a_arm

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != "ert_dynamic_s3_recovery_v1":
        raise DynamicS3RoutingError("dynamic S3 config does not carry the frozen contract")
    runs = payload.get("runs")
    if not isinstance(runs, dict) or run_key not in runs or not isinstance(runs[run_key], dict):
        raise DynamicS3RoutingError("dynamic S3 run key is missing")
    dynamic_arms = {"DYNBASE": "baseline", "S3FIX075": "fixed", "S3DYN075": "dynamic"}
    if arm not in dynamic_arms:
        raise DynamicS3RoutingError("dynamic S3 arm must be DYNBASE, S3FIX075, or S3DYN075")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise DynamicS3RoutingError("calibration artifact must be a JSON object")
    calibration["artifact_sha256"] = _sha256(calibration_path)
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, dict) or coefficients.get("beta_advce") != 0.075:
        raise DynamicS3RoutingError("dynamic S3 config must freeze beta_advce=0.075")
    run = runs[run_key]
    training_contract = payload.get("training_attack")
    endpoint_contract = payload.get("endpoint_attack")
    parent_config = load_config(Path(run["parent_config"]))
    if not isinstance(training_contract, dict) or training_contract != parent_config.method.attack.identity():
        raise DynamicS3RoutingError("dynamic S3 training attack contract does not exactly match the parent")
    if (
        not isinstance(endpoint_contract, dict)
        or parent_config.method.selection_attack is None
        or endpoint_contract != parent_config.method.selection_attack.identity()
    ):
        raise DynamicS3RoutingError("dynamic S3 endpoint attack contract does not exactly match the parent")
    try:
        result = run_stage_a_arm(
            parent_config_path=Path(run["parent_config"]),
            parent_checkpoint=Path(run["parent_checkpoint"]),
            mask_path=None,
            output_dir=output_dir,
            treatment=StageATreatment(arm=arm, mask_key=None, kind="baseline"),
            calibration=calibration,
            device=device,
            end_epoch=int(payload["end_epoch"]),
            horizon_epochs=tuple(int(epoch) for epoch in payload["horizons"]),
            run_namespace="dynamic-s3-recovery",
            dynamic_s3_arm=dynamic_arms[arm],
            dynamic_s3_beta_advce=float(coefficients["beta_advce"]),
            dynamic_s3_attack_contract=training_contract,
            dynamic_s3_endpoint_contract=endpoint_contract,
            dynamic_s3_peer_epoch80_state=peer_epoch80_state,
        )
    except StageARuntimeError as exc:
        raise DynamicS3RoutingError(str(exc)) from exc
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DynamicS3RoutingError(f"expected JSON object: {path}")
    return value


def _state_name(row: dict[str, Any]) -> str:
    if not bool(row["student_clean_correct"]):
        return "CW"
    if bool(row["student_adv_correct"]):
        return "AC"
    return "S3_TC" if bool(row["teacher_adv_correct"]) else "S3_TW"


def _transition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    outgoing: Counter[str] = Counter()
    previous: dict[int, tuple[int, str, bool]] = {}
    actions: dict[int, list[bool]] = {}
    active_count = Counter()
    current_count = Counter()
    epoch_size = Counter()
    for row in sorted(rows, key=lambda item: (int(item["epoch"]), int(item["sample_id"]))):
        epoch, sample_id, state = int(row["epoch"]), int(row["sample_id"]), _state_name(row)
        action = bool(row["action_active"])
        prior = previous.get(sample_id)
        if prior is not None:
            _, prior_state, _ = prior
            transitions[f"{prior_state}->{state}"] += 1
            outgoing[prior_state] += 1
        previous[sample_id] = (epoch, state, action)
        actions.setdefault(sample_id, []).append(action)
        active_count[epoch] += int(action)
        current_count[epoch] += int(bool(row["current_active"]))
        epoch_size[epoch] += 1
    entries = exits = reentries = switches = short_cycle_reentries = 0
    for sequence in actions.values():
        had_active = bool(sequence and sequence[0])
        for index in range(1, len(sequence)):
            before, after = sequence[index - 1], sequence[index]
            switches += int(before != after)
            entries += int(not before and after)
            exits += int(before and not after)
            if not before and after and had_active:
                reentries += 1
            if index >= 2 and sequence[index - 2] and not before and after:
                short_cycle_reentries += 1
            had_active = had_active or after
    recovery_numerator = transitions["S3_TC->AC"]
    recovery_denominator = outgoing["S3_TC"]
    relapse_numerator = transitions["AC->S3_TC"] + transitions["AC->S3_TW"]
    relapse_denominator = outgoing["AC"]
    return {
        "transitions": dict(sorted(transitions.items())),
        "transition_rates": {
            name: count / outgoing[name.split("->", 1)[0]]
            for name, count in sorted(transitions.items())
            if outgoing[name.split("->", 1)[0]]
        },
        "recovery": {
            "numerator": recovery_numerator,
            "denominator": recovery_denominator,
            "rate": recovery_numerator / recovery_denominator if recovery_denominator else None,
        },
        "relapse": {
            "numerator": relapse_numerator,
            "denominator": relapse_denominator,
            "rate": relapse_numerator / relapse_denominator if relapse_denominator else None,
        },
        "action_events": {
            "entries": entries,
            "exits": exits,
            "reentries": reentries,
            "switches": switches,
            "short_cycle_reentries": short_cycle_reentries,
        },
        "active_count_by_epoch": {str(key): value for key, value in sorted(active_count.items())},
        "active_fraction_by_epoch": {str(key): active_count[key] / epoch_size[key] for key in sorted(epoch_size)},
        "current_active_count_by_epoch": {str(key): value for key, value in sorted(current_count.items())},
    }


def _validate_endpoint_binding(
    *,
    metadata: dict[str, Any],
    rows_path: Path,
    checkpoint: Path,
    manifest: dict[str, Any],
    arm: str,
    dynamic_arm: str,
    seed: str,
    horizon: int,
    split: str,
    expected_attack: dict[str, object],
    expected_training_seed: int,
    expected_parent_sha256: str,
) -> None:
    if (
        metadata.get("checkpoint_epoch") != horizon
        or metadata.get("dataset_scope") != split
        or metadata.get("rows_sha256") != _sha256(rows_path)
        or metadata.get("checkpoint_sha256") != _sha256(checkpoint)
        or Path(str(metadata.get("checkpoint", ""))).resolve() != checkpoint.resolve()
        or metadata.get("attack") != expected_attack
        or not isinstance(metadata.get("source_git_sha"), str)
    ):
        raise DynamicS3RoutingError(f"endpoint metadata binding failed: {seed}/{arm}/epoch-{horizon}/{split}")
    fork = manifest.get("fork_lineage")
    if (
        not isinstance(fork, dict)
        or fork.get("arm") != arm
        or not isinstance(fork.get("dynamic_s3"), dict)
        or fork["dynamic_s3"].get("arm") != dynamic_arm
        or manifest.get("config_hash") != fork.get("child_config_hash")
        or manifest.get("training_seed") != expected_training_seed
        or fork.get("parent_checkpoint_sha256") != expected_parent_sha256
        or not isinstance(manifest.get("git", {}).get("sha"), str)
    ):
        raise DynamicS3RoutingError(f"training manifest lineage failed: {seed}/{arm}")
    artifacts = manifest.get("artifacts", [])
    checkpoint_sha = _sha256(checkpoint)
    if not any(
        isinstance(item, dict)
        and item.get("sha256") == checkpoint_sha
        and item.get("aliases") == [f"epoch-{horizon}"]
        for item in artifacts
    ):
        raise DynamicS3RoutingError(f"endpoint checkpoint is not attested by training manifest: {seed}/{arm}/{horizon}")


def build_dynamic_s3_report(
    *, endpoint_root: Path, training_root: Path, config_path: Path, output: Path
) -> dict[str, Any]:
    """Collect pre-registered endpoint effects and route transitions.

    Endpoint rows are generated by the existing independent CE-PGD20 CLI.  The
    report deliberately refuses partial arm/horizon inputs, preventing a
    post-hoc best-horizon comparison from masquerading as the screen result.
    """
    from ard.analysis.ert_stage_a_effect_decomposition import _bootstrap_ci, _paired_arrays, _paired_metrics, _rows
    from ard.tracking.adapter import collect_git_state

    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise DynamicS3RoutingError("dynamic S3 report requires a clean source tree")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") != "ert_dynamic_s3_recovery_v1":
        raise DynamicS3RoutingError("dynamic S3 report has the wrong config contract")
    arms = tuple(config.get("arms", ()))
    horizons = tuple(int(epoch) for epoch in config.get("horizons", ()))
    if arms != ("DYNBASE", "S3FIX075", "S3DYN075") or horizons != (84, 89, 94):
        raise DynamicS3RoutingError("dynamic S3 report requires the frozen three-arm 84/89/94 screen")
    endpoint_contract = config.get("endpoint_attack")
    if not isinstance(endpoint_contract, dict):
        raise DynamicS3RoutingError("dynamic S3 report requires the complete endpoint attack contract")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise DynamicS3RoutingError("dynamic routing report requires pyarrow") from exc
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_dynamic_s3_recovery_report_v1",
        "source_git_sha": source["sha"],
        "config_sha256": _sha256(config_path),
        "attack": None,
        "seeds": {},
    }
    for seed_index, seed in enumerate(("L2", "L4")):
        dynamic_arm_names = {"DYNBASE": "baseline", "S3FIX075": "fixed", "S3DYN075": "dynamic"}
        run_config = config.get("runs", {}).get(seed)
        if not isinstance(run_config, dict) or not isinstance(run_config.get("seed"), int):
            raise DynamicS3RoutingError(f"dynamic S3 config lacks run seed: {seed}")
        parent_checkpoint = Path(str(run_config.get("parent_checkpoint", "")))
        if not parent_checkpoint.is_file():
            raise DynamicS3RoutingError(f"dynamic S3 parent checkpoint is missing: {seed}")
        parent_sha256 = _sha256(parent_checkpoint)
        capture_paths = {arm: training_root / seed / arm / "routing-capture-mask.json" for arm in arms}
        captures = {arm: _read_json(path) for arm, path in capture_paths.items()}
        fixed_digest = captures["S3FIX075"].get("selected_ids_sha256")
        dynamic_digest = captures["S3DYN075"].get("selected_ids_sha256")
        if fixed_digest != dynamic_digest:
            raise DynamicS3RoutingError(f"epoch-80 fixed/dynamic capture mismatch: {seed}")
        fixed_epoch80 = _read_json(training_root / seed / "S3FIX075" / "epoch80-routing-state.json")
        dynamic_epoch80 = _read_json(training_root / seed / "S3DYN075" / "epoch80-routing-state.json")
        if fixed_epoch80.get("components") != dynamic_epoch80.get("components"):
            raise DynamicS3RoutingError(f"epoch-80 fixed/dynamic model/optimizer/scheduler mismatch: {seed}")
        selected = {int(item) for item in captures["S3FIX075"].get("selected_ids", [])}
        state_summary: dict[str, Any] = {}
        state_label_maps: dict[str, dict[int, int]] = {}
        for arm in arms:
            path = training_root / seed / arm / "dynamic-state.parquet"
            if not path.is_file():
                raise DynamicS3RoutingError(f"missing routing state artifact: {path}")
            rows = pq.read_table(path).to_pylist()
            state_label_maps[arm] = {int(row["sample_id"]): int(row["class_id"]) for row in rows}
            if _digest_labels(state_label_maps[arm]) != captures[arm].get("train_id_label_sha256"):
                raise DynamicS3RoutingError(f"routing-state class map does not match capture lineage: {seed}/{arm}")
            state_summary[arm] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "row_count": len(rows),
                **_transition_summary(rows),
            }
        seed_result: dict[str, Any] = {
            "capture": {
                "selected_count": len(selected),
                "selected_ids_sha256": fixed_digest,
                "paths": {
                    arm: {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for arm, path in capture_paths.items()
                },
            },
            "epoch80_equivalence": fixed_epoch80,
            "state": state_summary,
            "horizons": {},
        }
        for horizon in horizons:
            endpoint_rows: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
            endpoint_meta: dict[tuple[str, str], dict[str, Any]] = {}
            for arm in arms:
                for split in ("train", "validation"):
                    directory = endpoint_root / seed / arm / f"epoch-{horizon}" / split
                    metadata = _read_json(directory / "endpoint.json")
                    rows_path = directory / "endpoint-sample-stats.parquet"
                    checkpoint = training_root / seed / arm / "checkpoints" / f"epoch-{horizon}.pt"
                    manifest = _read_json(training_root / seed / arm / "run-bundle" / "manifest.json")
                    _validate_endpoint_binding(
                        metadata=metadata,
                        rows_path=rows_path,
                        checkpoint=checkpoint,
                        manifest=manifest,
                        arm=arm,
                        dynamic_arm=dynamic_arm_names[arm],
                        seed=seed,
                        horizon=horizon,
                        split=split,
                        expected_attack=endpoint_contract,
                        expected_training_seed=int(run_config["seed"]),
                        expected_parent_sha256=parent_sha256,
                    )
                    attack = metadata.get("attack_identity_sha256")
                    if not isinstance(attack, str):
                        raise DynamicS3RoutingError(f"endpoint attack identity missing: {directory}")
                    if report["attack"] is None:
                        report["attack"] = {"identity": metadata.get("attack"), "identity_sha256": attack}
                    elif report["attack"]["identity_sha256"] != attack:
                        raise DynamicS3RoutingError("endpoint attack identity differs between dynamic S3 arms")
                    endpoint_rows[(arm, split)] = _rows(rows_path)
                    endpoint_meta[(arm, split)] = metadata
            train_base = endpoint_rows[("DYNBASE", "train")]
            validation_base = endpoint_rows[("DYNBASE", "validation")]
            if set(train_base) & set(validation_base) or len(train_base) != 45000 or len(validation_base) != 5000:
                raise DynamicS3RoutingError(f"train/validation identity is invalid: {seed}/{horizon}")
            horizon_result: dict[str, Any] = {"arms": {}}
            for arm in arms:
                train = endpoint_rows[(arm, "train")]
                validation = endpoint_rows[(arm, "validation")]
                if set(train) != set(train_base) or set(validation) != set(validation_base):
                    raise DynamicS3RoutingError(f"endpoint stable-ID mismatch: {seed}/{horizon}/{arm}")
                if any(
                    int(train[sample_id]["true_label"]) != int(train_base[sample_id]["true_label"])
                    for sample_id in train
                ):
                    raise DynamicS3RoutingError(f"endpoint train label mismatch: {seed}/{horizon}/{arm}")
                if any(
                    int(validation[sample_id]["true_label"]) != int(validation_base[sample_id]["true_label"])
                    for sample_id in validation
                ):
                    raise DynamicS3RoutingError(f"endpoint validation label mismatch: {seed}/{horizon}/{arm}")
                if state_label_maps[arm] != {sample_id: int(row["true_label"]) for sample_id, row in train.items()}:
                    raise DynamicS3RoutingError(f"routing-state/train endpoint class map mismatch: {seed}/{arm}")
                if arm == "DYNBASE":
                    horizon_result["arms"][arm] = {
                        "heldout_clean_accuracy": endpoint_meta[(arm, "validation")]["clean_accuracy"],
                        "heldout_robust_accuracy": endpoint_meta[(arm, "validation")]["robust_accuracy"],
                    }
                    continue
                direct = _paired_arrays(train_base, train, selected)
                heldout = _paired_arrays(validation_base, validation, set(validation_base))
                base_seed = 921_337 + seed_index * 10_000 + horizon * 10 + (1 if arm == "S3FIX075" else 2)
                horizon_result["arms"][arm] = {
                    "heldout_clean_accuracy": endpoint_meta[(arm, "validation")]["clean_accuracy"],
                    "heldout_robust_accuracy": endpoint_meta[(arm, "validation")]["robust_accuracy"],
                    "capture_direct": {**_paired_metrics(direct), **_bootstrap_ci(direct, seed=base_seed)},
                    "heldout": {**_paired_metrics(heldout), **_bootstrap_ci(heldout, seed=base_seed + 1)},
                }
            fixed_train, dynamic_train = endpoint_rows[("S3FIX075", "train")], endpoint_rows[("S3DYN075", "train")]
            fixed_validation, dynamic_validation = (
                endpoint_rows[("S3FIX075", "validation")],
                endpoint_rows[("S3DYN075", "validation")],
            )
            fixed_dynamic_seed = 921_337 + seed_index * 10_000 + horizon * 10 + 7
            horizon_result["dynamic_vs_fixed"] = {
                "capture_direct": {
                    **_paired_metrics(_paired_arrays(fixed_train, dynamic_train, selected)),
                    **_bootstrap_ci(_paired_arrays(fixed_train, dynamic_train, selected), seed=fixed_dynamic_seed),
                },
                "heldout": {
                    **_paired_metrics(_paired_arrays(fixed_validation, dynamic_validation, set(fixed_validation))),
                    **_bootstrap_ci(
                        _paired_arrays(fixed_validation, dynamic_validation, set(fixed_validation)),
                        seed=fixed_dynamic_seed + 1,
                    ),
                },
            }
            seed_result["horizons"][str(horizon)] = horizon_result
        report["seeds"][seed] = seed_result
    if output.exists():
        raise DynamicS3RoutingError(f"refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_sha256"] = _sha256(output)
    return report
