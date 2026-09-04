"""Current-state Online-S2×T1 routing for the I100 preservation screen.

This module owns only detached state observation, immutable threshold binding,
and local trajectory artifacts.  It never constructs an attack, changes a
model mode, or performs an optimizer update.  The shared :class:`Trainer`
consumes its hard ``action_active`` mask before the outer boundary loss.

The router deliberately distinguishes two quantities:

* routing states use each model's *global* true-vs-strongest-nontrue logit
  margin; and
* the existing PMP/D-BDD loss continues to select the Student's current rival
  and reuse that detached rival for the Teacher.

They are related, but they are not interchangeable scientific definitions.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from ard.engine.distributed import gather_objects, get_rank


class OnlineStateS2RoutingError(RuntimeError):
    """The frozen Online-S2×T1 routing contract cannot be proven."""


OnlineStateS2Arm = Literal["prefix", "control", "pmp", "dbdp"]

_BRANCHES = frozenset({"CW", "S3", "S1", "S2_T1", "S2_T2", "S2_T3"})
_REQUIRED_TRANSITIONS = (
    "S1->S2_T1",
    "S2_T1->S1",
    "S2_T1->S3",
    "S2_T1->CW",
    "S3->S2_T1",
    "CW->S2_T1",
    "S2_T2->S2_T1",
    "S2_T3->S2_T1",
    "S2_T1->S2_T2",
    "S2_T1->S2_T3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_ids(ids: Mapping[int, int] | list[int] | tuple[int, ...]) -> str:
    values = sorted(int(item) for item in (ids if not isinstance(ids, Mapping) else ids.keys()))
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _digest_labels(labels: Mapping[int, int]) -> str:
    pairs = sorted((int(sample_id), int(label)) for sample_id, label in labels.items())
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def global_logit_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return detached ``z_y - max(z_nontrue)`` in finite FP32 math."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise OnlineStateS2RoutingError("global logit margin requires [batch, classes] logits and labels")
    if labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise OnlineStateS2RoutingError("global logit margin labels must be integer class indices")
    detached = logits.detach().float()
    if bool((labels < 0).any()) or bool((labels >= detached.shape[1]).any()):
        raise OnlineStateS2RoutingError("routing labels are outside the class range")
    true = detached.gather(1, labels[:, None]).squeeze(1)
    nontrue = detached.clone()
    nontrue.scatter_(1, labels[:, None], float("-inf"))
    margin = true - nontrue.amax(dim=1)
    if not bool(torch.isfinite(margin).all()):
        raise OnlineStateS2RoutingError("routing received non-finite global logit margins")
    return margin.detach()


def q10_linear(values: list[float]) -> float:
    """The frozen CPU float64 linear q10 convention.

    State membership remains the preregistered scalar predicate ``margin <=
    tau``.  Consequently, all values tied at the interpolated threshold are
    included; the artifact records the resulting count rather than silently
    imposing a stable-ID top-k tie breaker that would change the state
    definition.
    """
    if not values:
        raise OnlineStateS2RoutingError("q10 requires a non-empty positive-margin population")
    tensor = torch.tensor(values, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise OnlineStateS2RoutingError("q10 population contains a non-finite margin")
    return float(torch.quantile(tensor, 0.10, interpolation="linear").item())


def _threshold_population(rows: Mapping[int, Mapping[str, Any]], *, model: str) -> list[float]:
    if model not in {"student", "teacher"}:
        raise ValueError("model must be student or teacher")
    correct = f"{model}_adv_correct"
    margin = f"{model}_global_margin"
    values = [float(row[margin]) for row in rows.values() if bool(row[correct]) and float(row[margin]) > 0.0]
    if not values:
        raise OnlineStateS2RoutingError(f"epoch-100 has no positive adversarial-correct {model} margins")
    return values


def _threshold_summary(rows: Mapping[int, Mapping[str, Any]], *, model: str) -> dict[str, Any]:
    values = _threshold_population(rows, model=model)
    threshold = q10_linear(values)
    return {
        "quantile": 0.10,
        "quantile_method": "torch_quantile_linear_float64_cpu",
        "positive_adversarial_correct_count": len(values),
        "threshold": threshold,
        "at_or_below_threshold_count": sum(value <= threshold for value in values),
        "strictly_below_threshold_count": sum(value < threshold for value in values),
        "equal_threshold_count": sum(value == threshold for value in values),
        "tie_membership": "all positive margins <= scalar threshold",
    }


@dataclass(frozen=True)
class OnlineStateS2Decision:
    """Detached current routing state returned to the Trainer."""

    eligible_s2_t1: torch.Tensor
    action_active: torch.Tensor


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OnlineStateS2RoutingError(f"expected a JSON object: {path}")
    return value


class OnlineStateS2Router:
    """Checkpointable current-state router for the registered I100 screen."""

    CONTRACT = "ert_rslad_i100_online_state_s2_preservation_v1"
    # Schema v2 adds independent S2×T1 state persistence accounting.  The
    # earlier action-only counters were insufficient for the control arm,
    # where state re-entry can occur even though treatment action is always
    # off.
    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        arm: OnlineStateS2Arm,
        train_labels: Mapping[int, int],
        output_dir: Path,
        original_parent_checkpoint_sha256: str,
        prefix_epoch: int = 100,
        thresholds_path: Path | None = None,
    ) -> None:
        if arm not in {"prefix", "control", "pmp", "dbdp"}:
            raise OnlineStateS2RoutingError("online S2 arm must be prefix, control, pmp, or dbdp")
        if prefix_epoch < 0 or not train_labels:
            raise OnlineStateS2RoutingError("online S2 routing requires a non-empty exact train universe")
        if len(original_parent_checkpoint_sha256) != 64:
            raise OnlineStateS2RoutingError("online S2 routing requires the exact original parent SHA-256")
        self.arm = arm
        self.train_labels = {int(sample_id): int(label) for sample_id, label in train_labels.items()}
        self.output_dir = output_dir
        self.prefix_epoch = int(prefix_epoch)
        self.original_parent_checkpoint_sha256 = original_parent_checkpoint_sha256
        self._pending: list[dict[str, Any]] = []
        self._state_paths: dict[int, Path] = {}
        self._previous_branch: dict[int, str] = {}
        self._previous_s2_t1_state: dict[int, bool] = {}
        self._ever_s2_t1_state: dict[int, bool] = {}
        self._state_exposure_epochs: dict[int, int] = {}
        self._state_entries: dict[int, int] = {}
        self._state_exits: dict[int, int] = {}
        self._state_reentries: dict[int, int] = {}
        self._state_switches: dict[int, int] = {}
        self._previous_action: dict[int, bool] = {}
        self._ever_action: dict[int, bool] = {}
        self._exposure_epochs: dict[int, int] = {}
        self._action_switches: dict[int, int] = {}
        self._entries: dict[int, int] = {}
        self._exits: dict[int, int] = {}
        self._reentries: dict[int, int] = {}
        self._thresholds: dict[str, Any] | None = None
        self._prefix_state: dict[str, Any] | None = None
        self.epoch_statistics: dict[int, dict[str, int | float | None]] = {}
        if arm == "prefix":
            if thresholds_path is not None:
                raise OnlineStateS2RoutingError("shared e100 prefix cannot receive frozen thresholds")
        else:
            if thresholds_path is None:
                raise OnlineStateS2RoutingError("online treatment/control arm requires frozen e100 thresholds")
            self._thresholds = self._load_thresholds(thresholds_path)

    @property
    def thresholds(self) -> dict[str, Any] | None:
        return None if self._thresholds is None else dict(self._thresholds)

    @property
    def is_prefix(self) -> bool:
        return self.arm == "prefix"

    @property
    def state_paths(self) -> dict[int, Path]:
        return dict(self._state_paths)

    def _load_thresholds(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise OnlineStateS2RoutingError(f"frozen online threshold artifact is missing: {path}")
        value = _read_json(path)
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != _sha256(path):
            raise OnlineStateS2RoutingError("online threshold artifact SHA-256 sidecar is missing or mismatched")
        required = {
            "schema_version": self.SCHEMA_VERSION,
            "contract": self.CONTRACT,
            "kind": "frozen_thresholds",
            "prefix_epoch": self.prefix_epoch,
            "original_parent_checkpoint_sha256": self.original_parent_checkpoint_sha256,
            "train_ids_sha256": _digest_ids(self.train_labels),
            "train_id_label_sha256": _digest_labels(self.train_labels),
        }
        if any(value.get(key) != expected for key, expected in required.items()):
            raise OnlineStateS2RoutingError("online threshold artifact has incompatible lineage")
        thresholds = value.get("thresholds")
        if not isinstance(thresholds, dict):
            raise OnlineStateS2RoutingError("online threshold artifact lacks threshold values")
        for key in ("student_global_logit_q10", "teacher_global_logit_q10"):
            candidate = thresholds.get(key)
            if not isinstance(candidate, (int, float)) or isinstance(candidate, bool) or candidate <= 0:
                raise OnlineStateS2RoutingError(f"online threshold artifact has invalid {key}")
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "student_global_logit_q10": float(thresholds["student_global_logit_q10"]),
            "teacher_global_logit_q10": float(thresholds["teacher_global_logit_q10"]),
            "artifact": value,
        }

    def _require_thresholds(self) -> dict[str, Any]:
        if self._thresholds is None:
            raise OnlineStateS2RoutingError("Online-S2 state requested before frozen e100 thresholds")
        return self._thresholds

    def _validate_frozen_child_binding(
        self,
        *,
        prefix_router_state: Mapping[str, Any],
        prefix_checkpoint_sha256: str,
        source_git_sha: str,
        training_attack_identity_sha256: str,
    ) -> None:
        """Bind a child threshold artifact to its exact shared e100 state.

        The q10 values alone are not sufficient lineage.  A child must prove
        that the artifact was derived from the precise e100 checkpoint and
        raw state that it resumes, under the frozen source and KL10 attack
        contract.  This check intentionally happens before the first child
        observation or optimizer update.
        """
        if self.is_prefix:
            raise OnlineStateS2RoutingError("a shared prefix has no child threshold binding")
        thresholds = self._require_thresholds()
        artifact = thresholds["artifact"]
        raw_paths = prefix_router_state.get("state_paths")
        raw_state = None if not isinstance(raw_paths, Mapping) else raw_paths.get(str(self.prefix_epoch))
        if not isinstance(raw_state, Mapping) or not isinstance(raw_state.get("sha256"), str):
            raise OnlineStateS2RoutingError("shared e100 checkpoint lacks a hash-bound online state artifact")
        expected = {
            "prefix_checkpoint_sha256": prefix_checkpoint_sha256,
            "prefix_state_sha256": raw_state["sha256"],
            "source_git_sha": source_git_sha,
            "training_attack_identity_sha256": training_attack_identity_sha256,
        }
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise OnlineStateS2RoutingError("frozen threshold artifact is not bound to this exact e100 child lineage")

    def _classify_one(
        self,
        *,
        clean_correct: bool,
        student_adv_correct: bool,
        student_margin: float,
        teacher_margin: float,
    ) -> str:
        thresholds = self._require_thresholds()
        if not clean_correct:
            return "CW"
        if not student_adv_correct:
            return "S3"
        if student_margin <= 0.0:
            # An argmax tie that is resolved to the true class is not covered
            # by the preregistered strict positive S1/S2 partition.  Failing
            # closed avoids silently placing a boundary tie into S1.
            raise OnlineStateS2RoutingError("adv-correct Student has non-positive global logit margin")
        if student_margin > float(thresholds["student_global_logit_q10"]):
            return "S1"
        if teacher_margin <= 0.0:
            return "S2_T3"
        if teacher_margin <= float(thresholds["teacher_global_logit_q10"]):
            return "S2_T2"
        return "S2_T1"

    def _rows_from_prefix_state(self, path: Path, *, expected_sha256: str) -> dict[int, dict[str, Any]]:
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise OnlineStateS2RoutingError("shared e100 online state path/hash does not match checkpoint lineage")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise OnlineStateS2RoutingError("online state artifacts require pyarrow") from exc
        rows = pq.read_table(path).to_pylist()
        by_id = {int(row["sample_id"]): dict(row) for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(self.train_labels):
            raise OnlineStateS2RoutingError("shared e100 state does not cover the exact train universe")
        for sample_id, row in by_id.items():
            if int(row.get("class_id", -1)) != self.train_labels[sample_id]:
                raise OnlineStateS2RoutingError("shared e100 state has an ID/label mismatch")
        return by_id

    def adopt_prefix_state(
        self,
        value: Mapping[str, Any],
        *,
        materialized_state_path: Path | None = None,
        prefix_checkpoint_sha256: str | None = None,
        source_git_sha: str | None = None,
        training_attack_identity_sha256: str | None = None,
    ) -> None:
        """Adopt one exact e100 prefix after threshold freeze.

        The prefix checkpoint precedes threshold serialization.  This method
        therefore validates its raw per-ID state artifact, binds the frozen
        artifact, and derives only the *previous* e100 branch labels for
        e101 transition telemetry.  It never replays an attack or update.
        """
        if self.is_prefix:
            raise OnlineStateS2RoutingError("prefix router cannot adopt an e100 prefix")
        thresholds = self._require_thresholds()
        required = {
            "schema_version": self.SCHEMA_VERSION,
            "contract": self.CONTRACT,
            "arm": "prefix",
            "prefix_epoch": self.prefix_epoch,
            "original_parent_checkpoint_sha256": self.original_parent_checkpoint_sha256,
            "train_ids_sha256": _digest_ids(self.train_labels),
            "train_id_label_sha256": _digest_labels(self.train_labels),
        }
        if any(value.get(key) != expected for key, expected in required.items()):
            raise OnlineStateS2RoutingError("shared e100 online-router state has incompatible lineage")
        supplied = (prefix_checkpoint_sha256, source_git_sha, training_attack_identity_sha256)
        if any(item is not None for item in supplied):
            if any(not isinstance(item, str) or not item for item in supplied):
                raise OnlineStateS2RoutingError("child threshold lineage binding must be supplied completely")
            self._validate_frozen_child_binding(
                prefix_router_state=value,
                prefix_checkpoint_sha256=str(prefix_checkpoint_sha256),
                source_git_sha=str(source_git_sha),
                training_attack_identity_sha256=str(training_attack_identity_sha256),
            )
        raw_paths = value.get("state_paths")
        if not isinstance(raw_paths, dict):
            raise OnlineStateS2RoutingError("shared e100 online-router state lacks state paths")
        raw = raw_paths.get(str(self.prefix_epoch))
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str) or not isinstance(raw.get("sha256"), str):
            raise OnlineStateS2RoutingError("shared e100 online-router state lacks e100 artifact binding")
        # A child may execute on Ferret while the e100 prefix was generated
        # on Hamster.  The checkpoint deliberately retains the original path
        # as lineage, but transport is not scientific identity.  A caller may
        # supply one local materialization only when its bytes match the path
        # SHA sealed in the prefix checkpoint.
        path = Path(raw["path"]) if materialized_state_path is None else materialized_state_path
        rows = self._rows_from_prefix_state(path, expected_sha256=str(raw["sha256"]))
        self._previous_branch = {
            sample_id: self._classify_one(
                clean_correct=bool(row["student_clean_correct"]),
                student_adv_correct=bool(row["student_adv_correct"]),
                student_margin=float(row["student_global_margin"]),
                teacher_margin=float(row["teacher_global_margin"]),
            )
            for sample_id, row in rows.items()
        }
        self._previous_s2_t1_state = {
            sample_id: branch == "S2_T1" for sample_id, branch in self._previous_branch.items()
        }
        # e100 establishes the predecessor state for e101 transition
        # accounting, but is intentionally not counted as child exposure.
        self._ever_s2_t1_state = dict(self._previous_s2_t1_state)
        self._state_exposure_epochs = {sample_id: 0 for sample_id in self.train_labels}
        self._state_entries = {sample_id: 0 for sample_id in self.train_labels}
        self._state_exits = {sample_id: 0 for sample_id in self.train_labels}
        self._state_reentries = {sample_id: 0 for sample_id in self.train_labels}
        self._state_switches = {sample_id: 0 for sample_id in self.train_labels}
        self._previous_action = {sample_id: False for sample_id in self.train_labels}
        self._ever_action = {sample_id: False for sample_id in self.train_labels}
        self._exposure_epochs = {sample_id: 0 for sample_id in self.train_labels}
        self._action_switches = {sample_id: 0 for sample_id in self.train_labels}
        self._entries = {sample_id: 0 for sample_id in self.train_labels}
        self._exits = {sample_id: 0 for sample_id in self.train_labels}
        self._reentries = {sample_id: 0 for sample_id in self.train_labels}
        self._prefix_state = {
            "original_path": str(Path(raw["path"]).resolve()),
            "materialized_path": str(path.resolve()),
            "sha256": str(raw["sha256"]),
            "threshold_artifact_sha256": thresholds["sha256"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "contract": self.CONTRACT,
            "arm": self.arm,
            "prefix_epoch": self.prefix_epoch,
            "original_parent_checkpoint_sha256": self.original_parent_checkpoint_sha256,
            "train_ids_sha256": _digest_ids(self.train_labels),
            "train_id_label_sha256": _digest_labels(self.train_labels),
            "thresholds": self._thresholds,
            "prefix_state": self._prefix_state,
            "previous_branch": {str(key): value for key, value in sorted(self._previous_branch.items())},
            "previous_s2_t1_state": {
                str(key): bool(value) for key, value in sorted(self._previous_s2_t1_state.items())
            },
            "ever_s2_t1_state": {str(key): bool(value) for key, value in sorted(self._ever_s2_t1_state.items())},
            "state_exposure_epochs": {
                str(key): int(value) for key, value in sorted(self._state_exposure_epochs.items())
            },
            "state_entries": {str(key): int(value) for key, value in sorted(self._state_entries.items())},
            "state_exits": {str(key): int(value) for key, value in sorted(self._state_exits.items())},
            "state_reentries": {str(key): int(value) for key, value in sorted(self._state_reentries.items())},
            "state_switches": {str(key): int(value) for key, value in sorted(self._state_switches.items())},
            "previous_action": {str(key): bool(value) for key, value in sorted(self._previous_action.items())},
            "ever_action": {str(key): bool(value) for key, value in sorted(self._ever_action.items())},
            "exposure_epochs": {str(key): int(value) for key, value in sorted(self._exposure_epochs.items())},
            "action_switches": {str(key): int(value) for key, value in sorted(self._action_switches.items())},
            "entries": {str(key): int(value) for key, value in sorted(self._entries.items())},
            "exits": {str(key): int(value) for key, value in sorted(self._exits.items())},
            "reentries": {str(key): int(value) for key, value in sorted(self._reentries.items())},
            "epoch_statistics": {str(key): value for key, value in sorted(self.epoch_statistics.items())},
            "state_paths": {
                str(epoch): {"path": str(path.resolve()), "sha256": _sha256(path)}
                for epoch, path in sorted(self._state_paths.items())
            },
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        required = {
            "schema_version": self.SCHEMA_VERSION,
            "contract": self.CONTRACT,
            "arm": self.arm,
            "prefix_epoch": self.prefix_epoch,
            "original_parent_checkpoint_sha256": self.original_parent_checkpoint_sha256,
            "train_ids_sha256": _digest_ids(self.train_labels),
            "train_id_label_sha256": _digest_labels(self.train_labels),
        }
        if any(value.get(key) != expected for key, expected in required.items()):
            raise OnlineStateS2RoutingError("online S2 checkpoint state has incompatible lineage")
        raw_thresholds = value.get("thresholds")
        if self.is_prefix:
            if raw_thresholds is not None:
                raise OnlineStateS2RoutingError("e100 prefix checkpoint unexpectedly carries frozen thresholds")
        else:
            if not isinstance(raw_thresholds, dict) or self._thresholds is None:
                raise OnlineStateS2RoutingError("online S2 child checkpoint lacks frozen thresholds")
            if raw_thresholds.get("sha256") != self._thresholds.get("sha256"):
                raise OnlineStateS2RoutingError("online S2 resume threshold artifact differs from frozen child input")
        expected_ids = set(self.train_labels)

        def mapping(name: str, caster: Any, *, required_for_child: bool = False) -> dict[int, Any]:
            raw = value.get(name, {})
            if not isinstance(raw, dict):
                raise OnlineStateS2RoutingError(f"online S2 checkpoint {name} is malformed")
            parsed = {int(key): caster(item) for key, item in raw.items()}
            if required_for_child and not self.is_prefix and set(parsed) != expected_ids:
                raise OnlineStateS2RoutingError(
                    f"online S2 child checkpoint {name} lacks exact train-universe coverage"
                )
            if parsed and set(parsed) != expected_ids:
                raise OnlineStateS2RoutingError(f"online S2 checkpoint {name} lacks exact train-universe coverage")
            return parsed

        self._previous_branch = mapping("previous_branch", str, required_for_child=True)
        if any(branch not in _BRANCHES for branch in self._previous_branch.values()):
            raise OnlineStateS2RoutingError("online S2 checkpoint has an unknown prior branch")
        self._previous_s2_t1_state = mapping("previous_s2_t1_state", bool, required_for_child=True)
        self._ever_s2_t1_state = mapping("ever_s2_t1_state", bool, required_for_child=True)
        self._state_exposure_epochs = mapping("state_exposure_epochs", int, required_for_child=True)
        self._state_entries = mapping("state_entries", int, required_for_child=True)
        self._state_exits = mapping("state_exits", int, required_for_child=True)
        self._state_reentries = mapping("state_reentries", int, required_for_child=True)
        self._state_switches = mapping("state_switches", int, required_for_child=True)
        self._previous_action = mapping("previous_action", bool, required_for_child=True)
        self._ever_action = mapping("ever_action", bool, required_for_child=True)
        self._exposure_epochs = mapping("exposure_epochs", int, required_for_child=True)
        self._action_switches = mapping("action_switches", int, required_for_child=True)
        self._entries = mapping("entries", int, required_for_child=True)
        self._exits = mapping("exits", int, required_for_child=True)
        self._reentries = mapping("reentries", int, required_for_child=True)
        raw_statistics = value.get("epoch_statistics", {})
        if not isinstance(raw_statistics, dict):
            raise OnlineStateS2RoutingError("online S2 checkpoint statistics are malformed")
        self.epoch_statistics = {
            int(epoch): dict(item) for epoch, item in raw_statistics.items() if isinstance(item, dict)
        }
        raw_paths = value.get("state_paths", {})
        if not isinstance(raw_paths, dict):
            raise OnlineStateS2RoutingError("online S2 checkpoint state paths are malformed")
        if not self.is_prefix and not raw_paths:
            raise OnlineStateS2RoutingError("online S2 child checkpoint lacks persisted state-artifact lineage")
        parsed_paths: dict[int, Path] = {}
        for epoch, item in raw_paths.items():
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("sha256"), str)
            ):
                raise OnlineStateS2RoutingError("online S2 checkpoint state path descriptor is malformed")
            path = Path(str(item["path"]))
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise OnlineStateS2RoutingError("online S2 checkpoint state artifact differs from its saved SHA-256")
            parsed_paths[int(epoch)] = path
        self._state_paths = parsed_paths
        self._prefix_state = None if value.get("prefix_state") is None else dict(value["prefix_state"])

    def observe(
        self,
        *,
        epoch: int,
        sample_ids: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        student_clean_logits: torch.Tensor,
        student_adversarial_logits: torch.Tensor,
        teacher_adversarial_logits: torch.Tensor,
    ) -> OnlineStateS2Decision:
        if epoch < self.prefix_epoch:
            raise OnlineStateS2RoutingError("online S2 observation cannot run before the shared e100 prefix")
        if sample_ids.ndim != 1 or sample_ids.shape != labels.shape or valid_mask.shape != labels.shape:
            raise OnlineStateS2RoutingError("online S2 stable IDs, labels, and valid mask must align")
        for name, logits in {
            "student clean": student_clean_logits,
            "student adversarial": student_adversarial_logits,
            "teacher adversarial": teacher_adversarial_logits,
        }.items():
            if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
                raise OnlineStateS2RoutingError(f"{name} logits do not match the routing batch")
        if epoch == self.prefix_epoch and not self.is_prefix:
            raise OnlineStateS2RoutingError("only the shared prefix may observe epoch 100")
        if epoch > self.prefix_epoch and self.is_prefix:
            raise OnlineStateS2RoutingError("shared prefix may observe only epoch 100")

        student_clean_correct = student_clean_logits.detach().argmax(1).eq(labels)
        student_adv_correct = student_adversarial_logits.detach().argmax(1).eq(labels)
        teacher_adv_correct = teacher_adversarial_logits.detach().argmax(1).eq(labels)
        student_margin = global_logit_margin(student_adversarial_logits, labels)
        teacher_margin = global_logit_margin(teacher_adversarial_logits, labels)
        ids = [int(item) for item in sample_ids.detach().cpu().tolist()]
        classes = [int(item) for item in labels.detach().cpu().tolist()]
        valid = [bool(item) for item in valid_mask.detach().cpu().tolist()]
        clean_values = student_clean_correct.detach().cpu().tolist()
        student_correct_values = student_adv_correct.detach().cpu().tolist()
        teacher_correct_values = teacher_adv_correct.detach().cpu().tolist()
        student_margin_values = student_margin.detach().cpu().tolist()
        teacher_margin_values = teacher_margin.detach().cpu().tolist()
        eligible_values: list[bool] = []
        action_values: list[bool] = []
        for position, sample_id in enumerate(ids):
            if not valid[position]:
                eligible_values.append(False)
                action_values.append(False)
                continue
            expected_label = self.train_labels.get(sample_id)
            if expected_label is None or expected_label != classes[position]:
                raise OnlineStateS2RoutingError("routing observed an ID/label outside the exact train universe")
            if self.is_prefix:
                branch = "PREFIX"
                eligible = False
                action = False
                prior_branch = self._previous_branch.get(sample_id)
                prior_action = self._previous_action.get(sample_id, False)
                action_entry = action_exit = action_reentry = False
                state_active = state_entry = state_exit = state_reentry = False
            else:
                branch = self._classify_one(
                    clean_correct=bool(clean_values[position]),
                    student_adv_correct=bool(student_correct_values[position]),
                    student_margin=float(student_margin_values[position]),
                    teacher_margin=float(teacher_margin_values[position]),
                )
                eligible = branch == "S2_T1"
                state_active = eligible
                prior_state_active = self._previous_s2_t1_state.get(sample_id, False)
                state_entry = bool(state_active and not prior_state_active)
                state_exit = bool(not state_active and prior_state_active)
                state_reentry = bool(state_entry and self._ever_s2_t1_state.get(sample_id, False))
                action = eligible and self.arm in {"pmp", "dbdp"}
                prior_branch = self._previous_branch.get(sample_id)
                prior_action = self._previous_action.get(sample_id, False)
                action_entry = bool(action and not prior_action)
                action_exit = bool(not action and prior_action)
                action_reentry = bool(action_entry and self._ever_action.get(sample_id, False))
                if action and branch != "S2_T1":  # defensive, but a hard scientific invariant
                    raise OnlineStateS2RoutingError("preservation action escaped Online-S2×T1")
                if action_entry:
                    self._entries[sample_id] = self._entries.get(sample_id, 0) + 1
                    if action_reentry:
                        self._reentries[sample_id] = self._reentries.get(sample_id, 0) + 1
                if action_exit:
                    self._exits[sample_id] = self._exits.get(sample_id, 0) + 1
                if action != prior_action:
                    self._action_switches[sample_id] = self._action_switches.get(sample_id, 0) + 1
                if state_entry:
                    self._state_entries[sample_id] = self._state_entries.get(sample_id, 0) + 1
                    if state_reentry:
                        self._state_reentries[sample_id] = self._state_reentries.get(sample_id, 0) + 1
                if state_exit:
                    self._state_exits[sample_id] = self._state_exits.get(sample_id, 0) + 1
                if state_active != prior_state_active:
                    self._state_switches[sample_id] = self._state_switches.get(sample_id, 0) + 1
                if state_active:
                    self._ever_s2_t1_state[sample_id] = True
                    self._state_exposure_epochs[sample_id] = self._state_exposure_epochs.get(sample_id, 0) + 1
                if action:
                    self._ever_action[sample_id] = True
                    self._exposure_epochs[sample_id] = self._exposure_epochs.get(sample_id, 0) + 1
                self._previous_branch[sample_id] = branch
                self._previous_s2_t1_state[sample_id] = state_active
                self._previous_action[sample_id] = action
            transition = None if prior_branch is None or branch == "PREFIX" else f"{prior_branch}->{branch}"
            self._pending.append(
                {
                    "epoch": int(epoch),
                    "sample_id": sample_id,
                    "class_id": classes[position],
                    "student_clean_correct": bool(clean_values[position]),
                    "student_adv_correct": bool(student_correct_values[position]),
                    "teacher_adv_correct": bool(teacher_correct_values[position]),
                    "student_global_margin": float(student_margin_values[position]),
                    "teacher_global_margin": float(teacher_margin_values[position]),
                    "branch": branch,
                    "eligible_s2_t1": bool(eligible),
                    "s2_t1_state_active": bool(state_active),
                    "s2_t1_state_entry": bool(state_entry),
                    "s2_t1_state_exit": bool(state_exit),
                    "s2_t1_state_reentry": bool(state_reentry),
                    "action_active": bool(action),
                    "previous_branch": prior_branch,
                    "previous_action": bool(prior_action),
                    "action_entry": action_entry,
                    "action_exit": action_exit,
                    "action_reentry": action_reentry,
                    "transition": transition,
                    "rank": get_rank(),
                    "order": len(self._pending),
                }
            )
            eligible_values.append(eligible)
            action_values.append(action)
        eligible = torch.tensor(eligible_values, device=sample_ids.device, dtype=torch.bool) & valid_mask
        action = torch.tensor(action_values, device=sample_ids.device, dtype=torch.bool) & valid_mask
        if self.is_prefix and bool(action.any()):
            raise OnlineStateS2RoutingError("shared e100 prefix unexpectedly activated preservation")
        if bool((action & ~eligible).any()):
            raise OnlineStateS2RoutingError("online action is outside Online-S2×T1")
        return OnlineStateS2Decision(eligible_s2_t1=eligible.detach(), action_active=action.detach())

    def _flush_rows(self, epoch: int) -> dict[int, dict[str, Any]]:
        rows = [row for rank_rows in gather_objects(self._pending) for row in rank_rows]
        self._pending = []
        by_id: dict[int, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda item: (int(item["sample_id"]), int(item["rank"]), int(item["order"]))):
            sample_id = int(row["sample_id"])
            if sample_id in by_id:
                raise OnlineStateS2RoutingError(
                    f"online state observed duplicate valid stable ID in epoch {epoch}: {sample_id}"
                )
            by_id[sample_id] = row
        if set(by_id) != set(self.train_labels):
            missing = len(set(self.train_labels) - set(by_id))
            unexpected = len(set(by_id) - set(self.train_labels))
            raise OnlineStateS2RoutingError(
                f"online state epoch {epoch} lacks exact train coverage (missing={missing}, unexpected={unexpected})"
            )
        return by_id

    def _epoch_statistics(self, *, epoch: int, rows: Mapping[int, Mapping[str, Any]]) -> dict[str, int | float | None]:
        branches = Counter(str(row["branch"]) for row in rows.values())
        transitions = Counter(
            str(row["transition"])
            for row in rows.values()
            if isinstance(row.get("transition"), str) and row.get("transition") in _REQUIRED_TRANSITIONS
        )
        action_count = sum(bool(row["action_active"]) for row in rows.values())
        eligible_count = sum(bool(row["eligible_s2_t1"]) for row in rows.values())
        student_values = _threshold_population(rows, model="student")
        teacher_values = _threshold_population(rows, model="teacher")
        student_q10 = q10_linear(student_values)
        teacher_q10 = q10_linear(teacher_values)
        thresholds = self._thresholds
        return {
            "train_universe_count": len(rows),
            "clean_wrong_count": branches["CW"],
            "s3_non_clean_wrong_count": branches["S3"],
            "online_s2_count": branches["S2_T1"] + branches["S2_T2"] + branches["S2_T3"],
            "online_s1_count": branches["S1"],
            "online_s2_t1_count": branches["S2_T1"],
            "online_s2_t2_count": branches["S2_T2"],
            "online_s2_t3_count": branches["S2_T3"],
            "eligible_s2_t1_count": eligible_count,
            "s2_t1_state_active_count": sum(bool(row["s2_t1_state_active"]) for row in rows.values()),
            "s2_t1_state_entry_count": sum(bool(row["s2_t1_state_entry"]) for row in rows.values()),
            "s2_t1_state_exit_count": sum(bool(row["s2_t1_state_exit"]) for row in rows.values()),
            "s2_t1_state_reentry_count": sum(bool(row["s2_t1_state_reentry"]) for row in rows.values()),
            "active_treatment_count": action_count,
            "active_treatment_fraction": action_count / len(rows),
            "new_s2_t1_entrant_count": sum(
                bool(row["branch"] == "S2_T1" and row.get("previous_branch") != "S2_T1") for row in rows.values()
            ),
            "state_switch_count": sum(
                bool(row.get("previous_branch") is not None and row.get("previous_branch") != row["branch"])
                for row in rows.values()
            ),
            "action_switch_count": sum(bool(row["action_active"] != row["previous_action"]) for row in rows.values()),
            "action_entry_count": sum(bool(row["action_entry"]) for row in rows.values()),
            "action_exit_count": sum(bool(row["action_exit"]) for row in rows.values()),
            "action_reentry_count": sum(bool(row["action_reentry"]) for row in rows.values()),
            "transition_s1_to_s2_t1": transitions["S1->S2_T1"],
            "transition_s2_t1_to_s1": transitions["S2_T1->S1"],
            "transition_s2_t1_to_s3": transitions["S2_T1->S3"],
            "transition_s2_t1_to_clean_wrong": transitions["S2_T1->CW"],
            "transition_s3_to_s2_t1": transitions["S3->S2_T1"],
            "transition_clean_wrong_to_s2_t1": transitions["CW->S2_T1"],
            "transition_s2_t2_to_s2_t1": transitions["S2_T2->S2_T1"],
            "transition_s2_t3_to_s2_t1": transitions["S2_T3->S2_T1"],
            "transition_s2_t1_to_s2_t2": transitions["S2_T1->S2_T2"],
            "transition_s2_t1_to_s2_t3": transitions["S2_T1->S2_T3"],
            "student_current_q10": student_q10,
            "teacher_current_q10": teacher_q10,
            "student_current_q10_minus_frozen": (
                None if thresholds is None else student_q10 - float(thresholds["student_global_logit_q10"])
            ),
            "teacher_current_q10_minus_frozen": (
                None if thresholds is None else teacher_q10 - float(thresholds["teacher_global_logit_q10"])
            ),
        }

    def flush_epoch(self, epoch: int) -> None:
        rows = self._flush_rows(epoch)
        if self.is_prefix:
            if epoch != self.prefix_epoch:
                raise OnlineStateS2RoutingError("shared prefix wrote an unexpected epoch")
            if any(bool(row["action_active"]) for row in rows.values()):
                raise OnlineStateS2RoutingError("shared prefix applied a preservation action")
        else:
            if epoch <= self.prefix_epoch:
                raise OnlineStateS2RoutingError("online treatment wrote a pre-threshold epoch")
            if any(bool(row["action_active"]) and str(row["branch"]) != "S2_T1" for row in rows.values()):
                raise OnlineStateS2RoutingError("online action escaped Online-S2×T1 in persisted rows")
        self.epoch_statistics[epoch] = self._epoch_statistics(epoch=epoch, rows=rows)
        if get_rank() != 0:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise OnlineStateS2RoutingError("online state artifacts require pyarrow") from exc
        state_dir = self.output_dir / "online-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"epoch-{epoch}.parquet"
        if path.exists():
            raise OnlineStateS2RoutingError(f"refusing to overwrite online state artifact: {path}")
        public_rows = [
            {key: value for key, value in row.items() if key not in {"rank", "order"}} for row in rows.values()
        ]
        pq.write_table(pa.Table.from_pylist(public_rows), path, compression="zstd")
        self._state_paths[epoch] = path

    def record_runtime_metrics(self, *, epoch: int, metrics: Mapping[str, float]) -> None:
        """Bind already-computed boundary telemetry to the router checkpoint.

        ``flush_epoch`` deliberately runs before the common checkpoint write so
        that the per-ID state and its aggregate runtime diagnostics are saved
        atomically in the same continuation lineage.  This method only copies
        detached metrics that the Trainer has already computed; it neither
        changes the routing state nor performs any model work.
        """
        if epoch not in self.epoch_statistics:
            raise OnlineStateS2RoutingError("runtime telemetry arrived before the online state was flushed")
        boundary = {
            str(key): float(value)
            for key, value in metrics.items()
            if str(key).startswith("boundary_")
        }
        if not boundary:
            raise OnlineStateS2RoutingError("online runtime telemetry lacks boundary diagnostics")
        if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64)).item() for value in boundary.values()):
            raise OnlineStateS2RoutingError("online runtime telemetry contains a non-finite boundary diagnostic")
        overlap = set(boundary) & set(self.epoch_statistics[epoch])
        if overlap:
            raise OnlineStateS2RoutingError(
                f"online runtime telemetry would overwrite router statistics: {sorted(overlap)}"
            )
        self.epoch_statistics[epoch].update(boundary)

    def finalize(self) -> dict[str, Any]:
        if self.is_prefix:
            if set(self._state_paths) != {self.prefix_epoch}:
                raise OnlineStateS2RoutingError("shared e100 prefix did not produce exactly one state artifact")
        elif self._thresholds is None or not self._state_paths:
            raise OnlineStateS2RoutingError("online treatment cannot finalize without thresholds and state artifacts")
        if get_rank() != 0:
            return self.state_dict()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise OnlineStateS2RoutingError("online state artifacts require pyarrow") from exc
        paths = [self._state_paths[epoch] for epoch in sorted(self._state_paths)]
        combined_path = self.output_dir / "online-state.parquet"
        if combined_path.exists():
            raise OnlineStateS2RoutingError(f"refusing to overwrite online state artifact: {combined_path}")
        pq.write_table(pa.concat_tables([pq.read_table(path) for path in paths]), combined_path, compression="zstd")
        result = {
            "schema_version": self.SCHEMA_VERSION,
            "contract": self.CONTRACT,
            "arm": self.arm,
            "thresholds": self._thresholds,
            "prefix_state": self._prefix_state,
            "state": {
                "path": str(combined_path.resolve()),
                "sha256": _sha256(combined_path),
                "epochs": {
                    str(epoch): {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for epoch, path in sorted(self._state_paths.items())
                },
            },
            "per_id_action_summary": {
                "entries": sum(self._entries.values()),
                "exits": sum(self._exits.values()),
                "reentries": sum(self._reentries.values()),
                "action_switches": sum(self._action_switches.values()),
                "total_exposure_epochs": sum(self._exposure_epochs.values()),
            },
            "per_id_s2_t1_state_summary": {
                "entries": sum(self._state_entries.values()),
                "exits": sum(self._state_exits.values()),
                "reentries": sum(self._state_reentries.values()),
                "state_switches": sum(self._state_switches.values()),
                "total_active_epochs": sum(self._state_exposure_epochs.values()),
            },
        }
        manifest_path = self.output_dir / "online-state-manifest.json"
        if manifest_path.exists():
            raise OnlineStateS2RoutingError(f"refusing to overwrite online state manifest: {manifest_path}")
        manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**result, "manifest_path": str(manifest_path.resolve())}


def freeze_online_thresholds(
    *,
    prefix_router_state: Mapping[str, Any],
    prefix_checkpoint: Path,
    output_path: Path,
    source_git_sha: str,
    training_attack_identity_sha256: str,
    prefix_state_materialized_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze seed-specific e100 q10 thresholds from a hash-bound prefix state.

    ``prefix_state_materialized_path`` permits the controller to atomically
    promote a completed prefix from an attempt-scoped staging directory to its
    canonical output directory.  The original checkpoint path remains the
    lineage record; the replacement is accepted only when its bytes match the
    SHA-256 sealed in that checkpoint.
    """
    if output_path.exists() or output_path.with_name(output_path.name + ".sha256").exists():
        raise OnlineStateS2RoutingError(f"refusing to overwrite frozen online thresholds: {output_path}")
    required = {
        "schema_version": OnlineStateS2Router.SCHEMA_VERSION,
        "contract": OnlineStateS2Router.CONTRACT,
        "arm": "prefix",
    }
    if any(prefix_router_state.get(key) != expected for key, expected in required.items()):
        raise OnlineStateS2RoutingError("threshold freeze requires an Online-S2 e100 prefix state")
    raw_paths = prefix_router_state.get("state_paths")
    epoch = prefix_router_state.get("prefix_epoch")
    if not isinstance(raw_paths, dict) or not isinstance(epoch, int):
        raise OnlineStateS2RoutingError("prefix router state lacks epoch-state binding")
    raw = raw_paths.get(str(epoch))
    if not isinstance(raw, dict) or not isinstance(raw.get("path"), str) or not isinstance(raw.get("sha256"), str):
        raise OnlineStateS2RoutingError("prefix router state lacks e100 state artifact")
    if not prefix_checkpoint.is_file():
        raise OnlineStateS2RoutingError("prefix checkpoint is unavailable for threshold freeze")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise OnlineStateS2RoutingError("threshold freeze requires pyarrow") from exc
    original_state_path = Path(raw["path"])
    state_path = original_state_path if prefix_state_materialized_path is None else prefix_state_materialized_path
    if not state_path.is_file() or _sha256(state_path) != raw["sha256"]:
        raise OnlineStateS2RoutingError("prefix state artifact bytes do not match checkpoint lineage")
    rows = pq.read_table(state_path).to_pylist()
    by_id = {int(row["sample_id"]): dict(row) for row in rows}
    train_digest = prefix_router_state.get("train_ids_sha256")
    labels_digest = prefix_router_state.get("train_id_label_sha256")
    if len(by_id) != len(rows) or _digest_ids(list(by_id)) != train_digest:
        raise OnlineStateS2RoutingError("prefix threshold state has duplicate or mismatched stable IDs")
    labels = {sample_id: int(row["class_id"]) for sample_id, row in by_id.items()}
    if _digest_labels(labels) != labels_digest:
        raise OnlineStateS2RoutingError("prefix threshold state has mismatched ID/label digest")
    student = _threshold_summary(by_id, model="student")
    teacher = _threshold_summary(by_id, model="teacher")
    payload = {
        "schema_version": OnlineStateS2Router.SCHEMA_VERSION,
        "contract": OnlineStateS2Router.CONTRACT,
        "kind": "frozen_thresholds",
        "prefix_epoch": epoch,
        "prefix_checkpoint": str(prefix_checkpoint.resolve()),
        "prefix_checkpoint_sha256": _sha256(prefix_checkpoint),
        "prefix_state_original_path": str(original_state_path.resolve()),
        "prefix_state_path": str(state_path.resolve()),
        "prefix_state_sha256": _sha256(state_path),
        "original_parent_checkpoint_sha256": prefix_router_state.get("original_parent_checkpoint_sha256"),
        "train_ids_sha256": train_digest,
        "train_id_label_sha256": labels_digest,
        "source_git_sha": source_git_sha,
        "training_attack_identity_sha256": training_attack_identity_sha256,
        "thresholds": {
            "student_global_logit_q10": student["threshold"],
            "teacher_global_logit_q10": teacher["threshold"],
        },
        "population": {"student": student, "teacher": teacher},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    output_path.with_name(output_path.name + ".sha256").write_text(_sha256(output_path) + "\n", encoding="utf-8")
    return {**payload, "artifact_sha256": _sha256(output_path)}
