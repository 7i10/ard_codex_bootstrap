"""Read-only CE-PGD20 FF/current-wrong point analysis for Chen L2/L4."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.ffnr_forecasting import (
    FFNRForecastingError,
    _load_validation,
    _metric,
    _online_panel,
    deterministic_midranks,
    equal_rank_score,
    plateau_candidates,
    split_ff_nr,
)
from ard.analysis.ffnr_strong_replay import (
    CONTRACT_ID as REPLAY_CONTRACT,
)
from ard.analysis.ffnr_strong_replay import (
    EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
    OBSERVATION_COLUMNS,
    expected_selection_attack,
)
from ard.analysis.rslad_signal_replay import repository_root_from_source
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import canonical_json, sha256_file


class StrongPointError(FFNRForecastingError):
    """Raised when CE-PGD20 strong-point inputs are not scientifically bound."""


CONTRACT = "ffnr_strong_point_v1"
TOP_FRACTION = 0.10
ANCHORS = (39, 59, 79)


def _tracked_clean_provenance() -> dict[str, Any]:
    """Bind a point report to the exact clean, tracked analysis implementation."""
    root = repository_root_from_source()
    paths = {
        "ffnr_strong_point": Path(__file__).resolve(),
        "ffnr_strong_point_cli": root / "src/ard/cli/ffnr_strong_point.py",
        "ffnr_forecasting": root / "src/ard/analysis/ffnr_forecasting.py",
        "ffnr_strong_replay": root / "src/ard/analysis/ffnr_strong_replay.py",
    }
    if any(not path.is_file() for path in paths.values()):
        raise StrongPointError("strong point analysis source tree is incomplete")
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative],
            check=True,
            capture_output=True,
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise StrongPointError("strong point analysis requires tracked source and readable Git identity") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha) or dirty:
        raise StrongPointError("strong point analysis requires a tracked-clean Git revision")
    source_files = {name: sha256_file(path) for name, path in paths.items()}
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": source_files,
        "source_sha256": hashlib.sha256(canonical_json(source_files)).hexdigest(),
    }


def _validated_analysis_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    git, source_files = value.get("git"), value.get("source_files")
    expected_names = {"ffnr_strong_point", "ffnr_strong_point_cli", "ffnr_forecasting", "ffnr_strong_replay"}
    if (
        not isinstance(git, Mapping)
        or git.get("dirty") is not False
        or not isinstance(git.get("sha"), str)
        or len(str(git["sha"])) != 40
        or any(character not in "0123456789abcdef" for character in str(git["sha"]))
        or not isinstance(source_files, Mapping)
        or set(source_files) != expected_names
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in source_files.values()
        )
    ):
        raise StrongPointError("strong point analysis provenance is incomplete")
    normalized = {str(name): str(digest) for name, digest in source_files.items()}
    if value.get("source_sha256") != hashlib.sha256(canonical_json(normalized)).hexdigest():
        raise StrongPointError("strong point analysis provenance aggregate hash drifted")
    return {"git": {"sha": str(git["sha"]), "dirty": False}, "source_files": normalized, "source_sha256": value["source_sha256"]}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrongPointError(f"strong point JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise StrongPointError("strong point JSON must be an object")
    return value


def _finite(value: object, *, name: str, lo: float = -math.inf, hi: float = math.inf) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lo <= float(value) <= hi
    ):
        raise StrongPointError(f"{name} is outside its finite contract")
    return float(value)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:
        raise StrongPointError(f"strong replay Parquet is unreadable: {path}") from exc
    if set(table.column_names) != set(OBSERVATION_COLUMNS):
        raise StrongPointError("strong replay Parquet schema drifted")
    return [dict(row) for row in table.to_pylist()]


def _probabilities(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 10:
        raise StrongPointError(f"{name} must contain exactly ten probabilities")
    values = tuple(_finite(item, name=name, lo=0, hi=1) for item in value)
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
        raise StrongPointError(f"{name} must sum to one")
    return values


def _strong_lineage(
    *, path: Path, observations: Path, role: str, expected_count: int, expected_universe_sha256: str
) -> dict[str, Any]:
    lineage = _json(path)
    if (
        lineage.get("contract") != REPLAY_CONTRACT
        or lineage.get("schema_version") != 1
        or lineage.get("semantic_role") != role
        or lineage.get("observations_sha256") != sha256_file(observations)
        or lineage.get("train_expected_count") != expected_count
        or lineage.get("row_count") != expected_count * len(lineage.get("requested_epochs", []))
    ):
        raise StrongPointError("strong replay lineage byte/count/role contract drifted")
    required = {
        "run_id",
        "teacher",
        "dataset_identity",
        "attack_identity",
        "runtime",
        "stable_id_class_universe",
        "checkpoints",
        "requested_epochs",
        "saved_resolved_config_mapping_sha256",
        "manifest_sha256",
    }
    if not required.issubset(lineage) or lineage["attack_identity"] != expected_selection_attack():
        raise StrongPointError("strong replay lineage identity/CE-PGD20 contract drifted")
    runtime = lineage["runtime"]
    expected_flags = {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    if not isinstance(runtime, Mapping) or runtime.get("deterministic_backend") != expected_flags:
        raise StrongPointError("strong replay deterministic backend contract drifted")
    provenance = lineage["analysis_provenance"]
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("source_sha256"), str):
        raise StrongPointError("strong replay source SHA is missing")
    universe = lineage["stable_id_class_universe"]
    if (
        not isinstance(universe, Mapping)
        or universe.get("count") != expected_count
        or universe.get("sha256") != expected_universe_sha256
    ):
        raise StrongPointError("strong replay stable-ID/class universe lineage drifted")
    dataset = lineage["dataset_identity"]
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("dataset"), Mapping):
        raise StrongPointError("strong replay dataset identity is invalid")
    if dataset["dataset"].get("name") != "cifar10" or dataset["dataset"].get("split") != "train":
        raise StrongPointError("official-test or non-CIFAR-train input is forbidden")
    return lineage


def _teacher_registry_id(value: object) -> str:
    if not isinstance(value, Mapping) or not isinstance(value.get("registry_id"), str):
        raise StrongPointError("teacher registry identity is invalid")
    return str(value["registry_id"])


def _dataset_key(value: object) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise StrongPointError("dataset identity is invalid")
    dataset = value.get("dataset")
    if isinstance(dataset, Mapping):
        name, split = dataset.get("name"), dataset.get("split")
    else:
        name, split = dataset, value.get("split", "train")
    if name != "cifar10" or split != "train":
        raise StrongPointError("official-test or non-CIFAR-train input is forbidden")
    return str(name), str(split)


def _validation_identity(path: Path, history: Path, online: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a completed baseline manifest to online state without lineage migration."""
    if path.parent.resolve() != history.parent.resolve():
        raise StrongPointError("validation manifest must be a sibling of validation history")
    value = _json(path)
    git = value.get("git")
    if value.get("status") != "completed" or not isinstance(git, Mapping):
        raise StrongPointError("validation manifest must be completed with Git identity")
    observed = {
        "run_id": value.get("run_id"),
        "config_hash": value.get("config_hash"),
        "scientific_git_sha": git.get("sha"),
        "seed": value.get("seed"),
        "teacher_registry_id": _teacher_registry_id(value.get("teacher")),
    }
    expected = {
        "run_id": online.get("run_id"),
        "config_hash": online.get("config_hash"),
        "scientific_git_sha": online.get("scientific_git_sha"),
        "seed": online.get("seed"),
        "teacher_registry_id": _teacher_registry_id(online.get("teacher")),
    }
    if observed != expected:
        raise StrongPointError("validation manifest and online-state identity drifted")
    return observed


def _normalized_identity(
    *,
    feature: Mapping[str, Any],
    outcome: Mapping[str, Any],
    online: Mapping[str, Any],
    validation_manifest: Path,
    validation_history: Path,
) -> dict[str, Any]:
    """Bridge immutable e5 replay provenance to the online run identity."""
    if feature.get("run_id") != outcome.get("run_id") or feature.get("run_id") != online.get("run_id"):
        raise StrongPointError("strong feature/outcome/online run identity drifted")
    saved_config_hash = feature.get("saved_resolved_config_mapping_sha256")
    if (
        not isinstance(saved_config_hash, str)
        or saved_config_hash != outcome.get("saved_resolved_config_mapping_sha256")
        or saved_config_hash != online.get("config_hash")
    ):
        raise StrongPointError("strong replay saved config and online config hash drifted")
    manifest_sha = sha256_file(validation_manifest)
    if (
        feature.get("manifest_sha256") != outcome.get("manifest_sha256")
        or feature.get("manifest_sha256") != manifest_sha
    ):
        raise StrongPointError("strong replay and validation manifest hash drifted")
    teacher = _teacher_registry_id(feature.get("teacher"))
    if teacher != _teacher_registry_id(outcome.get("teacher")) or teacher != _teacher_registry_id(online.get("teacher")):
        raise StrongPointError("strong feature/outcome/online teacher identity drifted")
    dataset = _dataset_key(feature.get("dataset_identity"))
    if dataset != _dataset_key(outcome.get("dataset_identity")) or dataset != _dataset_key(online.get("dataset_identity")):
        raise StrongPointError("strong feature/outcome/online dataset identity drifted")
    feature_source = feature.get("analysis_provenance", {}).get("source_sha256")
    outcome_source = outcome.get("analysis_provenance", {}).get("source_sha256")
    if not isinstance(feature_source, str) or feature_source != outcome_source:
        raise StrongPointError("strong feature/outcome source SHA drifted")
    validation = _validation_identity(validation_manifest, validation_history, online)
    return {
        "run_id": str(online["run_id"]),
        "config_hash": str(online["config_hash"]),
        "scientific_git_sha": str(validation["scientific_git_sha"]),
        "seed": validation["seed"],
        "teacher_registry_id": teacher,
        "dataset": {"name": dataset[0], "split": dataset[1]},
        "strong_feature_source_sha256": feature_source,
        "strong_manifest_sha256": manifest_sha,
    }


def _strong_panel(
    rows: Sequence[Mapping[str, Any]], *, epochs: Sequence[int], expected_count: int, expected_universe_sha256: str
) -> dict[int, dict[int, dict[str, Any]]]:
    if tuple(epochs) != tuple(sorted(set(epochs))) or not epochs:
        raise StrongPointError("strong replay selected epochs must be sorted and unique")
    panel: dict[int, dict[int, dict[str, Any]]] = {epoch: {} for epoch in epochs}
    for row in rows:
        epoch, sample_id, class_id = row.get("epoch"), row.get("sample_id"), row.get("class_id")
        if (
            row.get("namespace") != "train"
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch not in panel
            or isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 10
            or sample_id in panel[epoch]
        ):
            raise StrongPointError("strong replay epoch/ID/class contract drifted")
        adv_margin = _finite(row.get("student_adversarial_probability_margin"), name="adv margin", lo=-1, hi=1)
        clean_margin = _finite(row.get("student_clean_probability_margin"), name="clean margin", lo=-1, hi=1)
        adv_logit = _finite(row.get("student_adversarial_logit_margin"), name="adv logit")
        clean_logit = _finite(row.get("student_clean_logit_margin"), name="clean logit")
        probability_delta = _finite(
            row.get("student_clean_to_adversarial_probability_margin_delta"), name="probability delta", lo=-2, hi=2
        )
        logit_delta = _finite(row.get("student_clean_to_adversarial_logit_margin_delta"), name="logit delta")
        if not math.isclose(clean_margin + probability_delta, adv_margin, rel_tol=0, abs_tol=1e-6) or not math.isclose(
            clean_logit + logit_delta, adv_logit, rel_tol=0, abs_tol=1e-6
        ):
            raise StrongPointError("strong clean-to-adversarial response algebra drifted")
        if not isinstance(row.get("student_robust_correct"), bool):
            raise StrongPointError("strong robust correctness must be boolean")
        panel[epoch][sample_id] = {
            "class_id": class_id,
            "correct": row["student_robust_correct"],
            "adv_probability_margin": adv_margin,
            "adv_logit_margin": adv_logit,
            "adv_ce": _finite(row.get("student_adversarial_ce"), name="adv CE", lo=0),
            "probability_drop": -probability_delta,
            "logit_drop": -logit_delta,
            "teacher_js": _finite(
                row.get("teacher_clean_adversarial_js"), name="teacher JS", lo=0, hi=math.log(2) + 1e-6
            ),
            "teacher_adv": _probabilities(
                row.get("teacher_adversarial_probabilities"), name="teacher adv probabilities"
            ),
        }
    if any(len(values) != expected_count for values in panel.values()):
        raise StrongPointError("strong replay lacks exact stable-ID coverage")
    reference = next(iter(panel.values()))
    pairs = [{"sample_id": item, "class_id": reference[item]["class_id"]} for item in sorted(reference)]
    if hashlib.sha256(canonical_json(pairs)).hexdigest() != expected_universe_sha256:
        raise StrongPointError("strong replay stable-ID/class universe hash drifted")
    if any(
        set(values) != set(reference)
        or any(values[sample_id]["class_id"] != reference[sample_id]["class_id"] for sample_id in reference)
        for values in panel.values()
    ):
        raise StrongPointError("strong replay stable-ID/class universe changed across epochs")
    return panel


def _teacher_wrong_confidence(probabilities: Sequence[float], label: int) -> tuple[float, float]:
    prediction = max(range(10), key=lambda index: probabilities[index])
    wrong = max(value for index, value in enumerate(probabilities) if index != label)
    return float(prediction != label), wrong - probabilities[label]


def _strong_scores(panel: Mapping[int, Mapping[int, Mapping[str, Any]]], anchor: int) -> dict[str, dict[int, float]]:
    if anchor not in panel:
        raise StrongPointError("strong feature panel lacks requested anchor")
    raw = {
        "L_adversarial_probability_margin_risk": {
            item: -float(row["adv_probability_margin"]) for item, row in panel[anchor].items()
        },
        "L_adversarial_logit_margin_risk": {
            item: -float(row["adv_logit_margin"]) for item, row in panel[anchor].items()
        },
        "L_adversarial_cross_entropy": {item: float(row["adv_ce"]) for item, row in panel[anchor].items()},
        "D_clean_to_adversarial_probability_margin_drop": {
            item: float(row["probability_drop"]) for item, row in panel[anchor].items()
        },
        "D_clean_to_adversarial_logit_margin_drop": {
            item: float(row["logit_drop"]) for item, row in panel[anchor].items()
        },
        "D_teacher_clean_adversarial_js": {item: float(row["teacher_js"]) for item, row in panel[anchor].items()},
        "D_teacher_adversarial_wrong_confidence": {
            item: _teacher_wrong_confidence(row["teacher_adv"], int(row["class_id"]))[1]
            for item, row in panel[anchor].items()
        },
        "D_teacher_adversarial_incorrect": {
            item: _teacher_wrong_confidence(row["teacher_adv"], int(row["class_id"]))[0]
            for item, row in panel[anchor].items()
        },
    }
    ranked = {name: deterministic_midranks(values) for name, values in raw.items()}
    ranked["strong_LD_equal_rank"] = equal_rank_score(raw)
    return ranked


def _online_scores(online: Mapping[int, Mapping[int, Mapping[str, Any]]], anchor: int) -> dict[str, dict[int, float]]:
    if anchor not in online:
        raise StrongPointError("online state lacks requested anchor")
    raw = {
        "S_online_correctness_frequency_risk": {
            item: float(row["frequency_risk"]) for item, row in online[anchor].items()
        },
        "S_online_margin_ema_risk": {item: float(row["margin_risk"]) for item, row in online[anchor].items()},
        "S_online_last_margin_risk": {item: float(row["last_margin_risk"]) for item, row in online[anchor].items()},
    }
    ranked = {name: deterministic_midranks(values) for name, values in raw.items()}
    ranked["online_history_equal_rank"] = equal_rank_score(raw)
    return ranked


def _conditional(scores: Mapping[str, Mapping[int, float]], ids: set[int]) -> dict[str, dict[int, float]]:
    return {name: {item: values[item] for item in sorted(ids)} for name, values in scores.items()}


def _top_mask(scores: Mapping[int, float]) -> set[int]:
    if not scores:
        return set()
    order = tuple(sorted(scores, key=lambda item: (-scores[item], item)))
    nominal = max(1, math.ceil(TOP_FRACTION * len(order)))
    boundary = scores[order[nominal - 1]]
    return {item for item in order if scores[item] >= boundary}


def _top_mask_key(key: object) -> tuple[str, int, str, str]:
    if not isinstance(key, str):
        raise StrongPointError("top10 mask key is invalid")
    parts = key.split(":", 4)
    if len(parts) != 5 or parts[0] != "top10" or not parts[1] or not parts[3] or not parts[4]:
        raise StrongPointError("top10 mask key is invalid")
    try:
        anchor = int(parts[2])
    except ValueError as exc:
        raise StrongPointError("top10 mask anchor is invalid") from exc
    if anchor not in ANCHORS:
        raise StrongPointError("top10 mask anchor is outside the frozen analysis contract")
    return parts[1], anchor, parts[3], parts[4]


def _selection_metadata(mask: set[int], values: Mapping[int, float], metric: Mapping[str, Any]) -> dict[str, float | int]:
    top_percent = metric.get("top_percent")
    summary = top_percent.get("0.1") if isinstance(top_percent, Mapping) else None
    realized_count = len(mask)
    realized_fraction = realized_count / len(values) if values else 0.0
    if isinstance(summary, Mapping):
        if summary.get("realized_count") != realized_count or not math.isclose(
            _finite(summary.get("realized_fraction"), name="top10 realized fraction", lo=0, hi=1),
            realized_fraction,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise StrongPointError("top10 mask and metric realized selection drifted")
    return {
        "eligible_count": len(values),
        "realized_count": realized_count,
        "realized_fraction": realized_fraction,
    }


def _mask_comparison(
    before: set[int], after: set[int], before_meta: Mapping[str, object], after_meta: Mapping[str, object]
) -> dict[str, float | int | None]:
    def selection(meta: Mapping[str, object], mask: set[int]) -> tuple[int, float]:
        eligible = meta.get("eligible_count")
        if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < 1:
            raise StrongPointError("top10 eligible selection count is invalid")
        count = meta.get("realized_count")
        if isinstance(count, bool) or not isinstance(count, int) or count != len(mask) or count > eligible:
            raise StrongPointError("top10 realized selection count is invalid")
        fraction = _finite(meta.get("realized_fraction"), name="top10 realized selection fraction", lo=0, hi=1)
        if not math.isclose(fraction, count / eligible, rel_tol=0, abs_tol=1e-12):
            raise StrongPointError("top10 realized selection fraction is invalid")
        return count, fraction

    before_count, before_fraction = selection(before_meta, before)
    after_count, after_fraction = selection(after_meta, after)
    union = before | after
    shared = before & after
    return {
        "jaccard": len(shared) / len(union) if union else None,
        "retention": len(shared) / len(before) if before else None,
        "entry": len(after - before),
        "exit": len(before - after),
        "from_realized_count": before_count,
        "from_realized_fraction": before_fraction,
        "to_realized_count": after_count,
        "to_realized_fraction": after_fraction,
    }


def _predictor_mask_stability(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    masks, metadata = report.get("_formula_masks"), report.get("_top_mask_metadata")
    if not isinstance(masks, Mapping) or not isinstance(metadata, Mapping):
        raise StrongPointError("predictor mask stability inputs are incomplete")
    indexed: dict[tuple[str, str, str], dict[int, tuple[set[int], Mapping[str, object]]]] = {}
    for key, raw_mask in masks.items():
        if not isinstance(key, str) or not key.startswith("top10:"):
            continue
        candidate, anchor, stratum, score = _top_mask_key(key)
        if not isinstance(raw_mask, set) or not isinstance(metadata.get(key), Mapping):
            raise StrongPointError("top10 predictor mask internal contract drifted")
        group = indexed.setdefault((candidate, stratum, score), {})
        if anchor in group:
            raise StrongPointError("top10 predictor masks duplicate an anchor")
        group[anchor] = (raw_mask, metadata[key])
    rows: list[dict[str, Any]] = []
    for (candidate, stratum, score), by_anchor in sorted(indexed.items()):
        if tuple(sorted(by_anchor)) != ANCHORS:
            raise StrongPointError("top10 predictor masks omit a frozen consecutive anchor")
        for left, right in zip(ANCHORS, ANCHORS[1:]):
            before, before_meta = by_anchor[left]
            after, after_meta = by_anchor[right]
            rows.append(
                {
                    "candidate_id": candidate,
                    "stratum": stratum,
                    "score": score,
                    "from_anchor": left,
                    "to_anchor": right,
                    **_mask_comparison(before, after, before_meta, after_meta),
                }
            )
    return rows


def _cross_seed_predictor_mask_stability(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[dict[str, Any]]:
    left_masks, right_masks = left.get("_formula_masks"), right.get("_formula_masks")
    left_meta, right_meta = left.get("_top_mask_metadata"), right.get("_top_mask_metadata")
    if not isinstance(left_masks, Mapping) or not isinstance(right_masks, Mapping):
        raise StrongPointError("cross-seed predictor mask stability inputs are incomplete")
    if not isinstance(left_meta, Mapping) or not isinstance(right_meta, Mapping):
        raise StrongPointError("cross-seed predictor mask stability inputs are incomplete")
    shared_candidates = {
        key.removeprefix("future_failure:")
        for key in set(left_masks) & set(right_masks)
        if isinstance(key, str) and key.startswith("future_failure:")
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left_masks) & set(right_masks)):
        if not isinstance(key, str) or not key.startswith("top10:"):
            continue
        candidate, anchor, stratum, score = _top_mask_key(key)
        if candidate not in shared_candidates:
            continue
        before, after = left_masks[key], right_masks[key]
        before_meta, after_meta = left_meta.get(key), right_meta.get(key)
        if not isinstance(before, set) or not isinstance(after, set) or not isinstance(before_meta, Mapping) or not isinstance(after_meta, Mapping):
            raise StrongPointError("cross-seed top10 predictor mask contract drifted")
        rows.append(
            {
                "candidate_id": candidate,
                "anchor_epoch": anchor,
                "stratum": stratum,
                "score": score,
                **_mask_comparison(before, after, before_meta, after_meta),
            }
        )
    return rows


def analyze_strong_run(
    *,
    label: str,
    feature_observations: Path,
    feature_lineage: Path,
    outcome_observations: Path,
    outcome_lineage: Path,
    online_states: Path,
    online_lineage: Path,
    validation_history: Path,
    validation_manifest: Path,
    expected_count: int,
    expected_universe_sha256: str = EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
    scheduler_stages: Sequence[Sequence[int]] = ((0, 99), (100, 149), (150, 199)),
    anchors: Sequence[int] = ANCHORS,
    deltas_pp: Sequence[float] = (0.25, 0.5, 1.0),
    window_sizes: Sequence[int] = (3, 5, 7),
    thresholds: Sequence[str] = ("majority", "two_thirds", "all"),
) -> dict[str, Any]:
    """Calculate all point estimates without selecting a GT, score, or intervention."""
    if label not in {"L2", "L4"} or tuple(anchors) != ANCHORS:
        raise StrongPointError("strong point analysis is restricted to L2/L4 and anchors 39/59/79")
    feature_meta = _strong_lineage(
        path=feature_lineage,
        observations=feature_observations,
        role="feature",
        expected_count=expected_count,
        expected_universe_sha256=expected_universe_sha256,
    )
    outcome_meta = _strong_lineage(
        path=outcome_lineage,
        observations=outcome_observations,
        role="outcome",
        expected_count=expected_count,
        expected_universe_sha256=expected_universe_sha256,
    )
    online, online_meta = _online_panel(online_states, online_lineage, expected_count)
    identity = _normalized_identity(
        feature=feature_meta,
        outcome=outcome_meta,
        online=online_meta,
        validation_manifest=validation_manifest,
        validation_history=validation_history,
    )
    feature_epochs, outcome_epochs = tuple(feature_meta["requested_epochs"]), tuple(outcome_meta["requested_epochs"])
    if feature_epochs != ANCHORS or not outcome_epochs or any(anchor >= min(outcome_epochs) for anchor in anchors):
        raise StrongPointError("strong replay epoch coverage creates temporal leakage or omits an anchor")
    feature = _strong_panel(
        _read_parquet(feature_observations),
        epochs=feature_epochs,
        expected_count=expected_count,
        expected_universe_sha256=expected_universe_sha256,
    )
    outcome = _strong_panel(
        _read_parquet(outcome_observations),
        epochs=outcome_epochs,
        expected_count=expected_count,
        expected_universe_sha256=expected_universe_sha256,
    )
    ids = set(feature[ANCHORS[0]])
    if ids != set(outcome[outcome_epochs[0]]) or ids != set(online[ANCHORS[0]]):
        raise StrongPointError("strong feature/outcome/online stable-ID join drifted")
    if any(
        feature[ANCHORS[0]][item]["class_id"] != outcome[outcome_epochs[0]][item]["class_id"]
        or feature[ANCHORS[0]][item]["class_id"] != online[ANCHORS[0]][item]["class_id"]
        for item in ids
    ):
        raise StrongPointError("strong feature/outcome/online class join drifted")
    candidates = plateau_candidates(
        _load_validation(validation_history),
        saved_epochs=outcome_epochs,
        scheduler_stages=scheduler_stages,
        deltas_pp=deltas_pp,
        window_sizes=window_sizes,
        thresholds=thresholds,
    )
    strong = {anchor: _strong_scores(feature, anchor) for anchor in anchors}
    online_scores = {anchor: _online_scores(online, anchor) for anchor in anchors}
    rows: list[dict[str, Any]] = []
    masks: dict[str, set[int]] = {}
    top_mask_metadata: dict[str, dict[str, float | int]] = {}
    reports: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = f"d{candidate['delta_pp']:g}_k{candidate['window_size']}_{candidate['threshold']}"
        report: dict[str, Any] = {**candidate, "candidate_id": candidate_id, "anchors": {}}
        if candidate["censored"]:
            reports.append(report)
            continue
        future = {
            item: int(
                sum(not outcome[epoch][item]["correct"] for epoch in candidate["window_epochs"])
                >= candidate["required_wrong_checkpoints"]
            )
            for item in ids
        }
        masks[f"future_failure:{candidate_id}"] = {item for item, value in future.items() if value}
        for anchor in anchors:
            current = {item: bool(online[anchor][item]["current_correct"]) for item in ids}
            ff, current_wrong = split_ff_nr(future_failure=future, online_current_correct=current)
            strata = {
                "FF": (ff, {item for item in ids if current[item]}),
                "current_wrong_future_failure": (current_wrong, {item for item in ids if not current[item]}),
            }
            anchor_report: dict[str, Any] = {}
            for stratum, (targets, eligible) in strata.items():
                metrics: dict[str, Any] = {}
                for name, values in _conditional({**online_scores[anchor], **strong[anchor]}, eligible).items():
                    metrics[name] = _metric(values, {item: targets[item] for item in eligible})
                    mask_key = f"top10:{candidate_id}:{anchor}:{stratum}:{name}"
                    masks[mask_key] = _top_mask(values)
                    top_mask_metadata[mask_key] = _selection_metadata(masks[mask_key], values, metrics[name])
                    rows.append(
                        {
                            "run": label,
                            "candidate_id": candidate_id,
                            "anchor_epoch": anchor,
                            "stratum": stratum,
                            "score": name,
                            **metrics[name],
                        }
                    )
                anchor_report[stratum] = metrics
            report["anchors"][str(anchor)] = anchor_report
        reports.append(report)
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "scientific_status": "development_point_analysis_only_no_gt_or_predictor_selection_no_bootstrap",
        "label": label,
        "input_identity": {
            **identity,
            "stable_id_class_universe_sha256": expected_universe_sha256,
        },
        "ground_truth_attack_status": "selection_CE_PGD20_primary_point_analysis",
        "candidates": reports,
        "_point_rows": rows,
        "_formula_masks": masks,
        "_top_mask_metadata": top_mask_metadata,
    }


def write_strong_point_report(
    *, output_dir: Path, reports: Mapping[str, Mapping[str, Any]], config_path: Path
) -> dict[str, Path]:
    """Write non-overwriting point results and formula-level L2/L4 Jaccard."""
    paths = {
        "report": output_dir / "ffnr-strong-point-report.json",
        "points": output_dir / "ffnr-strong-points.parquet",
    }
    if output_dir.exists() or any(path.exists() for path in paths.values()):
        raise StrongPointError("refusing to overwrite strong point-analysis output")
    if set(reports) != {"L2", "L4"}:
        raise StrongPointError("strong point report requires exactly L2 and L4")
    analysis_provenance = _validated_analysis_provenance(_tracked_clean_provenance())
    left, right = reports["L2"], reports["L4"]
    if left.get("input_identity", {}).get("stable_id_class_universe_sha256") != right.get("input_identity", {}).get(
        "stable_id_class_universe_sha256"
    ):
        raise StrongPointError("cross-seed formula Jaccard requires an identical stable universe")
    left_masks, right_masks = left.get("_formula_masks"), right.get("_formula_masks")
    if not isinstance(left_masks, Mapping) or not isinstance(right_masks, Mapping):
        raise StrongPointError("cross-seed formula masks are incomplete")
    shared = sorted(
        key for key in set(left_masks) & set(right_masks) if isinstance(key, str) and key.startswith("future_failure:")
    )
    if not shared:
        raise StrongPointError("cross-seed analysis has no shared ground-truth candidate masks")
    jaccard = {
        key: len(set(left_masks[key]) & set(right_masks[key])) / len(set(left_masks[key]) | set(right_masks[key]))
        if set(left_masks[key]) | set(right_masks[key])
        else None
        for key in shared
    }
    within_run_stability = {label: _predictor_mask_stability(report) for label, report in reports.items()}
    cross_seed_stability = _cross_seed_predictor_mask_stability(left, right)
    points = write_sample_parquet(
        [row for report in reports.values() for row in report.get("_point_rows", [])], paths["points"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "analysis_provenance": analysis_provenance,
        "reports": {
            label: {key: value for key, value in report.items() if not key.startswith("_")}
            for label, report in reports.items()
        },
        "formula_level_cross_seed_jaccard": jaccard,
        "predictor_mask_stability": {
            "within_run_consecutive_anchor": within_run_stability,
            "cross_seed_top10": cross_seed_stability,
        },
        "points_sha256": sha256_file(points),
    }
    paths["report"].write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return paths
