"""Frozen exploratory M2a treatment-utility point audit; no gradient replay."""

from __future__ import annotations

import hashlib
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ard.analysis.rescue_harm import RescueHarmError, _read_v3_observations
from ard.analysis.rslad_signal_replay import canonical_json
from ard.analysis.signal_audit import (
    _fit_logistic,
    _predict_logistic,
    binary_metrics,
    deterministic_hash_split,
    sha256_file,
)
from ard.engine.checkpoint import REQUIRED_KEYS


class TreatmentUtilityError(ValueError):
    """Raised when a point audit cannot prove its frozen inputs."""


ENDPOINT = {"PF": 119, "NR": 99}
MODEL_WIDTHS = {"S": 2, "T": 9, "S+T": 11}
EXPECTED_TRAIN_COUNT = 45_000
EXPECTED_ATTACK = {
    "loss": "ce",
    "norm": "linf",
    "epsilon": "8/255",
    "epsilon_value": 8 / 255,
    "step_size": "2/255",
    "step_size_value": 2 / 255,
    "steps": 20,
    "random_start": True,
    "input_domain": "pixel_0_1",
    "student_mode": "eval",
    "teacher_mode": "eval",
}


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise TreatmentUtilityError("expected lowercase SHA-256")
    return value


def _analysis_source() -> dict[str, Any]:
    """Bind the exact audit implementation and repository revision to the report."""
    analysis_path = Path(__file__).resolve()
    cli_path = analysis_path.parents[1] / "cli" / "treatment_utility.py"
    repository = analysis_path.parents[3]
    try:
        relative = [str(path.relative_to(repository)) for path in (analysis_path, cli_path)]
        subprocess.run(
            ("git", "-C", str(repository), "ls-files", "--error-unmatch", *relative),
            check=True,
            capture_output=True,
            text=True,
        )
        git_sha = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "--verify", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise TreatmentUtilityError("treatment-utility source Git identity is unavailable") from exc
    if len(git_sha) != 40 or any(char not in "0123456789abcdef" for char in git_sha) or dirty:
        raise TreatmentUtilityError("treatment-utility audit requires a tracked-clean source revision")
    files = {"analysis_module": sha256_file(analysis_path), "cli_module": sha256_file(cli_path)}
    return {
        "git": {"sha": git_sha, "dirty": False},
        "files": files,
        "sha256": hashlib.sha256(canonical_json(files)).hexdigest(),
    }


def _expected_attack(attack: Mapping[str, Any]) -> bool:
    for key, expected in EXPECTED_ATTACK.items():
        value = attack.get(key)
        if isinstance(expected, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15):
                return False
        elif value != expected:
            return False
    return True


def _panel(
    observations: Path, lineage: Path, *, arm: str
) -> tuple[dict[str, Any], dict[int, dict[int, dict[str, Any]]]]:
    try:
        meta, panels = _read_v3_observations(observations, lineage, arm=arm, require_parent_epoch=arm == "C")
    except RescueHarmError as exc:
        raise TreatmentUtilityError("v3 input contract drifted") from exc
    dataset = meta.get("dataset_identity")
    attack = meta.get("attack_identity")
    nested_dataset = dataset.get("dataset") if isinstance(dataset, Mapping) else None
    expected_rows = EXPECTED_TRAIN_COUNT * len(panels)
    if (
        not isinstance(nested_dataset, Mapping)
        or nested_dataset.get("name") != "cifar10"
        or nested_dataset.get("split") != "train"
        or dataset.get("train_expected_count") != EXPECTED_TRAIN_COUNT
        or meta.get("row_count") != expected_rows
        or any(len(rows) != EXPECTED_TRAIN_COUNT for rows in panels.values())
        or not isinstance(attack, Mapping)
        or not _expected_attack(attack)
        or "autoattack" in canonical_json(meta).decode().lower()
    ):
        raise TreatmentUtilityError("v3 lineage/dataset/attack contract drifted or official evaluation is forbidden")
    return meta, panels


def _state(path: Path, *, parent: Mapping[str, Any]) -> tuple[dict[int, tuple[int, float, float]], str, str]:
    if not path.is_file() or sha256_file(path) != _sha(parent.get("checkpoint_sha256")):
        raise TreatmentUtilityError("parent checkpoint bytes drifted")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover
        raise TreatmentUtilityError("parent checkpoint is unreadable") from exc
    state = payload.get("sample_state") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or REQUIRED_KEYS.difference(payload)
        or payload.get("epoch") != 79
        or not isinstance(state, Mapping)
    ):
        raise TreatmentUtilityError("parent must be an exact epoch-79 checkpoint with sample state")
    state_sha = hashlib.sha256(canonical_json(state)).hexdigest()
    if state_sha != _sha(parent.get("sample_state_sha256")):
        raise TreatmentUtilityError("parent sample-state SHA drifted")
    records = state.get("records")
    if not isinstance(records, Mapping):
        raise TreatmentUtilityError("parent sample-state records are unavailable")
    result: dict[int, tuple[int, float, float]] = {}
    for raw_id, row in records.items():
        if (
            not isinstance(raw_id, str)
            or not raw_id.isdigit()
            or str(int(raw_id)) != raw_id
            or not isinstance(row, Mapping)
        ):
            raise TreatmentUtilityError("parent stable-ID record drifted")
        label, seen, hits, margin = (
            row.get("true_label"),
            row.get("seen"),
            row.get("robust_correct_count"),
            row.get("margin_ema"),
        )
        if (
            isinstance(label, bool)
            or not isinstance(label, int)
            or isinstance(seen, bool)
            or not isinstance(seen, int)
            or isinstance(hits, bool)
            or not isinstance(hits, int)
            or seen < 1
            or hits < 0
            or hits > seen
            or not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
            or not -1.0 <= float(margin) <= 1.0
        ):
            raise TreatmentUtilityError("parent state feature is invalid")
        sample_id = int(raw_id)
        if sample_id in result:
            raise TreatmentUtilityError("parent stable-ID record collides")
        result[sample_id] = (label, 1.0 - hits / seen, -float(margin))
    return result, sha256_file(path), state_sha


def _features(control79: Mapping[str, Any], state: tuple[int, float, float]) -> tuple[float, ...]:
    fields = (
        "teacher_clean_correct",
        "teacher_clean_true_probability",
        "teacher_clean_probability_margin",
        "teacher_clean_entropy_normalized",
        "teacher_adversarial_correct",
        "teacher_clean_to_adversarial_prediction_flip",
        "teacher_clean_to_adversarial_true_probability_delta",
        "teacher_clean_to_adversarial_margin_delta",
        "teacher_clean_to_adversarial_kl",
    )
    values: list[float] = [state[1], state[2]]
    for name in fields:
        value = control79.get(name)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(math.log1p(float(value)) if name == "teacher_clean_to_adversarial_kl" else float(value))
        else:
            raise TreatmentUtilityError("epoch-79 teacher feature is unavailable/non-finite")
    return tuple(values)


def _calibration(target: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    return {"brier": sum((y - p) ** 2 for y, p in zip(target, score, strict=True)) / len(target)}


def _rank(rows: Sequence[dict[str, Any]], scores: Sequence[float], *, model: str) -> dict[str, Any]:
    ranked = sorted(zip(scores, rows, strict=True), key=lambda item: (-item[0], item[1]["sample_id"]))
    k = max(1, math.ceil(len(ranked) * 0.1))
    top = ranked[:k]
    rescues = sum(int(row["rescue"]) for _, row in top)
    total_rescues = sum(int(row["rescue"]) for row in rows)
    bins = []
    for index in range(10):
        group = ranked[index * len(ranked) // 10 : (index + 1) * len(ranked) // 10]
        bins.append({"count": len(group), "signed_net_rescue": sum(int(row["utility"]) for _, row in group)})
    return {
        "model": model,
        "top_ids": [row["sample_id"] for _, row in top],
        "precision_at_10pct": rescues / k,
        "recall_at_10pct": rescues / total_rescues if total_rescues else 0.0,
        "deciles": bins,
    }


def _metrics(rows: Sequence[dict[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    discordant = [index for index, row in enumerate(rows) if row["discordant"]]
    if not discordant:
        raise TreatmentUtilityError("held-out treatment utility has no discordant rows")
    target = [int(rows[index]["rescue"]) for index in discordant]
    predicted = [scores[index] for index in discordant]
    return {**binary_metrics(target, predicted), **_calibration(target, predicted), "count": len(discordant)}


def _model_features(row: Mapping[str, Any], *, model: str) -> tuple[float, ...]:
    features = row.get("features")
    if not isinstance(features, tuple) or len(features) != MODEL_WIDTHS["S+T"]:
        raise TreatmentUtilityError("frozen feature vector drifted")
    if model == "S":
        return features[: MODEL_WIDTHS["S"]]
    if model == "T":
        return features[MODEL_WIDTHS["S"] :]
    if model == "S+T":
        return features
    raise TreatmentUtilityError("unknown frozen treatment-utility model")


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def run_treatment_utility(
    *, panels: Mapping[str, tuple[Path, Path, Path]], output: Path, split_seed: int = 20260806
) -> dict[str, Any]:
    """Run the preregistered M2a point audit over L1/L3 PF/NR C/H/R panels."""
    if output.exists():
        raise FileExistsError("refusing to overwrite treatment-utility report")
    expected = {f"{seed}:{route}:{arm}" for seed in ("L1", "L3") for route in ("PF", "NR") for arm in ("C", "H", "R")}
    if set(panels) != expected:
        raise TreatmentUtilityError("audit requires exact L1/L3 PF/NR C/H/R panels")
    prepared: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    parent_identities: dict[str, dict[str, str]] = {}
    parent_observation_identities: dict[str, str] = {}
    for seed in ("L1", "L3"):
        for route, endpoint in ENDPOINT.items():
            control_meta, control = _panel(*panels[f"{seed}:{route}:C"][:2], arm="C")
            state, parent_sha, state_sha = _state(panels[f"{seed}:{route}:C"][2], parent=control_meta["parent"])
            parent_identities[f"{seed}:{route}"] = {
                "checkpoint_sha256": parent_sha,
                "sample_state_sha256": state_sha,
            }
            parent_observation_identities[f"{seed}:{route}"] = hashlib.sha256(
                canonical_json([control[79][sample_id] for sample_id in sorted(control[79])])
            ).hexdigest()
            for kind, path in zip(("observations", "lineage", "parent"), panels[f"{seed}:{route}:C"], strict=True):
                input_hashes[f"{seed}:{route}:C:{kind}"] = sha256_file(path)
            if endpoint not in control or 79 not in control:
                raise TreatmentUtilityError("control lacks frozen endpoint/parent observations")
            for arm in ("H", "R"):
                meta, child = _panel(*panels[f"{seed}:{route}:{arm}"][:2], arm=f"{route}-{arm}")
                parent = meta.get("parent")
                if (
                    not isinstance(parent, Mapping)
                    or parent.get("checkpoint_sha256") != parent_sha
                    or parent.get("sample_state_sha256") != state_sha
                ):
                    raise TreatmentUtilityError("child parent lineage drifted")
                if any(
                    meta.get(key) != control_meta.get(key)
                    for key in (
                        "seed",
                        "analysis_seed",
                        "teacher",
                        "dataset_identity",
                        "attack_identity",
                        "analysis_provenance",
                    )
                ):
                    raise TreatmentUtilityError("child/control lineage source/teacher/attack drifted")
                if (
                    endpoint not in child
                    or set(child[endpoint]) != set(control[endpoint])
                    or set(control[79]) != set(control[endpoint])
                ):
                    raise TreatmentUtilityError("endpoint sparse-ID population drifted")
                rows: list[dict[str, Any]] = []
                for sample_id, c in control[endpoint].items():
                    a, anchor = child[endpoint][sample_id], control[79][sample_id]
                    if (
                        a.get("class_id") != c.get("class_id")
                        or sample_id not in state
                        or state[sample_id][0] != c.get("class_id")
                    ):
                        raise TreatmentUtilityError("stable sample/class join drifted")
                    selected = bool(a.get("mask_selected"))
                    utility = int(bool(a.get("robust_correct"))) - int(bool(c.get("robust_correct")))
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "namespace": "train",
                            "class_id": int(c["class_id"]),
                            "selected": selected,
                            "utility": utility,
                            "discordant": utility != 0,
                            "rescue": int(utility == 1),
                            "features": _features(anchor, state[sample_id]),
                        }
                    )
                prepared[f"{seed}:{route}:{arm}"] = rows
                for kind, path in zip(
                    ("observations", "lineage", "parent"), panels[f"{seed}:{route}:{arm}"], strict=True
                ):
                    input_hashes[f"{seed}:{route}:{arm}:{kind}"] = sha256_file(path)
    for seed in ("L1", "L3"):
        if parent_identities[f"{seed}:PF"] != parent_identities[f"{seed}:NR"]:
            raise TreatmentUtilityError("PF/NR controls must bind one exact epoch-79 parent per seed")
        if parent_observation_identities[f"{seed}:PF"] != parent_observation_identities[f"{seed}:NR"]:
            raise TreatmentUtilityError("PF/NR controls must bind one exact epoch-79 observation panel per seed")
    reference_population = {row["sample_id"]: row["class_id"] for row in next(iter(prepared.values()))}
    if any({row["sample_id"]: row["class_id"] for row in rows} != reference_population for rows in prepared.values()):
        raise TreatmentUtilityError("all seeds/routes/arms must share one sparse sample-ID/class population")
    split_index: dict[int, dict[str, Any]] = {}
    for key, rows in prepared.items():
        for row in rows:
            prior = split_index.setdefault(int(row["sample_id"]), row)
            if prior["class_id"] != row["class_id"]:
                raise TreatmentUtilityError("one stable sample ID cannot span classes")
    train_ids, held_ids = deterministic_hash_split(list(split_index.values()), seed=split_seed, held_out_fraction=0.2)
    train_set, held_set = set(train_ids), set(held_ids)
    if train_set & held_set:
        raise TreatmentUtilityError("sample-ID split leakage")
    fits: dict[str, dict[str, Any]] = {}
    for route in ENDPOINT:
        pooled = [
            row
            for seed in ("L1", "L3")
            for row in prepared[f"{seed}:{route}:H"]
            if row["selected"] and row["discordant"] and row["sample_id"] in train_set
        ]
        fits[route] = {
            name: _fit_logistic([_model_features(row, model=name) for row in pooled], [row["rescue"] for row in pooled])
            for name in MODEL_WIDTHS
        }
    reports: dict[str, Any] = {}
    for key, rows in prepared.items():
        _, route, _ = key.split(":", maxsplit=2)
        subset = [row for row in rows if row["selected"] and row["sample_id"] in held_set]
        scores = {
            name: _predict_logistic(fits[route][name], [_model_features(row, model=name) for row in subset])
            for name in MODEL_WIDTHS
        }
        ranking = {name: _rank(subset, values, model=name) for name, values in scores.items()}
        reports[key] = {
            "selected_count": len(subset),
            "non_selected_spillover_net_rescue": sum(row["utility"] for row in rows if not row["selected"]),
            "models": {name: _metrics(subset, values) for name, values in scores.items()},
            "ranking": ranking,
            "top10_jaccard_pairwise": {
                "S_T": _jaccard(ranking["S"]["top_ids"], ranking["T"]["top_ids"]),
                "S_S+T": _jaccard(ranking["S"]["top_ids"], ranking["S+T"]["top_ids"]),
                "T_S+T": _jaccard(ranking["T"]["top_ids"], ranking["S+T"]["top_ids"]),
            },
        }
    route_prerequisites: dict[str, dict[str, bool]] = {}
    for route in ENDPOINT:
        h_keys = [f"{seed}:{route}:H" for seed in ("L1", "L3")]
        r_keys = [f"{seed}:{route}:R" for seed in ("L1", "L3")]
        route_prerequisites[route] = {
            "incremental_predictive_value": all(
                reports[key]["models"]["S+T"]["auroc"] - reports[key]["models"]["S"]["auroc"] >= 0.02
                and reports[key]["models"]["S+T"]["log_loss"] < reports[key]["models"]["S"]["log_loss"]
                for key in h_keys
            ),
            "top_decile_net_rescue_exceeds_bottom": all(
                reports[key]["ranking"]["S+T"]["deciles"][0]["signed_net_rescue"]
                > reports[key]["ranking"]["S+T"]["deciles"][-1]["signed_net_rescue"]
                for key in h_keys
            ),
            "matched_R_not_reversed": all(
                reports[key]["models"]["S+T"]["auroc"] >= reports[key]["models"]["S"]["auroc"] for key in r_keys
            ),
        }
    nominated_routes = [route for route, prerequisites in route_prerequisites.items() if all(prerequisites.values())]
    result = {
        "schema_version": 1,
        "contract": "prescriptive_v3_treatment_utility_m2a_point_v1",
        "exploratory_only": True,
        "bootstrap_required": bool(nominated_routes),
        "point_nomination": "pending_bootstrap" if nominated_routes else "not_nominated",
        "nominated_routes": nominated_routes,
        "equations": {
            "utility": "robust_correct_H - robust_correct_C",
            "S": "[1-robust_correct_frequency,-margin_ema]",
            "T": "epoch79 teacher clean/student-adversarial response primitives",
        },
        "analysis_source": _analysis_source(),
        "split": {
            "seed": split_seed,
            "train_ids_sha256": hashlib.sha256(canonical_json(sorted(train_set))).hexdigest(),
            "held_out_ids_sha256": hashlib.sha256(canonical_json(sorted(held_set))).hexdigest(),
        },
        "endpoints": ENDPOINT,
        "input_hashes": dict(sorted(input_hashes.items())),
        "parent_identities": parent_identities,
        "parent_epoch79_observation_identities": parent_observation_identities,
        "point_prerequisites": route_prerequisites,
        "reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result) + b"\n")
    return result
