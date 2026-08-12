"""Direct, spillover, and held-out decomposition for ERT Stage A."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ard.tracking.adapter import collect_git_state


class EffectDecompositionError(RuntimeError):
    """The effect-decomposition inputs violate a frozen contract."""


ARMS = (
    "C79",
    "ST1W",
    "ST1M",
    "ST1S",
    "ST2W",
    "ST2M",
    "ST2S",
    "ST3K1",
    "ST3K05",
    "ST3K0",
    "CW1",
    "CW2",
    "CW3",
)
MASK_KEYS = {
    "ST1W": "s3_t1_q10",
    "ST1M": "s3_t1_q10",
    "ST1S": "s3_t1_q10",
    "ST2W": "s3_t2_q10",
    "ST2M": "s3_t2_q10",
    "ST2S": "s3_t2_q10",
    "ST3K1": "s3_t3_q10",
    "ST3K05": "s3_t3_q10",
    "ST3K0": "s3_t3_q10",
    "CW1": "student_clean_wrong",
    "CW2": "student_clean_wrong",
    "CW3": "student_clean_wrong",
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260813


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EffectDecompositionError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        item = row.get("sample_id")
        label = row.get("true_label")
        if not isinstance(item, int) or item in result or not isinstance(label, int):
            raise EffectDecompositionError(f"invalid or duplicate stable ID/class: {path}")
        result[item] = row
    if not result:
        raise EffectDecompositionError(f"empty endpoint table: {path}")
    return result


def _cohort(mask_path: Path, key: str) -> set[int]:
    payload = _json(mask_path)
    if payload.get("anchor_epoch") != 79:
        raise EffectDecompositionError(f"mask is not an epoch-79 bundle: {mask_path}")
    raw = payload.get("masks", {}).get(key)
    values = raw.get("selected_ids") if isinstance(raw, dict) else None
    if not isinstance(values, list) or any(not isinstance(x, int) or isinstance(x, bool) for x in values):
        raise EffectDecompositionError(f"invalid mask: {mask_path}::{key}")
    result = set(values)
    if len(result) != len(values):
        raise EffectDecompositionError(f"duplicate IDs in mask: {mask_path}::{key}")
    return result


def _identity(rows: dict[int, dict[str, Any]], *, name: str) -> dict[str, Any]:
    ids = sorted(rows)
    labels = [int(rows[item]["true_label"]) for item in ids]
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    payload = {"split": name, "sample_ids": ids, "labels": labels, "class_counts": counts}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"name": name, "count": len(ids), "class_counts": counts, "sample_id_label_sha256": digest}


def _attack_identity(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = metadata.get("attack")
    digest = metadata.get("attack_identity_sha256")
    if not isinstance(identity, dict) or not isinstance(digest, str):
        raise EffectDecompositionError("endpoint lacks attack identity")
    required = {
        "loss": "ce",
        "norm": "linf",
        "epsilon": "8/255",
        "step_size": "2/255",
        "steps": 20,
        "random_start": True,
        "input_domain": "pixel_0_1",
        "student_mode": "eval",
    }
    for key, expected in required.items():
        if identity.get(key) != expected:
            raise EffectDecompositionError(f"wrong CE-PGD20 identity field {key}: {identity.get(key)!r}")
    return digest, identity


def _validate_manifest(
    *,
    manifest_path: Path,
    endpoint_metadata: dict[str, Any],
    expected_parent_sha: str,
    arm: str,
    manifest_arm: str,
) -> dict[str, Any]:
    manifest = _json(manifest_path)
    fork = manifest.get("fork_lineage")
    if not isinstance(fork, dict) or fork.get("parent_epoch") != 79:
        raise EffectDecompositionError(f"missing epoch-79 parent lineage: {manifest_path}")
    if fork.get("parent_checkpoint_sha256") != expected_parent_sha:
        raise EffectDecompositionError(f"wrong parent checkpoint for {manifest_path}")
    if fork.get("arm") != manifest_arm:
        raise EffectDecompositionError(f"arm mismatch in manifest {manifest_path}")
    if endpoint_metadata.get("checkpoint_epoch") != 84:
        raise EffectDecompositionError(f"endpoint is not epoch 84: {arm}")
    endpoint_sha = endpoint_metadata.get("checkpoint_sha256")
    artifacts = manifest.get("artifacts", [])
    if not isinstance(endpoint_sha, str) or not any(
        item.get("sha256") == endpoint_sha for item in artifacts if isinstance(item, dict)
    ):
        raise EffectDecompositionError(f"endpoint checkpoint is not attested by manifest: {manifest_path}")
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": endpoint_sha,
        "parent_checkpoint_sha256": expected_parent_sha,
        "parent_epoch": 79,
        "training_source_git_sha": manifest.get("git", {}).get("sha"),
        "treatment": manifest.get("intervention"),
    }


def _paired_arrays(
    control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: Iterable[int]
) -> dict[str, Any]:
    ordered = sorted(ids)
    if not ordered or not set(ordered).issubset(control) or not set(ordered).issubset(treatment):
        raise EffectDecompositionError("paired cohort is not contained in both endpoint universes")
    c = [control[item] for item in ordered]
    t = [treatment[item] for item in ordered]
    labels = np.asarray([int(row["true_label"]) for row in c], dtype=np.int64)
    robust = np.asarray(
        [int(trow["robust_correct"]) - int(crow["robust_correct"]) for crow, trow in zip(c, t)],
        dtype=np.int8,
    )
    clean = np.asarray(
        [int(trow["clean_correct"]) - int(crow["clean_correct"]) for crow, trow in zip(c, t)],
        dtype=np.int8,
    )
    adv_margin = np.asarray(
        [
            float(trow["adversarial_probability_margin"])
            - float(crow["adversarial_probability_margin"])
            for crow, trow in zip(c, t)
        ],
        dtype=np.float64,
    )
    clean_margin = np.asarray(
        [float(trow["clean_probability_margin"]) - float(crow["clean_probability_margin"]) for crow, trow in zip(c, t)],
        dtype=np.float64,
    )
    return {
        "ids": ordered,
        "labels": labels,
        "robust": robust,
        "clean": clean,
        "adv_margin": adv_margin,
        "clean_margin": clean_margin,
    }


def _paired_metrics(arrays: dict[str, Any]) -> dict[str, Any]:
    robust = arrays["robust"]
    clean = arrays["clean"]
    n = int(len(robust))
    if n == 0:
        raise EffectDecompositionError("empty paired metric cohort")
    rescue = int(np.count_nonzero(robust == 1))
    harm = int(np.count_nonzero(robust == -1))
    return {
        "count": n,
        "robust_accuracy_delta": float(robust.mean()),
        "clean_accuracy_delta": float(clean.mean()),
        "rescue_count": rescue,
        "harm_count": harm,
        "net_rescue_count": rescue - harm,
        "rescue_rate": rescue / n,
        "harm_rate": harm / n,
        "net_rescue_rate": (rescue - harm) / n,
        "adversarial_margin_delta": float(arrays["adv_margin"].mean()),
        "clean_margin_delta": float(arrays["clean_margin"].mean()),
    }


def _bootstrap_ci(arrays: dict[str, Any], *, seed: int) -> dict[str, list[float]]:
    """Class-stratified paired bootstrap CI for robust and clean deltas."""
    labels = arrays["labels"]
    robust = arrays["robust"].astype(np.float64)
    clean = arrays["clean"].astype(np.float64)
    rng = np.random.default_rng(seed)
    classes = sorted(set(int(item) for item in labels))
    robust_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    clean_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    by_class = {label: np.flatnonzero(labels == label) for label in classes}
    for replicate in range(BOOTSTRAP_REPLICATES):
        robust_sum = 0.0
        clean_sum = 0.0
        for indices in by_class.values():
            sampled = indices[rng.integers(0, len(indices), size=len(indices))]
            robust_sum += float(robust[sampled].sum())
            clean_sum += float(clean[sampled].sum())
        n = len(labels)
        robust_values[replicate] = robust_sum / n
        clean_values[replicate] = clean_sum / n
    return {
        "robust_accuracy_delta_ci95": [float(x) for x in np.quantile(robust_values, [0.025, 0.975])],
        "clean_accuracy_delta_ci95": [float(x) for x in np.quantile(clean_values, [0.025, 0.975])],
    }


def _full_metrics(
    control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: Iterable[int]
) -> dict[str, float | int]:
    ordered = sorted(ids)
    if not ordered:
        raise EffectDecompositionError("empty full cohort")
    c = [control[item] for item in ordered]
    t = [treatment[item] for item in ordered]
    return {
        "count": len(ordered),
        "control_clean_accuracy": sum(bool(row["clean_correct"]) for row in c) / len(c),
        "treatment_clean_accuracy": sum(bool(row["clean_correct"]) for row in t) / len(t),
        "control_robust_accuracy": sum(bool(row["robust_correct"]) for row in c) / len(c),
        "treatment_robust_accuracy": sum(bool(row["robust_correct"]) for row in t) / len(t),
        "clean_accuracy_delta": sum(
            int(trow["clean_correct"]) - int(crow["clean_correct"]) for crow, trow in zip(c, t)
        )
        / len(c),
        "robust_accuracy_delta": sum(
            int(trow["robust_correct"]) - int(crow["robust_correct"]) for crow, trow in zip(c, t)
        )
        / len(c),
    }


def _class_metrics(rows: dict[int, dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_class: dict[int, list[dict[str, Any]]] = {}
    for row in rows.values():
        by_class.setdefault(int(row["true_label"]), []).append(row)
    return {
        str(label): {
            "count": len(values),
            "clean_accuracy": sum(bool(row["clean_correct"]) for row in values) / len(values),
            "robust_accuracy": sum(bool(row["robust_correct"]) for row in values) / len(values),
        }
        for label, values in sorted(by_class.items())
    }


def build_effect_report(
    *,
    train_root: Path,
    validation_root: Path,
    train_output_root: Path,
    mask_paths: dict[str, Path],
    calibration_path: Path,
    stage_a_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise EffectDecompositionError("effect report requires a clean evaluator tree")
    calibration = _json(calibration_path)
    input_report = _json(stage_a_report_path)
    if input_report.get("contract") != "ert_stage_a_treatment_report_v1":
        raise EffectDecompositionError("unexpected Stage A input report contract")
    expected_parent = {
        seed: calibration.get("inputs", {}).get(seed, {}).get("checkpoint_sha256") for seed in ("L2", "L4")
    }
    if any(not isinstance(value, str) for value in expected_parent.values()):
        raise EffectDecompositionError("calibration artifact lacks parent checkpoint hashes")
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_stage_a_effect_decomposition_v1",
        "source_git_sha": source["sha"],
        "stage_a_report_sha256": _sha256(stage_a_report_path),
        "calibration_sha256": _sha256(calibration_path),
        "attack": None,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "stratified_by": "true_label"},
        "seeds": {},
        "inputs": {},
    }
    for seed in ("L2", "L4"):
        train_rows: dict[str, dict[int, dict[str, Any]]] = {}
        validation_rows: dict[str, dict[int, dict[str, Any]]] = {}
        manifests: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            train_arm = arm if not (seed == "L2" and arm in {"ST1W", "ST2S"}) else f"{arm}-rerun"
            train_dir = train_root / seed / arm
            val_dir = validation_root / seed / arm
            train_json = _json(train_dir / "endpoint.json")
            val_json = _json(val_dir / "endpoint.json")
            train_path = train_dir / "endpoint-sample-stats.parquet"
            val_path = val_dir / "endpoint-sample-stats.parquet"
            if train_json.get("checkpoint_epoch") != 84 or val_json.get("checkpoint_epoch") != 84:
                raise EffectDecompositionError(f"non-epoch-84 endpoint: {seed}/{arm}")
            if val_json.get("dataset_scope") != "validation":
                raise EffectDecompositionError(f"validation artifact has wrong scope: {seed}/{arm}")
            train_data, val_data = _rows(train_path), _rows(val_path)
            train_rows[arm], validation_rows[arm] = train_data, val_data
            train_digest, train_attack = _attack_identity(train_json)
            val_digest, val_attack = _attack_identity(val_json)
            if train_digest != val_digest or train_attack != val_attack:
                raise EffectDecompositionError(f"attack identity mismatch: {seed}/{arm}")
            if report["attack"] is None:
                report["attack"] = {"identity": train_attack, "identity_sha256": train_digest}
            elif report["attack"]["identity_sha256"] != train_digest:
                raise EffectDecompositionError("attack identity differs between arms")
            manifest_path = train_output_root / seed / train_arm / "run-bundle" / "manifest.json"
            manifests[arm] = _validate_manifest(
                manifest_path=manifest_path,
                endpoint_metadata=train_json,
                expected_parent_sha=expected_parent[seed],
                arm=arm,
                manifest_arm=train_arm,
            )
            report["inputs"][f"{seed}/{arm}"] = {
                "train_endpoint": str(train_path.resolve()),
                "train_endpoint_sha256": _sha256(train_path),
                "validation_endpoint": str(val_path.resolve()),
                "validation_endpoint_sha256": _sha256(val_path),
                "train_endpoint_json_sha256": _sha256(train_dir / "endpoint.json"),
                "validation_endpoint_json_sha256": _sha256(val_dir / "endpoint.json"),
                "manifest": manifests[arm],
                "train_arm": train_arm,
            }
        train_control, val_control = train_rows["C79"], validation_rows["C79"]
        train_ids, val_ids = set(train_control), set(val_control)
        if train_ids & val_ids or len(train_ids | val_ids) != 50000:
            raise EffectDecompositionError(f"train/validation sample-ID overlap or incomplete universe: {seed}")
        train_identity, val_identity = _identity(train_control, name="train"), _identity(val_control, name="validation")
        if len(train_ids) != 45000 or len(val_ids) != 5000:
            raise EffectDecompositionError(f"unexpected split sizes for {seed}: {len(train_ids)}, {len(val_ids)}")
        seed_report: dict[str, Any] = {
            "parent_checkpoint_sha256": expected_parent[seed],
            "train_identity": train_identity,
            "validation_identity": val_identity,
            "arms": {},
        }
        for arm in ARMS:
            if arm == "C79":
                seed_report["arms"][arm] = {
                    "selected_count": 0,
                    "nonselected_count": len(train_ids),
                    "global_train": {
                        "count": len(train_ids),
                        "clean_accuracy_delta": 0.0,
                        "robust_accuracy_delta": 0.0,
                    },
                    "direct": None,
                    "spillover": None,
                    "heldout": {
                        "count": len(val_ids),
                        "clean_accuracy_delta": 0.0,
                        "robust_accuracy_delta": 0.0,
                        "clean_accuracy": _class_metrics(val_control),
                        "robust_accuracy": _class_metrics(val_control),
                    },
                    "checkpoint": report["inputs"][f"{seed}/{arm}"]["manifest"],
                }
                continue
            selected = _cohort(mask_paths[seed], MASK_KEYS[arm])
            if not selected.issubset(train_ids):
                raise EffectDecompositionError(f"mask is outside train universe: {seed}/{arm}")
            unselected = train_ids - selected
            if selected & unselected or selected | unselected != train_ids:
                raise EffectDecompositionError(f"selected/non-selected partition failed: {seed}/{arm}")
            direct_arrays = _paired_arrays(train_control, train_rows[arm], selected)
            spill_arrays = _paired_arrays(train_control, train_rows[arm], unselected)
            heldout_arrays = _paired_arrays(val_control, validation_rows[arm], val_ids)
            global_train = _full_metrics(train_control, train_rows[arm], train_ids)
            weighted_robust = (
                len(selected) * direct_arrays["robust"].mean()
                + len(unselected) * spill_arrays["robust"].mean()
            ) / len(train_ids)
            weighted_clean = (
                len(selected) * direct_arrays["clean"].mean()
                + len(unselected) * spill_arrays["clean"].mean()
            ) / len(train_ids)
            if (
                abs(weighted_robust - float(global_train["robust_accuracy_delta"])) > 1e-12
                or abs(weighted_clean - float(global_train["clean_accuracy_delta"])) > 1e-12
            ):
                raise EffectDecompositionError(f"weighted direct/spillover identity failed: {seed}/{arm}")
            base_seed = BOOTSTRAP_SEED + (0 if seed == "L2" else 100000) + ARMS.index(arm) * 1000
            seed_report["arms"][arm] = {
                "selected_count": len(selected),
                "nonselected_count": len(unselected),
                "global_train": global_train,
                "direct": {**_paired_metrics(direct_arrays), **_bootstrap_ci(direct_arrays, seed=base_seed)},
                "spillover": {**_paired_metrics(spill_arrays), **_bootstrap_ci(spill_arrays, seed=base_seed + 1)},
                "heldout": {
                    **_paired_metrics(heldout_arrays),
                    **_bootstrap_ci(heldout_arrays, seed=base_seed + 2),
                    "clean_accuracy": _class_metrics(validation_rows[arm]),
                    "robust_accuracy": _class_metrics(validation_rows[arm]),
                },
                "weighted_identity": {
                    "robust_abs_error": abs(
                        weighted_robust - float(global_train["robust_accuracy_delta"])
                    ),
                    "clean_abs_error": abs(
                        weighted_clean - float(global_train["clean_accuracy_delta"])
                    ),
                },
                "checkpoint": report["inputs"][f"{seed}/{arm}"]["manifest"],
            }
        report["seeds"][seed] = seed_report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(digest + "\n", encoding="ascii")
    return {**report, "output_sha256": digest}
