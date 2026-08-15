"""Effect report for the fixed-anchor ERT T1/T2/T3 confirmatory screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ard.analysis.ert_stage_a_effect_decomposition import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _attack_identity,
    _bootstrap_ci,
    _cohort,
    _full_metrics,
    _identity,
    _json,
    _paired_arrays,
    _paired_metrics,
    _rows,
)
from ard.tracking.adapter import collect_git_state


class ConfirmatoryReportError(RuntimeError):
    """Raised when a confirmatory endpoint or lineage contract is invalid."""


ARMS = ("C79CONF", "T1WCONF", "T2WCONF", "T3LP05CONF")
MASK_KEYS = {"T1WCONF": "s3_t1_q10", "T2WCONF": "s3_t2_q10", "T3LP05CONF": "s3_t3_q10"}
HORIZONS = (84, 89, 94)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmatoryReportError(f"expected config mapping: {path}")
    return value


def _validate_fork(manifest_path: Path, *, parent_sha: str, arm: str, checkpoint: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    fork = manifest.get("fork_lineage")
    if not isinstance(fork, dict) or fork.get("parent_epoch") != 79:
        raise ConfirmatoryReportError(f"missing epoch-79 lineage: {manifest_path}")
    if fork.get("parent_checkpoint_sha256") != parent_sha or fork.get("arm") != arm:
        raise ConfirmatoryReportError(f"parent/arm lineage mismatch: {manifest_path}")
    checkpoint_sha = _sha256(checkpoint)
    artifacts = manifest.get("artifacts", [])
    if not any(
        isinstance(item, dict)
        and item.get("sha256") == checkpoint_sha
        and item.get("aliases") == [f"epoch-{checkpoint.stem.split('-')[-1]}"]
        for item in artifacts
    ):
        raise ConfirmatoryReportError(f"horizon checkpoint is not attested by its manifest: {manifest_path}")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "source_git_sha": manifest.get("git", {}).get("sha"),
        "run_id": manifest.get("run_id"),
    }


def _validate_rows_against(
    reference: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]], label: str
) -> None:
    if set(reference) != set(candidate):
        raise ConfirmatoryReportError(f"stable-ID universe mismatch: {label}")
    for sample_id in reference:
        if int(reference[sample_id]["true_label"]) != int(candidate[sample_id]["true_label"]):
            raise ConfirmatoryReportError(f"stable-ID class mismatch: {label}/{sample_id}")


def _endpoint(
    root: Path, seed: str, arm: str, horizon: int, split: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    directory = root / seed / arm / f"epoch-{horizon}" / split
    metadata_path = directory / "endpoint.json"
    rows_path = directory / "endpoint-sample-stats.parquet"
    if not metadata_path.is_file() or not rows_path.is_file():
        raise ConfirmatoryReportError(f"missing endpoint: {directory}")
    metadata = _json(metadata_path)
    if metadata.get("checkpoint_epoch") != horizon or metadata.get("dataset_scope") != split:
        raise ConfirmatoryReportError(f"wrong endpoint epoch/scope: {directory}")
    if metadata.get("rows_sha256") != _sha256(rows_path):
        raise ConfirmatoryReportError(f"endpoint row hash mismatch: {directory}")
    _attack_identity(metadata)
    return metadata, _rows(rows_path)


def build_confirmatory_report(
    *,
    endpoint_root: Path,
    training_root: Path,
    mask_paths: dict[str, Path],
    calibration_path: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise ConfirmatoryReportError("confirmatory report requires a clean source tree")
    config = _load_yaml(config_path)
    calibration = _json(calibration_path)
    if calibration.get("status") != "complete_no_update":
        raise ConfirmatoryReportError("calibration artifact is not a completed no-update artifact")
    if config.get("contract") != "ert_confirmatory_t123_v1":
        raise ConfirmatoryReportError("unexpected confirmatory config contract")
    expected_parents = {
        seed: calibration.get("inputs", {}).get(seed, {}).get("checkpoint_sha256") for seed in ("L2", "L4")
    }
    if any(not isinstance(value, str) for value in expected_parents.values()):
        raise ConfirmatoryReportError("calibration artifact lacks parent hashes")
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_confirmatory_t123_effect_report_v1",
        "source_git_sha": source["sha"],
        "config_sha256": _sha256(config_path),
        "calibration_sha256": _sha256(calibration_path),
        "horizons": list(HORIZONS),
        "arms": list(ARMS),
        "attack": None,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "stratified_by": "true_label"},
        "seeds": {},
        "inputs": {},
    }
    for seed_index, seed in enumerate(("L2", "L4")):
        seed_report: dict[str, Any] = {"parent_checkpoint_sha256": expected_parents[seed], "horizons": {}}
        for horizon in HORIZONS:
            train_rows: dict[str, dict[int, dict[str, Any]]] = {}
            validation_rows: dict[str, dict[int, dict[str, Any]]] = {}
            manifests: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                train_meta, train_data = _endpoint(endpoint_root, seed, arm, horizon, "train")
                val_meta, val_data = _endpoint(endpoint_root, seed, arm, horizon, "validation")
                train_digest, train_attack = _attack_identity(train_meta)
                val_digest, val_attack = _attack_identity(val_meta)
                if train_digest != val_digest or train_attack != val_attack:
                    raise ConfirmatoryReportError(f"train/validation attack mismatch: {seed}/{arm}/{horizon}")
                if report["attack"] is None:
                    report["attack"] = {"identity": train_attack, "identity_sha256": train_digest}
                elif report["attack"]["identity_sha256"] != train_digest:
                    raise ConfirmatoryReportError("endpoint attack identity differs between arms")
                checkpoint = training_root / seed / arm / "checkpoints" / f"epoch-{horizon}.pt"
                if train_meta.get("checkpoint_sha256") != _sha256(checkpoint) or val_meta.get(
                    "checkpoint_sha256"
                ) != _sha256(checkpoint):
                    raise ConfirmatoryReportError(f"endpoint checkpoint hash mismatch: {seed}/{arm}/{horizon}")
                manifest = _validate_fork(
                    training_root / seed / arm / "run-bundle" / "manifest.json",
                    parent_sha=expected_parents[seed],
                    arm=arm,
                    checkpoint=checkpoint,
                )
                train_rows[arm], validation_rows[arm], manifests[arm] = train_data, val_data, manifest
                report["inputs"][f"{seed}/{arm}/epoch-{horizon}"] = {
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "train_endpoint": str((endpoint_root / seed / arm / f"epoch-{horizon}" / "train").resolve()),
                    "validation_endpoint": str(
                        (endpoint_root / seed / arm / f"epoch-{horizon}" / "validation").resolve()
                    ),
                    "manifest": manifest,
                }
            train_control, val_control = train_rows["C79CONF"], validation_rows["C79CONF"]
            val_ids, train_ids = set(val_control), set(train_control)
            if len(train_ids) != 45000 or len(val_ids) != 5000 or train_ids & val_ids:
                raise ConfirmatoryReportError(f"unexpected split universe at {seed}/{horizon}")
            for arm in ARMS:
                _validate_rows_against(train_control, train_rows[arm], f"{seed}/{horizon}/train/{arm}")
                _validate_rows_against(val_control, validation_rows[arm], f"{seed}/{horizon}/validation/{arm}")
            horizon_report: dict[str, Any] = {
                "train_identity": _identity(train_control, name="train"),
                "validation_identity": _identity(val_control, name="validation"),
                "arms": {},
            }
            for arm in ARMS:
                if arm == "C79CONF":
                    horizon_report["arms"][arm] = {
                        "selected_count": 0,
                        "nonselected_count": len(train_ids),
                        "global_train": {
                            "count": len(train_ids),
                            "clean_accuracy_delta": 0.0,
                            "robust_accuracy_delta": 0.0,
                        },
                        "direct": None,
                        "spillover": None,
                        "heldout": {"count": len(val_ids), "clean_accuracy_delta": 0.0, "robust_accuracy_delta": 0.0},
                        "checkpoint": manifests[arm],
                    }
                    continue
                selected = _cohort(mask_paths[seed], MASK_KEYS[arm])
                if not selected.issubset(train_ids):
                    raise ConfirmatoryReportError(f"mask outside train universe: {seed}/{arm}")
                unselected = train_ids - selected
                direct = _paired_arrays(train_control, train_rows[arm], selected)
                spill = _paired_arrays(train_control, train_rows[arm], unselected)
                heldout = _paired_arrays(val_control, validation_rows[arm], val_ids)
                global_train = _full_metrics(train_control, train_rows[arm], train_ids)
                weighted_robust = (
                    len(selected) * direct["robust"].mean() + len(unselected) * spill["robust"].mean()
                ) / len(train_ids)
                weighted_clean = (
                    len(selected) * direct["clean"].mean() + len(unselected) * spill["clean"].mean()
                ) / len(train_ids)
                if (
                    abs(weighted_robust - float(global_train["robust_accuracy_delta"])) > 1e-12
                    or abs(weighted_clean - float(global_train["clean_accuracy_delta"])) > 1e-12
                ):
                    raise ConfirmatoryReportError(f"weighted direct/spillover identity failed: {seed}/{horizon}/{arm}")
                base_seed = BOOTSTRAP_SEED + seed_index * 100000 + (horizon - 84) * 1000 + list(ARMS).index(arm) * 10
                horizon_report["arms"][arm] = {
                    "selected_count": len(selected),
                    "nonselected_count": len(unselected),
                    "global_train": global_train,
                    "direct": {**_paired_metrics(direct), **_bootstrap_ci(direct, seed=base_seed)},
                    "spillover": {**_paired_metrics(spill), **_bootstrap_ci(spill, seed=base_seed + 1)},
                    "heldout": {**_paired_metrics(heldout), **_bootstrap_ci(heldout, seed=base_seed + 2)},
                    "weighted_identity": {
                        "robust_abs_error": abs(weighted_robust - float(global_train["robust_accuracy_delta"])),
                        "clean_abs_error": abs(weighted_clean - float(global_train["clean_accuracy_delta"])),
                    },
                    "checkpoint": manifests[arm],
                }
            seed_report["horizons"][str(horizon)] = horizon_report
        report["seeds"][seed] = seed_report
    if output.exists():
        raise ConfirmatoryReportError(f"refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(digest + "\n", encoding="ascii")
    return {**report, "output_sha256": digest}
