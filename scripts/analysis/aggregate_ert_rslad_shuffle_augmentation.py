#!/usr/bin/env python3
"""Aggregate the preregistered shuffle-vs-augmentation RNG campaign.

The script consumes only immutable local run bundles and independent endpoint
records.  It performs the matrix/lineage checks before calculating descriptive
effects; endpoint performance is never used to select an arm or checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TEACHERS = ("L2", "L4")
ARMS = ("REF1", "REF2", "SHUF1", "SHUF2", "AUG1", "AUG2", "BOTH1", "BOTH2")
HORIZONS = (84, 89, 94)
SPLITS = ("train", "validation")
SOURCE_ORDER = {
    "REF1": "S0/U0",
    "REF2": "S0/U0",
    "SHUF1": "S1/U0",
    "SHUF2": "S2/U0",
    "AUG1": "S0/U1",
    "AUG2": "S0/U2",
    "BOTH1": "S1/U1",
    "BOTH2": "S2/U2",
}
ARM_PURPOSE = {
    "REF1": "reference",
    "REF2": "exact-repeat residual control",
    "SHUF1": "shuffle only",
    "SHUF2": "shuffle only",
    "AUG1": "augmentation only",
    "AUG2": "augmentation only",
    "BOTH1": "shuffle and augmentation",
    "BOTH2": "shuffle and augmentation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def first_contract_record(path: Path) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        if '"contract":' not in line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON contract record in {path}")


def load_metrics(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        value = json.loads(line)
        epoch = int(value["epoch"])
        if epoch in rows:
            raise ValueError(f"duplicate epoch {epoch} in {path}")
        rows[epoch] = value
    expected = set(range(80, 95))
    if set(rows) != expected:
        raise ValueError(f"incomplete epoch range in {path}: {sorted(rows)}")
    return rows


def parse_endpoints(endpoint_root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in sorted(endpoint_root.glob("logs/*.log")):
        value = first_contract_record(path)
        if value.get("contract") != "ert_stage_a_common_ce_pgd20_endpoint_v1":
            raise ValueError(f"unexpected endpoint contract: {path}")
        checkpoint = Path(str(value["checkpoint"]))
        teacher = next((item for item in TEACHERS if item in checkpoint.parts), None)
        arm = next((item for item in ARMS if item in checkpoint.parts), None)
        if teacher is None or arm is None:
            raise ValueError(f"cannot infer endpoint identity from {checkpoint}")
        epoch = int(value["checkpoint_epoch"])
        split = str(value["dataset_scope"])
        key = (teacher, arm, epoch, split)
        if key in records:
            raise ValueError(f"duplicate endpoint record: {key}")
        expected_rows = 45_000 if split == "train" else 5_000
        if split not in SPLITS or epoch not in HORIZONS or int(value["row_count"]) != expected_rows:
            raise ValueError(f"bad endpoint identity in {path}")
        records[key] = {
            "teacher": teacher,
            "arm": arm,
            "epoch": epoch,
            "split": split,
            "clean_accuracy": float(value["clean_accuracy"]),
            "robust_accuracy": float(value["robust_accuracy"]),
            "checkpoint": str(value["checkpoint"]),
            "checkpoint_sha256": str(value["checkpoint_sha256"]),
            "rows_path": str(value["rows_path"]),
            "rows_sha256": str(value["rows_sha256"]),
            "row_count": expected_rows,
            "attack_identity_sha256": str(value["attack_identity_sha256"]),
            "attack": value.get("attack"),
            "contract": str(value["contract"]),
            "source_git_sha": str(value["source_git_sha"]),
            "source_log": str(path),
        }
    expected = {
        (teacher, arm, epoch, split) for teacher in TEACHERS for arm in ARMS for epoch in HORIZONS for split in SPLITS
    }
    if set(records) != expected:
        raise ValueError(
            "endpoint matrix mismatch; "
            f"missing={sorted(expected - set(records))}; extra={sorted(set(records) - expected)}"
        )
    return records


def load_trajectories(main_root: Path, registered: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for teacher in TEACHERS:
        expected_parent = registered["parents"][teacher]["parent_checkpoint_sha256"]
        expected_seeds = registered["seed_derivation"]["registry"][teacher]
        for arm in ARMS:
            run_root = main_root / teacher / arm
            manifest_path = run_root / "run-bundle/manifest.json"
            if not manifest_path.exists():
                raise ValueError(f"missing manifest: {manifest_path}")
            manifest = load_json(manifest_path)
            lineage = manifest.get("fork_lineage", {})
            if manifest.get("status") != "completed" or int(manifest.get("latest_progress", {}).get("epoch", -1)) != 94:
                raise ValueError(f"run is not complete: {run_root}")
            if (
                lineage.get("experiment_parent_epoch") != 79
                or lineage.get("experiment_parent_checkpoint_sha256") != expected_parent
            ):
                raise ValueError(f"wrong parent lineage: {run_root}")
            seeds = lineage.get("shuffle_augmentation_seeds")
            if not isinstance(seeds, dict):
                raise ValueError(f"missing split seed contract: {run_root}")
            arm_spec = next(item for item in registered["arms"] if item["arm"] == arm)
            expected_arm_seeds = {
                "shuffle_seed": expected_seeds[arm_spec["shuffle"]],
                "augmentation_seed": expected_seeds[arm_spec["augmentation"]],
                "attack_seed": expected_seeds[arm_spec["attack"]],
                "other_seed": expected_seeds[arm_spec["other"]],
            }
            if seeds != expected_arm_seeds:
                raise ValueError(f"split seed mismatch in {run_root}: {seeds} != {expected_arm_seeds}")
            metrics = load_metrics(run_root / "epoch-metrics.jsonl")
            checkpoint_hashes: dict[str, str] = {}
            for epoch in HORIZONS:
                checkpoint = run_root / "checkpoints" / f"epoch-{epoch}.pt"
                if not checkpoint.exists():
                    raise ValueError(f"missing horizon checkpoint: {checkpoint}")
                checkpoint_hashes[str(epoch)] = sha256_file(checkpoint)
            result[(teacher, arm)] = {
                "teacher": teacher,
                "arm": arm,
                "purpose": ARM_PURPOSE[arm],
                "source": SOURCE_ORDER[arm],
                "run_root": str(run_root),
                "manifest_path": str(manifest_path),
                "run_id": manifest.get("run_id"),
                "source_git_sha": manifest.get("git", {}).get("sha"),
                "config_hash": manifest.get("config_hash"),
                "training_seed": manifest.get("training_seed"),
                "shuffle_augmentation_seeds": seeds,
                "parent_checkpoint_sha256": lineage.get("experiment_parent_checkpoint_sha256"),
                "parent_epoch": lineage.get("experiment_parent_epoch"),
                "teacher_checkpoint_sha256": manifest.get("teacher", {}).get("checkpoint_sha256"),
                "training_execution_identity": manifest.get("training_execution_identity"),
                "summary": manifest.get("summary", {}),
                "epoch_metrics": {str(epoch): metrics[epoch] for epoch in sorted(metrics)},
                "horizon_checkpoint_sha256": checkpoint_hashes,
            }
    return result


def effect_rows(endpoints: dict[tuple[str, str, int, str], dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for teacher in TEACHERS:
        for epoch in HORIZONS:
            for split in SPLITS:
                base = endpoints[(teacher, "REF1", epoch, split)][metric]
                row: dict[str, Any] = {"teacher": teacher, "epoch": epoch, "split": split, "metric": metric}
                for arm in ARMS:
                    row[arm] = endpoints[(teacher, arm, epoch, split)][metric]
                row["reference"] = base
                row["ref_residual"] = row["REF2"] - base
                for source, pairs in {
                    "shuffle": ("SHUF1", "SHUF2"),
                    "augmentation": ("AUG1", "AUG2"),
                    "both": ("BOTH1", "BOTH2"),
                }.items():
                    for index, arm in enumerate(pairs, 1):
                        row[f"{source}_effect_{index}"] = row[arm] - base
                    values = [row[f"{source}_effect_{index}"] for index in (1, 2)]
                    row[f"{source}_mean"] = sum(values) / 2.0
                    row[f"{source}_mean_abs"] = sum(abs(value) for value in values) / 2.0
                row["interaction_1"] = (row["BOTH1"] - base) - (row["SHUF1"] - base) - (row["AUG1"] - base)
                row["interaction_2"] = (row["BOTH2"] - base) - (row["SHUF2"] - base) - (row["AUG2"] - base)
                row["interaction_mean"] = (row["interaction_1"] + row["interaction_2"]) / 2.0
                row["interaction_mean_abs"] = (abs(row["interaction_1"]) + abs(row["interaction_2"])) / 2.0
                rows.append(row)
    return rows


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(result: dict[str, Any]) -> str:
    endpoint = {
        tuple([parts[0], parts[1], int(parts[2]), parts[3]]): value
        for key, value in result["endpoint_records"].items()
        for parts in [key.split("|", 3)]
    }
    trajectories = {tuple(key.split("/", 1)): value for key, value in result["trajectories"].items()}
    lines = [
        "# ERT / RSLAD shuffle-vs-augmentation RNG decomposition results",
        "",
        "Status: completed for the preregistered 16 trajectories and 96 independent CE-PGD20 endpoints.",
        "",
        "## Scope and integrity",
        "",
        "This is a descriptive source decomposition. REF2 is reported as a residual control; the two perturbations "
        "are not population samples and no arm is promoted from this report.",
        "",
        f"- Source Git SHA: `{result['source_git_sha']}`",
        f"- Parent epoch: `79`; L2 `{result['parents']['L2']}`; L4 `{result['parents']['L4']}`",
        f"- Teacher checkpoint SHA: `{result['teacher_checkpoint_sha256']}`",
        f"- Training attack: `{json.dumps(result['training_attack'], sort_keys=True)}`",
        f"- Endpoint attack: `{json.dumps(result['endpoint_attack'], sort_keys=True)}`",
        f"- Endpoint attack identity SHA: `{result['endpoint_attack_identity_sha256']}`",
        f"- Trajectories: `{result['trajectory_count']}/16`; endpoints: `{result['endpoint_count']}/96`",
        f"- Input inventory SHA-256: `{result['input_inventory_sha256']}`",
        "",
        "All trajectory manifests reached epoch 94 and all endpoint rows have the registered train (45,000) or "
        "fixed validation (5,000) cardinality. No official test or AutoAttack was run.",
        "",
        "## Primary endpoint: epoch-94 validation CE-PGD20",
        "",
        "Absolute accuracy is shown; Δ is relative to REF1 within the same teacher and endpoint.",
        "",
    ]
    rows = []
    for teacher in TEACHERS:
        for arm in ARMS:
            record = endpoint[(teacher, arm, 94, "validation")]
            base = endpoint[(teacher, "REF1", 94, "validation")]
            rows.append(
                [
                    teacher,
                    arm,
                    fmt(record["clean_accuracy"]),
                    fmt(record["robust_accuracy"]),
                    fmt(record["robust_accuracy"] - base["robust_accuracy"]),
                ]
            )
    lines.append(table(["Teacher", "Arm", "Clean", "Robust", "Δ robust vs REF1"], rows))
    lines.extend(
        ["", "## Validation robust trajectory", "", "Independent CE-PGD20 robust accuracy at registered horizons.", ""]
    )
    rows = []
    for teacher in TEACHERS:
        for arm in ARMS:
            rows.append(
                [teacher, arm]
                + [fmt(endpoint[(teacher, arm, epoch, "validation")]["robust_accuracy"]) for epoch in HORIZONS]
            )
    lines.append(table(["Teacher", "Arm", "84", "89", "94"], rows))
    lines.extend(
        [
            "",
            "## Source decomposition at epoch 94 validation",
            "",
            "Effects use REF1 as the same-teacher reference. `I = both - shuffle - augmentation`.",
            "",
        ]
    )
    for metric, label in (("robust_accuracy", "Robust accuracy"), ("clean_accuracy", "Clean accuracy")):
        rows = []
        for teacher in TEACHERS:
            item = next(
                row
                for row in result["derived_effects"]
                if row["teacher"] == teacher
                and row["epoch"] == 94
                and row["split"] == "validation"
                and row["metric"] == metric
            )
            rows.append(
                [
                    teacher,
                    fmt(item["reference"]),
                    fmt(item["ref_residual"]),
                    fmt(item["shuffle_effect_1"]),
                    fmt(item["shuffle_effect_2"]),
                    fmt(item["augmentation_effect_1"]),
                    fmt(item["augmentation_effect_2"]),
                    fmt(item["both_effect_1"]),
                    fmt(item["both_effect_2"]),
                    fmt(item["interaction_1"]),
                    fmt(item["interaction_2"]),
                ]
            )
        lines.extend(
            [
                f"### {label}",
                "",
                table(["Teacher", "REF1", "REF2−REF1", "S1", "S2", "U1", "U2", "SU1", "SU2", "I1", "I2"], rows),
                "",
            ]
        )
    lines.extend(
        [
            "## Mean absolute source sensitivity",
            "",
            "Mean absolute effects average the two preregistered perturbations and are descriptive only.",
            "",
        ]
    )
    rows = []
    for teacher in TEACHERS:
        item = next(
            row
            for row in result["derived_effects"]
            if row["teacher"] == teacher
            and row["epoch"] == 94
            and row["split"] == "validation"
            and row["metric"] == "robust_accuracy"
        )
        rows.append(
            [
                teacher,
                fmt(item["shuffle_mean_abs"]),
                fmt(item["augmentation_mean_abs"]),
                fmt(item["both_mean_abs"]),
                fmt(item["interaction_mean_abs"]),
            ]
        )
    lines.append(table(["Teacher", "Shuffle", "Augmentation", "Both", "Interaction"], rows))
    lines.extend(
        [
            "",
            "## Training trajectory and best/last diagnostics",
            "",
            "These metrics come from the training log's fixed validation-PGD measurement; they are not the "
            "independent endpoint.",
            "",
        ]
    )
    rows = []
    for teacher in TEACHERS:
        for arm in ARMS:
            item = trajectories[(teacher, arm)]
            summary = item["summary"]
            epoch94 = item["epoch_metrics"]["94"]
            rows.append(
                [
                    teacher,
                    arm,
                    summary.get("best_epoch"),
                    fmt(summary.get("best_metric", float("nan"))),
                    fmt(epoch94.get("val_pgd_accuracy", float("nan"))),
                    fmt(epoch94.get("val_clean_accuracy", float("nan"))),
                    fmt(summary.get("best_metric", 0.0) - summary.get("last_pgd_accuracy", 0.0)),
                ]
            )
    lines.append(
        table(
            ["Teacher", "Arm", "Best epoch", "Best val PGD", "Epoch-94 val PGD", "Epoch-94 val clean", "Best−last gap"],
            rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation and stop boundary",
            "",
            "REF2−REF1 is retained as an observed same-source residual. The source effects describe this exact "
            "continuation and do not establish general dominance of shuffle or augmentation. No seed, checkpoint, "
            "attack, schedule, or future adaptive method was selected from the endpoint values.",
            "",
            "The preregistered campaign is complete. The result is recorded for human review; no stabilization run, "
            "new seed, official test, or AutoAttack was started automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(main_root: Path, endpoint_root: Path, experiment_path: Path) -> dict[str, Any]:
    registered = load_json(experiment_path)
    trajectories = load_trajectories(main_root, registered)
    endpoints = parse_endpoints(endpoint_root)
    source_shas = {str(value["source_git_sha"]) for value in trajectories.values()} | {
        value["source_git_sha"] for value in endpoints.values()
    }
    if len(source_shas) != 1:
        raise ValueError(f"mixed source Git SHAs: {source_shas}")
    parent_shas = {value["parent_checkpoint_sha256"] for value in trajectories.values()}
    expected_parents = {registered["parents"][teacher]["parent_checkpoint_sha256"] for teacher in TEACHERS}
    if parent_shas != expected_parents:
        raise ValueError(f"unexpected parent SHAs: {parent_shas}")
    attack_ids = {value["attack_identity_sha256"] for value in endpoints.values()}
    if len(attack_ids) != 1:
        raise ValueError(f"mixed endpoint attack identities: {attack_ids}")
    trajectory_json = {f"{teacher}/{arm}": value for (teacher, arm), value in trajectories.items()}
    endpoint_json = {
        "|".join((teacher, arm, str(epoch), split)): value for (teacher, arm, epoch, split), value in endpoints.items()
    }
    derived = effect_rows(endpoints, "robust_accuracy") + effect_rows(endpoints, "clean_accuracy")
    inventory = {
        "source_git_sha": sorted(source_shas),
        "parents": registered["parents"],
        "trajectory_keys": sorted(trajectory_json),
        "trajectory_checkpoints": {
            key: value["horizon_checkpoint_sha256"] for key, value in sorted(trajectory_json.items())
        },
        "endpoint_keys": sorted(endpoint_json),
        "endpoint_checkpoint_shas": {key: value["checkpoint_sha256"] for key, value in sorted(endpoint_json.items())},
        "endpoint_rows_shas": {key: value["rows_sha256"] for key, value in sorted(endpoint_json.items())},
    }
    inventory_sha = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment": registered["experiment"],
        "source_git_sha": sorted(source_shas)[0],
        "implementation_base_sha": registered["implementation_base_sha"],
        "production_source_sha": registered["production_source_sha"],
        "parent_epoch": 79,
        "parents": {teacher: registered["parents"][teacher]["parent_checkpoint_sha256"] for teacher in TEACHERS},
        "teacher_checkpoint_sha256": trajectories[("L2", "REF1")]["teacher_checkpoint_sha256"],
        "training_attack": registered["canonical_contract"]["training_attack"],
        "endpoint_attack": registered["canonical_contract"]["endpoint_attack"],
        "endpoint_attack_identity_sha256": sorted(attack_ids)[0],
        "endpoint_evaluation_seed": registered["canonical_contract"]["endpoint_evaluation_seed"],
        "input_inventory_sha256": inventory_sha,
        "trajectory_count": len(trajectories),
        "endpoint_count": len(endpoints),
        "trajectory_status": "16/16 completed through epoch 94",
        "endpoint_status": "96/96 complete (16 trajectories × 3 horizons × train/validation)",
        "trajectories": trajectory_json,
        "endpoint_records": endpoint_json,
        "derived_effects": derived,
        "inventory": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.main_root, args.endpoint_root, args.experiment)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(build_report(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "trajectory_count": result["trajectory_count"],
                "endpoint_count": result["endpoint_count"],
                "inventory_sha256": result["input_inventory_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
