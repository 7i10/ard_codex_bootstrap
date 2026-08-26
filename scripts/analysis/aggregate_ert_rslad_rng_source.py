#!/usr/bin/env python3
"""Aggregate the preregistered ERT/RSLAD RNG-source campaign.

The campaign is deliberately aggregated from the immutable local run bundles and
the independent CE-PGD20 endpoint logs.  No checkpoint is selected from the
endpoint results and no outcome is used to alter the registered arm mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TEACHERS = ("L2", "L4")
ARMS = ("REF1", "REF2", "ATTACK1", "ATTACK2", "DATA1", "DATA2", "BOTH1", "BOTH2")
HORIZONS = (84, 89, 94)
SPLITS = ("train", "validation")
SOURCE_ORDER = {
    "REF1": "D0/A0/O0",
    "REF2": "D0/A0/O0",
    "ATTACK1": "D0/A1/O0",
    "ATTACK2": "D0/A2/O0",
    "DATA1": "D1/A0/O0",
    "DATA2": "D2/A0/O0",
    "BOTH1": "D1/A1/O0",
    "BOTH2": "D2/A2/O0",
}
ARM_PURPOSE = {
    "REF1": "reference",
    "REF2": "exact-repeat residual control",
    "ATTACK1": "attack RNG only",
    "ATTACK2": "attack RNG only",
    "DATA1": "data RNG only",
    "DATA2": "data RNG only",
    "BOTH1": "both changed",
    "BOTH2": "both changed",
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
        raise ValueError(f"expected object: {path}")
    return value


def json_line(path: Path, predicate: str) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if predicate in line:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise ValueError(f"no {predicate} JSON record in {path}")


def load_metrics(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        value = json.loads(line)
        epoch = int(value["epoch"])
        if epoch in rows:
            raise ValueError(f"duplicate epoch {epoch} in {path}")
        rows[epoch] = value
    if set(rows) != set(range(80, 95)):
        raise ValueError(f"incomplete epoch range in {path}: {sorted(rows)}")
    return rows


def parse_endpoint_logs(endpoint_root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in sorted(endpoint_root.glob("logs/*.log")):
        value = json_line(path, '"contract":')
        if value.get("contract") != "ert_stage_a_common_ce_pgd20_endpoint_v1":
            raise ValueError(f"unexpected endpoint contract: {path}")
        checkpoint = Path(str(value["checkpoint"]))
        parts = checkpoint.parts
        teacher = next((item for item in TEACHERS if item in parts), None)
        if teacher is None:
            raise ValueError(f"cannot infer teacher from {checkpoint}")
        arm = next((item for item in ARMS if item in parts), None)
        if arm is None:
            raise ValueError(f"cannot infer arm from {checkpoint}")
        epoch = int(value["checkpoint_epoch"])
        split = str(value["dataset_scope"])
        key = (teacher, arm, epoch, split)
        if key in records:
            raise ValueError(f"duplicate endpoint record: {key}")
        expected_rows = 45000 if split == "train" else 5000
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
            "row_count": int(value["row_count"]),
            "attack_identity_sha256": str(value["attack_identity_sha256"]),
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
            f"missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )
    return records


def discover_run(root: Path, teacher: str, arm: str) -> tuple[Path, Path]:
    candidates = [root / teacher / arm, root.with_name(root.name + "-retry2") / teacher / arm]
    completed: list[Path] = []
    for candidate in candidates:
        manifest_path = candidate / "run-bundle/manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        if manifest.get("status") == "completed" and int(manifest.get("latest_progress", {}).get("epoch", -1)) == 94:
            completed.append(candidate)
    if len(completed) != 1:
        raise ValueError(f"expected one completed run bundle for {teacher}/{arm}, found {completed}")
    return completed[0], completed[0] / "run-bundle/manifest.json"


def load_trajectories(main_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    trajectories: dict[tuple[str, str], dict[str, Any]] = {}
    for teacher in TEACHERS:
        for arm in ARMS:
            run_root, manifest_path = discover_run(main_root, teacher, arm)
            manifest = load_json(manifest_path)
            lineage = manifest.get("fork_lineage", {})
            metrics = load_metrics(run_root / "epoch-metrics.jsonl")
            checkpoints = {
                str(epoch): sha256_file(run_root / "checkpoints" / f"epoch-{epoch}.pt") for epoch in HORIZONS
            }
            if manifest.get("status") != "completed" or int(manifest.get("latest_progress", {}).get("epoch", -1)) != 94:
                raise ValueError(f"run is not complete: {run_root}")
            if lineage.get("experiment_parent_epoch") != 79:
                raise ValueError(f"wrong parent epoch: {run_root}")
            trajectories[(teacher, arm)] = {
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
                "training_seeds": manifest.get("fork_lineage", {}).get("rng_source_seeds"),
                "parent_checkpoint_sha256": lineage.get("experiment_parent_checkpoint_sha256"),
                "parent_epoch": lineage.get("experiment_parent_epoch"),
                "teacher_checkpoint_sha256": manifest.get("teacher", {}).get("checkpoint_sha256"),
                "training_execution_identity": manifest.get("training_execution_identity"),
                "summary": manifest.get("summary", {}),
                "best_last_gap": float(manifest.get("summary", {}).get("best_metric", 0.0))
                - float(manifest.get("summary", {}).get("last_pgd_accuracy", 0.0)),
                "epoch_94_training_metrics": metrics[94],
                "horizon_checkpoint_sha256": checkpoints,
            }
    return trajectories


def effect_rows(records: dict[tuple[str, str, int, str], dict[str, Any]], *, metric: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for teacher in TEACHERS:
        for epoch in HORIZONS:
            for split in SPLITS:
                base = records[(teacher, "REF1", epoch, split)][metric]
                row: dict[str, Any] = {
                    "teacher": teacher,
                    "epoch": epoch,
                    "split": split,
                    "metric": metric,
                    "reference": base,
                }
                for arm in ARMS:
                    row[arm] = records[(teacher, arm, epoch, split)][metric]
                row["ref_residual"] = row["REF2"] - base
                for source in ("attack", "data", "both"):
                    arms = {"attack": ("ATTACK1", "ATTACK2"), "data": ("DATA1", "DATA2"), "both": ("BOTH1", "BOTH2")}[
                        source
                    ]
                    values = [row[arm] - base for arm in arms]
                    row[f"{source}_effect_1"] = values[0]
                    row[f"{source}_effect_2"] = values[1]
                    row[f"{source}_mean"] = sum(values) / 2.0
                    row[f"{source}_mean_abs"] = sum(abs(value) for value in values) / 2.0
                row["delta_A_1"] = row["attack_effect_1"]
                row["delta_A_2"] = row["attack_effect_2"]
                row["delta_D_1"] = row["data_effect_1"]
                row["delta_D_2"] = row["data_effect_2"]
                row["delta_AD_1"] = row["both_effect_1"]
                row["delta_AD_2"] = row["both_effect_2"]
                row["interaction_1"] = (row["BOTH1"] - base) - (row["ATTACK1"] - base) - (row["DATA1"] - base)
                row["interaction_2"] = (row["BOTH2"] - base) - (row["ATTACK2"] - base) - (row["DATA2"] - base)
                row["interaction_mean"] = (row["interaction_1"] + row["interaction_2"]) / 2.0
                row["interaction_mean_abs"] = (abs(row["interaction_1"]) + abs(row["interaction_2"])) / 2.0
                row["S_A"] = row["attack_mean_abs"]
                row["S_D"] = row["data_mean_abs"]
                row["S_AD"] = row["both_mean_abs"]
                row["delta_REF"] = row["ref_residual"]
                result.append(row)
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    text = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        text.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(text)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def build_report(result: dict[str, Any]) -> str:
    endpoint = {
        (parts[0], parts[1], int(parts[2]), parts[3]): value
        for key, value in result["endpoint_records"].items()
        for parts in [key.split("|", 3)]
    }
    trajectories = {tuple(key.split("/", 1)): value for key, value in result["trajectories"].items()}
    lines = [
        "# ERT / RSLAD baseline RNG-source decomposition results",
        "",
        "Status: completed for the preregistered 16 trajectories and independent CE-PGD20 endpoint matrix.",
        "",
        "## Scope and integrity",
        "",
        "This report is descriptive. It does not treat the two source perturbations as population samples, "
        "does not select a winner for a future method, and does not include official test or AutoAttack.",
        "",
        f"- Source Git SHA: `{result['source_git_sha']}`",
        f"- Parent epoch: `{result['parent_epoch']}`; "
        f"L2 parent `{result['parents']['L2']}`; L4 parent `{result['parents']['L4']}`",
        f"- Teacher checkpoint SHA: `{result['teacher_checkpoint_sha256']}`",
        f"- Training attack: `{result['training_attack']}`",
        f"- Endpoint attack: `{result['endpoint_attack']}`; "
        f"attack identity SHA `{result['endpoint_attack_identity_sha256']}`",
        f"- Endpoint records: `{len(endpoint)}/96`; trajectory records: `{len(trajectories)}/16`",
        f"- Input inventory SHA-256: `{result['input_inventory_sha256']}`",
        "",
        "All trajectories reached epoch 94. Every endpoint record uses the same independent eval-mode CE-PGD20 "
        "contract, with 45,000 train rows or 5,000 fixed validation rows. The evaluation seed is 0.",
        "",
        "## Primary endpoint: epoch 94 validation",
        "",
        "The primary endpoint is independent validation CE-PGD20 robust accuracy. Values are absolute accuracies; "
        "no checkpoint was chosen from this table.",
        "",
    ]
    rows: list[list[Any]] = []
    for teacher in TEACHERS:
        for arm in ARMS:
            record = endpoint[(teacher, arm, 94, "validation")]
            rows.append(
                [
                    teacher,
                    arm,
                    fmt(record["clean_accuracy"]),
                    fmt(record["robust_accuracy"]),
                    fmt(record["robust_accuracy"] - endpoint[(teacher, "REF1", 94, "validation")]["robust_accuracy"]),
                ]
            )
    lines.append(markdown_table(["Teacher", "Arm", "Clean", "Robust", "Δ robust vs REF1"], rows))
    lines.extend(
        [
            "",
            "## Validation robust trajectory",
            "",
            "Absolute CE-PGD20 validation robust accuracy at epochs 84, 89, and 94.",
            "",
        ]
    )
    rows = []
    for teacher in TEACHERS:
        for arm in ARMS:
            rows.append(
                [teacher, arm]
                + [fmt(endpoint[(teacher, arm, epoch, "validation")]["robust_accuracy"]) for epoch in HORIZONS]
            )
    lines.append(markdown_table(["Teacher", "Arm", "84", "89", "94"], rows))
    lines.extend(
        [
            "",
            "## Source decomposition at the primary endpoint",
            "",
            "For each teacher and metric, REF1 is the reference. `attack`, `data`, and `both` effects are the two "
            "preregistered perturbations relative to REF1. Interaction is `both - attack - data` using matched "
            "replicate indices.",
            "",
        ]
    )
    for metric, label in (("robust_accuracy", "Robust accuracy"), ("clean_accuracy", "Clean accuracy")):
        lines.extend([f"### {label} (validation, epoch 94)", ""])
        rows = []
        for teacher in TEACHERS:
            row = next(
                item
                for item in result["derived_effects"]
                if item["teacher"] == teacher
                and item["epoch"] == 94
                and item["split"] == "validation"
                and item["metric"] == metric
            )
            rows.append(
                [
                    teacher,
                    fmt(row["reference"]),
                    fmt(row["ref_residual"]),
                    fmt(row["attack_effect_1"]),
                    fmt(row["attack_effect_2"]),
                    fmt(row["data_effect_1"]),
                    fmt(row["data_effect_2"]),
                    fmt(row["both_effect_1"]),
                    fmt(row["both_effect_2"]),
                    fmt(row["interaction_1"]),
                    fmt(row["interaction_2"]),
                ]
            )
        lines.append(
            markdown_table(["Teacher", "REF1", "REF2−REF1", "A1", "A2", "D1", "D2", "AD1", "AD2", "I1", "I2"], rows)
        )
        lines.append("")
    lines.extend(
        [
            "## Training-side epoch-94 summary",
            "",
            "These are the training process's validation-PGD metrics, not the independent CE-PGD20 endpoint. "
            "They are included to expose any selection/evaluation distinction.",
            "",
        ]
    )
    rows = []
    for teacher in TEACHERS:
        for arm in ARMS:
            item = trajectories[(teacher, arm)]
            metrics = item["epoch_94_training_metrics"]
            summary = item["summary"]
            rows.append(
                [
                    teacher,
                    arm,
                    summary.get("best_epoch"),
                    fmt(float(summary.get("best_metric", float("nan")))),
                    fmt(float(metrics["val_pgd_accuracy"])),
                    fmt(float(metrics["val_clean_accuracy"])),
                    fmt(float(metrics["train_robust_accuracy"])),
                ]
            )
    lines.append(
        markdown_table(
            [
                "Teacher",
                "Arm",
                "Best epoch",
                "Best val PGD",
                "Epoch-94 val PGD",
                "Epoch-94 val clean",
                "Epoch-94 train robust",
            ],
            rows,
        )
    )
    lines.extend(["", "## Interpretation", "", "### Reference residual", ""])
    for teacher in TEACHERS:
        row = next(
            item
            for item in result["derived_effects"]
            if item["teacher"] == teacher
            and item["epoch"] == 94
            and item["split"] == "validation"
            and item["metric"] == "robust_accuracy"
        )
        lines.append(
            f"- {teacher}: REF2−REF1 = `{fmt(row['ref_residual'])}` robust-accuracy points (absolute fraction). "
            "This is the observed same-source residual, not a population uncertainty estimate."
        )
    lines.extend(["", "### Source sensitivity", ""])
    for teacher in TEACHERS:
        row = next(
            item
            for item in result["derived_effects"]
            if item["teacher"] == teacher
            and item["epoch"] == 94
            and item["split"] == "validation"
            and item["metric"] == "robust_accuracy"
        )
        effects = {
            "attack": row["attack_mean_abs"],
            "data": row["data_mean_abs"],
            "interaction": row["interaction_mean_abs"],
        }
        ranked = ", ".join(
            f"{name}={fmt(value)}" for name, value in sorted(effects.items(), key=lambda item: item[1], reverse=True)
        )
        lines.append(f"- {teacher}: mean absolute source effects — {ranked}.")
    lines.extend(
        [
            "",
            "The decomposition identifies how these particular post-epoch-79 continuations diverged under the frozen "
            "protocol. It does not establish that one RNG source dominates in general, and it does not justify "
            "changing the attack, data pipeline, or training objective.",
            "",
            "## Registered stop boundary",
            "",
            "The preregistered campaign is complete. No stabilization run, new seed, official test, or AutoAttack "
            "was started automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(main_root: Path, endpoint_root: Path, experiment_json: Path) -> dict[str, Any]:
    registered = load_json(experiment_json)
    trajectories = load_trajectories(main_root)
    endpoint = parse_endpoint_logs(endpoint_root)
    source_shas = {item["source_git_sha"] for item in trajectories.values()} | {
        item["source_git_sha"] for item in endpoint.values()
    }
    if source_shas != {"09e627e95a66a136a0cc7aa15bcb4deab141c719"}:
        raise ValueError(f"unexpected source SHAs: {source_shas}")
    parent_shas = {item["parent_checkpoint_sha256"] for item in trajectories.values()}
    if parent_shas != {
        registered["parents"]["L2"]["parent_checkpoint_sha256"],
        registered["parents"]["L4"]["parent_checkpoint_sha256"],
    }:
        raise ValueError(f"unexpected parent SHAs: {parent_shas}")
    attack_ids = {item["attack_identity_sha256"] for item in endpoint.values()}
    if len(attack_ids) != 1:
        raise ValueError(f"mixed endpoint attack identities: {attack_ids}")
    trajectory_json = {f"{teacher}/{arm}": value for (teacher, arm), value in trajectories.items()}
    endpoint_json = {
        "|".join((teacher, arm, str(epoch), split)): value for (teacher, arm, epoch, split), value in endpoint.items()
    }
    derived = effect_rows(endpoint, metric="robust_accuracy") + effect_rows(endpoint, metric="clean_accuracy")
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
        "schema_version": 2,
        "status": "completed",
        "experiment": registered["experiment"],
        "source_git_sha": sorted(source_shas)[0],
        "implementation_base_sha": registered["implementation_base_sha"],
        "parent_epoch": 79,
        "parents": {teacher: registered["parents"][teacher]["parent_checkpoint_sha256"] for teacher in TEACHERS},
        "teacher_checkpoint_sha256": trajectories[("L2", "REF1")]["teacher_checkpoint_sha256"],
        "training_attack": registered["canonical_contract"]["training_attack"],
        "endpoint_attack": registered["canonical_contract"]["endpoint_attack"],
        "endpoint_attack_identity_sha256": sorted(attack_ids)[0],
        "endpoint_evaluation_seed": registered["canonical_contract"]["endpoint_evaluation_seed"],
        "input_inventory_sha256": inventory_sha,
        "trajectory_count": len(trajectories),
        "endpoint_count": len(endpoint),
        "trajectory_status": "16/16 completed through epoch 94",
        "endpoint_status": "96/96 complete (16 trajectories × 3 horizons × 2 splits)",
        "trajectories": trajectory_json,
        "best_last_gap": {
            key: value["best_last_gap"] for key, value in sorted(trajectory_json.items())
        },
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
