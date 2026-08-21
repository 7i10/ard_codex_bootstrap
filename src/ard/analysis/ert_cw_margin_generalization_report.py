"""Aggregate the frozen ERT Clean-Wrong margin-screen endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.tracking.adapter import collect_git_state


class MarginGeneralizationReportError(RuntimeError):
    """Raised when the endpoint panel violates the frozen screen contract."""


ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")
HORIZONS = (84, 89, 94)
SPLITS = ("train", "validation")
MASK_KEY = "student_clean_wrong"
PARENTS = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}
MASK_SHA256 = {
    "L2": "0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b",
    "L4": "fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6",
}
CALIBRATION_SHA256 = "a625b43ec12277bbf698270193f27e0e1f62e0a2a9f9a6a49e7fc0702593b2b5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MarginGeneralizationReportError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result: dict[int, dict[str, Any]] = {}
    labels: dict[int, int] = {}
    for row in values:
        sample_id, label = row.get("sample_id"), row.get("true_label")
        if not isinstance(sample_id, int) or isinstance(sample_id, bool) or sample_id in result:
            raise MarginGeneralizationReportError(f"invalid or duplicate stable ID: {path}")
        if not isinstance(label, int) or isinstance(label, bool):
            raise MarginGeneralizationReportError(f"invalid class label: {path}")
        result[sample_id] = row
        labels[sample_id] = label
    if not result:
        raise MarginGeneralizationReportError(f"empty endpoint table: {path}")
    return result


def _mask(path: Path) -> set[int]:
    if _sha256(path) not in MASK_SHA256.values():
        raise MarginGeneralizationReportError(f"unexpected mask hash: {path}")
    payload = _json(path)
    if payload.get("anchor_epoch") != 79:
        raise MarginGeneralizationReportError(f"mask is not anchored at epoch 79: {path}")
    raw = payload.get("masks", {}).get(MASK_KEY)
    values = raw.get("selected_ids") if isinstance(raw, dict) else None
    if not isinstance(values, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise MarginGeneralizationReportError(f"invalid Clean-Wrong mask: {path}")
    selected = set(values)
    if len(selected) != len(values):
        raise MarginGeneralizationReportError(f"duplicate IDs in Clean-Wrong mask: {path}")
    return selected


def _attack(meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = meta.get("attack")
    digest = meta.get("attack_identity_sha256")
    if not isinstance(identity, dict) or not isinstance(digest, str):
        raise MarginGeneralizationReportError("endpoint is missing attack identity")
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
            raise MarginGeneralizationReportError(f"wrong CE-PGD20 field {key}: {identity.get(key)!r}")
    return digest, identity


def _lineage(manifest_path: Path, *, parent: str, arm: str) -> dict[str, Any]:
    manifest = _json(manifest_path)
    fork = manifest.get("fork_lineage")
    if not isinstance(fork, dict) or fork.get("parent_epoch") != 79:
        raise MarginGeneralizationReportError(f"missing epoch-79 fork lineage: {manifest_path}")
    if fork.get("parent_checkpoint_sha256") != parent:
        raise MarginGeneralizationReportError(f"wrong parent checkpoint: {manifest_path}")
    if fork.get("arm") != arm:
        raise MarginGeneralizationReportError(f"arm mismatch in lineage: {manifest_path}")
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "source_git_sha": fork.get("source_git_sha"),
        "parent_checkpoint_sha256": parent,
        "parent_epoch": 79,
        "tracker_run_id": fork.get("child_tracker_run_id"),
    }


def _effect(control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    ordered = sorted(ids)
    if not ordered or not set(ordered).issubset(control) or not set(ordered).issubset(treatment):
        raise MarginGeneralizationReportError("paired cohort is not contained in endpoint universes")
    pairs = [(control[item], treatment[item]) for item in ordered]
    n = len(pairs)
    robust_delta = [int(t["robust_correct"]) - int(c["robust_correct"]) for c, t in pairs]
    clean_delta = [int(t["clean_correct"]) - int(c["clean_correct"]) for c, t in pairs]
    robust_rescue = sum(value == 1 for value in robust_delta)
    robust_harm = sum(value == -1 for value in robust_delta)
    clean_rescue = sum(value == 1 for value in clean_delta)
    clean_harm = sum(value == -1 for value in clean_delta)
    return {
        "count": n,
        "control_clean_accuracy": sum(bool(c["clean_correct"]) for c, _ in pairs) / n,
        "treatment_clean_accuracy": sum(bool(t["clean_correct"]) for _, t in pairs) / n,
        "clean_accuracy_delta": sum(clean_delta) / n,
        "clean_rescue_count": clean_rescue,
        "clean_harm_count": clean_harm,
        "clean_rescue_rate": clean_rescue / n,
        "clean_harm_rate": clean_harm / n,
        "clean_net_rescue_rate": (clean_rescue - clean_harm) / n,
        "control_robust_accuracy": sum(bool(c["robust_correct"]) for c, _ in pairs) / n,
        "treatment_robust_accuracy": sum(bool(t["robust_correct"]) for _, t in pairs) / n,
        "robust_accuracy_delta": sum(robust_delta) / n,
        "robust_rescue_count": robust_rescue,
        "robust_harm_count": robust_harm,
        "robust_rescue_rate": robust_rescue / n,
        "robust_harm_rate": robust_harm / n,
        "robust_net_rescue_rate": (robust_rescue - robust_harm) / n,
        "clean_margin_delta": sum(
            float(t["clean_probability_margin"]) - float(c["clean_probability_margin"]) for c, t in pairs
        )
        / n,
        "robust_margin_delta": sum(
            float(t["adversarial_probability_margin"]) - float(c["adversarial_probability_margin"]) for c, t in pairs
        )
        / n,
    }


def _absolute(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "count": n,
        "clean_accuracy": sum(bool(row["clean_correct"]) for row in rows.values()) / n,
        "robust_accuracy": sum(bool(row["robust_correct"]) for row in rows.values()) / n,
        "clean_probability_margin": sum(float(row["clean_probability_margin"]) for row in rows.values()) / n,
        "robust_probability_margin": sum(float(row["adversarial_probability_margin"]) for row in rows.values()) / n,
    }


def _pareto(effects: dict[str, dict[str, Any]]) -> list[str]:
    candidates = [arm for arm in ARMS if arm != "A0" and arm in effects]
    frontier: list[str] = []
    for arm in candidates:
        clean_a, robust_a = effects[arm]["clean_accuracy_delta"], effects[arm]["robust_accuracy_delta"]
        dominated = False
        for other in candidates:
            if other == arm:
                continue
            clean_b, robust_b = effects[other]["clean_accuracy_delta"], effects[other]["robust_accuracy_delta"]
            if clean_b >= clean_a and robust_b >= robust_a and (clean_b > clean_a or robust_b > robust_a):
                dominated = True
                break
        if not dominated:
            frontier.append(arm)
    return frontier


def build_report(
    *,
    endpoint_root: Path,
    training_root: Path,
    mask_paths: dict[str, Path],
    calibration: Path,
    output_json: Path,
    output_markdown: Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if (source.get("dirty") is not False and not allow_dirty) or not isinstance(source.get("sha"), str):
        raise MarginGeneralizationReportError("report requires a clean source tree")
    if _sha256(calibration) != CALIBRATION_SHA256:
        raise MarginGeneralizationReportError("calibration artifact hash mismatch")
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_margin_generalization_screen_v1",
        "report_source_git_sha": source["sha"],
        "report_source_dirty": bool(source.get("dirty")),
        "training_source_git_sha": None,
        "arms": list(ARMS),
        "horizons": list(HORIZONS),
        "splits": list(SPLITS),
        "endpoint_attack": None,
        "calibration_sha256": CALIBRATION_SHA256,
        "seeds": {},
    }
    for seed in ("L2", "L4"):
        mask = _mask(mask_paths[seed])
        if _sha256(mask_paths[seed]) != MASK_SHA256[seed]:
            raise MarginGeneralizationReportError(f"unexpected {seed} mask hash")
        lineage: dict[str, Any] = {}
        by_horizon: dict[str, Any] = {}
        for arm in ARMS:
            manifest = training_root / seed / arm / "run-bundle" / "manifest.json"
            lineage[arm] = _lineage(manifest, parent=PARENTS[seed], arm=arm)
        for horizon in HORIZONS:
            horizon_result: dict[str, Any] = {"arms": {}, "effects_vs_A0": {}, "pareto": {}}
            rows_by_arm: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
            for arm in ARMS:
                rows_by_arm[arm] = {}
                for split in SPLITS:
                    directory = endpoint_root / seed / arm / f"epoch-{horizon}" / split
                    metadata_path, rows_path = directory / "endpoint.json", directory / "endpoint-sample-stats.parquet"
                    metadata = _json(metadata_path)
                    if metadata.get("checkpoint_epoch") != horizon:
                        raise MarginGeneralizationReportError(f"wrong endpoint epoch: {metadata_path}")
                    attack_digest, attack_identity = _attack(metadata)
                    if report["endpoint_attack"] is None:
                        report["endpoint_attack"] = {"identity": attack_identity, "identity_sha256": attack_digest}
                    elif report["endpoint_attack"]["identity_sha256"] != attack_digest:
                        raise MarginGeneralizationReportError("endpoint attack identity differs")
                    if report["training_source_git_sha"] is None:
                        report["training_source_git_sha"] = metadata.get("source_git_sha")
                    elif report["training_source_git_sha"] != metadata.get("source_git_sha"):
                        raise MarginGeneralizationReportError("training source SHA differs")
                    expected_checkpoint = training_root / seed / arm / "checkpoints" / f"epoch-{horizon}.pt"
                    if metadata.get("checkpoint_sha256") != _sha256(expected_checkpoint):
                        raise MarginGeneralizationReportError(f"checkpoint hash mismatch: {metadata_path}")
                    rows = _rows(rows_path)
                    expected_count = 45000 if split == "train" else 5000
                    if len(rows) != expected_count:
                        raise MarginGeneralizationReportError(f"wrong {split} row count: {rows_path}")
                    rows_by_arm[arm][split] = rows
                    horizon_result["arms"].setdefault(arm, {})[split] = {
                        "absolute": _absolute(rows),
                        "endpoint_json": str(metadata_path.resolve()),
                        "endpoint_json_sha256": _sha256(metadata_path),
                        "rows": str(rows_path.resolve()),
                        "rows_sha256": _sha256(rows_path),
                        "checkpoint_sha256": metadata["checkpoint_sha256"],
                    }
            for split in SPLITS:
                control = rows_by_arm["A0"][split]
                universe = set(control)
                regions = {"overall": universe}
                if split == "train":
                    regions["direct_clean_wrong"] = mask
                    regions["spillover_non_clean_wrong"] = universe - mask
                for arm in ARMS[1:]:
                    effects = {name: _effect(control, rows_by_arm[arm][split], ids) for name, ids in regions.items()}
                    horizon_result["effects_vs_A0"].setdefault(arm, {})[split] = effects
                if split == "validation":
                    compact = {
                        arm: horizon_result["effects_vs_A0"].get(arm, {}).get(split, {}).get("overall", {})
                        for arm in ARMS[1:]
                    }
                    horizon_result["pareto"][split] = _pareto(compact)
            by_horizon[str(horizon)] = horizon_result
        report["seeds"][seed] = {
            "parent_checkpoint_sha256": PARENTS[seed],
            "mask_sha256": _sha256(mask_paths[seed]),
            "mask_count": len(mask),
            "lineage": lineage,
            "horizons": by_horizon,
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_sha = _sha256(output_json)
    output_json.with_name(output_json.name + ".sha256").write_text(output_sha + "\n", encoding="ascii")
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    return {"output_json_sha256": output_sha, "output_markdown_sha256": _sha256(output_markdown), **report}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ERT Clean-Wrong Margin Generalization Screen",
        "",
        "This is a preregistered point report for the fixed epoch-79 Clean-Wrong screen. "
        "No arm is promoted automatically.",
        "",
        f"- Report source Git SHA: `{report['report_source_git_sha']}`",
        f"- Training source Git SHA: `{report['training_source_git_sha']}`",
        f"- Endpoint attack: `{report['endpoint_attack']['identity_sha256']}` "
        "(CE-PGD20, pixel-space $L_\\u221e$, $8/255$)",
        f"- Calibration: `{report['calibration_sha256']}`",
        "- Bootstrap: not run; this report contains fixed point estimates and sample-level paired effects.",
        "",
        "## Endpoint robust accuracy",
        "",
        "Values are held-out validation robust accuracy; `Δ` is paired against A0 at the same seed and horizon.",
        "",
        "| seed | epoch | arm | clean | robust | robust Δ vs A0 | clean Δ vs A0 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for seed in ("L2", "L4"):
        for epoch in HORIZONS:
            arms = report["seeds"][seed]["horizons"][str(epoch)]["arms"]
            for arm in ARMS:
                absolute = arms[arm]["validation"]["absolute"]
                effect = (
                    {}
                    if arm == "A0"
                    else report["seeds"][seed]["horizons"][str(epoch)]["effects_vs_A0"][arm]["validation"]["overall"]
                )
                delta = 0.0 if arm == "A0" else effect["robust_accuracy_delta"]
                clean_delta = 0.0 if arm == "A0" else effect["clean_accuracy_delta"]
                lines.append(
                    f"| {seed} | {epoch} | {arm} | {absolute['clean_accuracy']:.4f} | "
                    f"{absolute['robust_accuracy']:.4f} | {delta:+.4f} | {clean_delta:+.4f} |"
                )
    lines.extend(
        [
            "",
            "## Direct / spillover effects at epoch 94",
            "",
            "| seed | arm | region | n | clean Δ | robust Δ | robust rescue | robust harm |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("L2", "L4"):
        effects = report["seeds"][seed]["horizons"]["94"]["effects_vs_A0"]
        for arm in ARMS[1:]:
            for region in ("direct_clean_wrong", "spillover_non_clean_wrong", "overall"):
                item = effects[arm]["train"][region]
                lines.append(
                    f"| {seed} | {arm} | {region} | {item['count']} | "
                    f"{item['clean_accuracy_delta']:+.4f} | {item['robust_accuracy_delta']:+.4f} | "
                    f"{item['robust_rescue_rate']:.4f} | {item['robust_harm_rate']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Validation Pareto fronts at epoch 94",
            "",
            "| seed | non-dominated arms (Δ clean, Δ robust) |",
            "|---|---|",
        ]
    )
    for seed in ("L2", "L4"):
        frontier = report["seeds"][seed]["horizons"]["94"]["pareto"]["validation"]
        lines.append(f"| {seed} | {', '.join(frontier) if frontier else 'none'} |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Direct effects are paired effects within the fixed epoch-79 Clean-Wrong cohort.",
            "- Spillover is the complement within the 45,000-sample train universe.",
            "- Validation is held-out internal validation, not official CIFAR-10 test.",
            "- Bootstrap, AutoAttack, official test, and automatic winner selection were not performed.",
            "- These results do not by themselves justify a new threshold, coefficient, or follow-up training arm.",
            "",
        ]
    )
    return "\n".join(lines)
