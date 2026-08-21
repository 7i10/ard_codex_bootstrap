"""Aggregate the frozen ERT Clean-Wrong margin-screen endpoints."""

from __future__ import annotations

import hashlib
import json
import math
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
FEATURE_ATTACK_SHA256 = {
    "CE20": "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2",
    "KL10": "98194e2a6ee02add8c675b0df1146007f371ed1811ef34b9ef37d052997348bd",
}
FEATURE_CONTRACT = {
    "CE20": {
        "train": "ert_clean_wrong_c0_ce_pgd20_features_v1",
        "validation": "ert_clean_wrong_validation_ce_pgd20_features_v1",
    },
    "KL10": {
        "train": "ert_clean_wrong_c0_kl_pgd10_features_v1",
        "validation": "ert_clean_wrong_validation_kl_pgd10_features_v1",
    },
}


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


def _feature_bundle(
    path: Path,
    *,
    parent: str,
    contract: str,
    attack_sha256: str,
    mask_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load one hash-bound epoch-79 feature panel for subtype transfer."""
    metadata = _json(path)
    if metadata.get("contract") != contract or metadata.get("feature_epoch") != 79:
        raise MarginGeneralizationReportError(f"feature contract/epoch mismatch: {path}")
    if metadata.get("checkpoint_sha256") != parent:
        raise MarginGeneralizationReportError(f"feature parent checkpoint mismatch: {path}")
    if metadata.get("attack_identity_sha256") != attack_sha256:
        raise MarginGeneralizationReportError(f"feature attack identity mismatch: {path}")
    if mask_sha256 is not None and metadata.get("mask_sha256") != mask_sha256:
        raise MarginGeneralizationReportError(f"feature mask identity mismatch: {path}")
    rows_path = Path(str(metadata.get("rows_path", "")))
    if not rows_path.is_file() or metadata.get("rows_sha256") != _sha256(rows_path):
        raise MarginGeneralizationReportError(f"feature rows hash mismatch: {path}")
    rows = _rows(rows_path)
    if any(
        not isinstance(row.get("teacher_adv_margin"), (float, int))
        or not math.isfinite(float(row["teacher_adv_margin"]))
        for row in rows.values()
    ):
        raise MarginGeneralizationReportError(f"feature rows lack finite Teacher margin: {rows_path}")
    identity = {
        "metadata": str(path.resolve()),
        "metadata_sha256": _sha256(path),
        "rows": str(rows_path.resolve()),
        "rows_sha256": metadata["rows_sha256"],
        "contract": contract,
        "feature_epoch": 79,
        "checkpoint_sha256": parent,
        "attack_identity_sha256": attack_sha256,
        "mask_sha256": metadata.get("mask_sha256"),
        "source_git_sha": metadata.get("source_git_sha"),
    }
    return identity, rows


def _ids_sha256(ids: set[int] | list[int]) -> str:
    ordered = sorted(ids)
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()


def _quantile_edges(rows: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    ordered = sorted(ids, key=lambda item: (float(rows[item]["teacher_adv_margin"]), item))
    if len(ordered) < 5:
        raise MarginGeneralizationReportError("Clean-Wrong feature panel is too small for Q1-Q5")
    result: dict[str, Any] = {}
    for index in range(5):
        start = (len(ordered) * index) // 5
        end = (len(ordered) * (index + 1)) // 5
        members = ordered[start:end]
        result[f"Q{index + 1}"] = {
            "count": len(members),
            "lower": float(rows[members[0]]["teacher_adv_margin"]),
            "upper": float(rows[members[-1]]["teacher_adv_margin"]),
            "ids_sha256": _ids_sha256(members),
        }
    return result


def _assign_train_edges(rows: dict[int, dict[str, Any]], ids: set[int], edges: dict[str, Any]) -> dict[str, set[int]]:
    result = {f"Q{index}": set() for index in range(1, 6)}
    for item in ids:
        value = float(rows[item]["teacher_adv_margin"])
        for index in range(1, 6):
            if value <= float(edges[f"Q{index}"]["upper"]):
                result[f"Q{index}"].add(item)
                break
        else:
            result["Q5"].add(item)
    if any(not values for values in result.values()):
        raise MarginGeneralizationReportError("validation Clean-Wrong Q1-Q5 contains an empty bin")
    return result


def _feature_consistency(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]], *, label: str) -> None:
    if set(left) != set(right):
        raise MarginGeneralizationReportError(f"{label}: stable-ID universe mismatch")
    if any(int(left[item]["true_label"]) != int(right[item]["true_label"]) for item in left):
        raise MarginGeneralizationReportError(f"{label}: stable-ID class mapping mismatch")


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


def _pp(value: float) -> str:
    return f"{value * 100:+.3f}"


def _rate_pp(value: float) -> str:
    return f"{value * 100:.3f}"


def build_report(
    *,
    endpoint_root: Path,
    training_root: Path,
    mask_paths: dict[str, Path],
    calibration: Path,
    output_json: Path,
    output_markdown: Path,
    allow_dirty: bool = False,
    train_ce_feature_root: Path | None = None,
    train_kl_feature_root: Path | None = None,
    validation_ce_feature_root: Path | None = None,
    validation_kl_feature_root: Path | None = None,
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
    repo_root = Path.cwd()
    train_ce_feature_root = train_ce_feature_root or repo_root / ".cache/analysis/ert-clean-wrong-subtypes-v4"
    train_kl_feature_root = train_kl_feature_root or repo_root / ".cache/analysis/ert-clean-wrong-reliability-proxy-v1"
    validation_ce_feature_root = validation_ce_feature_root or repo_root / ".cache/analysis/ert-cw-generalization-v1"
    validation_kl_feature_root = validation_kl_feature_root or repo_root / ".cache/analysis/ert-cw-generalization-v1"
    for seed in ("L2", "L4"):
        mask = _mask(mask_paths[seed])
        if _sha256(mask_paths[seed]) != MASK_SHA256[seed]:
            raise MarginGeneralizationReportError(f"unexpected {seed} mask hash")
        feature_paths = {
            "train_ce20": train_ce_feature_root / seed / "clean-wrong-feature-replay.json",
            "train_kl10": train_kl_feature_root / seed / "clean-wrong-kl10-feature-replay.json",
            "validation_ce20": validation_ce_feature_root / seed / "CE20" / "validation-feature-replay.json",
            "validation_kl10": validation_kl_feature_root / seed / "KL10" / "validation-feature-replay.json",
        }
        feature_identity: dict[str, Any] = {}
        feature_rows: dict[str, dict[int, dict[str, Any]]] = {}
        for feature_name, feature_path in feature_paths.items():
            domain = "train" if feature_name.startswith("train") else "validation"
            attack_name = "CE20" if "ce20" in feature_name else "KL10"
            mask_sha = MASK_SHA256[seed] if domain == "train" else None
            identity, rows = _feature_bundle(
                feature_path,
                parent=PARENTS[seed],
                contract=FEATURE_CONTRACT[attack_name][domain],
                attack_sha256=FEATURE_ATTACK_SHA256[attack_name],
                mask_sha256=mask_sha,
            )
            feature_identity[feature_name] = identity
            feature_rows[feature_name] = rows
        train_ce = feature_rows["train_ce20"]
        train_kl = feature_rows["train_kl10"]
        validation_ce = feature_rows["validation_ce20"]
        validation_kl = feature_rows["validation_kl10"]
        if set(train_ce) != mask or set(train_kl) != mask:
            raise MarginGeneralizationReportError(f"{seed}: train feature/Clean-Wrong mask mismatch")
        _feature_consistency(train_ce, train_kl, label=f"{seed} train CE20/KL10 features")
        _feature_consistency(validation_ce, validation_kl, label=f"{seed} validation CE20/KL10 features")
        if any(
            bool(validation_ce[item].get("student_clean_correct"))
            != bool(validation_kl[item].get("student_clean_correct"))
            for item in validation_ce
        ):
            raise MarginGeneralizationReportError(f"{seed}: CE20/KL10 validation Clean-Wrong predicate mismatch")
        if len(validation_ce) != 5000:
            raise MarginGeneralizationReportError(f"{seed}: validation feature count is not 5000")
        train_ce_edges = _quantile_edges(train_ce, mask)
        train_kl_edges = _quantile_edges(train_kl, mask)
        validation_clean_wrong = {
            item for item, row in validation_ce.items() if not bool(row.get("student_clean_correct"))
        }
        if not validation_clean_wrong:
            raise MarginGeneralizationReportError(f"{seed}: validation Clean-Wrong cohort is empty")
        validation_ce_q = _assign_train_edges(validation_ce, validation_clean_wrong, train_ce_edges)
        validation_kl_q = _assign_train_edges(validation_kl, validation_clean_wrong, train_kl_edges)
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
            base_validation = rows_by_arm["A0"]["validation"]
            _feature_consistency(base_validation, validation_ce, label=f"{seed} endpoint/validation CE20 features")
            heldout = {
                "count": len(validation_clean_wrong),
                "ids_sha256": _ids_sha256(validation_clean_wrong),
                "effects_vs_A0": {},
            }
            heldout_ce20_q = {
                "counts": {q: len(ids) for q, ids in validation_ce_q.items()},
                "effects_vs_A0": {arm: {} for arm in ARMS[1:]},
            }
            heldout_kl10_q = {
                "counts": {q: len(ids) for q, ids in validation_kl_q.items()},
                "effects_vs_A0": {arm: {} for arm in ARMS[1:]},
            }
            for arm in ARMS[1:]:
                treatment_validation = rows_by_arm[arm]["validation"]
                heldout["effects_vs_A0"][arm] = _effect(base_validation, treatment_validation, validation_clean_wrong)
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                    heldout_ce20_q["effects_vs_A0"][arm][q] = _effect(
                        base_validation, treatment_validation, validation_ce_q[q]
                    )
                    heldout_kl10_q["effects_vs_A0"][arm][q] = _effect(
                        base_validation, treatment_validation, validation_kl_q[q]
                    )
            horizon_result["heldout_clean_wrong"] = heldout
            horizon_result["heldout_clean_wrong_ce20_q"] = heldout_ce20_q
            horizon_result["heldout_clean_wrong_kl10_q"] = heldout_kl10_q
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
            "feature_lineage": feature_identity,
            "train_derived_quantile_edges": {"CE20": train_ce_edges, "KL10": train_kl_edges},
            "validation_clean_wrong": {
                "count": len(validation_clean_wrong),
                "ids_sha256": _ids_sha256(validation_clean_wrong),
                "CE20_q_counts": {q: len(ids) for q, ids in validation_ce_q.items()},
                "KL10_q_counts": {q: len(ids) for q, ids in validation_kl_q.items()},
            },
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
            "## Held-out Clean Wrong",
            "",
            "The held-out Clean-Wrong cohort is defined only from the epoch-79 "
            "pre-treatment validation Student clean correctness. Effects are paired "
            "against A0 at the same seed and horizon.",
            "",
            "| seed | epoch | arm | n | clean Δ (pp) | robust Δ (pp) | clean rescue | clean harm | "
            "robust rescue | robust harm |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in ("L2", "L4"):
        for epoch in HORIZONS:
            heldout = report["seeds"][seed]["horizons"][str(epoch)]["heldout_clean_wrong"]
            for arm in ARMS[1:]:
                item = heldout["effects_vs_A0"][arm]
                lines.append(
                    f"| {seed} | {epoch} | {arm} | {item['count']} | "
                    f"{_pp(item['clean_accuracy_delta'])} | {_pp(item['robust_accuracy_delta'])} | "
                    f"{_rate_pp(item['clean_rescue_rate'])} | {_rate_pp(item['clean_harm_rate'])} | "
                    f"{_rate_pp(item['robust_rescue_rate'])} | {_rate_pp(item['robust_harm_rate'])} |"
                )

    def append_quantile_tables(feature_name: str, title: str) -> None:
        lines.extend(
            [
                "",
                f"## Held-out Clean Wrong Q1–Q5 ({title})",
                "",
                "Q1–Q5 boundaries are derived from the epoch-79 train Clean-Wrong "
                "cohort only; validation outcomes never define the bins.",
                "",
                "### Robust accuracy effect (pp)",
                "",
                "| seed | epoch | arm | Q1 | Q2 | Q3 | Q4 | Q5 |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for seed in ("L2", "L4"):
            for epoch in HORIZONS:
                quantiles = report["seeds"][seed]["horizons"][str(epoch)][f"heldout_clean_wrong_{feature_name}_q"]
                for arm in ARMS[1:]:
                    values = quantiles["effects_vs_A0"][arm]
                    lines.append(
                        f"| {seed} | {epoch} | {arm} | "
                        + " | ".join(_pp(values[q]["robust_accuracy_delta"]) for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
                        + " |"
                    )
        lines.extend(
            [
                "",
                "### Clean accuracy effect (pp)",
                "",
                "| seed | epoch | arm | Q1 | Q2 | Q3 | Q4 | Q5 |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for seed in ("L2", "L4"):
            for epoch in HORIZONS:
                quantiles = report["seeds"][seed]["horizons"][str(epoch)][f"heldout_clean_wrong_{feature_name}_q"]
                for arm in ARMS[1:]:
                    values = quantiles["effects_vs_A0"][arm]
                    lines.append(
                        f"| {seed} | {epoch} | {arm} | "
                        + " | ".join(_pp(values[q]["clean_accuracy_delta"]) for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
                        + " |"
                    )

    append_quantile_tables("ce20", "CE-PGD20 Teacher margin")
    append_quantile_tables("kl10", "KL-PGD10 Teacher margin")

    lines.extend(
        [
            "",
            "## Held-out Clean-Wrong quantile counts",
            "",
            "Counts may be unequal because validation values are assigned using "
            "train-derived upper boundaries, including ties.",
            "",
            "| seed | train CW | held-out CW | CE20 Q1–Q5 | KL10 Q1–Q5 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for seed in ("L2", "L4"):
        seed_report = report["seeds"][seed]
        validation = seed_report["validation_clean_wrong"]
        ce_counts = ", ".join(str(validation["CE20_q_counts"][q]) for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
        kl_counts = ", ".join(str(validation["KL10_q_counts"][q]) for q in ("Q1", "Q2", "Q3", "Q4", "Q5"))
        lines.append(f"| {seed} | {seed_report['mask_count']} | {validation['count']} | {ce_counts} | {kl_counts} |")
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
