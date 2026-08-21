"""Aggregate the frozen A5--A8 no-update mechanism replay."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.analysis.ert_cw_a7_mechanism_replay import PARENTS
from ard.analysis.ert_cw_margin_generalization_report import MASK_SHA256
from ard.tracking.adapter import collect_git_state


class A7MechanismReportError(RuntimeError):
    """Raised when replay or endpoint lineage is incomplete."""


ARMS = ("A5", "A6", "A7", "A8")
EPOCHS = (79, 84, 89, 94)
ENDPOINT_EPOCHS = (84, 89, 94)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise A7MechanismReportError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result: dict[int, dict[str, Any]] = {}
    labels: dict[int, int] = {}
    for row in values:
        item = int(row["sample_id"])
        label = int(row["true_label"])
        if item in result or (item in labels and labels[item] != label):
            raise A7MechanismReportError(f"duplicate/inconsistent stable ID: {path}")
        result[item] = row
        labels[item] = label
    if not result:
        raise A7MechanismReportError(f"empty rows: {path}")
    return result


def _summary(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    if rows:
        for field in ("target_active", "hinge_active"):
            result[field + "_fraction"] = sum(bool(row[field]) for row in rows) / len(rows)
    return result


def _effect(base: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(base) != set(treatment):
        raise A7MechanismReportError("endpoint stable-ID universe mismatch")
    ids = sorted(base)
    clean_before = [bool(base[item]["clean_correct"]) for item in ids]
    clean_after = [bool(treatment[item]["clean_correct"]) for item in ids]
    robust_before = [bool(base[item]["robust_correct"]) for item in ids]
    robust_after = [bool(treatment[item]["robust_correct"]) for item in ids]

    def one(before: list[bool], after: list[bool], margin: str) -> dict[str, Any]:
        rescue = sum(not left and right for left, right in zip(before, after, strict=True))
        harm = sum(left and not right for left, right in zip(before, after, strict=True))
        n = len(before)
        deltas = [float(treatment[item][margin]) - float(base[item][margin]) for item in ids]
        return {
            "n": n,
            "rescue": rescue,
            "harm": harm,
            "rescue_rate": rescue / n if n else None,
            "harm_rate": harm / n if n else None,
            "net_rescue_rate": (rescue - harm) / n if n else None,
            "accuracy_delta": (rescue - harm) / n if n else None,
            "margin_delta": statistics.fmean(deltas) if deltas else None,
        }

    return {
        "clean": one(clean_before, clean_after, "clean_probability_margin"),
        "robust": one(robust_before, robust_after, "adversarial_probability_margin"),
    }


def _subset(rows: dict[int, dict[str, Any]], ids: set[int]) -> dict[int, dict[str, Any]]:
    if not ids.issubset(rows):
        raise A7MechanismReportError("endpoint subset IDs are not present")
    return {item: rows[item] for item in ids}


def _load_replay(root: Path, run: str, arm: str, epoch: int) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    directory = root / run / arm / f"epoch-{epoch}"
    meta_path = directory / "a7-mechanism-replay.json"
    rows_path = directory / "a7-mechanism-sample-stats.parquet"
    if not meta_path.is_file() or not rows_path.is_file():
        raise A7MechanismReportError(f"missing replay artifact: {directory}")
    meta = _json(meta_path)
    if meta.get("run") != run or meta.get("arm") != arm or meta.get("feature_epoch") != epoch:
        raise A7MechanismReportError(f"replay identity mismatch: {meta_path}")
    if meta.get("rows_sha256") != _sha256(rows_path) or meta.get("no_training") is not True:
        raise A7MechanismReportError(f"replay hash/no-update contract mismatch: {meta_path}")
    if meta.get("parent_checkpoint_sha256") != PARENTS[run]:
        raise A7MechanismReportError(f"replay lineage mismatch: {meta_path}")
    # The replay consumes the hash-bound state-overlay container, whose byte
    # hash is intentionally different from the compact registered CW-mask
    # manifest used by the endpoint/feature panels.  Validate that container
    # itself here; the exact selected-ID universe is checked below against the
    # CE20 feature panel before any effect is reported.
    mask_path = Path(str(meta.get("mask_path", "")))
    if not mask_path.is_file() or meta.get("mask_sha256") != _sha256(mask_path):
        raise A7MechanismReportError(f"replay mask identity mismatch: {meta_path}")
    return meta, _rows(rows_path)


def _load_endpoint(root: Path, run: str, arm: str, epoch: int) -> dict[int, dict[str, Any]]:
    path = root / run / arm / f"epoch-{epoch}" / "train" / "endpoint-sample-stats.parquet"
    meta = _json(path.with_name("endpoint.json"))
    if meta.get("checkpoint_epoch") != epoch:
        raise A7MechanismReportError(f"endpoint epoch mismatch: {path}")
    if meta.get("rows_sha256") != _sha256(path):
        raise A7MechanismReportError(f"endpoint row hash mismatch: {path}")
    if meta.get("attack_identity_sha256") != "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2":
        raise A7MechanismReportError(f"endpoint is not the frozen CE-PGD20 attack: {path}")
    return _rows(path)


def _load_gradient(root: Path, run: str, arm: str, epoch: int, checkpoint_sha256: str) -> dict[str, Any]:
    path = root / run / arm / f"epoch-{epoch}.json"
    if not path.is_file():
        raise A7MechanismReportError(f"missing gradient probe: {path}")
    payload = _json(path)
    if (
        payload.get("contract") != "ert_cw_a7_gradient_probe_v1"
        or payload.get("no_update") is not True
        or payload.get("run") != run
        or payload.get("arm") != arm
        or payload.get("epoch") != epoch
        or payload.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise A7MechanismReportError(f"gradient probe lineage mismatch: {path}")
    return payload


def _quantile_members(feature_root: Path, ids: set[int]) -> dict[str, set[int]]:
    meta = _json(feature_root / "clean-wrong-feature-replay.json")
    rows = _rows(Path(str(meta["rows_path"])))
    if set(rows) != ids:
        raise A7MechanismReportError("pre-treatment CE20 feature IDs differ from replay IDs")
    ordered = sorted(ids, key=lambda item: (float(rows[item]["teacher_adv_margin"]), item))
    return {
        f"Q{index + 1}": set(ordered[(len(ordered) * index) // 5 : (len(ordered) * (index + 1)) // 5])
        for index in range(5)
    }


def _regime_summaries(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    fields = ("target", "student_adv_margin", "raw_deficit", "positive_deficit", "margin_loss")
    groups = {"all": list(rows.values())}
    for regime in ("R0", "R1", "R2", "R3"):
        groups[regime] = [row for row in rows.values() if row["regime"] == regime]
    return {name: _summary(values, fields) for name, values in groups.items()}


def _transitions(replay_by_epoch: dict[int, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    ids = sorted(next(iter(replay_by_epoch.values())))
    sequences = {item: [replay_by_epoch[epoch][item]["regime"] for epoch in EPOCHS] for item in ids}
    transition_counts = Counter(
        (sequence[index], sequence[index + 1]) for sequence in sequences.values() for index in range(len(sequence) - 1)
    )
    switches = [
        sum(left != right for left, right in zip(sequence, sequence[1:]))
        for sequence in sequences.values()
    ]
    return {
        "n": len(ids),
        "switch_count": sum(switches),
        "switches_per_sample": statistics.fmean(switches) if switches else None,
        "never_switch_fraction": sum(value == 0 for value in switches) / len(switches) if switches else None,
        "transition_counts": {f"{left}->{right}": count for (left, right), count in sorted(transition_counts.items())},
        "floor_to_continuous_rate": sum(
            sequence[0] in {"R0", "R1"} and sequence[-1] in {"R2", "R3"} for sequence in sequences.values()
        )
        / len(sequences),
        "continuous_to_floor_rate": sum(
            sequence[0] in {"R2", "R3"} and sequence[-1] in {"R0", "R1"} for sequence in sequences.values()
        )
        / len(sequences),
        "dominant_regime": Counter(Counter(sequence).most_common(1)[0][0] for sequence in sequences.values()),
    }


def build_report(
    *,
    replay_root: Path,
    endpoint_root: Path,
    ce_feature_root: Path,
    gradient_root: Path | None = None,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False:
        raise A7MechanismReportError("report requires a clean source tree")
    machine: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_a7_mechanism_diagnostic_v1",
        "no_training": True,
        "source_git_sha": source["sha"],
        "parents": PARENTS,
        "mask_sha256": MASK_SHA256,
        "arms": ARMS,
        "epochs": EPOCHS,
        "gradient_probe_root": str(gradient_root) if gradient_root is not None else None,
        "runs": {},
    }
    markdown: list[str] = [
        "# ERT Clean-Wrong A7 mechanism diagnostic",
        "",
        "> Read-only, no-update checkpoint replay. These results are descriptive mechanism evidence, not causal proof.",
        "",
        "## Frozen contract",
        "",
        "A5 fixed, A6 zero/cap, A7 positive-floor, and A8 abstain were replayed "
        "with unchanged Teacher-clean KL-PGD10. R0--R3 are training-time "
        "regimes; CE20 Q1--Q5 are a separate pre-treatment axis.",
        "",
    ]
    for run in ("L2", "L4"):
        replay: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(dict)
        run_machine: dict[str, Any] = {
            "epochs": {},
            "transitions": {},
            "endpoint_effects": {},
            "endpoint_effects_direct": {},
            "endpoint_effects_spillover": {},
        }
        replay_source_shas: set[str] = set()
        replay_mask_shas: set[str] = set()
        for arm in ARMS:
            for epoch in EPOCHS:
                meta, rows = _load_replay(replay_root, run, arm, epoch)
                replay[arm][epoch] = rows
                replay_source_shas.add(str(meta["source_git_sha"]))
                replay_mask_shas.add(str(meta["mask_sha256"]))
                floor = float(meta["treatment"].get("margin_floor") or 0.03221710026264191)
                buffer_fractions = {
                    label: sum(
                        (
                            float(row["student_adv_margin"]) < 0
                            if label == "negative"
                            else 0 <= float(row["student_adv_margin"]) < floor
                            if label == "barely_positive"
                            else float(row["student_adv_margin"]) >= floor
                        )
                        for row in rows.values()
                    )
                    / len(rows)
                    for label in ("negative", "barely_positive", "buffered_positive")
                }
                run_machine["epochs"].setdefault(str(epoch), {})[arm] = {
                    "meta_sha256": _sha256(replay_root / run / arm / f"epoch-{epoch}" / "a7-mechanism-replay.json"),
                    "rows_sha256": meta["rows_sha256"],
                    "regimes": _regime_summaries(rows),
                    "target_attainment": {
                        regime: {
                            "fraction": sum(
                                float(row["student_adv_margin"]) >= float(row["target"])
                                for row in rows.values()
                                if row["regime"] == regime
                            )
                            / max(1, sum(row["regime"] == regime for row in rows.values()))
                        }
                        for regime in ("R0", "R1", "R2", "R3")
                    },
                    "positive_buffer": {"floor": floor, **buffer_fractions},
                }
        run_machine["replay_source_git_shas"] = sorted(replay_source_shas)
        run_machine["replay_mask_sha256"] = sorted(replay_mask_shas)
        for arm in ARMS:
            run_machine["transitions"][arm] = _transitions({epoch: replay[arm][epoch] for epoch in EPOCHS})
        ids = set(replay["A7"][79])
        quantiles = _quantile_members(ce_feature_root / run, ids)
        run_machine["pre_treatment_ce20_quantiles"] = {
            label: {
                "n": len(members),
                "regime_by_epoch": {
                    str(epoch): Counter(replay["A7"][epoch][item]["regime"] for item in members) for epoch in EPOCHS
                },
            }
            for label, members in quantiles.items()
        }
        for epoch in ENDPOINT_EPOCHS:
            base = _load_endpoint(endpoint_root, run, "A0", epoch)
            effects: dict[str, Any] = {}
            direct_effects: dict[str, Any] = {}
            spillover_effects: dict[str, Any] = {}
            spillover_ids = set(base) - ids
            for arm in ARMS:
                treatment = _load_endpoint(endpoint_root, run, arm, epoch)
                effects[arm] = _effect(base, treatment)
                direct_effects[arm] = _effect(_subset(base, ids), _subset(treatment, ids))
                spillover_effects[arm] = _effect(
                    _subset(base, spillover_ids), _subset(treatment, spillover_ids)
                )
            run_machine["endpoint_effects"][str(epoch)] = effects
            run_machine["endpoint_effects_direct"][str(epoch)] = direct_effects
            run_machine["endpoint_effects_spillover"][str(epoch)] = spillover_effects
        if gradient_root is not None:
            run_machine["gradient_probes"] = {}
            for arm in ARMS:
                run_machine["gradient_probes"][arm] = {}
                for epoch in (79, 94):
                    replay_meta = _json(
                        replay_root / run / arm / f"epoch-{epoch}" / "a7-mechanism-replay.json"
                    )
                    run_machine["gradient_probes"][arm][str(epoch)] = _load_gradient(
                        gradient_root,
                        run,
                        arm,
                        epoch,
                        str(replay_meta["checkpoint_sha256"]),
                    )
        a7_endpoint = _load_endpoint(endpoint_root, run, "A7", 94)
        base_endpoint = _load_endpoint(endpoint_root, run, "A0", 94)
        outcome_by_regime: dict[str, dict[str, int]] = {
            regime: {"rescue": 0, "harm": 0} for regime in ("R0", "R1", "R2", "R3")
        }
        sequences = {item: [replay["A7"][epoch][item]["regime"] for epoch in EPOCHS] for item in ids}
        for item, sequence in sequences.items():
            mode = Counter(sequence).most_common()
            regime = sorted(mode, key=lambda pair: (-pair[1], sequence.index(pair[0])))[0][0]
            if not bool(base_endpoint[item]["robust_correct"]) and bool(a7_endpoint[item]["robust_correct"]):
                outcome_by_regime[regime]["rescue"] += 1
            if bool(base_endpoint[item]["robust_correct"]) and not bool(a7_endpoint[item]["robust_correct"]):
                outcome_by_regime[regime]["harm"] += 1
        for regime, values in outcome_by_regime.items():
            values["net_rescue"] = values["rescue"] - values["harm"]
        run_machine["a7_robust_outcome_by_dominant_regime"] = outcome_by_regime
        machine["runs"][run] = run_machine
        markdown.extend(
            [
                f"## {run}",
                "",
                "### Regime and hinge summary",
                "",
                "Epoch-wise regime/target/deficit/hinge summaries are stored in the machine artifact.",
                f"Replay source SHAs observed: {', '.join(run_machine['replay_source_git_shas'])}.",
                f"Replay mask-container SHA256: {', '.join(run_machine['replay_mask_sha256'])}.",
                "",
            ]
        )
        if gradient_root is not None:
            markdown.extend(
                [
                    "### No-update gradient probe",
                    "",
                    "The fixed 128-ID probes are diagnostics only; they do not tune coefficients "
                    "or use endpoint outcomes.",
                    "",
                    "| arm | epoch | base norm | margin/base ratio | margin/base cosine |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for arm in ARMS:
                for epoch in (79, 94):
                    summary = run_machine["gradient_probes"][arm][str(epoch)]["summary"]
                    markdown.append(
                        f"| {arm} | {epoch} | {summary['base_norm']['mean']:.6g} | "
                        f"{summary['weighted_margin_base_ratio']['mean']:.6g} | "
                        f"{summary['cosine_margin_base']['mean']:.6g} |"
                    )
            markdown.append("")
        markdown.extend(
            [
                "### CE20 pre-treatment Q1--Q5 mapping (epoch 79)",
                "",
                "Quantile boundaries are pre-treatment Teacher CE-PGD20 margins; "
                "they are not fitted to endpoint outcomes.",
                "",
                "| quantile | n | R0 | R1 | R2 | R3 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            q = run_machine["pre_treatment_ce20_quantiles"][label]
            counts = q["regime_by_epoch"]["79"]
            markdown.append(
                f"| {label} | {q['n']} | {counts.get('R0', 0)} | {counts.get('R1', 0)} | "
                f"{counts.get('R2', 0)} | {counts.get('R3', 0)} |"
            )
        markdown.append("")
        for epoch in EPOCHS:
            a7 = run_machine["epochs"][str(epoch)]["A7"]["regimes"]
            markdown.append(
                f"- epoch {epoch}: A7 R0={a7['R0']['n']}, R1={a7['R1']['n']}, "
                f"R2={a7['R2']['n']}, R3={a7['R3']['n']}; "
                f"positive-deficit mean={a7['all']['positive_deficit']['mean']:.6f}"
            )
        markdown.extend(
            [
                "",
                "### Endpoint effects vs BASE (epoch 94)",
                "",
                "| arm | clean Δ | robust Δ | clean rescue | clean harm | robust rescue | robust harm |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            effect = run_machine["endpoint_effects"]["94"][arm]
            markdown.append(
                f"| {arm} | {effect['clean']['accuracy_delta']:+.4f} | "
                f"{effect['robust']['accuracy_delta']:+.4f} | {effect['clean']['rescue']} | "
                f"{effect['clean']['harm']} | {effect['robust']['rescue']} | "
                f"{effect['robust']['harm']} |"
            )
        markdown.extend(
            [
                "",
                "### Endpoint effect partition (epoch 94)",
                "",
                "Direct is the fixed Clean-Wrong cohort; spillover is its complement in the train endpoint.",
                "",
                "| arm | direct clean Δ | direct robust Δ | spillover clean Δ | spillover robust Δ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            direct = run_machine["endpoint_effects_direct"]["94"][arm]
            spillover = run_machine["endpoint_effects_spillover"]["94"][arm]
            markdown.append(
                f"| {arm} | {direct['clean']['accuracy_delta']:+.4f} | "
                f"{direct['robust']['accuracy_delta']:+.4f} | "
                f"{spillover['clean']['accuracy_delta']:+.4f} | "
                f"{spillover['robust']['accuracy_delta']:+.4f} |"
            )
        markdown.extend(
            [
                "",
                "### A7 dominant-regime rescue/harm",
                "",
                "| dominant regime | rescue | harm | net |",
                "|---|---:|---:|---:|",
            ]
        )
        for regime, values in outcome_by_regime.items():
            markdown.append(f"| {regime} | {values['rescue']} | {values['harm']} | {values['net_rescue']} |")
        markdown.append("")
    machine["interpretation_boundary"] = (
        "No causal claim; endpoint outcomes are joined descriptively to no-update replay regimes."
    )
    markdown.extend(
        [
            "## Interpretation boundary and next priorities",
            "",
            "The endpoint tables are descriptive joins to replayed training-time regimes; "
            "they do not identify a causal effect of any target rule.",
            "The fixed 128-ID gradient probes are no-update diagnostics and were not used "
            "to select a coefficient or treatment.",
            "A7 follow-up priorities remain: (1) isolate A7 from extra CleanCE, "
            "(2) preregister a small lambda sensitivity check, and "
            "(3) separately test floor/cap sensitivity.",
            "No follow-up training is started by this report.",
            "",
        ]
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return machine
