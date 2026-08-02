"""Read-only H5-Early common-PGD panel analysis."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.history_cohort import HistoryCohortError, bind_reports_to_cohort, load_cohort_inventory
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS, PANEL_EMA_BETA, repository_root_from_source
from ard.analysis.signal_audit import SignalAuditError, binary_metrics, sha256_file


class HistoryEarlyError(ValueError):
    pass


ANCHORS = (39, 59, 79, 99)
PEAK_WINDOW = (99, 104, 109)
OBS_COLS = (
    "namespace",
    "sample_id",
    "class_id",
    "epoch",
    "teacher_entropy_normalized",
    "student_probability_margin",
    "student_margin_risk",
    "robust_correct",
)


def _tracked_clean_provenance() -> dict[str, Any]:
    """Bind a canonical H5-Early report to its exact clean analysis sources."""
    root = repository_root_from_source()
    paths = {
        "history_early": Path(__file__).resolve(),
        "history_early_cli": root / "src/ard/cli/history_early.py",
        "rslad_signal_replay": root / "src/ard/analysis/rslad_signal_replay.py",
    }
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
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
        raise HistoryEarlyError("H5-Early requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise HistoryEarlyError("H5-Early requires a tracked-clean revision")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": hashes}


def _json(path: Path) -> dict[str, Any]:
    try:
        v = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise HistoryEarlyError("unreadable lineage") from e
    if not isinstance(v, dict):
        raise HistoryEarlyError("lineage must be mapping")
    return v


def _parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        t = pq.read_table(path)
    except Exception as e:
        raise HistoryEarlyError("unreadable observation panel") from e
    if not set(OBS_COLS).issubset(t.column_names):
        raise HistoryEarlyError("observation column contract drifted")
    return [dict(x) for x in t.to_pylist()]


def _i(x: object, n: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise HistoryEarlyError(f"{n} must be nonnegative integer")
    return x


def _f(x: object, n: str, lo: float, hi: float) -> float:
    if (
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(float(x))
        or not lo <= float(x) <= hi
    ):
        raise HistoryEarlyError(f"{n} outside contract")
    return float(x)


def _panel(
    rows: Sequence[Mapping[str, Any]], epochs: Sequence[int], count: int, name: str
) -> dict[int, dict[int, dict[str, Any]]]:
    out = {e: {} for e in epochs}
    for r in rows:
        e = _i(r.get("epoch"), name + " epoch")
        sid = _i(r.get("sample_id"), name + " sample_id")
        cls = _i(r.get("class_id"), name + " class_id")
        if e not in out or r.get("namespace") != "train" or sid >= 50000 or cls >= 10 or sid in out[e]:
            raise HistoryEarlyError(name + " epoch/ID contract drifted")
        m = _f(r.get("student_probability_margin"), name + " margin", -1, 1)
        risk = _f(r.get("student_margin_risk"), name + " risk", 0, 1)
        _f(r.get("teacher_entropy_normalized"), name + " entropy", 0, 1)
        if not isinstance(r.get("robust_correct"), bool) or not math.isclose(risk, (1 - m) / 2, abs_tol=1e-7):
            raise HistoryEarlyError(name + " signal contract drifted")
        out[e][sid] = {"c": cls, "m": m, "r": risk, "ok": r["robust_correct"]}
    if any(len(v) != count for v in out.values()):
        raise HistoryEarlyError(name + " lacks exact stable ID coverage")
    ref = next(iter(out.values()))
    if any(set(v) != set(ref) or any(v[k]["c"] != ref[k]["c"] for k in ref) for v in out.values()):
        raise HistoryEarlyError(name + " class identity drifted")
    return out


def _lineage(
    path: Path,
    obs: Path,
    key: str,
    count: int,
    *,
    allow_historical_outcome_seed_omission: bool = False,
) -> dict[str, Any]:
    x = _json(path)
    if (
        x.get("schema_version") != 1
        or x.get("observation_schema_version") != 2
        or x.get("train_expected_count") != count
        or x.get(key) != sha256_file(obs)
    ):
        raise HistoryEarlyError("lineage byte/count binding drifted")
    for k in (
        "run_id",
        "config_hash",
        "scientific_git_sha",
        "attack_identity",
        "dataset_identity",
        "teacher",
        "feature_protocol" if key.startswith("feature") else "outcome_protocol",
    ):
        if k not in x:
            raise HistoryEarlyError("lineage identity incomplete")
    if "seed" in x:
        return {**x, "_outcome_seed_compatibility": "declared"}
    if allow_historical_outcome_seed_omission and key == "outcome_observations_sha256":
        return {**x, "_outcome_seed_compatibility": "historical_missing_seed"}
    raise HistoryEarlyError("lineage identity incomplete")


def _mid(v: Mapping[int, float]) -> dict[int, float]:
    order = sorted((x, k) for k, x in v.items())
    out = {}
    s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and order[e][0] == order[s][0]:
            e += 1
        for _, k in order[s:e]:
            out[k] = (s + (e - s) / 2) / len(order)
        s = e
    return out


def _score(panel: Mapping[int, Mapping[int, Mapping[str, Any]]], a: int) -> dict[str, dict[int, float]]:
    es = tuple(e for e in FEATURE_EPOCHS if e <= a)
    hist = es[:-1]
    if not hist or es[-1] != a:
        raise HistoryEarlyError("anchor schedule drifted")
    freq = {}
    margin = {}
    instant = {}
    curr = {}
    for sid in panel[a]:
        ema = None
        hits = 0
        for e in hist:
            ema = (
                panel[e][sid]["m"] if ema is None else PANEL_EMA_BETA * ema + (1 - PANEL_EMA_BETA) * panel[e][sid]["m"]
            )
            hits += int(panel[e][sid]["ok"])
        freq[sid] = 1 - hits / len(hist)
        margin[sid] = (1 - ema) / 2
        instant[sid] = panel[a][sid]["r"]
        curr[sid] = float(not panel[a][sid]["ok"])
    fr, mr, ir, cr = _mid(freq), _mid(margin), _mid(instant), _mid(curr)
    return {
        "adaptive_history": {k: (fr[k] + mr[k]) / 2 for k in freq},
        "frequency_only": freq,
        "margin_only": margin,
        "instantaneous_margin": instant,
        "current_correctness": curr,
        "outcome_free_current_rank": {k: (ir[k] + cr[k]) / 2 for k in freq},
    }


def _metric(scores: Mapping[int, float], y: Mapping[int, int]) -> dict[str, float | None]:
    if not y:
        return {"auroc": None, "auprc": None, "prevalence": None, "count": 0}
    try:
        r = binary_metrics([y[k] for k in sorted(y)], [scores[k] for k in sorted(y)])
    except SignalAuditError:
        return {"auroc": None, "auprc": None, "prevalence": sum(y.values()) / len(y), "count": len(y)}
    return {"auroc": r["auroc"], "auprc": r["auprc"], "prevalence": r["prevalence"], "count": len(y)}


def analyze_history_early(
    *,
    feature_observations: Path,
    outcome_observations: Path,
    feature_lineage: Path,
    outcome_lineage: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fmeta = _lineage(feature_lineage, feature_observations, "feature_observations_sha256", expected_count)
    ometa = _lineage(outcome_lineage, outcome_observations, "outcome_observations_sha256", expected_count)
    if any(
        fmeta[k] != ometa[k]
        for k in ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher")
    ):
        raise HistoryEarlyError("feature/outcome lineage identity drifted")
    f = _panel(_parquet(feature_observations), FEATURE_EPOCHS, expected_count, "feature")
    o = _panel(_parquet(outcome_observations), OUTCOME_EPOCHS, expected_count, "outcome")
    if set(f[99]) != set(o[99]) or any(f[99][k]["c"] != o[99][k]["c"] for k in f[99]):
        raise HistoryEarlyError("feature/outcome epoch99 join drifted")
    peak = {k: int(sum(not o[e][k]["ok"] for e in PEAK_WINDOW) >= 2) for k in o[99]}
    post = {}
    for k in o[109]:
        if o[109][k]["ok"]:
            prev = True
            forgot = False
            for e in (x for x in OUTCOME_EPOCHS if x > 109):
                now = o[e][k]["ok"]
                forgot |= prev and not now
                prev = now
            post[k] = int(forgot)
    tables = {}
    gate_rows: dict[str, list[dict[str, Any]]] = {}
    for a in ANCHORS:
        s = _score(f, a)
        post_scores = {n: {k: v for k, v in values.items() if k in post} for n, values in s.items()}
        tables[str(a)] = {
            "peak_window_error": {n: _metric(v, peak) for n, v in s.items()},
            "post_peak_forgetting": {n: _metric(values, post) for n, values in post_scores.items()},
        }
        gate_rows[str(a)] = [
            {
                "sample_id": sample_id,
                "class_id": o[109][sample_id]["c"],
                "outcome": post[sample_id],
                "baseline": post_scores["outcome_free_current_rank"][sample_id],
                "candidate": post_scores["adaptive_history"][sample_id],
            }
            for sample_id in sorted(post)
        ]
    return {
        "schema_version": 1,
        "contract": "h5_early_common_pgd_v1",
        "input_identity": {
            "run_id": fmeta["run_id"],
            "config_hash": fmeta["config_hash"],
            "scientific_git_sha": fmeta["scientific_git_sha"],
            "seed": fmeta["seed"],
            "teacher_registry_id": fmeta["teacher"].get("registry_id"),
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "feature_attack_domain": fmeta["feature_protocol"],
            "outcome_attack_domain": ometa["outcome_protocol"],
        },
        "anchors": list(ANCHORS),
        "outcomes": {
            "peak_window_error": "wrong_at_least_two_of_99_104_109",
            "post_peak_forgetting": "epoch109_correct_then_later_correct_to_wrong",
        },
        "tables": tables,
        "_post_peak_gate_rows": gate_rows,
        "analysis_provenance": dict(
            _tracked_clean_provenance() if analysis_provenance is None else analysis_provenance
        ),
    }


def collection_gate(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for a in ANCHORS:
        d = {}
        for label in ("L1", "L3"):
            t = reports.get(label, {}).get("tables", {}).get(str(a), {}).get("post_peak_forgetting", {})
            baseline = t.get("outcome_free_current_rank", {}).get("auroc")
            ad = t.get("adaptive_history", {}).get("auroc")
            d[label] = ad - baseline if isinstance(ad, float) and isinstance(baseline, float) else None
        point_pass = all(delta is not None and delta >= 0.02 for delta in d.values())
        result: dict[str, Any] = {
            "adaptive_minus_predeclared_current_rank_auroc": d,
            "predeclared_current_rank_diagnostic": "outcome_free_current_rank",
            "point_threshold_met": point_pass,
            "paired_lower_bound": None,
        }
        result["status"] = "point_gate_pass_bootstrap_task_required" if point_pass else "no_go_point_gate"
        rows[str(a)] = result
    return {
        "outcome": "post_peak_forgetting",
        "criterion": (
            "point delta>=0.02 on L1/L3, then frozen class-stratified paired bootstrap "
            "(seed=2026073102, n=2000) against outcome_free_current_rank; "
            "no automatic Go before admissibility review"
        ),
        "status": "sequential_no_automatic_go",
        "anchors": rows,
    }


def _online_panel(path: Path, lineage_path: Path, expected_count: int) -> dict[int, dict[int, dict[str, Any]]]:
    lineage = _json(lineage_path)
    if (
        lineage.get("contract") != "h5_online_state_anchor_v1"
        or lineage.get("expected_count") != expected_count
        or lineage.get("observations_sha256") != sha256_file(path)
    ):
        raise HistoryEarlyError("online state lineage is not the exact hash-bound anchor panel")
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise HistoryEarlyError("online state observations are unreadable") from exc
    # The revised primary route deliberately does not admit epoch 99 as an
    # online feature anchor: it is too late to affect the scheduled peak.
    expected_epochs = {39, 59, 79}
    result = {epoch: {} for epoch in expected_epochs}
    for row in rows:
        epoch = _i(row.get("anchor_epoch"), "online epoch")
        sample_id = _i(row.get("sample_id"), "online sample ID")
        label = _i(row.get("true_label"), "online label")
        if epoch not in result or label >= 10 or sample_id in result[epoch] or row.get("namespace") != "train":
            raise HistoryEarlyError("online state row ID/epoch contract drifted")
        hits = _i(row.get("robust_correct_count"), "online robust correct count")
        frequency = _f(row.get("robust_correct_frequency_inclusive"), "online inclusive frequency", 0, 1)
        if (
            not isinstance(row.get("previous_robust_correct"), bool)
            or hits > epoch + 1
            or not math.isclose(frequency, hits / (epoch + 1), abs_tol=1e-7)
        ):
            raise HistoryEarlyError("online state inclusive correctness contract drifted")
        margin_ema = _f(row.get("margin_ema"), "online margin EMA", -1, 1)
        last_margin = _f(row.get("last_margin"), "online last margin", -1, 1)
        result[epoch][sample_id] = {
            "label": label,
            "ok": row["previous_robust_correct"],
            "frequency": frequency,
            "ema": margin_ema,
            "last": last_margin,
        }
    if any(len(rows_by_id) != expected_count for rows_by_id in result.values()):
        raise HistoryEarlyError("online state lacks exact stable-ID coverage")
    reference = next(iter(result.values()))
    if any(
        set(panel) != set(reference) or any(panel[sid]["label"] != reference[sid]["label"] for sid in reference)
        for panel in result.values()
    ):
        raise HistoryEarlyError("online state stable ID/class join drifted")
    return result


def _precision_at_q(scores: Mapping[int, float], outcomes: Mapping[int, int]) -> float:
    k = max(1, math.floor(0.1 * len(scores)))
    chosen = sorted(scores, key=lambda sample_id: (-scores[sample_id], sample_id))[:k]
    return sum(outcomes[sample_id] for sample_id in chosen) / k


def _top_q(scores: Mapping[int, float]) -> set[int]:
    k = max(1, math.floor(0.1 * len(scores)))
    return set(sorted(scores, key=lambda sample_id: (-scores[sample_id], sample_id))[:k])


def _selector_overlap(
    candidate: Mapping[int, float], baseline: Mapping[int, float], outcomes: Mapping[int, int]
) -> dict[str, Any]:
    """Describe the actual 10% intervention masks, not only rank correlation."""
    selected_candidate = _top_q(candidate)
    selected_baseline = _top_q(baseline)
    groups = {
        "history_only": selected_candidate - selected_baseline,
        "instantaneous_margin_only": selected_baseline - selected_candidate,
        "shared": selected_candidate & selected_baseline,
    }

    def rate(ids: set[int]) -> dict[str, float | int | None]:
        return {
            "count": len(ids),
            "outcome_prevalence": sum(outcomes[sample_id] for sample_id in ids) / len(ids) if ids else None,
        }

    union = selected_candidate | selected_baseline
    return {
        "q": 0.1,
        "candidate_count": len(selected_candidate),
        "baseline_count": len(selected_baseline),
        "jaccard": len(groups["shared"]) / len(union) if union else None,
        **{name: rate(ids) for name, ids in groups.items()},
    }


def analyze_history_early_online(
    *,
    online_states: Path,
    online_lineage: Path,
    feature_observations: Path,
    feature_lineage: Path,
    outcome_observations: Path,
    outcome_lineage: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Primary revised-H5: online scores/risk sets with replay-only outcomes."""
    panel = _online_panel(online_states, online_lineage, expected_count)
    feature_meta = _lineage(feature_lineage, feature_observations, "feature_observations_sha256", expected_count)
    outcome_meta = _lineage(
        outcome_lineage,
        outcome_observations,
        "outcome_observations_sha256",
        expected_count,
        allow_historical_outcome_seed_omission=True,
    )
    online_meta = _json(online_lineage)
    identity_keys = (
        "run_id",
        "config_hash",
        "scientific_git_sha",
        "attack_identity",
        "dataset_identity",
        "teacher",
    )
    if any(
        online_meta.get(key) != feature_meta[key] or feature_meta[key] != outcome_meta[key] for key in identity_keys
    ):
        raise HistoryEarlyError("online/replay lineage identity drifted")
    if online_meta.get("seed") != feature_meta["seed"]:
        raise HistoryEarlyError("online/replay lineage identity drifted")
    outcome_seed_status = outcome_meta["_outcome_seed_compatibility"]
    if outcome_seed_status == "declared" and outcome_meta["seed"] != feature_meta["seed"]:
        raise HistoryEarlyError("online/replay lineage identity drifted")
    feature = _panel(_parquet(feature_observations), FEATURE_EPOCHS, expected_count, "feature replay")
    replay = _panel(_parquet(outcome_observations), OUTCOME_EPOCHS, expected_count, "outcome replay")
    if (
        set(panel[39]) != set(feature[39])
        or set(panel[39]) != set(replay[99])
        or any(
            panel[39][sid]["label"] != feature[39][sid]["c"] or panel[39][sid]["label"] != replay[99][sid]["c"]
            for sid in panel[39]
        )
    ):
        raise HistoryEarlyError("online/replay stable ID/class join drifted")
    reports: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    for anchor in (39, 59, 79):
        anchor_correct = {sid for sid, row in panel[anchor].items() if row["ok"]}
        anchor_wrong = set(panel[anchor]) - anchor_correct
        rank = {
            sid: value
            for sid, value in _mid({sid: 1 - panel[anchor][sid]["frequency"] for sid in panel[anchor]}).items()
        }
        ema_rank = _mid({sid: (1 - panel[anchor][sid]["ema"]) / 2 for sid in panel[anchor]})
        online_rank = {sid: (rank[sid] + ema_rank[sid]) / 2 for sid in panel[anchor]}
        baseline = {sid: (1 - panel[anchor][sid]["last"]) / 2 for sid in panel[anchor]}
        replay_rank = _score(feature, anchor)["adaptive_history"]
        peak = {sid: int(sum(not replay[epoch][sid]["ok"] for epoch in (99, 104, 109)) >= 2) for sid in anchor_correct}
        non_recovery = {sid: int(all(not replay[epoch][sid]["ok"] for epoch in (99, 104, 109))) for sid in anchor_wrong}
        ro = {}
        for sid in anchor_correct:
            values = [replay[epoch][sid]["ok"] for epoch in range(109, 200, 5)]
            ro[sid] = int(any(before and not after for before, after in zip(values, values[1:])))

        def summary(outcomes: Mapping[int, int]) -> dict[str, Any]:
            candidate = {sid: online_rank[sid] for sid in outcomes}
            current = {sid: baseline[sid] for sid in outcomes}
            replay_candidate = {sid: replay_rank[sid] for sid in outcomes}
            return {
                "online_rank": {
                    **_metric(candidate, outcomes),
                    "precision_at_10pct": _precision_at_q(candidate, outcomes) if outcomes else None,
                },
                "online_instantaneous_margin": {
                    **_metric(current, outcomes),
                    "precision_at_10pct": _precision_at_q(current, outcomes) if outcomes else None,
                },
                "top_10pct_mask_overlap": _selector_overlap(candidate, current, outcomes),
                "replay_pre_anchor_rank_diagnostic": {
                    "max_feature_epoch": anchor - 5,
                    "metrics": {
                        **_metric(replay_candidate, outcomes),
                        "precision_at_10pct": _precision_at_q(replay_candidate, outcomes) if outcomes else None,
                    },
                    "top_10pct_mask_overlap_vs_instantaneous": _selector_overlap(replay_candidate, current, outcomes),
                },
            }

        reports[str(anchor)] = {
            "anchor_inclusive": True,
            "anchor_correct_peak_failure": summary(peak),
            "anchor_wrong_non_recovery": summary(non_recovery),
            "robust_overfitting_secondary": {"gate_eligible": False, "metrics": summary(ro)},
        }
        raw[str(anchor)] = {
            "peak": peak,
            "non_recovery": non_recovery,
            "online_rank": online_rank,
            "baseline": baseline,
            "class_id": {sid: panel[anchor][sid]["label"] for sid in panel[anchor]},
        }
    return {
        "schema_version": 1,
        "contract": "h5_early_online_primary_v1",
        "status": "point_only",
        "input_identity": {
            **{key: feature_meta[key] for key in (*identity_keys, "seed")},
            "teacher_registry_id": outcome_meta["teacher"].get("registry_id"),
            "online_states_sha256": sha256_file(online_states),
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "outcome_seed_compatibility": {
                "status": outcome_seed_status,
                "effective_seed": feature_meta["seed"],
            },
        },
        "anchors": [39, 59, 79],
        "reports": reports,
        "analysis_provenance": dict(
            _tracked_clean_provenance() if analysis_provenance is None else analysis_provenance
        ),
        "_bootstrap_inputs": raw,
    }


def build_online_bootstrap_tasks(
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create only jointly admissible Bartoldson confirmation bootstrap tasks.

    L1/L3 are the pre-registered Bartoldson confirmation trajectories.  Other
    labels remain visible in the point report but cannot create a bootstrap
    task, so diagnostic runs never become accidental confirmation evidence.
    """
    primary_labels = ("L1", "L3")
    tasks: list[dict[str, Any]] = []
    gate: dict[str, Any] = {}
    for anchor in (39, 59, 79):
        for outcome_name, raw_key in (("peak_failure", "peak"), ("non_recovery", "non_recovery")):
            per_run: dict[str, Any] = {}
            ready = True
            for label in primary_labels:
                report = reports.get(label)
                raw = report.get("_bootstrap_inputs", {}).get(str(anchor)) if isinstance(report, Mapping) else None
                if not isinstance(raw, Mapping):
                    per_run[label] = {"pass": False, "reason": "missing_primary_run"}
                    ready = False
                    continue
                outcomes = raw.get(raw_key)
                online_rank = raw.get("online_rank")
                baseline = raw.get("baseline")
                class_id = raw.get("class_id")
                if not all(isinstance(value, Mapping) for value in (outcomes, online_rank, baseline, class_id)):
                    per_run[label] = {"pass": False, "reason": "invalid_raw_contract"}
                    ready = False
                    continue
                sample_ids = sorted(outcomes)
                labels = [outcomes[sample_id] for sample_id in sample_ids]
                if not sample_ids or not any(labels) or all(labels):
                    per_run[label] = {"pass": False, "reason": "degenerate_outcome"}
                    ready = False
                    continue
                candidate = [online_rank[sample_id] for sample_id in sample_ids]
                current = [baseline[sample_id] for sample_id in sample_ids]
                try:
                    delta = binary_metrics(labels, candidate)["auroc"] - binary_metrics(labels, current)["auroc"]
                    candidate_precision = _precision_at_q(
                        {sample_id: online_rank[sample_id] for sample_id in sample_ids}, outcomes
                    )
                    baseline_precision = _precision_at_q(
                        {sample_id: baseline[sample_id] for sample_id in sample_ids}, outcomes
                    )
                except (KeyError, TypeError, SignalAuditError):
                    per_run[label] = {"pass": False, "reason": "metric_contract_failure"}
                    ready = False
                    continue
                passed = delta >= 0.02 and candidate_precision >= baseline_precision - 0.01
                per_run[label] = {
                    "pass": passed,
                    "delta_auroc": delta,
                    "candidate_precision_at_10pct": candidate_precision,
                    "instantaneous_precision_at_10pct": baseline_precision,
                    "reason": "pass" if passed else "point_criterion_not_met",
                }
                ready &= passed
            gate_key = f"epoch{anchor}-{outcome_name}"
            gate[gate_key] = {
                "primary_runs": per_run,
                "pass": ready,
                "criterion": "both L1/L3: delta AUROC >= 0.02 and candidate precision@10% >= instantaneous - 0.01",
                "diagnostic_only_runs": sorted(set(reports) - set(primary_labels)),
            }
            if not ready:
                continue
            for label in primary_labels:
                raw = reports[label]["_bootstrap_inputs"][str(anchor)]
                outcomes = raw[raw_key]
                sample_ids = sorted(outcomes)
                tasks.append(
                    {
                        "task_id": f"{label}-epoch{anchor}-{outcome_name}",
                        "run": label,
                        "anchor": anchor,
                        "outcome": outcome_name,
                        "stratum": "online_anchor_correct" if outcome_name == "peak_failure" else "online_anchor_wrong",
                        "point_gate_pass": True,
                        "joint_primary_gate": gate_key,
                        "rows": [
                            {
                                "sample_id": sample_id,
                                "class_id": raw["class_id"][sample_id],
                                "outcome": outcomes[sample_id],
                                "baseline": raw["baseline"][sample_id],
                                "candidate": raw["online_rank"][sample_id],
                            }
                            for sample_id in sample_ids
                        ],
                    }
                )
    return tasks, gate


def bind_early_collection_to_cohort(*, cohort_inventory: Path, reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Bind a point-only H5-Early collection to the immutable L1--L4 inventory."""
    try:
        inventory, digest = load_cohort_inventory(cohort_inventory)
        bind_reports_to_cohort(inventory=inventory, reports=reports)
    except HistoryCohortError as exc:
        raise HistoryEarlyError(str(exc)) from exc
    return digest
