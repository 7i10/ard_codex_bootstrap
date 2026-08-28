"""Aggregate the preregistered ERT/RSLAD stage-wise augmentation screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1")
CONTROL_ROOT = Path(
    "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-static-trajstab-v1"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "bb68afc0ff505248f84c0263179ec24f0b346bcd"
CONTROL = {1: CONTROL_ROOT / "cropshift-s1-r2", 2: CONTROL_ROOT / "cropshift-s2-r1"}
CHILD = {
    "R100": {1: "re-s100-s1", 2: "re-s100-s2"},
    "I100": {1: "idbh-s100-s1", 2: "idbh-s100-s2"},
    "R150": {1: "re-s150-s1", 2: "re-s150-s2"},
    "I150": {1: "idbh-s150-s1", 2: "idbh-s150-s2"},
}
SWITCH = {"R100": 100, "I100": 100, "R150": 150, "I150": 150}
POLICY = {"R100": "CROP_RE", "I100": "IDBH_WEAK", "R150": "CROP_RE", "I150": "IDBH_WEAK"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["epoch"]))
    if [int(row["epoch"]) for row in rows] != list(range(rows[0]["epoch"], rows[-1]["epoch"] + 1)):
        raise ValueError(f"non-contiguous metrics: {path}")
    return rows


def auc(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("AUC needs at least two values")
    return (values[0] / 2 + sum(values[1:-1]) + values[-1] / 2) / (len(values) - 1)


def metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def control_endpoint(seed: int, display_epoch: int) -> dict[str, Any]:
    # Historical CROPSHIFT endpoint artifacts are stored beside the run
    # directories under the trajectory-stabilization root, not inside each
    # run directory.
    path = (
        CONTROL_ROOT
        / "endpoints"
        / ("cropshift-s1-r2" if seed == 1 else "cropshift-s2-r1")
        / f"epoch-{display_epoch:03d}"
        / "validation"
        / "endpoint.json"
    )
    data = read_json(path)
    if data["attack_identity_sha256"] != "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2":
        raise ValueError(f"control endpoint attack drift: {path}")
    if data["row_count"] != 5000:
        raise ValueError(f"control endpoint row count drift: {path}")
    return {
        "path": str(path.resolve()),
        "endpoint_json_sha256": sha256(path),
        "rows_sha256": data["rows_sha256"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "clean": float(data["clean_accuracy"]),
        "robust": float(data["robust_accuracy"]),
    }


def endpoint_record(seed: int, arm: str, display_epoch: int) -> dict[str, Any]:
    name = CHILD[arm][seed]
    path = ROOT / name / "stagewise-endpoints" / f"epoch-{display_epoch:03d}" / "validation" / "endpoint.json"
    if path.exists():
        data = read_json(path)
        if data["attack_identity_sha256"] != "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2":
            raise ValueError(f"stagewise endpoint attack drift: {path}")
        if data["row_count"] != 5000:
            raise ValueError(f"stagewise endpoint row count drift: {path}")
        return {
            "source": "stagewise",
            "path": str(path.resolve()),
            "endpoint_json_sha256": sha256(path),
            "rows_sha256": data["rows_sha256"],
            "checkpoint_sha256": data["checkpoint_sha256"],
            "clean": float(data["clean_accuracy"]),
            "robust": float(data["robust_accuracy"]),
        }
    if display_epoch == 149 and SWITCH[arm] == 150:
        return {"source": "historical_cropshift", **control_endpoint(seed, 149)}
    raise FileNotFoundError(path)


def build_arm(arm: str, seed: int) -> dict[str, Any]:
    switch = SWITCH[arm]
    control_rows = read_rows(CONTROL[seed] / "epoch-metrics.jsonl")
    child_dir = ROOT / CHILD[arm][seed]
    child_rows = read_rows(child_dir / "epoch-metrics.jsonl")
    if [int(row["epoch"]) for row in child_rows] != list(range(switch, 200)):
        raise ValueError(f"child coverage drift: {arm} seed{seed}")
    hybrid = control_rows[:switch] + child_rows
    if [int(row["epoch"]) for row in hybrid] != list(range(200)):
        raise ValueError(f"hybrid coverage drift: {arm} seed{seed}")
    control_pgd = metric(control_rows, "val_pgd_accuracy")
    hybrid_pgd = metric(hybrid, "val_pgd_accuracy")
    control_clean = metric(control_rows, "val_clean_accuracy")
    hybrid_clean = metric(hybrid, "val_clean_accuracy")
    child_throughput = sorted(float(row["train_images_per_second"]) for row in child_rows)[len(child_rows) // 2]
    control_throughput = sorted(float(row["train_images_per_second"]) for row in control_rows[switch:])[
        len(control_rows[switch:]) // 2
    ]
    delta = [hybrid_pgd[i] - control_pgd[i] for i in range(switch, 200)]
    negative = [i for i, value in enumerate(delta, start=switch) if value < 0]
    recovery = None
    if negative:
        for epoch, value in zip(range(switch, 200), delta):
            if epoch > min(negative) and value >= 0:
                recovery = epoch
                break
    endpoints = {}
    for display_epoch in (149, 199):
        endpoints[str(display_epoch)] = endpoint_record(seed, arm, display_epoch)
    control_endpoints = {str(display_epoch): control_endpoint(seed, display_epoch) for display_epoch in (149, 199)}
    endpoint_deltas = {
        key: {
            "clean": endpoints[key]["clean"] - control_endpoints[key]["clean"],
            "robust": endpoints[key]["robust"] - control_endpoints[key]["robust"],
        }
        for key in ("149", "199")
    }
    return {
        "arm": arm,
        "seed": seed,
        "policy": POLICY[arm],
        "switch_epoch": switch,
        "child_run": CHILD[arm][seed],
        "child_dir": str(child_dir.resolve()),
        "child_metrics_sha256": sha256(child_dir / "epoch-metrics.jsonl"),
        "child_final_checkpoint_sha256": sha256(child_dir / "epoch-199.pt"),
        "hybrid": {
            "rows": 200,
            "best_epoch": max(range(200), key=hybrid_pgd.__getitem__),
            "best_robust": max(hybrid_pgd),
            "last_robust": hybrid_pgd[-1],
            "last_clean": hybrid_clean[-1],
            "auc_0_199": auc(hybrid_pgd),
            "auc_post_switch": auc(hybrid_pgd[switch:]),
            "post_switch_mean": sum(hybrid_pgd[switch:]) / len(hybrid_pgd[switch:]),
            "post_switch_clean_mean": sum(hybrid_clean[switch:]) / len(hybrid_clean[switch:]),
            "median_images_per_second": child_throughput,
        },
        "control": {
            "auc_0_199": auc(control_pgd),
            "auc_post_switch": auc(control_pgd[switch:]),
            "last_robust": control_pgd[-1],
            "last_clean": control_clean[-1],
            "post_switch_mean": sum(control_pgd[switch:]) / len(control_pgd[switch:]),
            "median_images_per_second": control_throughput,
        },
        "shock": {
            "delta_at_plus_1": hybrid_pgd[switch + 1] - control_pgd[switch + 1],
            "delta_at_plus_5": hybrid_pgd[switch + 5] - control_pgd[switch + 5],
            "delta_at_plus_10": hybrid_pgd[switch + 10] - control_pgd[switch + 10],
            "maximum_negative_dip": min(delta),
            "recovery_epoch": recovery,
        },
        "deltas": {
            # Dense validation trajectory deltas; these are the values used
            # for AUC and are distinct from independent endpoint deltas.
            "final_robust": hybrid_pgd[-1] - control_pgd[-1],
            "final_clean": hybrid_clean[-1] - control_clean[-1],
            "auc_0_199": auc(hybrid_pgd) - auc(control_pgd),
            "auc_post_switch": auc(hybrid_pgd[switch:]) - auc(control_pgd[switch:]),
            "post_switch_mean": sum(hybrid_pgd[switch:]) / len(hybrid_pgd[switch:])
            - sum(control_pgd[switch:]) / len(control_pgd[switch:]),
            "throughput_fraction": child_throughput / control_throughput - 1.0,
        },
        "endpoint_deltas": endpoint_deltas,
        "endpoints": endpoints,
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value * 100:.{digits}f}%"


def main() -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"missing stagewise root: {ROOT}")
    arms = [build_arm(arm, seed) for arm in ("R100", "I100", "R150", "I150") for seed in (1, 2)]
    by_key = {(row["arm"], row["seed"]): row for row in arms}
    controls = {
        str(seed): {
            "metrics_sha256": sha256(CONTROL[seed] / "epoch-metrics.jsonl"),
            "run_dir": str(CONTROL[seed].resolve()),
            "final_robust": metric(read_rows(CONTROL[seed] / "epoch-metrics.jsonl"), "val_pgd_accuracy")[-1],
            "final_clean": metric(read_rows(CONTROL[seed] / "epoch-metrics.jsonl"), "val_clean_accuracy")[-1],
            "endpoint_149": control_endpoint(seed, 149),
            "endpoint_199": control_endpoint(seed, 199),
        }
        for seed in (1, 2)
    }
    promotion: dict[str, Any] = {}
    for arm in ("R100", "I100", "R150", "I150"):
        rows = [by_key[(arm, seed)] for seed in (1, 2)]
        promotion[arm] = {
            "final_robust_pass": all(r["endpoint_deltas"]["199"]["robust"] > 0 for r in rows),
            "full_auc_pass": all(r["deltas"]["auc_0_199"] >= 0 for r in rows),
            "post_switch_auc_pass": all(r["deltas"]["auc_post_switch"] > 0 for r in rows),
            "clean_guardrail_pass": all(r["endpoint_deltas"]["199"]["clean"] >= -0.01 for r in rows),
            "throughput_guardrail_pass": all(r["deltas"]["throughput_fraction"] >= -0.10 for r in rows),
            "per_seed": {
                str(r["seed"]): {
                    "final_robust_delta_pp": r["endpoint_deltas"]["199"]["robust"] * 100,
                    "final_clean_delta_pp": r["endpoint_deltas"]["199"]["clean"] * 100,
                    "full_auc_delta_pp": r["deltas"]["auc_0_199"] * 100,
                    "post_switch_auc_delta_pp": r["deltas"]["auc_post_switch"] * 100,
                }
                for r in rows
            },
        }
        promotion[arm]["qualifies"] = all(
            promotion[arm][key]
            for key in (
                "final_robust_pass",
                "full_auc_pass",
                "post_switch_auc_pass",
                "clean_guardrail_pass",
                "throughput_guardrail_pass",
            )
        )

    def timing_value(row: dict[str, Any], metric_name: str) -> float:
        if metric_name == "final_robust":
            return float(row["endpoint_deltas"]["199"]["robust"])
        return float(row["deltas"][metric_name])

    timing = {}
    for policy, early, late in (("CROP_RE", "R100", "R150"), ("IDBH_WEAK", "I100", "I150")):
        early_gate, late_gate = promotion[early], promotion[late]
        timing[policy] = {
            "early_dominates": all(
                timing_value(by_key[(early, seed)], metric_name) >= timing_value(by_key[(late, seed)], metric_name)
                for seed in (1, 2)
                for metric_name in ("final_robust", "auc_0_199")
            ),
            "late_dominates": all(
                timing_value(by_key[(late, seed)], metric_name) >= timing_value(by_key[(early, seed)], metric_name)
                for seed in (1, 2)
                for metric_name in ("final_robust", "auc_0_199")
            ),
            "early_qualifies": early_gate["qualifies"],
            "late_qualifies": late_gate["qualifies"],
        }
    qualified = [arm for arm in promotion if promotion[arm]["qualifies"]]
    decision = {
        "incumbent": "CROPSHIFT fixed",
        "qualified_schedules": qualified,
        "automatic_promotion": False,
        "freeze_decision": "human_review_required"
        if len(qualified) > 1
        else (qualified[0] if qualified else "CROPSHIFT"),
        "reason": (
            "No automatic promotion is permitted by the preregistered protocol; "
            "human review is required after the descriptive gates."
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "ert_rslad_stagewise_augmentation_results",
        "status": "complete",
        "source_git_sha": SOURCE_SHA,
        "teacher_checkpoint_sha256": "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983",
        "endpoint_attack_identity_sha256": "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2",
        "dataset": {"name": "cifar10", "train_count": 45000, "validation_count": 5000, "split_seed": 20260722},
        "controls": controls,
        "arms": arms,
        "promotion": promotion,
        "timing_comparison": timing,
        "decision": decision,
        "limitations": [
            "Two development seeds; no population-level seed inference.",
            "Validation endpoint is internal held-out CIFAR-10, not official test.",
            "No AutoAttack or additional timing/augmentation schedule was run.",
            "Hybrid AUC uses the accepted CROPSHIFT prefix and stage-wise suffix at the same epoch-metric contract.",
        ],
    }
    out_json = REPO_ROOT / "docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT / RSLAD Stage-Wise Augmentation Results",
        "",
        "Status: complete. This is the preregistered two-seed internal-validation "
        "screen; no official test or AutoAttack was run.",
        "",
        "## Contract and lineage",
        "",
        f"- Production source SHA: `{SOURCE_SHA}`.",
        "- Teacher: Chen2021LTD WRN34-10, SHA-256 `fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983`.",
        "- Parent: accepted CROPSHIFT seed-specific trajectories; hybrid prefix uses "
        "historical rows before the switch.",
        "- Training attack: KL-PGD10, epsilon 8/255, step 2/255, random start, teacher-clean target.",
        "- Endpoint: independent CE-PGD20, epsilon 8/255, step 2/255, 20 steps, random start, eval mode.",
        "- Trajectory AUC metric: per-epoch internal validation `val_pgd_accuracy` "
        "under the frozen selection attack contract.",
        "- W&B: metrics-only tracking; checkpoints and run bundles remain local.",
        "",
        "## Final endpoint (validation, CE-PGD20)",
        "",
        "| seed | schedule | late policy | switch | clean | robust | Δ robust vs CROPSHIFT | Δ clean |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        ep = row["endpoints"]["199"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['policy']} | "
            f"{row['switch_epoch']} | {fmt(ep['clean'])} | {fmt(ep['robust'])} | "
            f"{row['endpoint_deltas']['199']['robust'] * 100:+.2f} pp | "
            f"{row['endpoint_deltas']['199']['clean'] * 100:+.2f} pp |"
        )
    lines += [
        "",
        "## Hybrid trajectory and post-switch AUC",
        "",
        "AUC is normalized with the repository's trapezoidal epoch convention. "
        "Deltas are schedule minus the same-seed CROPSHIFT control.",
        "",
        "| seed | schedule | best robust (epoch) | last robust | full AUC | "
        "Δ full AUC | post-switch AUC | Δ post AUC | shock +1 | recovery epoch | "
        "throughput Δ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        h, s = row["hybrid"], row["shock"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {fmt(h['best_robust'])} ({h['best_epoch']}) | "
            f"{fmt(h['last_robust'])} | {h['auc_0_199']:.6f} | "
            f"{row['deltas']['auc_0_199'] * 100:+.3f} pp | "
            f"{h['auc_post_switch']:.6f} | {row['deltas']['auc_post_switch'] * 100:+.3f} pp | "
            f"{s['delta_at_plus_1'] * 100:+.2f} pp | "
            f"{s['recovery_epoch'] if s['recovery_epoch'] is not None else 'none'} | "
            f"{row['deltas']['throughput_fraction'] * 100:+.2f}% |"
        )
    lines += [
        "",
        "## Promotion gates (preregistered)",
        "",
        "A schedule must pass both seeds for final robustness, full hybrid AUC "
        "non-degradation, post-switch AUC improvement, and the final-clean "
        "guardrail. No automatic promotion is made.",
        "",
        "| schedule | final robust | full AUC | post-switch AUC | clean guardrail | throughput guardrail | qualifies |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, gates in promotion.items():
        lines.append(
            f"| {arm} | {'PASS' if gates['final_robust_pass'] else 'FAIL'} | "
            f"{'PASS' if gates['full_auc_pass'] else 'FAIL'} | "
            f"{'PASS' if gates['post_switch_auc_pass'] else 'FAIL'} | "
            f"{'PASS' if gates['clean_guardrail_pass'] else 'FAIL'} | "
            f"{'PASS' if gates['throughput_guardrail_pass'] else 'FAIL'} | "
            f"{'YES' if gates['qualifies'] else 'NO'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `R100`/`R150` are CropShift → CROP_RE; `I100`/`I150` are CropShift → IDBH_WEAK.",
        "- Endpoint values and trajectory AUC are reported separately; a final gain alone is not sufficient.",
        "- S150 pre-switch endpoint is the hash-bound historical CROPSHIFT epoch-149 endpoint and is not re-attacked.",
        "- Shock/recovery values are descriptive and do not trigger additional schedule tuning.",
        f"- Decision record: `{decision['freeze_decision']}`; incumbent remains "
        f"`{decision['incumbent']}` pending human review.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "Machine artifact: `docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json` "
        f"(SHA-256 `{sha256(out_json)}`).",
    ]
    (REPO_ROOT / "docs/ERT_RSLAD_STAGEWISE_AUGMENTATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"json": str(out_json), "json_sha256": sha256(out_json), "arm_count": len(arms), "decision": decision},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
