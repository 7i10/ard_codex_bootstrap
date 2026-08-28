"""Aggregate the final single-switch IDBH timing screen.

This is a read-only aggregator.  It joins the accepted CROPSHIFT prefix to
the six fresh suffix trajectories, verifies independent CE-PGD20 endpoint
identity, and applies the preregistered I100 replacement/freeze rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap")
CONTROL_ROOT = ROOT / "ert-rslad-static-trajstab-v1"
RUN_ROOT = ROOT / "ert-rslad-single-switch-timing-v1" / "runs"
CONTROL = {1: CONTROL_ROOT / "cropshift-s1-r2", 2: CONTROL_ROOT / "cropshift-s2-r1"}
CHILD = {
    "I50": {1: "idbh-s50-s1-final", 2: "idbh-s50-s2-final"},
    "I75": {1: "idbh-s75-s1-final", 2: "idbh-s75-s2-final"},
    "I125": {1: "idbh-s125-s1-final", 2: "idbh-s125-s2-final"},
}
SWITCH = {"I50": 50, "I75": 75, "I125": 125}
ENDPOINT_ATTACK = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
SOURCE_SHA = "54b637492309bcf4bb9f3c99ddec0398aa7cce1e"
TEACHER_SHA = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    rows.sort(key=lambda row: int(row["epoch"]))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise ValueError(f"non-contiguous trajectory: {path}")
    return rows


def auc(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("AUC requires at least two values")
    return (values[0] / 2 + sum(values[1:-1]) + values[-1] / 2) / (len(values) - 1)


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def endpoint(path: Path, *, expected_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = read_json(path)
    if data.get("attack_identity_sha256") != ENDPOINT_ATTACK:
        raise ValueError(f"endpoint attack identity drift: {path}")
    if data.get("row_count") != expected_rows:
        raise ValueError(f"endpoint row count drift: {path}")
    return {
        "path": str(path.resolve()),
        "endpoint_json_sha256": sha256(path),
        "rows_sha256": data["rows_sha256"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "clean": float(data["clean_accuracy"]),
        "robust": float(data["robust_accuracy"]),
        "row_count": int(data["row_count"]),
    }


def control_endpoint(seed: int, epoch: int, split: str = "validation") -> dict[str, Any]:
    name = "cropshift-s1-r2" if seed == 1 else "cropshift-s2-r1"
    path = CONTROL_ROOT / "endpoints" / name / f"epoch-{epoch:03d}" / split / "endpoint.json"
    return endpoint(path, expected_rows=5000 if split == "validation" else 45000)


def child_endpoint(seed: int, arm: str, epoch: int, split: str) -> dict[str, Any]:
    path = RUN_ROOT / CHILD[arm][seed] / "stagewise-endpoints" / f"epoch-{epoch:03d}" / split / "endpoint.json"
    return endpoint(path, expected_rows=5000 if split == "validation" else 45000)


def build_arm(arm: str, seed: int) -> dict[str, Any]:
    switch = SWITCH[arm]
    child = RUN_ROOT / CHILD[arm][seed]
    control_rows = read_rows(CONTROL[seed] / "epoch-metrics.jsonl")
    child_rows = read_rows(child / "epoch-metrics.jsonl")
    if [int(row["epoch"]) for row in child_rows] != list(range(switch, 200)):
        raise ValueError(f"child trajectory coverage drift: {arm} seed{seed}")
    hybrid = control_rows[:switch] + child_rows
    if [int(row["epoch"]) for row in hybrid] != list(range(200)):
        raise ValueError(f"hybrid trajectory coverage drift: {arm} seed{seed}")
    control_pgd = values(control_rows, "val_pgd_accuracy")
    hybrid_pgd = values(hybrid, "val_pgd_accuracy")
    control_clean = values(control_rows, "val_clean_accuracy")
    hybrid_clean = values(hybrid, "val_clean_accuracy")
    child_throughput = sorted(values(child_rows, "train_images_per_second"))[len(child_rows) // 2]
    control_suffix_throughput = sorted(values(control_rows[switch:], "train_images_per_second"))
    control_throughput = control_suffix_throughput[len(control_suffix_throughput) // 2]
    post_delta = [hybrid_pgd[i] - control_pgd[i] for i in range(switch, 200)]
    recovery = None
    for index, value in enumerate(post_delta, start=switch):
        if index > switch and value >= 0 and any(previous < 0 for previous in post_delta[: index - switch]):
            recovery = index
            break
    endpoint_records: dict[str, dict[str, dict[str, Any]]] = {}
    for epoch in (99, 149, 199):
        # Only evaluate the checkpoints explicitly required for each switch.
        if epoch < switch and epoch not in (149, 199):
            continue
        endpoint_records[str(epoch)] = {
            split: child_endpoint(seed, arm, epoch, split) for split in ("train", "validation")
        }
    control_endpoints = {
        str(epoch): {split: control_endpoint(seed, epoch, split) for split in ("train", "validation")}
        for epoch in endpoint_records
    }
    endpoint_deltas = {
        epoch: {
            split: {
                "clean": endpoint_records[epoch][split]["clean"] - control_endpoints[epoch][split]["clean"],
                "robust": endpoint_records[epoch][split]["robust"] - control_endpoints[epoch][split]["robust"],
            }
            for split in ("train", "validation")
        }
        for epoch in endpoint_records
    }
    return {
        "arm": arm,
        "seed": seed,
        "switch_epoch": switch,
        "child_run": CHILD[arm][seed],
        "child_dir": str(child.resolve()),
        "child_metrics_sha256": sha256(child / "epoch-metrics.jsonl"),
        "child_final_checkpoint_sha256": sha256(child / "epoch-199.pt"),
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
            "maximum_negative_dip": min(post_delta),
            "recovery_epoch": recovery,
        },
        "deltas": {
            "final_robust": hybrid_pgd[-1] - control_pgd[-1],
            "final_clean": hybrid_clean[-1] - control_clean[-1],
            "auc_0_199": auc(hybrid_pgd) - auc(control_pgd),
            "auc_post_switch": auc(hybrid_pgd[switch:]) - auc(control_pgd[switch:]),
            "post_switch_mean": sum(hybrid_pgd[switch:]) / len(hybrid_pgd[switch:])
            - sum(control_pgd[switch:]) / len(control_pgd[switch:]),
            "throughput_fraction": child_throughput / control_throughput - 1.0,
        },
        "endpoint_deltas": endpoint_deltas,
        "endpoints": endpoint_records,
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value * 100:.{digits}f}%"


def load_reference_profile() -> dict[str, Any]:
    static = read_json(REPO_ROOT / "docs/experiments/ert_rslad_static_augmentation_family_v1.json")
    old = read_json(REPO_ROOT / "docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json")
    profile: dict[str, Any] = {"I0": {}}
    for seed in (1, 2):
        item = static["candidates"]["IDBH_WEAK"]["runs"][f"seed{seed}"]
        ep = static["candidates"]["IDBH_WEAK"]["endpoint"][f"seed{seed}"]["199"]
        profile["I0"][str(seed)] = {
            "source": "static_IDBH_WEAK_reference",
            "final_robust": ep["robust"],
            "final_clean": ep["clean"],
            "auc_0_199": item["trajectory"]["auc"],
        }
    for old_arm in old["arms"]:
        arm = old_arm["arm"]
        if arm.startswith("I"):
            profile.setdefault(arm, {})[str(old_arm["seed"])] = {
                "source": "stagewise_v1_reference",
                "final_robust": old_arm["endpoints"]["199"]["robust"],
                "final_clean": old_arm["endpoints"]["199"]["clean"],
                "auc_0_199": old_arm["hybrid"]["auc_0_199"],
                "auc_post_switch": old_arm["hybrid"]["auc_post_switch"],
                "switch_epoch": old_arm["switch_epoch"],
            }
    return profile


def main() -> None:
    arms = [build_arm(arm, seed) for arm in ("I50", "I75", "I125") for seed in (1, 2)]
    by_key = {(row["arm"], row["seed"]): row for row in arms}
    promotion: dict[str, Any] = {}
    for arm in ("I50", "I75", "I125"):
        rows = [by_key[(arm, seed)] for seed in (1, 2)]
        promotion[arm] = {
            "final_robust_pass": all(row["endpoint_deltas"]["199"]["validation"]["robust"] > 0 for row in rows),
            "full_auc_pass": all(row["deltas"]["auc_0_199"] >= 0 for row in rows),
            "post_switch_auc_pass": all(row["deltas"]["auc_post_switch"] > 0 for row in rows),
            "clean_guardrail_pass": all(row["endpoint_deltas"]["199"]["validation"]["clean"] >= -0.01 for row in rows),
            "throughput_guardrail_pass": all(row["deltas"]["throughput_fraction"] >= -0.10 for row in rows),
            "per_seed": {
                str(row["seed"]): {
                    "final_robust_delta_pp": row["endpoint_deltas"]["199"]["validation"]["robust"] * 100,
                    "final_clean_delta_pp": row["endpoint_deltas"]["199"]["validation"]["clean"] * 100,
                    "full_auc_delta_pp": row["deltas"]["auc_0_199"] * 100,
                    "post_switch_auc_delta_pp": row["deltas"]["auc_post_switch"] * 100,
                }
                for row in rows
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
    reference = read_json(REPO_ROOT / "docs/experiments/ert_rslad_stagewise_augmentation_results_v1.json")
    i100 = {(row["seed"]): row for row in reference["arms"] if row["arm"] == "I100"}
    replacement: dict[str, Any] = {}
    for arm in ("I50", "I75", "I125"):
        replacement[arm] = {
            "final_robust_dominates_i100": all(
                by_key[(arm, seed)]["endpoints"]["199"]["validation"]["robust"]
                > i100[seed]["endpoints"]["199"]["robust"]
                for seed in (1, 2)
            ),
            "full_auc_not_lower_than_i100": all(
                by_key[(arm, seed)]["hybrid"]["auc_0_199"] >= i100[seed]["hybrid"]["auc_0_199"]
                for seed in (1, 2)
            ),
            "post_switch_not_lower_than_i100": all(
                by_key[(arm, seed)]["hybrid"]["auc_post_switch"] >= i100[seed]["hybrid"]["auc_post_switch"]
                for seed in (1, 2)
            ),
        }
        replacement[arm]["replaces_i100"] = all(replacement[arm].values())
    replacing = [arm for arm, result in replacement.items() if result["replaces_i100"]]
    freeze = replacing[0] if len(replacing) == 1 else "I100"
    result = {
        "schema_version": 1,
        "kind": "ert_rslad_single_switch_timing_results_v1",
        "status": "complete",
        "source_git_sha": SOURCE_SHA,
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "endpoint_attack_identity_sha256": ENDPOINT_ATTACK,
        "dataset": {"name": "cifar10", "train_count": 45000, "validation_count": 5000, "split_seed": 20260722},
        "candidate_set": [50, 75, 100, 125],
        "explored_reference": [0, 150],
        "controls": {
            str(seed): {
                "run_dir": str(CONTROL[seed].resolve()),
                "metrics_sha256": sha256(CONTROL[seed] / "epoch-metrics.jsonl"),
                "endpoint_199_validation": control_endpoint(seed, 199),
            }
            for seed in (1, 2)
        },
        "arms": arms,
        "promotion": promotion,
        "replacement_vs_i100": replacement,
        "timing_profile_reference": load_reference_profile(),
        "freeze": {
            "incumbent": "I100",
            "decision": freeze,
            "candidate_replacement_count": len(replacing),
            "additional_timing_allowed": False,
            "multi_stage_explored": False,
            "search_closed": True,
        },
        "limitations": [
            "Two development seeds; no population-level seed inference.",
            "Endpoint is fixed internal validation, not official CIFAR-10 test.",
            "No AutoAttack or additional timing was run.",
            "AUC uses accepted CROPSHIFT prefix and dense child suffix under the same trajectory metric contract.",
        ],
    }
    out_json = REPO_ROOT / "docs/experiments/ert_rslad_single_switch_timing_results_v1.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ERT / RSLAD Single-Switch Augmentation Timing",
        "",
        "Status: complete. This final finite timing screen uses fresh I50/I75/I125 suffixes, "
        "reuses hash-bound I100/I150 references, and does not use official test or AutoAttack.",
        "",
        "## Contract and lineage",
        "",
        f"- Production source SHA: `{SOURCE_SHA}`.",
        f"- Teacher SHA-256: `{TEACHER_SHA}`.",
        "- Prefix: accepted CROPSHIFT control; late policy: frozen IDBH_WEAK.",
        "- Training attack: KL-PGD10, epsilon 8/255, step 2/255, random start, teacher-clean target.",
        "- Endpoint: independent CE-PGD20, epsilon 8/255, step 2/255, 20 steps, random start, eval mode.",
        "- Endpoint table below is fixed internal validation (5,000 samples); train endpoints are retained in the "
        "machine artifact.",
        "",
        "## Fresh endpoint results",
        "",
        "| seed | arm | switch | clean | robust | Δ robust vs CROPSHIFT | Δ clean | full AUC Δ (pp) | "
        "post-switch AUC Δ (pp) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        ep = row["endpoints"]["199"]["validation"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['switch_epoch']} | {fmt(ep['clean'])} | {fmt(ep['robust'])} | "
            f"{row['endpoint_deltas']['199']['validation']['robust'] * 100:+.2f} pp | "
            f"{row['endpoint_deltas']['199']['validation']['clean'] * 100:+.2f} pp | "
            f"{row['deltas']['auc_0_199'] * 100:+.3f} | {row['deltas']['auc_post_switch'] * 100:+.3f} |"
        )
    lines += [
        "",
        "## Shock and throughput",
        "",
        "| seed | arm | +1 epoch Δ | +5 epoch Δ | +10 epoch Δ | max negative dip | recovery epoch | throughput Δ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['shock']['delta_at_plus_1'] * 100:+.3f} pp | "
            f"{row['shock']['delta_at_plus_5'] * 100:+.3f} pp | {row['shock']['delta_at_plus_10'] * 100:+.3f} pp | "
            f"{row['shock']['maximum_negative_dip'] * 100:+.3f} pp | {row['shock']['recovery_epoch'] or '—'} | "
            f"{row['deltas']['throughput_fraction'] * 100:+.1f}% |"
        )
    lines += ["", "## Preregistered gates and I100 replacement", ""]
    for arm in ("I50", "I75", "I125"):
        gate = promotion[arm]
        rep = replacement[arm]
        lines.append(
            f"- `{arm}`: qualifies={gate['qualifies']}; final={gate['final_robust_pass']}, "
            f"full-AUC={gate['full_auc_pass']}, post-AUC={gate['post_switch_auc_pass']}, "
            f"clean={gate['clean_guardrail_pass']}, throughput={gate['throughput_guardrail_pass']}; "
            f"replaces I100={rep['replaces_i100']}."
        )
    lines += [
        "",
        f"**Freeze decision: `{freeze}`.** The finite search is closed; no additional switch timing or "
        "multi-stage schedule was run.",
        "",
        "## Interpretation",
        "",
        "The timing profile is descriptive over I0/I50/I75/I100/I125/I150. A candidate is not called globally optimal "
        "from two development seeds. I100 remains the incumbent unless a candidate satisfies the preregistered "
        "two-seed final-robust and full-AUC dominance rules recorded in the machine artifact.",
        "",
        "## Next stage (not started)",
        "",
        "If human review accepts the freeze, the next experiment is three unseen paired seeds for confirmation and "
        "full-training stochasticity characterization. Student-History/Ordering work remains separate. No such "
        "run was started here.",
        "",
        "Machine artifact: `docs/experiments/ert_rslad_single_switch_timing_results_v1.json`.",
    ]
    (REPO_ROOT / "docs/ERT_RSLAD_SINGLE_SWITCH_TIMING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "complete", "freeze": freeze, "arms": len(arms), "output": str(out_json)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
