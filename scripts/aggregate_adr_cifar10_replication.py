#!/usr/bin/env python3
"""Aggregate plan 0097's ADR CIFAR-10 replication campaign (decision packet 0009, option B).

Eight arms, three seeds each (one diagnostic pilot arm uses a single seed),
CIFAR-10 ResNet-18 and a MobileNetV2 capacity-extension arm. Reads each
run's official-test evaluation (clean, CE-PGD-20, AutoAttack; best and last
checkpoints) plus, for the three ADR-family arms, a second EMA-weights
evaluation against `best-ema.pt`/`last.pt`. Two arms' seed 0 are historical
reuse (docs/plans/0097's confirmed field-by-field identity match) carried as
recorded constants with named provenance, not re-derived here.

The JSON record and the Markdown report are rendered from one result dict.
Every fresh run's bundle is re-verified before use (hand-run completion
contract, declared artifact hashes, dataset/split/count, threat identity,
AutoAttack provenance); a reused historical result is checked for internal
consistency against its recorded provenance instead, since no run bundle for
it survives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "adr_cifar10_replication_v1"
AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"
OFFICIAL_TEST_COUNT = 10_000
PLAN = "docs/plans/0097-adr-cifar10-replication.md"
DECISION_PACKET_INPUT = "docs/decisions/0009-mobile-robustness-direction-and-first-step.md"

RESULT_PATH = ROOT / "docs/experiments" / f"{CONTRACT}.json"
REPORT_PATH = ROOT / "docs/ADR_CIFAR10_REPLICATION_RESULTS.md"

# Each arm: config file (for documentation only; identity is checked from the
# evaluation rows themselves, not re-parsed from YAML here), the protocol id
# it must carry, the runtime method id, whether it has an independent
# EMA-selected checkpoint, its role in the preregistered comparison, and the
# seeds this campaign runs for it.
ARMS: dict[str, dict[str, Any]] = {
    "pgd_at": {
        "protocol": "controlled_cifar10_r18_v1",
        "method": "pgd_at",
        "has_ema": False,
        "role": "baseline_r18_plain_sgd",
        "seeds": (0, 1, 2),
    },
    "pgd_at_nesterov": {
        "protocol": "controlled_cifar10_r18_adr_v1",
        "method": "pgd_at",
        "has_ema": False,
        "role": "baseline_r18_nesterov_matched",
        "seeds": (0, 1, 2),
    },
    "adr": {
        "protocol": "controlled_cifar10_r18_adr_v1",
        "method": "adr",
        "has_ema": True,
        "role": "adr_r18",
        "seeds": (0, 1, 2),
    },
    "trades": {
        "protocol": "controlled_cifar10_r18_v1",
        "method": "trades",
        "has_ema": False,
        "role": "baseline_trades_r18",
        "seeds": (0, 1, 2),
    },
    "trades_adr": {
        "protocol": "controlled_cifar10_r18_adr_v1",
        "method": "adr_trades",
        "has_ema": True,
        "role": "adr_trades_r18",
        "seeds": (0, 1, 2),
    },
    "trades_49k_validation": {
        "protocol": "controlled_cifar10_r18_trades_49k_validation_v1",
        "method": "trades",
        "has_ema": False,
        "role": "trades_49k_validation_pilot",
        "seeds": (0,),
    },
    "mobilenetv2_pgd_at": {
        "protocol": "controlled_cifar10_mobilenetv2_adr_v1",
        "method": "pgd_at",
        "has_ema": False,
        "role": "baseline_mobilenetv2",
        "seeds": (0, 1, 2),
    },
    "mobilenetv2_adr": {
        "protocol": "controlled_cifar10_mobilenetv2_adr_v1",
        "method": "adr",
        "has_ema": True,
        "role": "adr_mobilenetv2",
        "seeds": (0, 1, 2),
    },
}

# The preregistered comparison (docs/plans/0097, docs/decisions/0009): ADR's
# AutoAttack gain over a matched plain-AT baseline, best checkpoint, compared
# between MobileNetV2 and the Nesterov-matched ResNet-18 pair.
COMPARISON_PAIRS = {
    "resnet18_nesterov_matched": {"treatment": "adr", "baseline": "pgd_at_nesterov"},
    "mobilenetv2": {"treatment": "mobilenetv2_adr", "baseline": "mobilenetv2_pgd_at"},
    # Reported but not decisive (docs/plans/0097's secondary quantities).
    "resnet18_trades_vs_plain_sgd_baseline": {"treatment": "trades_adr", "baseline": "trades"},
}

# Historical reuse, confirmed field-by-field identical to the current
# controlled_cifar10_r18_v1 contract (docs/plans/0097's "Historical reuse --
# confirmed" section). No run bundle survives for either; these are recorded
# constants with named provenance, matching the pattern already used by
# scripts/aggregate_controlled_trades_fix_official_test.py for its own
# "defective" comparison row.
REUSED: dict[tuple[str, int], dict[str, Any]] = {
    ("pgd_at", 0): {
        "run_id": "pgd-at-controlled-s0-c2220f1",
        "source_git_sha_short": "c2220f1",  # only this 7-char prefix is on record, not a full 40-char SHA
        "provenance": "outputs/scientific/pgd-at-controlled-s0-c2220f1/ (local, uncommitted); docs/plans/0097 M1a investigation, 2026-09-09",
        "checkpoints": {
            "best": {"clean_accuracy": 0.8201, "pgd_accuracy": 0.5112, "autoattack_accuracy": 0.4763},
            "last": {"clean_accuracy": 0.8446, "pgd_accuracy": 0.4189, "autoattack_accuracy": 0.4036},
        },
    },
    ("trades", 0): {
        "run_id": "trades-fix-v1-s0-attempt2",
        "source_git_sha_short": "ee9ced0",  # only this 7-char prefix is on record, not a full 40-char SHA
        "provenance": (
            "/home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap/runs/trades-fix-v1/seed0/ "
            "(local, uncommitted); NOT outputs/scientific/trades-controlled-s0-f0c3ace (pre-fix defective run, "
            "docs/debugging/0028-trades-clean-target-detached.md); docs/plans/0097 M1a investigation, 2026-09-09"
        ),
        "checkpoints": {
            "best": {"clean_accuracy": 0.8235, "pgd_accuracy": 0.5066, "autoattack_accuracy": 0.4787},
            "last": {"clean_accuracy": 0.8224, "pgd_accuracy": 0.4749, "autoattack_accuracy": 0.4499},
        },
    },
}

ALIASES = ("best", "last")


class AggregationError(RuntimeError):
    """Raised when an input fails a lineage, identity, or completeness check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_clean_source(expected: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if actual != expected or dirty:
        raise AggregationError("aggregation requires the frozen clean production source")


def _non_overwriting(path: Path) -> None:
    if path.exists():
        raise AggregationError(f"refusing to overwrite registered output: {path}")


def _verify_bundle(run_dir: Path, *, expected_protocol: str) -> dict[str, Any]:
    """Re-check the hand-run completion contract and every declared artifact hash."""
    bundle = run_dir / "run-bundle"
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise AggregationError(f"missing run bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (bundle / "completion.json").is_file():
        raise AggregationError(f"{manifest_path}: no completion.json, the run is not terminal")
    if manifest.get("status") not in {"completed", "sync_pending"}:
        raise AggregationError(f"{manifest_path}: manifest status is {manifest.get('status')!r}")
    marker = (bundle / "error-marker.txt").read_text(encoding="utf-8").strip()
    if marker != "no application error recorded":
        raise AggregationError(f"{manifest_path}: error marker reads {marker!r}")
    if manifest.get("protocol_id") != expected_protocol:
        raise AggregationError(
            f"{manifest_path}: protocol is {manifest.get('protocol_id')!r}, expected {expected_protocol!r}"
        )
    if manifest.get("git", {}).get("dirty"):
        raise AggregationError(f"{manifest_path}: the run source tree was dirty")
    artifacts = []
    for entry in manifest.get("artifacts", []):
        path = Path(entry["path"])
        if not path.is_file():
            raise AggregationError(f"{manifest_path}: declared artifact is missing: {path}")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise AggregationError(f"{path}: sha256 {actual} does not match the manifest {entry['sha256']}")
        artifacts.append({"name": entry["name"], "path": str(path), "sha256": actual})
    return {
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_git_sha": manifest["git"]["sha"],
        "finished_at": manifest["finished_at"],
        "execution": manifest["training_execution_identity"],
        "artifacts": artifacts,
    }


def _load_rows(path: Path, *, expected_method: str, expect_weights: str) -> dict[str, dict[str, Any]]:
    """Read one evaluation-results.json and verify it measured what this contract claims."""
    if not path.is_file():
        raise AggregationError(f"missing evaluation result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    by_alias: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row["count"]) != OFFICIAL_TEST_COUNT:
            raise AggregationError(f"{path}: expected the official {OFFICIAL_TEST_COUNT}-example test split")
        if row["dataset_identity"]["split"] != "test":
            raise AggregationError(f"{path}: split is not the official test split")
        if row["runtime_method"] != expected_method:
            raise AggregationError(f"{path}: method is {row['runtime_method']!r}, expected {expected_method!r}")
        if row.get("weights", "model") != expect_weights:
            raise AggregationError(f"{path}: weights is {row.get('weights')!r}, expected {expect_weights!r}")
        auto = row.get("autoattack")
        if not isinstance(auto, dict):
            raise AggregationError(f"{path}: AutoAttack did not run for checkpoint {row['checkpoint_alias']!r}")
        if auto["attack_version"] != "standard":
            raise AggregationError(f"{path}: AutoAttack version is {auto['attack_version']!r}, not standard")
        commit = auto.get("provenance", {}).get("expected_commit")
        if commit != AUTOATTACK_COMMIT:
            raise AggregationError(f"{path}: AutoAttack commit {commit} is not the pinned upstream")
        # best-ema.pt's own alias is "best-ema", not "best" -- normalize so
        # the rest of this script can treat every arm's rows uniformly.
        alias = "best" if row["checkpoint_alias"] == "best-ema" else row["checkpoint_alias"]
        by_alias[alias] = row
    missing = set(ALIASES) - set(by_alias)
    if missing:
        raise AggregationError(f"{path}: missing checkpoint aliases {sorted(missing)}")
    return by_alias


def _extract(row: dict[str, Any]) -> dict[str, float]:
    return {
        "checkpoint_filename": row["checkpoint_filename"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "clean_accuracy": float(row["clean_accuracy"]),
        "pgd_accuracy": float(row["pgd_accuracy"]),
        "autoattack_accuracy": float(row["autoattack"]["autoattack_accuracy"]),
    }


def _load_arm_seed(
    *, run_root: Path, arm_key: str, arm: dict[str, Any], seed: int
) -> dict[str, Any]:
    reused = REUSED.get((arm_key, seed))
    if reused is not None:
        return {
            "seed": seed,
            "kind": "reused_historical",
            "provenance": reused["provenance"],
            "run_id": reused["run_id"],
            "source_git_sha_short": reused["source_git_sha_short"],
            "model": {alias: dict(reused["checkpoints"][alias]) for alias in ALIASES},
            "ema": None,
        }
    run_dir = run_root / f"{arm_key}-s{seed}" / "train"
    bundle = _verify_bundle(run_dir, expected_protocol=arm["protocol"])
    model_rows = _load_rows(run_dir / "evaluation" / "evaluation-results.json", expected_method=arm["method"], expect_weights="model")
    model = {alias: _extract(model_rows[alias]) for alias in ALIASES}
    ema = None
    if arm["has_ema"]:
        ema_rows = _load_rows(
            run_dir / "evaluation-ema" / "evaluation-results.json", expected_method=arm["method"], expect_weights="ema"
        )
        ema = {alias: _extract(ema_rows[alias]) for alias in ALIASES}
    return {"seed": seed, "kind": "fresh_run", "bundle": bundle, "model": model, "ema": ema}


def _pp(value: float) -> float:
    return round(value * 100.0, 4)


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise AggregationError("cannot summarize an empty set of values")
    mean = sum(values) / len(values)
    variance = 0.0 if len(values) == 1 else sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return {
        "count": len(values),
        "mean": mean,
        "std": variance**0.5,
        "worst": min(values),
        "best": max(values),
    }


def _paired_delta(
    arm_results: dict[str, list[dict[str, Any]]], *, treatment: str, baseline: str, checkpoint: str, weights: str
) -> dict[str, Any]:
    """Per-seed AutoAttack delta (treatment minus baseline), paired by seed."""
    treatment_by_seed = {r["seed"]: r for r in arm_results[treatment]}
    baseline_by_seed = {r["seed"]: r for r in arm_results[baseline]}
    shared_seeds = sorted(set(treatment_by_seed) & set(baseline_by_seed))
    if not shared_seeds:
        raise AggregationError(f"no shared seeds between {treatment!r} and {baseline!r}")
    deltas = []
    for seed in shared_seeds:
        t_block = treatment_by_seed[seed]["ema" if weights == "ema" else "model"]
        b_block = baseline_by_seed[seed]["model"]
        if t_block is None:
            raise AggregationError(f"{treatment!r} seed {seed} has no {weights!r} evaluation")
        t_aa = t_block[checkpoint]["autoattack_accuracy"]
        b_aa = b_block[checkpoint]["autoattack_accuracy"]
        deltas.append(_pp(t_aa - b_aa))
    return {
        "seeds": shared_seeds,
        "deltas_pp": deltas,
        "summary": _summarize(deltas),
    }


def aggregate(*, run_root: Path, expected_source_sha: str) -> dict[str, Any]:
    _require_clean_source(expected_source_sha)
    run_root = run_root.resolve()
    arm_results: dict[str, list[dict[str, Any]]] = {}
    for arm_key, arm in ARMS.items():
        arm_results[arm_key] = [_load_arm_seed(run_root=run_root, arm_key=arm_key, arm=arm, seed=seed) for seed in arm["seeds"]]

    per_arm_summary: dict[str, Any] = {}
    for arm_key, results in arm_results.items():
        arm = ARMS[arm_key]
        summary: dict[str, Any] = {"role": arm["role"], "seeds": [r["seed"] for r in results]}
        for checkpoint in ALIASES:
            summary[f"{checkpoint}_model_autoattack_pp"] = _summarize(
                [_pp(r["model"][checkpoint]["autoattack_accuracy"]) for r in results]
            )
            summary[f"{checkpoint}_model_clean_pp"] = _summarize(
                [_pp(r["model"][checkpoint]["clean_accuracy"]) for r in results]
            )
            if arm["has_ema"]:
                summary[f"{checkpoint}_ema_autoattack_pp"] = _summarize(
                    [_pp(r["ema"][checkpoint]["autoattack_accuracy"]) for r in results if r["ema"] is not None]
                )
        per_arm_summary[arm_key] = summary

    comparisons = {
        name: {
            "treatment": pair["treatment"],
            "baseline": pair["baseline"],
            "best_checkpoint_model_weights": _paired_delta(
                arm_results, treatment=pair["treatment"], baseline=pair["baseline"], checkpoint="best", weights="model"
            ),
        }
        for name, pair in COMPARISON_PAIRS.items()
    }
    # The primary "ADR + WA" comparison also reads the EMA-weights delta for
    # the two arms that carry one.
    for name in ("resnet18_nesterov_matched", "mobilenetv2"):
        pair = COMPARISON_PAIRS[name]
        comparisons[name]["best_checkpoint_ema_weights"] = _paired_delta(
            arm_results, treatment=pair["treatment"], baseline=pair["baseline"], checkpoint="best", weights="ema"
        )

    r18_gain = comparisons["resnet18_nesterov_matched"]["best_checkpoint_model_weights"]["summary"]["mean"]
    mnv2_gain = comparisons["mobilenetv2"]["best_checkpoint_model_weights"]["summary"]["mean"]
    if mnv2_gain > 0 and mnv2_gain > r18_gain:
        verdict = "SIGN_CONFIRMED"
    elif mnv2_gain <= r18_gain and mnv2_gain <= 0:
        verdict = "NOT_CONFIRMED"
    else:
        verdict = "MIXED"

    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "source_git_sha": expected_source_sha,
        "plan": PLAN,
        "decision_packet_input": DECISION_PACKET_INPUT,
        "autoattack_upstream_commit": AUTOATTACK_COMMIT,
        "dataset_scope": {"name": "cifar10", "split": "test", "count": OFFICIAL_TEST_COUNT},
        "arms": {key: ARMS[key] for key in ARMS},
        "runs": arm_results,
        "per_arm_summary": per_arm_summary,
        "preregistered_comparison": {
            "primary_endpoint": "best-checkpoint official-test AutoAttack accuracy, ADR minus matched plain-AT baseline",
            "rule": (
                "SIGN_CONFIRMED if MobileNetV2's mean gain is positive and exceeds ResNet-18's "
                "(Nesterov-matched pair); NOT_CONFIRMED if MobileNetV2's gain is <= ResNet-18's and <= 0; "
                "otherwise MIXED"
            ),
            "comparisons": comparisons,
            "resnet18_mean_gain_pp": r18_gain,
            "mobilenetv2_mean_gain_pp": mnv2_gain,
            "verdict": verdict,
        },
        "stop": (
            "this contract measures three seeds (one pilot arm at one seed) at one protocol family; "
            "it authorizes no promotion, no ImageNet launch, and no method claim beyond the preregistered "
            "sign comparison -- see the plan's preregistered decision rule for the full scope"
        ),
    }


def _fmt_summary(block: dict[str, Any]) -> str:
    return f"{block['mean']:+.2f} pp (n={block['count']}, sd={block['std']:.2f}, range {block['worst']:+.2f}..{block['best']:+.2f})"


def _markdown(result: dict[str, Any]) -> str:
    rule = result["preregistered_comparison"]
    lines = [
        "# ADR CIFAR-10 replication: official test results",
        "",
        "## Decision",
        "",
        f"Verdict: **{rule['verdict']}**.",
        "",
        f"- ResNet-18 (Nesterov-matched pair, `adr` vs `pgd_at_nesterov`): mean AutoAttack gain "
        f"{_fmt_summary(rule['comparisons']['resnet18_nesterov_matched']['best_checkpoint_model_weights']['summary'])}",
        f"- MobileNetV2 (`mobilenetv2_adr` vs `mobilenetv2_pgd_at`): mean AutoAttack gain "
        f"{_fmt_summary(rule['comparisons']['mobilenetv2']['best_checkpoint_model_weights']['summary'])}",
        "",
        rule["rule"],
        "",
        "## Per-arm summary (best checkpoint, model weights, AutoAttack, percentage points)",
        "",
        "| arm | role | seeds | mean | sd |",
        "|---|---|---|---:|---:|",
    ]
    for arm_key, summary in result["per_arm_summary"].items():
        block = summary["best_model_autoattack_pp"]
        lines.append(
            f"| `{arm_key}` | {summary['role']} | {summary['seeds']} | {block['mean']:.2f} | {block['std']:.2f} |"
        )
    lines += [
        "",
        "## Secondary comparisons",
        "",
        f"- TRADES (`trades_adr` vs `trades`, no Nesterov-matched TRADES baseline exists): mean gain "
        f"{_fmt_summary(rule['comparisons']['resnet18_trades_vs_plain_sgd_baseline']['best_checkpoint_model_weights']['summary'])}",
        "",
        result["stop"],
        "",
        "## Provenance",
        "",
        f"Record: `docs/experiments/{RESULT_PATH.name}`, hash in `docs/experiments/{RESULT_PATH.name}.sha256`. "
        f"Contract `{result['contract']}`. Aggregation source SHA `{result['source_git_sha']}`. "
        f"Plan: `{result['plan']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="directory containing <arm>-s<seed>/train/ per arm")
    parser.add_argument("--expected-source-sha", required=True, help="clean source SHA this aggregation runs from")
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    result_path, report_path = args.result.resolve(), args.report.resolve()
    _non_overwriting(result_path)
    _non_overwriting(report_path)
    result = aggregate(run_root=args.run_root, expected_source_sha=args.expected_source_sha)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown(result), encoding="utf-8")
    sidecar = result_path.with_name(result_path.name + ".sha256")
    sidecar.write_text(_sha256(result_path) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result": str(result_path), "report": str(report_path), "verdict": result["preregistered_comparison"]["verdict"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
