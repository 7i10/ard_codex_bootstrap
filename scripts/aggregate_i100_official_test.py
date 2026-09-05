#!/usr/bin/env python3
"""Aggregate the plan-0091 official-test AutoAttack evaluations.

The screen asks one question: does the I100 augmentation schedule still beat its
matched CROP_SUFFIX control when the comparison moves from the 5,000-sample
internal validation split to the official 10,000-example CIFAR-10 test set, under
CE-PGD20 and standard AutoAttack.

The JSON record and the Markdown report are rendered from one result dict, so the
two can never disagree.  Every input is re-verified before it is used: the split
must be the official test split with 10,000 examples, the threat identity must be
the registered CE-PGD20 contract, and AutoAttack must have actually run under the
pinned upstream commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "ard_i100_official_test_autoattack_v1"
# The registered CE-PGD20 endpoint identity used throughout the ERT line.
ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
# Pinned AutoAttack upstream, per pyproject and docs/UPSTREAM_BASELINES.md.
AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"
OFFICIAL_TEST_COUNT = 10_000
SEEDS = ("confirm-a", "confirm-b", "confirm-c")
ARMS = ("i100", "crop")
RESULT_PATH = ROOT / "docs/experiments/ard_i100_official_test_autoattack_v1.json"
REPORT_PATH = ROOT / "docs/ERT_RSLAD_I100_OFFICIAL_TEST_AUTOATTACK.md"


class AggregationError(RuntimeError):
    """Raised when an input fails a lineage or identity check."""


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


def _load_arm(campaign: Path, seed: str, arm: str) -> dict[str, Any]:
    """Read one evaluation and verify it measured what this contract claims."""
    path = campaign / "eval" / f"{seed}-{arm}" / "evaluation-results.json"
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
        if row["threat_hash"] != ENDPOINT_ATTACK_SHA256:
            raise AggregationError(f"{path}: threat identity differs from the registered CE-PGD20 contract")
        auto = row.get("autoattack")
        if not isinstance(auto, dict):
            raise AggregationError(f"{path}: AutoAttack did not run for {row['checkpoint_alias']}")
        commit = auto.get("provenance", {}).get("expected_commit")
        if commit != AUTOATTACK_COMMIT:
            raise AggregationError(f"{path}: AutoAttack commit {commit} is not the pinned upstream")
        by_alias[row["checkpoint_alias"]] = row
    missing = {"best", "last"} - set(by_alias)
    if missing:
        raise AggregationError(f"{path}: missing checkpoint aliases {sorted(missing)}")
    return {
        "result_path": str(path),
        "result_sha256": _sha256(path),
        "train_run_id": by_alias["best"]["train_run_id"],
        "checkpoints": {
            alias: {
                "checkpoint_sha256": row["checkpoint_sha256"],
                "clean_accuracy": float(row["clean_accuracy"]),
                "pgd_accuracy": float(row["pgd_accuracy"]),
                "autoattack_accuracy": float(row["autoattack"]["autoattack_accuracy"]),
                "attack_version": row["autoattack"]["attack_version"],
            }
            for alias, row in by_alias.items()
        },
    }


def _pp(value: float) -> float:
    """Convert an accuracy fraction difference into percentage points."""
    return round(value * 100.0, 4)


def _decide(deltas: dict[str, float]) -> str:
    """Preregistered rule: positive in all three seeds confirms; two or more negative refutes."""
    positive = sum(1 for value in deltas.values() if value > 0.0)
    negative = sum(1 for value in deltas.values() if value < 0.0)
    if positive == len(deltas):
        return "CONFIRMED"
    if negative >= 2:
        return "NOT_CONFIRMED"
    return "MIXED"


def aggregate(*, campaign: Path, expected_source_sha: str) -> dict[str, Any]:
    _require_clean_source(expected_source_sha)
    campaign = campaign.resolve()
    arms = {seed: {arm: _load_arm(campaign, seed, arm) for arm in ARMS} for seed in SEEDS}
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "source_git_sha": expected_source_sha,
        "campaign_root": str(campaign),
        "endpoint_attack_identity_sha256": ENDPOINT_ATTACK_SHA256,
        "autoattack_upstream_commit": AUTOATTACK_COMMIT,
        "dataset_scope": {"split": "test", "count": OFFICIAL_TEST_COUNT, "name": "cifar10"},
        "seeds": arms,
        "comparisons": {},
    }
    primary: dict[str, float] = {}
    for seed in SEEDS:
        i100, crop = arms[seed]["i100"]["checkpoints"], arms[seed]["crop"]["checkpoints"]
        seed_block: dict[str, Any] = {}
        for alias in ("best", "last"):
            seed_block[alias] = {
                metric: _pp(i100[alias][metric] - crop[alias][metric])
                for metric in ("clean_accuracy", "pgd_accuracy", "autoattack_accuracy")
            }
        seed_block["robust_overfit_gap_pp"] = {
            arm: _pp(vals["best"]["autoattack_accuracy"] - vals["last"]["autoattack_accuracy"])
            for arm, vals in (("i100", i100), ("crop", crop))
        }
        result["comparisons"][seed] = seed_block
        primary[seed] = seed_block["last"]["autoattack_accuracy"]
    result["decision"] = {
        "primary_endpoint": (
            "epoch199 last-checkpoint official-test AutoAttack accuracy, I100 minus matched CROP_SUFFIX"
        ),
        "per_seed_pp": primary,
        "verdict": _decide(primary),
        "secondary_best_checkpoint_pp": {
            seed: result["comparisons"][seed]["best"]["autoattack_accuracy"] for seed in SEEDS
        },
        "internal_validation_reference_pp": {"confirm-a": 0.78, "confirm-b": 0.68, "confirm-c": 0.62},
        "stop": (
            "this screen measures the official test endpoint only; it authorizes no promotion, "
            "no new seed, no architecture or dataset extension, and no sample-level intervention"
        ),
    }
    return result


def _table(result: dict[str, Any], alias: str) -> str:
    lines = [
        "| seed | arm | clean | CE-PGD20 | AutoAttack |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        for arm in ARMS:
            row = result["seeds"][seed][arm]["checkpoints"][alias]
            lines.append(
                f"| {seed} | {arm.upper()} | {row['clean_accuracy'] * 100:.2f}% "
                f"| {row['pgd_accuracy'] * 100:.2f}% | {row['autoattack_accuracy'] * 100:.2f}% |"
            )
    return "\n".join(lines)


def _delta_table(result: dict[str, Any], alias: str) -> str:
    lines = ["| seed | clean Δ | CE-PGD20 Δ | AutoAttack Δ |", "| --- | ---: | ---: | ---: |"]
    for seed in SEEDS:
        d = result["comparisons"][seed][alias]
        lines.append(
            f"| {seed} | {d['clean_accuracy']:+.2f} pp | {d['pgd_accuracy']:+.2f} pp "
            f"| {d['autoattack_accuracy']:+.2f} pp |"
        )
    return "\n".join(lines)


def _markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    ro_rows = "\n".join(
        "| {seed} | {i:+.2f} pp | {c:+.2f} pp |".format(
            seed=seed,
            i=result["comparisons"][seed]["robust_overfit_gap_pp"]["i100"],
            c=result["comparisons"][seed]["robust_overfit_gap_pp"]["crop"],
        )
        for seed in SEEDS
    )
    lineage_rows = "\n".join(
        "| {seed} | {arm} | `{run}` | `{sha}...` |".format(
            seed=seed,
            arm=arm.upper(),
            run=result["seeds"][seed][arm]["train_run_id"],
            sha=result["seeds"][seed][arm]["result_sha256"][:16],
        )
        for seed in SEEDS
        for arm in ARMS
    )
    per_seed = ", ".join(f"{seed} {value:+.2f} pp" for seed, value in decision["per_seed_pp"].items())
    ref = decision["internal_validation_reference_pp"]
    return f"""# I100 official CIFAR-10 test and AutoAttack

## Decision

The preregistered primary endpoint is the epoch-199 last-checkpoint AutoAttack
accuracy on the official 10,000-example CIFAR-10 test set, I100 minus its matched
CROP_SUFFIX control, on the three previously unused confirmation seeds.

Verdict: **{decision["verdict"]}**.  Per seed: {per_seed}.

The rule was fixed before the runs: positive in all three seeds confirms the
claim, two or more negative refutes it, anything else is mixed.  For reference,
the internal-validation CE-PGD20 differences these runs were meant to test were
{ref["confirm-a"]:+.2f} / {ref["confirm-b"]:+.2f} / {ref["confirm-c"]:+.2f} pp.
One test example is 0.01 pp.

{decision["stop"].capitalize()}.

## Last checkpoint (epoch 199)

{_table(result, "last")}

{_delta_table(result, "last")}

## Best checkpoint

{_table(result, "best")}

{_delta_table(result, "best")}

## Robust overfitting

Best minus last AutoAttack accuracy, per arm.  A larger value means the arm lost
more robustness between its best epoch and the end of training.

| seed | I100 | CROP_SUFFIX |
| --- | ---: | ---: |
{ro_rows}

## Scope and lineage

Every evaluation read only saved weights in a separate process.  Each was checked
before use: the official test split with {OFFICIAL_TEST_COUNT:,} examples, the
registered CE-PGD20 threat identity `{ENDPOINT_ATTACK_SHA256[:16]}...`, and
standard AutoAttack from the pinned upstream commit
`{AUTOATTACK_COMMIT[:12]}...`.  Aggregation source `{result["source_git_sha"][:12]}...`;
campaign root `{result["campaign_root"]}`.

| seed | arm | training run | evaluation record SHA-256 |
| --- | --- | --- | --- |
{lineage_rows}

Record: `docs/experiments/{RESULT_PATH.name}` (SHA-256 in the sidecar).
Plan: `docs/plans/0091-i100-official-test-autoattack.md`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="campaign root holding eval/<seed>-<arm>/")
    parser.add_argument("--expected-source-sha", required=True, help="clean source SHA this aggregation runs from")
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    result_path, report_path = args.result.resolve(), args.report.resolve()
    _non_overwriting(result_path)
    _non_overwriting(report_path)
    result = aggregate(campaign=args.campaign, expected_source_sha=args.expected_source_sha)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_markdown(result), encoding="utf-8")
    sidecar = result_path.with_name(result_path.name + ".sha256")
    sidecar.write_text(_sha256(result_path) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"result": str(result_path), "report": str(report_path), "decision": result["decision"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
