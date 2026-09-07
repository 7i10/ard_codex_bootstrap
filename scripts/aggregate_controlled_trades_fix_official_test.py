#!/usr/bin/env python3
"""Aggregate the plan-0027 B5 official test of the fixed controlled TRADES baseline.

The question is narrow.  The controlled TRADES baseline was trained with a defect:
the clean branch of its KL term was detached, so the clean prediction received no
gradient from that term (`docs/debugging/0028-trades-clean-target-detached.md`).
That run scored 45.14 % AutoAttack at its best checkpoint against a published
ResNet-18 TRADES range of 49.0--49.4 %.  The objective was corrected and the run
repeated at identical config, protocol and seed.  This contract reports what the
corrected run scores on the official 10,000-example CIFAR-10 test set under clean
accuracy, CE-PGD-20 and standard AutoAttack, and applies the rule that
`docs/decisions/0005-trades-fix-official-evaluation.md` fixed before the numbers
were seen.

The JSON record and the Markdown report are rendered from one result dict, so the
two cannot disagree.  Every input is re-verified before use: the bundle must be a
successful hand-run, every artifact it declares must hash to the value it
declares, the split must be the official test split, the threat identity must be
the registered CE-PGD-20 contract, and AutoAttack must have run under the pinned
upstream commit.

The comparison arm -- the defective run's official values -- has no run bundle in
the runtime tree.  It is carried here as a recorded constant with its provenance
named, and the record marks it as such.  Nothing in this contract recomputes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "ard_controlled_trades_fix_official_test_v1"
# The registered CE-PGD20 endpoint identity, shared with the I100 official test.
ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
# Pinned AutoAttack upstream, per pyproject and docs/UPSTREAM_BASELINES.md.
AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"
OFFICIAL_TEST_COUNT = 10_000
TRAIN_RUN_ID = "trades-fix-v1-s0-attempt2"
TRAINING_CONFIG_HASH = "cbed20a1330c49e9deb9efb84856550e7e47872e1157019c6e979d3111cc8299"
PROTOCOL_ID = "controlled_cifar10_r18_v1"
ALIASES = ("best", "last")

# Published AutoAttack range for TRADES on ResNet-18, CIFAR-10, Linf 8/255.
LITERATURE_AA_BAND = (0.490, 0.494)
# The same engine's PGD-AT baseline, for the internal consistency note only.
PGD_AT_BEST_AA = 0.4763

# The defective run's official test values.  Source: plan 0027 progress log
# 2026-08-07 and docs/EXPERIMENT_DASHBOARD.md, run `trades-controlled-s0-f0c3ace`.
# No run bundle for it survives in the runtime tree, so these are recorded values,
# not values this aggregator re-derived.
DEFECTIVE = {
    "run_id": "trades-controlled-s0-f0c3ace",
    "source_git_sha": "f0c3acedbdda9b032531bd72f0ec54684bee6d47",
    "lineage": "recorded_value_no_bundle",
    "provenance": (
        "docs/plans/0027-controlled-teacherless-baselines.md progress log 2026-08-07; "
        "docs/EXPERIMENT_DASHBOARD.md controlled TRADES seed 0 row"
    ),
    "checkpoints": {
        "best": {"clean_accuracy": 0.8135, "pgd_accuracy": 0.4783, "autoattack_accuracy": 0.4514},
        "last": {"clean_accuracy": 0.8220, "pgd_accuracy": 0.4546, "autoattack_accuracy": 0.4325},
    },
}

# Preregistered in decision 0005 option A, before any official number was seen.
RULE_CONFIRM_AA = 0.480
RULE_PARTIAL_DELTA_AA = 0.020

RESULT_PATH = ROOT / "docs/experiments" / f"{CONTRACT}.json"
REPORT_PATH = ROOT / "docs/CONTROLLED_TRADES_FIX_OFFICIAL_TEST.md"


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


def _verify_bundle(run_dir: Path) -> dict[str, Any]:
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
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise AggregationError(f"{manifest_path}: protocol is {manifest.get('protocol_id')!r}")
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
        "wandb_url": manifest.get("wandb_url"),
        "execution": manifest["training_execution_identity"],
        "artifacts": artifacts,
    }


def _load_rows(run_dir: Path, *, require_autoattack: bool) -> dict[str, dict[str, Any]]:
    """Read one evaluation and verify it measured what this contract claims."""
    path = run_dir / "evaluation-results.json"
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
        if row["train_run_id"] != TRAIN_RUN_ID:
            raise AggregationError(f"{path}: training run is {row['train_run_id']!r}, not {TRAIN_RUN_ID!r}")
        if row["config_hash"] != TRAINING_CONFIG_HASH:
            raise AggregationError(f"{path}: training config hash {row['config_hash']} is not the fixed-run hash")
        if row["method"] != "trades" or row["teacher"] is not None:
            raise AggregationError(f"{path}: method/teacher identity is not teacherless TRADES")
        auto = row.get("autoattack")
        if require_autoattack:
            if not isinstance(auto, dict):
                raise AggregationError(f"{path}: AutoAttack did not run for {row['checkpoint_alias']}")
            if auto["attack_version"] != "standard":
                raise AggregationError(f"{path}: AutoAttack version is {auto['attack_version']!r}, not standard")
            commit = auto.get("provenance", {}).get("expected_commit")
            if commit != AUTOATTACK_COMMIT:
                raise AggregationError(f"{path}: AutoAttack commit {commit} is not the pinned upstream")
        elif auto is not None:
            raise AggregationError(f"{path}: expected the PGD-only evaluation to carry no AutoAttack result")
        by_alias[row["checkpoint_alias"]] = row
    missing = set(ALIASES) - set(by_alias)
    if missing:
        raise AggregationError(f"{path}: missing checkpoint aliases {sorted(missing)}")
    return by_alias


def _cross_check(aa_rows: dict[str, dict[str, Any]], pgd_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The CE-PGD-20 numbers were produced twice, in two separate processes.

    Both evaluations are deterministic at a fixed checkpoint and evaluation seed,
    so they must agree exactly.  Any disagreement means one of them did not
    evaluate what it claimed, and aggregation stops.
    """
    for alias in ALIASES:
        a, p = aa_rows[alias], pgd_rows[alias]
        if a["checkpoint_sha256"] != p["checkpoint_sha256"]:
            raise AggregationError(f"{alias}: the two evaluations loaded different checkpoint files")
        for metric in ("clean_accuracy", "pgd_accuracy"):
            if float(a[metric]) != float(p[metric]):
                raise AggregationError(f"{alias}: {metric} disagrees between the two evaluation processes")
    return {
        "status": "agree_exactly",
        "metrics": ["clean_accuracy", "pgd_accuracy"],
        "pgd_only_run_id": "eval-ae74acc5a69223ea747e",
    }


def _pp(value: float) -> float:
    """Convert an accuracy fraction difference into percentage points."""
    return round(value * 100.0, 4)


def _decide(best_aa: float) -> tuple[str, float]:
    """Apply decision 0005 option A, fixed before any official number was seen.

    best AA at or above 48.0 %                  -> CONFIRMED
    best AA at least 2.0 pp above the defective
    run but below 48.0 %                        -> PARTIAL
    anything less                               -> NOT_SUPPORTED
    """
    delta = best_aa - DEFECTIVE["checkpoints"]["best"]["autoattack_accuracy"]
    if best_aa >= RULE_CONFIRM_AA:
        return "CONFIRMED", delta
    if delta >= RULE_PARTIAL_DELTA_AA:
        return "PARTIAL", delta
    return "NOT_SUPPORTED", delta


def aggregate(*, evaluation: Path, pgd_evaluation: Path, expected_source_sha: str) -> dict[str, Any]:
    _require_clean_source(expected_source_sha)
    evaluation, pgd_evaluation = evaluation.resolve(), pgd_evaluation.resolve()
    aa_bundle = _verify_bundle(evaluation)
    pgd_bundle = _verify_bundle(pgd_evaluation)
    aa_rows = _load_rows(evaluation, require_autoattack=True)
    pgd_rows = _load_rows(pgd_evaluation, require_autoattack=False)
    cross_check = _cross_check(aa_rows, pgd_rows)

    fixed = {
        alias: {
            "checkpoint_filename": row["checkpoint_filename"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "clean_accuracy": float(row["clean_accuracy"]),
            "pgd_accuracy": float(row["pgd_accuracy"]),
            "autoattack_accuracy": float(row["autoattack"]["autoattack_accuracy"]),
        }
        for alias, row in aa_rows.items()
    }
    deltas = {
        alias: {
            metric: _pp(fixed[alias][metric] - DEFECTIVE["checkpoints"][alias][metric])
            for metric in ("clean_accuracy", "pgd_accuracy", "autoattack_accuracy")
        }
        for alias in ALIASES
    }
    verdict, delta_best_aa = _decide(fixed["best"]["autoattack_accuracy"])
    best_aa = fixed["best"]["autoattack_accuracy"]
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "source_git_sha": expected_source_sha,
        "plan": "docs/plans/0027-controlled-teacherless-baselines.md",
        "milestone": "B5",
        "decision_packet": "docs/decisions/0005-trades-fix-official-evaluation.md",
        "endpoint_attack_identity_sha256": ENDPOINT_ATTACK_SHA256,
        "autoattack_upstream_commit": AUTOATTACK_COMMIT,
        "dataset_scope": {"name": "cifar10", "split": "test", "count": OFFICIAL_TEST_COUNT},
        "identity": {
            "dataset": "cifar10",
            "student": "saad_resnet18_cifar_v1",
            "method": "trades",
            "trades_beta": 6.0,
            "teacher": None,
            "protocol_id": PROTOCOL_ID,
            "training_seed": 0,
            "evaluation_seed": 0,
            "world_size": aa_bundle["execution"]["world_size"],
            "effective_global_batch_size": aa_bundle["execution"]["effective_global_batch_size"],
            "training_config_hash": TRAINING_CONFIG_HASH,
            "train_run_id": TRAIN_RUN_ID,
        },
        "lineage": {
            "autoattack_evaluation": aa_bundle,
            "pgd_only_evaluation": pgd_bundle,
            "cross_check": cross_check,
        },
        "fixed": fixed,
        "defective": DEFECTIVE,
        "deltas_pp": deltas,
        "reference": {
            "literature_autoattack_band": list(LITERATURE_AA_BAND),
            "literature_band_source": "docs/debugging/0028-trades-clean-target-detached.md",
            "same_engine_pgd_at_best_autoattack": PGD_AT_BEST_AA,
        },
        "decision": {
            "primary_endpoint": "best-checkpoint official-test AutoAttack accuracy of the fixed TRADES run",
            "preregistered_rule": {
                "confirm_at_or_above": RULE_CONFIRM_AA,
                "partial_delta_at_or_above": RULE_PARTIAL_DELTA_AA,
                "source": "docs/decisions/0005-trades-fix-official-evaluation.md option A",
            },
            "best_autoattack_accuracy": best_aa,
            "delta_versus_defective_pp": _pp(delta_best_aa),
            "shortfall_below_literature_band_pp": [
                _pp(LITERATURE_AA_BAND[0] - best_aa),
                _pp(LITERATURE_AA_BAND[1] - best_aa),
            ],
            "verdict": verdict,
            "follow_up_required": (
                "the rule's PARTIAL branch requires a new investigation of the residual gap; "
                "the detach defect is confirmed as real but is not the whole shortfall"
                if verdict == "PARTIAL"
                else "none declared by the preregistered rule"
            ),
            "stop": (
                "this contract measures one seed at one protocol; it authorizes no promotion, "
                "no additional seed, no method claim, and no change to any training setting"
            ),
        },
    }


def _accuracy_table(result: dict[str, Any]) -> str:
    lines = [
        "| checkpoint | run | clean | CE-PGD-20 | AutoAttack |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for alias in ALIASES:
        for label, block in (("defective", result["defective"]["checkpoints"]), ("fixed", result["fixed"])):
            row = block[alias]
            lines.append(
                f"| {alias} | {label} | {row['clean_accuracy'] * 100:.2f}% "
                f"| {row['pgd_accuracy'] * 100:.2f}% | {row['autoattack_accuracy'] * 100:.2f}% |"
            )
    return "\n".join(lines)


def _delta_table(result: dict[str, Any]) -> str:
    lines = ["| checkpoint | clean Δ | CE-PGD-20 Δ | AutoAttack Δ |", "| --- | ---: | ---: | ---: |"]
    for alias in ALIASES:
        d = result["deltas_pp"][alias]
        lines.append(
            f"| {alias} | {d['clean_accuracy']:+.2f} pp | {d['pgd_accuracy']:+.2f} pp "
            f"| {d['autoattack_accuracy']:+.2f} pp |"
        )
    return "\n".join(lines)


def _markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    fixed, defective = result["fixed"], result["defective"]["checkpoints"]
    band_lo, band_hi = result["reference"]["literature_autoattack_band"]
    short_lo, short_hi = decision["shortfall_below_literature_band_pp"]
    gap_fixed = _pp(fixed["best"]["autoattack_accuracy"] - fixed["last"]["autoattack_accuracy"])
    gap_defective = _pp(defective["best"]["autoattack_accuracy"] - defective["last"]["autoattack_accuracy"])
    aa = result["lineage"]["autoattack_evaluation"]
    pgd = result["lineage"]["pgd_only_evaluation"]
    return f"""# Controlled TRADES after the clean-target fix: official CIFAR-10 test

## Decision

The preregistered primary endpoint is the best-checkpoint AutoAttack accuracy of
the corrected TRADES run on the official {OFFICIAL_TEST_COUNT:,}-example CIFAR-10
test set.  "Best checkpoint" means the epoch chosen during training by internal
validation CE-PGD-20; the official test set was never consulted for that choice.

Verdict: **{decision["verdict"]}**.

The corrected run scores **{fixed["best"]["autoattack_accuracy"] * 100:.2f}%**
AutoAttack at its best checkpoint, {decision["delta_versus_defective_pp"]:+.2f} pp
above the defective run's {defective["best"]["autoattack_accuracy"] * 100:.2f}%,
and {short_lo:.2f} to {short_hi:.2f} pp below the published range of
{band_lo * 100:.1f}--{band_hi * 100:.1f}% for TRADES on this architecture.

The rule was fixed in
`{result["decision_packet"]}` before any official number was seen:

- best AutoAttack at or above {decision["preregistered_rule"]["confirm_at_or_above"] * 100:.1f}%
  confirms that the detach defect accounted for the shortfall;
- at least {decision["preregistered_rule"]["partial_delta_at_or_above"] * 100:.1f} pp
  above the defective run but below that line means the defect is real and is not
  the whole shortfall;
- less than that means the defect's claimed magnitude is not supported.

The measured value falls in the middle branch, by
{(decision["preregistered_rule"]["confirm_at_or_above"] - decision["best_autoattack_accuracy"]) * 100:.2f} pp.
The threshold is not moved after the fact.  {decision["follow_up_required"].capitalize()}.

{decision["stop"].capitalize()}.

## Accuracies

"Clean" is accuracy on unmodified test images.  "CE-PGD-20" is accuracy under a
20-step projected gradient descent attack on the cross-entropy loss at radius
8/255 in the L-infinity norm.  "AutoAttack" is the standard AutoAttack ensemble at
the same radius.  The three are separate quantities and are never combined.

{_accuracy_table(result)}

Fixed minus defective, in percentage points:

{_delta_table(result)}

## Robust overfitting

Best minus last AutoAttack accuracy measures how much robustness the run lost
between its selected epoch and the end of training.  The corrected run loses
{gap_fixed:.2f} pp; the defective run lost {gap_defective:.2f} pp.

## Internal consistency

The same engine's PGD-AT baseline scores
{result["reference"]["same_engine_pgd_at_best_autoattack"] * 100:.2f}% best-checkpoint
AutoAttack under the identical protocol.  The corrected TRADES now sits
{_pp(fixed["best"]["autoattack_accuracy"] - result["reference"]["same_engine_pgd_at_best_autoattack"]):+.2f} pp
against it.  This is recorded as an observation about two single runs, not as a
comparison between the two methods.

## Scope and lineage

One training seed, one run, one protocol.  This is a directional result for seed 0
under `{result["identity"]["protocol_id"]}` and is not a distributional estimate or
a claim about TRADES as a method.

Fixed identity: CIFAR-10, `{result["identity"]["student"]}`, TRADES beta
{result["identity"]["trades_beta"]:.0f}, no teacher, training seed
{result["identity"]["training_seed"]}, evaluation seed
{result["identity"]["evaluation_seed"]}, world size {result["identity"]["world_size"]},
effective global batch {result["identity"]["effective_global_batch_size"]}, training
run `{result["identity"]["train_run_id"]}`, training config hash
`{result["identity"]["training_config_hash"][:16]}...`.

Both evaluations read only saved weights in a separate process.  Each was checked
before use: the official test split with {OFFICIAL_TEST_COUNT:,} examples, the
registered CE-PGD-20 threat identity `{ENDPOINT_ATTACK_SHA256[:16]}...`, and
standard AutoAttack from the pinned upstream commit `{AUTOATTACK_COMMIT[:12]}...`.

| role | evaluation run | source SHA | manifest SHA-256 |
| --- | --- | --- | --- |
| clean + CE-PGD-20 + AutoAttack | `{aa["run_id"]}` | `{aa["source_git_sha"][:12]}...` | `{aa["manifest_sha256"][:16]}...` |
| clean + CE-PGD-20 only | `{pgd["run_id"]}` | `{pgd["source_git_sha"][:12]}...` | `{pgd["manifest_sha256"][:16]}...` |

The two evaluations ran as separate processes over the same two checkpoint files.
Their clean and CE-PGD-20 values agree exactly, which is the expected behaviour of
a deterministic evaluation at a fixed checkpoint and seed; a disagreement would
have stopped this aggregation.

The defective run's official values are carried as recorded constants.  No run
bundle for `{result["defective"]["run_id"]}` survives in the runtime tree, so this
contract cites them rather than re-deriving them.  Source:
{result["defective"]["provenance"]}.

## Provenance

Record: `docs/experiments/{RESULT_PATH.name}`, hash in
`docs/experiments/{RESULT_PATH.name}.sha256`.  Contract `{result["contract"]}`.
Aggregation source SHA `{result["source_git_sha"]}`.
Plan: `{result["plan"]}`, milestone {result["milestone"]}.
Decision packet: `{result["decision_packet"]}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="eval run dir with AutoAttack")
    parser.add_argument("--pgd-evaluation", type=Path, required=True, help="eval run dir without AutoAttack")
    parser.add_argument("--expected-source-sha", required=True, help="clean source SHA this aggregation runs from")
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    result_path, report_path = args.result.resolve(), args.report.resolve()
    _non_overwriting(result_path)
    _non_overwriting(report_path)
    result = aggregate(
        evaluation=args.evaluation,
        pgd_evaluation=args.pgd_evaluation,
        expected_source_sha=args.expected_source_sha,
    )
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
