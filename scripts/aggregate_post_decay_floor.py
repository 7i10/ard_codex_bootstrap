"""Aggregate the plan 0092 post-decay floor calibration.

Six untreated control replicates -- three per parent -- forked from a shared
epoch-100 no-action prefix and run to epoch 114.  They differ only in
`continuation_seed`.  Every difference between them is instrument noise, so the
spread of their held-out CE-PGD20 endpoint accuracy *is* the floor.

The script emits one result dict and renders both the JSON record and the
Markdown report from it, so the two cannot drift.

Lineage is verified rather than assumed.  In particular each endpoint output is
required to name the exact checkpoint SHA-256 that the corresponding training
arm recorded for that horizon; without that link an endpoint sweep could have
scored some other replicate and the floor would be meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "docs/experiments/ard_post_decay_floor_v1.json"
REPORT_PATH = ROOT / "docs/POST_DECAY_FLOOR.md"

CONTRACT = "ard_post_decay_floor_v1"
ARM_CONTRACT = "ert_rslad_i100_online_state_s2_arm_v1"
ENDPOINT_SUMMARY_CONTRACT = "ert_rslad_i100_online_state_s2_endpoint_v1"
ENDPOINT_CONTRACT = "ert_stage_a_common_ce_pgd20_endpoint_v1"
ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
VALIDATION_SPLIT_SHA256 = "16ec66fbcdeae0b70261589b1ba5f1e7fd4128743ce0194eabc5bea53a0cc6c4"
TEACHER_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
PARENT_SHA256 = {
    "dev-1": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    "dev-2": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
SEEDS = ("dev-1", "dev-2")
REPLICATES = (1, 2, 3)
HORIZONS = (104, 109, 114)
VALIDATION_ROWS = 5_000
ALPHA = 0.05
POWER = 0.80


class AggregationError(RuntimeError):
    """A lineage, contract or coverage requirement failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregator_source_sha() -> str:
    """The SHA of the tree this script is being run from.

    This is NOT the SHA the campaign ran at, and conflating the two makes a
    record impossible to reproduce from a clean checkout the moment any later
    commit lands.  The campaign's own SHA is read from the runs themselves.
    """
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise AggregationError("the working tree must be clean so the record's aggregator SHA is meaningful")
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _campaign_source_sha(campaign: Path, *, declared: str | None) -> str:
    """The one source SHA every replicate ran at, read from the replicates."""
    observed: dict[str, str] = {}
    for seed in SEEDS:
        for replicate in REPLICATES:
            summary = _read_json(campaign / "arms" / seed / f"rep{replicate}" / "arm-summary.json")
            sha = str(summary.get("source_git_sha") or "")
            if len(sha) != 40:
                raise AggregationError(f"{seed} rep{replicate} does not record a source SHA")
            observed[f"{seed}/rep{replicate}"] = sha
    distinct = set(observed.values())
    if len(distinct) != 1:
        raise AggregationError(f"replicates did not all run at one source SHA: {observed}")
    sha = distinct.pop()
    if declared is not None and declared != sha:
        raise AggregationError(f"campaign ran at {sha[:12]}, not the declared {declared[:12]}")
    return sha


def _non_overwriting(path: Path) -> None:
    # A committed result is never edited in place; a correction is a new file.
    if path.exists():
        raise AggregationError(f"refusing to overwrite an existing record: {path}")


def _robust_accuracy(rows_path: Path, expected_sha256: str) -> tuple[float, float]:
    if sha256(rows_path) != expected_sha256:
        raise AggregationError(f"endpoint rows hash differs from the declared value: {rows_path}")
    rows = pq.read_table(rows_path).to_pylist()
    by_id = {int(row["sample_id"]): row for row in rows}
    if len(rows) != VALIDATION_ROWS or len(by_id) != VALIDATION_ROWS:
        raise AggregationError(f"endpoint rows do not cover exactly {VALIDATION_ROWS} stable IDs: {rows_path}")
    if any(int(row["true_label"]) < 0 for row in rows):
        raise AggregationError(f"endpoint rows lack valid class labels: {rows_path}")
    robust = sum(bool(row["robust_correct"]) for row in rows) / len(rows)
    clean = sum(bool(row["clean_correct"]) for row in rows) / len(rows)
    return robust, clean


def _prefix(campaign: Path, *, seed: str, source_sha: str) -> dict[str, Any]:
    """The shared epoch-100 no-action prefix, and its link to the registered parent.

    The replicates do not fork the epoch-99 parent directly; they fork the
    epoch-100 prefix that was grown from it.  Checking only one of those two
    links would leave the other unverified, so both are checked here: the prefix
    descends from the registered parent, and every replicate descends from this
    prefix.
    """
    summary = _read_json(campaign / "prefix" / seed / "prefix-summary.json")
    result = summary.get("result", {})
    if result.get("parent_checkpoint_sha256") != PARENT_SHA256[seed]:
        raise AggregationError(
            f"{seed}: the prefix was grown from {str(result.get('parent_checkpoint_sha256'))[:12]}, "
            f"not the registered parent {PARENT_SHA256[seed][:12]}"
        )
    if summary.get("source_git_sha") != source_sha:
        raise AggregationError(f"{seed}: the prefix ran at a different source SHA than the replicates")
    epoch100 = str(result.get("last_checkpoint_sha256") or "")
    if len(epoch100) != 64:
        raise AggregationError(f"{seed}: the prefix does not record its epoch-100 checkpoint")
    return {
        "seed": seed,
        "registered_parent_sha256": PARENT_SHA256[seed],
        "epoch100_checkpoint_sha256": epoch100,
        "online_state_sha256": ((result.get("online_state_s2") or {}).get("state") or {}).get("sha256"),
    }


def _replicate(campaign: Path, *, seed: str, replicate: int, source_sha: str, epoch100_sha256: str) -> dict[str, Any]:
    arm_root = campaign / "arms" / seed / f"rep{replicate}"
    arm = _read_json(arm_root / "arm-summary.json")
    result = arm.get("result", {})
    if arm.get("contract") != ARM_CONTRACT or arm.get("arm") != "control":
        raise AggregationError(f"arm summary identity differs: {arm_root}")
    if result.get("arm") != "I100_CONTROL":
        raise AggregationError(f"{seed} rep{replicate} is not an untreated control")
    if result.get("continuation_seed") != replicate:
        raise AggregationError(
            f"{seed} rep{replicate} records continuation_seed {result.get('continuation_seed')!r}; "
            "replicates that do not differ in the post-fork random stream are not replicates"
        )
    if result.get("parent_checkpoint_sha256") != epoch100_sha256:
        raise AggregationError(
            f"{seed} rep{replicate} forked {str(result.get('parent_checkpoint_sha256'))[:12]}, "
            f"not the shared epoch-100 prefix {epoch100_sha256[:12]}"
        )
    if arm.get("source_git_sha") != source_sha:
        raise AggregationError(f"{seed} rep{replicate} ran at a different source SHA than the rest")
    if result.get("dynamic_s3") is not None or result.get("dynamic_s3_epoch80") is not None:
        raise AggregationError(f"{seed} rep{replicate} carries a treatment payload")

    horizon_checkpoints = result.get("horizon_checkpoints") or {}
    if {int(key) for key in horizon_checkpoints} != set(HORIZONS):
        raise AggregationError(f"{seed} rep{replicate} lacks exactly the e104/e109/e114 horizons")

    endpoint_root = campaign / "endpoints" / seed / f"rep{replicate}"
    summary = _read_json(endpoint_root / "summary.json")
    if (
        summary.get("contract") != ENDPOINT_SUMMARY_CONTRACT
        or summary.get("seed") != seed
        or summary.get("arm") != "control"
        or summary.get("source_git_sha") != source_sha
    ):
        raise AggregationError(f"endpoint summary identity differs: {endpoint_root}")
    outputs = summary.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(HORIZONS):
        raise AggregationError(f"endpoint summary lacks exactly three horizons: {endpoint_root}")

    horizons: dict[int, dict[str, Any]] = {}
    for descriptor in outputs:
        epoch = descriptor.get("checkpoint_epoch")
        if (
            not isinstance(epoch, int)
            or epoch not in HORIZONS
            or descriptor.get("contract") != ENDPOINT_CONTRACT
            or descriptor.get("dataset_scope") != "validation"
            or descriptor.get("attack_identity_sha256") != ENDPOINT_ATTACK_SHA256
            or descriptor.get("split_identity", {}).get("sample_id_label_sha256") != VALIDATION_SPLIT_SHA256
        ):
            raise AggregationError(f"endpoint contract differs at {endpoint_root} e{epoch}")
        # The binding that makes the whole measurement meaningful: this endpoint
        # sweep scored THIS replicate's checkpoint and not some other one.
        trained = horizon_checkpoints[str(epoch)]["sha256"]
        if descriptor.get("checkpoint_sha256") != trained:
            raise AggregationError(
                f"{seed} rep{replicate} e{epoch}: the endpoint scored checkpoint "
                f"{str(descriptor.get('checkpoint_sha256'))[:12]} but training produced {trained[:12]}"
            )
        rows_path = endpoint_root / f"e{epoch}-validation" / "endpoint-sample-stats.parquet"
        robust, clean = _robust_accuracy(rows_path, str(descriptor["rows_sha256"]))
        declared = descriptor.get("robust_accuracy")
        if declared is not None and not math.isclose(float(declared), robust, abs_tol=1e-9):
            raise AggregationError(
                f"{seed} rep{replicate} e{epoch}: recomputed robust accuracy {robust} "
                f"differs from the declared {declared}"
            )
        horizons[epoch] = {
            "robust_accuracy": robust,
            "clean_accuracy": clean,
            "checkpoint_sha256": trained,
            "rows_sha256": str(descriptor["rows_sha256"]),
        }

    return {
        "seed": seed,
        "replicate": replicate,
        "continuation_seed": replicate,
        "config_hash": result.get("config_hash"),
        "parent_checkpoint_sha256": epoch100_sha256,
        "registered_e99_parent_sha256": PARENT_SHA256[seed],
        "prefix_state_sha256": (result.get("online_state_s2") or {}).get("prefix_state", {}).get("sha256"),
        "threshold_artifact_sha256": (result.get("online_state_s2") or {})
        .get("prefix_state", {})
        .get("threshold_artifact_sha256"),
        "horizons": horizons,
    }


def _pairwise(values: Mapping[int, float]) -> list[dict[str, Any]]:
    pairs = []
    keys = sorted(values)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            difference = values[left] - values[right]
            pairs.append(
                {
                    "left_replicate": left,
                    "right_replicate": right,
                    "signed_difference_pp": difference * 100.0,
                    "absolute_difference_pp": abs(difference) * 100.0,
                }
            )
    return pairs


def _mde(sigma_d_pp: float, blocks: int, *, df: int) -> float:
    """Minimum detectable effect for a paired two-arm design with k blocks.

    Two-sided alpha = 0.05, power = 0.80.

    The degrees of freedom are the calibration's, not the screen's.  The whole
    purpose of measuring the floor is that a future screen does not have to
    estimate sigma from its own handful of blocks: it uses this number.  Sizing
    a screen with t on k-1 degrees of freedom would be answering a different
    question -- what a screen that ignored this calibration could detect -- and
    at k = 2 that inflates the requirement fivefold.

    But the calibration's own four degrees of freedom are not free either.
    `docs/MEASUREMENT_DESIGN.md` sizes screens with sqrt(7.85 / k), which is the
    normal approximation, i.e. sigma treated as exactly known.  The values here
    are larger than that convention by the ratio 3.717 / 2.80 = 1.33, and that
    factor is the price of having estimated the floor from six runs.
    """
    t_alpha = float(stats.t.ppf(1.0 - ALPHA / 2.0, df))
    t_beta = float(stats.t.ppf(POWER, df))
    return (t_alpha + t_beta) * sigma_d_pp / math.sqrt(blocks)


def _horizon_statistics(replicates: Iterable[Mapping[str, Any]], *, epoch: int) -> dict[str, Any]:
    by_seed: dict[str, dict[int, float]] = {seed: {} for seed in SEEDS}
    for replicate in replicates:
        by_seed[replicate["seed"]][int(replicate["replicate"])] = replicate["horizons"][epoch]["robust_accuracy"]

    per_seed: dict[str, Any] = {}
    within_variances: list[float] = []
    all_absolute: list[float] = []
    for seed in SEEDS:
        values = by_seed[seed]
        if set(values) != set(REPLICATES):
            raise AggregationError(f"e{epoch} {seed}: expected three replicates, found {sorted(values)}")
        accuracies_pp = {key: value * 100.0 for key, value in values.items()}
        pairs = _pairwise(values)
        all_absolute.extend(pair["absolute_difference_pp"] for pair in pairs)
        variance = statistics.variance(accuracies_pp.values())  # 2 degrees of freedom
        within_variances.append(variance)
        per_seed[seed] = {
            "robust_accuracy_pp": accuracies_pp,
            "mean_pp": statistics.fmean(accuracies_pp.values()),
            "replicate_sd_pp": math.sqrt(variance),
            "pairs": pairs,
            "range_pp": max(accuracies_pp.values()) - min(accuracies_pp.values()),
        }

    # Preregistered statistic: the six pairwise absolute differences.
    preregistered = {
        "absolute_differences_pp": sorted(all_absolute),
        "count": len(all_absolute),
        "mean_pp": statistics.fmean(all_absolute),
        "sd_pp": statistics.stdev(all_absolute),
        "max_pp": max(all_absolute),
    }

    # Pooled estimator.  The six pairwise differences are not independent -- three
    # values yield only two degrees of freedom per seed -- so the SD of the six
    # absolute differences understates the uncertainty and is not sigma_d itself.
    # Pooling the within-seed variances gives 4 df, and the difference of two
    # independent runs has variance 2 * sigma_within^2.
    pooled_variance = statistics.fmean(within_variances)
    sigma_within = math.sqrt(pooled_variance)
    sigma_d = sigma_within * math.sqrt(2.0)
    df = 2 * (len(REPLICATES) - 1)
    lower = math.sqrt(df * pooled_variance / float(stats.chi2.ppf(1.0 - ALPHA / 2.0, df))) * math.sqrt(2.0)
    upper = math.sqrt(df * pooled_variance / float(stats.chi2.ppf(ALPHA / 2.0, df))) * math.sqrt(2.0)

    return {
        "epoch": epoch,
        "per_seed": per_seed,
        "preregistered_pairwise": preregistered,
        "pooled": {
            "sigma_within_pp": sigma_within,
            "sigma_d_pp": sigma_d,
            "degrees_of_freedom": df,
            "sigma_d_ci95_pp": [lower, upper],
        },
        "minimum_detectable_effect_pp": {str(k): _mde(sigma_d, k, df=df) for k in (2, 5, 10)},
        "minimum_detectable_effect_normal_approximation_pp": {
            str(k): sigma_d * math.sqrt(7.85 / k) for k in (2, 5, 10)
        },
    }


def aggregate(*, campaign: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    aggregator_sha = _aggregator_source_sha()
    source_sha = _campaign_source_sha(campaign, declared=expected_source_sha)
    prefixes = {seed: _prefix(campaign, seed=seed, source_sha=source_sha) for seed in SEEDS}
    replicates = [
        _replicate(
            campaign,
            seed=seed,
            replicate=replicate,
            source_sha=source_sha,
            epoch100_sha256=prefixes[seed]["epoch100_checkpoint_sha256"],
        )
        for seed in SEEDS
        for replicate in REPLICATES
    ]

    # All six must share one prefix per seed and one threshold artifact per seed,
    # or they are not forks of a common parent state.
    for seed in SEEDS:
        group = [item for item in replicates if item["seed"] == seed]
        for key in ("prefix_state_sha256", "threshold_artifact_sha256"):
            values = {item[key] for item in group}
            if len(values) != 1 or None in values:
                raise AggregationError(f"{seed}: replicates do not share one {key}: {values}")
        if len({item["continuation_seed"] for item in group}) != len(REPLICATES):
            raise AggregationError(f"{seed}: replicates do not have distinct continuation seeds")

    horizons = {epoch: _horizon_statistics(replicates, epoch=epoch) for epoch in HORIZONS}

    return {
        "contract": CONTRACT,
        "schema_version": 1,
        "campaign_source_git_sha": source_sha,
        "aggregator_source_git_sha": aggregator_sha,
        "campaign_root": str(campaign.resolve()),
        "design": {
            "description": "three untreated control replicates per parent, shared epoch-100 no-action prefix, "
            "epochs 101-114, differing only in continuation_seed",
            "seeds": list(SEEDS),
            "replicates_per_seed": len(REPLICATES),
            "horizons": list(HORIZONS),
            "endpoint": "held-out CE-PGD20 on the 5000-example validation split",
            "endpoint_attack_sha256": ENDPOINT_ATTACK_SHA256,
            "validation_split_sha256": VALIDATION_SPLIT_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "parent_checkpoint_sha256": dict(PARENT_SHA256),
        },
        "prefixes": prefixes,
        "replicates": replicates,
        "horizon_statistics": {str(epoch): value for epoch, value in horizons.items()},
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Post-decay noise floor")
    add("")
    add(
        f"Record: `{RESULT_PATH.relative_to(ROOT)}` (contract `{result['contract']}`). "
        f"The runs were produced at source `{result['campaign_source_git_sha'][:12]}`; "
        f"this report was rendered at `{result['aggregator_source_git_sha'][:12]}`."
    )
    add("")
    add("## What this measures")
    add("")
    add(
        "Six untreated control runs were forked from two epoch-99 parents, three from each, sharing a common "
        "epoch-100 no-action prefix and differing only in the random stream after the fork. They were trained "
        "to epoch 114 and scored on the held-out 5000-example validation split with the registered CE-PGD20 "
        "endpoint attack. No treatment was applied to any of them, so every difference between two of them is "
        "instrument noise. That spread is the floor: an effect smaller than it cannot be distinguished from "
        "nothing, no matter how the run is labelled."
    )
    add("")
    add("There is no pass or fail here. Only the number matters.")
    add("")
    for epoch in HORIZONS:
        stats_at = result["horizon_statistics"][str(epoch)]
        add(f"## Epoch {epoch}")
        add("")
        add("| parent | rep 1 | rep 2 | rep 3 | spread | replicate SD |")
        add("| --- | ---: | ---: | ---: | ---: | ---: |")
        for seed in SEEDS:
            per_seed = stats_at["per_seed"][seed]
            accuracies = per_seed["robust_accuracy_pp"]
            add(
                f"| {seed} | {_fmt(accuracies[1])} | {_fmt(accuracies[2])} | {_fmt(accuracies[3])} "
                f"| {_fmt(per_seed['range_pp'])} pp | {_fmt(per_seed['replicate_sd_pp'])} pp |"
            )
        add("")
        pre = stats_at["preregistered_pairwise"]
        pooled = stats_at["pooled"]
        add(
            f"The six pairwise absolute differences are "
            f"{', '.join(_fmt(value) for value in pre['absolute_differences_pp'])} pp: "
            f"mean {_fmt(pre['mean_pp'])} pp, largest {_fmt(pre['max_pp'])} pp."
        )
        add("")
        add(
            f"Pooled across both parents, one run differs from an identical one by a standard deviation of "
            f"**{_fmt(pooled['sigma_d_pp'])} pp** "
            f"(95% interval {_fmt(pooled['sigma_d_ci95_pp'][0])} to {_fmt(pooled['sigma_d_ci95_pp'][1])} pp, "
            f"{pooled['degrees_of_freedom']} degrees of freedom)."
        )
        add("")
        mde = stats_at["minimum_detectable_effect_pp"]
        normal = stats_at["minimum_detectable_effect_normal_approximation_pp"]
        add("| paired blocks | smallest detectable effect | under the existing sqrt(7.85/k) convention |")
        add("| ---: | ---: | ---: |")
        for blocks in ("2", "5", "10"):
            add(f"| {blocks} | {_fmt(mde[blocks])} pp | {_fmt(normal[blocks])} pp |")
        add("")
    add("## What this floor does and does not cover")
    add("")
    add(
        "These replicates share a parent, share an epoch-100 prefix, share a data order, and share a campaign. "
        "They differ in the random stream after the fork and in nothing else. That is exactly the structure of a "
        "screen that compares a treated arm against a control inside one campaign, so the number applies there "
        "directly."
    )
    add("")
    add(
        "It does not license comparisons across campaigns, but not for the reason usually given. The evidence "
        "cited for a cross-campaign penalty -- two nominally identical controls differing by 0.94 and 1.78 pp -- "
        "is measured at epoch 84, sixteen epochs BEFORE the learning-rate decay, and it sits inside the "
        "independently measured pre-decay floor of 1.14 to 1.25 pp. It is the pre-decay floor, not an extra term "
        "on top of it. Comparing it against the number on this page compares two different regimes."
    )
    add("")
    add(
        "Whether a cross-campaign penalty exists *after* the decay is simply untested: no two campaigns have ever "
        "run untreated controls from the same parent past epoch 100. Until one does, a post-decay comparison "
        "across campaigns has no measured floor at all, and the safe course is still not to make one."
    )
    add("")
    add(
        "It is also a floor for this design only: two parents, fourteen epochs past the decay, the registered "
        "CE-PGD20 endpoint on the validation split. Nothing here transfers to a different horizon, a different "
        "endpoint, or the official test split."
    )
    add("")
    add("## How to read the two spread numbers")
    add("")
    add(
        "The plan preregistered the six pairwise absolute differences and their standard deviation, and that is "
        "reported above unchanged. But three runs give only two degrees of freedom per parent, and the six "
        "differences drawn from them are not independent of each other, so the standard deviation of those six "
        "numbers is not the quantity a future screen needs. The pooled figure is: it combines the two "
        "within-parent variances into four degrees of freedom and converts to the difference of two runs. Both "
        "are reported so that neither the preregistration nor the correct estimator is hidden."
    )
    add("")
    add(
        "The interval on that figure is wide because four degrees of freedom is very little. It is stated rather "
        "than rounded away: the floor is now measured instead of guessed, but it is not measured precisely."
    )
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha",
        default=None,
        help="optional: fail unless every replicate ran at this SHA (it is read from the runs regardless)",
    )
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--print-only", action="store_true", help="render to stdout without writing the record or the report"
    )
    args = parser.parse_args()

    result = aggregate(campaign=args.campaign, expected_source_sha=args.expected_source_sha)
    report = _markdown(result)

    if args.print_only:
        print(report)
        return 0

    _non_overwriting(args.result)
    _non_overwriting(args.report)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.result.with_suffix(args.result.suffix + ".sha256").write_text(sha256(args.result) + "\n", encoding="utf-8")
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {args.result} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
