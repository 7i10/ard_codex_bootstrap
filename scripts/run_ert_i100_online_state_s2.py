#!/usr/bin/env python3
"""Public runtime for the I100 Online-State S2×T1 preservation screen.

The three subcommands intentionally expose the preregistered DAG boundary:
``prefix`` creates one treatment-free e100 state per seed, ``freeze`` derives
its immutable scalar thresholds, and ``arm`` forks exactly one e101–114 child.
No command offers a threshold, coefficient, horizon, or seed override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from ard.analysis.ert_i100_online_state_s2 import OnlineStateS2RoutingError, freeze_online_thresholds
from ard.analysis.ert_stage_a_runtime import SAMPLE_KEYED_KL10_ATTACK_IDENTITY_SHA256, StageATreatment, run_stage_a_arm
from ard.cli.ert_stage_a_runtime import _load_calibration_artifact
from ard.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    os.environ.get(
        "ARD_STAGEWISE_RUN_ROOT",
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1",
    )
).expanduser()
CALIBRATION = ROOT / "docs/experiments/ert_rslad_i100_s2_dynamic_bdd_calibration_v1.json"
CALIBRATION_SHA256 = "37bf0a0e1aa6ff12951f1c05f59f6df55700be0e28291c6925670d7b6cb56840"
TEACHER_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
ENDPOINT_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
PARENTS = {
    "dev-1": {
        "config": RUN_ROOT / "idbh-s100-s1/resolved_config.yaml",
        # Historical filename, but checkpoint payload is the exact e99 end state.
        "checkpoint": RUN_ROOT / "seed1/s100/epoch-100.pt",
        "sha256": "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    },
    "dev-2": {
        "config": RUN_ROOT / "idbh-s100-s2/resolved_config.yaml",
        # Historical filename, but checkpoint payload is the exact e99 end state.
        "checkpoint": RUN_ROOT / "seed2/s100/epoch-100.pt",
        "sha256": "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
    },
}
ARMS = {
    "control": {"name": "I100_CONTROL", "mode": None, "coefficient": None},
    "pmp": {"name": "OS_PMP", "mode": "pair_margin", "coefficient": 0.05380932585058825},
    "dbdp": {
        "name": "OS_DBDP",
        "mode": "detached_boundary_distance",
        "coefficient": 31.649566509850324,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_source(expected: str | None) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status:
        raise SystemExit("online-state runtime requires a clean scientific source tree")
    if expected is not None and actual != expected:
        raise SystemExit(f"source SHA mismatch: expected {expected}, got {actual}")
    return actual


def _contracts(seed: str) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    parent = PARENTS[seed]
    config_path, checkpoint = Path(parent["config"]), Path(parent["checkpoint"])
    if not config_path.is_file() or not checkpoint.is_file():
        raise SystemExit(f"registered input is missing for {seed}: config={config_path}, checkpoint={checkpoint}")
    if _sha256(checkpoint) != parent["sha256"]:
        raise SystemExit(f"registered e99 parent SHA mismatch for {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 99 or payload.get("epoch_boundary") != "end":
        raise SystemExit(f"registered e99 parent is not an exact epoch-99 end-boundary checkpoint: {seed}")
    config = load_config(config_path)
    if config.tracking.artifact_retention != "metrics_only":
        raise SystemExit("online screen requires the parent metrics-only W&B artifact-retention policy")
    teacher_path = Path(str(config.teacher.checkpoint)) if config.teacher is not None else None
    if teacher_path is None or not teacher_path.is_file() or _sha256(teacher_path) != TEACHER_SHA256:
        raise SystemExit(f"frozen Teacher SHA mismatch for {seed}")
    keyed_attack = config.method.attack.model_copy(update={"random_start_keying": "sample_keyed_v1"})
    training = keyed_attack.identity()
    endpoint = None if config.method.selection_attack is None else config.method.selection_attack.identity()
    if keyed_attack.identity_sha256() != SAMPLE_KEYED_KL10_ATTACK_IDENTITY_SHA256:
        raise SystemExit("registered sample-keyed KL-PGD10 attack identity mismatch")
    if endpoint is None or config.method.selection_attack.identity_sha256() != ENDPOINT_SHA256:
        raise SystemExit("registered CE-PGD20 endpoint attack identity mismatch")
    return config_path, checkpoint, training, endpoint


def _online_spec(
    *,
    seed: str,
    arm: str,
    training_attack: dict[str, object],
    endpoint_attack: dict[str, object],
    thresholds_path: Path | None,
    prefix_state_materialized_path: Path | None = None,
) -> dict[str, object]:
    parent = PARENTS[seed]
    spec: dict[str, object] = {
        "arm": arm,
        "prefix_epoch": 100,
        "original_parent_checkpoint": str(Path(parent["checkpoint"]).resolve()),
        "original_parent_checkpoint_sha256": parent["sha256"],
        "training_attack": training_attack,
        "endpoint_attack": endpoint_attack,
    }
    if thresholds_path is not None:
        spec["thresholds_path"] = str(thresholds_path.resolve())
    if prefix_state_materialized_path is not None:
        spec["prefix_state_materialized_path"] = str(prefix_state_materialized_path.resolve())
    return spec


def _calibration() -> dict[str, Any]:
    if not CALIBRATION.is_file() or _sha256(CALIBRATION) != CALIBRATION_SHA256:
        raise SystemExit("online screen requires the exact registered calibration artifact bytes")
    value = _load_calibration_artifact(CALIBRATION, require_hash_sidecar=True)
    if value.get("contract") != "ert_rslad_i100_s2_dynamic_bdd_calibration_v1":
        raise SystemExit("online screen requires the registered non-secant calibration artifact")
    coefficients = value.get("coefficients")
    if not isinstance(coefficients, dict):
        raise SystemExit("registered calibration has no coefficient map")
    for arm in ("pmp", "dbdp"):
        mode = ARMS[arm]["mode"]
        if coefficients.get(mode) != ARMS[arm]["coefficient"]:
            raise SystemExit(f"registered coefficient changed for {arm}")
    if value.get("artifact_sha256") not in {None, CALIBRATION_SHA256}:
        raise SystemExit("calibration self-reported artifact SHA differs from registered bytes")
    value = dict(value)
    value["artifact_sha256"] = CALIBRATION_SHA256
    return value


def _run_prefix(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs != 101:
        raise SystemExit("the registered shared prefix is exactly e100; --epochs must be the exclusive bound 101")
    source_sha = _require_source(args.expected_source_sha)
    config_path, checkpoint, training, endpoint = _contracts(args.seed)
    result = run_stage_a_arm(
        parent_config_path=config_path,
        parent_checkpoint=checkpoint,
        mask_path=None,
        output_dir=args.output / "training",
        treatment=StageATreatment(arm="I100_ONLINE_PREFIX", mask_key=None, kind="baseline", online_state_s2=True),
        calibration=_calibration(),
        device=torch.device(args.device),
        end_epoch=args.epochs,
        horizon_epochs=(100,),
        run_namespace=args.run_namespace,
        expected_parent_checkpoint_sha256=str(PARENTS[args.seed]["sha256"]),
        online_state_s2=_online_spec(
            seed=args.seed,
            arm="prefix",
            training_attack=training,
            endpoint_attack=endpoint,
            thresholds_path=None,
        ),
        resume_epoch=99,
        force_sample_keyed_attack=True,
        canary_max_train_batches=getattr(args, "canary_max_train_batches", None),
        canary_max_validation_batches=getattr(args, "canary_max_validation_batches", None),
    )
    payload = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_online_state_s2_prefix_v1",
        "seed": args.seed,
        "source_git_sha": source_sha,
        "result": result,
    }
    _write_json(args.output / "prefix-summary.json", payload)
    return payload


def _run_freeze(args: argparse.Namespace) -> dict[str, Any]:
    source_sha = _require_source(args.expected_source_sha)
    config_path, _, training, _ = _contracts(args.seed)
    checkpoint = args.prefix_output / "training" / "checkpoints" / "epoch-100.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"shared e100 prefix checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 100 or payload.get("epoch_boundary") != "end":
        raise SystemExit("threshold freeze requires an exact e100 end-boundary prefix checkpoint")
    lineage = payload.get("fork_lineage")
    if not isinstance(lineage, dict) or not isinstance(lineage.get("online_state_s2_state"), dict):
        raise SystemExit("shared e100 prefix checkpoint lacks online state capture")
    if lineage.get("source_git_sha") != source_sha:
        raise SystemExit("threshold freeze rejects an e100 prefix from a different source Git SHA")
    if lineage.get("training_attack_identity_sha256") != SAMPLE_KEYED_KL10_ATTACK_IDENTITY_SHA256:
        raise SystemExit("threshold freeze rejects an e100 prefix with a different training attack identity")
    result = freeze_online_thresholds(
        prefix_router_state=lineage["online_state_s2_state"],
        prefix_checkpoint=checkpoint,
        output_path=args.output,
        source_git_sha=source_sha,
        training_attack_identity_sha256=SAMPLE_KEYED_KL10_ATTACK_IDENTITY_SHA256,
    )
    _write_json(
        args.output.with_name(args.output.name + ".summary.json"),
        {
            "schema_version": 1,
            "contract": "ert_rslad_i100_online_state_s2_threshold_freeze_v1",
            "seed": args.seed,
            "config_path": str(config_path.resolve()),
            "threshold_artifact": str(args.output.resolve()),
            "threshold_artifact_sha256": result["artifact_sha256"],
            "thresholds": result["thresholds"],
        },
    )
    return result


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
    expected_epochs = 102 if args.canary_one_epoch else 115
    if args.epochs != expected_epochs:
        raise SystemExit(
            "the registered online-state child horizon is fixed; "
            f"--epochs must be the exclusive bound {expected_epochs}"
        )
    source_sha = _require_source(args.expected_source_sha)
    config_path, _, training, endpoint = _contracts(args.seed)
    if not args.prefix_checkpoint.is_file() or not args.thresholds.is_file():
        raise SystemExit("online child requires existing shared e100 prefix and frozen threshold artifact")
    calibration = _calibration()
    definition = ARMS[args.arm]
    if args.arm == "control":
        treatment = StageATreatment(arm=str(definition["name"]), mask_key=None, kind="baseline", online_state_s2=True)
    else:
        treatment = StageATreatment(
            arm=str(definition["name"]),
            mask_key=None,
            kind="broad",
            boundary_intervention=str(definition["mode"]),
            boundary_coefficient=float(definition["coefficient"]),
            boundary_epsilon=1e-12,
            online_state_s2=True,
        )
    if args.canary_one_epoch and args.run_namespace != "ert-i100-online-state-s2-v1-canary":
        raise SystemExit("one-epoch online canary is reserved for the registered public canary namespace")
    end_epoch = args.epochs
    horizons = (101,) if args.canary_one_epoch else (104, 109, 114)
    result = run_stage_a_arm(
        parent_config_path=config_path,
        parent_checkpoint=args.prefix_checkpoint,
        mask_path=None,
        output_dir=args.output / "training",
        treatment=treatment,
        calibration=calibration,
        device=torch.device(args.device),
        end_epoch=end_epoch,
        horizon_epochs=horizons,
        run_namespace=args.run_namespace,
        online_state_s2=_online_spec(
            seed=args.seed,
            arm=args.arm,
            training_attack=training,
            endpoint_attack=endpoint,
            thresholds_path=args.thresholds,
            prefix_state_materialized_path=args.prefix_state,
        ),
        resume_epoch=100,
        force_sample_keyed_attack=True,
        canary_max_train_batches=getattr(args, "canary_max_train_batches", None),
        canary_max_validation_batches=getattr(args, "canary_max_validation_batches", None),
        canary_source_ids=getattr(args, "canary_source_ids", None),
    )
    payload = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_online_state_s2_arm_v1",
        "seed": args.seed,
        "arm": args.arm,
        "source_git_sha": source_sha,
        "result": result,
    }
    _write_json(args.output / "arm-summary.json", payload)
    return payload


def _run_canary(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise the real prefix -> freeze -> child runtime once per branch.

    This is deliberately a bounded **public runtime** canary, not a direct
    formula probe: it executes full-batch loss reduction, one optimizer update
    epoch, checkpoint creation, and router checkpoint lineage through the
    same functions that production arms use.  Its only shortened dimension is
    the explicitly registered child horizon e101 (rather than e101–e114).
    """

    source_sha = _require_source(args.expected_source_sha)
    output = args.output.resolve()
    if output.exists() and any(path.name != "orchestration" for path in output.iterdir()):
        raise SystemExit(f"refusing to overwrite online-state public canary output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    namespace = "ert-i100-online-state-s2-v1-canary"
    prefix = _run_prefix(
        argparse.Namespace(
            expected_source_sha=source_sha,
            seed=args.seed,
            output=output / "prefix",
            device=args.device,
            run_namespace=namespace,
            canary_max_train_batches=4,
            canary_max_validation_batches=1,
            epochs=101,
        )
    )
    prefix_root = output / "prefix"
    thresholds = output / "thresholds" / "frozen-thresholds.json"
    frozen = _run_freeze(
        argparse.Namespace(
            expected_source_sha=source_sha,
            seed=args.seed,
            prefix_output=prefix_root,
            output=thresholds,
        )
    )
    checkpoint = prefix_root / "training/checkpoints/epoch-100.pt"
    prefix_canary = prefix.get("result", {}).get("canary")
    if not isinstance(prefix_canary, dict) or not isinstance(prefix_canary.get("source_ids"), list):
        raise SystemExit("public canary prefix did not persist its bounded stable-ID universe")
    source_ids = tuple(int(value) for value in prefix_canary["source_ids"])
    branches: dict[str, Any] = {}
    for arm in ARMS:
        arm_output = output / "arms" / arm
        branch = _run_arm(
            argparse.Namespace(
                expected_source_sha=source_sha,
                seed=args.seed,
                arm=arm,
                prefix_checkpoint=checkpoint,
                thresholds=thresholds,
                prefix_state=None,
                output=arm_output,
                device=args.device,
                run_namespace=namespace,
                canary_one_epoch=True,
                canary_max_train_batches=4,
                canary_max_validation_batches=1,
                canary_source_ids=source_ids,
                epochs=102,
            )
        )
        horizon = arm_output / "training/checkpoints/epoch-101.pt"
        if not horizon.is_file():
            raise SystemExit(f"public canary did not materialize e101 checkpoint for {arm}")
        payload = torch.load(horizon, map_location="cpu", weights_only=False)
        lineage = payload.get("fork_lineage") if isinstance(payload, dict) else None
        state = None if not isinstance(lineage, dict) else lineage.get("online_state_s2_state")
        if not isinstance(state, dict) or state.get("arm") != arm:
            raise SystemExit(f"public canary e101 checkpoint lacks resumable online state for {arm}")
        if arm != "control":
            statistics = state.get("epoch_statistics", {}).get("101", {})
            if not isinstance(statistics, dict) or float(statistics.get("active_treatment_count", 0.0)) <= 0.0:
                raise SystemExit(f"public canary did not exercise Online-S2×T1 action for {arm}")
            if float(statistics.get("boundary_active_count", 0.0)) <= 0.0:
                raise SystemExit(f"public canary did not exercise pair-gated boundary loss for {arm}")
        branches[arm] = {
            "result": branch,
            "checkpoint": str(horizon),
            "checkpoint_sha256": _sha256(horizon),
            "router_state_epoch": sorted(state.get("epoch_statistics", {})) if isinstance(state, dict) else [],
        }
    result = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_online_state_s2_public_runtime_canary_v1",
        "status": "pass",
        "seed": args.seed,
        "source_git_sha": source_sha,
        "scope": "full e100 prefix plus e101 public-runtime child update per arm; not a scientific endpoint",
        "prefix": prefix,
        "thresholds": frozen,
        "branches": branches,
    }
    _write_json(output / "canary-summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha")
    sub = parser.add_subparsers(dest="stage", required=True)
    prefix = sub.add_parser("prefix", help="run the one treatment-free shared e100 prefix")
    prefix.add_argument("--seed", choices=tuple(PARENTS), required=True)
    prefix.add_argument("--output", type=Path, required=True)
    prefix.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    prefix.add_argument("--run-namespace", default="ert-i100-online-state-s2-v1")
    prefix.add_argument(
        "--epochs",
        type=int,
        default=101,
        help="registered exclusive runtime bound; only 101 is accepted for the e100 shared prefix",
    )
    freeze = sub.add_parser("freeze", help="freeze e100 q10 thresholds from one prefix")
    freeze.add_argument("--seed", choices=tuple(PARENTS), required=True)
    freeze.add_argument("--prefix-output", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    arm = sub.add_parser("arm", help="fork one e101–114 online-state arm")
    arm.add_argument("--seed", choices=tuple(PARENTS), required=True)
    arm.add_argument("--arm", choices=tuple(ARMS), required=True)
    arm.add_argument("--prefix-checkpoint", type=Path, required=True)
    arm.add_argument("--thresholds", type=Path, required=True)
    arm.add_argument(
        "--prefix-state",
        type=Path,
        help="optional SHA-verified local materialization of the e100 state table for an external child",
    )
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    arm.add_argument("--run-namespace", default="ert-i100-online-state-s2-v1")
    arm.add_argument("--canary-one-epoch", action="store_true")
    arm.add_argument(
        "--epochs",
        type=int,
        default=115,
        help="registered exclusive runtime bound; only 115 is accepted outside the bounded canary",
    )
    canary = sub.add_parser("canary", help="run the registered full-public-runtime e100/e101 canary")
    canary.add_argument("--seed", choices=tuple(PARENTS), required=True)
    canary.add_argument("--output", type=Path, required=True)
    canary.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "prefix":
            result = _run_prefix(args)
        elif args.stage == "freeze":
            result = _run_freeze(args)
        elif args.stage == "arm":
            result = _run_arm(args)
        else:
            result = _run_canary(args)
    except OnlineStateS2RoutingError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
