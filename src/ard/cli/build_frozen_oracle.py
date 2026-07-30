"""Build one frozen oracle mask and three deterministic random controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from ard.analysis.frozen_oracle import (
    FrozenOracleError,
    build_frozen_oracle_manifests,
    builder_git_identity,
    replay_robust_correctness,
    train_labels,
    validate_wandb_checkpoint_inventory,
    write_frozen_oracle_manifests,
)
from ard.config import load_config
from ard.engine import config_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--wandb-checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Explicit CUDA replay device, for example cuda:0.")
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument(
        "--attack-seed-base",
        required=True,
        type=int,
        help="One frozen random-start base used identically for both source checkpoints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw_source_config = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FrozenOracleError("source resolved config YAML is unreadable") from exc
    if not isinstance(raw_source_config, dict):
        raise FrozenOracleError("source resolved config YAML must be a mapping")
    # This is deliberately the historical serialized mapping, before the
    # current Pydantic schema can inject newly introduced default fields.
    source_config_hash = config_digest(raw_source_config)
    source = load_config(args.source_config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise FrozenOracleError("frozen-oracle construction requires an available explicit CUDA device")
    builder_git = builder_git_identity()
    labels = train_labels(source)
    historical_replay = replay_robust_correctness(
        source_config=source,
        source_config_hash=source_config_hash,
        checkpoint=args.historical_checkpoint,
        device=device,
        batch_size=args.batch_size,
        attack_seed_base=args.attack_seed_base,
    )
    final_replay = replay_robust_correctness(
        source_config=source,
        source_config_hash=source_config_hash,
        checkpoint=args.final_checkpoint,
        device=device,
        batch_size=args.batch_size,
        attack_seed_base=args.attack_seed_base,
    )
    inventory_payload = yaml.safe_load(args.wandb_checkpoint_inventory.read_text(encoding="utf-8"))
    if not isinstance(inventory_payload, dict):
        raise FrozenOracleError("W&B checkpoint inventory must be a YAML mapping")
    source_manifest_payload = json.loads(args.source_run_manifest.read_text(encoding="utf-8"))
    if not isinstance(source_manifest_payload, dict):
        raise FrozenOracleError("source run manifest must be a JSON mapping")
    source_git = source_manifest_payload.get("git")
    if not isinstance(source_git, dict):
        raise FrozenOracleError("source run manifest has no Git identity")
    source_scientific_git_sha = source_git.get("sha")
    if not isinstance(source_scientific_git_sha, str):
        raise FrozenOracleError("source run manifest has no scientific Git SHA")
    wandb_inventory = validate_wandb_checkpoint_inventory(
        inventory_payload,
        historical_checkpoint=args.historical_checkpoint,
        final_checkpoint=args.final_checkpoint,
        source_config_hash=source_config_hash,
        source_run_id=historical_replay["tracker_run_id"],
        source_scientific_git_sha=source_scientific_git_sha,
    )
    manifests = build_frozen_oracle_manifests(
        source_config=source,
        source_manifest=args.source_run_manifest,
        source_config_hash=source_config_hash,
        historical_replay=historical_replay,
        final_replay=final_replay,
        labels=labels,
        builder_git=builder_git,
        wandb_checkpoint_inventory=wandb_inventory,
    )
    hashes = write_frozen_oracle_manifests(args.output_dir, manifests)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "sha256": hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except FrozenOracleError as exc:
        raise SystemExit(str(exc)) from exc
