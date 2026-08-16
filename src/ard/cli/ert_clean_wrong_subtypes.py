"""Replay baseline features and aggregate Clean-Wrong rescue subtypes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_clean_wrong_subtypes import build_report, replay_features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "report"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run", choices=("L2", "L4"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--l2-features", type=Path)
    parser.add_argument("--l4-features", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "replay":
        required = (args.config, args.checkpoint, args.mask, args.output, args.run)
        if any(item is None for item in required):
            parser.error("replay requires --config --checkpoint --mask --output --run")
        result = replay_features(
            config_path=args.config,
            checkpoint=args.checkpoint,
            mask_path=args.mask,
            output_dir=args.output,
            device=torch.device(args.device),
        )
    else:
        required = (args.root, args.l2_features, args.l4_features, args.output_json, args.output_markdown)
        if any(item is None for item in required):
            parser.error("report requires --root --l2-features --l4-features --output-json --output-markdown")
        result = build_report(
            root=args.root,
            feature_roots={"L2": args.l2_features, "L4": args.l4_features},
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    print(
        json.dumps(
            {"source_sha256": result.get("source_sha256"), "contract": result.get("contract")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
