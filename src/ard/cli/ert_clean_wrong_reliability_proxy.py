"""Replay KL-PGD10 Teacher reliability and aggregate CE20 safety effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ert_clean_wrong_reliability_proxy import build_proxy_report, replay_kl_features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "report"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run", choices=("L2", "L4"))
    parser.add_argument("--endpoint-root", type=Path)
    parser.add_argument("--ce-l2", type=Path)
    parser.add_argument("--ce-l4", type=Path)
    parser.add_argument("--kl-l2", type=Path)
    parser.add_argument("--kl-l4", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "replay":
        required = (args.config, args.checkpoint, args.mask, args.output, args.run)
        if any(item is None for item in required):
            parser.error("replay requires --config --checkpoint --mask --output --run")
        result = replay_kl_features(
            config_path=args.config,
            checkpoint=args.checkpoint,
            mask_path=args.mask,
            output_dir=args.output,
            device=torch.device(args.device),
        )
    else:
        required = (
            args.endpoint_root,
            args.ce_l2,
            args.ce_l4,
            args.kl_l2,
            args.kl_l4,
            args.output_json,
            args.output_markdown,
        )
        if any(item is None for item in required):
            parser.error("report requires endpoint and CE/KL feature roots plus output paths")
        result = build_proxy_report(
            endpoint_root=args.endpoint_root,
            ce_feature_roots={"L2": args.ce_l2, "L4": args.ce_l4},
            kl_feature_roots={"L2": args.kl_l2, "L4": args.kl_l4},
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    print(
        json.dumps({"contract": result.get("contract"), "source_sha256": result.get("source_sha256")}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
