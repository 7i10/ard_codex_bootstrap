"""Write the fixed ERT Clean-Wrong margin-screen report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_cw_margin_generalization_report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--mask-l2", type=Path, required=True)
    parser.add_argument("--mask-l4", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--train-ce-feature-root",
        type=Path,
        default=Path(".cache/analysis/ert-clean-wrong-subtypes-v4"),
    )
    parser.add_argument(
        "--train-kl-feature-root",
        type=Path,
        default=Path(".cache/analysis/ert-clean-wrong-reliability-proxy-v1"),
    )
    parser.add_argument(
        "--validation-ce-feature-root",
        type=Path,
        default=Path(".cache/analysis/ert-cw-generalization-v1"),
    )
    parser.add_argument(
        "--validation-kl-feature-root",
        type=Path,
        default=Path(".cache/analysis/ert-cw-generalization-v1"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(
        endpoint_root=args.endpoint_root,
        training_root=args.training_root,
        mask_paths={"L2": args.mask_l2, "L4": args.mask_l4},
        calibration=args.calibration,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        allow_dirty=args.allow_dirty,
        train_ce_feature_root=args.train_ce_feature_root,
        train_kl_feature_root=args.train_kl_feature_root,
        validation_ce_feature_root=args.validation_ce_feature_root,
        validation_kl_feature_root=args.validation_kl_feature_root,
    )
    print(
        json.dumps(
            {
                "output_json_sha256": result["output_json_sha256"],
                "output_markdown_sha256": result["output_markdown_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
