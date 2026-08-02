"""Export read-only revised-H5 online state anchors on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.history_online_state import HistoryOnlineStateError, export_online_anchors, write_online_anchors


def _checkpoint(value: str) -> tuple[int, Path]:
    epoch, separator, path = value.partition("=")
    if separator != "=" or not epoch or not path:
        raise argparse.ArgumentTypeError("expected EPOCH=PATH")
    try:
        return int(epoch), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint epoch must be integer") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, action="append", type=_checkpoint, metavar="EPOCH=PATH")
    parser.add_argument("--replay-lineage", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise HistoryOnlineStateError("duplicate online anchor checkpoint epoch")
    export = export_online_anchors(
        checkpoints={e: p.resolve() for e, p in checkpoints.items()},
        replay_lineage=args.replay_lineage.resolve(),
        expected_count=args.expected_count,
    )
    paths = write_online_anchors(output_dir=args.output_dir.resolve(), export=export)
    print(json.dumps({k: str(v) for k, v in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HistoryOnlineStateError as exc:
        raise SystemExit(str(exc)) from exc
