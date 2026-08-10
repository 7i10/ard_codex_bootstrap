"""Write the CPU-only frozen ERT online-routing proxy diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_online_routing_proxy import ERTOnlineRoutingProxyError, run_proxy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                name: str(path)
                for name, path in sorted(
                    run_proxy(config_path=args.config.resolve(), output_dir=args.output_dir.resolve()).items()
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ERTOnlineRoutingProxyError as exc:
        raise SystemExit(str(exc)) from exc
