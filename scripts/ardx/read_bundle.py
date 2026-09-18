#!/usr/bin/env python3
"""Read-only dump of a hand-run's run-bundle: manifest, completion, files, epoch metrics.

Exists because `/experiment-postrun` runs in a sandbox that only lets the shell's
file commands (`ls`, `cat`, `find`) touch the repo checkout, while run bundles live
under the runtime root. A Python subprocess can still read them, so this script is
the sanctioned way to look at a finished bundle from a postrun session.

Read-only by construction: it opens files and prints. It never writes, launches or
retries anything.

    python3 scripts/ardx/read_bundle.py <run-bundle/manifest.json> [--metrics-cols clean,pgd]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"__parse_error__": str(exc)}


def _sha256(path: Path, limit_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return digest.hexdigest()


def _epoch_rows(bundle_dir: Path) -> list[dict[str, Any]]:
    # written next to the checkpoints in the run dir; older runs also kept a copy in the bundle.
    for jsonl in (bundle_dir.parent / "epoch-metrics.jsonl", bundle_dir / "epoch-metrics.jsonl"):
        if jsonl.exists():
            break
    else:
        return []
    rows = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="path to run-bundle/manifest.json")
    parser.add_argument(
        "--metrics-cols",
        default="",
        help="comma-separated substrings; only epoch-metric keys containing one are printed",
    )
    parser.add_argument("--epochs", default="", help="comma-separated epoch numbers to print (default: all)")
    parser.add_argument("--checkpoint-sha", action="store_true", help="also hash *.pt files (slow)")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    bundle_dir = manifest_path.parent
    run_dir = bundle_dir.parent

    print("== manifest ==")
    print(json.dumps(_load_json(manifest_path), indent=2, sort_keys=True))

    completion = _load_json(bundle_dir / "completion.json")
    if completion is not None:
        print("\n== completion.json ==")
        print(json.dumps(completion, indent=2, sort_keys=True))

    print("\n== files (run dir, depth 2) ==")
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(run_dir)
        except ValueError:
            continue
        if len(rel.parts) > 3:
            continue
        line = f"{path.stat().st_size:>13,}  {rel}"
        if args.checkpoint_sha and path.suffix == ".pt":
            line += f"  sha256={_sha256(path)}"
        print(line)

    rows = _epoch_rows(bundle_dir)
    print(f"\n== epoch-metrics.jsonl ({len(rows)} rows) ==")
    if rows:
        wanted_cols = [c.strip() for c in args.metrics_cols.split(",") if c.strip()]
        wanted_epochs = {int(e) for e in args.epochs.split(",") if e.strip()}
        for row in rows:
            epoch = row.get("epoch")
            if wanted_epochs and epoch not in wanted_epochs:
                continue
            if wanted_cols:
                shown = {k: v for k, v in row.items() if any(c in k for c in wanted_cols) or k == "epoch"}
            else:
                shown = row
            print(json.dumps(shown, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
