#!/usr/bin/env python3
"""Build a pre-resized, declared-derivative copy of an ImageNet ``train/`` tree.

Plan 0103 loader speedup C (human-approved 2026-09-30). The output mirrors the
source ``train/<wnid>/<file>`` layout exactly -- same relative paths, same file
names -- so ``ImageNetDataset``'s class order, labels, source IDs and the
seeded held-out validation split are identical to the source's. For each file:

* shorter side > ``--short-side``: decode, ``convert("RGB")``, resize so the
  shorter side is exactly ``S`` (longer side ``round_half_up(long * S / short)``)
  with PIL's LANCZOS filter (anti-aliased; a higher-fidelity filter than the
  loader's own bilinear, chosen because this resize is done once), re-encode
  as baseline JPEG at ``--quality`` with 4:2:0 chroma subsampling and no
  metadata. EXIF orientation is ignored, as the loader ignores it.
* otherwise: the source bytes are copied unchanged.

Only ``train/`` is built (``val/`` is never derived; the official evaluation
keeps the original root). ``class_names.json`` is copied as provenance.

``derived_manifest.json`` at the output root records the build parameters, the
source's and the output's ``content_sha256`` -- both computed by
``ard.data.datasets.ImageNetDataset``'s own manifest digest, i.e. exactly what
a config's ``dataset.content_sha256`` is checked against at load time -- and a
digest of every output file's SHA-256. It is deterministic (no timestamps);
its own SHA-256 is what ``dataset.derived_from.manifest_sha256`` pins. Wall
time and host details go to the separate, unpinned ``build_log.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import PIL  # noqa: E402
from PIL import Image, features  # noqa: E402

from ard.data.datasets import DERIVED_DATASET_MANIFEST, ImageNetDataset  # noqa: E402

MANIFEST_KIND = "ard-derived-imagenet-v1"
TRANSFORM = "resize_short_side"
RESAMPLE = "lanczos"
JPEG_SUBSAMPLING = "4:2:0"
_PIL_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def resized_dimensions(width: int, height: int, short_side: int) -> tuple[int, int] | None:
    """Target size with the shorter side == ``short_side``, or None to copy unchanged."""
    short, long = min(width, height), max(width, height)
    if short <= short_side:
        return None
    scaled_long = (long * short_side + short // 2) // short  # round half up, integer-exact
    return (short_side, scaled_long) if width <= height else (scaled_long, short_side)


def encode_resized(source: Path, short_side: int, quality: int) -> bytes | None:
    """Resized JPEG bytes for ``source``, or None when it must be copied unchanged."""
    with Image.open(source) as image:
        target = resized_dimensions(image.width, image.height, short_side)
        if target is None:
            return None
        rgb = image.convert("RGB")
    resized = rgb.resize(target, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=_PIL_SUBSAMPLING[JPEG_SUBSAMPLING],
        optimize=False,
        progressive=False,
    )
    return buffer.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


_JOB: dict[str, object] = {}


def _init_worker(source_train: str, out_train: str, short_side: int, quality: int, resume: bool) -> None:
    _JOB.update(source=Path(source_train), out=Path(out_train), short_side=short_side, quality=quality, resume=resume)


def _process(relative: str) -> tuple[str, str, str, int]:
    source = Path(str(_JOB["source"])) / relative
    target = Path(str(_JOB["out"])) / relative
    if _JOB["resume"] and target.is_file():
        data = target.read_bytes()
        action = "resized" if data != source.read_bytes() else "copied"
        return relative, hashlib.sha256(data).hexdigest(), action, len(data)
    encoded = encode_resized(source, int(str(_JOB["short_side"])), int(str(_JOB["quality"])))
    if encoded is None:
        data, action = source.read_bytes(), "copied"
    else:
        data, action = encoded, "resized"
    _write_atomic(target, data)
    return relative, hashlib.sha256(data).hexdigest(), action, len(data)


def files_digest(records: list[tuple[str, str]]) -> str:
    """SHA-256 over sorted ``relative_path<TAB>file_sha256`` lines."""
    digest = hashlib.sha256()
    for relative, file_sha256 in sorted(records):
        digest.update(f"{relative}\t{file_sha256}\n".encode())
    return digest.hexdigest()


def _git_sha() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return completed.stdout.strip() or None


def build(
    *,
    source: Path,
    out: Path,
    short_side: int,
    quality: int,
    source_content_sha256: str,
    workers: int,
    resume: bool = False,
    determinism_sample: int = 64,
    log: bool = True,
) -> dict[str, object]:
    started = time.monotonic()
    if short_side < 1 or not 1 <= quality <= 100:
        raise ValueError("short side must be positive and quality in [1, 100]")
    source_dataset = ImageNetDataset(source, "train", image_size=224)
    observed_source = source_dataset.content_identity["observed_sha256"]
    if observed_source != source_content_sha256:
        raise ValueError(
            f"source content_sha256 mismatch: expected {source_content_sha256}, observed {observed_source}"
        )
    out_train = out / "train"
    if out_train.exists() and any(out_train.iterdir()) and not resume:
        raise FileExistsError(f"output train/ already populated (pass --resume to reuse it): {out_train}")
    if resume and out_train.exists():
        for stale in out_train.glob("*/.*.tmp-*"):
            stale.unlink()
    source_train = source / "train"
    relatives = [path.relative_to(source_train).as_posix() for path, _ in source_dataset.samples]
    for wnid in sorted({relative.split("/", 1)[0] for relative in relatives}):
        (out_train / wnid).mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str, str, int]] = []
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(source_train), str(out_train), short_side, quality, resume),
    ) as pool:
        for index, record in enumerate(pool.imap_unordered(_process, relatives, chunksize=256), start=1):
            records.append(record)
            if log and index % 50_000 == 0:
                rate = index / (time.monotonic() - started)
                print(f"[build] {index}/{len(relatives)} files ({rate:.0f}/s)", file=sys.stderr, flush=True)
    records.sort()
    if (source / "class_names.json").is_file():
        shutil.copyfile(source / "class_names.json", out / "class_names.json")
    derived_dataset = ImageNetDataset(out, "train", image_size=224)
    if [path.relative_to(out_train).as_posix() for path, _ in derived_dataset.samples] != relatives:
        raise RuntimeError("derived tree does not list exactly the source's relative paths in the same order")
    if derived_dataset.targets != source_dataset.targets:
        raise RuntimeError("derived tree labels differ from the source's")
    # Determinism: re-encoding the first resized files must reproduce their bytes.
    written = {relative: file_sha256 for relative, file_sha256, _, _ in records}
    resized = [relative for relative, _, action, _ in records if action == "resized"][:determinism_sample]
    for relative in resized:
        again = encode_resized(source_train / relative, short_side, quality)
        if again is None or hashlib.sha256(again).hexdigest() != written[relative]:
            raise RuntimeError(f"non-deterministic re-encode: {relative}")
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "split": "train",
        "transform": TRANSFORM,
        "short_side": short_side,
        "jpeg_quality": quality,
        "resample": RESAMPLE,
        "jpeg_subsampling": JPEG_SUBSAMPLING,
        "output_mode": "RGB",
        "copy_rule": "source shorter side <= short_side: source bytes copied unchanged",
        "pil_version": PIL.__version__,
        "libjpeg_version": features.version("jpg"),
        "libjpeg_turbo": bool(features.check_feature("libjpeg_turbo")),
        "content_manifest_algorithm": "imagenet-manifest-v1",
        "source_content_sha256": observed_source,
        "derived_content_sha256": derived_dataset.content_identity["observed_sha256"],
        "file_count": len(records),
        "resized_count": sum(1 for record in records if record[2] == "resized"),
        "copied_count": sum(1 for record in records if record[2] == "copied"),
        "total_bytes": sum(record[3] for record in records),
        "files_sha256": files_digest([(relative, file_sha256) for relative, file_sha256, _, _ in records]),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_atomic(out / DERIVED_DATASET_MANIFEST, manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    build_log = {
        "manifest_sha256": manifest_sha256,
        "wall_seconds": round(time.monotonic() - started, 1),
        "workers": workers,
        "resume": resume,
        "determinism_checked_files": len(resized),
        "source_root": str(source),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "builder_git_sha": _git_sha(),
    }
    (out / "build_log.json").write_text(json.dumps(build_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, **build_log}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="original ImageNet root (containing train/)")
    parser.add_argument("--out", type=Path, required=True, help="derived root to create (train/ is written)")
    parser.add_argument("--short-side", type=int, required=True)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--source-content-sha256", required=True, help="the source train root's pinned dataset.content_sha256"
    )
    parser.add_argument("--resume", action="store_true", help="reuse files already present under --out/train")
    args = parser.parse_args(argv)
    result = build(
        source=args.source,
        out=args.out,
        short_side=args.short_side,
        quality=args.quality,
        source_content_sha256=args.source_content_sha256,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
