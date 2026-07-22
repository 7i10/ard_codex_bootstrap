"""Process-wide GPU locks for marker-selected tests.

Pytest's ``gpu`` marker communicates hardware ownership to the verification
gate.  A lock is held for every visible physical device so two independent
gate processes cannot concurrently use the same CUDA device.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO


def visible_gpu_identities() -> tuple[str, ...]:
    """Return stable-ish identities for all devices visible to this process."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        import torch

        if not torch.cuda.is_available():
            return tuple(part.strip() for part in (configured or "").split(",") if part.strip())
        identities: list[str] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            identity = getattr(properties, "uuid", None)
            identities.append(str(identity or f"{properties.name}:{index}"))
        return tuple(identities)
    except ImportError:
        # A marker command with an unavailable torch installation must still
        # serialize any fallback GPU launcher by its explicit visibility.
        return tuple(part.strip() for part in (configured or "").split(",") if part.strip())


class GPULock(AbstractContextManager["GPULock"]):
    """Advisory exclusive locks, released even when a test fails."""

    def __init__(self, *, lock_dir: Path = Path("/tmp")) -> None:
        self.lock_dir = lock_dir
        self._handles: list[IO[str]] = []

    def __enter__(self) -> GPULock:
        identities = visible_gpu_identities()
        for identity in sorted(identities):
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            path = self.lock_dir / f"ard-test-gpu-{digest}.lock"
            handle = path.open("a+", encoding="utf-8")
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows has no CUDA CI here
                pass
            self._handles.append(handle)
        return self

    def __exit__(self, *_: object) -> None:
        for handle in reversed(self._handles):
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - Windows has no CUDA CI here
                pass
            handle.close()
        self._handles.clear()
