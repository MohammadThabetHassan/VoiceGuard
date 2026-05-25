"""PDPL cleanup middleware and shared rate limiter."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from slowapi import Limiter
from slowapi.util import get_remote_address

# Global rate limiter — 60 requests per minute per IP
limiter = Limiter(key_func=get_remote_address)

PDPL_MAX_AGE_SECONDS = int(os.environ.get("PDPL_MAX_AGE_SECONDS", "60"))


async def pdpl_auto_delete(path: str | Path, delay: float = PDPL_MAX_AGE_SECONDS) -> None:
    """Delete a file after `delay` seconds (PDPL §12 auto-erasure).

    Designed to be awaited as a background task:
        background_tasks.add_task(pdpl_auto_delete, tmp_path)
    """
    await asyncio.sleep(delay)
    try:
        os.unlink(path)
    except OSError:
        pass


def make_temp_audio_file(suffix: str = ".wav") -> tuple[int, str]:
    """Create a temporary audio file that will be auto-deleted."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="vg_upload_")
    return fd, path


class PDPLTimingMiddleware:
    """Records the upload time so we can enforce PDPL max-age in audit logs."""

    _registry: dict[str, float] = {}

    @classmethod
    def register(cls, file_path: str) -> None:
        cls._registry[file_path] = time.time()

    @classmethod
    def age_seconds(cls, file_path: str) -> float | None:
        ts = cls._registry.get(file_path)
        if ts is None:
            return None
        return time.time() - ts

    @classmethod
    def unregister(cls, file_path: str) -> None:
        cls._registry.pop(file_path, None)
