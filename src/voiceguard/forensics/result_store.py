"""Bounded in-memory store of server-verified detection results.

A forensic report must be based on a verdict VoiceGuard actually produced, not on
a verdict supplied by the client. ``/detect`` records its result here keyed by the
audio SHA-256; ``/forensic/report`` looks it up by hash and refuses (404) if there
is no server-side record. The store is process-local, LRU-bounded, and lost on
restart (acceptable for the demo's short-lived report flow).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

MAX_ENTRIES = 10_000

_lock = threading.Lock()
_store: OrderedDict[str, dict[str, Any]] = OrderedDict()


def record(
    audio_hash: str,
    *,
    label: str,
    confidence: float,
    model: str,
    user: str,
    timestamp: float,
) -> None:
    """Store (or refresh) the server-verified detection result for an audio hash."""
    with _lock:
        _store[audio_hash] = {
            "label": label,
            "confidence": confidence,
            "model": model,
            "user": user,
            "timestamp": timestamp,
        }
        _store.move_to_end(audio_hash)
        while len(_store) > MAX_ENTRIES:
            _store.popitem(last=False)  # evict least-recently-used


def get(audio_hash: str) -> dict[str, Any] | None:
    """Return a copy of the stored result for *audio_hash*, or None if unknown."""
    with _lock:
        rec = _store.get(audio_hash)
        if rec is None:
            return None
        _store.move_to_end(audio_hash)
        return dict(rec)


def clear() -> None:
    """Drop all records (used by tests)."""
    with _lock:
        _store.clear()
