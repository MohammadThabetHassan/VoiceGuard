"""
Checkpoint snapshot manager — prevents training data loss from corrupt writes.

Every call to save_snapshot() copies a .pt file to a timestamped snapshot
directory. Up to MAX_SNAPSHOTS are kept per model key; oldest are pruned.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

MAX_SNAPSHOTS = 10
SNAPSHOTS_DIR = Path.home() / "voiceguard-checkpoints" / "snapshots"


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:8]


def save_snapshot(
    src: Path,
    model_key: str,
    snapshots_dir: Path = SNAPSHOTS_DIR,
    max_keep: int = MAX_SNAPSHOTS,
) -> Path:
    """Copy src to snapshots_dir/<model_key>/<timestamp>_<sha256[:8]>.pt.

    Prunes oldest snapshots beyond max_keep. Returns snapshot path.
    Raises ValueError if src does not exist.
    """
    src = Path(src)
    if not src.exists():
        raise ValueError(f"Checkpoint not found: {src}")

    dest_dir = snapshots_dir / model_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = _sha256_short(src)
    dest = dest_dir / f"{ts}_{sha}.pt"
    shutil.copy2(src, dest)

    # Prune oldest beyond max_keep
    existing = sorted(dest_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    for old in existing[:-max_keep]:
        old.unlink(missing_ok=True)

    return dest


def list_snapshots(
    model_key: str,
    snapshots_dir: Path = SNAPSHOTS_DIR,
) -> list[dict]:
    """Return snapshot metadata dicts sorted newest-first."""
    snap_dir = snapshots_dir / model_key
    if not snap_dir.exists():
        return []
    snaps = []
    for p in sorted(snap_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = p.stat().st_size / 1_000_000
        snaps.append({"path": str(p), "name": p.name, "size_mb": round(size_mb, 2)})
    return snaps


def restore_snapshot(snapshot_path: Path, dest_path: Path) -> None:
    """Copy snapshot back to dest_path and verify SHA-256 matches filename hint."""
    snapshot_path = Path(snapshot_path)
    dest_path = Path(dest_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)
    shutil.copy2(snapshot_path, dest_path)
    actual = _sha256_short(dest_path)
    expected = snapshot_path.stem.split("_")[-1]
    if actual != expected:
        dest_path.unlink(missing_ok=True)
        raise ValueError(f"Snapshot integrity check failed: expected {expected}, got {actual}")
