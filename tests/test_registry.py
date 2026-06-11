"""Tests for the model registry checkpoint discovery (P0-4) and PDPL cleanup (P0-8)."""

from __future__ import annotations

import pytest

from voiceguard.api.middleware import PDPLTimingMiddleware, pdpl_auto_delete
from voiceguard.models import registry as reg_mod


def test_newest_pt_is_ext_aware(tmp_path, monkeypatch):
    """The classical detector is a .pkl; discovery must glob by the entry's ext."""
    monkeypatch.setattr(reg_mod, "_CHECKPOINTS_ROOT", tmp_path)
    d = tmp_path / "classical"
    d.mkdir()
    ckpt = d / "model_best.pkl"
    ckpt.write_bytes(b"x")

    assert reg_mod._newest_pt("classical", ".pkl") == ckpt
    # A .pt glob in the same dir must not match the .pkl
    assert reg_mod._newest_pt("classical", ".pt") is None


def test_resolve_path_classical_finds_pkl(tmp_path, monkeypatch):
    monkeypatch.setattr(reg_mod, "_CHECKPOINTS_ROOT", tmp_path)
    monkeypatch.delenv("CLASSICAL_MODEL_PATH", raising=False)
    d = tmp_path / "classical"
    d.mkdir()
    ckpt = d / "model_best.pkl"
    ckpt.write_bytes(b"x")

    r = reg_mod.ModelRegistry()
    assert r._resolve_path("classical") == ckpt


@pytest.mark.asyncio
async def test_pdpl_auto_delete_clears_registry(tmp_path):
    """After auto-delete runs, the timing registry entry must be gone (no leak)."""
    p = tmp_path / "upload.wav"
    p.write_bytes(b"x")
    PDPLTimingMiddleware.register(str(p))
    assert PDPLTimingMiddleware.age_seconds(str(p)) is not None

    await pdpl_auto_delete(str(p), delay=0)

    assert not p.exists()
    assert PDPLTimingMiddleware.age_seconds(str(p)) is None
