"""Tests for the true C2PA provenance signing layer.

Skipped automatically when the optional ``c2pa`` runtime is not installed, so CI
stays green on minimal environments.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voiceguard.watermark import c2pa_sign

pytestmark = pytest.mark.skipif(not c2pa_sign.is_available(), reason="c2pa runtime not installed")


def _write_wav(path, sr: int = 22050, dur: float = 1.0) -> None:
    t = np.arange(int(sr * dur)) / sr
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


def test_sign_and_verify_roundtrip(tmp_path, monkeypatch):
    # isolate the demo credential to a temp dir (auto-generated)
    monkeypatch.setenv("VG_C2PA_DIR", str(tmp_path / "c2pa"))
    monkeypatch.delenv("VG_C2PA_CERT", raising=False)
    monkeypatch.delenv("VG_C2PA_KEY", raising=False)

    src, dst = tmp_path / "a.wav", tmp_path / "a_signed.wav"
    _write_wav(src)

    result = c2pa_sign.sign_file(str(src), str(dst), software_agent="VoiceGuard/kokoro")
    assert result["signed"] is True
    assert dst.exists()

    info = c2pa_sign.verify_file(str(dst))
    assert info["has_manifest"] is True
    assert info["validation_state"] == "Valid"
    assert info["ai_generated"] is True
    assert info["software_agent"] == "VoiceGuard/kokoro"


def test_verify_unsigned_has_no_manifest(tmp_path):
    src = tmp_path / "plain.wav"
    _write_wav(src)
    info = c2pa_sign.verify_file(str(src))
    assert info["has_manifest"] is False


def test_generated_credential_is_persistent(tmp_path, monkeypatch):
    monkeypatch.setenv("VG_C2PA_DIR", str(tmp_path / "c2pa"))
    monkeypatch.delenv("VG_C2PA_CERT", raising=False)
    monkeypatch.delenv("VG_C2PA_KEY", raising=False)
    cert1, key1 = c2pa_sign._ensure_credentials()
    cert2, key2 = c2pa_sign._ensure_credentials()
    assert cert1 == cert2 and key1 == key2
    assert cert1.exists() and key1.exists()
