"""Tests for C2PA spectral watermarking."""

from __future__ import annotations

import numpy as np

from voiceguard.watermark.c2pa_watermark import (
    MIN_INAUDIBLE_CARRIER_HZ,
    _clamp_carrier,
    detect,
    embed,
    ensure_carrier_sr,
)

SR = 22050


def make_audio(duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, int(sr * duration_s)).astype(np.float32)


def test_embed_returns_audio_and_id():
    audio = make_audio()
    watermarked, wid = embed(audio, sr=SR)
    assert watermarked.shape == audio.shape
    assert isinstance(wid, str)
    assert len(wid) > 0


def test_embed_preserves_shape():
    audio = make_audio(2.0)
    watermarked, _ = embed(audio, sr=SR)
    assert watermarked.shape == audio.shape


def test_embed_clips_to_valid_range():
    audio = make_audio()
    watermarked, _ = embed(audio, sr=SR)
    assert watermarked.min() >= -1.0
    assert watermarked.max() <= 1.0


def test_embed_deterministic_with_same_id():
    audio = make_audio()
    w1, _ = embed(audio.copy(), sr=SR, watermark_id="fixed-id")
    w2, _ = embed(audio.copy(), sr=SR, watermark_id="fixed-id")
    np.testing.assert_array_equal(w1, w2)


def test_embed_different_ids_differ():
    audio = make_audio()
    w1, _ = embed(audio.copy(), sr=SR, watermark_id="id-a")
    w2, _ = embed(audio.copy(), sr=SR, watermark_id="id-b")
    assert not np.allclose(w1, w2)


def test_detect_watermark_present():
    audio = make_audio()
    watermarked, wid = embed(audio, sr=SR, amplitude=0.1)
    detected, corr = detect(watermarked, sr=SR, watermark_id=wid, threshold=0.01)
    assert detected is True
    assert corr > 0.0


def test_detect_wrong_id_not_detected():
    audio = make_audio()
    watermarked, _ = embed(audio, sr=SR, watermark_id="correct-id", amplitude=0.1)
    detected, corr = detect(watermarked, sr=SR, watermark_id="wrong-id", threshold=0.1)
    assert detected is False


def test_detect_returns_float_correlation():
    audio = make_audio()
    watermarked, wid = embed(audio, sr=SR)
    _, corr = detect(watermarked, sr=SR, watermark_id=wid)
    assert isinstance(corr, float)


def test_embed_auto_id_unique():
    audio = make_audio()
    _, wid1 = embed(audio.copy())
    _, wid2 = embed(audio.copy())
    assert wid1 != wid2


def test_carrier_clamped_at_nyquist():
    audio = make_audio()
    # carrier_hz > sr/2 should not crash, just clamp
    watermarked, wid = embed(audio, sr=SR, carrier_hz=SR * 10)
    assert watermarked.shape == audio.shape


def test_ensure_carrier_sr_keeps_carrier_inaudible_on_24k():
    """The Kokoro bug: 18kHz carrier on 24kHz audio clamps to ~11.9kHz (audible).
    ensure_carrier_sr must lift the rate so the carrier stays >= 16kHz, and the
    watermark must still be detectable after the resample."""
    # Quiet signal reflecting real synthesized speech (Kokoro RMS ~0.05), where an
    # inaudible 0.003 watermark is still recoverable.
    rng = np.random.default_rng(0)
    audio = (rng.uniform(-0.08, 0.08, 24000)).astype(np.float32)
    # On 24kHz the carrier would be audible:
    assert _clamp_carrier(18000.0, 24000) < MIN_INAUDIBLE_CARRIER_HZ
    wm_audio, wm_sr = ensure_carrier_sr(audio, 24000)
    assert wm_sr >= 36000
    assert _clamp_carrier(18000.0, wm_sr) >= MIN_INAUDIBLE_CARRIER_HZ
    marked, wid = embed(wm_audio, sr=wm_sr, amplitude=0.003)
    detected, corr = detect(marked, sr=wm_sr, watermark_id=wid, threshold=0.01)
    assert detected is True
    assert corr > 0.0


def test_ensure_carrier_sr_noop_when_rate_sufficient():
    audio = make_audio(0.5, sr=48000)
    out, sr = ensure_carrier_sr(audio, 48000)
    assert sr == 48000
    assert out.shape == audio.shape
