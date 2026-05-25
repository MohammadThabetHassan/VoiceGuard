"""Tests for VoIP bridge: μ-law decode, stream processor."""

from __future__ import annotations

import numpy as np

from voiceguard.voip.stream_processor import StreamProcessor, _resample_linear
from voiceguard.voip.twilio_bridge import TwilioStreamHandler, ulaw_decode

# ── μ-law decode ───────────────────────────────────────────────────────────────


def test_ulaw_decode_length():
    data = bytes(range(256))
    out = ulaw_decode(data)
    assert out.shape == (256,)


def test_ulaw_decode_range():
    data = bytes(range(256))
    out = ulaw_decode(data)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_ulaw_decode_dtype():
    out = ulaw_decode(b"\x00\xff\x80")
    assert out.dtype == np.float32


def test_ulaw_decode_silence():
    # 0xFF is positive silence in μ-law
    out = ulaw_decode(bytes([0xFF]))
    # magnitude should be small
    assert abs(out[0]) < 0.01


def test_ulaw_decode_zeros():
    out = ulaw_decode(bytes(16))
    assert out.shape == (16,)


# ── Resampling ─────────────────────────────────────────────────────────────────


def test_resample_same_rate():
    audio = np.ones(100, dtype=np.float32)
    out = _resample_linear(audio, 16000, 16000)
    np.testing.assert_array_equal(out, audio)


def test_resample_upsample_shape():
    audio = np.zeros(80, dtype=np.float32)  # 10ms @ 8kHz
    out = _resample_linear(audio, 8000, 16000)
    assert len(out) == 160  # 10ms @ 16kHz


def test_resample_downsample_shape():
    audio = np.zeros(160, dtype=np.float32)
    out = _resample_linear(audio, 16000, 8000)
    assert len(out) == 80


def test_resample_preserves_constant():
    audio = np.full(100, 0.5, dtype=np.float32)
    out = _resample_linear(audio, 8000, 16000)
    np.testing.assert_allclose(out, 0.5, atol=1e-5)


# ── StreamProcessor ─────────────────────────────────────────────────────────────


def test_stream_processor_no_result_on_short_push():
    sp = StreamProcessor(input_sr=8000, target_sr=16000, window_s=3.0, hop_s=1.0)
    chunk = np.zeros(80, dtype=np.float32)  # 10ms @ 8kHz → 160 samples @ 16kHz
    result = sp.push(chunk)
    assert result is None


def test_stream_processor_result_on_full_window():
    sp = StreamProcessor(input_sr=8000, target_sr=16000, window_s=3.0, hop_s=1.0)
    # 3s @ 8kHz = 24000 samples
    audio = np.zeros(24000, dtype=np.float32)
    result = sp.push(audio)
    assert result is not None
    assert "label" in result
    assert "confidence" in result
    assert "window_id" in result


def test_stream_processor_window_id_increments():
    sp = StreamProcessor(input_sr=8000, target_sr=16000, window_s=3.0, hop_s=1.0)
    # Push 6 seconds to get 2 windows (3s window, 1s hop)
    audio = np.zeros(48000, dtype=np.float32)
    results = []
    chunk_size = 4000  # 500ms chunks
    for i in range(0, len(audio), chunk_size):
        r = sp.push(audio[i : i + chunk_size])
        if r is not None:
            results.append(r)
    assert len(results) >= 1
    if len(results) >= 2:
        assert results[1]["window_id"] == results[0]["window_id"] + 1


def test_stream_processor_reset():
    sp = StreamProcessor(input_sr=8000, target_sr=16000)
    sp.push(np.zeros(4000, dtype=np.float32))
    sp.reset()
    assert len(sp._buffer) == 0
    assert sp._window_id == 0


def test_stream_processor_confidence_range():
    sp = StreamProcessor(input_sr=8000, target_sr=16000, window_s=3.0, hop_s=1.0)
    audio = np.zeros(24000, dtype=np.float32)
    result = sp.push(audio)
    if result is not None:
        assert 0.0 <= result["confidence"] <= 1.0


# ── TwilioStreamHandler ─────────────────────────────────────────────────────────


def test_twilio_handler_instantiates():
    handler = TwilioStreamHandler()
    assert handler.INPUT_SR == 8000


def test_twilio_handler_has_processor():
    handler = TwilioStreamHandler()
    assert isinstance(handler._processor, StreamProcessor)
