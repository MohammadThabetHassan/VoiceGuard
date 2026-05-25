"""Spectral watermarking stub compatible with C2PA v1.4 metadata embedding.

The watermark is embedded as a low-amplitude sinusoidal tone at an inaudible
frequency (18 kHz), amplitude-modulated by a PRNG sequence seeded from the
watermark_id.  Detection uses cross-correlation.
"""

from __future__ import annotations

import hashlib
import uuid

import numpy as np


def _clamp_carrier(carrier_hz: float, sr: int) -> float:
    """Clamp carrier below Nyquist."""
    nyquist = sr / 2
    return min(carrier_hz, nyquist - 100.0)


def _prng_sequence(seed: str, length: int) -> np.ndarray:
    """Reproducible ±1 spreading code from a string seed."""
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32))
    return rng.choice([-1.0, 1.0], size=length).astype(np.float32)


def embed(
    audio: np.ndarray,
    sr: int = 22050,
    watermark_id: str | None = None,
    amplitude: float = 0.002,
    carrier_hz: float = 18000.0,
) -> tuple[np.ndarray, str]:
    """Embed a spectral watermark into *audio* (float32, normalised to [-1, 1]).

    Args:
        audio: 1-D float32 PCM samples.
        sr: Sample rate.
        watermark_id: Unique ID for this watermark; auto-generated if None.
        amplitude: Watermark signal amplitude (inaudible ≤ 0.005 typical).
        carrier_hz: Carrier frequency in Hz (should be > 16 kHz for inaudibility).

    Returns:
        (watermarked_audio, watermark_id)
    """
    if watermark_id is None:
        watermark_id = str(uuid.uuid4())

    carrier_hz = _clamp_carrier(carrier_hz, sr)

    t = np.arange(len(audio), dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    code = _prng_sequence(watermark_id, len(audio))
    watermark = amplitude * carrier * code

    return (audio + watermark).clip(-1.0, 1.0), watermark_id


def detect(
    audio: np.ndarray,
    sr: int = 22050,
    watermark_id: str = "",
    carrier_hz: float = 18000.0,
    threshold: float = 0.3,
) -> tuple[bool, float]:
    """Detect whether *audio* contains a watermark for *watermark_id*.

    Returns:
        (detected: bool, correlation: float)
        `detected` is True when normalised cross-correlation exceeds *threshold*.
    """
    carrier_hz = _clamp_carrier(carrier_hz, sr)
    t = np.arange(len(audio), dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    code = _prng_sequence(watermark_id, len(audio))
    reference = carrier * code

    corr = float(np.dot(audio.astype(np.float32), reference))
    norm = float(np.linalg.norm(audio) * np.linalg.norm(reference) + 1e-12)
    normalised = corr / norm
    return normalised > threshold, normalised
