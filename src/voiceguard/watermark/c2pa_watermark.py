"""Spectral (in-signal) watermarking for synthesized audio.

This is the *robust* provenance layer: a low-amplitude sinusoidal tone at an
inaudible frequency (18 kHz), amplitude-modulated by a PRNG sequence seeded from
the watermark_id; detection uses cross-correlation. Because the mark lives in the
audio signal itself, it survives lossy re-encoding and playback-recapture that
strip file metadata.

This is distinct from — and complementary to — the *cryptographic* provenance
layer in ``c2pa_sign``, which embeds a real, signed C2PA manifest (metadata) into
the file. The synthesis API applies both: the C2PA manifest for verifiable,
standards-compliant provenance and this spectral mark for re-encode robustness.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)

# An 18 kHz carrier is only inaudible if the sample rate can carry it above
# ~16 kHz. Below this, a clamped carrier lands in the audible band.
MIN_INAUDIBLE_CARRIER_HZ = 16000.0


def _clamp_carrier(carrier_hz: float, sr: int) -> float:
    """Clamp carrier below Nyquist."""
    nyquist = sr / 2
    return min(carrier_hz, nyquist - 100.0)


def ensure_carrier_sr(
    audio: np.ndarray,
    sr: int,
    carrier_hz: float = 18000.0,
    min_carrier_hz: float = MIN_INAUDIBLE_CARRIER_HZ,
) -> tuple[np.ndarray, int]:
    """Resample *audio* up if *sr* is too low to carry an inaudible watermark tone.

    A carrier needs Nyquist (sr/2) comfortably above ``min_carrier_hz`` or the
    clamped carrier becomes audible (the Kokoro bug: 18 kHz carrier on 24 kHz
    audio clamps to 11.9 kHz). Returns ``(audio, sr)`` — possibly at a higher
    sample rate; a no-op when *sr* already suffices.
    """
    if sr / 2 - 100.0 >= min_carrier_hz:
        return np.asarray(audio, dtype=np.float32), sr
    target_sr = max(int(2 * (carrier_hz + 2000.0)), 48000)
    from scipy.signal import resample_poly

    g = gcd(target_sr, sr)
    out = resample_poly(np.asarray(audio, dtype=np.float32), target_sr // g, sr // g)
    return out.astype(np.float32), target_sr


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
    if carrier_hz < MIN_INAUDIBLE_CARRIER_HZ:
        logger.warning(
            "Watermark carrier clamped to %.0f Hz at sr=%d Hz — this is AUDIBLE; "
            "call ensure_carrier_sr() to resample up before embed().",
            carrier_hz,
            sr,
        )

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
    threshold: float = 0.02,
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
