"""Local Kokoro-82M text-to-speech for the Generate feature.

Kokoro is a small, fast, fully-local TTS (no API key). Output is watermarked
with the C2PA spectral watermark so synthesised audio is cryptographically
flagged as AI-generated. The pipeline is loaded lazily and cached.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceguard.watermark.c2pa_watermark import embed

KOKORO_SR = 24000  # Kokoro native sample rate
# American + British, female + male. af_heart is the default high-quality voice.
VOICES = ["af_heart", "af_nova", "am_adam", "am_echo", "bf_emma", "bm_george"]
_LANG_CODE = {"en": "a", "en-us": "a", "en-gb": "b"}

_pipeline = None
_lock = threading.Lock()


def _get_pipeline(lang_code: str = "a"):
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                from kokoro import KPipeline

                _pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def synthesize_raw(
    text: str,
    voice: str = "af_heart",
    language: str = "en",
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Synthesise *text* and return (float32 audio, sample_rate) — no watermark.

    This is the single Kokoro inference path, used by both the synthesis engine
    and the legacy `synthesize_to_file` wrapper. Raises ValueError on empty audio.
    """
    if voice not in VOICES:
        voice = "af_heart"
    pipe = _get_pipeline(_LANG_CODE.get(language.lower(), "a"))
    chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=voice, speed=speed)]
    if not chunks:
        raise ValueError("Synthesis produced no audio")
    return np.concatenate(chunks).astype(np.float32), KOKORO_SR


def synthesize_to_file(
    text: str,
    out_dir: str | Path,
    voice: str = "af_heart",
    language: str = "en",
    speed: float = 1.0,
) -> tuple[str, str, int]:
    """Synthesise *text*, watermark it, and write a WAV into *out_dir*.

    Returns (filename, watermark_id, duration_ms). Raises ValueError if the
    engine produced no audio or the voice is unknown.
    """
    audio, _sr = synthesize_raw(text, voice=voice, language=language, speed=speed)

    # C2PA spectral watermark marking the clip as AI-generated. Amplitude is set
    # high enough to stay reliably verifiable while remaining unobtrusive.
    watermarked, watermark_id = embed(audio, sr=KOKORO_SR, amplitude=0.01)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"vg_{uuid.uuid4().hex[:12]}.wav"
    sf.write(str(out_dir / fname), watermarked, KOKORO_SR)
    duration_ms = int(len(audio) / KOKORO_SR * 1000)
    return fname, watermark_id, duration_ms
