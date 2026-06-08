"""Kokoro-82M synthesis engine (in-process, preset voices, no cloning)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voiceguard.synthesis.base import SynthEngine
from voiceguard.synthesis.kokoro_synth import VOICES, synthesize_raw


class KokoroEngine(SynthEngine):
    name = "kokoro"
    label = "Kokoro-82M (local, preset voices)"
    requires_reference = False
    preset_voices = VOICES
    languages = ["en"]
    description = "Fast, fully-local TTS with high-quality preset voices. No cloning."

    def is_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("kokoro") is not None
        except Exception:  # pragma: no cover - defensive
            return False

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        reference_wav: str | Path | None = None,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        return synthesize_raw(text, voice=voice or "af_heart", language=language)
