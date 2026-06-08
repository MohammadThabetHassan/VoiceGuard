"""Synthesis engine abstraction.

A `SynthEngine` turns text (optionally conditioned on a reference voice) into raw
audio. Engines return *un-watermarked* float32 audio + sample rate; the API layer
centralises watermarking, file writing, and TTL cleanup so every engine's output
is uniformly flagged as AI-generated.

Heavy cloning engines (IndexTTS2, XTTS) have dependency stacks that conflict with
the API process (e.g. transformers 4.52 vs 5.x), so they run **out-of-process** via
an isolated venv + `clone_worker.py`; see `CloneEngine`. Lightweight in-process
engines (Kokoro) subclass `SynthEngine` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EngineInfo:
    """Serialisable description of an engine for `GET /synthesis/engines`."""

    name: str
    label: str
    requires_reference: bool
    available: bool
    preset_voices: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["en"])
    description: str = ""


class SynthEngine:
    """Base class for synthesis engines. Subclasses set the class attributes
    and implement `is_available()` and `synthesize()`."""

    name: str = "base"
    label: str = "Base"
    requires_reference: bool = False
    preset_voices: list[str] = []
    languages: list[str] = ["en"]
    description: str = ""

    def is_available(self) -> bool:  # pragma: no cover - overridden
        return False

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        reference_wav: str | Path | None = None,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio in [-1, 1], sample_rate). Override."""
        raise NotImplementedError

    def info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            label=self.label,
            requires_reference=self.requires_reference,
            available=self.is_available(),
            preset_voices=list(self.preset_voices),
            languages=list(self.languages),
            description=self.description,
        )
