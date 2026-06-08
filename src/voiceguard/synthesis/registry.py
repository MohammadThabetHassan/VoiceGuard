"""Central registry of synthesis engines (mirrors models.registry.ModelRegistry).

Lets the API list engines (`GET /synthesis/engines`) and resolve one by name for
`POST /synthesize`. Engines self-report availability, so an uninstalled cloning
engine is listed-but-unavailable rather than a hard error.
"""

from __future__ import annotations

from voiceguard.synthesis.base import EngineInfo, SynthEngine
from voiceguard.synthesis.clone_engine import IndexTTS2Engine, XTTSEngine
from voiceguard.synthesis.kokoro_engine import KokoroEngine


class SynthesisRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, SynthEngine] = {}

    def register(self, engine: SynthEngine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> SynthEngine | None:
        return self._engines.get(name)

    def info(self) -> list[EngineInfo]:
        return [e.info() for e in self._engines.values()]


registry = SynthesisRegistry()
registry.register(KokoroEngine())
registry.register(IndexTTS2Engine())
registry.register(XTTSEngine())
