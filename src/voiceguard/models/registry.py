"""
Central model registry for VoiceGuard.

Maps model keys to loaders + checkpoint env-var names. Provides
auto-discovery of the newest .pt in checkpoints/<key>/ when the env-var
is not set. The /detect endpoint and /health both use this registry.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHECKPOINTS_ROOT = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))


def _newest_pt(subdir: str, ext: str = ".pt") -> Path | None:
    """Newest model_best<ext> under checkpoints/<subdir>/ (ext-aware: the classical
    detector is a .pkl, SSL/CNN models are .pt)."""
    d = _CHECKPOINTS_ROOT / subdir
    pts = (
        sorted(d.glob(f"**/model_best{ext}"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    )
    return pts[-1] if pts else None


def _load_classical(path: Path) -> Any:
    from voiceguard.models.classical import ClassicalDetector

    return ClassicalDetector.from_file(str(path))


def _load_hf_spoof(model_id: str) -> Callable[[Path | None], Any]:
    """Loader for a HuggingFace Hub anti-spoofing model (no local checkpoint)."""

    def _loader(_path: Path | None = None) -> Any:
        from voiceguard.models.hf_spoof import HFSpoofDetector

        return HFSpoofDetector(model_id)

    return _loader


def _load_dsfnet(path: Path) -> Any:
    import torch

    from voiceguard.models.dsfnet import DSFNet

    model = DSFNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    return model


def _load_dsfnet_v2(path: Path) -> Any:
    import torch

    from voiceguard.models.dsfnet import DSFNetV2

    model = DSFNetV2()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    return model


def _load_aasist(path: Path) -> Any:
    import torch

    from voiceguard.models.aasist import AASIST

    model = AASIST()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    return model


def _load_wav2vec2(path: Path) -> Any:
    import torch

    from voiceguard.models.wav2vec2_ft import Wav2Vec2Classifier

    model = Wav2Vec2Classifier()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    return model


def _load_ssl(model_name: str) -> Callable[[Path], Any]:
    def _loader(path: Path) -> Any:
        import json

        import torch

        from voiceguard.models.ssl_classifier import SSLClassifier

        # Try to read model_name from sibling config.json
        cfg_path = path.parent / "config.json"
        name = model_name
        if cfg_path.exists():
            try:
                name = json.loads(cfg_path.read_text()).get("model_name", model_name)
            except Exception:  # noqa: S110
                pass
        model = SSLClassifier(name)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        model.eval()
        return model

    return _loader


def _load_ssl_aasist(model_name: str) -> Callable[[Path], Any]:
    """Loader for the SSLAASIST architecture (SSL front-end + graph-attention
    back-end), e.g. the production XLS-R-300m + AASIST detector. Reads
    model_name from a sibling config.json when present."""

    def _loader(path: Path) -> Any:
        import json

        import torch

        from voiceguard.models.ssl_classifier import SSLAASIST

        cfg_path = path.parent / "config.json"
        name = model_name
        if cfg_path.exists():
            try:
                name = json.loads(cfg_path.read_text()).get("model_name", model_name)
            except Exception:  # noqa: S110
                pass
        model = SSLAASIST(name)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        model.eval()
        return model

    return _loader


# Registry definition: key → {env, loader, discover_subdir}
# A "hub" entry loads a HuggingFace model (no local checkpoint) and is always
# reported available.
_REGISTRY_DEF: dict[str, dict] = {
    "classical": {
        "env": "CLASSICAL_MODEL_PATH",
        "loader": _load_classical,
        "discover": "classical",
        "ext": ".pkl",
    },
    # Pretrained wav2vec2-large anti-spoofing detector from the HuggingFace Hub.
    # Default working detector when the production v9c SSL checkpoint is absent —
    # it genuinely separates real speech from TTS / voice clones.
    "wav2vec2_spoof": {
        "hub": "alexandreacff/wav2vec2-large-ft-fake-detection",
        "loader": _load_hf_spoof("alexandreacff/wav2vec2-large-ft-fake-detection"),
    },
    "dsfnet": {"env": "DSFNET_MODEL_PATH", "loader": _load_dsfnet, "discover": "dsfnet"},
    "dsfnet_v2": {
        "env": "DSFNET_V2_MODEL_PATH",
        "loader": _load_dsfnet_v2,
        "discover": "dsfnet_v2",
    },
    "aasist": {"env": "AASIST_MODEL_PATH", "loader": _load_aasist, "discover": "aasist"},
    "wav2vec2": {"env": "WAV2VEC2_MODEL_PATH", "loader": _load_wav2vec2, "discover": "wav2vec2"},
    "wavlm_base_plus": {
        "env": "WAVLM_BASE_PLUS_PATH",
        "loader": _load_ssl("microsoft/wavlm-base-plus"),
        "discover": "wavlm_base_plus",
    },
    "wavlm_large": {
        "env": "WAVLM_LARGE_PATH",
        "loader": _load_ssl("microsoft/wavlm-large"),
        "discover": "wavlm_large",
    },
    "wav2vec2_large": {
        "env": "WAV2VEC2_LARGE_PATH",
        "loader": _load_ssl("facebook/wav2vec2-large"),
        "discover": "wav2vec2_large",
    },
    "xls_r": {
        "env": "XLS_R_PATH",
        "loader": _load_ssl("facebook/wav2vec2-xls-r-300m"),
        "discover": "xls_r",
    },
    # Production headline model (v9c): XLS-R-300m + AASIST graph-attention
    # back-end, official ASVspoof 2021 LA eval EER 2.84% (catches clones +
    # premium TTS). See docs/RESULTS_canonical.md.
    "xls_r_aasist": {
        "env": "XLS_R_AASIST_PATH",
        "loader": _load_ssl_aasist("facebook/wav2vec2-xls-r-300m"),
        "discover": "xls_r_aasist",
    },
}


class ModelRegistry:
    def __init__(self) -> None:
        import threading

        self._cache: dict[str, Any] = {}
        self._fingerprints: dict[str, str | None] = {}
        # load() now also runs on worker threads (asyncio.to_thread in the
        # streaming paths); without a lock two cold-start streams would both
        # miss the cache and load the same multi-GB checkpoint twice.
        self._load_lock = threading.Lock()

    def _resolve_path(self, key: str) -> Path | None:
        defn = _REGISTRY_DEF.get(key, {})
        if "hub" in defn:  # hub models have no local checkpoint path
            return None
        env_val = os.environ.get(defn.get("env", ""), "")
        if env_val:
            p = Path(env_val)
            return p if p.exists() else None
        # Auto-discover newest model_best<ext> (ext-aware: classical is .pkl)
        return _newest_pt(defn.get("discover", key), defn.get("ext", ".pt"))

    def load(self, key: str) -> Any | None:
        """Load and cache model by key. Returns None if no checkpoint found."""
        if key in self._cache:  # fast path, no lock once warm
            return self._cache[key]
        with self._load_lock:
            if key in self._cache:  # a concurrent caller loaded it while we waited
                return self._cache[key]
            defn = _REGISTRY_DEF.get(key)
            if defn is None:
                return None
            if "hub" in defn:  # HuggingFace model — no checkpoint file to resolve
                try:
                    model = defn["loader"](None)
                    self._cache[key] = model
                    return model
                except Exception:
                    logger.exception("Failed to load hub model '%s'", key)
                    self._cache[key] = None
                    return None
            path = self._resolve_path(key)
            if path is None:
                self._cache[key] = None
                return None
            try:
                model = defn["loader"](path)
                self._cache[key] = model
                return model
            except Exception:
                logger.exception("Failed to load model '%s' from %s", key, path)
                self._cache[key] = None
                return None

    def preload(self, keys: list[str] | None = None) -> None:
        """Eagerly load models whose env-vars are set (called at app startup)."""
        for key, defn in _REGISTRY_DEF.items():
            if keys and key not in keys:
                continue
            if os.environ.get(defn.get("env", ""), ""):
                self.load(key)

    def status(self) -> dict[str, dict]:
        """Return availability status for all registered model keys."""
        result = {}
        for key, defn in _REGISTRY_DEF.items():
            if "hub" in defn:  # hub models are always available (pulled on demand)
                result[key] = {
                    "available": True,
                    "loaded": key in self._cache and self._cache[key] is not None,
                    "path": defn["hub"],
                }
                continue
            path = self._resolve_path(key)
            result[key] = {
                "available": path is not None,
                "loaded": key in self._cache and self._cache[key] is not None,
                "path": str(path) if path else None,
            }
        return result

    def fingerprint(self, key: str) -> str | None:
        """SHA-256 of the checkpoint file behind *key* (None if unresolved).

        Computed once and cached — checkpoints are immutable for a running
        process, and the forensic report needs a stable model identity. The
        ~1 GB SSL checkpoint hashes in a few seconds on first request.
        """
        if key in self._fingerprints:
            return self._fingerprints[key]
        path = self._resolve_path(key)
        fp: str | None = None
        if path is not None:
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    while chunk := f.read(8 * 1024 * 1024):
                        h.update(chunk)
                fp = h.hexdigest()
            except OSError:
                logger.warning("could not fingerprint checkpoint for %s", key, exc_info=True)
        self._fingerprints[key] = fp
        return fp

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)
        self._fingerprints.pop(key, None)


registry = ModelRegistry()
