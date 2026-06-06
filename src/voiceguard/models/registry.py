"""
Central model registry for VoiceGuard.

Maps model keys to loaders + checkpoint env-var names. Provides
auto-discovery of the newest .pt in checkpoints/<key>/ when the env-var
is not set. The /detect endpoint and /health both use this registry.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

_CHECKPOINTS_ROOT = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))


def _newest_pt(subdir: str) -> Path | None:
    d = _CHECKPOINTS_ROOT / subdir
    pts = sorted(d.glob("**/model_best.pt"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return pts[-1] if pts else None


def _load_classical(path: Path) -> Any:
    from voiceguard.models.classical import ClassicalDetector
    return ClassicalDetector.from_file(str(path))


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
            except Exception:
                pass
        model = SSLClassifier(name)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        model.eval()
        return model
    return _loader


# Registry definition: key → {env, loader, discover_subdir}
_REGISTRY_DEF: dict[str, dict] = {
    "classical":        {"env": "CLASSICAL_MODEL_PATH",    "loader": _load_classical,
                         "discover": "classical", "ext": ".pkl"},
    "dsfnet":           {"env": "DSFNET_MODEL_PATH",       "loader": _load_dsfnet,
                         "discover": "dsfnet"},
    "dsfnet_v2":        {"env": "DSFNET_V2_MODEL_PATH",    "loader": _load_dsfnet_v2,
                         "discover": "dsfnet_v2"},
    "aasist":           {"env": "AASIST_MODEL_PATH",       "loader": _load_aasist,
                         "discover": "aasist"},
    "wav2vec2":         {"env": "WAV2VEC2_MODEL_PATH",     "loader": _load_wav2vec2,
                         "discover": "wav2vec2"},
    "wavlm_base_plus":  {"env": "WAVLM_BASE_PLUS_PATH",
                         "loader": _load_ssl("microsoft/wavlm-base-plus"),
                         "discover": "wavlm_base_plus"},
    "wavlm_large":      {"env": "WAVLM_LARGE_PATH",
                         "loader": _load_ssl("microsoft/wavlm-large"),
                         "discover": "wavlm_large"},
    "wav2vec2_large":   {"env": "WAV2VEC2_LARGE_PATH",
                         "loader": _load_ssl("facebook/wav2vec2-large"),
                         "discover": "wav2vec2_large"},
    "xls_r":            {"env": "XLS_R_PATH",
                         "loader": _load_ssl("facebook/wav2vec2-xls-r-300m"),
                         "discover": "xls_r"},
}


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _resolve_path(self, key: str) -> Path | None:
        defn = _REGISTRY_DEF.get(key, {})
        env_val = os.environ.get(defn.get("env", ""), "")
        if env_val:
            p = Path(env_val)
            return p if p.exists() else None
        # Auto-discover newest model_best.pt
        return _newest_pt(defn.get("discover", key))

    def load(self, key: str) -> Any | None:
        """Load and cache model by key. Returns None if no checkpoint found."""
        if key in self._cache:
            return self._cache[key]
        defn = _REGISTRY_DEF.get(key)
        if defn is None:
            return None
        path = self._resolve_path(key)
        if path is None:
            self._cache[key] = None
            return None
        try:
            ext = defn.get("ext", ".pt")
            if not str(path).endswith(ext) and ext != ".pt":
                path = path.with_suffix(ext)
            model = defn["loader"](path)
            self._cache[key] = model
            return model
        except Exception:
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
        for key in _REGISTRY_DEF:
            path = self._resolve_path(key)
            result[key] = {
                "available": path is not None,
                "loaded": key in self._cache and self._cache[key] is not None,
                "path": str(path) if path else None,
            }
        return result

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)


registry = ModelRegistry()
