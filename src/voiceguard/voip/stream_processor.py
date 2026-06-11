"""Rolling audio buffer with resampling for real-time deepfake detection."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _resample_linear(audio: np.ndarray, input_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampling (no scipy dependency)."""
    if input_sr == target_sr:
        return audio
    ratio = target_sr / input_sr
    n_out = int(len(audio) * ratio)
    x_in = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_in, np.arange(len(audio)), audio).astype(np.float32)


class StreamProcessor:
    """Stateful rolling buffer for streaming audio deepfake detection.

    Audio chunks are pushed in at *input_sr* Hz, resampled to *target_sr* Hz,
    and analysed in *window_s*-second windows with *hop_s*-second steps.
    """

    def __init__(
        self,
        input_sr: int = 8000,
        target_sr: int = 16000,
        window_s: float = 3.0,
        hop_s: float = 1.0,
    ) -> None:
        self.input_sr = input_sr
        self.target_sr = target_sr
        self._window_samples = int(target_sr * window_s)
        self._hop_samples = int(target_sr * hop_s)
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._window_id = 0

    def push(self, audio: np.ndarray) -> dict | None:
        """Push a chunk of audio; return detection result dict if a window is ready."""
        resampled = _resample_linear(audio.astype(np.float32), self.input_sr, self.target_sr)
        self._buffer = np.concatenate([self._buffer, resampled])

        if len(self._buffer) >= self._window_samples:
            window = self._buffer[: self._window_samples]
            self._buffer = self._buffer[self._hop_samples :]
            label, confidence, model = self._run_detection(window)
            result = {
                "window_id": self._window_id,
                "label": label,
                "confidence": round(float(confidence), 4),
                "model": model,
            }
            self._window_id += 1
            return result
        return None

    def _run_detection(self, audio: np.ndarray) -> tuple[str, float, str]:
        """Score the window, returning (label, confidence, model). `model` names which
        detector actually scored it ("xls_r_aasist" | "classical" | "stub") so callers
        know what produced the verdict and fallbacks are visible in logs."""
        # Preferred: SSL production detector (same model as /detect and /ws/stream).
        try:
            import torch

            from voiceguard.models.registry import registry

            model = registry.load("xls_r_aasist")
            if model is not None:
                wav = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    probs = torch.softmax(model(wav), dim=-1)[0]
                fake_p = float(probs[1])
                if fake_p >= 0.5:
                    return "fake", fake_p, "xls_r_aasist"
                return "real", float(probs[0]), "xls_r_aasist"
        except Exception:
            logger.warning("stream: SSL detection failed, falling back to classical", exc_info=True)
        # Fallback: classical detector (stub if no trained model is present).
        try:
            from voiceguard.features.extractor import extract_features
            from voiceguard.models.classical import ClassicalDetector

            features = extract_features(audio, self.target_sr)
            detector = ClassicalDetector()
            if detector._clf is None:
                logger.info("stream: no trained detector, returning neutral stub verdict")
                return "real", 0.5, "stub"
            label, confidence = detector.predict_features(features)
            return label, confidence, "classical"
        except Exception:
            logger.warning("stream: classical detection failed, returning stub", exc_info=True)
            return "real", 0.5, "stub"

    def reset(self) -> None:
        """Clear the buffer (call when a new call starts)."""
        self._buffer = np.array([], dtype=np.float32)
        self._window_id = 0
