"""Rolling audio buffer with resampling for real-time deepfake detection."""

from __future__ import annotations

import logging
import os

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


def _default_score_cap_s() -> float:
    """The streaming score cap, shared with /ws/stream: VG_WS_SCORE_SECONDS
    (default 15s), defensively parsed and clamped to the model's 3s minimum."""
    try:
        value = float(os.environ.get("VG_WS_SCORE_SECONDS", 15.0))
    except ValueError:
        logger.warning("VG_WS_SCORE_SECONDS is not a number; using default 15s")
        value = 15.0
    return max(value, 3.0)


class StreamProcessor:
    """Stateful growing-prefix buffer for streaming audio deepfake detection.

    Audio chunks are pushed in at *input_sr* Hz and resampled to *target_sr* Hz.
    The detector is only reliable on audio scored from the recording's natural
    start (mid-utterance windows read as synthetic regardless of content), so
    each verdict re-scores the growing prefix of the call — first at *window_s*
    seconds, then every *hop_s* seconds, capped at *score_cap_s* seconds of audio
    (default: VG_WS_SCORE_SECONDS, same knob as /ws/stream). The verdict covering
    the full cap carries ``"final": True`` and is the last one emitted.
    """

    def __init__(
        self,
        input_sr: int = 8000,
        target_sr: int = 16000,
        window_s: float = 3.0,
        hop_s: float = 2.0,
        score_cap_s: float | None = None,
    ) -> None:
        self.input_sr = input_sr
        self.target_sr = target_sr
        self._window_samples = int(target_sr * window_s)
        self._hop_samples = int(target_sr * hop_s)
        cap_s = _default_score_cap_s() if score_cap_s is None else score_cap_s
        self._cap_samples = max(int(target_sr * cap_s), self._window_samples)
        # Preallocated to the cap: np.concatenate per 20ms Twilio frame would
        # re-copy the whole growing buffer 50x/s (O(n^2) over the call).
        self._buffer: np.ndarray = np.empty(self._cap_samples, dtype=np.float32)
        self._buffered = 0
        self._received = 0
        self._next_score_at = self._window_samples
        self._final = False
        self._window_id = 0

    def push(self, audio: np.ndarray) -> dict | None:
        """Push a chunk of audio; return a detection result dict when one is due.

        Returns None once the final (cap-covering) verdict has been emitted —
        the caller can keep pushing, but no further inference runs.
        """
        resampled = _resample_linear(audio.astype(np.float32), self.input_sr, self.target_sr)
        self._received += len(resampled)
        if self._buffered < self._cap_samples:
            n = min(len(resampled), self._cap_samples - self._buffered)
            self._buffer[self._buffered : self._buffered + n] = resampled[:n]
            self._buffered += n

        if self._final or self._received < self._next_score_at:
            return None
        # Coalesce overdue milestones: score the freshest prefix once instead of
        # replaying every stale hop boundary after a burst or a slow pass.
        scored_to = min(self._received, self._buffered)
        label, confidence, model = self._run_detection(self._buffer[:scored_to])
        self._final = scored_to >= self._cap_samples
        result = {
            "window_id": self._window_id,
            "label": label,
            "confidence": round(float(confidence), 4),
            "model": model,
            "final": self._final,
        }
        self._window_id += 1
        self._next_score_at = scored_to + self._hop_samples
        return result

    def _run_detection(self, audio: np.ndarray) -> tuple[str, float, str]:
        """Score the call prefix, returning (label, confidence, model). `model` names
        which detector actually scored it ("xls_r_aasist" | "classical" | "stub") so
        callers know what produced the verdict and fallbacks are visible in logs."""
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
        """Clear all state (call when a new call starts)."""
        self._buffered = 0
        self._received = 0
        self._next_score_at = self._window_samples
        self._final = False
        self._window_id = 0
