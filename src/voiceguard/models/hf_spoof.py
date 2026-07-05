"""HuggingFace-backed audio anti-spoofing detector.

A self-contained deepfake-voice detector that loads a pretrained
wav2vec2 audio-classification model from the HuggingFace Hub (no local
checkpoint needed) and exposes the same ``(label, confidence)`` contract the
rest of VoiceGuard's detectors use. Unlike the classical MFCC baseline, this
model actually separates genuine human speech from modern TTS / voice clones,
so it is the default detector when the production SSL checkpoint (v9c) is not
staged on the host.

Loading is lazy and cached: the ~1.2 GB model is pulled/loaded on first use.
"""

from __future__ import annotations

import numpy as np

# Default anti-spoofing model. Labels are 'real' / 'fake'.
DEFAULT_MODEL_ID = "alexandreacff/wav2vec2-large-ft-fake-detection"


def _norm_label(raw: str) -> str:
    """Map a model's class label onto VoiceGuard's {'real','fake'} vocabulary."""
    s = raw.strip().lower()
    if any(k in s for k in ("fake", "spoof", "ai", "synth", "clone")):
        return "fake"
    if any(k in s for k in ("real", "human", "bona", "genuine")):
        return "real"
    # Unknown label: fall back to a conservative 'fake' so misses are safe.
    return "fake"


class HFSpoofDetector:
    """Lazy wrapper around a HuggingFace audio-classification pipeline."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1  # GPU when present
            self._pipe = pipeline("audio-classification", model=self.model_id, device=device)
        return self._pipe

    def predict_array(self, audio: np.ndarray, sr: int) -> tuple[str, float]:
        """Score a mono float32 waveform. Returns (label, confidence)."""
        pipe = self._pipeline()
        audio = np.asarray(audio, dtype=np.float32)
        results = pipe({"array": audio, "sampling_rate": int(sr)})
        top = max(results, key=lambda r: r["score"])
        return _norm_label(top["label"]), float(top["score"])
