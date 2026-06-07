"""
Integrated Gradients attribution for SSL-based deepfake detectors.

Works with Wav2Vec2Classifier, SSLClassifier (WavLM, XLS-R) and any model
that accepts a (1, T) waveform tensor and returns (1, 2) logits.

Usage::

    from voiceguard.xai.ssl_explain import explain_waveform
    result = explain_waveform(model, waveform_1d_tensor)
    # result["top_segments"] — list of {"start_s", "end_s", "importance"} dicts
    # result["attribution"]  — list of floats, one per 10ms frame
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

_SR = 16000
_FRAME_MS = 10  # Attribution granularity: 10 ms bins
_FRAME_SAMPLES = _SR * _FRAME_MS // 1000  # 160 samples
_IG_STEPS = 25  # Integration steps — fast enough for an API call


def _integrated_gradients(
    model: Any,
    waveform: torch.Tensor,  # (1, T)
    target_class: int,
    n_steps: int = _IG_STEPS,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return IG attribution of shape (T,) over the waveform."""
    waveform = waveform.detach()
    if baseline is None:
        baseline = torch.zeros_like(waveform)

    # Interpolate baseline → input in n_steps
    alphas = torch.linspace(0.0, 1.0, n_steps, device=waveform.device)  # (S,)
    # Batch: (S, T)
    interp = baseline + alphas[:, None] * (waveform - baseline)  # (S, T)
    interp = interp.requires_grad_(True)

    # Forward all steps in one batch
    logits = model(interp)  # (S, 2)
    scores = torch.softmax(logits, dim=-1)[:, target_class]  # (S,)
    scores.sum().backward()

    grads = interp.grad  # (S, T)
    avg_grads = grads.mean(dim=0)  # (T,)
    ig = (waveform.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return ig


def _to_frame_attribution(ig: torch.Tensor, T: int) -> np.ndarray:
    """Pool signed IG to 10ms frames, take absolute value, normalise to [0,1]."""
    ig_np = ig.detach().cpu().numpy()
    n_frames = math.ceil(T / _FRAME_SAMPLES)
    frames = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * _FRAME_SAMPLES
        e = min(s + _FRAME_SAMPLES, T)
        frames[i] = float(np.abs(ig_np[s:e]).mean())
    mx = frames.max()
    if mx > 0:
        frames /= mx
    return frames


def explain_waveform(
    model: Any,
    waveform: torch.Tensor,  # (1, T) on any device
    target_class: int = 1,  # 1 = fake
    top_k: int = 5,
    n_steps: int = _IG_STEPS,
) -> dict:
    """
    Compute Integrated Gradients attribution and return a structured explanation.

    Returns
    -------
    dict with keys:
        attribution_frames : list[float]  — normalised importance per 10ms frame
        top_segments       : list[dict]   — top-k suspicious time windows
        baseline           : "zeros"
        method             : "integrated_gradients"
        target_class       : int
    """
    model.eval()
    device = next(model.parameters()).device
    waveform = waveform.to(device)

    if waveform.dim() == 2 and waveform.shape[0] == 1:
        wav_1d = waveform  # (1, T)
    elif waveform.dim() == 1:
        wav_1d = waveform.unsqueeze(0)
    else:
        wav_1d = waveform.mean(0, keepdim=True)

    T = wav_1d.shape[-1]
    ig = _integrated_gradients(model, wav_1d, target_class, n_steps=n_steps)
    frames = _to_frame_attribution(ig, T)

    # Top-k contiguous suspicious windows (merge neighbouring top frames)
    top_indices = np.argsort(frames)[::-1]
    segments: list[dict] = []
    used: set[int] = set()
    for idx in top_indices:
        if idx in used or len(segments) >= top_k:
            continue
        # Grow the window while importance stays above threshold
        lo, hi = int(idx), int(idx)
        while lo > 0 and frames[lo - 1] >= 0.5:
            lo -= 1
        while hi < len(frames) - 1 and frames[hi + 1] >= 0.5:
            hi += 1
        for i in range(lo, hi + 1):
            used.add(i)
        segments.append(
            {
                "start_s": round(lo * _FRAME_MS / 1000, 3),
                "end_s": round((hi + 1) * _FRAME_MS / 1000, 3),
                "importance": round(float(frames[lo : hi + 1].mean()), 4),
            }
        )

    segments.sort(key=lambda s: s["importance"], reverse=True)

    return {
        "method": "integrated_gradients",
        "baseline": "zeros",
        "target_class": target_class,
        "frame_duration_ms": _FRAME_MS,
        "attribution_frames": frames.tolist(),
        "top_segments": segments[:top_k],
    }
