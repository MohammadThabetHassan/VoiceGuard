"""Grad-CAM attribution for DSFNet using captum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from voiceguard.models.dsfnet import DSFNet


def gradcam_waveform(
    model: DSFNet,
    waveform: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """Return Grad-CAM saliency over the waveform time axis.

    Args:
        model: Trained DSFNet instance.
        waveform: (1, 1, T) float32 tensor, normalised to [-1, 1].
        target_class: Class index for which to compute gradients (1 = fake).

    Returns:
        1-D numpy array of length T with per-sample importance scores (≥0).
    """
    from captum.attr import LayerGradCam

    model.eval()
    target_layer = model.wave_enc.blocks[-1]

    lgc = LayerGradCam(model, target_layer)

    waveform = waveform.requires_grad_(True)  # noqa: SIM910 – captum needs grad
    attributions = lgc.attribute(waveform, target=target_class)

    # attributions shape: (1, C, T')  — pool over channels, upsample to T
    saliency = attributions.squeeze(0).mean(dim=0)  # (T',)
    saliency = torch.relu(saliency)
    saliency_np = saliency.detach().cpu().numpy()

    # Upsample back to original waveform length
    T = waveform.shape[-1]
    indices = np.linspace(0, len(saliency_np) - 1, T)
    upsampled = np.interp(indices, np.arange(len(saliency_np)), saliency_np)

    # Normalise to [0, 1]
    mx = upsampled.max()
    if mx > 0:
        upsampled /= mx
    return upsampled.astype(np.float32)


def gradcam_spectrogram(
    model: DSFNet,
    waveform: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """Return Grad-CAM saliency over the Mel-spectrogram time axis.

    Returns:
        1-D numpy array of length T' (Mel time frames) with importance scores.
    """
    from captum.attr import LayerGradCam

    from voiceguard.models.dsfnet import MelSpectrogramTransform

    model.eval()
    transform = MelSpectrogramTransform(sr=16000)
    spectrogram = transform(waveform).requires_grad_(True)

    target_layer = model.spec_enc.blocks[-1]

    def forward_spec(spec: torch.Tensor) -> torch.Tensor:
        wav_feat = model.wave_enc(waveform)
        spec_feat = model.spec_enc(spec)
        wav_feat_exp = wav_feat.unsqueeze(1)
        spec_feat_exp = spec_feat.unsqueeze(1)
        wav_out, spec_out = model.cross_attn(wav_feat_exp, spec_feat_exp)
        fused = torch.cat([wav_out.squeeze(1), spec_out.squeeze(1)], dim=-1)
        return model.head(fused)

    lgc = LayerGradCam(forward_spec, target_layer)
    attributions = lgc.attribute(spectrogram, target=target_class)

    saliency = attributions.squeeze(0).mean(dim=(0, 1))  # (T',)
    saliency = torch.relu(saliency)
    saliency_np = saliency.detach().cpu().numpy()

    mx = saliency_np.max()
    if mx > 0:
        saliency_np /= mx
    return saliency_np.astype(np.float32)
