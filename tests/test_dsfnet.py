"""
DSFNet architecture unit tests.

All tests use synthetic tensors — no real audio files required.
"""

import time

import pytest
import torch

from voiceguard.models.dsfnet import (
    BidirectionalCrossAttention,
    ClassificationHead,
    DSFNet,
    MelSpectrogramTransform,
    SpectrogramEncoder,
    WaveformEncoder,
)

SR = 16000
CLIP_LEN = 3 * SR  # 3-second audio at 16kHz
BATCH = 4


@pytest.fixture
def waveform():
    """Batch of 3-second raw waveforms."""
    return torch.randn(BATCH, 1, CLIP_LEN)


@pytest.fixture
def model():
    return DSFNet()


def test_waveform_encoder_shape(waveform):
    enc = WaveformEncoder()
    out = enc(waveform)
    assert out.shape == (BATCH, 512), f"Expected (4, 512), got {out.shape}"


def test_spectrogram_encoder_shape():
    enc = SpectrogramEncoder()
    spec = torch.randn(BATCH, 1, 80, 300)  # (B, 1, F, T)
    out = enc(spec)
    assert out.shape == (BATCH, 512), f"Expected (4, 512), got {out.shape}"


def test_cross_attention_shape():
    attn = BidirectionalCrossAttention(d_model=512, n_heads=8)
    feat_a = torch.randn(BATCH, 512)
    feat_b = torch.randn(BATCH, 512)
    fused = attn(feat_a, feat_b)
    assert fused.shape == (BATCH, 1024), f"Expected (4, 1024), got {fused.shape}"


def test_classification_head_shape():
    head = ClassificationHead()
    x = torch.randn(BATCH, 1024)
    out = head(x)
    assert out.shape == (BATCH, 2), f"Expected (4, 2), got {out.shape}"


def test_mel_transform_shape(waveform):
    mel = MelSpectrogramTransform(sr=SR)
    spec = mel(waveform)
    assert spec.shape[0] == BATCH
    assert spec.shape[1] == 1
    assert spec.shape[2] == 80  # n_mels


def test_full_forward_pass(model, waveform):
    """Full forward pass: batch of 4, 3-second audio → [4, 2] logits."""
    logits = model(waveform)
    assert logits.shape == (BATCH, 2), f"Expected (4, 2), got {logits.shape}"


def test_forward_with_explicit_spectrogram(model, waveform):
    mel = MelSpectrogramTransform(sr=SR)
    spec = mel(waveform)
    logits = model(waveform, spectrogram=spec)
    assert logits.shape == (BATCH, 2)


def test_output_finite(model, waveform):
    logits = model(waveform)
    assert torch.all(torch.isfinite(logits)), "Model output contains NaN or Inf"


def test_softmax_sums_to_one(model, waveform):
    logits = model(waveform)
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


def test_parameter_count(model):
    n_params = model.count_parameters()
    assert n_params < 50_000_000, f"Model has {n_params:,} params — exceeds 50M limit"
    assert n_params > 1_000_000, f"Model has only {n_params:,} params — suspiciously small"


def test_predict_method(model, waveform):
    preds, probs = model.predict(waveform)
    assert preds.shape == (BATCH,)
    assert probs.shape == (BATCH, 2)
    assert all(p in {0, 1} for p in preds.tolist())


def test_cpu_inference_time(model, waveform):
    """Measure single-sample inference time on CPU (informational only)."""
    single = waveform[:1]
    # Warm up
    for _ in range(3):
        model(single)
    start = time.time()
    for _ in range(10):
        model(single)
    elapsed_ms = (time.time() - start) / 10 * 1000
    print(f"\nCPU inference (single sample): {elapsed_ms:.1f}ms")
    # No assertion on CPU — GPU target is ≤200ms per 3s window


def test_gradient_flow(waveform):
    """Verify gradients flow through the full model."""
    model = DSFNet()
    logits = model(waveform)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    # Check no NaN gradients
    for g in grads:
        assert torch.all(torch.isfinite(g)), "NaN/Inf gradient detected"
