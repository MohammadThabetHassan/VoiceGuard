"""Tests for SSL Integrated Gradients attribution."""
from __future__ import annotations

import pytest
import torch

from voiceguard.xai.ssl_explain import (
    _to_frame_attribution,
    explain_waveform,
)


class _FakeModel(torch.nn.Module):
    """Minimal model: scores second half of waveform as fake."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, x):  # x: (B, T)
        # Deliberately make model sensitive to the second half
        second_half = x[:, x.shape[1] // 2:].abs().mean(dim=-1, keepdim=True)
        return self.linear(second_half)


@pytest.fixture
def fake_model():
    m = _FakeModel()
    m.eval()
    return m


@pytest.fixture
def waveform():
    T = 16000  # 1s @ 16kHz
    w = torch.zeros(1, T)
    # Put a spike in the second half
    w[0, T // 2:] = 1.0
    return w


def test_explain_waveform_structure(fake_model, waveform):
    result = explain_waveform(fake_model, waveform, n_steps=5)
    assert "attribution_frames" in result
    assert "top_segments" in result
    assert result["method"] == "integrated_gradients"
    assert result["target_class"] == 1
    assert len(result["attribution_frames"]) > 0
    assert all(0.0 <= v <= 1.0 for v in result["attribution_frames"])


def test_explain_waveform_localizes_spike(fake_model, waveform):
    """Attribution should be higher in the spiked second half."""
    result = explain_waveform(fake_model, waveform, n_steps=5)
    frames = result["attribution_frames"]
    n = len(frames)
    first_half_avg = sum(frames[: n // 2]) / (n // 2)
    second_half_avg = sum(frames[n // 2 :]) / (n - n // 2)
    assert second_half_avg > first_half_avg, (
        f"Expected second half ({second_half_avg:.3f}) > first half ({first_half_avg:.3f})"
    )


def test_explain_top_segments_sorted(fake_model, waveform):
    result = explain_waveform(fake_model, waveform, n_steps=5)
    importances = [s["importance"] for s in result["top_segments"]]
    assert importances == sorted(importances, reverse=True)


def test_explain_segment_times_valid(fake_model, waveform):
    result = explain_waveform(fake_model, waveform, n_steps=5)
    for seg in result["top_segments"]:
        assert seg["start_s"] >= 0
        assert seg["end_s"] > seg["start_s"]


def test_explain_1d_input(fake_model):
    """Should accept (T,) input and not crash."""
    wav = torch.randn(16000)
    result = explain_waveform(fake_model, wav, n_steps=5)
    assert "attribution_frames" in result


def test_to_frame_attribution_normalised():
    ig = torch.tensor([0.0, 0.5, 1.0, 0.0])
    frames = _to_frame_attribution(ig, 4)
    assert frames.max() == pytest.approx(1.0, abs=1e-5)
    assert frames.min() >= 0.0
