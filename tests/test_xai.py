"""Tests for Grad-CAM and SHAP explainability modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voiceguard.xai.shap_explain import explain_prediction, top_features

# ── SHAP tests ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fitted_detector():
    """Return a ClassicalDetector fitted on synthetic data."""
    from voiceguard.models.classical import ClassicalDetector

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 18)).astype(np.float32)
    y = np.array(["real"] * 50 + ["fake"] * 50)
    det = ClassicalDetector()
    det.fit(X, y)
    return det


def test_explain_prediction_returns_dict(fitted_detector):
    features = np.random.default_rng(42).standard_normal(18).astype(np.float32)
    result = explain_prediction(fitted_detector, features)
    assert isinstance(result, dict)
    assert len(result) == 18


def test_explain_prediction_custom_names(fitted_detector):
    features = np.zeros(18, dtype=np.float32)
    names = [f"f{i}" for i in range(18)]
    result = explain_prediction(fitted_detector, features, feature_names=names)
    assert set(result.keys()) == set(names)


def test_explain_prediction_values_are_float(fitted_detector):
    features = np.random.default_rng(1).standard_normal(18).astype(np.float32)
    result = explain_prediction(fitted_detector, features)
    for v in result.values():
        assert isinstance(v, float)


def test_top_features_length(fitted_detector):
    features = np.random.default_rng(2).standard_normal(18).astype(np.float32)
    shap_dict = explain_prediction(fitted_detector, features)
    top = top_features(shap_dict, n=5)
    assert len(top) == 5


def test_top_features_sorted_by_abs(fitted_detector):
    features = np.random.default_rng(3).standard_normal(18).astype(np.float32)
    shap_dict = explain_prediction(fitted_detector, features)
    top = top_features(shap_dict, n=5)
    abs_vals = [abs(v) for _, v in top]
    assert abs_vals == sorted(abs_vals, reverse=True)


# ── Grad-CAM smoke tests (CPU, untrained model) ────────────────────────────────


@pytest.fixture
def dsfnet_model():
    from voiceguard.models.dsfnet import DSFNet

    return DSFNet()


def test_gradcam_waveform_shape(dsfnet_model):
    from voiceguard.xai.gradcam import gradcam_waveform

    T = 16000 * 1  # 1 second
    waveform = torch.randn(1, 1, T)
    saliency = gradcam_waveform(dsfnet_model, waveform, target_class=1)
    assert saliency.shape == (T,)


def test_gradcam_waveform_range(dsfnet_model):
    from voiceguard.xai.gradcam import gradcam_waveform

    T = 16000
    waveform = torch.randn(1, 1, T)
    saliency = gradcam_waveform(dsfnet_model, waveform, target_class=1)
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0 + 1e-6
