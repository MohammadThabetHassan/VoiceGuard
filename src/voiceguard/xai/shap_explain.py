"""SHAP explanations for the classical XGBoost detector."""

from __future__ import annotations

import numpy as np


def explain_prediction(
    detector,
    features: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Return per-feature SHAP values for a single prediction.

    Args:
        detector: ClassicalDetector instance (must be fitted).
        features: 1-D array of shape (n_features,).
        feature_names: Optional list of feature names; defaults to Feature_0..N.

    Returns:
        Dict mapping feature name → SHAP value (float).
    """
    import shap

    pipeline = detector._clf  # sklearn Pipeline stored as _clf
    X = features.reshape(1, -1)

    # Use TreeExplainer on the XGB step inside the sklearn Pipeline
    xgb_step = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(xgb_step)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values shape for binary: (1, n_features) or (n_classes, 1, n_features)
    if isinstance(shap_values, list):
        vals = shap_values[1][0]  # class 1 (fake)
    else:
        vals = shap_values[0]

    n = len(vals)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n)]

    return dict(zip(feature_names, vals.tolist(), strict=True))


def top_features(
    shap_dict: dict[str, float],
    n: int = 5,
    absolute: bool = True,
) -> list[tuple[str, float]]:
    """Return top-n features sorted by |SHAP value|."""
    key = (lambda item: abs(item[1])) if absolute else (lambda item: item[1])
    return sorted(shap_dict.items(), key=key, reverse=True)[:n]
