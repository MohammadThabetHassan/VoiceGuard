"""
SM2026 baseline reproduction test.

Loads osr_features.csv (475 samples, 18 features) and runs 5-fold stratified
cross-validation with the published Enhanced+XGBoost configuration.

Published result: F1 = 0.9500 on 475 samples.
This test asserts mean F1 >= 0.95.
"""

from pathlib import Path

import numpy as np

from voiceguard.models.classical import cross_validate, load_csv_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "osr_features.csv"


def test_fixture_exists():
    assert FIXTURE.exists(), f"Fixture not found: {FIXTURE}"


def test_fixture_shape():
    X, y = load_csv_dataset(FIXTURE)
    # 474 data rows (237 real + 237 fake), 18 features
    assert X.shape == (474, 18), f"Expected (474, 18), got {X.shape}"
    assert len(np.unique(y)) == 2, "Dataset must have exactly 2 classes"
    real_count = int(np.sum(y == 0))
    fake_count = int(np.sum(y == 1))
    assert real_count == 237 and fake_count == 237, (
        f"Expected 237/237 class split, got real={real_count} fake={fake_count}"
    )


def test_baseline_f1():
    """F1 >= 0.9500 on 5-fold stratified CV — reproduces SM2026 published result."""
    X, y = load_csv_dataset(FIXTURE)
    metrics = cross_validate(X, y, cv_folds=5, random_state=42)

    print(f"\nBaseline CV results: {metrics}")

    assert metrics["f1"] >= 0.95, (
        f"F1 {metrics['f1']:.4f} is below the published target of 0.9500. "
        "Check hyperparameters match the SM2026 configuration."
    )


def test_no_nan_in_features():
    X, _ = load_csv_dataset(FIXTURE)
    assert not np.any(np.isnan(X)), "NaN values found in feature matrix"
    assert not np.any(np.isinf(X)), "Inf values found in feature matrix"


def test_label_encoding():
    _, y = load_csv_dataset(FIXTURE)
    assert set(y).issubset({0, 1}), "Labels must be encoded as 0 (real) or 1 (fake)"


def test_feature_extractor_shape():
    """Extractor returns correct 18-dim vector on synthetic audio."""
    from voiceguard.features.extractor import extract_features

    sr = 16000
    signal = np.random.randn(sr * 2).astype(np.float32) * 0.1
    features = extract_features(signal, sr)

    assert features.shape == (18,), f"Expected (18,), got {features.shape}"
    assert not np.any(np.isnan(features)), "Extractor produced NaN features"
    assert not np.any(np.isinf(features)), "Extractor produced Inf features"
