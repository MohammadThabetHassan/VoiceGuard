"""
Classical ML models for voice deepfake detection.

Implements the SM2026 baseline: Standard-scaled XGBoost classifier
trained on the 18-dim feature vector (MFCC1-13 + spectral + jitter + shimmer).

Published result: F1 = 0.9500 on 475 samples (5-fold stratified CV).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from voiceguard.features.extractor import FEATURE_COLS

LABEL_MAP = {"real": 0, "fake": 1}
INV_LABEL_MAP = {0: "real", 1: "fake"}


def build_xgb_pipeline() -> Pipeline:
    """Return the SM2026 XGBoost pipeline (StandardScaler + XGBClassifier)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def encode_labels(labels: np.ndarray | list[str]) -> np.ndarray:
    """Encode string labels to 0/1 integers (real→0, fake→1)."""
    return np.array([LABEL_MAP.get(str(lbl), lbl) for lbl in labels], dtype=int)


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Run stratified k-fold CV and return mean F1, accuracy, precision, recall.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Integer label array (0=real, 1=fake).
        cv_folds: Number of folds.
        random_state: Random seed.

    Returns:
        Dict with keys: f1, accuracy, precision, recall.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    clf = build_xgb_pipeline()
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    return {
        "f1": float(f1_score(y, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, pos_label=1)),
        "recall": float(recall_score(y, y_pred, pos_label=1)),
    }


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train a final XGBoost pipeline on the full dataset."""
    clf = build_xgb_pipeline()
    clf.fit(X, y)
    return clf


def predict(clf: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return integer label predictions (0=real, 1=fake)."""
    return clf.predict(X)


def predict_proba(clf: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return probability array (n_samples, 2) — columns: [real, fake]."""
    return clf.predict_proba(X)


def load_csv_dataset(
    path: str | Path,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a pre-extracted feature CSV and return (X, y) arrays.

    Args:
        path: Path to CSV file with feature columns + 'label' column.
        feature_cols: List of feature column names. Defaults to FEATURE_COLS.

    Returns:
        X: float64 array (n_samples, n_features).
        y: int array (n_samples,) — 0=real, 1=fake.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df = pd.read_csv(path)
    X = df[feature_cols].values.astype(np.float64)
    y = encode_labels(df["label"].values)
    return X, y


def save_model(clf: Pipeline, path: str | Path) -> None:
    """Persist trained model to disk with joblib."""
    joblib.dump(clf, path)


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted model from disk."""
    return joblib.load(path)


class ClassicalDetector:
    """High-level wrapper for the SM2026 XGBoost deepfake detector.

    Example::

        detector = ClassicalDetector()
        detector.fit_csv("tests/fixtures/osr_features.csv")
        label, conf = detector.predict_audio("sample.wav")
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._clf: Pipeline | None = None
        if model_path is not None:
            self._clf = load_model(model_path)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassicalDetector:
        self._clf = train(X, y)
        return self

    def fit_csv(self, path: str | Path) -> ClassicalDetector:
        X, y = load_csv_dataset(path)
        return self.fit(X, y)

    def predict_features(self, features: np.ndarray) -> tuple[str, float]:
        """Predict from a pre-extracted 18-dim feature vector.

        Returns (label, confidence) where label ∈ {'real', 'fake'}.
        """
        if self._clf is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        features = features.reshape(1, -1)
        label_int = int(predict(self._clf, features)[0])
        prob = float(predict_proba(self._clf, features)[0, label_int])
        return INV_LABEL_MAP[label_int], prob

    def predict_audio(self, path: str | Path) -> tuple[str, float]:
        """Load audio, extract features, and return (label, confidence)."""
        from voiceguard.features.extractor import extract_from_file

        features = extract_from_file(path)
        return self.predict_features(features)

    def save(self, path: str | Path) -> None:
        if self._clf is None:
            raise RuntimeError("Model not trained.")
        save_model(self._clf, path)

    @classmethod
    def from_file(cls, path: str | Path) -> ClassicalDetector:
        return cls(model_path=path)
