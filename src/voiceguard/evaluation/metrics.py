"""
Evaluation metrics for voice deepfake detection.

Implements EER, minDCF, ROC curve generation, and a standard metrics dict.
"""

from __future__ import annotations

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate.

    Args:
        scores: Continuous fake-class probability scores (higher = more likely fake).
        labels: Binary ground-truth labels (1=fake, 0=real).

    Returns:
        EER as a fraction in [0, 1].
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find crossover point where FPR ≈ FNR
    abs_diff = np.abs(fnr - fpr)
    idx = int(np.argmin(abs_diff))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute minimum Detection Cost Function (minDCF).

    DCF(θ) = C_miss·P_miss(θ)·P_target + C_fa·P_fa(θ)·(1 - P_target)

    Args:
        scores: Fake-class probability scores.
        labels: Binary ground-truth labels (1=fake, 0=real).
        p_target: Prior probability of target class (default 0.01).
        c_miss: Cost of a miss (default 1.0).
        c_fa: Cost of a false alarm (default 1.0).

    Returns:
        minDCF as a float.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    norm = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(dcf) / (norm + 1e-12))


def compute_roc(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    return fpr, tpr, thresholds


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute the full metrics suite.

    Args:
        y_true: Ground-truth integer labels (0=real, 1=fake).
        y_pred: Predicted integer labels.
        scores: Fake-class probability scores.

    Returns:
        Dict with: accuracy, f1, precision, recall, roc_auc, eer, min_dcf.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "eer": compute_eer(scores, y_true),
        "min_dcf": compute_min_dcf(scores, y_true),
    }
