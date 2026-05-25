"""
Evaluation harness tests.

Uses a mock model and synthetic dataset — no real audio or GPU required.
"""

import numpy as np

from voiceguard.evaluation.harness import EvaluationResult, evaluate
from voiceguard.evaluation.metrics import (
    compute_all_metrics,
    compute_eer,
    compute_min_dcf,
    compute_roc,
)


class PerfectMockModel:
    """Mock model with perfect predictions."""

    def predict_batch(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = len(inputs)
        # Alternate real/fake to match synthetic dataset
        preds = np.array([i % 2 for i in range(n)])
        scores = np.array([0.9 if i % 2 == 1 else 0.1 for i in range(n)])
        return preds, scores


class RandomMockModel:
    """Mock model with random predictions."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def predict_batch(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        n = len(inputs)
        scores = self.rng.uniform(0, 1, n)
        preds = (scores > 0.5).astype(int)
        return preds, scores


def make_synthetic_dataset(n: int = 100) -> list[tuple[np.ndarray, int]]:
    """Alternating real/fake samples with 1-second synthetic audio."""
    rng = np.random.default_rng(42)
    dataset = []
    for i in range(n):
        audio = rng.standard_normal(16000).astype(np.float32)
        label = i % 2  # 0=real, 1=fake alternating
        dataset.append((audio, label))
    return dataset


# ── Metric unit tests ─────────────────────────────────────────────────────────


def test_eer_perfect_separation():
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 1, 0, 0])
    eer = compute_eer(scores, labels)
    assert eer < 0.05, f"EER should be near 0 for perfect separation, got {eer}"


def test_eer_random():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 200)
    labels = rng.integers(0, 2, 200)
    eer = compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0


def test_min_dcf_range():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 200)
    labels = rng.integers(0, 2, 200)
    dcf = compute_min_dcf(scores, labels)
    assert 0.0 <= dcf <= 10.0


def test_compute_roc_shapes():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    fpr, tpr, thresholds = compute_roc(scores, labels)
    # sklearn roc_curve: fpr and tpr have len(thresholds)+1 elements
    assert len(fpr) == len(tpr)
    assert len(thresholds) >= 1


def test_compute_all_metrics_keys():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_all_metrics(y_true, y_pred, scores)
    expected = {"accuracy", "f1", "precision", "recall", "roc_auc", "eer", "min_dcf"}
    assert expected.issubset(set(metrics.keys()))


# ── Harness integration tests ─────────────────────────────────────────────────


def test_evaluate_returns_result():
    dataset = make_synthetic_dataset(20)
    model = RandomMockModel()
    result = evaluate(model, dataset, batch_size=8)
    assert isinstance(result, EvaluationResult)


def test_evaluate_metrics_range():
    dataset = make_synthetic_dataset(50)
    model = RandomMockModel()
    result = evaluate(model, dataset, batch_size=16)
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["f1"] <= 1.0
    assert 0.0 <= result.metrics["eer"] <= 1.0


def test_evaluate_latency_populated():
    dataset = make_synthetic_dataset(20)
    model = RandomMockModel()
    result = evaluate(model, dataset, batch_size=8)
    assert "p95" in result.latency_ms
    assert result.latency_ms["p95"] > 0


def test_evaluation_result_to_dict():
    dataset = make_synthetic_dataset(20)
    model = RandomMockModel()
    result = evaluate(model, dataset)
    d = result.to_dict()
    assert "confusion_matrix" in d
    assert "latency_ms" in d


def test_evaluation_result_markdown():
    dataset = make_synthetic_dataset(20)
    model = RandomMockModel()
    result = evaluate(model, dataset)
    md = result.to_markdown()
    assert "| accuracy |" in md
    assert "| f1 |" in md


def test_evaluate_json_output(tmp_path):
    dataset = make_synthetic_dataset(20)
    model = RandomMockModel()
    out_path = tmp_path / "results.json"
    evaluate(model, dataset, output_path=out_path)
    assert out_path.exists()
    import json

    data = json.loads(out_path.read_text())
    assert "accuracy" in data
