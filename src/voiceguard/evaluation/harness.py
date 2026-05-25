"""
Evaluation harness for VoiceGuard models.

Works with any model that implements a common predict interface via
the ModelWrapper protocol. Supports classical ML, DSFNet, and Wav2Vec2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from voiceguard.evaluation.metrics import compute_all_metrics


class ModelWrapper(Protocol):
    """Protocol that all evaluatable models must satisfy."""

    def predict_batch(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Predict a batch of audio samples.

        Args:
            inputs: List of mono float32 audio arrays.

        Returns:
            (predictions, scores) — integer labels and fake-class probabilities.
        """
        ...


class EvaluationResult:
    """Container for evaluation results with JSON and markdown serialisation."""

    def __init__(self, metrics: dict, confusion_matrix: np.ndarray, latency_ms: dict) -> None:
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        cm = self.confusion_matrix.tolist() if self.confusion_matrix is not None else None
        return {
            **self.metrics,
            "confusion_matrix": cm,
            "latency_ms": self.latency_ms,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_markdown(self) -> str:
        lines = ["| Metric | Value |", "|--------|-------|"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
        if self.latency_ms:
            p95 = self.latency_ms.get("p95", float("nan"))
            lines.append(f"| latency_p95_ms | {p95:.1f} |")
        return "\n".join(lines)


def evaluate(
    model: ModelWrapper,
    dataset: list[tuple[np.ndarray, int]],
    batch_size: int = 32,
    output_path: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate a model on a dataset.

    Args:
        model: Any object satisfying the ModelWrapper protocol.
        dataset: List of (audio_array, label) tuples. label: 0=real, 1=fake.
        batch_size: Inference batch size.
        output_path: If set, save JSON results to this path.

    Returns:
        EvaluationResult with metrics, confusion matrix, and latency stats.
    """
    from sklearn.metrics import confusion_matrix

    all_preds: list[int] = []
    all_scores: list[float] = []
    all_labels: list[int] = []
    latencies_ms: list[float] = []

    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        t0 = time.perf_counter()
        preds, scores = model.predict_batch(inputs)
        elapsed_ms = (time.perf_counter() - t0) / len(inputs) * 1000

        all_preds.extend(preds.tolist())
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)
        latencies_ms.append(elapsed_ms)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    scores_arr = np.array(all_scores)

    metrics = compute_all_metrics(y_true, y_pred, scores_arr)
    cm = confusion_matrix(y_true, y_pred)

    latency_arr = np.array(latencies_ms)
    latency_stats = {
        "mean": float(np.mean(latency_arr)),
        "p50": float(np.percentile(latency_arr, 50)),
        "p95": float(np.percentile(latency_arr, 95)),
        "p99": float(np.percentile(latency_arr, 99)),
    }

    result = EvaluationResult(metrics=metrics, confusion_matrix=cm, latency_ms=latency_stats)

    if output_path is not None:
        result.save_json(output_path)

    return result
