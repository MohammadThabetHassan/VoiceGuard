"""Adversarial robustness evaluation — clean vs FGSM vs PGD accuracy.

Quantifies how a detector holds up under inaudible white-box perturbations
(L∞ budget ``epsilon``). A large gap between clean accuracy and PGD accuracy
means the model is adversarially fragile.

The attack primitives live in :mod:`voiceguard.training.adversarial`; this
module just sweeps a labelled set through clean / FGSM / PGD and tallies
accuracy. It is model-agnostic: any ``nn.Module`` whose ``forward`` takes the
input tensor and returns ``(B, n_classes)`` logits works (DSFNet, SSLAASIST, …).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from voiceguard.training.adversarial import fgsm_attack, pgd_attack


def measure_robustness(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.002,
    pgd_steps: int = 10,
    batch_size: int = 8,
) -> dict:
    """Return clean / FGSM / PGD accuracy of *model* on ``(x, y)``.

    Args:
        model: classifier returning ``(B, n_classes)`` logits from *x*-shaped input.
        x: ``(N, ...)`` inputs the model accepts (e.g. ``(N, T)`` waveforms).
        y: ``(N,)`` integer class labels.
        epsilon: L∞ perturbation budget for both attacks.
        alpha: PGD per-step size.
        pgd_steps: PGD iterations (FGSM is always single-step).
        batch_size: evaluation batch size.

    Returns:
        dict with ``n``, ``clean_acc``, ``fgsm_acc``, ``pgd_acc`` (all in [0, 1]),
        plus the ``epsilon`` / ``pgd_steps`` used. ``pgd_acc`` ≪ ``clean_acc``
        indicates adversarial fragility.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must align on dim 0: {x.shape[0]} vs {y.shape[0]}")
    model.eval()
    device = next(model.parameters()).device
    n = int(x.shape[0])
    clean_c = fgsm_c = pgd_c = 0

    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)
        with torch.no_grad():
            clean_c += int((model(xb).argmax(1) == yb).sum())
        x_fgsm = fgsm_attack(model, xb, yb, epsilon=epsilon)
        x_pgd = pgd_attack(model, xb, yb, epsilon=epsilon, alpha=alpha, n_steps=pgd_steps)
        with torch.no_grad():
            fgsm_c += int((model(x_fgsm).argmax(1) == yb).sum())
            pgd_c += int((model(x_pgd).argmax(1) == yb).sum())

    return {
        "n": n,
        "clean_acc": round(clean_c / n, 4),
        "fgsm_acc": round(fgsm_c / n, 4),
        "pgd_acc": round(pgd_c / n, 4),
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
    }
