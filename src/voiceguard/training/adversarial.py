"""PGD adversarial attack and adversarial training utilities for DSFNet.

Reference: Madry et al. (2018) "Towards Deep Learning Models Resistant to
Adversarial Attacks" — https://arxiv.org/abs/1706.06083
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def pgd_attack(
    model: nn.Module,
    waveform: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.001,
    n_steps: int = 10,
    random_start: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) white-box attack on *waveform*.

    Args:
        model: DSFNet (or any nn.Module) that accepts waveform as first arg.
        waveform: (B, 1, T) float32 input in [-1, 1].
        labels: (B,) integer target labels.
        epsilon: L∞ perturbation budget (default 0.01 ≈ inaudible).
        alpha: Per-step size.
        n_steps: Number of PGD iterations.
        random_start: Start from random point in ε-ball.

    Returns:
        Adversarial waveform of same shape as *waveform*, clamped to [-1, 1].
    """
    model.eval()
    waveform = waveform.detach().clone()

    if random_start:
        delta = torch.empty_like(waveform).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(waveform)

    delta = delta.requires_grad_(True)

    for _ in range(n_steps):
        adv = waveform + delta
        logits = model(adv)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (waveform + delta.data).clamp(-1.0, 1.0) - waveform
        delta.grad.zero_()

    return (waveform + delta.detach()).clamp(-1.0, 1.0)


def adversarial_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    waveform: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    pgd_steps: int = 3,
) -> float:
    """Single adversarial training step using PGD-generated examples.

    Generates adversarial examples, then computes loss on the mix of clean
    and adversarial inputs (50/50 split per batch).

    Returns:
        Scalar training loss (float).
    """
    model.train()

    adv_waveform = pgd_attack(
        model,
        waveform,
        labels,
        epsilon=epsilon,
        n_steps=pgd_steps,
        random_start=True,
    )

    # Mix clean + adversarial
    mixed = torch.cat([waveform, adv_waveform], dim=0)
    mixed_labels = torch.cat([labels, labels], dim=0)

    optimizer.zero_grad()
    logits = model(mixed)
    loss = F.cross_entropy(logits, mixed_labels)
    loss.backward()
    optimizer.step()

    return float(loss.item())


def fgsm_attack(
    model: nn.Module,
    waveform: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Fast Gradient Sign Method (FGSM) — single-step PGD for quick testing."""
    return pgd_attack(
        model,
        waveform,
        labels,
        epsilon=epsilon,
        alpha=epsilon,
        n_steps=1,
        random_start=False,
    )
