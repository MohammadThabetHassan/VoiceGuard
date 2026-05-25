"""Tests for PGD adversarial attack and adversarial training utilities."""

from __future__ import annotations

import pytest
import torch

from voiceguard.training.adversarial import adversarial_training_step, fgsm_attack, pgd_attack


@pytest.fixture
def model_and_inputs():
    from voiceguard.models.dsfnet import DSFNet

    model = DSFNet()
    model.eval()
    B, T = 2, 16000  # 1-second batch
    waveform = torch.randn(B, 1, T) * 0.1
    labels = torch.tensor([0, 1])
    return model, waveform, labels


def test_pgd_output_shape(model_and_inputs):
    model, waveform, labels = model_and_inputs
    adv = pgd_attack(model, waveform, labels, epsilon=0.01, n_steps=3)
    assert adv.shape == waveform.shape


def test_pgd_within_epsilon(model_and_inputs):
    model, waveform, labels = model_and_inputs
    epsilon = 0.01
    adv = pgd_attack(model, waveform, labels, epsilon=epsilon, n_steps=3)
    diff = (adv - waveform).abs().max().item()
    assert diff <= epsilon + 1e-5


def test_pgd_clipped_to_valid_range(model_and_inputs):
    model, waveform, labels = model_and_inputs
    adv = pgd_attack(model, waveform, labels, epsilon=0.01, n_steps=3)
    assert adv.min() >= -1.0 - 1e-5
    assert adv.max() <= 1.0 + 1e-5


def test_pgd_no_inplace_modification(model_and_inputs):
    model, waveform, labels = model_and_inputs
    original = waveform.clone()
    pgd_attack(model, waveform, labels, epsilon=0.01, n_steps=3)
    assert torch.allclose(waveform, original)


def test_pgd_detached_output(model_and_inputs):
    model, waveform, labels = model_and_inputs
    adv = pgd_attack(model, waveform, labels, epsilon=0.01, n_steps=3)
    assert not adv.requires_grad


def test_fgsm_shape(model_and_inputs):
    model, waveform, labels = model_and_inputs
    adv = fgsm_attack(model, waveform, labels, epsilon=0.01)
    assert adv.shape == waveform.shape


def test_fgsm_within_epsilon(model_and_inputs):
    model, waveform, labels = model_and_inputs
    epsilon = 0.01
    adv = fgsm_attack(model, waveform, labels, epsilon=epsilon)
    diff = (adv - waveform).abs().max().item()
    assert diff <= epsilon + 1e-5


def test_adversarial_training_step_returns_float(model_and_inputs):
    model, waveform, labels = model_and_inputs
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss = adversarial_training_step(model, optimizer, waveform, labels, epsilon=0.005, pgd_steps=2)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_adversarial_training_step_updates_params(model_and_inputs):
    model, waveform, labels = model_and_inputs
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    params_before = [p.clone() for p in model.parameters()]
    adversarial_training_step(model, optimizer, waveform, labels, epsilon=0.005, pgd_steps=2)
    params_after = list(model.parameters())
    changed = any(
        not torch.allclose(b, a) for b, a in zip(params_before, params_after, strict=True)
    )
    assert changed


def test_pgd_zero_epsilon_identity(model_and_inputs):
    """With epsilon=0, adversarial should equal original."""
    model, waveform, labels = model_and_inputs
    adv = pgd_attack(model, waveform, labels, epsilon=0.0, alpha=0.0, n_steps=3, random_start=False)  # noqa: E501
    assert torch.allclose(adv, waveform, atol=1e-6)
