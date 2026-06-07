"""Tests for RawBoost waveform augmentation (voiceguard.training.rawboost)."""

from __future__ import annotations

import numpy as np
import pytest

from voiceguard.training.rawboost import RawBoost, apply_rawboost

ALGOS = [1, 2, 3, 4, 5, 6]
FS = 16000


def _signal(n: int = FS) -> np.ndarray:
    """A 200 Hz sine — a stand-in for a normalised speech clip."""
    t = np.arange(n) / FS
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


@pytest.mark.parametrize("algo", ALGOS)
def test_apply_rawboost_shape_and_finite(algo: int) -> None:
    np.random.seed(0)
    x = _signal()
    y = apply_rawboost(x.copy(), fs=FS, algo=algo)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


@pytest.mark.parametrize("algo", ALGOS)
def test_rawboost_preserves_signal(algo: int) -> None:
    """Regression guard: a band-pass/band-stop mix-up once annihilated the
    signal (corr ~0). Every algorithm must keep the waveform correlated with
    the input."""
    np.random.seed(1)
    x = _signal()
    y = apply_rawboost(x.copy(), fs=FS, algo=algo)
    n = min(len(x), len(y))
    corr = np.corrcoef(x[:n], y[:n])[0, 1]
    assert corr > 0.3, f"algo {algo} decorrelated the signal (corr={corr:.3f})"
    assert y.std() > 1e-3, f"algo {algo} collapsed the signal (rms={y.std():.4f})"


def test_random_algo_selection_is_valid() -> None:
    np.random.seed(2)
    x = _signal()
    for _ in range(10):
        y = apply_rawboost(x.copy(), fs=FS, algo=0)  # 0 = random pick
        assert y.shape == x.shape and np.isfinite(y).all()


def test_rawboost_callable_probability() -> None:
    import torch

    x = torch.from_numpy(_signal())
    # p=0 → never augment → identical tensor returned
    assert torch.equal(RawBoost(p=0.0, fs=FS)(x), x)
    # p=1 → always augment → output differs but keeps shape/dtype
    np.random.seed(3)
    out = RawBoost(p=1.0, algo=3, fs=FS)(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert not torch.equal(out, x)
