"""
RawBoost data augmentation (Tak et al., ICASSP 2022) for anti-spoofing.

Three signal-processing noise components applied directly to the raw
waveform — no external noise/RIR databases needed:

  1. LnL convolutive noise   — linear+nonlinear multi-band FIR colouring
  2. ISD additive noise      — impulsive, signal-dependent perturbations
  3. SSI additive noise      — stationary, signal-independent coloured noise

Why we add it on top of the existing GPU `AudioAugment`: AudioAugment already
covers µ-law / bitcrush / bandpass / packet-loss / reverb (codec/channel),
but underweights impulsive (ISD) and stationary coloured-noise (SSI) cases.
RawBoost is therefore used here primarily as an **OOD-robustness** lever, not
an in-distribution EER lever (per-codec EER is already 2-3%).

Applied per-sample on CPU inside the Dataset (random algorithm per call).
Reference: github.com/TakHemlata/RawBoost-antispoofing (Apache-2.0).
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _rand_range(x1: float, x2: float, integer: bool) -> float:
    y = np.random.uniform(low=x1, high=x2, size=(1,))[0]
    return int(y) if integer else y


def _norm_wav(x: np.ndarray, always: bool) -> np.ndarray:
    peak = np.amax(np.abs(x))
    if peak < 1e-12:
        return x
    if always or peak > 1.0:
        x = x / peak
    return x


def _gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw,
                      min_coeff, max_coeff, min_g, max_g, fs) -> np.ndarray:
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = _rand_range(min_f, max_f, False)
        bw = _rand_range(min_bw, max_bw, False)
        c = _rand_range(min_coeff, max_coeff, True)
        if c % 2 == 0:
            c += 1
        f1 = max(fc - bw / 2.0, 1 / 1000.0)
        f2 = min(fc + bw / 2.0, fs / 2.0 - 1 / 1000.0)
        if f2 <= f1:
            continue
        # NOTCH (band-stop): keep everything except [f1,f2]. firwin's default
        # pass_zero=True gives passbands [0,f1] & [f2,nyq]. Using pass_zero=False
        # (bandpass) here annihilates the signal (corr~0) — the original bug.
        fir = signal.firwin(int(c), [float(f1), float(f2)],
                            window="hamming", fs=fs, pass_zero=True)
        b = np.convolve(fir, b)
    g = _rand_range(min_g, max_g, False)
    _, h = signal.freqz(b, 1, fs=fs)
    denom = np.amax(np.abs(h))
    if denom < 1e-12:
        return b
    return pow(10, g / 20.0) * b / denom


def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[0] + 1
    xpad = np.pad(x, (0, n), "constant")
    y = signal.lfilter(b, [1.0], xpad)
    return y[int(n / 2): int(y.shape[0] - n / 2)]


def _lnl_convolutive_noise(x, n_f, n_bands, min_f, max_f, min_bw, max_bw,
                           min_coeff, max_coeff, min_g, max_g,
                           min_bias, max_bias, fs) -> np.ndarray:
    y = np.zeros_like(x)
    for i in range(n_f):
        g_lo, g_hi = (min_g - min_bias, max_g - max_bias) if i == 1 else (min_g, max_g)
        b = _gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw,
                              min_coeff, max_coeff, g_lo, g_hi, fs)
        y = y + _filter_fir(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    return _norm_wav(y, False)


def _isd_additive_noise(x, p_pct, g_sd) -> np.ndarray:
    beta = _rand_range(0, p_pct, False)
    y = x.copy()
    n = int(x.shape[0] * (beta / 100.0))
    if n > 0:
        idx = np.random.permutation(x.shape[0])[:n]
        f_r = ((2 * np.random.rand(n)) - 1) * ((2 * np.random.rand(n)) - 1)
        y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _norm_wav(y, False)


def _ssi_additive_noise(x, snr_min, snr_max, n_bands, min_f, max_f, min_bw, max_bw,
                        min_coeff, max_coeff, min_g, max_g, fs) -> np.ndarray:
    noise = np.random.normal(0, 1, x.shape[0])
    b = _gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw,
                          min_coeff, max_coeff, min_g, max_g, fs)
    noise = _norm_wav(_filter_fir(noise, b), True)
    snr = _rand_range(snr_min, snr_max, False)
    nn = np.linalg.norm(noise, 2)
    if nn < 1e-12:
        return x
    noise = noise / nn * np.linalg.norm(x, 2) / (10.0 ** (0.05 * snr))
    return x + noise


# Default hyperparameters from the reference RawBoost LA configuration.
_DEFAULTS = dict(
    n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000,
    min_coeff=10, max_coeff=100, min_g=0, max_g=0,
    min_bias=5, max_bias=20, n_f=5, p_pct=10, g_sd=2,
    snr_min=10, snr_max=40,
)


def apply_rawboost(x: np.ndarray, fs: int = 16000, algo: int = 4,
                   **kw) -> np.ndarray:
    """Apply a RawBoost algorithm to a 1-D float waveform.

    algo: 1=LnL conv, 2=ISD, 3=SSI, 4=series(1->2->3), 5=series(1->2),
          6=parallel(1+2), 0=random pick of {1,2,3,4,5}.
    """
    p = {**_DEFAULTS, **kw}
    x = np.asarray(x, dtype=np.float64)
    if algo == 0:
        algo = int(np.random.choice([1, 2, 3, 4, 5]))

    def lnl(s):
        return _lnl_convolutive_noise(s, p["n_f"], p["n_bands"], p["min_f"], p["max_f"],
                                      p["min_bw"], p["max_bw"], p["min_coeff"], p["max_coeff"],
                                      p["min_g"], p["max_g"], p["min_bias"], p["max_bias"], fs)

    def isd(s):
        return _isd_additive_noise(s, p["p_pct"], p["g_sd"])

    def ssi(s):
        return _ssi_additive_noise(s, p["snr_min"], p["snr_max"], p["n_bands"], p["min_f"],
                                   p["max_f"], p["min_bw"], p["max_bw"], p["min_coeff"],
                                   p["max_coeff"], p["min_g"], p["max_g"], fs)

    if algo == 1:
        y = lnl(x)
    elif algo == 2:
        y = isd(x)
    elif algo == 3:
        y = ssi(x)
    elif algo == 4:
        y = ssi(isd(lnl(x)))
    elif algo == 5:
        y = isd(lnl(x))
    elif algo == 6:
        y = _norm_wav(lnl(x) + isd(x), False)
    else:
        y = x
    return y.astype(np.float32)


class RawBoost:
    """Stochastic per-sample RawBoost callable for use in a Dataset.

    Args:
        p: probability of applying RawBoost to a given sample.
        algo: RawBoost algorithm id (0 = random per call).
        fs: sample rate.
    """

    def __init__(self, p: float = 0.5, algo: int = 0, fs: int = 16000):
        self.p = p
        self.algo = algo
        self.fs = fs

    def __call__(self, wav):
        """wav: 1-D torch.Tensor or np.ndarray (float). Returns same type."""
        import torch
        is_tensor = isinstance(wav, torch.Tensor)
        if np.random.rand() > self.p:
            return wav
        arr = wav.detach().cpu().numpy() if is_tensor else np.asarray(wav)
        out = apply_rawboost(arr.reshape(-1), fs=self.fs, algo=self.algo)
        return torch.from_numpy(out).to(wav.dtype) if is_tensor else out
