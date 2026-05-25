"""
Feature extraction for voice deepfake detection.

Produces an 18-dimensional feature vector:
  MFCC1..MFCC13  — mean Mel-frequency cepstral coefficients
  centroid        — spectral centroid (Hz)
  bandwidth       — spectral bandwidth (Hz)
  rolloff         — 85% spectral rolloff frequency (Hz)
  jitter          — relative average perturbation of F0 periods
  shimmer         — mean amplitude variation between frames (dB)

This matches the published SM2026 baseline feature schema exactly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import scipy.fftpack as fftpack
from scipy import signal as scipy_signal
from scipy.io import wavfile

FEATURE_COLS: list[str] = [
    "MFCC1",
    "MFCC2",
    "MFCC3",
    "MFCC4",
    "MFCC5",
    "MFCC6",
    "MFCC7",
    "MFCC8",
    "MFCC9",
    "MFCC10",
    "MFCC11",
    "MFCC12",
    "MFCC13",
    "centroid",
    "bandwidth",
    "rolloff",
    "jitter",
    "shimmer",
]


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file to mono float32 at target_sr.

    Supports WAV files directly; uses scipy resampling.
    Returns (signal, sample_rate).
    """
    path = Path(path)
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    mx = np.max(np.abs(data))
    if mx > 0:
        data /= mx
    if sr != target_sr:
        new_len = int(len(data) * target_sr / sr)
        data = scipy_signal.resample(data, new_len)
        sr = target_sr
    return data, sr


def _mfcc(segment: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Return mean MFCC vector (n_mfcc dims) with pre-emphasis and mel filterbank."""
    pre_emphasis = 0.97
    emph = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])

    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(emph)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(emph, np.zeros(max(0, pad_len - n)))

    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)] * np.hamming(frame_len)

    nfft = 512
    mag = np.abs(np.fft.rfft(frames, nfft))
    power = (1.0 / nfft) * mag**2

    nfilt = 26
    low_mel = 0.0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_f = np.floor((nfft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((nfilt, nfft // 2 + 1))
    for m in range(1, nfilt + 1):
        lo, mid, hi = bin_f[m - 1], bin_f[m], bin_f[m + 1]
        for k in range(lo, mid):
            fbank[m - 1, k] = (k - lo) / (mid - lo + 1e-10)
        for k in range(mid, hi):
            fbank[m - 1, k] = (hi - k) / (hi - mid + 1e-10)

    fb = np.dot(power, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20 * np.log10(fb)
    mfcc = fftpack.dct(fb, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return np.mean(mfcc, axis=0)


def _spectral(segment: np.ndarray, sr: int) -> np.ndarray:
    """Return (centroid, bandwidth, rolloff) — 3 dims."""
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    eps = 1e-10
    total = np.sum(spectrum) + eps
    centroid = np.sum(freqs * spectrum) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total)
    cum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cum, 0.85 * cum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
    return np.array([centroid, bandwidth, rolloff])


def _jitter(segment: np.ndarray, sr: int) -> float:
    """Relative average perturbation of fundamental period (jitter)."""
    eps = 1e-10
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))
    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)]

    periods: list[float] = []
    for frame in frames:
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        corr = corr / (corr[0] + eps)
        min_lag = int(sr / 500)
        max_lag = int(sr / 50)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if max_lag > min_lag:
            peak_idx = min_lag + int(np.argmax(corr[min_lag:max_lag]))
            if corr[peak_idx] > 0.5:
                periods.append(float(peak_idx))

    if len(periods) < 3:
        return 0.0
    arr = np.array(periods)
    return float(np.mean(np.abs(np.diff(arr))) / (np.mean(arr) + eps))


def _shimmer(segment: np.ndarray, sr: int = 16000) -> float:
    """Mean dB amplitude variation between adjacent frames (shimmer)."""
    eps = 1e-10
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))
    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)]
    amplitudes = np.abs(np.mean(frames, axis=1))
    amplitudes = amplitudes[amplitudes > eps]
    if len(amplitudes) < 3:
        return 0.0
    amp_db = 20 * np.log10(amplitudes + eps)
    return float(np.mean(np.abs(np.diff(amp_db))))


def extract_features(segment: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract 18-dim feature vector matching the SM2026 baseline schema.

    Args:
        segment: Mono float32 audio signal.
        sr: Sample rate (default 16000).

    Returns:
        Array of shape (18,): MFCC1..13, centroid, bandwidth, rolloff, jitter, shimmer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = _mfcc(segment, sr)
        spectral = _spectral(segment, sr)
        jitter = _jitter(segment, sr)
        shimmer = _shimmer(segment, sr)
    return np.concatenate([mfcc, spectral, [jitter], [shimmer]])


def extract_from_file(path: str | Path) -> np.ndarray:
    """Load audio file and return 18-dim feature vector."""
    signal, sr = load_audio(path)
    return extract_features(signal, sr)
