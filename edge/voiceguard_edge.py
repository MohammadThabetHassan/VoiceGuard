#!/usr/bin/env python3
"""VoiceGuard edge detector — runs the 0.62 MB INT8 DSFNetTiny on a Raspberry Pi.

Dependencies: onnxruntime + numpy + soundfile ONLY. No torch, no transformers, no
librosa — the mel-spectrogram is reproduced in pure NumPy to byte-match the training
front-end (torchaudio MelSpectrogram sr=16k, n_fft=400, hop=160, n_mels=80, power=2,
htk, norm=None → AmplitudeToDB power, top_db=80).

Usage:
    python3 voiceguard_edge.py <audio.wav>                 # detect one file
    python3 voiceguard_edge.py --benchmark [n]             # latency benchmark
    python3 voiceguard_edge.py <audio.wav> --model path.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 16000
CLIP = 48000  # 3 s
N_FFT = 400  # 0.025 s
HOP = 160  # 0.010 s
N_MELS = 80
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "dsfnet_tiny_int8.onnx")


# ── pure-numpy mel front-end (matches torchaudio MelSpectrogram + AmplitudeToDB) ──
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_freqs=N_FFT // 2 + 1, n_mels=N_MELS, sr=SR):
    """htk mel, norm=None triangular filters → (n_freqs, n_mels), peak 1.0."""
    f_max = sr / 2.0
    all_freqs = np.linspace(0, f_max, n_freqs)
    m_pts = np.linspace(0.0, _hz_to_mel(f_max), n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        up = (all_freqs - lo) / (ctr - lo)
        down = (hi - all_freqs) / (hi - ctr)
        fb[:, i] = np.maximum(0.0, np.minimum(up, down))
    return fb


_FB = _mel_filterbank()
# periodic Hann window (matches torch.hann_window default periodic=True)
_WIN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """(CLIP,) float32 → (1, 1, N_MELS, 301) float32, in dB."""
    pad = N_FFT // 2
    p = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(wav)) // HOP
    frames = np.stack([p[i * HOP : i * HOP + N_FFT] * _WIN for i in range(n_frames)], axis=0)
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2.0  # power, (n_frames, n_freqs)
    mel = spec @ _FB  # (n_frames, n_mels)
    mel = mel.T  # (n_mels, n_frames)
    # AmplitudeToDB(power, top_db=80): 10*log10(max(mel,1e-10)), clamp to max-80
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    return db.astype(np.float32)[None, None, :, :]


# ── audio loading ────────────────────────────────────────────────────────────────
def _resample_linear(x, src, dst):
    if src == dst:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def load_audio(path: str) -> np.ndarray:
    data, sr = sf.read(path, always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = _resample_linear(data, sr, SR)
    if len(data) < CLIP:
        data = np.pad(data, (0, CLIP - len(data)))
    return data[:CLIP]


# ── inference ──────────────────────────────────────────────────────────────────--
def make_session(model_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0  # let ORT pick (good on Pi's few cores)
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


def detect(sess, wav: np.ndarray):
    waveform = wav.reshape(1, 1, CLIP).astype(np.float32)
    spec = mel_spectrogram(wav)
    logits = sess.run(["logits"], {"waveform": waveform, "spectrogram": spec})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    fake_p = float(probs[1])
    label = "fake" if fake_p >= 0.5 else "real"
    return label, fake_p


def main():
    ap = argparse.ArgumentParser(description="VoiceGuard edge detector (Raspberry Pi)")
    ap.add_argument("audio", nargs="?", help="audio file to classify")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--benchmark", nargs="?", const=50, type=int, metavar="N")
    args = ap.parse_args()

    size_kb = os.path.getsize(args.model) / 1024
    sess = make_session(args.model)

    if args.benchmark is not None:
        rng = np.random.default_rng(0)
        wav = (0.1 * rng.standard_normal(CLIP)).astype(np.float32)
        for _ in range(5):
            detect(sess, wav)  # warmup
        ts = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            detect(sess, wav)
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        print(f"VoiceGuard edge — INT8 DSFNetTiny ({size_kb:.0f} KB)")
        print(f"  runs={len(ts)}  p50={np.percentile(ts, 50):.1f} ms  "
              f"p95={np.percentile(ts, 95):.1f} ms  mean={ts.mean():.1f} ms")
        print(f"  real-time factor (3 s clip): {ts.mean() / 1000 / 3:.4f}  (<<1 = real-time)")
        return

    if not args.audio:
        ap.error("provide an audio file or --benchmark")
    wav = load_audio(args.audio)
    t0 = time.perf_counter()
    label, fake_p = detect(sess, wav)
    ms = (time.perf_counter() - t0) * 1000
    conf = fake_p if label == "fake" else 1 - fake_p
    print(f"  file:       {os.path.basename(args.audio)}")
    print(f"  verdict:    {label.upper()}  ({conf * 100:.1f}% confidence)")
    print(f"  fake_prob:  {fake_p:.4f}")
    print(f"  latency:    {ms:.1f} ms   |   model: {size_kb:.0f} KB INT8 on CPU")


if __name__ == "__main__":
    sys.exit(main())
