"""Real-world robustness evaluation harness.

Benchmark EER (ASVspoof) measures in-distribution accuracy. This harness instead
measures how the detector behaves on *real-world* audio: genuine human speech and
deepfakes, each replayed under several acoustic conditions (clean, noisy,
telephony-band, short). It reports the two numbers that matter in production:

    real-pass-rate     = % of genuine speech correctly accepted (1 - false-positive)
    fake-detection-rate = % of deepfakes correctly flagged

Usage:
    PYTHONPATH=src python -m voiceguard.evaluation.realworld_eval \
        --ckpt <model_best.pt> --arch ssl_aasist \
        --real-dir <dir-of-real-wavs> --fake-dir <dir-of-fake-wavs> \
        --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f

SR = 16000
TARGET_LEN = SR * 3
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def _load(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    wav = np.asarray(data, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio

        t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(t, sr, SR).squeeze(0).numpy()
    return wav


# ── acoustic conditions ─────────────────────────────────────────────────────


def _clean(w: np.ndarray) -> np.ndarray:
    return w


def _noisy(w: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    sig_pow = float(np.mean(w**2)) + 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = np.random.randn(len(w)).astype(np.float32) * np.sqrt(noise_pow)
    return (w + noise).astype(np.float32)


def _telephony(w: np.ndarray) -> np.ndarray:
    # Band-limit to ~300–3400 Hz and µ-law quantise — a phone-call channel.
    wt = torch.from_numpy(w).unsqueeze(0)
    import torchaudio

    wt = torchaudio.functional.highpass_biquad(wt, SR, 300.0)
    wt = torchaudio.functional.lowpass_biquad(wt, SR, 3400.0)
    x = wt.squeeze(0)
    mu = 255.0
    companded = torch.sign(x) * torch.log1p(mu * x.abs()) / np.log1p(mu)
    quantised = torch.round((companded + 1) / 2 * mu)
    decoded = 2 * (quantised / mu) - 1
    expanded = torch.sign(decoded) * (1 / mu) * ((1 + mu) ** decoded.abs() - 1)
    return expanded.numpy().astype(np.float32)


def _short(w: np.ndarray) -> np.ndarray:
    return w[: SR * 2]


CONDITIONS = {"clean": _clean, "noisy": _noisy, "telephony": _telephony, "short": _short}


def _to_input(w: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(w).float()
    if t.shape[0] < TARGET_LEN:
        t = torch_f.pad(t, (0, TARGET_LEN - t.shape[0]))
    else:
        t = t[:TARGET_LEN]
    return t.unsqueeze(0)


def _load_model(ckpt: str, arch: str, model_name: str, device: str):
    if arch == "ssl_aasist":
        from voiceguard.models.ssl_classifier import SSLAASIST

        model = SSLAASIST(model_name)
    else:
        from voiceguard.models.ssl_classifier import SSLClassifier

        model = SSLClassifier(model_name)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state", state), strict=False)
    return model.eval().to(device)


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in _AUDIO_EXTS)


def evaluate(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(args.ckpt, args.arch, args.model_name, device)

    real = [p for d in args.real_dir for p in _files(Path(d))]
    fake = [p for d in args.fake_dir for p in _files(Path(d))]
    report: dict = {"ckpt": args.ckpt, "n_real": len(real), "n_fake": len(fake), "conditions": {}}

    for cond, fn in CONDITIONS.items():
        scores = {"real": [], "fake": []}
        for kind, paths in (("real", real), ("fake", fake)):
            for p in paths:
                try:
                    w = fn(_load(p))
                except Exception:  # noqa: S112
                    continue
                with torch.no_grad():
                    prob = torch.softmax(model(_to_input(w).to(device)), -1)[0]
                scores[kind].append(float(prob[1]))  # fake probability
        real_arr, fake_arr = np.array(scores["real"]), np.array(scores["fake"])
        report["conditions"][cond] = {
            "real_pass_rate": round(float(np.mean(real_arr < 0.5)) * 100, 1)
            if real_arr.size
            else None,
            "fake_detection_rate": round(float(np.mean(fake_arr >= 0.5)) * 100, 1)
            if fake_arr.size
            else None,
            "real_mean_fake_score": round(float(np.mean(real_arr)), 3) if real_arr.size else None,
            "fake_mean_fake_score": round(float(np.mean(fake_arr)), 3) if fake_arr.size else None,
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Real-world robustness evaluation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", default="ssl_aasist", choices=["ssl_aasist", "ssl"])
    p.add_argument("--model-name", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--real-dir", nargs="+", required=True)
    p.add_argument("--fake-dir", nargs="+", required=True)
    p.add_argument("--out", default="realworld_report.json")
    args = p.parse_args()

    report = evaluate(args)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
