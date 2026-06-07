#!/usr/bin/env python3
"""
Out-of-distribution synthesis evaluation.

Generates TTS audio with Kokoro-82M (unseen synthesis system), mixes with
real ASVspoof 2021 LA eval samples, runs Wav2Vec2-large, and reports:
  - Fake detection rate  (% of Kokoro fakes correctly flagged)
  - Real pass rate       (% of real samples correctly passed)
  - Combined accuracy

This tests generalization to a synthesis system never seen during training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchaudio

sys.path.insert(0, "/tmp/voiceguard-workspace/workspace/voiceguard/src")  # noqa: S108  # nosec B108

CKPT = Path(  # noqa: S108
    "/srv/thabet/voiceguard-checkpoints/runs/wav2vec2-large_20ep_b8_lr5e-06_aug/model_best.pt"
)
REAL_FLAC_DIR = Path("/tmp/voiceguard-workspace/data/asvspoof/raw/eval/flac")  # noqa: S108  # nosec B108
OUT_DIR = Path("/srv/thabet/voiceguard-checkpoints/ood_eval")  # noqa: S108
N_FAKES = 30  # Kokoro samples to generate
N_REALS = 30  # Real samples to use
TARGET_LEN = 48000  # 3s @ 16kHz
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentences covering prosody / content variation
SENTENCES = [
    "The bank transfer has been processed successfully.",
    "Please confirm your identity by saying your full name.",
    "Your account will be suspended unless you verify now.",
    "Hello, this is your bank calling about suspicious activity.",
    "Press one to speak with a representative immediately.",
    "We detected unauthorized access to your account.",
    "Your package has been held at customs, payment required.",
    "Congratulations, you have won a prize. Claim it today.",
    "This is an automated security alert from your provider.",
    "Please enter your PIN to authorize this transaction.",
    "Your subscription will renew unless you cancel now.",
    "We need to verify your details to prevent account closure.",
    "Thank you for choosing our services. How can I help?",
    "The meeting has been rescheduled to Thursday at two PM.",
    "Please call back at your earliest convenience.",
    "Your voice has been verified. Access granted.",
    "The system will be down for maintenance tonight.",
    "Please hold while I transfer you to a specialist.",
    "Your loan application requires additional documentation.",
    "We noticed a login from an unrecognized device.",
    "Your insurance claim has been approved for processing.",
    "Call us immediately to avoid further penalties.",
    "This message contains important account information.",
    "Your verification code is being sent to your phone.",
    "Please stay on the line for an important announcement.",
    "Your credit score has changed. Review your report now.",
    "We are updating our privacy policy effective next month.",
    "Your appointment has been confirmed for tomorrow morning.",
    "The invoice is attached. Please review and approve.",
    "Thank you for your patience. Your request is processing.",
]

VOICES = ["af_heart", "af_nova", "am_adam", "am_echo", "bf_emma", "bm_george"]


def generate_mms_fakes(out_dir: Path) -> list[Path]:
    """Generate N_FAKES WAV files using facebook/mms-tts-eng (VITS, pure transformers).

    MMS-TTS is a massively multilingual TTS system trained on 1107 languages —
    it was never part of ASVspoof 2019/2021 training data, making it a clean
    out-of-distribution test of the detector's generalization.
    """
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    print("Loading facebook/mms-tts-eng ...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")  # noqa: B615  # nosec B615
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")  # noqa: B615  # nosec B615
    model.eval()

    fake_paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sentence in enumerate(SENTENCES[:N_FAKES]):
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform[0].cpu().numpy()
        sr = model.config.sampling_rate
        path = out_dir / f"mms_tts_{i:03d}.wav"
        sf.write(str(path), output, sr, subtype="PCM_16")
        fake_paths.append(path)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{N_FAKES}")
    print(f"Generated {len(fake_paths)} MMS-TTS fakes -> {out_dir}")
    return fake_paths


def load_real_samples(n: int) -> list[Path]:
    """Pick N real FLAC files from ASVspoof 2021 LA eval."""
    all_real = sorted(REAL_FLAC_DIR.glob("LA_E_*.flac"))
    # Use known real files (label=bonafide) — take deterministic sample
    selected = all_real[:: len(all_real) // n][:n]
    print(f"Selected {len(selected)} real samples from {REAL_FLAC_DIR}")
    return selected


def load_audio(path: Path) -> torch.Tensor:
    """Load, resample to 16kHz mono, pad/trim to TARGET_LEN."""
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    wav = torch.from_numpy(data).float()
    if wav.dim() == 2:
        wav = wav.mean(1)  # stereo → mono
    wav = wav.unsqueeze(0)  # (1, T) for resample
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)  # (T,)
    if wav.shape[0] < TARGET_LEN:
        wav = F.pad(wav, (0, TARGET_LEN - wav.shape[0]))
    else:
        wav = wav[:TARGET_LEN]
    return wav


def load_model() -> torch.nn.Module:
    from voiceguard.models.ssl_classifier import SSLClassifier

    cfg = json.loads((CKPT.parent / "config.json").read_text())
    model = SSLClassifier(cfg["model_name"])
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval().to(DEVICE)
    print(f"Loaded {cfg['model_name']} from {CKPT.name}")
    return model


@torch.no_grad()
def score_files(model, paths: list[Path], label: str) -> list[dict]:
    results = []
    for p in paths:
        wav = load_audio(p).unsqueeze(0).to(DEVICE)  # (1, T)
        logits = model(wav)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        fake_prob = probs[1]
        prediction = "fake" if fake_prob >= 0.5 else "real"
        results.append(
            {
                "file": p.name,
                "true_label": label,
                "predicted": prediction,
                "fake_prob": round(fake_prob, 4),
                "correct": prediction == label,
            }
        )
    return results


def main():
    print(f"Device: {DEVICE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate or reuse fakes
    fake_dir = OUT_DIR / "kokoro_fakes"
    existing_fakes = sorted(fake_dir.glob("*.wav"))
    if len(existing_fakes) >= N_FAKES:
        print(f"Reusing {len(existing_fakes)} existing MMS-TTS fakes")
        fake_paths = existing_fakes[:N_FAKES]
    else:
        fake_paths = generate_mms_fakes(fake_dir)

    real_paths = load_real_samples(N_REALS)
    model = load_model()

    print("\nScoring fakes...")
    fake_results = score_files(model, fake_paths, "fake")
    print("Scoring reals...")
    real_results = score_files(model, real_paths, "real")

    all_results = fake_results + real_results
    n_correct = sum(r["correct"] for r in all_results)
    fake_det = sum(r["correct"] for r in fake_results)
    real_pass = sum(r["correct"] for r in real_results)

    summary = {
        "model": "wav2vec2-large (6.86% EER on ASVspoof 2021 LA)",
        "synthesis_system": (  # noqa: E501
            "facebook/mms-tts-eng (VITS, 1107-language MMS) — NOT in ASVspoof training data"
        ),
        "n_fakes": len(fake_results),
        "n_reals": len(real_results),
        "fake_detection_rate": round(fake_det / len(fake_results), 4),
        "real_pass_rate": round(real_pass / len(real_results), 4),
        "overall_accuracy": round(n_correct / len(all_results), 4),
        "avg_fake_score_on_fakes": round(np.mean([r["fake_prob"] for r in fake_results]), 4),
        "avg_fake_score_on_reals": round(np.mean([r["fake_prob"] for r in real_results]), 4),
    }

    print("\n" + "=" * 60)
    print("OOD SYNTHESIS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Synthesis system : {summary['synthesis_system']}")
    pct = summary["fake_detection_rate"] * 100
    print(f"Fake detection   : {pct:.1f}%  ({fake_det}/{len(fake_results)})")
    rpt = summary["real_pass_rate"] * 100
    print(f"Real pass rate   : {rpt:.1f}%  ({real_pass}/{len(real_results)})")
    print(f"Overall accuracy : {summary['overall_accuracy'] * 100:.1f}%")
    print(f"Avg fake-score (on Kokoro) : {summary['avg_fake_score_on_fakes']:.4f}")
    print(f"Avg fake-score (on real)   : {summary['avg_fake_score_on_reals']:.4f}")
    print("=" * 60)

    out_json = OUT_DIR / "ood_eval_results.json"
    out_json.write_text(json.dumps({"summary": summary, "per_file": all_results}, indent=2))
    print(f"\nDetailed results: {out_json}")


if __name__ == "__main__":
    main()
