#!/usr/bin/env python3
"""Held-out clone score distributions for the deployed detector (Phase 2.3).

Characterises the IndexTTS-2 detectability ceiling with *fresh* score
distributions rather than a single accuracy number: for each held-out family
(real / XTTS / IndexTTS-2) it scores every clip and reports the fake-probability
distribution + detection rate. A family whose scores sit near the real
distribution is near the front-end ceiling — an honest limit, shown as data.

Usage:
    python3 scripts/clone_score_distributions.py \
        --checkpoint runs/xlsr_aasist_v9c/model_best.pt \
        --config runs/xlsr_aasist_v9c/config.json \
        --eval-dir heldout_eval_large --out runs/clone_scores_v9c.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torchaudio  # noqa: E402
from run_official_eval import CLIP, SR, load_model  # noqa: E402

# Subdir -> (display label, is_fake). "real" is the bonafide reference.
FAMILIES = {"real": ("real", False), "xtts": ("XTTS", True), "indextts2": ("IndexTTS-2", True)}


def load_wave(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.to(torch.float32)
    if wav.dim() == 2:
        wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
    if wav.shape[0] < CLIP:
        wav = torch.nn.functional.pad(wav, (0, CLIP - wav.shape[0]))
    return wav[:CLIP]


@torch.no_grad()
def score_dir(model, device, d: Path) -> list[float]:
    files = sorted(glob.glob(str(d / "*.wav")))
    scores = []
    for i in range(0, len(files), 16):
        batch = torch.stack([load_wave(f) for f in files[i : i + 16]]).to(device)
        probs = torch.softmax(model(batch), dim=-1)[:, 1]
        scores.extend(probs.cpu().tolist())
    return scores


def summarise(scores: list[float], is_fake: bool) -> dict:
    t = torch.tensor(scores)
    # detection: fakes should score >0.5; reals should score <0.5 (pass).
    correct = (t > 0.5) if is_fake else (t < 0.5)
    return {
        "n": len(scores),
        "mean_fake_prob": round(float(t.mean()), 4),
        "median": round(float(t.median()), 4),
        "p10": round(float(t.quantile(0.1)), 4),
        "p90": round(float(t.quantile(0.9)), 4),
        "rate": round(float(correct.float().mean()), 4),  # detect (fake) / pass (real)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name = load_model(args.checkpoint, args.config)
    model.to(device)

    out = {"model": name, "checkpoint": args.checkpoint, "families": {}}
    print(f"Model: {name}\n")
    print("family      |  n  | mean | median | p10–p90 | detect/pass")
    print("------------|-----|------|--------|---------|------------")
    for sub, (label, is_fake) in FAMILIES.items():
        d = Path(args.eval_dir) / sub
        if not d.is_dir():
            continue
        scores = score_dir(model, device, d)
        s = summarise(scores, is_fake)
        out["families"][label] = {**s, "is_fake": is_fake, "scores": [round(x, 4) for x in scores]}
        print(
            f"{label:<11} | {s['n']:>3} | {s['mean_fake_prob']:.2f} | {s['median']:.2f}   "
            f"| {s['p10']:.2f}–{s['p90']:.2f} | {s['rate'] * 100:.1f}%"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
