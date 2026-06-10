#!/usr/bin/env python3
"""Adversarial (white-box PGD/FGSM) robustness curve for a checkpoint (Phase 2.1).

Threat model: white-box L∞-bounded perturbations on the raw waveform (attacker
has the model + gradients), budget ``epsilon`` in [0, 1] on amplitude. Reports
clean vs FGSM vs PGD accuracy across an epsilon grid on a balanced sample of the
official ASVspoof 2021 LA eval. A large clean→PGD gap = adversarial fragility.

This MEASURES and CHARACTERISES the vulnerability honestly; it does not claim to
fix it (see docs/RESEARCH_RIGOR_PLAN.md Phase 2 / KNOWN_LIMITATIONS).

Usage:
    python3 scripts/pgd_curve.py --checkpoint runs/xlsr_aasist_v9c/model_best.pt \
        --config runs/xlsr_aasist_v9c/config.json --n-per-class 75 \
        --out runs/pgd_curve_v9c.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torchaudio  # noqa: E402
from run_official_eval import CLIP, DEFAULT_KEYS, SR, load_metadata, load_model  # noqa: E402

from voiceguard.evaluation.adversarial_eval import measure_robustness  # noqa: E402

EPS_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01]


def load_waveform(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.to(torch.float32)
    if wav.dim() == 2:
        wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
    if wav.shape[0] < CLIP:
        wav = torch.nn.functional.pad(wav, (0, CLIP - wav.shape[0]))
    return wav[:CLIP]


def build_sample(flac_dir: Path, meta: dict, n_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced sample from the eval phase: n_per_class real + n_per_class fake."""
    reals, fakes = [], []
    for fid, info in meta.items():
        if info["phase"] != "eval":
            continue
        bucket = reals if info["label"] == 0 else fakes
        if len(bucket) < n_per_class and (flac_dir / f"{fid}.flac").exists():
            bucket.append(fid)
        if len(reals) >= n_per_class and len(fakes) >= n_per_class:
            break
    ids = [(f, 0) for f in reals] + [(f, 1) for f in fakes]
    x = torch.stack([load_waveform(flac_dir / f"{fid}.flac") for fid, _ in ids])
    y = torch.tensor([lab for _, lab in ids], dtype=torch.long)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--flac-dir", required=True)
    ap.add_argument("--keys", default=DEFAULT_KEYS)
    ap.add_argument("--n-per-class", type=int, default=75)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name = load_model(args.checkpoint, args.config)
    model.to(device)
    meta = load_metadata(args.keys)
    x, y = build_sample(Path(args.flac_dir), meta, args.n_per_class)
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    print(f"Model: {name} | sample: {x.shape[0]} ({n_real} real / {n_fake} fake)")

    curve = []
    for eps in EPS_GRID:
        r = measure_robustness(model, x, y, epsilon=eps, alpha=eps / 5, pgd_steps=10)
        curve.append(r)
        print(
            f"  eps={eps:<7} clean={r['clean_acc']:.3f}  "
            f"FGSM={r['fgsm_acc']:.3f}  PGD={r['pgd_acc']:.3f}"
        )

    result = {"model": name, "checkpoint": args.checkpoint, "n": int(x.shape[0]), "curve": curve}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
