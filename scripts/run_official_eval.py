"""Official ASVspoof 2021 LA eval — faithful reproduction, reading FLAC directly.

Vendored into the repo (Phase 1.5) so `scripts/eval_official.sh` is self-contained
and reproducible from a clean clone — paths are derived from ``__file__`` / CLI
args instead of being hardcoded to one machine.

Mirrors the original harness recipe exactly:
  - CLIP = 48000 (3 s @ 16 kHz); pad if shorter, take first 48000 if longer
  - score = softmax(logits)[:, 1]  == P(spoof)
  - EER vs label (spoof=1) via voiceguard.evaluation.metrics.compute_eer
  - per-phase split on trial_metadata col 8 (eval / progress / hidden)

Usage:
  python scripts/run_official_eval.py --checkpoint runs/<run>/model_best.pt \
      --config runs/<run>/config.json --flac-dir <LA_eval>/flac \
      --keys <keys>/LA/CM/trial_metadata.txt --out runs/official_<run>.json \
      [--save-scores runs/scores_<run>.npz]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

# Repo-relative import: scripts/ -> ../src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voiceguard.evaluation.metrics import compute_all_metrics, compute_eer  # noqa: E402
from voiceguard.models.ssl_classifier import SSLAASIST, SSLClassifier  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
CLIP = 48000  # 3 s @ 16 kHz — identical to the original harness
DEFAULT_KEYS = os.environ.get(
    "VG_CM_KEYS",
    "/srv/thabet/voiceguard-checkpoints/asvspoof2021_LA_official/keys/LA/CM/trial_metadata.txt",
)


class FlacEvalDataset(Dataset):
    def __init__(self, flac_dir: Path, meta: dict[str, dict]) -> None:
        self.dir = flac_dir
        self.samples = []  # (file_id, label)
        for fid, info in meta.items():
            if (flac_dir / f"{fid}.flac").exists():
                self.samples.append((fid, info["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        fid, label = self.samples[idx]
        # NB: libsndfile 1.2.x (soundfile) has a FLAC-decoder regression on the
        # official ASVspoof .flac; torchaudio decodes them fine.
        wav, sr = torchaudio.load(str(self.dir / f"{fid}.flac"))
        wav = wav.to(torch.float32)
        if wav.dim() == 2:
            wav = wav.mean(0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
        if wav.shape[0] < CLIP:
            wav = nn.functional.pad(wav, (0, CLIP - wav.shape[0]))
        else:
            wav = wav[:CLIP]
        return wav, label, fid


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


def load_metadata(path: str) -> dict[str, dict]:
    meta = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            meta[parts[1]] = {
                "codec": parts[2],
                "attack": parts[4],
                "label": 1 if parts[5] == "spoof" else 0,
                "phase": parts[7],
            }
    return meta


def per_phase_eer(scores, labels, stems, meta) -> dict[str, dict]:
    groups = defaultdict(lambda: ([], []))
    for s, lab, st in zip(scores, labels, stems, strict=False):
        info = meta.get(st)
        if info is None:
            continue
        groups[info["phase"]][0].append(s)
        groups[info["phase"]][1].append(lab)
    out = {}
    for ph, (sc, lb) in sorted(groups.items()):
        sc, lb = np.array(sc), np.array(lb)
        nf, nr = int(lb.sum()), int((1 - lb).sum())
        eer = compute_eer(sc, lb) if (nf and nr) else float("nan")
        out[ph] = {"eer": round(float(eer) * 100, 4), "n_fake": nf, "n_real": nr}
    return out


def load_model(ckpt_path: str, config_path: str):
    cfg = json.loads(Path(config_path).read_text())
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if cfg.get("arch", "linear") == "aasist":
        model = SSLAASIST(cfg["model_name"])
    else:
        model = SSLClassifier(cfg["model_name"])
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    return model, f"{cfg['model_name']} [{cfg.get('arch', 'linear')}]"


@torch.no_grad()
def run_inference(model, loader):
    scores, labels, stems = [], [], []
    model.to(DEVICE)
    t0 = time.time()
    n = 0
    for wavs, labs, sts in loader:
        probs = torch.softmax(model(wavs.to(DEVICE)), dim=-1)[:, 1]
        scores.extend(probs.cpu().tolist())
        labels.extend(labs)
        stems.extend(sts)
        n += len(labs)
        if n % 5000 < len(labs):
            print(f"  scored {n} ({n / max(time.time() - t0, 1):.0f}/s)", flush=True)
    return np.array(scores), np.array(labels), stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--flac-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default=DEFAULT_KEYS, help="trial_metadata.txt (CM keys)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--save-scores",
        default=None,
        help="Optional .npz to cache raw scores/labels/phases for fast metric "
        "recompute (minDCF, bootstrap CIs) without re-running inference.",
    )
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    model, model_name = load_model(args.checkpoint, args.config)
    print(f"Model: {model_name}")
    meta = load_metadata(args.keys)
    print(f"Metadata entries: {len(meta)}")
    ds = FlacEvalDataset(Path(args.flac_dir), meta)
    nr = sum(1 for _, lab in ds.samples if lab == 0)
    nf = sum(1 for _, lab in ds.samples if lab == 1)
    print(f"Samples found: {len(ds)} ({nr} real, {nf} fake)")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=6, collate_fn=collate_fn)

    print("Running inference...")
    scores, labels, stems = run_inference(model, loader)

    preds = (scores >= 0.5).astype(int)
    metrics = compute_all_metrics(labels, preds, scores)
    per_phase = per_phase_eer(scores, labels, stems, meta)

    out = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "n_samples": len(ds),
        "n_real": nr,
        "n_fake": nf,
        "headline_full_pool": {
            k: (round(float(v) * 100, 4) if k == "eer" else round(float(v), 6))
            for k, v in metrics.items()
        },
        "per_phase_eer": per_phase,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    if args.save_scores:
        phases = np.array([meta.get(st, {}).get("phase", "?") for st in stems])
        attacks = np.array([meta.get(st, {}).get("attack", "?") for st in stems])
        np.savez_compressed(
            args.save_scores, scores=scores, labels=labels, phases=phases, attacks=attacks
        )
        print(f"Cached scores -> {args.save_scores}")
    summary = {"full_pool_eer": out["headline_full_pool"]["eer"], "per_phase": per_phase}
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
