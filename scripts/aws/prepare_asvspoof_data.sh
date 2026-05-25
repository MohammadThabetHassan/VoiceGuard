#!/usr/bin/env bash
# Download, preprocess, and upload ASVspoof 2021 LA data to S3.
#
# ASVspoof 2021 LA access:
#   Registration: https://www.asvspoof.org/
#   After registration you receive download credentials via email.
#   Dataset mirrors (requires auth token):
#     https://zenodo.org/record/4837263  — ASVspoof 2021 LA eval set
#     https://zenodo.org/record/4837268  — ASVspoof 2021 LA progress set
#
# This script assumes ZENODO_TOKEN is set in the environment.
# Run on a g5.xlarge or similar instance with sufficient disk (≥80GB).

set -euo pipefail

S3_BUCKET="voiceguard-mt-2026"
DATA_DIR="/data/asvspoof2021"
PROCESSED_DIR="$DATA_DIR/tensors"
SR=16000
CLIP_SECS=3

echo "ASVspoof 2021 LA — Data Preparation"
echo "======================================"

mkdir -p "$DATA_DIR/raw" "$PROCESSED_DIR/real" "$PROCESSED_DIR/fake"

# ── Step 1: Download ──────────────────────────────────────────────────────────
# ASVspoof 2021 LA evaluation set (FLAC files, ~12GB)
# Replace these URLs with actual download links from your registration email.
EVAL_URL="https://zenodo.org/record/4837263/files/ASVspoof2021_LA_eval.tar.gz"
KEYS_URL="https://zenodo.org/record/4837263/files/ASVspoof2021_LA_eval_keys.txt"

if [[ ! -f "$DATA_DIR/raw/eval.tar.gz" ]]; then
  echo "Downloading evaluation set (~12GB)..."
  curl -L -H "Authorization: Bearer $ZENODO_TOKEN" \
    "$EVAL_URL" -o "$DATA_DIR/raw/eval.tar.gz"
fi

if [[ ! -f "$DATA_DIR/raw/eval_keys.txt" ]]; then
  curl -L "$KEYS_URL" -o "$DATA_DIR/raw/eval_keys.txt"
fi

# ── Step 2: Extract ───────────────────────────────────────────────────────────
if [[ ! -d "$DATA_DIR/raw/eval" ]]; then
  echo "Extracting..."
  tar -xzf "$DATA_DIR/raw/eval.tar.gz" -C "$DATA_DIR/raw/"
fi

# ── Step 3: Preprocess with Python ───────────────────────────────────────────
python3 - <<PYEOF
import sys, os
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from scipy import signal as scipy_signal

data_dir = Path("$DATA_DIR/raw")
out_dir = Path("$PROCESSED_DIR")
keys_file = Path("$DATA_DIR/raw/eval_keys.txt")
sr_target = $SR
clip_samples = sr_target * $CLIP_SECS

# Parse keys file: format "- <filename> <system> <type> <label>"
# label is 'bonafide' or 'spoof'
label_map = {}
with open(keys_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            fname = parts[1]
            label = parts[4]  # 'bonafide' or 'spoof'
            label_map[fname] = 'real' if label == 'bonafide' else 'fake'

print(f"Label map: {sum(1 for v in label_map.values() if v=='real')} real, "
      f"{sum(1 for v in label_map.values() if v=='fake')} fake")

processed = 0
for flac_path in sorted(data_dir.rglob("*.flac")):
    fname = flac_path.stem
    label = label_map.get(fname)
    if label is None:
        continue

    out_path = out_dir / label / f"{fname}.pt"
    if out_path.exists():
        continue

    try:
        import soundfile as sf
        data, sr = sf.read(str(flac_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        mx = np.max(np.abs(data))
        if mx > 0:
            data /= mx
        if sr != sr_target:
            new_len = int(len(data) * sr_target / sr)
            data = scipy_signal.resample(data, new_len)

        # Pad or trim
        if len(data) < clip_samples:
            data = np.concatenate([data, np.zeros(clip_samples - len(data))])
        else:
            data = data[:clip_samples]

        tensor = torch.from_numpy(data).unsqueeze(0)  # (1, T)
        torch.save(tensor, out_path)
        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed} files...")
    except Exception as e:
        print(f"  WARN: {flac_path}: {e}")

print(f"Done. {processed} tensors saved to {out_dir}")
PYEOF

# ── Step 4: Upload to S3 ──────────────────────────────────────────────────────
echo "Uploading tensors to s3://$S3_BUCKET/data/asvspoof2021/..."
aws s3 sync "$PROCESSED_DIR" "s3://$S3_BUCKET/data/asvspoof2021/tensors/" \
  --storage-class STANDARD_IA

echo "Data preparation complete."
echo "Files available at: s3://$S3_BUCKET/data/asvspoof2021/tensors/"
