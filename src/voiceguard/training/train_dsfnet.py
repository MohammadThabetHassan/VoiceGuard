"""
DSFNet training script.

Usage:
    python -m voiceguard.training.train_dsfnet \\
        --data-path /path/to/features.h5 \\
        --s3-path s3://voiceguard-mt-2026/checkpoints/dsfnet \\
        --epochs 40 --batch-size 32 --lr 1e-4

Trains DSFNet with mixed-precision (fp16 on GPU), saves checkpoints to S3,
and logs metrics to stdout + JSON. Resumable from latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


def _try_import_amp():
    try:
        from torch.cuda import amp

        return amp
    except ImportError:
        return None


class AudioDataset(Dataset):
    """Dataset that loads preprocessed waveform tensors from a directory.

    Expected layout:
        data_path/
            real/<filename>.pt   — genuine audio tensors (1, T)
            fake/<filename>.pt   — synthetic audio tensors (1, T)

    Each .pt file contains a 1D or 2D float32 tensor.
    """

    def __init__(self, data_path: str | Path, clip_samples: int = 48000) -> None:
        self.clip_samples = clip_samples
        self.samples: list[tuple[Path, int]] = []
        data_path = Path(data_path)
        for label, idx in [("real", 0), ("fake", 1)]:
            label_dir = data_path / label
            if label_dir.exists():
                for f in sorted(label_dir.glob("*.pt")):
                    self.samples.append((f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        waveform = torch.load(path, weights_only=True)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)
        # Pad or trim to fixed length
        T = waveform.shape[-1]
        if T < self.clip_samples:
            pad = torch.zeros(1, self.clip_samples - T)
            waveform = torch.cat([waveform, pad], dim=-1)
        else:
            waveform = waveform[:, : self.clip_samples]
        return waveform, label


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate from fake scores and binary labels."""
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: x - np.interp(x, fpr, fnr), 0.0, 1.0)
    except ValueError:
        eer = float("nan")
    return float(eer)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    out_dir: Path,
    s3_path: str | None,
) -> Path:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dsfnet_epoch{epoch:03d}.pt"
    torch.save(ckpt, ckpt_path)

    if s3_path:
        dest = f"{s3_path}/dsfnet_epoch{epoch:03d}.pt"
        subprocess.run(["aws", "s3", "cp", str(ckpt_path), dest], check=False)

    return ckpt_path


def load_latest_checkpoint(
    out_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load latest checkpoint from out_dir. Returns start epoch (0 if none)."""
    ckpts = sorted(out_dir.glob("dsfnet_epoch*.pt"))
    if not ckpts:
        return 0
    ckpt = torch.load(ckpts[-1], weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Resumed from {ckpts[-1]} (epoch {ckpt['epoch']})")
    return int(ckpt["epoch"]) + 1


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = AudioDataset(args.data_path, clip_samples=args.sr * 3)
    if len(dataset) == 0:
        print(f"ERROR: No .pt files found under {args.data_path}/real/ or /fake/", file=sys.stderr)
        sys.exit(1)

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # Model
    from voiceguard.models.dsfnet import DSFNet

    model = DSFNet(sr=args.sr).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.checkpoint_dir)
    start_epoch = load_latest_checkpoint(out_dir, model, optimizer)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    history: list[dict] = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for waveform, labels in train_loader:
            waveform = waveform.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(waveform)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(waveform)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_scores: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for waveform, labels in val_loader:
                waveform = waveform.to(device)
                labels_dev = labels.to(device)
                logits = model(waveform)
                loss = criterion(logits, labels_dev)
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_scores.extend(probs.cpu().tolist())
                all_labels.extend(labels.tolist())

        val_loss /= len(val_loader)
        eer = compute_eer(np.array(all_scores), np.array(all_labels))
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "eer": round(eer, 4),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(metrics)
        print(json.dumps(metrics))

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, metrics, out_dir, args.s3_path)

    # Write final metrics JSON
    metrics_path = out_dir / "training_history.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. History saved to {metrics_path}")

    if args.s3_path:
        dest = f"{args.s3_path}/training_history.json"
        subprocess.run(["aws", "s3", "cp", str(metrics_path), dest], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DSFNet voice deepfake detector")
    parser.add_argument("--data-path", required=True, help="Path to dataset (real/ fake/ subdirs)")
    parser.add_argument("--s3-path", default=None, help="S3 prefix for checkpoint upload")
    parser.add_argument("--checkpoint-dir", default="checkpoints/dsfnet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
