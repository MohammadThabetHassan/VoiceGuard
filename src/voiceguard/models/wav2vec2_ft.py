"""
Wav2Vec2 fine-tuning for voice deepfake detection.

Uses facebook/wav2vec2-base as backbone with a binary classification head.
Fine-tuning script can be run on a GPU instance.
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


class Wav2Vec2Classifier(nn.Module):
    """Wav2Vec2 backbone with binary classification head.

    Attributes:
        backbone: Pretrained Wav2Vec2Model (feature extractor frozen by default).
        head: Linear classification head on top of pooled encoder output.
    """

    MODEL_NAME = "facebook/wav2vec2-base"
    # Pin model revision for reproducibility (not a secret — HuggingFace commit hash)
    MODEL_REVISION = "0b5b8e341f1393bce626f89a7cf5db20a7d24b1f"  # pragma: allowlist secret

    def __init__(
        self,
        num_labels: int = 2,
        freeze_feature_encoder: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model

        self.backbone = Wav2Vec2Model.from_pretrained(self.MODEL_NAME, revision=self.MODEL_REVISION)
        if freeze_feature_encoder:
            self.backbone.feature_extractor._freeze_parameters()

        hidden_size = self.backbone.config.hidden_size  # 768 for wav2vec2-base
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Args: input_values (B, T) raw waveform. Returns logits (B, 2)."""
        outputs = self.backbone(input_values)
        # Mean-pool over time
        hidden = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        return self.head(hidden)

    def predict(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(input_values)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Wav2Vec2Dataset(torch.utils.data.Dataset):
    """Dataset loading preprocessed .pt waveform tensors."""

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
        wav = torch.load(path, weights_only=True)
        if wav.ndim > 1:
            wav = wav.squeeze(0)  # (T,)
        T = wav.shape[0]
        if T < self.clip_samples:
            wav = nn.functional.pad(wav, (0, self.clip_samples - T))
        else:
            wav = wav[: self.clip_samples]
        return wav, label


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from torch.utils.data import DataLoader, random_split

    dataset = Wav2Vec2Dataset(args.data_path, clip_samples=args.sr * 3)
    if len(dataset) == 0:
        print(f"ERROR: No .pt files under {args.data_path}", file=sys.stderr)
        sys.exit(1)

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = Wav2Vec2Classifier(freeze_feature_encoder=True).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for wavs, labels in train_loader:
            wavs = wavs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(wavs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(wavs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_scores: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs = wavs.to(device)
                labels_dev = labels.to(device)
                logits = model(wavs)
                val_loss += criterion(logits, labels_dev).item()
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
            "elapsed_s": round(elapsed, 1),
        }
        history.append(metrics)
        print(json.dumps(metrics))

        ckpt_path = out_dir / f"wav2vec2_epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if args.s3_path:
            subprocess.run(["aws", "s3", "cp", str(ckpt_path), f"{args.s3_path}/"], check=False)

    metrics_path = out_dir / "wav2vec2_training_history.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    if args.s3_path:
        subprocess.run(["aws", "s3", "cp", str(metrics_path), f"{args.s3_path}/"], check=False)
    print(f"Training complete. {len(history)} epochs logged.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for deepfake detection")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--s3-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/wav2vec2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
