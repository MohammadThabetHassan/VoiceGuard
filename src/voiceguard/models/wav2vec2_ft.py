"""
Wav2Vec2 fine-tuning for voice deepfake detection.

Uses facebook/wav2vec2-base as backbone with a binary classification head.
Fine-tuning script can be run on a GPU instance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from voiceguard.evaluation.metrics import compute_eer


class Wav2Vec2Classifier(nn.Module):
    """Wav2Vec2 backbone with binary classification head.

    Attributes:
        backbone: Pretrained Wav2Vec2Model (feature extractor frozen by default).
        head: Linear classification head on top of pooled encoder output.
    """

    MODEL_NAME = "facebook/wav2vec2-base"
    # Pin revision for reproducibility — backbone weights must match saved checkpoints.
    MODEL_REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"  # pragma: allowlist secret

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
        if input_values.dtype != torch.float32:
            input_values = input_values.float()
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



def _eval_loader(model: nn.Module, loader, device: torch.device,
                  criterion: nn.Module) -> dict:
    model.eval()
    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for wavs, labels in loader:
            wavs, labels_dev = wavs.to(device), labels.to(device)
            logits = model(wavs)
            total_loss += criterion(logits, labels_dev).item()
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())
    total_loss /= len(loader)
    eer = compute_eer(np.array(all_scores), np.array(all_labels))
    return {"loss": total_loss, "eer": eer}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize CUDA context before forking DataLoader workers — forked workers
    # inherit partial CUDA state which can corrupt the NVML P2P check on .to(device).
    if device.type == "cuda":
        torch.cuda.init()
        torch.zeros(1, device=device)
    print(f"Device: {device}")

    from torch.utils.data import DataLoader

    train_ds = Wav2Vec2Dataset(args.train_path, clip_samples=args.sr * 3)
    val_ds = Wav2Vec2Dataset(args.val_path, clip_samples=args.sr * 3)
    test_ds = Wav2Vec2Dataset(args.test_path, clip_samples=args.sr * 3) if args.test_path else None

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: empty train or val dataset", file=sys.stderr)
        sys.exit(1)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}", end="")
    if test_ds:
        print(f" | Test: {len(test_ds)}", end="")
    print()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, num_workers=4) if test_ds else None
    )

    model = Wav2Vec2Classifier(freeze_feature_encoder=True).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Codec augmentation for the 2019→2021 domain gap (applied per-batch on GPU)
    augmentor = None
    if args.augment:
        from voiceguard.models.dsfnet import AudioAugment
        augmentor = AudioAugment(sample_rate=args.sr).to(device)
        augmentor.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for wavs, labels in train_loader:
            wavs = wavs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Apply codec augmentation (AudioAugment expects (B,1,T), wav2vec2 needs (B,T))
            if augmentor is not None:
                augmentor.train()
                wavs = augmentor(wavs.unsqueeze(1)).squeeze(1)

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

        val_m = _eval_loader(model, val_loader, device, criterion)
        model.train()
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_m["loss"], 4),
            "val_eer": round(val_m["eer"], 4),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(metrics)
        print(json.dumps(metrics))

        # Select best and early-stop on val_loss — test set never drives selection.
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "metrics": metrics}, out_dir / "model_best.pt")
            from voiceguard.models.checkpoint_manager import save_snapshot
            save_snapshot(out_dir / "model_best.pt", "wav2vec2")
        else:
            patience_counter += 1

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best val_loss={best_val_loss:.4f} @ epoch {best_epoch})"
            )
            break

    # Evaluate on test set once using the best val_loss checkpoint.
    test_eer = float("nan")
    if test_loader and (out_dir / "model_best.pt").exists():
        print("Loading best checkpoint for final test evaluation...")
        best_state = torch.load(out_dir / "model_best.pt", weights_only=True)
        model.load_state_dict(best_state["model_state"])
        test_m = _eval_loader(model, test_loader, device, criterion)
        test_eer = test_m["eer"]
        final = {"best_epoch": best_epoch, "best_val_loss": best_val_loss,
                 "test_eer": round(test_eer, 4), "test_loss": round(test_m["loss"], 4)}
        (out_dir / "final_results.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(final))

    metrics_path = out_dir / "training_history.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    print(
        f"Training complete. Best val_loss={best_val_loss:.4f} "
        f"@ epoch {best_epoch} | test EER={test_eer:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for deepfake detection")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/wav2vec2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument(
        "--early-stop", type=int, default=5, help="Early stop patience (0=disabled)"
    )
    parser.add_argument("--augment", action="store_true", help="Enable codec augmentation")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
