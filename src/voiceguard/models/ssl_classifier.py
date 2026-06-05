"""
Generic SSL backbone classifier for voice deepfake detection.
Supports: microsoft/wavlm-base-plus, facebook/wav2vec2-large, microsoft/wavlm-large, etc.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SSLClassifier(nn.Module):
    """Fine-tunable SSL backbone + 2-layer classification head."""

    def __init__(self, model_name: str, freeze_feature_encoder: bool = True,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.model_name = model_name
        from transformers import AutoModel, AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_feature_encoder and hasattr(self.backbone, 'feature_extractor'):
            self.backbone.feature_extractor._freeze_parameters()
        hidden = cfg.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dtype != torch.float32:
            input_values = input_values.float()
        out = self.backbone(input_values)
        hidden = out.last_hidden_state.mean(dim=1)
        return self.head(hidden)

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1), probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str | Path, clip_samples: int = 48000) -> None:
        self.clip_samples = clip_samples
        self.samples: list[tuple[Path, int]] = []
        for label, idx in [("real", 0), ("fake", 1)]:
            d = Path(data_path) / label
            if d.exists():
                for f in sorted(d.glob("*.pt")):
                    self.samples.append((f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = torch.load(path, weights_only=True)
        if wav.ndim > 1:
            wav = wav.squeeze(0)
        T = wav.shape[0]
        if T < self.clip_samples:
            wav = nn.functional.pad(wav, (0, self.clip_samples - T))
        else:
            wav = wav[:self.clip_samples]
        return wav, label


from voiceguard.evaluation.metrics import compute_eer


def _eval(model, loader, device, criterion, augmentor=None):
    model.eval()
    total_loss, all_scores, all_labels = 0.0, [], []
    with torch.no_grad():
        for wavs, labels in loader:
            wavs, labels_d = wavs.to(device), labels.to(device)
            logits = model(wavs)
            total_loss += criterion(logits, labels_d).item()
            all_scores.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
            all_labels.extend(labels.tolist())
    return {"loss": total_loss / len(loader),
            "eer": compute_eer(np.array(all_scores), np.array(all_labels))}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.init(); torch.zeros(1, device=device)
    print(f"Device: {device}")

    from torch.utils.data import DataLoader
    train_ds = AudioDataset(args.train_path, args.sr * 3)
    val_ds   = AudioDataset(args.val_path,   args.sr * 3)
    test_ds  = AudioDataset(args.test_path,  args.sr * 3) if args.test_path else None
    if not train_ds or not val_ds:
        sys.exit("ERROR: empty dataset")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}" + (f" | Test: {len(test_ds)}" if test_ds else ""))

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_ds,  args.batch_size, num_workers=4) if test_ds else None

    model = SSLClassifier(args.model_name).to(device)
    print(f"Model: {args.model_name} | Trainable: {model.count_parameters():,}")

    augmentor = None
    if args.augment:
        from voiceguard.models.dsfnet import AudioAugment
        augmentor = AudioAugment(sample_rate=args.sr).to(device)

    # Warmup 5% + cosine decay
    warmup = max(1, int(0.05 * args.epochs))
    def _lr(ep):
        if ep < warmup: return (ep + 1) / warmup
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Experiment label includes model short name
    short = args.model_name.split("/")[-1]
    expt = f"{short}_{args.epochs}ep_b{args.batch_size}_lr{args.lr}"
    if args.augment: expt += "_aug"
    out_dir = Path(args.checkpoint_dir) / expt
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    history, best_eer, best_loss, best_ep, patience = [], float("inf"), float("inf"), -1, 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, t0 = 0.0, time.time()

        for wavs, labels in train_loader:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            if augmentor is not None:
                augmentor.train()
                wavs = augmentor(wavs.unsqueeze(1)).squeeze(1)
            if use_amp and scaler:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(wavs), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = criterion(model(wavs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        vm = _eval(model, val_loader, device, criterion)
        m = {"epoch": epoch, "train_loss": round(train_loss / len(train_loader), 4),
             "val_loss": round(vm["loss"], 4),
             "val_eer":  None if math.isnan(vm["eer"]) else round(vm["eer"], 4),
             "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
        history.append(m); print(json.dumps(m))

        val_eer = vm["eer"]
        if not math.isnan(val_eer) and val_eer < best_eer:
            best_eer, best_loss, best_ep, patience = val_eer, vm["loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "best_val_eer": best_eer, "metrics": m}, out_dir / "model_best.pt")
        else:
            patience += 1

        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"Early stopping at epoch {epoch} (best val_eer={best_eer:.4f} @ ep {best_ep})")
            break

    # Final test evaluation
    test_eer = float("nan")
    if test_loader and (out_dir / "model_best.pt").exists():
        print("Loading best checkpoint for test evaluation...")
        ckpt = torch.load(out_dir / "model_best.pt", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        tm = _eval(model, test_loader, device, criterion)
        test_eer = tm["eer"]
        final = {"best_epoch": best_ep, "best_val_eer": round(best_eer, 4),
                 "test_eer": round(test_eer, 4), "test_loss": round(tm["loss"], 4)}
        (out_dir / "final_results.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(final))

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Training complete. Best val_EER={best_eer:.4f} @ ep{best_ep} | test EER={test_eer:.4f}")
    print(f"Results: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-path", required=True)
    p.add_argument("--val-path",   required=True)
    p.add_argument("--test-path",  default=None)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--sr",         type=int,   default=16000)
    p.add_argument("--early-stop", type=int,   default=6)
    p.add_argument("--augment",    action="store_true")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
