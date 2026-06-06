"""Unified training script: DSFNet / AASIST / Wav2Vec2 / WavLM + mixup + augment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from voiceguard.evaluation.metrics import compute_eer


class AudioDataset(Dataset):
    def __init__(self, data_paths: list[str | Path], clip_samples: int = 48000,
                 squeeze: bool = False) -> None:
        self.clip_samples = clip_samples
        self.squeeze = squeeze  # True for Wav2Vec2 (1D input)
        self.samples: list[tuple[Path, int]] = []
        for dp in data_paths:
            dp = Path(dp)
            for label, idx in [("real", 0), ("fake", 1)]:
                label_dir = dp / label
                if label_dir.exists():
                    for f in sorted(label_dir.glob("*.pt")):
                        self.samples.append((f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        wav = torch.load(path, weights_only=True)
        if wav.ndim > 1:
            wav = wav.squeeze(0)
        T = wav.shape[0]
        if T < self.clip_samples:
            wav = nn.functional.pad(wav, (0, self.clip_samples - T))
        else:
            wav = wav[:self.clip_samples]
        if not self.squeeze:
            wav = wav.unsqueeze(0)  # (1, T) for CNN models
        return wav, label


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[int] = []
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for waveform, labels in loader:
            waveform, labels_dev = waveform.to(device), labels.to(device)
            logits = model(waveform)
            total_loss += criterion(logits, labels_dev).item()
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    total_loss /= len(loader)
    eer = compute_eer(np.array(all_scores), np.array(all_labels))
    return {"loss": total_loss, "eer": eer}


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup: blend batch with shuffled version."""
    if alpha <= 0:
        return x, y, y, torch.ones(x.size(0), device=x.device)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_model(model_name: str, dropout: float, sr: int, augment: bool) -> nn.Module:
    from voiceguard.models.aasist import AASIST, AASISTLarge
    from voiceguard.models.dsfnet import DSFNet, DSFNetLarge, DSFNetTiny, DSFNetV2, DSFNetXL
    from voiceguard.models.wav2vec2_ft import Wav2Vec2Classifier

    if model_name == "aasist":
        return AASIST(sr=sr, dropout=dropout, augment=augment)
    elif model_name == "aasist_large":
        return AASISTLarge(sr=sr, augment=augment)
    elif model_name == "wav2vec2":
        return Wav2Vec2Classifier()
    elif model_name == "xl":
        return DSFNetXL(dropout=dropout, sr=sr, augment=augment)
    elif model_name == "large":
        return DSFNetLarge(dropout=dropout, sr=sr, augment=augment)
    elif model_name == "tiny":
        return DSFNetTiny(dropout=dropout, sr=sr, augment=augment)
    elif model_name == "v2":
        return DSFNetV2(dropout=dropout, sr=sr, augment=augment)
    return DSFNet(dropout=dropout, sr=sr, augment=augment)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.init()
        torch.zeros(1, device=device)
    print(f"Device: {device}")

    squeeze = args.model in ("wav2vec2",)
    train_paths = args.train_path
    val_paths = args.val_path if args.val_path else train_paths

    full_train_ds = AudioDataset(train_paths, squeeze=squeeze)
    if args.val_path:
        val_ds = AudioDataset(val_paths, squeeze=squeeze)
        train_ds = full_train_ds
    else:
        n_val = max(1, int(0.1 * len(full_train_ds)))
        n_train = len(full_train_ds) - n_val
        train_ds, val_ds = random_split(
            full_train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    test_ds = AudioDataset([args.test_path], squeeze=squeeze) if args.test_path else None

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: train or val dataset empty", file=sys.stderr)
        sys.exit(1)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}", end="")
    if test_ds:
        print(f" | Test: {len(test_ds)}", end="")
    print()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    test_loader = (DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)
                   if test_ds else None)

    model = get_model(args.model, args.dropout, args.sr, args.augment).to(device)
    print(f"Model: {args.model} | Parameters: {model.count_parameters():,}")

    # Experiment label
    eff_bs = args.batch_size * max(1, args.accum_steps)
    expt = f"{args.model}_{args.epochs}ep_b{eff_bs}_lr{args.lr}"
    if args.augment:
        expt += "_aug"
    if args.mixup > 0:
        expt += f"_mixup{args.mixup}"
    if args.label_smooth > 0:
        expt += f"_ls{args.label_smooth}"
    print(f"Experiment: {expt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Warm-up for 5% of epochs then cosine decay — critical for large models
    warmup_epochs = max(1, int(0.05 * args.epochs))
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if args.label_smooth > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.checkpoint_dir) / expt
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    start_epoch = 0
    if args.resume:
        ckpts = sorted(out_dir.glob("model_epoch*.pt"))
        if ckpts:
            ckpt = torch.load(ckpts[-1], weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"Resumed from {ckpts[-1]} (epoch {ckpt['epoch']})")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None
    history: list[dict] = []
    # Restore tracking state from checkpoint so resume doesn't clobber model_best.pt.
    resumed = args.resume and ckpts
    best_val_eer = float(ckpt.get("best_val_eer", float("inf"))) if resumed else float("inf")
    best_val_loss = float(ckpt.get("best_val_loss", float("inf"))) if resumed else float("inf")
    best_epoch = int(ckpt.get("best_epoch", -1)) if resumed else -1
    patience_counter = int(ckpt.get("patience_counter", 0)) if resumed else 0

    accum_steps = max(1, args.accum_steps)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (waveform, labels) in enumerate(train_loader):
            waveform, labels = waveform.to(device), labels.to(device)
            is_accum_step = (step + 1) % accum_steps != 0 and (step + 1) != len(train_loader)

            if args.mixup > 0:
                waveform, y_a, y_b, lam = mixup_data(waveform, labels, args.mixup)
                if use_amp and scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(waveform)
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
                else:
                    logits = model(waveform)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
            else:
                if use_amp and scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(waveform)
                        loss = criterion(logits, labels) / accum_steps
                else:
                    logits = model(waveform)
                    loss = criterion(logits, labels) / accum_steps

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if not is_accum_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if not is_accum_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * accum_steps

        scheduler.step()
        train_loss /= len(train_loader)

        val_metrics = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        def _safe(v: float, n: int = 4) -> float | None:
            return None if math.isnan(v) or math.isinf(v) else round(v, n)

        metrics = {
            "epoch": epoch,
            "train_loss": _safe(train_loss),
            "val_loss": _safe(val_metrics["loss"]),
            "val_eer": _safe(val_metrics["eer"]),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(metrics)
        print(json.dumps(metrics))

        # Select best checkpoint on val_EER (the actual target metric).
        # val_loss (with label smoothing) is noisy for EER optimisation.
        # Test set is touched only once at the end — never drives selection.
        val_eer_curr = val_metrics["eer"]
        val_loss_curr = val_metrics["loss"]
        if math.isnan(val_eer_curr) or math.isnan(val_loss_curr):
            print(f"WARNING: val metric is NaN at epoch {epoch} — skipping", flush=True)
        elif val_eer_curr < best_val_eer:
            best_val_eer = val_eer_curr
            best_val_loss = val_loss_curr
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_data = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
            }
            torch.save(best_ckpt_data, out_dir / "model_best.pt")
            try:
                from voiceguard.models.checkpoint_manager import save_snapshot
                save_snapshot(out_dir / "model_best.pt", args.model)
            except Exception:  # noqa: S110
                pass  # snapshot failure never interrupts training
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_eer": best_val_eer,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "metrics": metrics,
            }
            torch.save(ckpt, out_dir / f"model_epoch{epoch:03d}.pt")

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best val_eer={best_val_eer:.4f} @ epoch {best_epoch})"
            )
            break

    # Evaluate on test set once using the best val_EER checkpoint — never used for selection.
    test_eer = float("nan")
    if test_loader and (out_dir / "model_best.pt").exists():
        print("Loading best checkpoint for final test evaluation...")
        best_state = torch.load(out_dir / "model_best.pt", weights_only=True)
        model.load_state_dict(best_state["model_state"])
        test_metrics = evaluate(model, test_loader, device, criterion)
        test_eer = test_metrics["eer"]
        final = {"best_epoch": best_epoch, "best_val_eer": round(best_val_eer, 4),
                 "best_val_loss": best_val_loss,
                 "test_eer": round(test_eer, 4), "test_loss": round(test_metrics["loss"], 4)}
        (out_dir / "final_results.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(final))

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(
        f"Training complete. Best val_EER={best_val_eer:.4f} "
        f"@ epoch {best_epoch} | test EER={test_eer:.4f}"
    )
    print(f"Results: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", action="append", required=True)
    p.add_argument("--val-path", action="append", default=None)
    p.add_argument("--test-path", default=None)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--model", default="base",
                    choices=["base", "large", "xl", "tiny", "v2",
                             "aasist", "aasist_large", "wav2vec2"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha (0=disabled)")
    p.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing")
    p.add_argument("--early-stop", type=int, default=0, help="Early stop patience (0=disabled)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (effective_bs = batch_size × accum_steps)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
