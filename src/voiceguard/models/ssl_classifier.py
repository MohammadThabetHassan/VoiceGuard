"""
Generic SSL backbone classifier for voice deepfake detection.
Supports: microsoft/wavlm-base-plus, facebook/wav2vec2-large, microsoft/wavlm-large, etc.
"""

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

from voiceguard.evaluation.metrics import compute_eer


class SSLClassifier(nn.Module):
    """Fine-tunable SSL backbone + 2-layer classification head."""

    def __init__(
        self, model_name: str, freeze_feature_encoder: bool = True, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.model_name = model_name
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_name)  # nosec B615
        self.backbone = AutoModel.from_pretrained(model_name)  # nosec B615
        if freeze_feature_encoder and hasattr(self.backbone, "feature_extractor"):
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


class SSLAASIST(nn.Module):
    """SSL front-end + AASIST-style graph-attention back-end.

    The literature-SOTA recipe for ASVspoof21-LA: instead of mean-pooling the
    SSL hidden states (which left XLS-R at 10.47% here), take a *learnable
    weighted sum of all transformer layers*, project to d_model, prepend a CLS
    token, run multi-head graph-attention over the temporal frames, and classify
    the CLS output. Targets the hard hidden-track attacks (A10/A11/A18).
    """

    def __init__(
        self,
        model_name: str,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        freeze_feature_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        from transformers import AutoConfig, AutoModel

        from voiceguard.models.aasist import GraphAttentionLayer

        cfg = AutoConfig.from_pretrained(model_name)  # nosec B615
        self.backbone = AutoModel.from_pretrained(model_name)  # nosec B615
        if freeze_feature_encoder and hasattr(self.backbone, "feature_extractor"):
            self.backbone.feature_extractor._freeze_parameters()
        n_hidden = cfg.num_hidden_layers + 1  # +1 for embedding output
        self.layer_weights = nn.Parameter(torch.ones(n_hidden) / n_hidden)
        self.proj = nn.Linear(cfg.hidden_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.graph_layers = nn.ModuleList(
            [GraphAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dtype != torch.float32:
            input_values = input_values.float()
        out = self.backbone(input_values, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)  # (L, B, T, H)
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        feat = (hs * w).sum(dim=0)  # (B, T, H)
        feat = self.proj(feat)  # (B, T, d_model)
        cls = self.cls_token.expand(feat.size(0), -1, -1)
        x = torch.cat([cls, feat], dim=1)  # (B, T+1, d_model)
        for layer in self.graph_layers:
            x = layer(x)
        return self.head(x[:, 0])  # CLS → logits

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1), probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str | Path, clip_samples: int = 48000, rawboost=None) -> None:
        self.clip_samples = clip_samples
        self.rawboost = rawboost  # callable(wav)->wav applied per-sample (train only)
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
        wav = wav.reshape(-1).float()  # flatten (1,T) or (1,1,T) → (T,)
        if self.rawboost is not None:
            wav = self.rawboost(wav)  # CPU per-sample RawBoost (OOD robustness)
        T = wav.shape[0]
        if T < self.clip_samples:
            wav = nn.functional.pad(wav, (0, self.clip_samples - T))
        else:
            wav = wav[: self.clip_samples]
        return wav, label


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
    return {
        "loss": total_loss / len(loader),
        "eer": compute_eer(np.array(all_scores), np.array(all_labels)),
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.init()
        torch.zeros(1, device=device)
    print(f"Device: {device}")

    from torch.utils.data import ConcatDataset, DataLoader

    rawboost = None
    if getattr(args, "rawboost", False):
        from voiceguard.training.rawboost import RawBoost

        rawboost = RawBoost(p=args.rawboost_p, algo=0, fs=args.sr)
        print(f"RawBoost ENABLED (p={args.rawboost_p}, algo=random) — OOD robustness aug")

    # RawBoost applies only to training data (never val/test).
    train_ds = AudioDataset(args.train_path, args.sr * 3, rawboost=rawboost)
    for extra in args.extra_train_path or []:
        extra_ds = AudioDataset(extra, args.sr * 3, rawboost=rawboost)
        if extra_ds:
            train_ds = ConcatDataset([train_ds, extra_ds])
            print(f"  + {extra}: {len(extra_ds)} samples")
    val_ds = AudioDataset(args.val_path, args.sr * 3)
    test_ds = AudioDataset(args.test_path, args.sr * 3) if args.test_path else None
    if not train_ds or not val_ds:
        sys.exit("ERROR: empty dataset")
    test_info = f" | Test: {len(test_ds)}" if test_ds else ""
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}{test_info}")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=4) if test_ds else None

    if getattr(args, "arch", "linear") == "aasist":
        model = SSLAASIST(args.model_name).to(device)
    else:
        model = SSLClassifier(args.model_name).to(device)
    print(
        f"Model: {args.model_name} [{getattr(args, 'arch', 'linear')}] | "
        f"Trainable: {model.count_parameters():,}"
    )

    # Resume state (restored after optimizer/scheduler are built, below).
    resume_state = None
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(resume_state.get("model_state", resume_state), strict=False)
        print(f"Resumed weights from {args.resume}")

    augmentor = None
    if args.augment:
        from voiceguard.models.dsfnet import AudioAugment

        augmentor = AudioAugment(sample_rate=args.sr).to(device)

    # Warmup 5% + cosine decay
    warmup = max(1, int(0.05 * args.epochs))

    def _lr(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Experiment label includes model short name
    short = args.model_name.split("/")[-1]
    expt = f"{short}_{args.epochs}ep_b{args.batch_size}_lr{args.lr}"
    if getattr(args, "arch", "linear") != "linear":
        expt += f"_{args.arch}"
    if args.augment:
        expt += "_aug"
    if getattr(args, "rawboost", False):
        expt += "_rawboost"
    out_dir = Path(args.checkpoint_dir) / expt
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    history, best_eer, _best_loss, best_ep, patience = [], float("inf"), float("inf"), -1, 0
    start_epoch = 0

    # Full-state resume (optimizer/scheduler/scaler/epoch/best) for GPU-interrupt safety.
    if resume_state is not None and "optimizer_state" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        if scaler is not None and resume_state.get("scaler_state"):
            scaler.load_state_dict(resume_state["scaler_state"])
        start_epoch = resume_state.get("epoch", -1) + 1
        best_eer = resume_state.get("best_val_eer", float("inf"))
        best_ep = resume_state.get("best_epoch", -1)
        patience = resume_state.get("patience", 0)
        history = resume_state.get("history", [])
        print(f"Resumed full training state -> start_epoch={start_epoch}, best_eer={best_eer:.4f}")

    def _save_last(epoch):
        """Full-state checkpoint each epoch so a GPU reclaim resumes cleanly."""
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "best_val_eer": best_eer,
                "best_epoch": best_ep,
                "patience": patience,
                "history": history,
            },
            out_dir / "model_last.pt",
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, t0 = 0.0, time.time()

        for wavs, labels in train_loader:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            if augmentor is not None:
                augmentor.train()
                wavs = augmentor(wavs.unsqueeze(1)).squeeze(1)

            # PGD adversarial augmentation (50% adv mix when enabled)
            if getattr(args, "pgd_alpha", 0.0) > 0:
                eps = 0.01
                alpha = args.pgd_alpha
                delta = torch.zeros_like(wavs).uniform_(-eps, eps).requires_grad_(True)
                for _ in range(3):
                    loss_adv = criterion(model(wavs + delta), labels)
                    loss_adv.backward()
                    with torch.no_grad():
                        delta.data = (delta.data + alpha * delta.grad.sign()).clamp(-eps, eps)
                        delta.data = (wavs + delta.data).clamp(-1.0, 1.0) - wavs
                    delta.grad.zero_()
                adv_wavs = (wavs + delta.detach()).clamp(-1.0, 1.0)
                wavs = torch.cat([wavs, adv_wavs])
                labels = torch.cat([labels, labels])
                optimizer.zero_grad()

            if use_amp and scaler:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(wavs), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(wavs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        vm = _eval(model, val_loader, device, criterion)
        m = {
            "epoch": epoch,
            "train_loss": round(train_loss / len(train_loader), 4),
            "val_loss": round(vm["loss"], 4),
            "val_eer": None if math.isnan(vm["eer"]) else round(vm["eer"], 4),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(m)
        print(json.dumps(m))

        val_eer = vm["eer"]
        if not math.isnan(val_eer) and val_eer < best_eer:
            best_eer, _best_loss, best_ep, patience = (val_eer, vm["loss"], epoch, 0)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_eer": best_eer,
                    "best_epoch": best_ep,
                    "metrics": m,
                },
                out_dir / "model_best.pt",
            )
            try:
                from voiceguard.models.checkpoint_manager import save_snapshot

                save_snapshot(out_dir / "model_best.pt", args.model_name.split("/")[-1])
            except Exception:  # noqa: S110
                pass
        else:
            patience += 1

        _save_last(epoch)  # full-state resume checkpoint every epoch

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
        final = {
            "best_epoch": best_ep,
            "best_val_eer": round(best_eer, 4),
            "test_eer": round(test_eer, 4),
            "test_loss": round(tm["loss"], 4),
        }
        (out_dir / "final_results.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(final))

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Training complete. Best val_EER={best_eer:.4f} @ ep{best_ep} | test EER={test_eer:.4f}")
    print(f"Results: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--train-path", required=True)
    p.add_argument("--val-path", required=True)
    p.add_argument("--test-path", default=None)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--early-stop", type=int, default=6)
    p.add_argument(
        "--arch",
        choices=["linear", "aasist"],
        default="linear",
        help="Back-end: linear head (default) or AASIST graph-attention",
    )
    p.add_argument("--augment", action="store_true")
    p.add_argument(
        "--rawboost",
        action="store_true",
        help="Enable per-sample RawBoost waveform aug (OOD robustness)",
    )
    p.add_argument(
        "--rawboost-p",
        type=float,
        default=0.5,
        dest="rawboost_p",
        help="Probability of applying RawBoost per sample",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume from (model_last.pt restores full state)",
    )
    p.add_argument(
        "--extra-train-path",
        action="append",
        default=[],
        dest="extra_train_path",
        help="Extra training dirs merged with --train-path (repeat for multiple)",
    )
    p.add_argument(
        "--pgd-alpha",
        type=float,
        default=0.0,
        help="PGD step size (0=disabled). ε=0.01 fixed. Mixes 50%% adv per batch.",
    )
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
