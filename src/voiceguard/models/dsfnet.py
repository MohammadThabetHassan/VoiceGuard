"""
DSFNet — Dual-Stream Fusion Network for voice deepfake detection.

Architecture (midterm §5.2):
  Stream A: 1D-CNN waveform encoder (5 blocks, 1→512 channels)
  Stream B: 2D-ResNet spectrogram encoder (4 residual blocks, 1→512 channels)
  Fusion:   Bidirectional cross-attention (8 heads, 512-dim) → 1024-dim
  Head:     1024→512→256→128→2 with dropout p=0.3

Input:
  waveform   — raw 16kHz float32 tensor (B, 1, T)
  spectrogram — 80-bin Mel-spectrogram (B, 1, F, T')

Output: logits (B, 2) — class 0=genuine, class 1=synthetic
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as audio_tx


class ConvBlock1D(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 9, pool: int = 4) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveformEncoder(nn.Module):
    """Stream A: five 1D-CNN blocks, 1→32→64→128→256→512 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock1D(1, 32, kernel=9, pool=4),
            ConvBlock1D(32, 64, kernel=9, pool=4),
            ConvBlock1D(64, 128, kernel=9, pool=4),
            ConvBlock1D(128, 256, kernel=9, pool=4),
            ConvBlock1D(256, 512, kernel=9, pool=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 1, T). Returns (B, 512)."""
        out = self.blocks(x)  # (B, 512, T')
        return self.pool(out).squeeze(-1)  # (B, 512)


class ResBlock2D(nn.Module):
    """2D residual block: Conv→BN→ReLU→Conv→BN + skip."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class SpectrogramEncoder(nn.Module):
    """Stream B: four 2D-ResNet blocks, 1→64→128→256→512 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock2D(64, 64, stride=1),
            ResBlock2D(64, 128, stride=2),
            ResBlock2D(128, 256, stride=2),
            ResBlock2D(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 1, F, T). Returns (B, 512)."""
        out = self.stem(x)
        out = self.blocks(out)
        return self.pool(out).squeeze(-1).squeeze(-1)  # (B, 512)


class BidirectionalCrossAttention(nn.Module):
    """Symmetric cross-attention between two 512-dim token sequences.

    Q from A, K/V from B → attended_A
    Q from B, K/V from A → attended_B
    Output: concatenation → (B, 1024)
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8) -> None:
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Args: feat_a, feat_b both (B, 512). Returns (B, 1024)."""
        a = feat_a.unsqueeze(1)  # (B, 1, 512)
        b = feat_b.unsqueeze(1)
        # Q from A, K/V from B
        out_a, _ = self.attn_a2b(a, b, b)
        out_a = self.norm_a(out_a + a).squeeze(1)  # (B, 512)
        # Q from B, K/V from A
        out_b, _ = self.attn_b2a(b, a, a)
        out_b = self.norm_b(out_b + b).squeeze(1)  # (B, 512)
        return torch.cat([out_a, out_b], dim=-1)  # (B, 1024)


class ClassificationHead(nn.Module):
    """1024→512→256→128→2 with dropout p=0.3."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelSpectrogramTransform(nn.Module):
    """Convert raw waveform to 80-bin Mel-spectrogram (25ms/10ms frames)."""

    def __init__(self, sr: int = 16000) -> None:
        super().__init__()
        self.transform = audio_tx.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(0.025 * sr),
            hop_length=int(0.010 * sr),
            n_mels=80,
            power=2.0,
        )
        self.to_db = audio_tx.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Args: waveform (B, 1, T). Returns (B, 1, 80, T')."""
        # waveform (B, 1, T) → mel (B, 1, 80, T') via torchaudio broadcasting
        mel = self.transform(waveform)  # (B, 1, 80, T')
        return self.to_db(mel)  # (B, 1, 80, T')


class DSFNet(nn.Module):
    """Dual-Stream Fusion Network for voice deepfake detection.

    Can operate in three modes:
      - joint: accepts raw waveform, computes spectrogram internally
      - waveform_only: only Stream A (for ablation)
      - spectrogram_only: only Stream B (for ablation)

    Attributes:
        wave_enc: WaveformEncoder (Stream A)
        spec_enc: SpectrogramEncoder (Stream B)
        mel_transform: MelSpectrogramTransform
        cross_attn: BidirectionalCrossAttention
        head: ClassificationHead
    """

    def __init__(self, dropout: float = 0.3, sr: int = 16000) -> None:
        super().__init__()
        self.wave_enc = WaveformEncoder()
        self.spec_enc = SpectrogramEncoder()
        self.mel_transform = MelSpectrogramTransform(sr=sr)
        self.cross_attn = BidirectionalCrossAttention(d_model=512, n_heads=8)
        self.head = ClassificationHead(dropout=dropout)

    def forward(
        self,
        waveform: torch.Tensor,
        spectrogram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            waveform: (B, 1, T) raw 16kHz audio.
            spectrogram: (B, 1, 80, T') Mel-spectrogram. Computed from
                waveform if not provided.

        Returns:
            logits (B, 2) — class 0=genuine, class 1=synthetic.
        """
        if spectrogram is None:
            spectrogram = self.mel_transform(waveform)

        feat_a = self.wave_enc(waveform)  # (B, 512)
        feat_b = self.spec_enc(spectrogram)  # (B, 512)
        fused = self.cross_attn(feat_a, feat_b)  # (B, 1024)
        return self.head(fused)  # (B, 2)

    def predict(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class_indices, probabilities) for a waveform batch."""
        with torch.no_grad():
            logits = self.forward(waveform)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
