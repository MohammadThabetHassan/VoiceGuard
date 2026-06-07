"""
AASIST-inspired model for voice anti-spoofing.

Key components from AASIST (Jung et al., 2022):
  - Max Feature Map (MFM) activation
  - Spectro-temporal graph attention
  - Dual-stream (waveform + spectrogram) fusion

Reference: https://arxiv.org/abs/2110.01200
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchaudio.transforms as audio_tx


class MFM(nn.Module):
    """Max Feature Map activation: max across adjacent channel pairs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        return torch.max(x[:, : C // 2], x[:, C // 2 :])


class LCNNBlock(nn.Module):
    """Light CNN block: Conv2d(k=3) → BN → ReLU → MaxPool(k=2)."""

    def __init__(self, in_ch: int, out_ch: int, pool: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SpecEncoder(nn.Module):
    """Spectrogram encoder with LCNN blocks."""

    def __init__(self, channels: tuple[int, ...] = (1, 32, 32, 64, 64, 128)) -> None:
        super().__init__()
        blocks = []
        for i in range(len(channels) - 1):
            pool = 1 if channels[i] == channels[i + 1] else 2
            blocks.append(LCNNBlock(channels[i], channels[i + 1], pool=pool))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, F, T = x.shape
        return self.blocks(x)


class PositionalEncoding2D(nn.Module):
    """Learnable frequency + time positional encodings."""

    def __init__(self, d_model: int, max_freq: int = 80, max_time: int = 300) -> None:
        super().__init__()
        self.freq_embed = nn.Embedding(max_freq, d_model)
        self.time_embed = nn.Embedding(max_time, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, F, T = x.shape
        freq_idx = torch.arange(F, device=x.device)
        time_idx = torch.arange(T, device=x.device)
        freq_pos = self.freq_embed(freq_idx).unsqueeze(1)  # (F, 1, D)
        time_pos = self.time_embed(time_idx).unsqueeze(0)  # (1, T, D)
        return x.permute(0, 2, 3, 1) + freq_pos + time_pos  # (B, F, T, D)


class GraphAttentionLayer(nn.Module):
    """Multi-head self-attention over flattened spectro-temporal patches."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class WaveEncoder(nn.Module):
    """1D waveform encoder with MFM activation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.blocks = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class AASIST(nn.Module):
    """AASIST model for voice anti-spoofing.

    Combines spectrogram-based LCNN features with raw waveform features,
    processes the 2D feature map through graph attention, and fuses
    with waveform stream for final classification.
    """

    def __init__(
        self,
        sr: int = 16000,
        d_model: int = 256,
        n_heads: int = 4,
        n_graph_layers: int = 2,
        dropout: float = 0.2,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.augment = augment
        self.mel_transform = audio_tx.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(0.025 * sr),
            hop_length=int(0.010 * sr),
            n_mels=80,
            power=2.0,
        )
        self.to_db = audio_tx.AmplitudeToDB(stype="power", top_db=80)

        self.spec_enc = SpecEncoder(channels=(1, 32, 32, 64, 64, 128))
        self.pos_enc = PositionalEncoding2D(d_model, max_freq=20, max_time=300)

        self.proj = nn.Linear(128, d_model)

        self.graph_layers = nn.ModuleList(
            [GraphAttentionLayer(d_model, n_heads, dropout) for _ in range(n_graph_layers)]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.wave_enc = WaveEncoder()
        self.fusion = nn.Linear(d_model + 128, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        if augment:
            from .dsfnet import AudioAugment, SpecAugment

            self.audio_aug = AudioAugment(sample_rate=sr)
            self.spec_aug = SpecAugment(freq_mask=8, time_mask=30)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dtype != torch.float32:
            waveform = waveform.float()

        if self.augment and hasattr(self, "audio_aug"):
            waveform = self.audio_aug(waveform)

        # Spectrogram path
        mel = self.to_db(self.mel_transform(waveform))  # (B, 1, 80, T')
        if self.augment and hasattr(self, "spec_aug"):
            mel = self.spec_aug(mel)

        feat_spec = self.spec_enc(mel)  # (B, 256, F', T')
        feat_spec = self.proj(feat_spec.permute(0, 2, 3, 1))  # (B, F', T', D)
        feat_spec = self.pos_enc(feat_spec.permute(0, 3, 1, 2))  # (B, F', T', D)

        B, F, T, D = feat_spec.shape
        patches = feat_spec.view(B, F * T, D)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls, patches], dim=1)

        for layer in self.graph_layers:
            patches = layer(patches)

        cls_out = patches[:, 0]  # (B, D)

        # Waveform path
        feat_wave = self.wave_enc(waveform)  # (B, 256)

        fused = self.fusion(torch.cat([cls_out, feat_wave], dim=-1))
        return self.head(fused)

    def predict(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(waveform)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def AASISTLarge(sr: int = 16000, augment: bool = False) -> AASIST:  # noqa: N802
    return AASIST(sr=sr, d_model=384, n_heads=6, n_graph_layers=3, dropout=0.2, augment=augment)
