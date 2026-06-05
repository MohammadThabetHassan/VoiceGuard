"""
DSFNet — Dual-Stream Fusion Network for voice deepfake detection.

Variants:
  DSFNet (base):    5 block wave encoder, 4 ResBlock spec encoder, 9.26M params
  DSFNetLarge:      6 block wave encoder, 5 ResBlock spec encoder, ~14M params
  DSFNetXL:         7 block wave encoder, 6 ResBlock spec encoder, ~18M params

Input:
  waveform   — raw 16kHz float32 tensor (B, 1, T)
  spectrogram — 80-bin Mel-spectrogram (B, 1, F, T')

Output: logits (B, 2) — class 0=genuine, class 1=synthetic
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as audio_fn
import torchaudio.transforms as audio_tx


class ConvBlock1D(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 9, pool: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveformEncoder(nn.Module):
    """Stream A: 1D-CNN blocks, configurable depth."""

    def __init__(self, channels: tuple[int, ...] = (1, 32, 64, 128, 256, 512)) -> None:
        super().__init__()
        blocks = []
        for i in range(len(channels) - 1):
            pool = 2 if i == len(channels) - 2 else 4  # smaller pool at last block
            blocks.append(ConvBlock1D(channels[i], channels[i + 1], pool=pool))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 1, T). Returns (B, C_out)."""
        out = self.blocks(x)
        return self.pool(out).squeeze(-1)


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
    """Stream B: 2D-ResNet blocks, configurable depth."""

    def __init__(self, channels: tuple[int, ...] = (64, 64, 128, 256, 512)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(len(channels) - 1):
            stride = 1 if channels[i] == channels[i + 1] else 2
            blocks.append(ResBlock2D(channels[i], channels[i + 1], stride=stride))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 1, F, T). Returns (B, C_out)."""
        out = self.stem(x)
        out = self.blocks(out)
        return self.pool(out).squeeze(-1).squeeze(-1)


class BidirectionalCrossAttention(nn.Module):
    """Symmetric cross-attention between two 512-dim token sequences.

    Q from A, K/V from B → attended_A
    Q from B, K/V from A → attended_B
    Output: concatenation → (B, 1024)
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj_a = nn.Linear(d_model, d_model)
        self.k_proj_b = nn.Linear(d_model, d_model)
        self.v_proj_b = nn.Linear(d_model, d_model)
        self.out_a = nn.Linear(d_model, d_model)
        self.q_proj_b = nn.Linear(d_model, d_model)
        self.k_proj_a = nn.Linear(d_model, d_model)
        self.v_proj_a = nn.Linear(d_model, d_model)
        self.out_b = nn.Linear(d_model, d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def _attention(self, q, k, v):
        B, _, _ = q.shape
        q = q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return out

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Args: feat_a, feat_b both (B, 512). Returns (B, 1024)."""
        a = feat_a.unsqueeze(1)  # (B, 1, 512)
        b = feat_b.unsqueeze(1)
        q_a, k_b, v_b = self.q_proj_a(a), self.k_proj_b(b), self.v_proj_b(b)
        out_a = self.out_a(self._attention(q_a, k_b, v_b))
        out_a = self.norm_a(out_a + a).squeeze(1)  # (B, d_model)
        q_b, k_a, v_a = self.q_proj_b(b), self.k_proj_a(a), self.v_proj_a(a)
        out_b = self.out_b(self._attention(q_b, k_a, v_a))
        out_b = self.norm_b(out_b + b).squeeze(1)  # (B, d_model)
        return torch.cat([out_a, out_b], dim=-1)  # (B, 1024)


class ClassificationHead(nn.Module):
    """MLP classifier with configurable hidden dims."""

    def __init__(self, dims: tuple[int, ...] = (1024, 512, 256, 128, 2),
                 dropout: float = 0.3) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioAugment(nn.Module):
    """On-the-fly waveform augmentation targeting 2019→2021 domain shift.

    Applied only when self.training is True. Augmentations are applied
    stochastically to each batch. Codec/channel augmentations (µ-law,
    bitcrushing) are critical for closing the 2021 LA telephone-channel gap.

    Augmentations:
      - Random gain: [0.7, 1.3]
      - Gaussian noise: SNR [15, 30] dB
      - Random time masking: up to 10% of samples
      - Simulated reverb (RIR convolution), p=0.5
      - µ-law companding (G.711 telephone codec simulation), p=0.5
      - Bitcrushing (4–8 bits, simulates low-bitrate codec artifacts), p=0.4
      - Telephone bandpass filtering [300, 3400] Hz, p=0.3
      - Packet loss simulation (random zero-out of 20–80ms), p=0.2
      - Clipping distortion (tanh), p=0.3
    """

    MU = 255.0  # G.711 µ-law constant

    def __init__(self, sample_rate: int = 16000, rir_len: int = 8000) -> None:
        super().__init__()
        self.sr = sample_rate
        rir = torch.randn(rir_len)
        t = torch.arange(rir_len, dtype=torch.float32)
        rir = rir * torch.exp(-t / (sample_rate * 0.05))
        self.register_buffer('rir_base', rir / rir.norm())

        # Precompute bandpass kernels once (HP ~300 Hz + LP ~3400 Hz).
        # Registered as buffers so they move to GPU with the module.
        hp_len = 63  # ≥6σ for σ=sr/(2π·300)≈8.5 samples — avoids truncation distortion
        sigma_hp = sample_rate / (2 * math.pi * 300.0)
        t_hp = torch.arange(-(hp_len // 2), hp_len // 2 + 1, dtype=torch.float32)
        lp_hp = torch.exp(-0.5 * (t_hp / sigma_hp).pow(2))
        self.register_buffer('hp_kernel', (lp_hp / lp_hp.sum()).view(1, 1, -1))

        lp_len = 63
        sigma_lp = sample_rate / (2 * math.pi * 3400.0)
        t_lp = torch.arange(-(lp_len // 2), lp_len // 2 + 1, dtype=torch.float32)
        lp_3400 = torch.exp(-0.5 * (t_lp / sigma_lp).pow(2))
        self.register_buffer('lp3400_kernel', (lp_3400 / lp_3400.sum()).view(1, 1, -1))

    @staticmethod
    def _mulaw_compress(x: torch.Tensor, mu: float = 255.0) -> torch.Tensor:
        """µ-law encode then decode (net effect: non-linear quantization noise)."""
        x = x.clamp(-1.0, 1.0)
        encoded = x.sign() * torch.log1p(mu * x.abs()) / math.log(1.0 + mu)
        # Quantize encoded signal to 8 bits
        quantized = (encoded * 128).round().clamp(-128, 127) / 128.0
        decoded = quantized.sign() * ((1.0 + mu) ** quantized.abs() - 1.0) / mu
        return decoded

    @staticmethod
    def _bitcrush(x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Reduce bit depth to n_bits (4–8) then restore scale — codec artifact sim."""
        levels = 2 ** (n_bits - 1)
        return (x * levels).round().clamp(-levels, levels - 1) / levels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return waveform
        x = waveform.float()
        with torch.no_grad():
            B, C, T = x.shape

            # Gain
            gain = torch.empty(B, 1, 1, device=x.device).uniform_(0.7, 1.3)
            x = x * gain

            # Additive Gaussian noise
            noise_power = x.pow(2).mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            snr_db = torch.empty(B, 1, 1, device=x.device).uniform_(15, 30)
            noise_std = (noise_power / (10 ** (snr_db / 10))).sqrt()
            x = x + torch.randn_like(x) * noise_std

            # Time masking
            mask_len = int(T * 0.1)
            for i in range(B):
                start = torch.randint(0, max(1, T - mask_len), (1,)).item()
                x[i, :, start:start + mask_len] = 0.0

            # All stochastic augmentations below apply independently per sample.
            # Per-sample gates avoid the bias where a whole batch is either fully
            # augmented or fully clean, which inflates gradient variance.

            # Reverb (per-sample, p=0.5)
            for i in range(B):
                if torch.rand(1).item() < 0.5:
                    rir = self.rir_base * torch.empty(1).uniform_(0.3, 1.0).item()
                    rir = rir.to(x.device)
                    xi = F.pad(x[i:i+1], (len(rir) // 2, len(rir) // 2))
                    x[i:i+1] = F.conv1d(xi, rir.view(1, 1, -1), padding=0)[..., :T]

            # µ-law companding per-sample, p=0.5 (G.711 telephone codec simulation)
            for i in range(B):
                if torch.rand(1).item() < 0.5:
                    norm = x[i].abs().amax().clamp(min=1e-6)
                    x[i] = self._mulaw_compress(x[i] / norm) * norm

            # Bitcrushing per-sample, p=0.4 (low-bitrate codec artifact simulation)
            for i in range(B):
                if torch.rand(1).item() < 0.4:
                    n_bits = torch.randint(4, 9, (1,)).item()
                    norm = x[i].abs().amax().clamp(min=1e-6)
                    x[i] = self._bitcrush(x[i] / norm, n_bits) * norm

            # Telephone bandpass [300–3400 Hz] per-sample, p=0.3.
            # HP at ~300 Hz (Gaussian LP subtraction) + LP at 3400 Hz.
            for i in range(B):
                if torch.rand(1).item() < 0.3:
                    xi = x[i:i+1]
                    # High-pass at 300 Hz using precomputed registered buffer
                    x_lp = F.conv1d(xi, self.hp_kernel, padding=self.hp_kernel.shape[-1] // 2)
                    xi = xi - x_lp
                    # Low-pass at 3400 Hz using precomputed registered buffer
                    xi = F.conv1d(xi, self.lp3400_kernel, padding=self.lp3400_kernel.shape[-1] // 2)
                    x[i:i+1] = xi

            # Packet loss per-sample, p=0.2
            for i in range(B):
                if torch.rand(1).item() < 0.2:
                    loss_len = int(torch.randint(
                        int(0.02 * self.sr), int(0.08 * self.sr) + 1, (1,)).item())
                    start = torch.randint(0, max(1, T - loss_len), (1,)).item()
                    x[i, :, start:start + loss_len] = 0.0

            # Clipping distortion per-sample, p=0.3
            for i in range(B):
                if torch.rand(1).item() < 0.3:
                    clip_thresh = torch.empty(1, device=x.device).uniform_(0.5, 0.9).item()
                    x[i] = clip_thresh * torch.tanh(x[i] / max(clip_thresh, 1e-6))
        return x


class SpecAugment(nn.Module):
    """Spectrogram augmentation (applied only during training).

    - Frequency masking: 0-10 mel bins
    - Time masking: 0-40 time frames
    """

    def __init__(self, freq_mask: int = 10, time_mask: int = 40) -> None:
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        # Clone to avoid in-place mutation on a tensor that may have a grad_fn,
        # which would corrupt the autograd version counter at backward().
        spec = spec.clone()
        with torch.no_grad():
            B, _, F, T = spec.shape
            # Apply independent masks per sample (standard SpecAugment, Park et al. 2019).
            for i in range(B):
                f = torch.randint(0, self.freq_mask + 1, (1,)).item()
                if f > 0:
                    f_start = torch.randint(0, max(1, F - f), (1,)).item()
                    spec[i, :, f_start:f_start + f, :] = 0.0
                t = torch.randint(0, self.time_mask + 1, (1,)).item()
                if t > 0:
                    t_start = torch.randint(0, max(1, T - t), (1,)).item()
                    spec[i, :, :, t_start:t_start + t] = 0.0
        return spec


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
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        mel = self.transform(waveform)
        return self.to_db(mel)


class DSFNet(nn.Module):
    """Base DSFNet — Dual-Stream Fusion Network.

    Args:
        wave_channels: channel dims for WaveformEncoder
        spec_channels: channel dims for SpectrogramEncoder (after stem 1→64)
        d_model: cross-attention dimension
        head_dims: MLP hidden dims
        dropout: dropout rate
        sr: sample rate
        augment: enable on-the-fly augmentation
    """

    def __init__(
        self,
        wave_channels: tuple[int, ...] = (1, 32, 64, 128, 256, 512),
        spec_channels: tuple[int, ...] = (64, 64, 128, 256, 512),
        d_model: int = 512,
        head_dims: tuple[int, ...] = (1024, 512, 256, 128, 2),
        dropout: float = 0.3,
        sr: int = 16000,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.augment = augment
        self.wave_enc = WaveformEncoder(channels=wave_channels)
        self.spec_enc = SpectrogramEncoder(channels=spec_channels)
        self.mel_transform = MelSpectrogramTransform(sr=sr)
        self.cross_attn = BidirectionalCrossAttention(d_model=d_model, n_heads=8)
        self.head = ClassificationHead(dims=head_dims, dropout=dropout)
        if augment:
            self.audio_aug = AudioAugment(sample_rate=sr)
            self.spec_aug = SpecAugment()

    def forward(
        self,
        waveform: torch.Tensor,
        spectrogram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if self.augment and hasattr(self, 'audio_aug'):
            waveform = self.audio_aug(waveform)
        if spectrogram is None:
            spectrogram = self.mel_transform(waveform)
        if self.augment and hasattr(self, 'spec_aug'):
            spectrogram = self.spec_aug(spectrogram)

        feat_a = self.wave_enc(waveform)
        feat_b = self.spec_enc(spectrogram)
        fused = self.cross_attn(feat_a, feat_b)
        return self.head(fused)

    def predict(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(waveform)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def DSFNetLarge(dropout: float = 0.3, sr: int = 16000, augment: bool = False) -> DSFNet:
    """Larger variant: ~14M params."""
    return DSFNet(
        wave_channels=(1, 64, 128, 256, 384, 512, 640),
        spec_channels=(64, 128, 256, 384, 512, 640),
        d_model=640,
        head_dims=(1280, 768, 384, 128, 2),
        dropout=dropout,
        sr=sr,
        augment=augment,
    )


def DSFNetXL(dropout: float = 0.3, sr: int = 16000, augment: bool = False) -> DSFNet:
    """Extra-large variant: ~18M params."""
    return DSFNet(
        wave_channels=(1, 64, 128, 256, 384, 512, 640, 768),
        spec_channels=(64, 128, 256, 384, 512, 640, 768),
        d_model=768,
        head_dims=(1536, 1024, 512, 256, 128, 2),
        dropout=dropout,
        sr=sr,
        augment=augment,
    )


def DSFNetTiny(dropout: float = 0.2, sr: int = 16000, augment: bool = False) -> DSFNet:
    """Tiny variant for edge deployment: ~450K params, targets INT8 < 2MB.

    Designed for ONNX INT8 export: 450K fp32 params → ~450KB INT8.
    """
    return DSFNet(
        wave_channels=(1, 16, 32, 64, 128),
        spec_channels=(64, 64, 128),
        d_model=128,
        head_dims=(256, 64, 2),
        dropout=dropout,
        sr=sr,
        augment=augment,
    )


class WaveformEncoderSeq(nn.Module):
    """Waveform encoder that returns the temporal sequence (not globally pooled)."""

    def __init__(self, channels: tuple[int, ...] = (1, 32, 64, 128, 256, 512)) -> None:
        super().__init__()
        blocks = []
        for i in range(len(channels) - 1):
            pool = 2 if i == len(channels) - 2 else 4
            blocks.append(ConvBlock1D(channels[i], channels[i + 1], pool=pool))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, C, T') — temporal sequence preserved."""
        return self.blocks(x)


class DSFNetV2(nn.Module):
    """Improved DSFNet where cross-attention operates over temporal sequences.

    The original DSFNet applied global average pooling *before* cross-attention,
    making the attention between two single-token sequences — equivalent to a
    linear transform with no actual attention benefit.

    V2 keeps the temporal dimension through both encoders and applies cross-
    attention over T frames, then pools after fusion. This allows the model to
    identify *which time segments* contain spoofing artifacts.

    Args:
        wave_channels: channel dims for waveform encoder
        spec_channels: channel dims for spectrogram encoder
        d_model: transformer d_model
        n_heads: number of attention heads
        n_layers: number of cross-attention + self-attention layers
        dropout: dropout rate
        sr: sample rate
        augment: enable AudioAugment + SpecAugment during training
    """

    def __init__(
        self,
        wave_channels: tuple[int, ...] = (1, 32, 64, 128, 256, 512),
        spec_channels: tuple[int, ...] = (64, 64, 128, 256, 512),
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.3,
        sr: int = 16000,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.augment = augment
        wave_out_ch = wave_channels[-1]
        spec_out_ch = spec_channels[-1]

        self.wave_enc = WaveformEncoderSeq(channels=wave_channels)
        self.spec_enc = SpectrogramEncoder(channels=spec_channels)  # pools to (B, C)
        self.mel_transform = MelSpectrogramTransform(sr=sr)

        # Project both streams to d_model
        self.wave_proj = nn.Linear(wave_out_ch, d_model)
        self.spec_proj = nn.Linear(spec_out_ch, d_model)

        # Cross-attention: wave sequence (query) attends to spec (key/value)
        self.cross_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Self-attention on wave sequence after cross-attn enrichment
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                       dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model + spec_out_ch),
            nn.Linear(d_model + spec_out_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        if augment:
            self.audio_aug = AudioAugment(sample_rate=sr)
            self.spec_aug = SpecAugment()

    def forward(self, waveform: torch.Tensor,
                spectrogram: torch.Tensor | None = None) -> torch.Tensor:
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if self.augment and hasattr(self, 'audio_aug'):
            waveform = self.audio_aug(waveform)
        if spectrogram is None:
            spectrogram = self.mel_transform(waveform)
        if self.augment and hasattr(self, 'spec_aug'):
            spectrogram = self.spec_aug(spectrogram)

        # Wave: (B, wave_out_ch, T') → (B, T', d_model)
        wave_seq = self.wave_enc(waveform).permute(0, 2, 1)  # (B, T', C)
        wave_seq = self.wave_proj(wave_seq)                  # (B, T', d_model)

        # Spec: (B, spec_out_ch) → (B, 1, d_model) — serves as context token
        spec_feat = self.spec_enc(spectrogram)               # (B, spec_out_ch)
        spec_ctx = self.spec_proj(spec_feat).unsqueeze(1)    # (B, 1, d_model)

        # Cross-attention: wave queries, spec context as key/value
        x = wave_seq
        for cross, norm in zip(self.cross_layers, self.cross_norms):
            attn_out, _ = cross(x, spec_ctx, spec_ctx)
            x = norm(x + attn_out)

        # Self-attention over wave sequence
        for self_layer in self.self_layers:
            x = self_layer(x)

        # Pool over time, concatenate with spec features
        wave_pooled = x.mean(dim=1)                          # (B, d_model)
        fused = torch.cat([wave_pooled, spec_feat], dim=-1)  # (B, d_model + spec_out_ch)
        return self.head(fused)

    def predict(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(waveform)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
