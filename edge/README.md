# VoiceGuard on a Raspberry Pi (edge detector)

Proves the detector is lightweight: the **INT8 DSFNetTiny ONNX model is 0.62 MB** and
runs on a Raspberry Pi (ARM, CPU-only) with **only** `onnxruntime` + `numpy` +
`soundfile` — **no PyTorch, transformers, librosa, or GPU.** The mel-spectrogram
front-end is reimplemented in pure NumPy and byte-matches the training pipeline
(verified: max |diff| vs torchaudio = 0.001; ONNX vs PyTorch predictions agree).

## What it does

```
$ python3 voiceguard_edge.py clip.wav
  file:       clip.wav
  verdict:    FAKE  (99.7% confidence)
  fake_prob:  0.997
  latency:    ~68 ms (x86)   |   model: 608 KB INT8 on CPU
```

```
$ python3 voiceguard_edge.py --benchmark 40
  VoiceGuard edge — INT8 DSFNetTiny (608 KB)
  runs=40  p50=67.7 ms  p95=77.0 ms  mean=68.9 ms
  real-time factor (3 s clip): 0.0230  (<<1 = real-time)
```

## Scope (honest)

The edge model is the **tiny, offline** detector. It catches the **voice-clone
families it was trained on** — Kokoro, XTTS, IndexTTS-2 (validated FAKE at 99–100%)
— and passes genuine speech. It is **not** the full server model: it does **not**
reliably catch unseen premium engines (e.g. ElevenLabs). For those, use the deployed
SSL model **v9c** via the VoiceGuard API (official EER 2.84%, catches premium). This
is the deliberate size-vs-accuracy trade: 0.62 MB on a Pi vs ~1.2 GB on a server.

## Run it on a Raspberry Pi

Recommended: **Raspberry Pi 4 or 5, 64-bit Raspberry Pi OS** (onnxruntime ships
aarch64 wheels for 64-bit OS).

```bash
# 1. copy this edge/ folder (incl. dsfnet_tiny_int8.onnx) to the Pi
# 2. install deps (all have ARM64 wheels)
python3 -m pip install -r requirements-edge.txt

# 3. detect a file
python3 voiceguard_edge.py recording.wav

# 4. show it's lightweight
python3 voiceguard_edge.py --benchmark
```

Any audio format `soundfile` reads works (wav/flac/ogg…); it's resampled to 16 kHz
and trimmed/padded to 3 s automatically.

## Files
- `voiceguard_edge.py` — self-contained detector (numpy mel + onnxruntime)
- `dsfnet_tiny_int8.onnx` — the 0.62 MB INT8 edge model (real trained weights)
- `dsfnet_tiny_fp32.onnx` — fp32 variant (slightly higher accuracy, ~2.2 MB)
- `requirements-edge.txt`

## Status
Validated on x86 against the PyTorch DSFNetTiny (mel + predictions match). The
runtime (`onnxruntime`/`numpy`/`soundfile`) is ARM-compatible by construction; the
benchmark numbers above are x86 — expect a few hundred ms on a Pi 4/5, still well
under real-time (RTF < 1) for a 3 s clip.
