# VoiceGuard

[![Build](https://github.com/MohammadThabetHassan/VoiceGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/VoiceGuard/actions)
[![Coverage](https://img.shields.io/badge/coverage-≥80%25-brightgreen)](https://github.com/MohammadThabetHassan/VoiceGuard)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**VoiceGuard** is an AI-powered platform for real-time voice deepfake detection,
adversarial speech synthesis, and vishing (voice phishing) defence. The
production detector is an **XLS-R-300M + AASIST** model (headline eval EER 2.61%
on ASVspoof 2021 LA), alongside a validated classical baseline (Enhanced+XGBoost,
F1=0.9500) and the dual-stream DSFNet (research/edge model). It adds explainable
AI (Integrated Gradients), local **Kokoro-82M** synthesis with spectral
watermarking, and a Twilio WebSocket pipeline for live call interception — all
delivered through a FastAPI backend and React 18 frontend (self-hosted
Nginx + systemd; Docker Compose also provided).

Developed as a graduation project (GP2) at the Canadian University Dubai, in
fulfilment of the requirements for the Bachelor of Science in Computer Science.

---

## Running it

The frontend and API are served **same-origin** (`/api`): the React build is
static and the FastAPI backend runs locally — no audio leaves the host.

1. Start the API (deps installed; the package is run via `PYTHONPATH`):
   ```bash
   PYTHONPATH=src \
   XLS_R_AASIST_PATH=models/xls_r_aasist.pt \
   SECRET_KEY="$(openssl rand -hex 32)" \
   uvicorn voiceguard.api.main:app --host 127.0.0.1 --port 8000
   ```
2. Build & serve the frontend against `/api` (or use the combined dev server).
3. Detect and Generate run against the local backend. Demo login `admin` /
   `voiceguard2026` (change it — see [SECURITY.md](SECURITY.md)).

> `/detect` requires a JWT from `POST /token` (stored as `vg_token`). The
> production detector defaults to `xls_r_aasist`; set `XLS_R_AASIST_PATH` to the
> ~1.2 GB checkpoint (not in git) or `/detect` returns 503.

**Self-hosted deployment** (Nginx + systemd) is scripted in [`deploy/`](deploy/)
(`setup.sh`, `update.sh`, `DNS_SETUP.md`). Behind NAT, expose the API with a
userspace tunnel (e.g. `ngrok http 8000`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     VoiceGuard Platform                  │
├────────────────┬────────────────┬────────────────────────┤
│  React 18 UI   │  FastAPI 0.104 │  Twilio WebSocket VoIP │
│  Vite + Tailwind│  + JWT + TLS  │  (rolling deepfake score│
├────────────────┴────────────────┴────────────────────────┤
│                    Detection Engine                       │
│   XLS-R-300M + AASIST head (production, eval EER 2.61%)  │
│   + DSFNet / Wav2Vec2 / WavLM (research models)          │
│   + Enhanced+XGBoost baseline (F1=0.9500, 475 samples)   │
├──────────────────────────────────────────────────────────┤
│              Explainability & Forensics                   │
│   Integrated Gradients (captum) · SHA-256 audit chain    │
│   NIST SP 800-86 PDF report · spectral watermarking      │
├──────────────────────────────────────────────────────────┤
│                  Infrastructure                           │
│   Nginx + systemd / Docker · local RTX 5090 GPU          │
│   UAE PDPL compliant (no persistent raw audio, ≤60s TTL) │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone and launch
git clone https://github.com/MohammadThabetHassan/VoiceGuard.git
cd VoiceGuard
cp .env.example .env          # fill in TWILIO_*, JWT_SECRET, etc.
docker compose up --build
```

Frontend: http://localhost:3000 · API docs: http://localhost:8000/docs

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection model (production) | XLS-R-300M + AASIST head (PyTorch, transformers) |
| Other detectors | DSFNet, AASIST, Wav2Vec2 / WavLM |
| Classical baseline | XGBoost + hand-crafted 30-D features (librosa) |
| Backend | FastAPI 0.104+, Python 3.12, JWT, slowapi |
| Frontend | React 18, Vite, Tailwind CSS |
| Synthesis | Kokoro-82M (local TTS) + spectral watermarking |
| XAI | captum (Integrated Gradients) |
| Edge | ONNX INT8 (DSFNetTiny, 0.62 MB) |
| VoIP | Twilio Media Streams (WebSocket) |
| Infrastructure | Nginx + systemd, Docker Compose, GitHub Actions |
| Security | OWASP ASVS L2, bandit, semgrep |

---

## Edge Deployment (ONNX INT8)

`DSFNetTiny` (554K params) is the edge-target detector. Export it and apply
INT8 dynamic quantization with:

```bash
python scripts/export_onnx.py            # → checkpoints/onnx/dsfnet_tiny_int8.onnx
python scripts/export_onnx.py --no-quantize   # fp32 only
```

Validated export (size and CPU latency are weight-independent):

| Artifact | Size | CPU latency (p50) | Target |
|---|---|---|---|
| `dsfnet_tiny_fp32.onnx` | 2.23 MB | 26 ms | — |
| `dsfnet_tiny_int8.onnx` | **0.62 MB** | **30 ms** | <2 MB / <200 ms ✓ |

> The export pipeline, file size, and latency targets are validated. Accuracy is
> **not** claimed here — no trained `DSFNetTiny` checkpoint exists yet, so the
> export uses random weights for shape/size/latency validation. Pass
> `--checkpoint <path>` once a tiny model is trained. The `.onnx` artifacts are
> git-ignored (regenerable); see `checkpoints/onnx/export_report.json`.

---

## Acceptance Criteria

| Metric | Target | Status |
|---|---|---|
| Detector EER (ASVspoof 2021 LA, eval partition) | < 0.5% | 2.61% (XLS-R+AASIST; <0.5% not met — see KB Results) |
| Enhanced+XGBoost F1 (475 samples) | 0.9500 | ✅ 0.9500 (SM2026 published result) |
| Edge model size (INT8 ONNX) | < 2 MB | ✅ 0.62 MB (`DSFNetTiny`) |
| Edge inference latency (CPU) | ≤ 200 ms | ✅ p50 30 ms (INT8) |
| UAE PDPL raw audio retention | ≤ 60 seconds | ✅ background auto-delete |

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Mohammad Thabet Hassan | 20220002188 | DSFNet architecture, FastAPI backend, CI/CD, Docker |
| Fahad Sadek Al-Jazzeri | 20220001790 | Feature extraction, classical ML baseline, Wav2Vec2, evaluation |
| Ahmed Sami Alameri | 20220001166 | React frontend, speech synthesis, watermarking, Twilio VoIP, forensics, XAI |

**Supervisor:** Dr. Arash Kermani Kolankeh
**Institution:** Canadian University Dubai, Faculty of Engineering and Applied Sciences
**Academic Year:** 2025–2026

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{voiceguard2026,
  title     = {VoiceGuard: Real-Time Voice Deepfake Detection and Adversarial
               Speech Synthesis with Explainable AI},
  author    = {Hassan, Mohammad Thabet and Al-Jazzeri, Fahad Sadek and
               Alameri, Ahmed Sami},
  booktitle = {Proceedings of SM2026},
  year      = {2026},
  institution = {Canadian University Dubai}
}
```

---

## License

Licensed under the [Apache License 2.0](LICENSE).
