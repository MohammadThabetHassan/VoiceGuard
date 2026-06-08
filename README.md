# VoiceGuard

[![Build](https://github.com/MohammadThabetHassan/VoiceGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/VoiceGuard/actions)
[![Coverage](https://img.shields.io/badge/coverage-≥80%25-brightgreen)](https://github.com/MohammadThabetHassan/VoiceGuard)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**VoiceGuard** is an AI-powered platform for real-time voice deepfake detection,
adversarial speech synthesis, and vishing (voice phishing) defence. It combines
a novel Dual-Stream Fusion Network (DSFNet) trained on ASVspoof 2021 LA with a
validated classical baseline (Enhanced+XGBoost, F1=0.9500), explainable AI
outputs (Grad-CAM, SHAP), C2PA v1.4 watermarking on synthesised audio, and a
Twilio WebSocket pipeline for live call interception — all delivered through a
FastAPI backend and React 18 frontend in a single Docker Compose deployment.

Developed as a graduation project (GP2) at the Canadian University Dubai, in
fulfilment of the requirements for the Bachelor of Science in Computer Science.

---

## Live Demo

**Frontend:** https://mohammadthabethassan.github.io/VoiceGuard/

The hosted frontend is **static** — it runs entirely in your browser and talks
directly to a FastAPI backend running **on your own machine**. No audio is sent
to any remote server; all inference stays local.

**To use it with your local model:**

1. Install backend dependencies and start the API:
   ```bash
   pip install -e .
   uvicorn voiceguard.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Open the live demo, click the **⚙ gear** (bottom-right), enter your API URL
   (e.g. `http://localhost:8000`), and click **Test Connection** (✅ = reachable).
3. The Detect and Generate tabs now run against your local backend.

> **Share with others:** expose your local API with `ngrok http 8000` and paste
> the resulting `https://…ngrok.io` URL into the gear panel (under *Advanced*).
>
> **Note:** `/detect` requires a JWT (obtain one from `POST /token`) stored in
> `localStorage` as `vg_token`. `/synthesize` and `/forensic/report` return 501
> until the GPU stack is enabled.

Deployment is automated: pushing to `main` runs
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml), which builds the
Vite app and publishes `frontend/dist` to the `gh-pages` branch. Enable it once
via **Settings → Pages → Source: `gh-pages` branch**.

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
│   DSFNet (dual-stream: raw waveform + Mel-spectrogram)   │
│   Bidirectional cross-attention · EER target <0.5%       │
│   + Enhanced+XGBoost baseline (F1=0.9500, 475 samples)   │
├──────────────────────────────────────────────────────────┤
│              Explainability & Forensics                   │
│   Grad-CAM (captum) · SHAP · SHA-256 audit chain         │
│   NIST SP 800-86 PDF report · C2PA v1.4 watermarking     │
├──────────────────────────────────────────────────────────┤
│                  Infrastructure                           │
│   Docker Compose · AWS S3 checkpointing · g5.xlarge GPU  │
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
| Detection model | DSFNet (PyTorch 2.x), Wav2Vec2 (Hugging Face) |
| Classical baseline | XGBoost + hand-crafted 30-D features (librosa) |
| Backend | FastAPI 0.104+, Python 3.12, JWT, slowapi |
| Frontend | React 18, Vite, Tailwind CSS |
| Synthesis | Coqui XTTS v2 + C2PA v1.4 watermarking |
| XAI | captum (Grad-CAM), SHAP |
| VoIP | Twilio Media Streams (WebSocket) |
| Infrastructure | Docker Compose, AWS S3, GitHub Actions |
| Security | TLS 1.3, OWASP ASVS L2, bandit, semgrep, SBOM |

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
| Ahmed Sami Alameri | 20220001166 | React frontend, XTTS synthesis, watermarking, Twilio VoIP, forensics, XAI |

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
