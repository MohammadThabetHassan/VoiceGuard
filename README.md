<div align="center">

<img src="assets/banner.png" alt="VoiceGuard" width="100%">

# VoiceGuard

**Real-time voice deepfake detection, synthesis watermarking, and vishing defence.**

[![CI](https://github.com/MohammadThabetHassan/VoiceGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/VoiceGuard/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![IEEE SM2026](https://img.shields.io/badge/IEEE-SM2026-00629B.svg)](#-results)

<a href="#-quick-start">Quick start</a> ·
<a href="#-results">Results</a> ·
<a href="#-architecture">Architecture</a> ·
<a href="#-api">API</a> ·
<a href="CONTRIBUTING.md">Contributing</a>

</div>

---

## Why VoiceGuard?

AI voice cloning has turned phone fraud into a scalable weapon. In 2024 criminals
stole **US$25M** from a company using a deepfaked CFO on a video call, and reported
voice-phishing ("vishing") incidents **surged over 1,600%** in early 2025. Off-the-shelf
detectors collapse on *real-world* audio — phone codecs, background noise, and unseen
TTS engines — and offer no explanation a human analyst can act on.

**VoiceGuard** is an end-to-end platform that detects voice deepfakes in real time,
explains its decisions, watermarks any audio it generates, and ships small enough to
run at the edge. Built as a graduation project (GP2) at Canadian University Dubai; the
classical baseline was accepted at **IEEE SM2026**.

## ✨ Features

- 🛡️ **Detection** — production **XLS-R-300M + AASIST** model (eval EER **2.61%**), with DSFNet, Wav2Vec2/WavLM, and a classical XGBoost baseline all selectable. An input-quality guard rejects silent / too-short clips instead of guessing.
- 🌍 **Real-world robustness** — hardened against out-of-distribution TTS (Kokoro **93.3%**, IndexTTS2 **100%**) and noisy / telephony / short audio.
- 🔍 **Explainability** — Integrated-Gradients attribution shows *which moments* drove the verdict.
- 🗣️ **Synthesis + watermarking** — local **Kokoro-82M** TTS that spectrally watermarks every clip as AI-generated.
- 🧾 **Forensics** — SHA-256 chain-of-custody and NIST SP 800-86 PDF reports.
- ☎️ **VoIP** — Twilio Media Streams bridge for live call screening.
- ⚡ **Edge-ready** — ONNX INT8 export at **0.62 MB**, **~30 ms** CPU inference.

## 🖼️ Screenshots

| Detect | Generate | Results |
|:------:|:--------:|:-------:|
| ![Detect](assets/screenshots/01_detect.png) | ![Generate](assets/screenshots/03_generate.png) | ![Results](assets/screenshots/04_results.png) |

## 📊 Results

Trained on ASVspoof 2019 LA, evaluated on the full ASVspoof 2021 LA set (181,566 trials).

| Model | EER (eval) | EER (full-pool) | Role |
|-------|:----------:|:---------------:|------|
| **XLS-R + AASIST** (Kokoro-hardened) | **2.61%** | 8.21% | 🏆 headline |
| Wav2Vec2-large | 3.09% | 7.07% | baseline |
| WavLM-base-plus | 8.11% | — | baseline |
| AASIST | 10.90% | — | baseline |
| DSFNet-V2 | — | 12.67% | own architecture |

**Out-of-distribution & real-world:** IndexTTS2 100% · Kokoro 93.3% · genuine-voice
pass-rate 90% · real-world harness real-pass **87.5%** / fake-detect **90.3%**.
**Edge:** DSFNetTiny INT8 **0.62 MB**, CPU p50 **30 ms**.

> **Honest note.** The deployed checkpoint is a real-world-robustness fine-tune of the
> 2.61% Kokoro-hardened model; its ASVspoof EER was not separately benchmarked. PGD
> adversarial hardening is documented as a *negative result* (see [CHANGELOG](CHANGELOG.md)).
> The ONNX export validates size/latency with random weights (no trained tiny model yet).

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Client["React 18 UI"]
        UI["Detect · Generate · Results"]
    end
    subgraph API["FastAPI · JWT · rate-limit · PDPL auto-delete"]
        D["/detect"]
        S["/synthesize"]
        X["/explain"]
        F["/forensic/report"]
        W["/ws · /twilio"]
    end
    subgraph Engine["Detection Engine"]
        SSL["XLS-R + AASIST<br/>(production)"]
        ALT["DSFNet · Wav2Vec2 · classical"]
    end
    UI -->|audio| API
    D --> SSL & ALT
    X -->|Integrated Gradients| SSL
    S -->|Kokoro-82M + watermark| MEDIA[("/api/media")]
    F -->|SHA-256 chain · PDF| MEDIA
    SSL --> R["label · confidence · explanation"]
    R --> UI
    SSL -.ONNX INT8.-> EDGE["Edge (0.62 MB)"]
```

## 🚀 Quick start

```bash
git clone https://github.com/MohammadThabetHassan/VoiceGuard.git
cd VoiceGuard

# Backend (Python 3.12)
python3 -m venv venv && source venv/bin/activate
pip install -e .

PYTHONPATH=src SECRET_KEY="$(openssl rand -hex 32)" \
  uvicorn voiceguard.api.main:app --host 127.0.0.1 --port 8000
# API docs → http://127.0.0.1:8000/docs   (demo login: admin / voiceguard2026)

# Frontend (in another shell)
cd frontend && npm ci && npm run dev
```

The production detector (`xls_r_aasist`) needs a ~1.2 GB checkpoint (not in git);
without it, set `model=classical` or point `XLS_R_AASIST_PATH` at the checkpoint.
Self-hosted deployment (Nginx + systemd) is scripted in [`deploy/`](deploy/);
Docker Compose is also provided.

## 🔌 API

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|:----:|
| `POST` | `/token` | OAuth2 password → JWT | — |
| `POST` | `/detect` | Audio → verdict (`?model=`, `?explain=true`) | 🔑 |
| `POST` | `/explain` | Integrated-Gradients attribution | 🔑 |
| `POST` | `/synthesize` | Text → watermarked Kokoro speech | 🔑 |
| `POST` | `/forensic/report` | NIST SP 800-86 PDF report | 🔑 |
| `WS` | `/ws/stream` · `/twilio/stream` | Live mic / Twilio call screening | 🔑 |
| `GET` | `/models` · `/health` · `/docs` | Ops & Swagger | — |

## 🛠️ Tech stack

**ML** PyTorch · transformers (XLS-R, Wav2Vec2, WavLM) · AASIST · XGBoost · captum · ONNX Runtime
· **Backend** FastAPI · python-jose (JWT) · slowapi · **Frontend** React 18 · Vite · Tailwind · Recharts
· **Audio** librosa · torchaudio · Kokoro-82M · **Infra** Nginx + systemd · Docker · GitHub Actions · ruff · bandit

## 🗺️ Roadmap

- [ ] Train `DSFNetTiny` so the ONNX edge export carries accuracy
- [ ] Backbone adversarial fine-tuning for true PGD robustness
- [ ] GADC (Gulf-Arabic Deepfake Corpus) + human perception study
- [ ] Permanent hosted demo

## 🤝 Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). For vulnerabilities, see [SECURITY.md](SECURITY.md).

## 👥 Team

| Name | Role |
|------|------|
| **Mohammad Thabet Hassan** | Detection architecture, FastAPI backend, CI/CD, deployment |
| **Fahad Sadek Al-Jazzeri** | Feature extraction, classical ML, SSL models, evaluation |
| **Ahmed Sami Alameri** | React frontend, synthesis, watermarking, VoIP, forensics, XAI |

**Supervisor:** Dr. Arash Kermani Kolankeh · **Institution:** Canadian University Dubai · **2025–2026**

## 📚 Citation

```bibtex
@inproceedings{voiceguard2026,
  title     = {VoiceGuard: Real-Time Voice Deepfake Detection and Adversarial
               Speech Synthesis with Explainable AI},
  author    = {Hassan, Mohammad Thabet and Al-Jazzeri, Fahad Sadek and Alameri, Ahmed Sami},
  booktitle = {Proceedings of IEEE SM2026},
  year      = {2026},
  organization = {Canadian University Dubai}
}
```

## 📄 License

[Apache License 2.0](LICENSE).
