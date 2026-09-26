<div align="center">

<img src="assets/hero.png" alt="VoiceGuard" width="100%">

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
[![IEEE SM2026](https://img.shields.io/badge/IEEE-SM2026-00629B.svg)](https://doi.org/10.1109/SM69703.2026.11614145)

[**🎬 Demo video**](assets/demo.mp4) ·
<a href="#-quick-start">Quick start</a> ·
<a href="#-results">Results</a> ·
<a href="#-architecture">Architecture</a> ·
<a href="#-api">API</a> ·
<a href="#-documentation">Docs</a> ·
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

- 🛡️ **Detection** — production **XLS-R-300M + AASIST** model (official ASVspoof 2021 LA eval EER **2.84%**, v9c) that *also* catches modern voice clones and premium TTS, with DSFNet, Wav2Vec2/WavLM, and a classical XGBoost baseline all selectable. An input-quality guard rejects silent / too-short clips instead of guessing.
- 🌍 **Real-world robustness** — hardened against out-of-distribution TTS engines and noisy / telephony / short audio, with the limits *measured and documented*, not hidden (see [Results](#-results)).
- 🎙️ **Live microphone streaming** — the web app's **Live tab** streams your mic over WebSocket and shows a live real/fake verdict that re-scores the session from its start as more audio arrives (first verdict at 3s; see [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) for why not sliding windows).
- 🔍 **Explainability** — occlusion attribution (each time segment is silenced in turn and the clip re-scored) shows *which moments* drove the verdict. For the production model this covers the first 3 s in 6 segments (forward-only, so it takes seconds rather than minutes on CPU); an Integrated-Gradients implementation lives in `voiceguard.xai.ssl_explain` but is not wired to the API.
- 🗣️ **Synthesis + watermarking** — multi-engine Generate: local **Kokoro-82M** preset voices and optional **zero-shot voice cloning** (XTTS v2 / IndexTTS-2, admin-only) from a reference clip; every clip is spectrally watermarked as AI-generated and C2PA-signed. The **Verify tab** (`POST /watermark/verify`) closes the loop: prove any clip's provenance back. See [docs/SYNTHESIS_ENGINES.md](docs/SYNTHESIS_ENGINES.md).
- 🧾 **Forensics** — SHA-256 chain-of-custody and NIST SP 800-86 PDF reports.
- ☎️ **VoIP** — Twilio Media Streams bridge for live call screening.
- ⚡ **Edge-ready** — a *separate* tiny model, **DSFNetTiny** (8.47% balanced-mirror EER), exports to ONNX INT8 at **0.62 MB**, **~30 ms** CPU inference ([`edge/README.md`](edge/README.md) benchmarks p50 67.7 ms on x86 including its NumPy front-end). It covers the clone families it was trained on, not premium TTS; the XLS-R + AASIST production model stays on the server.

## 🎬 Demo

End-to-end on the recorded deployment: log in → upload a **premium ElevenLabs clip → flagged 100% FAKE** (588 ms) → generate **watermarked** speech. The hosted demo is currently offline; run it locally with the [Quick start](#-quick-start) below.

![VoiceGuard demo](assets/demo.gif)

> Full-resolution clip: [`assets/demo.mp4`](assets/demo.mp4). Recorded with Playwright (`deploy/demo_record.py`).

| Detect | Generate | Results |
|:------:|:--------:|:-------:|
| ![Detect](assets/screenshots/01_detect.png) | ![Generate](assets/screenshots/03_generate.png) | ![Results](assets/screenshots/04_results.png) |

## 📊 Results

The deployed detector — **XLS-R-300M + AASIST "v9c"** — is selected for *overall* performance,
not the lowest headline EER: a checkpoint with a lower official EER (v8, 2.49%) was **rejected**
for deployment because it is blind to modern voice clones.

| Benchmark | Result |
|-----------|--------|
| **Official ASVspoof 2021 LA eval** (181,566 trials) | **2.84% EER** [95% CI 2.67–3.02] |
| Real-audio pass rate (held-out, speaker/text-disjoint) | **96%** |
| Kokoro voice-clone detection (held-out, 100/family) | **100%** |
| XTTS v2 voice-clone detection | **100%** |
| IndexTTS-2 voice-clone detection | **97%** |
| **ElevenLabs-v3** — engine *never seen* in training | **95.8%** |

| Edge & provenance | Result |
|-------------------|--------|
| DSFNetTiny INT8 model size | **0.62 MB** |
| CPU inference latency (p50) | **~30 ms** |
| Edge model EER (trained weights, balanced mirror) | 8.47% |
| Synthesized audio provenance | signed **C2PA manifest** + spectral watermark |

<details>
<summary><b>Model lineage — why you may spot other EERs (2.61 / 2.49 / 3.38) in this repo</b></summary>

| Model | EER (eval) | EER (full-pool) | Catches clones | Catches premium TTS | Role |
|-------|:----------:|:---------------:|:--------------:|:-------------------:|------|
| **XLS-R + AASIST — v9c** | **2.84%** | 8.21% | ✓ all ≥97% | ✓ ElevenLabs 96% | 🏆 **deployed** |
| XLS-R + AASIST — v7 | 3.38% | 8.60% | ✓ all ≥96.7% | ✗ (85%) | previous production |
| XLS-R + AASIST (Kokoro-parent) | 2.61% | 8.21% | ✗ | ✗ | EER-only headline |
| XLS-R + AASIST — v8 | 2.49% | 9.91% | ✗ (Kokoro 62.5%) | — | lowest official EER |
| Wav2Vec2-large | 3.09% | 7.07% | — | — | baseline |

**On the "2.61%".** That figure is the **Kokoro-parent** checkpoint on the official
eval — **reproduced exactly from raw FLAC on 2026-06-09** (`run_official_eval.py`) — but
it does *not* catch modern clones. The deployed lineage (v7 → v9c) is measured on the
same official protocol: **v7 = 3.38%**, **v9c = 2.84%**. v9c recovers most of the EER
gap *and* catches clones + premium TTS, so it's the best model overall.

</details>

> **🔬 Reproducible & honestly bounded.** Every EER carries a 95% bootstrap CI, on a
> single provenance-tagged table, with same-protocol baselines and a fixed env manifest —
> and the hard limits are *measured*, not hidden. See the full
> [documentation index](#-documentation) below.

### Evidence and license scope

The headline results are not standalone claims. Start with [`docs/RESULTS_canonical.md`](docs/RESULTS_canonical.md) for the checkpoint identifier, dataset/protocol, confidence interval, and provenance record; use [`docs/EVAL_PROTOCOLS.md`](docs/EVAL_PROTOCOLS.md) for the exact evaluation path; and use [`docs/REPRODUCIBILITY_MANIFEST.md`](docs/REPRODUCIBILITY_MANIFEST.md) to reproduce the pinned environment. The canonical table covers the official ASVspoof EERs only; the held-out clone-family, real-pass and ElevenLabs-v3 rates are measured on a separate held-out set and are documented in [`docs/RESULTS.md`](docs/RESULTS.md) (including where the 95.8% ElevenLabs-v3 figure comes from) and [`docs/CLONE_DETECTION_LIMITS.md`](docs/CLONE_DETECTION_LIMITS.md). The deployment decision for v9c is documented in the model-lineage table above: the lowest official EER was not selected because it failed the clone- and premium-TTS-robustness requirement.

The repository code and original documentation are released under the [Apache License 2.0](LICENSE). That license does not automatically relicense third-party checkpoints, datasets, pretrained backbones, synthesis engines, fonts, images, or other bundled material. Check the relevant upstream license and attribution terms before redistributing those components or publishing a derived model. Dataset access and evaluation use must also follow the terms of the respective dataset providers.

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
    X -->|occlusion, first 3 s| SSL
    S -->|Kokoro-82M + watermark| MEDIA[("/api/media")]
    F -->|SHA-256 chain · PDF| MEDIA
    SSL --> R["label · confidence · explanation"]
    R --> UI
    ALT -.DSFNetTiny ONNX INT8.-> EDGE["Edge: separate DSFNetTiny<br/>(0.62 MB, 8.47% balanced EER)"]
```

<details>
<summary><b>📁 Repository layout</b></summary>

```
VoiceGuard/
├── src/voiceguard/        # Python package
│   ├── api/               #   FastAPI app — auth (JWT/roles), routes, middleware, WebSockets
│   ├── models/            #   XLS-R+AASIST, DSFNet, Wav2Vec2/WavLM, classical baseline
│   ├── features/          #   acoustic feature extraction
│   ├── preprocessing/     #   resampling, augmentation (RawBoost), input-quality guard
│   ├── training/          #   training loops & schedules
│   ├── evaluation/        #   EER / minDCF scoring, bootstrap CIs
│   ├── synthesis/         #   Kokoro-82M, XTTS v2, IndexTTS-2 engines
│   ├── watermark/         #   spectral watermark embed/verify + C2PA signing
│   ├── forensics/         #   SHA-256 chain-of-custody, NIST SP 800-86 PDF reports
│   ├── voip/              #   Twilio Media Streams bridge
│   └── xai/               #   Integrated Gradients, Grad-CAM, SHAP helpers (the API serves occlusion)
├── frontend/              # React 18 + Vite + Tailwind web app
├── edge/                  # 0.62 MB ONNX INT8 runtime (onnxruntime + numpy, no torch)
├── integrations/iped/     # IPED digital-forensics pipeline add-on
├── deploy/                # Nginx + systemd deployment scripts, demo recorder
├── docs/                  # results, protocols, limitations, reproducibility
└── tests/                 # 147 tests (pytest)
```

</details>

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

**Docker Compose:** `docker compose up --build` serves everything on `http://localhost`,
mounts `./checkpoints` into the backend (drop the checkpoint at
`checkpoints/xls_r_aasist/model_best.pt` or export `XLS_R_AASIST_PATH`), and falls
back to the classical baseline when no checkpoint is present.
Self-hosted deployment (Nginx + systemd) is scripted in [`deploy/`](deploy/).

### Three ways to run it

| Mode | What | Install |
|------|------|---------|
| 🌐 **Web app / API** | Full SSL model **v9c** (catches clones + premium); web app + REST API | this Quick start (hosted demo currently offline) |
| 🔬 **IPED forensic add-on** | Flags deepfake audio inside the [IPED](https://github.com/sepinf-inc/IPED) evidence pipeline (a capability IPED lacks) | [`integrations/iped/`](integrations/iped/) |
| 🍓 **Raspberry Pi / edge** | Separate 0.62 MB INT8 DSFNetTiny model, CPU-only, `onnxruntime`+`numpy`+`soundfile` (no torch) | [`edge/`](edge/) |

## ⚙️ Configuration

Everything is configured via environment variables; sensible defaults make local
development zero-config.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECRET_KEY` | dev placeholder | JWT signing key — **set in production** (`openssl rand -hex 32`) |
| `VG_ENV` | `development` | `production` enforces strict auth & Twilio signature checks |
| `XLS_R_AASIST_PATH` | — | Path to the production detector checkpoint |
| `VG_ADMIN_PASSWORD` / `VG_ANALYST_PASSWORD` | demo creds | Override the built-in demo users |
| `FRONTEND_ORIGIN` / `FRONTEND_ORIGINS` | — | CORS allowlist for the web app |
| `PDPL_MAX_AGE_SECONDS` | `60` | Auto-delete window for uploaded audio (PDPL compliance) |
| `VG_MAX_AUDIO_SECONDS` | `600` | Maximum accepted upload length |
| `VG_MEDIA_TTL_S` | `900` | TTL for generated/watermarked media |
| `VG_CLONE_QUOTA_PER_HOUR` | `10` | Per-admin voice-cloning quota |
| `VG_WS_MAX_CONNECTIONS` | `4` | Concurrent live-mic streaming slots |
| `TWILIO_AUTH_TOKEN` | — | Enables `X-Twilio-Signature` validation on the VoIP bridge |

## 🔌 API

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|:----:|
| `POST` | `/token` | OAuth2 password → JWT (carries a `role` claim) | — |
| `POST` | `/detect` | Audio → verdict (`?model=`, `?explain=true`) | 🔑 |
| `POST` | `/explain` | Per-moment occlusion attribution (first 3 s for XLS-R + AASIST and the other SSL / DSFNet models; first 10 s for `classical` and `wav2vec2_spoof`) | 🔑 |
| `POST` | `/synthesize` | Text → watermarked speech (voice *cloning* is admin-only + quota'd) | 🔑 |
| `POST` | `/watermark/verify` | Provenance check: spectral watermark + C2PA manifest | 🔑 |
| `POST` | `/forensic/report` | NIST SP 800-86 PDF report (audio metadata, model + checkpoint hash) | 🔑 |
| `WS` | `/ws/stream` | Live-mic streaming (JWT as first WS message; capped slots) | 🔑 |
| `WS` | `/twilio/stream` | Twilio call screening (`X-Twilio-Signature` when `TWILIO_AUTH_TOKEN` set) | ✍️ |
| `GET` | `/models` · `/health` · `/docs` | Ops & Swagger | — |

🔑 JWT bearer · ✍️ Twilio request signature (open in development; refused in
production unless `TWILIO_AUTH_TOKEN` is configured)

## 🧪 Testing & quality

**147 tests** across 17 modules — API auth & security hardening, watermark round-trip,
C2PA signing, forensics chain-of-custody, XAI, adversarial robustness, RawBoost
augmentation, and a simulated Twilio call stream. CI runs the suite plus `ruff`
(lint + format) and `bandit` (security static analysis) on every push.

```bash
pip install -e ".[dev]"
pytest --cov=src/voiceguard tests/
ruff check src/ tests/
```

## 📖 Documentation

| Document | What's inside |
|----------|---------------|
| [RESULTS_canonical.md](docs/RESULTS_canonical.md) | **Single source of truth** for every EER — auto-generated, 95% bootstrap CIs, minDCF, provenance per checkpoint |
| [EVAL_PROTOCOLS.md](docs/EVAL_PROTOCOLS.md) | Evaluation protocols & how to reproduce every number |
| [ADVERSARIAL_ROBUSTNESS.md](docs/ADVERSARIAL_ROBUSTNESS.md) | PGD attack curve — measured adversarial limits |
| [HIDDEN_TRACK_ANALYSIS.md](docs/HIDDEN_TRACK_ANALYSIS.md) | Where residual error concentrates (hard OOD track) |
| [CLONE_DETECTION_LIMITS.md](docs/CLONE_DETECTION_LIMITS.md) | Measured boundaries of clone detection |
| [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | Honest platform limitations (streaming, latency, scope) |
| [SYNTHESIS_ENGINES.md](docs/SYNTHESIS_ENGINES.md) | Kokoro / XTTS v2 / IndexTTS-2 engine guide |
| [REPRODUCIBILITY_MANIFEST.md](docs/REPRODUCIBILITY_MANIFEST.md) | Fixed environment manifest for all reported results |
| [EVALUATION_METADATA.md](docs/EVALUATION_METADATA.md) | Dataset identity, artifact hashes, commands, result fields, and licensing boundaries |

## 🛠️ Tech stack

**ML** PyTorch · transformers (XLS-R, Wav2Vec2, WavLM) · AASIST · XGBoost · captum · ONNX Runtime
· **Backend** FastAPI · python-jose (JWT) · slowapi · **Frontend** React 18 · Vite · Tailwind · Recharts
· **Audio** librosa · torchaudio · Kokoro-82M · **Infra** Nginx + systemd · Docker · GitHub Actions · ruff · bandit

## 🗺️ Roadmap

- [x] Train `DSFNetTiny` so the ONNX edge export carries accuracy
- [x] Reproduce the official ASVspoof 2021 LA 2.61% EER
- [x] True signed C2PA provenance on synthesized audio
- [ ] Permanent hosted demo (live via Cloudflare Tunnel from June 2026; currently offline)
- [x] Premium-TTS (ElevenLabs) hardening with a real-pass safety gate (v9c deployed: 95.8% held-out ElevenLabs-v3, 96% real-pass; see [Results](#-results))
- [ ] Backbone adversarial fine-tuning for true PGD robustness
- [ ] GADC (Gulf-Arabic Deepfake Corpus) + human perception study

## 🤝 Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). For vulnerabilities, see [SECURITY.md](SECURITY.md).

## 👥 Team

| Name | Role |
|------|------|
| **Mohammad Thabet Hassan** | Detection architecture, FastAPI backend, CI/CD, deployment |
| **Fahad Sadek Al-Jazzeri** | Feature extraction, classical ML, SSL models, evaluation |
| **Ahmed Sami Alameri** | React frontend, synthesis, watermarking, VoIP, forensics, XAI |

**Supervisor:** [Dr. Arash Kermani Kolankeh](https://github.com/arashkermaniprojects) · **Institution:** Canadian University Dubai · **2025–2026**

## 🙏 Acknowledgements

A heartfelt **thank you to our supervisor, [Dr. Arash Kermani Kolankeh](https://github.com/arashkermaniprojects)**,
whose guidance, insight, and encouragement shaped VoiceGuard at every stage. This project
would not have been possible without his mentorship — thank you, Dr. Arash.

## 📚 Citation

The **IEEE SM2026 acceptance covers the GP1 classical-baseline paper** (feature-based
detection, F1 = 0.95) — not the full platform or the XLS-R+AASIST results in this
repository, which post-date the submission. If you cite the accepted work:

```bibtex
@inproceedings{hassan2026lightweight,
  title     = {Lightweight Voice Deepfake Detection for Smart Mobility Using
               Multi-Feature Ensemble Learning},
  author    = {Hassan, Mohammad Thabet and Sadek, Fahad and Sami, Ahmed and
               Kolankeh, Arash Kermani},
  booktitle = {2026 International Conference on Smart Mobility (SM)},
  pages     = {1--2},
  year      = {2026},
  month     = may,
  publisher = {IEEE},
  doi       = {10.1109/SM69703.2026.11614145}
}
```

Paper: [IEEE Xplore 11614145](https://ieeexplore.ieee.org/document/11614145) ·
DOI [10.1109/SM69703.2026.11614145](https://doi.org/10.1109/SM69703.2026.11614145).
Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## 📄 License

The original VoiceGuard source code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). Third-party models, datasets, pretrained weights, synthesis engines, fonts, images, and other external assets remain subject to their own licenses and attribution requirements; see the linked provenance and documentation records before redistribution.
