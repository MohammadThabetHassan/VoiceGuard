# Changelog

All notable changes to VoiceGuard are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multi-engine Generate with zero-shot voice cloning.** A `SynthEngine`
  registry powers `GET /synthesis/engines`; `POST /synthesize` is now multipart
  with engine selection and an optional reference clip. Kokoro runs in-process;
  **XTTS v2** and **IndexTTS-2** cloning engines run in isolated venvs
  (subprocess) and degrade gracefully when not installed — see
  [docs/SYNTHESIS_ENGINES.md](docs/SYNTHESIS_ENGINES.md).
- Generate UI: engine picker, preset-voice selector, reference upload +
  authorization gate, and a **"Test against detector"** button closing the
  generate → watermark → detect loop.

### Notes
- Cloning engines are opt-in (install the engine venv to enable). XTTS weights
  are CPML (non-commercial).
- **Verified end-to-end**: an XTTS clone runs and the generate→detect loop works.
  Finding: the current production detector classifies XTTS clones as **real**
  (~0.99) — a new out-of-distribution gap (like pre-hardening Kokoro) that
  motivates XTTS-hardening. IndexTTS-2 clones are caught (~100%), so that engine
  is preferred for the "Test against detector" demo.

## [1.0.0] - 2026-06-08

First public release: a full voice-deepfake detection, synthesis-watermarking,
and vishing-defence platform.

### Added
- **Detection** — production **XLS-R-300M + AASIST** detector (headline eval EER
  **2.61%** on ASVspoof 2021 LA), selectable alongside DSFNet, Wav2Vec2/WavLM,
  and a classical XGBoost baseline (F1 = 0.9500). Input-quality guard rejects
  silent / too-short audio (422).
- **Out-of-distribution robustness** — Kokoro-hardened + real-world-tuned
  deployment model (Kokoro 93.3%, IndexTTS2 100%; real-world real-pass 87.5% /
  fake-detect 90.3%).
- **Explainability** — Integrated Gradients attribution via `/explain` and
  `/detect?explain=true`.
- **Synthesis** — local **Kokoro-82M** TTS with spectral (C2PA-style)
  watermarking (`/synthesize`).
- **Forensics** — SHA-256 chain-of-custody and NIST SP 800-86 PDF reports
  (`/forensic/report`).
- **VoIP** — Twilio Media Streams WebSocket bridge for live call screening.
- **Edge** — ONNX INT8 export of DSFNetTiny: **0.62 MB**, CPU p50 **30 ms**
  (`scripts/export_onnx.py`).
- **Adversarial robustness** — PGD/FGSM attack + a model-agnostic
  `measure_robustness()` evaluation (`src/voiceguard/evaluation/adversarial_eval.py`).
- **API & UI** — FastAPI backend (JWT, rate limiting, PDPL auto-deletion) and a
  React 18 + Vite + Tailwind frontend (Detect / Generate / Results).
- **Deployment** — self-hosted Nginx + systemd (`deploy/`) and Docker Compose.
- **Project hygiene** — `SECURITY.md`, CI (ruff, pytest, bandit, frontend lint).

### Notes
- The deployed checkpoint (`xlsr_aasist_realworld_v2`) is a real-world-robustness
  fine-tune of the 2.61% Kokoro-hardened model; its ASVspoof EER was not
  separately benchmarked.
- PGD adversarial hardening is documented as a **negative finding**: a
  frozen-backbone head-only fine-tune did not confer PGD robustness and was not
  promoted (the deployed model is unchanged).
- The ONNX export currently validates pipeline/size/latency with random weights
  (no trained DSFNetTiny checkpoint yet).

[1.0.0]: https://github.com/MohammadThabetHassan/VoiceGuard/releases/tag/v1.0.0
