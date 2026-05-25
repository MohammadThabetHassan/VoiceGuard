# VoiceGuard — Claude Code Context

Run `/plan` at session start. See `.claude/skills/` for all procedures.

## Project

VoiceGuard: AI-powered voice deepfake detection, adversarial speech synthesis,
and real-time vishing defence platform. University graduation project (GP2).

- **New repo (build target):** https://github.com/MohammadThabetHassan/VoiceGuard
- **Old repo (READ-ONLY reference):** /tmp/voiceguard-old
  (clone: `git clone https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator.git /tmp/voiceguard-old`)

## Acceptance Criteria (hard targets)

- **DSFNet** — dual-stream raw waveform + Mel-spectrogram, bidirectional cross-attention
  (8 heads, 512-dim). Target EER <0.5% on ASVspoof 2021 LA; fallback ≤1.0%.
- **SM2026 baseline** — Enhanced+XGBoost, 30-D feature vector, **F1=0.9500 on 475 samples**.
  Published result — must reproduce exactly. Never guess on anything affecting this.
- Real-time inference ≤200 ms per 3-second window on GPU.
- Grad-CAM (captum) + SHAP per prediction.
- SHA-256 chain-of-custody + automated PDF forensic report (NIST SP 800-86).
- C2PA v1.4 spectral watermarking on all XTTS v2 synthesis output.
- Twilio WebSocket VoIP interception with rolling deepfake scoring.
- FastAPI 0.104+ backend, React 18 + Vite + Tailwind frontend.
- Docker compose single-command deployment.
- Security: TLS 1.3, JWT auth, slowapi 60 req/min, OWASP ASVS L2, bandit+semgrep, SBOM.
- UAE PDPL: no persistent raw audio, auto-delete ≤60s.
- Backend test coverage ≥80%.

## Commit Identities

| Owner | Email | Scope |
|---|---|---|
| Mohammad Thabet | 20220002188@students.cud.ac.ae | DSFNet, architecture, FastAPI, CI/CD, Docker, tests |
| Fahad Sadek | 20220001790@students.cud.ac.ae | Feature extraction, classical ML, Wav2Vec2, evaluation |
| Ahmed Alameri | 20220001166@students.cud.ac.ae | React, XTTS, watermarking, Twilio VoIP, forensics, Grad-CAM/SHAP |

Push: `git push https://${GITHUB_TOKEN}@github.com/MohammadThabetHassan/VoiceGuard.git main`
Token: `source ~/.voiceguard-env` — never echo, never commit.
Conventional Commits: `feat|fix|docs|chore|test|refactor|ci(scope): message`

## Compute

- **This machine (t3.medium):** code editing, tests, Docker, frontend dev. No DSFNet training.
- **GPU (g5.xlarge Spot):** training only. Never launch without user confirmation.
  Write `scripts/aws/launch_<task>.sh`, show cost estimate, STOP.
- **S3:** `voiceguard-mt-2026` — all datasets, checkpoints, eval outputs.
- **Budget:** $200 total.
- **Region:** ap-southeast-1 (Singapore).

## Hard Rules

**NEVER:**
- Commit `.env`, `~/.voiceguard-env`, credentials, model weights, dataset files, audio files
- Push without lint + test + bandit passing
- Launch any AWS resource without user confirmation
- Modify `docs/midterm/` (published, frozen)
- Touch `/tmp/voiceguard-old` (read-only reference)
- Force-push or rewrite history
- Use FIXME/TODO without flagging
- Guess on anything affecting F1=0.9500

**ALWAYS ASK before:**
`aws ec2 terminate-instances` · `aws s3 rm --recursive` · `aws iam *` · `--force` flags ·
`rm -rf` outside /tmp · pip install non-standard packages · anything costing money ·
anything modifying `docs/midterm/`

## Build Phases

| Phase | Tag | Scope |
|---|---|---|
| 0 | v0.0-setup | Scaffold, skills, guardrails (current) |
| 1 | v0.1 | Classical baseline reproduction (SM2026 F1=0.9500) |
| 2 | v0.2 | DSFNet architecture + GPU training scripts |
| 3 | v0.3 | Evaluation harness + Wav2Vec2 baseline |
| 4 | v0.4 | FastAPI backend + React frontend |
| 5 | v0.5 | Forensics + XAI + watermarking |
| 6 | v0.6 | VoIP + adversarial hardening |
| 7 | v1.0 | ONNX INT8, security audit, release |

Current phase: see `PHASE` file at repo root.
