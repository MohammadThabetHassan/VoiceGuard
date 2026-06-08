# Security Policy

VoiceGuard is a graduation research project (GP2) that detects and watermarks
AI-generated speech. We take the security of the platform and of the people who
interact with it seriously.

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ active |
| < 1.0   | ❌ pre-release, unsupported |

Only the latest tagged release on the `main` branch receives security fixes.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through GitHub's coordinated disclosure:

1. Go to <https://github.com/MohammadThabetHassan/VoiceGuard/security/advisories>
2. Click **"Report a vulnerability"** to open a private advisory.

Please include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal proof-of-concept if possible).
- Affected component (API, model serving, synthesis, frontend, deploy scripts).
- Any suggested remediation.

### What to expect

- **Acknowledgement** within 5 business days.
- **Triage and severity assessment** within 10 business days.
- We will keep you updated on remediation progress and coordinate a disclosure
  timeline with you. With your consent we will credit you in the advisory.

## Scope

In scope:

- The FastAPI backend (`src/voiceguard/api/`) — auth, detection, synthesis,
  forensics, and streaming endpoints.
- Model-serving and input handling (e.g. malicious uploads, resource
  exhaustion, deserialization of checkpoints).
- The watermarking and forensic chain-of-custody integrity guarantees.
- The deployment tooling (`deploy/`) and CI workflows.

Out of scope:

- Third-party model weights and datasets (ASVspoof, Kokoro-82M, etc.).
- Vulnerabilities requiring a compromised host or physical access.
- The intrinsic statistical error of the detector (false accept / false reject
  rates) — these are tracked as model-quality metrics, not security issues.
- Findings that only affect the demo credentials (see below).

## Security posture (current)

- **Authentication:** JWT (HS256, python-jose); `SECRET_KEY` must be set to a
  strong random value in production (`openssl rand -hex 32`). The default
  `admin` / `voiceguard2026` credentials are **for local demos only** and must
  be changed before any real deployment (`src/voiceguard/api/auth.py`).
- **Rate limiting:** per-route limits via slowapi.
- **Data minimisation (UAE PDPL):** uploaded audio is auto-deleted via
  background tasks (≤60 s); generated media is removed on a TTL.
- **Input validation:** detection rejects silent / too-short audio (HTTP 422)
  and enforces an upload size limit.
- **Checkpoint loading:** model checkpoints are loaded with `weights_only`
  where supported; only load checkpoints from trusted sources.
- **Transport:** intended to run behind TLS-terminating Nginx (see `deploy/`).
- **Static analysis:** `bandit` and `semgrep` run in CI; `ruff` for linting.

## Hardening notes for operators

- Always set a unique `SECRET_KEY` and change the demo credentials.
- Restrict CORS via `VOICEGUARD_DOMAIN` to your own origin(s).
- Serve only over HTTPS; do not expose the uvicorn port (8000) directly.
- Treat model checkpoint files as trusted binaries — never load an untrusted
  `.pt` file (PyTorch deserialization can execute code on older formats).
