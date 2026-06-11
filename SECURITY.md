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
- **Roles:** tokens carry a `role` claim (`admin` / `analyst`). Voice *cloning*
  (synthesis from a reference clip) is admin-only and capped per user per hour
  (`VG_CLONE_QUOTA_PER_HOUR`); preset TTS, detection, and reports are open to
  both roles.
- **Rate limiting:** per-route limits via slowapi; WebSocket streams have a
  global concurrent-connection budget (`VG_WS_MAX_CONNECTIONS`), a per-session
  duration cap (`VG_WS_MAX_SECONDS`), and an ingest-rate cap (~4× realtime).
- **WebSocket auth:** `/ws/stream` takes the JWT as the *first WS message*
  (`?token=` is accepted for backward compatibility but deprecated — query
  strings end up in proxy access logs). `/twilio/stream` validates
  `X-Twilio-Signature` when `TWILIO_AUTH_TOKEN` is set and refuses to run
  unauthenticated in production.
- **Data minimisation (UAE PDPL):** uploaded audio is auto-deleted via
  background tasks (≤60 s); generated media is removed on a TTL and served
  under unguessable uuid4 filenames.
- **Input validation:** uploads are magic-byte sniffed (RIFF/WAVE, fLaC, MP3)
  before libsndfile parses them, size-capped (100 MB) and duration-capped
  (`VG_MAX_AUDIO_SECONDS`, default 600 s); detection rejects silent /
  too-short audio (HTTP 422).
- **Checkpoint loading:** model checkpoints are loaded with `weights_only`
  where supported; only load checkpoints from trusted sources.
- **Transport:** intended to run behind TLS-terminating Nginx (see `deploy/`);
  both nginx configs ship X-Frame-Options, nosniff, Referrer-Policy, and a CSP.
- **Static analysis:** `bandit` and `semgrep` run in CI; `ruff` for linting.

## Known limitations (accepted for the demo, documented honestly)

These are deliberate scope decisions for a graduation-project demo, not
oversights. Operators deploying VoiceGuard for real use should address them:

- **Token storage & revocation:** JWTs live 60 min in browser `localStorage`
  (XSS-exfiltratable) and there is no server-side revocation list — a stolen
  token is valid until expiry.
- **Single-process state:** the rate limiter, detection-result store, and clone
  quota are in-memory. Running uvicorn with multiple workers splits limits
  per-worker and makes `/forensic/report` lookups worker-dependent — deploy
  with one worker, or put these stores in Redis/SQLite first.
- **Crash-window cleanup:** PDPL deletion timers don't survive a crash;
  generated media is swept at startup, but an uploaded temp file could outlive
  a crash until the OS clears `/tmp`.
- **Public demo credentials:** the published demo runs with known credentials
  by design. Cloning on the demo is mitigated by the admin-only gate, the
  hourly quota, consent attestation, the spectral watermark, and C2PA signing
  of all output — not by secrecy of the account.

## Hardening notes for operators

- Always set a unique `SECRET_KEY` and change the demo credentials
  (`VG_ADMIN_PASSWORD` / `VG_ANALYST_PASSWORD`).
- Set `TWILIO_AUTH_TOKEN` if you expose the Twilio bridge.
- Restrict CORS via `VOICEGUARD_DOMAIN` to your own origin(s).
- Serve only over HTTPS; do not expose the uvicorn port (8000) directly.
- Run a single uvicorn worker (see "Single-process state" above).
- Treat model checkpoint files as trusted binaries — never load an untrusted
  `.pt` file (PyTorch deserialization can execute code on older formats).
