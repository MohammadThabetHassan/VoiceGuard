# Changelog

All notable changes to VoiceGuard are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **OGG upload support.** `/detect`, `/explain`, `/watermark/verify` and the
  forensic path now accept OGG (Vorbis) uploads — magic-byte sniffed via the
  `OggS` capture pattern, decoded by libsndfile like the other formats.
- **`seconds_analyzed`** on `DetectionResult` and in the forensic PDF
  ("Audio Scored (s)") — discloses exactly how much of a long clip the verdict
  covers (`min(duration, VG_SCORE_SECONDS)`); the PDF's misleading
  "Windows Analyzed (3s)" row is gone.
- **`final` flag on stream verdicts.** `/ws/stream` and the Twilio bridge mark
  the verdict that covers the full scoring cap with `final: true`, stop scoring,
  and stop re-emitting; the Live tab shows a "verdict final" notice instead of
  painting frozen verdicts as fresh per-second analysis.

### Fixed
- **Review follow-ups on the single-pass detection change:** the Twilio bridge
  now runs inference off the event loop (`asyncio.to_thread`, same as
  `/ws/stream`) and honors `VG_WS_SCORE_SECONDS` (was a hardcoded 15s); the
  silence guard checks the region actually scored, not the whole clip; scoring
  caps are defensively parsed and clamped (≥3s, even-aligned) so a bad env value
  can't crash a stream or score zero-padding; overdue stream milestones are
  coalesced into one pass over the freshest prefix (a burst no longer triggers
  a backlog of stale inferences); `ModelRegistry.load()` is now thread-safe
  (two cold-start streams could double-load the 1.2GB checkpoint); the Twilio
  buffer is preallocated (was O(n²) np.concatenate per 20ms frame); and the
  `/detect` OpenAPI docs / schema descriptions no longer describe the removed
  sliding-window behavior.
- **Real voices no longer flagged fake on long clips / Live streaming.**
  Sliding-window scoring (max, then mean aggregation) misclassified genuine
  recordings: mid-utterance 3 s windows are out-of-distribution for the SSL
  detector and read as synthetic regardless of content (held-out reals scored
  0.9+ fake from window 2 onward). `/detect` now scores **one full-clip pass
  from the recording's natural start** (capped at `VG_SCORE_SECONDS`, default
  60 s) — 16/16 held-out clips correct — and `/ws/stream` + the Twilio bridge
  re-score the **growing prefix from session start** (first verdict at 3 s,
  then every 2 s, capped at `VG_WS_SCORE_SECONDS`, default 15 s). Stream
  inference moved off the event loop (`asyncio.to_thread`). Documented, with
  the measured silence-prepend evasion this analysis surfaced, in
  `docs/KNOWN_LIMITATIONS.md`.

### Added
- **Live-mic streaming UI.** New **Live** tab streams the microphone over
  `/ws/stream` (AudioWorklet → 16 kHz int16 PCM) and renders a rolling
  real/fake verdict + window timeline — the "real-time" in the project name is
  now demonstrable in the browser.
- **`POST /watermark/verify` + Verify tab.** The read side of synthesis:
  checks the keyed spectral watermark (given a `watermark_id`) and the embedded
  C2PA manifest, closing the Generate → Verify provenance loop.
- **Roles.** JWTs carry a `role` claim; voice *cloning* is admin-only with a
  per-user hourly quota (`VG_CLONE_QUOTA_PER_HOUR`). Preset TTS stays open to
  analysts.
- **Forensic PDF upgrades:** evidence characteristics (duration, sample rate,
  channels, codec, windows analyzed) and analysis-tool identity (model key,
  app version, checkpoint SHA-256) — captured at detect time, before PDPL
  erasure.
- **Frontend test suite** (vitest + Testing Library, 16 tests) and a CI
  **Docker smoke test** that builds the backend image and probes `/health`.

### Security
- `/twilio/stream` validates `X-Twilio-Signature` when `TWILIO_AUTH_TOKEN` is
  set and refuses unauthenticated production use.
- `/ws/stream` takes the JWT as the first WS message (query-string tokens
  deprecated — they leak into proxy logs); WebSockets gained a global
  concurrent-connection budget, session-duration cap, and ingest-rate cap.
- Uploads are magic-byte sniffed before libsndfile parses them, and `/detect`
  enforces a duration cap (`VG_MAX_AUDIO_SECONDS`) instead of timing out.
- Generated media/report filenames switched from timestamps to uuid4
  (the unauthenticated `/media` mount was enumerable).
- nginx configs: per-location header inheritance fixed, CSP added, deprecated
  `X-XSS-Protection` dropped.

### Fixed
- Docker stack actually boots: healthcheck no longer needs curl (absent from
  the slim image), CPU torch resolves in one pass (the old `torch==2.1.2` pin
  crashed against numpy≥2.4), the frontend image builds from the lockfile
  (`npm ci`), compose mounts `./checkpoints`, and production nginx routes
  `/twilio/`.

### Removed
- Unused runtime dependencies `webrtcvad`, `datasets`, `matplotlib` (install
  time + image size).
- **Official ASVspoof 2021 LA EER reproduced (2.61%).** Re-downloaded the official
  eval (Zenodo + asvspoof.org keys) to durable disk and reproduced the headline
  **2.61%** eval-phase EER exactly from raw FLAC via `run_official_eval.py`. Scored
  the whole lineage on the official 181,566 trials — parent 2.61% / **v7 (deployed)
  3.38%** / v8 2.49% eval EER. EER is verifiable again.
- **True C2PA provenance signing** (`voiceguard.watermark.c2pa_sign`). Synthesized
  audio now embeds a real *signed* C2PA manifest tagging it `trainedAlgorithmicMedia`
  (validation `Valid`), with an auto-generated ES256 credential; `/synthesize`
  returns `c2pa_signed`. The spectral watermark remains as the re-encode-robust layer.
- **ONNX edge model with real weights.** Trained `DSFNetTiny` (8.47% balanced EER)
  so the INT8 export (`0.62 MB`, ~30 ms CPU) carries accuracy instead of random weights.
- **VoIP `/twilio/stream` on the SSL model** + an end-to-end simulated-call test
  (`test_twilio_stream_sim`) proving the Media-Stream bridge without a phone number.
- **Permanent custom-domain hosting via Cloudflare Tunnel** (set up through the CF
  API) + a `deploy/serve_demo.sh` watchdog supervising server + Cloudflare tunnel +
  pinned ngrok static domain across crashes/restarts.
- **Larger held-out clone eval** (100/family): v7 holds — real 97% / XTTS 100% /
  IndexTTS-2 96%.
- **Premium-TTS hardening (in progress).** v7 catches 85% of held-out ElevenLabs-v3;
  a balanced premium-hardened checkpoint (MLAAD: ElevenLabs/Cartesia/DeepGram/Gemini/…)
  is trained under a real-pass ≥90% safety gate.

### Changed
- `/ws/stream` (mic) now scores with the SSL production model (was classical),
  with graceful fallback.

### Fixed
- Corrected the spectral watermark's misleading "C2PA" docstring (it is the
  in-signal layer; real C2PA manifests live in `c2pa_sign`).

- **🏆 v7 — unified detector that catches all clone families (deployed).** Trained
  from the 2.61% base on the full ASVspoof-balanced anchor + a large diverse clone
  corpus (520 IndexTTS-2 + 280 XTTS + Kokoro, cloned from ASVspoof-real speakers to
  force vocoder-artifact learning), top-12 XLS-R layers unfrozen, 12 epochs. On the
  frozen speaker/text-disjoint held-out set: real-pass **96%**, Kokoro **100%**,
  XTTS **100%**, **IndexTTS-2 96.7%** (was 60%), ASVspoof-balanced EER **9.25%**
  (was ~31%). Live: held-out IndexTTS-2/XTTS → fake 1.00, real → real 0.999.
  Supersedes v3/v6 and disproves the earlier "IndexTTS-2 ceiling".
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
- **Detector hardened against voice cloning (v3, deployed).** A frozen-backbone
  head fine-tune from the v2 model, adding XTTS + IndexTTS-2 + Kokoro clone fakes,
  flips XTTS clones from `real ~0.99` to **`fake ~0.97`** while keeping real speech
  real. Real-world gate: fake-detect 90→94%, real-pass 87.5→~85%.
- **Cloning engines moved to GPU** (`torch cu128`, RTX 5090): IndexTTS-2 ~25s→~4s.
- **Unified anchored model (v6, deployed).** Built a trustworthy speaker/text-disjoint
  held-out eval and trained an anchored fine-tune (ASVspoof-balanced + diverse
  Kokoro/XTTS/IndexTTS-2 clones + LibriSpeech reals, top-6 XLS-R layers unfrozen).
  Held-out: real-pass 100%, Kokoro 100%, XTTS 90%, **IndexTTS-2 60→63%**, ASVspoof-
  balanced EER ~29%. Marginal gain over v3, no regression → deployed.
- **Honest limits:** IndexTTS-2 (BigVGAN, near-real) sits near the detectability
  ceiling (~60%) for this front-end — more clones/training barely move it.
  **EER < 2% is not currently achievable**: the official ASVspoof 2021 LA eval
  (where 2.61% was measured) is off-disk; the obtainable *balanced* mirror is a
  different, harder ruler (~29% for this lineage). See
  [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

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
