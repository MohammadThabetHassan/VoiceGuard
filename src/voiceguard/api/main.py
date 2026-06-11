"""
VoiceGuard FastAPI application.

Endpoints:
    POST /token              — JWT token issuance
    POST /detect             — Upload audio, get deepfake detection result
    POST /synthesize         — Text-to-speech with C2PA watermarking
    POST /forensic/report    — Generate PDF forensic report
    WS   /ws/stream          — Real-time microphone streaming detection
    WS   /twilio/stream      — Twilio Media Stream bridge
    GET  /health             — Healthcheck

Security: JWT auth, slowapi 60 req/min, PDPL auto-delete ≤60s.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from voiceguard.api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    check_production_security,
    create_access_token,
    get_current_claims,
    get_current_user,
    role_for,
)
from voiceguard.api.middleware import (
    PDPLTimingMiddleware,
    limiter,
    make_temp_audio_file,
    pdpl_auto_delete,
)
from voiceguard.api.schemas import (
    DetectionResult,
    ExplanationResult,
    ForensicReportRequest,
    ForensicReportResult,
    HealthResponse,
    ModelType,
    StreamDetectionEvent,
    SynthesisEngineInfo,
    SynthesisResult,
    TokenResponse,
    WatermarkVerifyResult,
)
from voiceguard.forensics import result_store

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────

from voiceguard.models.registry import registry  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_production_security()  # refuse to start with a default SECRET_KEY in production
    registry.preload()  # loads any model whose env-var is set at startup
    _sweep_media_dir()  # remove stale media whose TTL timers were lost on restart
    yield


def _sweep_media_dir() -> None:
    """Delete MEDIA_DIR files older than MEDIA_TTL_S (TTL timers don't survive restart)."""
    now = time.time()
    for f in MEDIA_DIR.glob("*"):
        try:
            if f.is_file() and now - f.stat().st_mtime > MEDIA_TTL_S:
                f.unlink(missing_ok=True)
        except OSError:
            logger.warning("media sweep: could not remove %s", f, exc_info=True)


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VoiceGuard API",
    version=__version__,
    description="Real-time voice deepfake detection and vishing defence API",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Allowed CORS origins: local dev + the self-hosted production domain. In the
# self-hosted deployment, Nginx serves the frontend and proxies /api on the SAME
# origin, so CORS is moot for the browser app; the domain is still allow-listed
# so direct-to-API requests work. Set VOICEGUARD_DOMAIN (e.g. "voiceguard.tech")
# to allow https://<domain> and https://www.<domain>. Extra origins (e.g. an
# ngrok tunnel) can be added via FRONTEND_ORIGINS (comma-separated); a single one
# via FRONTEND_ORIGIN.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
_domain = os.environ.get("VOICEGUARD_DOMAIN", "").strip()
_domain_origins = [f"https://{_domain}", f"https://www.{_domain}"] if _domain else []
_single_origin = os.environ.get("FRONTEND_ORIGIN", "")
_extra_origins = os.environ.get("FRONTEND_ORIGINS", "")
ALLOWED_ORIGINS = list(
    dict.fromkeys(
        _DEFAULT_ORIGINS
        + _domain_origins
        + ([_single_origin] if _single_origin else [])
        + [o.strip() for o in _extra_origins.split(",") if o.strip()]
    )
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Auth is via Bearer token (Authorization header), not cookies, so credentialed
    # CORS is unnecessary — keeping it False avoids relaxing the same-origin policy.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Synthesised audio is written here and served at /media (i.e. /api/media/<f>
# once behind Nginx or the demo's /api mount). Auto-deleted after MEDIA_TTL_S.
MEDIA_DIR = Path(os.environ.get("VG_MEDIA_DIR", "/tmp/voiceguard_media"))  # noqa: S108  # nosec B108
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_TTL_S = int(os.environ.get("VG_MEDIA_TTL_S", "900"))
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# ── Helpers ────────────────────────────────────────────────────────────────────

ACCEPTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/flac",
    "audio/ogg",
    "application/ogg",
}
ACCEPTED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".oga"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
_UPLOAD_CHUNK = 1024 * 1024  # 1MB
# Longest clip /detect will accept. libsndfile decode + one forward pass over up
# to VG_SCORE_SECONDS of audio run synchronously in the request, so an unbounded
# clip on CPU times out the 120s nginx proxy window — reject early with a clear
# message instead of a 504.
MAX_AUDIO_SECONDS = float(os.environ.get("VG_MAX_AUDIO_SECONDS", "600"))


def _sniff_is_audio(head: bytes) -> bool:
    """True when *head* starts like one of the advertised formats (WAV/FLAC/MP3/OGG).

    Extension and Content-Type are attacker-controlled; the bytes are what
    libsndfile will actually parse, so gate on them. MP3 is ID3-tagged or starts
    straight at an MPEG frame sync (0xFFEx); OGG containers always open with the
    "OggS" capture pattern.
    """
    if len(head) < 12:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    if head[:4] == b"fLaC":
        return True
    if head[:4] == b"OggS":
        return True
    if head[:3] == b"ID3":
        return True
    return head[0] == 0xFF and (head[1] & 0xE0) == 0xE0  # MPEG frame sync


async def save_upload(upload: UploadFile) -> tuple[str, str]:
    """Stream an upload to a temp file; return (path, sha256_hex).

    Rejects non-audio uploads (415) and streams in 1MB chunks — hashing
    incrementally and aborting with 413 as soon as the running size exceeds the
    limit — so a large body never lands fully in RAM.
    """
    import os as _os

    suffix = (Path(upload.filename or "audio.wav").suffix or ".wav").lower()
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    if ctype not in ACCEPTED_CONTENT_TYPES and suffix not in ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media type — upload WAV, MP3, FLAC, or OGG audio.",
        )

    fd, path = make_temp_audio_file(suffix=suffix)
    sha256 = hashlib.sha256()
    total = 0
    sniffed = False
    try:
        with _os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                if not sniffed:
                    # Magic bytes, not just extension/Content-Type: the bytes are
                    # what libsndfile (CVE history) will actually parse.
                    if not _sniff_is_audio(chunk[:12]):
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="File content is not WAV, MP3, FLAC, or OGG audio.",
                        )
                    sniffed = True
                total += len(chunk)
                if total > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"File exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit",
                    )
                sha256.update(chunk)
                out.write(chunk)
        if not sniffed:  # empty body never entered the loop
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Empty upload — send WAV, MP3, FLAC, or OGG audio.",
            )
    except HTTPException:
        Path(path).unlink(missing_ok=True)  # drop the partial file
        raise
    except OSError as exc:
        Path(path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save upload") from exc

    PDPLTimingMiddleware.register(path)
    return path, sha256.hexdigest()


def _probe_audio(path: str) -> dict:
    """Header-only probe via soundfile: duration/sample-rate/format without a full
    decode. Rejects clips over MAX_AUDIO_SECONDS (413) before any model work —
    synchronous CPU inference on an unbounded clip would 504 behind nginx."""
    import soundfile as sf

    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Could not decode audio — upload valid WAV, MP3, FLAC, or OGG.",
        ) from exc
    if info.duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Audio is {info.duration:.0f}s long; the analysis limit is "
                f"{MAX_AUDIO_SECONDS:.0f}s. Trim the clip or raise VG_MAX_AUDIO_SECONDS."
            ),
        )
    return {
        "duration_s": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": f"{info.format}/{info.subtype}",
    }


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file to (mono float32 array, sample_rate) via soundfile.

    soundfile (libsndfile) handles WAV/FLAC/OGG and MP3 (libsndfile ≥ 1.1) without
    the optional TorchCodec backend that newer ``torchaudio.load`` requires — so
    this works identically in CI and on the deploy host.
    """
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    return np.ascontiguousarray(data, dtype=np.float32), sr


def _detect_classical(path: str) -> tuple[str, float]:
    """Run classical detection. Returns (label, confidence).

    Loads via soundfile so MP3/FLAC (not just WAV) are supported, matching the
    formats the API advertises.
    """
    from voiceguard.features.extractor import extract_features

    data, sr = _read_audio(path)
    mx = np.max(np.abs(data)) + 1e-8
    data = (data / mx).astype(np.float32)
    features = extract_features(data, sr)

    detector = registry.load("classical")
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)


# SSL models take (B, T); CNN models (DSFNet/AASIST) take (B, 1, T).
_SSL_KEYS = {
    "wav2vec2",
    "wavlm_base_plus",
    "wavlm_large",
    "wav2vec2_large",
    "xls_r",
    "xls_r_aasist",
}
_WIN = 48000  # 3s @ 16kHz


def _score_seconds_env(name: str, default: float) -> float:
    """Parse a scoring-cap env var defensively: fall back on garbage, clamp to
    the 3s model minimum (a smaller cap would silently score zero-padding)."""
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is not a number; using default %.0fs", name, default)
        value = default
    return max(value, _WIN / 16000)


# Longest audio scored in the single forward pass (memory/latency bound on CPU).
_SCORE_MAX_S = _score_seconds_env("VG_SCORE_SECONDS", 60.0)


def _load_wav_mono16k(path: str):
    """Load *path* as a mono (1, T) float32 tensor at 16 kHz (single file read)."""
    import torch
    import torchaudio

    data, sr = _read_audio(path)
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, T)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def _guard_audio_quality(wav) -> None:
    """Reject too-short / near-silent clips — the detector is only meaningful on
    speech, and returns a confident (usually wrong) "fake" on silence/noise."""
    if wav.shape[-1] / 16000 < 0.8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio too short to analyse — please upload at least ~1 second of speech.",
        )
    if float(wav.pow(2).mean().sqrt()) < 1e-3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio is silent or near-silent — no speech detected to analyse.",
        )


def _ssl_fake_prob(model, wav, model_key: str) -> float:
    """One forward pass over `wav` (1, T) from its natural start; returns fake-prob.

    The SSL detector is only reliable on a recording scored *as recorded, from
    its start*: real audio opens with ambient lead-in while TTS starts mid-speech,
    and chunks cut from the middle of an utterance read as synthetic regardless
    of content (measured on the held-out set — sliding-window aggregations, max
    and mean alike, therefore flag real recordings as fake). Prepending silence
    is no fix either: it flips real fakes toward "real". So: single pass, capped
    at VG_SCORE_SECONDS; non-SSL research models keep their validated 3s crop.
    """
    import torch

    w = wav[..., : int(_SCORE_MAX_S * 16000)].squeeze(0)
    if model_key not in _SSL_KEYS:
        w = w[..., :_WIN]  # research CNNs are fixed-length 3s models
    if w.shape[-1] < _WIN:
        w = torch.nn.functional.pad(w, (0, _WIN - w.shape[-1]))
    inp = w.unsqueeze(0) if model_key in _SSL_KEYS else w.reshape(1, 1, -1)
    with torch.no_grad():
        return float(torch.softmax(model(inp), dim=-1)[0, 1])


def _detect_ssl_tensor(wav, model_key: str) -> tuple[str, float, float]:
    """Single-pass SSL detection on the clip from its natural start.

    Returns (label, confidence, seconds_analyzed). Earlier sliding-window
    aggregations (max, then mean) both misclassified real recordings because
    mid-utterance windows are out-of-distribution for the detector; the whole
    clip in one pass matches the regime the model's official EER and held-out
    real-pass were validated under. seconds_analyzed = min(duration,
    VG_SCORE_SECONDS) so callers and forensic reports can disclose truncation.
    """
    model = registry.load(model_key)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_key}' checkpoint not found. Set {model_key.upper()}_PATH.",
        )
    # Guard the region that will actually be scored, not the whole clip — a long
    # recording with a near-silent first VG_SCORE_SECONDS must 422, not let the
    # model confidently call silence "fake".
    scored = wav[..., : int(_SCORE_MAX_S * 16000)]
    _guard_audio_quality(scored)
    fake_prob = _ssl_fake_prob(model, wav, model_key)
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else 1.0 - fake_prob
    return label, round(confidence, 4), round(scored.shape[-1] / 16000, 2)


def _detect_ssl(path: str, model_key: str) -> tuple[str, float, int]:
    return _detect_ssl_tensor(_load_wav_mono16k(path), model_key)


def _first_window(wav):
    """First 3s window of `wav` (1, T) as a padded (1, _WIN) tensor for attribution."""
    import torch

    w = wav.squeeze(0)
    w = torch.nn.functional.pad(w, (0, _WIN - w.shape[-1])) if w.shape[-1] < _WIN else w[:_WIN]
    return w.unsqueeze(0)


def _build_explanation(model, wav_1xT) -> ExplanationResult | None:
    """Integrated-Gradients attribution → ExplanationResult (None on failure)."""
    try:
        from voiceguard.api.schemas import AttributionSegment
        from voiceguard.xai.ssl_explain import explain_waveform

        raw = explain_waveform(model, wav_1xT)
        return ExplanationResult(
            method=raw["method"],
            baseline=raw["baseline"],
            target_class=raw["target_class"],
            frame_duration_ms=raw["frame_duration_ms"],
            attribution_frames=raw["attribution_frames"],
            top_segments=[AttributionSegment(**s) for s in raw["top_segments"]],
        )
    except Exception:
        logger.warning("attribution failed", exc_info=True)
        return None


def _explain_ssl(path: str, model_key: str) -> ExplanationResult | None:
    """Run Integrated Gradients attribution on an SSL model (loads its own file)."""
    model = registry.load(model_key)
    if model is None:
        return None
    return _build_explanation(model, _first_window(_load_wav_mono16k(path)))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/token", response_model=TokenResponse, tags=["auth"])
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        logger.info("login failed for user=%r", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        {"sub": form_data.username, "role": role_for(form_data.username)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(access_token=token)


@app.post("/detect", response_model=DetectionResult, tags=["detection"])
@limiter.limit("60/minute")
async def detect(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: ModelType = ModelType.xls_r_aasist,
    explain: bool = False,
    _user: str = Depends(get_current_user),
):
    """Upload an audio file and receive a deepfake detection result.

    Accepted formats: WAV, MP3, FLAC, OGG (max 100MB). The clip is scored in a
    single forward pass from its natural start, exactly as recorded — sliding
    windows misclassify (mid-utterance segments read as synthetic to the model)
    — capped at VG_SCORE_SECONDS (default 60s) of audio; `seconds_analyzed`
    reports how much of the clip the verdict covers. Raw audio is auto-deleted
    after 60 seconds (PDPL compliance). Pass `explain=true` to include
    Integrated Gradients attribution showing which time segments drove the
    decision.
    """
    path, audio_hash = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    audio_info = _probe_audio(path)  # also enforces MAX_AUDIO_SECONDS (413)

    t0 = time.perf_counter()

    if model == ModelType.classical:
        label, confidence = _detect_classical(path)
        seconds_analyzed = audio_info.get("duration_s")
        explanation = None
    else:
        # Load the waveform once and share it between detection and attribution.
        wav = _load_wav_mono16k(path)
        label, confidence, seconds_analyzed = _detect_ssl_tensor(wav, str(model))
        explanation = None
        if explain:
            model_obj = registry.load(str(model))
            explanation = _build_explanation(model_obj, _first_window(wav)) if model_obj else None

    latency_ms = (time.perf_counter() - t0) * 1000

    # Record the server-verified result so /forensic/report can't be forged with a
    # client-supplied verdict (P0-10).
    result_store.record(
        audio_hash,
        label=label,
        confidence=round(confidence, 4),
        model=str(model),
        user=_user,
        timestamp=time.time(),
        windows_analyzed=1,
        seconds_analyzed=seconds_analyzed,
        audio_info=audio_info,
        app_version=__version__,
    )
    logger.info(
        "detect user=%s model=%s verdict=%s conf=%.3f analyzed_s=%s latency_ms=%.1f",
        _user,
        model,
        label,
        confidence,
        seconds_analyzed,
        latency_ms,
    )

    return DetectionResult(
        label=label,
        confidence=round(confidence, 4),
        model=model,
        latency_ms=round(latency_ms, 2),
        audio_hash=audio_hash,
        windows_analyzed=1,
        seconds_analyzed=seconds_analyzed,
        explanation=explanation,
    )


CLONE_QUOTA_PER_HOUR = int(os.environ.get("VG_CLONE_QUOTA_PER_HOUR", "10"))
_clone_log: dict[str, list[float]] = {}


def _claim_clone_quota(user: str) -> bool:
    """Per-user sliding-hour quota for voice cloning (the abuse-sensitive path)."""
    now = time.time()
    log = [t for t in _clone_log.get(user, []) if now - t < 3600]
    if len(log) >= CLONE_QUOTA_PER_HOUR:
        _clone_log[user] = log
        return False
    log.append(now)
    _clone_log[user] = log
    return True


def _schedule_media_cleanup(path: Path) -> None:
    """Delete a generated media file after MEDIA_TTL_S (PDPL minimisation)."""
    import threading

    def _rm() -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # Daemon timer: a pending best-effort cleanup must never block process
    # shutdown (otherwise the API — and the test suite — hangs for MEDIA_TTL_S).
    timer = threading.Timer(MEDIA_TTL_S, _rm)
    timer.daemon = True
    timer.start()


@app.get("/synthesis/engines", response_model=list[SynthesisEngineInfo], tags=["synthesis"])
async def synthesis_engines():
    """List synthesis engines and their availability (preset TTS + voice cloning)."""
    from voiceguard.synthesis.registry import registry as synth_registry

    return [SynthesisEngineInfo(**vars(i)) for i in synth_registry.info()]


@app.post("/synthesize", response_model=SynthesisResult, tags=["synthesis"])
@limiter.limit("20/minute")
async def synthesize(
    request: Request,
    background_tasks: BackgroundTasks,
    text: str = Form(..., min_length=1, max_length=2000),
    engine: str = Form("kokoro"),
    voice: str = Form("af_heart"),
    language: str = Form("en"),
    consent: bool = Form(False),  # UX acknowledgement only; gated in the UI
    reference: UploadFile | None = File(None),
    claims: tuple[str, str] = Depends(get_current_claims),
):
    """Synthesise speech and return a watermarked audio URL.

    `engine` selects a preset-voice TTS (Kokoro) or a zero-shot voice-cloning
    engine. Cloning engines require a `reference` audio clip (PDPL auto-deleted).
    Every output is spectrally watermarked as AI-generated.
    """
    import soundfile as sf
    from starlette.concurrency import run_in_threadpool

    from voiceguard.synthesis.registry import registry as synth_registry
    from voiceguard.watermark import c2pa_sign
    from voiceguard.watermark.c2pa_watermark import embed, ensure_carrier_sr

    _user, _role = claims

    eng = synth_registry.get(engine)
    if eng is None or not eng.is_available():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Synthesis engine '{engine}' is not available on this instance.",
        )

    ref_path: str | None = None
    if eng.requires_reference:
        # Voice cloning is the abuse-sensitive path: admin-only, with a per-user
        # hourly quota on top of the route's per-IP rate limit. Preset TTS stays
        # open to analysts.
        if _role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Voice cloning requires the 'admin' role.",
            )
        # Voice cloning consent is enforced server-side, not just in the UI.
        if not consent:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Voice cloning requires explicit consent that you are authorised to "
                    "clone this voice. Set consent=true to proceed."
                ),
            )
        if reference is None:
            raise HTTPException(
                status_code=422, detail=f"Engine '{engine}' requires a reference audio clip."
            )
        # Claimed only once the request is otherwise valid, so rejected attempts
        # don't burn the user's quota.
        if not _claim_clone_quota(_user):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Voice-clone quota reached ({CLONE_QUOTA_PER_HOUR}/hour per user).",
            )
        ref_path, _ = await save_upload(reference)
        background_tasks.add_task(pdpl_auto_delete, ref_path)
        logger.info("clone consent acknowledged user=%s engine=%s", _user, engine)

    t0 = time.perf_counter()
    try:
        audio, sr = await run_in_threadpool(
            eng.synthesize, text, voice=voice, reference_wav=ref_path, language=language
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

    # Resample up if needed so the 18 kHz watermark carrier stays inaudible
    # (Kokoro's 24 kHz would otherwise clamp the carrier into the audible band),
    # then embed at an inaudible amplitude (≤ the documented 0.005).
    wm_audio, wm_sr = ensure_carrier_sr(np.asarray(audio, dtype=np.float32), sr)
    watermarked, watermark_id = embed(wm_audio, sr=wm_sr, amplitude=0.003)
    # uuid4, not a timestamp: /media is unauthenticated, so names must be unguessable
    fname = f"vg_{uuid.uuid4().hex}_{engine}.wav"
    out_path = MEDIA_DIR / fname
    sf.write(str(out_path), watermarked, wm_sr)

    # Cryptographic C2PA provenance: sign the file with a real, signed manifest
    # marking it AI-generated (trainedAlgorithmicMedia) — on top of the spectral
    # watermark. Best-effort: if the c2pa runtime is unavailable the spectral mark
    # alone still ships.
    signed_tmp = out_path.with_suffix(".signed.wav")
    c2pa_status = c2pa_sign.sign_file(
        str(out_path), str(signed_tmp), software_agent=f"VoiceGuard/{engine}"
    )
    if c2pa_status.get("signed") and signed_tmp.exists():
        os.replace(str(signed_tmp), str(out_path))
    else:
        signed_tmp.unlink(missing_ok=True)

    _schedule_media_cleanup(out_path)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "synthesize user=%s engine=%s c2pa=%s latency_ms=%.1f",
        _user,
        engine,
        bool(c2pa_status.get("signed")),
        latency_ms,
    )
    return SynthesisResult(
        audio_url=f"/api/media/{fname}",
        watermark_id=watermark_id,
        synthesis_latency_ms=round(latency_ms, 2),
        engine=engine,
        c2pa_signed=bool(c2pa_status.get("signed")),
    )


@app.post("/forensic/report", response_model=ForensicReportResult, tags=["forensics"])
@limiter.limit("10/minute")
async def forensic_report(
    request: Request,
    body: ForensicReportRequest,
    _user: str = Depends(get_current_user),
):
    """Generate a NIST SP 800-86 compliant PDF forensic report.

    The verdict is taken from VoiceGuard's own server-side detection record for the
    audio hash (set by /detect), NOT from any client-supplied value — a report
    cannot be forged with a fabricated verdict. Returns 404 if no detection has
    been run for this hash.
    """
    from voiceguard.forensics.chain_of_custody import ChainOfCustody
    from voiceguard.forensics.pdf_report import generate_report

    record = result_store.get(body.audio_hash)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No server-side detection record for this audio hash — run /detect first.",
        )
    # Build the report from the verified record; the request's detection_result is ignored.
    detection_result = {
        "label": record["label"],
        "confidence": record["confidence"],
        "model": record["model"],
        "server_verified": True,
    }

    coc = ChainOfCustody()
    coc.add_event("evidence_received", body.analyst_name, body.audio_hash, "Audio submitted")
    coc.add_event(
        "analysis_completed",
        "VoiceGuard",
        body.audio_hash,
        f"Verdict: {record['label']} (server-verified)",
    )

    audio_meta = dict(record.get("audio_info") or {})
    if record.get("windows_analyzed") is not None:
        audio_meta["windows_analyzed"] = record["windows_analyzed"]
    if record.get("seconds_analyzed") is not None:
        audio_meta["seconds_analyzed"] = record["seconds_analyzed"]
    model_meta = {
        "model": record["model"],
        "app_version": record.get("app_version", __version__),
        # Cached after the first report; ~1 GB SSL checkpoint hashes in seconds.
        "checkpoint_sha256": registry.fingerprint(record["model"]),
    }

    # uuid4, not a timestamp: /media is unauthenticated, so names must be unguessable
    fname = f"report_{body.audio_hash[:12]}_{uuid.uuid4().hex}.pdf"
    out_path = MEDIA_DIR / fname
    try:
        generate_report(
            audio_hash=body.audio_hash,
            detection_result=detection_result,
            chain_of_custody=coc.to_dict(),
            analyst_name=body.analyst_name,
            output_path=out_path,
            audio_meta=audio_meta or None,
            model_meta=model_meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    _schedule_media_cleanup(out_path)
    return ForensicReportResult(
        report_url=f"/api/media/{fname}",
        chain_of_custody_hash=coc.chain_hash,
    )


class _ConnectionBudget:
    """Process-wide cap on concurrent streaming connections.

    Every stream window drives model inference on a CPU box, so an uncapped
    WebSocket is a one-client DoS. Single event loop => plain counter is safe.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)


_stream_budget = _ConnectionBudget(int(os.environ.get("VG_WS_MAX_CONNECTIONS", "4")))
_WS_MAX_SECONDS = float(os.environ.get("VG_WS_MAX_SECONDS", "900"))
_WS_PCM_BYTES_PER_S = 16000 * 2  # realtime int16 @ 16 kHz
_WS_RATE_SLACK = 4.0  # tolerate 4x realtime (buffered sends) before closing
_WS_BURST_BYTES = 512 * 1024


async def _ws_first_message_token(websocket: WebSocket) -> str:
    """Read the JWT from the first WS message: ``{"token": "<jwt>"}``.

    Preferred over ``?token=`` — query strings end up in nginx/proxy access
    logs. Returns "" on timeout, non-text frame, or malformed JSON.
    """
    import asyncio
    import json

    try:
        msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
    except TimeoutError:
        return ""
    if msg.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(code=int(msg.get("code") or 1000))
    text = msg.get("text")
    if not text:
        return ""
    try:
        return str(json.loads(text).get("token", ""))
    except (ValueError, AttributeError):
        return ""


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, token: str = ""):
    """Real-time microphone streaming detection.

    Auth: send ``{"token": "<jwt>"}`` as the first (text) message — the server
    replies ``{"type": "auth_ok"}`` — then stream raw PCM frames (int16, 16kHz,
    mono). ``?token=`` works too but is deprecated (it leaks into proxy logs).
    Server responds with JSON StreamDetectionEvent messages. Connections are
    capped globally (VG_WS_MAX_CONNECTIONS), per-session (VG_WS_MAX_SECONDS),
    and at ~4x realtime ingest.
    """
    await websocket.accept()
    if not token:
        try:
            token = await _ws_first_message_token(websocket)
        except WebSocketDisconnect:
            return
    try:
        verify_token_ws(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    if not _stream_budget.acquire():
        await websocket.close(code=1013)  # try again later — all slots busy
        return

    import asyncio

    # The detector is only reliable on audio scored from the recording's natural
    # start (see _ssl_fake_prob) — a mic stream's natural start is the moment the
    # user hit Start. So each verdict re-scores the growing prefix of the session
    # (first at 3s, then every ~2s), capped at VG_WS_SCORE_SECONDS of audio (CPU
    # cost grows with prefix length). The verdict that covers the full cap is sent
    # with final=true and scoring stops; the connection stays open only so the
    # session/rate limits keep being enforced.
    first_score_bytes = 3 * _WS_PCM_BYTES_PER_S
    stride_bytes = 2 * _WS_PCM_BYTES_PER_S
    cap_s = _score_seconds_env("VG_WS_SCORE_SECONDS", 15.0)
    # Even-align so a prefix slice is always whole int16 samples.
    score_cap_bytes = max(int(cap_s * _WS_PCM_BYTES_PER_S) & ~1, first_score_bytes)
    buffer = bytearray()
    window_id = 0
    next_score_at = first_score_bytes
    verdict_final = False
    started = time.monotonic()
    received = 0

    try:
        await websocket.send_json({"type": "auth_ok"})
        while True:
            data = await websocket.receive_bytes()
            received += len(data)
            elapsed = time.monotonic() - started
            if elapsed > _WS_MAX_SECONDS:
                await websocket.close(code=1000)
                return
            if received > _WS_BURST_BYTES + elapsed * _WS_PCM_BYTES_PER_S * _WS_RATE_SLACK:
                await websocket.close(code=1008)  # faster than any real microphone
                return
            if len(buffer) < score_cap_bytes:
                buffer.extend(data)
            if verdict_final or received < next_score_at:
                continue

            # Coalesce overdue milestones: score the freshest prefix once instead
            # of replaying every stale 2s boundary after a burst or a slow pass.
            scored_to = min(received, score_cap_bytes, len(buffer)) & ~1
            audio = (
                np.frombuffer(buffer, dtype=np.int16, count=scored_to // 2).astype(np.float32)
                / 32768.0
            )
            # to_thread: model inference must not block WS keepalives
            label, confidence, used_model = await asyncio.to_thread(_detect_ssl_array, audio, 16000)
            verdict_final = scored_to >= score_cap_bytes

            event = StreamDetectionEvent(
                timestamp_ms=time.time() * 1000,
                window_id=window_id,
                label=label,
                confidence=confidence,
                model=used_model,
                final=verdict_final,
            )
            await websocket.send_json(event.model_dump())
            window_id += 1
            next_score_at = scored_to + stride_bytes

    except WebSocketDisconnect:
        pass
    finally:
        _stream_budget.release()


def _verify_twilio_signature(websocket: WebSocket) -> bool:
    """Validate Twilio's X-Twilio-Signature on the WebSocket handshake.

    Twilio signs every request as base64(HMAC-SHA1(auth_token, public URL)).
    With TWILIO_AUTH_TOKEN unset the bridge stays open in development but is
    refused in production — an unauthenticated endpoint would let anyone burn
    model inference.
    """
    import base64
    import hmac

    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not token:
        return os.environ.get("VG_ENV", "development") != "production"
    signature = websocket.headers.get("x-twilio-signature", "")
    if not signature:
        return False
    # Rebuild the public URL Twilio signed (nginx preserves Host + sets
    # X-Forwarded-Proto; Twilio Media Streams always connects over wss).
    proto = websocket.headers.get("x-forwarded-proto", websocket.url.scheme)
    scheme = "wss" if proto in ("https", "wss") else "ws"
    host = websocket.headers.get("host", websocket.url.netloc)
    url = f"{scheme}://{host}{websocket.url.path}"
    if websocket.url.query:
        url += f"?{websocket.url.query}"
    expected = base64.b64encode(
        hmac.new(token.encode(), url.encode(), hashlib.sha1).digest()  # noqa: S324 — Twilio's scheme is HMAC-SHA1
    ).decode()
    return hmac.compare_digest(expected, signature)


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """Twilio Media Stream WebSocket bridge.

    Receives μ-law encoded 8kHz audio from Twilio and runs detection.
    Authenticated via X-Twilio-Signature when TWILIO_AUTH_TOKEN is set.
    """
    if not _verify_twilio_signature(websocket):
        await websocket.close(code=1008)  # rejects the handshake with HTTP 403
        return
    await websocket.accept()
    if not _stream_budget.acquire():
        await websocket.close(code=1013)  # all inference slots busy
        return
    try:
        from voiceguard.voip.twilio_bridge import TwilioStreamHandler

        handler = TwilioStreamHandler()
        await handler.handle(websocket)
    except WebSocketDisconnect:
        pass
    except ImportError:
        await websocket.close(code=1011)
    finally:
        _stream_budget.release()


@app.post("/watermark/verify", response_model=WatermarkVerifyResult, tags=["synthesis"])
@limiter.limit("30/minute")
async def watermark_verify(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    watermark_id: str = Form(""),
    _user: str = Depends(get_current_user),
):
    """Verify the provenance of an audio file — the read side of /synthesize.

    Checks two independent marks: the keyed spectral watermark (only when the
    `watermark_id` returned by /synthesize is supplied) and the embedded C2PA
    manifest (cryptographic provenance, no key needed). Closes the
    Generate → Detect loop: anything VoiceGuard synthesises can be proven so.
    """
    from voiceguard.watermark import c2pa_sign
    from voiceguard.watermark.c2pa_watermark import detect as wm_detect

    path, _ = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    result = WatermarkVerifyResult()

    if watermark_id:
        data, sr = _read_audio(path)
        detected, corr = wm_detect(data, sr=sr, watermark_id=watermark_id)
        result.spectral_checked = True
        result.spectral_detected = bool(detected)
        result.spectral_correlation = round(float(corr), 6)

    c2pa = c2pa_sign.verify_file(path)
    result.c2pa_has_manifest = bool(c2pa.get("has_manifest"))
    result.c2pa_validation_state = c2pa.get("validation_state")
    result.c2pa_ai_generated = c2pa.get("ai_generated")
    result.c2pa_software_agent = c2pa.get("software_agent")

    if result.spectral_detected:
        result.verdict = "voiceguard-generated"
    elif result.c2pa_ai_generated:
        result.verdict = "ai-generated"
    elif result.c2pa_has_manifest:
        result.verdict = "unknown"
    else:
        result.verdict = "no-provenance-found"

    logger.info(
        "watermark/verify user=%s spectral=%s c2pa=%s verdict=%s",
        _user,
        result.spectral_detected if result.spectral_checked else "skipped",
        result.c2pa_has_manifest,
        result.verdict,
    )
    return result


@app.post("/explain", response_model=ExplanationResult, tags=["detection"])
@limiter.limit("20/minute")
async def explain(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: ModelType = ModelType.xls_r_aasist,
    _user: str = Depends(get_current_user),
):
    """Return Integrated Gradients attribution for an uploaded audio file.

    Shows which time segments (10ms bins) drove the model's fake/real decision.
    Only SSL models (wav2vec2, wavlm_base_plus, wav2vec2_large, aasist, dsfnet*)
    support attribution; classical model returns 501.
    """
    if model == ModelType.classical:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Classical model does not support attribution. Use an SSL model.",
        )
    path, _ = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    explanation = _explain_ssl(path, str(model))
    if explanation is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model}' checkpoint not found or attribution failed.",
        )
    return explanation


@app.get("/models", tags=["ops"])
async def models_list():
    """List all registered model keys and their checkpoint availability."""
    return registry.status()


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    status_map = registry.status()
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded={k: v["available"] for k, v in status_map.items()},
    )


# ── Internal helpers ───────────────────────────────────────────────────────────


def verify_token_ws(token: str) -> str:
    """Verify JWT token for WebSocket connections."""
    from voiceguard.api.auth import verify_token

    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    return verify_token(token)


def _detect_classical_array(audio: np.ndarray, sr: int) -> tuple[str, float]:
    from voiceguard.features.extractor import extract_features

    features = extract_features(audio, sr)
    detector = registry.load("classical")
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)


def _detect_ssl_array(
    audio: np.ndarray, sr: int, model_key: str = "xls_r_aasist"
) -> tuple[str, float, str]:
    """Score a raw audio window with the SSL production detector (live streaming).

    Returns (label, confidence, model) where `model` names which detector actually
    scored the window: the SSL key, "classical" (SSL checkpoint unavailable), or
    "stub" (no detector at all) — so the client knows what produced the verdict.
    """
    import torch
    import torchaudio

    model = registry.load(model_key)
    if model is None:
        label, confidence = _detect_classical_array(audio, sr)
        used = "classical" if registry.load("classical") is not None else "stub"
        logger.info("stream: %s unavailable, scored with %s", model_key, used)
        return label, confidence, used
    wav = torch.as_tensor(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    # Score the audio as given, from its start (padded up to the 3s minimum,
    # capped at VG_SCORE_SECONDS) — see _ssl_fake_prob for why no windowing.
    fake_prob = _ssl_fake_prob(model, wav, model_key)
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else 1.0 - fake_prob
    return label, round(confidence, 4), model_key
