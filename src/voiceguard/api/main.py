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
import os
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
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
    create_access_token,
    get_current_user,
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
    SynthesisRequest,
    SynthesisResult,
    TokenResponse,
)

__version__ = "0.1.0"

# ── Model registry ─────────────────────────────────────────────────────────────

from voiceguard.models.registry import registry  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.preload()  # loads any model whose env-var is set at startup
    yield


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
    allow_credentials=True,
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

ACCEPTED_CONTENT_TYPES = {"audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/flac"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB


async def save_upload(upload: UploadFile) -> tuple[str, str]:
    """Save upload to temp file, return (path, sha256_hex)."""
    content = await upload.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit",
        )
    sha256 = hashlib.sha256(content).hexdigest()
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    fd, path = make_temp_audio_file(suffix=suffix)
    try:
        import os as _os

        _os.write(fd, content)
        _os.close(fd)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to save upload") from exc
    PDPLTimingMiddleware.register(path)
    return path, sha256


def _detect_classical(path: str) -> tuple[str, float]:
    """Run classical detection. Returns (label, confidence)."""
    from scipy.io import wavfile

    from voiceguard.features.extractor import extract_features

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    mx = np.max(np.abs(data)) + 1e-8
    data /= mx
    features = extract_features(data, sr)

    detector = registry.load("classical")
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)


def _explain_ssl(path: str, model_key: str) -> ExplanationResult | None:
    """Run Integrated Gradients attribution on an SSL model."""
    import torch
    import torchaudio

    from voiceguard.xai.ssl_explain import explain_waveform

    model = registry.load(model_key)
    if model is None:
        return None
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0)  # (T,)
    target_len = 48000
    if wav.shape[-1] < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
    else:
        wav = wav[..., :target_len]
    wav = wav.unsqueeze(0)  # (1, T)

    try:
        raw = explain_waveform(model, wav)
        from voiceguard.api.schemas import AttributionSegment

        return ExplanationResult(
            method=raw["method"],
            baseline=raw["baseline"],
            target_class=raw["target_class"],
            frame_duration_ms=raw["frame_duration_ms"],
            attribution_frames=raw["attribution_frames"],
            top_segments=[AttributionSegment(**s) for s in raw["top_segments"]],
        )
    except Exception:  # noqa: S110
        return None


def _detect_ssl(path: str, model_key: str) -> tuple[str, float]:
    """Run SSL/DSFNet/AASIST detection via the registry."""
    import torch
    import torchaudio

    model = registry.load(model_key)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_key}' checkpoint not found. Set {model_key.upper()}_PATH.",
        )
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0, keepdim=True)  # mono (1, T)

    # Input-quality guard: the detector is only meaningful on actual speech.
    # Reject clips that are too short or near-silent rather than returning a
    # confident (and usually wrong) "fake" on silence/noise/empty uploads.
    duration_s = wav.shape[-1] / 16000
    rms = float(wav.pow(2).mean().sqrt())
    if duration_s < 0.8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio too short to analyse — please upload at least ~1 second of speech.",
        )
    if rms < 1e-3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio is silent or near-silent — no speech detected to analyse.",
        )

    target_len = 48000
    if wav.shape[-1] < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
    else:
        wav = wav[..., :target_len]

    with torch.no_grad():
        # SSL models expect (B, T); CNN models expect (B, 1, T)
        if model_key in (
            "wav2vec2",
            "wavlm_base_plus",
            "wavlm_large",
            "wav2vec2_large",
            "xls_r",
            "xls_r_aasist",
        ):
            inp = wav.squeeze(0).unsqueeze(0)  # (1, T)
        else:
            inp = wav.unsqueeze(0)  # (1, 1, T)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = float(probs[1])
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else float(probs[0])
    return label, round(confidence, 4)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/token", response_model=TokenResponse, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        {"sub": form_data.username},
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

    Accepted formats: WAV, MP3, FLAC (max 100MB).
    Raw audio is auto-deleted after 60 seconds (PDPL compliance).
    Pass `explain=true` to include Integrated Gradients attribution showing
    which time segments drove the fake/real decision.
    """
    path, audio_hash = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    t0 = time.perf_counter()

    if model == ModelType.classical:
        label, confidence = _detect_classical(path)
        explanation = None
    else:
        label, confidence = _detect_ssl(path, str(model))
        explanation = _explain_ssl(path, str(model)) if explain else None

    latency_ms = (time.perf_counter() - t0) * 1000

    return DetectionResult(
        label=label,
        confidence=round(confidence, 4),
        model=model,
        latency_ms=round(latency_ms, 2),
        audio_hash=audio_hash,
        explanation=explanation,
    )


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


@app.post("/synthesize", response_model=SynthesisResult, tags=["synthesis"])
@limiter.limit("20/minute")
async def synthesize(
    request: Request,
    body: SynthesisRequest,
    _user: str = Depends(get_current_user),
):
    """Synthesise speech from text using Kokoro-82M with C2PA watermarking.

    Returns a URL to the synthesised (and watermarked) audio file.
    """
    from starlette.concurrency import run_in_threadpool

    from voiceguard.synthesis.kokoro_synth import synthesize_to_file

    t0 = time.perf_counter()
    try:
        fname, watermark_id, _dur = await run_in_threadpool(
            synthesize_to_file,
            body.text,
            str(MEDIA_DIR),
            body.voice,
            body.language,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Synthesis engine (Kokoro) is not installed on this instance.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

    _schedule_media_cleanup(MEDIA_DIR / fname)
    latency_ms = (time.perf_counter() - t0) * 1000
    return SynthesisResult(
        audio_url=f"/api/media/{fname}",
        watermark_id=watermark_id,
        synthesis_latency_ms=round(latency_ms, 2),
    )


@app.post("/forensic/report", response_model=ForensicReportResult, tags=["forensics"])
@limiter.limit("10/minute")
async def forensic_report(
    request: Request,
    body: ForensicReportRequest,
    _user: str = Depends(get_current_user),
):
    """Generate a NIST SP 800-86 compliant PDF forensic report."""
    from voiceguard.forensics.chain_of_custody import ChainOfCustody
    from voiceguard.forensics.pdf_report import generate_report

    coc = ChainOfCustody()
    coc.add_event("evidence_received", body.analyst_name, body.audio_hash, "Audio submitted")
    verdict = str(body.detection_result.get("label", "unknown"))
    coc.add_event("analysis_completed", "VoiceGuard", body.audio_hash, f"Verdict: {verdict}")

    fname = f"report_{body.audio_hash[:12]}_{int(time.time())}.pdf"
    out_path = MEDIA_DIR / fname
    try:
        generate_report(
            audio_hash=body.audio_hash,
            detection_result=body.detection_result,
            chain_of_custody=coc.to_dict(),
            analyst_name=body.analyst_name,
            output_path=out_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    _schedule_media_cleanup(out_path)
    return ForensicReportResult(
        report_url=f"/api/media/{fname}",
        chain_of_custody_hash=coc.chain_hash,
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, token: str = ""):
    """Real-time microphone streaming detection.

    Client sends raw PCM frames (int16, 16kHz, mono).
    Server responds with JSON StreamDetectionEvent messages.
    """
    await websocket.accept()
    try:
        verify_token_ws(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    buffer = bytearray()
    window_id = 0
    WINDOW_BYTES = 16000 * 2 * 3  # 3 seconds of int16 @ 16kHz

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)
            while len(buffer) >= WINDOW_BYTES:
                window = bytes(buffer[:WINDOW_BYTES])
                buffer = buffer[WINDOW_BYTES // 3 :]  # 1-second hop

                audio = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0

                label, confidence = _detect_classical_array(audio, sr=16000)

                event = StreamDetectionEvent(
                    timestamp_ms=time.time() * 1000,
                    window_id=window_id,
                    label=label,
                    confidence=confidence,
                )
                await websocket.send_json(event.model_dump())
                window_id += 1

    except WebSocketDisconnect:
        pass


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """Twilio Media Stream WebSocket bridge.

    Receives μ-law encoded 8kHz audio from Twilio and runs detection.
    """
    await websocket.accept()
    try:
        from voiceguard.voip.twilio_bridge import TwilioStreamHandler

        handler = TwilioStreamHandler()
        await handler.handle(websocket)
    except WebSocketDisconnect:
        pass
    except ImportError:
        await websocket.close(code=1011)


@app.post("/explain", response_model=ExplanationResult, tags=["detection"])
@limiter.limit("20/minute")
async def explain(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: ModelType = ModelType.wav2vec2,
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
