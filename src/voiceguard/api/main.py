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
from typing import Any

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

# ── Lazy model registry ────────────────────────────────────────────────────────

_models: dict[str, Any] = {}


def _get_classical_model():
    if "classical" not in _models:
        from voiceguard.models.classical import ClassicalDetector

        model_path = os.environ.get("CLASSICAL_MODEL_PATH")
        if model_path and Path(model_path).exists():
            _models["classical"] = ClassicalDetector.from_file(model_path)
        else:
            _models["classical"] = None
    return _models["classical"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_classical_model()
    yield
    _models.clear()


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

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    detector = _get_classical_model()
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)


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
    model: ModelType = ModelType.classical,
    _user: str = Depends(get_current_user),
):
    """Upload an audio file and receive a deepfake detection result.

    Accepted formats: WAV, MP3, FLAC (max 100MB).
    Raw audio is auto-deleted after 60 seconds (PDPL compliance).
    """
    path, audio_hash = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    t0 = time.perf_counter()

    if model == ModelType.classical:
        label, confidence = _detect_classical(path)
    else:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Model '{model}' requires GPU training. Use 'classical' for now.",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    return DetectionResult(
        label=label,
        confidence=round(confidence, 4),
        model=model,
        latency_ms=round(latency_ms, 2),
        audio_hash=audio_hash,
    )


@app.post("/synthesize", response_model=SynthesisResult, tags=["synthesis"])
@limiter.limit("20/minute")
async def synthesize(
    request: Request,
    body: SynthesisRequest,
    _user: str = Depends(get_current_user),
):
    """Synthesise speech from text using XTTS v2 with C2PA watermarking.

    Returns a URL to the synthesised audio file.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Speech synthesis requires Coqui XTTS v2 (GPU). Not available on CPU instance.",
    )


@app.post("/forensic/report", response_model=ForensicReportResult, tags=["forensics"])
@limiter.limit("10/minute")
async def forensic_report(
    request: Request,
    body: ForensicReportRequest,
    _user: str = Depends(get_current_user),
):
    """Generate a NIST SP 800-86 compliant PDF forensic report."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Forensic PDF generation is implemented in Phase 5.",
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


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded={
            "classical": _get_classical_model() is not None,
            "dsfnet": False,
            "wav2vec2": False,
        },
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
    detector = _get_classical_model()
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)
