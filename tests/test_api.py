"""
FastAPI endpoint tests.

Uses httpx.AsyncClient with ASGI transport — no network calls.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from voiceguard.api.main import app


def make_wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    samples = (np.random.randn(int(sr * duration_s)) * 0.1 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return make_wav_bytes()


async def get_token(client: AsyncClient) -> str:
    resp = await client.post(
        "/token",
        data={"username": "admin", "password": "voiceguard2026"},  # pragma: allowlist secret
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture
async def auth_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await get_token(client)
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


# ── Auth tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_valid_credentials():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/token",
            data={"username": "admin", "password": "voiceguard2026"},  # pragma: allowlist secret
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"  # noqa: S105


@pytest.mark.asyncio
async def test_token_invalid_credentials():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/token",
            data={"username": "wrong", "password": "wrong"},  # pragma: allowlist secret
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_detect_requires_auth(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/detect",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    assert resp.status_code == 401


# ── Detection tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_wav(auth_client, wav_bytes):
    # Use the lightweight classical detector — the SSL default (xls_r_aasist)
    # needs a ~1.2GB checkpoint that is not present in CI.
    resp = await auth_client.post(
        "/detect",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        params={"model": "classical"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("real", "fake")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model"] == "classical"
    assert data["latency_ms"] >= 0
    assert len(data["audio_hash"]) == 64  # SHA-256 hex


@pytest.mark.asyncio
async def test_detect_model_field(auth_client, wav_bytes):
    resp = await auth_client.post(
        "/detect",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        params={"model": "classical"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_detect_model_without_checkpoint(auth_client, wav_bytes):
    resp = await auth_client.post(
        "/detect",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        params={"model": "dsfnet"},
    )
    assert resp.status_code == 503


# ── Health test ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "models_loaded" in data


# ── Explain endpoint ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_explain_classical_returns_501(auth_client, wav_bytes):
    resp = await auth_client.post(
        "/explain",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        params={"model": "classical"},
    )
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_explain_missing_checkpoint_returns_503(auth_client, wav_bytes):
    resp = await auth_client.post(
        "/explain",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        params={"model": "dsfnet"},
    )
    assert resp.status_code == 503


# ── Synthesis / forensics ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesis_engines_lists_kokoro_and_clone(auth_client):
    resp = await auth_client.get("/synthesis/engines")
    assert resp.status_code == 200
    names = {e["name"] for e in resp.json()}
    assert "kokoro" in names
    assert {"indextts2", "xtts"} <= names  # cloning engines listed (may be unavailable)


@pytest.mark.asyncio
async def test_synthesize_kokoro(auth_client, monkeypatch):
    # Stub Kokoro so the contract is tested without the TTS model in CI.
    import numpy as np

    from voiceguard.synthesis import kokoro_engine

    monkeypatch.setattr(kokoro_engine.KokoroEngine, "is_available", lambda self: True)
    monkeypatch.setattr(
        kokoro_engine,
        "synthesize_raw",
        lambda text, voice="af_heart", language="en": (np.zeros(16000, dtype=np.float32), 16000),
    )
    resp = await auth_client.post("/synthesize", data={"text": "Hello world", "engine": "kokoro"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["engine"] == "kokoro"
    assert data["audio_url"].endswith(".wav")
    assert data["watermark_id"]


@pytest.mark.asyncio
async def test_synthesize_unavailable_engine_501(auth_client, monkeypatch):
    # Force the cloning engine unavailable so the assertion holds regardless of
    # whether this host happens to have the (optional, multi-GB) engine installed.
    from voiceguard.synthesis import clone_engine

    monkeypatch.setattr(clone_engine.IndexTTS2Engine, "is_available", lambda self: False)
    resp = await auth_client.post("/synthesize", data={"text": "hi", "engine": "indextts2"})
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_synthesize_clone_requires_reference_422(auth_client, monkeypatch):
    # Cloning engine "available" but no reference clip provided → 422.
    from voiceguard.synthesis import clone_engine

    monkeypatch.setattr(clone_engine.IndexTTS2Engine, "is_available", lambda self: True)
    resp = await auth_client.post("/synthesize", data={"text": "hi", "engine": "indextts2"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_forensic_report_returns_pdf_url(auth_client):
    resp = await auth_client.post(
        "/forensic/report",
        json={"audio_hash": "a" * 64},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["report_url"].endswith(".pdf")
    assert len(data["chain_of_custody_hash"]) == 64  # SHA-256 hex
