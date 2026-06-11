"""Tests for the security-hardening pass: magic-byte sniffing, duration cap,
role-gated cloning + quota, WS first-message auth, Twilio signature validation,
and the /watermark/verify provenance endpoint."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import wave

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.websockets import WebSocketDisconnect

import voiceguard.api.main as main_mod
from voiceguard.api.main import app


def make_wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    samples = (np.random.randn(int(sr * duration_s)) * 0.1 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


async def get_token(client: AsyncClient, username: str = "admin", password: str = "") -> str:
    password = password or {"admin": "voiceguard2026", "analyst": "analyst2026"}[username]
    resp = await client.post(
        "/token",
        data={"username": username, "password": password},
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


@pytest.fixture
async def analyst_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        token = await get_token(client, "analyst")
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


# ── Upload validation ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_rejects_disguised_text_415(auth_client):
    """A text payload named .wav with an audio content-type must fail the
    magic-byte sniff, not reach libsndfile."""
    resp = await auth_client.post(
        "/detect",
        files={"file": ("evil.wav", b"#!/bin/sh\necho pwned\n" * 4, "audio/wav")},
    )
    assert resp.status_code == 415
    assert "not WAV, MP3, FLAC, or OGG" in resp.json()["detail"]


def test_sniff_accepts_all_advertised_formats():
    from voiceguard.api.main import _sniff_is_audio

    assert _sniff_is_audio(b"RIFF\x24\x08\x00\x00WAVEfmt ")
    assert _sniff_is_audio(b"fLaC\x00\x00\x00\x22" + b"\x00" * 4)
    assert _sniff_is_audio(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00")  # OGG capture pattern
    assert _sniff_is_audio(b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00")
    assert _sniff_is_audio(b"\xff\xfb\x90\x00" + b"\x00" * 8)  # bare MPEG frame sync
    assert not _sniff_is_audio(b"#!/bin/sh\n\x00\x00")
    assert not _sniff_is_audio(b"OggX" + b"\x00" * 8)


@pytest.mark.asyncio
async def test_detect_duration_cap_413(auth_client, monkeypatch):
    monkeypatch.setattr(main_mod, "MAX_AUDIO_SECONDS", 1.0)
    resp = await auth_client.post(
        "/detect",
        params={"model": "classical"},
        files={"file": ("long.wav", make_wav_bytes(duration_s=3.0), "audio/wav")},
    )
    assert resp.status_code == 413
    assert "analysis limit" in resp.json()["detail"]


# ── Roles + clone quota ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_requires_admin_role_403(analyst_client, monkeypatch):
    from voiceguard.synthesis import clone_engine

    monkeypatch.setattr(clone_engine.IndexTTS2Engine, "is_available", lambda self: True)
    resp = await analyst_client.post(
        "/synthesize", data={"text": "hi", "engine": "indextts2", "consent": "true"}
    )
    assert resp.status_code == 403
    assert "admin" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_clone_quota_429(auth_client, monkeypatch):
    from voiceguard.synthesis import clone_engine

    monkeypatch.setattr(clone_engine.IndexTTS2Engine, "is_available", lambda self: True)
    monkeypatch.setattr(main_mod, "CLONE_QUOTA_PER_HOUR", 0)
    monkeypatch.setattr(main_mod, "_clone_log", {})
    resp = await auth_client.post(
        "/synthesize",
        data={"text": "hi", "engine": "indextts2", "consent": "true"},
        files={"reference": ("ref.wav", make_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_token_carries_role_claim():
    from jose import jwt

    from voiceguard.api.auth import ALGORITHM, SECRET_KEY

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        admin = jwt.decode(await get_token(client, "admin"), SECRET_KEY, algorithms=[ALGORITHM])
        analyst = jwt.decode(await get_token(client, "analyst"), SECRET_KEY, algorithms=[ALGORITHM])
    assert admin["role"] == "admin"
    assert analyst["role"] == "analyst"


# ── /watermark/verify ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_watermark_verify_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/watermark/verify", files={"file": ("a.wav", make_wav_bytes(), "audio/wav")}
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_watermark_verify_spectral_roundtrip(auth_client):
    """Embed → verify closes the provenance loop: the keyed mark must be found
    with the right watermark_id and missed with a wrong one."""
    from voiceguard.watermark.c2pa_watermark import embed

    sr = 22050
    audio = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    marked, watermark_id = embed(audio, sr=sr, amplitude=0.01)
    buf = io.BytesIO()
    sf.write(buf, marked, sr, format="WAV")

    resp = await auth_client.post(
        "/watermark/verify",
        data={"watermark_id": watermark_id},
        files={"file": ("marked.wav", buf.getvalue(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["spectral_checked"] is True
    assert data["spectral_detected"] is True
    assert data["verdict"] == "voiceguard-generated"

    buf.seek(0)
    resp = await auth_client.post(
        "/watermark/verify",
        data={"watermark_id": "definitely-the-wrong-id"},
        files={"file": ("marked.wav", buf.getvalue(), "audio/wav")},
    )
    assert resp.json()["spectral_detected"] is False


@pytest.mark.asyncio
async def test_watermark_verify_unmarked_audio(auth_client):
    resp = await auth_client.post(
        "/watermark/verify", files={"file": ("plain.wav", make_wav_bytes(), "audio/wav")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["spectral_checked"] is False
    assert data["verdict"] in ("no-provenance-found", "unknown")


# ── WebSocket auth ─────────────────────────────────────────────────────────────


def _sync_token(client: TestClient) -> str:
    resp = client.post(
        "/token",
        data={"username": "admin", "password": "voiceguard2026"},  # pragma: allowlist secret
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_ws_stream_first_message_auth_ok():
    with TestClient(app) as client:
        token = _sync_token(client)
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"token": token}))
            assert ws.receive_json() == {"type": "auth_ok"}


def test_ws_stream_first_message_bad_token_closed_1008():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_text(json.dumps({"token": "bogus"}))
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


def test_ws_stream_query_token_still_accepted():
    """Deprecated ?token= path stays working for old clients."""
    with TestClient(app) as client:
        token = _sync_token(client)
        with client.websocket_connect(f"/ws/stream?token={token}") as ws:
            assert ws.receive_json() == {"type": "auth_ok"}


# ── Twilio signature ───────────────────────────────────────────────────────────


def _twilio_signature(auth_token: str, url: str) -> str:
    return base64.b64encode(
        hmac.new(auth_token.encode(), url.encode(), hashlib.sha1).digest()  # noqa: S324
    ).decode()


def test_twilio_stream_rejects_missing_signature(monkeypatch):
    # The close happens BEFORE accept (handshake denial), so the disconnect
    # surfaces on connect itself.
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-secret")
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/twilio/stream"):
                pass
        assert exc.value.code == 1008


def test_twilio_stream_accepts_valid_signature(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "twilio-secret")
    sig = _twilio_signature("twilio-secret", "ws://testserver/twilio/stream")
    with TestClient(app) as client:
        with client.websocket_connect("/twilio/stream", headers={"x-twilio-signature": sig}) as ws:
            ws.send_text(json.dumps({"event": "stop"}))  # handler exits cleanly


def test_twilio_stream_open_in_development(monkeypatch):
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("VG_ENV", "development")
    with TestClient(app) as client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(json.dumps({"event": "stop"}))


def test_twilio_stream_refused_in_production_without_token(monkeypatch):
    from voiceguard.api import auth as auth_mod

    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("VG_ENV", "production")
    # Satisfy the production startup guard so the lifespan can run.
    monkeypatch.setattr(auth_mod, "SECRET_KEY", "a" * 64)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/twilio/stream"):
                pass
        assert exc.value.code == 1008
