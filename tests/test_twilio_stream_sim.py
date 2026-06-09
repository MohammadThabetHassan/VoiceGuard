"""Integration test: simulate a full Twilio Media Stream call against the live
/twilio/stream WebSocket — start → media frames → stop — and assert the bridge
emits a detection event. Proves the VoIP pipeline end-to-end without a real phone
number or Twilio account.
"""

from __future__ import annotations

import base64
import json

import numpy as np
from fastapi.testclient import TestClient

from voiceguard.api.main import app
from voiceguard.voip.twilio_bridge import _ULAW_TABLE

CALL_SID = "CAtest0000000000000000000000000000"
STREAM_SID = "MZtest0000000000000000000000000000"


def ulaw_encode(pcm: np.ndarray) -> bytes:
    """Encode float PCM [-1,1] to G.711 μ-law bytes by nearest-codeword lookup
    against the bridge's own decode table (exact inverse, no audioop dependency)."""
    pcm = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    idx = np.abs(_ULAW_TABLE[None, :] - pcm[:, None]).argmin(axis=1)
    return idx.astype(np.uint8).tobytes()


def _media_frames(seconds: float = 4.0, sr: int = 8000, frame: int = 160):
    """Yield base64 μ-law frames (20 ms each) of 8 kHz tone — enough audio to fill
    at least one 3 s detection window."""
    rng = np.random.default_rng(0)
    pcm = (0.2 * rng.standard_normal(int(sr * seconds))).astype(np.float32)
    for i in range(0, len(pcm) - frame, frame):
        yield base64.b64encode(ulaw_encode(pcm[i : i + frame])).decode()


def test_twilio_stream_emits_detection():
    client = TestClient(app)
    with client.websocket_connect("/twilio/stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(
            json.dumps(
                {
                    "event": "start",
                    "start": {"streamSid": STREAM_SID, "callSid": CALL_SID},
                }
            )
        )
        for payload in _media_frames():
            ws.send_text(
                json.dumps(
                    {"event": "media", "media": {"payload": payload}, "streamSid": STREAM_SID}
                )
            )
        # After ~4 s of audio, at least one 3 s window must have been scored.
        evt = ws.receive_json()
        ws.send_text(json.dumps({"event": "stop", "streamSid": STREAM_SID}))

    detections = [evt]

    assert detections, "bridge produced no detection event"
    d = detections[0]
    assert d["event"] == "detection"
    assert d["streamSid"] == STREAM_SID
    assert d["label"] in {"real", "fake"}
    assert 0.0 <= d["confidence"] <= 1.0
    assert "window_id" in d
