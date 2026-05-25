"""Twilio Media Stream WebSocket bridge.

Receives JSON messages from Twilio's Media Stream API, decodes μ-law
encoded audio frames (8kHz, mono), and pipes them through the streaming
VoiceGuard detector.
"""

from __future__ import annotations

import base64
import json
import time
from typing import TYPE_CHECKING

import numpy as np

from voiceguard.voip.stream_processor import StreamProcessor

if TYPE_CHECKING:
    from fastapi import WebSocket


# ITU-T G.711 μ-law decode table (256 entries)
def _build_ulaw_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        u = ~i & 0xFF
        sign = -1 if u & 0x80 else 1
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        magnitude = ((mantissa << 1) + 33) << exponent
        table[i] = sign * (magnitude - 33) / 32767.0
    return table


_ULAW_TABLE = _build_ulaw_table()


def ulaw_decode(data: bytes) -> np.ndarray:
    """Decode μ-law bytes to float32 normalised PCM in [-1, 1]."""
    indices = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    return _ULAW_TABLE[indices]


class TwilioStreamHandler:
    """Handle a single Twilio Media Stream WebSocket session."""

    INPUT_SR = 8000

    def __init__(self) -> None:
        self._processor = StreamProcessor(input_sr=self.INPUT_SR, target_sr=16000)
        self._stream_sid: str = ""
        self._call_sid: str = ""

    async def handle(self, websocket: WebSocket) -> None:
        """Main WebSocket loop — process incoming Twilio JSON messages."""
        from fastapi import WebSocketDisconnect

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                event = msg.get("event", "")

                if event == "start":
                    start = msg.get("start", {})
                    self._stream_sid = start.get("streamSid", "")
                    self._call_sid = start.get("callSid", "")

                elif event == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if not payload:
                        continue
                    pcm_bytes = base64.b64decode(payload)
                    audio = ulaw_decode(pcm_bytes)
                    result = self._processor.push(audio)
                    if result is not None:
                        await websocket.send_json(
                            {
                                "event": "detection",
                                "streamSid": self._stream_sid,
                                "timestamp_ms": time.time() * 1000,
                                **result,
                            }
                        )

                elif event == "stop":
                    break

        except WebSocketDisconnect:
            pass
