"""SHA-256 chain-of-custody log per NIST SP 800-86."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class CustodyEvent:
    event_type: str
    actor: str
    audio_hash: str
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "actor": self.actor,
            "audio_hash": self.audio_hash,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


class ChainOfCustody:
    """Append-only SHA-256 chain of custody log."""

    def __init__(self) -> None:
        self._events: list[CustodyEvent] = []
        self._chain_hash = "0" * 64

    def add_event(self, event_type: str, actor: str, audio_hash: str, notes: str = "") -> str:
        """Append event, return updated chain hash."""
        evt = CustodyEvent(
            event_type=event_type,
            actor=actor,
            audio_hash=audio_hash,
            notes=notes,
        )
        self._events.append(evt)
        payload = json.dumps(evt.to_dict(), sort_keys=True) + self._chain_hash
        self._chain_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self._chain_hash

    @property
    def chain_hash(self) -> str:
        return self._chain_hash

    def to_dict(self) -> dict:
        return {
            "chain_hash": self._chain_hash,
            "events": [e.to_dict() for e in self._events],
        }

    def verify(self) -> bool:
        """Re-derive chain hash from scratch; returns True if consistent."""
        running = "0" * 64
        for evt in self._events:
            payload = json.dumps(evt.to_dict(), sort_keys=True) + running
            running = hashlib.sha256(payload.encode()).hexdigest()
        return running == self._chain_hash
