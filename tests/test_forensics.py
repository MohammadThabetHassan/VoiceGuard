"""Tests for chain of custody and PDF report generation."""

from __future__ import annotations

import pytest

from voiceguard.forensics.chain_of_custody import ChainOfCustody, CustodyEvent
from voiceguard.forensics.pdf_report import generate_report

# ── Chain of custody ───────────────────────────────────────────────────────────


def test_initial_chain_hash_is_zeros():
    coc = ChainOfCustody()
    assert coc.chain_hash == "0" * 64


def test_add_event_changes_hash():
    coc = ChainOfCustody()
    h1 = coc.add_event("upload", "system", "a" * 64)
    assert h1 != "0" * 64
    assert len(h1) == 64


def test_chain_grows():
    coc = ChainOfCustody()
    coc.add_event("upload", "system", "a" * 64)
    h1 = coc.chain_hash
    coc.add_event("analysis", "analyst", "a" * 64)
    h2 = coc.chain_hash
    assert h1 != h2


def test_verify_integrity_passes():
    coc = ChainOfCustody()
    coc.add_event("upload", "system", "a" * 64)
    coc.add_event("analysis", "analyst", "a" * 64)
    assert coc.verify() is True


def test_verify_detects_tampering():
    coc = ChainOfCustody()
    coc.add_event("upload", "system", "a" * 64)
    coc._events[0].actor = "attacker"  # tamper
    assert coc.verify() is False


def test_to_dict_structure():
    coc = ChainOfCustody()
    coc.add_event("upload", "system", "a" * 64, notes="test run")
    d = coc.to_dict()
    assert "chain_hash" in d
    assert "events" in d
    assert len(d["events"]) == 1
    evt = d["events"][0]
    assert evt["event_type"] == "upload"
    assert evt["actor"] == "system"
    assert evt["notes"] == "test run"


def test_custody_event_to_dict():
    evt = CustodyEvent("upload", "system", "a" * 64, timestamp=0.0, notes="")
    d = evt.to_dict()
    assert d["event_type"] == "upload"
    assert d["timestamp"] == 0.0


def test_empty_chain_verify():
    coc = ChainOfCustody()
    assert coc.verify() is True


# ── PDF report ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_chain():
    coc = ChainOfCustody()
    coc.add_event("upload", "system", "a" * 64)
    coc.add_event("analysis", "analyst", "a" * 64)
    return coc.to_dict()


@pytest.fixture
def sample_detection():
    return {
        "label": "fake",
        "confidence": 0.9123,
        "model": "classical",
        "latency_ms": 42.5,
    }


def test_generate_report_returns_bytes(sample_detection, sample_chain):
    pdf = generate_report(
        audio_hash="a" * 64,
        detection_result=sample_detection,
        chain_of_custody=sample_chain,
    )
    assert isinstance(pdf, bytes)
    assert len(pdf) > 1000  # non-trivial PDF


def test_generate_report_is_pdf(sample_detection, sample_chain):
    pdf = generate_report("a" * 64, sample_detection, sample_chain)
    assert pdf[:4] == b"%PDF"


def test_generate_report_writes_file(tmp_path, sample_detection, sample_chain):
    out = tmp_path / "report.pdf"
    pdf = generate_report("a" * 64, sample_detection, sample_chain, output_path=out)
    assert out.exists()
    assert out.stat().st_size == len(pdf)


def test_generate_report_custom_analyst(sample_detection, sample_chain):
    pdf = generate_report(
        "a" * 64,
        sample_detection,
        sample_chain,
        analyst_name="Dr. Smith",
    )
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1000
