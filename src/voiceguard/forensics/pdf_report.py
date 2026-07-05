"""NIST SP 800-86 compliant PDF forensic report generator using ReportLab."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _ts(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def generate_report(
    audio_hash: str,
    detection_result: dict[str, Any],
    chain_of_custody: dict[str, Any],
    analyst_name: str = "Automated System",
    output_path: str | Path | None = None,
    audio_meta: dict[str, Any] | None = None,
    model_meta: dict[str, Any] | None = None,
    narrative: str | None = None,
) -> bytes:
    """Generate a PDF forensic report.

    Args:
        audio_hash: SHA-256 hex digest of the audio evidence.
        detection_result: dict with keys label, confidence, model, latency_ms.
        chain_of_custody: dict from ChainOfCustody.to_dict().
        analyst_name: Name of the reporting analyst.
        output_path: Optional path to write the PDF; bytes always returned.
        audio_meta: Evidence characteristics captured at analysis time —
            duration_s, sample_rate, channels, format, windows_analyzed.
        model_meta: Analysis-tool identity — model key, app version, checkpoint
            SHA-256 (NIST SP 800-86 wants the examination tool pinned down).
        narrative: Optional plain-language AI analysis of the verdict, rendered
            as a decision-support section (clearly labelled as such).

    Returns:
        PDF bytes.
    """
    from xml.sax.saxutils import escape as _xml_escape
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("VoiceGuard Forensic Report", styles["Title"]))
    story.append(Paragraph("NIST SP 800-86 Compliant", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    # Metadata table
    meta = [
        ["Report Generated", _ts(time.time())],
        ["Analyst", analyst_name],
        ["Evidence Hash (SHA-256)", audio_hash],
    ]
    t = Table(meta, colWidths=[5 * cm, 12 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Evidence characteristics (captured at analysis time, before PDPL erasure)
    if audio_meta:
        story.append(Paragraph("Evidence Characteristics", styles["Heading2"]))
        am = [
            ["Duration (s)", str(audio_meta.get("duration_s", "—"))],
            ["Sample Rate (Hz)", str(audio_meta.get("sample_rate", "—"))],
            ["Channels", str(audio_meta.get("channels", "—"))],
            ["Container / Codec", str(audio_meta.get("format", "—"))],
            ["Audio Scored (s)", str(audio_meta.get("seconds_analyzed", "—"))],
            ["Analysis Passes", str(audio_meta.get("windows_analyzed", "—"))],
        ]
        at = Table(am, colWidths=[5 * cm, 12 * cm])
        at.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        story.append(at)
        story.append(Spacer(1, 0.5 * cm))

    # Detection result
    story.append(Paragraph("Detection Result", styles["Heading2"]))
    det_data = [
        ["Attribute", "Value"],
        ["Label", str(detection_result.get("label", "—"))],
        ["Confidence", f"{detection_result.get('confidence', 0):.4f}"],
        ["Model", str(detection_result.get("model", "—"))],
        ["Latency (ms)", f"{detection_result.get('latency_ms', 0):.2f}"],
    ]
    dt = Table(det_data, colWidths=[5 * cm, 12 * cm])
    dt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(dt)
    story.append(Spacer(1, 0.5 * cm))

    # AI forensic analysis — plain-language interpretation of the verdict. Clearly
    # framed as automated decision-support (not independent proof) so the report
    # cannot be read as overstating the finding.
    if narrative:
        story.append(Paragraph("AI Forensic Analysis", styles["Heading2"]))
        story.append(Paragraph(_xml_escape(narrative), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * cm))
        disclaimer = (
            "Generated by an AI language model from the detector's numeric output "
            "(verdict, confidence, and time-segment attribution). Provided as "
            "decision-support to aid interpretation; it is not an independent "
            "determination and does not alter the model verdict above."
        )
        story.append(Paragraph(f"<i>{_xml_escape(disclaimer)}</i>", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

    # Analysis tool identity — pins the exact software + weights that produced
    # the verdict, so the examination is reproducible (NIST SP 800-86 §3.1).
    if model_meta:
        story.append(Paragraph("Analysis Tool", styles["Heading2"]))
        mm = [
            ["Detector", str(model_meta.get("model", "—"))],
            ["VoiceGuard Version", str(model_meta.get("app_version", "—"))],
            ["Checkpoint SHA-256", str(model_meta.get("checkpoint_sha256") or "unavailable")],
        ]
        mt = Table(mm, colWidths=[5 * cm, 12 * cm])
        mt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(mt)
        story.append(Spacer(1, 0.5 * cm))

    # Chain of custody
    story.append(Paragraph("Chain of Custody", styles["Heading2"]))
    story.append(
        Paragraph(f"Chain Hash: {chain_of_custody.get('chain_hash', '—')}", styles["Code"])
    )
    story.append(Spacer(1, 0.3 * cm))
    events = chain_of_custody.get("events", [])
    if events:
        rows = [["Timestamp", "Event", "Actor", "Audio Hash"]]
        for evt in events:
            rows.append(
                [
                    _ts(evt.get("timestamp", 0)),
                    evt.get("event_type", ""),
                    evt.get("actor", ""),
                    evt.get("audio_hash", "")[:16] + "…",
                ]
            )
        ct = Table(rows, colWidths=[4 * cm, 3 * cm, 4 * cm, 6 * cm])
        ct.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(ct)

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path is not None:
        Path(output_path).write_bytes(pdf_bytes)

    return pdf_bytes
