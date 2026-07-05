"""Pydantic models for all API request and response payloads."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ModelType(StrEnum):
    classical = "classical"
    dsfnet = "dsfnet"
    dsfnet_v2 = "dsfnet_v2"
    aasist = "aasist"
    wav2vec2 = "wav2vec2"
    wavlm_base_plus = "wavlm_base_plus"
    wavlm_large = "wavlm_large"
    wav2vec2_large = "wav2vec2_large"
    xls_r = "xls_r"
    xls_r_aasist = "xls_r_aasist"
    wav2vec2_spoof = "wav2vec2_spoof"


class AttributionSegment(BaseModel):
    start_s: float = Field(..., description="Segment start time in seconds")
    end_s: float = Field(..., description="Segment end time in seconds")
    importance: float = Field(..., ge=0.0, le=1.0, description="Normalised importance score")


class ExplanationResult(BaseModel):
    method: str = Field(..., description="Attribution method used")
    baseline: str
    target_class: int
    frame_duration_ms: int
    attribution_frames: list[float] = Field(
        ..., description="Per-frame importance (10ms bins, normalised 0–1)"
    )
    top_segments: list[AttributionSegment] = Field(..., description="Top suspicious time windows")
    narrative: str | None = Field(
        None,
        description=(
            "Plain-language forensic summary of the verdict, generated from the "
            "detector's own numbers by an LLM (null if unavailable)."
        ),
    )


class DetectionResult(BaseModel):
    label: str = Field(..., description="'real' or 'fake'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: ModelType
    latency_ms: float
    audio_hash: str = Field(..., description="SHA-256 of uploaded audio")
    windows_analyzed: int = Field(
        1,
        description=(
            "Number of model passes — always 1: the clip is scored in a single "
            "full-clip pass from its start (sliding windows misclassify)"
        ),
    )
    seconds_analyzed: float | None = Field(
        None,
        description=(
            "How many seconds of the clip the verdict covers — min(duration, "
            "VG_SCORE_SECONDS). Less than the duration means the tail was not scored."
        ),
    )
    explanation: ExplanationResult | None = Field(
        None, description="Integrated Gradients attribution (only when explain=true)"
    )


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    engine: str = Field(default="kokoro", description="Synthesis engine identifier")
    language: str = Field(default="en")
    voice: str = Field(default="af_heart", description="Kokoro voice id")


class SynthesisResult(BaseModel):
    audio_url: str
    watermark_id: str | None = None
    synthesis_latency_ms: float
    engine: str = "kokoro"
    # Cryptographic C2PA provenance (signed manifest) embedded in the output, in
    # addition to the spectral watermark. False if the c2pa runtime is unavailable.
    c2pa_signed: bool = False


class SynthesisEngineInfo(BaseModel):
    name: str
    label: str
    requires_reference: bool
    available: bool
    preset_voices: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    description: str = ""


class WatermarkVerifyResult(BaseModel):
    """Provenance verification for an uploaded audio file."""

    # Spectral watermark — only checkable when the client supplies the
    # watermark_id returned by /synthesize (the mark is keyed by it).
    spectral_checked: bool = Field(
        False, description="True when a watermark_id was supplied and the spectral check ran"
    )
    spectral_detected: bool = False
    spectral_correlation: float | None = Field(
        None, description="Normalised cross-correlation against the keyed carrier"
    )
    # C2PA manifest (cryptographic provenance), independent of watermark_id.
    c2pa_has_manifest: bool = False
    c2pa_validation_state: str | None = None
    c2pa_ai_generated: bool | None = None
    c2pa_software_agent: str | None = None
    verdict: str = Field(
        "unknown",
        description="'voiceguard-generated' | 'ai-generated' | 'no-provenance-found' | 'unknown'",
    )


class ForensicReportRequest(BaseModel):
    audio_hash: str
    analyst_name: str = Field(default="Automated System")
    detection_result: dict = Field(
        default_factory=dict,
        description=(
            "DEPRECATED and IGNORED — the report verdict now comes from VoiceGuard's "
            "server-side detection record for audio_hash (a client value cannot forge it)."
        ),
    )
    include_gradcam: bool = True
    include_shap: bool = True


class ForensicReportResult(BaseModel):
    report_url: str
    chain_of_custody_hash: str


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]


class StreamDetectionEvent(BaseModel):
    """Real-time streaming detection event."""

    timestamp_ms: float
    window_id: int
    label: str
    confidence: float
    model: str | None = Field(
        None, description="Which detector scored the window: xls_r_aasist | classical | stub"
    )
    final: bool = Field(
        False,
        description=(
            "True when this verdict covers the full scoring cap (VG_WS_SCORE_SECONDS) "
            "— it is the session's last scored verdict; later audio is not analyzed"
        ),
    )
    eer_estimate: float | None = None
