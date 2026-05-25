"""Pydantic models for all API request and response payloads."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ModelType(StrEnum):
    classical = "classical"
    dsfnet = "dsfnet"
    wav2vec2 = "wav2vec2"


class DetectionResult(BaseModel):
    label: str = Field(..., description="'real' or 'fake'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: ModelType
    latency_ms: float
    audio_hash: str = Field(..., description="SHA-256 of uploaded audio")


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    engine: str = Field(default="xtts_v2", description="Synthesis engine identifier")
    language: str = Field(default="en")


class SynthesisResult(BaseModel):
    audio_url: str
    watermark_id: str | None = None
    synthesis_latency_ms: float


class ForensicReportRequest(BaseModel):
    audio_hash: str
    analyst_name: str = Field(default="Automated System")
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
    eer_estimate: float | None = None
