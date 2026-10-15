"""FastAPI REST service for real-time speech deepfake detection."""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predictor import AudioForensicsPredictor

# Setup logging
logger = logging.getLogger("spectroguard.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global predictor instance
predictor: AudioForensicsPredictor = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize model into memory upon API startup."""
    global predictor
    model_type = os.getenv("MODEL_TYPE", "spectral_resnet")
    model_path = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/best_model.pt")
    calibrated_threshold = float(os.getenv("CALIBRATED_THRESHOLD", "0.50"))
    device = os.getenv("DEVICE", None)

    logger.info(f"Initializing SpectroGuard Forensic Predictor ({model_type})...")
    predictor = AudioForensicsPredictor(
        model_path=model_path,
        model_type=model_type,
        threshold=calibrated_threshold,
        device=device,
    )
    yield
    logger.info("Shutting down SpectroGuard API.")


app = FastAPI(
    title="SpectroGuard-Audio Forensics API",
    description="Acoustic Speech Deepfake, Voice Clone, and Synthetic Audio Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


class TelemetrySchema(BaseModel):
    """Acoustic DSP and hardware telemetry schema."""
    original_sample_rate: int = Field(..., description="Source audio sample rate in Hz")
    target_sample_rate: int = Field(..., description="Standardized sampling frequency (16000 Hz)")
    original_duration_seconds: float = Field(..., description="Raw input duration in seconds")
    processed_duration_seconds: float = Field(..., description="VAD-trimmed and normalized duration")
    processed_rms_dbfs: float = Field(..., description="RMS loudness in dBFS (-23 dBFS standard)")
    model_architecture: str = Field(..., description="Active forensic backbone architecture")
    device: str = Field(..., description="Compute device utilized (cpu or cuda)")


class DetectionResponse(BaseModel):
    """Forensic classification verdict and confidence schema."""
    status: str = Field("success", description="Status string")
    verdict: str = Field(..., description="'SPOOF' for AI clone / synthetic speech, 'BONAFIDE' for authentic human speech")
    confidence: float = Field(..., description="Decision confidence percentage [0 - 100]")
    spoof_probability: float = Field(..., description="Raw model probability of synthetic speech [0.0 - 1.0]")
    calibrated_threshold: float = Field(..., description="EER calibrated decision boundary")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    telemetry: TelemetrySchema


class HealthResponse(BaseModel):
    """Health check diagnostic schema."""
    status: str
    service: str
    model_type: str
    device: str
    version: str


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check() -> HealthResponse:
    """Check service availability, loaded model architecture, and execution device."""
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is still loading.")
    return HealthResponse(
        status="healthy",
        service="SpectroGuard-Audio",
        model_type=predictor.model_type,
        device=str(predictor.device),
        version="1.0.0",
    )


@app.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Forensics"],
    summary="Detect speech deepfake or voice clone in audio file",
)
async def detect_deepfake(
    file: UploadFile = File(..., description="Audio file payload (.wav, .mp3, .ogg, .flac)"),
    threshold: float = Query(
        None,
        description="Optional custom operating threshold override. If omitted, uses EER calibrated threshold.",
        ge=0.0,
        le=1.0,
    ),
) -> DetectionResponse:
    """Analyze an uploaded audio recording for synthetic speech artifacts and vocoder signatures.

    Returns forensic classification verdict, probability, EER threshold, and DSP telemetry.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized.",
        )

    filename = file.filename or "recording.wav"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio extension '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}",
        )

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        result = predictor.predict(audio_bytes, calibrated_threshold=threshold)

        return DetectionResponse(
            status="success",
            verdict=result["verdict"],
            confidence=result["confidence"],
            spoof_probability=result["spoof_probability"],
            calibrated_threshold=result["calibrated_threshold"],
            latency_ms=result["latency_ms"],
            telemetry=TelemetrySchema(**result["telemetry"]),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error during audio deepfake analysis: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process and decode audio: {str(exc)}",
        )
