"""Integration tests for SpectroGuard FastAPI REST inference endpoints."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.inference.api import app, predictor
from src.inference.predictor import AudioForensicsPredictor


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized predictor instance."""
    # Ensure predictor is initialized before tests run
    import src.inference.api as api_mod
    if api_mod.predictor is None:
        api_mod.predictor = AudioForensicsPredictor(
            model_type="spectral_resnet",
            threshold=0.50,
            device="cpu",
        )

    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Verify GET /api/v1/health returns healthy status and service info."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "SpectroGuard-Audio"
    assert "model_type" in payload
    assert "device" in payload


def test_detect_endpoint_valid_wav(client, dummy_wav_file: Path):
    """Verify POST /api/v1/detect handles valid audio and returns forensic verdict."""
    with open(dummy_wav_file, "rb") as f:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("sample.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    assert payload["verdict"] in ["BONAFIDE", "SPOOF"]
    assert 0.0 <= payload["spoof_probability"] <= 1.0
    assert 0.0 <= payload["confidence"] <= 100.0
    assert payload["latency_ms"] > 0.0

    telemetry = payload["telemetry"]
    assert telemetry["target_sample_rate"] == 16000
    assert telemetry["processed_duration_seconds"] == 3.0


def test_detect_endpoint_unsupported_extension(client):
    """Verify POST /api/v1/detect rejects unsupported file types with 400."""
    fake_txt_payload = io.BytesIO(b"this is not audio")
    response = client.post(
        "/api/v1/detect",
        files={"file": ("malicious.txt", fake_txt_payload, "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported audio extension" in response.json()["detail"]


def test_detect_endpoint_empty_file(client):
    """Verify POST /api/v1/detect rejects 0-byte audio file with 400."""
    empty_payload = io.BytesIO(b"")
    response = client.post(
        "/api/v1/detect",
        files={"file": ("empty.wav", empty_payload, "audio/wav")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
