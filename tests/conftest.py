"""Shared pytest fixtures for SpectroGuard-Audio test suite."""

import tempfile
from pathlib import Path
from typing import Tuple, Generator
import pytest
import numpy as np
import torch
import soundfile as sf


@pytest.fixture
def sample_rate() -> int:
    """Standard target sample rate (16 kHz)."""
    return 16000


@pytest.fixture
def dummy_audio_16k(sample_rate: int) -> np.ndarray:
    """Generate 3 seconds of 16kHz mono audio containing a 440Hz test tone."""
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Sine wave + small noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.normal(0, 1, len(t))
    return audio.astype(np.float32)


@pytest.fixture
def dummy_stereo_44k() -> Tuple[np.ndarray, int]:
    """Generate 2 seconds of 44.1kHz stereo audio with leading/trailing silence."""
    sr = 44100
    duration = 2.0
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # Left channel: 300Hz, Right channel: 600Hz
    ch1 = 0.4 * np.sin(2 * np.pi * 300 * t)
    ch2 = 0.4 * np.sin(2 * np.pi * 600 * t)
    stereo = np.column_stack([ch1, ch2])

    # Pad with 0.5s silence at start and end
    silence = np.zeros((int(sr * 0.5), 2))
    padded_stereo = np.vstack([silence, stereo, silence])

    return padded_stereo.astype(np.float32), sr


@pytest.fixture
def dummy_spectrogram() -> torch.Tensor:
    """Generate 2D Log-Mel spectrogram tensor of shape (batch=2, channel=1, mels=128, time=188)."""
    return torch.randn(2, 1, 128, 188, dtype=torch.float32)


@pytest.fixture
def dummy_wav_file(dummy_audio_16k: np.ndarray, sample_rate: int, tmp_path: Path) -> Path:
    """Create a temporary .wav audio file on disk for file reading tests."""
    file_path = tmp_path / "test_tone_16k.wav"
    sf.write(str(file_path), dummy_audio_16k, sample_rate)
    return file_path
