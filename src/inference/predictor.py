"""Real-time acoustic speech deepfake predictor and forensics inference engine."""

import io
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa

from src.dsp.preprocessor import AudioPreprocessor
from src.dsp.feature_extractor import AudioFeatureExtractor
from src.models.spectral_resnet import SpectralResNet
from src.models.wav2vec_detector import Wav2VecDetector

logger = logging.getLogger("spectroguard.inference")


class AudioForensicsPredictor:
    """Production inference engine for acoustic speech deepfake detection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "spectral_resnet",
        threshold: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the AudioForensicsPredictor.

        Args:
            model_path: Optional path to serialized checkpoint (.pt / .pth).
            model_type: Either 'spectral_resnet' or 'wav2vec'.
            threshold: Calibrated EER decision boundary for classifying as Spoof.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_type = model_type.lower()
        self.threshold = float(threshold)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate DSP preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor(
            target_sr=16000,
            target_lufs=-23.0,
            vad_top_db=25.0,
            target_duration=3.0,
        )
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )

        # Initialize neural network architecture
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Initialized {self.model_type} on {self.device} with threshold {self.threshold:.4f}")

    def _load_model(self, checkpoint_path: Optional[str]) -> nn.Module:
        """Instantiate and optionally load model weights.

        Args:
            checkpoint_path: Optional path to checkpoint file.

        Returns:
            nn.Module ready for evaluation.
        """
        if self.model_type == "wav2vec":
            model: nn.Module = Wav2VecDetector(pretrained_name="facebook/wav2vec2-base")
        else:
            model = SpectralResNet(in_channels=1, num_classes=1, dropout=0.3)

        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

        return model

    def _read_audio(self, audio_input: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
        """Decode audio from file path or binary buffer into raw floating-point array.

        Args:
            audio_input: Filepath, bytes, or in-memory BytesIO buffer.

        Returns:
            Tuple of (audio_waveform, sample_rate).
        """
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found at: {audio_path}")
            wav, sr = sf.read(str(audio_path), dtype="float32")
        elif isinstance(audio_input, bytes):
            buffer = io.BytesIO(audio_input)
            wav, sr = sf.read(buffer, dtype="float32")
        elif isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            wav, sr = sf.read(audio_input, dtype="float32")
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        return wav, sr

    @torch.no_grad()
    def predict(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        calibrated_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run acoustic forensics pipeline on an input audio recording.

        Args:
            audio_input: File path or raw audio bytes (.wav, .mp3, etc.).
            calibrated_threshold: Optional override for decision boundary.

        Returns:
            Dictionary containing:
                - verdict: "SPOOF" or "BONAFIDE"
                - spoof_probability: Float between 0.0 and 1.0
                - confidence: Confidence percentage [0-100]
                - threshold_used: Operating threshold
                - latency_ms: Execution time in milliseconds
                - telemetry: Audio properties and acoustic metadata
        """
        t0 = time.perf_counter()

        # Step 1: Decode raw audio
        raw_wav, orig_sr = self._read_audio(audio_input)
        orig_duration = len(raw_wav) / float(orig_sr) if orig_sr > 0 else 0.0

        # Step 2: DSP Standardization Pipeline
        # Convert to mono, resample to 16kHz, VAD silence trimming, RMS loudness normalization
        processed_wav = self.preprocessor.process(raw_wav, orig_sr)
        final_rms_dbfs = self.preprocessor.calculate_rms_dbfs(processed_wav)

        # Step 3: Feature Extraction & Tensor Construction
        if self.model_type == "spectral_resnet":
            log_mel = self.feature_extractor.extract_log_mel(processed_wav)
            # Shape: (1, 1, n_mels, time_frames)
            input_tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            # Wav2Vec expects (1, num_samples)
            input_tensor = torch.from_numpy(processed_wav).unsqueeze(0).to(self.device)

        # Step 4: Model Forward Pass
        logits = self.model(input_tensor).view(-1)
        prob = torch.sigmoid(logits).item()

        # Step 5: Decision Logic & Telemetry
        operating_thresh = (
            calibrated_threshold if calibrated_threshold is not None else self.threshold
        )
        is_spoof = prob >= operating_thresh
        verdict = "SPOOF" if is_spoof else "BONAFIDE"

        # Confidence: Distance from decision boundary or directional probability
        if is_spoof:
            confidence = prob * 100.0
        else:
            confidence = (1.0 - prob) * 100.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": verdict,
            "spoof_probability": round(prob, 4),
            "confidence": round(confidence, 2),
            "calibrated_threshold": operating_thresh,
            "latency_ms": round(latency_ms, 2),
            "telemetry": {
                "original_sample_rate": orig_sr,
                "target_sample_rate": self.preprocessor.target_sr,
                "original_duration_seconds": round(orig_duration, 3),
                "processed_duration_seconds": round(len(processed_wav) / self.preprocessor.target_sr, 3),
                "processed_rms_dbfs": round(final_rms_dbfs, 2),
                "model_architecture": self.model_type,
                "device": str(self.device),
            },
        }
