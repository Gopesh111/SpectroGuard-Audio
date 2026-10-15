"""Acoustic audio preprocessor: Resampling, VAD silence trimming, and RMS loudness normalization."""

import io
from pathlib import Path
from typing import Dict, Tuple, Union
import librosa
import numpy as np
import soundfile as sf
import torch


class AudioPreprocessor:
    """Preprocesses raw acoustic waveforms into standardized, loudness-normalized 16kHz audio."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_duration_sec: float = 3.0,
        target_rms_dbfs: float = -23.0,
        vad_top_db: int = 25,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_duration_sec = target_duration_sec
        self.target_samples = int(target_sample_rate * target_duration_sec)
        self.target_rms_dbfs = target_rms_dbfs
        self.vad_top_db = vad_top_db

    def load_audio(
        self, audio_source: Union[str, Path, bytes, io.BytesIO]
    ) -> Tuple[np.ndarray, int]:
        """Loads audio from a file path or raw bytes, converting to mono float32."""
        if isinstance(audio_source, (bytes, io.BytesIO)):
            if isinstance(audio_source, bytes):
                audio_source = io.BytesIO(audio_source)
            waveform, sr = sf.read(audio_source, dtype="float32")
        else:
            waveform, sr = sf.read(str(audio_source), dtype="float32")

        # Convert multi-channel to mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        return waveform, sr

    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples waveform to target sampling rate (16,000 Hz)."""
        if orig_sr == self.target_sample_rate:
            return waveform
        return librosa.resample(
            waveform,
            orig_sr=orig_sr,
            target_sr=self.target_sample_rate,
            res_type="soxr_hq",
        )

    def apply_vad_trimming(
        self, waveform: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Trims unvoiced leading, trailing, and inter-speech pauses using energy VAD."""
        orig_duration = len(waveform) / self.target_sample_rate
        non_silent_intervals = librosa.effects.split(waveform, top_db=self.vad_top_db)

        if len(non_silent_intervals) == 0:
            # Entire clip is silence
            return waveform, 0.0

        voiced_parts = [waveform[start:end] for start, end in non_silent_intervals]
        trimmed_waveform = np.concatenate(voiced_parts)
        trimmed_duration = len(trimmed_waveform) / self.target_sample_rate
        trimmed_silence_sec = max(0.0, orig_duration - trimmed_duration)

        return trimmed_waveform, round(trimmed_silence_sec, 3)

    def normalize_loudness(self, waveform: np.ndarray) -> np.ndarray:
        """Scales waveform to target RMS dBFS (-23.0 dBFS) to prevent gain bias."""
        rms = np.sqrt(np.mean(waveform**2))
        if rms < 1e-6:
            # Silent or near-silent signal
            return waveform

        current_dbfs = 20 * np.log10(rms)
        gain_db = self.target_rms_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20.0)

        normalized = waveform * gain_linear
        # Peak limit to prevent digital clipping (-0.5 dB peak headroom)
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = (normalized / max_val) * 0.95

        return normalized

    def standardize_length(self, waveform: np.ndarray) -> np.ndarray:
        """Pads with zeros or slices to achieve exact target length (48,000 samples)."""
        num_samples = len(waveform)
        if num_samples == self.target_samples:
            return waveform
        elif num_samples < self.target_samples:
            padding = self.target_samples - num_samples
            return np.pad(waveform, (0, padding), mode="constant")
        else:
            return waveform[: self.target_samples]

    def process(
        self, audio_source: Union[str, Path, bytes, io.BytesIO]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Executes full preprocessing pipeline and returns (tensor, telemetry_metadata)."""
        raw_wave, orig_sr = self.load_audio(audio_source)
        resampled_wave = self.resample(raw_wave, orig_sr)
        voiced_wave, trimmed_silence_sec = self.apply_vad_trimming(resampled_wave)
        norm_wave = self.normalize_loudness(voiced_wave)
        standard_wave = self.standardize_length(norm_wave)

        tensor_waveform = torch.from_numpy(standard_wave).unsqueeze(0).float()
        metadata = {
            "original_sample_rate": orig_sr,
            "processed_sample_rate": self.target_sample_rate,
            "duration_sec": round(len(standard_wave) / self.target_sample_rate, 2),
            "trimmed_silence_sec": trimmed_silence_sec,
        }

        return tensor_waveform, metadata
