"""Unit tests for acoustic DSP preprocessing and feature extraction modules."""

import numpy as np
import pytest

from src.dsp.preprocessor import AudioPreprocessor
from src.dsp.feature_extractor import AudioFeatureExtractor


class TestAudioPreprocessor:
    """Test suite for AudioPreprocessor acoustic standardization."""

    def test_to_mono(self, dummy_stereo_44k):
        audio_stereo, sr = dummy_stereo_44k
        preprocessor = AudioPreprocessor()

        mono = preprocessor.to_mono(audio_stereo)
        assert mono.ndim == 1
        assert len(mono) == len(audio_stereo)

    def test_resample(self, dummy_audio_16k):
        preprocessor = AudioPreprocessor(target_sr=16000)
        # Resample from 32000 to 16000
        audio_32k = np.repeat(dummy_audio_16k, 2)
        resampled = preprocessor.resample(audio_32k, orig_sr=32000)

        assert len(resampled) == len(dummy_audio_16k)

    def test_trim_silence(self, dummy_stereo_44k):
        audio_stereo, sr = dummy_stereo_44k
        preprocessor = AudioPreprocessor(target_sr=sr)
        mono = preprocessor.to_mono(audio_stereo)

        trimmed = preprocessor.trim_silence(mono, top_db=20.0)
        # Should be shorter than original because leading/trailing silence was removed
        assert len(trimmed) < len(mono)

    def test_normalize_loudness_rms(self, dummy_audio_16k):
        preprocessor = AudioPreprocessor(target_lufs=-23.0)
        # Scale to very quiet level
        quiet_audio = dummy_audio_16k * 0.001
        normalized = preprocessor.normalize_loudness_rms(quiet_audio, target_dbfs=-23.0)

        measured_dbfs = preprocessor.calculate_rms_dbfs(normalized)
        assert pytest.approx(measured_dbfs, abs=0.5) == -23.0

    def test_enforce_duration(self, sample_rate):
        preprocessor = AudioPreprocessor(target_sr=sample_rate, target_duration=3.0)
        target_len = int(sample_rate * 3.0)

        # Case 1: Short audio (1.5 seconds) -> zero pad
        short_audio = np.ones(int(sample_rate * 1.5), dtype=np.float32)
        padded = preprocessor.enforce_duration(short_audio)
        assert len(padded) == target_len
        assert np.all(padded[int(sample_rate * 1.5):] == 0)

        # Case 2: Long audio (5 seconds) -> truncate
        long_audio = np.ones(int(sample_rate * 5.0), dtype=np.float32)
        truncated = preprocessor.enforce_duration(long_audio)
        assert len(truncated) == target_len

    def test_end_to_end_process(self, dummy_stereo_44k):
        audio_stereo, orig_sr = dummy_stereo_44k
        preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0, target_lufs=-23.0)

        processed = preprocessor.process(audio_stereo, orig_sr=orig_sr)
        assert processed.ndim == 1
        assert len(processed) == 16000 * 3
        assert np.isfinite(processed).all()


class TestAudioFeatureExtractor:
    """Test suite for Log-Mel and LFCC feature extraction."""

    def test_log_mel_shape_and_finite(self, dummy_audio_16k, sample_rate):
        extractor = AudioFeatureExtractor(sample_rate=sample_rate, n_mels=128, hop_length=256)
        mel = extractor.extract_log_mel(dummy_audio_16k)

        # n_mels should be 128
        assert mel.shape[0] == 128
        # time frames should be approximately 3.0 * 16000 / 256 + 1
        expected_frames = 1 + int(len(dummy_audio_16k) / 256)
        assert abs(mel.shape[1] - expected_frames) <= 2
        assert np.isfinite(mel).all()

    def test_lfcc_shape(self, dummy_audio_16k, sample_rate):
        extractor = AudioFeatureExtractor(sample_rate=sample_rate, n_lfcc=40, hop_length=256)
        lfcc = extractor.extract_lfcc(dummy_audio_16k)

        assert lfcc.shape[0] == 40
        assert np.isfinite(lfcc).all()
