"""Unit tests for SpectralResNet and Wav2VecDetector forensics neural architectures."""

import pytest
import torch

from src.models.spectral_resnet import SpectralResNet
from src.models.wav2vec_detector import Wav2VecDetector


class TestSpectralResNet:
    """Test suite for 2D Spectral Residual CNN baseline."""

    def test_spectral_resnet_forward_pass(self, dummy_spectrogram):
        model = SpectralResNet(in_channels=1, num_classes=1, dropout=0.2)
        model.eval()

        with torch.no_grad():
            logits = model(dummy_spectrogram)

        # Batch size is 2
        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()

    def test_spectral_resnet_feature_extraction(self, dummy_spectrogram):
        model = SpectralResNet(in_channels=1, num_classes=1)
        model.eval()

        with torch.no_grad():
            features = model.extract_features(dummy_spectrogram)

        assert features.shape == (2, 512)

    def test_spectral_resnet_backward_gradients(self, dummy_spectrogram):
        model = SpectralResNet(in_channels=1, num_classes=1)
        model.train()

        logits = model(dummy_spectrogram)
        loss = logits.sum()
        loss.backward()

        # Verify gradients exist in initial conv layer
        assert model.conv1.weight.grad is not None
        assert not torch.isnan(model.conv1.weight.grad).any()


class TestWav2VecDetector:
    """Test suite for Wav2Vec2 fine-tuned classification backbone."""

    def test_wav2vec_detector_instantiation_and_freezing(self):
        model = Wav2VecDetector(pretrained_name="facebook/wav2vec2-base", freeze_encoder=True)

        # Confirm CNN encoder parameters are frozen
        if hasattr(model.backbone, "feature_extractor"):
            for param in model.backbone.feature_extractor.parameters():
                assert not param.requires_grad

        # Confirm classification head parameters are trainable
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_wav2vec_detector_forward_pass(self):
        model = Wav2VecDetector(pretrained_name="facebook/wav2vec2-base", freeze_encoder=True)
        model.eval()

        # Batch of 2 clips with 16000 samples (1 second at 16kHz)
        waveform = torch.randn(2, 16000, dtype=torch.float32)

        with torch.no_grad():
            logits = model(waveform)

        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all()
