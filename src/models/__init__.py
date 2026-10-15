"""Deep neural network models for acoustic forensics."""

from src.models.spectral_resnet import SpectralResNet
from src.models.wav2vec_detector import Wav2VecDetector

__all__ = [
    "SpectralResNet",
    "Wav2VecDetector",
]
