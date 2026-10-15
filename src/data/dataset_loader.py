"""PyTorch dataset, SpecAugment, and dataloader pipeline for acoustic deepfake training."""

from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T

from src.dsp.feature_extractor import AcousticFeatureExtractor
from src.dsp.preprocessor import AudioPreprocessor


class SpecAugment:
    """Acoustic SpecAugment transform: Frequency masking, time masking, and additive micro-noise."""

    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 24,
        gaussian_noise_std: float = 0.005,
    ):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.gaussian_noise_std = gaussian_noise_std

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Applies frequency and time masks to the 2D spectrogram tensor (1, n_mels, time)."""
        augmented = self.freq_mask(spec)
        augmented = self.time_mask(augmented)

        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(augmented) * self.gaussian_noise_std
            augmented = augmented + noise

        return augmented


class AudioDeepfakeDataset(Dataset):
    """PyTorch Dataset loading real (label 0) and synthetic (label 1) speech samples."""

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], int]],
        feature_type: str = "spectrogram",  # "spectrogram" or "waveform"
        augment: bool = False,
        preprocessor: Optional[AudioPreprocessor] = None,
        feature_extractor: Optional[AcousticFeatureExtractor] = None,
    ):
        self.samples = [(Path(p), lbl) for p, lbl in samples]
        self.feature_type = feature_type
        self.augment = augment
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.feature_extractor = feature_extractor or AcousticFeatureExtractor()
        self.spec_augment = SpecAugment() if augment else None

    @classmethod
    def from_directory(
        cls,
        root_dir: Union[str, Path],
        feature_type: str = "spectrogram",
        augment: bool = False,
    ) -> "AudioDeepfakeDataset":
        """Discovers audio files from standard folder structure: root_dir/{real,synthetic}/*.wav"""
        root = Path(root_dir)
        samples: List[Tuple[Path, int]] = []

        # Find real samples (Label 0)
        real_dirs = [root / "real", root / "bonafide"]
        for r_dir in real_dirs:
            if r_dir.exists():
                for ext in ("*.wav", "*.mp3", "*.flac"):
                    for file_path in r_dir.glob(ext):
                        samples.append((file_path, 0))

        # Find synthetic samples (Label 1)
        fake_dirs = [root / "synthetic", root / "spoof", root / "fake"]
        for f_dir in fake_dirs:
            if f_dir.exists():
                for ext in ("*.wav", "*.mp3", "*.flac"):
                    for file_path in f_dir.glob(ext):
                        samples.append((file_path, 1))

        return cls(samples=samples, feature_type=feature_type, augment=augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path, label = self.samples[idx]

        # 1. Acoustic Preprocessing (Resample, VAD, RMS Normalization, Padding)
        waveform, _ = self.preprocessor.process(path)

        # 2. Additive Gaussian noise on raw waveform if augmenting
        if self.augment and random.random() < 0.5:
            noise = torch.randn_like(waveform) * 0.003
            waveform = waveform + noise

        # 3. Feature Extraction
        if self.feature_type == "waveform":
            # For Wav2Vec2 models requiring 1D raw waveform: shape (48000,)
            feature_tensor = waveform.squeeze(0)
        else:
            # For 2D Spectral CNNs: shape (1, 128, 94)
            feature_tensor = self.feature_extractor.extract_log_mel_spectrogram(waveform)
            if self.augment and self.spec_augment:
                feature_tensor = self.spec_augment(feature_tensor)

        target_tensor = torch.tensor(label, dtype=torch.float32)
        return feature_tensor, target_tensor, str(path)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Builds standard PyTorch DataLoaders with shuffle enabled for training."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
