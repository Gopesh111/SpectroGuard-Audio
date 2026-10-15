"""Acoustic feature extractor: Log-Mel Spectrograms and Linear Frequency Cepstral Coefficients (LFCC)."""

from typing import Tuple, Union
import librosa
import numpy as np
import torch
import torchaudio.transforms as T


class AcousticFeatureExtractor:
    """Extracts high-resolution spectral acoustic representations for forensic deepfake modeling."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        n_lfcc: int = 40,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.n_lfcc = n_lfcc

        # TorchAudio Mel Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def extract_log_mel_spectrogram(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Extracts 128-band Log-Mel Spectrogram.

        Input:
            waveform: Tensor of shape (1, num_samples) or (batch, 1, num_samples)
        Output:
            log_mel: Tensor of shape (1, 128, time_frames) or (batch, 1, 128, time_frames)
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Compute Mel power spectrogram
        mel_spec = self.mel_transform(waveform)
        # Convert power to decibels (log-mel)
        log_mel = self.amplitude_to_db(mel_spec)

        # Standardize mean and variance across spectrogram
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        normalized_log_mel = (log_mel - mean) / std

        return normalized_log_mel

    def extract_lfcc(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Extracts Linear Frequency Cepstral Coefficients (LFCC) to isolate vocoder high-frequency artifacts.

        Unlike Mel-scale, Linear filterbanks preserve equal resolution at high frequencies (4kHz - 8kHz)
        where synthetic vocoders (HiFi-GAN, WaveGlow) exhibit phase and spectral discontinuity artifacts.
        """
        if isinstance(waveform, torch.Tensor):
            wave_np = waveform.squeeze().cpu().numpy()
        else:
            wave_np = waveform.squeeze()

        # Compute Short-Time Fourier Transform (STFT)
        stft = np.abs(
            librosa.stft(
                wave_np,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
            )
        )

        # Create linear triangular filterbank
        linear_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_lfcc,
            fmin=self.f_min,
            fmax=self.f_max,
            htk=False,
            norm=None,
        )

        # Filter power spectrum
        power_spec = stft**2
        linear_energies = np.dot(linear_filters, power_spec)
        linear_energies = np.maximum(linear_energies, 1e-10)
        log_linear = np.log10(linear_energies)

        # Discrete Cosine Transform (Type-II) to obtain cepstral coefficients
        from scipy.fftpack import dct
        lfcc = dct(log_linear, type=2, axis=0, norm="ortho")

        return torch.from_numpy(lfcc).unsqueeze(0).float()
