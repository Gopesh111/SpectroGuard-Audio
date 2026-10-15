"""Synthetic and Bonafide Audio Sample Dataset Generator.

Generates local test audio clips for Indian English and Hindi speech forensics.
Uses `edge-tts` (100% free, zero API key) when network is available, and provides
an algorithmic acoustic formant vocoder simulation fallback for full offline execution.
"""

import os
import sys
import asyncio
from pathlib import Path
import csv
import logging
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sample_dataset_generator")

SAMPLE_RATE = 16000
DURATION = 3.0  # seconds

# Sample scripts in Indian English & Hindi
SAMPLE_TEXTS = [
    ("en_in_01", "Your account security verification code is four eight two nine.", "en-IN-NeerjaNeural"),
    ("en_in_02", "Please confirm your identity by repeating this pass phrase clearly.", "en-IN-PrabhatNeural"),
    ("hi_in_01", "Aapka bank khata surakshit hai, kripya apna OTP kisi ko na dein.", "hi-IN-MadhurNeural"),
    ("hi_in_02", "Dilli police ki taraf se cyber suraksha jagrukta sandesh.", "hi-IN-SwaraNeural"),
    ("en_in_03", "Acoustic forensics pipeline analyzed the harmonic spectral density.", "en-IN-NeerjaNeural"),
    ("hi_in_03", "Yeh ek aawaz ki jaanch karne wala machine learning model hai.", "hi-IN-MadhurNeural"),
]


def generate_algorithmic_bonafide(duration_sec: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate realistic synthetic bonafide speech formant waveform (acoustic model).

    Simulates human glottal excitation pulse (F0 ~ 125 Hz) filtered through
    vocal tract formants (F1 ~ 500 Hz, F2 ~ 1500 Hz, F3 ~ 2500 Hz) with natural
    pitch vibrato and subtle micro-tremor.
    """
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # Glottal fundamental frequency (F0) with natural 4Hz vibrato and smooth inflection
    f0 = 130.0 + 5.0 * np.sin(2 * np.pi * 4.0 * t) + 15.0 * np.cos(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Rich harmonic spectrum (Rosenberg glottal pulse approximation)
    glottal = (
        0.5 * np.sin(phase)
        + 0.25 * np.sin(2 * phase)
        + 0.12 * np.sin(3 * phase)
        + 0.06 * np.sin(4 * phase)
        + 0.03 * np.sin(5 * phase)
    )

    # Human vocal tract formants (Vowel /a/ and /i/ transitions)
    formant1 = 0.4 * np.sin(2 * np.pi * 600 * t)
    formant2 = 0.25 * np.sin(2 * np.pi * 1800 * t)
    formant3 = 0.1 * np.sin(2 * np.pi * 2700 * t)

    speech = glottal * (1.0 + formant1 + formant2 + formant3)

    # Natural speech amplitude envelope (syllable modulation)
    syllables = np.abs(np.sin(2 * np.pi * 2.5 * t)) ** 1.5
    speech = speech * syllables

    # Soft ambient room noise (-45 dB)
    noise = np.random.normal(0, 0.005, total_samples)
    speech += noise

    # Normalize to -23 dBFS RMS
    rms = np.sqrt(np.mean(speech ** 2) + 1e-9)
    target_rms = 10.0 ** (-23.0 / 20.0)
    speech = speech * (target_rms / (rms + 1e-9))

    return speech.astype(np.float32)


def generate_algorithmic_vocoder_spoof(duration_sec: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate audio simulating neural vocoder artifacts (spoof).

    Includes characteristic TTS vocoder defects:
      1. High-frequency checkerboard / robotic harmonics (> 4000 Hz)
      2. Perfect mathematical pitch with zero micro-prosodic jitter
      3. Unnatural phase discontinuities and spectral flattening
    """
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # Completely rigid synthetic F0 (no natural human pitch tremor)
    f0 = 140.0
    phase = 2 * np.pi * f0 * t

    # Rectangular / metallic vocoder harmonic buzz
    synthetic_buzz = (
        0.4 * np.sin(phase)
        + 0.3 * np.sin(2 * phase)
        + 0.25 * np.sin(3 * phase)
        + 0.2 * np.sin(4 * phase)
        + 0.18 * np.sin(5 * phase)
        + 0.15 * np.sin(6 * phase)
        + 0.12 * np.sin(7 * phase)
        + 0.10 * np.sin(8 * phase)
    )

    # High frequency vocoder artifact (typical of HiFi-GAN / MelGAN upsampling aliasing)
    vocoder_artifact = 0.08 * np.sin(2 * np.pi * 6400 * t) + 0.05 * np.sin(2 * np.pi * 7200 * t)

    synthetic_audio = synthetic_buzz + vocoder_artifact

    # Artificial step-like envelope typical of TTS frame concatenations
    frame_steps = np.repeat(np.random.uniform(0.3, 1.0, int(duration_sec * 50)), int(sr / 50))
    if len(frame_steps) < total_samples:
        frame_steps = np.pad(frame_steps, (0, total_samples - len(frame_steps)), mode="edge")
    else:
        frame_steps = frame_steps[:total_samples]

    synthetic_audio = synthetic_audio * frame_steps

    # Normalize to -23 dBFS
    rms = np.sqrt(np.mean(synthetic_audio ** 2) + 1e-9)
    target_rms = 10.0 ** (-23.0 / 20.0)
    synthetic_audio = synthetic_audio * (target_rms / (rms + 1e-9))

    return synthetic_audio.astype(np.float32)


async def synthesize_edge_tts(text: str, voice: str, output_path: Path) -> bool:
    """Synthesize speech using edge-tts (free Microsoft Edge neural TTS).

    Args:
        text: Text to synthesize.
        voice: Voice identifier (e.g. 'en-IN-NeerjaNeural').
        output_path: Filepath where synthesized audio should be saved.

    Returns:
        True if synthesis succeeded, False otherwise.
    """
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except Exception as exc:
        logger.warning(f"edge-tts unavailable or failed ({exc}). Falling back to algorithmic vocoder synthesis.")
        return False


async def create_sample_dataset(output_dir: str = "data/sample_audio") -> None:
    """Generate balanced sample audio dataset with metadata CSV.

    Args:
        output_dir: Directory where samples will be organized.
    """
    base_dir = Path(output_dir)
    real_dir = base_dir / "real"
    synth_dir = base_dir / "synthetic"
    real_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = base_dir / "metadata.csv"
    records = []

    logger.info("Generating bonafide (genuine human acoustic model) audio samples...")
    for idx, (sample_id, _, _) in enumerate(SAMPLE_TEXTS):
        filename = f"bonafide_{sample_id}.wav"
        filepath = real_dir / filename
        audio = generate_algorithmic_bonafide(duration_sec=DURATION, sr=SAMPLE_RATE)
        sf.write(str(filepath), audio, SAMPLE_RATE)
        records.append({
            "filepath": str(filepath.relative_to(base_dir.parent.parent)),
            "label": "0",  # 0 = Bonafide
            "duration": DURATION,
            "sample_rate": SAMPLE_RATE,
            "category": "bonafide",
        })

    logger.info("Generating synthetic (AI voice clone / TTS) audio samples...")
    for idx, (sample_id, text, voice) in enumerate(SAMPLE_TEXTS):
        filename = f"synthetic_{sample_id}.wav"
        filepath = synth_dir / filename

        # Try free edge-tts first
        success = await synthesize_edge_tts(text, voice, filepath)
        if not success or not filepath.exists():
            # Use algorithmic vocoder simulator
            audio = generate_algorithmic_vocoder_spoof(duration_sec=DURATION, sr=SAMPLE_RATE)
            sf.write(str(filepath), audio, SAMPLE_RATE)

        records.append({
            "filepath": str(filepath.relative_to(base_dir.parent.parent)),
            "label": "1",  # 1 = Spoof
            "duration": DURATION,
            "sample_rate": SAMPLE_RATE,
            "category": "spoof",
        })

    # Write manifest CSV
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "duration", "sample_rate", "category"])
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Successfully generated dataset with {len(records)} samples at {base_dir}")
    logger.info(f"Manifest written to {metadata_path}")


if __name__ == "__main__":
    asyncio.run(create_sample_dataset())
