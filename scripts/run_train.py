"""Command-line interface to train SpectroGuard audio forensics models."""

import argparse
import logging
from pathlib import Path
import yaml
import torch

from src.data.dataset_loader import create_forensics_dataloaders
from src.models.spectral_resnet import SpectralResNet
from src.models.wav2vec_detector import Wav2VecDetector
from src.training.trainer import AudioForensicsTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spectroguard.train_cli")


def parse_args() -> argparse.Namespace:
    """Parse command line training arguments."""
    parser = argparse.ArgumentParser(
        description="Train SpectroGuard-Audio speech forensics deepfake classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML hyperparameter configuration file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="spectral_resnet",
        choices=["spectral_resnet", "wav2vec"],
        help="Architecture backbone to train: 'spectral_resnet' or 'wav2vec'.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample_audio",
        help="Path to dataset containing real/ and synthetic/ audio subdirectories.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model weights.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute training pipeline."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found. Using default configurations.")
        config = {
            "dsp": {"target_sample_rate": 16000, "target_lufs": -23.0, "vad_top_db": 25.0, "duration": 3.0},
            "features": {"n_fft": 1024, "hop_length": 256, "n_mels": 128},
            "training": {"epochs": 20, "learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": 16},
        }

    # Apply CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    batch_size = config["training"].get("batch_size", 16)
    model_type = args.model_type.lower()
    return_spectrogram = (model_type == "spectral_resnet")

    logger.info(f"Loading datasets from {args.data_dir} (return_spectrogram={return_spectrogram})...")
    data_dir = Path(args.data_dir)

    train_loader, val_loader = create_forensics_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
        return_spectrogram=return_spectrogram,
        apply_specaugment=True,
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Building model '{model_type}' on compute device: {device}...")

    if model_type == "spectral_resnet":
        model = SpectralResNet(in_channels=1, num_classes=1, dropout=0.3)
    elif model_type == "wav2vec":
        model = Wav2VecDetector(pretrained_name="facebook/wav2vec2-base", freeze_encoder=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Initialize trainer
    trainer = AudioForensicsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Run training
    results = trainer.train()
    logger.info("Training complete!")
    logger.info(f"Best Validation EER: {results['best_val_eer'] * 100:.2f}% (Epoch {results['best_epoch']})")
    logger.info(f"Model checkpoint saved to: {results['best_model_path']}")


if __name__ == "__main__":
    main()
