"""Command-line benchmark runner to evaluate speech deepfake models and compute EER."""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader import AudioDeepfakeDataset
from src.models.spectral_resnet import SpectralResNet
from src.models.wav2vec_detector import Wav2VecDetector
from src.evaluation.metrics import compute_forensic_metrics
from src.evaluation.visualizer import plot_roc_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spectroguard.eval_cli")


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark SpectroGuard models and compute exact ASVspoof Equal Error Rate (EER)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="spectral_resnet",
        choices=["spectral_resnet", "wav2vec"],
        help="Model architecture backbone.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample_audio",
        help="Path to evaluation dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save benchmark reports and visual ROC plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation benchmark."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating {args.model_type} on device: {device}")

    # Build dataset
    return_spectrogram = (args.model_type == "spectral_resnet")
    dataset = AudioDeepfakeDataset(
        data_dir=data_dir,
        return_spectrogram=return_spectrogram,
        apply_specaugment=False,
    )

    if len(dataset) == 0:
        logger.error(f"No audio files found in {data_dir}. Generate samples first.")
        return

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Initialize model
    if args.model_type == "spectral_resnet":
        model = SpectralResNet(in_channels=1, num_classes=1)
    else:
        model = Wav2VecDetector(pretrained_name="facebook/wav2vec2-base")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        logger.info(f"Loading weights from {checkpoint_path}...")
        chk = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in chk:
            model.load_state_dict(chk["model_state_dict"])
        else:
            model.load_state_dict(chk)
    else:
        logger.warning(f"Checkpoint {checkpoint_path} not found. Running benchmark with initialized weights.")

    model.to(device)
    model.eval()

    all_labels = []
    all_scores = []

    logger.info("Computing predictions across evaluation dataset...")
    with torch.no_grad():
        for batch in data_loader:
            if return_spectrogram:
                inputs = batch["spectrogram"].to(device)
            else:
                inputs = batch["waveform"].to(device)

            labels = batch["label"].cpu().long().tolist()
            logits = model(inputs).view(-1)
            probs = torch.sigmoid(logits).cpu().tolist()

            all_labels.extend(labels)
            all_scores.extend(probs)

    # Compute forensic metrics
    metrics = compute_forensic_metrics(all_labels, all_scores)

    # Generate plots
    roc_plot_path = output_dir / "roc_curve.png"
    cm_plot_path = output_dir / "confusion_matrix.png"

    plot_roc_curve(all_labels, all_scores, save_path=str(roc_plot_path))
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=str(cm_plot_path))

    # Print benchmark report
    print("\n" + "=" * 65)
    print("      SPECTROGUARD-AUDIO: ACOUSTIC FORENSICS BENCHMARK REPORT      ")
    print("=" * 65)
    print(f" Total Evaluation Samples    : {len(all_labels)}")
    print(f" Model Architecture          : {args.model_type}")
    print(f" Checkpoint Source           : {args.checkpoint}")
    print("-" * 65)
    print(f" Equal Error Rate (EER)      : {metrics['eer_percent']:.2f}% (ASVspoof standard)")
    print(f" Calibrated EER Threshold    : {metrics['calibrated_threshold']:.4f}")
    print(f" Receiver Operating AUC      : {metrics['roc_auc']:.4f}")
    print(f" Overall Accuracy            : {metrics['accuracy']:.2f}%")
    print(f" Precision (Spoof)           : {metrics['precision']:.2f}%")
    print(f" Recall / Sensitivity        : {metrics['recall']:.2f}%")
    print(f" Specificity (Bonafide)      : {metrics['specificity']:.2f}%")
    print(f" F1-Score                    : {metrics['f1_score']:.4f}")
    print("-" * 65)
    cm = metrics["confusion_matrix"]
    print(" Confusion Matrix Breakdown:")
    print(f"   True Bonafides Identified (TN)   : {cm['true_negative']}")
    print(f"   False Alarms on Bonafide (FP)    : {cm['false_positive']}")
    print(f"   Synthetic Clones Detected (TP)   : {cm['true_positive']}")
    print(f"   Missed Synthetic Clones (FN)     : {cm['false_negative']}")
    print("-" * 65)
    print(f" Visual ROC curve saved to       : {roc_plot_path}")
    print(f" Confusion matrix saved to       : {cm_plot_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
