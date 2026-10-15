"""Visualization utilities for audio forensics evaluation.

Renders ROC curves with EER operating points, side-by-side Log-Mel spectrograms
highlighting vocoder artifacts, and confusion matrices.
"""

from typing import Union, List, Dict, Optional
from pathlib import Path
import numpy as np

# Non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import calculate_roc_curve, compute_eer, compute_roc_auc


def plot_roc_curve(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    save_path: str = "roc_curve.png",
    title: str = "SpectroGuard Acoustic Forensics ROC Curve",
) -> str:
    """Generate and save Receiver Operating Characteristic (ROC) plot with EER marker.

    Args:
        labels: Ground truth binary labels (0=bonafide, 1=spoof).
        scores: Model spoof prediction probabilities.
        save_path: Output PNG path.
        title: Chart title.

    Returns:
        Absolute filepath of saved plot.
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    fpr, tpr, _ = calculate_roc_curve(labels_arr, scores_arr)
    auc_score = compute_roc_auc(fpr, tpr)
    eer, eer_thresh = compute_eer(labels_arr, scores_arr)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5)")

    # Mark EER operating point on ROC curve (where FPR == 1 - TPR or FPR == FNR)
    ax.plot(
        eer,
        1.0 - eer,
        marker="o",
        markersize=8,
        color="#DC2626",
        linestyle="None",
        label=f"EER = {eer * 100:.2f}% (Threshold = {eer_thresh:.3f})",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (FPR / False Alarm)", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate (TPR / Hit Rate)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=11)

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)

    return str(out_file.resolve())


def plot_spectrogram_comparison(
    bonafide_mel: np.ndarray,
    spoof_mel: np.ndarray,
    sample_rate: int = 16000,
    save_path: str = "spectrogram_comparison.png",
) -> str:
    """Plot side-by-side Log-Mel Spectrogram comparison between Bonafide and Spoof speech.

    Visualizes acoustic vocoder patterns such as unnatural harmonic cutoff,
    phase discontinuities, and checkerboard artifacts.

    Args:
        bonafide_mel: 2D numpy array of shape (n_mels, time_frames).
        spoof_mel: 2D numpy array of shape (n_mels, time_frames).
        sample_rate: Sampling frequency in Hz (typically 16000).
        save_path: Destination path for PNG image.

    Returns:
        Absolute filepath of saved plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)

    # Bonafide plot
    im1 = ax1.imshow(
        bonafide_mel,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax1.set_title("Bonafide (Genuine Human Voice)", fontsize=13, fontweight="bold", color="#1E293B")
    ax1.set_xlabel("Time Frames", fontsize=11)
    ax1.set_ylabel("Mel Frequency Bins (0 - 128)", fontsize=11)
    fig.colorbar(im1, ax=ax1, format="%+2.0f dB", fraction=0.046, pad=0.04)

    # Spoof plot
    im2 = ax2.imshow(
        spoof_mel,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax2.set_title("Spoof (Neural TTS / Voice Clone)", fontsize=13, fontweight="bold", color="#B91C1C")
    ax2.set_xlabel("Time Frames", fontsize=11)
    fig.colorbar(im2, ax=ax2, format="%+2.0f dB", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Acoustic Forensics: Log-Mel Spectral Artifact Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)

    return str(out_file.resolve())


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    save_path: str = "confusion_matrix.png",
) -> str:
    """Plot formatted 2x2 confusion matrix heatmap.

    Args:
        cm_dict: Dictionary with 'true_positive', 'false_positive',
                 'true_negative', 'false_negative'.
        save_path: Output PNG path.

    Returns:
        Absolute filepath of saved plot.
    """
    tp = cm_dict["true_positive"]
    fp = cm_dict["false_positive"]
    tn = cm_dict["true_negative"]
    fn = cm_dict["false_negative"]

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["True Neg\n(Bonafide)", "False Pos\n(False Alarm)"], ["False Neg\n(Missed Spoof)", "True Pos\n(Caught Spoof)"]]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            label = labels[i][j]
            ax.text(
                j,
                i,
                f"{label}\n{val}",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
                fontweight="bold",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Bonafide (0)", "Pred Spoof (1)"], fontsize=11)
    ax.set_yticklabels(["True Bonafide (0)", "True Spoof (1)"], fontsize=11)
    ax.set_title("Acoustic Forensics Confusion Matrix", fontsize=13, pad=20, fontweight="bold")

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)

    return str(out_file.resolve())
