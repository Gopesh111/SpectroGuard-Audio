"""Evaluation and benchmarking module for acoustic deepfake detection."""

from src.evaluation.metrics import (
    compute_eer,
    compute_forensic_metrics,
    calculate_roc_curve,
)
from src.evaluation.visualizer import (
    plot_roc_curve,
    plot_spectrogram_comparison,
    plot_confusion_matrix,
)

__all__ = [
    "compute_eer",
    "compute_forensic_metrics",
    "calculate_roc_curve",
    "plot_roc_curve",
    "plot_spectrogram_comparison",
    "plot_confusion_matrix",
]
