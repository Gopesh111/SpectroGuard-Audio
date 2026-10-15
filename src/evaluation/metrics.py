"""Acoustic forensics evaluation metrics and Equal Error Rate (EER) engine.

Adheres strictly to the ASVspoof challenge biometric benchmarking protocols
where False Acceptance Rate (FAR) and False Rejection Rate (FRR) cross.
"""

from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np


def compute_eer(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    num_thresholds: int = 1000,
) -> Tuple[float, float]:
    """Compute the exact Equal Error Rate (EER) and operating threshold.

    In audio deepfake forensics (ASVspoof standard):
      - Label 0 = Bonafide (Authentic human voice)
      - Label 1 = Spoof (Synthetic speech / voice clone)
      - Score = Spoof probability in [0, 1]

    At decision threshold theta:
      - Predicted Spoof if score >= theta
      - Predicted Bonafide if score < theta

    False Alarm / False Acceptance Rate (FAR):
      Bonafide misclassified as Spoof: FAR = FP / (FP + TN)
    False Rejection Rate (FRR):
      Spoof misclassified as Bonafide: FRR = FN / (TP + FN)

    Equal Error Rate (EER) is the point where FAR == FRR.

    Args:
        labels: Ground truth binary labels (0 for bonafide, 1 for spoof).
        scores: Model predicted spoof probabilities in [0.0, 1.0].
        num_thresholds: Number of discrete evaluation thresholds to sweep.

    Returns:
        Tuple of (eer_value, optimal_threshold). EER is returned as a float (0.0 to 1.0).
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    if len(labels_arr) != len(scores_arr):
        raise ValueError("Labels and scores arrays must have the same length.")

    n_bonafide = np.sum(labels_arr == 0)
    n_spoof = np.sum(labels_arr == 1)

    if n_bonafide == 0 or n_spoof == 0:
        # Cannot calculate EER with single-class evaluation data
        return 0.0, 0.5

    # Sweep thresholds across the score distribution
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    far_list = np.zeros(num_thresholds, dtype=float)
    frr_list = np.zeros(num_thresholds, dtype=float)

    for i, thresh in enumerate(thresholds):
        preds = (scores_arr >= thresh).astype(int)

        # FP: bonafide (0) predicted as spoof (1)
        fp = np.sum((labels_arr == 0) & (preds == 1))
        # FN: spoof (1) predicted as bonafide (0)
        fn = np.sum((labels_arr == 1) & (preds == 0))

        far_list[i] = fp / n_bonafide
        frr_list[i] = fn / n_spoof

    # Find the index where FAR and FRR cross (|FAR - FRR| is minimized)
    diff = far_list - frr_list
    min_idx = np.argmin(np.abs(diff))

    # Perform linear interpolation between adjacent points for sub-step precision
    if min_idx > 0 and min_idx < num_thresholds - 1:
        # Check sign change
        if diff[min_idx - 1] * diff[min_idx] <= 0:
            idx1, idx2 = min_idx - 1, min_idx
        elif diff[min_idx] * diff[min_idx + 1] <= 0:
            idx1, idx2 = min_idx, min_idx + 1
        else:
            idx1, idx2 = min_idx, min_idx

        if idx1 != idx2:
            denom = (frr_list[idx2] - frr_list[idx1]) - (far_list[idx2] - far_list[idx1])
            if abs(denom) > 1e-9:
                weight = (far_list[idx1] - frr_list[idx1]) / denom
                eer = far_list[idx1] + weight * (far_list[idx2] - far_list[idx1])
                opt_thresh = thresholds[idx1] + weight * (thresholds[idx2] - thresholds[idx1])
                return float(np.clip(eer, 0.0, 1.0)), float(np.clip(opt_thresh, 0.0, 1.0))

    # Discrete fallback
    eer = (far_list[min_idx] + frr_list[min_idx]) / 2.0
    opt_thresh = thresholds[min_idx]

    return float(np.clip(eer, 0.0, 1.0)), float(np.clip(opt_thresh, 0.0, 1.0))


def calculate_roc_curve(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    num_thresholds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) coordinates.

    Args:
        labels: Binary ground-truth array (0=bonafide, 1=spoof).
        scores: Spoof probability scores.
        num_thresholds: Number of threshold evaluation steps.

    Returns:
        Tuple of (fpr, tpr, thresholds).
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    n_bonafide = max(int(np.sum(labels_arr == 0)), 1)
    n_spoof = max(int(np.sum(labels_arr == 1)), 1)

    thresholds = np.linspace(1.0, 0.0, num_thresholds)
    fpr = np.zeros(num_thresholds)
    tpr = np.zeros(num_thresholds)

    for i, thresh in enumerate(thresholds):
        preds = (scores_arr >= thresh).astype(int)
        fp = np.sum((labels_arr == 0) & (preds == 1))
        tp = np.sum((labels_arr == 1) & (preds == 1))

        fpr[i] = fp / n_bonafide
        tpr[i] = tp / n_spoof

    return fpr, tpr, thresholds


def compute_roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute the Area Under the ROC Curve (ROC-AUC) using trapezoidal integration.

    Args:
        fpr: False positive rates sorted in ascending order.
        tpr: True positive rates.

    Returns:
        AUC float in range [0.0, 1.0].
    """
    # Ensure monotonically increasing FPR for proper trapezoidal rule
    order = np.argsort(fpr)
    sorted_fpr = fpr[order]
    sorted_tpr = tpr[order]
    return float(np.trapz(sorted_tpr, sorted_fpr))


def compute_forensic_metrics(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    calibrated_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate exhaustive biometric and forensics classification metrics.

    Args:
        labels: Ground-truth binary labels (0=bonafide, 1=spoof).
        scores: Model predicted spoof probabilities.
        calibrated_threshold: Optional decision boundary. If None, uses EER threshold.

    Returns:
        Dictionary containing EER, ROC-AUC, Accuracy, Precision, Recall,
        Specificity, F1 score, and Confusion Matrix.
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    # Calculate EER and operating threshold
    eer, eer_threshold = compute_eer(labels_arr, scores_arr)
    threshold = calibrated_threshold if calibrated_threshold is not None else eer_threshold

    # ROC & AUC
    fpr, tpr, _ = calculate_roc_curve(labels_arr, scores_arr)
    roc_auc = compute_roc_auc(fpr, tpr)

    # Classification metrics at operating threshold
    preds = (scores_arr >= threshold).astype(int)

    tp = int(np.sum((labels_arr == 1) & (preds == 1)))
    fp = int(np.sum((labels_arr == 0) & (preds == 1)))
    tn = int(np.sum((labels_arr == 0) & (preds == 0)))
    fn = int(np.sum((labels_arr == 1) & (preds == 0)))

    total = max(len(labels_arr), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    return {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "calibrated_threshold": threshold,
        "roc_auc": roc_auc,
        "accuracy": accuracy * 100.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "specificity": specificity * 100.0,
        "f1_score": f1,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
    }
