"""Unit tests for ASVspoof Equal Error Rate (EER) and biometric forensic evaluation metrics."""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    calculate_roc_curve,
    compute_forensic_metrics,
)


class TestAcousticForensicsMetrics:
    """Test suite for forensic evaluation algorithms."""

    def test_eer_perfect_separation(self):
        """When bonafide scores are strictly lower than spoof scores, EER must be 0.0."""
        # 50 bonafide samples with low spoof probability
        bonafide_scores = [0.05, 0.10, 0.12, 0.08, 0.15] * 10
        # 50 spoof samples with high spoof probability
        spoof_scores = [0.85, 0.90, 0.92, 0.88, 0.95] * 10

        labels = [0] * len(bonafide_scores) + [1] * len(spoof_scores)
        scores = bonafide_scores + spoof_scores

        eer, threshold = compute_eer(labels, scores)

        assert pytest.approx(eer, abs=1e-3) == 0.0
        # Threshold should lie cleanly between 0.15 and 0.85
        assert 0.15 <= threshold <= 0.85

    def test_eer_known_symmetric_crossover(self):
        """Construct synthetic distribution with an exact known 20% crossover error rate."""
        # 10 bonafide: 8 low (0.1) and 2 high (0.9) -> FAR at 0.5 is 2/10 = 0.20
        bonafide_scores = [0.1] * 8 + [0.9] * 2
        # 10 spoof: 2 low (0.1) and 8 high (0.9) -> FRR at 0.5 is 2/10 = 0.20
        spoof_scores = [0.1] * 2 + [0.9] * 8

        labels = [0] * 10 + [1] * 10
        scores = bonafide_scores + spoof_scores

        eer, threshold = compute_eer(labels, scores, num_thresholds=1000)

        # Expected EER is 0.20 (20%)
        assert pytest.approx(eer, abs=0.02) == 0.20
        assert pytest.approx(threshold, abs=0.05) == 0.50

    def test_roc_auc_perfect(self):
        """Verify ROC-AUC returns 1.0 for perfect class separation."""
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95]

        fpr, tpr, _ = calculate_roc_curve(labels, scores)
        auc = compute_roc_auc(fpr, tpr)

        assert pytest.approx(auc, abs=1e-4) == 1.0

    def test_compute_forensic_metrics_structure(self):
        """Verify all forensic metrics and confusion matrix values are populated."""
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]

        metrics = compute_forensic_metrics(labels, scores)

        assert "eer" in metrics
        assert "eer_percent" in metrics
        assert "calibrated_threshold" in metrics
        assert "roc_auc" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "specificity" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

        cm = metrics["confusion_matrix"]
        assert "true_positive" in cm
        assert "false_positive" in cm
        assert "true_negative" in cm
        assert "false_negative" in cm
        assert sum(cm.values()) == len(labels)
