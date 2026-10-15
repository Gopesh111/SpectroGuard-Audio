"""Loss functions for acoustic deepfake detection.

Implements Label Smoothing BCE and Focal Loss tailored for speech forensics
to prevent overconfidence on synthetic acoustic artifacts and mitigate
dataset class imbalance.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCELoss(nn.Module):
    """Binary Cross Entropy with Logits and Label Smoothing.

    Label smoothing reduces model overconfidence on specific vocoder signatures
    and room acoustic artifacts by converting hard 0/1 labels to soft targets:
        y_smooth = y * (1 - epsilon) + 0.5 * epsilon
    """

    def __init__(self, epsilon: float = 0.05, pos_weight: Optional[torch.Tensor] = None) -> None:
        """Initialize LabelSmoothingBCELoss.

        Args:
            epsilon: Smoothing parameter between 0.0 and 1.0.
            pos_weight: Weight for positive (synthetic) class to handle imbalance.
        """
        super().__init__()
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(f"Label smoothing epsilon must be in [0, 1), got {epsilon}")
        self.epsilon = epsilon
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smoothed BCE loss with logits.

        Args:
            logits: Predicted raw logits of shape (N,) or (N, 1).
            targets: Binary targets of shape (N,) or (N, 1) where 0=bonafide, 1=spoof.

        Returns:
            Scalar smoothed BCE loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing
        smoothed_targets = targets * (1.0 - self.epsilon) + 0.5 * self.epsilon

        return F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            pos_weight=self.pos_weight,
            reduction="mean",
        )


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss for hard example mining in acoustic spoof detection.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy bonafide/spoof samples to focus training gradients
    on subtle vocoder boundary artifacts.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize BinaryFocalLoss.

        Args:
            alpha: Weighting factor for positive class (spoof), balancing class frequency.
            gamma: Focusing parameter to penalize easily classified samples.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Binary Focal Loss.

        Args:
            logits: Predicted logits of shape (N,) or (N, 1).
            targets: Binary targets (0=bonafide, 1=spoof) of shape (N,) or (N, 1).

        Returns:
            Reduced focal loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        probs = torch.sigmoid(logits)
        # Compute p_t: prob of the true class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        # Clamp to avoid numerical instability
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)

        # Compute alpha_t
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal loss formula
        loss = -alpha_t * torch.pow(1.0 - p_t, self.gamma) * torch.log(p_t)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
