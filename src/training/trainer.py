"""PyTorch training loop and validation engine for speech forensics models."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.training.loss import LabelSmoothingBCELoss, BinaryFocalLoss
from src.evaluation.metrics import compute_eer

logger = logging.getLogger("spectroguard.training")


class AudioForensicsTrainer:
    """Trainer orchestrator for SpectralResNet and Wav2VecDetector deepfake classifiers."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Initialize AudioForensicsTrainer.

        Args:
            model: PyTorch nn.Module (SpectralResNet or Wav2VecDetector).
            train_loader: DataLoader yielding training batches.
            val_loader: DataLoader yielding validation batches.
            config: Hyperparameter dictionary containing training, optimizer, and loss settings.
            device: Optional torch.device. Defaults to auto-detecting cuda or cpu.
            checkpoint_dir: Directory to persist model weights and training metadata.
        """
        self.config = config
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 20))
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 5))

        # Setup Loss function
        loss_type = train_cfg.get("loss_type", "label_smoothing")
        if loss_type == "focal":
            self.criterion = BinaryFocalLoss(
                alpha=float(train_cfg.get("focal_alpha", 0.25)),
                gamma=float(train_cfg.get("focal_gamma", 2.0)),
            )
        else:
            self.criterion = LabelSmoothingBCELoss(
                epsilon=float(train_cfg.get("label_smoothing", 0.05))
            )

        # Setup Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Setup Scheduler
        scheduler_type = train_cfg.get("scheduler", "cosine")
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )

        # Best metrics tracking
        self.best_val_eer = float("inf")
        self.best_epoch = 0
        self.history: List[Dict[str, float]] = []

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Execute one complete training epoch.

        Args:
            epoch: Current epoch index (1-indexed).

        Returns:
            Tuple of (average_train_loss, train_accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # SpectralResNet expects spectrograms, Wav2Vec expects waveforms
            if "spectrogram" in batch:
                inputs = batch["spectrogram"].to(self.device)
            elif "waveform" in batch:
                inputs = batch["waveform"].to(self.device)
            else:
                raise KeyError("Batch must contain either 'spectrogram' or 'waveform'")

            targets = batch["label"].to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(inputs).view(-1)
            loss = self.criterion(logits, targets)

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Track statistics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            correct_preds += (preds == targets.long()).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = (correct_preds / max(total_samples, 1)) * 100.0
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model performance and calculate Equal Error Rate (EER).

        Args:
            data_loader: Optional DataLoader. Defaults to self.val_loader.

        Returns:
            Dictionary containing 'val_loss', 'val_acc', 'val_eer', 'val_threshold'.
        """
        loader = data_loader or self.val_loader
        self.model.eval()

        total_loss = 0.0
        all_targets: List[int] = []
        all_scores: List[float] = []
        total_samples = 0

        for batch in loader:
            if "spectrogram" in batch:
                inputs = batch["spectrogram"].to(self.device)
            elif "waveform" in batch:
                inputs = batch["waveform"].to(self.device)
            else:
                raise KeyError("Batch must contain either 'spectrogram' or 'waveform'")

            targets = batch["label"].to(self.device).float()
            logits = self.model(inputs).view(-1)
            loss = self.criterion(logits, targets)

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            all_targets.extend(targets.cpu().long().tolist())
            all_scores.extend(probs.cpu().tolist())

        avg_loss = total_loss / max(total_samples, 1)

        # Compute accuracy at default 0.5 threshold
        preds = [1 if s >= 0.5 else 0 for s in all_scores]
        acc = (
            sum(p == t for p, t in zip(preds, all_targets)) / max(total_samples, 1)
        ) * 100.0

        # Compute Equal Error Rate (EER) and calibrated threshold
        eer, optimal_threshold = compute_eer(all_targets, all_scores)

        return {
            "val_loss": avg_loss,
            "val_acc": acc,
            "val_eer": eer,
            "val_threshold": optimal_threshold,
        }

    def train(self) -> Dict[str, Any]:
        """Run full training lifecycle with learning rate scheduling and early stopping.

        Returns:
            Dictionary of training summary and checkpoint paths.
        """
        logger.info(f"Starting training on device: {self.device} for {self.epochs} epochs")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_acc"]
            val_eer = val_metrics["val_eer"]
            val_thresh = val_metrics["val_threshold"]

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_eer)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_eer": val_eer,
                "val_threshold": val_thresh,
                "lr": current_lr,
            }
            self.history.append(epoch_summary)

            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Val EER: {val_eer:.4f} (Thresh: {val_thresh:.4f}) | LR: {current_lr:.6f}"
            )

            # Checkpoint on best validation EER
            if val_eer < self.best_val_eer:
                self.best_val_eer = val_eer
                self.best_epoch = epoch
                patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path, epoch, val_metrics)
                logger.info(f"--> Saved new best checkpoint (EER: {val_eer:.4f}) to {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.early_stopping_patience})")
                    break

        # Save final epoch checkpoint
        final_path = self.checkpoint_dir / "latest_model.pt"
        self.save_checkpoint(final_path, epoch, val_metrics)

        return {
            "best_epoch": self.best_epoch,
            "best_val_eer": self.best_val_eer,
            "history": self.history,
            "best_model_path": str(self.checkpoint_dir / "best_model.pt"),
        }

    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Serialize model weights, optimizer state, and training metadata to disk.

        Args:
            filepath: Destination path for checkpoint file.
            epoch: Epoch index when checkpoint was taken.
            metrics: Evaluation metrics dictionary at this epoch.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, filepath)
