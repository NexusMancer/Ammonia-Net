import os
from dataclasses import asdict
from typing import Any, Mapping, Optional

import numpy as np
import torch
import wandb

from config.config import TrainConfig


class WandbLogger:
    """Logger class for tracking training metrics and configurations using Weights & Biases (WandB)."""
    def __init__(self, config: TrainConfig, fold: int, save_dir: str, timestamp: str):
        """Initialize WandB logger with training configuration and run metadata.
        
        Args:
            config: Training configuration object containing WandB project settings.
            fold: Current fold number for cross-validation tracking.
            save_dir: Directory path where model checkpoints are saved.
            timestamp: Timestamp string for the training run.
        """
        os.makedirs(config.wandb_dir, exist_ok=True)

        # Initialize WandB run with project configuration and training metadata
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.build_wandb_run_name(fold, timestamp),
            dir=config.wandb_dir,
            mode=config.wandb_mode,
            config=asdict(config),
            job_type="train",
            tags=[f"fold-{fold}"],
        )
        # Track the best validation loss across epochs
        self.best_val_loss_mtl: Optional[float] = None

        # Store fold number and checkpoint directory in WandB summary
        if self.run is not None:
            self.run.summary["fold"] = fold
            self.run.summary["save_dir"] = os.path.abspath(save_dir)

    @staticmethod
    def _serialize_metric(value: Any) -> Any:
        """Convert various metric types to scalar values compatible with WandB logging.
        
        Supports conversion of PyTorch tensors, NumPy arrays, and NumPy scalars to Python floats.
        
        Args:
            value: Metric value to serialize (can be tensor, array, or scalar).
            
        Returns:
            Serialized scalar value compatible with WandB.
            
        Raises:
            ValueError: If tensor or array contains more than one element.
        """
        # Convert PyTorch tensor to Python scalar
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be logged to wandb.")
            return value.detach().cpu().item()
        # Convert NumPy array to Python scalar
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError("Only scalar numpy arrays can be logged to wandb.")
            return value.reshape(-1)[0].item()
        # Convert NumPy generic type to Python scalar
        if isinstance(value, np.generic):
            return value.item()
        # Return value as-is if already a Python scalar
        return value

    def log_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Log epoch metrics to WandB for the current training step.
        
        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names and values to log. None values are skipped.
        """
        # Create payload with epoch information
        payload = {"epoch": epoch}
        # Serialize and add metrics to payload, skipping None values
        for key, value in metrics.items():
            if value is None:
                continue
            payload[key] = self._serialize_metric(value)
        # Log all metrics for this step
        self.run.log(payload, step=epoch)

    def update_best_val_loss(self, value: Any, epoch: int) -> bool:
        """Update best validation loss and track the epoch where it was achieved.
        
        Args:
            value: Current validation loss value to compare.
            epoch: Current epoch number.
            
        Returns:
            True if the current loss is better than the best recorded loss, False otherwise.
        """
        # Convert metric to float for comparison
        current = float(self._serialize_metric(value))
        # Check if this is a new best loss or the first recorded loss
        if self.best_val_loss_mtl is None or current <= self.best_val_loss_mtl:
            # Update best loss and record the epoch it was achieved
            self.best_val_loss_mtl = current
            self.run.summary["best/epoch"] = epoch
            self.run.summary["best/val_loss_mtl"] = current
            return True
        return False

    def finish(self) -> None:
        """Finalize the WandB run and upload all logged data."""
        if self.run is not None:
            self.run.finish()
