"""Shared segmentation metric utilities for computing TP/FP/FN and F-score."""

from typing import Tuple

import torch


def compute_tp_fp_fn(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute true positives, false positives, and false negatives for segmentation.
    
    Args:
        predictions: Predicted probabilities or masks of shape (batch_size, -1, num_classes).
        targets: Ground truth labels of shape (batch_size, -1, num_classes) excluding background.
        
    Returns:
        Tuple of (tp, fp, fn) tensors.
    """
    tp = torch.sum(targets * predictions, dim=(0, 1))
    fp = torch.sum(predictions, dim=(0, 1)) - tp
    fn = torch.sum(targets, dim=(0, 1)) - tp
    return tp, fp, fn


def compute_f_score(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    beta: float = 1,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Compute F-score from true positives, false positives, and false negatives.
    
    Args:
        tp: True positives tensor.
        fp: False positives tensor.
        fn: False negatives tensor.
        beta: Beta parameter for F-score. Default 1 for Dice score.
        smooth: Smoothing constant to avoid division by zero.
        
    Returns:
        F-score tensor computed as ((1 + beta^2) * tp + smooth) / ((1 + beta^2) * tp + beta^2 * fn + fp + smooth).
    """
    return ((1 + beta ** 2) * tp + smooth) / (
        (1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth
    )


def compute_f1_score(recall: float, precision: float) -> float:
    """Calculate F1-score from recall and precision values.
    
    F1-score is the harmonic mean of precision and recall, providing a balanced
    metric for classification performance.
    
    Args:
        recall: Recall value (True Positive Rate).
        precision: Precision value (Positive Prediction Value).
        
    Returns:
        F1-score as a float, or 0.0 if both recall and precision are 0.
    """
    denominator = recall + precision
    if denominator == 0:
        return 0.0
    return float(2 * recall * precision / denominator)


__all__ = ["compute_tp_fp_fn", "compute_f_score", "compute_f1_score"]
