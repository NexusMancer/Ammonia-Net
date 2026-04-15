import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.segmentation_utils import compute_tp_fp_fn, compute_f_score


def _resize_logits(inputs, target_height, target_width):
    """Resize logits to match target spatial dimensions using bilinear interpolation.
    
    Args:
        inputs: Input logits tensor of shape (batch_size, channels, height, width).
        target_height: Target height for resizing.
        target_width: Target width for resizing.
        
    Returns:
        Resized logits tensor matching the target dimensions.
    """
    # Extract spatial dimensions from input
    _, _, height, width = inputs.size()
    # Resize only if dimensions don't match
    if height != target_height or width != target_width:
        inputs = F.interpolate(
            inputs,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )
    return inputs


def CE_Loss(inputs, target, cls_weights, num_classes=3):
    """Cross Entropy Loss for multi-class semantic segmentation.
    
    Args:
        inputs: Model logits of shape (batch_size, num_classes, height, width).
        target: Ground truth labels of shape (batch_size, height, width).
        cls_weights: Class weights to handle class imbalance.
        num_classes: Number of segmentation classes. Default 3.
        
    Returns:
        Scalar cross entropy loss value.
    """
    # Extract dimensions from input tensor
    _, channels, _, _ = inputs.size()
    _, target_height, target_width = target.size()
    # Resize logits to match target dimensions
    inputs = _resize_logits(inputs, target_height, target_width)
    # Reshape tensors: (batch_size, channels, height, width) -> (batch_size*height*width, channels)
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, channels)
    # Flatten target labels
    temp_target = target.view(-1)
    # Compute weighted cross entropy loss, ignoring pixels with label >= num_classes
    return nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(
        temp_inputs,
        temp_target,
    )


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    """Focal Loss for handling class imbalance in semantic segmentation.
    
    Focal loss applies a modulating factor to the cross entropy loss to focus on
    hard negative examples and reduce the contribution of easy examples.
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    
    Args:
        inputs: Model logits of shape (batch_size, num_classes, height, width).
        target: Ground truth labels of shape (batch_size, height, width).
        cls_weights: Class weights to handle class imbalance.
        num_classes: Number of segmentation classes. Default 21.
        alpha: Weighting factor in [0, 1] to balance positive vs negative examples. Default 0.5.
        gamma: Exponent of the modulating factor (1 - p_t) to focus on hard examples. Default 2.
        
    Returns:
        Scalar focal loss value.
    """
    # Extract dimensions from input tensor
    _, channels, _, _ = inputs.size()
    _, target_height, target_width = target.size()
    # Resize logits to match target dimensions
    inputs = _resize_logits(inputs, target_height, target_width)
    # Reshape tensors: (batch_size, channels, height, width) -> (batch_size*height*width, channels)
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, channels)
    # Flatten target labels
    temp_target = target.view(-1)
    # Compute cross entropy loss per pixel without reduction
    logpt = -nn.CrossEntropyLoss(
        weight=cls_weights,
        ignore_index=num_classes,
        reduction="none",
    )(temp_inputs, temp_target)
    # Convert log-probability to probability
    pt = torch.exp(logpt)
    # Apply alpha weighting if specified
    if alpha is not None:
        logpt *= alpha
    # Apply focal term: -((1 - pt) ** gamma) * logpt
    # This down-weights easy examples and focuses on hard examples
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """Dice Loss (F-score loss) for semantic segmentation.
    
    Dice loss measures the overlap between predictions and ground truth.
    Loss = 1 - Dice coefficient. With beta=1, this becomes the standard Dice coefficient.
    
    Args:
        inputs: Model logits of shape (batch_size, num_classes, height, width).
        target: Ground truth labels of shape (batch_size, height, width, channels).
        beta: Beta parameter for F-score. Default 1 for standard Dice.
        smooth: Smoothing constant to avoid division by zero. Default 1e-5.
        
    Returns:
        Scalar Dice loss value (range 0 to 1, where 0 is perfect prediction).
    """
    # Extract batch size and channel dimensions
    batch_size, channels, _, _ = inputs.size()
    _, target_height, target_width, target_channels = target.size()
    # Resize logits to match target dimensions
    inputs = _resize_logits(inputs, target_height, target_width)
    # Apply softmax to get probabilities and flatten spatial dimensions
    temp_inputs = torch.softmax(
        inputs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, channels),
        dim=-1,
    )
    # Flatten target tensor
    temp_target = target.view(batch_size, -1, target_channels)
    # Calculate true positives, false positives, and false negatives
    tp, fp, fn = compute_tp_fp_fn(temp_inputs, temp_target[..., :-1])
    # Compute Dice coefficient
    score = compute_f_score(tp, fp, fn, beta=beta, smooth=smooth)
    # Return 1 - mean Dice coefficient as loss
    return 1 - torch.mean(score)


# Public API: Export loss functions for use in training
__all__ = ["CE_Loss", "Focal_Loss", "Dice_loss"]
