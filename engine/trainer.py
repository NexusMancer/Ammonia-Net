from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from config.config import TrainConfig
from utils.losses import CE_Loss, Dice_loss, Focal_Loss
from utils.metrics import f_score
from utils.preprocessing import get_lr

from .training_utils import (
    ClassificationMetrics,
    compute_classification_metrics,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)


@dataclass
class PhaseAccumulator:
    """Accumulates metrics and losses during a training/validation phase.
    
    Attributes:
        mean_loss_grade: Running average of grading classification loss
        mean_loss_detection: Running average of detection classification loss
        mean_loss_seg: Running average of segmentation loss
        mean_loss_mtl: Running average of multi-task learning combined loss
        mean_f_score_seg: Running average of segmentation F-score
        actual_grade_labels: List of true grading labels collected during phase
        predicted_grade_labels: List of predicted grading labels (gated by detection)
        actual_detection_labels: List of true detection labels (binary: 0 or 1)
        predicted_detection_labels: List of predicted detection labels
    """
    mean_loss_grade: torch.Tensor
    mean_loss_detection: torch.Tensor
    mean_loss_seg: torch.Tensor
    mean_loss_mtl: torch.Tensor
    mean_f_score_seg: torch.Tensor
    actual_grade_labels: list = field(default_factory=list)
    predicted_grade_labels: list = field(default_factory=list)
    actual_detection_labels: list = field(default_factory=list)
    predicted_detection_labels: list = field(default_factory=list)


@dataclass(frozen=True)
class PhaseSummary:
    """Summary statistics for a complete training/validation phase.
    
    Immutable dataclass containing aggregated metrics and losses over all batches.
    """
    mean_loss_grade: torch.Tensor  # Average grading loss
    mean_loss_detection: torch.Tensor  # Average detection loss
    mean_loss_seg: torch.Tensor  # Average segmentation loss
    mean_loss_mtl: torch.Tensor  # Average multi-task learning loss
    mean_f_score_seg: torch.Tensor  # Average segmentation F-score
    grade_metrics: ClassificationMetrics  # Accuracy, precision, recall, F-score for grading
    detection_metrics: ClassificationMetrics  # Metrics for detection (binary classification)


@dataclass(frozen=True)
class BatchStats:
    """Statistics computed for a single batch.
    
    Immutable dataclass containing model outputs and losses for one batch.
    """
    grading_logits: torch.Tensor  # Raw model outputs for grade classification
    detection_logits: torch.Tensor  # Raw model outputs for detection (binary)
    detection_labels: torch.Tensor  # Binary labels derived from grade_labels (0 or 1)
    loss_grade: torch.Tensor  # Classification loss for grading task
    loss_detection: torch.Tensor  # Classification loss for detection task
    loss_seg: torch.Tensor  # Loss for segmentation task
    loss_mtl: torch.Tensor  # Combined multi-task learning loss
    f_score_seg: torch.Tensor  # F-score for segmentation


def _create_phase_accumulator(device):
    """Initialize a PhaseAccumulator with zero tensors on the specified device.
    
    Args:
        device: torch device (CPU or CUDA)
        
    Returns:
        PhaseAccumulator with all metrics initialized to zero
    """
    return PhaseAccumulator(
        mean_loss_grade=torch.zeros(1, device=device),
        mean_loss_detection=torch.zeros(1, device=device),
        mean_loss_seg=torch.zeros(1, device=device),
        mean_loss_mtl=torch.zeros(1, device=device),
        mean_f_score_seg=torch.zeros(1, device=device),
    )


def _update_running_mean(current_mean, value, iteration):
    """Compute running mean incrementally for online statistics accumulation.
    
    Uses the formula: new_mean = (old_mean * n + value) / (n + 1)
    This avoids storing all values and is numerically stable.
    
    Args:
        current_mean: Previous running mean
        value: New value to incorporate
        iteration: Current iteration count (0-indexed)
        
    Returns:
        Updated running mean
    """
    return (current_mean * iteration + value.detach()) / (iteration + 1)


def _move_batch_to_device(batch, device):
    """Transfer batch tensors to the specified device.
    
    Args:
        batch: Tuple of (imgs, pngs, labels, class_labels)
        device: Target device (CPU or CUDA)
        
    Returns:
        Tuple with all tensors moved to device
    """
    imgs, pngs, labels, class_labels = batch
    return (
        imgs.to(device),
        pngs.to(device),
        labels.to(device),
        class_labels.to(device),
    )


def _compute_batch_stats(
    model_train,
    imgs,
    pngs,
    labels,
    grade_labels,
    weights,
    num_classes,
    dice_loss,
    focal_loss,
    loss_seg_weight,
    loss_grade_weight,
    loss_function,
):
    """Compute all losses and predictions for a batch.
    
    This function performs forward pass through the model and computes three task losses:
    1. Segmentation loss (semantic segmentation)
    2. Grade classification loss (multi-class as determined by grade_labels)
    3. Detection loss (binary classification: 0 = no toothmark, 1 = toothmark present)
    
    Args:
        model_train: The model in training mode
        imgs: Input images
        pngs: One-hot encoded segmentation masks
        labels: Segmentation labels
        grade_labels: Grade classification labels (includes background class 0)
        weights: Class weights for weighted loss computation
        num_classes: Number of segmentation classes
        dice_loss: Whether to use Dice loss
        focal_loss: Whether to use Focal loss
        loss_seg_weight: Weight for segmentation loss in MTL combination
        loss_grade_weight: Weight for grading loss in MTL combination
        loss_function: Loss function for classification tasks (typically CrossEntropyLoss)
        
    Returns:
        BatchStats containing all computed losses and model outputs
    """
    seg_logits, grading_logits, detection_logits = model_train(imgs)
    # Create binary detection labels: grade_labels != 0 becomes 1, grade_labels == 0 becomes 0
    detection_labels = (grade_labels != 0).long()

    loss_grade = loss_function(grading_logits, grade_labels)
    loss_detection = loss_function(detection_logits, detection_labels)

    if focal_loss:
        loss_seg = Focal_Loss(seg_logits, pngs, weights, num_classes=num_classes)
    else:
        loss_seg = CE_Loss(seg_logits, pngs, weights, num_classes=num_classes)

    if dice_loss:
        loss_seg = loss_seg + Dice_loss(seg_logits, labels)

    with torch.no_grad():
        seg_f_score = f_score(seg_logits, labels)

    # Combine losses using weighted multi-task learning approach
    loss_mtl = loss_seg_weight * loss_seg + loss_grade_weight * loss_grade
    return BatchStats(
        grading_logits=grading_logits,
        detection_logits=detection_logits,
        detection_labels=detection_labels,
        loss_grade=loss_grade,
        loss_detection=loss_detection,
        loss_seg=loss_seg,
        loss_mtl=loss_mtl,
        f_score_seg=seg_f_score,
    )


def _update_phase_accumulator(accumulator, batch_stats, grade_labels, iteration):
    """Update running statistics in the accumulator with current batch stats.
    
    Updates all loss metrics and accumulates predicted/actual labels for metric computation.
    For grading predictions, applies gating logic: if detection predicts negative (0),
    force predicted grade to 0 regardless of grading head output.
    
    Args:
        accumulator: PhaseAccumulator to update
        batch_stats: BatchStats from current batch
        grade_labels: True grade labels for current batch
        iteration: Batch index (0-indexed)
    """
    # Update running mean losses
    accumulator.mean_loss_grade = _update_running_mean(
        accumulator.mean_loss_grade,
        batch_stats.loss_grade,
        iteration,
    )
    accumulator.mean_loss_detection = _update_running_mean(
        accumulator.mean_loss_detection,
        batch_stats.loss_detection,
        iteration,
    )
    accumulator.mean_loss_seg = _update_running_mean(
        accumulator.mean_loss_seg,
        batch_stats.loss_seg,
        iteration,
    )
    accumulator.mean_loss_mtl = _update_running_mean(
        accumulator.mean_loss_mtl,
        batch_stats.loss_mtl,
        iteration,
    )
    accumulator.mean_f_score_seg = _update_running_mean(
        accumulator.mean_f_score_seg,
        batch_stats.f_score_seg,
        iteration,
    )

    # Get predictions from model outputs
    grading_predictions = torch.argmax(batch_stats.grading_logits, dim=1).detach().cpu().tolist()
    detection_predictions = torch.argmax(batch_stats.detection_logits, dim=1).detach().cpu().tolist()
    actual_grade_labels = grade_labels.detach().cpu().tolist()
    actual_detection_labels = batch_stats.detection_labels.detach().cpu().tolist()
    
    # Apply gating: if detection predicts negative (0), override grade prediction to 0
    # This ensures consistency between the two tasks
    gated_predictions = [
        0 if detection_prediction == 0 else grading_prediction
        for grading_prediction, detection_prediction in zip(
            grading_predictions,
            detection_predictions,
        )
    ]

    accumulator.actual_grade_labels.extend(actual_grade_labels)
    accumulator.predicted_grade_labels.extend(gated_predictions)
    accumulator.actual_detection_labels.extend(actual_detection_labels)
    accumulator.predicted_detection_labels.extend(detection_predictions)


def _finalize_phase(accumulator):
    """Convert accumulated batch statistics into final phase summary.
    
    Computes classification metrics (accuracy, precision, recall, F-score) from
    accumulated predictions and labels.
    
    Args:
        accumulator: PhaseAccumulator with all batch statistics
        
    Returns:
        PhaseSummary with finalized metrics
    """
    return PhaseSummary(
        mean_loss_grade=accumulator.mean_loss_grade,
        mean_loss_detection=accumulator.mean_loss_detection,
        mean_loss_seg=accumulator.mean_loss_seg,
        mean_loss_mtl=accumulator.mean_loss_mtl,
        mean_f_score_seg=accumulator.mean_f_score_seg,
        grade_metrics=compute_classification_metrics(
            accumulator.actual_grade_labels,
            accumulator.predicted_grade_labels,
        ),
        detection_metrics=compute_classification_metrics(
            accumulator.actual_detection_labels,
            accumulator.predicted_detection_labels,
        ),
    )


def _update_progress_bar(progress_bar, optimizer, accumulator):
    """Update tqdm progress bar with current metrics.
    
    Args:
        progress_bar: tqdm progress bar object
        optimizer: Optimizer to read learning rate from
        accumulator: Current phase accumulator with metrics
    """
    progress_bar.set_postfix(
        **{
            "lr": get_lr(optimizer),  # Current learning rate
            "seg_fscore": round(accumulator.mean_f_score_seg.item(), 3),  # Mean segmentation F-score
            "seg_loss": round(accumulator.mean_loss_seg.item(), 3),  # Mean segmentation loss
            "mtl_loss": round(accumulator.mean_loss_mtl.item(), 3),  # Mean multi-task loss
            "grade_loss": round(accumulator.mean_loss_grade.item(), 3),  # Mean grading loss
        }
    )
    progress_bar.update(1)


def _run_phase(
    phase_name,
    model_train,
    optimizer,
    epoch,
    epoch_step,
    data_loader,
    total_epochs,
    device,
    Pretrain_Epoch,
    dice_loss,
    focal_loss,
    cls_weights,
    num_classes,
    fp16,
    scaler,
    loss_seg_weight,
    loss_grade_weight,
    is_train,
    train_loss_mode="staged_classification",
):
    """Execute a complete training or validation phase.
    
    Iterates through the data loader, computes losses, updates metrics,
    and optionally performs backpropagation and parameter updates.
    
    Args:
        phase_name: Name of phase (e.g., 'Train', 'Validation')
        model_train: Model in training/eval mode
        optimizer: Optimizer for parameter updates
        epoch: Current epoch number
        epoch_step: Number of batches per epoch
        data_loader: PyTorch data loader
        total_epochs: Total number of epochs for the run
        device: Compute device
        Pretrain_Epoch: Epoch at which to switch loss functions in staged training
        dice_loss: Whether to include Dice loss for segmentation
        focal_loss: Whether to use Focal loss for segmentation
        cls_weights: Class weights for loss computation
        num_classes: Number of segmentation classes
        fp16: Whether to use mixed precision training
        scaler: Gradient scaler for mixed precision (required if fp16=True)
        loss_seg_weight: Weight of segmentation loss in MTL
        loss_grade_weight: Weight of grading loss in MTL
        is_train: True for training phase, False for validation
        train_loss_mode: Determines which loss to use for backpropagation:
                        - 'mtl': combines all losses (segmentation + grading + detection)
                        - 'staged_classification': uses detection loss early epochs, 
                          switches to grading loss after Pretrain_Epoch
        
    Returns:
        PhaseSummary with aggregated metrics for the phase
    """
    print(f"Start {phase_name}")
    progress_bar = tqdm(
        total=epoch_step,
        desc=f"Epoch {epoch + 1}/{total_epochs}",
        mininterval=0.3,
    )

    model_train.train(is_train)
    accumulator = _create_phase_accumulator(device)
    loss_function = torch.nn.CrossEntropyLoss()
    weights = torch.as_tensor(cls_weights, device=device)

    if fp16 and scaler is None:
        raise ValueError("scaler must be provided when fp16=True.")

    for iteration, batch in enumerate(data_loader):
        if iteration >= epoch_step:
            break

        imgs, pngs, labels, grade_labels = _move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=fp16):
                batch_stats = _compute_batch_stats(
                    model_train=model_train,
                    imgs=imgs,
                    pngs=pngs,
                    labels=labels,
                    grade_labels=grade_labels,
                    weights=weights,
                    num_classes=num_classes,
                    dice_loss=dice_loss,
                    focal_loss=focal_loss,
                    loss_seg_weight=loss_seg_weight,
                    loss_grade_weight=loss_grade_weight,
                    loss_function=loss_function,
                )

            if is_train:
                # Select loss for backpropagation based on training strategy
                if train_loss_mode == "mtl":
                    # Multi-task learning: use weighted combination of all losses
                    loss = batch_stats.loss_mtl
                else:
                    # Staged training: use detection loss early, switch to grading later
                    loss = batch_stats.loss_detection if epoch < Pretrain_Epoch else batch_stats.loss_grade
                if fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        _update_phase_accumulator(accumulator, batch_stats, grade_labels, iteration)
        _update_progress_bar(progress_bar, optimizer, accumulator)

    progress_bar.close()
    print(f"Finish {phase_name}")
    return _finalize_phase(accumulator)


def _save_epoch_weights(
    model,
    wandb_logger,
    save_dir,
    config: TrainConfig,
    epoch,
    total_epochs,
    train_summary,
    val_summary,
):
    """Save model weights and update best model tracking.
    
    Saves checkpoints at periodic intervals and maintains best/last models.
    
    Args:
        model: Model to save
        wandb_logger: Logger for tracking best validation loss
        save_dir: Directory for saving checkpoints
        config: Training configuration
        epoch: Current epoch number
        total_epochs: Total epochs in training
        train_summary: Training phase metrics
        val_summary: Validation phase metrics
    """
    save_dir = Path(save_dir)
    if (epoch + 1) % config.save_period == 0 or epoch + 1 == total_epochs:
        torch.save(
            model.state_dict(),
            save_dir / config.build_epoch_checkpoint_name(
                epoch=epoch + 1,
                train_loss_mtl=train_summary.mean_loss_mtl.item(),
                val_loss_mtl=val_summary.mean_loss_mtl.item(),
                train_loss_grade=train_summary.mean_loss_grade.item(),
                val_loss_grade=val_summary.mean_loss_grade.item(),
                train_loss_seg=train_summary.mean_loss_seg.item(),
                val_loss_seg=val_summary.mean_loss_seg.item(),
                train_accuracy_grade=train_summary.grade_metrics.accuracy,
                val_accuracy_grade=val_summary.grade_metrics.accuracy,
                train_f_score_seg=train_summary.mean_f_score_seg.item(),
                val_f_score_seg=val_summary.mean_f_score_seg.item(),
            ),
        )

    if wandb_logger.update_best_val_loss(val_summary.mean_loss_mtl.item(), epoch + 1):
        print(f"Save best model to {config.best_checkpoint_name}")
        torch.save(model.state_dict(), save_dir / config.best_checkpoint_name)

    torch.save(model.state_dict(), save_dir / config.last_checkpoint_name)


def staged_train_validation_epoch(model_train, model, Pretrain_Epoch, wandb_logger, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_dir, config: TrainConfig, batch_size=16, loss_seg_weight=0.1, loss_grade_weight=0.99):
    """Execute one complete epoch of training and validation.
    
    Runs both training and validation phases, logs metrics, evaluates segmentation
    performance, and saves model checkpoints.
    
    Args:
        model_train: Model with optional DataParallel wrapper
        model: Underlying model for checkpoint saving
        Pretrain_Epoch: Epoch to switch from detection to grading loss
        wandb_logger: Weights & Biases logger
        eval_callback: Segmentation evaluation callback
        optimizer: Optimizer instance
        epoch: Current epoch number
        epoch_step: Number of training batches
        epoch_step_val: Number of validation batches
        gen: Training data loader
        gen_val: Validation data loader
        Epoch: Total epochs
        cuda: Whether CUDA is available
        dice_loss: Use Dice loss flag
        focal_loss: Use Focal loss flag
        cls_weights: Class weights for segmentation loss
        num_classes: Number of segmentation classes
        fp16: Use mixed precision flag
        scaler: Gradient scaler for mixed precision
        save_dir: Checkpoint directory
        config: Training configuration
        batch_size: Batch size
        loss_seg_weight: Segmentation loss weight in MTL
        loss_grade_weight: Grading loss weight in MTL
        
    Returns:
        Tuple of validation metrics:
            (detection_accuracy, detection_loss, mtl_loss, 
             grade_accuracy, grade_precision, grade_recall, grade_f_score,
             seg_iou, seg_accuracy, seg_recall, seg_precision, seg_f_score)
    """
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    train_summary = _run_phase(
        phase_name="Train",
        model_train=model_train,
        optimizer=optimizer,
        epoch=epoch,
        epoch_step=epoch_step,
        data_loader=gen,
        total_epochs=Epoch,
        device=device,
        Pretrain_Epoch=Pretrain_Epoch,
        dice_loss=dice_loss,
        focal_loss=focal_loss,
        cls_weights=cls_weights,
        num_classes=num_classes,
        fp16=fp16,
        scaler=scaler,
        loss_seg_weight=loss_seg_weight,
        loss_grade_weight=loss_grade_weight,
        is_train=True,
        train_loss_mode="staged_classification",
    )
    val_summary = _run_phase(
        phase_name="Validation",
        model_train=model_train,
        optimizer=optimizer,
        epoch=epoch,
        epoch_step=epoch_step_val,
        data_loader=gen_val,
        total_epochs=Epoch,
        device=device,
        Pretrain_Epoch=Pretrain_Epoch,
        dice_loss=dice_loss,
        focal_loss=focal_loss,
        cls_weights=cls_weights,
        num_classes=num_classes,
        fp16=fp16,
        scaler=scaler,
        loss_seg_weight=loss_seg_weight,
        loss_grade_weight=loss_grade_weight,
        is_train=False,
        train_loss_mode="staged_classification",
    )
    mean_val_iou_seg = mean_val_recall_seg_eval = mean_val_precision_seg_eval = mean_val_accruacy_seg = None
    mean_val_iou_seg, mean_val_recall_seg_eval, mean_val_precision_seg_eval, mean_val_accruacy_seg = eval_callback.on_epoch_end(epoch + 1, model_train)
    wandb_logger.log_epoch(
        epoch + 1,
        {
            "train/loss_mtl": train_summary.mean_loss_mtl.item(),
            "train/loss_grade": train_summary.mean_loss_grade.item(),
            "train/loss_seg": train_summary.mean_loss_seg.item(),
            "train/loss_detection": train_summary.mean_loss_detection.item(),
            "train/accuracy_grade": train_summary.grade_metrics.accuracy,
            "train/accuracy_detection": train_summary.detection_metrics.accuracy,
            "train/precision_grade": train_summary.grade_metrics.precision,
            "train/precision_detection": train_summary.detection_metrics.precision,
            "train/recall_grade": train_summary.grade_metrics.recall,
            "train/recall_detection": train_summary.detection_metrics.recall,
            "train/f_score_grade": train_summary.grade_metrics.f_score,
            "train/f_score_detection": train_summary.detection_metrics.f_score,
            "train/f_score_seg": train_summary.mean_f_score_seg.item(),
            "val/loss_mtl": val_summary.mean_loss_mtl.item(),
            "val/loss_grade": val_summary.mean_loss_grade.item(),
            "val/loss_seg": val_summary.mean_loss_seg.item(),
            "val/loss_detection": val_summary.mean_loss_detection.item(),
            "val/accuracy_grade": val_summary.grade_metrics.accuracy,
            "val/accuracy_detection": val_summary.detection_metrics.accuracy,
            "val/precision_grade": val_summary.grade_metrics.precision,
            "val/precision_detection": val_summary.detection_metrics.precision,
            "val/recall_grade": val_summary.grade_metrics.recall,
            "val/recall_detection": val_summary.detection_metrics.recall,
            "val/f_score_grade": val_summary.grade_metrics.f_score,
            "val/f_score_detection": val_summary.detection_metrics.f_score,
            "val/f_score_seg": val_summary.mean_f_score_seg.item(),
            "val/miou_seg": mean_val_iou_seg,
            "val/recall_seg": mean_val_recall_seg_eval,
            "val/precision_seg": mean_val_precision_seg_eval,
            "val/accuracy_seg": mean_val_accruacy_seg,
            "train/learning_rate": get_lr(optimizer),
        },
    )
    _save_epoch_weights(
        model=model,
        wandb_logger=wandb_logger,
        save_dir=save_dir,
        config=config,
        epoch=epoch,
        total_epochs=Epoch,
        train_summary=train_summary,
        val_summary=val_summary,
    )
    return (
        val_summary.detection_metrics.accuracy,
        val_summary.mean_loss_detection.item(),
        val_summary.mean_loss_mtl.item(),
        val_summary.grade_metrics.accuracy,
        val_summary.grade_metrics.precision,
        val_summary.grade_metrics.recall,
        val_summary.grade_metrics.f_score,
        mean_val_iou_seg,
        mean_val_accruacy_seg,
        mean_val_recall_seg_eval,
        mean_val_precision_seg_eval,
        val_summary.mean_f_score_seg.item(),
    )


__all__ = [
    "CE_Loss",
    "Focal_Loss",
    "Dice_loss",
    "weights_init",
    "get_lr_scheduler",
    "set_optimizer_lr",
    "staged_train_validation_epoch",  # Main training function for one epoch with validation
]
