"""
Configuration module for Ammonia-Net project.

This module defines all dataclasses and configuration factories used throughout the project,
including dataset layout, inference, training, and prediction settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from torchvision import transforms


@dataclass(frozen=True)
class DatasetLayoutConfig:
    """
    Configuration for dataset directory structure and naming conventions.
    
    This class defines the standard directory layout and file extensions used
    by the dataset, following PASCAL VOC dataset conventions.
    """
    # Images directory name
    image_dir_name: str = "JPEGImages"
    # Segmentation masks directory name
    segmentation_dir_name: str = "SegmentationClass"
    # Root directory for train/val/test split files
    split_root_dir_name: str = "ImageSets"
    # Subdirectory for segmentation-specific splits
    split_subdir_name: str = "Segmentation"
    # Filename for trainval split (training + validation combined)
    trainval_split_filename: str = "trainval.txt"
    # Filename for training split
    train_split_filename: str = "train.txt"
    # Filename for validation split
    val_split_filename: str = "val.txt"
    # Filename for test/holdout split
    test_split_filename: str = "test.txt"
    # Image file extension
    image_extension: str = ".jpg"
    # Segmentation mask file extension
    mask_extension: str = ".png"
    # Temporary output directory for mIoU computation
    miou_output_dir: str = ".temp_miou_out"
    # Subdirectory for mIoU prediction results
    miou_prediction_subdir: str = "detection-results"

    def dataset_root(self, dataset_root_path: str | Path) -> Path:
        # Get the root dataset directory as a Path object
        return Path(dataset_root_path)

    def image_dir(self, dataset_root_path: str | Path) -> Path:
        # Get the directory containing image files
        return self.dataset_root(dataset_root_path) / self.image_dir_name

    def segmentation_dir(self, dataset_root_path: str | Path) -> Path:
        # Get the directory containing ground truth segmentation masks
        return self.dataset_root(dataset_root_path) / self.segmentation_dir_name

    def split_dir(self, dataset_root_path: str | Path) -> Path:
        # Get the directory containing train/val/test split files
        return (
            self.dataset_root(dataset_root_path)
            / self.split_root_dir_name
            / self.split_subdir_name
        )

    def split_path(self, dataset_root_path: str | Path, split_name: str) -> Path:
        # Get the full path to a specific dataset split file
        filename_by_split = {
            "trainval": self.trainval_split_filename,
            "train": self.train_split_filename,
            "val": self.val_split_filename,
            "test": self.test_split_filename,
        }
        if split_name not in filename_by_split:
            supported = ", ".join(sorted(filename_by_split))
            raise ValueError(f"Unsupported split name '{split_name}'. Expected one of: {supported}.")
        return self.split_dir(dataset_root_path) / filename_by_split[split_name]

    def miou_output_path(self) -> Path:
        # Get the temporary output directory for mIoU computation
        return Path(self.miou_output_dir)

    def miou_prediction_dir(self) -> Path:
        # Get the directory where mIoU predictions will be stored
        return self.miou_output_path() / self.miou_prediction_subdir


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for model inference."""
    # Path to the pre-trained model weights file
    model_path: str = "weight.pth"
    # Number of segmentation classes
    num_classes: int = 3
    # Input image shape (height, width) for the model
    input_shape: Tuple[int, int] = (512, 512)
    # Mix type flag for mixed precision or model variants (0 = default)
    mix_type: int = 0
    # Whether to use CUDA (GPU) for inference
    cuda: bool = True

# TODO
@dataclass(frozen=True)
class PredictConfig:
    """Configuration for prediction and visualization."""
    # Output directory name for saving predictions
    output_dir_name: str = "predictions"
    # Suffix to append to prediction filenames
    output_suffix: str = "_pred"
    # Output file extension for predictions
    output_extension: str = ".png"
    # Default font name for visualization labels
    default_font_name: str = "DejaVuSans.ttf"
    # Optional path to custom font file
    font_path: Optional[str] = None
    # Font size for visualization text
    font_size: int = 32
    # Segmentation class names
    seg_classes: Tuple[str, ...] = ("_background_", "Tongue", "toothmarks")
    # Grade/severity class names
    grade_classes: Tuple[str, ...] = ("NH", "LT", "MT", "ST")
    # Supported input image file extensions
    input_image_suffixes: Tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
    )
    # Nested inference configuration for model inference during prediction
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def default_output_dir(self, input_path: str | Path) -> Path:
        # Get the default output directory based on input path
        return Path(input_path) / self.output_dir_name


@dataclass(frozen=True)
class PrepareDatasetConfig:
    """Configuration for dataset preparation and splitting."""
    # Percentage of data to use for trainval (1.0 = use all data)
    trainval_percent: float = 1.0
    # Percentage of trainval data to allocate for training (remainder goes to validation)
    train_percent: float = 0.9
    # Random seed for reproducible train/val splits
    seed: int = 0
    # Dataset layout configuration
    dataset: DatasetLayoutConfig = field(default_factory=DatasetLayoutConfig)


@dataclass(frozen=True)
class MetricsOutputConfig:
    """Configuration for metric visualization and output files."""
    # Filename for mIoU plot visualization
    miou_plot_filename: str = "mIoU.png"
    # Filename for mean Pixel Accuracy plot
    mpa_plot_filename: str = "mPA.png"
    # Filename for recall metric plot
    recall_plot_filename: str = "Recall.png"
    # Filename for precision metric plot
    precision_plot_filename: str = "Precision.png"
    # Filename for confusion matrix CSV export
    confusion_matrix_filename: str = "confusion_matrix.csv"


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for model training."""
    # Learning rate for classification heads
    lr_c: float = 0.006
    # Final learning rate factor (for learning rate scheduling)
    lrf_c: float = 0.001
    # Enable validation evaluation during training
    eval_flag: bool = True
    # Number of iterations between evaluation runs (not epochs)
    eval_period: int = 1000
    # Checkpoint save period in epochs
    save_period: int = 1
    # Weight for the main segmentation loss
    loss_weight: float = 1.0
    # Weight for the grade/classification loss in multi-task learning
    loss_grade_weight: float = 1.0

    # Use CUDA (GPU) for training
    Cuda: bool = True
    # Enable mixed precision training (float16)
    fp16: bool = False

    # Number of segmentation classes
    num_classes: int = 3
    # Load pre-trained ImageNet weights for encoder
    pretrained: bool = False
    # Input image shape (height, width)
    input_shape: Tuple[int, int] = (512, 512)
    # Epoch to resume training from (0 = start from scratch)
    Init_Epoch: int = 0

    # Phase 1 (pretrain): number of epochs to train the full model (seg + cls)
    Pretrain_Epoch: int = 500
    # Batch size for Phase 1 training
    Pretrain_batch_size: int = 1
    # Phase 2 (post-training): number of epochs to train only classification heads (seg frozen).
    # Set to 0 to skip Phase 2 and run single-stage training for Pretrain_Epoch epochs only.
    PostTrain_Epoch: int = 1000
    # Batch size for Phase 2 training
    PostTrain_batch_size: int = 1

    # Shuffle dataset during training
    shuffle: bool = True

    # Use Dice loss in addition to cross-entropy
    dice_loss: bool = True
    # Use Focal loss for handling class imbalance
    focal_loss: bool = True
    # Number of worker threads for data loading
    num_workers: int = 10

    # Root path to the dataset
    dataset_root_path: str = os.getenv("AMN_DATASET_ROOT", "../MTL_nopush/AMN_dataset")
    # Path to pre-trained model checkpoint for resuming training
    model_path: str = ""
    # Template for checkpoint directory (format string with fold parameter)
    save_dir_template: str = "../MTL_nopush/log/logs_Usf_pro_fold={fold}/checkpoints"
    # File extension for checkpoint files
    checkpoint_suffix: str = ".pth"
    # Filename for the best epoch checkpoint
    best_checkpoint_name: str = "best_epoch_weights.pth"
    # Filename for the last epoch checkpoint
    last_checkpoint_name: str = "last_epoch_weights.pth"
    # Template for per-epoch checkpoint filenames (contains metrics)
    epoch_checkpoint_template: str = (
        "ep{epoch:03d}-"
        "loss_mtl{train_loss_mtl:.3f}-val_loss_mtl{val_loss_mtl:.3f}-"
        "loss_grade{train_loss_grade:.3f}-val_loss_grade{val_loss_grade:.3f}-"
        "loss_seg{train_loss_seg:.3f}-val_loss_seg{val_loss_seg:.3f}-"
        "acc_grade{train_accuracy_grade:.3f}-val_acc_grade{val_accuracy_grade:.3f}-"
        "fscore_seg{train_f_score_seg:.3f}-val_fscore_seg{val_f_score_seg:.3f}"
        "{suffix}"
    )
    # Template for checkpoint filenames without validation metrics (lightweight checkpoints)
    no_val_checkpoint_template: str = "ep{epoch:03d}-loss{train_loss_mtl:.3f}{suffix}"
    # Directory for Weights & Biases experiment tracking
    wandb_dir: str = os.getenv("WANDB_DIR", "../MTL_nopush/log/wandb")
    # W&B project name
    wandb_project: str = os.getenv("WANDB_PROJECT", "Ammonia_Net_refactoring")
    # W&B workspace entity (None = use default)
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY") or None
    # W&B mode (offline, online, disabled)
    wandb_mode: str = os.getenv("WANDB_MODE", "offline")
    # Template for W&B run name (format string with fold and timestamp)
    wandb_run_name_template: str = "ammonia-net-fold={fold}{timestamp}"
    # Dataset layout configuration
    dataset: DatasetLayoutConfig = field(default_factory=DatasetLayoutConfig)

    # Optimizer momentum for SGD
    optimizer_momentum: float = 0.9
    # L2 weight decay for regularization
    optimizer_weight_decay: float = 4e-5

    # Target image size for training data augmentation
    train_transform_size: int = 512
    # Target image size for validation preprocessing
    val_transform_size: int = 512

    def pretrain_end_epoch(self) -> int:
        """Epoch index where Phase 1 ends and Phase 2 begins.

        Fixed at Pretrain_Epoch regardless of Init_Epoch.
        Init_Epoch is only a resume offset: it may fall inside Phase 1
        (Init_Epoch < pretrain_end_epoch) or Phase 2 (Init_Epoch >= pretrain_end_epoch).
        """
        return self.Pretrain_Epoch

    def total_epochs(self) -> int:
        """Total training length = Pretrain_Epoch + PostTrain_Epoch.

        PostTrain_Epoch == 0 means single-stage training (Phase 2 is skipped).
        """
        return self.Pretrain_Epoch + self.PostTrain_Epoch

    def build_data_transform(self) -> Dict[str, transforms.Compose]:
        """Build training and validation data augmentation pipelines.
        
        Returns:
            Dictionary with 'train' and 'val' keys, each containing a transforms.Compose pipeline.
        """
        return {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(self.train_transform_size),
                transforms.RandomHorizontalFlip(),
            ]),
            "val": transforms.Compose([
                transforms.Resize(self.val_transform_size),
                transforms.CenterCrop(self.val_transform_size),
            ]),
        }

    def build_save_dir(self, fold: int) -> str:
        # Build the checkpoint directory path for a specific fold
        return self.save_dir_template.format(fold=fold)

    def build_wandb_run_name(self, fold: int, timestamp: str) -> str:
        # Build the W&B run name with fold number and timestamp
        return self.wandb_run_name_template.format(fold=fold, timestamp=timestamp)

    def train_split_path(self) -> Path:
        # Get the path to the training split file
        return self.dataset.split_path(self.dataset_root_path, "train")

    def val_split_path(self) -> Path:
        # Get the path to the validation split file
        return self.dataset.split_path(self.dataset_root_path, "val")

    def build_epoch_checkpoint_name(
        self,
        *,
        epoch: int,
        train_loss_mtl: float,
        val_loss_mtl: float,
        train_loss_grade: float,
        val_loss_grade: float,
        train_loss_seg: float,
        val_loss_seg: float,
        train_accuracy_grade: float,
        val_accuracy_grade: float,
        train_f_score_seg: float,
        val_f_score_seg: float,
    ) -> str:
        """Build a checkpoint filename with all training metrics.
        
        Args:
            epoch: Current training epoch number
            train_loss_mtl: Training multi-task loss
            val_loss_mtl: Validation multi-task loss
            train_loss_grade: Training grade classification loss
            val_loss_grade: Validation grade classification loss
            train_loss_seg: Training segmentation loss
            val_loss_seg: Validation segmentation loss
            train_accuracy_grade: Training grade classification accuracy
            val_accuracy_grade: Validation grade classification accuracy
            train_f_score_seg: Training segmentation F-score
            val_f_score_seg: Validation segmentation F-score
            
        Returns:
            Formatted checkpoint filename with all metrics embedded.
        """
        return self.epoch_checkpoint_template.format(
            epoch=epoch,
            train_loss_mtl=train_loss_mtl,
            val_loss_mtl=val_loss_mtl,
            train_loss_grade=train_loss_grade,
            val_loss_grade=val_loss_grade,
            train_loss_seg=train_loss_seg,
            val_loss_seg=val_loss_seg,
            train_accuracy_grade=train_accuracy_grade,
            val_accuracy_grade=val_accuracy_grade,
            train_f_score_seg=train_f_score_seg,
            val_f_score_seg=val_f_score_seg,
            suffix=self.checkpoint_suffix,
        )



# =========================
# Configuration Factory Functions
# =========================

def get_dataset_config() -> DatasetLayoutConfig:
    """Get default dataset layout configuration."""
    return DatasetLayoutConfig()


def get_inference_config() -> InferenceConfig:
    """Get default inference configuration."""
    return InferenceConfig()


def get_predict_config() -> PredictConfig:
    """Get default prediction configuration."""
    return PredictConfig()


def get_prepare_dataset_config() -> PrepareDatasetConfig:
    """Get default dataset preparation configuration."""
    return PrepareDatasetConfig()


def get_metrics_config() -> MetricsOutputConfig:
    """Get default metrics output configuration."""
    return MetricsOutputConfig()


def get_train_config() -> TrainConfig:
    """Get default training configuration."""
    return TrainConfig()
