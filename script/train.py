import argparse
import datetime
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from config.config import get_train_config
from dataset.dataset import AMNDataset, unet_dataset_collate
from engine.evaluation import EvalCallback
from engine.trainer import weights_init, staged_train_validation_epoch
from engine.training_utils import create_dataloaders, create_sgd_optimizer
from model import AmmoniaNet
from utils.checkpoint import extract_epoch_num
from utils.preprocessing import show_config
from utils.wandb_logger import WandbLogger


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train AmmoniaNet.")
    
    # Fold index for cross-validation, used in checkpoint directory naming
    parser.add_argument("--fold", type=int, default=0, help="Fold index used in save_dir naming.")
    
    # Optional custom dataset root path (overrides default path from config)
    # Expected structure: dataset_root/ImageSets/, JPEGImages/, SegmentationClass/
    parser.add_argument(
        "--dataset-root-path",
        default=None,
        help="Path to the dataset root that directly contains ImageSets, JPEGImages, and SegmentationClass.",
    )
    
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace):
    """Build runtime configuration by merging arg overrides with defaults.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Config object with command-line overrides applied
    """
    # Load default training configuration
    config = get_train_config()
    
    # Override dataset path if user provided a custom path via --dataset-root-path
    if args.dataset_root_path:
        config = replace(config, dataset_root_path=args.dataset_root_path)
    
    return config


def read_split_lines(split_path: Path, config) -> list[str]:
    """Read dataset split file containing image identifiers for training/validation.
    
    Args:
        split_path: Path to the split file (e.g., train.txt or val.txt)
        config: Configuration object containing dataset_root_path
        
    Returns:
        List of lines from the split file (image identifiers)
        
    Raises:
        FileNotFoundError: If split file not found, with helpful error message
    """
    # Verify the split file exists
    if not split_path.exists():
        # Raise informative error message with current paths and helpful instructions
        raise FileNotFoundError(
            "Dataset split file not found: "
            f"{split_path}\n"
            f"dataset_root_path={config.dataset_root_path!r}\n"
            "If your dataset root already contains ImageSets/JPEGImages/SegmentationClass, "
            "run with --dataset-root-path pointing to that directory."
        )
    
    # Read and return all lines from the split file
    with split_path.open("r", encoding="utf-8") as file:
        return file.readlines()


def main(fold: int = 0, config=None):
    """Main training loop for AmmoniaNet model.
    
    Handles model initialization, data loading, training across epochs,
    and evaluation with checkpoint management.
    
    Args:
        fold: Cross-validation fold index (default: 0)
        config: Training configuration object (uses default if None)
    """
    # Initialize or use provided configuration
    config = config or get_train_config()
    
    # Setup output directory and timestamp for this training run
    save_dir = Path(config.build_save_dir(fold))
    timestamp = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    
    # Initialize class weights (all ones by default)
    cls_weights = np.ones([config.num_classes], np.float32)
    
    # Build data augmentation transforms for train and validation
    data_transform = config.build_data_transform()
    
    # Determine compute device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if config.Cuda and torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # Initialize AmmoniaNet model and optionally initialize weights
    model = AmmoniaNet(num_classes=config.num_classes).train()
    weights_init(model)

    # Create output directory for checkpoints
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Weights & Biases logger for experiment tracking
    wandb_logger = WandbLogger(config, fold, save_dir, timestamp)

    # Setup automatic mixed precision (AMP) scaler if fp16 training enabled
    scaler = torch.amp.GradScaler(device.type, enabled=config.fp16) if config.fp16 else None
    
    try:
        # Move model to device and set to training mode
        model = model.to(device)
        model_train = model.train()

        # Enable multi-GPU training and optimization
        if use_cuda:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

        # Load training and validation split indices
        train_lines = read_split_lines(config.train_split_path(), config)
        val_lines = read_split_lines(config.val_split_path(), config)
        num_train = len(train_lines)
        num_val   = len(val_lines)

        # Display configuration summary
        show_config(
            num_classes=config.num_classes, model_path=config.model_path,
            input_shape=config.input_shape,
            Init_Epoch=config.Init_Epoch,
            Pretrain_Epoch=config.Pretrain_Epoch,
            PostTrain_Epoch=config.PostTrain_Epoch,
            total_epochs=config.total_epochs(),
            pretrain_end_epoch=config.pretrain_end_epoch(),
            Pretrain_batch_size=config.Pretrain_batch_size,
            PostTrain_batch_size=config.PostTrain_batch_size,
            num_workers=config.num_workers, num_train=num_train, num_val=num_val,
            wandb_project=config.wandb_project, wandb_mode=config.wandb_mode,
        )

        # total_epochs = Pretrain_Epoch + PostTrain_Epoch  (fixed training length)
        # pretrain_end = Pretrain_Epoch                    (Phase 1 / Phase 2 boundary)
        # Init_Epoch   = resume offset ∈ [0, total_epochs]; may fall in Phase 1 or Phase 2
        total_epochs = config.total_epochs()
        pretrain_end = config.pretrain_end_epoch()
        entered_post_training = False

        batch_size     = config.Pretrain_batch_size
        epoch_step     = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small for training with current batch size.")

        # Create training and validation datasets
        train_dataset = AMNDataset(
            train_lines, config.input_shape, config.num_classes, True, config.dataset_root_path,
            transform=data_transform["train"],
            config=config.dataset,
        )
        val_dataset   = AMNDataset(
            val_lines, config.input_shape, config.num_classes, False, config.dataset_root_path,
            transform=data_transform["val"],
            config=config.dataset,
        )

        shuffle = config.shuffle

        # Create data loaders for training and validation
        gen, gen_val = create_dataloaders(
            train_dataset, val_dataset, batch_size, config.num_workers,
            shuffle=shuffle, collate_fn=unet_dataset_collate,
        )

        # Setup evaluation callback for periodic validation during training
        eval_callback = EvalCallback(
            model, config.input_shape, config.num_classes, val_lines, config.dataset_root_path,
            use_cuda, config=config.dataset, eval_flag=config.eval_flag, period=config.eval_period,
        )

        # Create optimizer with trainable parameters
        optimizer = create_sgd_optimizer(
            model_train,
            lr=config.lr_c,
            momentum=config.optimizer_momentum,
            weight_decay=config.optimizer_weight_decay,
        )
        
        # Setup learning rate scheduler (cosine annealing over the full training span)
        lf        = lambda x: (
            (1 + math.cos(x * math.pi / total_epochs)) / 2
        ) * (1 - config.lrf_c) + config.lrf_c
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # Load existing checkpoints from previous epochs (for resuming training)
        # Sort by epoch number (natural sort) rather than lexicographic to avoid epoch_10 coming before epoch_2
        path_list = sorted(
            (path.name for path in save_dir.iterdir()
             if path.is_file() and path.suffix == config.checkpoint_suffix),
            key=extract_epoch_num
        )
        
        # Load the latest checkpoint if resuming training (only once, before the epoch loop)
        if path_list:
            latest_checkpoint = save_dir / path_list[-1]
            if latest_checkpoint.exists():
                weights_dict = torch.load(latest_checkpoint, map_location=device)
                model_train.load_state_dict(weights_dict, strict=False)
                print(f"Resumed training from checkpoint: {latest_checkpoint.name}")
            else:
                raise FileNotFoundError(f"Latest checkpoint not found: {latest_checkpoint}")

        # Main training loop: Init_Epoch → Init_Epoch + Pretrain_Epoch + PostTrain_Epoch
        for epoch in range(config.Init_Epoch, total_epochs):

            # Enter Phase 2 (post-training) once pretrain_end is reached.
            # PostTrain_Epoch == 0 means single-stage training; phase boundary is never crossed.
            # The flag ensures the transition runs exactly once even when resuming past pretrain_end.
            if config.PostTrain_Epoch > 0 and epoch >= pretrain_end and not entered_post_training:
                # Get underlying model (handle DataParallel wrapper if using multi-GPU)
                model_to_update = model_train.module if use_cuda else model
                model_to_update.freeze_segmentation_branch()
                batch_size     = config.PostTrain_batch_size
                epoch_step     = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small for post-training with current batch size.")

                gen, gen_val = create_dataloaders(
                    train_dataset, val_dataset, batch_size, config.num_workers,
                    shuffle=shuffle, collate_fn=unet_dataset_collate,
                )

                # Rebuild optimizer with only the now-trainable parameters (classification heads).
                # Without this, the optimizer retains momentum buffers for the frozen seg params.
                optimizer = create_sgd_optimizer(
                    model_train,
                    lr=config.lr_c,
                    momentum=config.optimizer_momentum,
                    weight_decay=config.optimizer_weight_decay,
                )
                entered_post_training = True

            # Execute one training epoch with validation
            (acc_val_detection, loss_val_detection, _,
             mean_val_accruacy_grade, mean_val_precision_grade, mean_val_recall_grade, mean_val_f_score_grade,
             mean_val_iou_seg, mean_val_accruacy_seg, mean_val_recall_seg, mean_val_precision_seg, mean_val_f_score_seg,
            ) = staged_train_validation_epoch(
                model_train, model, config.Pretrain_Epoch, wandb_logger, eval_callback, optimizer, epoch,
                epoch_step, epoch_step_val, gen, gen_val, total_epochs, use_cuda,
                config.dice_loss, config.focal_loss, cls_weights, config.num_classes, config.fp16, scaler,
                save_dir, config, batch_size,
                config.loss_weight, config.loss_grade_weight,
            )

            # Update learning rate for next epoch
            scheduler.step()

            # Print training metrics for this epoch
            print("[f {} epoch {}] Accd: {}; vlbd: {}; Accg: {}; Pg: {}; Rg: {}; Fg: {}; Iou: {}; Accs: {}; Ps: {}; Rs: {}; Fs: {}".format(
                fold, epoch + 1,
                round(acc_val_detection * 100, 4), round(loss_val_detection, 4),
                round(mean_val_accruacy_grade * 100, 4), round(mean_val_precision_grade * 100, 4),
                round(mean_val_recall_grade * 100, 4), round(mean_val_f_score_grade * 100, 4),
                "n/a" if mean_val_iou_seg is None else round(mean_val_iou_seg * 100, 4),
                "n/a" if mean_val_accruacy_seg is None else round(mean_val_accruacy_seg * 100, 4),
                "n/a" if mean_val_precision_seg is None else round(mean_val_precision_seg * 100, 4),
                "n/a" if mean_val_recall_seg is None else round(mean_val_recall_seg * 100, 4),
                round(mean_val_f_score_seg * 100, 4),
            ))
    finally:
        # Cleanup: finalize Weights & Biases logging
        wandb_logger.finish()


if __name__ == "__main__":
    args = parse_args()
    main(fold=args.fold, config=build_runtime_config(args))
