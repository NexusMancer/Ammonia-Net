from __future__ import annotations

# Standard library imports
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third-party imports for numerical operations, deep learning, and image processing
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Local imports for project configuration and utilities
from config.config import DatasetLayoutConfig, get_dataset_config
from utils.metrics import compute_mIoU

from .inference_utils import logits_to_class_index, predict_multitask_image


class EvalCallback:
    """Run end-of-epoch segmentation evaluation and report mIoU metrics."""

    def __init__(
        self,
        net,
        input_shape,
        num_classes,
        image_ids,
        dataset_path,
        cuda,
        config: Optional[DatasetLayoutConfig] = None,
        miou_out_path: Optional[str] = None,
        eval_flag=True,
        period=1,
    ):
        super().__init__()

        # Store the neural network model for evaluation
        self.net = net
        # Convert input shape to a tuple of integers for consistent formatting
        self.input_shape = tuple(int(value) for value in input_shape)
        # Number of segmentation classes
        self.num_classes = int(num_classes)
        # Normalize image IDs by removing any trailing whitespace or extra characters
        self.image_ids = tuple(self._normalize_image_id(image_id) for image_id in image_ids)
        # Path to the dataset root directory
        self.dataset_path = Path(dataset_path)
        # Dataset configuration (directory layout, file extensions, etc.)
        self.config = config or get_dataset_config()
        # Flag indicating whether to use CUDA (GPU)
        self.cuda = cuda
        # Path where mIoU prediction outputs will be saved
        self.miou_out_path = (
            Path(miou_out_path)
            if miou_out_path is not None
            else self.config.miou_output_path()
        )
        # Flag to enable/disable evaluation during training
        self.eval_flag = eval_flag
        # Evaluation period: run evaluation every N epochs
        self.period = period

    @staticmethod
    def _normalize_image_id(image_id: str) -> str:
        # Remove any trailing whitespace or extra characters from image ID
        return str(image_id).split()[0]

    def _model_device(self) -> torch.device:
        # Retrieve the device (CPU or GPU) where the model parameters are stored
        return next(self.net.parameters()).device

    def _image_path(self, image_id: str) -> Path:
        # Construct full path to the image file based on image ID and configured extension
        return self.config.image_dir(self.dataset_path) / f"{image_id}{self.config.image_extension}"

    def _ground_truth_dir(self) -> Path:
        # Get the directory containing ground truth segmentation masks
        return self.config.segmentation_dir(self.dataset_path)

    def _prediction_dir(self) -> Path:
        # Keep the historical directory name so compute_mIoU sees the same layout.
        return self.miou_out_path / self.config.miou_prediction_subdir

    def _generate_predictions(self, prediction_dir: Path) -> Dict[str, int]:
        """
        Generate predictions for all images and save segmentation masks.
        
        Returns a dictionary mapping image IDs to their predicted grade classifications
        after applying detection-grade gating logic.
        """
        classification_by_image: Dict[str, int] = {}

        # Process each image and generate prediction masks
        for image_id in tqdm(self.image_ids):
            with Image.open(self._image_path(image_id)) as image:
                # Get prediction mask and logits from the model
                prediction_image, grade_logits, detection_logits = self._predict_multitask(image)
            # Convert logits to class indices for grade and detection
            seg_mask = np.asarray(prediction_image)
            grade_index = logits_to_class_index(grade_logits)
            detection_index = logits_to_class_index(detection_logits)
            
            # Apply classification gating: if no defect detected or no grade assigned,
            # convert class 2 pixels to class 1 and reset grade to 0
            if detection_index == 0 or grade_index == 0:
                seg_mask = np.where(seg_mask == 2, 1, seg_mask)
                grade_index = 0
            
            # Store the gated grade classification
            classification_by_image[image_id] = grade_index
            # Save the gated segmentation mask as PNG file
            prediction_image = Image.fromarray(seg_mask.astype(np.uint8))
            prediction_image.save(prediction_dir / f"{image_id}{self.config.mask_extension}")

        return classification_by_image

    def _predict_multitask(self, image: Image.Image) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Generate segmentation predictions for a single image using the model.
        
        Returns:
            Tuple of (segmentation mask as PIL Image, grade logits tensor, detection logits tensor)
        """
        # Use the model to predict segmentation probabilities and grades
        prediction = predict_multitask_image(
            self.net,
            image=image,
            input_shape=self.input_shape,
            device=self._model_device(),
        )
        # Convert softmax probabilities to class indices (argmax operation)
        seg_mask = prediction.seg_probs.argmax(axis=-1).astype(np.uint8)
        # Return mask as PIL Image and both grade and detection logits for classification gating
        return Image.fromarray(seg_mask), prediction.grade_logits, prediction.detection_logits

    def on_epoch_end(self, epoch, model_eval):
        """
        Callback function executed at the end of each training epoch.
        Runs evaluation if the epoch matches the evaluation period.
        
        Returns:
            Tuple of mIoU, recall, precision, and accuracy metrics, or (None, None, None, None) if skipped.
        """
        # Skip evaluation if not on the right period or if evaluation is disabled
        if epoch % self.period != 0 or not self.eval_flag:
            return None, None, None, None

        # Update the network model to the latest state
        self.net = model_eval
        # Create directory for storing prediction masks
        prediction_dir = self._prediction_dir()
        prediction_dir.mkdir(parents=True, exist_ok=True)

        # Save the current training state to restore it later
        was_training = self.net.training
        # Switch model to evaluation mode (disables dropout, batch norm updates, etc.)
        self.net.eval()

        print("Get miou.")
        try:
            # Generate prediction masks for all images in the validation set
            classification_by_image = self._generate_predictions(prediction_dir)

            print("Calculate miou.")
            # Compute mIoU and other segmentation metrics by comparing predictions to ground truth
            _, ious, val_recall_seg, val_precision_seg, val_accuracy_seg = compute_mIoU(
                str(self._ground_truth_dir()),
                str(prediction_dir),
                self.image_ids,
                self.num_classes,
                classification_by_image,
                None,
                dataset_config=self.config,
            )
        finally:
            # Restore the original training/eval mode even if an error occurred
            self.net.train(was_training)
            # Clean up temporary prediction files
            shutil.rmtree(self.miou_out_path, ignore_errors=True)
        print("Get miou done.")

        # Return average metrics (using nanmean to handle any NaN values from empty classes)
        return (
            float(np.nanmean(ious)),
            float(np.nanmean(val_recall_seg)),
            float(np.nanmean(val_precision_seg)),
            float(np.nanmean(val_accuracy_seg)),
        )
