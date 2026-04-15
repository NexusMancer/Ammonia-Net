"""
AmmoniaNet Inference Module

This module provides the AmmoniaNetInferencer class for performing inference with the AmmoniaNet model.
It handles model loading, image preprocessing, multi-task predictions (segmentation and grading),
and visualization of results.
"""

from dataclasses import replace
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from config.config import InferenceConfig, get_inference_config
from model import AmmoniaNet
from utils.checkpoint import load_model_state
from utils.preprocessing import (
    build_color_palette,
    colorize_segmentation_mask,
    print_class_pixel_stats,
    show_config,
)

from .inference_utils import logits_to_class_index, predict_multitask_image


class AmmoniaNetInferencer:
    """
    Inference helper for AmmoniaNet segmentation and grading outputs.
    
    This class manages the inference pipeline including model initialization, 
    image prediction, mask processing, and result visualization.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize the AmmoniaNetInferencer with configuration and setup model.
        
        Args:
            config: InferenceConfig object containing model parameters. If None, uses default config.
            **kwargs: Additional keyword arguments to override config parameters.
        """
        resolved_config = config or get_inference_config()
        if kwargs:
            resolved_config = replace(resolved_config, **kwargs)
        self.config = replace(
            resolved_config,
            input_shape=tuple(int(value) for value in resolved_config.input_shape),
        )

        # Initialize model parameters
        self.model_path = self.config.model_path
        self.num_classes = self.config.num_classes
        self.input_shape = self.config.input_shape
        self.mix_type = self.config.mix_type
        self.cuda = self.config.cuda
        
        # Setup device (GPU or CPU)
        self.use_cuda = bool(self.config.cuda and torch.cuda.is_available())
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # Build color palette for visualization
        self.colors = build_color_palette(self.config.num_classes)
        
        # Initialize the model
        self.generate()

        # Display configuration summary
        show_config(
            model_path=self.config.model_path,
            num_classes=self.config.num_classes,
            input_shape=self.config.input_shape,
            mix_type=self.config.mix_type,
            cuda=self.config.cuda,
            use_cuda=self.use_cuda,
            device=str(self.device),
        )

    def _build_model(self, device: torch.device) -> AmmoniaNet:
        """
        Build and load the AmmoniaNet model from checkpoint.
        
        Args:
            device: PyTorch device to load the model onto (CPU or CUDA).
            
        Returns:
            Loaded AmmoniaNet model in evaluation mode.
        """
        model = AmmoniaNet(num_classes=self.num_classes)
        load_model_state(model, self.model_path, map_location=device)
        model = model.to(device)
        model.eval()
        return model

    def generate(self) -> None:
        """
        Initialize or reinitialize the model for inference.
        """
        self.net = self._build_model(self.device)
        print(f"{self.model_path} model, and classes loaded.")

    def _apply_classification_gate(
        self,
        seg_mask: np.ndarray,
        grade_index: int,
        detection_index: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply classification gating logic to filter classification results.
        
        If no defect is detected (detection_index == 0) or no grade is assigned (grade_index == 0),
        convert class 2 pixels back to class 1 (background) and reset grade to 0.
        
        Args:
            seg_mask: Segmentation mask array.
            grade_index: Predicted grade class index.
            detection_index: Predicted detection class index.
            
        Returns:
            Tuple of (processed segmentation mask as uint8, updated grade_index).
        """
        if detection_index == 0 or grade_index == 0:
            seg_mask = np.where(seg_mask == 2, 1, seg_mask)
            grade_index = 0
        return seg_mask.astype(np.uint8), grade_index

    def _predict_mask(self, image: Image.Image) -> Tuple[Image.Image, np.ndarray, int]:
        """
        Perform multi-task prediction on an image.
        
        Args:
            image: Input PIL Image.
            
        Returns:
            Tuple of (original_image, segmentation_mask, grade_prediction).
        """
        prediction = predict_multitask_image(
            self.net,
            image=image,
            input_shape=self.input_shape,
            device=self.device,
        )
        # Extract segmentation mask from probability predictions
        seg_mask = prediction.seg_probs.argmax(axis=-1)
        # Convert logits to class indices for grade and detection
        grade_index = logits_to_class_index(prediction.grade_logits)
        detection_index = logits_to_class_index(prediction.detection_logits)
        # Apply gating logic
        seg_mask, grade_index = self._apply_classification_gate(seg_mask, grade_index, detection_index)
        return prediction.original_image, seg_mask, grade_index

    def _render_prediction(self, original_image: Image.Image, seg_mask: np.ndarray) -> Image.Image:
        """
        Render the segmentation mask with visualization.
        
        Args:
            original_image: Original input image.
            seg_mask: Segmentation mask (class predictions).
            
        Returns:
            Rendered image (blended, colorized, or foreground only based on mix_type).
            
        Raises:
            ValueError: If mix_type is not 0, 1, or 2.
        """
        if self.mix_type == 0:
            # Blend original image with colorized mask (70% transparency)
            return Image.blend(original_image, colorize_segmentation_mask(seg_mask, self.colors), 0.7)
        if self.mix_type == 1:
            # Return only colorized mask
            return colorize_segmentation_mask(seg_mask, self.colors)
        if self.mix_type == 2:
            # Return only foreground (non-zero mask pixels)
            foreground = (
                np.expand_dims(seg_mask != 0, -1) * np.array(original_image, dtype=np.float32)
            ).astype(np.uint8)
            return Image.fromarray(foreground)
        raise ValueError(f"Unsupported mix_type: {self.mix_type}")

    def detect_image(
        self,
        image: Image.Image,
        count: bool = False,
        name_classes: Optional[Sequence[str]] = None,
    ) -> Tuple[Image.Image, int]:
        """
        Perform detection and segmentation on an image.
        
        Args:
            image: Input PIL Image.
            count: If True, print pixel statistics for each class.
            name_classes: Optional list of class names for statistics output.
            
        Returns:
            Tuple of (rendered_prediction_image, grade_index).
        """
        original_image, seg_mask, grade_index = self._predict_mask(image)
        if count:
            print_class_pixel_stats(seg_mask, self.num_classes, name_classes)
        rendered_image = self._render_prediction(original_image, seg_mask)
        return rendered_image, grade_index
