from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..classifiers.shufflenet_v2 import shufflenet_v2_x1_0
from ..layers.blocks import TensorNormalizer
from ..segmentation.unet import UNet


# Number of input image channels (RGB)
IMAGE_CHANNELS = 3
# Number of toothmark grading classes (None, Lightly, Moderate, Severe)
GRADING_NUM_CLASSES = 4
# Number of detection classes (background, foreground)
DETECTION_NUM_CLASSES = 2
# ImageNet normalization mean for segmentation input
SEGMENTATION_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
# ImageNet normalization standard deviation for segmentation input
SEGMENTATION_NORMALIZE_STD = (0.229, 0.224, 0.225)


class AmmoniaNet(nn.Module):
    """Multi-task neural network for toothmark assessment.
    
    Combines three complementary tasks:
    1. Segmentation: UNet that produces pixel-wise segmentation masks (3-channel output for toothmark regions).
    2. Grading: ShuffleNet classifier that predicts toothmark severity grade (None/Lightly/Moderate/Severe)
       based on concatenated RGB image and segmentation mask.
    3. Detection: ShuffleNet classifier that performs binary detection (foreground/background).
    
    The architecture enables end-to-end learning of toothmark detection, localization, and severity grading.
    """

    def __init__(self, num_classes: int = 3):
        """Initialize AmmoniaNet with segmentation, grading, and detection branches.
        
        Args:
            num_classes: Number of segmentation classes (must be 3 for compatibility with grading head).
                        Each class represents a toothmark region (background, light, moderate, severe).
                        
        Raises:
            ValueError: If num_classes is not 3 (required for grading head concatenation).
        """
        super().__init__()
        self._validate_num_classes(num_classes)

        # Segmentation branch: UNet produces pixel-wise segmentation masks
        self.unet = UNet(num_classes=num_classes)
        # Grading branch: Classifies toothmark severity from RGB + segmentation mask concatenation
        self.shufflenet_grading = shufflenet_v2_x1_0(
            num_classes=GRADING_NUM_CLASSES,
            input_channels=IMAGE_CHANNELS + num_classes,  # 3 (RGB) + 3 (segmentation masks)
        )
        # Detection branch: Binary classification from RGB image (damage present or not)
        self.shufflenet_detection = shufflenet_v2_x1_0(
            num_classes=DETECTION_NUM_CLASSES,
            input_channels=IMAGE_CHANNELS,
        )
        # Normalizer: Standardizes tensors using ImageNet normalization parameters
        self.normalizer = TensorNormalizer(
            mean=SEGMENTATION_NORMALIZE_MEAN,
            std=SEGMENTATION_NORMALIZE_STD,
        )

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through all three task branches.
        
        Processing pipeline:
        1. Generate segmentation masks from input image
        2. Normalize image and segmentation output
        3. Concatenate normalized image with normalized segmentation for grading
        4. Compute grading logits from concatenated features
        5. Compute detection logits from normalized image
        
        Args:
            inputs: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Tuple of three tensors:
                - seg_map: Raw segmentation output (batch_size, num_classes, height, width).
                - grading_logits: Toothmark severity logits (batch_size, 4).
                - detection_logits: Binary damage detection logits (batch_size, 2).
        """
        # Generate pixel-wise segmentation masks
        seg_map = self.unet(inputs)
        # Normalize image for classifier input
        inputs_norm = self.normalizer(inputs)
        # Normalize segmentation output for classifier input
        seg_norm = self.normalizer(seg_map)

        # Concatenate normalized image and segmentation for grading head input
        grading_logits = self.shufflenet_grading(torch.cat((inputs_norm, seg_norm), dim=1))
        # Compute detection logits from normalized image only
        detection_logits = self.shufflenet_detection(inputs_norm)
        return seg_map, grading_logits, detection_logits

    @staticmethod
    def _validate_num_classes(num_classes: int) -> None:
        """Validate that num_classes is compatible with network architecture.
        
        The grading head concatenates 3-channel RGB input with num_classes segmentation output.
        For the architecture to work correctly, num_classes must equal 3 (matching RGB channels).
        
        Args:
            num_classes: Number of segmentation classes to validate.
            
        Raises:
            ValueError: If num_classes is not 3.
        """
        if num_classes != IMAGE_CHANNELS:
            raise ValueError(
                "AmmoniaNet currently expects num_classes == 3 because the grading "
                "head concatenates RGB input with a 3-channel segmentation output."
            )

    def freeze_segmentation_branch(self) -> None:
        """Freeze segmentation branch parameters to prevent weight updates during training.
        
        Useful for transfer learning or fine-tuning scenarios where you want to use
        a pre-trained segmentation model and only train the grading and detection heads.
        """
        self.unet.freeze_segmentation_branch()


__all__ = ["AmmoniaNet"]
