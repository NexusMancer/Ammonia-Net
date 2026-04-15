from __future__ import annotations

"""Shared single-image inference helpers for prediction and evaluation flows."""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.preprocessing import cvtColor, preprocess_input, resize_image


@dataclass(frozen=True)
class PreparedImage:
    """Container for preprocessed image data ready for model inference.
    
    Attributes:
        original_image: The original input image in PIL format.
        image_tensor: Preprocessed image as a PyTorch tensor with batch dimension.
        original_shape: Height and width of the original input image.
        resized_shape: Height and width after resizing for model input.
    """
    original_image: Image.Image
    image_tensor: torch.Tensor
    original_shape: Tuple[int, int]
    resized_shape: Tuple[int, int]


@dataclass(frozen=True)
class MultiTaskImagePrediction:
    """Container for multi-task model predictions on a single image.
    
    Stores the output predictions from a multi-task model that performs
    segmentation, quality grading, and defect detection simultaneously.
    
    Attributes:
        original_image: The original input image in PIL format.
        seg_probs: Class probability map for semantic segmentation (H x W x C).
        grade_logits: Raw logits for quality grade classification.
        detection_logits: Raw logits for defect detection classification.
    """
    original_image: Image.Image
    seg_probs: np.ndarray
    grade_logits: torch.Tensor
    detection_logits: torch.Tensor


def prepare_image_for_model(
    image: Image.Image,
    input_shape: Tuple[int, int],
    device: torch.device,
) -> PreparedImage:
    """Prepare an image for model inference with preprocessing and resizing.
    
    Converts a PIL image to a PyTorch tensor with proper normalization and resizing.
    Handles padding to match the model's expected input shape while preserving aspect ratio.
    
    Args:
        image: Input image in PIL format.
        input_shape: Target shape (height, width) for model input.
        device: PyTorch device (CPU or GPU) to place the tensor on.
    
    Returns:
        PreparedImage containing the processed tensor and shape information.
    """
    rgb_image = cvtColor(image)
    original_image = rgb_image.copy()
    original_h, original_w = np.array(original_image).shape[:2]

    resized_image, resized_w, resized_h = resize_image(
        original_image,
        (int(input_shape[1]), int(input_shape[0])),
    )
    image_array = preprocess_input(np.array(resized_image, dtype=np.float32))
    image_tensor = torch.from_numpy(
        np.expand_dims(np.transpose(image_array, (2, 0, 1)), 0)
    ).to(device)
    return PreparedImage(
        original_image=original_image,
        image_tensor=image_tensor,
        original_shape=(original_h, original_w),
        resized_shape=(resized_h, resized_w),
    )


def forward_multitask_image(
    net,
    image_tensor: torch.Tensor,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Execute forward pass on a preprocessed image tensor.
    
    Runs the model in inference mode with gradient computation disabled.
    Converts segmentation logits to probability maps via softmax.
    
    Args:
        net: The multi-task model to run inference on.
        image_tensor: Preprocessed image tensor with batch dimension.
    
    Returns:
        Tuple of (seg_probs, grade_logits, detection_logits).
    """
    with torch.inference_mode():
        seg_logits, grade_logits, detection_logits = net(image_tensor)

    seg_probs = F.softmax(seg_logits[0].permute(1, 2, 0), dim=-1).cpu().numpy()
    return seg_probs, grade_logits, detection_logits


def remove_padding_and_resize(
    seg_probs: np.ndarray,
    input_shape: Tuple[int, int],
    resized_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Remove padding and resize segmentation probabilities to original image size.
    
    Reverses the preprocessing transformations: removes the padding that was added
    to match the model input shape and resizes back to the original image dimensions.
    
    Args:
        seg_probs: Segmentation probability map from model (input_shape dimensions).
        input_shape: Model input shape (height, width).
        resized_shape: Resized image shape before padding.
        original_shape: Original input image shape (height, width).
    
    Returns:
        Segmentation probabilities resized to original image dimensions.
    """
    input_h, input_w = (int(value) for value in input_shape)
    resized_h, resized_w = resized_shape
    original_h, original_w = original_shape

    top = int((input_h - resized_h) // 2)
    left = int((input_w - resized_w) // 2)
    seg_probs = seg_probs[top:top + resized_h, left:left + resized_w]
    return cv2.resize(seg_probs, (original_w, original_h), interpolation=cv2.INTER_LINEAR)


def predict_multitask_image(
    net,
    image: Image.Image,
    input_shape: Tuple[int, int],
    device: torch.device,
) -> MultiTaskImagePrediction:
    """Run complete multi-task inference pipeline on a single image.
    
    Orchestrates the full prediction workflow:
    1. Prepares the image with resizing and normalization
    2. Runs model forward pass
    3. Resizes predictions back to original image dimensions
    
    Args:
        net: The multi-task model.
        image: Input image in PIL format.
        input_shape: Model input shape (height, width).
        device: PyTorch device for computation.
    
    Returns:
        MultiTaskImagePrediction containing all predictions and original image.
    """
    prepared = prepare_image_for_model(image, input_shape, device)
    seg_probs, grade_logits, detection_logits = forward_multitask_image(
        net,
        prepared.image_tensor,
    )
    seg_probs = remove_padding_and_resize(
        seg_probs,
        input_shape,
        prepared.resized_shape,
        prepared.original_shape,
    )
    return MultiTaskImagePrediction(
        original_image=prepared.original_image,
        seg_probs=seg_probs,
        grade_logits=grade_logits,
        detection_logits=detection_logits,
    )


def logits_to_class_index(logits: torch.Tensor) -> int:
    """Convert raw logits to predicted class index using argmax.
    
    Extracts the class with the highest logit value.
    
    Args:
        logits: Raw model outputs (torch.Tensor).
    
    Returns:
        Predicted class index as an integer.
    """
    return int(torch.argmax(logits.detach().cpu(), dim=-1).reshape(-1)[0].item())


__all__ = [
    "MultiTaskImagePrediction",
    "PreparedImage",
    "forward_multitask_image",
    "logits_to_class_index",
    "predict_multitask_image",
    "prepare_image_for_model",
    "remove_padding_and_resize",
]
