"""Image preprocessing utilities for model inference and training.

This module provides functions to preprocess images for neural network input,
including format conversion, resizing, and normalization.

Image Processing Pipeline:
    Original Image (PIL)
        ↓
    cvtColor()          <- Convert to RGB format
        ↓
    resize_image()      <- Resize to target size (preserve aspect ratio, add gray borders)
        ↓
    np.array()          <- Convert to numpy array
        ↓
    preprocess_input()  <- Normalize [0,255] → [0,1]
        ↓
    transpose()         <- Change channel order (H,W,C) → (C,H,W)
        ↓
    torch.Tensor        <- Convert to PyTorch tensor
        ↓
    Model input ready

Key Functions:
    - cvtColor: Converts images to RGB format
    - resize_image: Resizes images with aspect ratio preservation (letterboxing)
    - preprocess_input: Normalizes pixel values to [0, 1] range
    - colorize_segmentation_mask: Visualizes segmentation results
    - print_class_pixel_stats: Displays class-wise pixel statistics
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image


RGB_IMAGE_MODE = "RGB"
MASK_IMAGE_MODE = "L"
LETTERBOX_FILL_COLOR = (128, 128, 128)
MASK_PADDING_VALUE = 0
# Color palette for 3-class semantic segmentation
SEGMENTATION_COLORS = [
    (0, 0, 0),        # Black - Class 0
    (128, 0, 0),      # Dark red - Class 1
    (0, 128, 0),      # Dark green - Class 2
]


def cvtColor(image):
    """Convert image to RGB format while preserving already-compatible inputs.
    
    Handles various input types (PIL Image, numpy array) and ensures output is RGB.
    Returns the input unchanged if already in RGB format.
    
    Args:
        image: Input image as PIL Image or numpy array.
    
    Returns:
        PIL Image in RGB format.
    
    Raises:
        TypeError: If image type is not supported.
    """
    if isinstance(image, Image.Image):
        # If PIL Image is already RGB, return as-is
        if image.mode == RGB_IMAGE_MODE:
            return image
        # Convert PIL Image to RGB
        return image.convert(RGB_IMAGE_MODE)

    # Check if numpy array with 3 channels (RGB format)
    image_shape = np.shape(image)
    if len(image_shape) == 3 and image_shape[2] == 3:
        return image

    # Attempt to convert using convert method if available
    if hasattr(image, "convert"):
        return image.convert(RGB_IMAGE_MODE)
    # Raise error for unsupported image types
    raise TypeError(f"Unsupported image type for cvtColor: {type(image)!r}")


def _calculate_resize_params(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[float, int, int, Tuple[int, int]]:
    """Calculate parameters for resizing image while preserving aspect ratio.
    
    Internal helper function that computes the scale factor, resized dimensions,
    and offset needed for aspect-ratio-preserving resize with padding.
    
    Args:
        image_width: Current image width.
        image_height: Current image height.
        target_width: Target canvas width.
        target_height: Target canvas height.
    
    Returns:
        Tuple of (scale, resized_width, resized_height, offset) where offset is
        (left, top) for centering the resized image on the target canvas.
    
    Raises:
        ValueError: If dimensions are invalid.
    """
    # Validate dimensions
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: ({image_width}, {image_height})")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target size: ({target_width}, {target_height})")

    # Calculate scale factor to fit image within target while preserving aspect ratio
    scale = min(target_width / image_width, target_height / image_height)
    # Calculate resized dimensions (ensure at least 1 pixel)
    resized_width = max(int(image_width * scale), 1)
    resized_height = max(int(image_height * scale), 1)
    # Calculate offset to center the resized image on the canvas
    offset = (
        (target_width - resized_width) // 2,
        (target_height - resized_height) // 2,
    )
    return scale, resized_width, resized_height, offset


def resize_image(
    image: Image.Image,
    size: Tuple[int, int],
) -> Tuple[Image.Image, int, int]:
    """Resize image to target size while preserving aspect ratio using letterboxing.
    
    Scales the image to fit within the target dimensions while maintaining
    aspect ratio, then pads with gray color to reach exact target dimensions.
    
    Args:
        image: Input image in PIL format.
        size: Target size as (width, height).
    
    Returns:
        Tuple of (resized_padded_image, resized_width, resized_height) where
        resized_width/height are the dimensions before padding.
    
    Raises:
        ValueError: If image or target size dimensions are invalid.
    """
    # Ensure image is in RGB format
    image = cvtColor(image)
    target_width, target_height = (int(value) for value in size)
    image_width, image_height = image.size

    # Calculate resize parameters using shared logic
    _, resized_width, resized_height, offset = _calculate_resize_params(
        image_width, image_height, target_width, target_height
    )

    # Resize image using high-quality bicubic interpolation
    resized_image = image.resize((resized_width, resized_height), Image.BICUBIC)
    # Create blank canvas with target dimensions filled with gray padding color
    canvas = Image.new(RGB_IMAGE_MODE, (target_width, target_height), LETTERBOX_FILL_COLOR)
    # Paste resized image onto the center of the canvas
    canvas.paste(resized_image, offset)

    return canvas, resized_width, resized_height


def resize_image_pair(
    image: Image.Image,
    mask: Image.Image,
    size: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    """Resize image and mask pair while preserving aspect ratio, then pad to target size.
    
    Simultaneously resizes an image and its segmentation mask while maintaining aspect ratio.
    Uses BICUBIC interpolation for images and NEAREST for masks to preserve label values.
    
    Args:
        image: PIL Image to resize (RGB).
        mask: PIL Image mask to resize (grayscale/L mode).
        size: Target size as (width, height).
    
    Returns:
        Tuple of (resized_padded_image, resized_padded_mask).
    
    Raises:
        ValueError: If image or target size dimensions are invalid.
    """
    # Ensure image is in RGB format
    image = cvtColor(image)
    target_width, target_height = (int(value) for value in size)
    image_width, image_height = image.size

    # Calculate resize parameters using shared logic
    _, resized_width, resized_height, offset = _calculate_resize_params(
        image_width, image_height, target_width, target_height
    )

    # Resize image using BICUBIC interpolation
    resized_image = image.resize((resized_width, resized_height), Image.BICUBIC)
    # Resize mask using NEAREST to preserve label values
    resized_mask = mask.resize((resized_width, resized_height), Image.NEAREST)

    # Create padded canvases with target size
    canvas_image = Image.new(RGB_IMAGE_MODE, (target_width, target_height), LETTERBOX_FILL_COLOR)
    canvas_mask = Image.new(MASK_IMAGE_MODE, (target_width, target_height), MASK_PADDING_VALUE)

    # Paste resized image and mask onto their respective canvases
    canvas_image.paste(resized_image, offset)
    canvas_mask.paste(resized_mask, offset)

    return canvas_image, canvas_mask


def preprocess_input(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values from [0, 255] to [0, 1] range.
    
    Standard preprocessing step for neural network inputs.
    
    Args:
        image: Image array with pixel values in range [0, 255].
    
    Returns:
        Normalized image array with pixel values in range [0, 1].
    """
    # Divide by 255 to normalize to [0, 1] range
    return image / 255.0


def build_color_palette(num_classes: int = 3) -> Sequence[Tuple[int, int, int]]:
    """Return color palette for semantic segmentation visualization.
    
    Provides predefined RGB colors for up to 3 segmentation classes.
    
    Args:
        num_classes: Number of classes (default 3, max 3).
    
    Returns:
        Sequence of RGB color tuples, one per class.
    
    Raises:
        ValueError: If num_classes exceeds 3.
    """
    # Validate requested number of classes
    if num_classes > len(SEGMENTATION_COLORS):
        raise ValueError(f"Requested {num_classes} colors but only {len(SEGMENTATION_COLORS)} available.")
    # Return colors for requested number of classes
    return SEGMENTATION_COLORS[:num_classes]


def colorize_segmentation_mask(
    seg_mask: np.ndarray,
    colors: Sequence[Tuple[int, int, int]],
) -> Image.Image:
    """Convert a segmentation mask to a colored image for visualization.
    
    Maps each class index in the mask to its corresponding RGB color,
    creating a visualizable segmentation overlay.
    
    Args:
        seg_mask: Segmentation mask with shape (height, width) containing class indices.
        colors: Sequence of RGB color tuples, indexed by class ID.
    
    Returns:
        Colored segmentation image as PIL Image.
    """
    # Convert color list to numpy array for efficient indexing
    color_array = np.array(colors, dtype=np.uint8)
    # Flatten mask, index into color array, then reshape back to (H, W, 3)
    seg_img = color_array[np.reshape(seg_mask, [-1])].reshape(seg_mask.shape[0], seg_mask.shape[1], -1)
    # Convert numpy array to PIL Image
    return Image.fromarray(seg_img.astype(np.uint8))


def print_class_pixel_stats(
    seg_mask: np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """
    Print pixel-level statistics for each class in a segmentation mask.
    
    This function analyzes the distribution of pixels across different classes
    in a segmentation map and displays a formatted table showing the count and
    percentage ratio for each class.
    """
    # Use provided class names, or generate default numeric names (0, 1, 2, ...)
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    
    # Validate that enough class names are provided
    if len(names) < num_classes:
        raise ValueError(f"class_names must provide at least {num_classes} entries.")

    # Initialize array to store pixel counts for each class
    classes_nums = np.zeros([num_classes])
    # Calculate total number of pixels in the segmentation mask (height × width)
    total_points_num = seg_mask.shape[0] * seg_mask.shape[1]
    
    # Print table header with borders
    print("-" * 63)
    print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
    print("-" * 63)
    
    # Iterate through each class and calculate statistics
    for class_index in range(num_classes):
        # Count pixels belonging to current class
        num = np.sum(seg_mask == class_index)
        # Calculate percentage ratio of this class relative to total pixels
        ratio = num / total_points_num * 100
        
        # Only display classes that have at least one pixel
        if num > 0:
            print("|%25s | %15s | %14.2f%%|" % (str(names[class_index]), str(num), ratio))
            print("-" * 63)
        
        # Store the pixel count for this class
        classes_nums[class_index] = num
    
    # Print final summary of total pixel counts per class
    print("classes_nums:", classes_nums)

