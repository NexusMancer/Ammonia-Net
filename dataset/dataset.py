from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from config.config import DatasetLayoutConfig, get_dataset_config
from utils.preprocessing import cvtColor, preprocess_input, resize_image_pair


# Mapping of class names to integer labels for toothmark classification
CLASS_LABELS: Dict[str, int] = {
    "None": 0,
    "LightlyTooth": 1,
    "ModerateTooth": 2,
    "SevereTooth": 3,
}


@dataclass(frozen=True)
class SamplePaths:
    """Immutable container for image and mask file paths of a single sample.
    
    Attributes:
        image: Path to the image file.
        mask: Path to the segmentation mask file.
    """
    image: Path
    mask: Path


class AMNDataset(Dataset):
    """PyTorch Dataset for Ammonia-Net segmentation and classification tasks.
    
    Loads images and corresponding segmentation masks, applies dynamic augmentations
    during training, and provides both segmentation labels and toothmark class labels.
    Supports various data augmentation techniques including HSV color augmentation,
    random flipping, and resized crop operations.
    """
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
        transform=None,
        config: Optional[DatasetLayoutConfig] = None,
    ):
        """Initialize the AMNDataset.
        
        Args:
            annotation_lines: List of annotation lines (sample names extracted from first token).
            input_shape: Target shape (height, width) for resized images and masks.
            num_classes: Number of segmentation classes.
            train: Boolean indicating training mode (enables augmentations if True).
            dataset_path: Root path to the dataset directory.
            transform: Optional torchvision transforms to apply during training.
            config: Optional dataset layout configuration. Uses default if not provided.
        """
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = Path(dataset_path)
        self.config = config or get_dataset_config()
        self.image_dir = self.config.image_dir(self.dataset_path)
        self.mask_dir = self.config.segmentation_dir(self.dataset_path)
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index):
        """Load and return a single sample with image, mask, segmentation labels, and class label.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple of (image_array, mask_array, seg_labels, class_label):
                - image_array: Preprocessed image as numpy array (channels, height, width).
                - mask_array: Segmentation mask as numpy array (height, width).
                - seg_labels: One-hot encoded segmentation labels (height, width, num_classes+1).
                - class_label: Toothmark classification label (0-3).
        """
        # Get sample name from annotation
        sample_name = self._get_sample_name(index)
        # Load image and mask files
        image, mask = self._load_sample(sample_name)
        # Apply data augmentation and resize
        image, mask = self._prepare_sample(image, mask)

        # Convert image to preprocessed numpy array with channels-first format
        image_array = np.transpose(
            preprocess_input(np.asarray(image, dtype=np.float64)),
            (2, 0, 1),
        )
        # Convert mask to numpy array with invalid class indices handled
        mask_array = np.array(mask, dtype=np.uint8, copy=True)
        mask_array[mask_array >= self.num_classes] = self.num_classes

        # Create one-hot encoded segmentation labels
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[mask_array.reshape(-1)]
        seg_labels = seg_labels.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1)
        )
        # Extract toothmark classification label from sample name
        class_label = self._parse_class_label(sample_name)

        return image_array, mask_array, seg_labels, class_label

    def _get_sample_name(self, index: int) -> str:
        """Extract sample name from annotation line.
        
        Args:
            index: Index in annotation_lines.
            
        Returns:
            Sample name (first token of the annotation line).
        """
        return self.annotation_lines[index].split()[0]

    def _resolve_sample_paths(self, sample_name: str) -> SamplePaths:
        """Resolve and validate file paths for image and mask.
        
        Args:
            sample_name: Name of the sample.
            
        Returns:
            SamplePaths containing validated image and mask file paths.
            
        Raises:
            FileNotFoundError: If image or mask file doesn't exist.
        """
        # Construct file paths using sample name and configured extensions
        sample_paths = SamplePaths(
            image=self.image_dir / f"{sample_name}{self.config.image_extension}",
            mask=self.mask_dir / f"{sample_name}{self.config.mask_extension}",
        )
        # Validate that both files exist
        if not sample_paths.image.exists():
            raise FileNotFoundError(f"Image file not found: {sample_paths.image}")
        if not sample_paths.mask.exists():
            raise FileNotFoundError(f"Mask file not found: {sample_paths.mask}")
        return sample_paths

    def _load_sample(self, sample_name: str) -> Tuple[Image.Image, Image.Image]:
        """Load image and mask files using PIL.
        
        Args:
            sample_name: Name of the sample.
            
        Returns:
            Tuple of (PIL Image, PIL Image mask).
        """
        # Resolve file paths and validate existence
        sample_paths = self._resolve_sample_paths(sample_name)
        # Load image and create a copy to avoid issues when file handle closes
        with Image.open(sample_paths.image) as image:
            loaded_image = image.copy()
        # Load mask and create a copy
        with Image.open(sample_paths.mask) as mask:
            loaded_mask = mask.copy()
        return loaded_image, loaded_mask

    def _prepare_sample(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Prepare image and mask by applying color conversion, augmentations, and resizing.
        
        Args:
            image: PIL Image to prepare.
            mask: PIL Image mask to prepare.
            
        Returns:
            Tuple of (prepared image, prepared mask).
        """
        # Convert image to RGB color space
        image = cvtColor(image)
        # Convert mask to numpy array and back to PIL for consistency
        mask = Image.fromarray(np.array(mask, dtype=np.uint8, copy=True))

        if self.train:
            # During training: apply augmentation transforms and HSV color augmentation
            image, mask = self._apply_training_transforms(image, mask)
            image = self._apply_hsv_augmentation(image)
        else:
            # During validation: resize with padding only
            image, mask = self._resize_pair_with_padding(image, mask)

        # Final resize check to ensure target dimensions
        if image.size != self._target_size or mask.size != self._target_size:
            image, mask = self._resize_pair_with_padding(image, mask)

        return image, mask

    @property
    def _target_size(self) -> Tuple[int, int]:
        """Get target image size (width, height) from input_shape.
        
        Returns:
            Tuple of (width, height) for PIL Image operations.
        """
        height, width = self.input_shape
        return width, height

    def _resize_pair_with_padding(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """Resize image and mask while maintaining aspect ratio, then pad to target size.
        
        Resizes the image/mask such that the largest dimension fits within the target,
        then centers them on padded canvases. Delegates to the shared preprocessing utility.
        
        Args:
            image: PIL Image to resize.
            mask: PIL Image mask to resize.
            
        Returns:
            Tuple of (resized_padded_image, resized_padded_mask).
        """
        return resize_image_pair(image, mask, self._target_size)

    def _apply_training_transforms(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply training augmentations from configured transform pipeline.
        
        Args:
            image: PIL Image to augment.
            mask: PIL Image mask to augment.
            
        Returns:
            Tuple of (augmented_image, augmented_mask).
        """
        # If no transforms configured, just resize and pad
        if self.transform is None:
            return self._resize_pair_with_padding(image, mask)
        # Apply transforms while keeping image and mask synchronized
        return self._apply_paired_transform(image, mask, self.transform)

    def _apply_paired_transform(
        self,
        image: Image.Image,
        mask: Image.Image,
        transform,
    ) -> Tuple[Image.Image, Image.Image]:
        """Recursively apply paired transforms to image and mask while keeping them synchronized.
        
        Handles Compose transforms by recursing through sub-transforms, and implements
        custom logic for spatial transforms to ensure mask uses appropriate interpolation.
        
        Args:
            image: PIL Image to transform.
            mask: PIL Image mask to transform.
            transform: Transform or transform.Compose object.
            
        Returns:
            Tuple of (transformed_image, transformed_mask).
            
        Raises:
            TypeError: If transform type is not supported.
        """
        # Recursively handle Compose transforms by applying each sub-transform
        if isinstance(transform, transforms.Compose):
            for sub_transform in transform.transforms:
                image, mask = self._apply_paired_transform(image, mask, sub_transform)
            return image, mask

        # Handle RandomResizedCrop: apply same crop to both image and mask
        if isinstance(transform, transforms.RandomResizedCrop):
            # Get random crop parameters
            crop_top, crop_left, crop_height, crop_width = transform.get_params(
                image,
                transform.scale,
                transform.ratio,
            )
            # Apply crop and resize with appropriate interpolation modes
            image = TF.resized_crop(
                image,
                crop_top,
                crop_left,
                crop_height,
                crop_width,
                transform.size,
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.resized_crop(
                mask,
                crop_top,
                crop_left,
                crop_height,
                crop_width,
                transform.size,
                interpolation=InterpolationMode.NEAREST,
            )
            return image, mask

        # Handle RandomHorizontalFlip: apply same horizontal flip to both
        if isinstance(transform, transforms.RandomHorizontalFlip):
            if torch.rand(1).item() < transform.p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            return image, mask

        # Handle RandomVerticalFlip: apply same vertical flip to both
        if isinstance(transform, transforms.RandomVerticalFlip):
            if torch.rand(1).item() < transform.p:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            return image, mask

        # Handle Resize: resize both with appropriate interpolation modes
        if isinstance(transform, transforms.Resize):
            image = TF.resize(
                image,
                transform.size,
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.resize(
                mask,
                transform.size,
                interpolation=InterpolationMode.NEAREST,
            )
            return image, mask

        # Handle CenterCrop: apply same center crop to both
        if isinstance(transform, transforms.CenterCrop):
            return TF.center_crop(image, transform.size), TF.center_crop(mask, transform.size)

        # Unsupported transform type
        raise TypeError(
            "Unsupported transform in AMNDataset. Supported transforms are "
            "Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, "
            f"Resize, and CenterCrop. Got: {type(transform).__name__}."
        )

    def _apply_hsv_augmentation(
        self,
        image: Image.Image,
        hue: float = 0.1,
        sat: float = 0.7,
        val: float = 0.3,
    ) -> Image.Image:
        """Apply random HSV color augmentation to image.
        
        Performs random shifts in Hue, Saturation, and Value channels to improve
        model robustness to color variations.
        
        Args:
            image: PIL Image to augment.
            hue: Maximum hue shift range. Default 0.1.
            sat: Maximum saturation multiplier range. Default 0.7.
            val: Maximum value multiplier range. Default 0.3.
            
        Returns:
            PIL Image with HSV augmentation applied.
        """
        # Convert PIL image to numpy array
        image_data = np.asarray(image, dtype=np.uint8)
        # Generate random multipliers for HSV channels: value between (1-range) and (1+range)
        random_gain = np.random.uniform(-1, 1, 3) * np.array([hue, sat, val]) + 1

        # Convert RGB to HSV color space and split channels
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # Create lookup tables for fast color transformation
        lookup = np.arange(0, 256, dtype=random_gain.dtype)
        # Apply transformations with clipping to valid ranges
        lut_hue = ((lookup * random_gain[0]) % 180).astype(dtype)  # Hue range [0, 180] in OpenCV
        lut_sat = np.clip(lookup * random_gain[1], 0, 255).astype(dtype)
        lut_val = np.clip(lookup * random_gain[2], 0, 255).astype(dtype)

        # Apply lookup tables to HSV channels
        image_data = cv2.merge(
            (
                cv2.LUT(hue_channel, lut_hue),
                cv2.LUT(sat_channel, lut_sat),
                cv2.LUT(val_channel, lut_val),
            )
        )
        # Convert back from HSV to RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return Image.fromarray(image_data)

    def _parse_class_label(self, sample_name: str) -> int:
        """Extract toothmark classification label from sample name prefix.
        
        Expects sample names to start with a class prefix (e.g., 'None-sample1', 'SevereTooth-sample2').
        
        Args:
            sample_name: Name of the sample.
            
        Returns:
            Integer class label (0-3).
            
        Raises:
            ValueError: If the class prefix is not recognized.
        """
        # Extract class prefix before first hyphen
        class_name = sample_name.split("-", maxsplit=1)[0]
        # Validate and lookup class label
        if class_name not in CLASS_LABELS:
            raise ValueError(
                f"Unknown class prefix '{class_name}' in sample name '{sample_name}'. "
                f"Expected one of: {', '.join(CLASS_LABELS)}."
            )
        return CLASS_LABELS[class_name]


def unet_dataset_collate(batch):
    """Collate function for DataLoader to stack batch samples into tensors.
    
    Converts a list of (image, mask, seg_labels, class_label) tuples into batch tensors.
    
    Args:
        batch: List of samples from AMNDataset.
        
    Returns:
        Tuple of (images_tensor, masks_tensor, seg_labels_tensor, class_labels_tensor):
            - images_tensor: Stacked images (batch_size, channels, height, width) as FloatTensor.
            - masks_tensor: Stacked masks (batch_size, height, width) as LongTensor.
            - seg_labels_tensor: Stacked segmentation labels (batch_size, height, width, num_classes+1) as FloatTensor.
            - class_labels_tensor: Class labels (batch_size,) as LongTensor.
    """
    # Separate components from batch samples
    images = []
    pngs = []
    seg_labels = []
    class_labels = []
    for image, png, labels, class_label in batch:
        images.append(image)
        pngs.append(png)
        seg_labels.append(labels)
        class_labels.append(class_label)

    # Stack all components and convert to appropriate tensor types
    images = torch.from_numpy(np.asarray(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.asarray(pngs)).long()
    seg_labels = torch.from_numpy(np.asarray(seg_labels)).type(torch.FloatTensor)
    class_labels = torch.as_tensor(class_labels, dtype=torch.long)

    return images, pngs, seg_labels, class_labels
