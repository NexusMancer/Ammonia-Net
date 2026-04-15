from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch.nn as nn
from torch import Tensor


# Type alias for VGG configuration items (channel counts or max pooling marker)
VGGConfigItem = Union[int, str]
# Slicing indices for extracting VGG16 stages from the features module
# Format: (start_index, end_index) for each of the 5 stages
_VGG16_STAGE_SLICES: Tuple[Tuple[int, int], ...] = (
    (0, 4),      # Stage 1: 2 conv layers
    (4, 9),      # Stage 2: 2 conv layers
    (9, 16),     # Stage 3: 3 conv layers
    (16, 23),    # Stage 4: 3 conv layers
    (23, -1),    # Stage 5: 3 conv layers
)
# VGG16 architecture configuration: channel numbers and max pooling markers
_VGG16_CFG: List[VGGConfigItem] = [
    # Block 1: 64 channels
    64, 64, "M",
    # Block 2: 128 channels
    128, 128, "M",
    # Block 3: 256 channels
    256, 256, 256, "M",
    # Block 4: 512 channels
    512, 512, 512, "M",
    # Block 5: 512 channels
    512, 512, 512, "M",
]


def make_layers(
    cfg: Sequence[VGGConfigItem],
    batch_norm: bool = False,
    in_channels: int = 3,
) -> nn.Sequential:
    """Build a VGG convolutional stack from a configuration list.
    
    Constructs a sequence of convolutional layers with optional batch normalization
    and ReLU activations, interleaved with max pooling layers as specified in the config.
    
    Args:
        cfg: VGG architecture configuration. Contains integers for channel counts
             and 'M' strings for max pooling layers.
        batch_norm: Whether to include batch normalization after convolutions. Default False.
        in_channels: Number of input channels (e.g., 3 for RGB images). Default 3.
        
    Returns:
        nn.Sequential module containing the complete convolutional stack.
        
    Raises:
        ValueError: If config contains unsupported items (not int or 'M').
    """
    layers: List[nn.Module] = []
    for value in cfg:
        # Add max pooling layer
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        # Validate config item is a valid channel count
        if not isinstance(value, int):
            raise ValueError(f"unsupported VGG config item: {value!r}")

        # Create convolution layer with 3x3 kernel and padding=1 (preserves spatial dimensions)
        conv2d = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
        if batch_norm:
            # Add batch norm and ReLU activation after convolution
            layers.extend([conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
        else:
            # Add ReLU activation only
            layers.extend([conv2d, nn.ReLU(inplace=True)])
        # Update input channels for next layer
        in_channels = value
    return nn.Sequential(*layers)


class VGGEncoder(nn.Module):
    """VGG feature extractor that returns multi-scale skip-connection maps.
    
    Segments a VGG network into 5 stages and extracts features after each stage.
    These multi-scale features are commonly used as skip connections in U-Net style
    architectures or for feature pyramid networks.
    
    The extracted features have progressively larger receptive fields and smaller spatial
    dimensions, making them ideal for decoder architectures that use skip connections.
    """

    def __init__(self, features: nn.Sequential):
        """Initialize VGGEncoder by segmenting VGG features into stages.
        
        Args:
            features: Complete VGG feature extraction module (from conv1 to final layers).
        """
        super().__init__()
        # Segment the feature module into 5 stages using predefined slice indices
        self.stages = nn.ModuleList([
            features[start:end] for start, end in _VGG16_STAGE_SLICES
        ])
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Extract multi-scale features from input by passing through all stages.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Tuple of 5 feature maps from each stage:
                - stage1: High resolution features with small receptive field
                - stage2: Features at 1/2 spatial resolution
                - stage3: Features at 1/4 spatial resolution
                - stage4: Features at 1/8 spatial resolution
                - stage5: Low resolution features with large receptive field
            Each feature map can be used as skip connections in decoder networks.
        """
        features: List[Tensor] = []
        current = x
        # Pass input through each stage and collect output
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        return tuple(features)

    def _initialize_weights(self) -> None:
        """Initialize network weights using standard initialization schemes.
        
        Uses:
        - Kaiming He initialization for convolutional layers (normal distribution)
        - Constant initialization for batch normalization and bias terms
        - Normal distribution for fully connected layer weights
        """
        for module in self.modules():
            # Convolution layers: Kaiming He initialization
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # Batch normalization: Set scale to 1 and bias to 0
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            # Fully connected layers: Normal distribution and zero bias
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def build_vgg16_encoder(in_channels: int = 3) -> VGGEncoder:
    """Create a VGG16 encoder with multi-scale feature extraction.
    
    Builds a VGG16 network configured for feature extraction (no fully connected
    classification layers) and wraps it with VGGEncoder for multi-scale output.
    
    Args:
        in_channels: Number of input channels (3 for RGB). Default 3.
                    Can be adjusted for different input modalities (e.g., grayscale=1).
        
    Returns:
        VGGEncoder instance with VGG16 architecture and standard configuration.
    """
    return VGGEncoder(make_layers(_VGG16_CFG, batch_norm=False, in_channels=in_channels))


# Public API: Export VGG encoder components and factory function
__all__ = ["VGGEncoder", "build_vgg16_encoder", "make_layers"]
