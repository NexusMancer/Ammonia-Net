from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..encoders.vgg_encoder import build_vgg16_encoder
from ..layers.blocks import DoubleConv, freeze_module


# Decoder block specifications: (skip_channels, in_channels, out_channels) for each decoder stage
# These specs define the architecture for progressively upsampling and refining features
_DECODER_SPECS = (
    (512, 512, 512),
    (256, 512, 256),
    (128, 256, 128),
    (64, 128, 64),
)


class UNetDecoderBlock(nn.Module):
    """Single decoder block for U-Net architecture.
    
    Performs upsampling followed by concatenation with skip connection and two sequential convolutions.
    This forms the basic building block for the U-Net decoder path during upsampling.
    
    Architecture:
    1. Upsample input by 2x using bilinear interpolation
    2. Concatenate with skip connection from encoder (channels: skip_channels + upsampled_channels)
    3. Apply DoubleConv to fuse and process concatenated features
    """

    def __init__(self, skip_channels: int, in_channels: int, out_channels: int):
        """Initialize UNet decoder block.
        
        Args:
            skip_channels: Number of channels in the skip connection from encoder.
            in_channels: Number of channels in the input feature map to upsample.
            out_channels: Number of output channels after convolution.
        """
        super().__init__()
        # Bilinear upsampling to increase spatial resolution by 2x
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # Concatenated feature dimension: skip_channels + upsampled_in_channels
        # DoubleConv processes the concatenated features
        self.block = DoubleConv(skip_channels + in_channels, out_channels)

    def forward(self, skip: Tensor, x: Tensor) -> Tensor:
        """Forward pass through decoder block.
        
        Args:
            skip: Skip connection tensor from encoder (lower resolution feature map).
            x: Input feature map to be upsampled.
            
        Returns:
            Processed output tensor after upsampling, concatenation, and convolution.
        """
        # Upsample input, concatenate with skip connection along channel dimension
        upsampled = self.up(x)
        # Concatenate skip connection and upsampled features: [skip, upsampled]
        concatenated = torch.cat([skip, upsampled], dim=1)
        # Apply double convolution to process concatenated features
        return self.block(concatenated)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.
    
    Combines a VGG16-based encoder with a symmetric decoder to produce pixel-wise predictions.
    The encoder extracts multi-scale features, and the decoder progressively upsamples while 
    combining features from corresponding encoder stages via skip connections.
    
    Architecture Overview:
    - Encoder: VGG16 with 5 stages, extracting features at multiple scales
    - Decoder: 4 upsampling stages with skip connections from encoder
    - Classifier: 1x1 convolution to map to num_classes
    """

    def __init__(self, num_classes: int = 21):
        """Initialize U-Net segmentation model.
        
        Args:
            num_classes: Number of output classes for semantic segmentation. Default 21 (VOC dataset).
        """
        super().__init__()
        # VGG16 encoder: extracts 5 levels of features with decreasing spatial resolution
        self.encoder = build_vgg16_encoder()
        # Decoder blocks (4 stages): progressively upsample from deepest to shallowest level
        # Each block concatenates encoder skip connection and applies DoubleConv
        self.decoder_blocks = nn.ModuleList([
            UNetDecoderBlock(skip_channels, in_channels, out_channels)
            for skip_channels, in_channels, out_channels in _DECODER_SPECS
        ])
        # Final 1x1 convolution to produce per-pixel class predictions
        # Maps from final decoder features to num_classes channels
        self.classifier = nn.Conv2d(_DECODER_SPECS[-1][-1], num_classes, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through U-Net for semantic segmentation.
        
        Args:
            inputs: Input image tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Segmentation logits of shape (batch_size, num_classes, height, width).
        """
        # Encoder: extract multi-scale features from VGG16 backbone
        # feat1 (shallowest, highest resolution) to feat5 (deepest, lowest resolution)
        feat1, feat2, feat3, feat4, feat5 = self.encoder(inputs)
        
        # Start from deepest features
        x = feat5
        
        # Decoder: progressively upsample and combine with encoder skip connections
        # Process from deep to shallow: feat5 -> feat4 -> feat3 -> feat2 -> feat1
        for skip, block in zip((feat4, feat3, feat2, feat1), self.decoder_blocks):
            x = block(skip, x)
        
        # Final classification: map to per-pixel class predictions
        return self.classifier(x)

    def freeze_segmentation_branch(self) -> None:
        """Freeze all parameters in the U-Net model.
        
        Disables gradient computation for entire network, useful for:
        - Using pre-trained U-Net without fine-tuning
        - Using as fixed feature extractor in multi-task learning architectures
        - Transfer learning where segmentation module shouldn't be updated
        """
        freeze_module(self)


# Public API: Export UNet architecture components
__all__ = [
    "UNet",  # Complete U-Net model for semantic segmentation with skip connections
    "UNetDecoderBlock",  # Single decoder block component for upsampling and feature fusion
]
