from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.blocks import ConvBNReLU


# ShuffleNet V2 X1.0 architecture parameters: number of inverted residual blocks per stage
_SHUFFLENET_V2_X1_0_STAGE_REPEATS = [4, 8, 4]
# ShuffleNet V2 X1.0 architecture parameters: output channels for each stage
_SHUFFLENET_V2_X1_0_OUT_CHANNELS = [24, 116, 232, 464, 1024]


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Shuffle channels across groups to enable cross-group information flow.
    
    Channel shuffle is a key operation in ShuffleNet that reorganizes channels
    to allow information exchange between different groups in grouped convolutions.
    The operation reshapes the tensor to group channels, transposes groups and
    channels, then reshapes back to the original spatial dimensions.
    
    Args:
        x: Input tensor of shape (batch_size, num_channels, height, width).
        groups: Number of groups to shuffle channels across.
        
    Returns:
        Tensor with shuffled channels, same shape as input.
    """
    # Get tensor dimensions
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # Reshape to group channels: (batch, groups, channels_per_group, height, width)
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # Swap groups and channels dimensions: (batch, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # Reshape back to original dimensions with shuffled channel order
    x = x.view(batch_size, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    """ShuffleNet V2 inverted residual block with depthwise separable convolutions.
    
    Implements the ShuffleNet V2 unit which uses:
    - Channel splitting: For stride=1, inputs are split into two branches
    - Depthwise separable convolutions: Reduces computation vs standard convolutions
    - Channel shuffle: Enables information exchange between branches
    - Identity connection: Preserves spatial dimensions when stride=1
    
    The block supports stride 1 (no spatial reduction) and stride 2 (spatial reduction),
    making it suitable for building multi-scale feature hierarchies.
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int):
        """Initialize an inverted residual block.
        
        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels. Must be even.
            stride: Stride for spatial reduction. Must be 1 or 2.
                   - stride=1: Channel splitting with identity + processing branch
                   - stride=2: No splitting; both input and processing branch are used
                   
        Raises:
            ValueError: If stride is not 1 or 2, output_channels is odd,
                       or if stride=1 and input_channels != 2 * output_channels.
        """
        super().__init__()

        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}.")
        if output_channels % 2 != 0:
            raise ValueError("output_channels must be even.")

        self.stride = stride
        # Each branch outputs half the total output channels
        branch_channels = output_channels // 2

        if self.stride == 1 and input_channels != branch_channels << 1:
            raise ValueError("input_channels must equal output_channels when stride is 1.")

        # Branch 1: Applies spatial reduction (stride=2) or identity (stride=1)
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                # Depthwise convolution with stride 2
                self.depthwise_conv(input_channels, input_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(input_channels),
                # 1x1 convolution to match output channels
                ConvBNReLU(input_channels, branch_channels, kernel_size=1),
            )
        else:
            # For stride=1, use identity since input is already split
            self.branch1 = nn.Identity()

        # Branch 2: Main processing path with depthwise separable convolutions
        branch2_input_channels = input_channels if self.stride > 1 else branch_channels
        self.branch2 = nn.Sequential(
            # 1x1 expansion convolution
            ConvBNReLU(branch2_input_channels, branch_channels, kernel_size=1),
            # Depthwise convolution with configured stride
            self.depthwise_conv(branch_channels, branch_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_channels),
            # 1x1 projection convolution
            ConvBNReLU(branch_channels, branch_channels, kernel_size=1),
        )

    @staticmethod
    def depthwise_conv(
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> nn.Conv2d:
        """Create a depthwise convolution layer.
        
        Depthwise convolution performs spatial convolution separately for each input channel,
        significantly reducing computation compared to standard convolution.
        
        Args:
            input_channels: Number of input channels (must equal output_channels for depthwise).
            output_channels: Number of output channels (must equal input_channels).
            kernel_size: Size of the convolution kernel.
            stride: Stride for the convolution. Default 1.
            padding: Padding for the convolution. Default 0.
            bias: Whether to include bias terms. Default False.
            
        Returns:
            nn.Conv2d layer configured as depthwise convolution (groups=input_channels).
        """
        return nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=input_channels,  # Depthwise: each input channel has its own set of filters
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the inverted residual block.
        
        Processing differs based on stride:
        - stride=1: Split input in half, apply branch2 to one half, concatenate with identity
        - stride=2: Apply both branches to full input and concatenate outputs
        Finally, shuffle channels across the two branches.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, output_channels, height//stride, width//stride).
        """
        if self.stride == 1:
            # Split input into two equal halves along channel dimension
            x1, x2 = x.chunk(2, dim=1)
            # Concatenate identity (x1) with processed branch (x2)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # Concatenate both branch outputs
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        # Shuffle channels across the two branches
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    """ShuffleNet V2 image classifier architecture.
    
    A lightweight CNN designed for efficient inference on mobile and edge devices.
    Key features:
    - Channel shuffling: Enables information flow between grouped convolutions
    - Depthwise separable convolutions: Reduces parameters and computation
    - Multi-scale features: Progressive spatial reduction through stages
    - Flexible input: Supports variable input channels and number of classes
    
    Reference: "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    """

    def __init__(
        self,
        stages_repeats: Sequence[int],
        stages_out_channels: Sequence[int],
        num_classes: int = 1000,
        input_channels: int = 3,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual,
    ):
        """Initialize ShuffleNet V2 architecture.
        
        Args:
            stages_repeats: Number of inverted residual blocks per stage (length 3).
                          Example: [4, 8, 4] for X1.0 variant.
            stages_out_channels: Output channels for each component (length 5).
                               Format: [initial_conv, stage1, stage2, stage3, final_conv].
                               Example: [24, 116, 232, 464, 1024] for X1.0 variant.
            num_classes: Number of output classes for classification. Default 1000.
            input_channels: Number of input channels (3 for RGB). Default 3.
            inverted_residual: Block type to use for stages. Default InvertedResidual.
            
        Raises:
            ValueError: If stages_repeats length is not 3 or stages_out_channels length is not 5.
        """
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("stages_repeats must have length 3.")
        if len(stages_out_channels) != 5:
            raise ValueError("stages_out_channels must have length 5.")

        self._stage_out_channels = list(stages_out_channels)

        # Initial convolution: reduce spatial dimensions and increase channels
        output_channels = self._stage_out_channels[0]
        self.conv1 = ConvBNReLU(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
        input_channels = output_channels

        # Max pooling: further spatial reduction
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build three stages with inverted residual blocks
        stages = []
        for repeats, output_channels in zip(stages_repeats, self._stage_out_channels[1:4]):
            stage = self._make_stage(
                input_channels=input_channels,
                output_channels=output_channels,
                repeats=repeats,
                inverted_residual=inverted_residual,
            )
            stages.append(stage)
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)

        # Final convolution: expand to target feature dimension for classification
        output_channels = self._stage_out_channels[-1]
        self.conv5 = ConvBNReLU(input_channels, output_channels, kernel_size=1)
        # Global average pooling: reduce spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Classification head: fully connected layer
        self.fc = nn.Linear(output_channels, num_classes)

    @staticmethod
    def _make_stage(
        *,
        input_channels: int,
        output_channels: int,
        repeats: int,
        inverted_residual: Callable[..., nn.Module],
    ) -> nn.Sequential:
        """Build a stage consisting of multiple inverted residual blocks.
        
        The first block in each stage uses stride=2 for spatial reduction,
        while subsequent blocks use stride=1 to maintain spatial dimensions.
        
        Args:
            input_channels: Number of input channels for the stage.
            output_channels: Number of output channels for all blocks in the stage.
            repeats: Number of inverted residual blocks in the stage.
            inverted_residual: Block class to instantiate (e.g., InvertedResidual).
            
        Returns:
            nn.Sequential containing all blocks for the stage.
        """
        # First block: stride=2 for spatial reduction and channel adjustment
        blocks = [inverted_residual(input_channels, output_channels, stride=2)]
        # Remaining blocks: stride=1 to maintain spatial dimensions
        for _ in range(repeats - 1):
            blocks.append(inverted_residual(output_channels, output_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ShuffleNet V2.
        
        Feature extraction pipeline:
        1. Initial convolution: spatial reduction and channel expansion
        2. Max pooling: further spatial reduction
        3. Three stages: progressive feature learning with spatial reduction
        4. Final convolution: expand to classification feature dimension
        5. Global average pooling: reduce spatial dimensions
        6. Fully connected layer: classification logits
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
            
        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Initial feature extraction with spatial reduction
        x = self.conv1(x)
        x = self.maxpool(x)
        # Multi-scale feature learning through stages
        for stage in self.stages:
            x = stage(x)
        # Final feature expansion
        x = self.conv5(x)
        # Global spatial pooling
        x = self.avgpool(x)
        # Flatten features for classification
        x = torch.flatten(x, 1)
        # Classification logits
        x = self.fc(x)
        return x


def shufflenet_v2_x1_0(num_classes: int = 1000, input_channels: int = 3) -> ShuffleNetV2:
    """Create a ShuffleNet V2 X1.0 model (base variant with 1x width multiplier).
    
    Args:
        num_classes: Number of output classes. Default 1000.
        input_channels: Number of input channels (3 for RGB). Default 3.
        
    Returns:
        Initialized ShuffleNetV2 model with X1.0 architecture parameters.
    """
    return ShuffleNetV2(
        stages_repeats=_SHUFFLENET_V2_X1_0_STAGE_REPEATS,
        stages_out_channels=_SHUFFLENET_V2_X1_0_OUT_CHANNELS,
        num_classes=num_classes,
        input_channels=input_channels,
    )


# Public API: Export ShuffleNet V2 components and factory function
__all__ = ["InvertedResidual", "ShuffleNetV2", "channel_shuffle", "shufflenet_v2_x1_0"]
