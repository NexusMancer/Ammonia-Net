from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set requires_grad flag for all parameters in a module.
    
    Controls whether gradients are computed and stored for module parameters.
    Useful for freezing parts of a model during transfer learning or multi-stage training.
    
    Args:
        module: PyTorch module whose parameters should be modified.
        requires_grad: Boolean flag to enable/disable gradient computation.
                      True: gradients will be computed during backward pass.
                      False: gradients will not be computed (frozen parameters).
    """
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def freeze_module(module: nn.Module) -> None:
    """Freeze a module by disabling gradient computation for all its parameters.
    
    Frozen modules won't be updated during training, which is useful for:
    - Transfer learning with pre-trained models
    - Fine-tuning specific parts while keeping others fixed
    - Reducing memory usage and computation during training
    
    Args:
        module: PyTorch module to freeze.
    """
    set_requires_grad(module, requires_grad=False)


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze a module by enabling gradient computation for all its parameters.
    
    Re-enables gradient computation for previously frozen modules, allowing them
    to be updated during training. Useful for staged fine-tuning strategies.
    
    Args:
        module: PyTorch module to unfreeze.
    """
    set_requires_grad(module, requires_grad=True)


class ConvBNReLU(nn.Sequential):
    """Sequential block combining convolution, batch normalization, and ReLU activation.
    
    A commonly used building block in modern CNNs that:
    - Applies convolution with configurable kernel size, stride, padding, and groups
    - Normalizes activations with batch normalization
    - Applies ReLU activation in-place to reduce memory
    
    The convolution uses bias=False since batch normalization will handle the bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        """Initialize ConvBNReLU block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Must be keyword-only argument.
            stride: Stride for convolution. Default 1.
            padding: Padding for convolution. Default 0.
            groups: Number of groups for grouped convolution (depthwise if groups=in_channels).
                   Default 1 (standard convolution).
        """
        super().__init__(
            # Convolution layer without bias (batch norm will handle bias)
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            # Batch normalization to normalize activations
            nn.BatchNorm2d(out_channels),
            # ReLU activation in-place to save memory
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    """Two sequential convolution layers with ReLU activations.
    
    A basic building block used in architectures like U-Net:
    - First convolution: changes channels from in_channels to out_channels
    - Second convolution: maintains out_channels (in_channels becomes out_channels)
    
    Both convolutions use 3x3 kernels with padding=1 to preserve spatial dimensions.
    Activations are applied in-place to reduce memory usage.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize DoubleConv block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (used for both conv layers after first one).
        """
        super().__init__(
            # First 3x3 convolution: in_channels -> out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # ReLU activation for first convolution
            nn.ReLU(inplace=True),
            # Second 3x3 convolution: out_channels -> out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # ReLU activation for second convolution
            nn.ReLU(inplace=True),
        )


class TensorNormalizer(nn.Module):
    """Normalize tensors using per-channel mean and standard deviation.
    
    Applies standardization to input tensors using learned or fixed normalization parameters.
    Commonly used for normalizing images to ImageNet statistics or other domain-specific values.
    
    The normalizer: output = (input - mean) / std
    - Mean and std are stored as buffers (part of model state but not trained parameters)
    - Supports any number of channels
    - Automatically handles different tensor dtypes
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Initialize TensorNormalizer with normalization parameters.
        
        Args:
            mean: Per-channel mean values (length must equal std length).
            std: Per-channel standard deviation values (length must equal mean length).
            
        Raises:
            ValueError: If mean and std have different lengths.
        """
        super().__init__()
        if len(mean) != len(std):
            raise ValueError("mean and std must contain the same number of channels.")

        # Register mean and std as buffers (not trainable parameters)
        # Reshape to (1, num_channels, 1, 1) for broadcasting with (N, C, H, W) tensors
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    @property
    def num_channels(self) -> int:
        """Get the number of channels this normalizer expects.
        
        Returns:
            Number of channels (length of mean/std vectors).
        """
        return int(self.mean.shape[1])

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize input tensor using registered mean and std.
        
        Applies: output = (input - mean) / std
        Both mean and std are automatically cast to match input dtype.
        
        Args:
            inputs: Input tensor of shape (batch_size, num_channels, height, width).
            
        Returns:
            Normalized tensor with same shape and dtype as input.
            
        Raises:
            ValueError: If input is not 4D or has wrong number of channels.
        """
        # Validate input shape
        if inputs.ndim != 4:
            raise ValueError(f"expected a 4D tensor shaped as (N, C, H, W), got {tuple(inputs.shape)}.")
        # Validate number of channels
        if inputs.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels for normalization, got {inputs.shape[1]}."
            )
        # Cast mean and std to match input dtype (for fp32/fp16 compatibility)
        mean = self.mean.to(dtype=inputs.dtype)
        std = self.std.to(dtype=inputs.dtype)
        # Apply normalization: (x - mean) / std
        return (inputs - mean) / std


# Public API: Export neural network building blocks and utility functions
__all__ = [
    "ConvBNReLU",  # Convolution + Batch Normalization + ReLU activatio block
    "DoubleConv",  # Double convolution block used in U-Net architectures
    "TensorNormalizer",  # Per-channel tensor normalization for preprocessing
    "freeze_module",  # Utility to freeze module parameters during training
    "set_requires_grad",  # Utility to control gradient computation for parameters
    "unfreeze_module",  # Utility to unfreeze module parameters for fine-tuning
]
