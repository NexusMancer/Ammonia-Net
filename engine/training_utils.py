import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


# Immutable dataclass to store classification evaluation metrics
@dataclass(frozen=True)
class ClassificationMetrics:
    """Container for classification performance metrics.
    
    Attributes:
        accuracy: Overall classification accuracy
        precision: Macro-averaged precision across classes
        recall: Macro-averaged recall across classes
        f_score: Macro-averaged F1 score across classes
    """
    accuracy: float
    precision: float
    recall: float
    f_score: float


def compute_classification_metrics(actual, predicted):
    """Compute classification metrics comparing actual and predicted labels.
    
    Uses macro-averaging to handle multi-class scenarios fairly by computing
    metrics for each class and averaging without considering class imbalance.
    
    Args:
        actual: Ground truth labels
        predicted: Model predictions
        
    Returns:
        ClassificationMetrics object containing accuracy, precision, recall, and F1 score
    """
    return ClassificationMetrics(
        accuracy=accuracy_score(actual, predicted, normalize=True, sample_weight=None),
        precision=precision_score(actual, predicted, average="macro", zero_division=0),
        recall=recall_score(actual, predicted, average="macro", zero_division=0),
        f_score=f1_score(actual, predicted, average="macro", zero_division=0),
    )


def weights_init(net, init_type="normal", init_gain=0.02):
    """Initialize network weights using specified initialization strategy.
    
    Applies different weight initialization methods to convolutional and batch norm layers.
    Proper weight initialization can improve training convergence and model performance.
    
    Args:
        net: Neural network module to initialize
        init_type: Type of initialization - 'normal', 'xavier', 'kaiming', or 'orthogonal'
        init_gain: Gain parameter for initialization (std dev for normal, gain for others)
        
    Raises:
        NotImplementedError: If init_type is not one of the supported methods
    """
    def init_func(module):
        # Get the class name of the current module
        classname = module.__class__.__name__
        
        # Initialize convolutional layers
        if hasattr(module, "weight") and "Conv" in classname:
            if init_type == "normal":
                # Initialize from normal distribution with mean=0, std=init_gain
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                # Xavier initialization (uniform variance based on fan-in/fan-out)
                torch.nn.init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                # Kaiming initialization for ReLU networks (uniform variance based on fan-in)
                torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                # Orthogonal initialization for stability
                torch.nn.init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        # Initialize batch normalization layers
        elif "BatchNorm2d" in classname:
            # Set scale (weight) to 1 with small noise
            torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
            # Set bias to 0
            torch.nn.init.constant_(module.bias.data, 0.0)

    print("initialize network with %s type" % init_type)
    # Apply initialization function recursively to all submodules
    net.apply(init_func)





def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    shuffle=True,
    collate_fn=None,
    pin_memory=True,
    drop_last=True,
):
    """Create training and validation DataLoaders with consistent configuration.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for both loaders
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data (default: True)
        collate_fn: Custom collate function (default: None)
        pin_memory: Whether to pin memory (default: True)
        drop_last: Whether to drop last incomplete batch (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    return gen, gen_val


def create_sgd_optimizer(model, lr, momentum, weight_decay):
    """Create SGD optimizer with trainable parameters only.
    
    Args:
        model: Model containing parameters (may be wrapped in DataParallel)
        lr: Learning rate
        momentum: Momentum value
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        SGD optimizer instance configured with model's trainable parameters
    """
    # Get only trainable parameters (works with both wrapped and unwrapped models)
    pg = [p for p in model.parameters() if p.requires_grad]
    
    # Create and return optimizer
    optimizer = optim.SGD(
        pg,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    
    return optimizer


def get_lr(optimizer: Any) -> float:
    """Extract the current learning rate from an optimizer.
    
    Retrieves the learning rate from the first parameter group, which is
    typically used when all parameters share the same learning rate.
    
    Args:
        optimizer: PyTorch optimizer object.
    
    Returns:
        Current learning rate as a float.
    
    Raises:
        ValueError: If optimizer has no parameter groups.
    """
    # Verify optimizer has parameter groups
    if not getattr(optimizer, "param_groups", None):
        raise ValueError("Optimizer does not contain any parameter groups.")
    # Return learning rate from first parameter group
    return float(optimizer.param_groups[0]["lr"])


def show_config(**kwargs: Any) -> None:
    """Display configuration parameters in a formatted table.
    
    Pretty-prints key/value pairs with aligned columns and borders for
    clear visualization of configuration settings.
    
    Args:
        **kwargs: Arbitrary configuration key/value pairs to display.
    """
    # Convert all keys and values to strings for display
    rows = [(str(key), str(value)) for key, value in kwargs.items()]
    # Calculate column widths based on content
    key_width = max([4] + [len(key) for key, _ in rows])
    value_width = max([6] + [len(value) for _, value in rows])
    # Create horizontal divider line
    divider = f"+-{'-' * key_width}-+-{'-' * value_width}-+"

    # Print table header
    print("Configurations:")
    print(divider)
    print(f"| {'keys':<{key_width}} | {'values':<{value_width}} |")
    print(divider)
    # Print each configuration row
    for key, value in rows:
        print(f"| {key:<{key_width}} | {value:<{value_width}} |")
    # Print table footer
    print(divider)


__all__ = [
    "ClassificationMetrics",
    "compute_classification_metrics",
    "weights_init",
    "create_dataloaders",
    "create_sgd_optimizer",
    "get_lr",
    "show_config",
]
