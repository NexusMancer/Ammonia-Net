import re
from typing import Any, Dict

import torch
from torch.nn import Module


def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    """Extract model state dictionary from a checkpoint file or dictionary.
    
    Handles various checkpoint formats by checking for standard keys like
    'state_dict', 'model_state_dict', or 'model'.
    
    Args:
        checkpoint: Checkpoint data, which can be a dictionary or a state dict itself.
        
    Returns:
        Model state dictionary containing the model weights and parameters.
    """
    # Check if checkpoint is a dictionary and extract the state dict using common keys
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    # Return checkpoint as-is if it's already a state dict
    return checkpoint


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'module.' prefix from state dictionary keys.
    
    When using nn.DataParallel, PyTorch adds 'module.' prefix to all keys.
    This function removes that prefix to allow loading into non-DataParallel models.
    
    Args:
        state_dict: Model state dictionary potentially with 'module.' prefixes.
        
    Returns:
        State dictionary with 'module.' prefixes removed if they existed.
    """
    # Check if any keys start with 'module.' prefix
    if not any(key.startswith("module.") for key in state_dict):
        # No prefix found, return as-is
        return state_dict
    # Remove 'module.' prefix from all keys
    return {key[len("module."):]: value for key, value in state_dict.items()}


def load_model_state(model: Module, checkpoint_path: str, map_location: torch.device) -> None:
    """Load model weights from a checkpoint file with automatic format handling.
    
    Handles various checkpoint formats and automatically strips 'module.' prefixes
    if needed (e.g., when loading a DataParallel model into a non-DataParallel model).
    
    Args:
        model: PyTorch model to load weights into.
        checkpoint_path: Path to the checkpoint file.
        map_location: Device to map checkpoint tensors to (e.g., 'cuda:0' or 'cpu').
        
    Raises:
        RuntimeError: If the state dict cannot be loaded after stripping prefixes.
    """
    # Load checkpoint file to the specified device
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # Extract state dict from checkpoint (handles different formats)
    state_dict = extract_state_dict(checkpoint)
    try:
        # Try to load state dict directly
        model.load_state_dict(state_dict)
    except RuntimeError:
        # If loading fails, try stripping 'module.' prefix for DataParallel models
        if isinstance(state_dict, dict):
            model.load_state_dict(strip_module_prefix(state_dict))
        else:
            # Re-raise if state dict is not a dictionary
            raise


def extract_epoch_num(filename: str) -> int:
    """Extract epoch number from checkpoint filename for natural sorting.
    
    Uses regex to find the first sequence of digits in the filename,
    which typically represents the epoch number (e.g., 'epoch_5.pth' -> 5).
    
    Args:
        filename: Checkpoint filename (e.g., 'epoch_5.pth', 'checkpoint_10.pt').
        
    Returns:
        Epoch number as integer, or 0 if extraction fails or no digits found.
    """
    try:
        # Search for the first sequence of digits in the filename
        match = re.search(r'\d+', filename)
        # Return the matched number, or 0 if no match found
        return int(match.group()) if match else 0
    except (ValueError, AttributeError):
        # Return 0 if any error occurs during conversion
        return 0
