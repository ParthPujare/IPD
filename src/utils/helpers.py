"""
Common utility functions for the AdaniGreenPredictor project.
Includes environment loading, device detection, and helper functions.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import torch


# Load environment variables
load_dotenv()


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Path object pointing to project root
    """
    current_file = Path(__file__).resolve()
    # Go up from src/utils to project root
    return current_file.parent.parent.parent


def ensure_dir(path):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path (str or Path): Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_accelerator():
    """
    Detect the best available accelerator for PyTorch.
    Priority: CUDA > MPS > CPU
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device():
    """
    Get the PyTorch device object.
    
    Returns:
        torch.device: Device object
    """
    accelerator = get_accelerator()
    if accelerator == "cuda":
        return torch.device("cuda")
    elif accelerator == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    # Test utilities
    print(f"Project root: {get_project_root()}")
    print(f"Accelerator: {get_accelerator()}")
    print(f"Device: {get_device()}")
    
    # Test directory creation
    test_dir = get_project_root() / "test_dir"
    ensure_dir(test_dir)
    print(f"Created test directory: {test_dir}")
    if test_dir.exists():
        test_dir.rmdir()
        print("Test directory removed successfully")

