"""
Device management utilities.

This module provides utilities for automatically detecting and configuring
the best available device (CUDA, MPS, XPU, or CPU).
"""

import torch
from typing import Optional


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.

    Automatically detects and returns the best available device in the
    following priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. XPU (Intel GPUs)
    4. CPU

    Args:
        preferred: Preferred device name ("cuda", "mps", "xpu", "cpu")
                  If None, automatically selects the best available

    Returns:
        PyTorch device object

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        elif preferred == "cpu":
            return torch.device("cpu")
        else:
            print(
                f"Warning: Preferred device '{preferred}' not available, "
                f"falling back to auto-detection"
            )

    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def setup_device(
    device: Optional[torch.device] = None, optimize_for_device: bool = True
) -> torch.device:
    """Setup and configure device for optimal performance.

    Args:
        device: Device to setup (if None, auto-detects)
        optimize_for_device: Whether to apply device-specific optimizations

    Returns:
        Configured device

    Example:
        >>> device = setup_device(optimize_for_device=True)
        >>> model = model.to(device)
    """
    if device is None:
        device = get_device()

    if optimize_for_device:
        if device.type == "cuda":
            # Enable TF32 for better performance on Ampere+ GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 for CUDA device")

        elif device.type == "mps":
            # MPS-specific optimizations (if any)
            print("Using MPS device (Apple Silicon)")

        elif device.type == "xpu":
            # XPU-specific optimizations (if any)
            print("Using XPU device (Intel GPU)")

        elif device.type == "cpu":
            print("Using CPU device")

    return device


def get_device_info(device: torch.device) -> dict:
    """Get detailed information about a device.

    Args:
        device: Device to query

    Returns:
        Dictionary with device information

    Example:
        >>> device = get_device()
        >>> info = get_device_info(device)
        >>> print(info["name"])
    """
    info = {"type": device.type, "index": device.index}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["memory_total"] = torch.cuda.get_device_properties(
            device
        ).total_memory / 1e9
        info["memory_allocated"] = torch.cuda.memory_allocated(device) / 1e9
        info["capability"] = torch.cuda.get_device_capability(device)

    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"

    elif device.type == "xpu":
        info["name"] = "Intel XPU"

    elif device.type == "cpu":
        info["name"] = "CPU"

    return info


def print_device_info(device: Optional[torch.device] = None):
    """Print detailed device information.

    Args:
        device: Device to print info for (if None, uses current device)

    Example:
        >>> device = setup_device()
        >>> print_device_info(device)
    """
    if device is None:
        device = get_device()

    info = get_device_info(device)
    print(f"\nDevice Information:")
    print(f"  Type: {info['type']}")
    print(f"  Name: {info.get('name', 'N/A')}")

    if "memory_total" in info:
        print(f"  Total Memory: {info['memory_total']:.2f} GB")
        print(f"  Allocated Memory: {info['memory_allocated']:.2f} GB")

    if "capability" in info:
        print(f"  Compute Capability: {info['capability']}")

    print()
