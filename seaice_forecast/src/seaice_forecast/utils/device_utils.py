"""
Device selection utilities optimized for Apple Silicon MPS.

Provides consistent device selection logic with proper fallbacks
and explicit MPS support.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_optimal_device(prefer_device=None, verbose=True):
    """
    Get optimal device with proper fallback hierarchy.

    Priority:
    1. MPS (Apple Silicon) if available
    2. CUDA if available
    3. CPU as fallback

    Args:
        prefer_device: Optional device preference ('mps', 'cuda', 'cpu')
        verbose: Whether to log device selection

    Returns:
        torch.device object
    """
    if prefer_device:
        # User specified preference
        if prefer_device == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                if verbose:
                    logger.info(f"✓ Using MPS device (Apple Silicon)")
            else:
                device = torch.device('cpu')
                if verbose:
                    logger.warning("⚠ MPS requested but not available, using CPU")
        elif prefer_device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                if verbose:
                    logger.info(f"✓ Using CUDA device")
            else:
                device = torch.device('cpu')
                if verbose:
                    logger.warning("⚠ CUDA requested but not available, using CPU")
        else:
            device = torch.device('cpu')
            if verbose:
                logger.info("Using CPU device")
    else:
        # Auto-detect best available device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            if verbose:
                logger.info(f"✓ Auto-selected MPS device (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            if verbose:
                logger.info(f"✓ Auto-selected CUDA device")
        else:
            device = torch.device('cpu')
            if verbose:
                logger.info("Auto-selected CPU device")

    if verbose:
        logger.info(f"Device: {device}")
        logger.info(f"PyTorch version: {torch.__version__}")

        if device.type == 'mps':
            logger.info("MPS optimization tips:")
            logger.info("  - Use batch sizes of 16-32 for better throughput")
            logger.info("  - Avoid pin_memory=True (MPS doesn't benefit)")
            logger.info("  - Try fp16 autocast if numerically stable")

    return device


def get_dataloader_config(device, num_workers=None):
    """
    Get optimal DataLoader configuration for device.

    Args:
        device: torch.device
        num_workers: Override number of workers (None = auto)

    Returns:
        Dictionary with DataLoader config
    """
    config = {}

    if device.type == 'mps':
        # MPS-specific config
        config['pin_memory'] = False  # MPS doesn't benefit
        config['num_workers'] = num_workers if num_workers is not None else 2
        config['persistent_workers'] = config['num_workers'] > 0
    elif device.type == 'cuda':
        # CUDA-specific config
        config['pin_memory'] = True  # CUDA benefits from pinned memory
        config['num_workers'] = num_workers if num_workers is not None else 4
        config['persistent_workers'] = config['num_workers'] > 0
    else:
        # CPU config
        config['pin_memory'] = False
        config['num_workers'] = num_workers if num_workers is not None else 2
        config['persistent_workers'] = config['num_workers'] > 0

    return config


def clear_device_cache(device):
    """Clear device cache to free memory."""
    if device.type == 'mps':
        torch.mps.empty_cache()
        logger.info("✓ Cleared MPS cache")
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("✓ Cleared CUDA cache")


def get_autocast_context(device, enabled=True, dtype=torch.float16):
    """
    Get autocast context manager for device.

    Args:
        device: torch.device
        enabled: Whether autocast is enabled
        dtype: Target dtype (typically float16)

    Returns:
        Context manager for autocast
    """
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)

    if device.type == 'mps':
        # MPS supports autocast
        return torch.autocast(device_type='mps', dtype=dtype)
    elif device.type == 'cuda':
        # CUDA supports autocast
        return torch.autocast(device_type='cuda', dtype=dtype)
    else:
        # CPU autocast with bfloat16
        return torch.autocast(device_type='cpu', dtype=torch.bfloat16 if enabled else torch.float32)


def check_mps_fallbacks(verbose=True):
    """
    Check if MPS fallback warnings are enabled.

    Returns:
        bool: True if fallback environment variable is set
    """
    import os
    fallback_enabled = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '1'

    if verbose:
        if fallback_enabled:
            logger.info("✓ MPS fallback warnings enabled")
            logger.info("  Will see warnings for ops that fall back to CPU")
        else:
            logger.info("MPS fallback warnings disabled")
            logger.info("  Set PYTORCH_ENABLE_MPS_FALLBACK=1 to enable")

    return fallback_enabled


def print_device_info(device):
    """Print detailed device information."""
    logger.info("\n" + "="*60)
    logger.info("DEVICE INFORMATION")
    logger.info("="*60)
    logger.info(f"Device type: {device.type}")
    logger.info(f"PyTorch version: {torch.__version__}")

    if device.type == 'mps':
        logger.info("Apple Silicon (MPS) detected")
        logger.info("Features:")
        logger.info("  ✓ Unified memory architecture")
        logger.info("  ✓ Fast matrix operations")
        logger.info("  ✓ FP16 autocast supported")
        logger.info("\nOptimizations:")
        logger.info("  - Use larger batch sizes (16-32)")
        logger.info("  - Use num_workers=2-4 for DataLoader")
        logger.info("  - Avoid pin_memory=True")
        logger.info("  - Try FP16 autocast if stable")
    elif device.type == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CPU device")

    logger.info("="*60 + "\n")
