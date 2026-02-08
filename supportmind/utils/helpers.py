"""
Utility functions for SupportMind.
"""

import logging
import sys
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """Detect and return device information."""
    info = {
        'cuda_available': False,
        'device_count': 0,
        'device_name': None,
        'device': 'cpu'
    }

    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['device'] = 'cuda'
            info['device_count'] = torch.cuda.device_count()
            info['device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return info


def check_gpu_memory() -> Dict[str, float]:
    """Check GPU memory usage."""
    usage = {'allocated_mb': 0, 'cached_mb': 0, 'total_mb': 0}

    try:
        import torch
        if torch.cuda.is_available():
            usage['allocated_mb'] = torch.cuda.memory_allocated() / 1e6
            usage['cached_mb'] = torch.cuda.memory_reserved() / 1e6
            usage['total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1e6
    except ImportError:
        pass

    return usage


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("supportmind")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
