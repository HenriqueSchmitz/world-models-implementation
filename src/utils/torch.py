import logging

import torch

from src.utils.logging import get_logger

def get_device(logger: logging.Logger = None) -> torch.device:
    logger = logger or get_logger()
    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else device)
    logger.info(f"Using device: {device}")
    return device