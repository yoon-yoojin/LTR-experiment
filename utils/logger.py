"""Logging utilities for training and evaluation."""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Set up logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log file name (if None, use timestamp)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """Logger for tracking training metrics."""

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """Initialize metric logger.

        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value.

        Args:
            tag: Metric name
            value: Metric value
            step: Training step
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalar values.

        Args:
            main_tag: Main tag name
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Training step
        """
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self) -> None:
        """Close logger."""
        if self.writer:
            self.writer.close()
