"""
Logging configuration for the ArUco Pose Estimation package.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name="aruco_pose", log_file=None, level=logging.INFO, 
                 console=True, file_mode='a'):
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
        console: Whether to log to console (default: True)
        file_mode: File mode for file handler ('a' for append, 'w' for overwrite)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("my_module", "logs/app.log", level=logging.DEBUG)
        >>> logger.info("Starting application")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_path, mode=file_mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def get_logger(name="aruco_pose"):
    """
    Get existing logger or create a default one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, setup with defaults
    if not logger.handlers:
        setup_logger(name)
    
    return logger
