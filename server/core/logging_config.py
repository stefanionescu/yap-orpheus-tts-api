import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "DEBUG",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up comprehensive logging for the Orpheus TTS server.
    
    Args:
        log_file: Path to log file. If None, defaults to 'logs/orpheus-tts.log'
        level: Logging level (DEBUG is default for maximum verbosity, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        log_format: Custom log format string
        
    Returns:
        Configured root logger
    """
    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_file = str(logs_dir / "orpheus-tts.log")
    else:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - "
            "%(funcName)s() - %(message)s"
        )
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Initialize logging on import if not already configured
_initialized = False

def ensure_logging_initialized():
    """Ensure logging is initialized with default settings."""
    global _initialized
    if not _initialized:
        # Get configuration from environment variables
        log_level = os.getenv("LOG_LEVEL", "DEBUG")
        log_file = os.getenv("LOG_FILE", None)
        max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        
        setup_logging(
            log_file=log_file,
            level=log_level,
            max_bytes=max_bytes,
            backup_count=backup_count
        )
        _initialized = True


# Auto-initialize with default settings
ensure_logging_initialized()
