
import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str = None,
    level: str = "INFO",
    format_string: str = None,
):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
    )
    
    # Suppress noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
