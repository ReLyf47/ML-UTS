import logging
import sys
from pathlib import Path
from kode.common.config import LOGS_DIR

def setup_logger(name: str = "pipeline", log_file: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """Setup and configure a unified logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(level)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    log_path = LOGS_DIR / log_file
    f_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    f_handler.setLevel(level)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger
