# Misc helpers (logging, color conversion, etc.)

# src/utils.py
import logging
import os
from datetime import datetime

def get_logger(name, level=logging.INFO, log_dir='logs'):
    """Returns a logger that writes to stdout and a rotating log file."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    # Create log directory if missing
    os.makedirs(log_dir, exist_ok=True)

    # Unique log file name for each run (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{name}_{timestamp}.log"

    # File handler
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.setLevel(level)
    logger.propagate = False
    return logger
