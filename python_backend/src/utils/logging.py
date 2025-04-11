import logging
import sys
import os
from logging.handlers import RotatingFileHandler

LOG_FILE_NAME = "simu_verse_backend.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs') # Assuming logs dir is at project root
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, LOG_FILE_NAME)

# Keep track if logging has been configured to avoid duplicate handlers
_logging_configured = False

def setup_logging(level=logging.INFO, log_to_console=True, log_to_file=True):
    """
    Configures logging for the application.

    Args:
        level: The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
        log_to_console: Whether to log messages to the console.
        log_to_file: Whether to log messages to a file.
    """
    global _logging_configured
    if _logging_configured:
        # logging.debug("Logging already configured.")
        return

    logger = logging.getLogger() # Get root logger
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    handlers = []

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File Handler (Rotating)
    if log_to_file:
        try:
            # Rotate logs, keep 5 backups of 5MB each
            file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging to {log_file_path}. Error: {e}", file=sys.stderr)


    # Remove existing handlers if any (to prevent duplicates in interactive environments)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add the configured handlers
    for handler in handlers:
        logger.addHandler(handler)

    _logging_configured = True
    # logging.info("Logging configured.") # Log initial configuration

# Example usage at the start of a module:
# import logging
# from .logging import setup_logging
# setup_logging()
# logger = logging.getLogger(__name__)
# logger.info("This is an info message.")
