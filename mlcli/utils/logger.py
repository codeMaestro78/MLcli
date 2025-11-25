"""
Logging Utilities

Provides structured logging setup with console and file handlers,
supporting different log levels and formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from joblib import Logger

# ANSI color code for the console output

class LogColors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"

class ColoredForamtter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    COLORS={
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }

    def format(self,record:logging.LogRecord)->str:
        """Foramt log records with colors"""
        # Add colot to level name
        levelname=record.levelname
        if record.levelno in self.COLORS:
            levelname_color=(self.COLORS[record.levelno]+ levelname+ LogColors.RESET)
            record.levelname=levelname_color

        # Format message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return result


def setup_logger(name:str="mlcli",level:str="INFO",log_file:Optional[str]=None,
                 console: bool= True,file_level:str="DEBUG")->logging.Logger:
    """
    Set up logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to enable console logging
        file_level: File logging level

    Returns:
        Configured logger instance
    """
    logger=logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler with colors
    if console:
        console_handler=logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging,level.upper()))

        console_format = ColoredForamtter(fmt="%(levelname)s | %(message)s",datefmt="%H:%M:%S")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    if log_file:
        log_path=Path(log_file)
        log_path.mkdir(parents=True,exist_ok=True)

        file_handler=logging.FileHandler(log_path,encoding="utf-8")
        file_handler.setLevel(getattr(logging,file_level.upper()))

        file_format=logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propogation to root logger
    logger.propagate =False

    return logger

def get_logger(name:str = "mlcli") -> logging.Logger:
    """
    Get existing logger or create new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger =  logging.getLogger(name)

    # Set up default logger if not configured
    if not logger.handlers:
        setup_logger(name)

    return logger

def create_run_logger(run_id:str,log_dir:str="runs")->logging.Logger:
    """
    Create a logger for a specific training run.

    Args:
        run_id: Unique identifier for the run
        log_dir: Directory to store log files

    Returns:
        Logger configured for the run
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True,exist_ok=True)

    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"run_{run_id}_{timestamp}.log"

    logger = setup_logger(name=f"mlcli.run.{run_id}",level="INFO",log_file=str(log_file),
                            console=True)
    return logger

class LoggerContext:
    """Context manager for temporary logger configuration."""

    def __init__(self,level:str ="INFO",log_file:Optional[str]=None):
        """
        Initialize logger context.

        Args:
            level: Logging level
            log_file: Optional log file path
        """

        self.level=level
        self.log_file= log_file
        self.logger = get_logger()
        self.original_level = self.logger.level
        self.added_handlers=[]

    def __enter__(self) -> logging.Logger:
        """Enter context and configure logger."""
        self.logger.setLevel(getattr(logging,self.level.upper()))

        if self.log_file:
            file_handler=logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.added_handlers.append(file_handler)

        return self.logger

    def __exit__(self,exc_type,exc_val,exc_tb):
        """Exit context and restore logger. """
        self.logger.setLevel(self.original_level)

        for handler in self.added_handlers:
            self.logger.removeHandler(handler)
            handler.close()
