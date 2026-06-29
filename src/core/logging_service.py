import logging
import functools
import time
import traceback
from pathlib import Path
from typing import Callable
from rich.logging import RichHandler


# Module lvl logger
_logger = None

def setup_logger(name: str = "nodule_detection", type: str = "INFO", log_file: Path = None) -> logging.Logger:
    """
    Initialize logger that is called at the start of training
    """

    global _logger

    if _logger is not None: return _logger

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, type))


    # console handler with formatting
    console_handler = RichHandler(rich_tracebacks=True, show_time=True)
    console_handler.setLevel(getattr(logging, type))
    logger.addHandler(console_handler)


    # Log file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter( '%(asctime)s | %(levelname)s | %(name)s | %(message)s' )) # !Adjust format here

        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger() ->logging.Logger:
    """
    Get configured logger, ensure to call setup_logger first
    """

    if _logger is None: raise RuntimeError("Please call setup_logger() first")
    return _logger


# Decorators: Used to decorate functions like training to view exection start, ends, runtime, errors

def log_execution(function: Callable) -> Callable:
    """
    Log function entry and exit
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.info(f"-> ENTERING {function.__name__}")
        result = function(*args, **kwargs)
        logger.info(f"<- EXITING {function.__name__}")

        return result
    
    return wrapper


def time_execution(function: Callable) -> Callable:
    """
    Time execution of inputted function
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = get_logger()

        # Time execution
        start = time.time()
        result = function(*args, **kwargs)
        elapsed_time = time.time() - start

        logger.info(f"{function.__name__} took {elapsed_time:.2f}s\n")
        
        return result
    
    return wrapper

def error_logging(function: Callable) -> Callable:
    """
    Display errors associated with running passed in function - full traceback and re-raise
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger =  get_logger()

        try:
            return function(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Exception in {function.__name__}: {exc}\n"f"{traceback.format_exc()}")
            raise
    return wrapper