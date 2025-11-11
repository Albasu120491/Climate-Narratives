"""API utilities for rate limiting and retries."""

import time
import logging
from functools import wraps
from typing import Callable
import os

logger = logging.getLogger(__name__)


def rate_limited(max_per_minute: int = None):
    """
    Decorator for rate limiting API calls.
    
    Args:
        max_per_minute: Maximum calls per minute
    """
    if max_per_minute is None:
        max_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 60))
    
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        
        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
):
    """
    Decorator for retrying with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        backoff_factor: Backoff multiplier
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        return wrapper
    return decorator
