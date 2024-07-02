"""Isolated calls to methods that may segfault."""

from __future__ import annotations

import functools
import multiprocessing
from typing import Callable


class AbortedError(RuntimeError):
    """Process aborted."""


def _turn_exception_as_return(func, return_dict):
    """Decorator to turn an exception into a return value."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return_dict["result"] = result
            return_dict["exception"] = None
        except Exception as e:
            return_dict["result"] = None
            return_dict["exception"] = e

    return wrapper


def safe_call(func: Callable, *args, **kwargs):
    """Call a function in a separate process.

    Args:
        func: The function to call.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        AbortedError: If the process was aborted.
        Exception: If the function raised an exception.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(
        target=_turn_exception_as_return(func, return_dict), args=args, kwargs=kwargs
    )
    process.start()
    process.join()
    process.close()
    if not return_dict:
        raise AbortedError
    if return_dict["error"] is not None:
        raise return_dict["error"]
    return return_dict["results"]
