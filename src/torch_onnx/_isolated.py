"""Isolated calls to methods that may segfault."""

from __future__ import annotations

import multiprocessing
from typing import Callable


def _call_function_and_return_exception(func, args, kwargs):
    """Call function and return a exception if there is one."""

    try:
        return func(*args, **kwargs)
    except Exception as e:
        return e


def safe_call(func: Callable, *args, **kwargs):
    """Call a function in a separate process.

    Args:
        func: The function to call.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        Exception: If the function raised an exception.
    """
    with multiprocessing.Pool(1) as pool:
        result = pool.apply(_call_function_and_return_exception, (func, args, kwargs))
    if isinstance(result, Exception):
        raise result
    return result
