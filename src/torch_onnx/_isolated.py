"""Isolated calls to methods that may segfault."""

from __future__ import annotations

import multiprocessing
from typing import Callable
import os
import warnings


_IS_WINDOWS = os.name == "nt"


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
    if _IS_WINDOWS:
        # On Windows, we cannot create a new process with fork.
        warnings.warn(
            f"A new process is not created for {func} on Windows.", stacklevel=1
        )
        return func(*args, **kwargs)

    with multiprocessing.get_context("fork").Pool(1) as pool:
        # It is important to fork a process here to prevent the main logic from
        # running again when the user does not place it under a `if __name__ == "__main__":`
        # block.
        result = pool.apply(_call_function_and_return_exception, (func, args, kwargs))
    if isinstance(result, Exception):
        raise result
    return result
